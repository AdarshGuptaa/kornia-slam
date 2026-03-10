//! Map-projection-based estimator: matches map points to frame features via PnP.

use std::collections::HashSet;

use kornia_3d::camera::{ImageSize, PinholeCamera};
use kornia_3d::pose::Pose3d;
use kornia_imgproc::features::OrbMatchConfig;

use crate::mapping::{Map, MapPoint};
use super::matching::{
    KeypointGrid, ProjectionMatchConfig, match_by_projection, match_map_to_frame,
    match_orb_descriptors,
};
use super::pnp::{pose_increment_ok, solve_pnp_from_correspondences};
use super::pnp::PnpConfig;
use super::Estimator;
use crate::frame::Frame;

/// Keyframe insertion heuristics.
#[derive(Debug, Clone)]
pub struct KeyframePolicy {
    /// Minimum frame gap before allowing keyframe insertion.
    pub min_frames_between: usize,
    /// Force a keyframe if this frame gap is reached.
    pub max_frames_between: usize,
    /// Relative inlier ratio threshold (vs reference keyframe tracked map points).
    pub ref_ratio: f64,
}

impl Default for KeyframePolicy {
    fn default() -> Self {
        Self {
            min_frames_between: 3,
            max_frames_between: 8,
            ref_ratio: 0.6,
        }
    }
}

/// Odometry-phase thresholds.
#[derive(Debug, Clone)]
pub struct OdometryConfig {
    /// ORB descriptor matcher settings for tracking against reference observations.
    pub match_config: OrbMatchConfig,
    /// Minimum pose inliers to accept a keyframe.
    pub min_inliers: usize,
    /// Minimum median parallax in degrees to accept a keyframe.
    pub min_parallax_deg: f64,
    /// Maximum consecutive tracking failures before resetting to bootstrap.
    pub max_consecutive_failures: usize,
    /// Enable local bundle adjustment after keyframe insertion.
    pub enable_local_ba: bool,
    /// Keyframe insertion policy.
    pub keyframe_policy: KeyframePolicy,
    /// PnP pose-estimation thresholds.
    pub pnp: PnpConfig,
}

impl Default for OdometryConfig {
    fn default() -> Self {
        Self {
            match_config: OrbMatchConfig {
                nn_ratio: 0.6,
                th_low: 50,
                check_orientation: true,
                histo_length: 30,
            },
            min_inliers: 30,
            min_parallax_deg: 1.0,
            max_consecutive_failures: 15,
            enable_local_ba: true,
            keyframe_policy: KeyframePolicy::default(),
            pnp: PnpConfig::default(),
        }
    }
}

/// Rejection reasons specific to the map-projection tracker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EstimationRejectReason {
    LowProjectionMatches,
    PnpFailed,
    PnpInconsistentMotion,
    LowPnpInliers,
    LowReferenceMatches,
    LowReferenceCorrespondences,
}

/// Map-projection-based pose estimator.
///
/// Estimates the camera pose by projecting map points into the current frame,
/// matching via ORB descriptors, and solving PnP.
pub struct MapProjectionEstimator {
    camera: PinholeCamera,
    config: OdometryConfig,
    /// Matches from the last successful `track()` call: `(map_point_idx, keypoint_idx)`.
    last_matches: Vec<(usize, usize)>,
    /// Inlier count from the last successful `track()` call.
    last_inliers: usize,
}

impl MapProjectionEstimator {
    pub fn new(camera: PinholeCamera, config: OdometryConfig) -> Self {
        Self {
            camera,
            config,
            last_matches: Vec::new(),
            last_inliers: 0,
        }
    }

    pub fn camera(&self) -> &PinholeCamera {
        &self.camera
    }

    pub fn config(&self) -> &OdometryConfig {
        &self.config
    }

    /// Matches from the last successful `track()` call.
    pub fn last_matches(&self) -> &[(usize, usize)] {
        &self.last_matches
    }

    /// Inlier count from the last successful `track()` call.
    pub fn last_inliers(&self) -> usize {
        self.last_inliers
    }

    /// Estimate the pose of `frame` against the map.
    pub fn track(
        &mut self,
        frame: &Frame,
        candidate_pose: &Pose3d,
        pose_before_tracking: &Pose3d,
        map: &Map,
        current_keyframe_idx: Option<usize>,
    ) -> Result<Pose3d, EstimationRejectReason> {
        let (pose, inliers, matches) = self.estimate_pose(
            frame,
            candidate_pose,
            pose_before_tracking,
            map,
            current_keyframe_idx,
        )?;
        self.last_matches = matches;
        self.last_inliers = inliers;
        Ok(pose)
    }

    /// Update `n_visible` and `n_found` for map points based on the current frame.
    pub fn update_map_point_observations(
        &self,
        map: &mut Map,
        matched: &[(usize, usize)],
        pose_world_to_cam: &Pose3d,
        image_size: ImageSize,
    ) {
        let (fx, fy, cx, cy) = self.camera.intrinsics();
        let img_w = image_size.width as f32;
        let img_h = image_size.height as f32;

        let mut matched_set: HashSet<usize> = HashSet::new();
        for &(mp_idx, _) in matched {
            matched_set.insert(mp_idx);
        }

        for (mp_idx, mp) in map.map_points_mut().iter_mut().enumerate() {
            if mp.culled {
                continue;
            }
            let p_cam = pose_world_to_cam.transform_point(&mp.position);
            if p_cam.z <= 0.0 {
                continue;
            }
            let u = (fx * p_cam.x / p_cam.z + cx) as f32;
            let v = (fy * p_cam.y / p_cam.z + cy) as f32;
            if u < 0.0 || v < 0.0 || u >= img_w || v >= img_h {
                continue;
            }
            mp.n_visible = mp.n_visible.saturating_add(1);
            if matched_set.contains(&mp_idx) {
                mp.n_found = mp.n_found.saturating_add(1);
            }
        }
    }

    /// Decide whether a new keyframe should be inserted.
    pub fn need_new_keyframe(
        &self,
        curr_idx: usize,
        last_keyframe_idx: Option<usize>,
        tracked_inliers: usize,
        n_ref_map_points: usize,
    ) -> bool {
        let Some(last_kf_idx) = last_keyframe_idx else {
            return true;
        };

        let frames_since_last_kf = curr_idx.saturating_sub(last_kf_idx);
        if frames_since_last_kf < self.config.keyframe_policy.min_frames_between {
            return false;
        }
        if frames_since_last_kf >= self.config.keyframe_policy.max_frames_between {
            return true;
        }

        if n_ref_map_points == 0 {
            return true;
        }

        let weak_threshold =
            (n_ref_map_points as f64 * self.config.keyframe_policy.ref_ratio) as usize;
        tracked_inliers >= 15 && tracked_inliers < weak_threshold
    }

    // ── private ──

    fn estimate_pose(
        &self,
        frame: &Frame,
        candidate_pose: &Pose3d,
        pose_before_tracking: &Pose3d,
        map: &Map,
        current_keyframe_idx: Option<usize>,
    ) -> Result<(Pose3d, usize, Vec<(usize, usize)>), EstimationRejectReason> {
        const MIN_PNP_CORRESPONDENCES: usize = 4;
        const MIN_STAGE1_INLIERS: usize = 10;

        // Stage 1: projection matching (narrow, then wide).
        let frame_match = match_map_to_frame(
            map.map_points(),
            frame,
            &self.camera,
            candidate_pose,
            ProjectionMatchConfig {
                min_depth: 0.0,
                search_radius: 15.0,
                max_hamming: 50,
            },
        );
        let curr_keypoints_undist = &frame_match.keypoints_undist;
        let grid = &frame_match.grid;
        let projection_matches = frame_match.matches;

        // Stage 2: PnP from projection matches.
        let last_reject = if projection_matches.len() >= MIN_PNP_CORRESPONDENCES {
            match try_track(
                map.map_points(),
                &self.camera,
                &self.config.pnp,
                &projection_matches,
                curr_keypoints_undist,
                candidate_pose,
                candidate_pose,
                MIN_STAGE1_INLIERS,
            ) {
                TrackAttempt::Success { pose, inliers } => {
                    return self.refine_pose(
                        curr_keypoints_undist,
                        &frame.features.descriptors,
                        grid,
                        frame.image_size,
                        candidate_pose,
                        pose,
                        inliers,
                        projection_matches,
                        map,
                        current_keyframe_idx,
                    );
                }
                TrackAttempt::Rejected(reason) => reason,
            }
        } else {
            EstimationRejectReason::LowProjectionMatches
        };

        // Stage 3: fallback — match against reference keyframe descriptors.
        let current_kf = current_keyframe_idx.and_then(|ki| map.get_keyframe(ki));
        let Some(current_kf) = current_kf else {
            return Err(last_reject);
        };

        let ref_matches = match_orb_descriptors(
            &current_kf.frame.features.orientations,
            &current_kf.frame.features.descriptors,
            &frame.features.orientations,
            &frame.features.descriptors,
            self.config.match_config,
        );

        const MIN_REF_MATCHES: usize = 15;
        if ref_matches.len() < MIN_REF_MATCHES {
            return Err(EstimationRejectReason::LowReferenceMatches);
        }

        let mut ref_correspondences = Vec::with_capacity(ref_matches.len());
        for (kf_desc_idx, curr_idx) in ref_matches {
            if let Some(Some(mp_idx)) = current_kf.map_point_by_desc_idx.get(kf_desc_idx) {
                if *mp_idx < map.map_points().len() {
                    ref_correspondences.push((*mp_idx, curr_idx));
                }
            }
        }

        if ref_correspondences.len() < MIN_PNP_CORRESPONDENCES {
            return Err(EstimationRejectReason::LowReferenceCorrespondences);
        }

        match try_track(
            map.map_points(),
            &self.camera,
            &self.config.pnp,
            &ref_correspondences,
            curr_keypoints_undist,
            pose_before_tracking,
            candidate_pose,
            MIN_STAGE1_INLIERS,
        ) {
            TrackAttempt::Success { pose, inliers } => self.refine_pose(
                curr_keypoints_undist,
                &frame.features.descriptors,
                grid,
                frame.image_size,
                candidate_pose,
                pose,
                inliers,
                ref_correspondences,
                map,
                current_keyframe_idx,
            ),
            TrackAttempt::Rejected(reason) => Err(reason),
        }
    }

    fn refine_pose(
        &self,
        curr_keypoints_undist: &[[f32; 2]],
        curr_descriptors: &[[u8; 32]],
        grid: &KeypointGrid,
        image_size: ImageSize,
        candidate_pose: &Pose3d,
        mut pose: Pose3d,
        mut inliers: usize,
        mut matches: Vec<(usize, usize)>,
        map: &Map,
        current_keyframe_idx: Option<usize>,
    ) -> Result<(Pose3d, usize, Vec<(usize, usize)>), EstimationRejectReason> {
        if let Some((local_matches, local_pose, local_inliers)) = refine_with_local_map(
            map,
            current_keyframe_idx,
            &self.camera,
            &self.config.pnp,
            &matches,
            curr_keypoints_undist,
            curr_descriptors,
            grid,
            image_size,
            &pose,
        ) {
            matches = local_matches;
            if local_inliers >= self.config.min_inliers
                && pose_increment_ok(candidate_pose, &local_pose, &self.config.pnp)
            {
                pose = local_pose;
                inliers = local_inliers;
            }
        }

        Ok((pose, inliers, matches))
    }
}

impl Estimator for MapProjectionEstimator {
    fn estimate(
        &mut self,
        frame: &Frame,
        predicted_pose: &Pose3d,
    ) -> Option<Pose3d> {
        // Delegates to the full map-aware method using the predicted pose
        // as both candidate and reference. Without a map or keyframe context,
        // this is the best we can do via the generic interface.
        self.track(frame, predicted_pose, predicted_pose, &Map::new(), None)
            .ok()
    }
}

// ── private helpers ──

/// Result of a PnP tracking attempt.
enum TrackAttempt {
    Success { pose: Pose3d, inliers: usize },
    Rejected(EstimationRejectReason),
}

fn try_track(
    map_points: &[MapPoint],
    camera: &PinholeCamera,
    pnp_config: &PnpConfig,
    correspondences: &[(usize, usize)],
    keypoints_undist: &[[f32; 2]],
    pose_init: &Pose3d,
    cand_pose: &Pose3d,
    min_inliers: usize,
) -> TrackAttempt {
    match solve_pnp_from_correspondences(
        map_points,
        correspondences,
        keypoints_undist,
        pose_init,
        camera,
        pnp_config,
    ) {
        Some((new_pose, inliers)) => {
            if inliers >= min_inliers {
                if pose_increment_ok(cand_pose, &new_pose, pnp_config) {
                    TrackAttempt::Success {
                        pose: new_pose,
                        inliers,
                    }
                } else {
                    TrackAttempt::Rejected(EstimationRejectReason::PnpInconsistentMotion)
                }
            } else {
                TrackAttempt::Rejected(EstimationRejectReason::LowPnpInliers)
            }
        }
        None => TrackAttempt::Rejected(EstimationRejectReason::PnpFailed),
    }
}

fn refine_with_local_map(
    map: &Map,
    current_kf_idx: Option<usize>,
    camera: &PinholeCamera,
    pnp_config: &PnpConfig,
    tracked_matches: &[(usize, usize)],
    curr_keypoints_undist: &[[f32; 2]],
    curr_descriptors: &[[u8; 32]],
    grid: &KeypointGrid,
    image_size: ImageSize,
    pose_init: &Pose3d,
) -> Option<(Vec<(usize, usize)>, Pose3d, usize)> {
    let current_kf = current_kf_idx.and_then(|ki| map.get_keyframe(ki));
    let (local_map_points, local_to_global) =
        map.build_local_map_points(tracked_matches, current_kf);
    if local_map_points.len() < 4 {
        return None;
    }

    let local_config = ProjectionMatchConfig {
        min_depth: 0.0,
        search_radius: 30.0,
        max_hamming: 60,
    };
    let local_matches = match_by_projection(
        &local_map_points,
        curr_keypoints_undist,
        curr_descriptors,
        grid,
        pose_init,
        camera,
        image_size,
        local_config,
    );
    if local_matches.len() < 4 {
        return None;
    }

    let global_matches: Vec<(usize, usize)> = local_matches
        .into_iter()
        .filter_map(|(local_mp_idx, curr_idx)| {
            local_to_global
                .get(local_mp_idx)
                .copied()
                .map(|global_mp_idx| (global_mp_idx, curr_idx))
        })
        .collect();
    if global_matches.len() < 4 {
        return None;
    }

    let (new_pose, inliers) = solve_pnp_from_correspondences(
        map.map_points(),
        &global_matches,
        curr_keypoints_undist,
        pose_init,
        camera,
        pnp_config,
    )?;
    Some((global_matches, new_pose, inliers))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_need_new_keyframe_forced_by_gap() {
        let camera = PinholeCamera {
            fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0,
            k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0,
        };
        let config = OdometryConfig::default();
        let tracker = MapProjectionEstimator::new(camera, config);
        assert!(tracker.need_new_keyframe(100, Some(90), 50, 100));
    }

    #[test]
    fn test_need_new_keyframe_too_soon() {
        let camera = PinholeCamera {
            fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0,
            k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0,
        };
        let config = OdometryConfig::default();
        let tracker = MapProjectionEstimator::new(camera, config);
        assert!(!tracker.need_new_keyframe(2, Some(1), 50, 100));
    }
}
