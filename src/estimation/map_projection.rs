//! Map-projection-based estimator: matching, PnP, and tracking flow.
//!
//! ```text
//!        Map Points (3D)
//!        *   *       *
//!         \  |      /
//!    project & match (ORB)
//!           \|/
//!    .---------------.
//!   /    * . *      /   current frame
//!  /       *       /    (2D keypoints)
//!  '---------------'
//!          |
//!      solve PnP
//!          |
//!     [pose_w2c]
//!          |
//!   refine with local map
//!          |
//! Estimated { pose, inliers, matches }
//! ```

use std::collections::HashSet;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pnp::{LMRefineParams, refine_pose_lm};
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3AF32, Mat3F64, Vec2F32, Vec3AF32, Vec3F64};
use kornia_image::ImageSize;
use kornia_imgproc::features::hamming_distance;
use kornia_imgproc::features::{OrbMatchConfig, match_orb_descriptors};

use crate::frame::Frame;
use crate::map::{Map, MapPoint};

use super::Estimate;


/// Result of matching map points against a frame.
struct FrameMatchResult {
    pub matches: Vec<(usize, usize)>,
    pub keypoints_undist: Vec<[f32; 2]>,
    pub grid: KeypointGrid,
}

/// Result of a PnP tracking attempt.
enum TrackAttempt {
    Success { pose: Pose3d, inliers: usize },
    Rejected(MapProjectionRejectReason),
}

/// A 2D grid that bins keypoints by their image position for O(1) spatial queries.
struct KeypointGrid {
    cells: Vec<Vec<usize>>,
    cell_w: f32,
    cell_h: f32,
    n_cols: usize,
    n_rows: usize,
    img_w: f32,
    img_h: f32,
}

impl KeypointGrid {
    /// Builds a grid over `keypoints_xy` (each `[x, y]`) for an image of size `img_w x img_h`.
    ///
    /// Each cell is `cell_size x cell_size` pixels. Points outside the image are clamped.
    fn new(keypoints_xy: &[[f32; 2]], img_w: f32, img_h: f32, cell_size: f32) -> Self {
        let n_cols = (img_w / cell_size).ceil() as usize;
        let n_rows = (img_h / cell_size).ceil() as usize;
        let n_cells = n_cols * n_rows;
        let mut cells = vec![Vec::new(); n_cells];

        for (i, kp) in keypoints_xy.iter().enumerate() {
            let col = ((kp[0] / cell_size) as usize).min(n_cols - 1);
            let row = ((kp[1] / cell_size) as usize).min(n_rows - 1);
            cells[row * n_cols + col].push(i);
        }

        Self {
            cells,
            cell_w: cell_size,
            cell_h: cell_size,
            n_cols,
            n_rows,
            img_w,
            img_h,
        }
    }

    /// Returns indices of keypoints within `radius` pixels of `(x, y)`.
    fn query_radius(
        &self,
        x: f32,
        y: f32,
        radius: f32,
        keypoints_xy: &[[f32; 2]],
    ) -> Vec<usize> {
        let r_sq = radius * radius;

        let col_min = ((x - radius).max(0.0) / self.cell_w) as usize;
        let col_max =
            (((x + radius).min(self.img_w - 1.0) / self.cell_w) as usize).min(self.n_cols - 1);
        let row_min = ((y - radius).max(0.0) / self.cell_h) as usize;
        let row_max =
            (((y + radius).min(self.img_h - 1.0) / self.cell_h) as usize).min(self.n_rows - 1);

        let mut result = Vec::new();
        for r in row_min..=row_max {
            for c in col_min..=col_max {
                for &idx in &self.cells[r * self.n_cols + c] {
                    let kp = keypoints_xy[idx];
                    let dx = kp[0] - x;
                    let dy = kp[1] - y;
                    if dx * dx + dy * dy <= r_sq {
                        result.push(idx);
                    }
                }
            }
        }
        result
    }
}

/// Tunable parameters for projection-guided matching.
#[derive(Debug, Clone, Copy)]
pub struct ProjectionMatchConfig {
    /// Reject projected points with depth `<= min_depth`.
    pub min_depth: f64,
    /// Keypoint search radius around each projected pixel.
    pub search_radius: f32,
    /// Maximum Hamming distance to accept a descriptor match.
    pub max_hamming: u32,
}

impl Default for ProjectionMatchConfig {
    fn default() -> Self {
        Self {
            min_depth: 0.0,
            search_radius: 15.0,
            max_hamming: 50,
        }
    }
}

/// PnP pose-estimation thresholds.
#[derive(Debug, Clone)]
pub struct PnpConfig {
    /// Reprojection threshold (px) for filtering correspondences against the prior pose.
    pub prior_reproj_threshold_px: f64,
    /// Reprojection threshold (px) for counting final inliers after LM refinement.
    pub final_reproj_threshold_px: f64,
    /// Minimum number of 3D-2D correspondences required for PnP solving.
    pub min_correspondences: usize,
    /// Minimum inliers to accept a PnP solution.
    pub min_inliers: usize,
}

impl Default for PnpConfig {
    fn default() -> Self {
        Self {
            prior_reproj_threshold_px: 25.0,
            final_reproj_threshold_px: 3.0,
            min_correspondences: 4,
            min_inliers: 30,
        }
    }
}

/// Map-projection tracking thresholds.
#[derive(Debug, Clone)]
pub struct MapProjectionConfig {
    /// ORB descriptor matcher settings for tracking against reference observations.
    pub match_config: OrbMatchConfig,
    /// PnP pose-estimation thresholds.
    pub pnp: PnpConfig,
    /// Projection matching config for initial tracking.
    pub projection: ProjectionMatchConfig,
    /// Projection matching config for local-map refinement (wider search).
    pub local_projection: ProjectionMatchConfig,
}

impl Default for MapProjectionConfig {
    fn default() -> Self {
        Self {
            match_config: OrbMatchConfig {
                nn_ratio: 0.6,
                th_low: 50,
                check_orientation: true,
                histo_length: 30,
            },
            pnp: PnpConfig::default(),
            projection: ProjectionMatchConfig::default(),
            local_projection: ProjectionMatchConfig {
                search_radius: 30.0,
                max_hamming: 60,
                ..ProjectionMatchConfig::default()
            },
        }
    }
}

/// Rejection reasons specific to the map-projection tracker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MapProjectionRejectReason {
    /// Too few projection-guided matches were available to run the first PnP stage.
    LowProjectionMatches,
    /// PnP solving failed for the provided correspondences.
    PnpFailed,
    /// PnP solved, but too few final inliers remained after validation.
    LowPnpInliers,
    /// Too few descriptor matches were found against the current reference keyframe.
    LowReferenceMatches,
    /// Too few valid 3D-2D correspondences could be built from reference matches.
    LowReferenceCorrespondences,
}

/// Map-projection-based pose estimator.
///
/// Estimates the camera pose by projecting map points into the current frame,
/// matching via ORB descriptors, and solving PnP.
pub struct MapProjectionEstimator {
    camera: PinholeCamera,
    config: MapProjectionConfig,
}

impl MapProjectionEstimator {
    pub fn new(camera: PinholeCamera, config: MapProjectionConfig) -> Self {
        Self { camera, config }
    }

    pub fn camera(&self) -> &PinholeCamera {
        &self.camera
    }

    pub fn config(&self) -> &MapProjectionConfig {
        &self.config
    }

    /// Estimate the pose of `frame` against the map.
    pub fn estimate_pose(
        &self,
        frame: &Frame,
        candidate_pose: &Pose3d,
        pose_before_tracking: &Pose3d,
        map: &Map,
        current_keyframe_idx: Option<usize>,
    ) -> Result<Estimate, MapProjectionRejectReason> {
        const MIN_PNP_CORRESPONDENCES: usize = 4;
        const MIN_STAGE1_INLIERS: usize = 10;

        let frame_match = self.match_map_to_frame(map.map_points(), frame, candidate_pose);
        let curr_keypoints_undist = &frame_match.keypoints_undist;
        let grid = &frame_match.grid;
        let projection_matches = frame_match.matches;

        // Shared logic: try_track → refine_pose → Estimate, or propagate rejection.
        let try_track_and_refine = |correspondences: Vec<(usize, usize)>,
                                    pose_init: &Pose3d|
         -> Result<Estimate, MapProjectionRejectReason> {
            match self.try_track(
                map.map_points(),
                &correspondences,
                curr_keypoints_undist,
                pose_init,
                MIN_STAGE1_INLIERS,
            ) {
                TrackAttempt::Success { pose, inliers } => {
                    let (pose, inliers, matches) = self.refine_pose(
                        curr_keypoints_undist,
                        &frame.features.descriptors,
                        grid,
                        frame.image_size,
                        pose,
                        inliers,
                        correspondences,
                        map,
                        current_keyframe_idx,
                    );
                    Ok(Estimate {
                        pose,
                        inliers,
                        matches,
                    })
                }
                TrackAttempt::Rejected(reason) => Err(reason),
            }
        };

        // PnP from projection matches.
        let last_reject = if projection_matches.len() >= MIN_PNP_CORRESPONDENCES {
            match try_track_and_refine(projection_matches, candidate_pose) {
                Ok(estimate) => return Ok(estimate),
                Err(reason) => reason,
            }
        } else {
            MapProjectionRejectReason::LowProjectionMatches
        };

        // Fallback: match against reference keyframe descriptors.
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
            return Err(MapProjectionRejectReason::LowReferenceMatches);
        }

        let mut ref_correspondences = Vec::with_capacity(ref_matches.len());
        for (kf_desc_idx, curr_idx) in ref_matches {
            if let Some(Some(mp_idx)) = current_kf.map_point_by_desc_idx.get(kf_desc_idx)
                && *mp_idx < map.map_points().len()
            {
                ref_correspondences.push((*mp_idx, curr_idx));
            }
        }

        if ref_correspondences.len() < MIN_PNP_CORRESPONDENCES {
            return Err(MapProjectionRejectReason::LowReferenceCorrespondences);
        }

        try_track_and_refine(ref_correspondences, pose_before_tracking)
    }

    /// Returns indices of map points visible in the current camera frustum.
    pub fn visible_map_points(
        &self,
        map: &Map,
        pose_world_to_cam: &Pose3d,
        image_size: ImageSize,
    ) -> HashSet<usize> {
        let mut visible = HashSet::new();
        for (mp_idx, mp) in map.map_points().iter().enumerate() {
            if mp.culled {
                continue;
            }
            let p_cam = pose_world_to_cam.transform_point(&mp.position);
            if self
                .camera
                .project_to_image(&p_cam, 0.0, image_size)
                .is_ok()
            {
                visible.insert(mp_idx);
            }
        }
        visible
    }

    #[allow(clippy::too_many_arguments)]
    fn refine_pose(
        &self,
        curr_keypoints_undist: &[[f32; 2]],
        curr_descriptors: &[[u8; 32]],
        grid: &KeypointGrid,
        image_size: ImageSize,
        mut pose: Pose3d,
        mut inliers: usize,
        mut matches: Vec<(usize, usize)>,
        map: &Map,
        current_keyframe_idx: Option<usize>,
    ) -> (Pose3d, usize, Vec<(usize, usize)>) {
        if let Some(local) = self.refine_with_local_map(
            map,
            current_keyframe_idx,
            &matches,
            curr_keypoints_undist,
            curr_descriptors,
            grid,
            image_size,
            &pose,
        ) {
            matches = local.matches;
            if local.inliers >= self.config.pnp.min_inliers {
                pose = local.pose;
                inliers = local.inliers;
            }
        }

        (pose, inliers, matches)
    }

    fn try_track(
        &self,
        map_points: &[MapPoint],
        correspondences: &[(usize, usize)],
        keypoints_undist: &[[f32; 2]],
        pose_init: &Pose3d,
        min_inliers: usize,
    ) -> TrackAttempt {
        match self.solve_pnp(map_points, correspondences, keypoints_undist, pose_init) {
            Some((new_pose, inliers)) => {
                if inliers >= min_inliers {
                    TrackAttempt::Success {
                        pose: new_pose,
                        inliers,
                    }
                } else {
                    TrackAttempt::Rejected(MapProjectionRejectReason::LowPnpInliers)
                }
            }
            None => TrackAttempt::Rejected(MapProjectionRejectReason::PnpFailed),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn refine_with_local_map(
        &self,
        map: &Map,
        current_kf_idx: Option<usize>,
        tracked_matches: &[(usize, usize)],
        curr_keypoints_undist: &[[f32; 2]],
        curr_descriptors: &[[u8; 32]],
        grid: &KeypointGrid,
        image_size: ImageSize,
        pose_init: &Pose3d,
    ) -> Option<Estimate> {
        let current_kf = current_kf_idx.and_then(|ki| map.get_keyframe(ki));
        let (local_map_points, local_to_global) =
            map.build_local_map_points(tracked_matches, current_kf);
        if local_map_points.len() < 4 {
            return None;
        }

        let local_matches = self.match_by_projection(
            &local_map_points,
            curr_keypoints_undist,
            curr_descriptors,
            grid,
            pose_init,
            image_size,
            self.config.local_projection,
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

        let (new_pose, inliers) = self.solve_pnp(
            map.map_points(),
            &global_matches,
            curr_keypoints_undist,
            pose_init,
        )?;
        Some(Estimate {
            pose: new_pose,
            inliers,
            matches: global_matches,
        })
    }

    /// Solve PnP from map-point to keypoint correspondences using LM refinement.
    fn solve_pnp(
        &self,
        map_points: &[MapPoint],
        correspondences: &[(usize, usize)],
        keypoints_undist: &[[f32; 2]],
        pose_init: &Pose3d,
    ) -> Option<(Pose3d, usize)> {
        let config = &self.config.pnp;
        let camera = &self.camera;

        let mut points_world = Vec::with_capacity(correspondences.len());
        let mut points_world_f64 = Vec::with_capacity(correspondences.len());
        let mut points_image = Vec::with_capacity(correspondences.len());
        for &(mp_idx, kp_idx) in correspondences {
            if let (Some(mp), Some(&kp)) = (map_points.get(mp_idx), keypoints_undist.get(kp_idx)) {
                points_world.push(Vec3AF32::new(
                    mp.position.x as f32,
                    mp.position.y as f32,
                    mp.position.z as f32,
                ));
                points_world_f64.push(mp.position);
                points_image.push(Vec2F32::new(kp[0], kp[1]));
            }
        }
        if points_world.len() < config.min_correspondences {
            return None;
        }

        let prior_th2 = config.prior_reproj_threshold_px * config.prior_reproj_threshold_px;
        let mut prior_inliers = Vec::new();
        for (i, pw) in points_world_f64.iter().enumerate() {
            if camera
                .reprojection_error_sq_world(
                    pose_init,
                    pw,
                    points_image[i].x as f64,
                    points_image[i].y as f64,
                )
                .is_some_and(|err_sq| err_sq <= prior_th2)
            {
                prior_inliers.push(i);
            }
        }
        if prior_inliers.len() < config.min_correspondences {
            return None;
        }

        let k = Mat3AF32::from_cols(
            Vec3AF32::new(camera.fx as f32, 0.0, 0.0),
            Vec3AF32::new(0.0, camera.fy as f32, 0.0),
            Vec3AF32::new(camera.cx as f32, camera.cy as f32, 1.0),
        );
        let mut world_inliers = Vec::with_capacity(prior_inliers.len());
        let mut image_inliers = Vec::with_capacity(prior_inliers.len());
        for &i in &prior_inliers {
            world_inliers.push(points_world[i]);
            image_inliers.push(points_image[i]);
        }

        let r_init_f32 = Mat3AF32::from_cols(
            Vec3AF32::new(
                pose_init.rotation.col(0).x as f32,
                pose_init.rotation.col(0).y as f32,
                pose_init.rotation.col(0).z as f32,
            ),
            Vec3AF32::new(
                pose_init.rotation.col(1).x as f32,
                pose_init.rotation.col(1).y as f32,
                pose_init.rotation.col(1).z as f32,
            ),
            Vec3AF32::new(
                pose_init.rotation.col(2).x as f32,
                pose_init.rotation.col(2).y as f32,
                pose_init.rotation.col(2).z as f32,
            ),
        );
        let t_init_f32 = Vec3AF32::new(
            pose_init.translation.x as f32,
            pose_init.translation.y as f32,
            pose_init.translation.z as f32,
        );

        let lm = refine_pose_lm(
            &world_inliers,
            &image_inliers,
            &k,
            &r_init_f32,
            &t_init_f32,
            None,
            &LMRefineParams::default(),
        )
        .ok()?;

        let r = &lm.rotation;
        let t = lm.translation;
        let r_new = Mat3F64::from_cols(
            Vec3F64::new(r.col(0).x as f64, r.col(0).y as f64, r.col(0).z as f64),
            Vec3F64::new(r.col(1).x as f64, r.col(1).y as f64, r.col(1).z as f64),
            Vec3F64::new(r.col(2).x as f64, r.col(2).y as f64, r.col(2).z as f64),
        );
        let t_new = Vec3F64::new(t.x as f64, t.y as f64, t.z as f64);
        let pose_new = Pose3d::new(r_new, t_new);
        let final_inliers = self.count_reprojection_inliers(
            &pose_new,
            &points_world_f64,
            &points_image,
            config.final_reproj_threshold_px,
        );
        Some((pose_new, final_inliers))
    }

    /// Undistorts keypoints, builds a spatial grid, and runs projection matching
    /// with narrow-to-wide fallback.
    fn match_map_to_frame(
        &self,
        map_points: &[MapPoint],
        frame: &Frame,
        pose: &Pose3d,
    ) -> FrameMatchResult {
        const KEYPOINT_GRID_CELL_SIZE: f32 = 64.0;
        const MIN_MATCHES_BEFORE_WIDE: usize = 20;

        let keypoints_undist: Vec<[f32; 2]> = frame
            .features
            .keypoints_xy
            .iter()
            .map(|kp| {
                let p = self.camera.undistort(kp[0] as f64, kp[1] as f64);
                [p.x as f32, p.y as f32]
            })
            .collect();

        let grid = KeypointGrid::new(
            &keypoints_undist,
            frame.image_size.width as f32,
            frame.image_size.height as f32,
            KEYPOINT_GRID_CELL_SIZE,
        );

        let config = self.config.projection;
        let mut matches = self.match_by_projection(
            map_points,
            &keypoints_undist,
            &frame.features.descriptors,
            &grid,
            pose,
            frame.image_size,
            config,
        );

        if matches.len() < MIN_MATCHES_BEFORE_WIDE {
            matches = self.match_by_projection(
                map_points,
                &keypoints_undist,
                &frame.features.descriptors,
                &grid,
                pose,
                frame.image_size,
                ProjectionMatchConfig {
                    search_radius: config.search_radius * 2.0,
                    ..config
                },
            );
        }

        FrameMatchResult {
            matches,
            keypoints_undist,
            grid,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn match_by_projection(
        &self,
        map_points: &[MapPoint],
        keypoints_xy: &[[f32; 2]],
        descriptors: &[[u8; 32]],
        grid: &KeypointGrid,
        pose_world_to_cam: &Pose3d,
        image_size: ImageSize,
        config: ProjectionMatchConfig,
    ) -> Vec<(usize, usize)> {
        let mut matched_kp = vec![false; keypoints_xy.len()];
        let mut matches = Vec::new();

        for (mp_idx, mp) in map_points.iter().enumerate() {
            if mp.culled {
                continue;
            }

            let p_cam = pose_world_to_cam.transform_point(&mp.position);
            let Ok(pixel) = self.camera.project_to_image(&p_cam, config.min_depth, image_size)
            else {
                continue;
            };
            let u = pixel.x as f32;
            let v = pixel.y as f32;

            let candidates = grid.query_radius(u, v, config.search_radius, keypoints_xy);
            let mut best_dist = u32::MAX;
            let mut best_kp = usize::MAX;

            for kp_idx in candidates {
                if matched_kp[kp_idx] {
                    continue;
                }
                let dist = hamming_distance(&mp.descriptor, &descriptors[kp_idx]);
                if dist < best_dist {
                    best_dist = dist;
                    best_kp = kp_idx;
                }
            }

            if best_dist <= config.max_hamming && best_kp != usize::MAX {
                matched_kp[best_kp] = true;
                matches.push((mp_idx, best_kp));
            }
        }

        matches
    }

    /// Count map-point reprojection inliers given a camera pose.
    fn count_reprojection_inliers(
        &self,
        pose_world_to_cam: &Pose3d,
        points_world: &[Vec3F64],
        points_image: &[Vec2F32],
        threshold_px: f64,
    ) -> usize {
        let th2 = threshold_px * threshold_px;
        let mut inliers = 0usize;
        for (pw, pi) in points_world.iter().zip(points_image.iter()) {
            if self
                .camera
                .reprojection_error_sq_world(pose_world_to_cam, pw, pi.x as f64, pi.y as f64)
                .is_some_and(|err_sq| err_sq <= th2)
            {
                inliers += 1;
            }
        }
        inliers
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod estimator_tests {
    use crate::system::KeyframePolicy;

    #[test]
    fn test_need_new_keyframe_forced_by_gap() {
        let policy = KeyframePolicy::default();
        assert!(policy.should_insert(100, Some(90), 50, 100));
    }

    #[test]
    fn test_need_new_keyframe_too_soon() {
        let policy = KeyframePolicy::default();
        assert!(!policy.should_insert(2, Some(1), 50, 100));
    }
}

#[cfg(test)]
mod matching_tests {
    use super::*;

    #[test]
    fn test_empty_grid() {
        let grid = KeypointGrid::new(&[], 640.0, 480.0, 64.0);
        let result = grid.query_radius(320.0, 240.0, 50.0, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_point_found() {
        let kps = [[100.0, 100.0]];
        let grid = KeypointGrid::new(&kps, 640.0, 480.0, 64.0);

        let result = grid.query_radius(100.0, 100.0, 10.0, &kps);
        assert_eq!(result, vec![0]);

        let result = grid.query_radius(500.0, 400.0, 10.0, &kps);
        assert!(result.is_empty());
    }

    #[test]
    fn test_radius_boundary() {
        let kps = [[50.0, 50.0], [60.0, 50.0], [100.0, 50.0]];
        let grid = KeypointGrid::new(&kps, 640.0, 480.0, 32.0);

        let mut result = grid.query_radius(55.0, 50.0, 15.0, &kps);
        result.sort();
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn test_corner_clamping() {
        let kps = [[0.0, 0.0], [639.0, 479.0]];
        let grid = KeypointGrid::new(&kps, 640.0, 480.0, 64.0);

        let result = grid.query_radius(0.0, 0.0, 5.0, &kps);
        assert_eq!(result, vec![0]);

        let result = grid.query_radius(639.0, 479.0, 5.0, &kps);
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_multiple_points_per_cell() {
        let kps = [[10.0, 10.0], [12.0, 11.0], [15.0, 13.0]];
        let grid = KeypointGrid::new(&kps, 640.0, 480.0, 64.0);

        let mut result = grid.query_radius(12.0, 12.0, 20.0, &kps);
        result.sort();
        assert_eq!(result, vec![0, 1, 2]);
    }

    fn make_test_estimator(camera: PinholeCamera) -> MapProjectionEstimator {
        MapProjectionEstimator::new(camera, MapProjectionConfig::default())
    }

    fn test_camera() -> PinholeCamera {
        PinholeCamera {
            fx: 200.0,
            fy: 200.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    #[test]
    fn test_match_by_projection_simple() {
        let estimator = make_test_estimator(test_camera());

        let desc_a = [0u8; 32];
        let map_points = vec![MapPoint {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            descriptor: desc_a,
            keyframe_idx: 0,
            n_visible: 0,
            n_found: 0,
            culled: false,
        }];

        let keypoints_xy = vec![[320.0f32, 240.0], [100.0, 100.0]];
        let descriptors = vec![desc_a, [0xFF; 32]];

        let grid = KeypointGrid::new(&keypoints_xy, 640.0, 480.0, 64.0);
        let pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO);

        let matches = estimator.match_by_projection(
            &map_points,
            &keypoints_xy,
            &descriptors,
            &grid,
            &pose,
            ImageSize {
                width: 640,
                height: 480,
            },
            ProjectionMatchConfig {
                min_depth: 0.0,
                search_radius: 15.0,
                max_hamming: 50,
            },
        );

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0], (0, 0));
    }

    #[test]
    fn test_behind_camera_rejected() {
        let estimator = make_test_estimator(test_camera());

        let map_points = vec![MapPoint {
            position: Vec3F64::new(0.0, 0.0, -5.0),
            descriptor: [0u8; 32],
            keyframe_idx: 0,
            n_visible: 0,
            n_found: 0,
            culled: false,
        }];

        let keypoints_xy = vec![[320.0f32, 240.0]];
        let descriptors = vec![[0u8; 32]];
        let grid = KeypointGrid::new(&keypoints_xy, 640.0, 480.0, 64.0);

        let matches = estimator.match_by_projection(
            &map_points,
            &keypoints_xy,
            &descriptors,
            &grid,
            &Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO),
            ImageSize {
                width: 640,
                height: 480,
            },
            ProjectionMatchConfig {
                min_depth: 0.0,
                search_radius: 15.0,
                max_hamming: 50,
            },
        );

        assert!(matches.is_empty());
    }

    #[test]
    fn test_high_hamming_rejected() {
        let estimator = make_test_estimator(test_camera());

        let map_points = vec![MapPoint {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            descriptor: [0u8; 32],
            keyframe_idx: 0,
            n_visible: 0,
            n_found: 0,
            culled: false,
        }];

        let keypoints_xy = vec![[320.0f32, 240.0]];
        let descriptors = vec![[0xFF; 32]];
        let grid = KeypointGrid::new(&keypoints_xy, 640.0, 480.0, 64.0);

        let matches = estimator.match_by_projection(
            &map_points,
            &keypoints_xy,
            &descriptors,
            &grid,
            &Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO),
            ImageSize {
                width: 640,
                height: 480,
            },
            ProjectionMatchConfig {
                min_depth: 0.0,
                search_radius: 15.0,
                max_hamming: 50,
            },
        );

        assert!(matches.is_empty());
    }
}
