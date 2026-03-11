//! ORB-SLAM pipeline: orchestrates tracking, mapping, and state transitions.
//!
//! This is an example-specific orchestrator that wires together the building
//! blocks from kornia-slam into a concrete ORB-based SLAM pipeline.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::pose::{
    TwoViewConfig, TwoViewModel, triangulate_midpoint_known_pose, two_view_estimate,
};
use kornia_algebra::Vec3F64;
use kornia_imgproc::features::{OrbMatchConfig, match_orb_descriptors};

use kornia_slam::mapping::ba::run_local_ba;
use kornia_slam::mapping::{
    cull_map_points, Keyframe, Map, MapPoint,
};
use kornia_slam::estimation::MapProjectionEstimator;
use kornia_slam::estimation::map_projection::{
    MapProjectionConfig, MapProjectionEstimateOutcome,
};
use kornia_slam::estimation::two_view::{
    TwoViewInitConfig, TwoViewInitOutcome, try_initialize_two_view,
};
use kornia_slam::odometry::{OdometryMode, OdometryResult, OdometryState, OdometryStatus};
use kornia_slam::{Frame, OrbFeatures};

fn materialize_bootstrap_map(
    map: &mut Map,
    reference_frame: Frame,
    current_frame: Frame,
    matches: &[(usize, usize)],
    points3d: &[Vec3F64],
    inlier_indices: &[usize],
    median_depth: Option<f64>,
) -> usize {
    let mut reference_kf = Keyframe::from_frame(reference_frame);
    let mut current_kf = Keyframe::from_frame(current_frame);
    let depth_scale = median_depth.filter(|&d| d > 1e-6).unwrap_or(1.0);
    let reference_pose_inv = reference_kf.frame.pose_world_to_cam.inverse();
    let mut added = 0usize;

    for (p_cam, &match_idx) in points3d.iter().zip(inlier_indices.iter()) {
        let Some(&(reference_desc_idx, current_desc_idx)) = matches.get(match_idx) else {
            continue;
        };
        if reference_desc_idx >= reference_kf.map_point_by_desc_idx.len()
            || current_desc_idx >= current_kf.map_point_by_desc_idx.len()
        {
            continue;
        }

        let descriptor = current_kf
            .frame
            .features
            .descriptors
            .get(current_desc_idx)
            .copied()
            .or_else(|| {
                reference_kf
                    .frame
                    .features
                    .descriptors
                    .get(reference_desc_idx)
                    .copied()
            });
        let Some(descriptor) = descriptor else {
            continue;
        };

        let p_world = reference_pose_inv.transform_point(&(*p_cam / depth_scale));
        let mp_idx = map.push_map_point(MapPoint::new(
            p_world,
            descriptor,
            reference_kf.frame.idx,
        ));
        reference_kf.associate_map_point(reference_desc_idx, mp_idx);
        current_kf.associate_map_point(current_desc_idx, mp_idx);
        added += 1;
    }

    map.upsert_keyframe(reference_kf);
    map.upsert_keyframe(current_kf);

    added
}

fn grow_map_points_from_keyframe_pair(
    map: &mut Map,
    camera: &PinholeCamera,
    curr_kf_idx: usize,
    prev_kf: &Keyframe,
    curr_features: &OrbFeatures,
    curr_kf_map_assoc: &mut [Option<usize>],
    pose_world_to_cam: &Pose3d,
    match_config: OrbMatchConfig,
    two_view_config: &TwoViewConfig,
    min_parallax_deg: f64,
) -> usize {
    const MIN_GROWTH_MATCHES: usize = 20;
    const MIN_GROWTH_INLIERS: usize = 15;
    const REPROJ_THRESHOLD_PX: f64 = 3.0;
    const MAX_TRIANGULATION_GAP: f64 = 0.25;

    let matches = match_orb_descriptors(
        &prev_kf.frame.features.orientations,
        &prev_kf.frame.features.descriptors,
        &curr_features.orientations,
        &curr_features.descriptors,
        match_config,
    );
    if matches.len() < MIN_GROWTH_MATCHES {
        return 0;
    }

    let mut pair_indices: Vec<(usize, usize)> = Vec::with_capacity(matches.len());
    for (prev_idx, curr_idx) in matches {
        if prev_idx >= prev_kf.frame.features.keypoints_xy.len()
            || curr_idx >= curr_features.keypoints_xy.len()
        {
            continue;
        }
        if curr_kf_map_assoc.get(curr_idx).is_some_and(|m| m.is_some()) {
            continue;
        }
        if prev_kf.map_point(prev_idx).is_some() {
            continue;
        }
        pair_indices.push((prev_idx, curr_idx));
    }

    let (prev_pts, curr_pts) = camera.undistort_matched_pairs(
        &prev_kf.frame.features.keypoints_xy,
        &curr_features.keypoints_xy,
        &pair_indices,
    );
    if pair_indices.len() < 8 {
        return 0;
    }

    let k = camera.intrinsic_matrix();
    let two_view = match two_view_estimate(&prev_pts, &curr_pts, &k, &k, two_view_config) {
        Ok(tv) if matches!(tv.model, TwoViewModel::Fundamental(_)) => tv,
        _ => return 0,
    };
    if two_view.inlier_indices.len() < MIN_GROWTH_INLIERS {
        return 0;
    }

    let prev_pose = prev_kf.frame.pose_world_to_cam;
    let curr_pose = *pose_world_to_cam;
    let relative_pose = Pose3d::between(&prev_pose, &curr_pose);
    let r_rel = relative_pose.rotation;
    let t_rel = relative_pose.translation;
    if t_rel.length() <= 1e-8 {
        return 0;
    }

    let prev_pose_inv = prev_pose.inverse();
    let reproj_th2 = REPROJ_THRESHOLD_PX * REPROJ_THRESHOLD_PX;
    let mut n_added = 0usize;

    for &inlier_idx in &two_view.inlier_indices {
        let Some(&(_prev_idx, curr_idx)) = pair_indices.get(inlier_idx) else {
            continue;
        };
        if curr_kf_map_assoc.get(curr_idx).is_some_and(|m| m.is_some()) {
            continue;
        }

        let p_prev = prev_pts[inlier_idx];
        let p_curr = curr_pts[inlier_idx];

        let ray_prev = Vec3F64::new(
            (p_prev.x - camera.cx) / camera.fx,
            (p_prev.y - camera.cy) / camera.fy,
            1.0,
        )
        .normalize();
        let ray_curr = Vec3F64::new(
            (p_curr.x - camera.cx) / camera.fx,
            (p_curr.y - camera.cy) / camera.fy,
            1.0,
        )
        .normalize();

        let Some((p_cam_prev, triang_gap)) =
            triangulate_midpoint_known_pose(&ray_prev, &ray_curr, &r_rel, &t_rel)
        else {
            continue;
        };
        if triang_gap > MAX_TRIANGULATION_GAP {
            continue;
        }

        if p_cam_prev.z <= 1e-6 {
            continue;
        }
        let p_cam_curr = r_rel * p_cam_prev + t_rel;
        if p_cam_curr.z <= 1e-6 {
            continue;
        }

        let c2 = -(r_rel.transpose() * t_rel);
        let d1 = p_cam_prev.normalize();
        let d2_vec = p_cam_prev - c2;
        if d2_vec.length() <= 1e-12 {
            continue;
        }
        let d2 = d2_vec.normalize();
        let parallax_deg_val = d1.dot(d2).clamp(-1.0, 1.0).acos().to_degrees();
        if parallax_deg_val < min_parallax_deg {
            continue;
        }

        let Some(err_prev) = camera.reprojection_error_sq_cam(&p_cam_prev, p_prev.x, p_prev.y)
        else {
            continue;
        };
        if err_prev > reproj_th2 {
            continue;
        }
        let Some(err_curr) = camera.reprojection_error_sq_cam(&p_cam_curr, p_curr.x, p_curr.y)
        else {
            continue;
        };
        if err_curr > reproj_th2 {
            continue;
        }

        let p_world = prev_pose_inv.transform_point(&p_cam_prev);
        let mp_idx = map.push_map_point(MapPoint::new(
            p_world,
            curr_features.descriptors[curr_idx],
            curr_kf_idx,
        ));

        if let Some(slot) = curr_kf_map_assoc.get_mut(curr_idx) {
            *slot = Some(mp_idx);
            n_added += 1;
        }
    }

    n_added
}

/// Top-level ORB-SLAM pipeline: orchestrates tracking, mapping, and state transitions.
pub struct ORBSLAMPipeline {
    estimator: MapProjectionEstimator,
    two_view_init_config: TwoViewInitConfig,
    map: Map,
    state: OdometryState,
}

impl ORBSLAMPipeline {
    /// Creates a new pipeline with identity pose.
    pub fn new(
        camera: PinholeCamera,
        two_view_init_config: TwoViewInitConfig,
        map_projection_config: MapProjectionConfig,
    ) -> Self {
        Self {
            estimator: MapProjectionEstimator::new(camera, map_projection_config),
            two_view_init_config,
            map: Map::new(),
            state: OdometryState::new(),
        }
    }

    /// Processes one frame (pre-extracted features) and returns the tracking result.
    pub fn process_frame(&mut self, frame: Frame) -> OdometryResult {
        match self.state.state {
            OdometryMode::Bootstrap => self.bootstrap(frame),
            OdometryMode::Tracking => self.track(frame),
        }
    }

    /// Returns a reference to the map.
    pub fn map(&self) -> &Map {
        &self.map
    }

    /// Returns all persistent map points.
    pub fn map_points(&self) -> &[MapPoint] {
        self.map.map_points()
    }

    /// Returns the index of the current reference keyframe, if tracking has one.
    pub fn current_keyframe_idx(&self) -> Option<usize> {
        self.state
            .current_keyframe_idx
            .and_then(|ki| self.map.get_keyframe(ki).map(|kf| kf.frame.idx))
    }

    /// Returns the total number of persistent map points.
    pub fn num_map_points(&self) -> usize {
        self.map.map_points().len()
    }

    /// Returns the current pose.
    pub fn pose(&self) -> &Pose3d {
        &self.state.pose_world_to_cam
    }

    fn bootstrap(&mut self, mut curr_frame: Frame) -> OdometryResult {
        // Stamp frames with current odometry pose so bootstrap builds
        // the new map in the existing coordinate frame (not at origin).
        curr_frame.pose_world_to_cam = self.state.pose_world_to_cam;

        let Some(prev_bootstrap_frame) = self.state.bootstrap_frame.take() else {
            // First bootstrap frame — store it and wait for a second frame.
            self.state.bootstrap_frame = Some(curr_frame);
            return OdometryResult {
                pose_world_to_cam: self.state.pose_world_to_cam,
                status: OdometryStatus::Skipped,
            };
        };

        let outcome = try_initialize_two_view(
            &prev_bootstrap_frame.features,
            &prev_bootstrap_frame.pose_world_to_cam,
            &curr_frame.features,
            &curr_frame.pose_world_to_cam,
            self.estimator.camera(),
            &self.two_view_init_config,
        );

        match outcome {
            TwoViewInitOutcome::Rejected { .. } => {
                self.state.bootstrap_frame = Some(prev_bootstrap_frame);
                OdometryResult {
                    pose_world_to_cam: self.state.pose_world_to_cam,
                    status: OdometryStatus::Skipped,
                }
            }
            TwoViewInitOutcome::Initialized {
                pose_world_to_cam,
                motion_increment,
                matches,
                points3d,
                inlier_indices,
                median_depth,
            } => {
                self.state.velocity = Some(motion_increment);
                self.state.pose_world_to_cam = pose_world_to_cam;
                curr_frame.pose_world_to_cam = self.state.pose_world_to_cam;

                let curr_idx = curr_frame.idx;
                materialize_bootstrap_map(
                    &mut self.map,
                    prev_bootstrap_frame,
                    curr_frame,
                    &matches,
                    &points3d,
                    &inlier_indices,
                    median_depth,
                );

                self.state.current_keyframe_idx = Some(curr_idx);
                self.state.last_keyframe_idx = Some(curr_idx);
                self.state.state = OdometryMode::Tracking;

                OdometryResult {
                    pose_world_to_cam: self.state.pose_world_to_cam,
                    status: OdometryStatus::KeyframeAccepted,
                }
            }
        }
    }

    fn track(&mut self, frame: Frame) -> OdometryResult {
        let pose_before_tracking = self.state.pose_world_to_cam;
        let image_size = frame.image_size;

        let candidate_pose = if let Some(vel) = self.state.velocity {
            vel.compose(&self.state.pose_world_to_cam)
        } else {
            self.state.pose_world_to_cam
        };

        // Phase 1: estimate pose.
        let estimate = self.estimator.estimate_pose(
            &frame,
            &candidate_pose,
            &pose_before_tracking,
            &self.map,
            self.state.current_keyframe_idx,
        );

        let (mut status, matches, tracked_inliers) = match estimate {
            MapProjectionEstimateOutcome::Estimated {
                pose_world_to_cam,
                inliers,
                matches,
            } => {
                self.state.velocity = Some(Pose3d::between(&pose_before_tracking, &pose_world_to_cam));
                self.state.pose_world_to_cam = pose_world_to_cam;
                (OdometryStatus::Tracked, matches, inliers)
            }
            MapProjectionEstimateOutcome::Rejected { .. } => {
                (OdometryStatus::Skipped, Vec::new(), 0)
            }
        };

        // Phase 2: update observations and maybe insert keyframe.
        if status == OdometryStatus::Tracked {
            self.estimator.update_map_point_observations(
                &mut self.map,
                &matches,
                &candidate_pose,
                image_size,
            );

            if self.try_insert_keyframe(&frame, tracked_inliers, &matches) {
                status = OdometryStatus::KeyframeAccepted;
            }
        }

        // Phase 3: handle tracking loss.
        if status == OdometryStatus::Skipped {
            self.state.consecutive_failures += 1;
            if self.state.consecutive_failures
                >= self.estimator.config().max_consecutive_failures
            {
                self.state.reset();
                return self.bootstrap(frame);
            }
        } else {
            self.state.consecutive_failures = 0;
        }

        OdometryResult {
            pose_world_to_cam: self.state.pose_world_to_cam,
            status,
        }
    }

    fn try_insert_keyframe(
        &mut self,
        frame: &Frame,
        tracked_inliers: usize,
        matches: &[(usize, usize)],
    ) -> bool {
        let n_ref_map_points = self
            .state
            .current_keyframe_idx
            .and_then(|ki| self.map.get_keyframe(ki))
            .map(|kf| kf.num_associated_points())
            .unwrap_or(0);

        if !self.estimator.need_new_keyframe(
            frame.idx,
            self.state.last_keyframe_idx,
            tracked_inliers,
            n_ref_map_points,
        ) {
            return false;
        }

        let Some(prev_kf) = self
            .state
            .current_keyframe_idx
            .and_then(|ki| self.map.get_keyframe(ki))
            .cloned()
        else {
            return false;
        };

        let mut curr_kf_map_assoc = vec![None; frame.features.descriptors.len()];
        for &(mp_idx, curr_idx) in matches {
            if let Some(slot) = curr_kf_map_assoc.get_mut(curr_idx) {
                *slot = Some(mp_idx);
            }
        }

        let config = self.estimator.config();
        grow_map_points_from_keyframe_pair(
            &mut self.map,
            self.estimator.camera(),
            frame.idx,
            &prev_kf,
            &frame.features,
            &mut curr_kf_map_assoc,
            &self.state.pose_world_to_cam,
            self.two_view_init_config.match_config,
            &self.two_view_init_config.estimation_config,
            config.min_parallax_deg,
        );

        let mut kf = Keyframe::from_frame(Frame::new(
            frame.idx,
            frame.features.clone(),
            self.state.pose_world_to_cam,
            frame.image_size,
        ));
        kf.map_point_by_desc_idx = curr_kf_map_assoc;
        self.map.upsert_keyframe(kf);
        self.state.current_keyframe_idx = Some(frame.idx);
        self.state.last_keyframe_idx = Some(frame.idx);

        if config.enable_local_ba {
            run_local_ba(&mut self.map, self.estimator.camera());

            if let Some(newest_kf) = self.map.keyframes().last() {
                self.state.pose_world_to_cam = newest_kf.frame.pose_world_to_cam;
            }
        }

        cull_map_points(&mut self.map);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_3d::camera::ImageSize;
    use kornia_algebra::Vec3F64;
    use kornia_imgproc::features::OrbMatchConfig;
    use kornia_imgproc::features::OrbFeatures;
    use kornia_3d::pose::TwoViewConfig;

    fn test_frame(idx: usize, keypoints_xy: Vec<[f32; 2]>, descriptors: Vec<[u8; 32]>) -> Frame {
        Frame::new(
            idx,
            OrbFeatures {
                orientations: vec![0.0; descriptors.len()],
                keypoints_xy,
                descriptors,
            },
            Pose3d::IDENTITY,
            ImageSize {
                width: 640.0,
                height: 480.0,
            },
        )
    }

    #[test]
    fn materialize_bootstrap_map_populates_map_and_keyframe_links() {
        let mut map = Map::new();
        let reference_frame = test_frame(0, vec![[100.0, 100.0]], vec![[7u8; 32]]);
        let current_frame = test_frame(1, vec![[101.0, 99.0]], vec![[9u8; 32]]);

        let added = materialize_bootstrap_map(
            &mut map,
            reference_frame,
            current_frame,
            &[(0usize, 0usize)],
            &[Vec3F64::new(0.0, 0.0, 2.0)],
            &[0],
            Some(2.0),
        );

        assert_eq!(added, 1);
        assert_eq!(map.map_points().len(), 1);
        assert_eq!(map.map_points()[0].position, Vec3F64::new(0.0, 0.0, 1.0));

        let kf0 = map.get_keyframe(0).expect("expected reference keyframe");
        let kf1 = map.get_keyframe(1).expect("expected current keyframe");
        assert_eq!(kf0.map_point_by_desc_idx[0], Some(0));
        assert_eq!(kf1.map_point_by_desc_idx[0], Some(0));
    }

    #[test]
    fn grow_map_points_from_keyframe_pair_returns_zero_when_too_few_matches() {
        let camera = PinholeCamera {
            fx: 458.654,
            fy: 457.296,
            cx: 367.215,
            cy: 248.375,
            k1: -0.28340811,
            k2: 0.07395907,
            p1: 0.00019359,
            p2: 1.76187114e-05,
        };
        let prev_kf = Keyframe::from_frame(test_frame(
            0,
            vec![[10.0, 10.0], [20.0, 20.0]],
            vec![[1u8; 32], [2u8; 32]],
        ));
        let curr_features = OrbFeatures {
            keypoints_xy: vec![[10.5, 10.5], [20.5, 20.5]],
            orientations: vec![0.0, 0.0],
            descriptors: vec![[1u8; 32], [2u8; 32]],
        };
        let mut map = Map::new();
        let mut curr_kf_map_assoc = vec![None; curr_features.descriptors.len()];

        let added = grow_map_points_from_keyframe_pair(
            &mut map,
            &camera,
            1,
            &prev_kf,
            &curr_features,
            &mut curr_kf_map_assoc,
            &Pose3d::IDENTITY,
            OrbMatchConfig::default(),
            &TwoViewConfig::default(),
            1.0,
        );

        assert_eq!(added, 0);
        assert_eq!(map.map_points().len(), 0);
        assert!(curr_kf_map_assoc.iter().all(|slot| slot.is_none()));
    }
}
