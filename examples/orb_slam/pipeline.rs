//! ORB-SLAM pipeline: orchestrates tracking, mapping, and state transitions.
//!
//! This is an example-specific orchestrator that wires together the building
//! blocks from kornia-slam into a concrete ORB-based SLAM pipeline.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;

use kornia_slam::mapping::ba::run_local_ba;
use kornia_slam::mapping::{
    build_initial_map, cull_map_points, grow_map_points_from_keyframe_pair, Keyframe, Map, MapPoint,
};
use kornia_slam::odometry::bootstrap::{BootstrapConfig, BootstrapOutcome, try_bootstrap};
use kornia_slam::odometry::estimation::MapProjectionEstimator;
use kornia_slam::odometry::estimation::map_projection::OdometryConfig;
use kornia_slam::odometry::{OdometryMode, OdometryResult, OdometryState, OdometryStatus};
use kornia_slam::Frame;

/// Top-level ORB-SLAM pipeline: orchestrates tracking, mapping, and state transitions.
pub struct ORBSLAMPipeline {
    estimator: MapProjectionEstimator,
    bootstrap_config: BootstrapConfig,
    map: Map,
    state: OdometryState,
}

impl ORBSLAMPipeline {
    /// Creates a new pipeline with identity pose.
    pub fn new(
        camera: PinholeCamera,
        bootstrap_config: BootstrapConfig,
        odometry_config: OdometryConfig,
    ) -> Self {
        Self {
            estimator: MapProjectionEstimator::new(camera, odometry_config),
            bootstrap_config,
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

        let outcome = try_bootstrap(
            &prev_bootstrap_frame.features,
            &prev_bootstrap_frame.pose_world_to_cam,
            &curr_frame.features,
            &curr_frame.pose_world_to_cam,
            self.estimator.camera(),
            &self.bootstrap_config,
        );

        match outcome {
            BootstrapOutcome::Rejected { .. } => {
                self.state.bootstrap_frame = Some(prev_bootstrap_frame);
                OdometryResult {
                    pose_world_to_cam: self.state.pose_world_to_cam,
                    status: OdometryStatus::Skipped,
                }
            }
            BootstrapOutcome::Initialized {
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
                build_initial_map(
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
        let estimate = self
            .estimator
            .track(
                &frame,
                &candidate_pose,
                &pose_before_tracking,
                &self.map,
                self.state.current_keyframe_idx,
            )
            .ok();

        let mut status = match estimate {
            Some(pose) => {
                self.state.velocity = Some(Pose3d::between(&pose_before_tracking, &pose));
                self.state.pose_world_to_cam = pose;
                OdometryStatus::Tracked
            }
            None => OdometryStatus::Skipped,
        };

        // Phase 2: update observations and maybe insert keyframe.
        if status == OdometryStatus::Tracked {
            let matches = self.estimator.last_matches();
            self.estimator.update_map_point_observations(
                &mut self.map,
                matches,
                &candidate_pose,
                image_size,
            );

            if self.try_insert_keyframe(&frame) {
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

    fn try_insert_keyframe(&mut self, frame: &Frame) -> bool {
        let tracked_inliers = self.estimator.last_inliers();
        let matches = self.estimator.last_matches();

        let n_ref_map_points = self
            .state
            .current_keyframe_idx
            .and_then(|ki| self.map.get_keyframe(ki))
            .map(|kf| {
                kf.map_point_by_desc_idx
                    .iter()
                    .filter(|m| m.is_some())
                    .count()
            })
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
            self.bootstrap_config.match_config,
            &self.bootstrap_config.two_view_config,
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
