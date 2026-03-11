//! ORB-SLAM pipeline: orchestrates tracking, mapping, and state transitions.
//!
//! This is an example-specific orchestrator that wires together the building
//! blocks from kornia-slam into a concrete ORB-based SLAM pipeline.

use crate::bootstrap::run_bootstrap;
use crate::tracking::run_tracking_step;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_slam::estimation::MapProjectionEstimator;
use kornia_slam::estimation::map_projection::MapProjectionConfig;
use kornia_slam::estimation::two_view::TwoViewInitConfig;
use kornia_slam::mapping::{Map, MapPoint};
use kornia_slam::odometry::{OdometryMode, OdometryResult, OdometryState, OdometryStatus};
use kornia_slam::{Frame, OrbFeatures};

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

    fn bootstrap(&mut self, curr_frame: Frame) -> OdometryResult {
        run_bootstrap(
            &mut self.state,
            &mut self.map,
            self.estimator.camera(),
            &self.two_view_init_config,
            curr_frame,
        )
    }

    fn track(&mut self, frame: Frame) -> OdometryResult {
        run_tracking_step(
            &mut self.state,
            &mut self.map,
            &self.estimator,
            &self.two_view_init_config,
            frame,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_3d::camera::ImageSize;

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
    fn process_frame_stores_first_bootstrap_frame_and_skips() {
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
        let mut system = ORBSLAMPipeline::new(
            camera,
            TwoViewInitConfig::default(),
            MapProjectionConfig::default(),
        );

        let result = system.process_frame(test_frame(0, vec![[10.0, 10.0]], vec![[1u8; 32]]));

        assert_eq!(result.status, OdometryStatus::Skipped);
        assert!(system.state.bootstrap_frame.is_some());
        assert_eq!(system.state.state, OdometryMode::Bootstrap);
    }

    #[test]
    fn process_frame_uses_tracking_path_when_tracking_mode_is_active() {
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
        let mut system = ORBSLAMPipeline::new(
            camera,
            TwoViewInitConfig::default(),
            MapProjectionConfig::default(),
        );
        system.state.state = OdometryMode::Tracking;

        let result = system.process_frame(test_frame(0, vec![[10.0, 10.0]], vec![[1u8; 32]]));

        assert_eq!(result.status, OdometryStatus::Skipped);
        assert_eq!(system.state.consecutive_failures, 1);
        assert_eq!(system.state.state, OdometryMode::Tracking);
    }
}
