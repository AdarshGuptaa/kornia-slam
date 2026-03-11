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
use kornia_slam::odometry::{OdometryMode, OdometryResult, OdometryState};
use kornia_slam::Frame;

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
    use kornia_imgproc::features::OrbMatchConfig;
    use kornia_slam::OrbFeatures;
    use kornia_slam::odometry::OdometryStatus;

    fn test_camera() -> PinholeCamera {
        PinholeCamera {
            fx: 300.0,
            fy: 300.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

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

    fn synthetic_bootstrap_frames() -> (Frame, Frame) {
        let camera = test_camera();
        let world_points = [
            (-0.8, -0.4, 3.0),
            (-0.5, 0.3, 3.4),
            (-0.2, -0.2, 3.8),
            (0.1, 0.5, 4.2),
            (0.4, -0.3, 4.6),
            (0.7, 0.2, 3.7),
            (-0.6, 0.1, 4.4),
            (0.3, -0.5, 3.3),
            (0.9, 0.4, 4.8),
            (-0.1, 0.0, 5.1),
        ];
        let baseline = 0.4;

        let project = |x: f64, y: f64, z: f64| -> [f32; 2] {
            [
                (camera.fx * x / z + camera.cx) as f32,
                (camera.fy * y / z + camera.cy) as f32,
            ]
        };
        let descriptor = |idx: usize| -> [u8; 32] {
            let mut bits = [0u8; 32];
            bits[idx] = 0xFF;
            bits
        };

        let reference_keypoints = world_points
            .iter()
            .map(|&(x, y, z)| project(x, y, z))
            .collect();
        let current_keypoints = world_points
            .iter()
            .map(|&(x, y, z)| project(x - baseline, y, z))
            .collect();
        let descriptors: Vec<[u8; 32]> = (0..world_points.len()).map(descriptor).collect();

        (
            test_frame(0, reference_keypoints, descriptors.clone()),
            test_frame(1, current_keypoints, descriptors),
        )
    }

    fn permissive_two_view_init_config() -> TwoViewInitConfig {
        let mut config = TwoViewInitConfig::default();
        config.match_config = OrbMatchConfig {
            nn_ratio: 0.8,
            th_low: 255,
            check_orientation: false,
            histo_length: 30,
        };
        config.estimation_config.ransac_f.min_inliers = 8;
        config.estimation_config.ransac_h.min_inliers = 4;
        config.acceptance_config.min_matches = 8;
        config.acceptance_config.min_inliers = 8;
        config.acceptance_config.min_triangulated = 8;
        config.estimation_config.min_parallax_deg = 0.5;
        config
    }

    #[test]
    fn process_frame_stores_first_bootstrap_frame_and_skips() {
        let camera = test_camera();
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
        let camera = test_camera();
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

    #[test]
    fn process_frame_bootstraps_into_tracking_mode() {
        let (reference_frame, current_frame) = synthetic_bootstrap_frames();
        let mut system = ORBSLAMPipeline::new(
            test_camera(),
            permissive_two_view_init_config(),
            MapProjectionConfig::default(),
        );

        let first = system.process_frame(reference_frame);
        let second = system.process_frame(current_frame);

        assert_eq!(first.status, OdometryStatus::Skipped);
        assert_eq!(second.status, OdometryStatus::KeyframeAccepted);
        assert_eq!(system.state.state, OdometryMode::Tracking);
        assert_eq!(system.current_keyframe_idx(), Some(1));
        assert!(system.num_map_points() >= 8);
    }
}
