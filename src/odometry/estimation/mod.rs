//! Pose estimation: single-frame pose estimators and supporting algorithms.

use kornia_3d::pose::Pose3d;

use crate::frame::Frame;

pub mod map_projection;
pub mod matching;
pub mod pnp;

/// Single-frame pose estimator.
///
/// Each implementation consumes sensor data (via the frame)
/// and produces a pose estimate. Different estimators can run different
/// algorithms (feature matching, photometric, ICP, IMU, etc.) and
/// be fused together by the odometry layer.
pub trait Estimator {
    /// Estimate the camera pose for the given frame.
    /// Returns `None` if estimation failed.
    fn estimate(
        &mut self,
        frame: &Frame,
        predicted_pose: &Pose3d,
    ) -> Option<Pose3d>;
}

pub use map_projection::MapProjectionEstimator;
