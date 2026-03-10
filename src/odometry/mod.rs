//! Odometry: pose estimation, bootstrap, and frame association.

use kornia_3d::pose::Pose3d;

use crate::frame::Frame;

pub mod bootstrap;
pub mod estimation;

pub use estimation::Estimator;

/// Status of an odometry step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdometryStatus {
    /// Frame tracked successfully.
    Tracked,
    /// Frame processed but rejected (includes bootstrap frames before the map is ready).
    Skipped,
    /// Keyframe accepted and pose chained.
    KeyframeAccepted,
}

/// Result of processing one frame.
#[derive(Debug, Clone)]
pub struct OdometryResult {
    /// Current accumulated world-to-camera pose.
    pub pose_world_to_cam: Pose3d,
    /// Status for this frame.
    pub status: OdometryStatus,
}

/// Mutable odometry state carried across frames.
#[derive(Debug, Clone)]
pub struct OdometryState {
    pub pose_world_to_cam: Pose3d,
    pub velocity: Option<Pose3d>,
    pub current_keyframe_idx: Option<usize>,
    pub last_keyframe_idx: Option<usize>,
    pub consecutive_failures: usize,
    pub bootstrap_frame: Option<Frame>,
    pub state: OdometryMode,
}

/// Odometry mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdometryMode {
    /// Bootstrap from two-view geometry before any map exists.
    Bootstrap,
    /// Track against the existing map and insert keyframes when needed.
    Tracking,
}

impl OdometryState {
    pub fn new() -> Self {
        Self {
            pose_world_to_cam: Pose3d::IDENTITY,
            velocity: None,
            current_keyframe_idx: None,
            last_keyframe_idx: None,
            consecutive_failures: 0,
            bootstrap_frame: None,
            state: OdometryMode::Bootstrap,
        }
    }

    pub fn reset(&mut self) {
        self.state = OdometryMode::Bootstrap;
        self.current_keyframe_idx = None;
        self.last_keyframe_idx = None;
        self.velocity = None;
        self.consecutive_failures = 0;
        self.bootstrap_frame = None;
    }
}
