//! System runtime state, mode transitions, and tracking results.

use kornia_3d::pose::Pose3d;

use crate::frame::Frame;

/// Status of processing one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackingStatus {
    /// Frame tracked successfully.
    Tracked,
    /// Frame processed but rejected (includes bootstrap frames before the map is ready).
    Skipped,
    /// Keyframe accepted and pose chained.
    KeyframeAccepted,
}

/// Result of processing one frame.
#[derive(Debug, Clone)]
pub struct TrackingResult {
    /// Current accumulated world-to-camera pose.
    pub pose_world_to_cam: Pose3d,
    /// Status for this frame.
    pub status: TrackingStatus,
}

/// Mutable pipeline state carried across frames.
#[derive(Debug, Clone)]
pub struct SystemState {
    pub pose_world_to_cam: Pose3d,
    pub velocity: Option<Pose3d>,
    pub current_keyframe_idx: Option<usize>,
    pub last_keyframe_idx: Option<usize>,
    pub consecutive_failures: usize,
    pub bootstrap_frame: Option<Frame>,
    pub mode: SystemMode,
}

/// Pipeline mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemMode {
    /// Bootstrap from two-view geometry before any map exists.
    Bootstrap,
    /// Track against the existing map and insert keyframes when needed.
    Tracking,
}

impl SystemState {
    pub fn new() -> Self {
        Self {
            pose_world_to_cam: Pose3d::IDENTITY,
            velocity: None,
            current_keyframe_idx: None,
            last_keyframe_idx: None,
            consecutive_failures: 0,
            bootstrap_frame: None,
            mode: SystemMode::Bootstrap,
        }
    }

    pub fn reset(&mut self) {
        self.mode = SystemMode::Bootstrap;
        self.current_keyframe_idx = None;
        self.last_keyframe_idx = None;
        self.velocity = None;
        self.consecutive_failures = 0;
        self.bootstrap_frame = None;
    }
}

impl Default for SystemState {
    fn default() -> Self {
        Self::new()
    }
}
