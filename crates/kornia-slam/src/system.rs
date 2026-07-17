//! System runtime state, mode transitions, and tracking results.

use kornia_3d::pose::Pose3d;

use crate::frame::Frame;
use kornia_algebra::Vec3F64;

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

impl KeyframePolicy {
    /// Decide whether a new keyframe should be inserted.
    pub fn should_insert(
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
        if frames_since_last_kf < self.min_frames_between {
            return false;
        }
        if frames_since_last_kf >= self.max_frames_between {
            return true;
        }

        if n_ref_map_points == 0 {
            return true;
        }

        let weak_threshold = (n_ref_map_points as f64 * self.ref_ratio) as usize;
        tracked_inliers >= 15 && tracked_inliers < weak_threshold
    }
}

/// Recently-lost grace period policy (mirrors ORB-SLAM3's RECENTLY_LOST vs
/// LOST distinction), bridging brief interruptions (motion blur, a few
/// dropped/occluded frames) without throwing the map away.
///
/// Deliberately short, unlike ORB-SLAM3's ~5s: our PnP has no RANSAC/robust
/// loss, so the projection search and the PnP prior-reprojection gate both
/// key off the same predicted pose. Once genuinely lost (not a brief blip),
/// that pose keeps compounding IMU/constant-velocity drift every extra frame
/// we wait, which does not improve recovery odds (verified against EuRoC
/// V101 frames ~600-770, a sustained-loss segment: granting several seconds
/// of patience there only delayed the same eventual reset, it never let
/// tracking resume early). A map that's too young, or an inertial state that
/// hasn't settled yet, gets no grace at all.
#[derive(Debug, Clone)]
pub struct TrackingLossRecoveryPolicy {
    /// Minimum keyframe count before any grace period is granted.
    pub min_keyframes_for_grace: usize,
    /// Grace period once the IMU has been initialized for at least
    /// `min_imu_confidence_sec`.
    pub timeout_imu_sec: f64,
    /// Grace period otherwise (no IMU, or too recently initialized).
    pub timeout_visual_sec: f64,
    /// How long the IMU must have been initialized before `timeout_imu_sec`
    /// applies instead of `timeout_visual_sec`.
    pub min_imu_confidence_sec: f64,
}

impl Default for TrackingLossRecoveryPolicy {
    fn default() -> Self {
        Self {
            min_keyframes_for_grace: 10,
            timeout_imu_sec: 1.0,
            timeout_visual_sec: 0.5,
            min_imu_confidence_sec: 2.0,
        }
    }
}

impl TrackingLossRecoveryPolicy {
    /// Grace period, in seconds, to allow before giving up and resetting.
    pub fn grace_period_sec(&self, imu_confident: bool) -> f64 {
        if imu_confident {
            self.timeout_imu_sec
        } else {
            self.timeout_visual_sec
        }
    }
}

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
    /// Metric body velocity in the world frame (m/s); valid once `imu_initialized`.
    pub velocity_world: Vec3F64,
    /// Timestamp of the previous processed frame, bounding the per-frame
    /// preintegration window.
    pub last_frame_timestamp_sec: f64,
    /// Whether visual-inertial initialization succeeded (gates IMU pose prediction).
    pub imu_initialized: bool,
    /// Timestamp (sec) at which inertial initialization completed. IMU-only
    /// pose prediction isn't trustworthy yet for a short window after this,
    /// so a tracking loss shortly after init is treated as fully lost rather
    /// than granted the longer inertial grace period.
    pub imu_init_timestamp_sec: Option<f64>,
    pub current_keyframe_idx: Option<usize>,
    pub last_keyframe_idx: Option<usize>,
    /// Timestamp (sec) of the first frame in the current run of tracking
    /// failures, or `None` while tracking is healthy. Drives the
    /// recently-lost grace period (mirrors ORB-SLAM3's `mTimeStampLost`).
    pub lost_since_sec: Option<f64>,
    pub bootstrap_frame: Option<Frame>,
    pub mode: SystemMode,
}

/// Pipeline mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemMode {
    /// Bootstrap from two-view geometry before any map exists.
    Bootstrap,
    /// IMU initialization for scale and gravity
    ImuInit,
    /// Track against the existing map and insert keyframes when needed.
    Tracking,
}

impl SystemState {
    pub fn new() -> Self {
        Self {
            pose_world_to_cam: Pose3d::IDENTITY,
            velocity: None,
            velocity_world: Vec3F64::ZERO,
            current_keyframe_idx: None,
            last_keyframe_idx: None,
            lost_since_sec: None,
            bootstrap_frame: None,
            imu_initialized: false,
            imu_init_timestamp_sec: None,
            last_frame_timestamp_sec: 0.0,
            mode: SystemMode::Bootstrap,
        }
    }

    pub fn reset(&mut self) {
        self.mode = SystemMode::Bootstrap;
        self.current_keyframe_idx = None;
        self.last_keyframe_idx = None;
        self.velocity = None;
        self.lost_since_sec = None;
        self.bootstrap_frame = None;
        // The new map starts at an unknown monocular scale, so the metric
        // IMU state no longer applies until inertial init runs again.
        self.imu_initialized = false;
        self.imu_init_timestamp_sec = None;
        self.velocity_world = Vec3F64::ZERO;
    }
}

impl Default for SystemState {
    fn default() -> Self {
        Self::new()
    }
}
