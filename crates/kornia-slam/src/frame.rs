//! Core frame-domain types shared across the crate.

use kornia_3d::pose::Pose3d;
use kornia_image::ImageSize;
use kornia_imgproc::features::OrbFeatures;

/// `Frame` carries ORB features directly today.
///
/// If kornia-slam grows multiple frontend feature pipelines, this module is the
/// place to introduce a higher-level abstraction above `OrbFeatures`.
/// A camera observation: features extracted at a known pose.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Frame index.
    pub idx: usize,
    /// Features extracted for this frame.
    pub features: OrbFeatures,
    /// Camera pose in world-to-camera convention.
    pub pose_world_to_cam: Pose3d,
    /// Image dimensions.
    pub image_size: ImageSize,
    /// Pixel color sampled at each keypoint location (one [u8; 3] per keypoint).
    pub keypoint_colors: Vec<[u8; 3]>,
    /// Right-image x-coordinate of the stereo match per left keypoint
    /// (ORB-SLAM3's `mvuRight`); `-1.0` = no stereo match. Parallel to
    /// `features.keypoints_xy`. Empty when the frame is monocular.
    pub u_right: Vec<f32>,
    /// Metric depth per left keypoint (ORB-SLAM3's `mvDepth`); `-1.0` = no
    /// stereo match. Parallel to `features.keypoints_xy`. Empty when monocular.
    pub depth: Vec<f32>,
}

impl Frame {
    /// Whether this frame carries per-keypoint stereo depth.
    pub fn is_stereo(&self) -> bool {
        !self.depth.is_empty()
    }

    /// Metric depth for keypoint `i`, if it has a valid stereo match.
    pub fn stereo_depth(&self, i: usize) -> Option<f32> {
        match self.depth.get(i) {
            Some(&z) if z > 0.0 => Some(z),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_imgproc::features::OrbFeatures;

    fn empty_frame(u_right: Vec<f32>, depth: Vec<f32>) -> Frame {
        Frame {
            idx: 0,
            features: OrbFeatures {
                keypoints_xy: Vec::new(),
                orientations: Vec::new(),
                descriptors: Vec::new(),
                octaves: Vec::new(),
            },
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: ImageSize {
                width: 1,
                height: 1,
            },
            keypoint_colors: Vec::new(),
            u_right,
            depth,
        }
    }

    #[test]
    fn monocular_frame_reports_no_stereo() {
        let frame = empty_frame(Vec::new(), Vec::new());
        assert!(!frame.is_stereo());
        assert_eq!(frame.stereo_depth(0), None);
    }

    #[test]
    fn stereo_frame_reports_valid_depth() {
        let frame = empty_frame(vec![10.0, -1.0], vec![2.5, -1.0]);
        assert!(frame.is_stereo());
        assert_eq!(frame.stereo_depth(0), Some(2.5));
        assert_eq!(frame.stereo_depth(1), None); // sentinel
        assert_eq!(frame.stereo_depth(2), None); // out of range
    }
}
