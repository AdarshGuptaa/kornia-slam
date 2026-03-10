//! Bootstrap-stage logic for monocular tracking using two-view triangulation.a
//              (3D point)
//                  X
//                /   \
//               /     \
//              /       \
//             /         \
//            /           \
//      .-----------. .-----------.
//     /   x1 *    / /    * x2   /
//    /           / /           /
//    '-----------' '-----------'
//      frame 1        frame 2

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::pose::{TwoViewConfig, TwoViewModel, two_view_estimate};
use kornia_algebra::Vec3F64;
use kornia_imgproc::features::OrbMatchConfig;

use crate::odometry::estimation::matching::match_orb_descriptors;
use crate::frame::OrbFeatures;

/// Configuration for the bootstrap stage.
#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    /// ORB descriptor matcher settings.
    pub match_config: OrbMatchConfig,
    /// Two-view model estimation settings.
    pub two_view_config: TwoViewConfig,
    /// Minimum descriptor matches required before two-view estimation.
    pub min_matches: usize,
    /// Minimum inlier count required to accept bootstrap.
    pub min_inliers: usize,
    /// Minimum median parallax in degrees required to accept bootstrap.
    pub min_parallax_deg: f64,
    /// Minimum triangulated points required to accept bootstrap.
    pub min_triangulated: usize,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            match_config: OrbMatchConfig {
                nn_ratio: 0.6,
                th_low: 50,
                check_orientation: true,
                histo_length: 30,
            },
            two_view_config: TwoViewConfig::default(),
            min_matches: 100,
            min_inliers: 30,
            min_parallax_deg: 1.0,
            min_triangulated: 50,
        }
    }
}

/// Bootstrap rejection reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BootstrapRejectReason {
    /// Not enough descriptor matches to run two-view estimation.
    LowMatches,
    /// Two-view estimation failed.
    EstimationFailed,
    /// Homography was selected instead of fundamental model.
    WrongModel,
    /// Too few triangulated points.
    LowTriangulated,
    /// Too few inliers in estimated model.
    LowInliers,
    /// Not enough parallax.
    LowParallax,
}

/// Result of one bootstrap step.
#[derive(Debug, Clone)]
pub enum BootstrapOutcome {
    /// Attempt was rejected by bootstrap gates.
    Rejected {
        /// Why the bootstrap attempt failed.
        reason: BootstrapRejectReason,
    },
    /// Bootstrap succeeded.
    Initialized {
        /// New world-to-camera pose for the current frame.
        pose_world_to_cam: Pose3d,
        /// Estimated constant-velocity increment (between previous pose and new pose).
        motion_increment: Pose3d,
        /// Descriptor match pairs `(reference_desc_idx, current_desc_idx)`.
        matches: Vec<(usize, usize)>,
        /// Triangulated 3D points in the reference camera frame.
        points3d: Vec<Vec3F64>,
        /// Indices into `matches` that were inliers in two-view estimation.
        inlier_indices: Vec<usize>,
        /// Median positive depth in the two-view triangulation (if available).
        median_depth: Option<f64>,
    },
}

/// Attempt two-view bootstrap between a reference frame and the current frame.
pub fn try_bootstrap(
    ref_features: &OrbFeatures,
    ref_pose: &Pose3d,
    curr_features: &OrbFeatures,
    curr_pose: &Pose3d,
    camera: &PinholeCamera,
    config: &BootstrapConfig,
) -> BootstrapOutcome {
    let matches = match_orb_descriptors(
        &ref_features.orientations,
        &ref_features.descriptors,
        &curr_features.orientations,
        &curr_features.descriptors,
        config.match_config,
    );
    if matches.len() < config.min_matches {
        return BootstrapOutcome::Rejected {
            reason: BootstrapRejectReason::LowMatches,
        };
    }

    let (reference_pts, current_pts) = camera.undistort_matched_pairs(
        &ref_features.keypoints_xy,
        &curr_features.keypoints_xy,
        &matches,
    );

    let k = camera.intrinsic_matrix();
    let Ok(result) = two_view_estimate(
        &reference_pts,
        &current_pts,
        &k,
        &k,
        &config.two_view_config,
    ) else {
        return BootstrapOutcome::Rejected {
            reason: BootstrapRejectReason::EstimationFailed,
        };
    };

    if !matches!(result.model, TwoViewModel::Fundamental(_)) {
        return BootstrapOutcome::Rejected {
            reason: BootstrapRejectReason::WrongModel,
        };
    }

    if result.points3d.len() < config.min_triangulated {
        return BootstrapOutcome::Rejected {
            reason: BootstrapRejectReason::LowTriangulated,
        };
    }

    if result.inlier_indices.len() < config.min_inliers {
        return BootstrapOutcome::Rejected {
            reason: BootstrapRejectReason::LowInliers,
        };
    }

    let median_parallax_deg = result.median_parallax_deg(&reference_pts, &current_pts, camera);
    if median_parallax_deg < config.min_parallax_deg {
        return BootstrapOutcome::Rejected {
            reason: BootstrapRejectReason::LowParallax,
        };
    }

    let mut t_scaled = result.translation;
    let mut depths: Vec<f64> = result
        .points3d
        .iter()
        .map(|p| p.z)
        .filter(|&z| z > 0.0)
        .collect();
    let median_depth = median_in_place(&mut depths).filter(|&d| d > 1e-6);
    if let Some(md) = median_depth {
        t_scaled /= md;
    }

    let new_pose = Pose3d::new(
        result.rotation * ref_pose.rotation,
        result.rotation * ref_pose.translation + t_scaled,
    );
    let motion_increment = Pose3d::between(curr_pose, &new_pose);

    BootstrapOutcome::Initialized {
        pose_world_to_cam: new_pose,
        motion_increment,
        matches,
        points3d: result.points3d,
        inlier_indices: result.inlier_indices,
        median_depth,
    }
}

/// Computes the median of a mutable slice in-place.
fn median_in_place(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mid = values.len() / 2;
    values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    Some(values[mid])
}
