//! Two-view geometric initialization for monocular tracking.
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
use kornia_imgproc::features::{OrbMatchConfig, match_orb_descriptors};

use crate::OrbFeatures;

/// Configuration for two-view initialization.
#[derive(Debug, Clone)]
pub struct TwoViewAcceptanceConfig {
    /// Minimum descriptor matches required before two-view estimation.
    pub min_matches: usize,
    /// Minimum inlier count required to accept initialization.
    pub min_inliers: usize,
    /// Minimum triangulated points required to accept initialization.
    pub min_triangulated: usize,
}

/// Configuration for two-view initialization.
#[derive(Debug, Clone)]
pub struct TwoViewInitConfig {
    /// ORB descriptor matcher settings.
    pub match_config: OrbMatchConfig,
    /// Two-view model estimation settings.
    pub estimation_config: TwoViewConfig,
    /// Acceptance thresholds applied on top of the estimator result.
    pub acceptance_config: TwoViewAcceptanceConfig,
}

impl Default for TwoViewInitConfig {
    fn default() -> Self {
        Self {
            match_config: OrbMatchConfig {
                nn_ratio: 0.6,
                th_low: 50,
                check_orientation: true,
                histo_length: 30,
            },
            estimation_config: TwoViewConfig::default(),
            acceptance_config: TwoViewAcceptanceConfig {
                min_matches: 100,
                min_inliers: 30,
                min_triangulated: 50,
            },
        }
    }
}

/// Two-view initialization rejection reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TwoViewInitRejectReason {
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

/// Result of one two-view initialization attempt.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum TwoViewInitOutcome {
    /// Attempt was rejected by two-view initialization gates.
    Rejected {
        /// Why the two-view initialization attempt failed.
        reason: TwoViewInitRejectReason,
    },
    /// Two-view initialization succeeded.
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

/// Attempt two-view initialization between a reference frame and the current frame.
pub fn try_initialize_two_view(
    ref_features: &OrbFeatures,
    ref_pose: &Pose3d,
    curr_features: &OrbFeatures,
    curr_pose: &Pose3d,
    camera: &PinholeCamera,
    config: &TwoViewInitConfig,
) -> TwoViewInitOutcome {
    let acceptance = &config.acceptance_config;

    let matches = match_orb_descriptors(
        &ref_features.orientations,
        &ref_features.descriptors,
        &curr_features.orientations,
        &curr_features.descriptors,
        config.match_config,
    );
    if matches.len() < acceptance.min_matches {
        return TwoViewInitOutcome::Rejected {
            reason: TwoViewInitRejectReason::LowMatches,
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
        &config.estimation_config,
    ) else {
        return TwoViewInitOutcome::Rejected {
            reason: TwoViewInitRejectReason::EstimationFailed,
        };
    };

    if !matches!(result.model, TwoViewModel::Fundamental(_)) {
        return TwoViewInitOutcome::Rejected {
            reason: TwoViewInitRejectReason::WrongModel,
        };
    }

    if result.points3d.len() < acceptance.min_triangulated {
        return TwoViewInitOutcome::Rejected {
            reason: TwoViewInitRejectReason::LowTriangulated,
        };
    }

    if result.inlier_indices.len() < acceptance.min_inliers {
        return TwoViewInitOutcome::Rejected {
            reason: TwoViewInitRejectReason::LowInliers,
        };
    }

    let median_parallax_deg = result.median_parallax_deg(&reference_pts, &current_pts, camera);
    if median_parallax_deg < config.estimation_config.triangulation.min_parallax_deg {
        return TwoViewInitOutcome::Rejected {
            reason: TwoViewInitRejectReason::LowParallax,
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

    TwoViewInitOutcome::Initialized {
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
