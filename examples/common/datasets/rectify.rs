//! EuRoC adapter for kornia-3d's Bouguet stereo rectifier.
//!
//! The Bouguet rectifier core ([`kornia_3d::stereo`]) is dataset-agnostic: it
//! consumes a generic [`CameraCalib`] plus the relative pose between the two
//! cameras. EuRoC ships raw cam0/cam1 images with independent intrinsics,
//! distortion, and a body-frame `T_BS` extrinsic each, so this module adapts an
//! [`EurocCameraCalibration`] pair into those generic inputs and builds the
//! rectifier. The OAK-D / MCAP sources feed `from_calib` directly via
//! [`StereoCalib`](super::StereoCalib).

use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_imgproc::calibration::distortion::PolynomialDistortion;

use super::euroc::EurocCameraCalibration;

// Re-export the upstream rectifier so existing `crate::datasets::rectify` and
// `crate::datasets::StereoRectifier` paths keep resolving.
pub use kornia_3d::stereo::{CameraCalib, StereoError, StereoRectifier};

/// Generic [`CameraCalib`] from an EuRoC per-camera calibration.
fn camera_calib_from_euroc(cam: &EurocCameraCalibration) -> CameraCalib {
    CameraCalib {
        width: cam.width,
        height: cam.height,
        fx: cam.fx,
        fy: cam.fy,
        cx: cam.cx,
        cy: cam.cy,
        distortion: PolynomialDistortion {
            k1: cam.k1,
            k2: cam.k2,
            k3: 0.0,
            k4: 0.0,
            k5: 0.0,
            k6: 0.0,
            p1: cam.p1,
            p2: cam.p2,
        },
    }
}

/// Builds a [`StereoRectifier`] from the left (`cam0`) and right (`cam1`) EuRoC
/// calibrations, deriving the relative pose left → right from their `T_BS`
/// body-frame extrinsics.
pub fn rectifier_from_euroc(
    left: &EurocCameraCalibration,
    right: &EurocCameraCalibration,
) -> Result<StereoRectifier, StereoError> {
    // Relative pose left -> right: X_right = R * X_left + t.
    let (r_l, t_l) = decompose_t_bs(&left.t_bs);
    let (r_r, t_r) = decompose_t_bs(&right.t_bs);
    let r_rt = r_r.transpose();
    let r_rel = r_rt * r_l;
    let t_rel = r_rt * (t_l - t_r);
    StereoRectifier::from_calib(
        &camera_calib_from_euroc(left),
        &camera_calib_from_euroc(right),
        r_rel,
        t_rel,
    )
}

/// Splits a row-major 4x4 `T_BS` into rotation (3x3) and translation (3).
fn decompose_t_bs(m: &[f64; 16]) -> (Mat3F64, Vec3F64) {
    let r = Mat3F64::from_cols(
        Vec3F64::new(m[0], m[4], m[8]),
        Vec3F64::new(m[1], m[5], m[9]),
        Vec3F64::new(m[2], m[6], m[10]),
    );
    let t = Vec3F64::new(m[3], m[7], m[11]);
    (r, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam(t_bs: [f64; 16], cx: f64, cy: f64) -> EurocCameraCalibration {
        EurocCameraCalibration {
            fx: 458.0,
            fy: 457.0,
            cx,
            cy,
            k1: -0.28,
            k2: 0.07,
            p1: 0.0,
            p2: 0.0,
            width: 752,
            height: 480,
            t_bs,
        }
    }

    #[test]
    fn rectified_baseline_matches_mh01() {
        // Real MH_01_easy cam0/cam1 T_BS (row-major) and principal points.
        let t_bs0 = [
            0.0148655429818,
            -0.999880929698,
            0.00414029679422,
            -0.0216401454975,
            0.999557249008,
            0.0149672133247,
            0.025715529948,
            -0.064676986768,
            -0.0257744366974,
            0.00375618835797,
            0.999660727178,
            0.00981073058949,
            0.0,
            0.0,
            0.0,
            1.0,
        ];
        let t_bs1 = [
            0.0125552670891,
            -0.999755099723,
            0.0182237714554,
            -0.0198435579556,
            0.999598781151,
            0.0130119051815,
            0.0251588363115,
            0.0453689425024,
            -0.0253898008918,
            0.0179005838253,
            0.999517347078,
            0.00786212447038,
            0.0,
            0.0,
            0.0,
            1.0,
        ];
        let left = cam(t_bs0, 367.215, 248.375);
        let right = cam(t_bs1, 379.999, 255.238);
        let rect = rectifier_from_euroc(&left, &right).expect("valid MH_01 stereo calib");

        // EuRoC VI-sensor stereo baseline is ~0.11 m.
        assert!(
            (rect.baseline() - 0.11).abs() < 0.01,
            "baseline {} not ~0.11 m",
            rect.baseline()
        );
        assert!(rect.bf() > 0.0);
    }
}
