use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_algebra::linalg::svd::svd3_f64;
use super::types::Sim3Alignment;

pub fn align_sim3(est: &[Vec3F64], gt: &[Vec3F64]) -> Sim3Alignment {
    let n = est.len() as f64;

    // Centroids
    let mut mu_est = Vec3F64::ZERO;
    let mut mu_gt = Vec3F64::ZERO;
    for i in 0..est.len() {
        mu_est += est[i];
        mu_gt += gt[i];
    }
    mu_est /= n;
    mu_gt /= n;

    // Cross-covariance and estimated variance
    let mut sigma = Mat3F64::ZERO;
    let mut var_est = 0.0_f64;
    for i in 0..est.len() {
        let pe = est[i] - mu_est;
        let pg = gt[i] - mu_gt;
        sigma += Mat3F64::from_cols(pg * pe.x, pg * pe.y, pg * pe.z);
        var_est += pe.dot(pe);
    }
    *sigma /= n;
    var_est /= n;

    // SVD of cross-covariance matrix
    let svd = svd3_f64(&sigma);
    let u = *svd.u();
    let s = *svd.s();
    let v = *svd.v();

    // Reflection correction: ensure det(U * V^T) > 0
    let mut diag_s = Mat3F64::IDENTITY;
    if (u * v.transpose()).determinant() < 0.0 {
        diag_s.z_axis.z = -1.0;
    }

    let r = u * diag_s * v.transpose();

    // Scale: trace(S_diag * diag_s) / var_est
    // S is diagonal, so only the diagonal elements matter
    let trace = s.x_axis.x * diag_s.x_axis.x
        + s.y_axis.y * diag_s.y_axis.y
        + s.z_axis.z * diag_s.z_axis.z;
    let scale = trace / var_est;

    let translation = mu_gt - (r * mu_est) * scale;

    Sim3Alignment { scale, rotation: r, translation }
}