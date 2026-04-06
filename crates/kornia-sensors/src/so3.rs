// TODO: migrate to kornia-algebra once SO3F64 is available

use kornia_algebra::{Mat3F64, Vec3F64};

/// Exponential map so(3) -> SO(3) via Rodrigues' formula.
///
/// Takes an angular velocity vector omega in the Lie algebra so(3)
/// and maps it to a rotation matrix in SO(3).
///
/// exp(omega) = I + (sin(theta)/theta) * [omega]_x + ((1 - cos(theta))/theta^2) * [omega]_x^2
pub fn exp(omega: &Vec3F64) -> Mat3F64 {
    let theta_sq = omega.x * omega.x + omega.y * omega.y + omega.z * omega.z;
    let theta = theta_sq.sqrt();

    let omega_hat = hat(omega);
    let omega_hat_sq = Mat3F64(omega_hat.mul_mat3(&omega_hat));

    if theta < 1e-8 {
        // First-order Taylor: exp(omega) ≈ I + [omega]_x
        Mat3F64(Mat3F64::IDENTITY.add_mat3(&omega_hat))
    } else {
        // Rodrigues: I + (sin/theta)*hat + ((1-cos)/theta^2)*hat^2
        let a = theta.sin() / theta;
        let b = (1.0 - theta.cos()) / theta_sq;

        let term1 = Mat3F64(omega_hat.mul_scalar(a));
        let term2 = Mat3F64(omega_hat_sq.mul_scalar(b));

        Mat3F64(Mat3F64::IDENTITY.add_mat3(&Mat3F64(term1.add_mat3(&term2))))
    }
}

/// Hat operator: R^3 -> so(3), maps a vector to a skew-symmetric matrix.
///
/// ```text
///          [ 0  -z   y]
/// [x,y,z] -> [ z   0  -x]
///          [-y   x   0]
/// ```
pub fn hat(v: &Vec3F64) -> Mat3F64 {
    Mat3F64::from_cols_array(&[
        0.0, v.z, -v.y, // col 0
        -v.z, 0.0, v.x, // col 1
        v.y, -v.x, 0.0, // col 2
    ])
}

/// Right Jacobian of SO(3).
///
/// Relates a perturbation in the Lie algebra to its effect through exp:
///   exp(omega + delta) ≈ exp(omega) · exp(J_r(omega) · delta)
///
/// J_r(omega) = I - ((1-cos(theta))/theta²) * [omega]_x + ((theta-sin(theta))/theta³) * [omega]_x²
pub fn right_jacobian(omega: &Vec3F64) -> Mat3F64 {
    let theta_sq = omega.x * omega.x + omega.y * omega.y + omega.z * omega.z;
    let theta = theta_sq.sqrt();

    let omega_hat = hat(omega);
    let omega_hat_sq = Mat3F64(omega_hat.mul_mat3(&omega_hat));

    if theta < 1e-8 {
        // Small angle: J_r ≈ I - ½ [omega]_x
        let half_hat = Mat3F64(omega_hat.mul_scalar(0.5));
        Mat3F64(Mat3F64::IDENTITY.sub_mat3(&half_hat))
    } else {
        // J_r = I - ((1-cos)/theta²)*hat + ((theta-sin)/theta³)*hat²
        let a = (1.0 - theta.cos()) / theta_sq;
        let b = (theta - theta.sin()) / (theta_sq * theta);

        let term1 = Mat3F64(omega_hat.mul_scalar(a));
        let term2 = Mat3F64(omega_hat_sq.mul_scalar(b));

        Mat3F64(Mat3F64::IDENTITY.sub_mat3(&Mat3F64(term1.sub_mat3(&term2))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq_mat(a: &Mat3F64, b: &Mat3F64, tol: f64) {
        let ac = a.to_cols_array();
        let bc = b.to_cols_array();
        for i in 0..9 {
            assert!(
                (ac[i] - bc[i]).abs() < tol,
                "mismatch at {i}: {} vs {}",
                ac[i],
                bc[i]
            );
        }
    }

    #[test]
    fn exp_zero_is_identity() {
        let r = exp(&Vec3F64::ZERO);
        approx_eq_mat(&r, &Mat3F64::IDENTITY, EPS);
    }

    #[test]
    fn exp_small_angle_approx() {
        // For tiny angles, exp(omega) ≈ I + [omega]_x
        let omega = Vec3F64::new(1e-10, 2e-10, 3e-10);
        let r = exp(&omega);
        let expected = Mat3F64(Mat3F64::IDENTITY.add_mat3(&hat(&omega)));
        approx_eq_mat(&r, &expected, EPS);
    }

    #[test]
    fn exp_pi_around_z() {
        // 180 degree rotation around z-axis: x -> -x, y -> -y, z -> z
        let omega = Vec3F64::new(0.0, 0.0, std::f64::consts::PI);
        let r = exp(&omega);
        let expected = Mat3F64::from_cols_array(&[
            -1.0, 0.0, 0.0, // col 0
            0.0, -1.0, 0.0, // col 1
            0.0, 0.0, 1.0, // col 2
        ]);
        approx_eq_mat(&r, &expected, 1e-9);
    }

    #[test]
    fn right_jacobian_zero_is_identity() {
        let jr = right_jacobian(&Vec3F64::ZERO);
        approx_eq_mat(&jr, &Mat3F64::IDENTITY, EPS);
    }

    #[test]
    fn right_jacobian_numerical() {
        // Verify J_r by finite difference:
        // exp(omega + eps*e_i) ≈ exp(omega) · exp(J_r · eps*e_i)
        let omega = Vec3F64::new(0.3, -0.2, 0.5);
        let jr = right_jacobian(&omega);
        let r = exp(&omega);
        let eps = 1e-6;

        for i in 0..3 {
            let mut delta = [0.0; 3];
            delta[i] = eps;
            let dv = Vec3F64::new(delta[0], delta[1], delta[2]);

            // Left side: exp(omega + delta)
            let r_perturbed = exp(&(omega + dv));

            // Right side: exp(omega) · exp(J_r · delta)
            let jr_delta = mat3_mul_vec3(&jr, &dv);
            let r_approx = Mat3F64(r.mul_mat3(&exp(&jr_delta)));

            approx_eq_mat(&r_perturbed, &r_approx, 1e-5);
        }
    }

    fn mat3_mul_vec3(m: &Mat3F64, v: &Vec3F64) -> Vec3F64 {
        let dv = glam::DVec3::new(v.x, v.y, v.z);
        Vec3F64::from(m.mul_vec3(dv))
    }

    #[test]
    fn hat_is_skew_symmetric() {
        let v = Vec3F64::new(1.0, 2.0, 3.0);
        let h = hat(&v);
        let ht = Mat3F64(*h.transpose());
        let sum = Mat3F64(h.add_mat3(&ht));
        approx_eq_mat(&sum, &Mat3F64::ZERO, EPS);
    }
}
