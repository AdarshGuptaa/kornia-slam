//! Schur-complement bundle adjustment with dense reduced camera system.
//!
//! The standard bipartite-Schur trick from Triggs et al. (1999). Each LM
//! iteration builds the Hessian in BLOCK form
//!
//! ```text
//!     H = [ A   B  ]    A = 6P × 6P pose blocks (block-diagonal),
//!         [ Bᵀ  C  ]    C = 3N × 3N point blocks (BLOCK-DIAGONAL),
//!                       B = 6P × 3N pose-point cross terms (sparse).
//! ```
//!
//! Block-diagonal C means C⁻¹ is cheap (per-3×3 invert). The **reduced
//! camera system**
//!
//! ```text
//!     M = A − B C⁻¹ Bᵀ           (dense 6P × 6P)
//!     m = g_pose − B C⁻¹ g_point
//! ```
//!
//! is solved with `faer`'s dense Cholesky on the small matrix; points are
//! recovered by back-substitution. For our SLAM problem (~170 poses ×
//! ~3000 points × ~15000 observations) the reduced system is just
//! 1020 × 1020 — Ceres's `DENSE_SCHUR` is exactly this regime.
//!
//! No sparse-matrix dependency is needed because the only "large" object
//! the Schur trick has to manipulate (B, 6P × 3N) is never materialised:
//! we walk observations and accumulate per-point contributions into M
//! directly.
//!
//! Jacobian conventions match [`kornia_3d::ba::ReprojFactor`]:
//!
//!   * Pose tangent layout `[ρ; ω]` (upsilon then omega), 6-dim.
//!   * Point parameters are the 3-dim world coordinates.
//!   * z is clamped to `MIN_Z` to handle mid-iteration cheirality flips.
//!
//! Currently supports: identity loss only, fixed-pose anchors, fixed-point
//! gauge (motion-only BA). Robust kernels and full LM-with-backtracking
//! are TODO.

use faer::prelude::Solve;
use faer::Mat;
use kornia_algebra::{Mat3AF32, Mat3F64, Vec3AF32, Vec3F64, SE3F32, SO3F32, SO3F64};
use thiserror::Error;

use kornia_3d::ba::{BaError, BaObservation, BaParams, BaPosePrior, BaResult};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::ransac::RobustKernelKind;
use kornia_sensors::imu::{ImuBias, PreintegratedImu};

const MIN_Z: f32 = 1e-3;

/// Errors specific to the Schur BA driver. Wraps existing [`BaError`].
#[derive(Debug, Error)]
pub enum SchurBaError {
    /// Linear system is rank-deficient / Cholesky failed.
    #[error("Reduced camera Cholesky failed (likely rank-deficient): {0}")]
    CholeskyFailed(String),
    /// No free variables after applying anchors.
    #[error("All variables are fixed — nothing to optimise")]
    NoFreeVariables,
    /// Other BA setup error.
    #[error(transparent)]
    Ba(#[from] BaError),
}


/// Errors specific to VI-BA.
#[derive(Debug, Error)]
pub enum ViBaError {
    #[error("Reduced system Cholesky failed (rank-deficient): {0}")]
    CholeskyFailed(String),
    #[error("All keyframe variables are fixed — nothing to optimise")]
    NoFreeVariables,
    #[error("IMU edge keyframe indices out of range: {0} → {1}")]
    ImuEdgeOutOfRange(usize, usize),
    #[error(transparent)]
    Ba(#[from] BaError),
}


/// A single keyframe state for VI-BA.
///
/// Holds the 6-DOF pose, 3-DOF world-frame velocity, and 6-DOF IMU bias.
/// Together these form the 15-DOF state vector that VI-BA optimises.
#[derive(Debug, Clone)]
pub struct ViBaKeyframe {
    /// World→camera pose. Same convention as [`Pose3d`] in schur_ba.
    pub pose: Pose3d,
    /// Velocity in world frame (m/s).
    pub velocity: Vec3F64,
    /// IMU bias estimate at this keyframe.
    pub bias: ImuBias,
    /// If true, pose+velocity+bias are held fixed (used for the first keyframe
    /// as the gauge anchor, or for marginalised frames).
    pub fixed: bool,
}

/// An IMU edge connecting two consecutive keyframes.
///
/// `from_idx` and `to_idx` must be consecutive keyframe indices (to_idx =
/// from_idx + 1 in a window), though non-consecutive indices work as long as
/// `preintegrated` covers the interval.
#[derive(Debug, Clone)]
pub struct ImuEdge {
    pub from_idx: usize,
    pub to_idx: usize,
    /// Preintegrated measurements between the two keyframes.
    pub preintegrated: PreintegratedImu,
}


/// Parameters for VI-BA.
#[derive(Debug, Clone)]
pub struct ViBaParams {
    pub max_iterations: usize,
    /// Levenberg-Marquardt initial damping.
    pub initial_lambda: f64,
    /// Relative cost-decrease threshold for convergence.
    pub cost_tolerance: f64,
    /// Gravity vector in world frame, e.g. [0, 0, -9.81].
    pub gravity: Vec3F64,
}

impl Default for ViBaParams {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            initial_lambda: 1e-4,
            cost_tolerance: 1e-6,
            gravity: Vec3F64::new(0.0, 0.0, -9.81),
        }
    }
}

/// Output of VI-BA.
#[derive(Debug, Clone)]
pub struct ViBaResult {
    pub keyframes: Vec<ViBaKeyframe>,
    pub points: Vec<Vec3F64>,
    pub iterations: usize,
    pub converged: bool,
    pub final_cost: f64,
}

// ── f32 ↔ f64 conversion helpers (shared shape with ba.rs) ───────────────

fn pose_to_se3(pose: &Pose3d) -> SE3F32 {
    let r = Mat3AF32::from_cols(
        Vec3AF32::new(
            pose.rotation.col(0).x as f32,
            pose.rotation.col(0).y as f32,
            pose.rotation.col(0).z as f32,
        ),
        Vec3AF32::new(
            pose.rotation.col(1).x as f32,
            pose.rotation.col(1).y as f32,
            pose.rotation.col(1).z as f32,
        ),
        Vec3AF32::new(
            pose.rotation.col(2).x as f32,
            pose.rotation.col(2).y as f32,
            pose.rotation.col(2).z as f32,
        ),
    );
    let so3 = SO3F32::from_matrix(&r);
    SE3F32::new(
        so3,
        Vec3AF32::new(
            pose.translation.x as f32,
            pose.translation.y as f32,
            pose.translation.z as f32,
        ),
    )
}

fn se3_to_pose(se3: &SE3F32) -> Pose3d {
    let r = se3.r.matrix();
    let t = se3.t;
    Pose3d::new(
        Mat3F64::from_cols(
            Vec3F64::new(r.col(0).x as f64, r.col(0).y as f64, r.col(0).z as f64),
            Vec3F64::new(r.col(1).x as f64, r.col(1).y as f64, r.col(1).z as f64),
            Vec3F64::new(r.col(2).x as f64, r.col(2).y as f64, r.col(2).z as f64),
        ),
        Vec3F64::new(t.x as f64, t.y as f64, t.z as f64),
    )
}

// ── Per-observation residual + analytical Jacobian (matches ReprojFactor) ──

/// Computes (residual, J_pose 2×6, J_point 2×3) at the current state.
/// Returns the camera-frame point and the clamped z too, for back-substitution
/// reasoning.
///
/// Jacobian layout (row-major flat):
///   J_pose[0..6]:  [du/dρ_x, du/dρ_y, du/dρ_z, du/dω_x, du/dω_y, du/dω_z]
///   J_pose[6..12]: [dv/dρ_x, dv/dρ_y, dv/dρ_z, dv/dω_x, dv/dω_y, dv/dω_z]
///   J_point[0..3]: [du/dx,   du/dy,   du/dz]
///   J_point[3..6]: [dv/dx,   dv/dy,   dv/dz]
fn residual_and_jacobians(
    pose: &SE3F32,
    point_w: &Vec3F64,
    pixel: [f32; 2],
    camera: &PinholeCamera,
) -> ([f32; 2], [f32; 12], [f32; 6]) {
    let fx = camera.fx as f32;
    let fy = camera.fy as f32;
    let cx = camera.cx as f32;
    let cy = camera.cy as f32;

    let pw = Vec3AF32::new(point_w.x as f32, point_w.y as f32, point_w.z as f32);
    let pc = *pose * pw;
    let z = if pc.z.abs() < MIN_Z {
        if pc.z >= 0.0 {
            MIN_Z
        } else {
            -MIN_Z
        }
    } else {
        pc.z
    };
    let inv_z = 1.0 / z;
    let inv_z2 = inv_z * inv_z;

    let u = fx * pc.x * inv_z + cx;
    let v = fy * pc.y * inv_z + cy;
    let r = [u - pixel[0], v - pixel[1]];

    // J_proj row coefficients (∂[u; v] / ∂[X_c]).
    let a0 = fx * inv_z;
    let a2 = -fx * pc.x * inv_z2;
    let b1 = fy * inv_z;
    let b2 = -fy * pc.y * inv_z2;

    // Rotation matrix elements (R: world→cam).
    let rm = pose.r.matrix();
    let r00 = rm.col(0).x;
    let r01 = rm.col(1).x;
    let r02 = rm.col(2).x;
    let r10 = rm.col(0).y;
    let r11 = rm.col(1).y;
    let r12 = rm.col(2).y;
    let r20 = rm.col(0).z;
    let r21 = rm.col(1).z;
    let r22 = rm.col(2).z;

    let (px, py, pz) = (pw.x, pw.y, pw.z);

    // S = -R · skew(p_w) — for the omega part.
    let s00 = -pz * r01 + py * r02;
    let s10 = -pz * r11 + py * r12;
    let s20 = -pz * r21 + py * r22;

    let s01 = pz * r00 - px * r02;
    let s11 = pz * r10 - px * r12;
    let s21 = pz * r20 - px * r22;

    let s02 = -py * r00 + px * r01;
    let s12 = -py * r10 + px * r11;
    let s22 = -py * r20 + px * r21;

    // J_pt = J_proj · R (3 cols).
    let jpt_00 = a0 * r00 + a2 * r20;
    let jpt_01 = a0 * r01 + a2 * r21;
    let jpt_02 = a0 * r02 + a2 * r22;
    let jpt_10 = b1 * r10 + b2 * r20;
    let jpt_11 = b1 * r11 + b2 * r21;
    let jpt_12 = b1 * r12 + b2 * r22;

    // J_omega = J_proj · S (3 cols).
    let jom_00 = a0 * s00 + a2 * s20;
    let jom_01 = a0 * s01 + a2 * s21;
    let jom_02 = a0 * s02 + a2 * s22;
    let jom_10 = b1 * s10 + b2 * s20;
    let jom_11 = b1 * s11 + b2 * s21;
    let jom_12 = b1 * s12 + b2 * s22;

    // Layout J_pose 2×6 row-major: [ρ(3) | ω(3)] per row.
    let j_pose: [f32; 12] = [
        jpt_00, jpt_01, jpt_02, jom_00, jom_01, jom_02, jpt_10, jpt_11, jpt_12, jom_10, jom_11,
        jom_12,
    ];
    // J_point 2×3 row-major.
    let j_point: [f32; 6] = [jpt_00, jpt_01, jpt_02, jpt_10, jpt_11, jpt_12];

    (r, j_pose, j_point)
}
/// Computes the 15-DOF IMU residual and its Jacobians wrt the two keyframe states.
///
/// Returns:
///   residual: [f64; 15]  — [r_R(3), r_v(3), r_p(3), r_bg(3), r_ba(3)]
///   J_i: [f64; 15*15]   — Jacobian wrt state_i (row-major, 15 rows × 15 cols)
///   J_j: [f64; 15*15]   — Jacobian wrt state_j
///
/// State layout (per keyframe): [ρ(6) | v(3) | bg(3) | ba(3)]
///   ρ = [upsilon(3), omega(3)] — SE3 tangent, matching ReprojFactor convention
fn imu_residual_and_jacobians(
    kf_i: &ViBaKeyframe,
    kf_j: &ViBaKeyframe,
    pim: &PreintegratedImu,
    gravity: &Vec3F64,
) -> ([f64; 15], [f64; 225], [f64; 225]) {
    let dt = pim.dt;

    // Extracting state
    let r_i = &kf_i.pose.rotation;  // world <- body
    let p_i = &kf_i.pose.translation;
    let v_i = &kf_i.velocity;
    let bg_i = &kf_i.bias.gyro;
    let ba_i = &kf_i.bias.accel;

    let r_j = &kf_j.pose.rotation;
    let p_j = &kf_j.pose.translation;
    let v_j = &kf_j.velocity;
    let bg_j = &kf_j.bias.gyro;
    let ba_j = &kf_j.bias.accel;

    // First Order bias corrections
    let dr_corrected = pim.delta_rotation_with_bias(&kf_i.bias);
    let dv_corrected = pim.delta_velocity_with_bias(&kf_i.bias);
    let dp_corrected = pim.delta_position_with_bias(&kf_i.bias);

    // Rotation Residuals r_R = Log(ΔR^T · R_i^T · R_j)
    let r_i_t = Mat3F64(*r_i.transpose());
    let dr_t = Mat3F64(*dr_corrected.transpose());
    let lhs_r = Mat3F64(dr_t.mul_mat3(&r_i_t.mul_mat3(r_j)));
    let r_rot = SO3F64::from_matrix(&lhs_r).log();

    // Velocity Residuals r_v = R_i^T·(v_j - v_i - g·dt) - Δv
    let dv_world = *v_j - *v_i - *gravity * dt;
    let r_vel = r_i_t * dv_world - dv_corrected;

    // Position Residuals r_p = R_i^T·(p_j - p_i - v_i·dt - ½g·dt²) - Δp
    let dp_world = *p_j - *p_i - *v_i * dt - *gravity * (0.5 * dt * dt);
    let r_pos = r_i_t * dp_world - dp_corrected;

    // Bias random-walk residuals
    let r_bg = *bg_j - *bg_i;
    let r_ba = *ba_j - *ba_i;

    let mut residual = [0.0f64; 15];
    residual[0..3].copy_from_slice(&[r_rot.x, r_rot.y, r_rot.z]);
    residual[3..6].copy_from_slice(&[r_vel.x, r_vel.y, r_vel.z]);
    residual[6..9].copy_from_slice(&[r_pos.x, r_pos.y, r_pos.z]);
    residual[9..12].copy_from_slice(&[r_bg.x, r_bg.y, r_bg.z]);
    residual[12..15].copy_from_slice(&[r_ba.x, r_ba.y, r_ba.z]);



    // Jacobians
    // State layout: [ω(3) | ρ(3) | v(3) | bg(3) | ba(3)]  ← note ω first
    // to match the Lie group convention; map to your [ρ|ω] layout at end.
    // J_i is 15×15, J_j is 15×15, all zeros initially.

    let mut j_i = [0.0f64; 225];
    let mut j_j = [0.0f64; 225];

    let r_it_mat = r_i_t; // R_i^T
    let dp_hat = SO3F64::hat(dp_world); // skew symmetric matrix
    let dv_hat = SO3F64::hat(dv_world);

    let jr_inv = SO3F64::right_jacobian(r_rot).inverse();

    // ∂r_R / ∂ω_i = -J_r^{-1} · R_j^T · R_i
    // (perturbation of R_i via right-multiply exp(δω))
    let r_j_t = Mat3F64(*r_j.transpose());
    let neg_jr_inv_rjt_ri = mat3_neg(mat3_mul(&jr_inv, &mat3_mul(&r_j_t, r_i)));
    set_block_15x15(&mut j_i, 0, 0, &neg_jr_inv_rjt_ri);

    // ∂r_R / ∂bg_i = -J_r^{-1} · ΔR^T_corrected · J_r(ΔR) · ∂ΔR/∂bg
    // From Forster eq 9d: ∂r_R/∂bg_i = -J_r^{-1} · (ΔR_corrected)^{-T} · Jr · d_R_d_bg
    let d_r_bg = &pim.d_rotation_d_bias_gyro;
    let neg_jr_inv_dr_bg = mat3_neg(mat3_mul(&jr_inv, d_r_bg));
    set_block_15x15(&mut j_i, 0, 9, &neg_jr_inv_dr_bg);

    // ∂r_R / ∂ω_j = J_r^{-1}
    set_block_15x15(&mut j_j, 0, 0, &jr_inv);

    // ∂r_v / ∂ω_i = R_i^T · [v_j - v_i - g·dt]×
    let dr_vel_domega_i = mat3_mul(&r_it_mat, &dv_hat);
    set_block_15x15(&mut j_i, 3, 0, &dr_vel_domega_i);

    // ∂r_v / ∂v_i = -R_i^T
    let neg_ri_t = mat3_neg(r_it_mat);
    set_block_15x15(&mut j_i, 3, 3, &neg_ri_t);

    // ∂r_v / ∂bg_i = -∂Δv/∂bg
    let neg_dv_dba = mat3_neg(pim.d_velocity_d_bias_accel);
    set_block_15x15(&mut j_i, 3, 12, &neg_dv_dba);

    let neg_dv_dbg = mat3_neg(pim.d_velocity_d_bias_gyro);
    set_block_15x15(&mut j_i, 3, 9, &neg_dv_dbg);

    // ∂r_v / ∂v_j = R_i^T
    set_block_15x15(&mut j_j, 3, 3, &r_it_mat);

    // ∂r_p / ∂ω_i = R_i^T · [p_j - p_i - v_i·dt - ½g·dt²]×
    let dr_pos_domega_i = mat3_mul(&r_it_mat, &dp_hat);
    set_block_15x15(&mut j_i, 6, 0, &dr_pos_domega_i);

    // ∂r_p / ∂v_i = -R_i^T · dt
    let neg_ri_t_dt = mat3_scalar_f64(&r_it_mat, -dt);
    set_block_15x15(&mut j_i, 6, 3, &neg_ri_t_dt);

    // ∂r_p / ∂bg_i = -∂Δp/∂bg
    let neg_dp_dbg = mat3_neg(pim.d_position_d_bias_gyro);
    set_block_15x15(&mut j_i, 6, 9, &neg_dp_dbg);

    // ∂r_p / ∂ba_i = -∂Δp/∂ba
    let neg_dp_dba = mat3_neg(pim.d_position_d_bias_accel);
    set_block_15x15(&mut j_i, 6, 12, &neg_dp_dba);

    // ── ∂r_p / ∂ω_j = 0, ∂r_p / ∂v_j = 0 (no coupling via position) ───
    // ∂r_p / ∂t_j — wait, p_j IS the translation of pose_j, so:
    // ∂r_p / ∂ρ_j (translation part) = R_i^T
    // This goes in cols 3-5 of j_j (ρ = [upsilon|omega], upsilon cols 3-5)
    // NOTE: adjust col offset to match your [ρ|ω] = [upsilon(3)|omega(3)] layout
    set_block_15x15(&mut j_j, 6, 3, &r_it_mat); // ∂r_p/∂p_j = R_i^T (via translation)

    // ── ∂r_bg / ∂bg_i = -I, ∂r_bg / ∂bg_j = +I ─────────────────────────
    set_block_15x15(&mut j_i, 9, 9, &mat3_neg(Mat3F64::IDENTITY));
    set_block_15x15(&mut j_j, 9, 9, &Mat3F64::IDENTITY);

    // ── ∂r_ba / ∂ba_i = -I, ∂r_ba / ∂ba_j = +I ─────────────────────────
    set_block_15x15(&mut j_i, 12, 12, &mat3_neg(Mat3F64::IDENTITY));
    set_block_15x15(&mut j_j, 12, 12, &Mat3F64::IDENTITY);

    (residual, j_i, j_j)
}

/// Accumulate J_a^T · Ω · J_b into m_mat[row_base:, col_base:] (15×15 block).
/// If `update_rhs`, also subtract J_a^T · Ω · r from m_vec[row_base:].
fn accum_jt_omega_j(
    m_mat: &mut Mat<f64>,
    m_vec: &mut Vec<f64>,
    row_base: usize,
    col_base: usize,
    j_a: &[f64; 225],    // 15×15 row-major
    j_b: &[f64; 225],    // 15×15 row-major
    omega: &[f64; 225],  // 15×15 row-major information matrix
    res: &[f64; 15],
    update_rhs: bool,
) {
    // omega_jb[r,c] = Σ_k omega[r,k] · j_b[k,c]
    let mut omega_jb = [0.0f64; 225];
    for r in 0..15 { for c in 0..15 {
        let mut s = 0.0f64;
        for k in 0..15 { s += omega[r*15+k] * j_b[k*15+c]; }
        omega_jb[r*15+c] = s;
    }}

    // H[row_base+i, col_base+j] += Σ_k j_a[k,i] · omega_jb[k,j]
    for i in 0..15 { for j in 0..15 {
        let mut s = 0.0f64;
        for k in 0..15 { s += j_a[k*15+i] * omega_jb[k*15+j]; }
        m_mat[(row_base+i, col_base+j)] += s;
    }}

    // g[row_base+i] -= Σ_k j_a[k,i] · (Σ_m omega[k,m] · res[m])
    if update_rhs {
        let mut omega_r = [0.0f64; 15];
        for k in 0..15 { for m in 0..15 { omega_r[k] += omega[k*15+m] * res[m]; }}
        for i in 0..15 {
            let mut s = 0.0f64;
            for k in 0..15 { s += j_a[k*15+i] * omega_r[k]; }
            m_vec[row_base+i] -= s;
        }
    }
}

/// Diagonal information matrix from preintegrated IMU covariances.
/// Uses diagonal approximation (off-diagonal terms ignored).
/// Replace with full 15×15 inversion once FD tests pass.
fn imu_information_matrix(pim: &PreintegratedImu) -> [f64; 225] {
    let mut omega = [0.0f64; 225];
    // Nav block (9×9) from pim.covariance (column-major [f64;81]).
    // Diagonal of column-major: element [i,i] is at index i + i*9.
    for i in 0..9 {
        let var = pim.covariance[i + i*9];
        omega[i*15+i] = if var > 1e-20 { 1.0/var } else { 1e6 };
    }
    // Bias block (6×6) from pim.bias_covariance (column-major [f64;36]).
    for i in 0..6 {
        let var = pim.bias_covariance[i + i*6];
        omega[(9+i)*15+(9+i)] = if var > 1e-20 { 1.0/var } else { 1e6 };
    }
    omega
}

/// f64 version of invert_3x3 (the existing one uses f32).
fn invert_3x3_f64(m: &[f64; 9]) -> Option<[f64; 9]> {
    let det = m[0]*(m[4]*m[8]-m[5]*m[7])
            - m[1]*(m[3]*m[8]-m[5]*m[6])
            + m[2]*(m[3]*m[7]-m[4]*m[6]);
    if det.abs() < 1e-20 { return None; }
    let inv = 1.0/det;
    Some([
        (m[4]*m[8]-m[5]*m[7])*inv, (m[2]*m[7]-m[1]*m[8])*inv, (m[1]*m[5]-m[2]*m[4])*inv,
        (m[5]*m[6]-m[3]*m[8])*inv, (m[0]*m[8]-m[2]*m[6])*inv, (m[2]*m[3]-m[0]*m[5])*inv,
        (m[3]*m[7]-m[4]*m[6])*inv, (m[1]*m[6]-m[0]*m[7])*inv, (m[0]*m[4]-m[1]*m[3])*inv,
    ])
}

/// f64 version of matmul_6x3_3x3.
fn matmul_6x3_3x3_f64(a: &[f64; 18], b: &[f64; 9]) -> [f64; 18] {
    let mut out = [0.0f64; 18];
    for i in 0..6 { for k in 0..3 {
        let mut s = 0.0f64;
        for r in 0..3 { s += a[i*3+r] * b[r*3+k]; }
        out[i*3+k] = s;
    }}
    out
}

#[inline]
fn set_block_15x15(m: &mut [f64; 225], row: usize, col: usize, block: &Mat3F64) {
    let cols = block.to_cols_array(); // column-major from glam
    for c in 0..3 {
        for r in 0..3 {
            m[(row + r) * 15 + (col + c)] = cols[c * 3 + r];
        }
    }
}

#[inline]
fn mat3_mul(a: &Mat3F64, b: &Mat3F64) -> Mat3F64 {
    Mat3F64(a.mul_mat3(b))
}

#[inline]
fn mat3_neg(a: Mat3F64) -> Mat3F64 {
    mat3_scalar_f64(&a, -1.0)
}

#[inline]
fn mat3_scalar_f64(m: &Mat3F64, s: f64) -> Mat3F64 {
    Mat3F64(m.mul_scalar(s))
}

fn ata_6x6_into(acc: &mut [f32; 36], j: &[f32; 12]) {
    // acc += J.T @ J  where J is 2×6 row-major.
    let r0 = &j[0..6];
    let r1 = &j[6..12];
    for i in 0..6 {
        for k in 0..6 {
            acc[i * 6 + k] += r0[i] * r0[k] + r1[i] * r1[k];
        }
    }
}

#[inline]
fn ata_3x3_into(acc: &mut [f32; 9], j: &[f32; 6]) {
    let r0 = &j[0..3];
    let r1 = &j[3..6];
    for i in 0..3 {
        for k in 0..3 {
            acc[i * 3 + k] += r0[i] * r0[k] + r1[i] * r1[k];
        }
    }
}

#[inline]
fn atb_6x3_into(acc: &mut [f32; 18], jp: &[f32; 12], jx: &[f32; 6]) {
    // acc += J_pose.T @ J_point  →  6 × 3 row-major.
    let jp0 = &jp[0..6];
    let jp1 = &jp[6..12];
    let jx0 = &jx[0..3];
    let jx1 = &jx[3..6];
    for i in 0..6 {
        for k in 0..3 {
            acc[i * 3 + k] += jp0[i] * jx0[k] + jp1[i] * jx1[k];
        }
    }
}

#[inline]
fn atb_6x1_into(acc: &mut [f32; 6], j: &[f32; 12], r: &[f32; 2]) {
    // acc -= J.T @ r  (note negative for gradient convention).
    for i in 0..6 {
        acc[i] -= j[i] * r[0] + j[6 + i] * r[1];
    }
}

#[inline]
fn atb_3x1_into(acc: &mut [f32; 3], j: &[f32; 6], r: &[f32; 2]) {
    for i in 0..3 {
        acc[i] -= j[i] * r[0] + j[3 + i] * r[1];
    }
}

/// Invert a 3×3 row-major matrix. Returns None if singular.
fn invert_3x3(m: &[f32; 9]) -> Option<[f32; 9]> {
    let a = m[0];
    let b = m[1];
    let c = m[2];
    let d = m[3];
    let e = m[4];
    let f = m[5];
    let g = m[6];
    let h = m[7];
    let i = m[8];
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-20 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        (e * i - f * h) * inv_det,
        (c * h - b * i) * inv_det,
        (b * f - c * e) * inv_det,
        (f * g - d * i) * inv_det,
        (a * i - c * g) * inv_det,
        (c * d - a * f) * inv_det,
        (d * h - e * g) * inv_det,
        (b * g - a * h) * inv_det,
        (a * e - b * d) * inv_det,
    ])
}

#[inline]
fn matmul_6x3_3x3(a: &[f32; 18], b: &[f32; 9]) -> [f32; 18] {
    let mut out = [0.0_f32; 18];
    for i in 0..6 {
        for k in 0..3 {
            let mut s = 0.0_f32;
            for r in 0..3 {
                s += a[i * 3 + r] * b[r * 3 + k];
            }
            out[i * 3 + k] = s;
        }
    }
    out
}

#[inline]
fn matvec_6x3_3(a: &[f32; 18], b: &[f32; 3]) -> [f32; 6] {
    let mut out = [0.0_f32; 6];
    for i in 0..6 {
        out[i] = a[i * 3] * b[0] + a[i * 3 + 1] * b[1] + a[i * 3 + 2] * b[2];
    }
    out
}

#[inline]
fn matvec_3x3_3(a: &[f32; 9], b: &[f32; 3]) -> [f32; 3] {
    [
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
        a[3] * b[0] + a[4] * b[1] + a[5] * b[2],
        a[6] * b[0] + a[7] * b[1] + a[8] * b[2],
    ]
}

#[inline]
fn matvec_6x3t_6(a: &[f32; 18], b: &[f32; 6]) -> [f32; 3] {
    // returns a.T @ b  →  3-vector; a is stored row-major 6×3
    let mut out = [0.0_f32; 3];
    for k in 0..3 {
        out[k] = a[k] * b[0]
            + a[3 + k] * b[1]
            + a[6 + k] * b[2]
            + a[9 + k] * b[3]
            + a[12 + k] * b[4]
            + a[15 + k] * b[5];
    }
    out
}

// ── Driver ───────────────────────────────────────────────────────────────

/// Bundle adjustment via dense Schur-complement reduction. Same external
/// contract as [`kornia_3d::ba::bundle_adjust`] but uses Schur internally:
/// the reduced 6P×6P camera system is solved with `faer`'s dense Cholesky;
/// points are recovered by back-substitution.
///
/// Currently respects `fixed_pose` and `fixed_point` flags on each
/// observation but does not yet implement `BaParams::robust` (treats as
/// identity loss). All other params (max_iterations, initial_lambda,
/// cost_tolerance, gradient_tolerance) are honoured.
pub fn bundle_adjust_schur(
    poses: &[Pose3d],
    points: &[Vec3F64],
    observations: &[BaObservation],
    camera: &PinholeCamera,
    params: &BaParams,
) -> Result<BaResult, SchurBaError> {
    bundle_adjust_schur_with_priors(poses, points, observations, camera, params, None)
}

/// Bundle adjustment via dense Schur-complement reduction with optional
/// per-pose translation priors.
///
/// Identical to [`bundle_adjust_schur`] but accepts a slice of
/// `Option<BaPosePrior>` of length `poses.len()` (entries may be `None` for
/// unconstrained poses). When a prior is present for pose `i`, the BA cost
/// gains a position residual
///
/// ```text
///     r_pos_i = (C_i_world − prior_i.center_world) / prior_i.sigma
/// ```
///
/// where `C_i_world = -R^T · t`. This anchors all three world-frame axes of
/// the pose translation simultaneously — the durable fix for lateral /
/// vertical drift that the per-observation depth residual alone (which only
/// constrains cam-frame Z) cannot close.
///
/// The pose-prior Jacobian decomposes into a 3×6 block per pose with no
/// coupling to point variables, so it augments only the on-diagonal
/// camera-block A_ii in the Schur reduction (B and C are untouched).
///
/// Poses marked fixed via `BaObservation::fixed_pose` have no free
/// parameters; any prior on them is silently ignored.
pub fn bundle_adjust_schur_with_priors(
    poses: &[Pose3d],
    points: &[Vec3F64],
    observations: &[BaObservation],
    camera: &PinholeCamera,
    params: &BaParams,
    pose_priors: Option<&[Option<BaPosePrior>]>,
) -> Result<BaResult, SchurBaError> {
    // Validate prior slice length matches poses.
    if let Some(pp) = pose_priors {
        if pp.len() != poses.len() {
            return Err(SchurBaError::Ba(BaError::InvalidInput(format!(
                "pose_priors length {} != poses length {}",
                pp.len(),
                poses.len()
            ))));
        }
    }
    let p_total = poses.len();
    let n_total = points.len();

    // Index map: which poses / points are touched by any free observation.
    let mut pose_is_free = vec![false; p_total];
    let mut point_is_free = vec![false; n_total];
    for obs in observations {
        if obs.pose_idx >= p_total || obs.point_idx >= n_total {
            continue;
        }
        if !obs.fixed_pose {
            pose_is_free[obs.pose_idx] = true;
        }
        if !obs.fixed_point {
            point_is_free[obs.point_idx] = true;
        }
    }
    let pose_local: Vec<i64> = {
        let mut v = vec![-1_i64; p_total];
        let mut next = 0;
        for i in 0..p_total {
            if pose_is_free[i] {
                v[i] = next;
                next += 1;
            }
        }
        v
    };
    let point_local: Vec<i64> = {
        let mut v = vec![-1_i64; n_total];
        let mut next = 0;
        for i in 0..n_total {
            if point_is_free[i] {
                v[i] = next;
                next += 1;
            }
        }
        v
    };
    let n_free_poses = pose_local.iter().filter(|&&x| x >= 0).count();
    let n_free_points = point_local.iter().filter(|&&x| x >= 0).count();

    if n_free_poses == 0 {
        return Err(SchurBaError::NoFreeVariables);
    }

    // Mutable state.
    let mut se3s: Vec<SE3F32> = poses.iter().map(pose_to_se3).collect();
    let mut xyz: Vec<Vec3F64> = points.to_vec();

    let mut lambda = params.initial_lambda;
    let mut prev_cost: Option<f32> = None;
    let mut iters_done = 0usize;
    let mut converged = false;

    for _iter in 0..params.max_iterations {
        iters_done += 1;

        // ── Linearise: build A, C, B (per-obs), g_pose, g_point ──────────
        // A: n_free_poses × [36] (6×6 blocks).
        // C: n_free_points × [9]  (3×3 blocks).
        // We also keep observation-aligned B blocks (6×3) so we can iterate
        // by point during the Schur reduction.
        let mut a_blocks = vec![[0.0_f32; 36]; n_free_poses];
        let mut c_blocks = vec![[0.0_f32; 9]; n_free_points];
        let mut g_pose = vec![[0.0_f32; 6]; n_free_poses];
        let mut g_point = vec![[0.0_f32; 3]; n_free_points];

        // Per-observation B contributions, grouped by point (for the Schur
        // pass). We store (pose_local_idx, B_6x3) lists per free-point index.
        let mut b_by_point: Vec<Vec<(usize, [f32; 18])>> = vec![Vec::new(); n_free_points];

        // Also record observations that touch FIXED point but FREE pose —
        // contribute to A and g_pose only, no B.
        // (Symmetric case: free point + fixed pose contributes to C and
        //  g_point only. Both we handle below.)
        // Robust-loss IRLS weight per observation. weight w = min(1, scale/‖r‖)
        // for Huber, w = scale²/(scale²+‖r‖²) for Cauchy. Identity uses w=1.
        // Apply √w to both residual and Jacobian rows (equivalent to multiplying
        // the obs's contribution to the normal equations by w).
        let robust = params.robust;
        let robust_scale = params.robust_scale_sq.sqrt().max(1e-6);
        let huber_w = |r_sq: f32| -> f32 {
            // ‖r‖ ≤ scale → w=1; else w = scale/‖r‖
            let r_norm = r_sq.sqrt();
            if r_norm <= robust_scale {
                1.0
            } else {
                robust_scale / r_norm
            }
        };
        let cauchy_w = |r_sq: f32| -> f32 {
            let s2 = robust_scale * robust_scale;
            s2 / (s2 + r_sq)
        };

        let mut cost = 0.0_f32;
        let mut n_depth_obs_iter = 0usize;

        for obs in observations {
            if obs.pose_idx >= p_total || obs.point_idx >= n_total {
                continue;
            }
            let pose = &se3s[obs.pose_idx];
            let point = &xyz[obs.point_idx];
            let (mut r, mut j_pose, mut j_point) =
                residual_and_jacobians(pose, point, obs.pixel, camera);
            let r_sq = r[0] * r[0] + r[1] * r[1];

            // IRLS weight; apply √w to r and J.
            let w = match robust {
                RobustKernelKind::Identity => 1.0,
                RobustKernelKind::Huber => huber_w(r_sq),
                RobustKernelKind::Cauchy | RobustKernelKind::Tukey => cauchy_w(r_sq),
            };
            if w != 1.0 {
                let sw = w.sqrt();
                r[0] *= sw;
                r[1] *= sw;
                for v in j_pose.iter_mut() {
                    *v *= sw;
                }
                for v in j_point.iter_mut() {
                    *v *= sw;
                }
            }
            cost += 0.5 * (r[0] * r[0] + r[1] * r[1]);

            let pli = pose_local[obs.pose_idx];
            let xli = point_local[obs.point_idx];

            if pli >= 0 {
                let pli = pli as usize;
                ata_6x6_into(&mut a_blocks[pli], &j_pose);
                atb_6x1_into(&mut g_pose[pli], &j_pose, &r);
            }
            if xli >= 0 {
                let xli = xli as usize;
                ata_3x3_into(&mut c_blocks[xli], &j_point);
                atb_3x1_into(&mut g_point[xli], &j_point, &r);
            }
            if pli >= 0 && xli >= 0 {
                let mut b_block = [0.0_f32; 18];
                atb_6x3_into(&mut b_block, &j_pose, &j_point);
                b_by_point[xli as usize].push((pli as usize, b_block));
            }

            // ── Depth residual (optional metric anchor) ─────────────────
            // r_z = (Z_pred − d_meas) / σ_depth
            // ∂Z/∂ρ  = e_z   (translation tangent contributes 1 to z)
            // ∂Z/∂ω  = row 2 of S = -R · skew(p_w)
            // ∂Z/∂Xw = row 2 of R
            // We treat the depth residual as a single extra row in the
            // stacked Jacobian, weighted by 1/σ. Its outer products are
            // added to A_p, C_p, B as for any other residual.
            if let Some(d_meas) = obs.depth_meas {
                let sigma = obs.depth_sigma.max(1e-6);
                let inv_sigma = 1.0_f32 / sigma;

                // Recompute Z_pred + jacobian rows. We need the same z-clamp
                // semantics, and the geometry-only Jacobians (no projection
                // coefficients a0/b1/a2/b2).
                let pw = Vec3AF32::new(point.x as f32, point.y as f32, point.z as f32);
                let pc = *pose * pw;
                let z_pred = if pc.z.abs() < MIN_Z {
                    if pc.z >= 0.0 {
                        MIN_Z
                    } else {
                        -MIN_Z
                    }
                } else {
                    pc.z
                };

                // Depth residual (scaled by 1/σ).
                let r_z = (z_pred - d_meas) * inv_sigma;

                // J rows (1×6 pose, 1×3 point), all scaled by 1/σ.
                let rm = pose.r.matrix();
                let r20 = rm.col(0).z;
                let r21 = rm.col(1).z;
                let r22 = rm.col(2).z;
                let (px, py, pz) = (pw.x, pw.y, pw.z);
                // Row 2 of S = -R · skew(p_w):
                //   col0: -pz·r21 + py·r22
                //   col1:  pz·r20 - px·r22
                //   col2: -py·r20 + px·r21
                let s20 = -pz * r21 + py * r22;
                let s21 = pz * r20 - px * r22;
                let s22 = -py * r20 + px * r21;

                // J_pose_depth (1×6): [ρ(0,0,1) | ω(s20, s21, s22)] / σ
                let jpd = [
                    0.0_f32 * inv_sigma,
                    0.0_f32 * inv_sigma,
                    1.0_f32 * inv_sigma,
                    s20 * inv_sigma,
                    s21 * inv_sigma,
                    s22 * inv_sigma,
                ];
                // J_point_depth (1×3): [r20, r21, r22] / σ
                let jxd = [r20 * inv_sigma, r21 * inv_sigma, r22 * inv_sigma];

                // ── Apply IRLS robust weight to the depth residual ────────
                // The depth residual is a single scalar r_z (already scaled by
                // 1/σ_depth). Use the same Huber/Cauchy gate as the
                // reprojection path so outlier depth measurements (e.g.
                // boundary mis-samples) do not dominate the normal equations.
                // The gate uses ‖r_z‖² of the *whitened* residual, matching
                // the χ² interpretation (ORB-SLAM3 §IV.B uses χ²=7.815 for
                // 3-DoF RGB-D; we reuse `robust_scale_sq` for simplicity).
                let r_sq_d = r_z * r_z;
                let w_d = match robust {
                    RobustKernelKind::Identity => 1.0,
                    RobustKernelKind::Huber => huber_w(r_sq_d),
                    RobustKernelKind::Cauchy | RobustKernelKind::Tukey => cauchy_w(r_sq_d),
                };
                cost += 0.5 * w_d * r_sq_d;
                n_depth_obs_iter += 1;

                // Accumulate into A (6×6) — w · outer product jpd·jpdᵀ.
                if pli >= 0 {
                    let pli_u = pli as usize;
                    let ab = &mut a_blocks[pli_u];
                    for i in 0..6 {
                        for k in 0..6 {
                            ab[i * 6 + k] += w_d * jpd[i] * jpd[k];
                        }
                    }
                    // g_pose -= w · jpdᵀ · r_z
                    let gp = &mut g_pose[pli_u];
                    for i in 0..6 {
                        gp[i] -= w_d * jpd[i] * r_z;
                    }
                }
                // Accumulate into C (3×3) — w · outer product jxd·jxdᵀ.
                if xli >= 0 {
                    let xli_u = xli as usize;
                    let cb = &mut c_blocks[xli_u];
                    for i in 0..3 {
                        for k in 0..3 {
                            cb[i * 3 + k] += w_d * jxd[i] * jxd[k];
                        }
                    }
                    let gx = &mut g_point[xli_u];
                    for i in 0..3 {
                        gx[i] -= w_d * jxd[i] * r_z;
                    }
                }
                // Accumulate into B (6×3) — w · jpd·jxdᵀ.
                if pli >= 0 && xli >= 0 {
                    let mut b_block = [0.0_f32; 18];
                    for i in 0..6 {
                        for k in 0..3 {
                            b_block[i * 3 + k] = w_d * jpd[i] * jxd[k];
                        }
                    }
                    b_by_point[xli as usize].push((pli as usize, b_block));
                }
            }
        }
        let _ = n_depth_obs_iter; // currently unused; reserved for future telemetry

        // ── Per-pose translation prior (3-D position residual) ──────────────
        // For each pose i with a Some(prior), contribute a 3-row residual
        //
        //     r_pos = (C - C_prior) / σ
        //
        // with C = -R^T · t (camera centre in world frame). Jacobian wrt the
        // pose tangent ξ = [ρ; ω] is
        //
        //     ∂C/∂ρ = -I                 (3×3)
        //     ∂C/∂ω = [C]_×              (3×3, skew of C)
        //
        // derived from the right-perturbation retract `T·exp(ξ)` matching the
        // convention used by `residual_and_jacobians` above (see ReprojFactor
        // docs). With no coupling to point variables, this only augments the
        // pose-block A_ii and g_pose[i]; B and C in the Schur reduction are
        // untouched.
        if let Some(pp_slice) = pose_priors {
            for i_global in 0..p_total {
                let Some(prior) = pp_slice[i_global] else {
                    continue;
                };
                let pli = pose_local[i_global];
                if pli < 0 {
                    // Pose fixed — prior is moot.
                    continue;
                }
                let pli_u = pli as usize;
                let sigma = prior.sigma.max(1e-6);
                let inv_sigma = 1.0_f32 / sigma;

                // Camera centre C = -R^T · t.
                let pose = &se3s[i_global];
                let rm = pose.r.matrix();
                let t = pose.t;
                // R^T · t (i.e. R-transpose-times-t — apply R as world←cam to t).
                // rm.col(j) is column j of R (cam→world if you read it as R^T … but
                // our convention has R as world→cam). So R^T · t = sum over rows.
                // R^T_row0 = (r00, r10, r20) = R.col(0); so R^T · t = (R.col(0)·t,
                // R.col(1)·t, R.col(2)·t).
                let r_col0 = rm.col(0);
                let r_col1 = rm.col(1);
                let r_col2 = rm.col(2);
                let rt_t_x = r_col0.x * t.x + r_col0.y * t.y + r_col0.z * t.z;
                let rt_t_y = r_col1.x * t.x + r_col1.y * t.y + r_col1.z * t.z;
                let rt_t_z = r_col2.x * t.x + r_col2.y * t.y + r_col2.z * t.z;
                let c_pred = [-rt_t_x, -rt_t_y, -rt_t_z];

                // Residual r_pos = (C − C_prior) / σ  (3-vector).
                let r_pos = [
                    (c_pred[0] - prior.center_world[0]) * inv_sigma,
                    (c_pred[1] - prior.center_world[1]) * inv_sigma,
                    (c_pred[2] - prior.center_world[2]) * inv_sigma,
                ];

                // ── Apply IRLS robust weight to the pose-prior residual ───
                // The gate uses ‖r_pos‖² (sum of three whitened squared
                // components). This dampens single-pose VO glitches (a
                // mis-aligned chain step) so they cannot dominate the prior
                // term. We reuse `robust_scale_sq` for consistency with the
                // reprojection path; the residual is already whitened by 1/σ
                // so the gate is on the χ²-equivalent magnitude.
                let r_sq_p = r_pos[0] * r_pos[0] + r_pos[1] * r_pos[1] + r_pos[2] * r_pos[2];
                let w_p = match robust {
                    RobustKernelKind::Identity => 1.0,
                    RobustKernelKind::Huber => huber_w(r_sq_p),
                    RobustKernelKind::Cauchy | RobustKernelKind::Tukey => cauchy_w(r_sq_p),
                };
                cost += 0.5 * w_p * r_sq_p;

                // Jacobian (3×6), all scaled by 1/σ:
                //   ∂C/∂ρ = -I
                //   ∂C/∂ω = [C]_× =  [ 0   -cz   cy ]
                //                    [ cz   0   -cx ]
                //                    [-cy   cx   0  ]
                let cx_ = c_pred[0];
                let cy_ = c_pred[1];
                let cz_ = c_pred[2];
                // Row-major 3×6 layout: [ρ(3) | ω(3)] per row.
                let j_pose_prior: [f32; 18] = [
                    // Row 0 (dCx)
                    -inv_sigma,
                    0.0,
                    0.0,
                    0.0,
                    -cz_ * inv_sigma,
                    cy_ * inv_sigma,
                    // Row 1 (dCy)
                    0.0,
                    -inv_sigma,
                    0.0,
                    cz_ * inv_sigma,
                    0.0,
                    -cx_ * inv_sigma,
                    // Row 2 (dCz)
                    0.0,
                    0.0,
                    -inv_sigma,
                    -cy_ * inv_sigma,
                    cx_ * inv_sigma,
                    0.0,
                ];

                // Accumulate into A_ii (6×6) — w · Σ_r J_r.T · J_r over 3 rows.
                let ab = &mut a_blocks[pli_u];
                for r_idx in 0..3 {
                    let row = &j_pose_prior[r_idx * 6..(r_idx + 1) * 6];
                    for ii in 0..6 {
                        for kk in 0..6 {
                            ab[ii * 6 + kk] += w_p * row[ii] * row[kk];
                        }
                    }
                }
                // RHS: g_pose -= w · Σ_r J_r.T · r_pos[r]
                let gp = &mut g_pose[pli_u];
                for r_idx in 0..3 {
                    let row = &j_pose_prior[r_idx * 6..(r_idx + 1) * 6];
                    for ii in 0..6 {
                        gp[ii] -= w_p * row[ii] * r_pos[r_idx];
                    }
                }
            }
        }

        // Cost convergence (post-step convergence will follow successful steps below).
        if let Some(pc) = prev_cost {
            // Only declare convergence here on a *successful* step path; we'll
            // do that after accepting a step. For now, just log.
            let _ = pc;
        }

        // ── Apply LM damping: A[i] += λ·I, C[j] += λ·I ──────────────────
        for ab in &mut a_blocks {
            for d in 0..6 {
                ab[d * 6 + d] += lambda;
            }
        }
        for cb in &mut c_blocks {
            for d in 0..3 {
                cb[d * 3 + d] += lambda;
            }
        }

        // ── Build M (dense 6Pf × 6Pf) + m (6Pf) ─────────────────────────
        let dim = n_free_poses * 6;
        let mut m_mat = Mat::<f64>::zeros(dim, dim);
        let mut m_vec = vec![0.0_f64; dim];

        // Place A blocks on diagonal of M.
        for (k, ab) in a_blocks.iter().enumerate() {
            for i in 0..6 {
                for j in 0..6 {
                    m_mat[(k * 6 + i, k * 6 + j)] = ab[i * 6 + j] as f64;
                }
            }
            for i in 0..6 {
                m_vec[k * 6 + i] = g_pose[k][i] as f64;
            }
        }

        // For each free point j: invert C_j, accumulate Schur correction
        //   M[i1, i2] -= B[i1, j] · C_j⁻¹ · B[i2, j].T
        //   m[i]     -= B[i, j]  · C_j⁻¹ · g_point[j]
        // Skip if C_j is singular (rare, but be safe).
        let mut c_inv_blocks: Vec<Option<[f32; 9]>> = Vec::with_capacity(n_free_points);
        for cb in &c_blocks {
            c_inv_blocks.push(invert_3x3(cb));
        }

        for (j, b_for_j) in b_by_point.iter().enumerate() {
            let Some(c_inv_j) = c_inv_blocks[j] else {
                continue;
            };
            // Pre-compute B_i · C⁻¹ for each i in this point's edge list.
            let bc: Vec<(usize, [f32; 18])> = b_for_j
                .iter()
                .map(|(i_loc, b)| (*i_loc, matmul_6x3_3x3(b, &c_inv_j)))
                .collect();

            // RHS: m[i] -= (B_i · C⁻¹) · g_point[j]
            let gp = g_point[j];
            for (i_loc, bc_block) in &bc {
                let bc_g = matvec_6x3_3(bc_block, &gp);
                let base = i_loc * 6;
                for r in 0..6 {
                    m_vec[base + r] -= bc_g[r] as f64;
                }
            }

            // LHS: M[i1, i2] -= (B_i1 · C⁻¹) · B_i2.T   (6×6 block)
            for (idx1, (i1_loc, bc1)) in bc.iter().enumerate() {
                for (idx2, (i2_loc, _bc2_unused)) in bc.iter().enumerate() {
                    let b2 = &b_for_j[idx2].1;
                    // (6×3) @ (3×6) — bc1 (6×3) times b2.T (3×6).
                    // Compute element (r, c): sum_k bc1[r, k] · b2[c, k]
                    let row0 = i1_loc * 6;
                    let col0 = i2_loc * 6;
                    let _ = idx1;
                    let _ = idx2;
                    for r in 0..6 {
                        for c in 0..6 {
                            let mut s = 0.0_f32;
                            for k in 0..3 {
                                s += bc1[r * 3 + k] * b2[c * 3 + k];
                            }
                            m_mat[(row0 + r, col0 + c)] -= s as f64;
                        }
                    }
                }
            }
        }

        // ── Solve M · δ_pose = m via Cholesky ────────────────────────────
        // Symmetrize numerically (the construction above should already be
        // symmetric to within roundoff; do an average to guarantee).
        for i in 0..dim {
            for j in (i + 1)..dim {
                let avg = 0.5 * (m_mat[(i, j)] + m_mat[(j, i)]);
                m_mat[(i, j)] = avg;
                m_mat[(j, i)] = avg;
            }
        }
        let chol = match m_mat.llt(faer::Side::Lower) {
            Ok(c) => c,
            Err(e) => {
                // Bump damping and retry next outer iteration.
                lambda *= 10.0;
                if lambda > 1e10 {
                    return Err(SchurBaError::CholeskyFailed(format!("{e:?}")));
                }
                continue;
            }
        };
        // RHS as faer column.
        let m_col = Mat::<f64>::from_fn(dim, 1, |i, _| m_vec[i]);
        let d_pose_col = chol.solve(&m_col);

        // ── Back-substitute for points: δ_x[j] = C⁻¹ (g_x - B.T · δ_p) ──
        let mut d_pose = vec![0.0_f64; dim];
        for i in 0..dim {
            d_pose[i] = d_pose_col[(i, 0)];
        }
        let mut d_point = vec![[0.0_f32; 3]; n_free_points];
        for (j, b_for_j) in b_by_point.iter().enumerate() {
            let Some(c_inv_j) = c_inv_blocks[j] else {
                continue;
            };
            // rhs = g_point[j] - sum_i B[i, j].T · δ_pose[i]
            let mut rhs = g_point[j];
            for (i_loc, b_block) in b_for_j {
                let mut dp6 = [0.0_f32; 6];
                let base = i_loc * 6;
                for r in 0..6 {
                    dp6[r] = d_pose[base + r] as f32;
                }
                let contrib = matvec_6x3t_6(b_block, &dp6);
                for c in 0..3 {
                    rhs[c] -= contrib[c];
                }
            }
            d_point[j] = matvec_3x3_3(&c_inv_j, &rhs);
        }

        // ── Trial: retract poses, add to points, recompute cost ─────────
        let mut se3s_trial = se3s.clone();
        for i_global in 0..p_total {
            let pli = pose_local[i_global];
            if pli < 0 {
                continue;
            }
            let pli = pli as usize;
            let delta: [f32; 6] = [
                d_pose[pli * 6] as f32,
                d_pose[pli * 6 + 1] as f32,
                d_pose[pli * 6 + 2] as f32,
                d_pose[pli * 6 + 3] as f32,
                d_pose[pli * 6 + 4] as f32,
                d_pose[pli * 6 + 5] as f32,
            ];
            se3s_trial[i_global] = se3s[i_global].retract(&delta);
        }
        let mut xyz_trial = xyz.clone();
        for i_global in 0..n_total {
            let xli = point_local[i_global];
            if xli < 0 {
                continue;
            }
            let xli = xli as usize;
            let dp = d_point[xli];
            xyz_trial[i_global] = Vec3F64::new(
                xyz[i_global].x + dp[0] as f64,
                xyz[i_global].y + dp[1] as f64,
                xyz[i_global].z + dp[2] as f64,
            );
        }

        let mut new_cost = 0.0_f32;
        for obs in observations {
            if obs.pose_idx >= p_total || obs.point_idx >= n_total {
                continue;
            }
            let pose = &se3s_trial[obs.pose_idx];
            let point = &xyz_trial[obs.point_idx];
            let (r, _, _) = residual_and_jacobians(pose, point, obs.pixel, camera);
            let r_sq = r[0] * r[0] + r[1] * r[1];
            let w = match robust {
                RobustKernelKind::Identity => 1.0,
                RobustKernelKind::Huber => huber_w(r_sq),
                RobustKernelKind::Cauchy | RobustKernelKind::Tukey => cauchy_w(r_sq),
            };
            new_cost += 0.5 * w * r_sq;

            // Depth residual contribution to trial cost (same Huber/Cauchy
            // weighting as the linearisation pass, so accept/reject decisions
            // reflect the robust loss).
            if let Some(d_meas) = obs.depth_meas {
                let sigma = obs.depth_sigma.max(1e-6);
                let pw = Vec3AF32::new(point.x as f32, point.y as f32, point.z as f32);
                let pc = *pose * pw;
                let z_pred = if pc.z.abs() < MIN_Z {
                    if pc.z >= 0.0 {
                        MIN_Z
                    } else {
                        -MIN_Z
                    }
                } else {
                    pc.z
                };
                let r_z = (z_pred - d_meas) / sigma;
                let r_sq_d = r_z * r_z;
                let w_d = match robust {
                    RobustKernelKind::Identity => 1.0,
                    RobustKernelKind::Huber => huber_w(r_sq_d),
                    RobustKernelKind::Cauchy | RobustKernelKind::Tukey => cauchy_w(r_sq_d),
                };
                new_cost += 0.5 * w_d * r_sq_d;
            }
        }

        // Pose-prior contribution to trial cost.
        if let Some(pp_slice) = pose_priors {
            for i_global in 0..p_total {
                let Some(prior) = pp_slice[i_global] else {
                    continue;
                };
                if pose_local[i_global] < 0 {
                    continue;
                }
                let sigma = prior.sigma.max(1e-6);
                let inv_sigma = 1.0_f32 / sigma;
                let pose = &se3s_trial[i_global];
                let rm = pose.r.matrix();
                let t = pose.t;
                let r_col0 = rm.col(0);
                let r_col1 = rm.col(1);
                let r_col2 = rm.col(2);
                let rt_t_x = r_col0.x * t.x + r_col0.y * t.y + r_col0.z * t.z;
                let rt_t_y = r_col1.x * t.x + r_col1.y * t.y + r_col1.z * t.z;
                let rt_t_z = r_col2.x * t.x + r_col2.y * t.y + r_col2.z * t.z;
                let c_pred = [-rt_t_x, -rt_t_y, -rt_t_z];
                let r0 = (c_pred[0] - prior.center_world[0]) * inv_sigma;
                let r1 = (c_pred[1] - prior.center_world[1]) * inv_sigma;
                let r2 = (c_pred[2] - prior.center_world[2]) * inv_sigma;
                // Match the linearisation pass's Huber/Cauchy gate so
                // accept/reject reflects the robust loss.
                let r_sq_p = r0 * r0 + r1 * r1 + r2 * r2;
                let w_p = match robust {
                    RobustKernelKind::Identity => 1.0,
                    RobustKernelKind::Huber => huber_w(r_sq_p),
                    RobustKernelKind::Cauchy | RobustKernelKind::Tukey => cauchy_w(r_sq_p),
                };
                new_cost += 0.5 * w_p * r_sq_p;
            }
        }

        if new_cost < cost {
            // Accept step.
            let rel = if cost > 1e-12 {
                (cost - new_cost) / cost
            } else {
                0.0
            };
            se3s = se3s_trial;
            xyz = xyz_trial;
            prev_cost = Some(new_cost);
            lambda = (lambda / 3.0).max(1e-8);
            if rel < params.cost_tolerance {
                converged = true;
                break;
            }
        } else {
            // Reject — bump damping and retry.
            lambda *= 10.0;
            if lambda > 1e10 {
                break;
            }
        }
    }

    // Pack results.
    let mut out_poses = Vec::with_capacity(p_total);
    for i in 0..p_total {
        if pose_is_free[i] {
            out_poses.push(se3_to_pose(&se3s[i]));
        } else {
            out_poses.push(poses[i]);
        }
    }
    let mut out_points = Vec::with_capacity(n_total);
    for i in 0..n_total {
        if point_is_free[i] {
            out_points.push(xyz[i]);
        } else {
            out_points.push(points[i]);
        }
    }

    Ok(BaResult {
        poses: out_poses,
        points: out_points,
        iterations: iters_done,
        converged,
    })
}

/// Visual-inertial bundle adjustment via dense Schur-complement reduction.
    ///
    /// Jointly optimises keyframe poses, velocities, and IMU biases (15 DOF each)
    /// together with 3D landmark positions, connected by both reprojection factors
    /// and IMU preintegration factors.
    ///
    /// The Schur trick marginalises points exactly as in pure BA. After point
    /// elimination, IMU edge normal equations are added directly to the reduced
    /// 15P × 15P camera system before Cholesky solve. This matches Ceres's
    /// DENSE_SCHUR strategy as used in ORB-SLAM3.
    ///
    /// # Gauge freedom
    /// Fix at least one keyframe (`fixed = true`) to anchor the metric scale and
    /// world frame. The first keyframe is the natural choice.
    
    pub fn visual_inertial_bundle_adjust(
        keyframes: &[ViBaKeyframe],
        points: &[Vec3F64],
        observations: &[BaObservation],
        imu_edges: &[ImuEdge],
        camera: &PinholeCamera,
        params: &ViBaParams,
    ) -> Result<ViBaResult, ViBaError> {
        const KF_DOF: usize = 15; // keyframe dimensions

        let n_kf = keyframes.len();
        let n_pts = points.len();

        // ── Index maps: free keyframes and free points ────────────────────────
        // A keyframe is free if it is not marked fixed AND appears in at least
        // one observation or IMU edge (otherwise it has no gradient, and we
        // skip it to avoid rank deficiency).

        let mut kf_touched = vec![false; n_kf];
        for obs in observations {
            if obs.pose_idx < n_kf {kf_touched[obs.pose_idx] = true; }
        }

        for edge in imu_edges {
            if edge.from_idx < n_kf {kf_touched[edge.from_idx] = true; }
            if edge.to_idx < n_kf {kf_touched[edge.to_idx] = true; }
        }

        let kf_local: Vec<i64> = {
            let mut v = vec![-1i64; n_kf];
            let mut next = 0i64;
            for i in 0..n_kf {
                if kf_touched[i] && !keyframes[i].fixed {
                    v[i] = next;
                    next += 1;
                }
            }
            v
        };

        let point_local: Vec<i64> = {
            let mut v = vec![-1i64; n_pts];
            let mut next = 0i64;
            for obs in observations {
                if obs.point_idx < n_pts && !obs.fixed_point {
                    if v[obs.point_idx] < 0 {
                        v[obs.point_idx] = next;
                        next += 1;
                    }
                }
            }
            v
        };

        let n_free_kf  = kf_local.iter().filter(|&&x| x >= 0).count();
        let n_free_pts = point_local.iter().filter(|&&x| x >= 0).count();

        if n_free_kf == 0 {
            return Err(ViBaError::NoFreeVariables);
        }

        // Validate IMU edge indices.
        for edge in imu_edges {
            if edge.from_idx >= n_kf || edge.to_idx >= n_kf {
                return Err(ViBaError::ImuEdgeOutOfRange(edge.from_idx, edge.to_idx));
            }
        }

        // ── Mutable state (f64 throughout — IMU needs the precision) ─────────
        let mut kfs: Vec<ViBaKeyframe> = keyframes.to_vec();
        let mut xyz: Vec<Vec3F64> = points.to_vec();
        // SE3F32 mirrors for the reprojection Jacobian (matches residual_and_jacobians).
        let mut se3s: Vec<SE3F32> = kfs.iter().map(|kf| pose_to_se3(&kf.pose)).collect();

        let mut lambda = params.initial_lambda;
        let mut iters_done = 0usize;
        let mut converged = false;
        let mut final_cost = f64::MAX;

        for _iter in 0..params.max_iterations {
            iters_done += 1;
            // ── 1. Schur blocks for visual residuals ─────────────────────────
            // These are exactly the a_blocks / c_blocks / b_by_point / g_pose /
            // g_point from bundle_adjust_schur_with_priors, but a_blocks are
            // 6×6 (pose only) and we scatter them into the 15×15 keyframe slot
            // when building m_mat.

            // a_blocks[k]: 6×6 visual Hessian for free keyframe k (pose DOF only).
            let mut a_blocks  = vec![[0.0f64; 36]; n_free_kf];
            // c_blocks[j]: 3×3 point Hessian block.
            let mut c_blocks  = vec![[0.0f64; 9];  n_free_pts];
            // g_pose[k]: 6-vector RHS for free keyframe k (pose DOF).
            let mut g_pose    = vec![[0.0f64; 6];  n_free_kf];
            // g_point[j]: 3-vector RHS for free point j.
            let mut g_point   = vec![[0.0f64; 3];  n_free_pts];
            // B cross-terms grouped by free point index.
            let mut b_by_point: Vec<Vec<(usize, [f64; 18])>> = vec![Vec::new(); n_free_pts];

            let mut cost_vis = 0.0f64;

            for obs in observations {
                if obs.pose_idx >= n_kf || obs.point_idx >= n_pts { continue; }

                let pose  = &se3s[obs.pose_idx];
                let point = &xyz[obs.point_idx];
                // residual_and_jacobians is f32 internally; cast results to f64.
                let (r_f32, jp_f32, jx_f32) =
                    residual_and_jacobians(pose, point, obs.pixel, camera);

                let r  = [r_f32[0]  as f64, r_f32[1]  as f64];
                let jp: [f64; 12] = std::array::from_fn(|i| jp_f32[i] as f64);
                let jx: [f64; 6]  = std::array::from_fn(|i| jx_f32[i] as f64);

                cost_vis += 0.5 * (r[0]*r[0] + r[1]*r[1]);

                let pli = kf_local[obs.pose_idx];
                let xli = point_local[obs.point_idx];
            
                // Accumulate A block (6×6).
                if pli >= 0 {
                    let p = pli as usize;
                    let ab = &mut a_blocks[p];
                    for i in 0..6 { for k in 0..6 {
                        ab[i*6+k] += jp[i]*jp[k] + jp[6+i]*jp[6+k];
                    }}
                    let gp = &mut g_pose[p];
                    for i in 0..6 { gp[i] -= jp[i]*r[0] + jp[6+i]*r[1]; }
                }

                // Accumulate C block (3×3).
                if xli >= 0 {
                    let x = xli as usize;
                    let cb = &mut c_blocks[x];
                    for i in 0..3 { for k in 0..3 {
                        cb[i*3+k] += jx[i]*jx[k] + jx[3+i]*jx[3+k];
                    }}
                    let gx = &mut g_point[x];
                    for i in 0..3 { gx[i] -= jx[i]*r[0] + jx[3+i]*r[1]; }
                }

                // Accumulate B block (6×3) for Schur.
                if pli >= 0 && xli >= 0 {
                    let mut b = [0.0f64; 18];
                    for i in 0..6 { for k in 0..3 {
                        b[i*3+k] += jp[i]*jx[k] + jp[6+i]*jx[3+k];
                    }}
                    b_by_point[xli as usize].push((pli as usize, b));
                }

            }

            // ── 2. Invert C blocks, build Schur-reduced system ───────────────
            // dim is 15*n_free_kf. Visual terms only touch the first 6 DOF of
            // each 15-DOF block; IMU terms fill the rest.
            let dim = n_free_kf * KF_DOF;
            let mut m_mat = Mat::<f64>::zeros(dim, dim);
            let mut m_vec = vec![0.0f64; dim];

            // Scatter A blocks into the upper-left 6×6 of each 15×15 slot.
            for (k, ab) in a_blocks.iter().enumerate() {
                let base = k * KF_DOF;
                for i in 0..6 { for j in 0..6 {
                    m_mat[(base+i, base+j)] = ab[i*6+j];
                }}
                for i in 0..6 { m_vec[base+i] = g_pose[k][i]; }
            }

            // Schur point elimination: M -= B C⁻¹ Bᵀ,  m -= B C⁻¹ g_point.
            // Point back-substitution only needs the first 6 elements of each
            // keyframe's delta (pose DOF), so the B blocks are 6×3 as before.

            let c_inv: Vec<Option<[f64; 9]>> = c_blocks.iter()
                .map(|cb| invert_3x3_f64(cb))
                .collect();

            for (j, b_for_j) in b_by_point.iter().enumerate() {
                let Some(ci) = c_inv[j] else { continue; };

                // Pre-multiply: BC_inv[i] = B[i,j] · C⁻¹  (6×3).
                let bc: Vec<(usize, [f64; 18])> = b_for_j.iter()
                    .map(|(i_loc, b)| (*i_loc, matmul_6x3_3x3_f64(b, &ci)))
                    .collect();

                // RHS correction: m[i] -= BC_inv[i] · g_point[j].
                for (i_loc, bc_block) in &bc {
                    let base = i_loc * KF_DOF;
                    for r in 0..6 {
                        let mut s = 0.0f64;
                        for k in 0..3 { s += bc_block[r*3+k] * g_point[j][k]; }
                        m_vec[base+r] -= s;
                    }
                }
                // LHS correction: M[i1,i2] -= BC_inv[i1] · B[i2,j]ᵀ.
                for (i1_loc, bc1) in &bc {
                    for (i2_loc, b2) in b_for_j.iter() {
                        let row0 = i1_loc * KF_DOF;
                        let col0 = i2_loc * KF_DOF;
                        for r in 0..6 { for c in 0..6 {
                            let mut s = 0.0f64;
                            for k in 0..3 { s += bc1[r*3+k] * b2[c*3+k]; }
                            m_mat[(row0+r, col0+c)] -= s;
                        }}
                    }
                }
            }

            // ── 3. IMU normal equations ───────────────────────────────────────
            // Each edge contributes J_i^T Ω J_i, J_j^T Ω J_j (diagonal 15×15
            // blocks) and J_i^T Ω J_j, J_j^T Ω J_i (off-diagonal 15×15 blocks).
            let mut cost_imu = 0.0f64;

            for edge in imu_edges {
                let i = edge.from_idx;
                let j = edge.to_idx;
                let li = kf_local[i];
                let lj = kf_local[j];
                if li < 0 && lj < 0 { continue; }

                let (res, ji, jj) = imu_residual_and_jacobians(
                    &kfs[i], &kfs[j], &edge.preintegrated, &params.gravity,
                );

                let omega = imu_information_matrix(&edge.preintegrated);

                // Cost: ½ rᵀ Ω r.
                let mut omega_r = [0.0f64; 15];
                for r in 0..15 {
                    for k in 0..15 { omega_r[r] += omega[r*15+k] * res[k]; }
                }
                for k in 0..15 { cost_imu += 0.5 * res[k] * omega_r[k]; }

                // Accumulate H_ii, g_i.
                if li >= 0 {
                    let base_i = li as usize * KF_DOF;
                    accum_jt_omega_j(&mut m_mat, &mut m_vec,
                                    base_i, base_i, &ji, &ji, &omega, &res, true);
                }
                // Accumulate H_jj, g_j.
                if lj >= 0 {
                    let base_j = lj as usize * KF_DOF;
                    accum_jt_omega_j(&mut m_mat, &mut m_vec,
                                    base_j, base_j, &jj, &jj, &omega, &res, true);
                }
                // Off-diagonal H_ij and H_ji (symmetric pair).
                if li >= 0 && lj >= 0 {
                    let base_i = li as usize * KF_DOF;
                    let base_j = lj as usize * KF_DOF;
                    accum_jt_omega_j(&mut m_mat, &mut m_vec,
                                    base_i, base_j, &ji, &jj, &omega, &res, false);
                    accum_jt_omega_j(&mut m_mat, &mut m_vec,
                                    base_j, base_i, &jj, &ji, &omega, &res, false);
                }
            }

            let cost = (cost_vis + cost_imu) as f32;

            // ── 4. LM damping on full 15P system, then Cholesky ──────────────
            // Damping goes on AFTER IMU terms so all 15 diagonal entries are
            // regularised uniformly. Do NOT damp a_blocks separately.
            for d in 0..dim {
                m_mat[(d, d)] += lambda;
            }

            // Symmetrize (should already be symmetric to roundoff).
            for i in 0..dim {
                for j in (i+1)..dim {
                    let avg = 0.5 * (m_mat[(i,j)] + m_mat[(j,i)]);
                    m_mat[(i,j)] = avg;
                    m_mat[(j,i)] = avg;
                }
            }

            let chol = match m_mat.llt(faer::Side::Lower) {
                Ok(c) => c,
                Err(e) => {
                    lambda *= 10.0;
                    if lambda > 1e10 {
                        return Err(ViBaError::CholeskyFailed(format!("{e:?}")));
                    }
                    continue;
                }
            };

            let m_col = Mat::<f64>::from_fn(dim, 1, |i, _| m_vec[i]);
            let d_full_col = chol.solve(&m_col);
            let d_full: Vec<f64> = (0..dim).map(|i| d_full_col[(i,0)]).collect();

            // ── 5. Point back-substitution (uses pose-DOF slice only) ────────
            let mut d_point = vec![[0.0f64; 3]; n_free_pts];
            for (j, b_for_j) in b_by_point.iter().enumerate() {
                let Some(ci) = c_inv[j] else { continue; };
                let mut rhs = g_point[j];
                for (i_loc, b_block) in b_for_j {
                    // Only the first 6 elements of d_full for this keyframe (pose).
                    let base = i_loc * KF_DOF;
                    let dp6: [f64; 6] = std::array::from_fn(|k| d_full[base+k]);
                    // rhs -= B[i,j]ᵀ · δ_pose[i]  (B is 6×3, so Bᵀ is 3×6).
                    for c in 0..3 {
                        let mut s = 0.0f64;
                        for r in 0..6 { s += b_block[r*3+c] * dp6[r]; }
                        rhs[c] -= s;
                    }
                }
                // δ_x = C⁻¹ · rhs.
                for r in 0..3 {
                    d_point[j][r] = ci[r*3]*rhs[0] + ci[r*3+1]*rhs[1] + ci[r*3+2]*rhs[2];
                }
            }

            // ── 6. Trial retraction ───────────────────────────────────────────
            let mut kfs_trial = kfs.clone();
            let mut se3s_trial = se3s.clone();

            for i in 0..n_kf {
                let li = kf_local[i];
                if li < 0 { continue; }
                let base = li as usize * KF_DOF;

                // Pose delta: first 6 elements [ρ(3)|ω(3)].
                let pose_delta: [f32; 6] = std::array::from_fn(|k| d_full[base+k] as f32);
                let new_se3 = se3s[i].retract(&pose_delta);
                se3s_trial[i] = new_se3;
                kfs_trial[i].pose = se3_to_pose(&new_se3);

                // Velocity: additive.
                kfs_trial[i].velocity.x += d_full[base+6];
                kfs_trial[i].velocity.y += d_full[base+7];
                kfs_trial[i].velocity.z += d_full[base+8];

                // Bias: additive.
                kfs_trial[i].bias.gyro.x  += d_full[base+9];
                kfs_trial[i].bias.gyro.y  += d_full[base+10];
                kfs_trial[i].bias.gyro.z  += d_full[base+11];
                kfs_trial[i].bias.accel.x += d_full[base+12];
                kfs_trial[i].bias.accel.y += d_full[base+13];
                kfs_trial[i].bias.accel.z += d_full[base+14];
            }

            let mut xyz_trial = xyz.clone();
            for i in 0..n_pts {
                let xli = point_local[i];
                if xli < 0 { continue; }
                let dp = d_point[xli as usize];
                xyz_trial[i].x += dp[0];
                xyz_trial[i].y += dp[1];
                xyz_trial[i].z += dp[2];
            }

            // ── 7. Trial cost (visual + IMU) ─────────────────────────────────
            let mut new_cost_vis = 0.0f64;
            for obs in observations {
                if obs.pose_idx >= n_kf || obs.point_idx >= n_pts { continue; }
                let (r_f32, _, _) = residual_and_jacobians(
                    &se3s_trial[obs.pose_idx], &xyz_trial[obs.point_idx], obs.pixel, camera,
                );
                new_cost_vis += 0.5 * (r_f32[0]*r_f32[0] + r_f32[1]*r_f32[1]) as f64;
            }
            let mut new_cost_imu = 0.0f64;
            for edge in imu_edges {
                let i = edge.from_idx;
                let j = edge.to_idx;
                if kf_local[i] < 0 && kf_local[j] < 0 { continue; }
                let (res, _, _) = imu_residual_and_jacobians(
                    &kfs_trial[i], &kfs_trial[j], &edge.preintegrated, &params.gravity,
                );
                let omega = imu_information_matrix(&edge.preintegrated);
                let mut omega_r = [0.0f64; 15];
                for r in 0..15 { for k in 0..15 { omega_r[r] += omega[r*15+k]*res[k]; }}
                for k in 0..15 { new_cost_imu += 0.5 * res[k] * omega_r[k]; }
            }
            let new_cost = (new_cost_vis + new_cost_imu) as f32;

            // ── 8. LM accept / reject ─────────────────────────────────────────
            if new_cost < cost {
                let rel = if cost > 1e-12 { (cost - new_cost) as f64 / cost as f64 } else { 0.0 };
                kfs    = kfs_trial;
                se3s   = se3s_trial;
                xyz    = xyz_trial;
                final_cost = new_cost as f64;
                lambda = (lambda / 3.0).max(1e-8);
                if rel < params.cost_tolerance {
                    converged = true;
                    break;
                }
            } else {
                lambda *= 10.0;
                if lambda > 1e10 { break; }
            }
        }

         // ── Pack output ───────────────────────────────────────────────────────
    // Fixed keyframes get their original state back; free keyframes get
    // the optimised state.
    let out_kfs: Vec<ViBaKeyframe> = (0..n_kf).map(|i| {
        if kf_local[i] >= 0 { kfs[i].clone() } else { keyframes[i].clone() }
    }).collect();
    let out_pts: Vec<Vec3F64> = (0..n_pts).map(|i| {
        if point_local[i] >= 0 { xyz[i] } else { points[i] }
    }).collect();

    Ok(ViBaResult {
        keyframes: out_kfs,
        points: out_pts,
        iterations: iters_done,
        converged,
        final_cost,
    })
}
