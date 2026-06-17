use std::collections::HashMap;

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, QuatF64, SO3F64, Vec3F64};
use kornia_sensors::imu::ImuBias;

use crate::map::{Keyframe, Map};
use crate::system::{SystemMode, SystemState};

const GRAVITY_MAGNITUDE: f64 = 9.81;

fn vec_axis(v: Vec3F64, axis: usize) -> f64 {
    match axis {
        0 => v.x,
        1 => v.y,
        2 => v.z,
        _ => unreachable!("axis must be in 0..3"),
    }
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 || a.iter().any(|row| row.len() != n) || b.len() != n {
        return None;
    }

    for i in 0..n {
        let mut pivot = i;
        for r in i + 1..n {
            if a[r][i].abs() > a[pivot][i].abs() {
                pivot = r;
            }
        }
        if a[pivot][i].abs() < 1e-12 {
            return None;
        }
        a.swap(i, pivot);
        b.swap(i, pivot);

        let diag = a[i][i];
        for value in a[i].iter_mut().skip(i) {
            *value /= diag;
        }
        b[i] /= diag;

        for r in 0..n {
            if r == i {
                continue;
            }
            let factor = a[r][i];
            if factor == 0.0 {
                continue;
            }
            let pivot_row = a[i].clone();
            for (dst, src) in a[r].iter_mut().skip(i).zip(pivot_row.iter().skip(i)) {
                *dst -= factor * *src;
            }
            b[r] -= factor * b[i];
        }
    }

    Some(b)
}

fn solve_least_squares(rows: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    if rows.is_empty() || rows.len() != rhs.len() {
        return None;
    }
    let cols = rows.first()?.len();
    if cols == 0 {
        return None;
    }

    let mut ata = vec![vec![0.0; cols]; cols];
    let mut atb = vec![0.0; cols];
    for (row, &b) in rows.iter().zip(rhs.iter()) {
        for i in 0..cols {
            atb[i] += row[i] * b;
            for j in 0..cols {
                ata[i][j] += row[i] * row[j];
            }
        }
    }

    solve_linear_system(ata, atb)
}

fn rotation_from_to(from: Vec3F64, to: Vec3F64) -> SO3F64 {
    let from = from.normalize();
    let to = to.normalize();
    let cross = from.cross(to);
    let dot = from.dot(to).clamp(-1.0, 1.0);

    if dot < -1.0 + 1e-9 {
        let perp = if from.x.abs() < 0.9 {
            Vec3F64::new(1.0, 0.0, 0.0)
        } else {
            Vec3F64::new(0.0, 1.0, 0.0)
        };
        let axis = from.cross(perp).normalize();
        return SO3F64::from_quaternion(QuatF64::from_array([axis.x, axis.y, axis.z, 0.0]));
    }

    let w = ((1.0 + dot) / 2.0).sqrt();
    let s = 1.0 / (2.0 * w);
    SO3F64::from_quaternion(QuatF64::from_array([
        cross.x * s,
        cross.y * s,
        cross.z * s,
        w,
    ]))
}

/// Configuration for visual-inertial initialization.
#[derive(Debug, Clone)]
pub struct InertialInitConfig {
    pub min_keyframes: usize,
    pub min_time_sec: f64,
    pub min_motion: f64,
}

/// Result of a successful inertial initialization.
#[derive(Debug, Clone)]
pub struct ImuInitResult {
    pub scale: f64,
    pub gravity_world: Vec3F64,
    pub velocities_world: Vec<Vec3F64>,
    pub bias: ImuBias,
}

struct InertialSolveResult {
    scale: f64,
    gravity_world: Vec3F64,
    velocities_world: Vec<Vec3F64>,
}

/// Visual-inertial initialization helpers.
pub struct InertialInitializer {
    pub config: InertialInitConfig,
}

impl InertialInitializer {
    pub fn new(config: InertialInitConfig) -> Self {
        Self { config }
    }

    pub fn ready(&self, map: &Map, start_idx: Option<usize>) -> bool {
        let Some(start_idx) = start_idx else {
            return false;
        };

        let init_kfs: Vec<&Keyframe> = map
            .keyframes()
            .iter()
            .filter(|kf| kf.frame.idx >= start_idx)
            .collect();
        if init_kfs.len() < self.config.min_keyframes {
            return false;
        }

        let imu_time: f64 = map
            .imu_edges()
            .iter()
            .filter(|edge| edge.curr_kf_idx >= start_idx)
            .map(|edge| edge.preintegrated.dt)
            .sum();
        if imu_time < self.config.min_time_sec {
            return false;
        }

        let Some(first) = init_kfs.first() else {
            return false;
        };
        let Some(last) = init_kfs.last() else {
            return false;
        };
        let first_center = first.frame.pose_world_to_cam.inverse().translation;
        let last_center = last.frame.pose_world_to_cam.inverse().translation;
        (last_center - first_center).length() >= self.config.min_motion
    }

    pub fn try_initialize(
        &self,
        map: &Map,
        imu_t_bc: Option<Pose3d>,
        imu_bias: ImuBias,
        start_idx: usize,
    ) -> Option<ImuInitResult> {
        let imu_t_bc = imu_t_bc?;

        let keyframes: Vec<&Keyframe> = map
            .keyframes()
            .iter()
            .filter(|kf| kf.frame.idx >= start_idx)
            .collect();
        let n = keyframes.len();
        if n < self.config.min_keyframes {
            return None;
        }

        let mut frame_to_local = HashMap::new();
        for (local_idx, kf) in keyframes.iter().enumerate() {
            frame_to_local.insert(kf.frame.idx, local_idx);
        }

        let mut bias = imu_bias;
        if let Some(gyro_bias) = self.estimate_gyro_bias(map, &keyframes, &frame_to_local, imu_t_bc) {
            bias.gyro = gyro_bias;
        }

        let solve_scale = !keyframes
            .first()
            .map(|kf| kf.frame.is_stereo())
            .unwrap_or(false);

        let first = self.solve_scale_gravity(
            map,
            &keyframes,
            &frame_to_local,
            imu_t_bc,
            &bias,
            None,
            solve_scale,
        )?;
        let mut gravity_dir = first.gravity_world;
        if !gravity_dir.length().is_finite() || gravity_dir.length() < 1e-6 {
            return None;
        }
        gravity_dir /= gravity_dir.length();

        let mut refined = None;
        for _ in 0..2 {
            let solved = self.solve_scale_gravity(
                map,
                &keyframes,
                &frame_to_local,
                imu_t_bc,
                &bias,
                Some(gravity_dir),
                solve_scale,
            )?;
            gravity_dir = solved.gravity_world / solved.gravity_world.length();
            refined = Some(solved);
        }
        let solved = refined?;

        if !solved.scale.is_finite() || solved.scale <= 1e-6 {
            return None;
        }

        Some(ImuInitResult {
            scale: solved.scale,
            gravity_world: gravity_dir * GRAVITY_MAGNITUDE,
            velocities_world: solved.velocities_world,
            bias,
        })
    }

    pub fn apply_initialization(
        &self,
        map: &mut Map,
        state: &mut SystemState,
        imu_bias: &mut ImuBias,
        gravity_world: &mut Vec3F64,
        init: ImuInitResult,
        start_idx: usize,
    ) {
        map.scale_world(init.scale);

        let g_est = init.gravity_world;
        let g_norm = g_est / g_est.length();
        let rwg = rotation_from_to(g_norm, Vec3F64::new(0.0, 1.0, 0.0));
        map.rotate_world(&rwg);

        let gravity_aligned = Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0);

        let mut velocity_iter = init.velocities_world.into_iter();
        for kf in map
            .keyframes_mut()
            .iter_mut()
            .filter(|kf| kf.frame.idx >= start_idx)
        {
            if let Some(velocity) = velocity_iter.next() {
                kf.velocity_world = rwg * velocity;
                kf.imu_bias = init.bias;
            }
        }

        if let Some(last_kf) = map
            .keyframes()
            .iter()
            .rfind(|kf| kf.frame.idx >= start_idx)
        {
            state.velocity_world = last_kf.velocity_world;
            state.pose_world_to_cam = last_kf.frame.pose_world_to_cam;
        }
        state.velocity = None;
        state.imu_initialized = true;
        *gravity_world = gravity_aligned;
        *imu_bias = init.bias;
    }

    pub fn activate_tracking(&self, state: &mut SystemState) {
        state.mode = SystemMode::Tracking;
    }

    fn estimate_gyro_bias(
        &self,
        map: &Map,
        keyframes: &[&Keyframe],
        frame_to_local: &HashMap<usize, usize>,
        imu_t_bc: Pose3d,
    ) -> Option<Vec3F64> {
        let r_cb = imu_t_bc.inverse().rotation;
        let mut bias = ImuBias::default();

        for _ in 0..3 {
            let mut h = vec![vec![0.0f64; 3]; 3];
            let mut b = vec![0.0f64; 3];
            let mut n_edges = 0usize;

            for edge in map.imu_edges() {
                let Some(&i) = frame_to_local.get(&edge.prev_kf_idx) else {
                    continue;
                };
                let Some(&j) = frame_to_local.get(&edge.curr_kf_idx) else {
                    continue;
                };
                if edge.preintegrated.dt <= 0.0 {
                    continue;
                }

                let r_wb_i = keyframes[i].frame.pose_world_to_cam.inverse().rotation * r_cb;
                let r_wb_j = keyframes[j].frame.pose_world_to_cam.inverse().rotation * r_cb;
                let r_vis = Mat3F64(*r_wb_i.transpose()) * r_wb_j;
                let d_rot = edge.preintegrated.delta_rotation_with_bias(&bias);
                let resid = SO3F64::from_matrix(&(Mat3F64(*d_rot.transpose()) * r_vis)).log();
                let jac = edge.preintegrated.d_rotation_d_bias_gyro.to_cols_array();
                let resid = [resid.x, resid.y, resid.z];

                for r in 0..3 {
                    for c in 0..3 {
                        for k in 0..3 {
                            h[r][c] += jac[r * 3 + k] * jac[c * 3 + k];
                        }
                    }
                    for k in 0..3 {
                        b[r] += jac[r * 3 + k] * resid[k];
                    }
                }
                n_edges += 1;
            }

            if n_edges < 2 {
                return None;
            }
            let delta = solve_linear_system(h, b)?;
            bias.gyro += Vec3F64::new(delta[0], delta[1], delta[2]);
            if (delta[0].powi(2) + delta[1].powi(2) + delta[2].powi(2)).sqrt() < 1e-8 {
                break;
            }
        }

        if !bias.gyro.length().is_finite() || bias.gyro.length() > 0.1 {
            return None;
        }
        Some(bias.gyro)
    }

    fn solve_scale_gravity(
        &self,
        map: &Map,
        keyframes: &[&Keyframe],
        frame_to_local: &HashMap<usize, usize>,
        imu_t_bc: Pose3d,
        bias: &ImuBias,
        gravity_dir: Option<Vec3F64>,
        solve_scale: bool,
    ) -> Option<InertialSolveResult> {
        let t_cb = imu_t_bc.inverse();
        let r_cb = t_cb.rotation;
        let lever = t_cb.translation;
        let n = keyframes.len();

        let gravity_basis = gravity_dir.map(|dir| {
            let pick = if dir.x.abs() < 0.9 {
                Vec3F64::new(1.0, 0.0, 0.0)
            } else {
                Vec3F64::new(0.0, 1.0, 0.0)
            };
            let b1 = dir.cross(pick).normalize();
            let b2 = dir.cross(b1).normalize();
            (b1, b2)
        });
        let n_gravity_cols = if gravity_basis.is_some() { 2 } else { 3 };

        let unknowns = 3 * n + n_gravity_cols + usize::from(solve_scale);
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut rhs: Vec<f64> = Vec::new();

        let vel_col = |kf_local: usize, axis: usize| -> usize { 3 * kf_local + axis };
        let gravity_col = |axis: usize| -> usize { 3 * n + axis };
        let scale_col = 3 * n + n_gravity_cols;

        for edge in map.imu_edges() {
            let Some(&i) = frame_to_local.get(&edge.prev_kf_idx) else {
                continue;
            };
            let Some(&j) = frame_to_local.get(&edge.curr_kf_idx) else {
                continue;
            };

            let cam_i_world = keyframes[i].frame.pose_world_to_cam.inverse();
            let cam_j_world = keyframes[j].frame.pose_world_to_cam.inverse();
            let r_wb_i = cam_i_world.rotation * r_cb;
            let dt = edge.preintegrated.dt;
            if dt <= 0.0 {
                continue;
            }

            let delta_p_world = r_wb_i * edge.preintegrated.delta_position_with_bias(bias);
            let delta_v_world = r_wb_i * edge.preintegrated.delta_velocity_with_bias(bias);
            let visual_dp = cam_j_world.translation - cam_i_world.translation;
            let lever_dp = cam_j_world.rotation * lever - cam_i_world.rotation * lever;

            for axis in 0..3 {
                let mut row = vec![0.0; unknowns];
                row[vel_col(i, axis)] = dt;
                let mut b = vec_axis(lever_dp - delta_p_world, axis);
                if solve_scale {
                    row[scale_col] = -vec_axis(visual_dp, axis);
                } else {
                    b += vec_axis(visual_dp, axis);
                }
                match gravity_basis {
                    None => row[gravity_col(axis)] = 0.5 * dt * dt,
                    Some((b1, b2)) => {
                        row[gravity_col(0)] = 0.5 * dt * dt * vec_axis(b1, axis);
                        row[gravity_col(1)] = 0.5 * dt * dt * vec_axis(b2, axis);
                        let g0 = gravity_dir.unwrap() * GRAVITY_MAGNITUDE;
                        b -= 0.5 * dt * dt * vec_axis(g0, axis);
                    }
                }
                rows.push(row);
                rhs.push(b);
            }

            for axis in 0..3 {
                let mut row = vec![0.0; unknowns];
                row[vel_col(j, axis)] = 1.0;
                row[vel_col(i, axis)] = -1.0;
                let mut b = vec_axis(delta_v_world, axis);
                match gravity_basis {
                    None => row[gravity_col(axis)] = -dt,
                    Some((b1, b2)) => {
                        row[gravity_col(0)] = -dt * vec_axis(b1, axis);
                        row[gravity_col(1)] = -dt * vec_axis(b2, axis);
                        let g0 = gravity_dir.unwrap() * GRAVITY_MAGNITUDE;
                        b += dt * vec_axis(g0, axis);
                    }
                }
                rows.push(row);
                rhs.push(b);
            }
        }

        if rows.len() < unknowns {
            return None;
        }

        let solution = solve_least_squares(&rows, &rhs)?;
        let gravity_world = match gravity_basis {
            None => Vec3F64::new(
                solution[gravity_col(0)],
                solution[gravity_col(1)],
                solution[gravity_col(2)],
            ),
            Some((b1, b2)) => {
                gravity_dir.unwrap() * GRAVITY_MAGNITUDE
                    + b1 * solution[gravity_col(0)]
                    + b2 * solution[gravity_col(1)]
            }
        };
        if !gravity_world.length().is_finite() || gravity_world.length() < 1e-6 {
            return None;
        }

        let velocities_world = (0..n)
            .map(|i| {
                Vec3F64::new(
                    solution[vel_col(i, 0)],
                    solution[vel_col(i, 1)],
                    solution[vel_col(i, 2)],
                )
            })
            .collect();

        Some(InertialSolveResult {
            scale: if solve_scale { solution[scale_col] } else { 1.0 },
            gravity_world,
            velocities_world,
        })
    }
}
