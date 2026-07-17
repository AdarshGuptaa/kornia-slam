use std::collections::HashMap;

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, QuatF64, SO3F64, Vec3F64};
use kornia_sensors::imu::{GRAVITY_MAGNITUDE, ImuBias};

use crate::estimation::inertial_init_factor::{InertialInitFactor, KfConst, WeightedZeroPrior};
use crate::map::{Keyframe, Map};
use crate::system::SystemState;
use kornia_algebra::optim::{LevenbergMarquardt, Problem, Variable, VariableType};
// ─────────────────────────────────────────────────────────────────────────────
// Small numeric helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Rotation that takes unit vector `from` to unit vector `to`.
fn rotation_from_to(from: Vec3F64, to: Vec3F64) -> SO3F64 {
    let from = from.normalize();
    let to = to.normalize();
    let dot = from.dot(to).clamp(-1.0, 1.0);
    let cross = from.cross(to);

    // Anti-parallel: pick an arbitrary perpendicular axis.
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

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ImuInitConfig {
    pub min_keyframes: usize,
    pub min_time_sec: f64,
    pub min_motion: f64,
}

#[derive(Debug, Clone)]
pub struct ImuInitResult {
    pub scale: f64,
    pub gravity_world: Vec3F64,
    pub velocities_world: Vec<Vec3F64>,
    pub bias: ImuBias,
}

// ─────────────────────────────────────────────────────────────────────────────
// ImuInitializer
// ─────────────────────────────────────────────────────────────────────────────

pub struct ImuInitializer {
    pub config: ImuInitConfig,
}

impl ImuInitializer {
    pub fn new(config: ImuInitConfig) -> Self {
        Self { config }
    }

    // ── readiness gate ────────────────────────────────────────────────────────

    pub fn ready(&self, map: &Map, start_idx: Option<usize>) -> bool {
        let Some(start_idx) = start_idx else {
            return false;
        };

        let kfs: Vec<&Keyframe> = map
            .keyframes()
            .iter()
            .filter(|kf| kf.frame.idx >= start_idx)
            .collect();
        if kfs.len() < self.config.min_keyframes {
            return false;
        }

        // ORB-SLAM3 uses minTime=2.0s (mono) vs 1.0s (stereo) — `config.min_time_sec`
        // holds the stereo (stricter/smaller) value; mono doubles it.
        let is_mono = !kfs.first().map(|kf| kf.frame.is_stereo()).unwrap_or(false);
        let min_time_sec = if is_mono {
            self.config.min_time_sec * 2.0
        } else {
            self.config.min_time_sec
        };

        let imu_time: f64 = map
            .imu_factors()
            .iter()
            .filter(|f| f.curr_kf_idx >= start_idx)
            .map(|f| f.preintegrated.dt)
            .sum();
        if imu_time < min_time_sec {
            return false;
        }

        let t0 = kfs
            .first()
            .unwrap()
            .frame
            .pose_world_to_cam
            .inverse()
            .translation;
        let t1 = kfs
            .last()
            .unwrap()
            .frame
            .pose_world_to_cam
            .inverse()
            .translation;
        (t1 - t0).length() >= self.config.min_motion
    }

    // ── main entry point ──────────────────────────────────────────────────────
    #[allow(clippy::too_many_arguments)]
    pub fn inertial_optimizer(
        &self,
        map: &Map,
        frame_to_local: &HashMap<usize, usize>,
        keyframes: Vec<&Keyframe>,
        rwg: Mat3F64,
        seed_velocities: &[Vec3F64],
        scale_init: f64,
        mbg: Vec3F64,
        mba: Vec3F64,
        is_mono: bool,
        prior_g: f64,
        prior_a: f64,
        imu_t_bc: Pose3d,
    ) -> Option<(Mat3F64, f64, Vec3F64, Vec3F64, Vec<Vec3F64>)> {
        let n = keyframes.len();
        let mut problem = Problem::new();

        for (local_idx, vel) in seed_velocities.iter().enumerate().take(n) {
            let vel_f32 = vec![vel.x as f32, vel.y as f32, vel.z as f32];
            let name = format!("v{}", local_idx);
            problem
                .add_variable(Variable::euclidean(&name, 3), vel_f32)
                .ok()?;
        }
        let bg_f32 = vec![mbg.x as f32, mbg.y as f32, mbg.z as f32];
        let ba_f32 = vec![mba.x as f32, mba.y as f32, mba.z as f32];
        problem
            .add_variable(Variable::euclidean("bg", 3), bg_f32)
            .ok()?;
        problem
            .add_variable(Variable::euclidean("ba", 3), ba_f32)
            .ok()?;
        let q = SO3F64::from_matrix(&rwg).to_array();
        let g_f32: Vec<f32> = q.iter().map(|&v| v as f32).collect();
        problem
            .add_variable(
                Variable::new("gdir", VariableType::SO3, g_f32.clone()),
                g_f32,
            )
            .ok()?;
        problem
            .add_variable(Variable::euclidean("scale", 1), vec![scale_init as f32])
            .ok()?;
        let kf_const: Vec<KfConst> = keyframes
            .iter()
            .map(|kf| KfConst::new(&kf.frame.pose_world_to_cam, &imu_t_bc))
            .collect();

        let mut edge_count = 0;
        for factor in map.imu_factors() {
            // Factors crossing the window boundary (e.g. the mono bootstrap's
            // reference-keyframe -> first-in-window-keyframe factor, whose
            // prev_kf_idx sits before start_idx) are skipped, not fatal —
            // using `?` here previously made every call silently return None
            // as soon as any such factor existed, which mono always has and
            // stereo never does.
            let Some(&i) = frame_to_local.get(&factor.prev_kf_idx) else {
                continue;
            };
            let Some(&j) = frame_to_local.get(&factor.curr_kf_idx) else {
                continue;
            };

            if factor.preintegrated.dt <= 0.0 {
                continue;
            }

            let f = InertialInitFactor::new(
                kf_const[i],
                kf_const[j],
                factor.preintegrated.clone(),
                is_mono,
            );
            problem
                .add_factor(
                    Box::new(f),
                    vec![
                        format!("v{i}"),
                        format!("v{j}"),
                        "bg".to_string(),
                        "ba".to_string(),
                        "gdir".to_string(),
                        "scale".to_string(),
                    ],
                )
                .ok()?;
            edge_count += 1;
        }

        if edge_count == 0 {
            return None;
        }

        problem
            .add_factor(
                Box::new(WeightedZeroPrior {
                    sqrt_weight: prior_g.sqrt(),
                }),
                vec!["bg".to_string()],
            )
            .ok()?;
        problem
            .add_factor(
                Box::new(WeightedZeroPrior {
                    sqrt_weight: prior_a.sqrt(),
                }),
                vec!["ba".to_string()],
            )
            .ok()?;

        let lm = LevenbergMarquardt {
            max_iterations: 200,
            lambda_init: if prior_g != 0.0 { 1e3 } else { 1e-3 },
            ..Default::default()
        };

        let result = lm.optimize(&mut problem).ok()?;
        println!(
            "[inertial_optimizer] {:?} after {} iters, final_cost={:.6}",
            result.termination_reason, result.iterations, result.final_cost
        );
        let vars = problem.get_variables(); // &HashMap<String, Variable>

        let bg_out = Vec3F64::new(
            vars["bg"].values[0] as f64,
            vars["bg"].values[1] as f64,
            vars["bg"].values[2] as f64,
        );
        let ba_out = Vec3F64::new(
            vars["ba"].values[0] as f64,
            vars["ba"].values[1] as f64,
            vars["ba"].values[2] as f64,
        );
        let gdir_v = &vars["gdir"].values;
        let rwg_out = SO3F64::from_array([
            gdir_v[0] as f64,
            gdir_v[1] as f64,
            gdir_v[2] as f64,
            gdir_v[3] as f64,
        ])
        .matrix();
        let scale_out = if is_mono {
            vars["scale"].values[0] as f64
        } else {
            1.0
        };

        let velocities_out: Vec<Vec3F64> = (0..n)
            .map(|i| {
                let v = &vars[&format!("v{i}")].values;
                Vec3F64::new(v[0] as f64, v[1] as f64, v[2] as f64)
            })
            .collect();

        Some((rwg_out, scale_out, bg_out, ba_out, velocities_out))
    }

    /// Mirrors ORB-SLAM3's `LocalMapping::InitializeIMU(priorG, priorA, bFIBA)`:
    /// a *single* joint LM solve per call. The pipeline is responsible for the
    /// progressive VIBA0 (immediate) / VIBA1 (mTinit>5s) / VIBA2 (mTinit>15s)
    /// re-triggering schedule with progressively relaxed priors — this
    /// function does not chain multiple passes internally.
    ///
    /// `already_initialized` selects the same branch ORB-SLAM3 does at
    /// LocalMapping.cc:1226 (`!isImuInitialized()`): on the first (VIBA0)
    /// call, Rwg and per-keyframe velocities are derived from scratch via the
    /// visual trajectory (finite-difference velocity + gravity-direction
    /// accumulation); on later refinement calls the map is already
    /// gravity-aligned and metric (VIBA0's result was applied), so Rwg seeds
    /// as identity and velocities seed from each keyframe's current
    /// IMU-propagated estimate instead of being re-derived visually.
    #[allow(clippy::too_many_arguments)]
    pub fn try_initialize(
        &self,
        map: &Map,
        imu_t_bc: Option<Pose3d>,
        imu_bias: ImuBias,
        start_idx: usize,
        prior_g: f64,
        prior_a: f64,
        already_initialized: bool,
    ) -> Option<ImuInitResult> {
        let imu_t_bc = imu_t_bc?;

        let mut keyframes: Vec<&Keyframe> = map
            .keyframes()
            .iter()
            .filter(|kf| kf.frame.idx >= start_idx)
            .collect();
        keyframes.sort_by_key(|kf| kf.frame.idx);
        let n = keyframes.len();
        if n < self.config.min_keyframes {
            return None;
        }

        let frame_to_local: HashMap<usize, usize> = keyframes
            .iter()
            .enumerate()
            .map(|(i, kf)| (kf.frame.idx, i))
            .collect();

        let is_mono = !keyframes
            .first()
            .map(|kf| kf.frame.is_stereo())
            .unwrap_or(false);

        let (velocities, rwg): (Vec<Vec3F64>, Mat3F64) = if already_initialized {
            let vels = keyframes.iter().map(|kf| kf.velocity_world).collect();
            // NOT Identity: ORB-SLAM3 can seed Identity here because its
            // ApplyScaledRotation rotates the map so gravity lands back on
            // its own internal reference gI=(0,0,-1) — so "already aligned"
            // means "already at gI". `apply_initialization` here instead
            // rotates the map to kornia-slam's own canonical (0,+G,0), a
            // fixed ~125° offset from gI. Seeding Identity would tell this
            // solve gravity is still at gI when it is actually at (0,+G,0),
            // which is not a small perturbation and would fight convergence.
            let gi_to_canonical =
                rotation_from_to(Vec3F64::new(0.0, 0.0, -1.0), Vec3F64::new(0.0, 1.0, 0.0))
                    .matrix();
            (vels, gi_to_canonical)
        } else {
            let t_cb = imu_t_bc.inverse();
            let r_cb = t_cb.rotation;
            let lever = t_cb.translation;

            let mut velocities: Vec<Vec3F64> = vec![Vec3F64::ZERO; n];
            let mut dir_g = Vec3F64::ZERO;

            for factor in map.imu_factors() {
                let Some(&i) = frame_to_local.get(&factor.prev_kf_idx) else {
                    continue;
                };
                let Some(&j) = frame_to_local.get(&factor.curr_kf_idx) else {
                    continue;
                };
                let dt = factor.preintegrated.dt;
                if dt <= 0.0 {
                    continue;
                }

                let cam_i = keyframes[i].frame.pose_world_to_cam.inverse();
                let cam_j = keyframes[j].frame.pose_world_to_cam.inverse();
                let r_wb_i = cam_i.rotation * r_cb;

                dir_g -= r_wb_i * factor.preintegrated.delta_velocity_with_bias(&imu_bias);

                let p_wb_i = cam_i.translation + cam_i.rotation * lever;
                let p_wb_j = cam_j.translation + cam_j.rotation * lever;
                let vel = (p_wb_j - p_wb_i) / dt;
                velocities[i] = vel;
                velocities[j] = vel;
            }

            let rwg = if dir_g.length() > 1e-9 {
                rotation_from_to(Vec3F64::new(0.0, 0.0, -1.0), dir_g.normalize()).matrix()
            } else {
                Mat3F64::IDENTITY
            };
            (velocities, rwg)
        };

        let (rwg_out, scale_out, bg_out, ba_out, velocities_out) = self.inertial_optimizer(
            map,
            &frame_to_local,
            keyframes,
            rwg,
            &velocities,
            1.0, // scale_init — ORB-SLAM3 always seeds mScale=1.0 before InertialOptimization
            imu_bias.gyro,
            imu_bias.accel,
            is_mono,
            prior_g,
            prior_a,
            imu_t_bc,
        )?;

        // ── Sanity check — mirrors ORB-SLAM3's *only* gate on this result,
        // `if (mScale<1e-1) { bInitializing=false; return; }` (LocalMapping.cc
        // ~1271). There is no bg/ba magnitude gate in ORB-SLAM3: it trusts the
        // prior-regularized joint solve and refines further on the next VIBA
        // pass. A hard |bg|>0.05 reject was added here previously and is what
        // broke real-data initialization — it rejected results ORB-SLAM3
        // would have accepted and simply refined at VIBA1/VIBA2.
        if !scale_out.is_finite() || scale_out < 0.1 {
            println!("[imu_init] rejected: bad scale {:.4}", scale_out);
            return None;
        }
        if !bg_out.length().is_finite() || !ba_out.length().is_finite() {
            println!("[imu_init] rejected: non-finite bias");
            return None;
        }
        // Reconstruct the PHYSICAL gravity vector using the *same* gI the factor
        // used internally — do not reuse rwg_out assuming any other convention.
        let gravity_world = rwg_out * Vec3F64::new(0.0, 0.0, -GRAVITY_MAGNITUDE);

        // NOTE: a hard gravity-misalignment gate was tried here and reverted.
        // VIBA0's own bootstrap (the `!already_initialized` branch above) is a
        // crude, unweighted-sum estimate that does NOT reliably converge with a
        // larger window — on V101 it got *worse* the longer `ready()` retried it
        // (23°  →  27° over the full sequence), because gating VIBA0 on
        // misalignment prevents `state.imu_initialized` from ever becoming true,
        // which starves VIBA1/VIBA2 (`refine_inertial_init`, gated on that same
        // flag) of the chance to run at all. VIBA1/VIBA2 are what actually fix
        // VIBA0's roughness — confirmed on the same V101 sequence pre-gate:
        // VIBA0 23.0° → VIBA1 1.13° → VIBA2 0.78°, all within the first 15s of
        // IMU time. Trust that chain; don't gate its entry point.
        println!(
            "[imu_init] accepted  scale={:.4}  gravity=({:.3},{:.3},{:.3})  bg=({:.5},{:.5},{:.5})  ba=({:.6},{:.6},{:.6})",
            scale_out,
            gravity_world.x,
            gravity_world.y,
            gravity_world.z,
            bg_out.x,
            bg_out.y,
            bg_out.z,
            ba_out.x,
            ba_out.y,
            ba_out.z,
        );

        Some(ImuInitResult {
            scale: scale_out,
            gravity_world,
            velocities_world: velocities_out,
            bias: ImuBias {
                gyro: bg_out,
                accel: ba_out,
            },
        })
    }

    // ── apply to map & state ──────────────────────────────────────────────────

    pub fn apply_initialization(
        &self,
        map: &mut Map,
        state: &mut SystemState,
        imu_bias: &mut ImuBias,
        gravity_world: &mut Vec3F64,
        init: ImuInitResult,
        start_idx: usize,
    ) {
        println!(
            "[imu_init] applying: scale={:.4}  g=({:.3},{:.3},{:.3})",
            init.scale, init.gravity_world.x, init.gravity_world.y, init.gravity_world.z
        );

        // 1. Bring the monocular map to metric scale.
        map.scale_world(init.scale);

        // 2. Rotate world so that gravity aligns with +Y (OpenCV convention).
        let g_norm = init.gravity_world / init.gravity_world.length();
        let rwg = rotation_from_to(g_norm, Vec3F64::new(0.0, 1.0, 0.0));
        map.rotate_world(&rwg);

        // 3. Assign velocities and biases to every keyframe in the window.
        let mut vel_iter = init.velocities_world.into_iter();
        for kf in map
            .keyframes_mut()
            .iter_mut()
            .filter(|kf| kf.frame.idx >= start_idx)
        {
            if let Some(v) = vel_iter.next() {
                kf.velocity_world = rwg * v;
                kf.imu_bias = init.bias;
            }
        }

        // 4. Update the tracker state from the last initialized keyframe.
        if let Some(last_kf) = map.keyframes().iter().rfind(|kf| kf.frame.idx >= start_idx) {
            state.velocity_world = last_kf.velocity_world;
            state.pose_world_to_cam = last_kf.frame.pose_world_to_cam;
        }

        state.velocity = None;
        state.imu_initialized = true;
        *gravity_world = Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0);
        *imu_bias = init.bias;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;
    use kornia_image::ImageSize;
    use kornia_imgproc::features::OrbFeatures;
    use kornia_sensors::imu::{ImuCalib, ImuMeasurement};

    const RADIUS: f64 = 2.0;
    const OMEGA: f64 = 0.5;

    /// True trajectory: circular translation *plus* the body yawing in sync
    /// with it (like a vehicle banking into the turn) — needed so `R_wb(t)`
    /// actually varies across keyframes. Without body rotation, a constant
    /// accel-bias offset and a wrong scale/gravity estimate are genuinely
    /// indistinguishable from the data (classic VIO init degeneracy); gyro
    /// bias stays observable either way since it only needs rotation-only
    /// consistency, which is why `bg` recovered correctly even with the
    /// bug below.
    fn circular_trajectory(t: f64) -> (Vec3F64, Vec3F64, Vec3F64, Mat3F64) {
        let theta = OMEGA * t;
        let (s, c) = theta.sin_cos();
        let p = Vec3F64::new(RADIUS * c, 0.0, RADIUS * s);
        let v = Vec3F64::new(-RADIUS * OMEGA * s, 0.0, RADIUS * OMEGA * c);
        let a = Vec3F64::new(
            -RADIUS * OMEGA * OMEGA * c,
            0.0,
            -RADIUS * OMEGA * OMEGA * s,
        );
        // Rotation about the (world) Y axis by theta — angular velocity is
        // exactly (0, OMEGA, 0) in both body and world frame since Y is a
        // fixed eigenvector of this rotation for all theta.
        let r_wb = SO3F64::exp(Vec3F64::new(0.0, theta, 0.0)).matrix();
        (p, v, a, r_wb)
    }

    /// Builds the *corrupted* vision-frame keyframe pose: true trajectory
    /// scaled by `s_true` and rotated by an arbitrary, non-gravity-aligned
    /// `r_arb` — models what a monocular front-end actually hands the
    /// initializer (unknown scale, unknown relation to gravity).
    fn synth_pose_world_to_cam(
        r_arb: Mat3F64,
        s_true: f64,
        p_true: Vec3F64,
        r_wb_true: Mat3F64,
    ) -> Pose3d {
        let vision_rotation = r_arb * r_wb_true;
        let vision_position = (r_arb * p_true) * s_true;
        let cam_to_world = Pose3d::new(vision_rotation, vision_position);
        cam_to_world.inverse()
    }

    fn synth_frame(idx: usize, pose_world_to_cam: Pose3d) -> Frame {
        Frame {
            idx,
            features: OrbFeatures {
                keypoints_xy: vec![],
                orientations: vec![],
                descriptors: vec![],
                octaves: vec![],
            },
            pose_world_to_cam,
            image_size: ImageSize {
                width: 640,
                height: 480,
            },
            keypoint_colors: vec![],
            u_right: vec![],
            depth: vec![], // empty => is_stereo() == false => monocular path
            keypoints_undist: vec![],
        }
    }

    /// Integrates the TRUE (metric, gravity-aligned) trajectory's IMU signal
    /// between two keyframe timestamps, with a known bias baked into the raw
    /// measurement — `PreintegratedImu` is seeded with zero reference bias,
    /// so it integrates the biased signal uncorrected, exactly like a real
    /// biased sensor (`PreintegratedImu::integrate` subtracts `self.bias`,
    /// which is zero here — see `imu.rs:140-143`).
    fn integrate_true_imu(
        t0: f64,
        t1: f64,
        imu_rate_hz: f64,
        gravity_true: Vec3F64,
        bias_gyro_true: Vec3F64,
        bias_accel_true: Vec3F64,
        calib: ImuCalib,
    ) -> kornia_sensors::imu::PreintegratedImu {
        let mut pim = kornia_sensors::imu::PreintegratedImu::new(ImuBias::default(), calib);
        let dt = 1.0 / imu_rate_hz;
        let mut t = t0;
        while t < t1 - 1e-9 {
            let (_, _, a_true, r_wb_true) = circular_trajectory(t + 0.5 * dt);
            let gyro_meas = Vec3F64::new(0.0, OMEGA, 0.0) + bias_gyro_true; // true body-frame ω + bias
            let accel_meas = r_wb_true.transpose() * (a_true - gravity_true) + bias_accel_true; // specific force in body frame + bias
            pim.integrate(
                &ImuMeasurement {
                    timestamp: t,
                    gyro: gyro_meas,
                    accel: accel_meas,
                },
                dt,
            );
            t += dt;
        }
        pim
    }

    #[test]
    fn recovers_scale_bias_gravity_from_synthetic_trajectory() {
        let s_true = 0.5; // vision map is half true metric size
        let r_arb = SO3F64::exp(Vec3F64::new(0.3, -0.5, 0.2)).matrix(); // arbitrary, non-gravity-aligned
        let bias_gyro_true = Vec3F64::new(0.01, -0.02, 0.005);
        let bias_accel_true = Vec3F64::new(0.05, -0.03, 0.02);
        let gravity_true = Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0); // kornia's Y-down convention

        let calib = ImuCalib {
            gyro_noise: 1e-3,
            accel_noise: 1e-2,
            gyro_bias_noise: 1e-5,
            accel_bias_noise: 1e-4,
        };

        let n_keyframes = 30;
        let kf_dt = 0.2; // 5 Hz keyframes -> 5.8s window, ~166° of yaw excitation
        let imu_rate = 200.0; // Hz

        let mut map = Map::new();
        for k in 0..n_keyframes {
            let t = k as f64 * kf_dt;
            let (p_true, _, _, r_wb_true) = circular_trajectory(t);
            let pose = synth_pose_world_to_cam(r_arb, s_true, p_true, r_wb_true);
            map.upsert_keyframe(Keyframe::from_frame(synth_frame(k, pose)));

            if k > 0 {
                let pim = integrate_true_imu(
                    t - kf_dt,
                    t,
                    imu_rate,
                    gravity_true,
                    bias_gyro_true,
                    bias_accel_true,
                    calib,
                );
                map.add_imu_factor(k - 1, k, pim, Vec::new(), t - kf_dt, t);
            }
        }

        let initializer = ImuInitializer::new(ImuInitConfig {
            min_keyframes: 10,
            min_time_sec: 1.0,
            min_motion: 0.1,
        });

        // `try_initialize` now does exactly one joint solve per call (mirrors
        // ORB-SLAM3's `InitializeIMU`/`InertialOptimization` one-call-per-
        // invocation structure). Reproduce the VIBA0 -> apply -> VIBA1
        // (loosened-prior refinement) schedule explicitly here: VIBA0 alone
        // deliberately suppresses accel bias with a huge prior_a to avoid the
        // scale/accel-bias degeneracy on a short/early window, which also
        // biases scale by several percent even on clean synthetic data.
        let viba0 = initializer
            .try_initialize(
                &map,
                Some(Pose3d::IDENTITY),
                ImuBias::default(),
                0,
                1e2,
                1e10,
                false,
            )
            .expect("VIBA0 pass should recover a rough solution from clean synthetic data");

        // Capture VIBA0's scale/gravity before it's consumed by apply_initialization
        // (needed below to reconstruct what frame the second call's raw output —
        // scale correction, velocities — is expressed relative to).
        let viba0_scale = viba0.scale;
        let rwg_viba0 =
            rotation_from_to(viba0.gravity_world.normalize(), Vec3F64::new(0.0, 1.0, 0.0)).matrix();

        let mut state = SystemState::new();
        let mut bias = ImuBias::default();
        let mut gravity_world = Vec3F64::ZERO;
        initializer.apply_initialization(
            &mut map,
            &mut state,
            &mut bias,
            &mut gravity_world,
            viba0,
            0,
        );

        let result = initializer
            .try_initialize(&map, Some(Pose3d::IDENTITY), bias, 0, 0.0, 0.0, true)
            .expect("VIBA1 (loosened-prior) pass should refine to the true solution");

        // `result.scale` is only the *residual* correction on top of what
        // VIBA0 already applied to the map (apply_initialization composes
        // scale/rotation multiplicatively, mirroring ORB-SLAM3's
        // ApplyScaledRotation) — compare the cumulative effect, not the raw
        // second-call output in isolation.
        let total_scale = viba0_scale * result.scale;
        assert!(
            (total_scale - 1.0 / s_true).abs() < 0.05,
            "cumulative scale: got {:.4} (viba0={:.4} * refine={:.4}), want {:.4}",
            total_scale,
            viba0_scale,
            result.scale,
            1.0 / s_true,
        );
        assert!(
            (result.bias.gyro - bias_gyro_true).length() < 0.01,
            "gyro bias: got {:?}, want {:?}",
            result.bias.gyro,
            bias_gyro_true,
        );
        assert!(
            (result.bias.accel - bias_accel_true).length() < 0.01,
            "accel bias: got {:?}, want {:?}",
            result.bias.accel,
            bias_accel_true,
        );

        // apply_initialization already rotated the map so gravity sits at
        // kornia-slam's own canonical (0,+G,0) — the second call's residual
        // gravity-direction correction should converge there too, not to the
        // original (pre-rotation) r_arb-relative direction.
        assert!(
            result
                .gravity_world
                .normalize()
                .dot(Vec3F64::new(0.0, 1.0, 0.0))
                > 0.99,
            "gravity direction: got {:?}, want ~(0,1,0)",
            result.gravity_world.normalize(),
        );

        // Velocity is expressed in the frame at the time of this second
        // solve: after VIBA0's position scale/rotation was applied to the
        // map, but before this call's own residual scale correction (that
        // correction is only reflected in `result.scale`, applied to the map
        // by a subsequent apply_initialization — not folded back into
        // `result.velocities_world` here).
        for k in 0..n_keyframes {
            let t = k as f64 * kf_dt;
            let (_, v_true, _, _) = circular_trajectory(t);
            let expected_v = rwg_viba0 * ((r_arb * v_true) * s_true * viba0_scale);
            let err = (result.velocities_world[k] - expected_v).length();
            assert!(
                err < 0.15,
                "velocity[{k}]: got {:?}, want {:?}",
                result.velocities_world[k],
                expected_v
            );
        }
    }
}
