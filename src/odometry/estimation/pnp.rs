//! PnP pose estimation and validation.

use kornia_algebra::{Mat3AF32, Mat3F64, Vec2F32, Vec3AF32, Vec3F64};
use kornia_3d::pnp::{LMRefineParams, refine_pose_lm};

use crate::mapping::MapPoint;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;

/// PnP pose-estimation thresholds.
#[derive(Debug, Clone)]
pub struct PnpConfig {
    /// Reprojection threshold (px) for filtering correspondences against the prior pose.
    pub prior_reproj_threshold_px: f64,
    /// Reprojection threshold (px) for counting final inliers after LM refinement.
    pub final_reproj_threshold_px: f64,
    /// Maximum acceptable rotation change in degrees between consecutive poses.
    pub max_rot_deg: f64,
    /// Maximum acceptable translation change (norm) between consecutive poses.
    pub max_trans_norm: f64,
    /// Minimum number of 3D–2D correspondences required for PnP solving.
    pub min_correspondences: usize,
}

impl Default for PnpConfig {
    fn default() -> Self {
        Self {
            prior_reproj_threshold_px: 25.0,
            final_reproj_threshold_px: 3.0,
            max_rot_deg: 20.0,
            max_trans_norm: 0.50,
            min_correspondences: 4,
        }
    }
}

/// Count map-point reprojection inliers given a camera pose.
pub fn count_reprojection_inliers(
    pose_world_to_cam: &Pose3d,
    points_world: &[Vec3F64],
    points_image: &[Vec2F32],
    camera: &PinholeCamera,
    threshold_px: f64,
) -> usize {
    let th2 = threshold_px * threshold_px;
    let mut inliers = 0usize;
    for (pw, pi) in points_world.iter().zip(points_image.iter()) {
        if camera
            .reprojection_error_sq_world(pose_world_to_cam, pw, pi.x as f64, pi.y as f64)
            .is_some_and(|err_sq| err_sq <= th2)
        {
            inliers += 1;
        }
    }
    inliers
}

/// Check whether a pose increment is geometrically plausible.
pub fn pose_increment_ok(prev_pose: &Pose3d, new_pose: &Pose3d, config: &PnpConfig) -> bool {
    let r_delta = new_pose.rotation * prev_pose.rotation.transpose();
    let trace = r_delta.col(0).x + r_delta.col(1).y + r_delta.col(2).z;
    let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    let rot_deg = cos_theta.acos().to_degrees();
    let trans_norm = (new_pose.translation - prev_pose.translation).length();

    rot_deg <= config.max_rot_deg && trans_norm <= config.max_trans_norm
}

/// Solve PnP from map-point ↔ keypoint correspondences using LM refinement.
///
/// Returns `(pose_world_to_cam, num_inliers)` or `None` if not enough data.
pub fn solve_pnp_from_correspondences(
    map_points: &[MapPoint],
    correspondences: &[(usize, usize)],
    keypoints_undist: &[[f32; 2]],
    pose_init: &Pose3d,
    camera: &PinholeCamera,
    config: &PnpConfig,
) -> Option<(Pose3d, usize)> {
    let mut points_world = Vec::with_capacity(correspondences.len());
    let mut points_world_f64 = Vec::with_capacity(correspondences.len());
    let mut points_image = Vec::with_capacity(correspondences.len());
    for &(mp_idx, kp_idx) in correspondences {
        if let (Some(mp), Some(&kp)) = (map_points.get(mp_idx), keypoints_undist.get(kp_idx)) {
            points_world.push(Vec3AF32::new(
                mp.position.x as f32,
                mp.position.y as f32,
                mp.position.z as f32,
            ));
            points_world_f64.push(mp.position);
            points_image.push(Vec2F32::new(kp[0], kp[1]));
        }
    }
    if points_world.len() < config.min_correspondences {
        return None;
    }

    // Keep correspondences that are geometrically consistent with the prior pose.
    let prior_th2 = config.prior_reproj_threshold_px * config.prior_reproj_threshold_px;
    let mut prior_inliers = Vec::new();
    for (i, pw) in points_world_f64.iter().enumerate() {
        if camera
            .reprojection_error_sq_world(
                pose_init,
                pw,
                points_image[i].x as f64,
                points_image[i].y as f64,
            )
            .is_some_and(|err_sq| err_sq <= prior_th2)
        {
            prior_inliers.push(i);
        }
    }
    if prior_inliers.len() < config.min_correspondences {
        return None;
    }

    let k = Mat3AF32::from_cols(
        Vec3AF32::new(camera.fx as f32, 0.0, 0.0),
        Vec3AF32::new(0.0, camera.fy as f32, 0.0),
        Vec3AF32::new(camera.cx as f32, camera.cy as f32, 1.0),
    );
    let mut world_inliers = Vec::with_capacity(prior_inliers.len());
    let mut image_inliers = Vec::with_capacity(prior_inliers.len());
    for &i in &prior_inliers {
        world_inliers.push(points_world[i]);
        image_inliers.push(points_image[i]);
    }

    let r_init_f32 = Mat3AF32::from_cols(
        Vec3AF32::new(
            pose_init.rotation.col(0).x as f32,
            pose_init.rotation.col(0).y as f32,
            pose_init.rotation.col(0).z as f32,
        ),
        Vec3AF32::new(
            pose_init.rotation.col(1).x as f32,
            pose_init.rotation.col(1).y as f32,
            pose_init.rotation.col(1).z as f32,
        ),
        Vec3AF32::new(
            pose_init.rotation.col(2).x as f32,
            pose_init.rotation.col(2).y as f32,
            pose_init.rotation.col(2).z as f32,
        ),
    );
    let t_init_f32 = Vec3AF32::new(
        pose_init.translation.x as f32,
        pose_init.translation.y as f32,
        pose_init.translation.z as f32,
    );

    let lm = refine_pose_lm(
        &world_inliers,
        &image_inliers,
        &k,
        &r_init_f32,
        &t_init_f32,
        None,
        &LMRefineParams::default(),
    )
    .ok()?;

    let r = &lm.rotation;
    let t = lm.translation;
    let r_new = Mat3F64::from_cols(
        Vec3F64::new(r.col(0).x as f64, r.col(0).y as f64, r.col(0).z as f64),
        Vec3F64::new(r.col(1).x as f64, r.col(1).y as f64, r.col(1).z as f64),
        Vec3F64::new(r.col(2).x as f64, r.col(2).y as f64, r.col(2).z as f64),
    );
    let t_new = Vec3F64::new(t.x as f64, t.y as f64, t.z as f64);
    let pose_new = Pose3d::new(r_new, t_new);
    let final_inliers = count_reprojection_inliers(
        &pose_new,
        &points_world_f64,
        &points_image,
        camera,
        config.final_reproj_threshold_px,
    );
    Some((pose_new, final_inliers))
}
