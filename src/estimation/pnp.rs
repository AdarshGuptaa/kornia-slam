//! PnP pose estimation: solve 3D-2D correspondences with LM refinement.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pnp::{LMRefineParams, refine_pose_lm};
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3AF32, Mat3F64, Vec2F32, Vec3AF32, Vec3F64};

/// PnP pose-estimation thresholds.
#[derive(Debug, Clone)]
pub struct PnpConfig {
    /// Reprojection threshold (px) for filtering correspondences against the prior pose.
    pub prior_reproj_threshold_px: f64,
    /// Reprojection threshold (px) for counting final inliers after LM refinement.
    pub final_reproj_threshold_px: f64,
    /// Minimum number of 3D-2D correspondences required for PnP solving.
    pub min_correspondences: usize,
    /// Minimum inliers for early acceptance before local map refinement.
    pub min_inliers_early: usize,
    /// Minimum inliers to accept a PnP solution.
    pub min_inliers: usize,
}

impl Default for PnpConfig {
    fn default() -> Self {
        Self {
            prior_reproj_threshold_px: 25.0,
            final_reproj_threshold_px: 3.0,
            min_correspondences: 4,
            min_inliers_early: 10,
            min_inliers: 30,
        }
    }
}

/// Solve PnP from 3D world points and 2D image points with LM refinement.
///
/// Filters correspondences by reprojection error against `pose_init`, runs LM,
/// then counts final inliers. Returns the refined pose and inlier count.
pub fn solve_pnp(
    points_world_f64: &[Vec3F64],
    points_image: &[Vec2F32],
    camera: &PinholeCamera,
    pose_init: &Pose3d,
    config: &PnpConfig,
) -> Option<(Pose3d, usize)> {
    if points_world_f64.len() < config.min_correspondences {
        return None;
    }

    // Filter by reprojection error against the prior pose.
    let prior_th2 = config.prior_reproj_threshold_px * config.prior_reproj_threshold_px;
    let mut prior_inlier_indices = Vec::new();
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
            prior_inlier_indices.push(i);
        }
    }
    if prior_inlier_indices.len() < config.min_correspondences {
        return None;
    }

    // Build f32 arrays for LM solver.
    let k = Mat3AF32::from_cols(
        Vec3AF32::new(camera.fx as f32, 0.0, 0.0),
        Vec3AF32::new(0.0, camera.fy as f32, 0.0),
        Vec3AF32::new(camera.cx as f32, camera.cy as f32, 1.0),
    );
    let mut world_inliers = Vec::with_capacity(prior_inlier_indices.len());
    let mut image_inliers = Vec::with_capacity(prior_inlier_indices.len());
    for &i in &prior_inlier_indices {
        let pw = &points_world_f64[i];
        world_inliers.push(Vec3AF32::new(pw.x as f32, pw.y as f32, pw.z as f32));
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
        points_world_f64,
        points_image,
        camera,
        config.final_reproj_threshold_px,
    );

    Some((pose_new, final_inliers))
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
