//! Tracking-phase helpers for the ORB-SLAM example.

use crate::bootstrap::run_bootstrap;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::pose::{
    TwoViewConfig, TwoViewModel, triangulate_midpoint_known_pose, two_view_estimate,
};
use kornia_algebra::Vec3F64;
use kornia_imgproc::features::{OrbMatchConfig, match_orb_descriptors};
use kornia_slam::estimation::MapProjectionEstimator;
use kornia_slam::estimation::map_projection::MapProjectionEstimateOutcome;
use kornia_slam::estimation::two_view::TwoViewInitConfig;
use kornia_slam::mapping::ba::run_local_ba;
use kornia_slam::mapping::{Keyframe, Map, MapPoint, cull_map_points};
use kornia_slam::odometry::{OdometryResult, OdometryState, OdometryStatus};
use kornia_slam::{Frame, OrbFeatures};

fn grow_map_points_from_keyframe_pair(
    map: &mut Map,
    camera: &PinholeCamera,
    curr_kf_idx: usize,
    prev_kf: &Keyframe,
    curr_features: &OrbFeatures,
    curr_kf_map_assoc: &mut [Option<usize>],
    pose_world_to_cam: &Pose3d,
    match_config: OrbMatchConfig,
    two_view_config: &TwoViewConfig,
    min_parallax_deg: f64,
) -> usize {
    const MIN_GROWTH_MATCHES: usize = 20;
    const MIN_GROWTH_INLIERS: usize = 15;
    const REPROJ_THRESHOLD_PX: f64 = 3.0;
    const MAX_TRIANGULATION_GAP: f64 = 0.25;

    let matches = match_orb_descriptors(
        &prev_kf.frame.features.orientations,
        &prev_kf.frame.features.descriptors,
        &curr_features.orientations,
        &curr_features.descriptors,
        match_config,
    );
    if matches.len() < MIN_GROWTH_MATCHES {
        return 0;
    }

    let mut pair_indices: Vec<(usize, usize)> = Vec::with_capacity(matches.len());
    for (prev_idx, curr_idx) in matches {
        if prev_idx >= prev_kf.frame.features.keypoints_xy.len()
            || curr_idx >= curr_features.keypoints_xy.len()
        {
            continue;
        }
        if curr_kf_map_assoc.get(curr_idx).is_some_and(|m| m.is_some()) {
            continue;
        }
        if prev_kf.map_point(prev_idx).is_some() {
            continue;
        }
        pair_indices.push((prev_idx, curr_idx));
    }

    let (prev_pts, curr_pts) = camera.undistort_matched_pairs(
        &prev_kf.frame.features.keypoints_xy,
        &curr_features.keypoints_xy,
        &pair_indices,
    );
    if pair_indices.len() < 8 {
        return 0;
    }

    let k = camera.intrinsic_matrix();
    let two_view = match two_view_estimate(&prev_pts, &curr_pts, &k, &k, two_view_config) {
        Ok(tv) if matches!(tv.model, TwoViewModel::Fundamental(_)) => tv,
        _ => return 0,
    };
    if two_view.inlier_indices.len() < MIN_GROWTH_INLIERS {
        return 0;
    }

    let prev_pose = prev_kf.frame.pose_world_to_cam;
    let curr_pose = *pose_world_to_cam;
    let relative_pose = Pose3d::between(&prev_pose, &curr_pose);
    let r_rel = relative_pose.rotation;
    let t_rel = relative_pose.translation;
    if t_rel.length() <= 1e-8 {
        return 0;
    }

    let prev_pose_inv = prev_pose.inverse();
    let reproj_th2 = REPROJ_THRESHOLD_PX * REPROJ_THRESHOLD_PX;
    let mut n_added = 0usize;

    for &inlier_idx in &two_view.inlier_indices {
        let Some(&(_prev_idx, curr_idx)) = pair_indices.get(inlier_idx) else {
            continue;
        };
        if curr_kf_map_assoc.get(curr_idx).is_some_and(|m| m.is_some()) {
            continue;
        }

        let p_prev = prev_pts[inlier_idx];
        let p_curr = curr_pts[inlier_idx];

        let ray_prev = Vec3F64::new(
            (p_prev.x - camera.cx) / camera.fx,
            (p_prev.y - camera.cy) / camera.fy,
            1.0,
        )
        .normalize();
        let ray_curr = Vec3F64::new(
            (p_curr.x - camera.cx) / camera.fx,
            (p_curr.y - camera.cy) / camera.fy,
            1.0,
        )
        .normalize();

        let Some((p_cam_prev, triang_gap)) =
            triangulate_midpoint_known_pose(&ray_prev, &ray_curr, &r_rel, &t_rel)
        else {
            continue;
        };
        if triang_gap > MAX_TRIANGULATION_GAP {
            continue;
        }

        if p_cam_prev.z <= 1e-6 {
            continue;
        }
        let p_cam_curr = r_rel * p_cam_prev + t_rel;
        if p_cam_curr.z <= 1e-6 {
            continue;
        }

        let c2 = -(r_rel.transpose() * t_rel);
        let d1 = p_cam_prev.normalize();
        let d2_vec = p_cam_prev - c2;
        if d2_vec.length() <= 1e-12 {
            continue;
        }
        let d2 = d2_vec.normalize();
        let parallax_deg_val = d1.dot(d2).clamp(-1.0, 1.0).acos().to_degrees();
        if parallax_deg_val < min_parallax_deg {
            continue;
        }

        let Some(err_prev) = camera.reprojection_error_sq_cam(&p_cam_prev, p_prev.x, p_prev.y)
        else {
            continue;
        };
        if err_prev > reproj_th2 {
            continue;
        }
        let Some(err_curr) = camera.reprojection_error_sq_cam(&p_cam_curr, p_curr.x, p_curr.y)
        else {
            continue;
        };
        if err_curr > reproj_th2 {
            continue;
        }

        let p_world = prev_pose_inv.transform_point(&p_cam_prev);
        let mp_idx = map.push_map_point(MapPoint::new(
            p_world,
            curr_features.descriptors[curr_idx],
            curr_kf_idx,
        ));

        if let Some(slot) = curr_kf_map_assoc.get_mut(curr_idx) {
            *slot = Some(mp_idx);
            n_added += 1;
        }
    }

    n_added
}

fn try_insert_keyframe(
    state: &mut OdometryState,
    map: &mut Map,
    estimator: &MapProjectionEstimator,
    two_view_init_config: &TwoViewInitConfig,
    frame: &Frame,
    tracked_inliers: usize,
    matches: &[(usize, usize)],
) -> bool {
    let n_ref_map_points = state
        .current_keyframe_idx
        .and_then(|ki| map.get_keyframe(ki))
        .map(|kf| kf.num_associated_points())
        .unwrap_or(0);

    if !estimator.need_new_keyframe(
        frame.idx,
        state.last_keyframe_idx,
        tracked_inliers,
        n_ref_map_points,
    ) {
        return false;
    }

    let Some(prev_kf) = state
        .current_keyframe_idx
        .and_then(|ki| map.get_keyframe(ki))
        .cloned()
    else {
        return false;
    };

    let mut curr_kf_map_assoc = vec![None; frame.features.descriptors.len()];
    for &(mp_idx, curr_idx) in matches {
        if let Some(slot) = curr_kf_map_assoc.get_mut(curr_idx) {
            *slot = Some(mp_idx);
        }
    }

    let config = estimator.config();
    grow_map_points_from_keyframe_pair(
        map,
        estimator.camera(),
        frame.idx,
        &prev_kf,
        &frame.features,
        &mut curr_kf_map_assoc,
        &state.pose_world_to_cam,
        two_view_init_config.match_config,
        &two_view_init_config.estimation_config,
        config.min_parallax_deg,
    );

    let mut kf = Keyframe::from_frame(Frame::new(
        frame.idx,
        frame.features.clone(),
        state.pose_world_to_cam,
        frame.image_size,
    ));
    kf.map_point_by_desc_idx = curr_kf_map_assoc;
    map.upsert_keyframe(kf);
    state.current_keyframe_idx = Some(frame.idx);
    state.last_keyframe_idx = Some(frame.idx);

    if config.enable_local_ba {
        run_local_ba(map, estimator.camera());

        if let Some(newest_kf) = map.keyframes().last() {
            state.pose_world_to_cam = newest_kf.frame.pose_world_to_cam;
        }
    }

    cull_map_points(map);
    true
}

pub(crate) fn run_tracking_step(
    state: &mut OdometryState,
    map: &mut Map,
    estimator: &MapProjectionEstimator,
    two_view_init_config: &TwoViewInitConfig,
    frame: Frame,
) -> OdometryResult {
    let pose_before_tracking = state.pose_world_to_cam;
    let image_size = frame.image_size;

    let candidate_pose = if let Some(vel) = state.velocity {
        vel.compose(&state.pose_world_to_cam)
    } else {
        state.pose_world_to_cam
    };

    let estimate = estimator.estimate_pose(
        &frame,
        &candidate_pose,
        &pose_before_tracking,
        map,
        state.current_keyframe_idx,
    );

    let (mut status, matches, tracked_inliers) = match estimate {
        MapProjectionEstimateOutcome::Estimated {
            pose_world_to_cam,
            inliers,
            matches,
        } => {
            state.velocity = Some(Pose3d::between(&pose_before_tracking, &pose_world_to_cam));
            state.pose_world_to_cam = pose_world_to_cam;
            (OdometryStatus::Tracked, matches, inliers)
        }
        MapProjectionEstimateOutcome::Rejected { .. } => (OdometryStatus::Skipped, Vec::new(), 0),
    };

    if status == OdometryStatus::Tracked {
        estimator.update_map_point_observations(map, &matches, &candidate_pose, image_size);

        if try_insert_keyframe(
            state,
            map,
            estimator,
            two_view_init_config,
            &frame,
            tracked_inliers,
            &matches,
        ) {
            status = OdometryStatus::KeyframeAccepted;
        }
    }

    if status == OdometryStatus::Skipped {
        state.consecutive_failures += 1;
        if state.consecutive_failures >= estimator.config().max_consecutive_failures {
            state.reset();
            return run_bootstrap(
                state,
                map,
                estimator.camera(),
                two_view_init_config,
                frame,
            );
        }
    } else {
        state.consecutive_failures = 0;
    }

    OdometryResult {
        pose_world_to_cam: state.pose_world_to_cam,
        status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_3d::camera::{ImageSize, PinholeCamera};
    use kornia_3d::pose::{Pose3d, TwoViewConfig};
    use kornia_imgproc::features::{OrbFeatures, OrbMatchConfig};
    use kornia_slam::mapping::{Keyframe, Map};
    use kornia_slam::Frame;

    fn test_frame(idx: usize, keypoints_xy: Vec<[f32; 2]>, descriptors: Vec<[u8; 32]>) -> Frame {
        Frame::new(
            idx,
            OrbFeatures {
                orientations: vec![0.0; descriptors.len()],
                keypoints_xy,
                descriptors,
            },
            Pose3d::IDENTITY,
            ImageSize {
                width: 640.0,
                height: 480.0,
            },
        )
    }

    #[test]
    fn grow_map_points_from_keyframe_pair_returns_zero_when_too_few_matches() {
        let camera = PinholeCamera {
            fx: 458.654,
            fy: 457.296,
            cx: 367.215,
            cy: 248.375,
            k1: -0.28340811,
            k2: 0.07395907,
            p1: 0.00019359,
            p2: 1.76187114e-05,
        };
        let prev_kf = Keyframe::from_frame(test_frame(
            0,
            vec![[10.0, 10.0], [20.0, 20.0]],
            vec![[1u8; 32], [2u8; 32]],
        ));
        let curr_features = OrbFeatures {
            keypoints_xy: vec![[10.5, 10.5], [20.5, 20.5]],
            orientations: vec![0.0, 0.0],
            descriptors: vec![[1u8; 32], [2u8; 32]],
        };
        let mut map = Map::new();
        let mut curr_kf_map_assoc = vec![None; curr_features.descriptors.len()];

        let added = grow_map_points_from_keyframe_pair(
            &mut map,
            &camera,
            1,
            &prev_kf,
            &curr_features,
            &mut curr_kf_map_assoc,
            &Pose3d::IDENTITY,
            OrbMatchConfig::default(),
            &TwoViewConfig::default(),
            1.0,
        );

        assert_eq!(added, 0);
        assert_eq!(map.map_points().len(), 0);
        assert!(curr_kf_map_assoc.iter().all(|slot| slot.is_none()));
    }
}
