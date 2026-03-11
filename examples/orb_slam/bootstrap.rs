//! Bootstrap-phase helpers for the ORB-SLAM example.

use kornia_3d::camera::PinholeCamera;
use kornia_algebra::Vec3F64;
use kornia_slam::estimation::two_view::{
    TwoViewInitConfig, TwoViewInitOutcome, try_initialize_two_view,
};
use kornia_slam::mapping::{Keyframe, Map, MapPoint};
use kornia_slam::odometry::{OdometryMode, OdometryResult, OdometryState, OdometryStatus};
use kornia_slam::Frame;

fn materialize_bootstrap_map(
    map: &mut Map,
    reference_frame: Frame,
    current_frame: Frame,
    matches: &[(usize, usize)],
    points3d: &[Vec3F64],
    inlier_indices: &[usize],
    median_depth: Option<f64>,
) -> usize {
    let mut reference_kf = Keyframe::from_frame(reference_frame);
    let mut current_kf = Keyframe::from_frame(current_frame);
    let depth_scale = median_depth.filter(|&d| d > 1e-6).unwrap_or(1.0);
    let reference_pose_inv = reference_kf.frame.pose_world_to_cam.inverse();
    let mut added = 0usize;

    for (p_cam, &match_idx) in points3d.iter().zip(inlier_indices.iter()) {
        let Some(&(reference_desc_idx, current_desc_idx)) = matches.get(match_idx) else {
            continue;
        };
        if reference_desc_idx >= reference_kf.map_point_by_desc_idx.len()
            || current_desc_idx >= current_kf.map_point_by_desc_idx.len()
        {
            continue;
        }

        let descriptor = current_kf
            .frame
            .features
            .descriptors
            .get(current_desc_idx)
            .copied()
            .or_else(|| {
                reference_kf
                    .frame
                    .features
                    .descriptors
                    .get(reference_desc_idx)
                    .copied()
            });
        let Some(descriptor) = descriptor else {
            continue;
        };

        let p_world = reference_pose_inv.transform_point(&(*p_cam / depth_scale));
        let mp_idx = map.push_map_point(MapPoint::new(
            p_world,
            descriptor,
            reference_kf.frame.idx,
        ));
        reference_kf.associate_map_point(reference_desc_idx, mp_idx);
        current_kf.associate_map_point(current_desc_idx, mp_idx);
        added += 1;
    }

    map.upsert_keyframe(reference_kf);
    map.upsert_keyframe(current_kf);

    added
}

pub(crate) fn run_bootstrap(
    state: &mut OdometryState,
    map: &mut Map,
    camera: &PinholeCamera,
    two_view_init_config: &TwoViewInitConfig,
    mut curr_frame: Frame,
) -> OdometryResult {
    // Stamp frames with current odometry pose so bootstrap builds
    // the new map in the existing coordinate frame (not at origin).
    curr_frame.pose_world_to_cam = state.pose_world_to_cam;

    let Some(prev_bootstrap_frame) = state.bootstrap_frame.take() else {
        // First bootstrap frame - store it and wait for a second frame.
        state.bootstrap_frame = Some(curr_frame);
        return OdometryResult {
            pose_world_to_cam: state.pose_world_to_cam,
            status: OdometryStatus::Skipped,
        };
    };

    let outcome = try_initialize_two_view(
        &prev_bootstrap_frame.features,
        &prev_bootstrap_frame.pose_world_to_cam,
        &curr_frame.features,
        &curr_frame.pose_world_to_cam,
        camera,
        two_view_init_config,
    );

    match outcome {
        TwoViewInitOutcome::Rejected { .. } => {
            state.bootstrap_frame = Some(prev_bootstrap_frame);
            OdometryResult {
                pose_world_to_cam: state.pose_world_to_cam,
                status: OdometryStatus::Skipped,
            }
        }
        TwoViewInitOutcome::Initialized {
            pose_world_to_cam,
            motion_increment,
            matches,
            points3d,
            inlier_indices,
            median_depth,
        } => {
            state.velocity = Some(motion_increment);
            state.pose_world_to_cam = pose_world_to_cam;
            curr_frame.pose_world_to_cam = state.pose_world_to_cam;

            let curr_idx = curr_frame.idx;
            materialize_bootstrap_map(
                map,
                prev_bootstrap_frame,
                curr_frame,
                &matches,
                &points3d,
                &inlier_indices,
                median_depth,
            );

            state.current_keyframe_idx = Some(curr_idx);
            state.last_keyframe_idx = Some(curr_idx);
            state.state = OdometryMode::Tracking;

            OdometryResult {
                pose_world_to_cam: state.pose_world_to_cam,
                status: OdometryStatus::KeyframeAccepted,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_3d::camera::ImageSize;
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::Vec3F64;
    use kornia_imgproc::features::OrbFeatures;
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
    fn materialize_bootstrap_map_populates_map_and_keyframe_links() {
        let mut map = Map::new();
        let reference_frame = test_frame(0, vec![[100.0, 100.0]], vec![[7u8; 32]]);
        let current_frame = test_frame(1, vec![[101.0, 99.0]], vec![[9u8; 32]]);

        let added = materialize_bootstrap_map(
            &mut map,
            reference_frame,
            current_frame,
            &[(0usize, 0usize)],
            &[Vec3F64::new(0.0, 0.0, 2.0)],
            &[0],
            Some(2.0),
        );

        assert_eq!(added, 1);
        assert_eq!(map.map_points().len(), 1);
        assert_eq!(map.map_points()[0].position, Vec3F64::new(0.0, 0.0, 1.0));

        let kf0 = map.get_keyframe(0).expect("expected reference keyframe");
        let kf1 = map.get_keyframe(1).expect("expected current keyframe");
        assert_eq!(kf0.map_point_by_desc_idx[0], Some(0));
        assert_eq!(kf1.map_point_by_desc_idx[0], Some(0));
    }
}
