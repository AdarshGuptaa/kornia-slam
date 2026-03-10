//! Local bundle adjustment using `kornia_3d::ba`.
//!
//! Collects observations from recent keyframes and delegates to
//! `kornia_3d::ba::bundle_adjust` for joint pose + point optimization.

use std::collections::HashSet;

use kornia_3d::ba::{self, BaObservation, BaParams};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;

use crate::mapping::Map;

/// Run local bundle adjustment over recent keyframes and their observed map points.
///
/// Collects the last N active keyframes, gathers observations (undistorting keypoints
/// via camera), calls `kornia_3d::ba::bundle_adjust`, and writes back optimized poses
/// and point positions into the map.
pub fn run_local_ba(map: &mut Map, camera: &PinholeCamera) {
    const MAX_ACTIVE_KFS: usize = 3;
    const MIN_OBSERVATIONS: usize = 8;

    let n_kfs = map.keyframes().len();
    if n_kfs < 2 {
        return;
    }

    let active_start = n_kfs.saturating_sub(MAX_ACTIVE_KFS);

    // Collect map point indices observed by active keyframes.
    let mut mp_set: HashSet<usize> = HashSet::new();
    for kf in &map.keyframes()[active_start..] {
        for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
            if let Some(mp) = map.map_points().get(*mp_idx) {
                if !mp.culled {
                    mp_set.insert(*mp_idx);
                }
            }
        }
    }
    if mp_set.is_empty() {
        return;
    }

    // Build sorted map point list and index mapping (global → local).
    let mut mp_global_indices: Vec<usize> = mp_set.iter().copied().collect();
    mp_global_indices.sort_unstable();

    let mp_global_to_local: std::collections::HashMap<usize, usize> = mp_global_indices
        .iter()
        .enumerate()
        .map(|(local, &global)| (global, local))
        .collect();

    let points: Vec<Vec3F64> = mp_global_indices
        .iter()
        .map(|&idx| map.map_points()[idx].position)
        .collect();

    // Build flat pose list: all keyframes, with fixed/active determined per observation.
    let poses: Vec<Pose3d> = map
        .keyframes()
        .iter()
        .map(|kf| kf.frame.pose_world_to_cam)
        .collect();

    let mut observations = Vec::new();

    // All keyframe observations.
    for (kf_idx, kf) in map.keyframes().iter().enumerate() {
        let is_fixed = kf_idx < active_start;
        for (desc_idx, mp_opt) in kf.map_point_by_desc_idx.iter().enumerate() {
            if let Some(mp_idx) = mp_opt {
                let Some(&point_idx) = mp_global_to_local.get(mp_idx) else {
                    continue;
                };
                if let Some(kp) = kf.frame.features.keypoints_xy.get(desc_idx) {
                    let p = camera.undistort(kp[0] as f64, kp[1] as f64);
                    observations.push(BaObservation {
                        pose_idx: kf_idx,
                        point_idx,
                        pixel: [p.x as f32, p.y as f32],
                        fixed_pose: is_fixed,
                    });
                }
            }
        }
    }

    if observations.len() < MIN_OBSERVATIONS {
        return;
    }

    let ba_result = match ba::bundle_adjust(&poses, &points, &observations, camera, &BaParams::default()) {
        Ok(r) => r,
        Err(_) => return,
    };

    // Write back optimized poses (active keyframes only).
    let keyframes = map.keyframes_mut();
    for (kf_idx, pose) in ba_result.poses.iter().enumerate() {
        if kf_idx >= active_start {
            keyframes[kf_idx].frame.pose_world_to_cam = *pose;
        }
    }

    // Write back optimized point positions.
    let map_points = map.map_points_mut();
    for (local_idx, &global_idx) in mp_global_indices.iter().enumerate() {
        if let Some(mp) = map_points.get_mut(global_idx) {
            mp.position = ba_result.points[local_idx];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Mat3F64;

    fn test_camera() -> PinholeCamera {
        PinholeCamera {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    #[test]
    fn test_local_ba_converges() {
        let cam = test_camera();

        let pose0 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO);
        let pose1 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.5, 0.0, 0.0));

        let true_points = vec![
            Vec3F64::new(-1.0, -1.0, 5.0),
            Vec3F64::new(1.0, -1.0, 5.0),
            Vec3F64::new(1.0, 1.0, 5.0),
            Vec3F64::new(-1.0, 1.0, 5.0),
        ];

        let project = |pose: &Pose3d, pw: &Vec3F64| -> [f32; 2] {
            let pc = pose.transform_point(pw);
            let u = cam.fx * pc.x / pc.z + cam.cx;
            let v = cam.fy * pc.y / pc.z + cam.cy;
            [u as f32, v as f32]
        };

        let mut observations = Vec::new();
        for (pi, pt) in true_points.iter().enumerate() {
            observations.push(BaObservation {
                pose_idx: 0,
                point_idx: pi,
                pixel: project(&pose0, pt),
                fixed_pose: true,
            });
            observations.push(BaObservation {
                pose_idx: 1,
                point_idx: pi,
                pixel: project(&pose1, pt),
                fixed_pose: false,
            });
        }

        let perturbed: Vec<Vec3F64> = true_points
            .iter()
            .map(|p| *p + Vec3F64::new(0.05, -0.03, 0.02))
            .collect();

        let result = ba::bundle_adjust(
            &[pose0, pose1],
            &perturbed,
            &observations,
            &cam,
            &BaParams {
                max_iterations: 20,
                ..BaParams::default()
            },
        )
        .unwrap();

        for (i, refined) in result.points.iter().enumerate() {
            let err = (*refined - true_points[i]).length();
            assert!(err < 0.1, "point {i} error {err} too large");
        }
    }
}
