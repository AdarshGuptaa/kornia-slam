//! Map: keyframes, map points, triangulation, and culling.

use std::collections::{HashMap, HashSet};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::{
    Pose3d, TwoViewConfig, TwoViewModel, triangulate_midpoint_known_pose, two_view_estimate,
};
use kornia_algebra::Vec3F64;
use kornia_imgproc::features::OrbMatchConfig;

use crate::odometry::estimation::matching::match_orb_descriptors;
use crate::frame::{Frame, OrbFeatures};

// ── Domain types ─────────────────────────────────────────────────────────────

/// A frame promoted into the map, with descriptor-to-map-point associations.
#[derive(Debug, Clone)]
pub struct Keyframe {
    pub frame: Frame,
    /// For each descriptor index in `frame.features`, associated map-point index.
    pub map_point_by_desc_idx: Vec<Option<usize>>,
}

impl Keyframe {
    /// Creates a keyframe from a frame, with empty map-point associations.
    pub fn from_frame(frame: Frame) -> Self {
        let map_point_by_desc_idx = vec![None; frame.features.descriptors.len()];
        Self {
            frame,
            map_point_by_desc_idx,
        }
    }
}

/// A persistent 3D landmark in the map.
#[derive(Debug, Clone)]
pub struct MapPoint {
    /// 3D position in world frame.
    pub position: Vec3F64,
    /// ORB descriptor used for projection-guided matching.
    pub descriptor: [u8; 32],
    /// Index of the keyframe that first observed this point.
    pub keyframe_idx: usize,
    /// Number of frames where this point was in the camera frustum.
    pub n_visible: u32,
    /// Number of frames where this point was successfully matched.
    pub n_found: u32,
    /// Whether this point has been culled (logically deleted).
    pub culled: bool,
}

// ── Map container ────────────────────────────────────────────────────────────

/// In-memory map storage for keyframes and persistent map points.
#[derive(Debug, Clone, Default)]
pub struct Map {
    keyframes: Vec<Keyframe>,
    map_points: Vec<MapPoint>,
}

impl Map {
    /// Creates an empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns all keyframes.
    pub fn keyframes(&self) -> &[Keyframe] {
        &self.keyframes
    }

    /// Returns all map points.
    pub fn map_points(&self) -> &[MapPoint] {
        &self.map_points
    }

    /// Returns the keyframe with frame index `idx`, if present.
    pub fn get_keyframe(&self, idx: usize) -> Option<&Keyframe> {
        self.keyframes.iter().find(|kf| kf.frame.idx == idx)
    }

    /// Inserts or replaces a keyframe by frame index.
    pub fn upsert_keyframe(&mut self, keyframe: Keyframe) {
        if let Some(pos) = self
            .keyframes
            .iter()
            .position(|kf| kf.frame.idx == keyframe.frame.idx)
        {
            self.keyframes[pos] = keyframe;
        } else {
            self.keyframes.push(keyframe);
        }
    }

    /// Appends a map point and returns its index.
    pub fn push_map_point(&mut self, map_point: MapPoint) -> usize {
        let idx = self.map_points.len();
        self.map_points.push(map_point);
        idx
    }

    /// Returns a mutable reference to all map points.
    pub fn map_points_mut(&mut self) -> &mut Vec<MapPoint> {
        &mut self.map_points
    }

    /// Returns a mutable reference to all keyframes.
    pub fn keyframes_mut(&mut self) -> &mut Vec<Keyframe> {
        &mut self.keyframes
    }

    /// Build a local map of map points visible from nearby keyframes.
    ///
    /// Finds keyframes related by covisibility (shared map point observations)
    /// and recency, then collects their visible map points.
    pub fn build_local_map_points(
        &self,
        tracked_matches: &[(usize, usize)],
        current_keyframe: Option<&Keyframe>,
    ) -> (Vec<MapPoint>, Vec<usize>) {
        const MAX_VOTED_KEYFRAMES: usize = 10;
        const MAX_RECENT_KEYFRAMES: usize = 10;

        let mut keyframe_votes: HashMap<usize, usize> = HashMap::new();
        for &(mp_idx, _) in tracked_matches {
            if let Some(mp) = self.map_points.get(mp_idx) {
                *keyframe_votes.entry(mp.keyframe_idx).or_insert(0) += 1;
            }
        }

        let mut voted_kfs: Vec<(usize, usize)> = keyframe_votes.into_iter().collect();
        voted_kfs.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.cmp(&a.0)));

        let mut local_kf_indices: HashSet<usize> = HashSet::new();
        if let Some(kf) = current_keyframe {
            local_kf_indices.insert(kf.frame.idx);
        }
        for (kf_idx, _) in voted_kfs.into_iter().take(MAX_VOTED_KEYFRAMES) {
            local_kf_indices.insert(kf_idx);
        }
        for kf in self.keyframes.iter().rev().take(MAX_RECENT_KEYFRAMES) {
            local_kf_indices.insert(kf.frame.idx);
        }

        let mut mp_indices: HashSet<usize> = HashSet::new();
        for &(mp_idx, _) in tracked_matches {
            if mp_idx < self.map_points.len() {
                mp_indices.insert(mp_idx);
            }
        }
        for kf in &self.keyframes {
            if !local_kf_indices.contains(&kf.frame.idx) {
                continue;
            }
            for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
                if *mp_idx < self.map_points.len() {
                    mp_indices.insert(*mp_idx);
                }
            }
        }

        let mut global_indices: Vec<usize> = mp_indices.into_iter().collect();
        global_indices.sort_unstable();

        if global_indices.len() < 4 && self.map_points.len() >= 4 {
            global_indices = (0..self.map_points.len()).collect();
        }

        let local_map_points: Vec<MapPoint> = global_indices
            .iter()
            .filter_map(|&idx| self.map_points.get(idx).filter(|mp| !mp.culled).cloned())
            .collect();
        (local_map_points, global_indices)
    }
}

// ── Triangulation ────────────────────────────────────────────────────────────

/// Build the initial map from a successful bootstrap step.
///
/// Takes the two-view data directly rather than the full `BootstrapOutcome` enum.
/// Returns the number of map points added.
pub fn build_initial_map(
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
        let mp_idx = map.push_map_point(MapPoint {
            position: p_world,
            descriptor,
            keyframe_idx: reference_kf.frame.idx,
            n_visible: 1,
            n_found: 1,
            culled: false,
        });
        reference_kf.map_point_by_desc_idx[reference_desc_idx] = Some(mp_idx);
        current_kf.map_point_by_desc_idx[current_desc_idx] = Some(mp_idx);
        added += 1;
    }

    map.upsert_keyframe(reference_kf);
    map.upsert_keyframe(current_kf);

    added
}

/// Triangulate new map points from a pair of keyframes.
///
/// Matches descriptors between `prev_kf` and `curr_features`, filters pairs that
/// already have map-point associations, runs two-view estimation, and triangulates
/// inlier matches. Returns the number of new map points added.
pub fn grow_map_points_from_keyframe_pair(
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
        if prev_kf.map_point_by_desc_idx
            .get(prev_idx)
            .is_some_and(|m| m.is_some())
        {
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
        let mp_idx = map.push_map_point(MapPoint {
            position: p_world,
            descriptor: curr_features.descriptors[curr_idx],
            keyframe_idx: curr_kf_idx,
            n_visible: 1,
            n_found: 1,
            culled: false,
        });

        if let Some(slot) = curr_kf_map_assoc.get_mut(curr_idx) {
            *slot = Some(mp_idx);
            n_added += 1;
        }
    }

    n_added
}

// ── Culling ──────────────────────────────────────────────────────────────────

/// Cull map points with poor observation ratios or that project behind cameras.
pub fn cull_map_points(map: &mut Map) {
    const MIN_OBSERVATIONS: u32 = 5;
    const MIN_FOUND_RATIO: f64 = 0.20;

    let mut n_culled = 0usize;

    for mp in map.map_points_mut().iter_mut() {
        if mp.culled || mp.n_visible < MIN_OBSERVATIONS {
            continue;
        }
        let ratio = mp.n_found as f64 / mp.n_visible as f64;
        if ratio < MIN_FOUND_RATIO {
            mp.culled = true;
            n_culled += 1;
        }
    }

    let mut behind_camera: Vec<usize> = Vec::new();
    for kf in map.keyframes() {
        for mp_opt in &kf.map_point_by_desc_idx {
            if let Some(mp_idx) = mp_opt {
                if let Some(mp) = map.map_points().get(*mp_idx) {
                    if !mp.culled {
                        let p_cam = kf.frame.pose_world_to_cam.transform_point(&mp.position);
                        if p_cam.z <= 1e-8 {
                            behind_camera.push(*mp_idx);
                        }
                    }
                }
            }
        }
    }

    for mp_idx in &behind_camera {
        if let Some(mp) = map.map_points_mut().get_mut(*mp_idx) {
            if !mp.culled {
                mp.culled = true;
                n_culled += 1;
            }
        }
    }

    if n_culled > 0 {
        let culled_set: HashSet<usize> = map
            .map_points()
            .iter()
            .enumerate()
            .filter(|(_, mp)| mp.culled)
            .map(|(i, _)| i)
            .collect();

        for kf in map.keyframes_mut() {
            for mp_opt in &mut kf.map_point_by_desc_idx {
                if let Some(mp_idx) = mp_opt {
                    if culled_set.contains(mp_idx) {
                        *mp_opt = None;
                    }
                }
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

    #[test]
    fn keyframe_from_frame_initializes_map_point_slots() {
        let features = OrbFeatures {
            keypoints_xy: vec![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            orientations: vec![0.1, 0.2, 0.3],
            descriptors: vec![[0u8; 32], [1u8; 32], [2u8; 32]],
        };

        let image_size = ImageSize { width: 640.0, height: 480.0 };
        let frame = Frame::new(7, features, Pose3d::IDENTITY, image_size);
        let keyframe = Keyframe::from_frame(frame);

        assert_eq!(keyframe.frame.idx, 7);
        assert_eq!(keyframe.frame.features.descriptors.len(), 3);
        assert_eq!(keyframe.map_point_by_desc_idx.len(), 3);
        assert!(
            keyframe
                .map_point_by_desc_idx
                .iter()
                .all(|slot| slot.is_none())
        );
    }

    #[test]
    fn upsert_keyframe_replaces_existing_idx() {
        let mut map = Map::new();
        let image_size = ImageSize { width: 640.0, height: 480.0 };

        let first = Keyframe::from_frame(Frame::new(
            10,
            OrbFeatures {
                keypoints_xy: vec![[0.0, 0.0], [1.0, 1.0]],
                orientations: vec![0.1, 0.2],
                descriptors: vec![[0u8; 32], [1u8; 32]],
            },
            Pose3d::IDENTITY,
            image_size,
        ));
        map.upsert_keyframe(first);
        assert_eq!(map.keyframes().len(), 1);

        let second = Keyframe::from_frame(Frame::new(
            10,
            OrbFeatures {
                keypoints_xy: vec![[2.0, 2.0]],
                orientations: vec![0.3],
                descriptors: vec![[2u8; 32]],
            },
            Pose3d::IDENTITY,
            image_size,
        ));
        map.upsert_keyframe(second);

        assert_eq!(map.keyframes().len(), 1);
        assert_eq!(
            map.get_keyframe(10)
                .expect("expected keyframe with idx 10")
                .frame
                .features
                .descriptors
                .len(),
            1
        );
    }

    #[test]
    fn push_map_point_returns_sequential_index() {
        let mut map = Map::new();

        let first_idx = map.push_map_point(MapPoint {
            position: Vec3F64::new(0.0, 0.0, 1.0),
            descriptor: [0u8; 32],
            keyframe_idx: 0,
            n_visible: 0,
            n_found: 0,
            culled: false,
        });
        let second_idx = map.push_map_point(MapPoint {
            position: Vec3F64::new(1.0, 0.0, 1.0),
            descriptor: [1u8; 32],
            keyframe_idx: 0,
            n_visible: 0,
            n_found: 0,
            culled: false,
        });

        assert_eq!(first_idx, 0);
        assert_eq!(second_idx, 1);
        assert_eq!(map.map_points().len(), 2);
    }

    #[test]
    fn test_cull_map_points_removes_low_ratio() {
        let mut map = Map::new();

        map.push_map_point(MapPoint {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            descriptor: [0u8; 32],
            keyframe_idx: 0,
            n_visible: 10,
            n_found: 1,
            culled: false,
        });
        map.push_map_point(MapPoint {
            position: Vec3F64::new(1.0, 0.0, 5.0),
            descriptor: [1u8; 32],
            keyframe_idx: 0,
            n_visible: 10,
            n_found: 5,
            culled: false,
        });

        cull_map_points(&mut map);

        assert!(map.map_points()[0].culled);
        assert!(!map.map_points()[1].culled);
    }

    #[test]
    fn build_initial_map_populates_map_and_keyframe_links() {
        let mut map = Map::new();

        let reference_features = OrbFeatures {
            keypoints_xy: vec![[100.0, 100.0]],
            orientations: vec![0.1],
            descriptors: vec![[7u8; 32]],
        };
        let image_size = ImageSize { width: 640.0, height: 480.0 };
        let reference_frame = Frame::new(0, reference_features, Pose3d::IDENTITY, image_size);

        let current_features = OrbFeatures {
            keypoints_xy: vec![[101.0, 99.0]],
            orientations: vec![0.2],
            descriptors: vec![[9u8; 32]],
        };

        let pose = Pose3d::IDENTITY;
        let matches = vec![(0usize, 0usize)];
        let points3d = vec![Vec3F64::new(0.0, 0.0, 2.0)];
        let inlier_indices = vec![0];
        let median_depth = Some(2.0);

        let current_frame = Frame::new(1, current_features, pose, image_size);

        let added = build_initial_map(
            &mut map,
            reference_frame,
            current_frame,
            &matches,
            &points3d,
            &inlier_indices,
            median_depth,
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
