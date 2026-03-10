//! Feature matching: spatial indexing and projection-guided association.

use kornia_3d::camera::{ImageSize, PinholeCamera};
use kornia_3d::pose::Pose3d;
use kornia_imgproc::features::hamming_distance;

pub use kornia_imgproc::features::{OrbMatchConfig, match_orb_descriptors};

use crate::mapping::MapPoint;
use crate::frame::Frame;

/// Tunable parameters for projection-guided matching.
#[derive(Debug, Clone, Copy)]
pub struct ProjectionMatchConfig {
    /// Reject projected points with depth `<= min_depth`.
    pub min_depth: f64,
    /// Keypoint search radius around each projected pixel.
    pub search_radius: f32,
    /// Maximum Hamming distance to accept a descriptor match.
    pub max_hamming: u32,
}

/// Result of matching map points against a frame.
pub(crate) struct FrameMatchResult {
    pub matches: Vec<(usize, usize)>,
    pub keypoints_undist: Vec<[f32; 2]>,
    pub grid: KeypointGrid,
}

/// A 2D grid that bins keypoints by their image position for O(1) spatial queries.
pub struct KeypointGrid {
    cells: Vec<Vec<usize>>,
    cell_w: f32,
    cell_h: f32,
    n_cols: usize,
    n_rows: usize,
    img_w: f32,
    img_h: f32,
}

impl KeypointGrid {
    /// Builds a grid over `keypoints_xy` (each `[x, y]`) for an image of size `img_w x img_h`.
    ///
    /// Each cell is `cell_size x cell_size` pixels. Points outside the image are clamped.
    pub fn new(keypoints_xy: &[[f32; 2]], img_w: f32, img_h: f32, cell_size: f32) -> Self {
        let n_cols = (img_w / cell_size).ceil() as usize;
        let n_rows = (img_h / cell_size).ceil() as usize;
        let n_cells = n_cols * n_rows;
        let mut cells = vec![Vec::new(); n_cells];

        for (i, kp) in keypoints_xy.iter().enumerate() {
            let col = ((kp[0] / cell_size) as usize).min(n_cols - 1);
            let row = ((kp[1] / cell_size) as usize).min(n_rows - 1);
            cells[row * n_cols + col].push(i);
        }

        Self {
            cells,
            cell_w: cell_size,
            cell_h: cell_size,
            n_cols,
            n_rows,
            img_w,
            img_h,
        }
    }

    /// Returns indices of keypoints within `radius` pixels of `(x, y)`.
    ///
    /// Queries a neighborhood of grid cells around the target position and filters
    /// by squared Euclidean distance, avoiding a sqrt per point.
    pub fn query_radius(
        &self,
        x: f32,
        y: f32,
        radius: f32,
        keypoints_xy: &[[f32; 2]],
    ) -> Vec<usize> {
        let r_sq = radius * radius;

        // Determine the range of cells that overlap the search circle.
        let col_min = ((x - radius).max(0.0) / self.cell_w) as usize;
        let col_max =
            (((x + radius).min(self.img_w - 1.0) / self.cell_w) as usize).min(self.n_cols - 1);
        let row_min = ((y - radius).max(0.0) / self.cell_h) as usize;
        let row_max =
            (((y + radius).min(self.img_h - 1.0) / self.cell_h) as usize).min(self.n_rows - 1);

        let mut result = Vec::new();
        for r in row_min..=row_max {
            for c in col_min..=col_max {
                for &idx in &self.cells[r * self.n_cols + c] {
                    let kp = keypoints_xy[idx];
                    let dx = kp[0] - x;
                    let dy = kp[1] - y;
                    if dx * dx + dy * dy <= r_sq {
                        result.push(idx);
                    }
                }
            }
        }
        result
    }
}

/// Matches map points to current-frame keypoints by projecting each map point
/// into the image and searching for the best descriptor match within a radius.
///
/// Points with camera-frame depth `<= config.min_depth` are rejected.
///
/// Returns pairs of `(map_point_idx, keypoint_idx)`.
pub fn match_by_projection(
    map_points: &[MapPoint],
    keypoints_xy: &[[f32; 2]],
    descriptors: &[[u8; 32]],
    grid: &KeypointGrid,
    pose_world_to_cam: &Pose3d,
    camera: &PinholeCamera,
    image_size: ImageSize,
    config: ProjectionMatchConfig,
) -> Vec<(usize, usize)> {
    // Track which keypoints have already been matched to avoid duplicates.
    let mut matched_kp = vec![false; keypoints_xy.len()];
    let mut matches = Vec::new();

    for (mp_idx, mp) in map_points.iter().enumerate() {
        if mp.culled {
            continue;
        }

        // Project world point into camera frame.
        let p_cam = pose_world_to_cam.transform_point(&mp.position);
        let Ok(pixel) = camera.project_to_image(&p_cam, config.min_depth, image_size) else {
            continue;
        };
        let u = pixel.x as f32;
        let v = pixel.y as f32;

        // Find candidate keypoints near the projection.
        let candidates = grid.query_radius(u, v, config.search_radius, keypoints_xy);

        // Find the best (lowest Hamming distance) unmatched descriptor.
        let mut best_dist = u32::MAX;
        let mut best_kp = usize::MAX;

        for kp_idx in candidates {
            if matched_kp[kp_idx] {
                continue;
            }
            let dist = hamming_distance(&mp.descriptor, &descriptors[kp_idx]);
            if dist < best_dist {
                best_dist = dist;
                best_kp = kp_idx;
            }
        }

        if best_dist <= config.max_hamming && best_kp != usize::MAX {
            matched_kp[best_kp] = true;
            matches.push((mp_idx, best_kp));
        }
    }

    matches
}

/// Undistorts keypoints, builds a spatial grid, and runs projection matching
/// with narrow→wide fallback. Returns matches, undistorted keypoints, and grid.
pub(crate) fn match_map_to_frame(
    map_points: &[MapPoint],
    frame: &Frame,
    camera: &PinholeCamera,
    pose: &Pose3d,
    config: ProjectionMatchConfig,
) -> FrameMatchResult {
    const KEYPOINT_GRID_CELL_SIZE: f32 = 64.0;
    const MIN_MATCHES_BEFORE_WIDE: usize = 20;

    let keypoints_undist: Vec<[f32; 2]> = frame
        .features
        .keypoints_xy
        .iter()
        .map(|kp| {
            let p = camera.undistort(kp[0] as f64, kp[1] as f64);
            [p.x as f32, p.y as f32]
        })
        .collect();

    let grid = KeypointGrid::new(
        &keypoints_undist,
        frame.image_size.width as f32,
        frame.image_size.height as f32,
        KEYPOINT_GRID_CELL_SIZE,
    );

    // Narrow search first.
    let mut matches = match_by_projection(
        map_points,
        &keypoints_undist,
        &frame.features.descriptors,
        &grid,
        pose,
        camera,
        frame.image_size,
        config,
    );

    // Widen if too few matches.
    if matches.len() < MIN_MATCHES_BEFORE_WIDE {
        matches = match_by_projection(
            map_points,
            &keypoints_undist,
            &frame.features.descriptors,
            &grid,
            pose,
            camera,
            frame.image_size,
            ProjectionMatchConfig {
                search_radius: config.search_radius * 2.0,
                ..config
            },
        );
    }

    FrameMatchResult {
        matches,
        keypoints_undist,
        grid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::{Mat3F64, Vec3F64};

    #[test]
    fn test_empty_grid() {
        let grid = KeypointGrid::new(&[], 640.0, 480.0, 64.0);
        let result = grid.query_radius(320.0, 240.0, 50.0, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_point_found() {
        let kps = [[100.0, 100.0]];
        let grid = KeypointGrid::new(&kps, 640.0, 480.0, 64.0);

        let result = grid.query_radius(100.0, 100.0, 10.0, &kps);
        assert_eq!(result, vec![0]);

        let result = grid.query_radius(500.0, 400.0, 10.0, &kps);
        assert!(result.is_empty());
    }

    #[test]
    fn test_radius_boundary() {
        let kps = [[50.0, 50.0], [60.0, 50.0], [100.0, 50.0]];
        let grid = KeypointGrid::new(&kps, 640.0, 480.0, 32.0);

        let mut result = grid.query_radius(55.0, 50.0, 15.0, &kps);
        result.sort();
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn test_corner_clamping() {
        let kps = [[0.0, 0.0], [639.0, 479.0]];
        let grid = KeypointGrid::new(&kps, 640.0, 480.0, 64.0);

        let result = grid.query_radius(0.0, 0.0, 5.0, &kps);
        assert_eq!(result, vec![0]);

        let result = grid.query_radius(639.0, 479.0, 5.0, &kps);
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_multiple_points_per_cell() {
        let kps = [[10.0, 10.0], [12.0, 11.0], [15.0, 13.0]];
        let grid = KeypointGrid::new(&kps, 640.0, 480.0, 64.0);

        let mut result = grid.query_radius(12.0, 12.0, 20.0, &kps);
        result.sort();
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_match_by_projection_simple() {
        let camera = PinholeCamera {
            fx: 200.0,
            fy: 200.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };

        let desc_a = [0u8; 32];
        let map_points = vec![MapPoint {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            descriptor: desc_a,
            keyframe_idx: 0,
            n_visible: 0,
            n_found: 0,
            culled: false,
        }];

        let keypoints_xy = vec![[320.0f32, 240.0], [100.0, 100.0]];
        let descriptors = vec![desc_a, [0xFF; 32]];

        let grid = KeypointGrid::new(&keypoints_xy, 640.0, 480.0, 64.0);
        let pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO);

        let matches = match_by_projection(
            &map_points,
            &keypoints_xy,
            &descriptors,
            &grid,
            &pose,
            &camera,
            ImageSize {
                width: 640.0,
                height: 480.0,
            },
            ProjectionMatchConfig {
                min_depth: 0.0,
                search_radius: 15.0,
                max_hamming: 50,
            },
        );

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0], (0, 0));
    }

    #[test]
    fn test_behind_camera_rejected() {
        let camera = PinholeCamera {
            fx: 200.0,
            fy: 200.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };

        let map_points = vec![MapPoint {
            position: Vec3F64::new(0.0, 0.0, -5.0),
            descriptor: [0u8; 32],
            keyframe_idx: 0,
            n_visible: 0,
            n_found: 0,
            culled: false,
        }];

        let keypoints_xy = vec![[320.0f32, 240.0]];
        let descriptors = vec![[0u8; 32]];
        let grid = KeypointGrid::new(&keypoints_xy, 640.0, 480.0, 64.0);

        let matches = match_by_projection(
            &map_points,
            &keypoints_xy,
            &descriptors,
            &grid,
            &Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO),
            &camera,
            ImageSize {
                width: 640.0,
                height: 480.0,
            },
            ProjectionMatchConfig {
                min_depth: 0.0,
                search_radius: 15.0,
                max_hamming: 50,
            },
        );

        assert!(matches.is_empty());
    }

    #[test]
    fn test_high_hamming_rejected() {
        let camera = PinholeCamera {
            fx: 200.0,
            fy: 200.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };

        let map_points = vec![MapPoint {
            position: Vec3F64::new(0.0, 0.0, 5.0),
            descriptor: [0u8; 32],
            keyframe_idx: 0,
            n_visible: 0,
            n_found: 0,
            culled: false,
        }];

        let keypoints_xy = vec![[320.0f32, 240.0]];
        let descriptors = vec![[0xFF; 32]];

        let grid = KeypointGrid::new(&keypoints_xy, 640.0, 480.0, 64.0);

        let matches = match_by_projection(
            &map_points,
            &keypoints_xy,
            &descriptors,
            &grid,
            &Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO),
            &camera,
            ImageSize {
                width: 640.0,
                height: 480.0,
            },
            ProjectionMatchConfig {
                min_depth: 0.0,
                search_radius: 15.0,
                max_hamming: 50,
            },
        );

        assert!(matches.is_empty());
    }
}
