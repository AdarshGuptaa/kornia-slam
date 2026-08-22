//! Persistent KLT track state and map-point associations.

use std::collections::HashMap;

use kornia_image::Image;
use kornia_imgproc::optical_flow_pyr_lk::{
    BorderMode, PyrLKError, PyrLKParams, TermCriteria, calc_optical_flow_pyr_lk,
};

/// One point tracked frame-to-frame by optical flow.
#[derive(Debug, Clone, Copy)]
pub struct Track {
    /// Stable identity across frames.
    pub track_id: u64,
    /// Raw, distorted pixel position in the current frame.
    pub pixel: [f32; 2],
    /// Associated map-point index, if any.
    pub map_point_idx: Option<usize>,
    /// Consecutive frames this track has survived. 1 = just detected.
    pub age: u32,
}

/// Persistent state for points carried between frames.
#[derive(Debug, Default)]
pub struct TrackState {
    next_track_id: u64,
    tracks: Vec<Track>,
}

/// KLT solver parameters and survivor acceptance thresholds.
#[derive(Debug)]
pub struct OpticalFlowConfig {
    /// LK integration window size in pixels (currently required to be 21).
    pub win_size: usize,
    /// Pyramid depth: more levels track larger displacements, at higher cost
    /// and less precision at the coarsest level.
    pub max_level: usize,
    /// Gauss-Newton iterations per pyramid level before giving up.
    pub max_iter: usize,
    /// Convergence threshold on the incremental flow update.
    pub epsilon: f32,
    /// Minimum structure-tensor eigenvalue for a patch to be considered
    /// trackable at all (rejects flat/aliased regions).
    pub min_eigen_threshold: f32,
    /// Seed the search from a provided initial-flow estimate.
    pub use_initial_flow: bool,
    /// Iteration stopping rule (count / epsilon / both).
    pub term_criteria: TermCriteria,
    /// Sampling policy near image borders during iteration.
    pub border_mode: BorderMode,

    /// Maximum accepted KLT residual when tracking succeeds.
    pub max_error: f32,
    /// Maximum accepted frame-to-frame displacement in pixels.
    pub max_movement_px: f32,
    /// Required distance in pixels from the output image border.
    pub border_margin_px: f32,
}

impl Default for OpticalFlowConfig {
    fn default() -> Self {
        Self {
            win_size: 21,
            max_level: 3,
            max_iter: 30,
            epsilon: 0.01,
            min_eigen_threshold: 1e-4,
            use_initial_flow: false,
            term_criteria: TermCriteria::Both,
            border_mode: BorderMode::Clamp,
            max_error: 20.0,
            max_movement_px: 60.0,
            border_margin_px: 10.0,
        }
    }
}

impl OpticalFlowConfig {
    /// Converts the public configuration into the underlying LK parameters.
    fn to_pyr_lk_params(&self) -> PyrLKParams {
        PyrLKParams {
            win_size: self.win_size,
            max_level: self.max_level,
            max_iter: self.max_iter,
            epsilon: self.epsilon,
            min_eigen_threshold: self.min_eigen_threshold,
            use_initial_flow: self.use_initial_flow,
            term_criteria: self.term_criteria.clone(),
            border_mode: self.border_mode,
        }
    }
}

impl TrackState {
    /// Creates an empty track state.
    pub fn new() -> Self {
        Self {
            next_track_id: 0,
            tracks: Vec::new(),
        }
    }

    /// Returns current tracks in KLT input order.
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Number of points currently tracked.
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Returns pixel positions in the same order as [`Self::tracks`].
    pub fn pixels(&self) -> Vec<[f32; 2]> {
        self.tracks.iter().map(|t| t.pixel).collect()
    }

    /// Creates a uniquely identified, unassociated track.
    pub fn new_track(&mut self, pixel: [f32; 2]) -> Track {
        let track_id = self.next_track_id;
        self.next_track_id += 1;
        Track {
            track_id,
            pixel,
            map_point_idx: None,
            age: 1,
        }
    }

    /// Replaces the current track set.
    pub fn set_tracks(&mut self, tracks: Vec<Track>) {
        self.tracks = tracks;
    }

    /// Tracks current points between raw grayscale images and returns the
    /// accepted survivors without mutating this state.
    pub fn track(
        &self,
        prev_img: &Image<u8, 1>,
        next_img: &Image<u8, 1>,
        config: &OpticalFlowConfig,
    ) -> Result<Vec<Track>, PyrLKError> {
        if self.is_empty() {
            return Ok(Vec::new());
        }

        // Every u8 value is exactly representable as f32.
        let prev_f32 = prev_img
            .cast::<f32>()
            .expect("u8 -> f32 image cast is always representable");
        let next_f32 = next_img
            .cast::<f32>()
            .expect("u8 -> f32 image cast is always representable");

        let prev_pts = self.pixels();
        let result = calc_optical_flow_pyr_lk(
            &prev_f32,
            &next_f32,
            &prev_pts,
            None,
            &config.to_pyr_lk_params(),
        )?;

        let width = next_img.width() as f32;
        let height = next_img.height() as f32;
        let margin = config.border_margin_px;

        let mut survivors = Vec::with_capacity(self.tracks.len());
        for (i, track) in self.tracks.iter().enumerate() {
            if result.status[i] == 0 {
                continue; // KLT itself failed for this point.
            }
            if result.error[i] > config.max_error {
                continue;
            }
            let next_pt = result.next_pts[i];
            if next_pt[0] < margin
                || next_pt[1] < margin
                || next_pt[0] > width - margin
                || next_pt[1] > height - margin
            {
                continue; // Too close to the edge to trust.
            }
            let dx = next_pt[0] - track.pixel[0];
            let dy = next_pt[1] - track.pixel[1];
            if (dx * dx + dy * dy).sqrt() > config.max_movement_px {
                continue; // Implausible jump — likely aliased onto the wrong patch.
            }
            survivors.push(Track {
                track_id: track.track_id,
                pixel: next_pt,
                map_point_idx: track.map_point_idx,
                age: track.age + 1,
            });
        }
        Ok(survivors)
    }

    /// Rebuilds tracks from `(map_point_idx, keypoint_idx)` matches, preserving
    /// identity and age for map points already present in the state.
    pub fn refresh_from_matches(&mut self, matches: &[(usize, usize)], keypoints: &[[f32; 2]]) {
        let mut old_by_mp: HashMap<usize, Track> = self
            .tracks
            .drain(..)
            .filter_map(|t| t.map_point_idx.map(|mp| (mp, t)))
            .collect();

        let mut new_tracks = Vec::with_capacity(matches.len());
        for &(mp_idx, kp_idx) in matches {
            let Some(&pixel) = keypoints.get(kp_idx) else {
                continue;
            };
            let track = match old_by_mp.remove(&mp_idx) {
                Some(old) => Track {
                    track_id: old.track_id,
                    pixel,
                    map_point_idx: Some(mp_idx),
                    age: old.age + 1,
                },
                None => {
                    let mut t = self.new_track(pixel);
                    t.map_point_idx = Some(mp_idx);
                    t
                }
            };
            new_tracks.push(track);
        }
        self.tracks = new_tracks;
    }
}

/// Finds the nearest keypoint within `radius_px`.
fn nearest_keypoint(predicted: [f32; 2], keypoints: &[[f32; 2]], radius_px: f32) -> Option<usize> {
    let r2 = radius_px * radius_px;
    let mut best: Option<(usize, f32)> = None;
    for (i, &kp) in keypoints.iter().enumerate() {
        let dx = kp[0] - predicted[0];
        let dy = kp[1] - predicted[1];
        let d2 = dx * dx + dy * dy;
        if d2 > r2 {
            continue;
        }
        match best {
            Some((_, best_d2)) if d2 >= best_d2 => {}
            _ => best = Some((i, d2)),
        }
    }
    best.map(|(i, _)| i)
}

/// Snaps associated survivors to detected keypoints, producing
/// `(map_point_idx, keypoint_idx)` correspondences.
pub fn snap_survivors(
    survivors: &[Track],
    curr_keypoints: &[[f32; 2]],
    snap_radius_px: f32,
) -> Vec<(usize, usize)> {
    let mut correspondences = Vec::new();
    for t in survivors {
        if let Some(mp_idx) = t.map_point_idx
            && let Some(kp_idx) = nearest_keypoint(t.pixel, curr_keypoints, snap_radius_px)
        {
            correspondences.push((mp_idx, kp_idx));
        }
    }
    correspondences
}

/// Tracks existing points and snaps survivors to current-frame keypoints.
pub fn klt_correspondences(
    track_state: &TrackState,
    prev_img: &Image<u8, 1>,
    next_img: &Image<u8, 1>,
    curr_keypoints: &[[f32; 2]],
    config: &OpticalFlowConfig,
    snap_radius_px: f32,
) -> Result<Vec<(usize, usize)>, PyrLKError> {
    let survivors = track_state.track(prev_img, next_img, config)?;
    Ok(snap_survivors(&survivors, curr_keypoints, snap_radius_px))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    #[test]
    fn new_track_gets_unique_ids() {
        let mut state = TrackState::new();
        let a = state.new_track([1.0, 2.0]);
        let b = state.new_track([3.0, 4.0]);
        assert_ne!(a.track_id, b.track_id);
        assert_eq!(a.map_point_idx, None);
        assert_eq!(a.age, 1);
    }

    #[test]
    fn pixels_are_positional_with_tracks() {
        let mut state = TrackState::new();
        let t0 = state.new_track([1.0, 2.0]);
        let t1 = state.new_track([3.0, 4.0]);
        state.set_tracks(vec![t0, t1]);
        assert_eq!(state.pixels(), vec![[1.0, 2.0], [3.0, 4.0]]);
    }

    /// Builds a black image with a white square whose corner is trackable.
    fn square_image(origin: [usize; 2]) -> Image<u8, 1> {
        const SIZE: usize = 200;
        let mut data = vec![0u8; SIZE * SIZE];
        for y in origin[1]..(origin[1] + 40).min(SIZE) {
            for x in origin[0]..(origin[0] + 40).min(SIZE) {
                data[y * SIZE + x] = 255;
            }
        }
        Image::new(
            ImageSize {
                width: SIZE,
                height: SIZE,
            },
            data,
        )
        .unwrap()
    }

    #[test]
    fn track_follows_a_known_shift() {
        let prev_img = square_image([90, 90]);
        let next_img = square_image([93, 92]); // shifted by (+3, +2)

        let mut state = TrackState::new();
        // Seed the square's top-left corner, which has gradients on both axes.
        let t = state.new_track([90.0, 90.0]);
        state.set_tracks(vec![t]);

        let survivors = state
            .track(&prev_img, &next_img, &OpticalFlowConfig::default())
            .expect("KLT should succeed on a clean synthetic shift");

        assert_eq!(survivors.len(), 1, "the corner should track successfully");
        let got = survivors[0].pixel;
        assert!(
            (got[0] - 93.0).abs() < 1.5 && (got[1] - 92.0).abs() < 1.5,
            "expected tracked pixel near (93, 92), got {got:?}"
        );
        assert_eq!(survivors[0].track_id, t.track_id);
        assert_eq!(survivors[0].age, 2);
    }

    #[test]
    fn refresh_from_matches_preserves_identity_for_known_map_points() {
        let mut state = TrackState::new();
        let mut t0 = state.new_track([1.0, 2.0]);
        t0.map_point_idx = Some(42);
        state.set_tracks(vec![t0]);

        // Map point 42 persists while map point 7 appears for the first time.
        let keypoints = [[5.0, 6.0], [10.0, 11.0]];
        state.refresh_from_matches(&[(42, 0), (7, 1)], &keypoints);

        let tracks = state.tracks();
        assert_eq!(tracks.len(), 2);
        let mp42 = tracks.iter().find(|t| t.map_point_idx == Some(42)).unwrap();
        assert_eq!(mp42.track_id, t0.track_id, "identity must carry forward");
        assert_eq!(mp42.age, 2);
        assert_eq!(mp42.pixel, [5.0, 6.0]);

        let mp7 = tracks.iter().find(|t| t.map_point_idx == Some(7)).unwrap();
        assert_ne!(mp7.track_id, t0.track_id, "new map point gets a fresh id");
        assert_eq!(mp7.age, 1);
    }
}
