//! AprilTag-based metric anchor.
//!
//! Observes a known physical tag from streamed keyframes and accumulates
//! map-frame snapshots of the tag's four corners. Once the tag has been seen
//! from at least two distinct keyframes, [`AprilTagAnchor::solve`] fits a
//! Sim3 (scale, rotation, translation) mapping the SLAM map frame onto the
//! metric tag frame, which can then be applied globally with
//! [`crate::map::Map::apply_world_sim3`].

use std::collections::HashSet;

use thiserror::Error;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
pub use kornia_apriltag::family::TagFamilyKind;
use kornia_apriltag::{AprilTagDecoder, DecodeTagsConfig};
use kornia_image::{Image, ImageSize};

use crate::estimation::sim3::{HuberIrlsConfig, Sim3Alignment, align_sim3_robust};

/// Configuration for the AprilTag anchor.
#[derive(Debug, Clone)]
pub struct AprilTagAnchorConfig {
    /// Decode only this tag id.
    pub tag_id: u16,
    /// Physical tag size in metres (square tag edge length).
    pub tag_size_m: f64,
    /// Tag family used by the decoder.
    pub family: TagFamilyKind,
    /// Minimum acceptable decision margin (tag finder confidence score).
    pub min_decision_margin: f32,
    /// Maximum tolerable Hamming distance from the canonical tag code.
    pub max_hamming: u8,
    /// Robust-weights configuration used by [`AprilTagAnchor::solve`].
    pub robust: HuberIrlsConfig,
}

impl Default for AprilTagAnchorConfig {
    fn default() -> Self {
        Self {
            tag_id: 0,
            tag_size_m: 0.2,
            family: TagFamilyKind::Tag36H11,
            min_decision_margin: 30.0,
            max_hamming: 0,
            robust: HuberIrlsConfig::default(),
        }
    }
}

/// One map-frame snapshot of the anchor tag's four corners.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TagObservation {
    /// Index of the keyframe in which the tag was observed.
    pub kf_idx: usize,
    /// Tag corners in map frame, ordered top-right, bottom-right,
    /// bottom-left, top-left (matching the decoder's quad corner winding).
    pub corners_map: [Vec3F64; 4],
}

/// Errors produced by [`AprilTagAnchor::solve`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum AprilTagAnchorError {
    /// Too few distinct keyframes have observed the tag to constrain the full
    /// similarity transform.
    #[error(
        "insufficient tag observations: {keyframes} distinct keyframe(s) observed the tag (need at least 2)"
    )]
    InsufficientObservations {
        /// Number of distinct keyframes that observed the tag.
        keyframes: usize,
    },
    /// The pipeline has no AprilTag anchor configured.
    #[error("no AprilTag anchor configured")]
    AnchorNotConfigured,
}

/// Accumulates anchor-tag observations across keyframes.
pub struct AprilTagAnchor {
    config: AprilTagAnchorConfig,
    /// Lazily constructed on the first keyframe, once the image size is known.
    decoder: Option<AprilTagDecoder>,
    observations: Vec<TagObservation>,
}

/// Canonical tag-corner positions in the tag frame, ordered top-right,
/// bottom-right, bottom-left, top-left. This winding matches
/// `Detection::estimate_pose`'s object points (`(+s, -s)`, `(+s, +s)`,
/// `(-s, +s)`, `(-s, -s)`, z = 0), so each observed corner lands on the same
/// canonical coordinate across keyframes.
fn canonical_corners(tag_size_m: f64) -> [Vec3F64; 4] {
    let s = tag_size_m / 2.0;
    [
        Vec3F64::new(s, -s, 0.0),  // TR
        Vec3F64::new(s, s, 0.0),   // BR
        Vec3F64::new(-s, s, 0.0),  // BL
        Vec3F64::new(-s, -s, 0.0), // TL
    ]
}

impl AprilTagAnchor {
    /// Creates an anchor with the given configuration.
    pub fn new(config: AprilTagAnchorConfig) -> Self {
        Self {
            config,
            decoder: None,
            observations: Vec::new(),
        }
    }

    /// Returns the accumulated observations.
    pub fn observations(&self) -> &[TagObservation] {
        &self.observations
    }

    /// Total tag detections accumulated across keyframes.
    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }

    /// Number of distinct keyframes that observed the tag.
    pub fn num_distinct_keyframes(&self) -> usize {
        self.observations
            .iter()
            .map(|o| o.kf_idx)
            .collect::<std::collections::HashSet<_>>()
            .len()
    }

    /// Returns the anchor configuration.
    pub fn config(&self) -> &AprilTagAnchorConfig {
        &self.config
    }

    /// Ingests one keyframe image.
    ///
    /// On the first call the image size is unknown at construction time, so the
    /// decoder is lazily built here. The image is decoded, detections for the
    /// configured tag are filtered (`id`, `hamming`, `decision_margin`), and
    /// the best match's corner geometry is stored in map frame.
    /// Decoding or pose errors are ignored: a failed frame must never poison
    /// the SLAM chain.
    pub fn on_keyframe(
        &mut self,
        kf_idx: usize,
        image: &Image<u8, 1>,
        pose_world_to_cam: &Pose3d,
        camera: &PinholeCamera,
    ) {
        if self.decoder.is_none() {
            let decode_config = match DecodeTagsConfig::new(vec![self.config.family.clone()]) {
                Ok(config) => config,
                Err(_) => return,
            };
            let size = ImageSize {
                width: image.width(),
                height: image.height(),
            };
            self.decoder = match AprilTagDecoder::new(decode_config, size) {
                Ok(decoder) => Some(decoder),
                Err(_) => return,
            };
        }

        let detections = match self.decoder.as_mut() {
            Some(decoder) => match decoder.decode(image) {
                Ok(decoded) => decoded,
                Err(_) => return,
            },
            None => return,
        };

        let Some(detection) = detections
            .into_iter()
            .filter(|det| {
                det.id == self.config.tag_id
                    && det.hamming <= self.config.max_hamming
                    && det.decision_margin >= self.config.min_decision_margin
            })
            .max_by(|a, b| a.decision_margin.total_cmp(&b.decision_margin))
        else {
            return;
        };

        // T_cam<-tag from the planar-pose solver.
        let pose_pair = match detection.estimate_pose(camera, self.config.tag_size_m, 50) {
            Ok(pair) => pair,
            Err(_) => return,
        };

        // Compose into map frame: tag -> cam -> map.
        let tag_to_map = pose_world_to_cam.inverse().compose(&pose_pair.best.pose);
        let corners_map =
            canonical_corners(self.config.tag_size_m).map(|p| tag_to_map.transform_point(&p));

        self.observations.push(TagObservation {
            kf_idx,
            corners_map,
        });
    }

    /// Fits the map -> tag-frame Sim3 from the accumulated observations.
    ///
    /// Observations are screened first: a pose-estimation error corrupts all
    /// four corners of an observation together, so each observation is scored
    /// by the median distance of its corners to the canonical tag corners and
    /// observations scoring more than 4x the global median are dropped
    /// (residual-based robustifiers alone can collapse toward zero scale and
    /// get stuck when a bad tag pose contributes gross outliers). The
    /// survivors are flattened into corner pairs (`corners_map[i]` ->
    /// canonical tag corner `i`) and fitted robustly with `align_sim3_robust`.
    /// At least two distinct keyframes are required.
    pub fn solve(&self) -> Result<Sim3Alignment, AprilTagAnchorError> {
        let distinct: HashSet<usize> = self.observations.iter().map(|o| o.kf_idx).collect();
        if distinct.len() < 2 {
            return Err(AprilTagAnchorError::InsufficientObservations {
                keyframes: distinct.len(),
            });
        }

        let canonical = canonical_corners(self.config.tag_size_m);

        // Score each observation by the median distance of its corners to the
        // canonical corners, then drop observations at > 4x the median score.
        let mut scores: Vec<f64> = self
            .observations
            .iter()
            .map(|obs| observation_score(obs, &canonical))
            .collect();
        let med_score = median_of(&mut scores);
        let screened: Vec<&TagObservation> = if med_score > 0.0 {
            self.observations
                .iter()
                .filter(|obs| observation_score(obs, &canonical) <= 4.0 * med_score)
                .collect()
        } else {
            self.observations.iter().collect()
        };

        let distinct_survivors: HashSet<usize> = screened.iter().map(|o| o.kf_idx).collect();
        if distinct_survivors.len() < 2 {
            return Err(AprilTagAnchorError::InsufficientObservations {
                keyframes: distinct_survivors.len(),
            });
        }

        let mut est = Vec::with_capacity(screened.len() * 4);
        let mut gt = Vec::with_capacity(screened.len() * 4);
        for obs in &screened {
            est.extend(obs.corners_map);
            gt.extend(canonical);
        }

        Ok(align_sim3_robust(&est, &gt, self.config.robust))
    }
}

/// Median of four = average of the two middle values after sorting.
fn median_of(sorted: &mut [f64]) -> f64 {
    let m = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[m]
    } else if m > 0 {
        0.5 * (sorted[m - 1] + sorted[m])
    } else {
        0.0
    }
}

/// Per-observation score: median distance of the observed corners to the
/// canonical tag corners.
fn observation_score(obs: &TagObservation, canonical: &[Vec3F64; 4]) -> f64 {
    let mut ds = [0.0_f64; 4];
    for i in 0..4 {
        ds[i] = (obs.corners_map[i] - canonical[i]).length();
    }
    ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    0.5 * (ds[1] + ds[2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Mat3F64;

    fn vec3_norm(v: Vec3F64) -> f64 {
        (v.x * v.x + v.y * v.y + v.z * v.z).sqrt()
    }

    fn mat3_frobenius_diff(a: &Mat3F64, b: &Mat3F64) -> f64 {
        let d = *a - *b;
        let cols = [d.x_axis, d.y_axis, d.z_axis];
        cols.iter()
            .map(|c| c.x * c.x + c.y * c.y + c.z * c.z)
            .sum::<f64>()
            .sqrt()
    }

    /// Builds observations for two keyframes by transforming the canonical tag
    /// corners through the given tag->map pose (the same convention
    /// `on_keyframe` uses to store `corners_map`).
    #[test]
    fn winding_convention_yields_identity_when_tag_aligned_with_map() {
        // Tag at the map origin, axes aligned: the solved anchor must be the
        // identity Sim3. If the canonical winding order ever diverges from the
        // quad corner order, the fits would disagree and this dies loudly.
        let mut anchor = AprilTagAnchor::new(AprilTagAnchorConfig::default());
        for kf_idx in [0, 1] {
            anchor.observations.push(TagObservation {
                kf_idx,
                corners_map: canonical_corners(anchor.config.tag_size_m),
            });
        }

        let fit = anchor.solve().unwrap();
        assert!((fit.scale - 1.0).abs() < 1e-9);
        assert!(mat3_frobenius_diff(&fit.rotation, &Mat3F64::IDENTITY) < 1e-9);
        assert!(vec3_norm(fit.translation) < 1e-9);
    }

    #[test]
    fn scale_recovery_is_inverse_of_map_scale() {
        // A "synthetic map" whose tag geometry is 0.5x the true metric tag,
        // rotated and translated arbitrarily: solving must recover scale ≈ 2.
        let tag_size_m = 0.2;
        let r_m = rotation_from_axis_angle(Vec3F64::new(0.0, 0.0, 1.0), 0.5);
        let t_m = Vec3F64::new(1.0, 2.0, 0.5);
        let scale_map = 0.5;

        let mut anchor = AprilTagAnchor::new(AprilTagAnchorConfig {
            tag_size_m,
            ..Default::default()
        });
        for kf_idx in [0, 1, 2] {
            let corners_map = canonical_corners(tag_size_m).map(|p| scale_map * (r_m * p + t_m));
            anchor.observations.push(TagObservation {
                kf_idx,
                corners_map,
            });
        }

        let fit = anchor.solve().unwrap();
        // Map -> tag: scale closes the 0.5x map factor, rotation/translation
        // invert the tag->map pose.
        let expected_rot = r_m.transpose();
        assert!((fit.scale - 2.0).abs() < 1e-9, "scale error: {}", fit.scale);
        assert!(mat3_frobenius_diff(&fit.rotation, &expected_rot) < 1e-9);
        let expected_t = -(expected_rot * t_m);
        assert!(vec3_norm(fit.translation - expected_t) < 1e-9);
    }

    #[test]
    fn robust_solve_ignores_corrupted_observations() {
        let tag_size_m = 0.2;
        let r_m = rotation_from_axis_angle(Vec3F64::new(0.0, 0.0, 1.0), 1.1);
        let t_m = Vec3F64::new(0.5, -0.5, 1.0);
        let scale_map = 0.5;

        let mut anchor = AprilTagAnchor::new(AprilTagAnchorConfig::default());
        for kf_idx in 0..10 {
            let mut corners_map =
                canonical_corners(tag_size_m).map(|p| scale_map * (r_m * p + t_m));
            // Grossly corrupt two of the ten observations.
            if kf_idx == 3 || kf_idx == 6 {
                for c in corners_map.iter_mut() {
                    *c += Vec3F64::new(5.0, -4.0, 3.0);
                }
            }
            anchor.observations.push(TagObservation {
                kf_idx,
                corners_map,
            });
        }

        let fit = anchor.solve().unwrap();
        let expected_rot = r_m.transpose();
        let expected_t = -(expected_rot * t_m);
        assert!((fit.scale - 2.0).abs() < 1e-3, "scale error: {}", fit.scale);
        assert!(mat3_frobenius_diff(&fit.rotation, &expected_rot) < 5e-3);
        assert!(vec3_norm(fit.translation - expected_t) < 1e-2);
    }

    #[test]
    fn solve_rejects_single_keyframe() {
        let mut anchor = AprilTagAnchor::new(AprilTagAnchorConfig::default());
        anchor.observations.push(TagObservation {
            kf_idx: 0,
            corners_map: canonical_corners(anchor.config.tag_size_m),
        });

        let err = match anchor.solve() {
            Err(err) => err,
            Ok(_) => panic!("expected solve to fail with a single keyframe"),
        };
        assert_eq!(
            err,
            AprilTagAnchorError::InsufficientObservations { keyframes: 1 }
        );
    }

    fn rotation_from_axis_angle(axis: Vec3F64, angle: f64) -> Mat3F64 {
        let len = axis.length();
        let (nx, ny, nz) = (axis.x / len, axis.y / len, axis.z / len);
        let (c, s, ti) = (angle.cos(), angle.sin(), 1.0 - angle.cos());

        let col0 = Vec3F64::new(
            ti * nx * nx + c,
            ti * nx * ny + s * nz,
            ti * nx * nz - s * ny,
        );
        let col1 = Vec3F64::new(
            ti * nx * ny - s * nz,
            ti * ny * ny + c,
            ti * ny * nz + s * nx,
        );
        let col2 = Vec3F64::new(
            ti * nx * nz + s * ny,
            ti * ny * nz - s * nx,
            ti * nz * nz + c,
        );
        Mat3F64::from_cols(col0, col1, col2)
    }
}
