//! EuRoC MAV dataset as a [`FrameSource`].

use std::path::Path;

use kornia_3d::camera::PinholeCamera;
use kornia_io::png::read_image_png_mono8;

use super::{FrameItem, FrameSource, SourceError};
use crate::datasets::EurocDataset;
use crate::datasets::euroc::GroundTruthPose;
/// Reads cam0 PNG frames from an EuRoC dataset in order.
pub struct EurocSource {
    dataset: EurocDataset,
    cursor: usize,
    start: usize,
    end: usize,
}

impl EurocSource {
    /// Opens the dataset and configures the iteration window.
    ///
    /// `max_frames == 0` means "until the dataset is exhausted". `start_frame`
    /// is the index into `cam0_samples` of the first sample to yield; later
    /// samples retain their absolute index in `FrameItem::idx`.
    pub fn open(
        root: impl AsRef<Path>,
        start_frame: usize,
        max_frames: usize,
    ) -> Result<Self, SourceError> {
        let dataset = EurocDataset::open(root).map_err(SourceError::other)?;
        let n = dataset.samples().len();
        let start = start_frame.min(n);
        let end = if max_frames > 0 {
            (start + max_frames).min(n)
        } else {
            n
        };
        Ok(Self {
            dataset,
            cursor: start,
            start,
            end,
        })
    }

    pub fn ground_truth_at(&self, query_sec: f64) -> Option<&GroundTruthPose> {
        let gt = self.dataset.ground_truth();
        if gt.is_empty() {
            return None;
        }
        // Linear scan is fine; GT is ~200 Hz and sequences are short.
        let best = gt.iter().min_by(|a, b| {
            let da = (a.timestamp_sec - query_sec).abs();
            let db = (b.timestamp_sec - query_sec).abs();
            da.partial_cmp(&db).unwrap()
        })?;
        if (best.timestamp_sec - query_sec).abs() < 0.5 {
            Some(best)
        } else {
            None
        }
    }

    pub fn ground_truth_poses_cloned(&self) -> Vec<GroundTruthPose> {
        self.dataset.ground_truth().to_vec()
    }

    /// Total sample count in the dataset (ignoring start/max).
    pub fn dataset_len(&self) -> usize {
        self.dataset.samples().len()
    }
}

impl FrameSource for EurocSource {
    fn camera(&self) -> PinholeCamera {
        self.dataset.camera()
    }

    fn n_frames_hint(&self) -> Option<usize> {
        Some(self.end - self.start)
    }

    fn next_frame(&mut self) -> Result<Option<FrameItem>, SourceError> {
        if self.cursor >= self.end {
            return Ok(None);
        }
        let sample = &self.dataset.samples()[self.cursor];
        let idx = self.cursor;
        let timestamp_sec = sample.timestamp_sec;
        let image = read_image_png_mono8(&sample.image_path)
            .map_err(SourceError::other)?
            .into_inner();
        self.cursor += 1;
        Ok(Some(FrameItem {
            idx,
            timestamp_sec,
            image,
        }))
    }
}
