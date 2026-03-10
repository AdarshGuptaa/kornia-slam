//! Dataset readers for visual odometry benchmarks.

use std::{fs::File, io::BufRead, io::BufReader, path::Path, path::PathBuf};

/// Error type used by dataset readers.
#[derive(thiserror::Error, Debug)]
pub enum DatasetError {
    /// Generic I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Parse failure with contextual message.
    #[error("parse error: {0}")]
    Parse(String),

    /// Referenced file does not exist.
    #[error("file not found: {0}")]
    FileNotFound(PathBuf),
}

/// One dataset image sample.
#[derive(Debug, Clone)]
pub struct DatasetSample {
    /// Timestamp in seconds.
    pub timestamp_sec: f64,
    /// Path to the image file.
    pub image_path: PathBuf,
}

/// One ground-truth pose from `state_groundtruth_estimate0/data.csv`.
#[derive(Debug, Clone, Copy)]
pub struct GroundTruthPose {
    /// Timestamp in seconds.
    pub timestamp_sec: f64,
    /// Position x (meters).
    pub tx: f64,
    /// Position y (meters).
    pub ty: f64,
    /// Position z (meters).
    pub tz: f64,
    /// Quaternion scalar part.
    pub qw: f64,
    /// Quaternion x.
    pub qx: f64,
    /// Quaternion y.
    pub qy: f64,
    /// Quaternion z.
    pub qz: f64,
}

/// Reader for the EuRoC MAV dataset (ASL format).
///
/// Expects `<root>/mav0/cam0/data.csv` with nanosecond timestamps and
/// PNG images in `<root>/mav0/cam0/data/`.
#[derive(Debug, Clone)]
pub struct EurocDataset {
    /// Base directory of the extracted dataset.
    pub root: std::path::PathBuf,
    /// Ordered camera samples.
    pub cam0_samples: Vec<DatasetSample>,
    /// Ground-truth poses (empty if GT file not present).
    pub ground_truth: Vec<GroundTruthPose>,
}

impl EurocDataset {
    /// Opens the dataset from `<root>/mav0/cam0/data.csv`.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, DatasetError> {
        let root = root.as_ref().to_path_buf();
        let csv = root.join("mav0").join("cam0").join("data.csv");
        let data_dir = root.join("mav0").join("cam0").join("data");

        if !csv.exists() {
            return Err(DatasetError::FileNotFound(csv));
        }
        if !data_dir.exists() {
            return Err(DatasetError::FileNotFound(data_dir));
        }

        let file = File::open(&csv)?;
        let reader = BufReader::new(file);
        let mut samples = Vec::new();

        for (line_idx, line) in reader.lines().enumerate() {
            let line = line?;
            // Skip header line (starts with '#').
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let mut parts = line.split(',');
            let ts_str = parts.next().ok_or_else(|| {
                DatasetError::Parse(format!("missing timestamp at line {}", line_idx + 1))
            })?;
            let file_str = parts.next().ok_or_else(|| {
                DatasetError::Parse(format!("missing filename at line {}", line_idx + 1))
            })?;

            let timestamp_ns = ts_str.trim().parse::<u64>().map_err(|e| {
                DatasetError::Parse(format!("invalid timestamp at line {}: {e}", line_idx + 1))
            })?;
            let timestamp_sec = timestamp_ns as f64 * 1e-9;

            samples.push(DatasetSample {
                timestamp_sec,
                image_path: data_dir.join(file_str.trim()),
            });
        }

        let ground_truth = Self::load_ground_truth(&root);

        Ok(Self {
            root,
            cam0_samples: samples,
            ground_truth,
        })
    }

    /// Returns ordered cam0 samples.
    pub fn samples(&self) -> &[DatasetSample] {
        &self.cam0_samples
    }

    /// Returns parsed ground-truth poses (possibly empty).
    pub fn ground_truth(&self) -> &[GroundTruthPose] {
        &self.ground_truth
    }

    /// Loads ground-truth poses from `mav0/state_groundtruth_estimate0/data.csv`.
    ///
    /// Returns an empty Vec if the file does not exist.
    fn load_ground_truth(root: &Path) -> Vec<GroundTruthPose> {
        let csv = root
            .join("mav0")
            .join("state_groundtruth_estimate0")
            .join("data.csv");
        let file = match File::open(&csv) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        let reader = BufReader::new(file);
        let mut poses = Vec::new();

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            // Columns: timestamp_ns, px, py, pz, qw, qx, qy, qz, ...
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() < 8 {
                continue;
            }
            let Ok(ts_ns) = cols[0].trim().parse::<u64>() else {
                continue;
            };
            let f = |i: usize| cols[i].trim().parse::<f64>();
            let (Ok(px), Ok(py), Ok(pz), Ok(qw), Ok(qx), Ok(qy), Ok(qz)) =
                (f(1), f(2), f(3), f(4), f(5), f(6), f(7))
            else {
                continue;
            };

            poses.push(GroundTruthPose {
                timestamp_sec: ts_ns as f64 * 1e-9,
                tx: px,
                ty: py,
                tz: pz,
                qw,
                qx,
                qy,
                qz,
            });
        }
        poses
    }
}
