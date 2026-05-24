//! Frame sources for SLAM examples.
//!
//! Both offline datasets (EuRoC) and live cameras (OAK-D) feed the same
//! `process_frame` loop. This module exposes a single trait, [`FrameSource`],
//! so the main binary can stay source-agnostic.

pub mod euroc;
#[cfg(feature = "oakd")]
pub mod oakd;
#[cfg(feature = "webcam")]
pub mod webcam;

use kornia_3d::camera::PinholeCamera;
use kornia_image::Image;
use kornia_tensor::CpuAllocator;

pub use euroc::EurocSource;
#[cfg(feature = "oakd")]
pub use oakd::OakdSource;
#[cfg(feature = "webcam")]
pub use webcam::WebcamSource;

/// One frame yielded by a source.
pub struct FrameItem {
    /// Absolute frame index (source-defined; EuRoC counts samples, OAK-D counts received frames).
    pub idx: usize,
    /// Capture timestamp in seconds (host clock for live sources).
    #[allow(dead_code)]
    pub timestamp_sec: f64,
    /// Grayscale image.
    pub image: Image<u8, 1, CpuAllocator>,
}

/// Pull-based interface for monocular SLAM frame producers.
///
/// `next_frame` returns `Ok(None)` when the stream is exhausted. Offline
/// datasets exhaust after their last sample; live sources may exhaust when
/// a CLI-imposed cap is reached.
pub trait FrameSource {
    /// Camera intrinsics. Must be valid before the first `next_frame` call.
    fn camera(&self) -> PinholeCamera;

    /// Total frames the source will yield, if known.
    ///
    /// Live sources without a cap return `None`. The TUI uses this to render
    /// a progress bar; absent it, the bar shows elapsed-only.
    fn n_frames_hint(&self) -> Option<usize>;

    /// Pull the next frame. `Ok(None)` ⇒ end of stream.
    fn next_frame(&mut self) -> Result<Option<FrameItem>, SourceError>;
}

/// Errors returned from a [`FrameSource`].
#[derive(thiserror::Error, Debug)]
pub enum SourceError {
    /// I/O error reading a frame from disk.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Other error (dataset parse, image decode, device error).
    #[error("{0}")]
    Other(Box<dyn std::error::Error + Send + Sync + 'static>),
}

impl SourceError {
    /// Wrap any boxable error as a `SourceError::Other`.
    pub fn other<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync + 'static>>,
    {
        Self::Other(err.into())
    }
}
