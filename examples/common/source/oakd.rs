//! Live OAK-D mono camera as a [`FrameSource`].
//!
//! Pulls GRAY8 frames from CamB at the requested resolution and feeds them
//! through the same `process_frame` orchestrator as the EuRoC dataset.
//! Intrinsics are placeholder (rough scale of the OAK-D Pro factory fx/fy at
//! 1280×800); reading the on-device factory calibration is a TODO.

use std::time::{Duration, Instant};

use depthai::camera::{CameraNode, CameraOutputConfig, OutputQueue};
use depthai::common::{CameraBoardSocket, ImageFrameType};
use depthai::{Device, Pipeline as DaiPipeline};
use kornia_3d::camera::PinholeCamera;
use kornia_image::{Image, ImageSize};
use kornia_tensor::CpuAllocator;

use super::{FrameItem, FrameSource, SourceError};

/// Live OAK-D mono frame source.
pub struct OakdSource {
    // Order matters for drop: the queue must outlive the pipeline/device only
    // if depthai needs it that way. The crate stores everything by-value with
    // shared_ptr semantics on the C++ side, so we keep all handles alive in
    // this struct and let drop run in field-declared order (top-to-bottom).
    queue: OutputQueue,
    _pipeline: DaiPipeline,
    _device: Device,
    image_size: ImageSize,
    n_pixels: usize,
    camera: PinholeCamera,
    max_frames: usize,
    cursor: usize,
    start: Instant,
}

impl OakdSource {
    /// Opens the device and starts a single GRAY8 stream from CamB.
    ///
    /// `max_frames == 0` ⇒ run until the caller stops the loop (Ctrl-C).
    pub fn open(width: u32, height: u32, fps: f32, max_frames: usize) -> Result<Self, SourceError> {
        eprintln!("[oakd] opening device…");
        let device = Device::new().map_err(SourceError::other)?;
        let platform = device.platform().map_err(SourceError::other)?;
        eprintln!("[oakd] platform: {platform:?}");

        let pipeline = DaiPipeline::new()
            .with_device(&device)
            .build()
            .map_err(SourceError::other)?;
        let cam = pipeline
            .create_with::<CameraNode, _>(CameraBoardSocket::CamB)
            .map_err(SourceError::other)?;
        let out = cam
            .request_output(CameraOutputConfig {
                size: (width, height),
                frame_type: Some(ImageFrameType::GRAY8),
                fps: Some(fps),
                ..Default::default()
            })
            .map_err(SourceError::other)?;
        let queue = out.create_queue(4, false).map_err(SourceError::other)?;
        pipeline.start().map_err(SourceError::other)?;

        let image_size = ImageSize {
            width: width as usize,
            height: height as usize,
        };
        let n_pixels = image_size.width * image_size.height;

        // Placeholder intrinsics: OAK-D Pro mono is 1280×800 native with
        // roughly fx≈fy≈880 px; principal point at image center after scale.
        let scale = width as f64 / 1280.0;
        let camera = PinholeCamera {
            fx: 880.0 * scale,
            fy: 880.0 * scale,
            cx: width as f64 * 0.5,
            cy: height as f64 * 0.5,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        eprintln!(
            "[oakd] placeholder intrinsics: fx={:.2} fy={:.2} cx={:.2} cy={:.2}",
            camera.fx, camera.fy, camera.cx, camera.cy
        );

        Ok(Self {
            queue,
            _pipeline: pipeline,
            _device: device,
            image_size,
            n_pixels,
            camera,
            max_frames,
            cursor: 0,
            start: Instant::now(),
        })
    }
}

impl FrameSource for OakdSource {
    fn camera(&self) -> PinholeCamera {
        self.camera.clone()
    }

    fn n_frames_hint(&self) -> Option<usize> {
        if self.max_frames > 0 {
            Some(self.max_frames)
        } else {
            None
        }
    }

    fn next_frame(&mut self) -> Result<Option<FrameItem>, SourceError> {
        if self.max_frames > 0 && self.cursor >= self.max_frames {
            return Ok(None);
        }
        loop {
            let frame_msg = match self.queue.try_next().map_err(SourceError::other)? {
                Some(f) => f,
                None => {
                    std::thread::sleep(Duration::from_millis(1));
                    continue;
                }
            };
            if frame_msg.width() as usize != self.image_size.width
                || frame_msg.height() as usize != self.image_size.height
            {
                eprintln!(
                    "[oakd] unexpected frame size: {}x{} (want {}x{})",
                    frame_msg.width(),
                    frame_msg.height(),
                    self.image_size.width,
                    self.image_size.height,
                );
                continue;
            }
            let bytes = frame_msg.bytes();
            if bytes.len() != self.n_pixels {
                eprintln!(
                    "[oakd] unexpected payload {} bytes (want {})",
                    bytes.len(),
                    self.n_pixels,
                );
                continue;
            }
            let image: Image<u8, 1, CpuAllocator> =
                Image::new(self.image_size, bytes, CpuAllocator).map_err(SourceError::other)?;
            let idx = self.cursor;
            self.cursor += 1;
            let timestamp_sec = self.start.elapsed().as_secs_f64();
            return Ok(Some(FrameItem {
                idx,
                timestamp_sec,
                image,
                right_image: None,
            }));
        }
    }
}
