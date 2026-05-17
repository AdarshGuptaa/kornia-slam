//! Smoke test for OAK-D via the `depthai` Rust crate.
//!
//! Opens the device, configures one CamA mono output at 640x400/30fps, and
//! waits up to 10 s for the first frame. Prints `(w, h)` and exits 0 on
//! success; non-zero otherwise.

use std::time::{Duration, Instant};

use depthai::camera::{CameraNode, CameraOutputConfig};
use depthai::common::{CameraBoardSocket, ImageFrameType};
use depthai::{Device, Pipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("[oakd_smoke] opening device…");
    let device = Device::new()?;
    eprintln!("[oakd_smoke] platform: {:?}", device.platform()?);

    eprintln!("[oakd_smoke] building pipeline…");
    let pipeline = Pipeline::new().with_device(&device).build()?;
    let camera = pipeline.create_with::<CameraNode, _>(CameraBoardSocket::CamA)?;
    let out = camera.request_output(CameraOutputConfig {
        size: (640, 400),
        frame_type: Some(ImageFrameType::GRAY8),
        fps: Some(30.0),
        ..Default::default()
    })?;
    let queue = out.create_queue(4, false)?;

    eprintln!("[oakd_smoke] starting pipeline…");
    pipeline.start()?;

    eprintln!("[oakd_smoke] waiting for first frame (timeout 10 s)…");
    let start = Instant::now();
    loop {
        if let Some(frame) = queue.try_next()? {
            let t = start.elapsed().as_secs_f64();
            eprintln!(
                "[oakd_smoke] got frame in {:.2}s: {}x{}",
                t,
                frame.width(),
                frame.height()
            );
            break;
        }
        if start.elapsed() > Duration::from_secs(10) {
            return Err("timeout waiting for first frame".into());
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    eprintln!("[oakd_smoke] OK");
    Ok(())
}
