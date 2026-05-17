//! Live monocular ORB-SLAM on an OAK-D camera.
//!
//! Pulls 640x400 GRAY8 frames from CamB (left mono), runs ORB feature
//! extraction, and feeds the result into the same `Pipeline` orchestrator
//! used by the EuRoC `orb_slam` example.
//!
//! Currently uses hand-tuned intrinsics; reading the on-device factory
//! calibration is a follow-up.

mod config;
mod pipeline;
#[path = "../../orb_slam/src/utils.rs"]
mod utils;

use std::time::{Duration, Instant};

use depthai::camera::{CameraNode, CameraOutputConfig};
use depthai::common::{CameraBoardSocket, ImageFrameType};
use depthai::{Device, Pipeline as DaiPipeline};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_image::{Image, ImageSize};
use kornia_slam::Frame;
use kornia_tensor::CpuAllocator;

use config::PipelineConfig;
use pipeline::Pipeline;

#[cfg(feature = "viz")]
use utils::{
    log_camera_to_rerun, log_frame_to_rerun, log_map_points_to_rerun, log_trajectory_to_rerun,
    trajectory_point_from_pose,
};

/// CLI arguments.
#[derive(argh::FromArgs)]
#[argh(description = "Monocular ORB-SLAM on a live OAK-D mono camera")]
struct Args {
    /// maximum number of frames to process (0 = run forever)
    #[argh(option, default = "0")]
    max_frames: usize,

    /// frame width in pixels (depthai resizes the source to this)
    #[argh(option, default = "640")]
    width: u32,

    /// frame height in pixels
    #[argh(option, default = "400")]
    height: u32,

    /// camera FPS
    #[argh(option, default = "30.0")]
    fps: f32,

    /// spawn a Rerun viewer and stream to it (requires `--features viz`)
    #[argh(switch)]
    #[cfg(feature = "viz")]
    rerun_stream: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // ── Open device + start a single GRAY8 mono stream ────────────────────
    eprintln!("[oakd_slam] opening device…");
    let device = Device::new()?;
    eprintln!("[oakd_slam] platform: {:?}", device.platform()?);

    let dai = DaiPipeline::new().with_device(&device).build()?;
    let cam = dai.create_with::<CameraNode, _>(CameraBoardSocket::CamB)?;
    let out = cam.request_output(CameraOutputConfig {
        size: (args.width, args.height),
        frame_type: Some(ImageFrameType::GRAY8),
        fps: Some(args.fps),
        ..Default::default()
    })?;
    let queue = out.create_queue(4, false)?;
    dai.start()?;

    // ── Camera intrinsics (placeholder — TODO: read from device) ──────────
    // OAK-D Pro mono is 1280x800 native (~880 px focal). Scale to requested
    // output. Principal point assumed image center.
    let scale = args.width as f64 / 1280.0;
    let camera = PinholeCamera {
        fx: 880.0 * scale,
        fy: 880.0 * scale,
        cx: args.width as f64 * 0.5,
        cy: args.height as f64 * 0.5,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    };
    eprintln!(
        "[oakd_slam] placeholder intrinsics: fx={:.2} fy={:.2} cx={:.2} cy={:.2}",
        camera.fx, camera.fy, camera.cx, camera.cy
    );

    let image_size = ImageSize {
        width: args.width as usize,
        height: args.height as usize,
    };
    let n_pixels = image_size.width * image_size.height;

    // ── SLAM ───────────────────────────────────────────────────────────────
    let mut system = Pipeline::new(camera.clone(), PipelineConfig::default());
    let detector = kornia_imgproc::features::OrbDetector {
        n_keypoints: 1000,
        ..Default::default()
    };

    // ── Rerun ──────────────────────────────────────────────────────────────
    #[cfg(feature = "viz")]
    let rec = if args.rerun_stream {
        let r = rerun::RecordingStreamBuilder::new("oakd_slam").spawn()?;
        r.log("/", &rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN())?;
        r.log("world/camera", &rerun::ViewCoordinates::RDF())?;
        Some(r)
    } else {
        None
    };
    let mut trajectory: Vec<[f32; 3]> = Vec::new();

    // ── Main loop ──────────────────────────────────────────────────────────
    eprintln!("[oakd_slam] entering main loop (Ctrl-C to stop)");
    let mut idx: usize = 0;
    let max = args.max_frames;
    let start = Instant::now();
    loop {
        let Some(frame_msg) = queue.try_next()? else {
            std::thread::sleep(Duration::from_millis(1));
            continue;
        };
        if frame_msg.width() as usize != image_size.width
            || frame_msg.height() as usize != image_size.height
        {
            eprintln!(
                "[oakd_slam] unexpected frame size: {}x{}",
                frame_msg.width(),
                frame_msg.height()
            );
            continue;
        }
        let bytes = frame_msg.bytes();
        if bytes.len() != n_pixels {
            eprintln!(
                "[oakd_slam] unexpected payload {} bytes (want {})",
                bytes.len(),
                n_pixels
            );
            continue;
        }
        let gray_u8: Image<u8, 1, CpuAllocator> =
            Image::new(image_size, bytes, CpuAllocator)?;

        let features = detector.detect_and_extract_u8(&gray_u8)?;
        #[cfg(feature = "viz")]
        if let Some(ref rec) = rec {
            log_frame_to_rerun(rec, &gray_u8, &features.keypoints_xy);
        }
        let image_bytes = gray_u8.as_slice();
        let keypoint_colors: Vec<[u8; 3]> = features
            .keypoints_xy
            .iter()
            .map(|kp| {
                let x = (kp[0] as usize).min(image_size.width.saturating_sub(1));
                let y = (kp[1] as usize).min(image_size.height.saturating_sub(1));
                let g = image_bytes[y * image_size.width + x];
                [g, g, g]
            })
            .collect();

        let frame = Frame {
            idx,
            features,
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size,
            keypoint_colors,
        };

        let t0 = Instant::now();
        let result = system.process_frame(frame);
        let frame_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let kf = system.current_keyframe_idx().unwrap_or(idx);
        let mp = system.num_active_map_points();

        eprintln!(
            "[{idx:>5}] {:?}  kf={:<5} mp={:<5} {frame_ms:>6.1} ms",
            result.status, kf, mp
        );

        #[cfg(feature = "viz")]
        {
            trajectory.push(trajectory_point_from_pose(&result.pose_world_to_cam));
            if let Some(ref rec) = rec {
                log_trajectory_to_rerun(rec, &trajectory);
                log_camera_to_rerun(rec, &result.pose_world_to_cam, &camera, image_size);
                log_map_points_to_rerun(rec, system.map_points());
            }
        }

        idx += 1;
        if max > 0 && idx >= max {
            break;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "[oakd_slam] processed {idx} frames in {:.2} s ({:.1} fps avg)",
        elapsed,
        idx as f64 / elapsed.max(1e-6)
    );
    Ok(())
}
