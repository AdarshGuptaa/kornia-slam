//! ORB-SLAM example with minimal Rerun visualization.
//!
//! Runs an ORB-based SLAM pipeline on EuRoC MAV images, extracting ORB features
//! externally and feeding them to `process_frame`.
//!
//! ```text
//! cargo run --example orb_slam -- --data /path/to/euroc/V1_01_easy
//! ```

#[path = "../common/datasets/mod.rs"]
mod datasets;
mod pipeline;
mod utils;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_io::png::read_image_png_mono8;
use kornia_slam::Frame;
use kornia_slam::estimation::map_projection::MapProjectionConfig;
use kornia_slam::estimation::two_view::TwoViewInitConfig;

use pipeline::Pipeline;

use datasets::EurocDataset;

use utils::{
    log_camera_to_rerun, log_frame_to_rerun, log_map_points_to_rerun, log_trajectory_to_rerun,
    trajectory_point_from_pose,
};

/// CLI arguments.
#[derive(argh::FromArgs)]
#[argh(description = "Monocular visual odometry on EuRoC dataset")]
struct Args {
    /// path to EuRoC dataset root (e.g. V1_01_easy/)
    #[argh(option)]
    data: String,

    /// maximum number of frames to process (0 = all)
    #[argh(option, default = "0")]
    max_frames: usize,

    /// number of ORB keypoints to detect per frame
    #[argh(option, default = "1000")]
    n_keypoints: usize,

    /// connect to a Rerun viewer via TCP (e.g. "127.0.0.1:9876")
    #[argh(option)]
    rerun_addr: Option<String>,

    /// spawn a Rerun viewer and stream to it
    #[argh(switch)]
    rerun_stream: bool,

    /// skip this many initial frames
    #[argh(option, default = "0")]
    start_frame: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // ── Dataset ────────────────────────────────────────────────────────────
    let dataset = EurocDataset::open(&args.data)?;
    let samples = dataset.samples();

    let n_frames = if args.max_frames > 0 {
        args.max_frames.min(samples.len() - args.start_frame)
    } else {
        samples.len() - args.start_frame
    };
    eprintln!(
        "Dataset: {} frames (processing {}..{})",
        samples.len(),
        args.start_frame,
        args.start_frame + n_frames
    );

    // ── Camera (EuRoC cam0 intrinsics) ─────────────────────────────────────
    let camera = PinholeCamera {
        fx: 458.654,
        fy: 457.296,
        cx: 367.215,
        cy: 248.375,
        k1: -0.28340811,
        k2: 0.07395907,
        p1: 0.00019359,
        p2: 1.76187114e-05,
    };

    // ── ORB detector (used externally before feeding Pipeline) ─────────────
    let detector = kornia_imgproc::features::OrbDetector {
        n_keypoints: args.n_keypoints,
        ..Default::default()
    };

    // ── SLAM config & system ───────────────────────────────────────────────
    let mut two_view_init_config = TwoViewInitConfig::default();
    two_view_init_config
        .estimation_config
        .triangulation
        .max_midpoint_gap = 0.25;
    two_view_init_config
        .estimation_config
        .triangulation
        .max_reprojection_error = 3.0;
    let map_projection_config = MapProjectionConfig::default();
    let mut system = Pipeline::new(camera, two_view_init_config, map_projection_config);

    // ── Rerun ──────────────────────────────────────────────────────────────
    let rec = if args.rerun_stream {
        let r = rerun::RecordingStreamBuilder::new("mono_vo").spawn()?;
        r.log("/", &rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN())?;
        r.log("world/camera", &rerun::ViewCoordinates::RDF())?;
        Some(r)
    } else if let Some(addr) = args.rerun_addr.as_deref() {
        // Connect to an already-running viewer/server over gRPC.
        // Accepts host:port (e.g. 127.0.0.1:9876).
        let r = rerun::RecordingStreamBuilder::new("mono_vo")
            .connect_grpc_opts(format!("rerun+http://{addr}/proxy"))?;
        r.log("/", &rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN())?;
        r.log("world/camera", &rerun::ViewCoordinates::RDF())?;
        Some(r)
    } else {
        None
    };

    // ── Main loop ──────────────────────────────────────────────────────────
    let mut trajectory: Vec<[f32; 3]> = Vec::with_capacity(n_frames);

    for (i, sample) in samples
        .iter()
        .skip(args.start_frame)
        .take(n_frames)
        .enumerate()
    {
        let idx = args.start_frame + i;

        // Load grayscale image and convert u8 → f32.
        let gray_u8 = read_image_png_mono8(&sample.image_path)?;
        let image_size = kornia_3d::camera::ImageSize {
            width: gray_u8.width() as f64,
            height: gray_u8.height() as f64,
        };
        let gray_f32 = {
            let mut dst = kornia_image::Image::from_size_val(
                gray_u8.size(),
                0.0f32,
                kornia_tensor::CpuAllocator,
            )
            .unwrap();
            gray_u8
                .as_slice()
                .iter()
                .zip(dst.as_slice_mut())
                .for_each(|(&s, d)| *d = s as f32 / 255.0);
            dst
        };

        // Extract ORB features.
        let features = detector.detect_and_extract(&gray_f32)?;
        if let Some(ref rec) = rec {
            log_frame_to_rerun(rec, &gray_u8, &features.keypoints_xy);
        }

        // Run SLAM.
        let frame = Frame::new(idx, features, Pose3d::IDENTITY, image_size);
        let result = system.process_frame(frame);
        let keyframe_idx = system.current_keyframe_idx().unwrap_or(idx);
        let map_point_count = system.num_map_points();

        // Status line.
        let status_line = format!(
            "[{idx:>5}] {:?}  kf={:<4} pts={:<5}",
            result.status, keyframe_idx, map_point_count,
        );
        eprintln!("{status_line}");

        // Collect trajectory.
        trajectory.push(trajectory_point_from_pose(&result.pose_world_to_cam));

        // Rerun logging.
        if let Some(ref rec) = rec {
            log_trajectory_to_rerun(rec, &trajectory);
            log_camera_to_rerun(rec, &result.pose_world_to_cam);
            log_map_points_to_rerun(rec, system.map_points());
        }
    }

    let final_line = format!("Done. Final map: {} points", system.map_points().len());
    eprintln!("{final_line}");
    Ok(())
}
