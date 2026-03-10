//! ORB-SLAM example with Rerun visualization.
//!
//! Runs an ORB-based SLAM pipeline on EuRoC MAV images, extracting ORB features
//! externally and feeding them to `process_frame`.
//!
//! ```text
//! cargo run --example orb_slam -- --data /path/to/euroc/V1_01_easy
//! ```

mod pipeline;
mod utils;
#[path = "../common/datasets/mod.rs"]
mod datasets;

use std::collections::BTreeMap;
use std::path::PathBuf;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_io::png::read_image_png_mono8;
use kornia_slam::odometry::bootstrap::BootstrapConfig;
use kornia_slam::odometry::estimation::map_projection::OdometryConfig;
use kornia_slam::{Frame, OdometryStatus};

use pipeline::ORBSLAMPipeline;

use datasets::EurocDataset;

use utils::{
    RunArtifactWriter, SummaryRecord, TumPose, default_run_dir, frame_record_from_tracking_result,
    log_ground_truth_to_rerun, log_keyframes_to_rerun, log_map_points_to_rerun,
    log_trajectory_to_rerun, pose_to_tum, write_tum_trajectory,
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

    /// output TUM trajectory file path
    #[argh(option)]
    output: Option<String>,

    /// output directory for machine-readable run artifacts
    #[argh(option)]
    run_dir: Option<String>,

    /// skip this many initial frames
    #[argh(option, default = "0")]
    start_frame: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // ── Dataset ────────────────────────────────────────────────────────────
    let dataset = EurocDataset::open(&args.data)?;
    let samples = dataset.samples();
    let ground_truth = dataset.ground_truth();

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
    // Temporary debugging artifacts for comparing this port against the original crate.
    let run_dir = args
        .run_dir
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| default_run_dir("mono_vo_run"));
    let run_name = run_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("mono_vo_run")
        .to_string();
    let mut run_writer = RunArtifactWriter::create(&run_dir)?;
    let dataset_line = format!(
        "Dataset: {} frames (processing {}..{})",
        samples.len(),
        args.start_frame,
        args.start_frame + n_frames
    );
    run_writer.write_stdout_line(&dataset_line)?;

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

    // ── ORB detector (used externally before feeding ORBSLAMPipeline) ─────
    let detector = kornia_imgproc::features::OrbDetector {
        n_keypoints: args.n_keypoints,
        ..Default::default()
    };

    // ── SLAM config & system ───────────────────────────────────────────────
    let bootstrap_config = BootstrapConfig::default();
    let odometry_config = OdometryConfig::default();
    let mut system = ORBSLAMPipeline::new(camera, bootstrap_config, odometry_config);

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
    let mut trajectory: Vec<TumPose> = Vec::with_capacity(n_frames);
    let mut keyframe_poses: Vec<(usize, kornia_3d::pose::Pose3d)> = Vec::new();
    let mut keyframe_tum: Vec<TumPose> = Vec::new();
    let mut status_counts: BTreeMap<String, usize> = BTreeMap::new();

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
            let inner = gray_u8.into_inner();
            let mut dst = kornia_image::Image::from_size_val(
                inner.size(),
                0.0f32,
                kornia_tensor::CpuAllocator,
            )
            .unwrap();
            inner
                .as_slice()
                .iter()
                .zip(dst.as_slice_mut())
                .for_each(|(&s, d)| *d = s as f32 / 255.0);
            dst
        };

        // Extract ORB features.
        let features = detector.detect_and_extract(&gray_f32)?;

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
        run_writer.write_stdout_line(&status_line)?;
        *status_counts
            .entry(format!("{:?}", result.status))
            .or_insert(0) += 1;

        // Collect trajectory.
        let tum_pose = pose_to_tum(sample.timestamp_sec, &result.pose_world_to_cam);
        trajectory.push(tum_pose);

        // Track keyframes for visualization.
        if result.status == OdometryStatus::KeyframeAccepted {
            keyframe_poses.push((keyframe_idx, result.pose_world_to_cam));
            keyframe_tum.push(tum_pose);
        }
        run_writer.write_frame(&frame_record_from_tracking_result(
            idx,
            sample.timestamp_sec,
            keyframe_idx,
            map_point_count,
            &result,
        ))?;

        // Rerun logging.
        if let Some(ref rec) = rec {
            log_trajectory_to_rerun(rec, &trajectory, &result.pose_world_to_cam);
            log_map_points_to_rerun(rec, system.map_points());
            log_keyframes_to_rerun(rec, &keyframe_poses);
            log_ground_truth_to_rerun(rec, ground_truth, sample.timestamp_sec);
        }
    }

    // ── Write output ───────────────────────────────────────────────────────
    if let Some(ref out) = args.output {
        write_tum_trajectory(&PathBuf::from(out), &trajectory)?;
        eprintln!("Wrote {}-pose trajectory to {out}", trajectory.len());
        run_writer.write_stdout_line(&format!(
            "Wrote {}-pose trajectory to {out}",
            trajectory.len()
        ))?;
    }

    let summary = SummaryRecord {
        dataset_path: args.data.clone(),
        run_name,
        frame_count: trajectory.len(),
        status_counts,
        final_map_points: system.map_points().len(),
    };
    run_writer.write_summary(&summary, &trajectory, &keyframe_tum)?;

    let final_line = format!("Done. Final map: {} points", system.map_points().len());
    eprintln!("{final_line}");
    run_writer.write_stdout_line(&final_line)?;
    let run_dir_line = format!("Wrote run artifacts to {}", run_dir.display());
    eprintln!("{run_dir_line}");
    run_writer.write_stdout_line(&run_dir_line)?;
    Ok(())
}
