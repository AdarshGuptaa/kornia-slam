//! Monocular ORB-SLAM example with a selectable frame source.
//!
//! Run on an offline EuRoC dataset:
//! ```text
//! cargo run --release -p orb_slam -- euroc --data /path/to/V1_01_easy
//! ```
//!
//! Run live on an OAK-D camera (requires `--features oakd`):
//! ```text
//! cargo run --release -p orb_slam --features oakd -- oakd
//! ```

mod config;
#[path = "../../common/datasets/mod.rs"]
mod datasets;
mod pipeline;
#[path = "../../common/source/mod.rs"]
mod source;
#[cfg(feature = "tui")]
mod tui;
mod utils;

use config::PipelineConfig;
use kornia_3d::pose::Pose3d;
use kornia_slam::Frame;
use pipeline::Pipeline;
use source::{EurocSource, FrameItem, FrameSource};
#[cfg(feature = "oakd")]
use source::OakdSource;
use utils::trajectory_point_from_pose;

#[cfg(feature = "viz")]
use utils::{
    log_camera_to_rerun, log_frame_to_rerun, log_map_points_to_rerun, log_trajectory_to_rerun,
};

/// CLI arguments.
#[derive(argh::FromArgs)]
#[argh(description = "Monocular ORB-SLAM (EuRoC dataset or live OAK-D)")]
struct Args {
    #[argh(subcommand)]
    source: SourceCmd,

    /// spawn a Rerun viewer and stream to it (requires `--features viz`)
    #[argh(switch)]
    #[cfg(feature = "viz")]
    rerun_stream: bool,

    /// render a terminal UI with a live BEV view (requires `--features tui`)
    #[argh(switch)]
    #[cfg(feature = "tui")]
    tui: bool,
}

#[derive(argh::FromArgs)]
#[argh(subcommand)]
enum SourceCmd {
    Euroc(EurocCmd),
    #[cfg(feature = "oakd")]
    Oakd(OakdCmd),
}

/// Run on an EuRoC MAV dataset.
#[derive(argh::FromArgs)]
#[argh(subcommand, name = "euroc")]
struct EurocCmd {
    /// path to EuRoC dataset root (e.g. V1_01_easy/)
    #[argh(option)]
    data: String,

    /// maximum number of frames to process (0 = all)
    #[argh(option, default = "0")]
    max_frames: usize,

    /// skip this many initial frames
    #[argh(option, default = "0")]
    start_frame: usize,
}

/// Run live on an OAK-D camera (CamB mono).
#[cfg(feature = "oakd")]
#[derive(argh::FromArgs)]
#[argh(subcommand, name = "oakd")]
struct OakdCmd {
    /// maximum number of frames to process (0 = run forever, Ctrl-C to stop)
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    #[cfg(all(feature = "tui", feature = "viz"))]
    if args.tui && args.rerun_stream {
        eprintln!("--tui and --rerun-stream are mutually exclusive");
        std::process::exit(2);
    }

    #[cfg(feature = "tui")]
    let tui_active = args.tui;
    #[cfg(not(feature = "tui"))]
    let tui_active = false;

    // ── Source ─────────────────────────────────────────────────────────────
    let mut source: Box<dyn FrameSource> = match args.source {
        SourceCmd::Euroc(e) => {
            let src = EurocSource::open(&e.data, e.start_frame, e.max_frames)?;
            if !tui_active {
                let total = src.dataset_len();
                let n = src.n_frames_hint().unwrap_or(0);
                eprintln!(
                    "Dataset: {total} frames (processing {}..{})",
                    e.start_frame,
                    e.start_frame + n,
                );
            }
            Box::new(src)
        }
        #[cfg(feature = "oakd")]
        SourceCmd::Oakd(o) => Box::new(OakdSource::open(o.width, o.height, o.fps, o.max_frames)?),
    };

    let camera = source.camera();
    #[cfg(feature = "tui")]
    let n_frames_hint = source.n_frames_hint();

    // ── ORB detector ───────────────────────────────────────────────────────
    let detector = kornia_imgproc::features::OrbDetector {
        n_keypoints: 1000,
        ..Default::default()
    };

    // ── SLAM system ────────────────────────────────────────────────────────
    let mut system = Pipeline::new(camera.clone(), PipelineConfig::default());

    // ── Rerun ──────────────────────────────────────────────────────────────
    #[cfg(feature = "viz")]
    let rec = if args.rerun_stream {
        let r = rerun::RecordingStreamBuilder::new("orb_slam").spawn()?;
        r.log("/", &rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN())?;
        r.log("world/camera", &rerun::ViewCoordinates::RDF())?;
        Some(r)
    } else {
        None
    };

    // ── TUI ────────────────────────────────────────────────────────────────
    #[cfg(feature = "tui")]
    let mut tui_state = if args.tui {
        let (term, guard) = tui::setup_terminal(std::path::Path::new("tui_stderr.log"))?;
        let app = tui::TuiApp::new(n_frames_hint.unwrap_or(0));
        Some((term, app, guard))
    } else {
        None
    };

    // ── Main loop ──────────────────────────────────────────────────────────
    let mut trajectory: Vec<[f32; 3]> = Vec::new();
    #[cfg(feature = "tui")]
    let mut processed: usize = 0;

    while let Some(item) = source.next_frame()? {
        let FrameItem {
            idx,
            timestamp_sec: _,
            image: gray_u8,
        } = item;
        let image_size = gray_u8.size();

        // Extract ORB features.
        let features = detector.detect_and_extract_u8(&gray_u8)?;
        #[cfg(feature = "viz")]
        if let Some(ref rec) = rec {
            log_frame_to_rerun(rec, &gray_u8, &features.keypoints_xy);
        }

        // Sample pixel colors at each keypoint location.
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

        // Run SLAM.
        let frame = Frame {
            idx,
            features,
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size,
            keypoint_colors,
        };
        let t0 = std::time::Instant::now();
        let result = system.process_frame(frame);
        let frame_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let keyframe_idx = system.current_keyframe_idx().unwrap_or(idx);
        let map_point_count = system.num_active_map_points();
        #[cfg(feature = "tui")]
        {
            processed += 1;
        }

        // Status line.
        if !tui_active {
            let status_line = format!(
                "[{idx:>5}] {:?}  kf={:<4} pts={:<5} {frame_ms:>6.1}ms",
                result.status, keyframe_idx, map_point_count,
            );
            eprintln!("{status_line}");
        }

        // Trajectory.
        trajectory.push(trajectory_point_from_pose(&result.pose_world_to_cam));

        // Rerun logging.
        #[cfg(feature = "viz")]
        if let Some(ref rec) = rec {
            log_trajectory_to_rerun(rec, &trajectory);
            log_camera_to_rerun(rec, &result.pose_world_to_cam, &camera, image_size);
            log_map_points_to_rerun(rec, system.map_points());
        }

        // TUI render.
        #[cfg(feature = "tui")]
        if let Some((term, app, _guard)) = tui_state.as_mut() {
            app.frame_idx = idx;
            app.n_frames = n_frames_hint.unwrap_or(processed);
            app.frame_ms = frame_ms;
            app.status = match result.status {
                kornia_slam::TrackingStatus::Tracked => tui::TuiStatus::Tracked,
                kornia_slam::TrackingStatus::KeyframeAccepted => tui::TuiStatus::KeyframeAccepted,
                kornia_slam::TrackingStatus::Skipped => tui::TuiStatus::Skipped,
            };
            app.kf_idx = keyframe_idx;
            app.n_active_mp = map_point_count;
            let n_so_far = processed as f64;
            app.mean_ms = app.mean_ms + (frame_ms - app.mean_ms) / n_so_far;
            app.update_pose(&result.pose_world_to_cam);
            app.draw(term)?;
            if tui::poll_quit()? {
                break;
            }
        }
    }

    // Restore terminal before printing the final summary.
    #[cfg(feature = "tui")]
    if let Some((mut term, _, _guard)) = tui_state.take() {
        tui::restore_terminal(&mut term)?;
    }

    let total_pts = system.map_points().len();
    let active_pts = system.map_points().iter().filter(|mp| !mp.culled).count();
    let mut obs_total: usize = 0;
    let mut obs_max: usize = 0;
    for mp in system.map_points().iter().filter(|mp| !mp.culled) {
        let n = mp.observation_kf_indices.len();
        obs_total += n;
        if n > obs_max {
            obs_max = n;
        }
    }
    let obs_mean = if active_pts > 0 {
        obs_total as f64 / active_pts as f64
    } else {
        0.0
    };
    eprintln!(
        "Done. Final map: total={total_pts}  active={active_pts}  obs_per_active_mp={obs_mean:.2}  max_obs={obs_max}"
    );
    Ok(())
}
