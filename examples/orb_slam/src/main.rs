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
//!
//! Run live on a UVC camera (built-in webcam, USB cam, etc.; requires
//! `--features uvc`):
//! ```text
//! cargo run --release -p orb_slam --features uvc -- uvc \
//!     --fx 600 --fy 600 --cx 320 --cy 240
//! ```

mod config;
#[path = "../../common/datasets/mod.rs"]
mod datasets;
mod pipeline;
#[path = "../../common/source/mod.rs"]
mod source;
mod tui;
mod utils;

use config::PipelineConfig;
use kornia_3d::pose::Pose3d;
use kornia_slam::Frame;
use pipeline::Pipeline;
use source::{EurocSource, FrameItem, FrameSource};
#[cfg(feature = "oakd")]
use source::OakdSource;
#[cfg(feature = "uvc")]
use source::UvcSource;
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

    /// disable the terminal UI (status lines stream to stderr instead)
    #[argh(switch)]
    no_tui: bool,

    /// print per-frame diagnostics: bootstrap skip/reject reasons,
    /// map-projection reject reasons, keyframe growth and fuse counters
    #[argh(switch)]
    debug: bool,
}

#[derive(argh::FromArgs)]
#[argh(subcommand)]
enum SourceCmd {
    Euroc(EurocCmd),
    #[cfg(feature = "oakd")]
    Oakd(OakdCmd),
    #[cfg(feature = "uvc")]
    Uvc(UvcCmd),
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

/// Run live on a UVC camera (laptop webcam, USB cam, CSI-to-UVC adapter…),
/// using V4L2 / AVFoundation / MSMF via nokhwa.
///
/// Intrinsics flags must match the resolution the device actually streams at
/// — nokhwa may pick the closest supported mode if the exact one is missing.
#[cfg(feature = "uvc")]
#[derive(argh::FromArgs)]
#[argh(subcommand, name = "uvc")]
struct UvcCmd {
    /// camera device index (0 = first camera)
    #[argh(option, default = "0")]
    index: u32,

    /// frame width in pixels
    #[argh(option, default = "640")]
    width: u32,

    /// frame height in pixels
    #[argh(option, default = "480")]
    height: u32,

    /// maximum number of frames to process (0 = run forever, Ctrl-C to stop)
    #[argh(option, default = "0")]
    max_frames: usize,

    /// focal length x (pixels)
    #[argh(option)]
    fx: f64,

    /// focal length y (pixels)
    #[argh(option)]
    fy: f64,

    /// principal point x (pixels)
    #[argh(option)]
    cx: f64,

    /// principal point y (pixels)
    #[argh(option)]
    cy: f64,

    /// radial distortion k1
    #[argh(option, default = "0.0")]
    k1: f64,

    /// radial distortion k2
    #[argh(option, default = "0.0")]
    k2: f64,

    /// tangential distortion p1
    #[argh(option, default = "0.0")]
    p1: f64,

    /// tangential distortion p2
    #[argh(option, default = "0.0")]
    p2: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // TUI is the default; --rerun-stream or --no-tui falls back to plain stderr.
    #[cfg(feature = "viz")]
    let tui_active = !args.no_tui && !args.rerun_stream;
    #[cfg(not(feature = "viz"))]
    let tui_active = !args.no_tui;

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
        #[cfg(feature = "uvc")]
        SourceCmd::Uvc(w) => {
            let camera = kornia_3d::camera::PinholeCamera {
                fx: w.fx,
                fy: w.fy,
                cx: w.cx,
                cy: w.cy,
                k1: w.k1,
                k2: w.k2,
                p1: w.p1,
                p2: w.p2,
            };
            Box::new(UvcSource::open(
                w.index,
                w.width,
                w.height,
                camera,
                w.max_frames,
            )?)
        }
    };

    let camera = source.camera();
    let n_frames_hint = source.n_frames_hint();

    // ── ORB detector ───────────────────────────────────────────────────────
    let detector = kornia_imgproc::features::OrbDetector {
        n_keypoints: 1000,
        ..Default::default()
    };

    // ── SLAM system ────────────────────────────────────────────────────────
    let pipeline_config = PipelineConfig {
        debug: args.debug,
        ..PipelineConfig::default()
    };
    let mut system = Pipeline::new(camera.clone(), pipeline_config);

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
    let mut tui_state = if tui_active {
        let (term, guard) = tui::setup_terminal(std::path::Path::new("tui_stderr.log"))?;
        let mut app = tui::TuiApp::new(n_frames_hint.unwrap_or(0));
        app.debug_enabled = args.debug;
        Some((term, app, guard))
    } else {
        None
    };

    // ── Main loop ──────────────────────────────────────────────────────────
    let mut trajectory: Vec<[f32; 3]> = Vec::new();
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
        let debug_msgs = system.drain_debug_messages();
        processed += 1;

        // Status line.
        if !tui_active {
            for line in &debug_msgs {
                eprintln!("{line}");
            }
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
        if let Some((term, app, _guard)) = tui_state.as_mut() {
            for line in debug_msgs {
                app.push_debug_line(line);
            }
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
            match tui::poll_action()? {
                tui::TuiAction::Quit => break,
                tui::TuiAction::ToggleDebug => {
                    app.debug_enabled = !app.debug_enabled;
                    system.set_debug(app.debug_enabled);
                }
                tui::TuiAction::None => {}
            }
        }
    }

    // Restore terminal before printing the final summary.
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
