//! `kornia-slam` command-line application: runs the SLAM runtime over the
//! source a run file describes.
//!
//! ```text
//! cargo run --release -p kornia-slam-app -- --config configs/euroc.ron
//! ```
//!
//! Anchor the final map to a physical AprilTag (requires `--features apriltag`):
//!
//! ```text
//! cargo run --release -p kornia-slam-app --features apriltag -- \
//!     --config configs/apriltag-video.ron --apriltag-anchor \
//!     --apriltag-id 2 --apriltag-size 0.1 --apriltag-family tag36h11
//! ```
//!
//! A run file names the source (EuRoC, Hilti, MCAP, or a live OAK-D or UVC
//! camera) and the system to run on it; see `configs/` and [`config`].

mod config;
mod datasets;
mod evaluation;
mod source;
mod tui;
mod utils;
use config::RunConfig;
use evaluation::associate_gt;
use kornia_algebra::Vec3F64;
#[cfg(feature = "apriltag")]
use kornia_slam::apriltag_anchor::AprilTagAnchorConfig;
use kornia_slam::system::PipelineDefinition;
use kornia_slam::{LoopClosureEvent, SensorFrame, SlamSystem};
use source::{FrameItem, OpenedSource};
use std::time::{Duration, Instant};
use utils::trajectory_point_from_pose;

#[cfg(feature = "viz")]
use utils::{
    log_camera_to_rerun, log_frame_to_rerun, log_map_points_to_rerun, log_trajectory_to_rerun,
};

#[derive(argh::FromArgs)]
#[argh(description = "Visual and visual-inertial ORB-SLAM over the source a run file describes")]
struct Args {
    /// run file (RON): the source to read and the system to run on it; see configs/
    #[argh(option)]
    config: String,

    /// after the run, align the trajectory to ground truth and report
    /// ATE/RPE/drift (writes kornia_slam_raw.csv and kornia_slam_aligned.csv);
    /// needs a source with ground truth (EuRoC, Hilti)
    #[argh(switch)]
    evaluate: bool,

    /// directory for the evaluation CSVs (created if missing; default: current dir)
    #[argh(option, default = "String::from(\".\")")]
    eval_out: String,

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

    /// after the run, anchor map scale/orientation to a physical AprilTag
    /// solved once as a Sim3 (requires building with --features apriltag)
    #[argh(switch)]
    apriltag_anchor: bool,

    /// apriltag side length in metres (with --apriltag-anchor)
    #[argh(option, default = "0.1")]
    apriltag_size: f64,

    /// apriltag id to anchor to (with --apriltag-anchor)
    #[argh(option, default = "0")]
    apriltag_id: u16,

    /// tag family: tag16h5 | tag25h9 | tag36h10 | tag36h11 | tagcircle21h7 |
    /// tagcircle49h12 | tagcustom48h12 | tagstandard41h12 | tagstandard52h13
    /// (with --apriltag-anchor)
    #[argh(option, default = "String::from(\"tag36h11\")")]
    apriltag_family: String,
}

/// Maps a CLI family name (case-insensitive) to a [`TagFamilyKind`].
#[cfg(feature = "apriltag")]
fn parse_tag_family(name: &str) -> Option<kornia_slam::apriltag_anchor::TagFamilyKind> {
    use kornia_slam::apriltag_anchor::TagFamilyKind::{
        Tag16H5, Tag25H9, Tag36H10, Tag36H11, TagCircle21H7, TagCircle49H12, TagCustom48H12,
        TagStandard41H12, TagStandard52H13,
    };
    match name.to_ascii_lowercase().as_str() {
        "tag16h5" => Some(Tag16H5),
        "tag25h9" => Some(Tag25H9),
        "tag36h10" => Some(Tag36H10),
        "tag36h11" => Some(Tag36H11),
        "tagcircle21h7" => Some(TagCircle21H7),
        "tagcircle49h12" => Some(TagCircle49H12),
        "tagcustom48h12" => Some(TagCustom48H12),
        "tagstandard41h12" => Some(TagStandard41H12),
        "tagstandard52h13" => Some(TagStandard52H13),
        _ => None,
    }
}

/// Rejects the AprilTag anchor flags when their inputs are invalid (or, in a
/// build without the `apriltag` feature, whenever the flag is used at all).
#[cfg(feature = "apriltag")]
fn validate_apriltag_mode(
    apriltag_anchor: bool,
    _tag_id: u16,
    family: &str,
    tag_size_m: f64,
) -> Result<(), &'static str> {
    if !apriltag_anchor {
        return Ok(());
    }
    if parse_tag_family(family).is_none() {
        return Err(
            "--apriltag-anchor: unknown --apriltag-family (supported: tag16h5, tag25h9, tag36h10, tag36h11, tagcircle21h7, tagcircle49h12, tagcustom48h12, tagstandard41h12, tagstandard52h13)",
        );
    }
    if tag_size_m <= 0.0 {
        return Err("--apriltag-anchor: --apriltag-size must be positive (metres)");
    }
    Ok(())
}

#[cfg(not(feature = "apriltag"))]
fn validate_apriltag_mode(
    apriltag_anchor: bool,
    _tag_id: u16,
    _family: &str,
    _tag_size_m: f64,
) -> Result<(), &'static str> {
    if apriltag_anchor {
        return Err("--apriltag-anchor requires building with --features apriltag");
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    #[cfg(feature = "viz")]
    let tui_active = !args.no_tui && !args.rerun_stream;
    #[cfg(not(feature = "viz"))]
    let tui_active = !args.no_tui;

    let run = RunConfig::from_ron_file(&args.config)?;
    if let Err(error) = validate_apriltag_mode(
        args.apriltag_anchor,
        args.apriltag_id,
        &args.apriltag_family,
        args.apriltag_size,
    ) {
        return Err(error.into());
    }
    #[cfg(feature = "apriltag")]
    let run = {
        let mut run = run;
        if args.apriltag_anchor {
            let PipelineDefinition::OrbSlam(orb) = &mut run.system.pipeline;
            if let Some(family) = parse_tag_family(&args.apriltag_family) {
                orb.apriltag = Some(AprilTagAnchorConfig {
                    tag_id: args.apriltag_id,
                    tag_size_m: args.apriltag_size,
                    family,
                    ..AprilTagAnchorConfig::default()
                });
            }
        }
        run
    };
    if !tui_active {
        eprint!("{}", run.system);
    }

    let OpenedSource {
        mut source,
        ground_truth,
        summary,
    } = run.source.open(run.system.sensors)?;
    if args.evaluate && ground_truth.is_none() {
        return Err("--evaluate needs ground truth, and this source has none".into());
    }
    if !tui_active && let Some(summary) = summary {
        eprintln!("{summary}");
    }
    let target_dt = Duration::from_secs_f64(1.0 / 30.0);
    let mut last_frame_walltime = Instant::now();

    let n_frames_hint = source.n_frames_hint();
    let PipelineDefinition::OrbSlam(orb) = &run.system.pipeline;
    let vocabulary = orb
        .loop_closing
        .vocabulary()
        .map(std::path::Path::to_path_buf);
    let mut system = SlamSystem::build(run.system, source.rig()).map_err(|e| e.to_string())?;
    system.set_debug(args.debug);
    let camera = system.rig().camera.clone();
    if let Some(path) = vocabulary {
        eprintln!(
            "[place-recognition] loaded vocabulary from {}",
            path.display()
        );
    }
    if !tui_active && let Some(baseline) = system.rig().stereo_baseline_m {
        eprintln!(
            "Stereo: rectified fx={:.2} baseline={baseline:.4}m bf={:.2}",
            camera.fx,
            baseline * camera.fx,
        );
    }

    #[cfg(feature = "viz")]
    let rec = if args.rerun_stream {
        let r = rerun::RecordingStreamBuilder::new("kornia-slam").spawn()?;
        r.log("/", &rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN())?;
        r.log("world/camera", &rerun::ViewCoordinates::RDF())?;
        Some(r)
    } else {
        None
    };

    let mut tui_state = if tui_active {
        let (term, guard) = tui::setup_terminal(std::path::Path::new("tui_stderr.log"))?;
        let mut app = tui::TuiApp::new(n_frames_hint.unwrap_or(0));
        app.debug_enabled = args.debug;
        Some((term, app, guard))
    } else {
        None
    };
    let (mut est_positions, mut gt_positions): (Vec<Vec3F64>, Vec<Vec3F64>) =
        (Vec::new(), Vec::new());
    let mut trajectory: Vec<[f32; 3]> = Vec::new();
    let mut processed: usize = 0;

    while let Some(item) = source.next_frame()? {
        let now = Instant::now();
        let elapsed = now.duration_since(last_frame_walltime);

        if elapsed < target_dt {
            std::thread::sleep(target_dt - elapsed);
        }

        last_frame_walltime = Instant::now();

        let FrameItem {
            idx,
            timestamp_sec,
            image,
            right_image,
            imu_samples,
        } = item;
        #[cfg(feature = "viz")]
        if let Some(ref rec) = rec {
            rec.set_time_sequence("frame", idx as i64);
            rec.set_duration_secs("timestamp", timestamp_sec);
        }

        let result = system.process(SensorFrame {
            idx,
            timestamp_sec,
            image: &image,
            right_image: right_image.as_ref(),
            imu_samples: &imu_samples,
        })?;
        let frame_ms = system.tracking_duration().as_secs_f64() * 1000.0;
        let frontend_ms = system.frontend_observation().duration.as_secs_f64() * 1000.0;
        #[cfg(feature = "viz")]
        if let Some(ref rec) = rec {
            log_frame_to_rerun(rec, &image, &system.frontend_observation().keypoints_xy);
        }
        let keyframe_idx = system.current_keyframe_idx().unwrap_or(idx);
        let map_point_count = system.num_active_map_points();
        let debug_msgs = system.drain_debug_messages();
        processed += 1;

        for event in system.drain_loop_closure_events() {
            match event {
                LoopClosureEvent::Accepted { edge, applied } => eprintln!(
                    "[loop-closure] accepted kf={} ~ kf={} inliers={} rmse={:.2}px applied={applied}",
                    edge.query_kf_idx,
                    edge.candidate_kf_idx,
                    edge.inliers,
                    edge.reprojection_rmse_px,
                ),
                LoopClosureEvent::PgoFailed {
                    query_kf_idx,
                    candidate_kf_idx,
                    reason,
                } => eprintln!("[pgo] failed kf={query_kf_idx} ~ kf={candidate_kf_idx}: {reason}"),
            }
        }

        if !tui_active {
            for line in &debug_msgs {
                eprintln!("{line}");
            }
            let status_line = format!(
                "[{idx:>5}] {:?}  kf={:<4} pts={:<5} {frame_ms:>6.1}ms fe={frontend_ms:.1}ms",
                result.status, keyframe_idx, map_point_count,
            );
            eprintln!("{status_line}");
        }

        let traj_pt = trajectory_point_from_pose(&result.pose_world_to_cam);
        trajectory.push(traj_pt);

        // Evaluation pairs each estimate with the nearest ground-truth pose.
        if args.evaluate
            && let Some(gt) = ground_truth
                .as_deref()
                .and_then(|gt| associate_gt(timestamp_sec, gt))
        {
            est_positions.push(Vec3F64::new(
                traj_pt[0] as f64,
                traj_pt[1] as f64,
                traj_pt[2] as f64,
            ));
            gt_positions.push(Vec3F64::new(gt.tx, gt.ty, gt.tz));
        }
        #[cfg(feature = "viz")]
        if let Some(ref rec) = rec {
            log_trajectory_to_rerun(rec, &trajectory);
            log_camera_to_rerun(rec, &result.pose_world_to_cam, &camera, image.size());
            system.with_map_points(|map_points| log_map_points_to_rerun(rec, map_points));
        }

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

    let (total_pts, active_pts, obs_total, obs_max) = system.with_map_points(|map_points| {
        let mut active_pts: usize = 0;
        let mut obs_total: usize = 0;
        let mut obs_max: usize = 0;
        for mp in map_points.iter().filter(|mp| !mp.culled) {
            let n = mp.observations().len();
            active_pts += 1;
            obs_total += n;
            if n > obs_max {
                obs_max = n;
            }
        }
        (map_points.len(), active_pts, obs_total, obs_max)
    });
    let obs_mean = if active_pts > 0 {
        obs_total as f64 / active_pts as f64
    } else {
        0.0
    };
    eprintln!(
        "Done. Final map: total={total_pts}  active={active_pts}  obs_per_active_mp={obs_mean:.2}  max_obs={obs_max}"
    );

    // Metric anchor: solve and apply once, then report. On IMU-initialized
    // maps the correction is gravity-preserving (scale stays 1); otherwise it
    // is the full Sim3 the tag geometry defines.
    #[cfg(feature = "apriltag")]
    if args.apriltag_anchor {
        match system.apply_apriltag_anchor() {
            Ok(a) => eprintln!(
                "[apriltag-anchor] scale={:.6} translation=({:.4}, {:.4}, {:.4})",
                a.scale, a.translation.x, a.translation.y, a.translation.z
            ),
            Err(e) => eprintln!("[apriltag-anchor] failed: {e}"),
        }
    }

    // Median/max distance of active map points from the map origin, in map
    // units: a stable telemetry signal for auditing the anchor's metric scale.
    let (median_depth, max_depth) = system.with_map_points(|map_points| {
        let mut depths: Vec<f64> = map_points
            .iter()
            .filter(|mp| !mp.culled)
            .map(|mp| mp.position.length())
            .collect();
        if depths.is_empty() {
            return (0.0, 0.0);
        }
        depths.sort_by(|a, b| a.total_cmp(b));
        let median = depths[depths.len() / 2];
        (median, *depths.last().unwrap_or(&0.0))
    });
    eprintln!("Map depth (map units): median={median_depth:.3}  max={max_depth:.3}");

    if args.evaluate {
        evaluation::report(
            &est_positions,
            &gt_positions,
            std::path::Path::new(&args.eval_out),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod cli_tests {
    use argh::FromArgs;

    use super::*;

    fn parse(args: &[&str]) -> Result<Args, argh::EarlyExit> {
        Args::from_args(&["kornia-slam"], args)
    }

    #[test]
    fn parses_the_run_file_and_evaluation_options() {
        assert!(parse(&[]).is_err(), "a run file is required");

        let args = parse(&["--config", "configs/euroc.ron"]).unwrap();
        assert_eq!(args.config, "configs/euroc.ron");
        assert!(!args.evaluate);
        assert_eq!(args.eval_out, ".");

        let args = parse(&["--config", "r.ron", "--evaluate", "--eval-out", "out"]).unwrap();
        assert!(args.evaluate);
        assert_eq!(args.eval_out, "out");
    }
}

#[cfg(all(test, feature = "apriltag"))]
mod apriltag_mode_tests {
    use super::{parse_tag_family, validate_apriltag_mode};

    #[test]
    fn parses_builtin_families_and_rejects_unknown() {
        for (name, valid) in [
            ("tag16h5", true),
            ("tag25h9", true),
            ("tag36h10", true),
            ("tag36h11", true),
            ("tagcircle21h7", true),
            ("tagcircle49h12", true),
            ("tagcustom48h12", true),
            ("tagstandard41h12", true),
            ("tagstandard52h13", true),
            ("TAG25H9", true), // case-insensitive
            ("tag36h12", false),
            ("", false),
        ] {
            assert_eq!(parse_tag_family(name).is_some(), valid, "family {name}");
        }
    }

    #[test]
    fn validates_family_and_positive_size() {
        assert!(validate_apriltag_mode(true, 2, "tag36h11", 0.1).is_ok());
        assert!(validate_apriltag_mode(true, 2, "nope", 0.1).is_err());
        assert!(validate_apriltag_mode(true, 2, "tag36h11", 0.0).is_err());
        assert!(validate_apriltag_mode(false, 2, "nope", 0.0).is_ok());
    }
}

#[cfg(all(test, not(feature = "apriltag")))]
mod apriltag_mode_tests {
    use super::validate_apriltag_mode;

    #[test]
    fn flag_requires_building_with_feature() {
        assert!(validate_apriltag_mode(true, 2, "tag36h11", 0.1).is_err());
        assert!(validate_apriltag_mode(false, 2, "tag36h11", 0.1).is_ok());
    }
}
