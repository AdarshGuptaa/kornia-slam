//! Rerun visualization helpers and TUM trajectory I/O.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use kornia_3d::pose::Pose3d;
use kornia_algebra::Mat3F64;
use kornia_slam::mapping::MapPoint;
use serde::{Deserialize, Serialize};

const EUROC_FX: f32 = 458.654;
const EUROC_FY: f32 = 457.296;
const EUROC_CX: f32 = 367.215;
const EUROC_CY: f32 = 248.375;
const EUROC_IMAGE_WIDTH: f32 = 752.0;
const EUROC_IMAGE_HEIGHT: f32 = 480.0;
const CAMERA_IMAGE_PLANE_DISTANCE: f32 = 0.15;

#[derive(Debug, Clone, Copy)]
struct CameraVisualizationSpec {
    translation: [f32; 3],
    rotation_wxyz: [f32; 4],
    focal_length: [f32; 2],
    resolution: [f32; 2],
    principal_point: [f32; 2],
    image_plane_distance: f32,
}

// ── TUM trajectory format ──────────────────────────────────────────────────

/// One row of a TUM-format trajectory file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TumPose {
    pub timestamp: f64,
    pub tx: f64,
    pub ty: f64,
    pub tz: f64,
    pub qx: f64,
    pub qy: f64,
    pub qz: f64,
    pub qw: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PoseRecord {
    pub translation: [f64; 3],
    pub quaternion_xyzw: [f64; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FrameRecord {
    pub frame_idx: usize,
    pub timestamp_sec: f64,
    pub status: String,
    pub keyframe_idx: usize,
    pub world_to_cam: PoseRecord,
    pub cam_to_world: PoseRecord,
    pub map_point_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SummaryRecord {
    pub dataset_path: String,
    pub run_name: String,
    pub frame_count: usize,
    pub status_counts: BTreeMap<String, usize>,
    pub final_map_points: usize,
}

pub struct RunArtifactWriter {
    run_dir: PathBuf,
    frames: BufWriter<File>,
    stdout: BufWriter<File>,
}

impl RunArtifactWriter {
    pub fn create(run_dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let run_dir = run_dir.as_ref().to_path_buf();
        fs::create_dir_all(&run_dir)?;
        let frames = BufWriter::new(File::create(run_dir.join("frames.jsonl"))?);
        let stdout = BufWriter::new(File::create(run_dir.join("stdout.log"))?);
        Ok(Self {
            run_dir,
            frames,
            stdout,
        })
    }

    pub fn write_stdout_line(&mut self, line: &str) -> std::io::Result<()> {
        writeln!(self.stdout, "{line}")?;
        self.stdout.flush()
    }

    pub fn write_frame(&mut self, record: &FrameRecord) -> std::io::Result<()> {
        serde_json::to_writer(&mut self.frames, record)?;
        writeln!(self.frames)?;
        self.frames.flush()
    }

    pub fn write_summary(
        &mut self,
        summary: &SummaryRecord,
        trajectory: &[TumPose],
        keyframes: &[TumPose],
    ) -> std::io::Result<()> {
        self.frames.flush()?;
        self.stdout.flush()?;

        let summary_file = File::create(self.run_dir.join("summary.json"))?;
        let mut summary_writer = BufWriter::new(summary_file);
        serde_json::to_writer_pretty(&mut summary_writer, summary)?;
        writeln!(summary_writer)?;
        summary_writer.flush()?;

        write_tum_trajectory(&self.run_dir.join("trajectory.tum"), trajectory)?;
        write_tum_trajectory(&self.run_dir.join("keyframes.tum"), keyframes)?;
        Ok(())
    }
}

pub fn default_run_dir(prefix: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    PathBuf::from(format!("{prefix}-{}", now.as_secs()))
}

pub fn pose_record_from_pose(pose: &Pose3d) -> PoseRecord {
    let t = pose.translation;
    let (qx, qy, qz, qw) = quat_xyzw_from_matrix(&pose.rotation);
    PoseRecord {
        translation: [t.x, t.y, t.z],
        quaternion_xyzw: [qx, qy, qz, qw],
    }
}

pub fn frame_record_from_tracking_result(
    frame_idx: usize,
    timestamp_sec: f64,
    keyframe_idx: usize,
    map_point_count: usize,
    result: &kornia_slam::OdometryResult,
) -> FrameRecord {
    let cam_to_world = result.pose_world_to_cam.inverse();
    FrameRecord {
        frame_idx,
        timestamp_sec,
        status: format!("{:?}", result.status),
        keyframe_idx,
        world_to_cam: pose_record_from_pose(&result.pose_world_to_cam),
        cam_to_world: pose_record_from_pose(&cam_to_world),
        map_point_count,
    }
}

/// Convert a world-to-camera pose into a TUM row (camera-to-world).
pub fn pose_to_tum(timestamp: f64, pose_world_to_cam: &Pose3d) -> TumPose {
    let cam_to_world = pose_world_to_cam.inverse();
    let t = cam_to_world.translation;
    let (qx, qy, qz, qw) = quat_xyzw_from_matrix(&cam_to_world.rotation);
    TumPose {
        timestamp,
        tx: t.x,
        ty: t.y,
        tz: t.z,
        qx,
        qy,
        qz,
        qw,
    }
}

/// Write a sequence of TUM poses to a file.
pub fn write_tum_trajectory(path: &Path, poses: &[TumPose]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    for p in poses {
        writeln!(
            f,
            "{:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            p.timestamp, p.tx, p.ty, p.tz, p.qx, p.qy, p.qz, p.qw
        )?;
    }
    Ok(())
}

/// Extract quaternion (x, y, z, w) from a 3x3 rotation matrix.
///
/// Uses Shepperd's method for numerical stability.
fn quat_xyzw_from_matrix(r: &Mat3F64) -> (f64, f64, f64, f64) {
    let trace = r.x_axis.x + r.y_axis.y + r.z_axis.z;

    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        let w = 0.25 * s;
        let x = (r.y_axis.z - r.z_axis.y) / s;
        let y = (r.z_axis.x - r.x_axis.z) / s;
        let z = (r.x_axis.y - r.y_axis.x) / s;
        (x, y, z, w)
    } else if r.x_axis.x > r.y_axis.y && r.x_axis.x > r.z_axis.z {
        let s = (1.0 + r.x_axis.x - r.y_axis.y - r.z_axis.z).sqrt() * 2.0;
        let w = (r.y_axis.z - r.z_axis.y) / s;
        let x = 0.25 * s;
        let y = (r.x_axis.y + r.y_axis.x) / s;
        let z = (r.z_axis.x + r.x_axis.z) / s;
        (x, y, z, w)
    } else if r.y_axis.y > r.z_axis.z {
        let s = (1.0 + r.y_axis.y - r.x_axis.x - r.z_axis.z).sqrt() * 2.0;
        let w = (r.z_axis.x - r.x_axis.z) / s;
        let x = (r.x_axis.y + r.y_axis.x) / s;
        let y = 0.25 * s;
        let z = (r.y_axis.z + r.z_axis.y) / s;
        (x, y, z, w)
    } else {
        let s = (1.0 + r.z_axis.z - r.x_axis.x - r.y_axis.y).sqrt() * 2.0;
        let w = (r.x_axis.y - r.y_axis.x) / s;
        let x = (r.z_axis.x + r.x_axis.z) / s;
        let y = (r.y_axis.z + r.z_axis.y) / s;
        let z = 0.25 * s;
        (x, y, z, w)
    }
}

// ── Rerun visualization ────────────────────────────────────────────────────

/// Log the estimated trajectory and current camera frustum.
pub fn log_trajectory_to_rerun(
    rec: &rerun::RecordingStream,
    poses: &[TumPose],
    pose_world_to_cam: &Pose3d,
) {
    // Trajectory as a 3D line strip.
    if poses.len() >= 2 {
        let points: Vec<[f32; 3]> = poses
            .iter()
            .map(|p| [p.tx as f32, p.ty as f32, p.tz as f32])
            .collect();
        rec.log("world/trajectory", &rerun::LineStrips3D::new([points]))
            .ok();
    }

    // Camera frustum at current pose.
    log_camera_frustum(rec, "world/camera", pose_world_to_cam);
}

/// Log map points as a 3D point cloud.
pub fn log_map_points_to_rerun(rec: &rerun::RecordingStream, map_points: &[MapPoint]) {
    let positions: Vec<rerun::Position3D> = map_points
        .iter()
        .filter(|mp| !mp.culled)
        .map(|mp| {
            rerun::Position3D::new(
                mp.position.x as f32,
                mp.position.y as f32,
                mp.position.z as f32,
            )
        })
        .collect();
    if !positions.is_empty() {
        rec.log(
            "world/map_points",
            &rerun::Points3D::new(positions).with_radii([rerun::Radius::new_scene_units(0.01)]),
        )
        .ok();
    }
}

/// Log keyframe positions as labeled 3D points.
pub fn log_keyframes_to_rerun(rec: &rerun::RecordingStream, keyframe_poses: &[(usize, Pose3d)]) {
    let positions: Vec<rerun::Position3D> = keyframe_poses
        .iter()
        .map(|(_, pose)| {
            let c2w = pose.inverse();
            rerun::Position3D::new(
                c2w.translation.x as f32,
                c2w.translation.y as f32,
                c2w.translation.z as f32,
            )
        })
        .collect();
    if !positions.is_empty() {
        rec.log(
            "world/keyframes",
            &rerun::Points3D::new(positions).with_radii([rerun::Radius::new_scene_units(0.03)]),
        )
        .ok();
    }
}

/// Log ground-truth trajectory and nearest GT pose.
pub fn log_ground_truth_to_rerun(
    rec: &rerun::RecordingStream,
    gt: &[crate::datasets::GroundTruthPose],
    current_timestamp: f64,
) {
    if gt.is_empty() {
        return;
    }

    // Full GT path.
    let gt_points: Vec<[f32; 3]> = gt
        .iter()
        .map(|g| [g.tx as f32, g.ty as f32, g.tz as f32])
        .collect();
    rec.log(
        "world/ground_truth/path",
        &rerun::LineStrips3D::new([gt_points]),
    )
    .ok();

    // Nearest GT pose to current timestamp.
    if let Some(nearest) = gt.iter().min_by(|a, b| {
        let da = (a.timestamp_sec - current_timestamp).abs();
        let db = (b.timestamp_sec - current_timestamp).abs();
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    }) {
        rec.log(
            "world/ground_truth/current",
            &rerun::Points3D::new([rerun::Position3D::new(
                nearest.tx as f32,
                nearest.ty as f32,
                nearest.tz as f32,
            )])
            .with_radii([rerun::Radius::new_scene_units(0.04)]),
        )
        .ok();
    }
}

/// Log a camera frustum at the given world-to-camera pose.
fn log_camera_frustum(rec: &rerun::RecordingStream, entity: &str, pose_world_to_cam: &Pose3d) {
    let spec = camera_visualization_spec(pose_world_to_cam);

    rec.log(
        entity,
        &rerun::Transform3D::from_translation_rotation(
            spec.translation,
            rerun::Quaternion::from_wxyz(spec.rotation_wxyz),
        ),
    )
    .ok();
    rec.log(
        format!("{entity}/pinhole"),
        &rerun::Pinhole::from_focal_length_and_resolution(spec.focal_length, spec.resolution)
            .with_principal_point(spec.principal_point)
            .with_image_plane_distance(spec.image_plane_distance),
    )
    .ok();
}

fn camera_visualization_spec(pose_world_to_cam: &Pose3d) -> CameraVisualizationSpec {
    let cam_to_world = pose_world_to_cam.inverse();
    let t = cam_to_world.translation;
    let (qx, qy, qz, qw) = quat_xyzw_from_matrix(&cam_to_world.rotation);

    CameraVisualizationSpec {
        translation: [t.x as f32, t.y as f32, t.z as f32],
        rotation_wxyz: [qw as f32, qx as f32, qy as f32, qz as f32],
        focal_length: [EUROC_FX, EUROC_FY],
        resolution: [EUROC_IMAGE_WIDTH, EUROC_IMAGE_HEIGHT],
        principal_point: [EUROC_CX, EUROC_CY],
        image_plane_distance: CAMERA_IMAGE_PLANE_DISTANCE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::fs;

    use kornia_algebra::{Mat3F64, Vec3F64};
    use kornia_slam::{OdometryResult, OdometryStatus};
    use tempfile::tempdir;

    #[test]
    fn camera_visualization_spec_uses_camera_to_world_pose() {
        let pose_world_to_cam = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(1.0, -2.0, 3.0));

        let spec = camera_visualization_spec(&pose_world_to_cam);

        assert_eq!(spec.translation, [-1.0, 2.0, -3.0]);
        assert_eq!(spec.rotation_wxyz, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn camera_visualization_spec_matches_euroc_intrinsics() {
        let spec = camera_visualization_spec(&Pose3d::IDENTITY);

        assert_eq!(spec.focal_length, [EUROC_FX, EUROC_FY]);
        assert_eq!(spec.resolution, [EUROC_IMAGE_WIDTH, EUROC_IMAGE_HEIGHT]);
        assert_eq!(spec.principal_point, [EUROC_CX, EUROC_CY]);
        assert_eq!(spec.image_plane_distance, CAMERA_IMAGE_PLANE_DISTANCE);
    }

    #[test]
    fn run_artifact_writer_writes_expected_files() {
        let temp_dir = tempdir().unwrap();
        let run_dir = temp_dir.path().join("run");
        let mut writer = RunArtifactWriter::create(&run_dir).unwrap();

        writer.write_stdout_line("frame 0000 | Bootstrap").unwrap();
        writer.write_frame(&sample_frame_record()).unwrap();
        writer
            .write_summary(
                &sample_summary_record(),
                &[sample_tum_pose(1.0)],
                &[sample_tum_pose(2.0)],
            )
            .unwrap();

        assert!(run_dir.join("frames.jsonl").exists());
        assert!(run_dir.join("summary.json").exists());
        assert!(run_dir.join("trajectory.tum").exists());
        assert!(run_dir.join("keyframes.tum").exists());
        assert!(run_dir.join("stdout.log").exists());
    }

    #[test]
    fn run_artifact_frame_record_is_valid_jsonl() {
        let temp_dir = tempdir().unwrap();
        let run_dir = temp_dir.path().join("run");
        let mut writer = RunArtifactWriter::create(&run_dir).unwrap();

        writer.write_frame(&sample_frame_record()).unwrap();

        let frames_text = fs::read_to_string(run_dir.join("frames.jsonl")).unwrap();
        let value: serde_json::Value = serde_json::from_str(frames_text.trim()).unwrap();
        assert_eq!(value["frame_idx"], 7);
        assert_eq!(value["status"], "Tracked");
    }

    #[test]
    fn frame_record_uses_tracking_result() {
        let result = OdometryResult {
            pose_world_to_cam: Pose3d::IDENTITY,
            status: OdometryStatus::Tracked,
        };

        let record = frame_record_from_tracking_result(7, 1.25, 4, 89, &result);

        assert_eq!(record.frame_idx, 7);
        assert_eq!(record.timestamp_sec, 1.25);
        assert_eq!(record.status, "Tracked");
        assert_eq!(record.keyframe_idx, 4);
        assert_eq!(record.map_point_count, 89);
    }

    fn sample_frame_record() -> FrameRecord {
        FrameRecord {
            frame_idx: 7,
            timestamp_sec: 1.25,
            status: "Tracked".to_string(),
            keyframe_idx: 4,
            world_to_cam: sample_pose_record(),
            cam_to_world: sample_pose_record(),
            map_point_count: 89,
        }
    }

    fn sample_summary_record() -> SummaryRecord {
        let mut status_counts = BTreeMap::new();
        status_counts.insert("Tracked".to_string(), 1);

        SummaryRecord {
            dataset_path: "/tmp/euroc".to_string(),
            run_name: "run".to_string(),
            frame_count: 1,
            status_counts,
            final_map_points: 89,
        }
    }

    fn sample_pose_record() -> PoseRecord {
        PoseRecord {
            translation: [1.0, 2.0, 3.0],
            quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
        }
    }

    fn sample_tum_pose(timestamp: f64) -> TumPose {
        TumPose {
            timestamp,
            tx: 1.0,
            ty: 2.0,
            tz: 3.0,
            qx: 0.0,
            qy: 0.0,
            qz: 0.0,
            qw: 1.0,
        }
    }
}
