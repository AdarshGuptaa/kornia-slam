//! Trajectory evaluation: ground-truth association, Sim3 alignment,
//! ATE/RPE/drift metrics, and CSV reporting.

use std::io::Write;
use std::path::Path;

use kornia_algebra::Vec3F64;
use kornia_slam::sim3::align_sim3;

use crate::datasets::euroc::GroundTruthPose;

// ── Association ──────────────────────────────────────────────────────────

/// Nearest ground-truth pose to timestamp `t` (by absolute time difference).
pub fn associate_gt(t: f64, gt: &[GroundTruthPose]) -> Option<&GroundTruthPose> {
    gt.iter().min_by(|a, b| {
        (a.timestamp_sec - t)
            .abs()
            .partial_cmp(&(b.timestamp_sec - t).abs())
            .unwrap()
    })
}

// ── Metrics ──────────────────────────────────────────────────────────────

pub fn compute_ate(est: &[Vec3F64], gt: &[Vec3F64]) -> f64 {
    let sum: f64 = est
        .iter()
        .zip(gt)
        .map(|(e, g)| {
            let d = *e - *g;
            d.dot(d)
        })
        .sum();
    (sum / est.len() as f64).sqrt()
}

pub fn compute_rpe(est: &[Vec3F64], gt: &[Vec3F64]) -> f64 {
    let sum: f64 = est
        .windows(2)
        .zip(gt.windows(2))
        .map(|(e, g)| {
            let d = (e[1] - e[0]) - (g[1] - g[0]);
            d.dot(d)
        })
        .sum();
    (sum / (est.len() - 1) as f64).sqrt()
}

pub fn compute_drift(est: &[Vec3F64], gt: &[Vec3F64]) -> f64 {
    let diff = *est.last().unwrap() - *gt.last().unwrap();
    let final_error = diff.dot(diff).sqrt();
    let length: f64 = gt
        .windows(2)
        .map(|w| {
            let d = w[1] - w[0];
            d.dot(d).sqrt()
        })
        .sum();
    final_error / length
}

// ── Reporting ────────────────────────────────────────────────────────────

/// Writes the trajectory as CSVs under `out_dir` (created if missing) and
/// prints ATE/RPE/drift. A no-op for fewer than two samples or a length
/// mismatch.
///
/// `anchored` is true only when the AprilTag metric correction was actually
/// applied to the positions (its Sim3 is already in them), so they are scored
/// as-is — no second best-fit Sim3, which would otherwise hide whether the
/// anchor succeeded. When `false` — no stage, or the solve failed — the
/// trajectory is still in the map's arbitrary monocular frame and is first
/// aligned to ground truth by a best-fit Sim3.
pub fn report(
    est: &[Vec3F64],
    gt: &[Vec3F64],
    out_dir: &Path,
    anchored: bool,
) -> std::io::Result<()> {
    if est.len() < 2 || gt.len() != est.len() {
        return Ok(());
    }
    std::fs::create_dir_all(out_dir)?;

    // Raw trajectory as recorded (post-anchor when the anchor applied).
    let mut raw = std::fs::File::create(out_dir.join("kornia_slam_raw.csv"))?;
    writeln!(raw, "idx,x,y,z")?;
    for (i, p) in est.iter().enumerate() {
        writeln!(raw, "{i},{:.9},{:.9},{:.9}", p.x, p.y, p.z)?;
    }

    let (aligned, alignment_label): (Vec<Vec3F64>, String) = if anchored {
        (est.to_vec(), "none (AprilTag metric anchor)".to_string())
    } else {
        // The length guard above means the fit cannot be refused; a refusal is
        // treated as "nothing to report".
        let Ok(a) = align_sim3(est, gt) else {
            return Ok(());
        };
        let aligned = est
            .iter()
            .map(|p| a.translation + (a.rotation * *p) * a.scale)
            .collect();
        (
            aligned,
            format!("best-fit Sim3 to ground truth (scale {:.6})", a.scale),
        )
    };

    let mut af = std::fs::File::create(out_dir.join("kornia_slam_aligned.csv"))?;
    writeln!(af, "idx,x,y,z,gt_x,gt_y,gt_z")?;
    for (i, (p, g)) in aligned.iter().zip(gt).enumerate() {
        writeln!(
            af,
            "{i},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}",
            p.x, p.y, p.z, g.x, g.y, g.z
        )?;
    }

    eprintln!();
    eprintln!("================ Trajectory Evaluation ================");
    eprintln!("Frames evaluated : {}", aligned.len());
    eprintln!("Alignment        : {alignment_label}");
    eprintln!("ATE RMSE         : {:.4} m", compute_ate(&aligned, gt));
    eprintln!("RPE RMSE         : {:.4} m", compute_rpe(&aligned, gt));
    eprintln!(
        "Final Drift      : {:.4} %",
        compute_drift(&aligned, gt) * 100.0
    );
    eprintln!("=======================================================");
    Ok(())
}
