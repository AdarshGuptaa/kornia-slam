use crate::datasets::euroc::GroundTruthPose;

pub fn associate_gt(t: f64, gt: &[GroundTruthPose]) -> Option<&GroundTruthPose> {
    gt.iter().min_by(|a, b| {
        (a.timestamp_sec - t)
            .abs()
            .partial_cmp(&(b.timestamp_sec - t).abs())
            .unwrap()
    })
}
