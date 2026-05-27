use kornia_algebra::Vec3F64;

pub fn compute_ate(est: &[Vec3F64], gt: &[Vec3F64]) -> f64 {
    let mut sum = 0.0_f64;
    for i in 0..est.len() {
        let d = est[i] - gt[i];
        sum += d.dot(d);
    }
    (sum / est.len() as f64).sqrt()
}

pub fn compute_rpe(est: &[Vec3F64], gt: &[Vec3F64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for i in 0..est.len() - 1 {
        let de = est[i + 1] - est[i];
        let dg = gt[i + 1] - gt[i];
        let d = de - dg;
        sum += d.dot(d);
        count += 1;
    }
    (sum / count as f64).sqrt()
}

pub fn compute_drift(est: &[Vec3F64], gt: &[Vec3F64]) -> f64 {
    let diff = *est.last().unwrap() - *gt.last().unwrap();
    let final_error = diff.dot(diff).sqrt();
    let mut length = 0.0_f64;
    for i in 1..gt.len() {
        let d = gt[i] - gt[i - 1];
        length += d.dot(d).sqrt();
    }
    final_error / length
}
