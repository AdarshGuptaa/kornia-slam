use kornia_algebra::{Vec3F64, Mat3F64};

#[derive(Clone)]
pub struct Trajectory {
    pub timestamps: Vec<f64>,
    pub positions: Vec<Vec3F64>,
}

pub struct Sim3Alignment {
    pub scale: f64,
    pub rotation: Mat3F64,
    pub translation: Vec3F64,
}

pub struct EvaluationResult {
    pub ate: f64,
    pub rpe: f64,
    pub drift: f64,
}