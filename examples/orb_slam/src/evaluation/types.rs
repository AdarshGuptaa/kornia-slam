use kornia_algebra::{Mat3F64, Vec3F64};

pub struct Sim3Alignment {
    pub scale: f64,
    pub rotation: Mat3F64,
    pub translation: Vec3F64,
}
