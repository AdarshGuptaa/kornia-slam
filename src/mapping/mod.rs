//! Mapping: map storage, local-map selection, culling, and optimization.

pub mod ba;
pub mod map;

pub use map::cull_map_points;
pub use map::{Keyframe, Map, MapPoint};
