//! Mapping: map storage, triangulation, culling, and optimization.

pub mod ba;
pub mod map;

pub use map::{build_initial_map, grow_map_points_from_keyframe_pair};
pub use map::cull_map_points;
pub use map::{Keyframe, Map, MapPoint};
