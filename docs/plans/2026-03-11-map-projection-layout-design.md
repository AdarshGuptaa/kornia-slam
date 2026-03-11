# Map-Projection Layout Design

**Date:** 2026-03-11

## Goal

Align the `odometry/estimation/` layout with the current dependency structure:

- `two_view` is one estimator technique
- `map_projection` is another estimator technique
- `matching` and `pnp` are currently support modules used by `map_projection`

The filesystem and module boundaries should reflect that `map_projection` is the parent technique, not `pnp`.

## Current Problem

The current layout places these files side-by-side:

- `map_projection.rs`
- `matching.rs`
- `pnp.rs`
- `two_view.rs`

This is workable, but it obscures the fact that `map_projection` is the real tracking estimator while `matching` and `pnp` are support layers for it.

## Design

### Module structure

Restructure:

- `src/odometry/estimation/map_projection.rs`

into:

- `src/odometry/estimation/map_projection/mod.rs`
- `src/odometry/estimation/map_projection/matching.rs`
- `src/odometry/estimation/map_projection/pnp.rs`

Keep:

- `src/odometry/estimation/two_view.rs`

as a sibling estimator module.

### Responsibility split

`map_projection/mod.rs` should own:

- `MapProjectionEstimator`
- `OdometryConfig`
- `KeyframePolicy`
- tracker-specific rejection reasons
- the high-level pose-tracking flow

`map_projection/matching.rs` should own:

- projection-guided matching
- ORB descriptor matching re-exports if still needed here
- keypoint grid utilities

`map_projection/pnp.rs` should own:

- PnP solving
- reprojection inlier counting
- pose plausibility checks
- `PnpConfig`

### Public surface

`src/odometry/estimation/mod.rs` should continue to expose:

- `pub mod map_projection;`
- `pub mod two_view;`

and should re-export:

- `MapProjectionEstimator`

Call sites should keep importing:

- `odometry::estimation::map_projection::OdometryConfig`
- `odometry::estimation::map_projection::KeyframePolicy`
- `odometry::estimation::map_projection::pnp::PnpConfig`

or, if desired, `PnpConfig` can later be re-exported from `map_projection`.

## Non-goals

This change should not alter:

- pose-estimation behavior
- tracking thresholds
- two-view initialization behavior
- public names of `MapProjectionEstimator`, `OdometryConfig`, or `KeyframePolicy`

This is a layout refactor only.

## Testing

Verification should be:

- `cargo test`

If import paths change for tests or the example, update them without changing behavior.
