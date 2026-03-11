# Odometry Estimation Layout Design

**Date:** 2026-03-11

## Goal

Align the `odometry/` module layout with the actual responsibilities in the code:

- `odometry/mod.rs` should define odometry runtime and state-machine types.
- `odometry/estimation/` should contain estimation algorithms.
- the current `bootstrap.rs` should be renamed to `two_view.rs` because the implementation is two-view geometric initialization, not a generic "bootstrap" algorithm.

## Current Problem

The current layout mixes phase-oriented and algorithm-oriented naming:

- `src/odometry/bootstrap.rs` contains ORB matching, two-view estimation, triangulation gating, and pose initialization.
- `src/odometry/estimation/` already contains algorithm modules such as `map_projection.rs`, `matching.rs`, and `pnp.rs`.

This makes `bootstrap.rs` look like a runtime-state concern, even though it is really an estimation algorithm module.

## Design

### Module boundary

Keep `src/odometry/mod.rs` focused on:

- `OdometryState`
- `OdometryMode`
- `OdometryResult`
- `OdometryStatus`

Move the two-view initialization code under `src/odometry/estimation/`.

### Naming

Rename:

- `src/odometry/bootstrap.rs`

to:

- `src/odometry/estimation/two_view.rs`

This name matches the actual method implemented in the file:

- ORB feature matching
- two-view model estimation
- inlier and parallax gating
- initial pose and triangulation output

### Public surface

`src/odometry/estimation/mod.rs` should declare and re-export the new module:

- `pub mod two_view;`

Call sites should import from:

- `crate::odometry::estimation::two_view`

or through explicit re-exports if desired.

### Non-goals

This change should not alter:

- bootstrap behavior
- estimator math
- config values
- result structs or rejection logic

This is a naming and module-boundary cleanup only.

## Testing

Verification should be limited to:

- `cargo test`

If any imports break, fix them without changing behavior.
