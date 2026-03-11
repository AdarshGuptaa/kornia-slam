# Top-Level Estimation Design

**Date:** 2026-03-11

## Goal

Flatten the current `odometry/` layout so that:

- `odometry` runtime types live in a single top-level file
- `estimation/` becomes a top-level module directory for the algorithm code

This keeps the conceptual distinction between runtime state and estimation algorithms, while reducing one directory layer.

## Current Problem

The current layout is:

- `src/odometry/mod.rs`
- `src/odometry/estimation/mod.rs`
- `src/odometry/estimation/two_view.rs`
- `src/odometry/estimation/map_projection.rs`

At this point:

- `odometry/mod.rs` is small and only contains runtime state/mode/result types
- most of the real code volume now lives under `estimation`

That makes `odometry/` feel like an extra directory layer that is no longer pulling its weight.

## Design

### Filesystem layout

Replace:

- `src/odometry/mod.rs`
- `src/odometry/estimation/`

with:

- `src/odometry.rs`
- `src/estimation/mod.rs`
- `src/estimation/two_view.rs`
- `src/estimation/map_projection.rs`

### Responsibility split

`odometry.rs` should contain only runtime concepts:

- `OdometryState`
- `OdometryMode`
- `OdometryResult`
- `OdometryStatus`

`estimation/` should contain only estimator modules and their re-exports:

- `two_view`
- `map_projection`
- `MapProjectionEstimator`

### Public surface

The crate root should continue to expose:

- `OdometryState`, `OdometryMode`, `OdometryResult`, `OdometryStatus`
- `MapProjectionEstimator`

Call sites should switch from:

- `crate::odometry::estimation::...`

to:

- `crate::estimation::...`

## Non-goals

This change should not alter:

- estimator behavior
- odometry state behavior
- public type names

This is a module-layout simplification only.

## Testing

Verification should be:

- `cargo test`

If imports break, update them as part of the move without changing semantics.
