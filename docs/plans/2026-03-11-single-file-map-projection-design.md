# Single-File Map-Projection Design

**Date:** 2026-03-11

## Goal

Collapse the `map_projection/` directory back into a single `map_projection.rs` file while preserving the conceptual separation inside the file.

The intended outcome is:

- fewer module boundaries
- one file per estimator technique
- internal section headers that keep matching, PnP, and estimator flow easy to navigate

## Current Problem

The current `map_projection` layout is technically coherent:

- `map_projection/mod.rs`
- `map_projection/matching.rs`
- `map_projection/pnp.rs`

But if the priority is simplicity, this is now more structure than the current estimator surface needs.

## Design

### Filesystem layout

Replace:

- `src/odometry/estimation/map_projection/mod.rs`
- `src/odometry/estimation/map_projection/matching.rs`
- `src/odometry/estimation/map_projection/pnp.rs`

with:

- `src/odometry/estimation/map_projection.rs`

### Internal organization

Keep the file structured with clear section headers, for example:

- public config and outcome types
- matching
- PnP
- estimator
- tests

The goal is not to flatten everything into an undifferentiated block. The file should still read as one estimator with clear subparts.

### Public surface

Keep the existing public API unchanged:

- `MapProjectionConfig`
- `MapProjectionRejectReason`
- `MapProjectionEstimateOutcome`
- `MapProjectionEstimator`
- `KeyframePolicy`
- `PnpConfig`

If `ProjectionMatchConfig` and `KeypointGrid` stay public today, they can remain public in the single file as well.

## Non-goals

This change should not alter:

- estimator behavior
- matching behavior
- PnP behavior
- public names

This is a layout simplification only.

## Testing

Verification should be:

- `cargo test`

If import paths change, update them without changing semantics.
