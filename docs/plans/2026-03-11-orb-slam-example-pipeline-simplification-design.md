# ORB-SLAM Example Pipeline Simplification Design

**Date:** 2026-03-11

## Goal

Simplify the ORB-SLAM example pipeline by moving phase-specific logic out of `examples/orb_slam/pipeline.rs` into example-local modules, while keeping all reusable library boundaries unchanged.

## Current Problem

The current example keeps three different concerns in one file:

- top-level runtime orchestration in `examples/orb_slam/pipeline.rs`
- bootstrap map creation and two-view initialization flow in the same file
- tracking, keyframe insertion, map growth, local BA, and culling in the same file

That makes `pipeline.rs` harder to scan because the state machine is mixed with dense geometry and map-mutation procedures.

## Design

### Filesystem layout

Keep:

- `examples/orb_slam/pipeline.rs`

Add:

- `examples/orb_slam/bootstrap.rs`
- `examples/orb_slam/tracking.rs`

Update `examples/orb_slam/main.rs` to declare the new modules only if needed by the example module layout.

### Responsibility split

`pipeline.rs` should contain:

- `ORBSLAMPipeline`
- constructor and simple getters
- `process_frame`
- thin phase dispatch between bootstrap and tracking

`bootstrap.rs` should contain:

- bootstrap-phase entry point
- first-frame storage logic
- second-frame rejection or acceptance flow
- bootstrap map materialization helper

`tracking.rs` should contain:

- tracking-phase entry point
- map-projection pose estimation flow
- map-point observation updates
- keyframe insertion logic
- map growth helper
- local BA and culling follow-up

### Data flow

Per frame:

1. `pipeline.rs` receives a `Frame`.
2. It selects bootstrap or tracking based on `OdometryState`.
3. The phase module mutates `Map`, `OdometryState`, and estimator state as needed.
4. The phase module returns `OdometryResult`.
5. `main.rs` continues to own dataset I/O, feature extraction, logging, and visualization.

### Module boundary rule

This refactor is intentionally example-local. It should not introduce new public API in `kornia_slam` yet.

The reason is that the current bootstrap and keyframe-maintenance code is still tightly shaped around this example's control flow. Moving it into sibling files improves readability without prematurely freezing library interfaces.

## Non-goals

This change should not:

- alter tracking or bootstrap behavior
- change example CLI behavior
- move new runtime units into `src/`
- add a public top-level pipeline type to `kornia_slam`

## Error Handling

- Keep I/O and fallible external setup in `examples/orb_slam/main.rs`
- Keep `bootstrap.rs` and `tracking.rs` as runtime logic returning `OdometryResult`
- Preserve current `OdometryStatus` behavior exactly
- Preserve current bootstrap fallback after repeated tracking failure

## Testing

- Move bootstrap-focused unit tests next to `examples/orb_slam/bootstrap.rs`
- Move map-growth and keyframe-focused tests next to `examples/orb_slam/tracking.rs`
- Keep `pipeline.rs` tests focused on dispatch and phase-transition behavior
- Run `cargo test --example orb_slam` if available, otherwise `cargo test`

## Recommendation

Use a phase-based split:

- `pipeline.rs` as facade and state machine
- `bootstrap.rs` for initialization logic
- `tracking.rs` for tracking and keyframe maintenance

This is the smallest structural change that materially improves readability and reviewability.
