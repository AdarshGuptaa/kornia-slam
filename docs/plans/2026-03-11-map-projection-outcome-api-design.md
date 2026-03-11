# Map-Projection Outcome API Design

**Date:** 2026-03-11

## Goal

Make the public API of the map-projection estimator follow the same shape as the two-view estimator.

Today:

- `two_view` exposes one public entrypoint and one explicit outcome enum
- `map_projection` exposes `track(...) -> Result<Pose3d, ...>` and stores matches/inliers as side-channel state on the estimator

The map-projection API should become explicit and self-contained.

## Design

### Public surface

Keep:

- `MapProjectionConfig`
- `MapProjectionRejectReason`
- `MapProjectionEstimator`

Add:

- `MapProjectionEstimateOutcome`

with shape:

```rust
pub enum MapProjectionEstimateOutcome {
    Rejected { reason: MapProjectionRejectReason },
    Estimated {
        pose_world_to_cam: Pose3d,
        inliers: usize,
        matches: Vec<(usize, usize)>,
    },
}
```

### Estimator entrypoint

Replace:

- `track(...) -> Result<Pose3d, MapProjectionRejectReason>`

with:

- `estimate_pose(...) -> MapProjectionEstimateOutcome`

This makes the estimator parallel the two-view module:

- `try_initialize_two_view(...) -> TwoViewInitOutcome`
- `estimate_pose(...) -> MapProjectionEstimateOutcome`

### Internal state

Remove:

- `last_matches`
- `last_inliers`
- `last_matches()`
- `last_inliers()`

These fields only exist today because the public API does not return the full estimation result directly.

### Call-site impact

The ORB pipeline should match on `MapProjectionEstimateOutcome` instead of:

- calling `track(...)`
- then separately calling `last_matches()` and `last_inliers()`

This keeps the estimator boundary self-contained and reduces hidden state.

## Non-goals

This change should not alter:

- tracking behavior
- PnP math
- matching strategy
- keyframe heuristics

This is an API and naming cleanup only.

## Testing

Verification should be:

- `cargo test`

If existing tests assume `track()` or `last_*`, update them to the new explicit outcome API.
