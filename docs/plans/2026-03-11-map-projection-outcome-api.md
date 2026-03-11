# Map-Projection Outcome API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current `track()` plus `last_*` side-channel API with a single explicit `estimate_pose()` outcome API for the map-projection estimator.

**Architecture:** Mirror the `two_view` module shape. `MapProjectionEstimator` should expose one public entrypoint returning a concrete outcome enum, and the caller should consume the pose, inliers, and matches directly from that outcome instead of reading mutable estimator state.

**Tech Stack:** Rust, Cargo test runner

---

### Task 1: Introduce the explicit outcome type

**Files:**
- Modify: `src/odometry/estimation/map_projection/mod.rs`
- Modify: `tests/public_module_layout.rs`

**Step 1: Write the failing test**

Update the public API layout test to reference `MapProjectionEstimateOutcome`.

**Step 2: Run test to verify it fails**

Run: `cargo test --test public_module_layout`
Expected: FAIL with unresolved symbol errors until the outcome type is added.

**Step 3: Write minimal implementation**

- Add:

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

**Step 4: Run test to verify it passes**

Run: `cargo test --test public_module_layout`
Expected: PASS

**Step 5: Commit**

```bash
git add src/odometry/estimation/map_projection/mod.rs tests/public_module_layout.rs
git commit -m "refactor: add map projection estimate outcome"
```

### Task 2: Replace `track()` with `estimate_pose()`

**Files:**
- Modify: `src/odometry/estimation/map_projection/mod.rs`
- Modify: `examples/orb_slam/pipeline.rs`

**Step 1: Write the failing test**

Update the example pipeline to call `estimate_pose()` and match on the outcome before changing the estimator implementation.

**Step 2: Run test to verify it fails**

Run: `cargo test`
Expected: FAIL until the new method exists and the old state access is removed.

**Step 3: Write minimal implementation**

- Rename/remove `track()`
- Expose `estimate_pose(...) -> MapProjectionEstimateOutcome`
- Have it return the pose, inliers, and matches directly
- Update the pipeline to consume the outcome explicitly

**Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS

**Step 5: Commit**

```bash
git add src/odometry/estimation/map_projection/mod.rs examples/orb_slam/pipeline.rs
git commit -m "refactor: use explicit map projection estimate outcome"
```

### Task 3: Remove last-result side-channel state

**Files:**
- Modify: `src/odometry/estimation/map_projection/mod.rs`

**Step 1: Write the failing test**

Use compile failures from removing `last_matches` / `last_inliers` and their accessors as the failing signal.

**Step 2: Run test to verify it fails**

Run: `cargo test`
Expected: FAIL if any stale uses remain.

**Step 3: Write minimal implementation**

- Remove:
  - `last_matches`
  - `last_inliers`
  - `last_matches()`
  - `last_inliers()`
- Keep only persistent estimator configuration and camera state

**Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS

**Step 5: Commit**

```bash
git add src/odometry/estimation/map_projection/mod.rs examples/orb_slam/pipeline.rs
git commit -m "refactor: remove map projection last-result state"
```
