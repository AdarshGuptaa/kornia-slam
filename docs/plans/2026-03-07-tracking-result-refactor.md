# Tracking Result Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove reporting metadata from `TrackingResult`, expose explicit `SlamSystem` accessors, and update the example/reporting path accordingly.

**Architecture:** Keep `TrackingResult` limited to per-frame outcome data. Move keyframe and map-point reporting to `SlamSystem` accessors so callers read current tracker state directly after processing a frame.

**Tech Stack:** Rust, cargo test, crate-local unit tests

---

### Task 1: Lock the new public shape with tests

**Files:**
- Modify: `src/backend/tracker.rs`
- Modify: `examples/mono_vo/utils.rs`

**Step 1: Write the failing test**

- Add a tracker test that expects `SlamSystem::current_keyframe_idx()` and `SlamSystem::num_map_points()` to expose tracker state.
- Update the example utility test to build a frame record from explicit `keyframe_idx` and `map_point_count` inputs instead of reading them from `TrackingResult`.

**Step 2: Run test to verify it fails**

Run: `cargo test frame_record_uses_tracking_result --example mono_vo`

Expected: fail because the helper still reads removed fields or the new API is missing.

### Task 2: Implement the minimal tracker API change

**Files:**
- Modify: `src/backend/tracker.rs`
- Modify: `src/lib.rs`

**Step 1: Write minimal implementation**

- Remove `keyframe_idx` and `num_map_points` from `TrackingResult`.
- Add `SlamSystem::current_keyframe_idx(&self) -> Option<usize>`.
- Add `SlamSystem::num_map_points(&self) -> usize`.

**Step 2: Run focused tests**

Run: `cargo test --lib tracker`

Expected: tracker tests pass.

### Task 3: Update the example/reporting path

**Files:**
- Modify: `examples/mono_vo/main.rs`
- Modify: `examples/mono_vo/utils.rs`

**Step 1: Write minimal implementation**

- Query `SlamSystem` for `keyframe_idx` and `map_point_count` after each processed frame.
- Pass those values into frame-record generation and status logging.

**Step 2: Run focused tests**

Run: `cargo test --example mono_vo`

Expected: example tests pass.

### Task 4: Full verification

**Files:**
- No code changes expected

**Step 1: Run verification**

Run: `cargo test`

Expected: all tests pass.
