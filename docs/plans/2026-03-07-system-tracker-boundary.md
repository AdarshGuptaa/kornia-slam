# System Tracker Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract system orchestration into `src/backend/system.rs` and leave `src/backend/tracker.rs` focused on tracking-stage helpers.

**Architecture:** `system.rs` will become the home of the SLAM state machine and public API. `tracker.rs` will keep the tracking-specific helper types and functions used by the system during the tracking stage.

**Tech Stack:** Rust, cargo test, crate-local unit tests

---

### Task 1: Create a red test around the public system surface

**Files:**
- Modify: `src/backend/tracker.rs`
- Create: `src/backend/system.rs`
- Modify: `src/backend/mod.rs`
- Modify: `src/lib.rs`

**Step 1: Write the failing test**

Add or move one existing `SlamSystem` test so it compiles from `system.rs` and proves the public API is still exposed through `backend` and crate root exports.

**Step 2: Run test to verify it fails**

Run: `cargo test test_bootstrap_needs_first_frame --lib`
Expected: FAIL due to missing/moved types during extraction.

### Task 2: Move system-level types and orchestration into `system.rs`

**Files:**
- Create: `src/backend/system.rs`
- Modify: `src/backend/tracker.rs`
- Modify: `src/backend/mod.rs`
- Modify: `src/lib.rs`

**Step 1: Write minimal implementation**

Move these items into `system.rs`:

- `SystemState`
- `TrackingStatus`
- `FrameRejectReason`
- `TrackingResult`
- `SlamSystem`
- `process_frame`
- `process_bootstrap`
- `process_tracking`
- public accessors and reset

Keep `TrackRejectReason` and tracking helpers in `tracker.rs`.

**Step 2: Run focused tests**

Run: `cargo test --lib`
Expected: tracker/system tests compile or expose the next import break.

### Task 3: Reconnect system-to-tracker dependencies cleanly

**Files:**
- Modify: `src/backend/system.rs`
- Modify: `src/backend/tracker.rs`
- Modify: `src/backend/mod.rs`

**Step 1: Write minimal implementation**

Expose only the tracking helpers needed by `system.rs`, with imports adjusted so:

- `system.rs` can call tracking helpers
- `tracker.rs` no longer depends on owning `SlamSystem`
- backend re-exports still present the same public API

**Step 2: Run focused tests**

Run: `cargo test test_consecutive_failures_resets --lib`
Expected: PASS

### Task 4: Final verification

**Files:**
- No code changes expected

**Step 1: Run verification**

Run: `cargo test`
Expected: all tests pass

**Step 2: Verify example surface**

Run: `cargo test --example mono_vo`
Expected: PASS
