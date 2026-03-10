# SLAM Run Artifacts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make both monocular VO examples emit comparable run directories with per-frame debug telemetry, summaries, and trajectories.

**Architecture:** Add a temporary debug telemetry payload to the ported tracker result, normalize both examples onto a shared artifact schema, and write run directories through small example-local helpers. Keep the new instrumentation isolated so it can be removed after debugging.

**Tech Stack:** Rust, `argh`, existing `mono_vo` example utilities, JSON/JSONL serialization via `serde`/`serde_json`, TUM trajectory files.

---

### Task 1: Add artifact-writing dependencies and scaffolding in the ported crate

**Files:**
- Modify: `/home/christie/projects/kornia-slam/crates/kornia-slam/Cargo.toml`
- Modify: `/home/christie/projects/kornia-slam/crates/kornia-slam/examples/mono_vo/main.rs`
- Modify: `/home/christie/projects/kornia-slam/crates/kornia-slam/examples/mono_vo/utils.rs`
- Test: `/home/christie/projects/kornia-slam/crates/kornia-slam/examples/mono_vo/utils.rs`

**Step 1: Write the failing tests**

Add example-local tests for:
- creating a run directory layout
- writing one `frames.jsonl` record
- writing `summary.json`
- writing `keyframes.tum`

**Step 2: Run test to verify it fails**

Run: `cargo test --example mono_vo run_artifact`
Expected: FAIL because the run-writer helpers do not exist yet.

**Step 3: Write minimal implementation**

Add:
- CLI option for output directory
- serializable run artifact structs
- helper to create output paths and write JSON/JSONL/TUM files

**Step 4: Run test to verify it passes**

Run: `cargo test --example mono_vo run_artifact`
Expected: PASS

### Task 2: Add temporary debug telemetry to the ported tracker

**Files:**
- Modify: `/home/christie/projects/kornia-slam/crates/kornia-slam/src/backend/tracker.rs`
- Modify: `/home/christie/projects/kornia-slam/crates/kornia-slam/src/backend/mod.rs`
- Test: `/home/christie/projects/kornia-slam/crates/kornia-slam/src/backend/tracker.rs`

**Step 1: Write the failing tests**

Add tracker tests that verify:
- bootstrap results populate debug feature counts and map counts
- rejected initialization carries stable reject codes
- rejected tracking carries stage-match counts and reject codes

**Step 2: Run test to verify it fails**

Run: `cargo test backend::tracker::tests::debug_`
Expected: FAIL because `DebugFrameInfo` and reject-code helpers are missing.

**Step 3: Write minimal implementation**

Add:
- `DebugFrameInfo`
- stable reject-code helpers
- temporary debug fields on `TrackingResult`
- instrumentation in bootstrap and tracking paths

**Step 4: Run test to verify it passes**

Run: `cargo test backend::tracker::tests::debug_`
Expected: PASS

### Task 3: Wire the ported example to emit shared run artifacts

**Files:**
- Modify: `/home/christie/projects/kornia-slam/crates/kornia-slam/examples/mono_vo/main.rs`
- Modify: `/home/christie/projects/kornia-slam/crates/kornia-slam/examples/mono_vo/utils.rs`
- Test: `/home/christie/projects/kornia-slam/crates/kornia-slam/examples/mono_vo/utils.rs`

**Step 1: Write the failing tests**

Add a test covering conversion from `TrackingResult` + example bookkeeping into one frame artifact record.

**Step 2: Run test to verify it fails**

Run: `cargo test --example mono_vo frame_record`
Expected: FAIL because the conversion helper does not exist yet.

**Step 3: Write minimal implementation**

Update the example loop to:
- create a run writer
- mirror console lines to `stdout.log`
- emit one JSONL record per frame
- accumulate summary counters
- write `trajectory.tum`, `keyframes.tum`, and `summary.json` at the end
- keep Rerun logging intact

**Step 4: Run test to verify it passes**

Run: `cargo test --example mono_vo frame_record`
Expected: PASS

### Task 4: Add artifact-writing dependencies and scaffolding in the original crate

**Files:**
- Modify: `/home/christie/projects/kornia-rs/crates/kornia-slam/Cargo.toml`
- Modify: `/home/christie/projects/kornia-rs/crates/kornia-slam/examples/mono_vo/main.rs`
- Modify: `/home/christie/projects/kornia-rs/crates/kornia-slam/examples/mono_vo/utils.rs`
- Test: `/home/christie/projects/kornia-rs/crates/kornia-slam/examples/mono_vo/utils.rs`

**Step 1: Write the failing tests**

Add example-local tests for the same run directory helpers used by the ported example.

**Step 2: Run test to verify it fails**

Run: `cargo test -p kornia-slam --example mono_vo run_artifact`
Expected: FAIL because the original example has no run-writer helpers yet.

**Step 3: Write minimal implementation**

Add the same artifact structs and file-writing helpers, adjusted for the original example’s utility module.

**Step 4: Run test to verify it passes**

Run: `cargo test -p kornia-slam --example mono_vo run_artifact`
Expected: PASS

### Task 5: Wire the original example to emit the shared run artifacts

**Files:**
- Modify: `/home/christie/projects/kornia-rs/crates/kornia-slam/examples/mono_vo/main.rs`
- Modify: `/home/christie/projects/kornia-rs/crates/kornia-slam/examples/mono_vo/utils.rs`
- Test: `/home/christie/projects/kornia-rs/crates/kornia-slam/examples/mono_vo/utils.rs`

**Step 1: Write the failing tests**

Add a test for converting the original `TrackingResult` into the shared frame artifact schema.

**Step 2: Run test to verify it fails**

Run: `cargo test -p kornia-slam --example mono_vo frame_record`
Expected: FAIL because the shared frame-record conversion does not exist yet.

**Step 3: Write minimal implementation**

Update the original example loop to:
- create the run writer
- write per-frame JSONL
- write end-of-run summary and TUM files
- preserve existing console and Rerun behavior

**Step 4: Run test to verify it passes**

Run: `cargo test -p kornia-slam --example mono_vo frame_record`
Expected: PASS

### Task 6: Run end-to-end verification

**Files:**
- Modify: none
- Test: both example crates

**Step 1: Run focused tests**

Run: `cargo test --example mono_vo`
Expected: PASS in `/home/christie/projects/kornia-slam/crates/kornia-slam`

**Step 2: Run focused tests in the original crate**

Run: `cargo test -p kornia-slam --example mono_vo`
Expected: PASS in `/home/christie/projects/kornia-rs`

**Step 3: Run short smoke tests**

Run: `cargo run --example mono_vo -- --data /home/christie/projects/kornia-rs/tests/data/MH_01_easy --max-frames 5`
Expected: run directory created with `summary.json`, `frames.jsonl`, `trajectory.tum`, `keyframes.tum`, `stdout.log`

Run: `cargo run -p kornia-slam --example mono_vo -- --data /home/christie/projects/kornia-rs/tests/data/MH_01_easy --max-frames 5`
Expected: same artifact set in the original crate

### Task 7: Document temporary status

**Files:**
- Modify: `/home/christie/projects/kornia-slam/crates/kornia-slam/examples/mono_vo/main.rs`
- Modify: `/home/christie/projects/kornia-rs/crates/kornia-slam/examples/mono_vo/main.rs`

**Step 1: Add short comments**

Add a brief comment near the run-artifact/debug telemetry wiring that it is temporary debugging instrumentation intended for removal after tracker comparison work is complete.

**Step 2: Re-run relevant tests**

Run: `cargo test --example mono_vo`
Expected: PASS

Run: `cargo test -p kornia-slam --example mono_vo`
Expected: PASS
