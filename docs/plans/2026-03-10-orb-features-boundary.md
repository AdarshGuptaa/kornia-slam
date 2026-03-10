# ORB Features Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the local `FrameFeatures` struct with the upstream `kornia-imgproc` `OrbFeatures` type while keeping `Frame.features` as the field name.

**Architecture:** Re-export `OrbFeatures` from `kornia-slam`, update `Frame` to store that type directly, and align example/test code with the upstream field names. Document in `frame.rs` that a higher-level abstraction can be introduced later if more feature frontends are supported.

**Tech Stack:** Rust, Cargo, `kornia-imgproc`

---

### Task 1: Replace the local feature payload type

**Files:**
- Modify: `src/frame.rs`
- Modify: `src/lib.rs`
- Modify: `src/odometry/bootstrap.rs`
- Modify: `src/mapping/map.rs`
- Modify: `examples/orb_slam/main.rs`
- Modify: `tests/public_module_layout.rs`

**Step 1: Update the public type boundary**

Remove the local `FrameFeatures` struct, re-export `OrbFeatures` from `frame.rs`, and change `Frame.features` plus `Frame::new` to use `OrbFeatures`.

**Step 2: Update internal library imports**

Switch internal modules that currently reference `FrameFeatures` to use `crate::frame::OrbFeatures`.

**Step 3: Update app/test call sites**

In the ORB example, use `detector.detect_and_extract(...)` directly as the frame feature payload and update tests to construct `OrbFeatures` using the upstream `descriptors` field name.

**Step 4: Verify the refactor**

Run: `cargo test`
Expected: PASS, with only the pre-existing example dead-code warnings.

**Step 5: Commit**

```bash
git add docs/plans/2026-03-10-orb-features-boundary-design.md \
        docs/plans/2026-03-10-orb-features-boundary.md \
        src/frame.rs \
        src/lib.rs \
        src/odometry/bootstrap.rs \
        src/mapping/map.rs \
        examples/orb_slam/main.rs \
        tests/public_module_layout.rs
git commit -m "refactor: use upstream orb feature payload"
```
