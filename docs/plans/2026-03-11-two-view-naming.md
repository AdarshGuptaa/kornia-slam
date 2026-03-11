# Two-View Naming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rename the `two_view` module surface to `TwoViewInit*`, introduce `TwoViewAcceptanceConfig`, and keep `kornia_3d::pose::TwoViewConfig` as the estimator config without changing behavior.

**Architecture:** Treat `src/odometry/estimation/two_view.rs` as an algorithm module. Its public config should compose three concerns: ORB matching, upstream two-view estimation, and SLAM acceptance gating. All naming should align with the `two_view` algorithm rather than the old bootstrap phase naming.

**Tech Stack:** Rust, Cargo test runner

---

### Task 1: Rename the two-view public API in place

**Files:**
- Modify: `src/odometry/estimation/two_view.rs`
- Modify: `examples/orb_slam/main.rs`
- Modify: `examples/orb_slam/pipeline.rs`
- Modify: `tests/public_module_layout.rs`

**Step 1: Write the failing test**

Use the existing public API layout test as the failing test by updating it first to the intended names.

**Step 2: Run test to verify it fails**

Run: `cargo test tests::public_module_layout`
Expected: FAIL with unresolved symbols until the rename is implemented.

**Step 3: Write minimal implementation**

- Rename:
  - `BootstrapConfig` -> `TwoViewInitConfig`
  - `BootstrapOutcome` -> `TwoViewInitOutcome`
  - `BootstrapRejectReason` -> `TwoViewInitRejectReason`
  - `try_bootstrap` -> `try_initialize_two_view`
- Update imports and uses at all call sites

**Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: compile progresses with any remaining failures limited to config-struct changes.

**Step 5: Commit**

```bash
git add src/odometry/estimation/two_view.rs examples/orb_slam/main.rs examples/orb_slam/pipeline.rs tests/public_module_layout.rs
git commit -m "refactor: rename two-view initialization API"
```

### Task 2: Split estimator config from acceptance config

**Files:**
- Modify: `src/odometry/estimation/two_view.rs`
- Modify: `examples/orb_slam/main.rs`
- Modify: `examples/orb_slam/pipeline.rs`
- Modify: `tests/public_module_layout.rs`

**Step 1: Write the failing test**

Extend the public API layout test to reference `TwoViewAcceptanceConfig`, and update any construction sites to the target nested config shape.

**Step 2: Run test to verify it fails**

Run: `cargo test`
Expected: FAIL until the new config struct and field accesses are implemented.

**Step 3: Write minimal implementation**

- Add:

```rust
pub struct TwoViewAcceptanceConfig {
    pub min_matches: usize,
    pub min_inliers: usize,
    pub min_triangulated: usize,
}
```

- Reshape the top-level config:

```rust
pub struct TwoViewInitConfig {
    pub match_config: OrbMatchConfig,
    pub estimation_config: kornia_3d::pose::TwoViewConfig,
    pub acceptance_config: TwoViewAcceptanceConfig,
}
```

- Update field reads in `try_initialize_two_view`
- Move `min_parallax_deg` use to `estimation_config.min_parallax_deg`

**Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS

**Step 5: Commit**

```bash
git add src/odometry/estimation/two_view.rs examples/orb_slam/main.rs examples/orb_slam/pipeline.rs tests/public_module_layout.rs
git commit -m "refactor: split two-view acceptance config"
```

### Task 3: Clean docs and comments for consistency

**Files:**
- Modify: `src/odometry/estimation/two_view.rs`
- Modify: `src/odometry/mod.rs`

**Step 1: Write the failing test**

No code failure expected here; use this as a consistency cleanup task after the code passes.

**Step 2: Run test to verify it fails**

Skip.

**Step 3: Write minimal implementation**

- Update doc comments to use `two-view initialization` terminology instead of `bootstrap` where appropriate
- Keep runtime state docs in `odometry/mod.rs` phase-oriented where they describe `OdometryMode::Bootstrap`

**Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS

**Step 5: Commit**

```bash
git add src/odometry/estimation/two_view.rs src/odometry/mod.rs
git commit -m "docs: align two-view initialization terminology"
```
