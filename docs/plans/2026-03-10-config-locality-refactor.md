# Config Locality Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move subsystem-specific config types next to their owning modules while keeping a thin top-level `SLAMConfig` composition layer.

**Architecture:** `BootstrapConfig` will live in the bootstrap module, `PnpConfig` in the PnP module, and tracking-related config in the map-projection estimator module. The root config module will keep only `SLAMConfig` plus re-exports so the crate still has one obvious entry point for system construction.

**Tech Stack:** Rust, Cargo, module refactors, compile-driven tests

---

### Task 1: Drive the refactor with a failing public API test

**Files:**
- Modify: `tests/public_module_layout.rs`

**Step 1: Update imports to the desired config locations**

Change the test to import:
- `BootstrapConfig` from `kornia_slam::odometry::bootstrap`
- `PnpConfig` from `kornia_slam::odometry::estimation::pnp`
- `OdometryConfig` and `KeyframePolicy` from `kornia_slam::odometry::estimation::map_projection`

**Step 2: Run the test and verify it fails**

Run: `cargo test public_module_layout`

Expected: compile failure because those config types are not yet exposed from those modules.

### Task 2: Move config ownership to the subsystem modules

**Files:**
- Modify: `src/odometry/bootstrap.rs`
- Modify: `src/odometry/estimation/pnp.rs`
- Modify: `src/odometry/estimation/map_projection.rs`

**Step 1: Define `BootstrapConfig` in `bootstrap.rs`**

Move the struct and its `Default` impl into the bootstrap module.

**Step 2: Define `PnpConfig` in `pnp.rs`**

Move the struct and its `Default` impl into the PnP module.

**Step 3: Define tracking-related config in `map_projection.rs`**

Move `KeyframePolicy` and `OdometryConfig` into the map-projection module.

### Task 3: Keep a thin top-level config composition layer

**Files:**
- Modify: `src/config.rs`
- Modify: imports in files that currently use moved config types

**Step 1: Re-export moved config types from `config.rs`**

Keep:
- `pub use` re-exports for moved config types
- `SLAMConfig` with `bootstrap` and `odometry`

**Step 2: Update intra-module imports**

Expected final usage:
- subsystem modules use local config types directly
- callers can still construct `SLAMConfig::default()`

### Task 4: Verify the refactor

**Files:**
- Test: `tests/public_module_layout.rs`

**Step 1: Run targeted test**

Run: `cargo test public_module_layout`

Expected: pass

**Step 2: Run full test suite**

Run: `cargo test`

Expected: pass
