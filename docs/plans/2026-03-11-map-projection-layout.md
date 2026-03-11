# Map-Projection Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the map-projection estimator into its own submodule, with `matching` and `pnp` nested under it, while keeping behavior and public estimator names unchanged.

**Architecture:** Treat `two_view` and `map_projection` as the two current estimator techniques. Move `matching.rs` and `pnp.rs` under `map_projection/` because they are implementation details of the map-projection estimator today. Keep `odometry/estimation/mod.rs` as the top-level estimator registry.

**Tech Stack:** Rust, Cargo test runner

---

### Task 1: Create the map_projection submodule layout

**Files:**
- Create: `src/odometry/estimation/map_projection/mod.rs`
- Create: `src/odometry/estimation/map_projection/matching.rs`
- Create: `src/odometry/estimation/map_projection/pnp.rs`
- Delete: `src/odometry/estimation/map_projection.rs`
- Delete: `src/odometry/estimation/matching.rs`
- Delete: `src/odometry/estimation/pnp.rs`
- Modify: `src/odometry/estimation/mod.rs`

**Step 1: Write the failing test**

Use the existing compile surface as the test by moving files first and relying on broken module declarations/imports to fail.

**Step 2: Run test to verify it fails**

Run: `cargo test`
Expected: FAIL with module/import errors until declarations are updated.

**Step 3: Write minimal implementation**

- Move:
  - `map_projection.rs` -> `map_projection/mod.rs`
  - `matching.rs` -> `map_projection/matching.rs`
  - `pnp.rs` -> `map_projection/pnp.rs`
- Update `estimation/mod.rs` to keep `pub mod map_projection;`
- Add nested `pub mod matching;` and `pub mod pnp;` inside `map_projection/mod.rs`

**Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: compile progresses, with any remaining failures limited to stale import paths.

**Step 5: Commit**

```bash
git add src/odometry/estimation
git commit -m "refactor: nest map projection estimator internals"
```

### Task 2: Update imports to nested paths

**Files:**
- Modify: Rust files importing `odometry::estimation::matching`
- Modify: Rust files importing `odometry::estimation::pnp`
- Modify: `tests/public_module_layout.rs`

**Step 1: Write the failing test**

Use unresolved import errors from the move as the failing signal.

**Step 2: Run test to verify it fails**

Run: `cargo test`
Expected: FAIL with stale imports.

**Step 3: Write minimal implementation**

- Update imports to:
  - `odometry::estimation::map_projection::matching`
  - `odometry::estimation::map_projection::pnp`
- Keep `MapProjectionEstimator` and `OdometryConfig` imports stable at `map_projection`

**Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS

**Step 5: Commit**

```bash
git add src examples tests
git commit -m "refactor: update imports for map projection layout"
```

### Task 3: Clean comments and optional re-exports

**Files:**
- Modify: `src/odometry/estimation/map_projection/mod.rs`
- Modify: `src/odometry/estimation/mod.rs`

**Step 1: Write the failing test**

No dedicated failing test required; this is consistency cleanup after green.

**Step 2: Run test to verify it fails**

Skip.

**Step 3: Write minimal implementation**

- Update module docs to describe `map_projection` as a concrete estimator with nested support modules
- Optionally re-export `PnpConfig` from `map_projection` if that improves the public surface without widening it unnecessarily

**Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS

**Step 5: Commit**

```bash
git add src/odometry/estimation
git commit -m "docs: align map projection module layout"
```
