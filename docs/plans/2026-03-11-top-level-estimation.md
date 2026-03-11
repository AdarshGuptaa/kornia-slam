# Top-Level Estimation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move estimation to a top-level module directory and flatten odometry runtime types into a single `src/odometry.rs` file.

**Architecture:** Keep runtime odometry types separate from estimation algorithms, but simplify the filesystem. `odometry.rs` becomes a small runtime-state file, and `estimation/` becomes the top-level home for estimator techniques and re-exports.

**Tech Stack:** Rust, Cargo test runner

---

### Task 1: Move modules to the new layout

**Files:**
- Create: `src/odometry.rs`
- Create: `src/estimation/mod.rs`
- Create: `src/estimation/two_view.rs`
- Create: `src/estimation/map_projection.rs`
- Delete: `src/odometry/mod.rs`
- Delete: `src/odometry/estimation/mod.rs`
- Delete: `src/odometry/estimation/two_view.rs`
- Delete: `src/odometry/estimation/map_projection.rs`
- Delete: `src/odometry/estimation/`

**Step 1: Write the failing test**

Use the existing compile surface as the test by moving files first and relying on stale module declarations/imports to fail.

**Step 2: Run test to verify it fails**

Run: `cargo test`
Expected: FAIL with module/import errors until declarations are updated.

**Step 3: Write minimal implementation**

- Move the files into the new top-level layout
- Update module declarations in `lib.rs`

**Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: compile progresses, with any remaining failures limited to stale import paths.

**Step 5: Commit**

```bash
git add src
git commit -m "refactor: promote estimation to top-level module"
```

### Task 2: Update imports to the new top-level paths

**Files:**
- Modify: Rust files importing `odometry::estimation`
- Modify: `tests/public_module_layout.rs`

**Step 1: Write the failing test**

Use unresolved import errors from the move as the failing signal.

**Step 2: Run test to verify it fails**

Run: `cargo test`
Expected: FAIL with stale `odometry::estimation` paths.

**Step 3: Write minimal implementation**

- Replace imports from:
  - `crate::odometry::estimation::...`
- with:
  - `crate::estimation::...`

**Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS

**Step 5: Commit**

```bash
git add src examples tests
git commit -m "refactor: update imports for top-level estimation"
```

### Task 3: Clean docs and verify module clarity

**Files:**
- Modify: `src/odometry.rs`
- Modify: `src/estimation/mod.rs`
- Modify: `src/lib.rs`

**Step 1: Write the failing test**

No dedicated failing test required; this is a consistency cleanup after green.

**Step 2: Run test to verify it fails**

Skip.

**Step 3: Write minimal implementation**

- Make `odometry.rs` docs clearly runtime-oriented
- Make `estimation/mod.rs` docs clearly algorithm-oriented
- Ensure `lib.rs` re-exports remain easy to read

**Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS

**Step 5: Commit**

```bash
git add src
git commit -m "docs: align top-level odometry and estimation modules"
```
