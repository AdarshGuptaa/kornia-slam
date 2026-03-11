# Odometry Estimation Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the two-view initialization code into the estimation module and rename it from `bootstrap` to `two_view` without changing behavior.

**Architecture:** Keep `src/odometry/mod.rs` as the odometry runtime/state boundary and make `src/odometry/estimation/` the home for algorithm modules. This change is a pure module-layout cleanup: move the file, update imports/re-exports, and verify the existing behavior still passes tests.

**Tech Stack:** Rust, Cargo test runner

---

### Task 1: Move the module and update declarations

**Files:**
- Create: `src/odometry/estimation/two_view.rs`
- Modify: `src/odometry/mod.rs`
- Modify: `src/odometry/estimation/mod.rs`

**Step 1: Write the failing test**

Use the existing compile surface as the test: the crate should fail to build if module declarations are inconsistent after the move.

**Step 2: Run test to verify it fails**

Run: `cargo test`
Expected: FAIL after the file move until module declarations/imports are updated.

**Step 3: Write minimal implementation**

- Move the contents of `src/odometry/bootstrap.rs` to `src/odometry/estimation/two_view.rs`
- Remove `pub mod bootstrap;` from `src/odometry/mod.rs`
- Add `pub mod two_view;` to `src/odometry/estimation/mod.rs`

**Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: compile progresses, with any remaining failures limited to stale imports.

**Step 5: Commit**

```bash
git add src/odometry/mod.rs src/odometry/estimation/mod.rs src/odometry/estimation/two_view.rs
git commit -m "refactor: move two-view init under estimation"
```

### Task 2: Update imports and references

**Files:**
- Modify: `examples/orb_slam/pipeline.rs`
- Modify: any Rust files importing `crate::odometry::bootstrap`

**Step 1: Write the failing test**

Use the existing compile surface again: unresolved import errors are the failing signal.

**Step 2: Run test to verify it fails**

Run: `cargo test`
Expected: FAIL with unresolved imports referencing `bootstrap`.

**Step 3: Write minimal implementation**

- Replace imports from `odometry::bootstrap` with `odometry::estimation::two_view`
- Keep the existing type and function names for now:
  - `BootstrapConfig`
  - `BootstrapOutcome`
  - `try_bootstrap`

**Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS

**Step 5: Commit**

```bash
git add examples/orb_slam/pipeline.rs src
git commit -m "refactor: update odometry imports for two-view module"
```

### Task 3: Optional follow-up naming cleanup

**Files:**
- Modify: `src/odometry/estimation/two_view.rs`
- Modify: any import sites if the public symbols are renamed later

**Step 1: Write the failing test**

Do not rename public symbols in this change unless there is a clear follow-up need. Treat this as optional and separate.

**Step 2: Run test to verify it fails**

Run: `cargo test`
Expected: only needed if symbol renames are attempted.

**Step 3: Write minimal implementation**

If desired later, rename:
- `BootstrapConfig` -> `TwoViewConfig` or `TwoViewInitConfig`
- `BootstrapOutcome` -> `TwoViewInitOutcome`

This should be deferred unless the naming inconsistency becomes painful.

**Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS

**Step 5: Commit**

```bash
git add src examples
git commit -m "refactor: align two-view estimation naming"
```
