# Single-File Map-Projection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Collapse the `map_projection/` directory into a single `map_projection.rs` file while keeping the code internally sectioned and preserving the public API.

**Architecture:** Treat `map_projection` as one concrete estimator technique. Move the matching and PnP support code into the same file under explicit section headers so the estimator remains self-contained without spreading logic across multiple nested modules.

**Tech Stack:** Rust, Cargo test runner

---

### Task 1: Collapse the directory into a single file

**Files:**
- Create: `src/odometry/estimation/map_projection.rs`
- Delete: `src/odometry/estimation/map_projection/mod.rs`
- Delete: `src/odometry/estimation/map_projection/matching.rs`
- Delete: `src/odometry/estimation/map_projection/pnp.rs`
- Modify: `src/odometry/estimation/mod.rs`

**Step 1: Write the failing test**

Use the existing compile surface as the test by moving the content into one file and relying on stale module references to fail until fixed.

**Step 2: Run test to verify it fails**

Run: `cargo test`
Expected: FAIL with module/import errors until the single-file layout is wired up.

**Step 3: Write minimal implementation**

- Merge the contents of the three current files into `map_projection.rs`
- Remove nested `pub mod matching;` and `pub mod pnp;`
- Keep the public items and section headers clear

**Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: compile progresses, with any remaining failures limited to stale imports.

**Step 5: Commit**

```bash
git add src/odometry/estimation
git commit -m "refactor: collapse map projection into one file"
```

### Task 2: Update imports to the flattened file

**Files:**
- Modify: `examples/orb_slam/pipeline.rs`
- Modify: `src/odometry/estimation/two_view.rs`
- Modify: `tests/public_module_layout.rs`

**Step 1: Write the failing test**

Use unresolved imports from the flattening as the failing signal.

**Step 2: Run test to verify it fails**

Run: `cargo test`
Expected: FAIL with stale nested import paths such as `map_projection::pnp`.

**Step 3: Write minimal implementation**

- Update imports to the flattened `map_projection` module
- Keep public names unchanged

**Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS

**Step 5: Commit**

```bash
git add src examples tests
git commit -m "refactor: update imports for single-file map projection"
```

### Task 3: Clean section comments and verify readability

**Files:**
- Modify: `src/odometry/estimation/map_projection.rs`

**Step 1: Write the failing test**

No dedicated failing test required; this is a readability cleanup after green.

**Step 2: Run test to verify it fails**

Skip.

**Step 3: Write minimal implementation**

- Add or adjust section headers:
  - public types
  - matching
  - PnP
  - estimator flow
  - tests

**Step 4: Run test to verify it passes**

Run: `cargo test`
Expected: PASS

**Step 5: Commit**

```bash
git add src/odometry/estimation/map_projection.rs
git commit -m "docs: organize single-file map projection sections"
```
