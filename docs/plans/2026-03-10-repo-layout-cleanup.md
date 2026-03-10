# Repo Layout Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Flatten the repository so the single `kornia-slam` crate lives at the repository root instead of under `crates/kornia-slam`.

**Architecture:** The repo currently contains one real Rust crate plus repo-level `docs/` and `runs/` directories. The cleanup will move the crate root files and directories (`Cargo.toml`, `Cargo.lock`, `src/`, `examples/`, `tests/`) to the repository root, preserve repo-level docs and artifacts, and then re-run Cargo commands from the new root.

**Tech Stack:** Rust, Cargo, shell file moves, README/docs updates if paths change

---

### Task 1: Define the target root-crate layout

**Files:**
- Create: `docs/plans/2026-03-10-repo-layout-cleanup.md`
- Modify: `README.md` only if path or command references need adjustment

**Step 1: Confirm the current crate contents**

Run: `find crates/kornia-slam -maxdepth 2 -mindepth 1 | sort`

Expected: crate-local `Cargo.toml`, `Cargo.lock`, `src/`, `examples/`, `tests/`, plus generated artifacts like `target/`.

**Step 2: Define the files that move to repo root**

Move only:
- `crates/kornia-slam/Cargo.toml`
- `crates/kornia-slam/Cargo.lock`
- `crates/kornia-slam/src/`
- `crates/kornia-slam/examples/`
- `crates/kornia-slam/tests/`

Leave repo-level:
- `README.md`
- `docs/`
- `runs/`

**Step 3: Exclude generated artifacts from the move**

Do not move:
- `crates/kornia-slam/target/`
- `crates/kornia-slam/runs/`
- `crates/kornia-slam/mono_vo_run-*`
- nested crate-local `docs/`

### Task 2: Move the crate root to the repository root

**Files:**
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`
- Move: `src/`, `examples/`, `tests/`

**Step 1: Move Cargo manifests and source directories**

Run the file moves from the repo root.

Expected result:
- `/home/christie/projects/kornia-slam/Cargo.toml`
- `/home/christie/projects/kornia-slam/Cargo.lock`
- `/home/christie/projects/kornia-slam/src/`
- `/home/christie/projects/kornia-slam/examples/`
- `/home/christie/projects/kornia-slam/tests/`

**Step 2: Remove now-empty crate scaffolding if safe**

If `crates/kornia-slam` only contains generated leftovers after the move, delete those leftovers or leave them if they are user artifacts that should be preserved outside this cleanup.

### Task 3: Fix path references after the move

**Files:**
- Modify: `Cargo.toml`
- Modify: `README.md` if commands or paths mention `crates/kornia-slam`

**Step 1: Update path dependencies**

Adjust relative paths in `Cargo.toml` from:
- `../../../kornia-rs/...`

to:
- `../kornia-rs/...`

Expected: dependency resolution works from the repo root.

**Step 2: Update example path declarations if needed**

Verify the `[[example]]` path still points to `examples/orb_slam/main.rs`.

**Step 3: Search for stale nested-crate references**

Run: `rg -n "crates/kornia-slam|../../../kornia-rs|cargo run --example" .`

Expected: no broken path references remain.

### Task 4: Verify the new layout

**Files:**
- Test: root crate build/tests/examples

**Step 1: Run tests from the repo root**

Run: `cargo test`

Expected: test suite passes from `/home/christie/projects/kornia-slam`.

**Step 2: Run a lightweight example build check**

Run: `cargo check --example orb_slam`

Expected: the demo example resolves and compiles from the new root layout.

**Step 3: Review the final diff**

Run: `git status --short`

Expected: moved crate files are visible at repo root, and the nested crate path is gone or reduced to intentional leftovers only.
