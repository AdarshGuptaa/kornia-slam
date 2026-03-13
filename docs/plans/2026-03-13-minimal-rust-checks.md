# Minimal Rust Checks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a minimal Rust quality gate for kornia-slam with local commands and CI checks for formatting, clippy, and tests.

**Architecture:** Keep the policy surface small. Reuse the split used in `kornia-rs`: one lint workflow and one test workflow. Make the codebase pass those checks first rather than weakening the checks to fit current issues.

**Tech Stack:** Rust, Cargo, GitHub Actions, rustfmt, clippy

---

### Task 1: Define the local check baseline

**Files:**
- Modify: `README.md`

**Step 1: Document the local commands**

Add a short contributor-facing note listing:
- `cargo fmt -- --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test`

**Step 2: Keep the guidance narrow**

Do not add broader repo tooling yet. This task only documents the minimal Rust baseline.

### Task 2: Make formatting and doctests compatible with the baseline

**Files:**
- Modify: `src/estimation/map_projection.rs`
- Modify: files reported by `cargo fmt -- --check`

**Step 1: Fix the doctest failure**

Convert the ASCII-art module comment in `src/estimation/map_projection.rs` so rustdoc no longer tries to compile it as code.

**Step 2: Apply rustfmt-compatible formatting**

Run formatting and keep the changes mechanical.

### Task 3: Make clippy pass under `-D warnings`

**Files:**
- Modify: `src/frame.rs`
- Modify: `src/odometry.rs`
- Modify: `src/map.rs`
- Modify: `src/estimation/map_projection.rs`
- Modify: `src/estimation/two_view.rs`

**Step 1: Fix low-risk lint issues directly**

Address:
- empty doc comment spacing
- collapsible `if`
- `new_without_default`

**Step 2: Make intentional API shape explicit**

For the estimator helpers that currently exceed clippy’s argument-count threshold, either:
- refactor into small parameter structs if that stays clean, or
- add narrow `#[allow(...)]` annotations where the current shape is intentional and clearer than artificial indirection.

**Step 3: Handle large enum / type complexity warnings pragmatically**

Prefer a small type alias or narrow allow where it reduces lint noise without distorting the API.

### Task 4: Add minimal CI workflows

**Files:**
- Create: `.github/workflows/rust_lint.yml`
- Create: `.github/workflows/rust_test.yml`

**Step 1: Add lint workflow**

Create a small GitHub Actions workflow that runs:
- `cargo fmt -- --check`
- `cargo clippy --all-targets -- -D warnings`

**Step 2: Add test workflow**

Create a second workflow that runs:
- `cargo test`

**Step 3: Keep the workflows simple**

Do not pull in the full `kornia-rs` pixi/cache/action stack yet unless required by this repo. Use direct Cargo commands.

### Task 5: Verify the baseline end-to-end

**Files:**
- No code changes expected

**Step 1: Run the lint commands fresh**

Run:
- `cargo fmt -- --check`
- `cargo clippy --all-targets -- -D warnings`

**Step 2: Run the tests fresh**

Run:
- `cargo test`

**Step 3: Report the actual status**

If any command still fails, capture the exact blocker instead of claiming completion.
