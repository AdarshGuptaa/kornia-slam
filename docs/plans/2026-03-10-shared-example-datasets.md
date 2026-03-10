# Shared Example Datasets Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the EuRoC dataset loader out of `examples/orb_slam` into a shared example-side datasets module that future example apps can reuse.

**Architecture:** Keep dataset support outside the public library API and create a shared `examples/common/datasets` module. Update the ORB-SLAM example to import the shared dataset module by path, preserving existing runtime behavior.

**Tech Stack:** Rust, Cargo examples, module path imports

---

### Task 1: Create shared example dataset module

**Files:**
- Create: `examples/common/datasets/mod.rs`
- Create: `examples/common/datasets/euroc.rs`
- Modify: `examples/orb_slam/main.rs`
- Delete: `examples/orb_slam/euroc.rs`

**Step 1: Write the failing integration point**

Update `examples/orb_slam/main.rs` to import `EurocDataset` from a shared datasets module path instead of the local `mod euroc;`.

**Step 2: Run the build to verify it fails**

Run: `cargo test`
Expected: build failure because the shared datasets module does not exist yet.

**Step 3: Write the minimal implementation**

Create `examples/common/datasets/mod.rs` that re-exports the EuRoC dataset types and move the current EuRoC loader implementation into `examples/common/datasets/euroc.rs`.

**Step 4: Wire the ORB-SLAM example to the shared module**

Replace the local dataset module import in `examples/orb_slam/main.rs` with a `#[path = "../common/datasets/mod.rs"]` module declaration and import `EurocDataset` from it.

**Step 5: Run tests to verify the refactor**

Run: `cargo test`
Expected: PASS, with only the existing example warnings.

**Step 6: Commit**

```bash
git add docs/plans/2026-03-10-shared-example-datasets-design.md \
        docs/plans/2026-03-10-shared-example-datasets.md \
        examples/common/datasets/mod.rs \
        examples/common/datasets/euroc.rs \
        examples/orb_slam/main.rs \
        examples/orb_slam/euroc.rs
git commit -m "refactor: share example dataset loaders"
```
