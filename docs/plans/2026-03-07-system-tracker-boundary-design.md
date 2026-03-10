# System And Tracker Boundary Design

**Date:** 2026-03-07

## Goal

Split the current `src/backend/tracker.rs` into a clearer architectural boundary where system orchestration lives in `system.rs` and tracking-stage logic lives in `tracker.rs`.

## Current Problem

`tracker.rs` currently owns both:

- top-level orchestration: `SlamSystem`, `SystemState`, `process_frame`, bootstrap/tracking dispatch
- tracking-stage behavior: pose tracking, local map refinement, keyframe decisions, map point observation updates, culling

That makes the file read like the whole SLAM system while also being named `tracker.rs`.

## Approved Boundary

### `system.rs`

Owns the system-level state machine and public API:

- `SlamSystem`
- `SystemState`
- `TrackingResult`
- `TrackingStatus`
- `FrameRejectReason`
- `process_frame`
- `process_bootstrap`
- `process_tracking`
- reset and public accessors

This file is responsible for deciding which stage runs and for coordinating bootstrap versus tracking.

### `tracker.rs`

Owns tracking-stage helper logic only:

- `TrackRejectReason`
- `TrackAttempt`
- `try_track`
- `refine_with_local_map`
- `need_new_keyframe`
- `update_map_point_observations`
- `cull_map_points`

This file should not define the whole SLAM system.

## Rationale

- `SlamSystem` is an orchestrator, not a tracking primitive.
- Bootstrap/tracking stage transitions belong at the system level.
- Tracking-specific helper logic stays grouped without pretending to be the whole backend.

## Non-Goals

- Do not redesign the tracking pipeline itself.
- Do not change public behavior unless needed for the file split.
- Do not move two-view initialization out of `src/frontend/init.rs`.
