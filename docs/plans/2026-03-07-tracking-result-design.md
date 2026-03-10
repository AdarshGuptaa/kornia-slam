# Tracking Result API Design

**Date:** 2026-03-07

## Goal

Keep `TrackingResult` focused on the outcome of processing a frame and move reporting-oriented metadata to explicit `SlamSystem` queries.

## Current Problem

`TrackingResult` currently mixes:

- core outcome fields: pose, status, reject reason
- reporting fields: keyframe index, map point count

That shape leaks internal tracker state into the per-frame result and creates an ambiguity bug: `keyframe_idx` is not consistently "reference keyframe" or "newly inserted keyframe".

## Options Considered

### Option 1: Keep the fields and document them better

This is the smallest patch, but it preserves the ambiguous API and keeps example/logging concerns in the tracker result.

### Option 2: Replace the fields with a nested diagnostics struct

This keeps reporting data available, but it still makes every caller pay for diagnostics they may not need.

### Option 3: Make `TrackingResult` outcome-only and query diagnostics from `SlamSystem`

This separates concerns cleanly:

- `TrackingResult` reports what happened to the frame
- `SlamSystem` reports current tracker/map state

This is the recommended approach.

## Approved Design

Use Option 3.

`TrackingResult` will keep:

- `pose_world_to_cam`
- `status`
- `reject_reason`

`SlamSystem` will expose explicit accessors for:

- current keyframe index
- current map point count

The `mono_vo` example and its JSON/frame-record helpers will compute reporting fields from `SlamSystem` after each `process_frame` call.

## Testing

- Add tests for the new example helper signature so frame records are built from explicit inputs instead of `TrackingResult` metadata.
- Add tests for the new `SlamSystem` accessors.
