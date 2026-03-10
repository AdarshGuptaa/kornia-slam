# SLAM Run Artifacts Design

**Date:** 2026-03-06

**Goal:** Make both `mono_vo` examples emit comparable run directories so tracking behavior can be analyzed offline frame by frame.

## Problem

The current ported example and the original `kornia-rs` example expose different amounts of runtime state. The original example already has access to matches, reject codes, initialization details, and per-frame diagnostics. The ported example mostly exposes final status, pose, keyframe index, and map size. That makes direct A/B comparison hard even when both runs use the same dataset slice.

## Decision

Add a temporary debug telemetry surface to the tracker results and have both examples write the same machine-readable run directory schema.

This is intentionally a debugging-oriented design, not a permanent public API. The extra telemetry will be grouped under a dedicated debug payload so it can be removed cleanly once the regression is understood.

## Output Artifacts

Each example run will write a directory such as `run-2026-03-06T21-00-00/` containing:

- `summary.json`
  - dataset path
  - example arguments
  - camera intrinsics
  - run start time
  - total processed frames
  - counts by `TrackingStatus`
  - counts by reject reason/code
  - final map size
  - optional evaluation metrics when available
- `frames.jsonl`
  - one JSON record per processed frame
  - frame index and timestamp
  - pose in both world-to-camera and TUM-friendly camera-to-world form
  - status and keyframe index
  - map-point count
  - feature count
  - match counts by stage
  - inlier counts by stage
  - reject reason as display text and stable code string
  - initialization diagnostics when present
  - keyframe insertion diagnostics
- `trajectory.tum`
  - estimated trajectory for the full run
- `keyframes.tum`
  - accepted keyframe poses
- `stdout.log`
  - mirrored human-readable log lines for convenience

## Shared Frame Schema

The examples should serialize the same logical schema even if some fields are `null` in one crate:

- frame identity
  - `frame_idx`
  - `timestamp_sec`
- status
  - `status`
  - `keyframe_idx`
  - `reject_reason_debug`
  - `reject_reason_code`
- pose
  - `world_to_cam.translation`
  - `world_to_cam.quaternion_xyzw`
  - `cam_to_world.translation`
  - `cam_to_world.quaternion_xyzw`
- counts
  - `feature_count`
  - `map_point_count`
  - `output_match_count`
- tracking debug
  - `projection_matches_narrow`
  - `projection_matches_wide`
  - `reference_matches`
  - `reference_correspondences`
  - `stage1_pnp_inliers`
  - `local_map_matches`
  - `local_map_inliers`
- initialization debug
  - `two_view_model`
  - `init_match_count`
  - `init_inlier_count`
  - `init_triangulated_count`
  - `median_parallax_deg`
- keyframe debug
  - `inserted`
  - `local_ba_enabled`
  - `map_points_before`
  - `map_points_after`

## Telemetry Source

### Original crate

The original crate already exposes most of the needed state through `TrackingResult`. The example will mostly derive the run artifacts from existing fields and a small amount of example-local bookkeeping.

### Ported crate

The port needs extra temporary telemetry. The cleanest path is:

- add a `DebugFrameInfo` struct in the tracker module
- attach it to `TrackingResult` as `debug: DebugFrameInfo`
- populate it inside bootstrap and tracking flows
- add stable reject-reason code helpers so the JSON schema aligns with the original crate

This avoids a permanent scatter of one-off fields while keeping the example code simple.

## Example Changes

Each example should:

- accept an explicit output directory option
- create a default timestamped run directory when the option is omitted
- stream per-frame JSON records while the run is active
- accumulate summary counters in memory
- write the summary and TUM files on completion

The examples should keep their existing console output, but also mirror it into `stdout.log`.

## Testing Strategy

Add focused tests around the run-writer helpers rather than trying to test full SLAM behavior:

- `summary.json` shape and required fields
- `frames.jsonl` record emission
- `keyframes.tum` writing
- stable reject-code formatting helpers

For the ported crate, add tracker-level tests that verify the debug payload is populated in representative bootstrap and rejected-tracking cases.

## Removal Plan

The debug payload is temporary. Once the run artifacts are no longer needed:

- remove the `DebugFrameInfo` payload from the tracker results
- remove the JSONL/summary writer helpers if they are no longer useful
- keep only the trajectory export if that still has ongoing value

The implementation should keep the temporary code isolated enough that this becomes a small cleanup patch rather than another cross-cutting migration.
