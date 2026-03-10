# ORB Features Boundary Design

## Context

`kornia-slam` currently defines `FrameFeatures` in `src/frame.rs`, but that struct duplicates the ORB feature payload already exposed by `kornia-imgproc` as `OrbFeatures`.

The current SLAM pipeline is ORB-specific, so the local duplicate adds API noise without providing a real abstraction boundary.

## Decision

Keep `Frame.features` as the field name on `Frame`, but change its type to `kornia_imgproc::features::OrbFeatures`.

Re-export `OrbFeatures` from `kornia-slam` so callers can still build frames ergonomically from the crate root or `frame` module.

Add a note in `src/frame.rs` making the current boundary explicit: the crate uses ORB features directly today, and a higher-level feature abstraction can be introduced later if multiple frontend algorithms need to coexist.

## Rationale

- Removes a redundant local type.
- Makes the current ORB dependency explicit instead of pretending the feature payload is generic.
- Keeps future flexibility by documenting where a higher abstraction would belong if more feature frontends arrive.
