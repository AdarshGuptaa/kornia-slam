# oakd_slam

Live monocular ORB-SLAM on an OAK-D camera via the [`depthai`](https://crates.io/crates/depthai) Rust crate. Replaces the EuRoC PNG loop in [`orb_slam`](../orb_slam) with a depthai queue and feeds the same `Pipeline::process_frame` orchestrator.

Reuses `examples/orb_slam/src/utils.rs` for the Rerun logging helpers (path-included via `#[path = "../../orb_slam/src/utils.rs"]`).

## Prerequisites

See [`../oakd_smoke/README.md`](../oakd_smoke/README.md) — same depthai-core / libclang / udev setup applies.

The Rerun feature (`viz`, default-on) introduces one additional gotcha: both `depthai-sys` (vendoring depthai-core's static archive) and `rerun` (via `lz4_sys`) statically link **lz4** and therefore collide at link time. Until either upstream stops vendoring lz4, pass:

```text
RUSTFLAGS="-C link-arg=-Wl,--allow-multiple-definition"
```

at cargo invocation time. The linker picks one impl; lz4's ABI is stable so this is safe in practice.

## Run

With Rerun visualization:

```bash
RUSTFLAGS="-C link-arg=-Wl,--allow-multiple-definition" \
  cargo run --release -p oakd_slam -- --rerun-stream
```

Lean (no Rerun, stderr only):

```bash
RUSTFLAGS="-C link-arg=-Wl,--allow-multiple-definition" \
  cargo run --release -p oakd_slam --no-default-features
```

CLI flags:

- `--max-frames N` — stop after N frames (default 0 = run forever, Ctrl-C to stop)
- `--width W` `--height H` — depthai resizes the source to this resolution (default 640×400)
- `--fps F` — request a specific frame rate (default 30)
- `--rerun-stream` — spawn a Rerun viewer and stream image/keypoints/trajectory/camera/map_points (requires `--features viz`, on by default)

## Status

- Frame pipeline (device → mono → ORB → SLAM) works end-to-end at ~30 fps with ~9 ms per-frame cost.
- **Intrinsics are placeholder** (rough scale of the OAK-D Pro factory fx/fy at 1280×800). Real calibration via `device.read_calibration()` is a TODO.
- IMU is not exposed yet by the `depthai` Rust crate (v0.1.3).
