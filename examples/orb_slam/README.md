# ORB-SLAM Example

This package is the current runnable slice of `kornia-slam`: a monocular ORB-based SLAM pipeline with three interchangeable frame sources — offline EuRoC MAV image sequences, a live OAK-D mono camera, and any UVC-class camera (laptop webcams, USB cams, CSI-to-UVC adapters on a Pi…). All three feed the same `process_frame` orchestrator, and the TUI / Rerun visualizers work for any of them.

## Frame sources

Selectable via subcommand:

```text
orb_slam euroc --data /path/to/V1_01_easy [--start-frame N] [--max-frames N]
orb_slam oakd  [--width 640 --height 400 --fps 30] [--max-frames N]
orb_slam uvc   --fx F --fy F --cx C --cy C [--index 0] [--width 640 --height 480] [--max-frames N]
```

`oakd` requires `--features oakd`; `uvc` requires `--features uvc`. The default build needs no extra system dependencies.

## EuRoC dataset

Download the EuRoC MAV dataset from the OpenVINS dataset guide:
<https://docs.openvins.com/gs-datasets.html#gs-data-euroc>

Standard directory layout:

```text
V1_01_easy/
└── mav0/
    ├── cam0/
    │   ├── data.csv
    │   ├── sensor.yaml
    │   └── data/
    │       ├── 1403636579763555584.png
    │       └── ...
    └── state_groundtruth_estimate0/
        └── data.csv
```

`mav0/cam0/{data.csv,sensor.yaml}` and the PNGs under `data/` are required. Ground truth is optional and only parsed by the dataset reader.

[Machine Hall sequences](https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f) (MH_01–MH_05) are recommended for initial testing.

```bash
cargo run --release -p orb_slam -- euroc --data /path/to/MH_01_easy
```

## OAK-D camera

Build prerequisites (`depthai-sys` builds [depthai-core](https://github.com/luxonis/depthai-core) v3 from source on first compile — ~5–10 min wall, several GB of `target/`):

- `cmake` (3.20+) and a C/C++ toolchain (`gcc`/`g++` or `clang`)
- `pkg-config`
- udev rules for non-root device access — see `/etc/udev/rules.d/80-movidius.rules` in the [depthai docs](https://docs.luxonis.com/projects/api/en/latest/install/)
- **libclang 14** (or older) so `autocxx`/bindgen can parse depthai-core headers. Clang 19+ rejects a libnop template construct used in vcpkg-installed deps; libclang is pinned at the workspace level via `.cargo/config.toml`:

  ```toml
  [env]
  LIBCLANG_PATH = "/usr/lib/llvm-14/lib"
  ```

  Adjust for your system, or set the env var when invoking cargo.

The Rerun feature (`viz`, default-on) collides with `depthai-sys`'s vendored lz4 at link time. Until either upstream stops vendoring lz4, pass:

```text
RUSTFLAGS="-C link-arg=-Wl,--allow-multiple-definition"
```

at cargo invocation time when building with both `viz` and `oakd`.

```bash
# Live, with Rerun visualization:
RUSTFLAGS="-C link-arg=-Wl,--allow-multiple-definition" \
  cargo run --release -p orb_slam --features oakd -- --rerun-stream oakd

# Live, TUI only (no Rerun, no lz4 clash):
cargo run --release -p orb_slam --no-default-features --features oakd -- oakd
```

Intrinsics are placeholder (rough scale of the OAK-D Pro factory fx/fy at 1280×800); reading the on-device factory calibration is a TODO.

## UVC camera

Any UVC-class device works (built-in laptop webcam, USB camera, CSI-to-UVC adapter on a Raspberry Pi). Unlike EuRoC and OAK-D, there's no on-device calibration, so you must pass intrinsics on the command line — they have to match the resolution the device actually streams at (nokhwa picks the closest supported mode if the exact one is missing).

```bash
# /dev/video0 at 640x480, rough pinhole calibration:
cargo run --release -p orb_slam --features uvc -- \
    uvc --index 0 --fx 600 --fy 600 --cx 320 --cy 240
```

## Visualizers

The TUI is the default — just run the example. Override with one of:

- `--rerun-stream` — spawn a Rerun viewer and stream image / keypoints / trajectory / camera / map points (requires `--features viz`, default on). Disables the TUI.
- `--no-tui` — fall back to plain stderr status lines (no TUI, no Rerun).
- `--debug` — show the debug panel inside the TUI (or extra diagnostic lines on stderr in `--no-tui` mode). Toggle live with the `d` key while the TUI is running.

## Local checks

```bash
cargo fmt -p orb_slam -- --check
cargo clippy -p orb_slam --all-targets -- -D warnings
cargo run -p orb_slam -- --help
```
