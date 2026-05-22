# ORB-SLAM Example

This package is the current runnable slice of `kornia-slam`: a monocular ORB-based SLAM pipeline with two interchangeable frame sources — offline EuRoC MAV image sequences and a live OAK-D mono camera. Both feed the same `process_frame` orchestrator, and TUI / Rerun work for both.

## Frame sources

Selectable via subcommand:

```text
orb_slam euroc --data /path/to/V1_01_easy [--start-frame N] [--max-frames N]
orb_slam oakd  [--width 640 --height 400 --fps 30] [--max-frames N]
```

`oakd` requires `--features oakd`; the default build needs no extra system dependencies.

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
  cargo run --release -p orb_slam --features oakd -- oakd --rerun-stream

# Live, TUI only (no Rerun, no lz4 clash):
cargo run --release -p orb_slam --no-default-features --features "oakd tui" -- oakd --tui
```

Intrinsics are placeholder (rough scale of the OAK-D Pro factory fx/fy at 1280×800); reading the on-device factory calibration is a TODO.

## Visualizers

Mutually exclusive flags (apply to either source):

- `--rerun-stream` — spawn a Rerun viewer and stream image / keypoints / trajectory / camera / map points (requires `--features viz`, default on).
- `--tui` — render a terminal UI with live status + bird's-eye view (requires `--features tui`).

## Local checks

```bash
cargo fmt -p orb_slam -- --check
cargo clippy -p orb_slam --all-targets -- -D warnings
cargo run -p orb_slam -- --help
```
