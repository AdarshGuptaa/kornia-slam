# ORB-SLAM Example

This package is the current runnable slice of `kornia-slam`: a monocular ORB-based SLAM pipeline over EuRoC MAV image sequences with optional Rerun visualization.

Temporary dependency note: until [kornia-rs PR #803](https://github.com/kornia/kornia-rs/pull/803) is merged and released, this package depends on unpublished `kornia-rs` crates from the `feat/slam-utils` branch. Cargo fetches them automatically on first build.

## Dataset setup

Download the EuRoC MAV dataset from the OpenVINS dataset guide:
<https://docs.openvins.com/gs-datasets.html#gs-data-euroc>

The example expects the standard EuRoC directory layout:

```text
V1_01_easy/
└── mav0/
    ├── cam0/
    │   ├── data.csv
    │   └── data/
    │       ├── 1403636579763555584.png
    │       └── ...
    └── state_groundtruth_estimate0/
        └── data.csv
```

`mav0/cam0/data.csv`, `mav0/cam0/data/*.png`, and `mav0/cam0/sensor.yaml` are required to run the example. Ground truth is optional and currently only loaded by the dataset reader.

The example reads EuRoC `cam0` calibration from `sensor.yaml`. Running on another dataset requires a dataset adapter and the corresponding camera calibration.

## Run

The [Machine Hall sequences](https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f) (MH_01–MH_05) are recommended for initial testing.

```bash
cargo run --release --manifest-path examples/orb_slam/Cargo.toml -- --data /path/to/MH_01_easy
```

Useful options:

- `--max-frames 500` to limit the run length
- `--start-frame 100` to skip initial frames
- `--rerun-stream` to spawn a Rerun viewer

## Local checks

```bash
cargo fmt --manifest-path examples/orb_slam/Cargo.toml -- --check
cargo clippy --manifest-path examples/orb_slam/Cargo.toml --all-targets -- -D warnings
cargo run --manifest-path examples/orb_slam/Cargo.toml -- --help
```
