# Shared Example Datasets Design

## Context

`examples/orb_slam` currently owns the EuRoC dataset loader in `examples/orb_slam/euroc.rs`. That works for a single runnable, but it couples benchmark I/O to one algorithm-specific example.

Future example apps such as ORB-SLAM and ElasticFusion should be able to reuse the same dataset readers without pulling those readers into the public `kornia-slam` library API.

## Decision

Keep dataset support on the app/example side and move shared readers under `examples/common/datasets/`.

The first shared module will be the EuRoC loader:

- `examples/common/datasets/mod.rs`
- `examples/common/datasets/euroc.rs`

Example apps will import the shared module by path from their `main.rs`, keeping the library crate free of benchmark-specific dataset code.

## Rationale

- Datasets are operational support code for runners and benchmarks, not core SLAM library abstractions.
- Multiple example apps can share loaders without duplicating parsing code.
- This layout leaves room for future shared app-side modules such as evaluation or visualization helpers under `examples/common/`.

## Consequences

- Example binaries will need a small `#[path = "../common/datasets/mod.rs"]` module hook.
- Dataset APIs should stay generic enough to support multiple example apps, but they do not need the stability bar of the public library API.
