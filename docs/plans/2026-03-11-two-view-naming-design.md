# Two-View Naming Design

**Date:** 2026-03-11

## Goal

Make the public surface of `src/odometry/estimation/two_view.rs` consistently algorithm-oriented.

The current module name is `two_view`, but its public API still uses phase-oriented bootstrap names:

- `BootstrapConfig`
- `BootstrapOutcome`
- `BootstrapRejectReason`
- `try_bootstrap`

This should be renamed so the module and its exported symbols describe the same concept.

## Design

### Naming

Rename the public surface as follows:

- `BootstrapConfig` -> `TwoViewInitConfig`
- `BootstrapOutcome` -> `TwoViewInitOutcome`
- `BootstrapRejectReason` -> `TwoViewInitRejectReason`
- `try_bootstrap` -> `try_initialize_two_view`

### Config split

Keep `kornia_3d::pose::TwoViewConfig` as the upstream geometry-estimation config.

Add a local acceptance-policy struct:

- `TwoViewAcceptanceConfig`

The new top-level config becomes:

```rust
pub struct TwoViewInitConfig {
    pub match_config: OrbMatchConfig,
    pub estimation_config: kornia_3d::pose::TwoViewConfig,
    pub acceptance_config: TwoViewAcceptanceConfig,
}
```

And:

```rust
pub struct TwoViewAcceptanceConfig {
    pub min_matches: usize,
    pub min_inliers: usize,
    pub min_triangulated: usize,
}
```

`min_parallax_deg` should not be duplicated in the local acceptance struct, because it already belongs in `kornia_3d::pose::TwoViewConfig`.

### Rationale

This keeps the boundary honest:

- `kornia-3d` owns generic two-view estimation settings
- `kornia-slam` owns ORB matching and SLAM-stage acceptance policy

The result is a clearer API without pushing SLAM-specific gating into `kornia-3d`.

## Non-goals

This change should not alter:

- the two-view math
- initialization acceptance behavior
- default threshold values
- ORB matching behavior

It is a naming and struct-shape cleanup only.

## Testing

Verification should be:

- `cargo test`

If public API tests or example imports break, update them as part of the rename.
