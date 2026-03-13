# kornia-slam 📷🧭🗺️📍🤖

Spatial runtime for real-time pose estimation, mapping, and agent interaction.

> **Early stage.** This README describes the long-term vision for kornia-slam, while the current implementation is a much narrower slice: monocular ORB-based odometry running end-to-end on EuRoC datasets. The current codebase is not yet a general SLAM framework: it is an ORB-specific monocular pipeline whose system orchestration still lives in the example layer rather than behind a stable library abstraction. The API, module layout, and internal abstractions are still taking shape, and broader multi-sensor SLAM, map serving, and agent integration remain roadmap work. Expect breaking changes. Contributions and feedback welcome.

kornia-slam is a modular SLAM framework that estimates poses in real time from cameras, IMU, LiDAR, and GNSS, builds a persistent map of the environment, and makes that spatial state available to agents through MCP.

## Vision

kornia-slam is taking shape as a spatial runtime within a larger robotics system. The intended direction is to consume sensor streams, estimate pose, build and maintain a persistent 3D map, and serve that spatial state to agent runtimes through an MCP-facing control layer.

### High-Level Architecture

```
 Sensors / Nodes         Zenoh data plane          Agent control plane
┌──────────────┐   pub/sub   ┌─────────────┐        ┌──────────────┐
│ Camera / IMU │────────────>│ kornia-slam │───────>│ MCP / Agents │
│ LiDAR / GNSS │             │ spatial node│        │              │
└──────────────┘             └──────┬──────┘        └──────────────┘
                                    │
                                    v
                              Persistent Map
```

### Odometry

Odometry is the real-time state estimation layer of kornia-slam. It turns incoming sensor observations into a continuous pose stream, using the map both as a source of geometric constraints and as the persistent context that keeps localization grounded over time.

```
 Visual tracking   ──┐
 Inertial updates  ──┤
 Geometric cues    ──┼──> Odometry ──> Pose stream
 Learned priors    ──┤
 Map constraints   ──┘
```

Odometry can combine multiple estimation strategies across cameras, IMU, LiDAR, and other sensors. Rather than treating each estimator as an isolated output, kornia-slam uses them as complementary sources of motion and structure that contribute to a shared spatial state.

### Map

The map is a persistent 3D representation of the environment — points, poses, and spatial relationships. Odometry and other estimators update this 3D map as they track, while the surrounding system can expose spatial queries and actions to agents through an MCP layer.

- **Odometry** reads the map to localize against known structure and writes new observations as it tracks.
- **Agents** query the spatial state through MCP — nearby landmarks, current pose, environment geometry.

### Integration with bubbaloop

kornia-slam is intended to integrate with [bubbaloop](https://github.com/kornia/bubbaloop) as a spatial service node. In that direction, bubbaloop would provide the node runtime, Zenoh data plane, and agent-facing MCP layer, while kornia-slam would provide pose estimation and mapping.

```
 Sensors / nodes        Zenoh pub/sub        Spatial service        MCP
┌──────────────┐   ┌──────────────────┐   ┌─────────────┐   ┌──────────────┐
│ bubbaloop    │──>│ sensor streams   │──>│ kornia-slam │──>│ agents/tools │
│ node runtime │   │ and commands     │   │             │   │ via bubbaloop│
└──────────────┘   └──────────────────┘   └─────────────┘   └──────────────┘
```

- **Sensors via Zenoh** — bubbaloop nodes publish camera, IMU, LiDAR, and other streams on the data plane. kornia-slam subscribes and tracks against them.
- **Spatial state via MCP** — bubbaloop's MCP layer can expose pose, map, and spatial query tools backed by kornia-slam.
- **Agent-facing spatial tools** — through bubbaloop's daemon-side agent runtime and MCP server, agents would be able to inspect the live 3D map, query current pose and nearby structure, and invoke higher-level spatial computations such as localization, landmark lookup, geometric relationships, and map-based reasoning.

## Setup

The current runnable package is the standalone ORB-SLAM example in [examples/orb_slam/README.md](examples/orb_slam/README.md).

Temporary dependency note: until [kornia-rs PR #803](https://github.com/kornia/kornia-rs/pull/803) is merged and released, this repository depends on unpublished `kornia-rs` crates from the `feat/slam-utils` branch. Cargo fetches them automatically on first build.

### Local checks

The current baseline checks for this repository are:

```bash
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

## Roadmap

**Now — monocular ORB odometry**
- [x] ORB feature extraction and matching
- [x] Two-view bootstrap (essential matrix + triangulation)
- [x] Map-projection tracking (PnP with RANSAC)
- [x] Keyframe insertion and map point triangulation
- [x] Local bundle adjustment
- [x] Map point culling

**Next — complete monocular SLAM ideas**
- [ ] Rich keyframe and map-point observation model
- [ ] Covisibility graph and local map maintenance
- [ ] Neighbor search, landmark fusion, and redundant keyframe culling
- [ ] Relocalization on tracking loss
- [ ] Loop closing (place recognition + pose graph optimization)

**After that — robustness and evaluation**
- [ ] Match a strong monocular ORB baseline on trajectory quality and tracking robustness
- [ ] Trajectory evaluation (ATE/RPE against ground truth)

**Later — multi-sensor and real-time**
- [ ] Stereo and RGB-D estimators
- [ ] IMU preintegration estimator
- [ ] Estimator fusion in odometry
- [ ] Zenoh sensor integration
- [ ] bubbaloop node integration
- [ ] Map server (MCP)
- [ ] LiDAR ICP estimator

## License

Apache-2.0
