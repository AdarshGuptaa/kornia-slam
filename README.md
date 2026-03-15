# kornia-slam 📷🧭🗺️📍🤖

Spatial runtime for real-time pose estimation, mapping, and agent interaction.

> **Work in progress**

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

kornia-slam is intended to integrate with [bubbaloop](https://github.com/kornia/bubbaloop) as a spatial service within a broader robotics runtime. In that direction, kornia-slam would consume sensor streams, maintain its own map and localization state, and expose them through an MCP server. bubbaloop agents could then connect to that spatial interface as part of the surrounding system.

```
 Sensors / nodes        Zenoh pub/sub        Spatial service        MCP clients
┌──────────────┐   ┌──────────────────┐   ┌─────────────┐   ┌──────────────┐
│ cameras/IMU/ │──>│ sensor streams   │──>│ kornia-slam │<──│ bubbaloop    │
│ LiDAR/GNSS   │   │ and commands     │   │             │   │ agents/tools │
└──────────────┘   └──────────────────┘   └─────────────┘   └──────────────┘
```

- **Sensor access** — kornia-slam should be able to subscribe to camera, IMU, LiDAR, and other streams as part of the system data plane.
- **Spatial state via MCP** — kornia-slam should expose pose, map, and spatial query tools through an MCP server.
- **Agent-facing spatial tools** — bubbaloop agents should be able to connect to that MCP interface to inspect the live 3D map, query current pose and nearby structure, and invoke higher-level spatial computations such as localization, landmark lookup, geometric relationships, and map-based reasoning.

### Agentic SLAM — rethinking SLAM in the age of agents

Beyond serving spatial state to external agents, kornia-slam is exploring the idea of agents operating *within* the SLAM system itself — monitoring and improving subsystems at runtime. For example, an agent that detects degraded feature matching in low-texture scenes and switches extraction strategy, or one that tunes bundle adjustment parameters based on observed residuals. This treats SLAM not as a fixed pipeline but as a system whose components can be inspected, tuned, and swapped by agents operating in sandboxed environments.

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
- [ ] Comprehensive testing across multiple datasets (EuRoC, TUM-VI, etc.) and challenging scenarios

**Later — multi-sensor and real-time**
- [ ] IMU preintegration
- [ ] Stereo and RGB-D estimators
- [ ] Estimator fusion in odometry
- [ ] Map server (MCP)
- [ ] bubbaloop node integration
- [ ] Agentic SLAM (subsystem monitoring, strategy switching, parameter tuning)

## License

Apache-2.0
