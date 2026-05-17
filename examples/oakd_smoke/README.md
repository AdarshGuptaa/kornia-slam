# oakd_smoke

Smoke test for OAK-D devices via the [`depthai`](https://crates.io/crates/depthai) Rust crate.

Opens the device, configures one CamA mono output at 640×400 / 30 fps, waits up to 10 s for the first frame, prints `(w, h)`, and exits. Used to verify that the build environment and the device are both healthy before wiring up the full SLAM example.

## Prerequisites

`depthai-sys` builds [depthai-core](https://github.com/luxonis/depthai-core) v3 from source on first compile (≈5–10 min wall, several GB of `target/`). It needs:

- `cmake` (3.20+) and a C/C++ toolchain (`gcc`/`g++` or `clang`)
- `pkg-config`
- udev rules for non-root access — see `/etc/udev/rules.d/80-movidius.rules` in the [depthai docs](https://docs.luxonis.com/projects/api/en/latest/install/)
- A working **libclang 14** (or older) so `autocxx`/bindgen can parse depthai-core headers. Clang 19+ rejects a libnop template construct used in vcpkg-installed deps; we pin libclang at the workspace level via `.cargo/config.toml`:

  ```toml
  [env]
  LIBCLANG_PATH = "/usr/lib/llvm-14/lib"
  ```

  Adjust the path for your system, or set the env var directly when running cargo.

## Run

```bash
cargo run -p oakd_smoke
```

Expected output:

```
[oakd_smoke] opening device…
[oakd_smoke] platform: Rvc2
[oakd_smoke] building pipeline…
[oakd_smoke] starting pipeline…
[oakd_smoke] waiting for first frame (timeout 10 s)…
[oakd_smoke] got frame in 0.11s: 640x400
[oakd_smoke] OK
```
