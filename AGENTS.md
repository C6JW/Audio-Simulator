# AGENTS.md

## Project

Ground-truth audio simulator. Physics-first renderer for real-world sound:
multiphase fluids, solids, fracture, supersonic flow, and acoustic probes.
Target is path-tracer fidelity for audio, not sample playback or procedural foley.

The current quality gap is the main work item. Improve physical fidelity and
resolution, not post-processing.

## Non-negotiables

- Output must come from simulated physics. No synthesized or random replacement
  for missing physics.
- SI units everywhere at API and scene level (see `crates/sim-core/src/units.rs`).
- Determinism: fixed seeds (`splitmix64`), no time- or hash-order-dependent behavior.
- Every new phenomenon needs a measurable check: telemetry, assertion, or test.
  Do not add "sounds plausible" changes.
- Keep GPU memory under `DEFAULT_VRAM_CAP_BYTES` (16 GB-class card; RTX 5080 here).
- `cargo test --release` must stay green.

## Layout

- `crates/sim-cuda/` CUDA kernels plus Rust wrappers. `build.rs` invokes `nvcc`
  (`-arch=sm_120`) and `ar`.
  - `fluid_multiphase.cu` incompressible two-phase water/air.
  - `fluid_compressible.cu` Euler solver and shocks.
  - `fluid_amr.cu` adaptive mesh refinement.
  - `coupled_ib.cu` immersed boundaries.
  - `solid_mpm.cu` MPM solids.
  - `solid_shell_fracture.cu` shells and tearing.
  - `probes_reduce.cu` microphone sampling and reductions.
- `crates/sim-core/` scene config, material models, engine loop, diagnostics,
  render prep.
  - `engine.rs`: `MultiphysicsEngine::run` is the pipeline. Keep it the single
    source of orchestration.
  - `scene.rs`: all scene, source, solid, and microphone specs. New features go
    here first.
- `crates/sim-io/` WAV I/O, OBJ loading, telemetry JSON/CSV, isosurface/PLY
  export, Blender/ffmpeg video muxing.
- `examples/` each scenario is a validation harness with assertions.
  `render_all_videos.rs` drives Blender and ffmpeg.
- `assets/` input audio. `output/` generated artifacts (ignored by git).

## Build and run

Requires CUDA toolkit 13.x at `/opt/cuda` (override `CUDA_PATH`), an NVIDIA GPU
with compute capability 12.0, and Blender 5.x plus ffmpeg for videos.

```sh
cargo build --release
cargo test --release
cargo run --release --example water_drop_plop
cargo run --release --example render_all_videos
```

Debug builds compile but are not usable for real scenes. Always measure with
`--release`.

Outputs land in `output/`: `<name>.wav`, `*_telemetry.json`, `*_timeseries.csv`,
and videos in `output/videos/`.

## Audio pipeline

Probe microphones sample pressure on the oversampled fluid grid
(`sample_rate * oversample_factor`), then `sim-io` decimates and resamples to
the target WAV rate. Air and hydrophone probes are supported. Telemetry records
peak pressure, spectral centroid, kurtosis, NaN count, VRAM, energies, and
fracture or bubble metrics. Use telemetry and tests to judge quality, not ears
alone.

## Conventions

- `f32` on GPU, `f64` for global accumulators and timestep math. Do not silently
  downcast.
- All timestep and CFL logic goes through `compute_adaptive_timestep` and the
  substep helpers.
- New CUDA entry points: add to `cuda_api.cu`, declare in `ffi.rs`, wrap in
  `fluid.rs` or `solid.rs`, and check errors with `check_cuda`.
- Scene defaults live in `Scene::minimal`; every scenario must set its own
  physical scales explicitly.
- Time-varying sources and fluid emitters disable CUDA graphs (see `can_graph`).
- Comments only for non-obvious physics or numerics. No dead code.

## Gotchas

- `crates/sim-cuda/build.rs` hardcodes `sm_120`. Older GPUs need that changed.
- `.cargo/config.toml` hardcodes the runtime path `/opt/cuda/lib64`.
- CUDA allocations are tracked by `VramBudgetManager`. Untracked `cudaMalloc`
  bypasses the budget.
- `output/` is large. Do not commit generated media.
- `render_all_videos` reuses cached preview PNGs in `output/videos/previews/`.
- `.codegraph/` and `.projectmem/events.jsonl` are local state, not source.
