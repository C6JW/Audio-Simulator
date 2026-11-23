# Audio Simulator

A physics-first ground-truth acoustic simulator written in Rust and CUDA. Rather than relying on sample playback or procedural foley, sound is synthesized directly from coupled solid, fluid, fracture, and compressible shockwave physics sampled by virtual microphones.

## Capabilities & Solvers

- **Multiphase Fluids (`fluid_multiphase.cu`, `fluid_amr.cu`)** — Incompressible two-phase water/air VOF solver with adaptive mesh refinement (AMR), surface tension, and acoustic propagation.
- **Compressible Gas Dynamics (`fluid_compressible.cu`)** — HLLC Euler solver with sliding AMR shock windows for supersonic projectiles, Mach cones, and blast waves.
- **Coupled Fluid–Structure Interaction (`coupled_ib.cu`)** — Two-way immersed-boundary coupling between vibrating bulk/shell solids and surrounding air or water.
- **Bulk Solids & Thin-Shell Fracture (`solid_mpm.cu`, `solid_shell_fracture.cu`)** — MPM solids, discrete Kirchhoff–Love thin shells with Zener SLS viscoelasticity, AT2 phase-field fracture, and wet capillary/lubrication contact.
- **Virtual Acoustic Probes (`probes_reduce.cu`, `sim-io/audio.rs`)** — Air condenser and hydrophone pressure sampling on the fluid grid with linear-phase Kaiser FIR decimation.
- **Cycles Video & Setup Export (`sim-io/video.rs`, `render/blender_cycles_render.py`)** — Automatic Blender Cycles OptiX rendering, FFmpeg audio/video muxing with silent pre-roll/post-roll buffers, and per-scene setup sheets (`*_setup.txt`).

## Workspace Layout

~~~text
crates/
  sim-core/   # MultiphysicsEngine orchestration, Scene specs, materials, geometry, diagnostics, render prep
  sim-cuda/   # CUDA kernels (sm_120) + Rust FFI & VRAM budget tracking
  sim-io/     # WAV DSP, PLY/isosurface export, telemetry JSON/CSV, Cycles/FFmpeg muxing, setup .txt export
examples/     # Self-contained validation & showcase scenarios
render/       # Blender Cycles OptiX rendering script (blender_cycles_render.py)
assets/       # Input audio assets (e.g., whispering gallery dry speech)
~~~

## Included Scenes (`examples/`)

| Example | Physics Modeled |
| :--- | :--- |
| `rip_paper` | Slow, high-poly orthotropic cellulose thin-shell phase-field tearing |
| `sinking_bell_grotto` | High-Q bronze bell struck in air and lowered into live VOF water |
| `water_drop_plop` | Water droplet impact, cavity collapse, and Minnaert bubble resonance |
| `pouring_square_jar` / `pouring_wine_glass` | Live liquid pouring into thin glass vessels with cavity resonance |
| `ballistics_and_blast` | Supersonic projectiles, shockwaves, and canyon blast propagation |
| `drop_wood_vs_metal` | Material-dependent rigid impact acoustics (orthotropic oak vs. steel) |
| `drop_on_wet_leaves` / `rain_leaves_gallery` | Wet shell fracture, capillary adhesion, and rainfall on foliage |
| `whispering_gallery` | Curved stone enclosure acoustic focusing |
| `wind_flutter_hall` | Aeroelastic flutter driven by turbulent wind inlets |
| `render_all_videos` | Batch Cycles video + setup sheet rendering across showcase scenes |

## Output Standard

Every scene outputs synchronized **MP4 video** (not audio-only) into `output/videos/`:
- `<scene>.mp4` — High-quality Cycles OptiX video with physics-simulated audio and silent pre-roll/post-roll buffers so playback devices do not clip the opening transient or decaying tail.
- `<scene>_setup.txt` — Companion setup sheet recording mesh polycount, simulation parameters, sound source coordinates/trajectories, microphone coordinates, camera placement, and whether the microphone is co-located with the camera.
- Telemetry (`output/<scene>_telemetry.json`, `output/<scene>_timeseries.csv`) for quantitative checks.

## Prerequisites & Build

- **Rust** (2021 edition; use `--release` for real scenes)
- **NVIDIA CUDA Toolkit 13.x** (`/opt/cuda`, override with `CUDA_PATH`) and an NVIDIA GPU (`sm_120` in `crates/sim-cuda/build.rs`)
- **Blender 5.x** (Cycles OptiX) and **FFmpeg / FFprobe** on `PATH`

~~~sh
cargo build --release
cargo test --release
cargo run --release --example rip_paper
cargo run --release --example render_all_videos
~~~

## License

Licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0-or-later)**. See [LICENSE](LICENSE) for details.
