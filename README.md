# Audio Simulator

A physics-driven acoustic simulator written in Rust and CUDA that synthesizes sound directly from coupled solid-fluid-fracture physics—covering rigid/elastic impacts, compressible blasts, submerged resonating objects, fluid pouring and droplets, architectural acoustics, and thin-shell tearing.

## Architecture

- **`crates/sim-core`** — Core simulation engine, scene definitions, geometry builders, material models, diagnostics, units, and render preparation (`engine.rs`, `scene.rs`, `geometry.rs`, `material.rs`, `render_prep.rs`, `diagnostics.rs`).
- **`crates/sim-cuda`** — GPU-accelerated physics kernels and Rust FFI/VRAM management:
  - Acoustic, adaptive mesh refinement (AMR), compressible, and multiphase fluid solvers (`fluid_acoustic.cu`, `fluid_amr.cu`, `fluid_compressible.cu`, `fluid_multiphase.cu`)
  - Immersed-boundary fluid–structure coupling (`coupled_ib.cu`)
  - Solid MPM and thin-shell fracture solvers (`solid_mpm.cu`, `solid_shell_fracture.cu`)
  - Acoustic probe reduction (`probes_reduce.cu`)
- **`crates/sim-io`** — WAV/audio export, video muxing, OBJ mesh loading, telemetry, and render scene export (`audio.rs`, `video.rs`, `obj_loader.rs`, `render_export.rs`, `telemetry.rs`).
- **`render/blender_cycles_render.py`** — High-quality Cycles video rendering pipeline driven by exported simulation geometry and camera/scene data.

## Included Scenes (`examples/`)

| Example | Description |
| :--- | :--- |
| `rip_paper` | Slow thin-shell fracture and tearing acoustics of a high-poly paper sheet |
| `sinking_bell_grotto` | Ringing bell submerging into water inside an acoustic cavity |
| `ballistics_and_blast` | Compressible shockwave and blast sound synthesis |
| `drop_wood_vs_metal` | Material-dependent rigid impact acoustics |
| `drop_on_wet_leaves` | Droplet impacts on compliant wet surfaces |
| `rain_leaves_gallery` | Multi-drop rain scene on foliage |
| `pouring_square_jar` / `pouring_wine_glass` | Multiphase liquid pouring and container cavity resonance |
| `water_drop_plop` | Single water drop cavity/bubble resonance |
| `whispering_gallery` | Architectural acoustic focusing (`assets/hall_whisper_speech.wav`) |
| `wind_flutter_hall` | Aeroelastic flutter inside a hall |
| `render_all_videos` | Batch video generation across scenes |

## Output Standard

Every scene is designed to export:
1. **A rendered video file (with synchronized physics-generated audio)** — never audio-only — including short silent pre-roll and post-roll padding so playback devices do not clip the opening transient or decaying tail.
2. **A companion scene setup file** documenting mesh resolution, sound source coordinates/trajectories, microphone placement and directivity, camera placement, and whether the microphone is co-located with the camera.

## Building & Running

> Requires a Rust toolchain, NVIDIA CUDA Toolkit (for `sim-cuda`), FFmpeg (for video/audio muxing), and Blender (for `render/blender_cycles_render.py`).

```bash
# Run tests
cargo test --release

# Run a single scene example (e.g., paper rip)
cargo run --release --example rip_paper

# Batch render all scene videos
cargo run --release --example render_all_videos
