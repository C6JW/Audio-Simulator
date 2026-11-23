use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use sim_cuda::ffi::*;
use sim_cuda::{
    device_sync, CudaError, CudaGraphExec, CudaStream, DeviceBuffer, FluidLevelDevice,
    FluidLevelSpec, MaterialParamsGpu, MicProbeGpu, MpmGridDevice, MpmGridGpu, MpmGridSpec,
    MpmMetricsGpu, MpmParticlesDevice, MpmParticlesHost, ShellMeshDevice, ShellMeshHost,
    SimReduceMetricsGpu, VramBudgetManager, DEFAULT_VRAM_CAP_BYTES,
};
use sim_io::{
    mux_realtime_video_with_audio, mux_video_with_audio, prepare_probe_wav_signal_f64,
    render_frames_with_blender, write_probe_wav_f32_with_gain, write_stereo_wav_f32,
    write_telemetry_json, write_timeseries_csv, RunTelemetry, TimeSeriesRow, VideoRenderOptions,
};
use thiserror::Error;

use crate::diagnostics::{analyze_probe_signal, analyze_subsurface_bubble};
use crate::geometry::{
    generate_shell_uvs, normalize3, sample_bulk_solid_into, sample_shell_solid_into, GeometryShape,
};
use crate::render_prep::PreparedRenderExporter;
use crate::scene::Scene;
use crate::units::AIR_C;

/// Format a duration in seconds into a concise human-readable string (`14.2s`, `4m 08s`, `1h 23m 05s`).
pub fn format_duration_hms(secs: f64) -> String {
    if !secs.is_finite() || secs <= 0.0 {
        return "0.0s".to_string();
    }
    if secs < 60.0 {
        return format!("{secs:.1}s");
    }
    let total = secs.round() as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}h {m:02}m {s:02}s")
    } else {
        format!("{m}m {s:02}s")
    }
}

fn is_audio_only_override() -> bool {
    for var in ["AUDIO_SIM_NO_VIDEO", "AUDIO_SIM_AUDIO_ONLY"] {
        if let Ok(val) = std::env::var(var) {
            let v = val.trim().to_ascii_lowercase();
            if v == "1" || v == "true" || v == "yes" {
                return true;
            }
        }
    }
    std::env::args().any(|a| a == "--audio-only" || a == "--no-video")
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn rand_unit_f32(seed: u64, step: u64) -> f32 {
    let h = splitmix64(seed ^ step.wrapping_mul(0xD1B5_4A32_D192_ED03));
    ((h >> 40) as f32) / ((1u32 << 24) as f32)
}

#[inline]
fn empty_mpm_grid_gpu(has_floor: i32, floor_z: f32, floor_mu: f32) -> MpmGridGpu {
    MpmGridGpu {
        nx: 0,
        ny: 0,
        nz: 0,
        dx: 0.01,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
        has_floor,
        floor_z,
        floor_mu,
        _pad: 0.0,
        _pad2: 0.0,
        mass: std::ptr::null_mut(),
        vx: std::ptr::null_mut(),
        vy: std::ptr::null_mut(),
        vz: std::ptr::null_mut(),
        fx: std::ptr::null_mut(),
        fy: std::ptr::null_mut(),
        fz: std::ptr::null_mut(),
        bc_w: std::ptr::null_mut(),
        bc_vx: std::ptr::null_mut(),
        bc_vy: std::ptr::null_mut(),
        bc_vz: std::ptr::null_mut(),
        d_field: std::ptr::null_mut(),
        d_new: std::ptr::null_mut(),
        h_field: std::ptr::null_mut(),
        gc_field: std::ptr::null_mut(),
        lc_field: std::ptr::null_mut(),
        eta_field: std::ptr::null_mut(),
        w_field: std::ptr::null_mut(),
    }
}

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Audio IO error: {0}")]
    Audio(#[from] sim_io::AudioIoError),
    #[error("Render export error: {0}")]
    RenderExport(#[from] sim_io::RenderExportError),
    #[error("Video pipeline error: {0}")]
    Video(#[from] sim_io::VideoPipelineError),
    #[error("Invalid scene configuration: {0}")]
    InvalidScene(String),
}

pub type Result<T> = std::result::Result<T, EngineError>;

#[derive(Clone, Debug)]
pub struct RenderExportConfig {
    pub frames_dir: PathBuf,
    pub num_frames: usize,
    pub export_fluid_isosurface: bool,
    pub export_pressure_slice: bool,
    pub pressure_slice_z: f32,
}

#[derive(Clone, Debug)]
pub struct SimRunConfig {
    pub duration_s: f64,
    pub pre_roll_s: f64,
    pub sample_rate: u32,
    pub oversample_factor: usize,
    pub target_wav_sample_rate: Option<u32>,
    pub shared_wav_gain: bool,
    pub phase_field_stride_steps: usize,
    pub telemetry_interval_s: f64,
    pub output_dir: Option<PathBuf>,
    pub water_surface_z_for_diag: Option<f32>,
    pub use_cuda_graph: bool,
    pub cfl_safety: f64,
    pub print_progress: bool,
    pub render_export: Option<RenderExportConfig>,
    /// Automatically export scene frames, render with Blender Cycles, and mux an MP4 video
    /// into `<output_dir>/videos/<scenario>.mp4` whenever `output_dir` is `Some(...)` and
    /// `render_export` is not explicitly configured. Defaults to `true`.
    /// Disable with `auto_video: false`, `--audio-only` / `--no-video`, or `AUDIO_SIM_NO_VIDEO=1`.
    pub auto_video: bool,
    /// Optional Blender / ffmpeg settings override used when `auto_video` runs.
    pub video_options: Option<VideoRenderOptions>,
}

impl Default for SimRunConfig {
    fn default() -> Self {
        Self {
            duration_s: 0.01,
            pre_roll_s: 0.0,
            sample_rate: 48_000,
            oversample_factor: 4,
            target_wav_sample_rate: None,
            shared_wav_gain: false,
            phase_field_stride_steps: 1,
            telemetry_interval_s: 0.001,
            output_dir: None,
            water_surface_z_for_diag: None,
            use_cuda_graph: true,
            cfl_safety: 0.38,
            print_progress: false,
            render_export: None,
            auto_video: true,
            video_options: None,
        }
    }
}

pub struct SimRunResult {
    pub telemetry: RunTelemetry,
    pub timeseries: Vec<TimeSeriesRow>,
    pub raw_probe_buffers: Vec<Vec<f32>>,
    pub video_path: Option<PathBuf>,
}

pub struct MultiphysicsEngine {
    pub scene: Scene,
    pub vram: VramBudgetManager,
    pub fluid_const: FluidConstantsGpu,
    pub levels: Vec<FluidLevelDevice>,
    pub level_parents: Vec<usize>,
    pub sliding_level_idx: Option<usize>,
    pub d_levels: DeviceBuffer<FluidLevelGpu>,
    pub d_probes: DeviceBuffer<MicProbeGpu>,
    pub d_metrics: DeviceBuffer<SimReduceMetricsGpu>,
    pub d_proj_force: DeviceBuffer<f64>,
    pub particles: Option<MpmParticlesDevice>,
    pub mpm_grid: Option<MpmGridDevice>,
    pub shell_mesh: Option<ShellMeshDevice>,
    pub shell_host: Option<ShellMeshHost>,
    pub p_rest: Vec<[f32; 3]>,
    pub p_rest_mat: Vec<i32>,
    pub p_rest_uv: Vec<[f32; 2]>,
    pub d_mats: Option<DeviceBuffer<MaterialParamsGpu>>,
    pub d_mpm_metrics: DeviceBuffer<MpmMetricsGpu>,
    pub has_dynamic_solids: bool,
    pub has_shells: bool,
    pub has_bulk_dynamic_solids: bool,
    pub has_fracture: bool,
    pub has_contact: bool,
    pub max_solid_wave_speed: f64,
    pub min_shell_edge: f64,
    pub min_shell_bend_dt: f64,
    pub solid_friction_mu: f32,
    pub stream: CudaStream,
    pub initial_water_volume: f64,
    pub initial_total_energy: f64,
}

impl MultiphysicsEngine {
    pub fn new(scene: Scene, vram_cap_bytes: Option<usize>) -> Result<Self> {
        let cap = vram_cap_bytes.unwrap_or(DEFAULT_VRAM_CAP_BYTES);
        let vram = VramBudgetManager::new(0, cap)?;
        let stream = CudaStream::new()?;
        let fluid_const = scene.fluid.to_gpu();

        if scene.amr.nx0 == 0 || scene.amr.ny0 == 0 || scene.amr.nz0 == 0 {
            return Err(EngineError::InvalidScene(
                "Level 0 dimensions must be positive".to_string(),
            ));
        }

        let any_dynamic_solids = scene
            .solids
            .iter()
            .any(|b| b.is_kinematic_rigid || !b.material.is_rigid_static);
        let is_euler = scene.enable_compressible_euler;
        let alloc_amr_old = !is_euler && !scene.amr.patches.is_empty();

        // Build AMR hierarchy specs (supporting nested chains, sibling multi-patches, and sliding shock windows)
        let mut specs = Vec::with_capacity(1 + scene.amr.patches.len() + usize::from(scene.sliding_patch.is_some()));
        let mut level_parents = Vec::with_capacity(1 + scene.amr.patches.len() + 1);
        specs.push(FluidLevelSpec {
            nx: scene.amr.nx0,
            ny: scene.amr.ny0,
            nz: scene.amr.nz0,
            dx: scene.amr.dx0,
            ox: scene.amr.origin[0],
            oy: scene.amr.origin[1],
            oz: scene.amr.origin[2],
            level_idx: 0,
            refine_ratio: 1,
            pml_thickness: scene.amr.pml_thickness,
            pml_face_mask: scene.amr.pml_face_mask,
            off_x: 0,
            off_y: 0,
            off_z: 0,
            alloc_flow: !is_euler && scene.enable_flow_advection,
            alloc_amr_old,
            alloc_ib_vel: any_dynamic_solids,
            alloc_compressible: is_euler,
        });
        level_parents.push(0);

        for (idx, p) in scene.amr.patches.iter().enumerate() {
            let parent_idx = p.parent_level.unwrap_or(idx);
            if parent_idx >= specs.len() {
                return Err(EngineError::InvalidScene(format!(
                    "AMR level {} invalid parent_level {}",
                    idx + 1,
                    parent_idx
                )));
            }
            let parent = &specs[parent_idx];
            if p.refine_ratio < 2
                || p.nx % p.refine_ratio != 0
                || p.ny % p.refine_ratio != 0
                || p.nz % p.refine_ratio != 0
            {
                return Err(EngineError::InvalidScene(format!(
                    "AMR level {} dims ({},{},{}) must be divisible by refine_ratio {}",
                    idx + 1,
                    p.nx,
                    p.ny,
                    p.nz,
                    p.refine_ratio
                )));
            }
            let nx_c = p.nx / p.refine_ratio;
            let ny_c = p.ny / p.refine_ratio;
            let nz_c = p.nz / p.refine_ratio;
            if p.off_x + nx_c > parent.nx
                || p.off_y + ny_c > parent.ny
                || p.off_z + nz_c > parent.nz
            {
                return Err(EngineError::InvalidScene(format!(
                    "AMR level {} patch exceeds parent level {} bounds",
                    idx + 1,
                    parent_idx
                )));
            }
            let dx = parent.dx / (p.refine_ratio as f32);
            let ox = parent.ox + (p.off_x as f32) * parent.dx;
            let oy = parent.oy + (p.off_y as f32) * parent.dx;
            let oz = parent.oz + (p.off_z as f32) * parent.dx;
            let p_flow = p.enable_flow.unwrap_or(scene.enable_flow_advection);
            specs.push(FluidLevelSpec {
                nx: p.nx,
                ny: p.ny,
                nz: p.nz,
                dx,
                ox,
                oy,
                oz,
                level_idx: idx + 1,
                refine_ratio: p.refine_ratio,
                pml_thickness: 0,
                pml_face_mask: 0,
                off_x: p.off_x,
                off_y: p.off_y,
                off_z: p.off_z,
                alloc_flow: !is_euler && p_flow,
                alloc_amr_old,
                alloc_ib_vel: any_dynamic_solids,
                alloc_compressible: is_euler,
            });
            level_parents.push(parent_idx);
        }

        let mut sliding_level_idx: Option<usize> = None;
        if is_euler {
            if let Some(ref sp) = scene.sliding_patch {
                let p0 = &specs[0];
                if sp.refine_ratio < 2
                    || sp.nx % sp.refine_ratio != 0
                    || sp.ny % sp.refine_ratio != 0
                    || sp.nz % sp.refine_ratio != 0
                {
                    return Err(EngineError::InvalidScene(
                        "Sliding AMR patch dims must be divisible by refine_ratio".to_string(),
                    ));
                }
                let off_x = ((sp.initial_ox - p0.ox) / p0.dx).round().max(0.0) as usize;
                let off_y = ((sp.oy - p0.oy) / p0.dx).round().max(0.0) as usize;
                let off_z = ((sp.oz - p0.oz) / p0.dx).round().max(0.0) as usize;
                let sli = specs.len();
                specs.push(FluidLevelSpec {
                    nx: sp.nx,
                    ny: sp.ny,
                    nz: sp.nz,
                    dx: p0.dx / (sp.refine_ratio as f32),
                    ox: p0.ox + (off_x as f32) * p0.dx,
                    oy: p0.oy + (off_y as f32) * p0.dx,
                    oz: p0.oz + (off_z as f32) * p0.dx,
                    level_idx: sli,
                    refine_ratio: sp.refine_ratio,
                    pml_thickness: 0,
                    pml_face_mask: 0,
                    off_x,
                    off_y,
                    off_z,
                    alloc_flow: false,
                    alloc_amr_old: false,
                    alloc_ib_vel: any_dynamic_solids,
                    alloc_compressible: true,
                });
                level_parents.push(0);
                sliding_level_idx = Some(sli);
            }
        }

        let mut levels = Vec::with_capacity(specs.len());
        for spec in specs {
            levels.push(FluidLevelDevice::new(&vram, spec)?);
        }

        if is_euler {
            // Compressible HLLC Euler initialization + direct static geometry IB rasterization
            for lvl in &mut levels {
                let d_lvl = lvl.descriptor();
                unsafe {
                    sim_cuda::check_cuda(
                        sim_euler_init_ambient(d_lvl, fluid_const, stream.raw()),
                        "sim_euler_init_ambient",
                    )?;
                }
            }
            let static_shapes: Vec<&GeometryShape> = scene
                .solids
                .iter()
                .filter(|b| b.material.is_rigid_static && !b.is_kinematic_rigid)
                .map(|b| &b.shape)
                .collect();
            if !static_shapes.is_empty() {
                for lvl in &mut levels {
                    let nxp = lvl.nxp;
                    let nyp = lvl.nyp;
                    let nzp = lvl.nzp;
                    let h = lvl.halo as isize;
                    let dx = lvl.spec.dx;
                    let ox = lvl.spec.ox;
                    let oy = lvl.spec.oy;
                    let oz = lvl.spec.oz;
                    let mut ib_static = vec![0.0_f32; nxp * nyp * nzp];
                    for shape in &static_shapes {
                        let (bmin, bmax) = shape.bounding_box();
                        let pad = 2.0 * dx;
                        let i_lo = (((bmin[0] - pad - ox) / dx).floor() as isize + h)
                            .clamp(0, nxp as isize - 1) as usize;
                        let i_hi = (((bmax[0] + pad - ox) / dx).ceil() as isize + h)
                            .clamp(0, nxp as isize - 1) as usize;
                        let j_lo = (((bmin[1] - pad - oy) / dx).floor() as isize + h)
                            .clamp(0, nyp as isize - 1) as usize;
                        let j_hi = (((bmax[1] + pad - oy) / dx).ceil() as isize + h)
                            .clamp(0, nyp as isize - 1) as usize;
                        let k_lo = (((bmin[2] - pad - oz) / dx).floor() as isize + h)
                            .clamp(0, nzp as isize - 1) as usize;
                        let k_hi = (((bmax[2] + pad - oz) / dx).ceil() as isize + h)
                            .clamp(0, nzp as isize - 1) as usize;
                        for k in k_lo..=k_hi {
                            let z = oz + ((k as isize - h) as f32 + 0.5) * dx;
                            for j in j_lo..=j_hi {
                                let y = oy + ((j as isize - h) as f32 + 0.5) * dx;
                                for i in i_lo..=i_hi {
                                    let x = ox + ((i as isize - h) as f32 + 0.5) * dx;
                                    let f = shape.subcell_volume_fraction([x, y, z], dx);
                                    if f > 0.0 {
                                        let cid = i + nxp * (j + nyp * k);
                                        ib_static[cid] = ib_static[cid].max(f).clamp(0.0, 1.0);
                                    }
                                }
                            }
                        }
                    }
                    lvl.cell_ib_static.as_mut().unwrap().upload(&ib_static)?;
                    lvl.cell_ib.as_mut().unwrap().upload(&ib_static)?;
                }
            }
            stream.sync()?;
        } else {
        // 1. Initialize background water alpha & hydrostatic equilibrium on each level
        for lvl_idx in 0..levels.len() {
            let nxp = levels[lvl_idx].nxp;
            let nyp = levels[lvl_idx].nyp;
            let nzp = levels[lvl_idx].nzp;
            let halo = levels[lvl_idx].halo;
            let dx = levels[lvl_idx].spec.dx;
            let ox = levels[lvl_idx].spec.ox;
            let oy = levels[lvl_idx].spec.oy;
            let oz = levels[lvl_idx].spec.oz;

            let mut alpha_host = vec![0.0_f32; nxp * nyp * nzp];
            if !scene.background_water.is_empty() {
                for k in 0..nzp {
                    let z = oz + ((k as isize - halo as isize) as f32 + 0.5) * dx;
                    for j in 0..nyp {
                        let y = oy + ((j as isize - halo as isize) as f32 + 0.5) * dx;
                        for i in 0..nxp {
                            let x = ox + ((i as isize - halo as isize) as f32 + 0.5) * dx;
                            let mut a = 0.0_f32;
                            for shape in &scene.background_water {
                                a = a.max(shape.subcell_volume_fraction([x, y, z], dx));
                            }
                            alpha_host[i + nxp * (j + nyp * k)] = a;
                        }
                    }
                }
            }
            levels[lvl_idx].alpha.as_mut().unwrap().upload(&alpha_host)?;

            if lvl_idx == 0 {
                let d_lvl0 = levels[0].descriptor();
                unsafe {
                    sim_cuda::check_cuda(
                        sim_fluid_init_hydrostatic_root(
                            d_lvl0,
                            fluid_const,
                            scene.p_top_ref,
                            stream.raw(),
                        ),
                        "sim_fluid_init_hydrostatic_root",
                    )?;
                }
            } else {
                let p_idx = level_parents[lvl_idx];
                let d_parent = levels[p_idx].descriptor();
                let d_child = levels[lvl_idx].descriptor();
                unsafe {
                    sim_cuda::check_cuda(
                        sim_fluid_init_hydrostatic_child(
                            d_child,
                            d_parent,
                            fluid_const,
                            stream.raw(),
                        ),
                        "sim_fluid_init_hydrostatic_child",
                    )?;
                }
            }
        }
        stream.sync()?;

        // 2. Apply dynamic fluid regions (e.g., falling droplets) & velocities & Laplace overpressure
        if !scene.dynamic_fluid.is_empty() {
            for lvl in &mut levels {
                let nxp = lvl.nxp;
                let nyp = lvl.nyp;
                let nzp = lvl.nzp;
                let halo = lvl.halo;
                let dx = lvl.spec.dx;
                let ox = lvl.spec.ox;
                let oy = lvl.spec.oy;
                let oz = lvl.spec.oz;

                let mut alpha_host = lvl.alpha.as_ref().unwrap().to_vec()?;
                let mut ux_host = lvl.ux.as_ref().unwrap().to_vec()?;
                let mut uy_host = lvl.uy.as_ref().unwrap().to_vec()?;
                let mut uz_host = lvl.uz.as_ref().unwrap().to_vec()?;

                for reg in &scene.dynamic_fluid {
                    let (bmin, bmax) = reg.shape.bounding_box();
                    let pad = 2.0 * dx;
                    let i_lo = (((bmin[0] - pad - ox) / dx).floor() as isize + halo as isize)
                        .clamp(0, nxp as isize - 1) as usize;
                    let i_hi = (((bmax[0] + pad - ox) / dx).ceil() as isize + halo as isize)
                        .clamp(0, nxp as isize - 1) as usize;
                    let j_lo = (((bmin[1] - pad - oy) / dx).floor() as isize + halo as isize)
                        .clamp(0, nyp as isize - 1) as usize;
                    let j_hi = (((bmax[1] + pad - oy) / dx).ceil() as isize + halo as isize)
                        .clamp(0, nyp as isize - 1) as usize;
                    let k_lo = (((bmin[2] - pad - oz) / dx).floor() as isize + halo as isize)
                        .clamp(0, nzp as isize - 1) as usize;
                    let k_hi = (((bmax[2] + pad - oz) / dx).ceil() as isize + halo as isize)
                        .clamp(0, nzp as isize - 1) as usize;

                    for k in k_lo..=k_hi {
                        let z = oz + ((k as isize - halo as isize) as f32 + 0.5) * dx;
                        for j in j_lo..=j_hi {
                            let y = oy + ((j as isize - halo as isize) as f32 + 0.5) * dx;
                            for i in i_lo..=i_hi {
                                let x = ox + ((i as isize - halo as isize) as f32 + 0.5) * dx;
                                let frac = reg.shape.subcell_volume_fraction([x, y, z], dx);
                                if frac > 0.0 {
                                    let cid = i + nxp * (j + nyp * k);
                                    if reg.is_water {
                                        alpha_host[cid] = (alpha_host[cid] + frac).min(1.0);
                                    } else {
                                        alpha_host[cid] = (alpha_host[cid] * (1.0 - frac)).max(0.0);
                                    }
                                    // Set adjacent face velocities cleanly when cell has phase > 0.01
                                    if frac > 0.01 {
                                        let fx0 = i + (nxp + 1) * (j + nyp * k);
                                        let fx1 = (i + 1) + (nxp + 1) * (j + nyp * k);
                                        let fy0 = i + nxp * (j + (nyp + 1) * k);
                                        let fy1 = i + nxp * ((j + 1) + (nyp + 1) * k);
                                        let fz0 = i + nxp * (j + nyp * k);
                                        let fz1 = i + nxp * (j + nyp * (k + 1));
                                        ux_host[fx0] = reg.velocity[0];
                                        ux_host[fx1] = reg.velocity[0];
                                        uy_host[fy0] = reg.velocity[1];
                                        uy_host[fy1] = reg.velocity[1];
                                        uz_host[fz0] = reg.velocity[2];
                                        uz_host[fz1] = reg.velocity[2];
                                    }
                                }
                            }
                        }
                    }
                }

                lvl.alpha.as_mut().unwrap().upload(&alpha_host)?;
                lvl.ux.as_mut().unwrap().upload(&ux_host)?;
                lvl.uy.as_mut().unwrap().upload(&uy_host)?;
                lvl.uz.as_mut().unwrap().upload(&uz_host)?;

                for reg in &scene.dynamic_fluid {
                    if reg.laplace_overpressure_pa != 0.0 {
                        if let GeometryShape::Sphere { center, radius } = &reg.shape {
                            let d_lvl = lvl.descriptor();
                            unsafe {
                                sim_cuda::check_cuda(
                                    sim_fluid_add_sphere_laplace(
                                        d_lvl,
                                        center[0],
                                        center[1],
                                        center[2],
                                        *radius,
                                        reg.laplace_overpressure_pa,
                                        stream.raw(),
                                    ),
                                    "sim_fluid_add_sphere_laplace",
                                )?;
                            }
                        }
                    }
                }
            }
            stream.sync()?;
        }

        // 3. Inject initial Gaussian acoustic pulses
        for pulse in &scene.gaussian_pulses {
            for lvl in &mut levels {
                let d_lvl = lvl.descriptor();
                unsafe {
                    sim_cuda::check_cuda(
                        sim_fluid_inject_gaussian_pulse(
                            d_lvl,
                            pulse.center[0],
                            pulse.center[1],
                            pulse.center[2],
                            pulse.sigma_r,
                            pulse.amplitude_pa,
                            stream.raw(),
                        ),
                        "sim_fluid_inject_gaussian_pulse",
                    )?;
                }
            }
        }

        // 4. Sync AMR boundary ghosts & compute initial slow CSF/viscous forces
        {
            let has_amr_children = levels.len() > 1;
            let d_lvl0 = levels[0].descriptor();
            unsafe {
                sim_cuda::check_cuda(
                    sim_fluid_apply_outer_bc(d_lvl0, stream.raw()),
                    "sim_fluid_apply_outer_bc",
                )?;
                if has_amr_children {
                    sim_cuda::check_cuda(
                        sim_fluid_save_old(d_lvl0, stream.raw()),
                        "sim_fluid_save_old",
                    )?;
                }
            }
            for idx in 1..levels.len() {
                let p_idx = level_parents[idx];
                let d_parent = levels[p_idx].descriptor();
                let d_child = levels[idx].descriptor();
                unsafe {
                    sim_cuda::check_cuda(
                        sim_fluid_amr_fill_ghosts(d_child, d_parent, 1.0, 1, stream.raw()),
                        "sim_fluid_amr_fill_ghosts",
                    )?;
                    sim_cuda::check_cuda(
                        sim_fluid_save_old(d_child, stream.raw()),
                        "sim_fluid_save_old",
                    )?;
                }
            }
            for lvl in &mut levels {
                if lvl.spec.alloc_flow {
                    let d_lvl = lvl.descriptor();
                    unsafe {
                        sim_cuda::check_cuda(
                            sim_fluid_update_slow_forces_only(d_lvl, fluid_const, stream.raw()),
                            "sim_fluid_update_slow_forces_only",
                        )?;
                    }
                }
            }
        }
        stream.sync()?;
        }

        // 5. Upload level descriptors & probe descriptors
        let level_descs: Vec<FluidLevelGpu> = levels.iter_mut().map(|l| l.descriptor()).collect();
        let d_levels = vram.alloc_from_slice(&level_descs)?;
        let d_proj_force = vram.alloc_zeros::<f64>(3)?;

        let mut probes_gpu = Vec::with_capacity(scene.microphones.len());
        for m in &scene.microphones {
            let mut best_lvl = 0i32;
            for (li, lvl) in levels.iter().enumerate() {
                let s = &lvl.spec;
                let margin = s.dx * 1.5;
                let x_hi = s.ox + (s.nx as f32) * s.dx;
                let y_hi = s.oy + (s.ny as f32) * s.dx;
                let z_hi = s.oz + (s.nz as f32) * s.dx;
                if m.position[0] >= s.ox + margin
                    && m.position[0] <= x_hi - margin
                    && m.position[1] >= s.oy + margin
                    && m.position[1] <= y_hi - margin
                    && m.position[2] >= s.oz + margin
                    && m.position[2] <= z_hi - margin
                {
                    best_lvl = li as i32;
                }
            }
            probes_gpu.push(MicProbeGpu {
                x: m.position[0],
                y: m.position[1],
                z: m.position[2],
                level_idx: best_lvl,
                is_hydrophone: if m.is_hydrophone { 1 } else { 0 },
                _pad: [0.0; 3],
            });
        }
        let d_probes = vram.alloc_from_slice(&probes_gpu)?;
        let d_metrics = vram.alloc_zeros::<SimReduceMetricsGpu>(1)?;
        let d_mpm_metrics = vram.alloc_zeros::<MpmMetricsGpu>(1)?;

        // 6. Initialize solid MPM particles, shells, background MPM grid, and Immersed Boundary coupling
        let mut particles: Option<MpmParticlesDevice> = None;
        let mut mpm_grid: Option<MpmGridDevice> = None;
        let mut shell_mesh: Option<ShellMeshDevice> = None;
        let mut d_mats: Option<DeviceBuffer<MaterialParamsGpu>> = None;
        let mut has_dynamic_solids = false;
        let mut has_shells = false;
        let mut has_bulk_dynamic_solids = false;
        let mut has_fracture = false;
        let mut has_contact = false;
        let mut max_solid_wave_speed = 0.0_f64;
        let mut min_shell_edge = f64::INFINITY;
        let mut min_shell_bend_dt = f64::INFINITY;
        let mut solid_friction_mu = 0.35_f32;

        let mut shell_host_saved: Option<ShellMeshHost> = None;
        let mut p_rest: Vec<[f32; 3]> = Vec::new();
        let mut p_rest_mat: Vec<i32> = Vec::new();
        let mut p_rest_uv: Vec<[f32; 2]> = Vec::new();

        if !scene.solids.is_empty() {
            let mut mats_gpu = Vec::with_capacity(scene.solids.len());
            let mut p_host = MpmParticlesHost::default();
            let mut static_ib_bodies: Vec<usize> = Vec::new();
            let mut shell_host = ShellMeshHost::default();
            let mut q0_list = Vec::new();
            let mut uv_list: Vec<[f32; 2]> = Vec::new();
            let mut body_ids = std::collections::BTreeSet::new();

            for (idx, body) in scene.solids.iter().enumerate() {
                mats_gpu.push(body.material.to_gpu());
                if !body.material.is_rigid_static || body.is_kinematic_rigid {
                    body_ids.insert(body.body_id);
                }
                if body.is_kinematic_rigid {
                    has_dynamic_solids = true;
                    solid_friction_mu = solid_friction_mu.max(body.material.friction_mu);
                } else if !body.material.is_rigid_static {
                    has_dynamic_solids = true;
                    max_solid_wave_speed =
                        max_solid_wave_speed.max(body.material.max_wave_speed() as f64);
                    solid_friction_mu = solid_friction_mu.max(body.material.friction_mu);
                    if body.material.is_shell {
                        has_shells = true;
                    } else {
                        has_bulk_dynamic_solids = true;
                    }
                } else if body.material.is_shell {
                    has_shells = true;
                }
                if body.material.g_c > 0.0 {
                    has_fracture = true;
                }
                let is_static_only = body.material.is_rigid_static && !body.is_kinematic_rigid;
                let skip_sample = is_euler && is_static_only;
                let separate_static_ib = !is_euler && is_static_only && !body.material.is_shell;
                if separate_static_ib {
                    static_ib_bodies.push(idx);
                } else if !skip_sample {
                    let p_before = p_host.len();
                    if body.material.is_shell {
                        sample_shell_solid_into(
                            body,
                            idx as i32,
                            &mut p_host,
                            &mut q0_list,
                            &mut shell_host,
                        );
                        if let GeometryShape::CurvedShell { nx, ny, .. } = &body.shape {
                            uv_list.extend(generate_shell_uvs(*nx, *ny));
                        } else {
                            uv_list.resize(p_host.len(), [0.0, 0.0]);
                        }
                    } else {
                        sample_bulk_solid_into(body, idx as i32, &mut p_host, &mut q0_list);
                        let added = p_host.len().saturating_sub(p_before);
                        uv_list.resize(uv_list.len() + added, [0.0, 0.0]);
                    }
                }
            }

            // Analytical subcell face-area IB baking for static rigid bodies in non-Euler mode:
            // eliminates particle-splat staircasing on curved acoustic walls (e.g. Whispering Gallery)
            // and guarantees watertight boundaries with zero thin-wall leakage.
            if !static_ib_bodies.is_empty() {
                for lvl in &mut levels {
                    let nxp = lvl.nxp;
                    let nyp = lvl.nyp;
                    let nzp = lvl.nzp;
                    let halo = lvl.halo;
                    let dx = lvl.spec.dx;
                    let ox = lvl.spec.ox;
                    let oy = lvl.spec.oy;
                    let oz = lvl.spec.oz;

                    let mut bx_h = vec![0.0_f32; (nxp + 1) * nyp * nzp];
                    let mut by_h = vec![0.0_f32; nxp * (nyp + 1) * nzp];
                    let mut bz_h = vec![0.0_f32; nxp * nyp * (nzp + 1)];

                    for &bidx in &static_ib_bodies {
                        let shape = &scene.solids[bidx].shape;
                        let (bmin, bmax) = shape.bounding_box();
                        let pad = 1.5 * dx;

                        let ix0 = (((bmin[0] - pad - ox) / dx).floor() as isize + halo as isize)
                            .clamp(0, nxp as isize) as usize;
                        let ix1 = (((bmax[0] + pad - ox) / dx).ceil() as isize + halo as isize)
                            .clamp(0, nxp as isize) as usize;
                        let iy0 = (((bmin[1] - pad - oy) / dx).floor() as isize + halo as isize)
                            .clamp(0, nyp as isize) as usize;
                        let iy1 = (((bmax[1] + pad - oy) / dx).ceil() as isize + halo as isize)
                            .clamp(0, nyp as isize) as usize;
                        let iz0 = (((bmin[2] - pad - oz) / dx).floor() as isize + halo as isize)
                            .clamp(0, nzp as isize) as usize;
                        let iz1 = (((bmax[2] + pad - oz) / dx).ceil() as isize + halo as isize)
                            .clamp(0, nzp as isize) as usize;

                        let cx1 = ix1.min(nxp - 1);
                        let cy1 = iy1.min(nyp - 1);
                        let cz1 = iz1.min(nzp - 1);

                        // X-faces: (nxp + 1) x nyp x nzp
                        for k in iz0..=cz1 {
                            let z = oz + ((k as isize - halo as isize) as f32 + 0.5) * dx;
                            for j in iy0..=cy1 {
                                let y = oy + ((j as isize - halo as isize) as f32 + 0.5) * dx;
                                let row = (nxp + 1) * (j + nyp * k);
                                for i in ix0..=ix1 {
                                    let x = ox + (i as isize - halo as isize) as f32 * dx;
                                    let f = shape.subcell_volume_fraction([x, y, z], dx);
                                    if f > 0.0 {
                                        let slot = &mut bx_h[i + row];
                                        if f > *slot {
                                            *slot = f;
                                        }
                                    }
                                }
                            }
                        }

                        // Y-faces: nxp x (nyp + 1) x nzp
                        for k in iz0..=cz1 {
                            let z = oz + ((k as isize - halo as isize) as f32 + 0.5) * dx;
                            for j in iy0..=iy1 {
                                let y = oy + (j as isize - halo as isize) as f32 * dx;
                                let row = nxp * (j + (nyp + 1) * k);
                                for i in ix0..=cx1 {
                                    let x = ox + ((i as isize - halo as isize) as f32 + 0.5) * dx;
                                    let f = shape.subcell_volume_fraction([x, y, z], dx);
                                    if f > 0.0 {
                                        let slot = &mut by_h[i + row];
                                        if f > *slot {
                                            *slot = f;
                                        }
                                    }
                                }
                            }
                        }

                        // Z-faces: nxp x nyp x (nzp + 1)
                        for k in iz0..=iz1 {
                            let z = oz + (k as isize - halo as isize) as f32 * dx;
                            for j in iy0..=cy1 {
                                let y = oy + ((j as isize - halo as isize) as f32 + 0.5) * dx;
                                let row = nxp * (j + nyp * k);
                                for i in ix0..=cx1 {
                                    let x = ox + ((i as isize - halo as isize) as f32 + 0.5) * dx;
                                    let f = shape.subcell_volume_fraction([x, y, z], dx);
                                    if f > 0.0 {
                                        let slot = &mut bz_h[i + row];
                                        if f > *slot {
                                            *slot = f;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if let Some(ref mut buf) = lvl.ib_beta_x {
                        buf.upload(&bx_h)?;
                    }
                    if let Some(ref mut buf) = lvl.ib_beta_static_x {
                        buf.upload(&bx_h)?;
                    }
                    if let Some(ref mut buf) = lvl.ib_beta_y {
                        buf.upload(&by_h)?;
                    }
                    if let Some(ref mut buf) = lvl.ib_beta_static_y {
                        buf.upload(&by_h)?;
                    }
                    if let Some(ref mut buf) = lvl.ib_beta_z {
                        buf.upload(&bz_h)?;
                    }
                    if let Some(ref mut buf) = lvl.ib_beta_static_z {
                        buf.upload(&bz_h)?;
                    }
                }
            }

            if !p_host.is_empty() {
                has_contact = has_shells && body_ids.len() > 1;
                let needs_mpm_grid = has_bulk_dynamic_solids || has_fracture;
                if has_dynamic_solids && (needs_mpm_grid || scene.mpm_grid.is_some()) {
                    let gcfg = scene.mpm_grid.as_ref().ok_or_else(|| {
                        EngineError::InvalidScene(
                            "Dynamic bulk or fracturing solid bodies require scene.mpm_grid to be configured"
                                .to_string(),
                        )
                    })?;
                    if !has_shells {
                        p_host.reorder_color_disjoint(&mut q0_list, gcfg.origin, gcfg.dx);
                    }
                    mpm_grid = Some(MpmGridDevice::new(
                        &vram,
                        MpmGridSpec {
                            nx: gcfg.nx,
                            ny: gcfg.ny,
                            nz: gcfg.nz,
                            dx: gcfg.dx,
                            ox: gcfg.origin[0],
                            oy: gcfg.origin[1],
                            oz: gcfg.origin[2],
                            has_floor: gcfg.has_floor,
                            floor_z: gcfg.floor_z,
                            floor_mu: gcfg.floor_mu,
                        },
                        has_fracture,
                    )?);
                }

                p_rest = (0..p_host.len())
                    .map(|pi| [p_host.px[pi], p_host.py[pi], p_host.pz[pi]])
                    .collect();
                p_rest_mat = p_host.mat_id.clone();
                if has_shells {
                    p_rest_uv = uv_list;
                    shell_host_saved = Some(shell_host.clone());
                    for t in 0..shell_host.tri_v0.len() {
                        let mat_idx = shell_host.tri_mat_id[t] as usize;
                        if scene.solids[mat_idx].material.is_rigid_static {
                            continue;
                        }
                        let v0 = shell_host.tri_v0[t] as usize;
                        let v1 = shell_host.tri_v1[t] as usize;
                        let v2 = shell_host.tri_v2[t] as usize;
                        let all_fixed = (p_host.flags[v0] & 3u32) != 0
                            && (p_host.flags[v1] & 3u32) != 0
                            && (p_host.flags[v2] & 3u32) != 0;
                        if !all_fixed {
                            let h_tri = (2.0 * shell_host.tri_area0[t] as f64).sqrt().max(0.0015);
                            min_shell_edge = min_shell_edge.min(h_tri);
                        }
                    }
                    // Flexural bending wave CFL limit: dt_bend = 0.35 * l^2 / sqrt(D / (rho * h))
                    for h_idx in 0..shell_host.hinge_e0.len() {
                        let mat_idx = shell_host.hinge_mat_id[h_idx] as usize;
                        let mat = &scene.solids[mat_idx].material;
                        if mat.is_rigid_static {
                            continue;
                        }
                        let e0 = shell_host.hinge_e0[h_idx] as usize;
                        let e1 = shell_host.hinge_e1[h_idx] as usize;
                        let o0 = shell_host.hinge_o0[h_idx] as usize;
                        let o1 = shell_host.hinge_o1[h_idx] as usize;
                        let fixed_opp = (p_host.flags[o0] & 3u32) != 0 && (p_host.flags[o1] & 3u32) != 0;
                        if !fixed_opp {
                            let dx = (p_host.px[e1] - p_host.px[e0]) as f64;
                            let dy = (p_host.py[e1] - p_host.py[e0]) as f64;
                            let dz = (p_host.pz[e1] - p_host.pz[e0]) as f64;
                            let l_e = (dx * dx + dy * dy + dz * dz).sqrt().max(0.0015);
                            let e_eff = 0.5 * (mat.e1 as f64 + mat.e2 as f64);
                            let nu = (mat.nu12 as f64).clamp(0.0, 0.45);
                            let h_th = (mat.thickness as f64).max(1e-5);
                            let rho = (mat.rho as f64).max(1.0);
                            let alpha_b = h_th * (e_eff / (12.0 * rho * (1.0 - nu * nu))).sqrt();
                            if alpha_b > 1e-12 {
                                let dt_b = 0.35 * l_e * l_e / alpha_b;
                                min_shell_bend_dt = min_shell_bend_dt.min(dt_b);
                            }
                        }
                    }
                    let mut max_r_eff = 0.001_f32;
                    for p in 0..p_host.len() {
                        let r_eff = if (p_host.flags[p] & 8u32) != 0 {
                            (0.5 * p_host.thickness[p]).max(0.42 * p_host.area[p].sqrt())
                        } else {
                            0.55 * p_host.vol0[p].cbrt()
                        };
                        max_r_eff = max_r_eff.max(r_eff);
                    }
                    let cell_size = (3.0 * max_r_eff).clamp(0.002, 0.015);
                    shell_mesh = Some(ShellMeshDevice::from_host(
                        &vram,
                        &shell_host,
                        p_host.len(),
                        cell_size,
                    )?);
                }

                p_host.finalize_soa(&q0_list);
                let mut p_dev = MpmParticlesDevice::from_host(&vram, &p_host)?;
                let m_dev = vram.alloc_from_slice(&mats_gpu)?;

                // Splat initial IB blocking & surface velocities onto all fluid levels
                let d_pts = p_dev.descriptor();
                for lvl in &mut levels {
                    let d_lvl = lvl.descriptor();
                    unsafe {
                        if is_euler {
                            sim_cuda::check_cuda(
                                sim_euler_ib_splat_level(d_pts, d_lvl, stream.raw()),
                                "sim_euler_ib_splat_level",
                            )?;
                        } else {
                            sim_cuda::check_cuda(
                                sim_ib_splat_level(d_pts, d_lvl, stream.raw()),
                                "sim_ib_splat_level",
                            )?;
                        }
                    }
                }
                stream.sync()?;

                particles = Some(p_dev);
                d_mats = Some(m_dev);
            }
        }

        let mut engine = Self {
            scene,
            vram,
            fluid_const,
            levels,
            level_parents,
            sliding_level_idx,
            d_levels,
            d_probes,
            d_metrics,
            d_proj_force,
            particles,
            mpm_grid,
            shell_mesh,
            shell_host: shell_host_saved,
            p_rest,
            p_rest_mat,
            p_rest_uv,
            d_mats,
            d_mpm_metrics,
            has_dynamic_solids,
            has_shells,
            has_bulk_dynamic_solids,
            has_fracture,
            has_contact,
            max_solid_wave_speed,
            min_shell_edge,
            min_shell_bend_dt,
            solid_friction_mu,
            stream,
            initial_water_volume: 0.0,
            initial_total_energy: 0.0,
        };
        let init_m = engine.reduce_level_metrics(0)?;
        let init_sm = engine.reduce_mpm_metrics()?;
        engine.initial_water_volume = init_m.water_volume;
        engine.initial_total_energy = init_m.acoustic_energy
            + init_m.kinetic_energy
            + init_sm.kinetic_energy
            + init_sm.elastic_energy;
        Ok(engine)
    }

    pub fn reduce_mpm_metrics(&mut self) -> Result<MpmMetricsGpu> {
        if let (Some(ref mut pts), Some(ref mats)) = (&mut self.particles, &self.d_mats) {
            let d_pts = pts.descriptor();
            unsafe {
                sim_cuda::check_cuda(
                    sim_mpm_reduce_energy(
                        d_pts,
                        mats.as_ptr(),
                        self.d_mpm_metrics.as_mut_ptr(),
                        self.stream.raw(),
                    ),
                    "sim_mpm_reduce_energy",
                )?;
            }
            self.stream.sync()?;
            let out = self.d_mpm_metrics.to_vec()?;
            Ok(out[0])
        } else {
            Ok(MpmMetricsGpu::default())
        }
    }

    pub fn reduce_level_metrics(&mut self, level_idx: usize) -> Result<SimReduceMetricsGpu> {
        let is_comp = self.levels[level_idx].spec.alloc_compressible;
        let d_lvl = self.levels[level_idx].descriptor();
        unsafe {
            if is_comp {
                sim_cuda::check_cuda(
                    sim_euler_reduce_metrics(
                        d_lvl,
                        self.fluid_const,
                        self.d_metrics.as_mut_ptr(),
                        self.stream.raw(),
                    ),
                    "sim_euler_reduce_metrics",
                )?;
            } else {
                sim_cuda::check_cuda(
                    sim_fluid_reduce_metrics(
                        d_lvl,
                        self.fluid_const,
                        self.d_metrics.as_mut_ptr(),
                        self.stream.raw(),
                    ),
                    "sim_fluid_reduce_metrics",
                )?;
            }
        }
        self.stream.sync()?;
        let out = self.d_metrics.to_vec()?;
        Ok(out[0])
    }

    fn build_amr_children(&self) -> Vec<Vec<usize>> {
        let mut children = vec![Vec::new(); self.levels.len()];
        for li in 1..self.levels.len() {
            if Some(li) == self.sliding_level_idx {
                continue;
            }
            let p = self.level_parents[li];
            children[p].push(li);
        }
        children
    }

    #[inline]
    fn compute_solid_dt_substeps(&self, dt_step: f64, mpm_dx_fallback: f64) -> (usize, f32) {
        if !self.has_dynamic_solids {
            return (0, 0.0);
        }
        let solid_h = mpm_dx_fallback.min(self.min_shell_edge);
        let dt_mem_lim = 0.25 * solid_h / self.max_solid_wave_speed.max(1.0);
        let dt_s_lim = if self.min_shell_bend_dt.is_finite() {
            dt_mem_lim.min(self.min_shell_bend_dt.max(0.25 * dt_mem_lim))
        } else {
            dt_mem_lim
        };
        let n = (dt_step / dt_s_lim).ceil().max(1.0) as usize;
        (n, (dt_step / (n as f64)) as f32)
    }

    /// Compute decoupled CFL-safe timesteps `dt_ac_0`, `k0`, `t_refine[li]`, `flow_substeps[li]`, `mult_from_0[li]`.
    fn compute_adaptive_timestep(
        &mut self,
        cfg: &SimRunConfig,
    ) -> Result<(f64, usize, f64, f64, Vec<usize>, Vec<usize>, Vec<usize>)> {
        let dt_probe = 1.0 / ((cfg.sample_rate as f64) * (cfg.oversample_factor as f64));
        let mut c_max_global = AIR_C;
        let mut u_max_global = 0.0_f64;
        let mut dt_lim = vec![f64::INFINITY; self.levels.len()];
        let has_water_injection =
            !self.scene.fluid_inflows.is_empty() || !self.scene.droplet_emitters.is_empty();

        for li in 0..self.levels.len() {
            let m = self.reduce_level_metrics(li)?;
            if m.nan_count > 0 {
                return Err(EngineError::InvalidScene(format!(
                    "NaN detected on level {} ({} cells)",
                    li, m.nan_count
                )));
            }
            let mut c_max_lvl = m.c_max_sq.sqrt().max(AIR_C) * 1.02;
            if self.levels[li].spec.alloc_flow && has_water_injection {
                c_max_lvl = c_max_lvl.max(self.fluid_const.c_w * 1.02);
            }
            let u_max_lvl = m.u_max_sq.sqrt();
            c_max_global = c_max_global.max(c_max_lvl);
            u_max_global = u_max_global.max(u_max_lvl);

            let dx = self.levels[li].spec.dx as f64;
            // 4th-order staggered FDTD stability bound is 1 / (sqrt(3) * 7/6) = 0.4949.
            let eff_cfl = cfg.cfl_safety.min(0.44);
            dt_lim[li] = eff_cfl * dx / (c_max_lvl + u_max_lvl);
        }

        let k0 = (dt_probe / dt_lim[0]).ceil().max(1.0) as usize;
        let dt_ac_0 = dt_probe / (k0 as f64);

        let mut t_refine = vec![1usize; self.levels.len()];
        let mut dt_actual = vec![dt_ac_0; self.levels.len()];
        let mut mult_from_0 = vec![k0; self.levels.len()];
        for li in 1..self.levels.len() {
            let p = self.level_parents[li];
            let r = self.levels[li].spec.refine_ratio;
            let tr = (dt_actual[p] / dt_lim[li]).ceil().max(r as f64) as usize;
            t_refine[li] = tr;
            dt_actual[li] = dt_actual[p] / (tr as f64);
            mult_from_0[li] = mult_from_0[p] * tr;
        }

        let mut flow_substeps = vec![1usize; self.levels.len()];
        let nu_air = self.fluid_const.mu_a / self.fluid_const.rho_a;
        for li in 0..self.levels.len() {
            if !self.levels[li].spec.alloc_flow {
                continue;
            }
            let dx = self.levels[li].spec.dx as f64;
            let dt_u = 0.25 * dx / u_max_global.max(1.0);
            let dt_sig = 0.25
                * ((self.fluid_const.rho_a + self.fluid_const.rho_w) * dx.powi(3)
                    / (4.0 * std::f64::consts::PI * self.fluid_const.sigma.max(1e-6)))
                .sqrt();
            let dt_nu = 0.20 * (dx * dx) / (6.0 * nu_air);
            let dt_flow_lim = dt_u.min(dt_sig).min(dt_nu);
            let n_sub_max = (dt_flow_lim / dt_actual[li]).floor().max(1.0) as usize;
            let cap = n_sub_max.min(16).max(1);
            let mut best = 1usize;
            for d in 1..=cap {
                if mult_from_0[li] % d == 0 {
                    best = d;
                }
            }
            flow_substeps[li] = best;
        }

        Ok((
            dt_ac_0,
            k0,
            c_max_global,
            u_max_global,
            flow_substeps,
            t_refine,
            mult_from_0,
        ))
    }

    /// Recursively advance fluid AMR level `li` by `dt_lvl` with decoupled temporal refinement.
    #[allow(clippy::too_many_arguments)]
    fn launch_advance_level(
        levels: &mut [FluidLevelDevice],
        children: &[Vec<usize>],
        t_refine: &[usize],
        fluid_const: FluidConstantsGpu,
        flow_substeps: &[usize],
        step_counters: &mut [u64],
        li: usize,
        dt_lvl: f64,
        stream: &CudaStream,
    ) -> Result<()> {
        let has_children = !children[li].is_empty();
        let d_lvl = levels[li].descriptor();

        if has_children {
            unsafe {
                sim_cuda::check_cuda(
                    sim_fluid_save_old(d_lvl, stream.raw()),
                    "sim_fluid_save_old",
                )?;
            }
        }

        unsafe {
            sim_cuda::check_cuda(
                sim_fluid_acoustic_step(d_lvl, fluid_const, dt_lvl as f32, stream.raw()),
                "sim_fluid_acoustic_step",
            )?;
        }

        for &child_li in &children[li] {
            let tr = t_refine[child_li].max(1);
            let dt_child = dt_lvl / (tr as f64);
            let d_child = levels[child_li].descriptor();
            unsafe {
                sim_cuda::check_cuda(
                    sim_fluid_amr_reset_fluxes(d_child, stream.raw()),
                    "sim_fluid_amr_reset_fluxes",
                )?;
            }
            for m in 0..tr {
                let d_parent = levels[li].descriptor();
                let d_child = levels[child_li].descriptor();
                let theta = (m as f64) / (tr as f64);
                let fill_alpha = if m == 0 { 1 } else { 0 };
                unsafe {
                    sim_cuda::check_cuda(
                        sim_fluid_amr_fill_ghosts(
                            d_child,
                            d_parent,
                            theta,
                            fill_alpha,
                            stream.raw(),
                        ),
                        "sim_fluid_amr_fill_ghosts",
                    )?;
                }
                Self::launch_advance_level(
                    levels,
                    children,
                    t_refine,
                    fluid_const,
                    flow_substeps,
                    step_counters,
                    child_li,
                    dt_child,
                    stream,
                )?;
            }
            let d_parent = levels[li].descriptor();
            let d_child = levels[child_li].descriptor();
            unsafe {
                sim_cuda::check_cuda(
                    sim_fluid_amr_reflux_and_restrict(
                        d_child,
                        d_parent,
                        fluid_const,
                        dt_lvl as f32,
                        stream.raw(),
                    ),
                    "sim_fluid_amr_reflux_and_restrict",
                )?;
            }
        }

        step_counters[li] += 1;
        if levels[li].spec.alloc_flow && (step_counters[li] % (flow_substeps[li] as u64) == 0) {
            let dt_flow = (dt_lvl * (flow_substeps[li] as f64)) as f32;
            let parity = (step_counters[li] / (flow_substeps[li] as u64)) as i32;
            let d_lvl = levels[li].descriptor();
            unsafe {
                sim_cuda::check_cuda(
                    sim_fluid_flow_step(d_lvl, fluid_const, dt_flow, parity, stream.raw()),
                    "sim_fluid_flow_step",
                )?;
            }
            for &child_li in &children[li] {
                let d_parent = levels[li].descriptor();
                let d_child = levels[child_li].descriptor();
                unsafe {
                    sim_cuda::check_cuda(
                        sim_fluid_amr_restrict_only(
                            d_child,
                            d_parent,
                            fluid_const,
                            stream.raw(),
                        ),
                        "sim_fluid_amr_restrict_only",
                    )?;
                }
            }
        }

        Ok(())
    }

    #[inline]
    fn level_max_p(m: &SimReduceMetricsGpu) -> f64 {
        if m.max_p > -1e20 && m.min_p < 1e20 {
            m.min_p.abs().max(m.max_p.abs())
        } else {
            0.0
        }
    }

    fn dump_level_anomaly(&self, li: usize, t_now: f64) -> Result<()> {
        let lvl = &self.levels[li];
        if lvl.spec.alloc_compressible {
            return Ok(());
        }
        let s = &lvl.spec;
        let h = lvl.halo;
        let nxp = lvl.nxp;
        let nyp = lvl.nyp;
        let nzp = lvl.nzp;
        let p_g = lvl.p_gauge.as_ref().unwrap().to_vec()?;
        let p_h = lvl.p_hydro.as_ref().unwrap().to_vec()?;
        let ux = lvl.ux.as_ref().unwrap().to_vec()?;
        let uy = lvl.uy.as_ref().unwrap().to_vec()?;
        let uz = lvl.uz.as_ref().unwrap().to_vec()?;
        let alpha = lvl.alpha.as_ref().unwrap().to_vec()?;
        let asx = lvl
            .a_slow_x
            .as_ref()
            .map(|b| b.to_vec())
            .transpose()?
            .unwrap_or_default();
        let asy = lvl
            .a_slow_y
            .as_ref()
            .map(|b| b.to_vec())
            .transpose()?
            .unwrap_or_default();
        let asz = lvl
            .a_slow_z
            .as_ref()
            .map(|b| b.to_vec())
            .transpose()?
            .unwrap_or_default();
        let kappa = lvl
            .kappa
            .as_ref()
            .map(|b| b.to_vec())
            .transpose()?
            .unwrap_or_default();

        let mut first_nan: Option<(usize, usize, usize)> = None;
        let mut max_p = (-1.0_f64, 0usize, 0usize, 0usize);
        let mut max_u = (-1.0_f64, 0usize, 0usize, 0usize);

        for kp in 0..nzp {
            for jp in 0..nyp {
                for ip in 0..nxp {
                    let cid = ip + nxp * (jp + nyp * kp);
                    let fxi = ip + (nxp + 1) * (jp + nyp * kp);
                    let fyi = ip + nxp * (jp + (nyp + 1) * kp);
                    let fzi = ip + nxp * (jp + nyp * kp);
                    let pg = p_g[cid];
                    let vx = ux[fxi];
                    let vy = uy[fyi];
                    let vz = uz[fzi];
                    let al = alpha[cid];
                    let ax = asx.get(fxi).copied().unwrap_or(0.0);
                    let ay = asy.get(fyi).copied().unwrap_or(0.0);
                    let az = asz.get(fzi).copied().unwrap_or(0.0);
                    let ka = kappa.get(cid).copied().unwrap_or(0.0);
                    if !pg.is_finite()
                        || !vx.is_finite()
                        || !vy.is_finite()
                        || !vz.is_finite()
                        || !al.is_finite()
                        || !ax.is_finite()
                        || !ay.is_finite()
                        || !az.is_finite()
                        || !ka.is_finite()
                    {
                        if first_nan.is_none() {
                            first_nan = Some((ip, jp, kp));
                        }
                    } else {
                        let pa = pg.abs() as f64;
                        if pa > max_p.0 {
                            max_p = (pa, ip, jp, kp);
                        }
                        let um = (vx * vx + vy * vy + vz * vz).sqrt() as f64;
                        if um > max_u.0 {
                            max_u = (um, ip, jp, kp);
                        }
                    }
                }
            }
        }

        let describe = |tag: &str, ip: usize, jp: usize, kp: usize| {
            let cid = ip + nxp * (jp + nyp * kp);
            let fxi = ip + (nxp + 1) * (jp + nyp * kp);
            let fyi = ip + nxp * (jp + (nyp + 1) * kp);
            let fzi = ip + nxp * (jp + nyp * kp);
            let ii = (ip as isize) - (h as isize);
            let ji = (jp as isize) - (h as isize);
            let ki = (kp as isize) - (h as isize);
            let is_int = ii >= 0
                && ii < s.nx as isize
                && ji >= 0
                && ji < s.ny as isize
                && ki >= 0
                && ki < s.nz as isize;
            let region = if is_int { "INT" } else { "GHOST" };
            let x = s.ox + (ii as f32 + 0.5) * s.dx;
            let y = s.oy + (ji as f32 + 0.5) * s.dx;
            let z = s.oz + (ki as f32 + 0.5) * s.dx;
            println!(
                "  [anomaly-detail L{} {}({}) @ t={:.6}s] pad=({},{},{}) int=({},{},{}) xyz=({:.5},{:.5},{:.5})m | alpha={:.4} p_g={:.4e} p_h={:.4e} u=({:.3e},{:.3e},{:.3e}) a_s=({:.3e},{:.3e},{:.3e}) kappa={:.3e}",
                li,
                tag,
                region,
                t_now,
                ip,
                jp,
                kp,
                ii,
                ji,
                ki,
                x,
                y,
                z,
                alpha[cid],
                p_g[cid],
                p_h[cid],
                ux[fxi],
                uy[fyi],
                uz[fzi],
                asx.get(fxi).copied().unwrap_or(0.0),
                asy.get(fyi).copied().unwrap_or(0.0),
                asz.get(fzi).copied().unwrap_or(0.0),
                kappa.get(cid).copied().unwrap_or(0.0)
            );
        };
        if let Some((ip, jp, kp)) = first_nan {
            describe("FIRST_NAN", ip, jp, kp);
        }
        describe("MAX_P", max_p.1, max_p.2, max_p.3);
        describe("MAX_U", max_u.1, max_u.2, max_u.3);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_coupled_step(
        &mut self,
        children: &[Vec<usize>],
        t_refine: &[usize],
        dt_ac_0: f64,
        n_mpm: usize,
        dt_s: f32,
        flow_substeps: &[usize],
        step_counters: &mut [u64],
        run_phase_field_inside: bool,
    ) -> Result<()> {
        self.step_dynamic_solids_only(dt_ac_0, n_mpm, dt_s, run_phase_field_inside, false, None)?;

        Self::launch_advance_level(
            &mut self.levels,
            children,
            t_refine,
            self.fluid_const,
            flow_substeps,
            step_counters,
            0,
            dt_ac_0,
            &self.stream,
        )
    }

    fn sync_d_levels(&mut self, sliding_active: bool) -> Result<()> {
        let sli = self.sliding_level_idx;
        let mut descs: Vec<FluidLevelGpu> = self.levels.iter_mut().map(|l| l.descriptor()).collect();
        if let Some(si) = sli {
            if !sliding_active {
                descs[si].ox = -1.0e9;
            }
        }
        self.d_levels.upload(&descs)?;
        Ok(())
    }

    fn update_kinematics(
        &mut self,
        t_now: f64,
        dt_probe: f64,
        kinematic_pivots: &mut [[f32; 3]],
    ) -> Result<()> {
        if self.scene.kinematic_trajectories.is_empty() {
            return Ok(());
        }
        if let Some(ref mut pts) = self.particles {
            let d_pts = pts.descriptor();
            for (ti, traj) in self.scene.kinematic_trajectories.iter().enumerate() {
                let (v, w) = traj.velocity_at(t_now);
                let piv = &mut kinematic_pivots[ti];
                unsafe {
                    sim_cuda::check_cuda(
                        sim_mpm_set_body_prescribed_vel(
                            d_pts,
                            traj.body_id,
                            v[0],
                            v[1],
                            v[2],
                            w[0],
                            w[1],
                            w[2],
                            piv[0],
                            piv[1],
                            piv[2],
                            self.stream.raw(),
                        ),
                        "sim_mpm_set_body_prescribed_vel",
                    )?;
                }
                piv[0] += v[0] * (dt_probe as f32);
                piv[1] += v[1] * (dt_probe as f32);
                piv[2] += v[2] * (dt_probe as f32);
            }
        }
        Ok(())
    }

    fn step_dynamic_solids_only(
        &mut self,
        dt_step: f64,
        n_mpm: usize,
        dt_s: f32,
        run_phase_field_inside: bool,
        is_euler: bool,
        active_levels: Option<&[bool]>,
    ) -> Result<()> {
        if !self.has_dynamic_solids {
            return Ok(());
        }
        let d_pts = self.particles.as_mut().unwrap().descriptor();
        let d_grid = if let Some(ref mut mg) = self.mpm_grid {
            mg.descriptor()
        } else {
            let (hf, fz, fmu) = self
                .scene
                .mpm_grid
                .as_ref()
                .map(|g| (if g.has_floor { 1 } else { 0 }, g.floor_z, g.floor_mu))
                .unwrap_or((0, -10.0, 0.4));
            empty_mpm_grid_gpu(hf, fz, fmu)
        };
        let d_mats = self.d_mats.as_ref().unwrap().as_ptr();
        let d_shell_opt = self.shell_mesh.as_mut().map(|s| s.descriptor());
        let hash_meta = self.shell_mesh.as_mut().map(|s| {
            (
                s.hash_head.as_mut_ptr(),
                s.hash_next.as_mut_ptr(),
                s.hash_size,
                s.cell_size,
            )
        });
        let (ex, ey, ez) = if let Some(ref ds) = d_shell_opt {
            (ds.p_fx as *const f32, ds.p_fy as *const f32, ds.p_fz as *const f32)
        } else {
            (std::ptr::null(), std::ptr::null(), std::ptr::null())
        };

        unsafe {
            if is_euler {
                sim_cuda::check_cuda(
                    sim_euler_ib_gather_to_solid(
                        d_pts,
                        d_mats,
                        self.d_levels.as_ptr(),
                        self.levels.len() as i32,
                        self.fluid_const,
                        self.stream.raw(),
                    ),
                    "sim_euler_ib_gather_to_solid",
                )?;
            } else {
                sim_cuda::check_cuda(
                    sim_ib_gather_to_solid(
                        d_pts,
                        d_mats,
                        self.d_levels.as_ptr(),
                        self.levels.len() as i32,
                        self.fluid_const,
                        dt_step as f32,
                        self.stream.raw(),
                    ),
                    "sim_ib_gather_to_solid",
                )?;
            }
            if self.has_shells && !self.has_bulk_dynamic_solids && d_pts.count <= 8192 {
                if let Some(d_shell) = d_shell_opt {
                    let (hh, hn, hsz, csz) =
                        hash_meta.unwrap_or((std::ptr::null_mut(), std::ptr::null_mut(), 0, 0.01));
                    sim_cuda::check_cuda(
                        sim_shell_substep_fused(
                            d_shell,
                            d_pts,
                            d_mats,
                            hh,
                            hn,
                            hsz,
                            csz,
                            self.fluid_const.gx,
                            self.fluid_const.gy,
                            self.fluid_const.gz,
                            dt_s,
                            n_mpm as i32,
                            if self.has_contact { 1 } else { 0 },
                            d_grid.has_floor,
                            d_grid.floor_z,
                            d_grid.floor_mu,
                            self.fluid_const.sigma as f32,
                            self.fluid_const.mu_w as f32,
                            self.stream.raw(),
                        ),
                        "sim_shell_substep_fused",
                    )?;
                }
            } else {
                for _ in 0..n_mpm {
                    if let Some(d_shell) = d_shell_opt {
                        sim_cuda::check_cuda(
                            sim_shell_forces(d_shell, d_pts, d_mats, dt_s, self.stream.raw()),
                            "sim_shell_forces",
                        )?;
                        if self.has_contact {
                            if let Some((hh, hn, hsz, csz)) = hash_meta {
                                sim_cuda::check_cuda(
                                    sim_capillary_contact(
                                        d_pts,
                                        d_shell,
                                        d_mats,
                                        hh,
                                        hn,
                                        hsz,
                                        csz,
                                        dt_s,
                                        self.fluid_const.sigma as f32,
                                        self.fluid_const.mu_w as f32,
                                        self.stream.raw(),
                                    ),
                                    "sim_capillary_contact",
                                )?;
                            }
                        }
                    }
                    if self.has_bulk_dynamic_solids {
                        sim_cuda::check_cuda(
                            sim_mpm_step(
                                d_pts,
                                d_grid,
                                d_mats,
                                self.fluid_const.gx,
                                self.fluid_const.gy,
                                self.fluid_const.gz,
                                dt_s,
                                self.solid_friction_mu,
                                ex,
                                ey,
                                ez,
                                self.stream.raw(),
                            ),
                            "sim_mpm_step",
                        )?;
                    }
                    if self.has_shells {
                        if let Some(d_shell) = d_shell_opt {
                            sim_cuda::check_cuda(
                                sim_shell_integrate(
                                    d_pts,
                                    d_shell,
                                    d_mats,
                                    self.fluid_const.gx,
                                    self.fluid_const.gy,
                                    self.fluid_const.gz,
                                    dt_s,
                                    d_grid.has_floor,
                                    d_grid.floor_z,
                                    d_grid.floor_mu,
                                    self.stream.raw(),
                                ),
                                "sim_shell_integrate",
                            )?;
                        }
                    }
                }
            }
            if self.has_fracture && run_phase_field_inside {
                sim_cuda::check_cuda(
                    sim_phase_field_step(
                        d_pts,
                        d_grid,
                        d_mats,
                        dt_step as f32,
                        self.stream.raw(),
                    ),
                    "sim_phase_field_step",
                )?;
            }
            let multi_lvls = self.levels.len() > 1;
            for (li, lvl) in self.levels.iter_mut().enumerate() {
                if let Some(mask) = active_levels {
                    if !mask[li] {
                        continue;
                    }
                }
                if !is_euler && multi_lvls && li == 0 && lvl.spec.dx >= 0.08 {
                    continue;
                }
                if is_euler && multi_lvls && li == 0 && lvl.spec.dx >= 0.25 {
                    continue;
                }
                let d_lvl = lvl.descriptor();
                if is_euler {
                    sim_cuda::check_cuda(
                        sim_euler_ib_splat_level(d_pts, d_lvl, self.stream.raw()),
                        "sim_euler_ib_splat_level",
                    )?;
                } else {
                    sim_cuda::check_cuda(
                        sim_ib_splat_level(d_pts, d_lvl, self.stream.raw()),
                        "sim_ib_splat_level",
                    )?;
                }
            }
        }
        Ok(())
    }

    fn resolve_effective_render_config(
        &self,
        cfg: &SimRunConfig,
        active_dur_s: f64,
    ) -> (Option<RenderExportConfig>, bool) {
        if let Some(ref rcfg) = cfg.render_export {
            return (Some(rcfg.clone()), false);
        }
        if !cfg.auto_video || is_audio_only_override() {
            return (None, false);
        }
        let Some(ref out_dir) = cfg.output_dir else {
            return (None, false);
        };

        let frames_dir = out_dir
            .join("videos")
            .join("sim_data")
            .join(&self.scene.name);
        let num_frames = if active_dur_s < 0.5 {
            90
        } else {
            ((active_dur_s * 30.0).round() as usize).clamp(60, 240)
        };
        let has_water = !self.scene.enable_compressible_euler
            && (!self.scene.background_water.is_empty()
                || !self.scene.dynamic_fluid.is_empty()
                || !self.scene.fluid_inflows.is_empty()
                || !self.scene.droplet_emitters.is_empty());
        let export_pressure_slice = self.scene.enable_compressible_euler
            || !self.scene.point_sources.is_empty()
            || !self.scene.directional_sources.is_empty()
            || (!self.has_dynamic_solids && !has_water);
        let pressure_slice_z = if let Some(p) = self.scene.projectiles.first() {
            p.initial_pos[2]
        } else if let Some(b) = self.scene.blast_regions.first() {
            b.center[2]
        } else if let Some(s) = self.scene.point_sources.first() {
            s.center[2]
        } else if let Some(s) = self.scene.directional_sources.first() {
            s.center[2]
        } else if let Some(m) = self.scene.microphones.first() {
            m.position[2]
        } else {
            let s0 = &self.levels[0].spec;
            s0.oz + 0.5 * (s0.nz as f32) * s0.dx
        };

        (
            Some(RenderExportConfig {
                frames_dir,
                num_frames,
                export_fluid_isosurface: has_water,
                export_pressure_slice,
                pressure_slice_z,
            }),
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finalize_run_outputs(
        &self,
        cfg: &SimRunConfig,
        actual_pre_roll_s: f64,
        total_samples: usize,
        pre_roll_samples: usize,
        active_samples: usize,
        d_probe_out: &DeviceBuffer<f32>,
        final_m0: SimReduceMetricsGpu,
        final_sm: MpmMetricsGpu,
        peak_driver_used: usize,
        wall_time_s: f64,
        total_substeps: u64,
        timeseries: Vec<TimeSeriesRow>,
        render_exporter: Option<PreparedRenderExporter>,
        effective_render_cfg: Option<&RenderExportConfig>,
        run_auto_video_render: bool,
    ) -> Result<SimRunResult> {
        let num_probes = self.scene.microphones.len();
        let mut raw_probe_buffers = vec![vec![0.0_f32; total_samples]; num_probes];
        if num_probes > 0 {
            let interleaved = d_probe_out.to_vec()?;
            for s in 0..active_samples {
                for p in 0..num_probes {
                    raw_probe_buffers[p][pre_roll_samples + s] = interleaved[s * num_probes + p];
                }
            }
        }

        let mut probe_summaries = Vec::with_capacity(num_probes);
        for (p, mic) in self.scene.microphones.iter().enumerate() {
            let summary = analyze_probe_signal(
                &mic.name,
                mic.is_hydrophone,
                mic.position,
                &raw_probe_buffers[p][pre_roll_samples..],
                cfg.oversample_factor,
                cfg.sample_rate,
            );
            probe_summaries.push(summary);
        }

        if let Some(ref out_dir) = cfg.output_dir {
            let mut prepared = Vec::with_capacity(num_probes);
            for p in 0..num_probes {
                prepared.push(prepare_probe_wav_signal_f64(
                    &raw_probe_buffers[p],
                    cfg.oversample_factor,
                    cfg.sample_rate,
                    cfg.target_wav_sample_rate,
                    20.0,
                )?);
            }
            let fixed_gain = if cfg.shared_wav_gain {
                let peak_all = prepared
                    .iter()
                    .flat_map(|(_, sig, _, _)| sig.iter())
                    .fold(0.0_f64, |a, &b| a.max(b.abs()));
                Some(if peak_all > 1e-18 {
                    10.0_f64.powf(-1.0 / 20.0) / peak_all
                } else {
                    1.0
                })
            } else {
                None
            };
            for (p, mic) in self.scene.microphones.iter().enumerate() {
                let (wav_sr, ref sig, peak_pa, rms_pa) = prepared[p];
                let wav_path = if mic.name.ends_with(".wav") {
                    out_dir.join(&mic.name)
                } else if mic.name.is_empty() {
                    out_dir.join(format!("{}.wav", self.scene.name))
                } else {
                    out_dir.join(format!("{}_{}.wav", self.scene.name, mic.name))
                };
                let wav_stats = write_probe_wav_f32_with_gain(
                    &wav_path, sig, wav_sr, peak_pa, rms_pa, fixed_gain,
                )?;
                probe_summaries[p].wav = Some(wav_stats);
            }
        }

        let mut video_path: Option<PathBuf> = None;
        if let (Some(exp), Some(rcfg)) = (render_exporter, effective_render_cfg) {
            exp.write_manifest(
                self,
                rcfg,
                cfg.duration_s,
                actual_pre_roll_s,
                &probe_summaries,
                cfg.output_dir.as_deref(),
            )?;

            if run_auto_video_render {
                if let Some(ref out_dir) = cfg.output_dir {
                    // Prefer binaural stereo WAV when a left/right ear pair is present;
                    // otherwise pick the primary non-hydrophone mic (falling back to mic 0).
                    let left_idx = self
                        .scene
                        .microphones
                        .iter()
                        .position(|m| m.name.to_ascii_lowercase().contains("left"));
                    let right_idx = self
                        .scene
                        .microphones
                        .iter()
                        .position(|m| m.name.to_ascii_lowercase().contains("right"));
                    let video_wav_path = if let (Some(li), Some(ri)) = (left_idx, right_idx) {
                        if li != ri {
                            let stereo_p =
                                out_dir.join(format!("{}_stereo.wav", self.scene.name));
                            write_stereo_wav_f32(
                                &stereo_p,
                                &raw_probe_buffers[li],
                                &raw_probe_buffers[ri],
                                cfg.oversample_factor,
                                cfg.sample_rate,
                                cfg.target_wav_sample_rate,
                                20.0,
                            )?;
                            Some(stereo_p)
                        } else {
                            probe_summaries
                                .first()
                                .and_then(|s| s.wav.as_ref().map(|w| PathBuf::from(&w.path)))
                        }
                    } else {
                        probe_summaries
                            .iter()
                            .find(|s| !s.is_hydrophone && s.wav.is_some())
                            .or_else(|| probe_summaries.iter().find(|s| s.wav.is_some()))
                            .and_then(|s| s.wav.as_ref().map(|w| PathBuf::from(&w.path)))
                    };

                    if let Some(wav_p) = video_wav_path {
                        let render_wall = Instant::now();
                        let manifest_path = rcfg.frames_dir.join("manifest.json");
                        let png_dir = out_dir
                            .join("videos")
                            .join("frames")
                            .join(&self.scene.name);
                        let mp4_path = out_dir
                            .join("videos")
                            .join(format!("{}.mp4", self.scene.name));
                        let mut vopts = cfg.video_options.clone().unwrap_or_default();
                        vopts.wav_pre_roll_s = actual_pre_roll_s;
                        let active_dur_s = (cfg.duration_s - actual_pre_roll_s).max(1e-6);

                        println!(
                            "[{}] Rendering {} video frames with Blender Cycles ({}x{}, {} samples)...",
                            self.scene.name, rcfg.num_frames, vopts.width, vopts.height, vopts.samples
                        );
                        render_frames_with_blender(&manifest_path, &png_dir, &vopts)?;

                        if active_dur_s < 0.5 {
                            mux_video_with_audio(
                                &png_dir,
                                &wav_p,
                                cfg.duration_s,
                                rcfg.num_frames,
                                &mp4_path,
                                &vopts,
                            )?;
                        } else {
                            mux_realtime_video_with_audio(
                                &png_dir,
                                &wav_p,
                                cfg.duration_s,
                                actual_pre_roll_s,
                                rcfg.num_frames,
                                &mp4_path,
                                vopts.post_roll_s,
                            )?;
                        }
                        println!(
                            "[{}] Video saved -> {} (audio: {}, render+mux: {})",
                            self.scene.name,
                            mp4_path.display(),
                            wav_p.display(),
                            format_duration_hms(render_wall.elapsed().as_secs_f64())
                        );
                        video_path = Some(mp4_path);
                    }
                }
            }
        }

        let water_vol_err = if self.initial_water_volume > 1e-12 {
            (final_m0.water_volume - self.initial_water_volume).abs() / self.initial_water_volume
        } else {
            0.0
        };
        let final_e = final_m0.acoustic_energy
            + final_m0.kinetic_energy
            + final_sm.elastic_energy
            + final_sm.kinetic_energy;
        let energy_err = if self.initial_total_energy > 1e-12 {
            (final_e - self.initial_total_energy).abs() / self.initial_total_energy
        } else {
            0.0
        };

        let info = self.vram.device_info();
        let telemetry = RunTelemetry {
            scenario: self.scene.name.clone(),
            compute_capability: format!("sm_{}{}", info.cc_major, info.cc_minor),
            total_vram_bytes: info.total_vram_bytes,
            vram_cap_bytes: self.vram.cap_bytes(),
            peak_vram_tracked_bytes: self.vram.peak_bytes(),
            peak_vram_driver_used_bytes: peak_driver_used,
            sim_duration_s: cfg.duration_s,
            wall_time_s,
            total_substeps,
            nan_count: final_m0.nan_count + final_sm.nan_count,
            initial_water_volume: self.initial_water_volume,
            final_water_volume: final_m0.water_volume,
            water_volume_rel_error: water_vol_err,
            initial_total_energy: self.initial_total_energy,
            final_total_energy: final_e,
            energy_conservation_error: energy_err,
            probes: probe_summaries,
            notes: BTreeMap::new(),
        };

        if let Some(ref out_dir) = cfg.output_dir {
            write_telemetry_json(
                out_dir.join(format!("{}_telemetry.json", self.scene.name)),
                &telemetry,
            )?;
            write_timeseries_csv(
                out_dir.join(format!("{}_timeseries.csv", self.scene.name)),
                &timeseries,
            )?;
        }

        Ok(SimRunResult {
            telemetry,
            timeseries,
            raw_probe_buffers,
            video_path,
        })
    }

    pub fn run(&mut self, cfg: &SimRunConfig) -> Result<SimRunResult> {
        if self.scene.enable_compressible_euler {
            return self.run_compressible_euler(cfg);
        }
        let wall_start = Instant::now();
        let num_probes = self.scene.microphones.len();
        let fs_oversampled = (cfg.sample_rate as f64) * (cfg.oversample_factor as f64);
        let dt_probe = 1.0 / fs_oversampled;

        let pre_roll_s_clamped = cfg.pre_roll_s.clamp(0.0, cfg.duration_s);
        let total_samples = (cfg.duration_s * fs_oversampled).round().max(1.0) as usize;
        let pre_roll_samples = if pre_roll_s_clamped > 0.0 {
            ((pre_roll_s_clamped * fs_oversampled).round() as usize)
                .min(total_samples.saturating_sub(1))
        } else {
            0
        };
        let active_samples = total_samples - pre_roll_samples;
        let actual_pre_roll_s = (pre_roll_samples as f64) * dt_probe;
        let active_dur_s = (active_samples as f64) * dt_probe;

        let mut d_probe_out = if num_probes > 0 {
            self.vram.alloc_zeros::<f32>(active_samples * num_probes)?
        } else {
            self.vram.alloc_zeros::<f32>(1)?
        };

        let children = self.build_amr_children();
        let (mut dt_ac_0, mut k0, mut c_max, mut u_max, mut flow_substeps, mut t_refine, mut mult_from_0) =
            self.compute_adaptive_timestep(cfg)?;
        let mpm_dx_fallback = self
            .mpm_grid
            .as_ref()
            .map(|g| g.spec.dx as f64)
            .unwrap_or(0.005);
        let has_water_f = !self.scene.background_water.is_empty()
            || !self.scene.fluid_inflows.is_empty()
            || !self.scene.droplet_emitters.is_empty();
        let solid_once_per_probe =
            cfg.duration_s >= 1.0 && cfg.oversample_factor == 1 && !has_water_f;
        let dt_solid_step = if solid_once_per_probe { dt_probe } else { dt_ac_0 };
        let (mut n_mpm, mut dt_s) = self.compute_solid_dt_substeps(dt_solid_step, mpm_dx_fallback);
        let mut step_counters = vec![0u64; self.levels.len()];
        let finest_mult = (*mult_from_0.iter().max().unwrap_or(&k0) / k0.max(1)) as u64;

        let mut timeseries = Vec::new();
        let mut next_diag_t = 0.0_f64;
        if pre_roll_samples > 0 {
            let init_m0 = self.reduce_level_metrics(0)?;
            let init_sm = self.reduce_mpm_metrics()?;
            let p_count = self.particles.as_ref().map(|p| p.count).unwrap_or(0);
            let frac_init = if p_count > 0 {
                init_sm.damage_sum / (p_count as f64)
            } else {
                0.0
            };
            let dt_finest = dt_ac_0 / (finest_mult as f64);
            while next_diag_t < actual_pre_roll_s - 1e-12 {
                timeseries.push(TimeSeriesRow {
                    time_s: next_diag_t,
                    dt_ac_min: dt_finest,
                    c_max,
                    u_max,
                    acoustic_energy: init_m0.acoustic_energy,
                    kinetic_energy: init_m0.kinetic_energy,
                    solid_elastic_energy: init_sm.elastic_energy,
                    solid_kinetic_energy: init_sm.kinetic_energy,
                    total_energy: init_m0.acoustic_energy
                        + init_m0.kinetic_energy
                        + init_sm.elastic_energy
                        + init_sm.kinetic_energy,
                    water_volume: init_m0.water_volume,
                    max_abs_p: init_m0.min_p.abs().max(init_m0.max_p.abs()),
                    vram_tracked_bytes: self.vram.tracked_bytes(),
                    fractured_fraction: frac_init,
                    crater_min_z: 0.0,
                    subsurface_air_volume: 0.0,
                    bubble_radius_mm: 0.0,
                    minnaert_freq_hz: 0.0,
                });
                next_diag_t += cfg.telemetry_interval_s.max(1e-9);
            }
        }

        let mut total_substeps = 0u64;
        let mut peak_driver_used = 0usize;
        let mut last_print_wall = Instant::now();
        let pf_stride = cfg.phase_field_stride_steps.max(1);
        let run_pf_inside = pf_stride <= 1;

        if cfg.print_progress {
            println!(
                "[{}] start: dur={:.1}ms (pre={:.1}ms, active={:.1}ms) | k0={} | t_ref={:?} | n_mpm={} | dt_ac0={:.3e}s | VRAM={:.1}MB",
                self.scene.name,
                cfg.duration_s * 1e3,
                actual_pre_roll_s * 1e3,
                active_dur_s * 1e3,
                k0,
                t_refine,
                n_mpm,
                dt_ac_0,
                (self.vram.tracked_bytes() as f64) / 1e6
            );
        }

        let has_point_sources = !self.scene.point_sources.is_empty();
        let has_directional_sources = !self.scene.directional_sources.is_empty();
        let has_fluid_emitters =
            !self.scene.fluid_inflows.is_empty() || !self.scene.droplet_emitters.is_empty();
        let has_time_varying_sources = has_point_sources || has_directional_sources;
        let can_graph = cfg.use_cuda_graph && !has_time_varying_sources && !has_fluid_emitters;
        let mut cached_graph: Option<CudaGraphExec> = None;

        let mut drop_accum = vec![0.0_f64; self.scene.droplet_emitters.len()];
        let mut drop_counter = vec![0u64; self.scene.droplet_emitters.len()];
        let mut kinematic_pivots: Vec<[f32; 3]> = self
            .scene
            .kinematic_trajectories
            .iter()
            .map(|t| t.initial_pivot)
            .collect();

        if num_probes > 0 {
            unsafe {
                sim_cuda::check_cuda(
                    sim_fluid_sample_probes(
                        self.d_levels.as_ptr(),
                        self.levels.len() as i32,
                        self.d_probes.as_ptr(),
                        num_probes as i32,
                        d_probe_out.as_mut_ptr(),
                        0,
                        self.stream.raw(),
                    ),
                    "sim_fluid_sample_probes",
                )?;
            }
        }

        let (effective_render_cfg, run_auto_video_render) =
            self.resolve_effective_render_config(cfg, active_dur_s);
        let mut render_exporter = if let Some(ref rcfg) = effective_render_cfg {
            Some(PreparedRenderExporter::new(self, rcfg)?)
        } else {
            None
        };
        let mut next_render_frame = 0usize;
        let total_render_frames = effective_render_cfg
            .as_ref()
            .map(|r| r.num_frames)
            .unwrap_or(0);
        let t_end_active = (active_samples.saturating_sub(1) as f64) * dt_probe;

        for active_idx in 0..active_samples {
            let t_active = (active_idx as f64) * dt_probe;
            let t_now = actual_pre_roll_s + t_active;

            if t_now >= next_diag_t - 1e-12 {
                self.stream.sync()?;
                let mut total_fluid_nan = 0u64;
                let mut anomaly_lvl: Option<usize> = None;
                let m0 = self.reduce_level_metrics(0)?;
                let mut m_top = m0;
                let mut u_top = m0.u_max_sq.sqrt();
                let mut c_top = m0.c_max_sq.sqrt();
                for li in 0..self.levels.len() {
                    let ml = if li == 0 { m0 } else { self.reduce_level_metrics(li)? };
                    if li + 1 == self.levels.len() {
                        m_top = ml;
                    }
                    u_top = u_top.max(ml.u_max_sq.sqrt());
                    c_top = c_top.max(ml.c_max_sq.sqrt());
                    total_fluid_nan += ml.nan_count;
                    if anomaly_lvl.is_none()
                        && (ml.nan_count > 0
                            || ml.u_max_sq.sqrt() > 500.0
                            || Self::level_max_p(&ml) > 1.0e8)
                    {
                        anomaly_lvl = Some(li);
                    }
                }
                let sm = self.reduce_mpm_metrics()?;
                if anomaly_lvl.is_some() {
                    for li in (0..self.levels.len()).rev() {
                        let ml = self.reduce_level_metrics(li)?;
                        println!(
                            "  [anomaly-L{}] nan={} max|p|={:.3e}Pa u_max={:.3e}m/s",
                            li,
                            ml.nan_count,
                            Self::level_max_p(&ml),
                            ml.u_max_sq.sqrt()
                        );
                        let _ = self.dump_level_anomaly(li, t_now);
                    }
                }
                if total_fluid_nan > 0 || sm.nan_count > 0 {
                    return Err(EngineError::InvalidScene(format!(
                        "NaN detected at t={:.6}s (fluid={}, solid={})",
                        t_now, total_fluid_nan, sm.nan_count
                    )));
                }
                let (drv_free, drv_total) = self.vram.query_driver_mem_info()?;
                peak_driver_used = peak_driver_used.max(drv_total.saturating_sub(drv_free));

                if u_top > u_max * 1.5 + 2.0 || c_top > c_max * 1.15 {
                    let (new_dt0, new_k0, new_c, new_u, new_fs, new_tr, new_mult) =
                        self.compute_adaptive_timestep(cfg)?;
                    if new_k0 != k0 || new_fs != flow_substeps || new_tr != t_refine {
                        dt_ac_0 = new_dt0;
                        k0 = new_k0;
                        c_max = new_c;
                        u_max = new_u;
                        flow_substeps = new_fs;
                        t_refine = new_tr;
                        mult_from_0 = new_mult;
                        let dt_solid_step = if solid_once_per_probe { dt_probe } else { dt_ac_0 };
                        let (nn, ndt) = self.compute_solid_dt_substeps(dt_solid_step, mpm_dx_fallback);
                        n_mpm = nn;
                        dt_s = ndt;
                        cached_graph = None;
                    }
                }

                let mut bdiag = Default::default();
                if let Some(wz) = cfg.water_surface_z_for_diag {
                    let top_li = self.levels.len() - 1;
                    if let Some(ref abuf) = self.levels[top_li].alpha {
                        let alpha_snap = abuf.to_vec()?;
                        let s = &self.levels[top_li].spec;
                        bdiag = analyze_subsurface_bubble(
                            &alpha_snap,
                            s.nx,
                            s.ny,
                            s.nz,
                            self.levels[top_li].halo,
                            s.dx,
                            s.ox,
                            s.oy,
                            s.oz,
                            wz,
                        );
                    }
                }

                let p_count = self.particles.as_ref().map(|p| p.count).unwrap_or(0);
                let frac_frac = if p_count > 0 {
                    sm.damage_sum / (p_count as f64)
                } else {
                    0.0
                };
                let fmult_now = (*mult_from_0.iter().max().unwrap_or(&k0) / k0.max(1)) as u64;
                let dt_finest = dt_ac_0 / (fmult_now.max(1) as f64);
                timeseries.push(TimeSeriesRow {
                    time_s: t_now,
                    dt_ac_min: dt_finest,
                    c_max,
                    u_max,
                    acoustic_energy: m0.acoustic_energy,
                    kinetic_energy: m0.kinetic_energy,
                    solid_elastic_energy: sm.elastic_energy,
                    solid_kinetic_energy: sm.kinetic_energy,
                    total_energy: m0.acoustic_energy
                        + m0.kinetic_energy
                        + sm.elastic_energy
                        + sm.kinetic_energy,
                    water_volume: m0.water_volume,
                    max_abs_p: m0.min_p.abs().max(m0.max_p.abs()),
                    vram_tracked_bytes: self.vram.tracked_bytes(),
                    fractured_fraction: frac_frac,
                    crater_min_z: bdiag.crater_min_z,
                    subsurface_air_volume: bdiag.subsurface_air_volume,
                    bubble_radius_mm: bdiag.bubble_radius_m * 1e3,
                    minnaert_freq_hz: bdiag.minnaert_freq_hz,
                });
                next_diag_t += cfg.telemetry_interval_s.max(1e-9);
                if cfg.print_progress
                    && (cfg.water_surface_z_for_diag.is_some()
                        || last_print_wall.elapsed().as_secs_f64() >= 2.0
                        || active_idx == 0)
                {
                    let wall_s = wall_start.elapsed().as_secs_f64();
                    let frac = (t_active / active_dur_s.max(1e-12)).clamp(0.0, 1.0);
                    let wall_str = format_duration_hms(wall_s);
                    let eta_str = if frac > 0.01 {
                        format_duration_hms(wall_s * (1.0 - frac) / frac)
                    } else {
                        "--".to_string()
                    };
                    if cfg.water_surface_z_for_diag.is_some() {
                        println!(
                            "[{}] t={:6.2}/{:6.2}ms ({:5.1}%) | wall={} | ETA={} | max|p|={:.3e}Pa | u_top={:.2}m/s | p_L2={:.3e}Pa | crater={:+.2}mm | Rb={:.3}mm | VRAM={:.1}MB",
                            self.scene.name,
                            t_now * 1e3,
                            cfg.duration_s * 1e3,
                            frac * 100.0,
                            wall_str,
                            eta_str,
                            Self::level_max_p(&m0),
                            u_top,
                            Self::level_max_p(&m_top),
                            bdiag.crater_min_z * 1e3,
                            bdiag.bubble_radius_m * 1e3,
                            (self.vram.tracked_bytes() as f64) / 1e6
                        );
                    } else {
                        println!(
                            "[{}] t={:6.2}/{:6.2}ms ({:5.1}%) | wall={} | ETA={} | max|p|={:.3e}Pa | VRAM={:.1}MB",
                            self.scene.name,
                            t_now * 1e3,
                            cfg.duration_s * 1e3,
                            frac * 100.0,
                            wall_str,
                            eta_str,
                            m0.min_p.abs().max(m0.max_p.abs()),
                            (self.vram.tracked_bytes() as f64) / 1e6
                        );
                    }
                    last_print_wall = Instant::now();
                }
            }

            if let (Some(ref mut exp), Some(ref rcfg)) = (&mut render_exporter, &effective_render_cfg)
            {
                while next_render_frame < total_render_frames {
                    let t_target = if total_render_frames > 1 {
                        (next_render_frame as f64) * t_end_active
                            / ((total_render_frames - 1) as f64)
                    } else {
                        0.0
                    };
                    if t_active + 1e-12 >= t_target || active_idx + 1 >= active_samples {
                        exp.export_frame(self, next_render_frame, rcfg)?;
                        next_render_frame += 1;
                    } else {
                        break;
                    }
                }
            }

            if active_idx + 1 >= active_samples {
                break;
            }

            self.update_kinematics(t_now, dt_probe, &mut kinematic_pivots)?;

            // Apply fluid nozzles and falling droplet emitters on flow-enabled levels
            for inf in &self.scene.fluid_inflows {
                if t_now >= inf.start_time_s && t_now <= inf.end_time_s {
                    let ramp = if inf.ramp_s > 0.0 {
                        ((t_now - inf.start_time_s) / inf.ramp_s).clamp(0.0, 1.0) as f32
                    } else {
                        1.0
                    };
                    let relax = 0.85 * ramp;
                    let mut best_li = None;
                    for (li, lvl) in self.levels.iter().enumerate() {
                        if !lvl.spec.alloc_flow {
                            continue;
                        }
                        let s = &lvl.spec;
                        if inf.center[0] >= s.ox
                            && inf.center[0] <= s.ox + s.nx as f32 * s.dx
                            && inf.center[1] >= s.oy
                            && inf.center[1] <= s.oy + s.ny as f32 * s.dx
                            && inf.center[2] >= s.oz
                            && inf.center[2] <= s.oz + s.nz as f32 * s.dx
                        {
                            best_li = Some(li);
                        }
                    }
                    if let Some(li) = best_li {
                        let d_lvl = self.levels[li].descriptor();
                        unsafe {
                            sim_cuda::check_cuda(
                                sim_fluid_apply_inflow_nozzle(
                                    d_lvl,
                                    inf.center[0],
                                    inf.center[1],
                                    inf.center[2],
                                    inf.radius,
                                    inf.height,
                                    inf.velocity[0],
                                    inf.velocity[1],
                                    inf.velocity[2],
                                    relax,
                                    self.stream.raw(),
                                ),
                                "sim_fluid_apply_inflow_nozzle",
                            )?;
                        }
                    }
                }
            }
            for (ei, em) in self.scene.droplet_emitters.iter().enumerate() {
                if t_now >= em.start_time_s && t_now <= em.end_time_s {
                    drop_accum[ei] += em.drops_per_sec * dt_probe;
                    while drop_accum[ei] >= 1.0 {
                        drop_accum[ei] -= 1.0;
                        let dc = drop_counter[ei];
                        drop_counter[ei] += 1;
                        let u1 = rand_unit_f32(em.seed, dc * 3);
                        let u2 = rand_unit_f32(em.seed, dc * 3 + 1);
                        let u3 = rand_unit_f32(em.seed, dc * 3 + 2);
                        let x = em.area_min_xy[0] + u1 * (em.area_max_xy[0] - em.area_min_xy[0]);
                        let y = em.area_min_xy[1] + u2 * (em.area_max_xy[1] - em.area_min_xy[1]);
                        let r = em.radius_min + u3 * (em.radius_max - em.radius_min);
                        let mut best_li = None;
                        for (li, lvl) in self.levels.iter().enumerate() {
                            if !lvl.spec.alloc_flow {
                                continue;
                            }
                            let s = &lvl.spec;
                            let m = 2.0 * s.dx;
                            if x >= s.ox + m
                                && x <= s.ox + s.nx as f32 * s.dx - m
                                && y >= s.oy + m
                                && y <= s.oy + s.ny as f32 * s.dx - m
                                && em.spawn_z >= s.oz + m
                                && em.spawn_z <= s.oz + s.nz as f32 * s.dx - m
                            {
                                best_li = Some(li);
                            }
                        }
                        if let Some(li) = best_li {
                            let d_lvl = self.levels[li].descriptor();
                            unsafe {
                                sim_cuda::check_cuda(
                                    sim_fluid_spawn_droplet(
                                        d_lvl,
                                        self.fluid_const,
                                        x,
                                        y,
                                        em.spawn_z,
                                        r,
                                        0.0,
                                        0.0,
                                        em.velocity_z,
                                        self.stream.raw(),
                                    ),
                                    "sim_fluid_spawn_droplet",
                                )?;
                            }
                        }
                    }
                }
            }

            if can_graph {
                let mut flow_repeats = true;
                for li in 0..self.levels.len() {
                    if self.levels[li].spec.alloc_flow && mult_from_0[li] % flow_substeps[li] != 0 {
                        flow_repeats = false;
                    }
                }
                if cached_graph.is_none() && flow_repeats {
                    self.stream.sync()?;
                    self.stream.begin_capture()?;
                    if solid_once_per_probe {
                        self.step_dynamic_solids_only(
                            dt_probe,
                            n_mpm,
                            dt_s,
                            run_pf_inside,
                            false,
                            None,
                        )?;
                        for _ in 0..k0 {
                            Self::launch_advance_level(
                                &mut self.levels,
                                &children,
                                &t_refine,
                                self.fluid_const,
                                &flow_substeps,
                                &mut step_counters,
                                0,
                                dt_ac_0,
                                &self.stream,
                            )?;
                        }
                    } else {
                        for _ in 0..k0 {
                            self.launch_coupled_step(
                                &children,
                                &t_refine,
                                dt_ac_0,
                                n_mpm,
                                dt_s,
                                &flow_substeps,
                                &mut step_counters,
                                run_pf_inside,
                            )?;
                        }
                    }
                    cached_graph = Some(self.stream.end_capture()?);
                }

                if let Some(ref graph) = cached_graph {
                    graph.launch(&self.stream)?;
                } else if solid_once_per_probe {
                    self.step_dynamic_solids_only(
                        dt_probe,
                        n_mpm,
                        dt_s,
                        run_pf_inside,
                        false,
                        None,
                    )?;
                    for _ in 0..k0 {
                        Self::launch_advance_level(
                            &mut self.levels,
                            &children,
                            &t_refine,
                            self.fluid_const,
                            &flow_substeps,
                            &mut step_counters,
                            0,
                            dt_ac_0,
                            &self.stream,
                        )?;
                    }
                } else {
                    for _ in 0..k0 {
                        self.launch_coupled_step(
                            &children,
                            &t_refine,
                            dt_ac_0,
                            n_mpm,
                            dt_s,
                            &flow_substeps,
                            &mut step_counters,
                            run_pf_inside,
                        )?;
                    }
                }
            } else {
                if solid_once_per_probe {
                    self.step_dynamic_solids_only(
                        dt_probe,
                        n_mpm,
                        dt_s,
                        run_pf_inside,
                        false,
                        None,
                    )?;
                }
                for sub in 0..k0 {
                    let t_sub = t_now + (sub as f64) * dt_ac_0;
                    if has_point_sources {
                        for src in &self.scene.point_sources {
                            let rate = src.rate_at(t_sub);
                            for lvl in &mut self.levels {
                                let d_lvl = lvl.descriptor();
                                unsafe {
                                    sim_cuda::check_cuda(
                                        sim_fluid_set_point_source_ricker(
                                            d_lvl,
                                            src.center[0],
                                            src.center[1],
                                            src.center[2],
                                            src.radius,
                                            rate,
                                            self.stream.raw(),
                                        ),
                                        "sim_fluid_set_point_source_ricker",
                                    )?;
                                }
                            }
                        }
                    }
                    if has_directional_sources {
                        for (s_idx, dsrc) in self.scene.directional_sources.iter().enumerate() {
                            let rate = dsrc.rate_at(t_sub);
                            let dir = normalize3(dsrc.direction);
                            let clear_prev = if !has_point_sources && s_idx == 0 {
                                1
                            } else {
                                0
                            };
                            for lvl in &mut self.levels {
                                let d_lvl = lvl.descriptor();
                                unsafe {
                                    sim_cuda::check_cuda(
                                        sim_fluid_apply_directional_source_bbox(
                                            d_lvl,
                                            self.fluid_const,
                                            dsrc.center[0],
                                            dsrc.center[1],
                                            dsrc.center[2],
                                            dir[0],
                                            dir[1],
                                            dir[2],
                                            dsrc.cardioid_beta,
                                            dsrc.radius,
                                            rate,
                                            dt_ac_0 as f32,
                                            clear_prev,
                                            self.stream.raw(),
                                        ),
                                        "sim_fluid_apply_directional_source_bbox",
                                    )?;
                                }
                            }
                        }
                    }
                    if solid_once_per_probe {
                        Self::launch_advance_level(
                            &mut self.levels,
                            &children,
                            &t_refine,
                            self.fluid_const,
                            &flow_substeps,
                            &mut step_counters,
                            0,
                            dt_ac_0,
                            &self.stream,
                        )?;
                    } else {
                        self.launch_coupled_step(
                            &children,
                            &t_refine,
                            dt_ac_0,
                            n_mpm,
                            dt_s,
                            &flow_substeps,
                            &mut step_counters,
                            run_pf_inside,
                        )?;
                    }
                }
            }
            if self.has_fracture && !run_pf_inside && (active_idx + 1) % pf_stride == 0 {
                let d_pts = self.particles.as_mut().unwrap().descriptor();
                let d_grid = self.mpm_grid.as_mut().unwrap().descriptor();
                let d_mats = self.d_mats.as_ref().unwrap().as_ptr();
                unsafe {
                    sim_cuda::check_cuda(
                        sim_phase_field_step(
                            d_pts,
                            d_grid,
                            d_mats,
                            (dt_probe * (pf_stride as f64)) as f32,
                            self.stream.raw(),
                        ),
                        "sim_phase_field_step",
                    )?;
                }
            }
            let fmult_step = (*mult_from_0.iter().max().unwrap_or(&k0)) as u64;
            total_substeps += fmult_step;

            if num_probes > 0 {
                unsafe {
                    sim_cuda::check_cuda(
                        sim_fluid_sample_probes(
                            self.d_levels.as_ptr(),
                            self.levels.len() as i32,
                            self.d_probes.as_ptr(),
                            num_probes as i32,
                            d_probe_out.as_mut_ptr(),
                            (active_idx + 1) as i32,
                            self.stream.raw(),
                        ),
                        "sim_fluid_sample_probes",
                    )?;
                }
            }
        }

        self.stream.sync()?;
        device_sync()?;
        let final_m0 = self.reduce_level_metrics(0)?;
        let final_sm = self.reduce_mpm_metrics()?;
        let (drv_free, drv_total) = self.vram.query_driver_mem_info()?;
        peak_driver_used = peak_driver_used.max(drv_total.saturating_sub(drv_free));
        let wall_time_s = wall_start.elapsed().as_secs_f64();

        self.finalize_run_outputs(
            cfg,
            actual_pre_roll_s,
            total_samples,
            pre_roll_samples,
            active_samples,
            &d_probe_out,
            final_m0,
            final_sm,
            peak_driver_used,
            wall_time_s,
            total_substeps,
            timeseries,
            render_exporter,
            effective_render_cfg.as_ref(),
            run_auto_video_render,
        )
    }

    fn run_compressible_euler(&mut self, cfg: &SimRunConfig) -> Result<SimRunResult> {
        let wall_start = Instant::now();
        let num_probes = self.scene.microphones.len();
        let fs_oversampled = (cfg.sample_rate as f64) * (cfg.oversample_factor as f64);
        let dt_probe = 1.0 / fs_oversampled;

        let pre_roll_s_clamped = cfg.pre_roll_s.clamp(0.0, cfg.duration_s);
        let total_samples = (cfg.duration_s * fs_oversampled).round().max(1.0) as usize;
        let pre_roll_samples = if pre_roll_s_clamped > 0.0 {
            ((pre_roll_s_clamped * fs_oversampled).round() as usize)
                .min(total_samples.saturating_sub(1))
        } else {
            0
        };
        let active_samples = total_samples - pre_roll_samples;
        let actual_pre_roll_s = (pre_roll_samples as f64) * dt_probe;
        let active_dur_s = (active_samples as f64) * dt_probe;

        let mut d_probe_out = if num_probes > 0 {
            self.vram.alloc_zeros::<f32>(active_samples * num_probes)?
        } else {
            self.vram.alloc_zeros::<f32>(1)?
        };

        let mpm_dx_fallback = self
            .mpm_grid
            .as_ref()
            .map(|g| g.spec.dx as f64)
            .unwrap_or(0.005);
        let (n_mpm, dt_s) = self.compute_solid_dt_substeps(dt_probe, mpm_dx_fallback);

        let m0_init = self.reduce_level_metrics(0)?;
        let cfl_eu = cfg.cfl_safety.min(0.42);
        let dx0 = self.levels[0].spec.dx as f64;
        let nu_eff = ((self.fluid_const.mu_a + self.fluid_const.bulk_viscosity_a as f64)
            / self.fluid_const.rho_a)
            * 2.5;
        let has_fine_patches = self.levels.len() > 1;
        let compute_coarse_stride = |m: &SimReduceMetricsGpu, in_blast_burst: bool| -> (usize, f64) {
            let c0 = m.c_max_sq.sqrt().max(AIR_C) * 1.05;
            let u0 = m.u_max_sq.sqrt();
            let dt0_lim = cfl_eu * dx0 / (1.25 * c0 + 1.15 * u0 + 2.0 * nu_eff / dx0);
            let max_s = if in_blast_burst {
                2.0
            } else if has_fine_patches {
                6.0
            } else {
                32.0
            };
            let stride = (dt0_lim / dt_probe).floor().clamp(1.0, max_s) as usize;
            (stride, dt0_lim)
        };
        let (mut coarse_stride, mut dt0_lim) = compute_coarse_stride(&m0_init, false);

        struct ProjState {
            active: bool,
            finished: bool,
            pos: [f32; 3],
            vel: [f32; 3],
        }
        let mut proj_states: Vec<ProjState> = self
            .scene
            .projectiles
            .iter()
            .map(|p| ProjState {
                active: false,
                finished: false,
                pos: p.initial_pos,
                vel: p.initial_vel,
            })
            .collect();
        let mut blast_triggered = vec![false; self.scene.blast_regions.len()];

        let mut sliding_active = false;
        let mut handoff_done = false;
        self.sync_d_levels(false)?;

        let mut timeseries = Vec::new();
        let mut next_diag_t = 0.0_f64;
        if pre_roll_samples > 0 {
            let init_sm = self.reduce_mpm_metrics()?;
            let p_count = self.particles.as_ref().map(|p| p.count).unwrap_or(0);
            let frac_init = if p_count > 0 {
                init_sm.damage_sum / (p_count as f64)
            } else {
                0.0
            };
            while next_diag_t < actual_pre_roll_s - 1e-12 {
                timeseries.push(TimeSeriesRow {
                    time_s: next_diag_t,
                    dt_ac_min: dt_probe,
                    c_max: m0_init.c_max_sq.sqrt().max(AIR_C),
                    u_max: m0_init.u_max_sq.sqrt(),
                    acoustic_energy: m0_init.acoustic_energy,
                    kinetic_energy: m0_init.kinetic_energy,
                    solid_elastic_energy: init_sm.elastic_energy,
                    solid_kinetic_energy: init_sm.kinetic_energy,
                    total_energy: m0_init.acoustic_energy
                        + m0_init.kinetic_energy
                        + init_sm.elastic_energy
                        + init_sm.kinetic_energy,
                    water_volume: 0.0,
                    max_abs_p: Self::level_max_p(&m0_init),
                    vram_tracked_bytes: self.vram.tracked_bytes(),
                    fractured_fraction: frac_init,
                    crater_min_z: 0.0,
                    subsurface_air_volume: 0.0,
                    bubble_radius_mm: 0.0,
                    minnaert_freq_hz: 0.0,
                });
                next_diag_t += cfg.telemetry_interval_s.max(1e-9);
            }
        }

        if cfg.print_progress {
            println!(
                "[{}] EULER start: dur={:.1}ms (pre={:.1}ms, active={:.1}ms) | levels={} | dx0={:.3}m | coarse_stride={} | VRAM={:.1}MB",
                self.scene.name,
                cfg.duration_s * 1e3,
                actual_pre_roll_s * 1e3,
                active_dur_s * 1e3,
                self.levels.len(),
                dx0,
                coarse_stride,
                (self.vram.tracked_bytes() as f64) / 1e6
            );
        }

        if num_probes > 0 {
            unsafe {
                sim_cuda::check_cuda(
                    sim_euler_sample_probes(
                        self.d_levels.as_ptr(),
                        self.levels.len() as i32,
                        self.d_probes.as_ptr(),
                        num_probes as i32,
                        d_probe_out.as_mut_ptr(),
                        0,
                        self.fluid_const,
                        self.stream.raw(),
                    ),
                    "sim_euler_sample_probes",
                )?;
            }
        }

        let (effective_render_cfg, run_auto_video_render) =
            self.resolve_effective_render_config(cfg, active_dur_s);
        let mut render_exporter = if let Some(ref rcfg) = effective_render_cfg {
            Some(PreparedRenderExporter::new(self, rcfg)?)
        } else {
            None
        };
        let mut next_render_frame = 0usize;
        let total_render_frames = effective_render_cfg
            .as_ref()
            .map(|r| r.num_frames)
            .unwrap_or(0);
        let t_end_active = (active_samples.saturating_sub(1) as f64) * dt_probe;

        let mut kinematic_pivots: Vec<[f32; 3]> = self
            .scene
            .kinematic_trajectories
            .iter()
            .map(|t| t.initial_pivot)
            .collect();

        let mut steps_since_coarse = 0usize;
        let mut dt_since_coarse = 0.0_f64;
        let mut blast_cooldown_until = -1.0_f64;
        let mut total_substeps = 0u64;
        let mut peak_driver_used = 0usize;
        let mut last_print_wall = Instant::now();
        let mut active_mask = vec![false; self.levels.len()];
        let mut prev_active_mask = vec![false; self.levels.len()];
        let mut fine_k = vec![1usize; self.levels.len()];
        let mut force_diag_stride_update = true;
        let mut last_recompute_idx = 0usize;
        self.sync_d_levels(false)?;

        for active_idx in 0..active_samples {
            let t_active = (active_idx as f64) * dt_probe;
            let t_now = actual_pre_roll_s + t_active;

            active_mask[0] = true;
            for li in 1..self.levels.len() {
                if Some(li) == self.sliding_level_idx {
                    active_mask[li] = sliding_active;
                } else {
                    let amrp = &self.scene.amr.patches[li - 1];
                    let active_end = amrp.active_end_s.unwrap_or(f64::INFINITY);
                    active_mask[li] =
                        t_now + 1e-12 >= amrp.active_start_s && t_now <= active_end + 1e-12;
                }
            }
            if active_mask != prev_active_mask {
                for li in 1..self.levels.len() {
                    if Some(li) == self.sliding_level_idx {
                        continue;
                    }
                    if active_mask[li] && !prev_active_mask[li] && active_idx > 0 {
                        let p_idx = self.level_parents[li];
                        let d_dst = self.levels[li].descriptor();
                        let d_src = self.levels[p_idx].descriptor();
                        unsafe {
                            sim_cuda::check_cuda(
                                sim_euler_copy_overlap(d_dst, d_src, self.stream.raw()),
                                "sim_euler_copy_overlap_activate",
                            )?;
                        }
                        force_diag_stride_update = true;
                    } else if !active_mask[li] && prev_active_mask[li] {
                        let p_idx = self.level_parents[li];
                        let d_child = self.levels[li].descriptor();
                        let d_parent = self.levels[p_idx].descriptor();
                        unsafe {
                            sim_cuda::check_cuda(
                                sim_euler_amr_restrict(d_child, d_parent, self.stream.raw()),
                                "sim_euler_amr_restrict_deactivate",
                            )?;
                        }
                    }
                }
                let mut descs: Vec<FluidLevelGpu> =
                    self.levels.iter_mut().map(|l| l.descriptor()).collect();
                for li in 1..descs.len() {
                    if !active_mask[li] {
                        descs[li].ox = -1.0e9;
                    }
                }
                self.d_levels.upload(&descs)?;
                prev_active_mask.clone_from(&active_mask);
            }

            // Recompute per-level CFL substeps periodically or after blast events
            let need_cfl_check = force_diag_stride_update
                || (t_now <= blast_cooldown_until && (active_idx - last_recompute_idx) >= 20)
                || ((active_idx - last_recompute_idx) >= 250);
            if need_cfl_check {
                self.stream.sync()?;
                let m0 = self.reduce_level_metrics(0)?;
                if m0.nan_count > 0 {
                    return Err(EngineError::InvalidScene(format!(
                        "Euler NaN on Level 0 at t={t_now:.5}s"
                    )));
                }
                let (ns, nl) = compute_coarse_stride(&m0, t_now <= blast_cooldown_until);
                coarse_stride = ns;
                dt0_lim = nl;
                for li in 1..self.levels.len() {
                    if !active_mask[li] {
                        continue;
                    }
                    let ml = self.reduce_level_metrics(li)?;
                    if ml.nan_count > 0 {
                        return Err(EngineError::InvalidScene(format!(
                            "Euler NaN on Level {li} at t={t_now:.5}s"
                        )));
                    }
                    let cl = ml.c_max_sq.sqrt().max(AIR_C) * 1.05;
                    let mut ul = ml.u_max_sq.sqrt();
                    for ps in &proj_states {
                        if ps.active {
                            let s = &self.levels[li].spec;
                            if ps.pos[0] >= s.ox - 0.5
                                && ps.pos[0] <= s.ox + s.nx as f32 * s.dx + 0.5
                            {
                                let sp = ((ps.vel[0] * ps.vel[0]
                                    + ps.vel[1] * ps.vel[1]
                                    + ps.vel[2] * ps.vel[2])
                                    as f64)
                                    .sqrt();
                                ul = ul.max(sp * 1.15);
                            }
                        }
                    }
                    let dx_l = self.levels[li].spec.dx as f64;
                    let dt_lim_f = cfl_eu * dx_l / (1.25 * cl + 1.15 * ul + 2.0 * nu_eff / dx_l);
                    fine_k[li] = (dt_probe / dt_lim_f).ceil().max(1.0) as usize;
                }
                last_recompute_idx = active_idx;
                force_diag_stride_update = false;
            }

            if t_now >= next_diag_t - 1e-12 {
                self.stream.sync()?;
                let m0 = self.reduce_level_metrics(0)?;
                let sm = self.reduce_mpm_metrics()?;
                if m0.nan_count > 0 || sm.nan_count > 0 {
                    return Err(EngineError::InvalidScene(format!(
                        "NaN at t={:.6}s (fluid={}, solid={})",
                        t_now, m0.nan_count, sm.nan_count
                    )));
                }
                let (drv_free, drv_total) = self.vram.query_driver_mem_info()?;
                peak_driver_used = peak_driver_used.max(drv_total.saturating_sub(drv_free));

                let p_count = self.particles.as_ref().map(|p| p.count).unwrap_or(0);
                let frac_frac = if p_count > 0 {
                    sm.damage_sum / (p_count as f64)
                } else {
                    0.0
                };
                let max_p0 = Self::level_max_p(&m0);
                timeseries.push(TimeSeriesRow {
                    time_s: t_now,
                    dt_ac_min: dt_probe,
                    c_max: m0.c_max_sq.sqrt().max(AIR_C),
                    u_max: m0.u_max_sq.sqrt(),
                    acoustic_energy: m0.acoustic_energy,
                    kinetic_energy: m0.kinetic_energy,
                    solid_elastic_energy: sm.elastic_energy,
                    solid_kinetic_energy: sm.kinetic_energy,
                    total_energy: m0.acoustic_energy
                        + m0.kinetic_energy
                        + sm.elastic_energy
                        + sm.kinetic_energy,
                    water_volume: 0.0,
                    max_abs_p: max_p0,
                    vram_tracked_bytes: self.vram.tracked_bytes(),
                    fractured_fraction: frac_frac,
                    crater_min_z: 0.0,
                    subsurface_air_volume: 0.0,
                    bubble_radius_mm: 0.0,
                    minnaert_freq_hz: 0.0,
                });
                next_diag_t += cfg.telemetry_interval_s.max(1e-9);
                if cfg.print_progress
                    && (last_print_wall.elapsed().as_secs_f64() >= 2.0 || active_idx == 0)
                {
                    let wall_s = wall_start.elapsed().as_secs_f64();
                    let frac = (t_active / active_dur_s.max(1e-12)).clamp(0.0, 1.0);
                    let wall_str = format_duration_hms(wall_s);
                    let eta_str = if frac > 0.01 {
                        format_duration_hms(wall_s * (1.0 - frac) / frac)
                    } else {
                        "--".to_string()
                    };
                    println!(
                        "[{}] t={:6.2}/{:6.2}ms ({:5.1}%) | wall={} | ETA={} | stride0={} | max|dp|={:.3e}Pa | VRAM={:.1}MB",
                        self.scene.name,
                        t_now * 1e3,
                        cfg.duration_s * 1e3,
                        frac * 100.0,
                        wall_str,
                        eta_str,
                        coarse_stride,
                        max_p0,
                        (self.vram.tracked_bytes() as f64) / 1e6
                    );
                    last_print_wall = Instant::now();
                }
            }

            if let (Some(ref mut exp), Some(ref rcfg)) = (&mut render_exporter, &effective_render_cfg)
            {
                while next_render_frame < total_render_frames {
                    let t_target = if total_render_frames > 1 {
                        (next_render_frame as f64) * t_end_active
                            / ((total_render_frames - 1) as f64)
                    } else {
                        0.0
                    };
                    if t_active + 1e-12 >= t_target || active_idx + 1 >= active_samples {
                        exp.export_frame(self, next_render_frame, rcfg)?;
                        next_render_frame += 1;
                    } else {
                        break;
                    }
                }
            }

            if active_idx + 1 >= active_samples {
                break;
            }

            // 1. Trigger scheduled blasts
            for (bi, b) in self.scene.blast_regions.iter().enumerate() {
                if !blast_triggered[bi] && t_now + 1e-12 >= b.trigger_time_s {
                    blast_triggered[bi] = true;
                    blast_cooldown_until = t_now + 0.025;
                    coarse_stride = 1;
                    force_diag_stride_update = true;
                    for fk in fine_k.iter_mut().skip(1) {
                        *fk = (*fk).max(4);
                    }
                    for (li, lvl) in self.levels.iter_mut().enumerate() {
                        if Some(li) == self.sliding_level_idx && !sliding_active {
                            continue;
                        }
                        if li > 0 && Some(li) != self.sliding_level_idx && !active_mask[li] {
                            continue;
                        }
                        let s = &lvl.spec;
                        let span = b.radius.max(b.length) + 4.0 * s.dx;
                        if b.center[0] + span >= s.ox
                            && b.center[0] - span <= s.ox + s.nx as f32 * s.dx
                            && b.center[1] + span >= s.oy
                            && b.center[1] - span <= s.oy + s.ny as f32 * s.dx
                            && b.center[2] + span >= s.oz
                            && b.center[2] - span <= s.oz + s.nz as f32 * s.dx
                        {
                            let d_lvl = lvl.descriptor();
                            unsafe {
                                sim_cuda::check_cuda(
                                    sim_euler_add_blast_region(
                                        d_lvl,
                                        self.fluid_const,
                                        b.center[0],
                                        b.center[1],
                                        b.center[2],
                                        b.radius,
                                        b.length,
                                        if b.is_cylinder_x { 1 } else { 0 },
                                        b.overpressure_pa,
                                        b.density_kg_m3,
                                        b.velocity[0],
                                        b.velocity[1],
                                        b.velocity[2],
                                        self.stream.raw(),
                                    ),
                                    "sim_euler_add_blast_region",
                                )?;
                            }
                        }
                    }
                }
            }

            // 2. Kinematic trajectories & dynamic MPM / shell solids
            self.update_kinematics(t_now, dt_probe, &mut kinematic_pivots)?;
            let any_fine_active = self.levels.len() == 1 || active_mask.iter().skip(1).any(|&a| a);
            if any_fine_active {
                let run_pf_now =
                    self.has_fracture && (active_idx % cfg.phase_field_stride_steps.max(1) == 0);
                self.step_dynamic_solids_only(
                    dt_probe,
                    n_mpm,
                    dt_s,
                    run_pf_now,
                    true,
                    Some(&active_mask),
                )?;
            }

            // 3. Advance supersonic projectiles & sliding AMR shock patch
            let mut fine_proj_steps: Vec<(usize, [f32; 3], [f32; 3], f32, f32)> = Vec::new();
            for pi in 0..self.scene.projectiles.len() {
                let pspec = self.scene.projectiles[pi].clone();
                let ps = &mut proj_states[pi];
                if ps.finished || t_now + 1e-12 < pspec.trigger_time_s {
                    continue;
                }
                if !ps.active {
                    ps.active = true;
                    force_diag_stride_update = true;
                    for fk in fine_k.iter_mut().skip(1) {
                        *fk = (*fk).max(4);
                    }
                }

                let pos_old = ps.pos;
                let vel_cur = ps.vel;
                let dt_f32 = dt_probe as f32;
                let pos_new = [
                    pos_old[0] + vel_cur[0] * dt_f32,
                    pos_old[1] + vel_cur[1] * dt_f32,
                    pos_old[2] + vel_cur[2] * dt_f32,
                ];

                if let (Some(sli), Some(sp_cfg)) = (self.sliding_level_idx, self.scene.sliding_patch.clone()) {
                    let dx0_f = self.levels[0].spec.dx;
                    let dx_s = self.levels[sli].spec.dx;
                    let win_len_x = sp_cfg.nx as f32 * dx_s;
                    if !sliding_active && !handoff_done && pos_new[0] >= sp_cfg.initial_ox + 0.1 {
                        sliding_active = true;
                        active_mask[sli] = true;
                        let mut copied_from_fine = false;
                        for li in 1..self.levels.len() {
                            if li == sli || !active_mask[li] {
                                continue;
                            }
                            let d_dst = self.levels[sli].descriptor();
                            let d_src = self.levels[li].descriptor();
                            unsafe {
                                sim_cuda::check_cuda(
                                    sim_euler_copy_overlap(d_dst, d_src, self.stream.raw()),
                                    "sim_euler_copy_overlap",
                                )?;
                            }
                            copied_from_fine = true;
                        }
                        if !copied_from_fine {
                            let d_dst = self.levels[sli].descriptor();
                            let d_src = self.levels[0].descriptor();
                            unsafe {
                                sim_cuda::check_cuda(
                                    sim_euler_copy_overlap(d_dst, d_src, self.stream.raw()),
                                    "sim_euler_copy_overlap",
                                )?;
                            }
                        }
                        self.sync_d_levels(true)?;
                        force_diag_stride_update = true;
                    }

                    if sliding_active {
                        let target_ox = pos_new[0] - sp_cfg.lead_fraction * win_len_x;
                        let cur_ox = self.levels[sli].spec.ox;
                        if target_ox - cur_ox >= dx0_f {
                            let max_off_x = self.levels[0]
                                .spec
                                .nx
                                .saturating_sub(sp_cfg.nx / sp_cfg.refine_ratio);
                            let raw_shift = ((target_ox - cur_ox) / dx0_f).floor() as usize;
                            let avail = max_off_x.saturating_sub(self.levels[sli].spec.off_x);
                            let shift_p = raw_shift.min(avail);
                            if shift_p > 0 {
                                self.levels[sli].spec.off_x += shift_p;
                                self.levels[sli].spec.ox += (shift_p as f32) * dx0_f;
                                self.sync_d_levels(true)?;
                                let d_patch = self.levels[sli].descriptor();
                                let d_p0 = self.levels[0].descriptor();
                                unsafe {
                                    sim_cuda::check_cuda(
                                        sim_euler_shift_x(
                                            d_patch,
                                            d_p0,
                                            (shift_p * sp_cfg.refine_ratio) as i32,
                                            self.stream.raw(),
                                        ),
                                        "sim_euler_shift_x",
                                    )?;
                                }
                            }
                        }
                        if !handoff_done && pos_new[0] >= sp_cfg.handoff_x_end {
                            for li in 1..self.levels.len() {
                                if li == sli {
                                    continue;
                                }
                                let s = &self.levels[li].spec;
                                if pos_new[0] >= s.ox && pos_new[0] <= s.ox + s.nx as f32 * s.dx {
                                    active_mask[li] = true;
                                    let d_dst = self.levels[li].descriptor();
                                    let d_src = self.levels[sli].descriptor();
                                    unsafe {
                                        sim_cuda::check_cuda(
                                            sim_euler_copy_overlap(d_dst, d_src, self.stream.raw()),
                                            "sim_euler_copy_overlap_handoff",
                                        )?;
                                    }
                                    force_diag_stride_update = true;
                                }
                            }
                            sliding_active = false;
                            active_mask[sli] = false;
                            handoff_done = true;
                            self.sync_d_levels(false)?;
                        }
                    }
                }

                if pos_new[0] >= pspec.impact_x_max {
                    proj_states[pi].finished = true;
                    proj_states[pi].active = false;
                    // Physical bullet momentum impact onto target barrier/plate
                    if let (Some(tbid), Some(ref mut pts)) = (pspec.impact_target_body_id, &mut self.particles) {
                        self.stream.sync()?;
                        let px_h = pts.px.to_vec()?;
                        let py_h = pts.py.to_vec()?;
                        let pz_h = pts.pz.to_vec()?;
                        let mass_h = pts.mass.to_vec()?;
                        let bid_h = pts.body_id.to_vec()?;
                        let flag_h = pts.flags.to_vec()?;
                        let mut vx_h = pts.vx.to_vec()?;
                        let mut vy_h = pts.vy.to_vec()?;
                        let mut vz_h = pts.vz.to_vec()?;
                        let r2 = pspec.impact_radius_m * pspec.impact_radius_m;
                        let mut hits = Vec::new();
                        let mut m_tot = 0.0_f64;
                        for idx_p in 0..pts.count {
                            if bid_h[idx_p] == tbid && (flag_h[idx_p] & 1) == 0 {
                                let d2 = (py_h[idx_p] - pos_new[1]).powi(2)
                                    + (pz_h[idx_p] - pos_new[2]).powi(2)
                                    + (px_h[idx_p] - pspec.impact_x_max).powi(2);
                                if d2 <= r2 {
                                    hits.push(idx_p);
                                    m_tot += mass_h[idx_p] as f64;
                                }
                            }
                        }
                        if m_tot > 1e-12 {
                            let jx = (pspec.mass_kg as f64) * (vel_cur[0] as f64);
                            let jy = (pspec.mass_kg as f64) * (vel_cur[1] as f64);
                            let jz = (pspec.mass_kg as f64) * (vel_cur[2] as f64);
                            for idx_p in hits {
                                vx_h[idx_p] += (jx / m_tot) as f32;
                                vy_h[idx_p] += (jy / m_tot) as f32;
                                vz_h[idx_p] += (jz / m_tot) as f32;
                            }
                            pts.vx.upload(&vx_h)?;
                            pts.vy.upload(&vy_h)?;
                            pts.vz.upload(&vz_h)?;
                        }
                    }
                    continue;
                }

                // Apply bullet piston + drag on finest active level containing bullet.
                // When inside a fine AMR patch (li > 0), queue for smooth per-substep forcing inside
                // Step 5 so Mach-2+ projectiles advance ~0.15*dx per RK2 substep without probe-rate buzzing.
                let mut best_b_li = 0usize;
                for li in 1..self.levels.len() {
                    if !active_mask[li] {
                        continue;
                    }
                    let s = &self.levels[li].spec;
                    let m = 2.0 * s.dx;
                    if pos_new[0] >= s.ox + m
                        && pos_new[0] + pspec.length_m <= s.ox + s.nx as f32 * s.dx - m
                        && pos_new[1] >= s.oy + m
                        && pos_new[1] <= s.oy + s.ny as f32 * s.dx - m
                        && pos_new[2] >= s.oz + m
                        && pos_new[2] <= s.oz + s.nz as f32 * s.dx - m
                    {
                        best_b_li = li;
                    }
                }
                if best_b_li == 0 {
                    let d_lvl = self.levels[0].descriptor();
                    unsafe {
                        sim_cuda::check_cuda(
                            sim_euler_apply_projectile(
                                d_lvl,
                                self.fluid_const,
                                pos_old[0],
                                pos_old[1],
                                pos_old[2],
                                pos_new[0],
                                pos_new[1],
                                pos_new[2],
                                vel_cur[0],
                                vel_cur[1],
                                vel_cur[2],
                                pspec.radius_m,
                                pspec.length_m,
                                dt_f32,
                                self.d_proj_force.as_mut_ptr(),
                                self.stream.raw(),
                            ),
                            "sim_euler_apply_projectile",
                        )?;
                    }
                } else {
                    fine_proj_steps.push((
                        best_b_li,
                        pos_old,
                        vel_cur,
                        pspec.radius_m,
                        pspec.length_m,
                    ));
                }
                if active_idx % 8 == 0 {
                    let spd = ((vel_cur[0] * vel_cur[0]
                        + vel_cur[1] * vel_cur[1]
                        + vel_cur[2] * vel_cur[2]) as f64)
                        .sqrt()
                        .max(1e-3);
                    let area_front =
                        std::f64::consts::PI * (pspec.radius_m as f64) * (pspec.radius_m as f64);
                    let cd_g1 = 0.18_f64;
                    let a_drag = (-0.5 * self.fluid_const.rho_a * cd_g1 * area_front * spd * spd
                        / (pspec.mass_kg as f64).max(1e-6))
                    .clamp(-750.0, 0.0);
                    let dt8 = 8.0 * dt_probe;
                    let new_spd = (spd + a_drag * dt8).max(460.0);
                    let scale = (new_spd / spd) as f32;
                    proj_states[pi].vel[0] = vel_cur[0] * scale;
                    proj_states[pi].vel[1] = vel_cur[1] * scale;
                    proj_states[pi].vel[2] = vel_cur[2] * scale;
                }
                proj_states[pi].pos = pos_new;
            }

            // 4. Wind inlets & directional audio sources
            for w in &self.scene.wind_inlets {
                if t_now >= w.start_time_s && t_now <= w.end_time_s {
                    let ramp = if w.ramp_s > 0.0 {
                        ((t_now - w.start_time_s) / w.ramp_s).clamp(0.0, 1.0) as f32
                    } else {
                        1.0
                    };
                    let relax = 0.55 * ramp;
                    for (li, lvl) in self.levels.iter_mut().enumerate() {
                        if !active_mask[li] {
                            continue;
                        }
                        let s = &lvl.spec;
                        if w.max[0] >= s.ox && w.min[0] <= s.ox + s.nx as f32 * s.dx {
                            let d_lvl = lvl.descriptor();
                            unsafe {
                                sim_cuda::check_cuda(
                                    sim_euler_apply_wind_inlet(
                                        d_lvl,
                                        self.fluid_const,
                                        w.min[0],
                                        w.min[1],
                                        w.min[2],
                                        w.max[0],
                                        w.max[1],
                                        w.max[2],
                                        w.velocity[0],
                                        w.velocity[1],
                                        w.velocity[2],
                                        w.turb_amp,
                                        t_now as f32,
                                        relax,
                                        self.stream.raw(),
                                    ),
                                    "sim_euler_apply_wind_inlet",
                                )?;
                            }
                        }
                    }
                }
            }
            for dsrc in &self.scene.directional_sources {
                let rate = dsrc.rate_at(t_now);
                if rate.abs() > 0.0 {
                    let dir = normalize3(dsrc.direction);
                    for (li, lvl) in self.levels.iter_mut().enumerate() {
                        if !active_mask[li] {
                            continue;
                        }
                        let d_lvl = lvl.descriptor();
                        unsafe {
                            sim_cuda::check_cuda(
                                sim_euler_apply_directional_source(
                                    d_lvl,
                                    self.fluid_const,
                                    dsrc.center[0],
                                    dsrc.center[1],
                                    dsrc.center[2],
                                    dir[0],
                                    dir[1],
                                    dir[2],
                                    dsrc.cardioid_beta,
                                    dsrc.radius,
                                    rate,
                                    dt_probe as f32,
                                    self.stream.raw(),
                                ),
                                "sim_euler_apply_directional_source",
                            )?;
                        }
                    }
                }
            }

            // 5. Advance active fine patches by dt_probe (with smooth per-substep projectile forcing)
            for li in 1..self.levels.len() {
                if !active_mask[li] {
                    continue;
                }
                let kf = fine_k[li].max(1);
                let dt_f = (dt_probe / (kf as f64)) as f32;
                let p_idx = self.level_parents[li];
                let d_parent = self.levels[p_idx].descriptor();
                let d_child = self.levels[li].descriptor();
                for sub in 0..kf {
                    for &(p_li, p_pos0, p_vel, p_rad, p_len) in &fine_proj_steps {
                        if p_li != li {
                            continue;
                        }
                        let t0_sub = (sub as f32) * dt_f;
                        let t1_sub = ((sub + 1) as f32) * dt_f;
                        unsafe {
                            sim_cuda::check_cuda(
                                sim_euler_apply_projectile(
                                    d_child,
                                    self.fluid_const,
                                    p_pos0[0] + p_vel[0] * t0_sub,
                                    p_pos0[1] + p_vel[1] * t0_sub,
                                    p_pos0[2] + p_vel[2] * t0_sub,
                                    p_pos0[0] + p_vel[0] * t1_sub,
                                    p_pos0[1] + p_vel[1] * t1_sub,
                                    p_pos0[2] + p_vel[2] * t1_sub,
                                    p_vel[0],
                                    p_vel[1],
                                    p_vel[2],
                                    p_rad,
                                    p_len,
                                    dt_f,
                                    self.d_proj_force.as_mut_ptr(),
                                    self.stream.raw(),
                                ),
                                "sim_euler_apply_projectile_substep",
                            )?;
                        }
                    }
                    unsafe {
                        sim_cuda::check_cuda(
                            sim_euler_amr_fill_ghosts(d_child, d_parent, 0, self.stream.raw()),
                            "sim_euler_amr_fill_ghosts_s0",
                        )?;
                        sim_cuda::check_cuda(
                            sim_euler_rk_stage(d_child, self.fluid_const, 0, dt_f, self.stream.raw()),
                            "sim_euler_rk_stage_fine0",
                        )?;
                        sim_cuda::check_cuda(
                            sim_euler_amr_fill_ghosts(d_child, d_parent, 1, self.stream.raw()),
                            "sim_euler_amr_fill_ghosts_s1",
                        )?;
                        sim_cuda::check_cuda(
                            sim_euler_rk_stage(d_child, self.fluid_const, 1, dt_f, self.stream.raw()),
                            "sim_euler_rk_stage_fine1",
                        )?;
                    }
                }
                total_substeps += kf as u64;
            }

            // 6. Adaptive subcycled coarse Level 0 advance
            steps_since_coarse += 1;
            dt_since_coarse += dt_probe;
            let flush_coarse = steps_since_coarse >= coarse_stride
                || dt_since_coarse + dt_probe > 0.92 * dt0_lim
                || active_idx + 2 >= active_samples;
            if flush_coarse {
                for li in (1..self.levels.len()).rev() {
                    if !active_mask[li] {
                        continue;
                    }
                    let p_idx = self.level_parents[li];
                    let d_parent = self.levels[p_idx].descriptor();
                    let d_child = self.levels[li].descriptor();
                    unsafe {
                        sim_cuda::check_cuda(
                            sim_euler_amr_restrict(d_child, d_parent, self.stream.raw()),
                            "sim_euler_amr_restrict",
                        )?;
                    }
                }
                let d_lvl0 = self.levels[0].descriptor();
                let dt0_f32 = dt_since_coarse as f32;
                unsafe {
                    sim_cuda::check_cuda(
                        sim_euler_rk_stage(d_lvl0, self.fluid_const, 0, dt0_f32, self.stream.raw()),
                        "sim_euler_rk_stage_0",
                    )?;
                    sim_cuda::check_cuda(
                        sim_euler_rk_stage(d_lvl0, self.fluid_const, 1, dt0_f32, self.stream.raw()),
                        "sim_euler_rk_stage_1",
                    )?;
                }
                steps_since_coarse = 0;
                dt_since_coarse = 0.0;
                total_substeps += 1;
            }

            // 7. Sample probes
            if num_probes > 0 {
                unsafe {
                    sim_cuda::check_cuda(
                        sim_euler_sample_probes(
                            self.d_levels.as_ptr(),
                            self.levels.len() as i32,
                            self.d_probes.as_ptr(),
                            num_probes as i32,
                            d_probe_out.as_mut_ptr(),
                            (active_idx + 1) as i32,
                            self.fluid_const,
                            self.stream.raw(),
                        ),
                        "sim_euler_sample_probes",
                    )?;
                }
            }
        }

        self.stream.sync()?;
        device_sync()?;
        let final_m0 = self.reduce_level_metrics(0)?;
        let final_sm = self.reduce_mpm_metrics()?;
        let (drv_free, drv_total) = self.vram.query_driver_mem_info()?;
        peak_driver_used = peak_driver_used.max(drv_total.saturating_sub(drv_free));
        let wall_time_s = wall_start.elapsed().as_secs_f64();

        self.finalize_run_outputs(
            cfg,
            actual_pre_roll_s,
            total_samples,
            pre_roll_samples,
            active_samples,
            &d_probe_out,
            final_m0,
            final_sm,
            peak_driver_used,
            wall_time_s,
            total_substeps,
            timeseries,
            render_exporter,
            effective_render_cfg.as_ref(),
            run_auto_video_render,
        )
    }
}
