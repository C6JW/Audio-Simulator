use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use sim_core::*;
use sim_io::{
    compose_showcase_reel, compose_side_by_side_comparison, mux_video_with_audio,
    render_frames_with_blender, render_preview_frames_with_blender, AudioSlowmoMode,
    RenderManifest, VideoRenderOptions,
};

struct CliOpts {
    target: String,
    preview_only: bool,
    skip_sim: bool,
}

fn parse_cli() -> CliOpts {
    let mut target = "all".to_string();
    let mut preview_only = false;
    let mut skip_sim = false;
    for arg in env::args().skip(1) {
        if arg == "--preview-only" {
            preview_only = true;
        } else if arg == "--skip-sim" {
            skip_sim = true;
        } else if !arg.starts_with('-') {
            target = arg;
        }
    }
    CliOpts {
        target,
        preview_only,
        skip_sim,
    }
}

fn build_drop_scene(name: &str, plate_material: SolidMaterial) -> Scene {
    let mut scene = Scene::minimal(
        name,
        AmrDomainConfig {
            origin: [-0.160, -0.160, -0.112],
            dx0: 0.004,
            nx0: 80,
            ny0: 80,
            nz0: 56,
            pml_thickness: 10,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![AmrPatchConfig {
                refine_ratio: 2,
                off_x: 22,
                off_y: 27,
                off_z: 20,
                nx: 72,
                ny: 52,
                nz: 32,
                ..Default::default()
            }],
        },
    );
    scene.mpm_grid = Some(MpmGridConfig {
        origin: [-0.072, -0.052, -0.024],
        dx: 0.002,
        nx: 72,
        ny: 52,
        nz: 44,
        has_floor: false,
        floor_z: -0.10,
        floor_mu: 0.3,
    });
    scene.solids.push(SolidBodySpec {
        name: "plate".to_string(),
        body_id: 0,
        shape: GeometryShape::Box {
            min: [-0.060, -0.040, -0.0025],
            max: [0.060, 0.040, 0.0025],
        },
        material: plate_material,
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.001,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![
            SolidBoundaryCondition::ClampedRegion {
                min: [-0.062, -0.042, -0.004],
                max: [-0.054, 0.042, 0.004],
            },
            SolidBoundaryCondition::ClampedRegion {
                min: [0.054, -0.042, -0.004],
                max: [0.062, 0.042, 0.004],
            },
        ],
    });
    scene.solids.push(SolidBodySpec {
        name: "sphere".to_string(),
        body_id: 1,
        shape: GeometryShape::Sphere {
            center: [0.0, 0.0, 0.0120],
            radius: 0.008,
        },
        material: SolidMaterial::structural_steel(),
        initial_velocity: [0.0, 0.0, -1.5],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.001,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![],
    });
    scene.microphones.push(MicrophoneSpec {
        name: "mic".to_string(),
        position: [0.0, 0.018, 0.025],
        is_hydrophone: false,
    });
    scene
}

fn build_whispering_gallery_scene() -> Scene {
    let r_inner = 0.360_f32;
    let r_outer = 0.400_f32;
    let height = 0.100_f32;
    let z_mid = 0.5 * height;

    let mut scene = Scene::minimal(
        "whispering_gallery",
        AmrDomainConfig {
            origin: [-0.430, -0.430, -0.030],
            dx0: 0.0025,
            nx0: 344,
            ny0: 344,
            nz0: 64,
            pml_thickness: 8,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![],
        },
    );
    scene.fluid.gravity = [0.0, 0.0, 0.0];

    let stone = SolidMaterial::granite_stone();
    scene.solids.push(SolidBodySpec {
        name: "stone_wall".to_string(),
        body_id: 0,
        shape: GeometryShape::CylindricalRing {
            center_xy: [0.0, 0.0],
            r_inner,
            r_outer,
            z_min: 0.0,
            z_max: height,
        },
        material: stone.clone(),
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.0025,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![],
    });
    scene.solids.push(SolidBodySpec {
        name: "stone_floor".to_string(),
        body_id: 0,
        shape: GeometryShape::Disc {
            center_xy: [0.0, 0.0],
            radius: r_outer,
            z_min: -0.015,
            z_max: 0.0,
        },
        material: stone.clone(),
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.0025,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![],
    });
    scene.solids.push(SolidBodySpec {
        name: "stone_ceiling".to_string(),
        body_id: 0,
        shape: GeometryShape::Disc {
            center_xy: [0.0, 0.0],
            radius: r_outer,
            z_min: height,
            z_max: height + 0.015,
        },
        material: stone,
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.0025,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![],
    });
    scene.point_sources.push(PointRickerSourceSpec {
        center: [r_inner - 0.015, 0.0, z_mid],
        radius: 0.0075,
        peak_rate_pa_per_s: 8.0e6,
        f0_hz: 6000.0,
        t0_s: 0.00035,
    });
    scene.microphones.push(MicrophoneSpec {
        name: "mic_180_wall".to_string(),
        position: [-(r_inner - 0.015), 0.0, z_mid],
        is_hydrophone: false,
    });
    scene.microphones.push(MicrophoneSpec {
        name: "mic_90_wall".to_string(),
        position: [0.0, r_inner - 0.015, z_mid],
        is_hydrophone: false,
    });
    scene.microphones.push(MicrophoneSpec {
        name: "mic_center".to_string(),
        position: [0.0, 0.0, z_mid],
        is_hydrophone: false,
    });
    scene.microphones.push(MicrophoneSpec {
        name: "mic_mid_radial".to_string(),
        position: [-0.5 * r_inner, 0.0, z_mid],
        is_hydrophone: false,
    });
    scene
}

fn build_water_drop_plop_scene() -> Scene {
    let r_drop = 0.0015_f32;
    let mut scene = Scene::minimal(
        "water_drop_plop",
        AmrDomainConfig {
            origin: [-0.128, -0.128, -0.128],
            dx0: 0.0032,
            nx0: 80,
            ny0: 80,
            nz0: 80,
            pml_thickness: 8,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![
                AmrPatchConfig {
                    refine_ratio: 4,
                    off_x: 32,
                    off_y: 32,
                    off_z: 32,
                    nx: 64,
                    ny: 64,
                    nz: 64,
                    ..Default::default()
                },
                AmrPatchConfig {
                    refine_ratio: 4,
                    off_x: 24,
                    off_y: 24,
                    off_z: 18,
                    nx: 64,
                    ny: 64,
                    nz: 84,
                    ..Default::default()
                },
            ],
        },
    );
    scene.background_water = vec![GeometryShape::HalfSpaceZ { z_max: 0.0 }];
    scene.enable_flow_advection = true;
    scene.dynamic_fluid.push(DynamicFluidRegion {
        shape: GeometryShape::Sphere {
            center: [0.0, 0.0, 0.0020],
            radius: r_drop,
        },
        velocity: [0.0, 0.0, -1.65],
        is_water: true,
        laplace_overpressure_pa: 2.0 * WATER_SIGMA / (r_drop as f64),
    });
    scene.microphones.push(MicrophoneSpec {
        name: "air_mic".to_string(),
        position: [0.0, 0.0, 0.045],
        is_hydrophone: false,
    });
    scene.microphones.push(MicrophoneSpec {
        name: "hydrophone".to_string(),
        position: [0.0, 0.0, -0.045],
        is_hydrophone: true,
    });
    scene
}

fn build_rip_paper_scene() -> Scene {
    let size_x = 0.056_f32;
    let size_y = 0.044_f32;
    let notch_x_len = 0.010_f32;

    let mut scene = Scene::minimal(
        "rip_paper",
        AmrDomainConfig {
            origin: [-0.08, -0.08, -0.08],
            dx0: 0.0025,
            nx0: 64,
            ny0: 64,
            nz0: 64,
            pml_thickness: 8,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![],
        },
    );
    scene.mpm_grid = Some(MpmGridConfig {
        origin: [-0.042, -0.040, -0.036],
        dx: 0.002,
        nx: 42,
        ny: 40,
        nz: 36,
        has_floor: false,
        floor_z: -1.0,
        floor_mu: 0.0,
    });
    scene.solids.push(SolidBodySpec {
        name: "cellulose_sheet".to_string(),
        body_id: 0,
        shape: GeometryShape::CurvedShell {
            center: [0.0, 0.0, 0.0],
            size_x,
            size_y,
            nx: 32,
            ny: 24,
            dome_amp: 0.0,
            saddle_amp: 0.0,
            yaw_rad: 0.0,
            notch_x_len,
            notch_y_pos: 0.0,
            notch_half_w: 0.0018,
        },
        material: SolidMaterial::cellulose_paper(1.1e-4),
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.0018,
        gc_heterogeneity_amp: 0.28,
        is_kinematic_rigid: false,
        bcs: vec![
            SolidBoundaryCondition::PrescribedVelocityRegion {
                min: [-0.035, 0.002, -0.01],
                max: [-0.020, 0.030, 0.01],
                velocity: [0.0, 0.25, 0.45],
            },
            SolidBoundaryCondition::PrescribedVelocityRegion {
                min: [-0.035, -0.030, -0.01],
                max: [-0.020, -0.002, 0.01],
                velocity: [0.0, -0.25, -0.45],
            },
            SolidBoundaryCondition::ClampedRegion {
                min: [0.024, -0.030, -0.01],
                max: [0.035, 0.030, 0.01],
            },
        ],
    });
    scene.microphones.push(MicrophoneSpec {
        name: "rip_paper".to_string(),
        position: [0.0, 0.0, 0.040],
        is_hydrophone: false,
    });
    scene
}

fn build_wet_leaves_scene() -> Scene {
    let mut scene = Scene::minimal(
        "drop_on_wet_leaves",
        AmrDomainConfig {
            origin: [-0.08, -0.08, -0.02],
            dx0: 0.0025,
            nx0: 64,
            ny0: 64,
            nz0: 64,
            pml_thickness: 8,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![],
        },
    );
    scene.mpm_grid = Some(MpmGridConfig {
        origin: [-0.044, -0.044, -0.004],
        dx: 0.0025,
        nx: 36,
        ny: 36,
        nz: 20,
        has_floor: true,
        floor_z: 0.0,
        floor_mu: 0.4,
    });

    let leaf_z = [0.002_f32, 0.0055, 0.009, 0.0125, 0.016, 0.0195];
    let leaf_yaw = [0.0_f32, 0.55, -0.45, 1.10, -0.95, 0.25];
    let leaf_dome = [0.0035_f32, 0.0042, 0.0038, 0.0048, 0.0040, 0.0050];
    let leaf_saddle = [0.0015_f32, -0.0020, 0.0022, -0.0018, 0.0025, -0.0022];
    let leaf_xy = [
        [0.000_f32, 0.000],
        [0.004, -0.003],
        [-0.004, 0.003],
        [0.002, 0.004],
        [-0.003, -0.003],
        [0.001, -0.001],
    ];
    for idx in 0..6 {
        scene.solids.push(SolidBodySpec {
            name: format!("wet_leaf_{idx}"),
            body_id: (idx + 1) as i32,
            shape: GeometryShape::CurvedShell {
                center: [leaf_xy[idx][0], leaf_xy[idx][1], leaf_z[idx]],
                size_x: 0.044,
                size_y: 0.032,
                nx: 14,
                ny: 10,
                dome_amp: leaf_dome[idx],
                saddle_amp: leaf_saddle[idx],
                yaw_rad: leaf_yaw[idx],
                notch_x_len: 0.0,
                notch_y_pos: 0.0,
                notch_half_w: 0.0,
            },
            material: SolidMaterial::wet_leaf(2.8e-4, 0.85),
            initial_velocity: [0.0, 0.0, 0.0],
            orientation_euler_xyz: [0.0, 0.0, 0.0],
            particle_spacing: 0.0032,
            gc_heterogeneity_amp: 0.25,
            is_kinematic_rigid: false,
            bcs: vec![],
        });
    }
    scene.solids.push(SolidBodySpec {
        name: "steel_plate".to_string(),
        body_id: 0,
        shape: GeometryShape::Box {
            min: [-0.026, -0.022, 0.025],
            max: [0.026, 0.022, 0.030],
        },
        material: SolidMaterial::structural_steel(),
        initial_velocity: [0.0, 0.0, -1.8],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.0025,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![],
    });
    scene.microphones.push(MicrophoneSpec {
        name: "drop_on_wet_leaves".to_string(),
        position: [0.0, 0.0, 0.055],
        is_hydrophone: false,
    });
    scene
}

fn read_manifest(path: &Path) -> Result<RenderManifest> {
    let f = File::open(path)?;
    Ok(serde_json::from_reader(f).map_err(|e| EngineError::Io(std::io::Error::other(e)))?)
}

#[allow(clippy::too_many_arguments)]
fn run_render_job(
    scene_opt: Option<Scene>,
    cache_key: &str,
    mut base_cfg: SimRunConfig,
    num_frames: usize,
    export_fluid_isosurface: bool,
    export_pressure_slice: bool,
    pressure_slice_z: f32,
    samples: u32,
    audio_mode: AudioSlowmoMode,
    preview_frames: &[usize],
    outputs: &[(usize, &str)],
    cli: &CliOpts,
) -> Result<()> {
    let cache_dir = PathBuf::from("output/render_cache").join(cache_key);
    let data_dir = cache_dir.join("data");
    let png_dir = cache_dir.join("png");
    let wav_dir = cache_dir.join("wav");
    let previews_dir = PathBuf::from("output/videos/previews");
    fs::create_dir_all(&previews_dir)?;

    let manifest_path = data_dir.join("manifest.json");
    if !cli.skip_sim {
        let scene = scene_opt.expect("scene required when --skip-sim is false");
        fs::create_dir_all(&wav_dir)?;
        base_cfg.output_dir = Some(wav_dir.clone());
        base_cfg.render_export = Some(RenderExportConfig {
            frames_dir: data_dir.clone(),
            num_frames,
            export_fluid_isosurface,
            export_pressure_slice,
            pressure_slice_z,
        });
        println!("\n=== [{cache_key}] Running CUDA Multiphysics Simulation + Mesh Export ({num_frames} frames) ===");
        let mut engine = MultiphysicsEngine::new(scene, None)?;
        let res = engine.run(&base_cfg)?;
        assert_eq!(res.telemetry.nan_count, 0, "{cache_key} produced NaNs!");
        // Drop CUDA device buffers cleanly BEFORE Cycles OptiX opens
        drop(engine);
    } else {
        println!("\n=== [{cache_key}] Skipping CUDA simulation; reusing {} ===", manifest_path.display());
    }

    let manifest = read_manifest(&manifest_path)?;
    let opts = VideoRenderOptions {
        samples,
        audio_mode,
        ..Default::default()
    };

    if cli.preview_only {
        println!("[{cache_key}] Rendering {} Blender Cycles OptiX preview frames...", preview_frames.len());
        render_preview_frames_with_blender(&manifest_path, &png_dir, &opts, preview_frames)
            .map_err(|e| EngineError::Io(std::io::Error::other(e)))?;
        for &pf in preview_frames {
            let src = png_dir.join(format!("frame_{pf:04}.png"));
            let dst = previews_dir.join(format!("{cache_key}_frame_{pf:04}.png"));
            fs::copy(&src, &dst)?;
        }
        println!("[{cache_key}] Preview PNGs saved into output/videos/previews/");
        return Ok(());
    }

    println!("[{cache_key}] Rendering all {} Blender Cycles OptiX frames ({samples} spp)...", manifest.num_frames);
    render_frames_with_blender(&manifest_path, &png_dir, &opts)
        .map_err(|e| EngineError::Io(std::io::Error::other(e)))?;

    for &pf in preview_frames {
        let src = png_dir.join(format!("frame_{pf:04}.png"));
        if src.exists() {
            let dst = previews_dir.join(format!("{cache_key}_frame_{pf:04}.png"));
            fs::copy(&src, &dst)?;
        }
    }

    for &(mic_idx, out_mp4) in outputs {
        let wav_path = manifest.microphones[mic_idx]
            .wav_path
            .as_ref()
            .expect("missing mic wav_path in manifest");
        println!("[{cache_key}] Muxing {out_mp4} with {wav_path}...");
        mux_video_with_audio(
            &png_dir,
            wav_path,
            manifest.duration_s,
            manifest.num_frames,
            out_mp4,
            &opts,
        )
        .map_err(|e| EngineError::Io(std::io::Error::other(e)))?;
    }

    Ok(())
}

fn run_rip_paper(cli: &CliOpts) -> Result<()> {
    let cfg = SimRunConfig {
        duration_s: 0.060,
        sample_rate: 48_000,
        oversample_factor: 4,
        telemetry_interval_s: 0.002,
        use_cuda_graph: false,
        cfl_safety: 0.38,
        print_progress: true,
        ..Default::default()
    };
    run_render_job(
        (!cli.skip_sim).then(build_rip_paper_scene),
        "rip_paper",
        cfg,
        120,
        false,
        false,
        0.0,
        64,
        AudioSlowmoMode::PhysicalRateScale,
        &[0, 45, 95],
        &[(0, "output/videos/04_rip_paper.mp4")],
        cli,
    )
}

fn run_drop_on_wet_leaves(cli: &CliOpts) -> Result<()> {
    let cfg = SimRunConfig {
        duration_s: 0.055,
        sample_rate: 48_000,
        oversample_factor: 4,
        telemetry_interval_s: 0.002,
        use_cuda_graph: false,
        cfl_safety: 0.38,
        print_progress: true,
        ..Default::default()
    };
    run_render_job(
        (!cli.skip_sim).then(build_wet_leaves_scene),
        "drop_on_wet_leaves",
        cfg,
        120,
        false,
        false,
        0.0,
        64,
        AudioSlowmoMode::PitchPreservingAtempo,
        &[0, 35, 90],
        &[(0, "output/videos/05_drop_on_wet_leaves.mp4")],
        cli,
    )
}

fn run_whispering_gallery(cli: &CliOpts) -> Result<()> {
    let cfg = SimRunConfig {
        duration_s: 0.025,
        sample_rate: 48_000,
        oversample_factor: 4,
        telemetry_interval_s: 0.001,
        use_cuda_graph: false,
        cfl_safety: 0.38,
        print_progress: true,
        ..Default::default()
    };
    run_render_job(
        (!cli.skip_sim).then(build_whispering_gallery_scene),
        "whispering_gallery",
        cfg,
        120,
        false,
        true,
        0.05,
        64,
        AudioSlowmoMode::PhysicalRateScale,
        &[12, 25, 60],
        &[(0, "output/videos/02_whispering_gallery.mp4")],
        cli,
    )
}

fn run_drop_wood_vs_metal(cli: &CliOpts) -> Result<()> {
    let cfg = SimRunConfig {
        duration_s: 0.120,
        sample_rate: 48_000,
        oversample_factor: 4,
        telemetry_interval_s: 0.002,
        use_cuda_graph: true,
        cfl_safety: 0.38,
        print_progress: true,
        ..Default::default()
    };
    run_render_job(
        (!cli.skip_sim).then(|| build_drop_scene("drop_wood", SolidMaterial::oak_wood())),
        "drop_wood",
        cfg.clone(),
        120,
        false,
        false,
        0.0,
        64,
        AudioSlowmoMode::PitchPreservingAtempo,
        &[0, 20, 60],
        &[(0, "output/videos/01_drop_wood.mp4")],
        cli,
    )?;
    run_render_job(
        (!cli.skip_sim).then(|| build_drop_scene("drop_steel", SolidMaterial::structural_steel())),
        "drop_steel",
        cfg,
        120,
        false,
        false,
        0.0,
        64,
        AudioSlowmoMode::PitchPreservingAtempo,
        &[0, 20, 60],
        &[(0, "output/videos/01_drop_steel.mp4")],
        cli,
    )?;
    if !cli.preview_only {
        println!("[drop_wood_vs_metal] Composing side-by-side comparison video...");
        compose_side_by_side_comparison(
            "output/videos/01_drop_wood.mp4",
            "output/videos/01_drop_steel.mp4",
            "output/videos/01_drop_wood_vs_metal.mp4",
        )
        .map_err(|e| EngineError::Io(std::io::Error::other(e)))?;
    }
    Ok(())
}

fn run_water_drop_plop(cli: &CliOpts) -> Result<()> {
    let cfg = SimRunConfig {
        duration_s: 0.025,
        sample_rate: 96_000,
        oversample_factor: 2,
        telemetry_interval_s: 0.0005,
        water_surface_z_for_diag: Some(0.0),
        use_cuda_graph: true,
        cfl_safety: 0.45,
        print_progress: true,
        ..Default::default()
    };
    run_render_job(
        (!cli.skip_sim).then(build_water_drop_plop_scene),
        "water_drop_plop",
        cfg,
        125,
        true,
        false,
        0.0,
        72,
        AudioSlowmoMode::PhysicalRateScale,
        &[0, 65, 103, 115],
        &[
            (0, "output/videos/03_water_drop_plop_air_mic.mp4"),
            (1, "output/videos/03_water_drop_plop_hydrophone.mp4"),
        ],
        cli,
    )
}

fn run_showcase() -> Result<()> {
    let inputs = vec![
        PathBuf::from("output/videos/01_drop_wood_vs_metal.mp4"),
        PathBuf::from("output/videos/02_whispering_gallery.mp4"),
        PathBuf::from("output/videos/03_water_drop_plop_air_mic.mp4"),
        PathBuf::from("output/videos/04_rip_paper.mp4"),
        PathBuf::from("output/videos/05_drop_on_wet_leaves.mp4"),
    ];
    println!("[showcase] Composing output/videos/multiphysics_ground_truth_showcase.mp4...");
    compose_showcase_reel(&inputs, "output/videos/multiphysics_ground_truth_showcase.mp4")
        .map_err(|e| EngineError::Io(std::io::Error::other(e)))?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = parse_cli();
    fs::create_dir_all("output/videos")?;
    match cli.target.as_str() {
        "rip_paper" => run_rip_paper(&cli)?,
        "drop_on_wet_leaves" => run_drop_on_wet_leaves(&cli)?,
        "whispering_gallery" => run_whispering_gallery(&cli)?,
        "drop_wood_vs_metal" => run_drop_wood_vs_metal(&cli)?,
        "water_drop_plop" => run_water_drop_plop(&cli)?,
        "showcase" => run_showcase()?,
        "all" => {
            run_rip_paper(&cli)?;
            run_drop_on_wet_leaves(&cli)?;
            run_whispering_gallery(&cli)?;
            run_drop_wood_vs_metal(&cli)?;
            run_water_drop_plop(&cli)?;
            if !cli.preview_only {
                run_showcase()?;
            }
        }
        other => {
            return Err(EngineError::InvalidScene(format!(
                "Unknown render target '{other}'"
            )));
        }
    }
    Ok(())
}
