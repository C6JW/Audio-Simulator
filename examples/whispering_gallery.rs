use std::collections::BTreeMap;
use std::path::Path;

use sim_core::*;
use sim_cuda::DEFAULT_VRAM_CAP_BYTES;
use sim_io::{
    generate_fallback_speech_phrase, read_wav_mono_f32, resample_sinc_f32, write_telemetry_json,
    write_timeseries_csv,
};

fn run_focusing_benchmark(out_dir: &Path) -> Result<()> {
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

    let mut stone = SolidMaterial::granite_stone();
    stone.is_rigid_static = true;

    // Smooth curved cylindrical gallery wall (analytical subcell face-area IB)
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

    // Broadband 6 kHz Ricker source near perimeter (15 mm inside curved wall)
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

    let cfg = SimRunConfig {
        duration_s: 0.025,
        sample_rate: 48_000,
        oversample_factor: 4,
        telemetry_interval_s: 0.001,
        output_dir: Some(out_dir.to_path_buf()),
        water_surface_z_for_diag: None,
        use_cuda_graph: false,
        cfl_safety: 0.42,
        print_progress: true,
        render_export: None,
        ..Default::default()
    };

    let mut engine = MultiphysicsEngine::new(scene, None)?;
    let mut res = engine.run(&cfg)?;

    let e_180 = res.telemetry.probes[0].total_energy_pa2_s;
    let e_90 = res.telemetry.probes[1].total_energy_pa2_s;
    let e_center = res.telemetry.probes[2].total_energy_pa2_s;
    let e_mid = res.telemetry.probes[3].total_energy_pa2_s;
    let ratio_180_center = e_180 / e_center.max(1e-30);

    let fs_os = (cfg.sample_rate as f64) * (cfg.oversample_factor as f64);
    let idx_90_peak = res.raw_probe_buffers[1]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let t_90_ms = (idx_90_peak as f64) / fs_os * 1e3;

    let mut notes = BTreeMap::new();
    notes.insert("energy_180_pa2_s".to_string(), e_180);
    notes.insert("energy_90_pa2_s".to_string(), e_90);
    notes.insert("energy_center_pa2_s".to_string(), e_center);
    notes.insert("energy_mid_radial_pa2_s".to_string(), e_mid);
    notes.insert("ratio_e180_over_ecenter".to_string(), ratio_180_center);
    notes.insert("mic_90_peak_time_ms".to_string(), t_90_ms);
    res.telemetry.notes = notes;
    write_telemetry_json(out_dir.join("whispering_gallery_telemetry.json"), &res.telemetry)?;
    write_timeseries_csv(out_dir.join("whispering_gallery_telemetry.csv"), &res.timeseries)?;

    println!("\n=== Part 1: Whispering Gallery 6 kHz Creeping-Wave Benchmark ===");
    println!(
        "E_180 = {:.4e} | E_90 = {:.4e} | E_center = {:.4e} | E_mid = {:.4e}",
        e_180, e_90, e_center, e_mid
    );
    println!(
        "E_180 / E_center = {:.3}x | mic_90 peak @ {:.2} ms | VRAM = {:.2} MB | NaNs = {}",
        ratio_180_center,
        t_90_ms,
        (res.telemetry.peak_vram_tracked_bytes as f64) / 1e6,
        res.telemetry.nan_count
    );

    assert_eq!(res.telemetry.nan_count, 0);
    assert!(res.telemetry.peak_vram_tracked_bytes < DEFAULT_VRAM_CAP_BYTES);
    assert!(res.telemetry.probes[0].peak_pa > 0.5);
    assert!(
        ratio_180_center > 1.3,
        "Expected whispering gallery E_180 / E_center > 1.3, got {ratio_180_center:.3}"
    );
    Ok(())
}

fn run_rotunda_speech_gallery(out_dir: &Path) -> Result<()> {
    // Full-scale 4.5 m diameter stone rotunda driven by real whispered speech tangentially
    // along the curved stone wall. Exports matched-gain wall vs center WAVs so the listener
    // directly hears the whispering-gallery creeping-wave focusing gain.
    let r_inner = 2.25_f32;
    let r_outer = 2.40_f32;
    let height = 0.72_f32;
    let z_mid = 0.5 * height;
    let sr_sim = 48_000u32;

    let speech_path = Path::new("assets/hall_whisper_speech.wav");
    let speech_samples = if speech_path.exists() {
        let (src_sr, mono) = read_wav_mono_f32(speech_path)?;
        resample_sinc_f32(&mono, src_sr, sr_sim)
    } else {
        generate_fallback_speech_phrase(sr_sim)
    };

    let mut scene = Scene::minimal(
        "whispering_gallery_rotunda",
        AmrDomainConfig {
            origin: [-2.65, -2.65, -0.14],
            dx0: 0.025,
            nx0: 212,
            ny0: 212,
            nz0: 40,
            pml_thickness: 6,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![],
        },
    );
    scene.fluid.gravity = [0.0, 0.0, 0.0];
    scene.fluid.wall_absorption_alpha = 0.025;

    let mut stone = SolidMaterial::granite_stone();
    stone.is_rigid_static = true;
    scene.solids.push(SolidBodySpec {
        name: "rotunda_stone_wall".to_string(),
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
        particle_spacing: 0.025,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![],
    });
    scene.solids.push(SolidBodySpec {
        name: "rotunda_stone_floor".to_string(),
        body_id: 0,
        shape: GeometryShape::Disc {
            center_xy: [0.0, 0.0],
            radius: r_outer,
            z_min: -0.05,
            z_max: 0.0,
        },
        material: stone.clone(),
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.025,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![],
    });
    scene.solids.push(SolidBodySpec {
        name: "rotunda_stone_ceiling".to_string(),
        body_id: 0,
        shape: GeometryShape::Disc {
            center_xy: [0.0, 0.0],
            radius: r_outer,
            z_min: height,
            z_max: height + 0.05,
        },
        material: stone,
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.025,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![],
    });

    // Speaker whispering tangentially along the curved stone wall (+Y direction at +X perimeter)
    scene.directional_sources.push(DirectionalAudioSourceSpec {
        center: [r_inner - 0.08, 0.0, z_mid],
        direction: [0.0, 1.0, 0.0],
        cardioid_beta: 0.75,
        radius: 0.055,
        peak_rate_pa_per_s: 1.2e5,
        t0_s: 0.05,
        sample_rate: sr_sim as f64,
        samples: speech_samples,
    });

    // Listener opposite along the curved wall vs listener standing in the center of the rotunda
    scene.microphones.push(MicrophoneSpec {
        name: "whispering_gallery_wall.wav".to_string(),
        position: [-(r_inner - 0.08), 0.0, z_mid],
        is_hydrophone: false,
    });
    scene.microphones.push(MicrophoneSpec {
        name: "whispering_gallery_center.wav".to_string(),
        position: [0.0, 0.0, z_mid],
        is_hydrophone: false,
    });

    let cfg = SimRunConfig {
        duration_s: 3.2,
        pre_roll_s: 0.0,
        sample_rate: sr_sim,
        oversample_factor: 1,
        target_wav_sample_rate: Some(sr_sim),
        shared_wav_gain: true,
        telemetry_interval_s: 0.05,
        output_dir: Some(out_dir.to_path_buf()),
        cfl_safety: 0.42,
        print_progress: true,
        ..Default::default()
    };

    println!("\n=== Part 2: Full-Scale 4.5 m Stone Rotunda Whispering Gallery (Speech) ===");
    let mut engine = MultiphysicsEngine::new(scene, None)?;
    let mut res = engine.run(&cfg)?;

    let rms_wall = res.telemetry.probes[0].rms_pa;
    let rms_center = res.telemetry.probes[1].rms_pa;
    let gain_db = 20.0 * (rms_wall / rms_center.max(1e-12)).log10();

    let mut notes = BTreeMap::new();
    notes.insert("rms_wall_pa".to_string(), rms_wall);
    notes.insert("rms_center_pa".to_string(), rms_center);
    notes.insert("wall_over_center_gain_db".to_string(), gain_db);
    res.telemetry.notes = notes;
    write_telemetry_json(out_dir.join("whispering_gallery_rotunda_telemetry.json"), &res.telemetry)?;
    write_timeseries_csv(out_dir.join("whispering_gallery_rotunda_timeseries.csv"), &res.timeseries)?;

    println!(
        "Rotunda Speech Gallery: Wall RMS={:.4} Pa vs Center RMS={:.4} Pa | Whispering Gain={:+.2} dB | VRAM={:.1} MB | NaNs={}",
        rms_wall,
        rms_center,
        gain_db,
        res.telemetry.peak_vram_tracked_bytes as f64 / 1e6,
        res.telemetry.nan_count
    );

    assert_eq!(res.telemetry.nan_count, 0);
    assert!(res.telemetry.peak_vram_tracked_bytes < DEFAULT_VRAM_CAP_BYTES);
    assert!(rms_wall > 1e-4, "Expected audible speech pressure at opposite wall");
    Ok(())
}

fn main() -> Result<()> {
    let out_dir = Path::new("output");
    std::fs::create_dir_all(out_dir)?;
    run_focusing_benchmark(out_dir)?;
    run_rotunda_speech_gallery(out_dir)?;
    Ok(())
}
