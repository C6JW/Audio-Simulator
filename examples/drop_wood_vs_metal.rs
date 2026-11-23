use std::collections::BTreeMap;
use std::path::Path;

use sim_core::*;
use sim_cuda::DEFAULT_VRAM_CAP_BYTES;
use sim_io::{write_probe_wav_f32, write_telemetry_json, write_timeseries_csv, RunTelemetry};

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

    // Edge-clamped 120 x 80 x 5 mm plate (body_id = 0)
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

    // Identical 8 mm radius steel impact sphere (body_id = 1) starting 1.5 mm above plate top
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

fn main() -> Result<()> {
    let out_dir = Path::new("output");
    std::fs::create_dir_all(out_dir)?;

    let cfg = SimRunConfig {
        duration_s: 0.12,
        sample_rate: 48_000,
        oversample_factor: 4,
        telemetry_interval_s: 0.002,
        output_dir: None,
        water_surface_z_for_diag: None,
        use_cuda_graph: true,
        cfl_safety: 0.38,
        print_progress: true,
        render_export: None,
        ..Default::default()
    };

    println!("=== Running Drop Test (a): Orthotropic Oak Wood ===");
    let mut eng_wood = MultiphysicsEngine::new(
        build_drop_scene("drop_wood", SolidMaterial::oak_wood()),
        None,
    )?;
    let res_wood = eng_wood.run(&cfg)?;
    let wav_wood = write_probe_wav_f32(
        out_dir.join("drop_wood.wav"),
        &res_wood.raw_probe_buffers[0],
        cfg.oversample_factor,
        cfg.sample_rate,
        20.0,
    )?;
    drop(eng_wood);

    println!("=== Running Drop Test (b): Isotropic Structural Steel ===");
    let mut eng_steel = MultiphysicsEngine::new(
        build_drop_scene("drop_steel", SolidMaterial::structural_steel()),
        None,
    )?;
    let res_steel = eng_steel.run(&cfg)?;
    let wav_steel = write_probe_wav_f32(
        out_dir.join("drop_steel.wav"),
        &res_steel.raw_probe_buffers[0],
        cfg.oversample_factor,
        cfg.sample_rate,
        20.0,
    )?;
    drop(eng_steel);

    let mut w_probe = res_wood.telemetry.probes[0].clone();
    w_probe.name = "wood_mic".to_string();
    w_probe.wav = Some(wav_wood);

    let mut s_probe = res_steel.telemetry.probes[0].clone();
    s_probe.name = "steel_mic".to_string();
    s_probe.wav = Some(wav_steel);

    let tail_ratio_steel_over_wood =
        s_probe.decay_tail_ratio / w_probe.decay_tail_ratio.max(1e-12);

    let mut notes = BTreeMap::new();
    notes.insert("wood_peak_freq_hz".to_string(), w_probe.peak_freq_hz);
    notes.insert("steel_peak_freq_hz".to_string(), s_probe.peak_freq_hz);
    notes.insert("wood_centroid_hz".to_string(), w_probe.spectral_centroid_hz);
    notes.insert("steel_centroid_hz".to_string(), s_probe.spectral_centroid_hz);
    notes.insert("wood_decay_tail_ratio".to_string(), w_probe.decay_tail_ratio);
    notes.insert("steel_decay_tail_ratio".to_string(), s_probe.decay_tail_ratio);
    notes.insert("steel_to_wood_tail_ratio".to_string(), tail_ratio_steel_over_wood);
    notes.insert("wood_wall_time_s".to_string(), res_wood.telemetry.wall_time_s);
    notes.insert("steel_wall_time_s".to_string(), res_steel.telemetry.wall_time_s);

    let combined_tele = RunTelemetry {
        scenario: "drop_wood_vs_metal".to_string(),
        compute_capability: res_steel.telemetry.compute_capability.clone(),
        total_vram_bytes: res_steel.telemetry.total_vram_bytes,
        vram_cap_bytes: DEFAULT_VRAM_CAP_BYTES,
        peak_vram_tracked_bytes: res_wood
            .telemetry
            .peak_vram_tracked_bytes
            .max(res_steel.telemetry.peak_vram_tracked_bytes),
        peak_vram_driver_used_bytes: res_wood
            .telemetry
            .peak_vram_driver_used_bytes
            .max(res_steel.telemetry.peak_vram_driver_used_bytes),
        sim_duration_s: cfg.duration_s,
        wall_time_s: res_wood.telemetry.wall_time_s + res_steel.telemetry.wall_time_s,
        total_substeps: res_wood.telemetry.total_substeps + res_steel.telemetry.total_substeps,
        nan_count: res_wood.telemetry.nan_count + res_steel.telemetry.nan_count,
        initial_water_volume: 0.0,
        final_water_volume: 0.0,
        water_volume_rel_error: 0.0,
        initial_total_energy: res_steel.telemetry.initial_total_energy,
        final_total_energy: res_steel.telemetry.final_total_energy,
        energy_conservation_error: res_steel.telemetry.energy_conservation_error,
        probes: vec![w_probe.clone(), s_probe.clone()],
        notes,
    };

    write_telemetry_json(out_dir.join("drop_wood_vs_metal_telemetry.json"), &combined_tele)?;
    write_timeseries_csv(
        out_dir.join("drop_wood_vs_metal_wood_timeseries.csv"),
        &res_wood.timeseries,
    )?;
    write_timeseries_csv(
        out_dir.join("drop_wood_vs_metal_steel_timeseries.csv"),
        &res_steel.timeseries,
    )?;
    write_timeseries_csv(
        out_dir.join("drop_wood_vs_metal_telemetry.csv"),
        &res_steel.timeseries,
    )?;

    println!("\n=== Validation Test 1 Results: drop_wood_vs_metal ===");
    println!(
        "Wood : peak={:.3} Pa, f_peak={:.1} Hz, centroid={:.1} Hz, tail_ratio={:.4}, wall={:.2}s",
        w_probe.peak_pa,
        w_probe.peak_freq_hz,
        w_probe.spectral_centroid_hz,
        w_probe.decay_tail_ratio,
        res_wood.telemetry.wall_time_s
    );
    println!(
        "Steel: peak={:.3} Pa, f_peak={:.1} Hz, centroid={:.1} Hz, tail_ratio={:.4}, wall={:.2}s",
        s_probe.peak_pa,
        s_probe.peak_freq_hz,
        s_probe.spectral_centroid_hz,
        s_probe.decay_tail_ratio,
        res_steel.telemetry.wall_time_s
    );
    println!(
        "Steel/Wood decay tail ratio = {:.2}x | Peak VRAM = {:.2} MB | NaNs = {}",
        tail_ratio_steel_over_wood,
        (combined_tele.peak_vram_tracked_bytes as f64) / 1e6,
        combined_tele.nan_count
    );

    assert_eq!(combined_tele.nan_count, 0);
    assert!(combined_tele.peak_vram_tracked_bytes < DEFAULT_VRAM_CAP_BYTES);
    assert!(w_probe.peak_pa > 0.05 && s_probe.peak_pa > 0.05);
    assert!(
        tail_ratio_steel_over_wood >= 1.8,
        "Expected steel decay tail >= 1.8x oak, got {tail_ratio_steel_over_wood:.2}x"
    );

    Ok(())
}
