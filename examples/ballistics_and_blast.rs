use std::collections::BTreeMap;
use std::path::Path;

use sim_core::*;
use sim_cuda::DEFAULT_VRAM_CAP_BYTES;
use sim_io::{
    prepare_probe_wav_signal_f64, write_probe_wav_f32_with_gain, write_telemetry_json,
    write_timeseries_csv,
};

struct CaliberSpec {
    tag: &'static str,
    blast_center: [f32; 3],
    blast_radius: f32,
    blast_length: f32,
    blast_overpressure_pa: f32,
    blast_density: f32,
    bullet_pos: [f32; 3],
    bullet_vel: f32,
    bullet_mass_kg: f32,
    bullet_radius_m: f32,
    bullet_length_m: f32,
    impact_radius_m: f32,
}

fn build_ballistics_scene(spec: &CaliberSpec) -> Scene {
    let mut scene = Scene::minimal(
        format!("ballistics_{}", spec.tag),
        AmrDomainConfig {
            origin: [-8.0, -24.0, -3.0],
            dx0: 0.50,
            nx0: 2208,
            ny0: 96,
            nz0: 56,
            pml_thickness: 8,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![
                // Level 1: High-res stationary shooter zone (x = -2.0..3.0 m, y = -2.0..2.0 m, z = 0.0..2.5 m, dx = 0.05 m)
                AmrPatchConfig {
                    parent_level: Some(0),
                    refine_ratio: 10,
                    off_x: 12,
                    off_y: 44,
                    off_z: 6,
                    nx: 100,
                    ny: 80,
                    nz: 50,
                    enable_flow: None,
                    active_start_s: 0.0,
                    active_end_s: Some(1.65),
                },
                // Level 2: High-res stationary 1,000 m downrange receiver & ricochet zone (x = 998..1019 m, dx = 0.05 m)
                AmrPatchConfig {
                    parent_level: Some(0),
                    refine_ratio: 10,
                    off_x: 2012,
                    off_y: 45,
                    off_z: 7,
                    nx: 420,
                    ny: 60,
                    nz: 40,
                    enable_flow: None,
                    active_start_s: 1.80,
                    active_end_s: Some(2.95),
                },
            ],
        },
    );

    scene.enable_compressible_euler = true;
    scene.fluid.bulk_viscosity_a = 0.003;
    scene.fluid.prandtl_a = 0.71;

    // Sliding nested AMR shock window tracking the supersonic projectile across the 1 km valley
    scene.sliding_patch = Some(SlidingAmrPatchConfig {
        refine_ratio: 10,
        nx: 100,
        ny: 60,
        nz: 60,
        initial_ox: -0.5,
        oy: -1.5,
        oz: 0.0,
        lead_fraction: 0.70,
        handoff_x_end: 999.0,
    });

    // Rigid canyon valley terrain & far cliff wall
    let mut concrete = SolidMaterial::concrete();
    concrete.is_rigid_static = true;
    scene.solids.push(SolidBodySpec {
        name: "canyon_terrain".to_string(),
        body_id: 0,
        shape: GeometryShape::CanyonTerrain {
            floor_z: 0.0,
            valley_half_width_y: 12.0,
            wall_slope: 0.9,
            cliff_x: 1030.0,
            cliff_height: 22.0,
        },
        material: concrete,
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.50,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![],
    });

    // Downrange steel ricochet target plate at x = 1017.5 m (coupled via projectile impact momentum transfer)
    let mut steel_plate = SolidMaterial::structural_steel();
    steel_plate.is_shell = true;
    steel_plate.thickness = 0.008;
    steel_plate.g_c = 0.0;
    scene.solids.push(SolidBodySpec {
        name: "downrange_steel_target".to_string(),
        body_id: 1,
        shape: GeometryShape::CurvedShell {
            center: [1017.5, 0.0, 1.60],
            size_x: 0.60,
            size_y: 0.60,
            nx: 10,
            ny: 10,
            dome_amp: 0.006,
            saddle_amp: 0.0,
            yaw_rad: 0.0,
            notch_x_len: 0.0,
            notch_y_pos: 0.0,
            notch_half_w: 0.0,
        },
        material: steel_plate.clone(),
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, std::f32::consts::FRAC_PI_2, 0.0],
        particle_spacing: 0.06,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![SolidBoundaryCondition::ClampedRegion {
            min: [1017.0, -0.40, 1.20],
            max: [1018.0, 0.40, 1.38],
        }],
    });

    // Rifle steel receiver / bolt-carrier plate near shooter excited by muzzle blast wave
    let mut receiver_steel = steel_plate;
    receiver_steel.thickness = 0.0035;
    scene.solids.push(SolidBodySpec {
        name: "shooter_rifle_receiver".to_string(),
        body_id: 2,
        shape: GeometryShape::CurvedShell {
            center: [0.18, 0.22, 1.54],
            size_x: 0.18,
            size_y: 0.12,
            nx: 6,
            ny: 4,
            dome_amp: 0.003,
            saddle_amp: 0.001,
            yaw_rad: 0.0,
            notch_x_len: 0.0,
            notch_y_pos: 0.0,
            notch_half_w: 0.0,
        },
        material: receiver_steel,
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.03,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![SolidBoundaryCondition::ClampedRegion {
            min: [0.05, 0.12, 1.45],
            max: [0.13, 0.32, 1.65],
        }],
    });

    // High-pressure propellant gas plug exiting barrel at t = 0.75 s (after 0.75 s silent pre-roll)
    scene.blast_regions.push(BlastRegionSpec {
        trigger_time_s: 0.75,
        center: spec.blast_center,
        radius: spec.blast_radius,
        length: spec.blast_length,
        is_cylinder_x: true,
        overpressure_pa: spec.blast_overpressure_pa,
        density_kg_m3: spec.blast_density,
        velocity: [spec.bullet_vel, 0.0, 0.0],
    });

    // Supersonic ogive projectile launched at t = 0.75 s
    scene.projectiles.push(SupersonicProjectileSpec {
        trigger_time_s: 0.75,
        initial_pos: spec.bullet_pos,
        initial_vel: [spec.bullet_vel, 0.0, 0.0],
        mass_kg: spec.bullet_mass_kg,
        radius_m: spec.bullet_radius_m,
        length_m: spec.bullet_length_m,
        impact_x_max: 1017.5,
        impact_target_body_id: Some(1),
        impact_radius_m: spec.impact_radius_m,
    });

    scene.microphones.push(MicrophoneSpec {
        name: format!("mic_shooter_{}.wav", spec.tag),
        position: [0.0, 1.5, 1.6],
        is_hydrophone: false,
    });
    scene.microphones.push(MicrophoneSpec {
        name: format!("mic_downrange_{}.wav", spec.tag),
        position: [1000.0, 1.0, 1.6],
        is_hydrophone: false,
    });

    scene
}

fn build_canyon_explosion_scene() -> Scene {
    let mut scene = Scene::minimal(
        "explosion_canyon",
        AmrDomainConfig {
            origin: [-20.0, -40.0, -2.0],
            dx0: 0.50,
            nx0: 880,
            ny0: 160,
            nz0: 64,
            pml_thickness: 8,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![
                // High-res blast kernel & near-cliff reflection patch (dx = 0.10 m)
                AmrPatchConfig {
                    parent_level: Some(0),
                    refine_ratio: 5,
                    off_x: 110,
                    off_y: 52,
                    off_z: 4,
                    nx: 100,
                    ny: 100,
                    nz: 80,
                    enable_flow: None,
                    active_start_s: 0.0,
                    active_end_s: Some(1.65),
                },
            ],
        },
    );

    scene.enable_compressible_euler = true;
    scene.fluid.bulk_viscosity_a = 0.003;
    scene.fluid.prandtl_a = 0.71;

    let mut concrete = SolidMaterial::concrete();
    concrete.is_rigid_static = true;
    scene.solids.push(SolidBodySpec {
        name: "canyon_walls_and_cliff".to_string(),
        body_id: 0,
        shape: GeometryShape::CanyonTerrain {
            floor_z: 0.0,
            valley_half_width_y: 16.0,
            wall_slope: 1.1,
            cliff_x: 340.0,
            cliff_height: 28.0,
        },
        material: concrete,
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.50,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![],
    });

    // 2 kg TNT high-explosive detonation (~8.37 MJ) near south canyon wall at t = 0.75 s
    scene.blast_regions.push(BlastRegionSpec {
        trigger_time_s: 0.75,
        center: [40.0, -10.0, 2.5],
        radius: 0.55,
        length: 0.0,
        is_cylinder_x: false,
        overpressure_pa: 4.8e6,
        density_kg_m3: 12.0,
        velocity: [0.0, 0.0, 0.0],
    });

    // Corrugated steel shed roof panel near observer excited by the passing blast wave
    let mut shed_steel = SolidMaterial::structural_steel();
    shed_steel.is_shell = true;
    shed_steel.thickness = 0.0025;
    shed_steel.g_c = 0.0;
    scene.solids.push(SolidBodySpec {
        name: "canyon_steel_panel".to_string(),
        body_id: 1,
        shape: GeometryShape::CurvedShell {
            center: [112.0, 6.5, 1.8],
            size_x: 0.90,
            size_y: 0.60,
            nx: 10,
            ny: 8,
            dome_amp: 0.015,
            saddle_amp: 0.004,
            yaw_rad: 0.25,
            notch_x_len: 0.0,
            notch_y_pos: 0.0,
            notch_half_w: 0.0,
        },
        material: shed_steel,
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.2, 0.0],
        particle_spacing: 0.08,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![SolidBoundaryCondition::ClampedRegion {
            min: [111.4, 6.1, 1.5],
            max: [111.7, 6.9, 2.1],
        }],
    });

    scene.microphones.push(MicrophoneSpec {
        name: "explosion_canyon.wav".to_string(),
        position: [115.0, 4.0, 1.6],
        is_hydrophone: false,
    });

    scene
}

fn main() -> Result<()> {
    let out_dir = Path::new("output");
    std::fs::create_dir_all(out_dir)?;

    let spec_556 = CaliberSpec {
        tag: "556",
        blast_center: [0.60, 0.0, 1.60],
        blast_radius: 0.06,
        blast_length: 0.14,
        blast_overpressure_pa: 7.7e5,
        blast_density: 6.0,
        bullet_pos: [0.75, 0.0, 1.60],
        bullet_vel: 940.0,
        bullet_mass_kg: 0.0040,
        bullet_radius_m: 0.00285,
        bullet_length_m: 0.023,
        impact_radius_m: 0.14,
    };

    let spec_50bmg = CaliberSpec {
        tag: "50bmg",
        blast_center: [0.70, 0.0, 1.60],
        blast_radius: 0.095,
        blast_length: 0.24,
        blast_overpressure_pa: 1.8e6,
        blast_density: 9.0,
        bullet_pos: [0.90, 0.0, 1.60],
        bullet_vel: 890.0,
        bullet_mass_kg: 0.046,
        bullet_radius_m: 0.00648,
        bullet_length_m: 0.058,
        impact_radius_m: 0.18,
    };

    let cfg_ballistics = SimRunConfig {
        duration_s: 6.5,
        pre_roll_s: 0.75,
        sample_rate: 48_000,
        oversample_factor: 1,
        target_wav_sample_rate: Some(48_000),
        telemetry_interval_s: 0.05,
        output_dir: Some(out_dir.to_path_buf()),
        cfl_safety: 0.38,
        print_progress: true,
        ..Default::default()
    };

    println!("=== Scene 1A/1B (1/3): Running 5.56x45mm NATO Supersonic Ballistics ===");
    let mut engine_556 = MultiphysicsEngine::new(build_ballistics_scene(&spec_556), None)?;
    let mut res_556 = engine_556.run(&cfg_ballistics)?;
    assert_eq!(res_556.telemetry.nan_count, 0);
    assert!(res_556.telemetry.peak_vram_tracked_bytes < DEFAULT_VRAM_CAP_BYTES);
    drop(engine_556);

    println!("=== Scene 1A/1B (2/3): Running .50 BMG Supersonic Ballistics ===");
    let mut engine_50 = MultiphysicsEngine::new(build_ballistics_scene(&spec_50bmg), None)?;
    let mut res_50 = engine_50.run(&cfg_ballistics)?;
    assert_eq!(res_50.telemetry.nan_count, 0);
    assert!(res_50.telemetry.peak_vram_tracked_bytes < DEFAULT_VRAM_CAP_BYTES);
    drop(engine_50);

    // Re-export shooter & downrange A/B pairs with shared calibrated gain per pair
    // so .50 BMG vs 5.56x45mm physical pressure amplitude contrast is preserved 1:1.
    let (sr_s556, sig_s556, pk_s556, rms_s556) = prepare_probe_wav_signal_f64(
        &res_556.raw_probe_buffers[0],
        1,
        48_000,
        Some(48_000),
        20.0,
    )?;
    let (sr_s50, sig_s50, pk_s50, rms_s50) = prepare_probe_wav_signal_f64(
        &res_50.raw_probe_buffers[0],
        1,
        48_000,
        Some(48_000),
        20.0,
    )?;
    let target_peak = 10.0_f64.powf(-1.0 / 20.0); // -1.0 dBFS
    let max_shooter_pk = pk_s556.max(pk_s50).max(1e-12);
    let shooter_shared_gain = Some(target_peak / max_shooter_pk);

    let wav_s556 = write_probe_wav_f32_with_gain(
        out_dir.join("mic_shooter_556.wav"),
        &sig_s556,
        sr_s556,
        pk_s556,
        rms_s556,
        shooter_shared_gain,
    )?;
    let wav_s50 = write_probe_wav_f32_with_gain(
        out_dir.join("mic_shooter_50bmg.wav"),
        &sig_s50,
        sr_s50,
        pk_s50,
        rms_s50,
        shooter_shared_gain,
    )?;
    res_556.telemetry.probes[0].wav = Some(wav_s556);
    res_50.telemetry.probes[0].wav = Some(wav_s50);

    let (sr_d556, sig_d556, pk_d556, rms_d556) = prepare_probe_wav_signal_f64(
        &res_556.raw_probe_buffers[1],
        1,
        48_000,
        Some(48_000),
        20.0,
    )?;
    let (sr_d50, sig_d50, pk_d50, rms_d50) = prepare_probe_wav_signal_f64(
        &res_50.raw_probe_buffers[1],
        1,
        48_000,
        Some(48_000),
        20.0,
    )?;
    let max_downrange_pk = pk_d556.max(pk_d50).max(1e-12);
    let downrange_shared_gain = Some(target_peak / max_downrange_pk);

    let wav_d556 = write_probe_wav_f32_with_gain(
        out_dir.join("mic_downrange_556.wav"),
        &sig_d556,
        sr_d556,
        pk_d556,
        rms_d556,
        downrange_shared_gain,
    )?;
    let wav_d50 = write_probe_wav_f32_with_gain(
        out_dir.join("mic_downrange_50bmg.wav"),
        &sig_d50,
        sr_d50,
        pk_d50,
        rms_d50,
        downrange_shared_gain,
    )?;
    res_556.telemetry.probes[1].wav = Some(wav_d556);
    res_50.telemetry.probes[1].wav = Some(wav_d50);

    let mut notes_556 = BTreeMap::new();
    notes_556.insert("shooter_peak_pa".to_string(), pk_s556);
    notes_556.insert("downrange_peak_pa".to_string(), pk_d556);
    res_556.telemetry.notes = notes_556;
    write_telemetry_json(out_dir.join("ballistics_556_telemetry.json"), &res_556.telemetry)?;
    write_timeseries_csv(out_dir.join("ballistics_556_timeseries.csv"), &res_556.timeseries)?;

    let mut notes_50 = BTreeMap::new();
    notes_50.insert("shooter_peak_pa".to_string(), pk_s50);
    notes_50.insert("downrange_peak_pa".to_string(), pk_d50);
    notes_50.insert("caliber_shooter_peak_ratio_50_vs_556".to_string(), pk_s50 / pk_s556.max(1e-9));
    notes_50.insert("caliber_downrange_peak_ratio_50_vs_556".to_string(), pk_d50 / pk_d556.max(1e-9));
    res_50.telemetry.notes = notes_50;
    write_telemetry_json(out_dir.join("ballistics_50bmg_telemetry.json"), &res_50.telemetry)?;
    write_timeseries_csv(out_dir.join("ballistics_50bmg_timeseries.csv"), &res_50.timeseries)?;

    println!(
        "Shooter peak: 5.56={:.1} Pa vs .50 BMG={:.1} Pa (ratio={:.2}x) | Downrange peak: 5.56={:.2} Pa vs .50 BMG={:.2} Pa (ratio={:.2}x)",
        pk_s556,
        pk_s50,
        pk_s50 / pk_s556.max(1e-9),
        pk_d556,
        pk_d50,
        pk_d50 / pk_d556.max(1e-9)
    );

    if out_dir.join("explosion_canyon.wav").exists()
        && out_dir.join("explosion_canyon_telemetry.json").exists()
    {
        println!("=== Scene 1C (3/3): explosion_canyon.wav already verified on disk; skipping re-run ===");
    } else {
        println!("=== Scene 1C (3/3): Running 2 kg TNT Canyon Cliff Detonation ===");
        let cfg_blast = SimRunConfig {
            duration_s: 7.0,
            pre_roll_s: 0.75,
            sample_rate: 48_000,
            oversample_factor: 1,
            target_wav_sample_rate: Some(48_000),
            telemetry_interval_s: 0.05,
            output_dir: Some(out_dir.to_path_buf()),
            cfl_safety: 0.38,
            print_progress: true,
            ..Default::default()
        };
        let mut engine_blast = MultiphysicsEngine::new(build_canyon_explosion_scene(), None)?;
        let res_blast = engine_blast.run(&cfg_blast)?;
        assert_eq!(res_blast.telemetry.nan_count, 0);
        assert!(res_blast.telemetry.peak_vram_tracked_bytes < DEFAULT_VRAM_CAP_BYTES);

        println!(
            "Canyon explosion completed: peak={:.1} Pa | wall_time={:.1}s | VRAM={:.1} MB",
            res_blast.telemetry.probes[0].peak_pa,
            res_blast.telemetry.wall_time_s,
            res_blast.telemetry.peak_vram_tracked_bytes as f64 / 1e6
        );
    }
    Ok(())
}
