use std::collections::BTreeMap;
use std::path::Path;

use sim_core::*;
use sim_cuda::DEFAULT_VRAM_CAP_BYTES;
use sim_io::{write_telemetry_json, write_timeseries_csv};

fn main() -> Result<()> {
    let out_dir = Path::new("output");
    std::fs::create_dir_all(out_dir)?;

    let mut scene = Scene::minimal(
        "sinking_bell_grotto",
        AmrDomainConfig {
            origin: [-0.288, -0.288, -0.352],
            dx0: 0.008,
            nx0: 72,
            ny0: 72,
            nz0: 88,
            pml_thickness: 6,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![],
        },
    );

    scene.fluid.wall_absorption_alpha = 0.05;
    scene.background_water = vec![GeometryShape::HalfSpaceZ { z_max: -0.020 }];
    scene.enable_flow_advection = false;

    // Reflective stone grotto partial cylindrical enclosure
    let mut stone = SolidMaterial::granite_stone();
    stone.is_rigid_static = true;
    scene.solids.push(SolidBodySpec {
        name: "grotto_stone_wall".to_string(),
        body_id: 2,
        shape: GeometryShape::CylindricalRing {
            center_xy: [0.0, 0.0],
            r_inner: 0.22,
            r_outer: 0.26,
            z_min: -0.28,
            z_max: 0.28,
        },
        material: stone,
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.016,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![],
    });

    // High-Q cast bronze bell (rim at z = +0.020 m, crown at z = +0.190 m, 4 cm above water pool)
    let mut bell_mat = SolidMaterial::bell_bronze();
    bell_mat.is_shell = true;
    bell_mat.thickness = 0.0044;
    bell_mat.eta_visc = 4.0;
    bell_mat.sls_beta = 0.0004;
    bell_mat.g_c = 0.0;

    let profile = vec![
        [0.020, 0.084],
        [0.035, 0.078],
        [0.080, 0.062],
        [0.140, 0.048],
        [0.182, 0.032],
        [0.190, 0.000],
    ];
    scene.solids.push(SolidBodySpec {
        name: "bronze_bell".to_string(),
        body_id: 0,
        shape: GeometryShape::RevolutionShell {
            center_xy: [0.0, 0.0],
            profile_zr: profile,
            n_theta: 48,
            n_meridian: 20,
            hyd_len: 0.042,
        },
        material: bell_mat,
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.007,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![
            // Only the top crown suspension cap is velocity-prescribed so the bell skirt vibrates freely
            SolidBoundaryCondition::PrescribedVelocityRegion {
                min: [-0.045, -0.045, 0.175],
                max: [0.045, 0.045, 0.210],
                velocity: [0.0, 0.0, 0.0],
            },
        ],
    });

    // Smooth lowering trajectory for the bell crown:
    // - 0.00..2.25 s: suspended stationary in air (struck at t = 0.766 s)
    // - 2.25..4.45 s: lowered at vz = -0.10 m/s by 22 cm into the water pool (z_rim: +0.02 -> -0.20 m)
    // - 4.45..7.00 s: held stationary underwater while ringing at its added-mass-shifted frequency
    scene.kinematic_trajectories.push(KinematicTrajectorySpec {
        body_id: 0,
        initial_pivot: [0.0, 0.0, 0.185],
        segments: vec![
            KinematicWaypoint {
                t_end_s: 2.25,
                linear_vel: [0.0, 0.0, 0.0],
                angular_vel: [0.0, 0.0, 0.0],
            },
            KinematicWaypoint {
                t_end_s: 4.45,
                linear_vel: [0.0, 0.0, -0.10],
                angular_vel: [0.0, 0.0, 0.0],
            },
            KinematicWaypoint {
                t_end_s: 7.00,
                linear_vel: [0.0, 0.0, 0.0],
                angular_vel: [0.0, 0.0, 0.0],
            },
        ],
    });

    // Kinematic bronze clapper striking the bell rim in air at t = 0.76..0.766 s, then retracting dry
    scene.solids.push(SolidBodySpec {
        name: "bell_clapper".to_string(),
        body_id: 1,
        shape: GeometryShape::Sphere {
            center: [0.095, 0.0, 0.026],
            radius: 0.010,
        },
        material: SolidMaterial::bell_bronze(),
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.006,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: true,
        bcs: vec![],
    });
    scene.kinematic_trajectories.push(KinematicTrajectorySpec {
        body_id: 1,
        initial_pivot: [0.095, 0.0, 0.026],
        segments: vec![
            KinematicWaypoint {
                t_end_s: 0.760,
                linear_vel: [0.0, 0.0, 0.0],
                angular_vel: [0.0, 0.0, 0.0],
            },
            KinematicWaypoint {
                t_end_s: 0.766,
                linear_vel: [-1.5, 0.0, 0.0],
                angular_vel: [0.0, 0.0, 0.0],
            },
            KinematicWaypoint {
                t_end_s: 0.820,
                linear_vel: [1.5, 0.0, 0.8],
                angular_vel: [0.0, 0.0, 0.0],
            },
            KinematicWaypoint {
                t_end_s: 7.000,
                linear_vel: [0.0, 0.0, 0.0],
                angular_vel: [0.0, 0.0, 0.0],
            },
        ],
    });

    scene.microphones.push(MicrophoneSpec {
        name: "bell_air_mic.wav".to_string(),
        position: [0.0, 0.16, 0.12],
        is_hydrophone: false,
    });
    scene.microphones.push(MicrophoneSpec {
        name: "bell_hydrophone.wav".to_string(),
        position: [0.0, 0.15, -0.14],
        is_hydrophone: true,
    });

    let cfg = SimRunConfig {
        duration_s: 7.0,
        pre_roll_s: 0.75,
        sample_rate: 48_000,
        oversample_factor: 1,
        target_wav_sample_rate: Some(48_000),
        telemetry_interval_s: 0.05,
        output_dir: Some(out_dir.to_path_buf()),
        cfl_safety: 0.35,
        print_progress: true,
        ..Default::default()
    };

    println!("=== Scene 2: Running Sinking Bell in Stone Grotto (Air -> Water Pitch Glide) ===");
    let mut engine = MultiphysicsEngine::new(scene, None)?;
    let mut res = engine.run(&cfg)?;

    let sr = 48_000.0_f64;
    let slice_win = |buf: &[f32], t0: f64, t1: f64| -> Vec<f64> {
        let i0 = ((t0 * sr) as usize).min(buf.len());
        let i1 = ((t1 * sr) as usize).min(buf.len());
        buf[i0..i1].iter().map(|&v| v as f64).collect()
    };

    let air_early = slice_win(&res.raw_probe_buffers[0], 0.90, 2.15);
    let air_sub = slice_win(&res.raw_probe_buffers[0], 4.55, 6.50);
    let hyd_sub = slice_win(&res.raw_probe_buffers[1], 4.55, 6.50);

    let (f_air_dom, f_air_cent) = dominant_freq_and_centroid(&air_early, sr, 180.0, 5000.0);
    let (f_air_sub_dom, _) = dominant_freq_and_centroid(&air_sub, sr, 120.0, 4000.0);
    let (f_hyd_sub_dom, f_hyd_cent) = dominant_freq_and_centroid(&hyd_sub, sr, 120.0, 4000.0);
    let drop_pct = (f_air_dom - f_hyd_sub_dom) / f_air_dom.max(1.0) * 100.0;

    let mut notes = BTreeMap::new();
    notes.insert("in_air_dominant_freq_hz".to_string(), f_air_dom);
    notes.insert("in_air_centroid_hz".to_string(), f_air_cent);
    notes.insert("submerged_air_mic_freq_hz".to_string(), f_air_sub_dom);
    notes.insert("submerged_hydrophone_freq_hz".to_string(), f_hyd_sub_dom);
    notes.insert("submerged_hydrophone_centroid_hz".to_string(), f_hyd_cent);
    notes.insert("pitch_drop_percent".to_string(), drop_pct);
    res.telemetry.notes = notes;

    write_telemetry_json(out_dir.join("sinking_bell_grotto_telemetry.json"), &res.telemetry)?;
    write_timeseries_csv(out_dir.join("sinking_bell_grotto_timeseries.csv"), &res.timeseries)?;

    println!(
        "Sinking Bell Results: in-air={:.1} Hz -> submerged={:.1} Hz (air_mic={:.1} Hz) | drop={:.1}% | VRAM={:.1} MB | NaNs={}",
        f_air_dom,
        f_hyd_sub_dom,
        f_air_sub_dom,
        drop_pct,
        res.telemetry.peak_vram_tracked_bytes as f64 / 1e6,
        res.telemetry.nan_count
    );

    assert_eq!(res.telemetry.nan_count, 0);
    assert!(res.telemetry.peak_vram_tracked_bytes < DEFAULT_VRAM_CAP_BYTES);
    Ok(())
}
