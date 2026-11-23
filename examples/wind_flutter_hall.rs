use std::collections::BTreeMap;
use std::path::Path;

use sim_core::*;
use sim_cuda::DEFAULT_VRAM_CAP_BYTES;
use sim_io::{
    generate_fallback_speech_phrase, read_wav_mono_f32, write_telemetry_json, write_timeseries_csv,
};

fn main() -> Result<()> {
    let out_dir = Path::new("output");
    std::fs::create_dir_all(out_dir)?;

    // 6.0 m diameter curved stone hall (R_inner = 3.0 m) with a narrow stone archway at x = -3.0 m
    // Level 0: 132 x 120 x 54, dx0 = 0.06 m (7.92 m x 7.20 m x 3.24 m)
    // Level 1: 96 x 72 x 72, r=3, dx1 = 0.02 m (archway vortex-shedding jet + paper + 3 steel chimes)
    let mut scene = Scene::minimal(
        "wind_flutter_hall",
        AmrDomainConfig {
            origin: [-4.20, -3.60, -0.12],
            dx0: 0.06,
            nx0: 132,
            ny0: 120,
            nz0: 54,
            pml_thickness: 6,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![AmrPatchConfig {
                parent_level: Some(0),
                refine_ratio: 3,
                off_x: 16,
                off_y: 48,
                off_z: 14,
                nx: 96,
                ny: 72,
                nz: 72,
                enable_flow: None,
                active_start_s: 0.0,
                active_end_s: None,
            }],
        },
    );

    scene.enable_compressible_euler = true;
    scene.fluid.bulk_viscosity_a = 0.001;
    scene.fluid.prandtl_a = 0.71;
    scene.fluid.wall_absorption_alpha = 0.03;

    let mut stone = SolidMaterial::concrete();
    stone.is_rigid_static = true;
    scene.solids.push(SolidBodySpec {
        name: "stone_hall_wall_with_archway".to_string(),
        body_id: 0,
        shape: GeometryShape::CylindricalRingWithArch {
            center_xy: [0.0, 0.0],
            r_inner: 3.00,
            r_outer: 3.35,
            z_min: 0.00,
            z_max: 2.90,
            arch_center_angle_rad: std::f32::consts::PI,
            arch_half_angle_rad: 0.13,
            arch_z_min: 0.00,
            arch_z_max: 2.15,
        },
        material: stone.clone(),
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.06,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![],
    });
    scene.solids.push(SolidBodySpec {
        name: "stone_hall_floor".to_string(),
        body_id: 0,
        shape: GeometryShape::Disc {
            center_xy: [0.0, 0.0],
            radius: 3.40,
            z_min: -0.10,
            z_max: 0.00,
        },
        material: stone.clone(),
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.06,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![],
    });
    scene.solids.push(SolidBodySpec {
        name: "stone_hall_ceiling".to_string(),
        body_id: 0,
        shape: GeometryShape::Disc {
            center_xy: [0.0, 0.0],
            radius: 3.40,
            z_min: 2.90,
            z_max: 3.00,
        },
        material: stone,
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.06,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![],
    });

    // Hanging cellulose paper sheet inside the archway wind jet (buckles & flutters without tearing)
    let mut paper_mat = SolidMaterial::cellulose_paper(1.1e-4);
    paper_mat.g_c = 0.0;
    scene.solids.push(SolidBodySpec {
        name: "hanging_paper_sheet".to_string(),
        body_id: 1,
        shape: GeometryShape::CurvedShell {
            center: [-2.35, -0.14, 1.45],
            size_x: 0.26,
            size_y: 0.18,
            nx: 10,
            ny: 8,
            dome_amp: 0.006,
            saddle_amp: 0.003,
            yaw_rad: 0.18,
            notch_x_len: 0.0,
            notch_y_pos: 0.0,
            notch_half_w: 0.0,
        },
        material: paper_mat,
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.35, 0.20],
        particle_spacing: 0.022,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![SolidBoundaryCondition::ClampedRegion {
            min: [-2.55, -0.32, 1.46],
            max: [-2.38, 0.06, 1.68],
        }],
    });

    // 3 suspended thin-shell steel chime strips of distinct lengths (0.28 m, 0.23 m, 0.19 m)
    let mut chime_mat = SolidMaterial::structural_steel();
    chime_mat.is_shell = true;
    chime_mat.thickness = 0.0014;
    chime_mat.eta_visc = 3.5;
    chime_mat.sls_beta = 0.0003;
    chime_mat.g_c = 0.0;

    let chime_lengths = [0.28_f32, 0.23, 0.19];
    let chime_x = [-1.98_f32, -1.94, -1.90];
    let chime_y = [0.10_f32, 0.16, 0.22];
    for (ci, &len) in chime_lengths.iter().enumerate() {
        let z_top = 1.66_f32;
        let z_center = z_top - 0.5 * len;
        scene.solids.push(SolidBodySpec {
            name: format!("steel_chime_{ci}"),
            body_id: (ci + 2) as i32,
            shape: GeometryShape::CurvedShell {
                center: [chime_x[ci], chime_y[ci], z_center],
                size_x: len,
                size_y: 0.036,
                nx: 8,
                ny: 3,
                dome_amp: 0.002,
                saddle_amp: 0.0,
                yaw_rad: 0.0,
                notch_x_len: 0.0,
                notch_y_pos: 0.0,
                notch_half_w: 0.0,
            },
            material: chime_mat.clone(),
            initial_velocity: [0.0, 0.0, 0.0],
            orientation_euler_xyz: [0.0, 1.38, 0.22],
            particle_spacing: 0.018,
            gc_heterogeneity_amp: 0.0,
            is_kinematic_rigid: false,
            bcs: vec![SolidBoundaryCondition::ClampedRegion {
                min: [chime_x[ci] - 0.18, chime_y[ci] - 0.06, z_top - 0.045],
                max: [chime_x[ci] + 0.18, chime_y[ci] + 0.06, z_top + 0.05],
            }],
        });
    }

    // 12 m/s turbulent wind gust entering through the western stone archway during t = 0.75..6.25 s
    scene.wind_inlets.push(WindInletSpec {
        min: [-3.95, -0.55, 0.20],
        max: [-3.40, 0.55, 2.05],
        velocity: [12.0, 0.0, 0.0],
        turb_amp: 0.35,
        start_time_s: 0.75,
        end_time_s: 6.25,
        ramp_s: 0.35,
    });

    // Load 3-second human speech recording for opposite-wall directional source at t = 2.0..5.0 s
    let speech_path = Path::new("assets/hall_whisper_speech.wav");
    let (speech_sr, mut speech_samples) = if speech_path.exists() {
        read_wav_mono_f32(speech_path)?
    } else {
        (48_000, generate_fallback_speech_phrase(48_000))
    };
    let max_speech_samples = (3.0 * (speech_sr as f64)).round() as usize;
    if speech_samples.len() > max_speech_samples {
        speech_samples.truncate(max_speech_samples);
    }
    let pk_sp = speech_samples
        .iter()
        .fold(0.0_f32, |a, &b| a.max(b.abs()))
        .max(1e-6);
    for s in &mut speech_samples {
        *s /= pk_sp;
    }

    scene.directional_sources.push(DirectionalAudioSourceSpec {
        center: [2.82, 0.0, 1.48],
        direction: [0.0, 1.0, 0.0],
        cardioid_beta: 0.5,
        radius: 0.12,
        peak_rate_pa_per_s: 1.8e5,
        t0_s: 2.00,
        sample_rate: speech_sr as f64,
        samples: speech_samples,
    });

    scene.microphones.push(MicrophoneSpec {
        name: "wind_flutter_hall.wav".to_string(),
        position: [-1.58, 0.04, 1.46],
        is_hydrophone: false,
    });

    let cfg = SimRunConfig {
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

    println!("=== Scene 5: Running Turbulent Wind, Paper Flutter, Chimes & Curved-Hall Speech ===");
    let mut engine = MultiphysicsEngine::new(scene, None)?;
    let mut res = engine.run(&cfg)?;

    let sr = 48_000.0_f64;
    let slice_win = |buf: &[f32], t0: f64, t1: f64| -> Vec<f64> {
        let i0 = ((t0 * sr) as usize).min(buf.len());
        let i1 = ((t1 * sr) as usize).min(buf.len());
        buf[i0..i1].iter().map(|&v| v as f64).collect()
    };

    let wind_only = slice_win(&res.raw_probe_buffers[0], 0.90, 1.95);
    let speech_mix = slice_win(&res.raw_probe_buffers[0], 2.15, 4.95);
    let tail_chimes = slice_win(&res.raw_probe_buffers[0], 5.10, 6.80);

    let (f_wind, c_wind) = dominant_freq_and_centroid(&wind_only, sr, 40.0, 6000.0);
    let (f_speech, c_speech) = dominant_freq_and_centroid(&speech_mix, sr, 100.0, 6000.0);
    let (f_tail, c_tail) = dominant_freq_and_centroid(&tail_chimes, sr, 100.0, 8000.0);

    let mut notes = BTreeMap::new();
    notes.insert("wind_dominant_freq_hz".to_string(), f_wind);
    notes.insert("wind_centroid_hz".to_string(), c_wind);
    notes.insert("speech_mix_dominant_freq_hz".to_string(), f_speech);
    notes.insert("speech_mix_centroid_hz".to_string(), c_speech);
    notes.insert("tail_chimes_dominant_freq_hz".to_string(), f_tail);
    notes.insert("tail_chimes_centroid_hz".to_string(), c_tail);
    res.telemetry.notes = notes;

    write_telemetry_json(out_dir.join("wind_flutter_hall_telemetry.json"), &res.telemetry)?;
    write_timeseries_csv(out_dir.join("wind_flutter_hall_timeseries.csv"), &res.timeseries)?;

    println!(
        "Wind Flutter Hall Results: peak={:.2} Pa | wind_cent={:.1} Hz | speech_cent={:.1} Hz | tail_cent={:.1} Hz | VRAM={:.1} MB | NaNs={}",
        res.telemetry.probes[0].peak_pa,
        c_wind,
        c_speech,
        c_tail,
        res.telemetry.peak_vram_tracked_bytes as f64 / 1e6,
        res.telemetry.nan_count
    );

    assert_eq!(res.telemetry.nan_count, 0);
    assert!(res.telemetry.peak_vram_tracked_bytes < DEFAULT_VRAM_CAP_BYTES);
    Ok(())
}
