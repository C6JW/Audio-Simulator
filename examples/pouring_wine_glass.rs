use std::collections::BTreeMap;
use std::path::Path;

use sim_core::*;
use sim_cuda::DEFAULT_VRAM_CAP_BYTES;
use sim_io::{write_telemetry_json, write_timeseries_csv};

fn main() -> Result<()> {
    let out_dir = Path::new("output");
    std::fs::create_dir_all(out_dir)?;

    let mut scene = Scene::minimal(
        "pouring_wine_glass",
        AmrDomainConfig {
            origin: [-0.24, -0.24, -0.06],
            dx0: 0.0075,
            nx0: 64,
            ny0: 64,
            nz0: 64,
            pml_thickness: 6,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![
                // Level 1: 2.5 mm multiphase VOF + Kapila bubble + crystal wine glass patch
                // Covers x,y in [-0.090, +0.090] m, z in [0.0075, +0.2775] m
                AmrPatchConfig {
                    parent_level: Some(0),
                    refine_ratio: 3,
                    off_x: 20,
                    off_y: 20,
                    off_z: 9,
                    nx: 72,
                    ny: 72,
                    nz: 108,
                    enable_flow: Some(true),
                    active_start_s: 0.0,
                    active_end_s: None,
                },
            ],
        },
    );

    // Level 0 propagates room air acoustics; Level 1 runs full two-phase VOF + CSF + Kapila bubbles
    scene.enable_flow_advection = false;

    let mut glass_mat = SolidMaterial::crystal_glass();
    glass_mat.is_shell = true;
    glass_mat.thickness = 0.0012;
    glass_mat.eta_visc = 3.0;
    glass_mat.sls_beta = 0.0008;
    glass_mat.g_c = 0.0;

    // Watertight crystal wine glass bowl (bottom pole at z=0.020m, open rim at z=0.115m, R_max=36mm)
    let profile = vec![
        [0.020, 0.000],
        [0.022, 0.018],
        [0.032, 0.030],
        [0.055, 0.036],
        [0.085, 0.035],
        [0.115, 0.032],
    ];
    scene.solids.push(SolidBodySpec {
        name: "crystal_wine_glass".to_string(),
        body_id: 0,
        shape: GeometryShape::RevolutionShell {
            center_xy: [0.0, 0.0],
            profile_zr: profile,
            n_theta: 64,
            n_meridian: 28,
            hyd_len: 0.011,
        },
        material: glass_mat,
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.0032,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![SolidBoundaryCondition::ClampedRegion {
            min: [-0.024, -0.024, 0.015],
            max: [0.024, 0.024, 0.024],
        }],
    });

    // Water pouring stream from 15 cm above the glass bottom (z = 0.170 m) during t = 0.75..5.75 s
    scene.fluid_inflows.push(FluidInflowSpec {
        center: [0.006, 0.0, 0.170],
        radius: 0.0045,
        height: 0.006,
        velocity: [0.0, 0.0, -0.80],
        start_time_s: 0.75,
        end_time_s: 5.75,
        ramp_s: 0.08,
    });

    scene.microphones.push(MicrophoneSpec {
        name: "pouring_wine_glass.wav".to_string(),
        position: [0.065, 0.0, 0.135],
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
        cfl_safety: 0.42,
        print_progress: true,
        ..Default::default()
    };

    println!("=== Scene 3: Running Pouring Water into Crystal Wine Glass (dx1 = 2.5 mm AMR) ===");
    let mut engine = MultiphysicsEngine::new(scene, None)?;
    let mut res = engine.run(&cfg)?;

    let m1_final = engine.reduce_level_metrics(1)?;
    res.telemetry.final_water_volume = m1_final.water_volume;

    let sr = 48_000.0_f64;
    let slice_win = |buf: &[f32], t0: f64, t1: f64| -> Vec<f64> {
        let i0 = ((t0 * sr) as usize).min(buf.len());
        let i1 = ((t1 * sr) as usize).min(buf.len());
        buf[i0..i1].iter().map(|&v| v as f64).collect()
    };

    let early_pour = slice_win(&res.raw_probe_buffers[0], 1.10, 2.30);
    let late_pour = slice_win(&res.raw_probe_buffers[0], 4.40, 5.65);

    // Low-band structural crystal glass resonance (bending mode shifts downward with added water mass)
    let (f_struct_early, _) = dominant_freq_and_centroid(&early_pour, sr, 250.0, 1100.0);
    let (f_struct_late, _) = dominant_freq_and_centroid(&late_pour, sr, 250.0, 1100.0);

    // High-band air-column cavity & bubble plop band (shifts upward as remaining air column shortens)
    let (f_cav_early, c_cav_early) = dominant_freq_and_centroid(&early_pour, sr, 1150.0, 4800.0);
    let (f_cav_late, c_cav_late) = dominant_freq_and_centroid(&late_pour, sr, 1150.0, 4800.0);

    let mut notes = BTreeMap::new();
    notes.insert("final_water_volume_m3".to_string(), m1_final.water_volume);
    notes.insert("final_water_volume_ml".to_string(), m1_final.water_volume * 1e6);
    notes.insert("structural_freq_early_hz".to_string(), f_struct_early);
    notes.insert("structural_freq_late_hz".to_string(), f_struct_late);
    notes.insert("cavity_freq_early_hz".to_string(), f_cav_early);
    notes.insert("cavity_freq_late_hz".to_string(), f_cav_late);
    notes.insert("cavity_centroid_early_hz".to_string(), c_cav_early);
    notes.insert("cavity_centroid_late_hz".to_string(), c_cav_late);
    res.telemetry.notes = notes;

    write_telemetry_json(out_dir.join("pouring_wine_glass_telemetry.json"), &res.telemetry)?;
    write_timeseries_csv(out_dir.join("pouring_wine_glass_timeseries.csv"), &res.timeseries)?;

    println!(
        "Pouring Wine Glass Results: water={:.1} mL | struct {:.1}->{:.1} Hz | cavity centroid {:.1}->{:.1} Hz | VRAM={:.1} MB | NaNs={}",
        m1_final.water_volume * 1e6,
        f_struct_early,
        f_struct_late,
        c_cav_early,
        c_cav_late,
        res.telemetry.peak_vram_tracked_bytes as f64 / 1e6,
        res.telemetry.nan_count
    );

    assert_eq!(res.telemetry.nan_count, 0);
    assert!(res.telemetry.peak_vram_tracked_bytes < DEFAULT_VRAM_CAP_BYTES);
    Ok(())
}
