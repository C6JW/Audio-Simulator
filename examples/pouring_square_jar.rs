use std::collections::BTreeMap;
use std::path::Path;

use sim_core::*;
use sim_cuda::DEFAULT_VRAM_CAP_BYTES;
use sim_io::{write_telemetry_json, write_timeseries_csv};

fn main() -> Result<()> {
    let out_dir = Path::new("output");
    std::fs::create_dir_all(out_dir)?;

    let mut scene = Scene::minimal(
        "pouring_square_jar",
        AmrDomainConfig {
            origin: [-0.24, -0.24, -0.06],
            dx0: 0.006,
            nx0: 80,
            ny0: 80,
            nz0: 76,
            pml_thickness: 6,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![
                // Level 1: 2.0 mm multiphase VOF + Kapila bubble + crystal square jar patch
                // Covers x,y in [-0.096, +0.096] m, z in [0.000, +0.288] m
                AmrPatchConfig {
                    parent_level: Some(0),
                    refine_ratio: 3,
                    off_x: 24,
                    off_y: 24,
                    off_z: 10,
                    nx: 96,
                    ny: 96,
                    nz: 144,
                    enable_flow: Some(true),
                    active_start_s: 0.0,
                    active_end_s: None,
                },
            ],
        },
    );

    // Level 0 propagates room air acoustics; Level 1 solves full two-phase VOF + CSF + Kapila bubbles
    scene.enable_flow_advection = false;

    let mut jar_glass = SolidMaterial::crystal_glass();
    jar_glass.is_shell = true;
    jar_glass.thickness = 0.0024;
    jar_glass.eta_visc = 3.5;
    jar_glass.sls_beta = 0.0012;
    jar_glass.g_c = 0.0;

    // Rounded-square glass jar: 9.2 cm x 9.2 cm square body, smooth shoulder, 5.2 cm open mouth
    // Bottom at z = 0.018 m, top rim at z = 0.180 m
    scene.solids.push(SolidBodySpec {
        name: "square_glass_jar".to_string(),
        body_id: 0,
        shape: GeometryShape::SquareJarShell {
            center_xy: [0.0, 0.0],
            z_bottom: 0.018,
            body_half_width: 0.046,
            body_height: 0.112,
            shoulder_height: 0.028,
            neck_radius: 0.026,
            neck_height: 0.022,
            corner_radius: 0.011,
            n_perim: 56,
            n_height: 30,
            hyd_len: 0.014,
        },
        material: jar_glass,
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.004,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![SolidBoundaryCondition::ClampedRegion {
            min: [-0.052, -0.052, 0.014],
            max: [0.052, 0.052, 0.022],
        }],
    });

    // Steady water stream poured through the jar neck (z = 0.235 m) from t = 0.30..4.50 s
    scene.fluid_inflows.push(FluidInflowSpec {
        center: [0.007, 0.005, 0.235],
        radius: 0.0045,
        height: 0.006,
        velocity: [0.0, 0.0, -0.95],
        start_time_s: 0.30,
        end_time_s: 4.50,
        ramp_s: 0.06,
    });

    // Near-mouth mic inside Level 1 (2 mm resolution) + room mic on Level 0
    scene.microphones.push(MicrophoneSpec {
        name: "pouring_square_jar.wav".to_string(),
        position: [0.080, 0.025, 0.215],
        is_hydrophone: false,
    });
    scene.microphones.push(MicrophoneSpec {
        name: "pouring_square_jar_room.wav".to_string(),
        position: [0.160, 0.050, 0.220],
        is_hydrophone: false,
    });

    let cfg = SimRunConfig {
        duration_s: 5.5,
        pre_roll_s: 0.30,
        sample_rate: 48_000,
        oversample_factor: 1,
        target_wav_sample_rate: Some(48_000),
        telemetry_interval_s: 0.05,
        output_dir: Some(out_dir.to_path_buf()),
        cfl_safety: 0.42,
        print_progress: true,
        ..Default::default()
    };

    println!("=== Running Pouring Water into Square Glass Jar (dx1 = 2.0 mm AMR) ===");
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

    let early_pour = slice_win(&res.raw_probe_buffers[0], 0.65, 1.75);
    let late_pour = slice_win(&res.raw_probe_buffers[0], 3.30, 4.40);

    let (f_helm_early, c_helm_early) = dominant_freq_and_centroid(&early_pour, sr, 250.0, 2200.0);
    let (f_helm_late, c_helm_late) = dominant_freq_and_centroid(&late_pour, sr, 250.0, 2200.0);
    let (f_bub_early, c_bub_early) = dominant_freq_and_centroid(&early_pour, sr, 1200.0, 6000.0);
    let (f_bub_late, c_bub_late) = dominant_freq_and_centroid(&late_pour, sr, 1200.0, 6000.0);

    let mut notes = BTreeMap::new();
    notes.insert("final_water_volume_m3".to_string(), m1_final.water_volume);
    notes.insert("final_water_volume_ml".to_string(), m1_final.water_volume * 1e6);
    notes.insert("helmholtz_dom_early_hz".to_string(), f_helm_early);
    notes.insert("helmholtz_dom_late_hz".to_string(), f_helm_late);
    notes.insert("helmholtz_centroid_early_hz".to_string(), c_helm_early);
    notes.insert("helmholtz_centroid_late_hz".to_string(), c_helm_late);
    notes.insert("bubble_dom_early_hz".to_string(), f_bub_early);
    notes.insert("bubble_dom_late_hz".to_string(), f_bub_late);
    notes.insert("bubble_centroid_early_hz".to_string(), c_bub_early);
    notes.insert("bubble_centroid_late_hz".to_string(), c_bub_late);
    res.telemetry.notes = notes;

    write_telemetry_json(out_dir.join("pouring_square_jar_telemetry.json"), &res.telemetry)?;
    write_timeseries_csv(out_dir.join("pouring_square_jar_timeseries.csv"), &res.timeseries)?;

    println!(
        "Pouring Square Jar Results: water={:.1} mL | cavity centroid {:.1}->{:.1} Hz | bubble centroid {:.1}->{:.1} Hz | VRAM={:.1} MB | NaNs={}",
        m1_final.water_volume * 1e6,
        c_helm_early,
        c_helm_late,
        c_bub_early,
        c_bub_late,
        res.telemetry.peak_vram_tracked_bytes as f64 / 1e6,
        res.telemetry.nan_count
    );

    assert_eq!(res.telemetry.nan_count, 0);
    assert!(res.telemetry.peak_vram_tracked_bytes < DEFAULT_VRAM_CAP_BYTES);
    assert!(
        m1_final.water_volume > 1e-5,
        "Expected water accumulation inside square jar, got {:.2e} m^3",
        m1_final.water_volume
    );
    Ok(())
}
