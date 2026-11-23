use std::f64::consts::PI;

use sim_core::*;
use sim_cuda::{CudaError, VramBudgetManager, DEFAULT_VRAM_CAP_BYTES};
use sim_io::{
    read_wav_f32, write_probe_wav_f32, write_telemetry_json, write_timeseries_csv, RunTelemetry,
    SampleFormat, TimeSeriesRow,
};

#[test]
fn test_cuda_device_and_vram_budget() {
    let vram = VramBudgetManager::new(0, DEFAULT_VRAM_CAP_BYTES).expect("init CUDA device 0");
    let info = vram.device_info();
    assert_eq!(info.cc_major, 12, "Expected Blackwell sm_12x");
    assert_eq!(info.cc_minor, 0, "Expected sm_120 (RTX 5080)");
    assert!(info.total_vram_bytes > 15_000_000_000);

    let (free, total) = vram.query_driver_mem_info().expect("cuMemGetInfo_v2");
    assert!(total > 15_000_000_000);
    assert!(free > 0 && free <= total);

    // Over-15 GB allocation must be rejected immediately with VramBudgetExceeded
    let over_cap_elems = 4_000_000_000usize; // 16 GB of f32
    let res = vram.alloc_zeros::<f32>(over_cap_elems);
    assert!(matches!(res, Err(CudaError::VramBudgetExceeded { .. })));

    // 1 MiB allocation succeeds and cleans up via RAII Drop
    {
        let buf = vram.alloc_zeros::<f32>(262_144).expect("alloc 1 MiB");
        assert_eq!(vram.tracked_bytes(), 1_048_576);
        let host = buf.to_vec().expect("download");
        assert_eq!(host.len(), 262_144);
        assert!(host.iter().all(|&x| x == 0.0));
    }
    assert_eq!(vram.tracked_bytes(), 0);
}

#[test]
fn test_single_phase_air_wave_speed() {
    let dx = 0.004_f32;
    let mut scene = Scene::minimal(
        "test_air_wave_speed",
        AmrDomainConfig {
            origin: [0.0, 0.0, 0.0],
            dx0: dx,
            nx0: 140,
            ny0: 48,
            nz0: 48,
            pml_thickness: 10,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![],
        },
    );
    scene.fluid.gravity = [0.0, 0.0, 0.0];
    scene.gaussian_pulses.push(GaussianPulseSpec {
        center: [0.16, 0.096, 0.096],
        sigma_r: 0.012,
        amplitude_pa: 100.0,
    });
    scene.microphones.push(MicrophoneSpec {
        name: "mic1".to_string(),
        position: [0.24, 0.096, 0.096], // d1 = 0.08 m
        is_hydrophone: false,
    });
    scene.microphones.push(MicrophoneSpec {
        name: "mic2".to_string(),
        position: [0.36, 0.096, 0.096], // d2 = 0.20 m (delta_d = 0.12 m)
        is_hydrophone: false,
    });

    let mut engine = MultiphysicsEngine::new(scene, None).expect("init engine");
    let res = engine
        .run(&SimRunConfig {
            duration_s: 0.0020,
            sample_rate: 96_000,
            oversample_factor: 4, // 384 kHz probe sampling (2.6 us resolution)
            telemetry_interval_s: 0.0005,
            output_dir: None,
            water_surface_z_for_diag: None,
            use_cuda_graph: true,
            cfl_safety: 0.38,
            print_progress: false,
            render_export: None,
            ..Default::default()
        })
        .expect("run air wave test");

    assert_eq!(res.telemetry.nan_count, 0);
    assert!(res.telemetry.peak_vram_tracked_bytes < DEFAULT_VRAM_CAP_BYTES);

    let fs = 96_000.0 * 4.0;
    let s1 = &res.raw_probe_buffers[0];
    let s2 = &res.raw_probe_buffers[1];

    let (idx1, pk1) = s1
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let (idx2, pk2) = s2
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    assert!(*pk1 > 2.0, "mic1 peak too small: {pk1}");
    assert!(*pk2 > 1.0, "mic2 peak too small: {pk2}");

    let dt_arrival = ((idx2 - idx1) as f64) / fs;
    let c_meas = 0.12 / dt_arrival;
    let rel_err = (c_meas - AIR_C).abs() / AIR_C;
    assert!(
        rel_err < 0.03,
        "Measured air speed {c_meas:.2} m/s vs expected {AIR_C:.2} m/s (err={:.2}%)",
        rel_err * 100.0
    );

    // Check PML absorption at mic2 after pulse has exited domain (t > 1.25 ms; reflected wave from x=0.56 would return around t=1.2-1.7 ms)
    let tail_start = (0.00125 * fs) as usize;
    let tail_max = s2[tail_start..]
        .iter()
        .fold(0.0_f32, |acc, &x| acc.max(x.abs()));
    let refl_ratio = (tail_max as f64) / (*pk2 as f64);
    let refl_db = 20.0 * refl_ratio.max(1e-12).log10();
    assert!(
        refl_db < -35.0,
        "PML reflection too high: {refl_db:.2} dB (ratio={refl_ratio:.5})"
    );
}

#[test]
fn test_two_phase_water_air_interface_stability() {
    // Part A: 2000+ acoustic steps of quiescent air-water tank under gravity + surface tension
    let dx = 0.002_f32;
    let mut scene = Scene::minimal(
        "test_hydrostatic_stability",
        AmrDomainConfig {
            origin: [-0.064, -0.064, -0.064],
            dx0: dx,
            nx0: 64,
            ny0: 64,
            nz0: 64,
            pml_thickness: 8,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![],
        },
    );
    scene.background_water.push(GeometryShape::HalfSpaceZ { z_max: 0.0 });
    scene.enable_flow_advection = true;

    let mut engine = MultiphysicsEngine::new(scene, None).expect("init hydrostatic tank");
    // dt_ac ~ 0.38 * 0.002 / 1500 ~ 5.06e-7 s -> 0.0012 s is ~2370 acoustic steps!
    let res = engine
        .run(&SimRunConfig {
            duration_s: 0.0012,
            sample_rate: 48_000,
            oversample_factor: 4,
            telemetry_interval_s: 0.0004,
            output_dir: None,
            water_surface_z_for_diag: None,
            use_cuda_graph: true,
            cfl_safety: 0.38,
            print_progress: false,
            render_export: None,
            ..Default::default()
        })
        .expect("run hydrostatic stability");

    assert_eq!(res.telemetry.nan_count, 0);
    assert!(
        res.telemetry.total_substeps >= 2000,
        "Expected >= 2000 substeps, got {}",
        res.telemetry.total_substeps
    );
    assert!(
        res.telemetry.water_volume_rel_error < 0.005,
        "Water volume error {} >= 0.5%",
        res.telemetry.water_volume_rel_error
    );
    let max_u = res
        .timeseries
        .iter()
        .fold(0.0_f64, |acc, r| acc.max(r.u_max));
    assert!(max_u < 1e-3, "Spurious hydrostatic velocity too high: {max_u}");

    // Part B: Underwater acoustic wave speed test (c_w = 1480 m/s +- 3%)
    let mut uw_scene = Scene::minimal(
        "test_underwater_speed",
        AmrDomainConfig {
            origin: [0.0, 0.0, -0.100],
            dx0: dx,
            nx0: 120,
            ny0: 48,
            nz0: 64,
            pml_thickness: 8,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![],
        },
    );
    uw_scene
        .background_water
        .push(GeometryShape::HalfSpaceZ { z_max: 0.0 });
    uw_scene.gaussian_pulses.push(GaussianPulseSpec {
        center: [0.060, 0.048, -0.052],
        sigma_r: 0.006,
        amplitude_pa: 500.0,
    });
    uw_scene.microphones.push(MicrophoneSpec {
        name: "h1".to_string(),
        position: [0.100, 0.048, -0.052], // d1 = 0.04 m
        is_hydrophone: true,
    });
    uw_scene.microphones.push(MicrophoneSpec {
        name: "h2".to_string(),
        position: [0.160, 0.048, -0.052], // d2 = 0.10 m (delta_d = 0.06 m)
        is_hydrophone: true,
    });

    let mut uw_engine = MultiphysicsEngine::new(uw_scene, None).expect("init underwater wave");
    let uw_res = uw_engine
        .run(&SimRunConfig {
            duration_s: 0.00018,
            sample_rate: 96_000,
            oversample_factor: 8, // 768 kHz probe rate (1.3 us resolution for fast 1480 m/s wave)
            telemetry_interval_s: 0.00009,
            output_dir: None,
            water_surface_z_for_diag: None,
            use_cuda_graph: true,
            cfl_safety: 0.38,
            print_progress: false,
            render_export: None,
            ..Default::default()
        })
        .expect("run underwater speed");

    assert_eq!(uw_res.telemetry.nan_count, 0);
    let fs_uw = 96_000.0 * 8.0;
    let (i1, pk1) = uw_res.raw_probe_buffers[0]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let (i2, pk2) = uw_res.raw_probe_buffers[1]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    assert!(*pk1 > 10.0 && *pk2 > 4.0, "Weak hydrophone signal: {pk1}, {pk2}");
    let c_uw_meas = 0.06 / (((i2 - i1) as f64) / fs_uw);
    let err_uw = (c_uw_meas - WATER_C).abs() / WATER_C;
    assert!(
        err_uw < 0.03,
        "Measured water speed {c_uw_meas:.2} vs expected {WATER_C:.2} (err={:.2}%)",
        err_uw * 100.0
    );
}

#[test]
fn test_amr_multilevel_acoustic_transmission() {
    // 3-level nested AMR in air: Level 0 (dx=4mm, 72^3), Level 1 (dx=2mm, 56^3), Level 2 (dx=1mm, 48^3)
    let mut scene = Scene::minimal(
        "test_amr_transmission",
        AmrDomainConfig {
            origin: [-0.144, -0.144, -0.144],
            dx0: 0.004,
            nx0: 72,
            ny0: 72,
            nz0: 72,
            pml_thickness: 10,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![
                AmrPatchConfig {
                    refine_ratio: 2,
                    off_x: 22,
                    off_y: 22,
                    off_z: 22,
                    nx: 56,
                    ny: 56,
                    nz: 56,
                    ..Default::default()
                },
                AmrPatchConfig {
                    refine_ratio: 2,
                    off_x: 16,
                    off_y: 16,
                    off_z: 16,
                    nx: 48,
                    ny: 48,
                    nz: 48,
                    ..Default::default()
                },
            ],
        },
    );
    scene.fluid.gravity = [0.0, 0.0, 0.0];
    // Pulse at center (0, 0, 0) inside Level 2
    scene.gaussian_pulses.push(GaussianPulseSpec {
        center: [0.0, 0.0, 0.0],
        sigma_r: 0.008,
        amplitude_pa: 80.0,
    });
    // Probe inside Level 2 & two probes outside Level 1 in Level 0 (Level 1 ends at +0.056m, PML starts at +0.104m)
    scene.microphones.push(MicrophoneSpec {
        name: "inside_l2".to_string(),
        position: [0.008, 0.0, 0.0],
        is_hydrophone: false,
    });
    scene.microphones.push(MicrophoneSpec {
        name: "outside_l0_near".to_string(),
        position: [0.064, 0.0, 0.0],
        is_hydrophone: false,
    });
    scene.microphones.push(MicrophoneSpec {
        name: "outside_l0_far".to_string(),
        position: [0.098, 0.0, 0.0],
        is_hydrophone: false,
    });

    let mut engine = MultiphysicsEngine::new(scene, None).expect("init AMR test");
    let res = engine
        .run(&SimRunConfig {
            duration_s: 0.0007,
            sample_rate: 96_000,
            oversample_factor: 4,
            telemetry_interval_s: 0.0002,
            output_dir: None,
            water_surface_z_for_diag: None,
            use_cuda_graph: true,
            cfl_safety: 0.38,
            print_progress: false,
            render_export: None,
            ..Default::default()
        })
        .expect("run AMR test");

    assert_eq!(res.telemetry.nan_count, 0);
    let fs = 96_000.0 * 4.0;
    let s_in = &res.raw_probe_buffers[0];
    let s_out1 = &res.raw_probe_buffers[1];
    let s_out2 = &res.raw_probe_buffers[2];

    let (_, pk_in) = s_in
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let (idx_out1, pk_out1) = s_out1
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let (idx_out2, pk_out2) = s_out2
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    assert!(*pk_in > 5.0);
    assert!(*pk_out1 > 1.0 && *pk_out2 > 0.7, "Pulse failed to transmit cleanly into Level 0: {pk_out1}, {pk_out2}");

    // 3D N-wave leading peak sits at r - sigma_r = c * t_peak
    let t_out1 = (idx_out1 as f64) / fs;
    let c_nwave = (0.064 - 0.008) / t_out1;
    assert!(
        (c_nwave - AIR_C).abs() / AIR_C < 0.04,
        "AMR N-wave peak speed {c_nwave:.2} m/s diverged from {AIR_C:.2} m/s"
    );

    // Two-probe differential arrival cancels sigma_r offset identically
    let c_delta = (0.098 - 0.064) / (((idx_out2 - idx_out1) as f64) / fs);
    assert!(
        (c_delta - AIR_C).abs() / AIR_C < 0.05,
        "AMR two-probe wave speed {c_delta:.2} m/s diverged from {AIR_C:.2} m/s"
    );

    // Verify no strong reflection returns inside Level 2 after pulse leaves (t > 0.30 ms)
    let late_idx = (0.00030 * fs) as usize;
    let late_max = s_in[late_idx..]
        .iter()
        .fold(0.0_f32, |acc, &x| acc.max(x.abs()));
    let spurious_ratio = (late_max as f64) / (*pk_in as f64);
    assert!(
        spurious_ratio < 0.10,
        "AMR boundary reflection too high: {spurious_ratio:.4}"
    );
}

#[test]
fn test_wav_and_telemetry_io() {
    let dir = std::env::temp_dir().join("audio_sim_m1_io_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    for &sr in &[48_000u32, 96_000u32] {
        let oversample = 4usize;
        let n = (sr as usize) * oversample / 10; // 0.1 s
        let fs_os = (sr as f64) * (oversample as f64);
        let raw: Vec<f32> = (0..n)
            .map(|i| {
                let t = (i as f64) / fs_os;
                (12.5 * (2.0 * PI * 1000.0 * t).sin() + 100.0) as f32 // 100 Pa DC offset + 1 kHz tone
            })
            .collect();

        let wav_path = dir.join(format!("test_{sr}.wav"));
        let stats = write_probe_wav_f32(&wav_path, &raw, oversample, sr, 20.0).expect("write wav");
        assert_eq!(stats.sample_rate, sr);
        assert!(stats.peak_pa > 10.0 && stats.peak_pa < 15.0);
        assert!((stats.normalized_peak - 0.89125).abs() < 0.02);

        let (spec, samples) = read_wav_f32(&wav_path).expect("read wav");
        assert_eq!(spec.sample_rate, sr);
        assert_eq!(spec.bits_per_sample, 32);
        assert_eq!(spec.sample_format, SampleFormat::Float);
        assert!(samples.iter().all(|x| x.is_finite()));
    }

    let tele = RunTelemetry {
        scenario: "io_test".to_string(),
        compute_capability: "sm_120".to_string(),
        ..Default::default()
    };
    write_telemetry_json(dir.join("tele.json"), &tele).expect("write json");
    write_timeseries_csv(dir.join("ts.csv"), &[TimeSeriesRow::default()]).expect("write csv");
    assert!(dir.join("tele.json").exists());
    assert!(dir.join("ts.csv").exists());
    let _ = std::fs::remove_dir_all(&dir);
}
