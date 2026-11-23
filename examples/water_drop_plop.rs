use std::collections::BTreeMap;
use std::path::Path;

use sim_core::*;
use sim_cuda::DEFAULT_VRAM_CAP_BYTES;
use sim_io::{apply_highpass_2nd_order, write_telemetry_json, write_timeseries_csv};

fn main() -> Result<()> {
    let out_dir = Path::new("output");
    std::fs::create_dir_all(out_dir)?;

    // Exact face-aligned 3-level nested AMR:
    // L0: [-0.1280, +0.1280]^3, dx0 = 3.20 mm, 80^3, PML = 8
    // L1: off=[32,32,32], r=4, 64^3 (dx1 = 0.80 mm, covers [-25.6, +25.6]^3 mm)
    // L2: off=[24,24,18], r=4, 64x64x84 (dx2 = 0.20 mm, covers [-6.4,+6.4]x[-6.4,+6.4]x[-11.2,+5.6] mm)
    let r_drop = 0.0015_f32; // 3.0 mm diameter sphere droplet
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

    let cfg = SimRunConfig {
        duration_s: 0.025,
        sample_rate: 96_000,
        oversample_factor: 2,
        telemetry_interval_s: 0.0005,
        output_dir: Some(out_dir.to_path_buf()),
        water_surface_z_for_diag: Some(0.0),
        use_cuda_graph: true,
        cfl_safety: 0.45,
        print_progress: true,
        render_export: None,
        ..Default::default()
    };

    let mut engine = MultiphysicsEngine::new(scene, None)?;
    let mut res = engine.run(&cfg)?;

    let fs_os = (cfg.sample_rate as f64) * (cfg.oversample_factor as f64);

    // Find first pinch-off and peak trapped subsurface bubble row
    let first_pinch = res
        .timeseries
        .iter()
        .find(|r| r.bubble_radius_mm >= 0.20 && r.minnaert_freq_hz > 500.0);
    let max_bubble = res
        .timeseries
        .iter()
        .max_by(|a, b| a.bubble_radius_mm.partial_cmp(&b.bubble_radius_mm).unwrap());

    let (t_pinch_s, r_b_mm, f_minnaert_theory) = if let Some(b) = max_bubble {
        let t0 = first_pinch.map(|p| p.time_s).unwrap_or(b.time_s);
        (t0, b.bubble_radius_mm, b.minnaert_freq_hz)
    } else {
        (0.0, 0.0, 0.0)
    };

    // Analyze acoustic bubble ringdown around pinch-off after rejecting the ~60 Hz
    // hydrodynamic crater-collapse Bernoulli pressure wave with a zero-phase 4th-order HPF.
    let analyze_bubble_mic = |buf: &[f32]| -> (f64, f64, f64) {
        let mut hp: Vec<f64> = buf.iter().map(|&x| x as f64).collect();
        let fs_u32 = fs_os.round() as u32;
        apply_highpass_2nd_order(&mut hp, fs_u32, 2000.0);
        hp.reverse();
        apply_highpass_2nd_order(&mut hp, fs_u32, 2000.0);
        hp.reverse();

        let slice_ms = |t0_s: f64, t1_s: f64| -> &[f64] {
            let s = (t0_s.max(0.0) * fs_os) as usize;
            let e = ((t1_s.max(0.0) * fs_os) as usize).min(hp.len());
            if s < e && e - s >= 64 {
                &hp[s..e]
            } else {
                &hp[..]
            }
        };

        let win_band = slice_ms(t_pinch_s - 0.0005, t_pinch_s + 0.0012);
        let win_early = slice_ms(t_pinch_s - 0.0005, t_pinch_s + 0.0005);
        let win_late = slice_ms(t_pinch_s, t_pinch_s + 0.0010);

        let (f_peak_band, _) = dominant_freq_and_centroid(win_band, fs_os, 2500.0, 25000.0);
        let (f_early, _) = dominant_freq_and_centroid(win_early, fs_os, 2500.0, 25000.0);
        let (f_late, _) = dominant_freq_and_centroid(win_late, fs_os, 2500.0, 25000.0);
        (f_peak_band, f_early, f_late)
    };

    let (air_f_band, air_f_early, air_f_late) = analyze_bubble_mic(&res.raw_probe_buffers[0]);
    let (hydro_f_band, hydro_f_early, hydro_f_late) = analyze_bubble_mic(&res.raw_probe_buffers[1]);

    let mut notes = BTreeMap::new();
    notes.insert("pinch_off_time_ms".to_string(), t_pinch_s * 1e3);
    notes.insert("peak_bubble_radius_mm".to_string(), r_b_mm);
    notes.insert("minnaert_freq_hz".to_string(), f_minnaert_theory);
    notes.insert("air_mic_band_peak_hz".to_string(), air_f_band);
    notes.insert("air_mic_chirp_early_hz".to_string(), air_f_early);
    notes.insert("air_mic_chirp_late_hz".to_string(), air_f_late);
    notes.insert("hydro_band_peak_hz".to_string(), hydro_f_band);
    notes.insert("hydro_chirp_early_hz".to_string(), hydro_f_early);
    notes.insert("hydro_chirp_late_hz".to_string(), hydro_f_late);
    res.telemetry.notes = notes;

    // Save diagnostics before running assertions so crater/bubble CSV is always on disk
    write_telemetry_json(out_dir.join("water_drop_plop_telemetry.json"), &res.telemetry)?;
    write_timeseries_csv(out_dir.join("water_drop_plop_telemetry.csv"), &res.timeseries)?;

    println!("\n=== Validation Test 3 Results: water_drop_plop ===");
    println!(
        "Bubble: t_pinch={:.2}ms | R_b={:.3}mm | Minnaert_f={:.1}Hz",
        t_pinch_s * 1e3,
        r_b_mm,
        f_minnaert_theory
    );
    println!(
        "Air mic   : peak={:.3}Pa | summary_f={:.1}Hz | band_f={:.1}Hz | chirp {:.1}->{:.1}Hz",
        res.telemetry.probes[0].peak_pa,
        res.telemetry.probes[0].peak_freq_hz,
        air_f_band,
        air_f_early,
        air_f_late
    );
    println!(
        "Hydrophone: peak={:.3}Pa | summary_f={:.1}Hz | band_f={:.1}Hz | chirp {:.1}->{:.1}Hz",
        res.telemetry.probes[1].peak_pa,
        res.telemetry.probes[1].peak_freq_hz,
        hydro_f_band,
        hydro_f_early,
        hydro_f_late
    );
    println!(
        "Water vol rel err = {:.3e} | VRAM = {:.2} MB | NaNs = {}",
        res.telemetry.water_volume_rel_error,
        (res.telemetry.peak_vram_tracked_bytes as f64) / 1e6,
        res.telemetry.nan_count
    );

    assert_eq!(res.telemetry.nan_count, 0);
    assert!(res.telemetry.peak_vram_tracked_bytes < DEFAULT_VRAM_CAP_BYTES);
    assert!(
        (0.25..=1.2).contains(&r_b_mm),
        "Expected pinched-off subsurface bubble radius R_b in [0.25, 1.2] mm, got {r_b_mm:.3} mm"
    );
    assert!(f_minnaert_theory > 0.0);

    let air_err = ((air_f_band - f_minnaert_theory) / f_minnaert_theory).abs();
    let hydro_err = ((hydro_f_band - f_minnaert_theory) / f_minnaert_theory).abs();
    assert!(
        air_err <= 0.35,
        "Air mic peak {air_f_band:.1} Hz deviates {:.1}% from Minnaert {f_minnaert_theory:.1} Hz",
        air_err * 100.0
    );
    assert!(
        hydro_err <= 0.35,
        "Hydrophone peak {hydro_f_band:.1} Hz deviates {:.1}% from Minnaert {f_minnaert_theory:.1} Hz",
        hydro_err * 100.0
    );
    assert!(
        hydro_f_late > hydro_f_early,
        "Expected positive upward STFT chirp in hydrophone: early={hydro_f_early:.1}Hz, late={hydro_f_late:.1}Hz"
    );
    assert!(
        air_f_late > air_f_early,
        "Expected positive upward STFT chirp in air mic: early={air_f_early:.1}Hz, late={air_f_late:.1}Hz"
    );

    Ok(())
}
