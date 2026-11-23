use std::collections::BTreeMap;
use std::path::Path;

use sim_core::*;
use sim_cuda::DEFAULT_VRAM_CAP_BYTES;
use sim_io::{write_telemetry_json, write_timeseries_csv};

fn main() -> Result<()> {
    let out_dir = Path::new("output");
    std::fs::create_dir_all(out_dir)?;

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

    let cfg = SimRunConfig {
        duration_s: 0.060,
        sample_rate: 48_000,
        oversample_factor: 4,
        telemetry_interval_s: 0.002,
        output_dir: Some(out_dir.to_path_buf()),
        water_surface_z_for_diag: None,
        use_cuda_graph: false,
        cfl_safety: 0.38,
        print_progress: true,
        render_export: None,
        ..Default::default()
    };

    let mut engine = MultiphysicsEngine::new(scene, None)?;
    let mut res = engine.run(&cfg)?;

    let px = engine.particles.as_ref().unwrap().px.to_vec()?;
    let dmg = engine.particles.as_ref().unwrap().damage.to_vec()?;
    let notch_tip_x = -0.5 * size_x + notch_x_len;
    let mut max_cracked_x = notch_tip_x;
    for (&x, &d) in px.iter().zip(dmg.iter()) {
        if d > 0.7 && x > max_cracked_x {
            max_cracked_x = x;
        }
    }
    let crack_fraction = ((max_cracked_x - notch_tip_x) / (size_x - notch_x_len)).clamp(0.0, 1.0) as f64;
    let summary = &res.telemetry.probes[0];

    let mut notes = BTreeMap::new();
    notes.insert("crack_propagation_fraction".to_string(), crack_fraction);
    notes.insert("final_fractured_fraction".to_string(), res.timeseries.last().map(|r| r.fractured_fraction).unwrap_or(0.0));
    res.telemetry.notes = notes;
    write_telemetry_json(out_dir.join("rip_paper_telemetry.json"), &res.telemetry)?;
    write_timeseries_csv(out_dir.join("rip_paper_telemetry.csv"), &res.timeseries)?;

    println!("\n=== Validation Test 4 Results: rip_paper ===");
    println!(
        "Crack extent = {:.1}% | peak={:.3} Pa | centroid={:.1} Hz | kurtosis={:.2} | VRAM={:.2} MB | NaNs={}",
        crack_fraction * 100.0,
        summary.peak_pa,
        summary.spectral_centroid_hz,
        summary.kurtosis,
        (res.telemetry.peak_vram_tracked_bytes as f64) / 1e6,
        res.telemetry.nan_count
    );

    assert_eq!(res.telemetry.nan_count, 0);
    assert!(res.telemetry.peak_vram_tracked_bytes < DEFAULT_VRAM_CAP_BYTES);
    assert!(
        crack_fraction > 0.30,
        "Expected progressive paper tearing > 30%, got {:.1}%",
        crack_fraction * 100.0
    );
    assert!(summary.peak_pa > 0.1, "Expected peak_pa > 0.1 Pa, got {}", summary.peak_pa);
    assert!(
        summary.spectral_centroid_hz > 1000.0,
        "Expected broadband centroid > 1000 Hz, got {}",
        summary.spectral_centroid_hz
    );
    assert!(
        summary.kurtosis > 4.0,
        "Expected tearing transient kurtosis > 4.0, got {}",
        summary.kurtosis
    );

    Ok(())
}
