use std::collections::BTreeMap;
use std::path::Path;

use sim_core::*;
use sim_cuda::DEFAULT_VRAM_CAP_BYTES;
use sim_io::{write_telemetry_json, write_timeseries_csv};

fn main() -> Result<()> {
    let out_dir = Path::new("output");
    std::fs::create_dir_all(out_dir)?;

    let mut scene = Scene::minimal(
        "drop_on_wet_leaves",
        AmrDomainConfig {
            origin: [-0.08, -0.08, -0.02],
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
        origin: [-0.044, -0.044, -0.004],
        dx: 0.0025,
        nx: 36,
        ny: 36,
        nz: 20,
        has_floor: true,
        floor_z: 0.0,
        floor_mu: 0.4,
    });

    // 6 stacked curved wet leaves with varied yaw, dome & saddle camber
    let leaf_z = [0.002_f32, 0.0055, 0.009, 0.0125, 0.016, 0.0195];
    let leaf_yaw = [0.0_f32, 0.55, -0.45, 1.10, -0.95, 0.25];
    let leaf_dome = [0.0035_f32, 0.0042, 0.0038, 0.0048, 0.0040, 0.0050];
    let leaf_saddle = [0.0015_f32, -0.0020, 0.0022, -0.0018, 0.0025, -0.0022];
    let leaf_xy = [
        [0.000_f32, 0.000],
        [0.004, -0.003],
        [-0.004, 0.003],
        [0.002, 0.004],
        [-0.003, -0.003],
        [0.001, -0.001],
    ];

    for idx in 0..6 {
        scene.solids.push(SolidBodySpec {
            name: format!("wet_leaf_{idx}"),
            body_id: (idx + 1) as i32,
            shape: GeometryShape::CurvedShell {
                center: [leaf_xy[idx][0], leaf_xy[idx][1], leaf_z[idx]],
                size_x: 0.044,
                size_y: 0.032,
                nx: 14,
                ny: 10,
                dome_amp: leaf_dome[idx],
                saddle_amp: leaf_saddle[idx],
                yaw_rad: leaf_yaw[idx],
                notch_x_len: 0.0,
                notch_y_pos: 0.0,
                notch_half_w: 0.0,
            },
            material: SolidMaterial::wet_leaf(2.8e-4, 0.85),
            initial_velocity: [0.0, 0.0, 0.0],
            orientation_euler_xyz: [0.0, 0.0, 0.0],
            particle_spacing: 0.0032,
            gc_heterogeneity_amp: 0.25,
            is_kinematic_rigid: false,
            bcs: vec![],
        });
    }

    // Heavy flat steel plate dropping onto the wet leaf stack
    scene.solids.push(SolidBodySpec {
        name: "steel_plate".to_string(),
        body_id: 0,
        shape: GeometryShape::Box {
            min: [-0.026, -0.022, 0.025],
            max: [0.026, 0.022, 0.030],
        },
        material: SolidMaterial::structural_steel(),
        initial_velocity: [0.0, 0.0, -1.8],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.0025,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![],
    });

    scene.microphones.push(MicrophoneSpec {
        name: "drop_on_wet_leaves".to_string(),
        position: [0.0, 0.0, 0.055],
        is_hydrophone: false,
    });

    let cfg = SimRunConfig {
        duration_s: 0.055,
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

    let frac_final = res
        .timeseries
        .last()
        .map(|r| r.fractured_fraction)
        .unwrap_or(0.0);
    let summary = &res.telemetry.probes[0];

    let mut notes = BTreeMap::new();
    notes.insert("final_fractured_fraction".to_string(), frac_final);
    notes.insert("crunch_kurtosis".to_string(), summary.kurtosis);
    notes.insert("crunch_centroid_hz".to_string(), summary.spectral_centroid_hz);
    res.telemetry.notes = notes;
    write_telemetry_json(out_dir.join("drop_on_wet_leaves_telemetry.json"), &res.telemetry)?;
    write_timeseries_csv(out_dir.join("drop_on_wet_leaves_telemetry.csv"), &res.timeseries)?;

    println!("\n=== Validation Test 5 Results: drop_on_wet_leaves ===");
    println!(
        "Fractured = {:.2}% | peak={:.3} Pa | centroid={:.1} Hz | kurtosis={:.2} | VRAM={:.2} MB | NaNs={}",
        frac_final * 100.0,
        summary.peak_pa,
        summary.spectral_centroid_hz,
        summary.kurtosis,
        (res.telemetry.peak_vram_tracked_bytes as f64) / 1e6,
        res.telemetry.nan_count
    );

    assert_eq!(res.telemetry.nan_count, 0);
    assert!(res.telemetry.peak_vram_tracked_bytes < DEFAULT_VRAM_CAP_BYTES);
    assert!(
        frac_final > 0.03,
        "Expected leaf brittle fracture > 3%, got {:.2}%",
        frac_final * 100.0
    );
    assert!(summary.peak_pa > 0.1, "Expected peak_pa > 0.1 Pa, got {}", summary.peak_pa);
    assert!(
        summary.kurtosis > 4.0,
        "Expected crunch transient kurtosis > 4.0, got {}",
        summary.kurtosis
    );

    Ok(())
}
