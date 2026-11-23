use std::f32::consts::FRAC_PI_2;

use sim_core::*;

fn run_cantilever_vibration(mat: SolidMaterial, yaw_rad: f32) -> (f64, f64) {
    let mut scene = Scene::minimal(
        "test_cantilever",
        AmrDomainConfig {
            origin: [-0.04, -0.04, -0.04],
            dx0: 0.004,
            nx0: 36,
            ny0: 24,
            nz0: 24,
            pml_thickness: 6,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![],
        },
    );
    scene.fluid.gravity = [0.0, 0.0, 0.0];
    scene.mpm_grid = Some(MpmGridConfig {
        origin: [-0.01, -0.02, -0.02],
        dx: 0.002,
        nx: 52,
        ny: 20,
        nz: 20,
        has_floor: false,
        floor_z: -1.0,
        floor_mu: 0.0,
    });
    scene.solids.push(SolidBodySpec {
        name: "bar".to_string(),
        body_id: 0,
        shape: GeometryShape::Box {
            min: [0.0, -0.006, -0.003],
            max: [0.080, 0.006, 0.003],
        },
        material: mat,
        initial_velocity: [0.0, 0.0, -0.35],
        orientation_euler_xyz: [0.0, 0.0, yaw_rad],
        particle_spacing: 0.001,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![SolidBoundaryCondition::ClampedRegion {
            min: [-0.001, -0.01, -0.01],
            max: [0.006, 0.01, 0.01],
        }],
    });
    scene.microphones.push(MicrophoneSpec {
        name: "mic".to_string(),
        position: [0.055, 0.0, 0.020],
        is_hydrophone: false,
    });

    let mut engine = MultiphysicsEngine::new(scene, None).expect("init bar engine");
    let res = engine
        .run(&SimRunConfig {
            duration_s: 0.012,
            sample_rate: 48_000,
            oversample_factor: 4,
            telemetry_interval_s: 0.004,
            output_dir: None,
            water_surface_z_for_diag: None,
            use_cuda_graph: true,
            cfl_safety: 0.38,
            print_progress: false,
            render_export: None,
            ..Default::default()
        })
        .expect("run bar vibration");

    assert_eq!(res.telemetry.nan_count, 0);
    let sum = &res.telemetry.probes[0];
    (sum.peak_freq_hz, sum.peak_pa)
}

#[test]
fn test_mpm_orthotropic_wave_speeds() {
    let (f_oak_0, p_oak_0) = run_cantilever_vibration(SolidMaterial::oak_wood(), 0.0);
    let (f_oak_90, p_oak_90) = run_cantilever_vibration(SolidMaterial::oak_wood(), FRAC_PI_2);
    let (f_steel_0, p_steel_0) = run_cantilever_vibration(SolidMaterial::structural_steel(), 0.0);
    let (f_steel_90, p_steel_90) =
        run_cantilever_vibration(SolidMaterial::structural_steel(), FRAC_PI_2);

    println!(
        "Oak 0deg: {f_oak_0:.1} Hz (pk={p_oak_0:.3} Pa) | Oak 90deg: {f_oak_90:.1} Hz (pk={p_oak_90:.3} Pa)"
    );
    println!(
        "Steel 0deg: {f_steel_0:.1} Hz (pk={p_steel_0:.3} Pa) | Steel 90deg: {f_steel_90:.1} Hz (pk={p_steel_90:.3} Pa)"
    );

    assert!(p_oak_0 > 0.02 && p_oak_90 > 0.02);
    assert!(
        f_oak_0 > f_oak_90 * 1.4,
        "Expected orthotropic oak 0deg ({f_oak_0:.1} Hz) > 1.4x 90deg ({f_oak_90:.1} Hz)"
    );
    let steel_diff = (f_steel_0 - f_steel_90).abs() / f_steel_0.max(1.0);
    assert!(
        steel_diff < 0.05,
        "Expected isotropic steel frequencies to match within 5%, got {f_steel_0:.1} vs {f_steel_90:.1}"
    );
}

#[test]
fn test_plate_bending_radiation() {
    let mut scene = Scene::minimal(
        "test_plate_radiation",
        AmrDomainConfig {
            origin: [-0.072, -0.072, -0.056],
            dx0: 0.004,
            nx0: 36,
            ny0: 36,
            nz0: 32,
            pml_thickness: 6,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![AmrPatchConfig {
                refine_ratio: 2,
                off_x: 7,
                off_y: 7,
                off_z: 8,
                nx: 44,
                ny: 44,
                nz: 32,
                ..Default::default()
            }],
        },
    );
    scene.fluid.gravity = [0.0, 0.0, 0.0];
    scene.mpm_grid = Some(MpmGridConfig {
        origin: [-0.048, -0.048, -0.016],
        dx: 0.002,
        nx: 48,
        ny: 48,
        nz: 16,
        has_floor: false,
        floor_z: -1.0,
        floor_mu: 0.0,
    });
    scene.solids.push(SolidBodySpec {
        name: "plate".to_string(),
        body_id: 0,
        shape: GeometryShape::Box {
            min: [-0.040, -0.040, -0.002],
            max: [0.040, 0.040, 0.002],
        },
        material: SolidMaterial::structural_steel(),
        initial_velocity: [0.0, 0.0, -0.50],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.001,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![
            SolidBoundaryCondition::ClampedRegion {
                min: [-0.042, -0.042, -0.004],
                max: [-0.035, 0.042, 0.004],
            },
            SolidBoundaryCondition::ClampedRegion {
                min: [0.035, -0.042, -0.004],
                max: [0.042, 0.042, 0.004],
            },
        ],
    });
    scene.microphones.push(MicrophoneSpec {
        name: "air_mic".to_string(),
        position: [0.0, 0.0, 0.025],
        is_hydrophone: false,
    });

    let mut engine = MultiphysicsEngine::new(scene, None).expect("init plate radiation");
    let res = engine
        .run(&SimRunConfig {
            duration_s: 0.008,
            sample_rate: 48_000,
            oversample_factor: 4,
            telemetry_interval_s: 0.002,
            output_dir: None,
            water_surface_z_for_diag: None,
            use_cuda_graph: true,
            cfl_safety: 0.38,
            print_progress: false,
            render_export: None,
            ..Default::default()
        })
        .expect("run plate radiation");

    assert_eq!(res.telemetry.nan_count, 0);
    let pk = res.telemetry.probes[0].peak_pa;
    println!(
        "Plate bending radiation: peak_pa={pk:.4} Pa, peak_freq={:.1} Hz, wall={:.2}s",
        res.telemetry.probes[0].peak_freq_hz, res.telemetry.wall_time_s
    );
    assert!(pk > 0.05, "Expected measurable radiation > 0.05 Pa, got {pk}");
}
