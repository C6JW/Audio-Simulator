use sim_core::*;

#[test]
fn test_phase_field_tensile_fracture() {
    let mut scene = Scene::minimal(
        "test_phase_field_tensile_fracture",
        AmrDomainConfig {
            origin: [-0.048, -0.048, -0.048],
            dx0: 0.003,
            nx0: 32,
            ny0: 32,
            nz0: 32,
            pml_thickness: 6,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![],
        },
    );
    scene.fluid.gravity = [0.0, 0.0, 0.0];
    scene.mpm_grid = Some(MpmGridConfig {
        origin: [-0.032, -0.022, -0.018],
        dx: 0.002,
        nx: 36,
        ny: 22,
        nz: 18,
        has_floor: false,
        floor_z: -1.0,
        floor_mu: 0.0,
    });

    scene.solids.push(SolidBodySpec {
        name: "notched_paper".to_string(),
        body_id: 0,
        shape: GeometryShape::CurvedShell {
            center: [0.0, 0.0, 0.0],
            size_x: 0.040,
            size_y: 0.020,
            nx: 20,
            ny: 10,
            dome_amp: 0.001,
            saddle_amp: 0.0,
            yaw_rad: 0.0,
            notch_x_len: 0.008,
            notch_y_pos: 0.0,
            notch_half_w: 0.0018,
        },
        material: SolidMaterial::cellulose_paper(1.1e-4),
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.002,
        gc_heterogeneity_amp: 0.25,
        is_kinematic_rigid: false,
        bcs: vec![
            SolidBoundaryCondition::ClampedRegion {
                min: [-0.025, -0.02, -0.01],
                max: [-0.016, 0.02, 0.01],
            },
            SolidBoundaryCondition::PrescribedVelocityRegion {
                min: [0.016, -0.02, -0.01],
                max: [0.025, 0.02, 0.01],
                velocity: [0.8, 0.0, 0.0],
            },
        ],
    });
    scene.microphones.push(MicrophoneSpec {
        name: "mic".to_string(),
        position: [0.0, 0.0, 0.022],
        is_hydrophone: false,
    });

    let mut engine = MultiphysicsEngine::new(scene, None).expect("init fracture test");
    let res = engine
        .run(&SimRunConfig {
            duration_s: 0.015,
            sample_rate: 48_000,
            oversample_factor: 4,
            telemetry_interval_s: 0.003,
            output_dir: None,
            water_surface_z_for_diag: None,
            use_cuda_graph: false,
            cfl_safety: 0.38,
            print_progress: false,
            render_export: None,
            ..Default::default()
        })
        .expect("run tensile fracture");

    let dmg = engine.particles.as_ref().unwrap().damage.to_vec().unwrap();
    let max_d = dmg.iter().copied().fold(0.0_f32, f32::max);
    let frac = res.timeseries.last().map(|r| r.fractured_fraction).unwrap_or(0.0);
    let pk = res.telemetry.probes[0].peak_pa;
    println!("Tensile fracture: max_damage={max_d:.3} | frac={frac:.3} | peak_pa={pk:.3} Pa");

    assert_eq!(res.telemetry.nan_count, 0);
    assert!(max_d > 0.85, "Expected crack nucleation max_d > 0.85, got {max_d}");
    assert!(frac > 0.05, "Expected fractured_fraction > 0.05, got {frac}");
    assert!(pk > 0.05, "Expected fracture acoustic emission > 0.05 Pa, got {pk}");
}

#[test]
fn test_shell_snap_through_buckling() {
    let mut mat = SolidMaterial::wet_leaf(2.8e-4, 0.0);
    mat.g_c = 0.0; // Pure elastic-viscoplastic snap-through without tearing

    let mut scene = Scene::minimal(
        "test_shell_snap_through",
        AmrDomainConfig {
            origin: [-0.056, -0.056, -0.056],
            dx0: 0.0035,
            nx0: 32,
            ny0: 32,
            nz0: 32,
            pml_thickness: 6,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![],
        },
    );
    scene.mpm_grid = Some(MpmGridConfig {
        origin: [-0.032, -0.032, -0.024],
        dx: 0.003,
        nx: 22,
        ny: 22,
        nz: 16,
        has_floor: false,
        floor_z: -1.0,
        floor_mu: 0.0,
    });

    scene.solids.push(SolidBodySpec {
        name: "buckling_dome".to_string(),
        body_id: 0,
        shape: GeometryShape::CurvedShell {
            center: [0.0, 0.0, 0.0],
            size_x: 0.044,
            size_y: 0.044,
            nx: 14,
            ny: 14,
            dome_amp: 0.004,
            saddle_amp: 0.0,
            yaw_rad: 0.0,
            notch_x_len: 0.0,
            notch_y_pos: 0.0,
            notch_half_w: 0.0,
        },
        material: mat,
        initial_velocity: [0.0, 0.0, -2.5],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.0031,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![
            SolidBoundaryCondition::ClampedRegion {
                min: [-0.025, -0.025, -0.01],
                max: [-0.018, 0.025, 0.01],
            },
            SolidBoundaryCondition::ClampedRegion {
                min: [0.018, -0.025, -0.01],
                max: [0.025, 0.025, 0.01],
            },
            SolidBoundaryCondition::PrescribedVelocityRegion {
                min: [-0.004, -0.004, -0.01],
                max: [0.004, 0.004, 0.01],
                velocity: [0.0, 0.0, -1.2],
            },
        ],
    });
    scene.microphones.push(MicrophoneSpec {
        name: "mic".to_string(),
        position: [0.0, 0.0, 0.028],
        is_hydrophone: false,
    });

    let mut engine = MultiphysicsEngine::new(scene, None).expect("init snap-through test");
    let res = engine
        .run(&SimRunConfig {
            duration_s: 0.012,
            sample_rate: 48_000,
            oversample_factor: 4,
            telemetry_interval_s: 0.003,
            output_dir: None,
            water_surface_z_for_diag: None,
            use_cuda_graph: false,
            cfl_safety: 0.38,
            print_progress: false,
            render_export: None,
            ..Default::default()
        })
        .expect("run snap-through");

    let pz = engine.particles.as_ref().unwrap().pz.to_vec().unwrap();
    let min_z = pz.iter().copied().fold(f32::INFINITY, f32::min);
    let pk = res.telemetry.probes[0].peak_pa;
    println!("Snap-through buckling: min_z={min_z:.4} m | peak_pa={pk:.3} Pa");

    assert_eq!(res.telemetry.nan_count, 0);
    assert!(min_z < -0.001, "Expected dome inversion min_z < -0.001 m, got {min_z}");
    assert!(pk > 0.05, "Expected acoustic snap peak_pa > 0.05 Pa, got {pk}");
}

fn run_capillary_drop(wetness: f32) -> (f64, f64, u64) {
    let mut bottom_mat = SolidMaterial::wet_leaf(2.8e-4, wetness);
    bottom_mat.g_c = 0.0;
    let mut top_mat = SolidMaterial::wet_leaf(2.8e-4, wetness);
    top_mat.g_c = 0.0;

    let mut scene = Scene::minimal(
        format!("capillary_drop_{wetness:.1}"),
        AmrDomainConfig {
            origin: [-0.048, -0.048, -0.030],
            dx0: 0.003,
            nx0: 32,
            ny0: 32,
            nz0: 32,
            pml_thickness: 6,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![],
        },
    );
    scene.mpm_grid = Some(MpmGridConfig {
        origin: [-0.030, -0.030, -0.006],
        dx: 0.0025,
        nx: 24,
        ny: 24,
        nz: 16,
        has_floor: true,
        floor_z: -0.001,
        floor_mu: 0.3,
    });

    scene.solids.push(SolidBodySpec {
        name: "bottom_sheet".to_string(),
        body_id: 0,
        shape: GeometryShape::CurvedShell {
            center: [0.0, 0.0, 0.0],
            size_x: 0.040,
            size_y: 0.040,
            nx: 16,
            ny: 16,
            dome_amp: 0.0,
            saddle_amp: 0.0,
            yaw_rad: 0.0,
            notch_x_len: 0.0,
            notch_y_pos: 0.0,
            notch_half_w: 0.0,
        },
        material: bottom_mat,
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.0025,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![SolidBoundaryCondition::ClampedRegion {
            min: [-1.0, -1.0, -0.001],
            max: [1.0, 1.0, 0.001],
        }],
    });
    scene.solids.push(SolidBodySpec {
        name: "top_sheet".to_string(),
        body_id: 1,
        shape: GeometryShape::CurvedShell {
            center: [0.0, 0.0, 0.0022],
            size_x: 0.040,
            size_y: 0.040,
            nx: 16,
            ny: 16,
            dome_amp: 0.0,
            saddle_amp: 0.0,
            yaw_rad: 0.0,
            notch_x_len: 0.0,
            notch_y_pos: 0.0,
            notch_half_w: 0.0,
        },
        material: top_mat,
        initial_velocity: [0.0, 0.0, -1.2],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.0025,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![],
    });
    scene.microphones.push(MicrophoneSpec {
        name: "mic".to_string(),
        position: [0.0, 0.0, 0.030],
        is_hydrophone: false,
    });

    let mut engine = MultiphysicsEngine::new(scene, None).expect("init capillary drop");
    let res = engine
        .run(&SimRunConfig {
            duration_s: 0.010,
            sample_rate: 48_000,
            oversample_factor: 4,
            telemetry_interval_s: 0.002,
            output_dir: None,
            water_surface_z_for_diag: None,
            use_cuda_graph: false,
            cfl_safety: 0.38,
            print_progress: false,
            render_export: None,
            ..Default::default()
        })
        .expect("run capillary drop");

    let final_ke = res.timeseries.last().map(|r| r.solid_kinetic_energy).unwrap_or(0.0);
    let late_rms = res.telemetry.probes[0].late_rms_pa;
    (final_ke, late_rms, res.telemetry.nan_count)
}

#[test]
fn test_capillary_lubrication_damping() {
    let (dry_ke, dry_late, dry_nan) = run_capillary_drop(0.0);
    let (wet_ke, wet_late, wet_nan) = run_capillary_drop(0.9);
    println!("Dry: final_KE={dry_ke:.3e} J, late_RMS={dry_late:.3e} Pa | Wet: final_KE={wet_ke:.3e} J, late_RMS={wet_late:.3e} Pa");

    assert_eq!(dry_nan, 0);
    assert_eq!(wet_nan, 0);
    assert!(
        wet_ke < dry_ke,
        "Expected wet squeeze-film damping to yield lower final KE ({wet_ke:.3e} < {dry_ke:.3e})"
    );
}
