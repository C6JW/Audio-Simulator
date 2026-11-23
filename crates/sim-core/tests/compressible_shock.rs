use sim_core::*;

#[test]
fn test_compressible_spherical_blast_and_negative_phase() {
    let mut scene = Scene::minimal(
        "test_compressible_blast",
        AmrDomainConfig {
            origin: [-0.8, -0.8, -0.8],
            dx0: 0.025,
            nx0: 64,
            ny0: 64,
            nz0: 64,
            pml_thickness: 8,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![],
        },
    );
    scene.enable_compressible_euler = true;
    scene.fluid.gravity = [0.0, 0.0, 0.0];
    scene.fluid.bulk_viscosity_a = 0.0;
    scene.blast_regions.push(BlastRegionSpec {
        trigger_time_s: 0.0,
        center: [0.0, 0.0, 0.0],
        radius: 0.08,
        length: 0.0,
        is_cylinder_x: false,
        overpressure_pa: 4.0e5,
        density_kg_m3: 4.0,
        velocity: [0.0, 0.0, 0.0],
    });
    scene.microphones.push(MicrophoneSpec {
        name: "blast_mic".to_string(),
        position: [0.55, 0.0, 0.0],
        is_hydrophone: false,
    });

    let mut engine = MultiphysicsEngine::new(scene, None).expect("init euler blast");
    let res = engine
        .run(&SimRunConfig {
            duration_s: 0.008,
            pre_roll_s: 0.0,
            sample_rate: 48_000,
            oversample_factor: 2,
            telemetry_interval_s: 0.001,
            cfl_safety: 0.35,
            use_cuda_graph: false,
            print_progress: false,
            ..Default::default()
        })
        .expect("run euler blast");

    assert_eq!(res.telemetry.nan_count, 0);
    let sig = &res.raw_probe_buffers[0];
    let sr_os = 96_000.0_f64;
    let max_p = sig.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_p = sig.iter().copied().fold(f32::INFINITY, f32::min);
    let idx_peak = sig
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let t_peak = (idx_peak as f64) / sr_os;
    let t_acoustic = (0.55 - 0.08) / 343.0;
    println!(
        "Spherical blast: max_p={:.1} Pa | min_p={:.1} Pa | t_peak={:.3} ms (linear={:.3} ms)",
        max_p,
        min_p,
        t_peak * 1e3,
        t_acoustic * 1e3
    );

    assert!(max_p > 1000.0, "Expected strong blast shock > 1000 Pa, got {max_p}");
    assert!(
        min_p < -50.0,
        "Expected Friedlander rarefaction negative phase < -50 Pa, got {min_p}"
    );
    assert!(
        t_peak < t_acoustic + 1.5e-4,
        "Expected nonlinear shock arrival <= linear acoustic time ({t_acoustic:.5}s), got {t_peak:.5}s"
    );
}

#[test]
fn test_supersonic_projectile_mach_cone_nwave() {
    let mut scene = Scene::minimal(
        "test_supersonic_nwave",
        AmrDomainConfig {
            origin: [-1.6, -0.6, -0.6],
            dx0: 0.025,
            nx0: 128,
            ny0: 48,
            nz0: 48,
            pml_thickness: 6,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![],
        },
    );
    scene.enable_compressible_euler = true;
    scene.fluid.gravity = [0.0, 0.0, 0.0];
    scene.fluid.bulk_viscosity_a = 0.0;
    scene.projectiles.push(SupersonicProjectileSpec {
        trigger_time_s: 0.0,
        initial_pos: [-1.2, 0.0, 0.0],
        initial_vel: [900.0, 0.0, 0.0],
        mass_kg: 0.046,
        radius_m: 0.008,
        length_m: 0.06,
        impact_x_max: 1.5,
        impact_target_body_id: None,
        impact_radius_m: 0.05,
    });
    scene.microphones.push(MicrophoneSpec {
        name: "lateral_mic".to_string(),
        position: [0.4, 0.35, 0.0],
        is_hydrophone: false,
    });

    let mut engine = MultiphysicsEngine::new(scene, None).expect("init supersonic bullet");
    let res = engine
        .run(&SimRunConfig {
            duration_s: 0.004,
            pre_roll_s: 0.0,
            sample_rate: 48_000,
            oversample_factor: 2,
            telemetry_interval_s: 0.001,
            cfl_safety: 0.35,
            use_cuda_graph: false,
            print_progress: false,
            ..Default::default()
        })
        .expect("run supersonic bullet");

    assert_eq!(res.telemetry.nan_count, 0);
    let sig = &res.raw_probe_buffers[0];
    let max_p = sig.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_p = sig.iter().copied().fold(f32::INFINITY, f32::min);
    let idx_peak = sig
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let t_peak = (idx_peak as f64) / 96_000.0;
    println!(
        "Supersonic N-wave: max_p={:.1} Pa | min_p={:.1} Pa | t_peak={:.3} ms",
        max_p,
        min_p,
        t_peak * 1e3
    );

    assert!(max_p > 50.0, "Expected bow shock peak > 50 Pa, got {max_p}");
    assert!(
        min_p < -40.0,
        "Expected N-wave tail rarefaction < -40 Pa, got {min_p}"
    );
    assert!(
        t_peak > 1.8e-3 && t_peak < 3.6e-3,
        "Expected Mach cone arrival around 2.7 ms, got {t_peak:.5}s"
    );
}

fn run_bell_mode_freq(submerged: bool) -> (f64, f64, f64, f64) {
    let mut scene = Scene::minimal(
        if submerged { "bell_water" } else { "bell_air" },
        AmrDomainConfig {
            origin: [-0.11, -0.11, -0.10],
            dx0: 0.0055,
            nx0: 40,
            ny0: 40,
            nz0: 40,
            pml_thickness: 6,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![],
        },
    );
    scene.fluid.gravity = [0.0, 0.0, 0.0];
    if submerged {
        scene.background_water = vec![GeometryShape::HalfSpaceZ { z_max: 0.20 }];
    }

    let mut bell_mat = SolidMaterial::bell_bronze();
    bell_mat.is_shell = true;
    bell_mat.thickness = 0.0035;
    // Crown top [z=0.055, r=0.0] to open rim [z=0.0, r=0.055] (ordered bottom to top)
    let profile = vec![
        [0.000, 0.055],
        [0.015, 0.044],
        [0.035, 0.038],
        [0.050, 0.025],
        [0.055, 0.000],
    ];
    scene.solids.push(SolidBodySpec {
        name: "bronze_bell".to_string(),
        body_id: 0,
        shape: GeometryShape::RevolutionShell {
            center_xy: [0.0, 0.0],
            profile_zr: profile,
            n_theta: 32,
            n_meridian: 12,
            hyd_len: 0.035,
        },
        material: bell_mat,
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.005,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![
            SolidBoundaryCondition::ClampedRegion {
                min: [-0.03, -0.03, 0.048],
                max: [0.03, 0.03, 0.07],
            },
            SolidBoundaryCondition::PrescribedVelocityRegion {
                min: [0.045, -0.015, -0.005],
                max: [0.065, 0.015, 0.012],
                velocity: [-0.8, 0.0, 0.0],
            },
        ],
    });
    // Release striker velocity cleanly after 0.5 ms by using a kinematic striker sphere hitting rim instead:
    scene.solids[0].bcs.truncate(1);
    scene.solids.push(SolidBodySpec {
        name: "clapper".to_string(),
        body_id: 1,
        shape: GeometryShape::Sphere {
            center: [0.0595, 0.0, 0.006],
            radius: 0.007,
        },
        material: SolidMaterial::bell_bronze(),
        initial_velocity: [-1.5, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.004,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: true,
        bcs: vec![],
    });
    scene.kinematic_trajectories.push(KinematicTrajectorySpec {
        body_id: 1,
        initial_pivot: [0.0595, 0.0, 0.006],
        segments: vec![
            KinematicWaypoint {
                t_end_s: 0.0022,
                linear_vel: [-1.5, 0.0, 0.0],
                angular_vel: [0.0, 0.0, 0.0],
            },
            KinematicWaypoint {
                t_end_s: 0.006,
                linear_vel: [1.5, 0.0, 0.0],
                angular_vel: [0.0, 0.0, 0.0],
            },
        ],
    });
    scene.microphones.push(MicrophoneSpec {
        name: "mic".to_string(),
        position: [0.0, 0.075, 0.015],
        is_hydrophone: submerged,
    });

    let mut engine = MultiphysicsEngine::new(scene, None).expect("init bell test");
    let res = engine
        .run(&SimRunConfig {
            duration_s: 0.070,
            pre_roll_s: 0.0,
            sample_rate: 48_000,
            oversample_factor: 2,
            telemetry_interval_s: 0.01,
            use_cuda_graph: false,
            cfl_safety: 0.35,
            print_progress: false,
            ..Default::default()
        })
        .expect("run bell test");

    assert_eq!(res.telemetry.nan_count, 0);
    let sig = &res.raw_probe_buffers[0];
    let ring_start = (0.014 * 96_000.0) as usize;
    let early_end = (0.032 * 96_000.0) as usize;
    let late_start = (0.045 * 96_000.0) as usize;
    let late_end = (0.068_f64 * 96_000.0).min(sig.len() as f64) as usize;
    let rms_of = |a: usize, b: usize| -> f64 {
        let sl = &sig[a..b];
        (sl.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / (sl.len().max(1) as f64)).sqrt()
    };
    let early_rms = rms_of(ring_start, early_end);
    let late_rms = rms_of(late_start, late_end);
    let ring_slice: Vec<f64> = sig[ring_start..].iter().map(|&v| v as f64).collect();
    let (dom_freq, cent_freq) = dominant_freq_and_centroid(&ring_slice, 96_000.0, 150.0, 6000.0);
    println!(
        "Bell (submerged={}): ring_dom={:.1} Hz | ring_cent={:.1} Hz | peak={:.3} Pa | early_rms={:.3} | late_rms={:.3}",
        submerged, dom_freq, cent_freq, res.telemetry.probes[0].peak_pa, early_rms, late_rms
    );
    (dom_freq, res.telemetry.probes[0].peak_pa, early_rms, late_rms)
}

#[test]
fn test_added_mass_bell_pitch_drop() {
    let (f_air, peak_air, early_air, late_air) = run_bell_mode_freq(false);
    let (f_water, peak_water, early_water, late_water) = run_bell_mode_freq(true);
    let drop_ratio = (f_air - f_water) / f_air.max(1.0);
    println!(
        "Added-mass bell frequency drop: {f_air:.1} Hz (peak={peak_air:.2} Pa, rms {early_air:.2}->{late_air:.2}) -> {f_water:.1} Hz (peak={peak_water:.2} Pa, rms {early_water:.2}->{late_water:.2}) ({:.1}%)",
        drop_ratio * 100.0
    );
    assert!(
        peak_water.is_finite() && peak_water < 5.0e6,
        "Expected stable physical water pressure < 5e6 Pa, got {peak_water}"
    );
    assert!(
        late_water < early_water * 1.05,
        "Expected clean underwater bell decay (late_rms={late_water} >= early_rms*1.05={})",
        early_water * 1.05
    );
    assert!(
        f_water > 100.0 && f_air > 200.0 && drop_ratio >= 0.20,
        "Expected >= 20% downward added-mass frequency drop, got {f_air:.1} -> {f_water:.1} Hz"
    );
}

#[test]
fn test_hygro_leaf_wetting() {
    // Verify hygro-elastic leaf wetting via falling VOF rain droplets & softening
    let mut scene = Scene::minimal(
        "test_hygro_leaf",
        AmrDomainConfig {
            origin: [-0.048, -0.048, -0.032],
            dx0: 0.004,
            nx0: 24,
            ny0: 24,
            nz0: 24,
            pml_thickness: 4,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![],
        },
    );
    scene.enable_flow_advection = true;
    scene.mpm_grid = Some(MpmGridConfig {
        origin: [-0.048, -0.048, -0.032],
        dx: 0.004,
        nx: 24,
        ny: 24,
        nz: 24,
        has_floor: true,
        floor_z: -0.015,
        floor_mu: 0.4,
    });
    scene.solids.push(SolidBodySpec {
        name: "dry_leaf".to_string(),
        body_id: 0,
        shape: GeometryShape::CurvedShell {
            center: [0.0, 0.0, 0.0],
            size_x: 0.032,
            size_y: 0.020,
            nx: 12,
            ny: 8,
            dome_amp: 0.002,
            saddle_amp: 0.001,
            yaw_rad: 0.0,
            notch_x_len: 0.0,
            notch_y_pos: 0.0,
            notch_half_w: 0.0,
        },
        material: SolidMaterial::autumn_leaf_dry(0.0004),
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.003,
        gc_heterogeneity_amp: 0.15,
        is_kinematic_rigid: false,
        bcs: vec![SolidBoundaryCondition::ClampedRegion {
            min: [-0.025, -0.02, -0.01],
            max: [-0.012, 0.02, 0.01],
        }],
    });
    scene.droplet_emitters.push(DropletEmitterSpec {
        area_min_xy: [-0.006, -0.006],
        area_max_xy: [0.006, 0.006],
        spawn_z: 0.016,
        radius_min: 0.0045,
        radius_max: 0.0055,
        velocity_z: -2.0,
        drops_per_sec: 180.0,
        start_time_s: 0.0,
        end_time_s: 0.065,
        seed: 42,
    });
    scene.microphones.push(MicrophoneSpec {
        name: "mic".to_string(),
        position: [0.0, 0.0, 0.025],
        is_hydrophone: false,
    });

    let mut engine = MultiphysicsEngine::new(scene, None).expect("init hygro leaf test");
    let res = engine
        .run(&SimRunConfig {
            duration_s: 0.070,
            pre_roll_s: 0.0,
            sample_rate: 48_000,
            oversample_factor: 2,
            telemetry_interval_s: 0.01,
            use_cuda_graph: false,
            cfl_safety: 0.35,
            print_progress: false,
            ..Default::default()
        })
        .expect("run hygro leaf test");

    assert_eq!(res.telemetry.nan_count, 0);
    let wetness = engine.particles.as_ref().unwrap().wetness.to_vec().unwrap();
    let max_wet = wetness.iter().copied().fold(0.0_f32, f32::max);
    println!("Hygro leaf max_wetness after rain impact: {max_wet:.3}");
    assert!(
        max_wet > 0.05,
        "Expected VOF rain wetting to raise particle wetness above 0.05, got {max_wet}"
    );
}
