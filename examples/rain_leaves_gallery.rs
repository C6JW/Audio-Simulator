use std::collections::BTreeMap;
use std::path::Path;

use sim_core::*;
use sim_cuda::DEFAULT_VRAM_CAP_BYTES;
use sim_io::{write_telemetry_json, write_timeseries_csv};

fn main() -> Result<()> {
    let out_dir = Path::new("output");
    std::fs::create_dir_all(out_dir)?;

    // Full-scale 34 m St. Paul's Cathedral Whispering Gallery (R_inner = 17.0 m)
    // Level 0: 384 x 384 x 40, dx0 = 0.096 m (36.864 m x 36.864 m x 3.840 m), pure air acoustics
    // Level 1: 48 x 48 x 36, r=4, dx1 = 0.024 m, pure air coupling zone around x = -16.6 m
    // Level 2: 64 x 64 x 48, r=4, dx2 = 0.006 m, two-phase VOF rain + 40 hygro-elastic leaves
    let mut scene = Scene::minimal(
        "rain_leaves_gallery",
        AmrDomainConfig {
            origin: [-18.432, -18.432, -0.192],
            dx0: 0.096,
            nx0: 384,
            ny0: 384,
            nz0: 40,
            pml_thickness: 6,
            pml_face_mask: PML_ALL_FACES,
            patches: vec![
                AmrPatchConfig {
                    parent_level: Some(0),
                    refine_ratio: 4,
                    off_x: 13,
                    off_y: 186,
                    off_z: 2,
                    nx: 48,
                    ny: 48,
                    nz: 36,
                    enable_flow: Some(false),
                    active_start_s: 0.0,
                    active_end_s: None,
                },
                AmrPatchConfig {
                    parent_level: Some(1),
                    refine_ratio: 4,
                    off_x: 18,
                    off_y: 16,
                    off_z: 0,
                    nx: 64,
                    ny: 64,
                    nz: 48,
                    enable_flow: Some(true),
                    active_start_s: 0.0,
                    active_end_s: None,
                },
            ],
        },
    );

    scene.fluid.wall_absorption_alpha = 0.02;
    scene.enable_flow_advection = false;

    scene.mpm_grid = Some(MpmGridConfig {
        origin: [-16.752, -0.192, 0.000],
        dx: 0.006,
        nx: 64,
        ny: 64,
        nz: 48,
        has_floor: true,
        floor_z: 0.006,
        floor_mu: 0.50,
    });

    // Rigid circular stone masonry gallery wall (34.0 m internal diameter) & stone floor
    let mut stone = SolidMaterial::granite_stone();
    stone.is_rigid_static = true;
    scene.solids.push(SolidBodySpec {
        name: "cathedral_gallery_wall".to_string(),
        body_id: 0,
        shape: GeometryShape::CylindricalRing {
            center_xy: [0.0, 0.0],
            r_inner: 17.00,
            r_outer: 17.80,
            z_min: -0.15,
            z_max: 3.60,
        },
        material: stone.clone(),
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.096,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![],
    });
    scene.solids.push(SolidBodySpec {
        name: "cathedral_stone_floor".to_string(),
        body_id: 0,
        shape: GeometryShape::Disc {
            center_xy: [0.0, 0.0],
            radius: 17.80,
            z_min: -0.15,
            z_max: 0.004,
        },
        material: stone,
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.096,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: false,
        bcs: vec![],
    });

    // 40 curled dry autumn leaves stacked near the western gallery wall (x ≈ -16.56 m, y ≈ 0.0 m)
    let base_x = -16.560_f32;
    let base_y = 0.000_f32;
    for idx in 0..40usize {
        let layer = idx / 8; // 5 vertical layers of 8 leaves
        let slot = idx % 8;
        let angle = (slot as f32) * std::f32::consts::PI * 0.25 + (layer as f32) * 0.37;
        let rad = if slot % 2 == 0 { 0.032_f32 } else { 0.056_f32 };
        let cx = base_x + rad * angle.cos();
        let cy = base_y + rad * angle.sin();
        let cz = 0.011 + (layer as f32) * 0.0072 + ((slot % 3) as f32) * 0.0015;
        let dome = 0.0038 + ((idx % 5) as f32) * 0.0006;
        let saddle = if idx % 2 == 0 { 0.0020 } else { -0.0022 };
        let yaw = ((idx as f32) * 0.618 * std::f32::consts::TAU) % std::f32::consts::TAU;

        scene.solids.push(SolidBodySpec {
            name: format!("autumn_leaf_{idx}"),
            body_id: (idx + 1) as i32,
            shape: GeometryShape::CurvedShell {
                center: [cx, cy, cz],
                size_x: 0.062,
                size_y: 0.044,
                nx: 6,
                ny: 4,
                dome_amp: dome,
                saddle_amp: saddle,
                yaw_rad: yaw,
                notch_x_len: 0.0,
                notch_y_pos: 0.0,
                notch_half_w: 0.0,
            },
            material: SolidMaterial::autumn_leaf_dry(0.00035),
            initial_velocity: [0.0, 0.0, 0.0],
            orientation_euler_xyz: [0.0, 0.0, 0.0],
            particle_spacing: 0.009,
            gc_heterogeneity_amp: 0.25,
            is_kinematic_rigid: false,
            bcs: vec![],
        });
    }

    // Heavy VOF rainstorm wetting and softening the leaf pile from t = 0.75 s to 4.50 s
    scene.droplet_emitters.push(DropletEmitterSpec {
        area_min_xy: [-16.65, -0.09],
        area_max_xy: [-16.47, 0.09],
        spawn_z: 0.235,
        radius_min: 0.0065,
        radius_max: 0.0085,
        velocity_z: -3.1,
        drops_per_sec: 140.0,
        start_time_s: 0.75,
        end_time_s: 4.50,
        seed: 1337,
    });

    // Kinematic rigid boot sole: held to the side (y = -0.14 m) during rainstorm so raindrops hit
    // the leaves unobstructed, then steps over the pile (4.6..5.0 s) and crushes it (5.0..7.2 s).
    scene.solids.push(SolidBodySpec {
        name: "heavy_boot_sole".to_string(),
        body_id: 41,
        shape: GeometryShape::Box {
            min: [-16.63, -0.18, 0.068],
            max: [-16.49, -0.10, 0.094],
        },
        material: SolidMaterial::structural_steel(),
        initial_velocity: [0.0, 0.0, 0.0],
        orientation_euler_xyz: [0.0, 0.0, 0.0],
        particle_spacing: 0.012,
        gc_heterogeneity_amp: 0.0,
        is_kinematic_rigid: true,
        bcs: vec![],
    });
    scene.kinematic_trajectories.push(KinematicTrajectorySpec {
        body_id: 41,
        initial_pivot: [-16.56, -0.14, 0.081],
        segments: vec![
            KinematicWaypoint {
                t_end_s: 4.60,
                linear_vel: [0.0, 0.0, 0.0],
                angular_vel: [0.0, 0.0, 0.0],
            },
            KinematicWaypoint {
                t_end_s: 5.00,
                linear_vel: [0.0, 0.35, 0.0],
                angular_vel: [0.0, 0.0, 0.0],
            },
            KinematicWaypoint {
                t_end_s: 6.40,
                linear_vel: [0.0, 0.0, -0.041],
                angular_vel: [0.0, 0.0, 0.08],
            },
            KinematicWaypoint {
                t_end_s: 7.20,
                linear_vel: [0.0, 0.0, -0.005],
                angular_vel: [0.0, 0.0, -0.18],
            },
            KinematicWaypoint {
                t_end_s: 8.00,
                linear_vel: [0.0, 0.0, 0.0],
                angular_vel: [0.0, 0.0, 0.0],
            },
        ],
    });

    scene.microphones.push(MicrophoneSpec {
        name: "rain_leaves_gallery_near.wav".to_string(),
        position: [-16.41, 0.0, 0.21],
        is_hydrophone: false,
    });
    scene.microphones.push(MicrophoneSpec {
        name: "rain_leaves_gallery_sweet_spot.wav".to_string(),
        position: [16.82, 0.0, 1.40],
        is_hydrophone: false,
    });

    let cfg = SimRunConfig {
        duration_s: 8.0,
        pre_roll_s: 0.75,
        sample_rate: 24_000,
        oversample_factor: 1,
        target_wav_sample_rate: Some(48_000),
        phase_field_stride_steps: 8,
        telemetry_interval_s: 0.05,
        output_dir: Some(out_dir.to_path_buf()),
        cfl_safety: 0.35,
        print_progress: true,
        ..Default::default()
    };

    println!("=== Scene 4: Running Rain-Soaked Autumn Leaves in 34m St. Paul's Whispering Gallery ===");
    let mut engine = MultiphysicsEngine::new(scene, None)?;
    let mut res = engine.run(&cfg)?;

    let wetness = engine.particles.as_ref().unwrap().wetness.to_vec()?;
    let max_wet = wetness.iter().copied().fold(0.0_f32, f32::max) as f64;
    let avg_wet = wetness.iter().map(|&w| w as f64).sum::<f64>() / (wetness.len().max(1) as f64);

    let sr = 24_000.0_f64;
    let slice_win = |buf: &[f32], t0: f64, t1: f64| -> Vec<f64> {
        let i0 = ((t0 * sr) as usize).min(buf.len());
        let i1 = ((t1 * sr) as usize).min(buf.len());
        buf[i0..i1].iter().map(|&v| v as f64).collect()
    };

    let dry_rain = slice_win(&res.raw_probe_buffers[0], 0.85, 1.85);
    let wet_rain = slice_win(&res.raw_probe_buffers[0], 3.40, 4.45);
    let boot_crush = slice_win(&res.raw_probe_buffers[0], 5.05, 7.15);

    let (_, cent_dry) = dominant_freq_and_centroid(&dry_rain, sr, 120.0, 10000.0);
    let (_, cent_wet) = dominant_freq_and_centroid(&wet_rain, sr, 120.0, 10000.0);
    let (_, cent_crush) = dominant_freq_and_centroid(&boot_crush, sr, 120.0, 10000.0);

    let mut notes = BTreeMap::new();
    notes.insert("max_leaf_wetness".to_string(), max_wet);
    notes.insert("avg_leaf_wetness".to_string(), avg_wet);
    notes.insert("dry_rain_centroid_hz".to_string(), cent_dry);
    notes.insert("wet_rain_centroid_hz".to_string(), cent_wet);
    notes.insert("boot_crush_centroid_hz".to_string(), cent_crush);
    res.telemetry.notes = notes;

    write_telemetry_json(out_dir.join("rain_leaves_gallery_telemetry.json"), &res.telemetry)?;
    write_timeseries_csv(out_dir.join("rain_leaves_gallery_timeseries.csv"), &res.timeseries)?;

    println!(
        "Rain Leaves Gallery Results: max_wet={:.3} (avg={:.3}) | centroid dry={:.1} Hz -> wet={:.1} Hz -> crush={:.1} Hz | VRAM={:.1} MB | NaNs={}",
        max_wet,
        avg_wet,
        cent_dry,
        cent_wet,
        cent_crush,
        res.telemetry.peak_vram_tracked_bytes as f64 / 1e6,
        res.telemetry.nan_count
    );

    assert_eq!(res.telemetry.nan_count, 0);
    assert!(res.telemetry.peak_vram_tracked_bytes < DEFAULT_VRAM_CAP_BYTES);
    Ok(())
}
