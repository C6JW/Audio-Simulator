use serde::{Deserialize, Serialize};

use crate::geometry::GeometryShape;
use crate::material::{FluidParams, SolidMaterial};

pub const PML_X_LO: u32 = 1 << 0;
pub const PML_X_HI: u32 = 1 << 1;
pub const PML_Y_LO: u32 = 1 << 2;
pub const PML_Y_HI: u32 = 1 << 3;
pub const PML_Z_LO: u32 = 1 << 4;
pub const PML_Z_HI: u32 = 1 << 5;
pub const PML_ALL_FACES: u32 = 0b11_1111;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AmrPatchConfig {
    pub refine_ratio: usize,
    pub off_x: usize,
    pub off_y: usize,
    pub off_z: usize,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    #[serde(default)]
    pub parent_level: Option<usize>,
    #[serde(default)]
    pub enable_flow: Option<bool>,
    #[serde(default)]
    pub active_start_s: f64,
    #[serde(default)]
    pub active_end_s: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AmrDomainConfig {
    pub origin: [f32; 3],
    pub dx0: f32,
    pub nx0: usize,
    pub ny0: usize,
    pub nz0: usize,
    pub pml_thickness: usize,
    pub pml_face_mask: u32,
    pub patches: Vec<AmrPatchConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynamicFluidRegion {
    pub shape: GeometryShape,
    pub velocity: [f32; 3],
    pub is_water: bool,
    pub laplace_overpressure_pa: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GaussianPulseSpec {
    pub center: [f32; 3],
    pub sigma_r: f32,
    pub amplitude_pa: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PointRickerSourceSpec {
    pub center: [f32; 3],
    pub radius: f32,
    pub peak_rate_pa_per_s: f32,
    pub f0_hz: f64,
    pub t0_s: f64,
}

impl PointRickerSourceSpec {
    /// Standard zero-phase Ricker wavelet evaluated at time t.
    pub fn rate_at(&self, t: f64) -> f32 {
        let tau = t - self.t0_s;
        let a = std::f64::consts::PI * self.f0_hz * tau;
        let a2 = a * a;
        let ricker = (1.0 - 2.0 * a2) * (-a2).exp();
        (self.peak_rate_pa_per_s as f64 * ricker) as f32
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DirectionalAudioSourceSpec {
    pub center: [f32; 3],
    pub direction: [f32; 3],
    pub cardioid_beta: f32,
    pub radius: f32,
    pub peak_rate_pa_per_s: f32,
    pub t0_s: f64,
    pub sample_rate: f64,
    pub samples: Vec<f32>,
}

impl DirectionalAudioSourceSpec {
    /// Linearly interpolated acoustic injection rate at world time `t`.
    pub fn rate_at(&self, t: f64) -> f32 {
        if self.samples.len() < 2 || self.sample_rate <= 0.0 {
            return 0.0;
        }
        let tau = t - self.t0_s;
        if tau < 0.0 {
            return 0.0;
        }
        let f_idx = tau * self.sample_rate;
        let max_idx = (self.samples.len() - 1) as f64;
        if f_idx >= max_idx {
            return 0.0;
        }
        let i0 = f_idx.floor() as usize;
        let frac = (f_idx - i0 as f64) as f32;
        let v = self.samples[i0] * (1.0 - frac) + self.samples[i0 + 1] * frac;
        self.peak_rate_pa_per_s * v
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KinematicWaypoint {
    pub t_end_s: f64,
    pub linear_vel: [f32; 3],
    pub angular_vel: [f32; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KinematicTrajectorySpec {
    pub body_id: i32,
    pub initial_pivot: [f32; 3],
    pub segments: Vec<KinematicWaypoint>,
}

impl KinematicTrajectorySpec {
    /// Piecewise-constant `(linear_vel, angular_vel)` at time `t`.
    pub fn velocity_at(&self, t: f64) -> ([f32; 3], [f32; 3]) {
        for seg in &self.segments {
            if t <= seg.t_end_s {
                return (seg.linear_vel, seg.angular_vel);
            }
        }
        ([0.0; 3], [0.0; 3])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SolidBoundaryCondition {
    ClampedRegion {
        min: [f32; 3],
        max: [f32; 3],
    },
    PrescribedVelocityRegion {
        min: [f32; 3],
        max: [f32; 3],
        velocity: [f32; 3],
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolidBodySpec {
    pub name: String,
    pub body_id: i32,
    pub shape: GeometryShape,
    pub material: SolidMaterial,
    pub initial_velocity: [f32; 3],
    pub orientation_euler_xyz: [f32; 3],
    pub particle_spacing: f32,
    pub gc_heterogeneity_amp: f32,
    pub bcs: Vec<SolidBoundaryCondition>,
    #[serde(default)]
    pub is_kinematic_rigid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MpmGridConfig {
    pub origin: [f32; 3],
    pub dx: f32,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub has_floor: bool,
    pub floor_z: f32,
    pub floor_mu: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MicrophoneSpec {
    pub name: String,
    pub position: [f32; 3],
    pub is_hydrophone: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlastRegionSpec {
    pub trigger_time_s: f64,
    pub center: [f32; 3],
    pub radius: f32,
    pub length: f32,
    pub is_cylinder_x: bool,
    pub overpressure_pa: f32,
    pub density_kg_m3: f32,
    pub velocity: [f32; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SupersonicProjectileSpec {
    pub trigger_time_s: f64,
    pub initial_pos: [f32; 3],
    pub initial_vel: [f32; 3],
    pub mass_kg: f32,
    pub radius_m: f32,
    pub length_m: f32,
    pub impact_x_max: f32,
    pub impact_target_body_id: Option<i32>,
    pub impact_radius_m: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlidingAmrPatchConfig {
    pub refine_ratio: usize,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub initial_ox: f32,
    pub oy: f32,
    pub oz: f32,
    pub lead_fraction: f32,
    pub handoff_x_end: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FluidInflowSpec {
    pub center: [f32; 3],
    pub radius: f32,
    pub height: f32,
    pub velocity: [f32; 3],
    pub start_time_s: f64,
    pub end_time_s: f64,
    pub ramp_s: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DropletEmitterSpec {
    pub area_min_xy: [f32; 2],
    pub area_max_xy: [f32; 2],
    pub spawn_z: f32,
    pub radius_min: f32,
    pub radius_max: f32,
    pub velocity_z: f32,
    pub drops_per_sec: f64,
    pub start_time_s: f64,
    pub end_time_s: f64,
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindInletSpec {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub velocity: [f32; 3],
    pub turb_amp: f32,
    pub start_time_s: f64,
    pub end_time_s: f64,
    pub ramp_s: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scene {
    pub name: String,
    pub fluid: FluidParams,
    pub amr: AmrDomainConfig,
    pub background_water: Vec<GeometryShape>,
    pub dynamic_fluid: Vec<DynamicFluidRegion>,
    pub gaussian_pulses: Vec<GaussianPulseSpec>,
    pub point_sources: Vec<PointRickerSourceSpec>,
    #[serde(default)]
    pub directional_sources: Vec<DirectionalAudioSourceSpec>,
    pub solids: Vec<SolidBodySpec>,
    #[serde(default)]
    pub kinematic_trajectories: Vec<KinematicTrajectorySpec>,
    pub mpm_grid: Option<MpmGridConfig>,
    pub microphones: Vec<MicrophoneSpec>,
    pub p_top_ref: f64,
    pub enable_flow_advection: bool,
    #[serde(default)]
    pub enable_compressible_euler: bool,
    #[serde(default)]
    pub blast_regions: Vec<BlastRegionSpec>,
    #[serde(default)]
    pub projectiles: Vec<SupersonicProjectileSpec>,
    #[serde(default)]
    pub sliding_patch: Option<SlidingAmrPatchConfig>,
    #[serde(default)]
    pub fluid_inflows: Vec<FluidInflowSpec>,
    #[serde(default)]
    pub droplet_emitters: Vec<DropletEmitterSpec>,
    #[serde(default)]
    pub wind_inlets: Vec<WindInletSpec>,
}

impl Scene {
    pub fn minimal(name: impl Into<String>, amr: AmrDomainConfig) -> Self {
        Self {
            name: name.into(),
            fluid: FluidParams::default(),
            amr,
            background_water: Vec::new(),
            dynamic_fluid: Vec::new(),
            gaussian_pulses: Vec::new(),
            point_sources: Vec::new(),
            directional_sources: Vec::new(),
            solids: Vec::new(),
            kinematic_trajectories: Vec::new(),
            mpm_grid: None,
            microphones: Vec::new(),
            p_top_ref: 0.0,
            enable_flow_advection: false,
            enable_compressible_euler: false,
            blast_regions: Vec::new(),
            projectiles: Vec::new(),
            sliding_patch: None,
            fluid_inflows: Vec::new(),
            droplet_emitters: Vec::new(),
            wind_inlets: Vec::new(),
        }
    }
}
