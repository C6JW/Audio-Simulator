pub mod diagnostics;
pub mod engine;
pub mod geometry;
pub mod material;
pub mod render_prep;
pub mod scene;
pub mod units;

pub use diagnostics::{
    analyze_probe_signal, analyze_subsurface_bubble, dominant_freq_and_centroid,
    power_spectrum_hann, BubbleDiagnostic,
};
pub use engine::{
    format_duration_hms, EngineError, MultiphysicsEngine, RenderExportConfig, Result, SimRunConfig,
    SimRunResult,
};
pub use geometry::{generate_shape_surface_mesh, generate_shell_uvs, normalize3, GeometryShape};
pub use material::{FluidParams, SolidMaterial};
pub use scene::{
    AmrDomainConfig, AmrPatchConfig, BlastRegionSpec, DirectionalAudioSourceSpec,
    DropletEmitterSpec, DynamicFluidRegion, FluidInflowSpec, GaussianPulseSpec,
    KinematicTrajectorySpec, KinematicWaypoint, MicrophoneSpec, MpmGridConfig,
    PointRickerSourceSpec, Scene, SlidingAmrPatchConfig, SolidBodySpec, SolidBoundaryCondition,
    SupersonicProjectileSpec, WindInletSpec, PML_ALL_FACES, PML_X_HI, PML_X_LO, PML_Y_HI,
    PML_Y_LO, PML_Z_HI, PML_Z_LO,
};
pub use units::*;
