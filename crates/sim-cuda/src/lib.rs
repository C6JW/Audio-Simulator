pub mod error;
pub mod ffi;
pub mod fluid;
pub mod solid;
pub mod vram;

pub use error::{check_cuda, CudaError, Result};
pub use ffi::{
    FluidConstantsGpu, FluidLevelGpu, MaterialParamsGpu, MicProbeGpu, MpmGridGpu,
    MpmMetricsGpu, MpmParticlesGpu, ShellMeshGpu, SimReduceMetricsGpu,
};
pub use fluid::{device_sync, CudaGraphExec, CudaStream, FluidLevelDevice, FluidLevelSpec};
pub use solid::{
    MpmGridDevice, MpmGridSpec, MpmParticlesDevice, MpmParticlesHost, ShellMeshDevice,
    ShellMeshHost,
};
pub use vram::{DeviceBuffer, DeviceInfo, VramBudgetManager, DEFAULT_VRAM_CAP_BYTES};
