use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA runtime/driver call `{func}` failed with code {code}")]
    CallFailed { func: &'static str, code: i32 },

    #[error(
        "VRAM budget exceeded: requested {requested} bytes, tracked {tracked} bytes, cap {cap} bytes (driver free: {driver_free} bytes)"
    )]
    VramBudgetExceeded {
        requested: usize,
        tracked: usize,
        cap: usize,
        driver_free: usize,
    },

    #[error("Invalid buffer length: expected {expected} elements, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    #[error("Compute capability {major}.{minor} is incompatible or unexpected: {msg}")]
    DeviceError {
        major: i32,
        minor: i32,
        msg: String,
    },
}

pub type Result<T> = std::result::Result<T, CudaError>;

#[inline]
pub fn check_cuda(code: i32, func: &'static str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaError::CallFailed { func, code })
    }
}
