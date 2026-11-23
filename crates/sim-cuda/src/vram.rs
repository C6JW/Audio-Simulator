use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::{Arc, Mutex};

use crate::error::{check_cuda, CudaError, Result};
use crate::ffi::{
    cuMemGetInfo_v2, sim_cuda_free, sim_cuda_init_device, sim_cuda_malloc, sim_cuda_memcpy_d2h,
    sim_cuda_memcpy_h2d, sim_cuda_memset,
};

pub const DEFAULT_VRAM_CAP_BYTES: usize = 15_000_000_000;

#[derive(Debug)]
struct VramState {
    tracked_bytes: usize,
    peak_bytes: usize,
    cap_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub device_id: i32,
    pub cc_major: i32,
    pub cc_minor: i32,
    pub total_vram_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct VramBudgetManager {
    state: Arc<Mutex<VramState>>,
    info: DeviceInfo,
}

impl VramBudgetManager {
    pub fn new(device_id: i32, cap_bytes: usize) -> Result<Self> {
        let mut cc_major = 0;
        let mut cc_minor = 0;
        let mut total_vram_bytes = 0;
        unsafe {
            check_cuda(
                sim_cuda_init_device(device_id, &mut cc_major, &mut cc_minor, &mut total_vram_bytes),
                "sim_cuda_init_device",
            )?;
        }
        Ok(Self {
            state: Arc::new(Mutex::new(VramState {
                tracked_bytes: 0,
                peak_bytes: 0,
                cap_bytes,
            })),
            info: DeviceInfo {
                device_id,
                cc_major,
                cc_minor,
                total_vram_bytes,
            },
        })
    }

    pub fn device_info(&self) -> &DeviceInfo {
        &self.info
    }

    pub fn query_driver_mem_info(&self) -> Result<(usize, usize)> {
        let mut free = 0usize;
        let mut total = 0usize;
        unsafe {
            check_cuda(cuMemGetInfo_v2(&mut free, &mut total), "cuMemGetInfo_v2")?;
        }
        Ok((free, total))
    }

    pub fn tracked_bytes(&self) -> usize {
        self.state.lock().unwrap().tracked_bytes
    }

    pub fn peak_bytes(&self) -> usize {
        self.state.lock().unwrap().peak_bytes
    }

    pub fn cap_bytes(&self) -> usize {
        self.state.lock().unwrap().cap_bytes
    }

    pub fn alloc_uninit<T: Copy>(&self, len: usize) -> Result<DeviceBuffer<T>> {
        if len == 0 {
            return Ok(DeviceBuffer {
                ptr: std::ptr::null_mut(),
                len: 0,
                bytes: 0,
                manager: self.clone(),
                _marker: PhantomData,
            });
        }
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(CudaError::VramBudgetExceeded {
                requested: usize::MAX,
                tracked: self.tracked_bytes(),
                cap: self.cap_bytes(),
                driver_free: 0,
            })?;

        let (driver_free, _) = self.query_driver_mem_info()?;
        {
            let mut st = self.state.lock().unwrap();
            let new_tracked = st.tracked_bytes.saturating_add(bytes);
            if new_tracked > st.cap_bytes || bytes > driver_free {
                return Err(CudaError::VramBudgetExceeded {
                    requested: bytes,
                    tracked: st.tracked_bytes,
                    cap: st.cap_bytes,
                    driver_free,
                });
            }
            st.tracked_bytes = new_tracked;
            if new_tracked > st.peak_bytes {
                st.peak_bytes = new_tracked;
            }
        }

        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { sim_cuda_malloc(&mut ptr, bytes) };
        if rc != 0 {
            let mut st = self.state.lock().unwrap();
            st.tracked_bytes = st.tracked_bytes.saturating_sub(bytes);
            return Err(CudaError::CallFailed {
                func: "sim_cuda_malloc",
                code: rc,
            });
        }

        Ok(DeviceBuffer {
            ptr: ptr as *mut T,
            len,
            bytes,
            manager: self.clone(),
            _marker: PhantomData,
        })
    }

    pub fn alloc_zeros<T: Copy>(&self, len: usize) -> Result<DeviceBuffer<T>> {
        let buf = self.alloc_uninit::<T>(len)?;
        if buf.bytes > 0 {
            unsafe {
                check_cuda(
                    sim_cuda_memset(buf.ptr as *mut c_void, 0, buf.bytes),
                    "sim_cuda_memset",
                )?;
            }
        }
        Ok(buf)
    }

    pub fn alloc_from_slice<T: Copy>(&self, data: &[T]) -> Result<DeviceBuffer<T>> {
        let mut buf = self.alloc_uninit::<T>(data.len())?;
        buf.upload(data)?;
        Ok(buf)
    }
}

pub struct DeviceBuffer<T: Copy> {
    ptr: *mut T,
    len: usize,
    bytes: usize,
    manager: VramBudgetManager,
    _marker: PhantomData<T>,
}

unsafe impl<T: Copy + Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Copy + Sync> Sync for DeviceBuffer<T> {}

impl<T: Copy> DeviceBuffer<T> {
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn zero(&mut self) -> Result<()> {
        if self.bytes > 0 {
            unsafe {
                check_cuda(
                    sim_cuda_memset(self.ptr as *mut c_void, 0, self.bytes),
                    "sim_cuda_memset",
                )?;
            }
        }
        Ok(())
    }

    pub fn upload(&mut self, src: &[T]) -> Result<()> {
        if src.len() != self.len {
            return Err(CudaError::LengthMismatch {
                expected: self.len,
                actual: src.len(),
            });
        }
        if self.bytes > 0 {
            unsafe {
                check_cuda(
                    sim_cuda_memcpy_h2d(
                        self.ptr as *mut c_void,
                        src.as_ptr() as *const c_void,
                        self.bytes,
                    ),
                    "sim_cuda_memcpy_h2d",
                )?;
            }
        }
        Ok(())
    }

    pub fn download(&self, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.len {
            return Err(CudaError::LengthMismatch {
                expected: self.len,
                actual: dst.len(),
            });
        }
        if self.bytes > 0 {
            unsafe {
                check_cuda(
                    sim_cuda_memcpy_d2h(
                        dst.as_mut_ptr() as *mut c_void,
                        self.ptr as *const c_void,
                        self.bytes,
                    ),
                    "sim_cuda_memcpy_d2h",
                )?;
            }
        }
        Ok(())
    }

    pub fn to_vec(&self) -> Result<Vec<T>> {
        let mut out = Vec::with_capacity(self.len);
        if self.len > 0 {
            // Safe because T: Copy and download initializes every byte before exposing slice.
            unsafe {
                out.set_len(self.len);
            }
            self.download(&mut out)?;
        }
        Ok(out)
    }
}

impl<T: Copy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = sim_cuda_free(self.ptr as *mut c_void);
            }
            let mut st = self.manager.state.lock().unwrap();
            st.tracked_bytes = st.tracked_bytes.saturating_sub(self.bytes);
            self.ptr = std::ptr::null_mut();
        }
    }
}
