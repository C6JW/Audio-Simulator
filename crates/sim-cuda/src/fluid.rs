use std::ffi::c_void;

use crate::error::{check_cuda, Result};
use crate::ffi::*;
use crate::vram::{DeviceBuffer, VramBudgetManager};

pub struct CudaStream {
    raw: *mut c_void,
}

unsafe impl Send for CudaStream {}

impl CudaStream {
    pub fn new() -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        unsafe {
            check_cuda(sim_cuda_stream_create(&mut raw), "sim_cuda_stream_create")?;
        }
        Ok(Self { raw })
    }

    pub fn raw(&self) -> *mut c_void {
        self.raw
    }

    pub fn sync(&self) -> Result<()> {
        unsafe { check_cuda(sim_cuda_stream_sync(self.raw), "sim_cuda_stream_sync") }
    }

    pub fn begin_capture(&self) -> Result<()> {
        unsafe {
            check_cuda(
                sim_cuda_graph_begin_capture(self.raw),
                "sim_cuda_graph_begin_capture",
            )
        }
    }

    pub fn end_capture(&self) -> Result<CudaGraphExec> {
        let mut exec = std::ptr::null_mut();
        unsafe {
            check_cuda(
                sim_cuda_graph_end_capture(self.raw, &mut exec),
                "sim_cuda_graph_end_capture",
            )?;
        }
        Ok(CudaGraphExec { raw: exec })
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = sim_cuda_stream_destroy(self.raw);
            }
            self.raw = std::ptr::null_mut();
        }
    }
}

pub struct CudaGraphExec {
    raw: *mut c_void,
}

unsafe impl Send for CudaGraphExec {}

impl CudaGraphExec {
    pub fn launch(&self, stream: &CudaStream) -> Result<()> {
        unsafe {
            check_cuda(
                sim_cuda_graph_launch(self.raw, stream.raw()),
                "sim_cuda_graph_launch",
            )
        }
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = sim_cuda_graph_destroy(self.raw);
            }
            self.raw = std::ptr::null_mut();
        }
    }
}

#[derive(Clone, Debug)]
pub struct FluidLevelSpec {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f32,
    pub ox: f32,
    pub oy: f32,
    pub oz: f32,
    pub level_idx: usize,
    pub refine_ratio: usize,
    pub pml_thickness: usize,
    pub pml_face_mask: u32,
    pub off_x: usize,
    pub off_y: usize,
    pub off_z: usize,
    pub alloc_flow: bool,
    pub alloc_amr_old: bool,
    pub alloc_ib_vel: bool,
    pub alloc_compressible: bool,
}

pub struct FluidLevelDevice {
    pub spec: FluidLevelSpec,
    pub halo: usize,
    pub nxp: usize,
    pub nyp: usize,
    pub nzp: usize,

    pub p_gauge: Option<DeviceBuffer<f32>>,
    pub p_gauge_old: Option<DeviceBuffer<f32>>,
    pub p_hydro: Option<DeviceBuffer<f32>>,
    pub alpha: Option<DeviceBuffer<f32>>,
    pub alpha_tmp: Option<DeviceBuffer<f32>>,
    pub alpha_smooth: Option<DeviceBuffer<f32>>,
    pub kappa: Option<DeviceBuffer<f32>>,
    pub div_u: Option<DeviceBuffer<f32>>,
    pub src_p: Option<DeviceBuffer<f32>>,

    pub ux: Option<DeviceBuffer<f32>>,
    pub ux_old: Option<DeviceBuffer<f32>>,
    pub a_slow_x: Option<DeviceBuffer<f32>>,
    pub ib_beta_x: Option<DeviceBuffer<f32>>,
    pub ib_beta_static_x: Option<DeviceBuffer<f32>>,
    pub ib_vel_x: Option<DeviceBuffer<f32>>,
    pub mflux_x: Option<DeviceBuffer<f32>>,
    pub vflux_x: Option<DeviceBuffer<f32>>,

    pub uy: Option<DeviceBuffer<f32>>,
    pub uy_old: Option<DeviceBuffer<f32>>,
    pub a_slow_y: Option<DeviceBuffer<f32>>,
    pub ib_beta_y: Option<DeviceBuffer<f32>>,
    pub ib_beta_static_y: Option<DeviceBuffer<f32>>,
    pub ib_vel_y: Option<DeviceBuffer<f32>>,
    pub mflux_y: Option<DeviceBuffer<f32>>,
    pub vflux_y: Option<DeviceBuffer<f32>>,

    pub uz: Option<DeviceBuffer<f32>>,
    pub uz_old: Option<DeviceBuffer<f32>>,
    pub a_slow_z: Option<DeviceBuffer<f32>>,
    pub ib_beta_z: Option<DeviceBuffer<f32>>,
    pub ib_beta_static_z: Option<DeviceBuffer<f32>>,
    pub ib_vel_z: Option<DeviceBuffer<f32>>,
    pub mflux_z: Option<DeviceBuffer<f32>>,
    pub vflux_z: Option<DeviceBuffer<f32>>,

    pub psi_ux: Option<DeviceBuffer<f32>>,
    pub psi_uy: Option<DeviceBuffer<f32>>,
    pub psi_uz: Option<DeviceBuffer<f32>>,
    pub psi_px: Option<DeviceBuffer<f32>>,
    pub psi_py: Option<DeviceBuffer<f32>>,
    pub psi_pz: Option<DeviceBuffer<f32>>,

    pub bnd_vflux_xlo: Option<DeviceBuffer<f32>>,
    pub bnd_vflux_xhi: Option<DeviceBuffer<f32>>,
    pub bnd_vflux_ylo: Option<DeviceBuffer<f32>>,
    pub bnd_vflux_yhi: Option<DeviceBuffer<f32>>,
    pub bnd_vflux_zlo: Option<DeviceBuffer<f32>>,
    pub bnd_vflux_zhi: Option<DeviceBuffer<f32>>,

    pub bnd_aflux_xlo: Option<DeviceBuffer<f32>>,
    pub bnd_aflux_xhi: Option<DeviceBuffer<f32>>,
    pub bnd_aflux_ylo: Option<DeviceBuffer<f32>>,
    pub bnd_aflux_yhi: Option<DeviceBuffer<f32>>,
    pub bnd_aflux_zlo: Option<DeviceBuffer<f32>>,
    pub bnd_aflux_zhi: Option<DeviceBuffer<f32>>,

    pub u_rho: Option<DeviceBuffer<f32>>,
    pub u_mx: Option<DeviceBuffer<f32>>,
    pub u_my: Option<DeviceBuffer<f32>>,
    pub u_mz: Option<DeviceBuffer<f32>>,
    pub u_ene: Option<DeviceBuffer<f32>>,
    pub u1_rho: Option<DeviceBuffer<f32>>,
    pub u1_mx: Option<DeviceBuffer<f32>>,
    pub u1_my: Option<DeviceBuffer<f32>>,
    pub u1_mz: Option<DeviceBuffer<f32>>,
    pub u1_ene: Option<DeviceBuffer<f32>>,
    pub cell_ib: Option<DeviceBuffer<f32>>,
    pub cell_ib_static: Option<DeviceBuffer<f32>>,
    pub cell_vx: Option<DeviceBuffer<f32>>,
    pub cell_vy: Option<DeviceBuffer<f32>>,
    pub cell_vz: Option<DeviceBuffer<f32>>,
}

impl FluidLevelDevice {
    pub fn new(vram: &VramBudgetManager, spec: FluidLevelSpec) -> Result<Self> {
        let halo = 2usize;
        let nxp = spec.nx + 2 * halo;
        let nyp = spec.ny + 2 * halo;
        let nzp = spec.nz + 2 * halo;
        let n_cells = nxp * nyp * nzp;
        let n_fx = (nxp + 1) * nyp * nzp;
        let n_fy = nxp * (nyp + 1) * nzp;
        let n_fz = nxp * nyp * (nzp + 1);

        if spec.alloc_compressible {
            let alloc_cell_vel = spec.alloc_ib_vel;
            return Ok(Self {
                p_gauge: None,
                p_gauge_old: None,
                p_hydro: None,
                alpha: None,
                alpha_tmp: None,
                alpha_smooth: None,
                kappa: None,
                div_u: None,
                src_p: None,

                ux: None,
                ux_old: None,
                a_slow_x: None,
                ib_beta_x: None,
                ib_beta_static_x: None,
                ib_vel_x: None,
                mflux_x: None,
                vflux_x: None,

                uy: None,
                uy_old: None,
                a_slow_y: None,
                ib_beta_y: None,
                ib_beta_static_y: None,
                ib_vel_y: None,
                mflux_y: None,
                vflux_y: None,

                uz: None,
                uz_old: None,
                a_slow_z: None,
                ib_beta_z: None,
                ib_beta_static_z: None,
                ib_vel_z: None,
                mflux_z: None,
                vflux_z: None,

                psi_ux: None,
                psi_uy: None,
                psi_uz: None,
                psi_px: None,
                psi_py: None,
                psi_pz: None,

                bnd_vflux_xlo: None,
                bnd_vflux_xhi: None,
                bnd_vflux_ylo: None,
                bnd_vflux_yhi: None,
                bnd_vflux_zlo: None,
                bnd_vflux_zhi: None,

                bnd_aflux_xlo: None,
                bnd_aflux_xhi: None,
                bnd_aflux_ylo: None,
                bnd_aflux_yhi: None,
                bnd_aflux_zlo: None,
                bnd_aflux_zhi: None,

                u_rho: Some(vram.alloc_zeros(n_cells)?),
                u_mx: Some(vram.alloc_zeros(n_cells)?),
                u_my: Some(vram.alloc_zeros(n_cells)?),
                u_mz: Some(vram.alloc_zeros(n_cells)?),
                u_ene: Some(vram.alloc_zeros(n_cells)?),
                u1_rho: Some(vram.alloc_zeros(n_cells)?),
                u1_mx: Some(vram.alloc_zeros(n_cells)?),
                u1_my: Some(vram.alloc_zeros(n_cells)?),
                u1_mz: Some(vram.alloc_zeros(n_cells)?),
                u1_ene: Some(vram.alloc_zeros(n_cells)?),
                cell_ib: Some(vram.alloc_zeros(n_cells)?),
                cell_ib_static: Some(vram.alloc_zeros(n_cells)?),
                cell_vx: if alloc_cell_vel { Some(vram.alloc_zeros(n_cells)?) } else { None },
                cell_vy: if alloc_cell_vel { Some(vram.alloc_zeros(n_cells)?) } else { None },
                cell_vz: if alloc_cell_vel { Some(vram.alloc_zeros(n_cells)?) } else { None },

                spec,
                halo,
                nxp,
                nyp,
                nzp,
            });
        }

        let has_pml = spec.pml_thickness > 0 && spec.level_idx == 0;
        let has_amr = spec.level_idx > 0;
        let alloc_flow = spec.alloc_flow;
        let alloc_amr_old = spec.alloc_amr_old;
        let alloc_ib_vel = spec.alloc_ib_vel;

        Ok(Self {
            p_gauge: Some(vram.alloc_zeros(n_cells)?),
            p_gauge_old: if alloc_amr_old { Some(vram.alloc_zeros(n_cells)?) } else { None },
            p_hydro: Some(vram.alloc_zeros(n_cells)?),
            alpha: Some(vram.alloc_zeros(n_cells)?),
            alpha_tmp: if alloc_flow { Some(vram.alloc_zeros(n_cells)?) } else { None },
            alpha_smooth: if alloc_flow { Some(vram.alloc_zeros(n_cells)?) } else { None },
            kappa: if alloc_flow { Some(vram.alloc_zeros(n_cells)?) } else { None },
            div_u: Some(vram.alloc_zeros(n_cells)?),
            src_p: Some(vram.alloc_zeros(n_cells)?),

            ux: Some(vram.alloc_zeros(n_fx)?),
            ux_old: if alloc_amr_old { Some(vram.alloc_zeros(n_fx)?) } else { None },
            a_slow_x: if alloc_flow { Some(vram.alloc_zeros(n_fx)?) } else { None },
            ib_beta_x: Some(vram.alloc_zeros(n_fx)?),
            ib_beta_static_x: Some(vram.alloc_zeros(n_fx)?),
            ib_vel_x: if alloc_ib_vel { Some(vram.alloc_zeros(n_fx)?) } else { None },
            mflux_x: if alloc_flow { Some(vram.alloc_zeros(n_fx)?) } else { None },
            vflux_x: if alloc_flow { Some(vram.alloc_zeros(n_fx)?) } else { None },

            uy: Some(vram.alloc_zeros(n_fy)?),
            uy_old: if alloc_amr_old { Some(vram.alloc_zeros(n_fy)?) } else { None },
            a_slow_y: if alloc_flow { Some(vram.alloc_zeros(n_fy)?) } else { None },
            ib_beta_y: Some(vram.alloc_zeros(n_fy)?),
            ib_beta_static_y: Some(vram.alloc_zeros(n_fy)?),
            ib_vel_y: if alloc_ib_vel { Some(vram.alloc_zeros(n_fy)?) } else { None },
            mflux_y: if alloc_flow { Some(vram.alloc_zeros(n_fy)?) } else { None },
            vflux_y: if alloc_flow { Some(vram.alloc_zeros(n_fy)?) } else { None },

            uz: Some(vram.alloc_zeros(n_fz)?),
            uz_old: if alloc_amr_old { Some(vram.alloc_zeros(n_fz)?) } else { None },
            a_slow_z: if alloc_flow { Some(vram.alloc_zeros(n_fz)?) } else { None },
            ib_beta_z: Some(vram.alloc_zeros(n_fz)?),
            ib_beta_static_z: Some(vram.alloc_zeros(n_fz)?),
            ib_vel_z: if alloc_ib_vel { Some(vram.alloc_zeros(n_fz)?) } else { None },
            mflux_z: if alloc_flow { Some(vram.alloc_zeros(n_fz)?) } else { None },
            vflux_z: if alloc_flow { Some(vram.alloc_zeros(n_fz)?) } else { None },

            psi_ux: if has_pml { Some(vram.alloc_zeros(n_cells)?) } else { None },
            psi_uy: if has_pml { Some(vram.alloc_zeros(n_cells)?) } else { None },
            psi_uz: if has_pml { Some(vram.alloc_zeros(n_cells)?) } else { None },
            psi_px: if has_pml { Some(vram.alloc_zeros(n_fx)?) } else { None },
            psi_py: if has_pml { Some(vram.alloc_zeros(n_fy)?) } else { None },
            psi_pz: if has_pml { Some(vram.alloc_zeros(n_fz)?) } else { None },

            bnd_vflux_xlo: if has_amr { Some(vram.alloc_zeros(spec.ny * spec.nz)?) } else { None },
            bnd_vflux_xhi: if has_amr { Some(vram.alloc_zeros(spec.ny * spec.nz)?) } else { None },
            bnd_vflux_ylo: if has_amr { Some(vram.alloc_zeros(spec.nx * spec.nz)?) } else { None },
            bnd_vflux_yhi: if has_amr { Some(vram.alloc_zeros(spec.nx * spec.nz)?) } else { None },
            bnd_vflux_zlo: if has_amr { Some(vram.alloc_zeros(spec.nx * spec.ny)?) } else { None },
            bnd_vflux_zhi: if has_amr { Some(vram.alloc_zeros(spec.nx * spec.ny)?) } else { None },

            bnd_aflux_xlo: if has_amr { Some(vram.alloc_zeros(spec.ny * spec.nz)?) } else { None },
            bnd_aflux_xhi: if has_amr { Some(vram.alloc_zeros(spec.ny * spec.nz)?) } else { None },
            bnd_aflux_ylo: if has_amr { Some(vram.alloc_zeros(spec.nx * spec.nz)?) } else { None },
            bnd_aflux_yhi: if has_amr { Some(vram.alloc_zeros(spec.nx * spec.nz)?) } else { None },
            bnd_aflux_zlo: if has_amr { Some(vram.alloc_zeros(spec.nx * spec.ny)?) } else { None },
            bnd_aflux_zhi: if has_amr { Some(vram.alloc_zeros(spec.nx * spec.ny)?) } else { None },

            u_rho: None,
            u_mx: None,
            u_my: None,
            u_mz: None,
            u_ene: None,
            u1_rho: None,
            u1_mx: None,
            u1_my: None,
            u1_mz: None,
            u1_ene: None,
            cell_ib: None,
            cell_ib_static: None,
            cell_vx: None,
            cell_vy: None,
            cell_vz: None,

            spec,
            halo,
            nxp,
            nyp,
            nzp,
        })
    }

    pub fn descriptor(&mut self) -> FluidLevelGpu {
        fn opt_ptr(b: &mut Option<DeviceBuffer<f32>>) -> *mut f32 {
            match b {
                Some(buf) => buf.as_mut_ptr(),
                None => std::ptr::null_mut(),
            }
        }
        FluidLevelGpu {
            nx: self.spec.nx as i32,
            ny: self.spec.ny as i32,
            nz: self.spec.nz as i32,
            halo: self.halo as i32,
            nxp: self.nxp as i32,
            nyp: self.nyp as i32,
            nzp: self.nzp as i32,
            dx: self.spec.dx,
            ox: self.spec.ox,
            oy: self.spec.oy,
            oz: self.spec.oz,
            level_idx: self.spec.level_idx as i32,
            refine_ratio: self.spec.refine_ratio as i32,
            pml_thickness: self.spec.pml_thickness as i32,
            pml_face_mask: self.spec.pml_face_mask,
            off_x: self.spec.off_x as i32,
            off_y: self.spec.off_y as i32,
            off_z: self.spec.off_z as i32,

            p_gauge: opt_ptr(&mut self.p_gauge),
            p_gauge_old: opt_ptr(&mut self.p_gauge_old),
            p_hydro: opt_ptr(&mut self.p_hydro),
            alpha: opt_ptr(&mut self.alpha),
            alpha_tmp: opt_ptr(&mut self.alpha_tmp),
            alpha_smooth: opt_ptr(&mut self.alpha_smooth),
            kappa: opt_ptr(&mut self.kappa),
            div_u: opt_ptr(&mut self.div_u),
            src_p: opt_ptr(&mut self.src_p),

            ux: opt_ptr(&mut self.ux),
            ux_old: opt_ptr(&mut self.ux_old),
            a_slow_x: opt_ptr(&mut self.a_slow_x),
            ib_beta_x: opt_ptr(&mut self.ib_beta_x),
            ib_beta_static_x: opt_ptr(&mut self.ib_beta_static_x),
            ib_vel_x: opt_ptr(&mut self.ib_vel_x),
            mflux_x: opt_ptr(&mut self.mflux_x),
            vflux_x: opt_ptr(&mut self.vflux_x),

            uy: opt_ptr(&mut self.uy),
            uy_old: opt_ptr(&mut self.uy_old),
            a_slow_y: opt_ptr(&mut self.a_slow_y),
            ib_beta_y: opt_ptr(&mut self.ib_beta_y),
            ib_beta_static_y: opt_ptr(&mut self.ib_beta_static_y),
            ib_vel_y: opt_ptr(&mut self.ib_vel_y),
            mflux_y: opt_ptr(&mut self.mflux_y),
            vflux_y: opt_ptr(&mut self.vflux_y),

            uz: opt_ptr(&mut self.uz),
            uz_old: opt_ptr(&mut self.uz_old),
            a_slow_z: opt_ptr(&mut self.a_slow_z),
            ib_beta_z: opt_ptr(&mut self.ib_beta_z),
            ib_beta_static_z: opt_ptr(&mut self.ib_beta_static_z),
            ib_vel_z: opt_ptr(&mut self.ib_vel_z),
            mflux_z: opt_ptr(&mut self.mflux_z),
            vflux_z: opt_ptr(&mut self.vflux_z),

            psi_ux: opt_ptr(&mut self.psi_ux),
            psi_uy: opt_ptr(&mut self.psi_uy),
            psi_uz: opt_ptr(&mut self.psi_uz),
            psi_px: opt_ptr(&mut self.psi_px),
            psi_py: opt_ptr(&mut self.psi_py),
            psi_pz: opt_ptr(&mut self.psi_pz),

            bnd_vflux_xlo: opt_ptr(&mut self.bnd_vflux_xlo),
            bnd_vflux_xhi: opt_ptr(&mut self.bnd_vflux_xhi),
            bnd_vflux_ylo: opt_ptr(&mut self.bnd_vflux_ylo),
            bnd_vflux_yhi: opt_ptr(&mut self.bnd_vflux_yhi),
            bnd_vflux_zlo: opt_ptr(&mut self.bnd_vflux_zlo),
            bnd_vflux_zhi: opt_ptr(&mut self.bnd_vflux_zhi),

            bnd_aflux_xlo: opt_ptr(&mut self.bnd_aflux_xlo),
            bnd_aflux_xhi: opt_ptr(&mut self.bnd_aflux_xhi),
            bnd_aflux_ylo: opt_ptr(&mut self.bnd_aflux_ylo),
            bnd_aflux_yhi: opt_ptr(&mut self.bnd_aflux_yhi),
            bnd_aflux_zlo: opt_ptr(&mut self.bnd_aflux_zlo),
            bnd_aflux_zhi: opt_ptr(&mut self.bnd_aflux_zhi),

            u_rho: opt_ptr(&mut self.u_rho),
            u_mx: opt_ptr(&mut self.u_mx),
            u_my: opt_ptr(&mut self.u_my),
            u_mz: opt_ptr(&mut self.u_mz),
            u_ene: opt_ptr(&mut self.u_ene),
            u1_rho: opt_ptr(&mut self.u1_rho),
            u1_mx: opt_ptr(&mut self.u1_mx),
            u1_my: opt_ptr(&mut self.u1_my),
            u1_mz: opt_ptr(&mut self.u1_mz),
            u1_ene: opt_ptr(&mut self.u1_ene),
            cell_ib: opt_ptr(&mut self.cell_ib),
            cell_ib_static: opt_ptr(&mut self.cell_ib_static),
            cell_vx: opt_ptr(&mut self.cell_vx),
            cell_vy: opt_ptr(&mut self.cell_vy),
            cell_vz: opt_ptr(&mut self.cell_vz),
        }
    }

    pub fn cell_count_padded(&self) -> usize {
        self.nxp * self.nyp * self.nzp
    }
}

pub fn device_sync() -> Result<()> {
    unsafe { check_cuda(sim_cuda_device_sync(), "sim_cuda_device_sync") }
}
