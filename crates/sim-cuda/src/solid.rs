use crate::error::Result;
use crate::ffi::{MpmGridGpu, MpmParticlesGpu, ShellMeshGpu};
use crate::vram::{DeviceBuffer, VramBudgetManager};

#[derive(Clone, Debug, Default)]
pub struct MpmParticlesHost {
    pub px: Vec<f32>,
    pub py: Vec<f32>,
    pub pz: Vec<f32>,
    pub vx: Vec<f32>,
    pub vy: Vec<f32>,
    pub vz: Vec<f32>,
    pub c_mat: Vec<f32>,
    pub f_mat: Vec<f32>,
    pub q0_mat: Vec<f32>,
    pub q_visc: Vec<f32>,
    pub nx0: Vec<f32>,
    pub ny0: Vec<f32>,
    pub nz0: Vec<f32>,
    pub nx: Vec<f32>,
    pub ny: Vec<f32>,
    pub nz: Vec<f32>,
    pub area: Vec<f32>,
    pub mass: Vec<f32>,
    pub vol0: Vec<f32>,
    pub thickness: Vec<f32>,
    pub damage: Vec<f32>,
    pub hist_h: Vec<f32>,
    pub gc_scale: Vec<f32>,
    pub wetness: Vec<f32>,
    pub mat_id: Vec<i32>,
    pub body_id: Vec<i32>,
    pub flags: Vec<u32>,
    pub v_presc_x: Vec<f32>,
    pub v_presc_y: Vec<f32>,
    pub v_presc_z: Vec<f32>,
    pub hyd_len: Vec<f32>,
    pub fluid_rho: Vec<f32>,
}

impl MpmParticlesHost {
    pub fn len(&self) -> usize {
        self.px.len()
    }

    pub fn is_empty(&self) -> bool {
        self.px.is_empty()
    }

    /// Reorder particles by (sub_slot, 64-color bucket, z, y, x) so consecutive particles
    /// in every GPU warp have disjoint 3x3x3 B-spline stencils (zero L2 atomic contention).
    pub fn reorder_color_disjoint(
        &mut self,
        q0_per_particle: &mut Vec<[f32; 9]>,
        origin: [f32; 3],
        dx: f32,
    ) {
        let n = self.px.len();
        if n <= 1 || dx <= 0.0 {
            return;
        }
        use std::collections::HashMap;
        let mut cell_counts: HashMap<(isize, isize, isize), usize> = HashMap::with_capacity(n / 4);
        let mut keys: Vec<(usize, usize, isize, isize, isize, usize)> = Vec::with_capacity(n);
        for p in 0..n {
            let cx = ((self.px[p] - origin[0]) / dx).floor() as isize;
            let cy = ((self.py[p] - origin[1]) / dx).floor() as isize;
            let cz = ((self.pz[p] - origin[2]) / dx).floor() as isize;
            let slot = {
                let e = cell_counts.entry((cx, cy, cz)).or_insert(0);
                let s = *e;
                *e += 1;
                s
            };
            let color = (cx.rem_euclid(4) as usize)
                + 4 * (cy.rem_euclid(4) as usize)
                + 16 * (cz.rem_euclid(4) as usize);
            keys.push((slot, color, cz, cy, cx, p));
        }
        keys.sort_unstable();
        let perm: Vec<usize> = keys.into_iter().map(|k| k.5).collect();

        fn permute_vec<T: Copy>(v: &mut Vec<T>, perm: &[usize]) {
            if v.is_empty() {
                return;
            }
            let out: Vec<T> = perm.iter().map(|&i| v[i]).collect();
            *v = out;
        }

        permute_vec(&mut self.px, &perm);
        permute_vec(&mut self.py, &perm);
        permute_vec(&mut self.pz, &perm);
        permute_vec(&mut self.vx, &perm);
        permute_vec(&mut self.vy, &perm);
        permute_vec(&mut self.vz, &perm);
        permute_vec(&mut self.nx0, &perm);
        permute_vec(&mut self.ny0, &perm);
        permute_vec(&mut self.nz0, &perm);
        permute_vec(&mut self.nx, &perm);
        permute_vec(&mut self.ny, &perm);
        permute_vec(&mut self.nz, &perm);
        permute_vec(&mut self.area, &perm);
        permute_vec(&mut self.mass, &perm);
        permute_vec(&mut self.vol0, &perm);
        permute_vec(&mut self.thickness, &perm);
        permute_vec(&mut self.damage, &perm);
        permute_vec(&mut self.hist_h, &perm);
        permute_vec(&mut self.gc_scale, &perm);
        permute_vec(&mut self.wetness, &perm);
        permute_vec(&mut self.mat_id, &perm);
        permute_vec(&mut self.body_id, &perm);
        permute_vec(&mut self.flags, &perm);
        permute_vec(&mut self.v_presc_x, &perm);
        permute_vec(&mut self.v_presc_y, &perm);
        permute_vec(&mut self.v_presc_z, &perm);
        permute_vec(&mut self.hyd_len, &perm);
        permute_vec(&mut self.fluid_rho, &perm);
        permute_vec(q0_per_particle, &perm);
    }

    /// Resize/fill SoA matrix arrays cleanly from per-particle fields before upload.
    pub fn finalize_soa(&mut self, q0_per_particle: &[[f32; 9]]) {
        let n = self.px.len();
        self.c_mat.resize(9 * n, 0.0);
        self.f_mat.resize(9 * n, 0.0);
        self.q0_mat.resize(9 * n, 0.0);
        self.q_visc.resize(6 * n, 0.0);
        if self.hyd_len.len() != n {
            self.hyd_len.resize(n, 0.0);
        }
        if self.fluid_rho.len() != n {
            self.fluid_rho.resize(n, 1.204);
        }

        for p in 0..n {
            for r in 0..3 {
                for c in 0..3 {
                    let m = r * 3 + c;
                    // Store displacement gradient perturbation H = F - I to avoid FP32 quantization around 1.0
                    self.f_mat[m * n + p] = 0.0;
                    self.q0_mat[m * n + p] = q0_per_particle[p][m];
                }
            }
        }
    }
}

pub struct MpmParticlesDevice {
    pub count: usize,
    pub px: DeviceBuffer<f32>,
    pub py: DeviceBuffer<f32>,
    pub pz: DeviceBuffer<f32>,
    pub vx: DeviceBuffer<f32>,
    pub vy: DeviceBuffer<f32>,
    pub vz: DeviceBuffer<f32>,
    pub c_mat: DeviceBuffer<f32>,
    pub f_mat: DeviceBuffer<f32>,
    pub q0_mat: DeviceBuffer<f32>,
    pub q_visc: DeviceBuffer<f32>,
    pub nx0: DeviceBuffer<f32>,
    pub ny0: DeviceBuffer<f32>,
    pub nz0: DeviceBuffer<f32>,
    pub nx: DeviceBuffer<f32>,
    pub ny: DeviceBuffer<f32>,
    pub nz: DeviceBuffer<f32>,
    pub area: DeviceBuffer<f32>,
    pub mass: DeviceBuffer<f32>,
    pub vol0: DeviceBuffer<f32>,
    pub thickness: DeviceBuffer<f32>,
    pub damage: DeviceBuffer<f32>,
    pub hist_h: DeviceBuffer<f32>,
    pub gc_scale: DeviceBuffer<f32>,
    pub wetness: DeviceBuffer<f32>,
    pub fx_ext: DeviceBuffer<f32>,
    pub fy_ext: DeviceBuffer<f32>,
    pub fz_ext: DeviceBuffer<f32>,
    pub mat_id: DeviceBuffer<i32>,
    pub body_id: DeviceBuffer<i32>,
    pub flags: DeviceBuffer<u32>,
    pub v_presc_x: DeviceBuffer<f32>,
    pub v_presc_y: DeviceBuffer<f32>,
    pub v_presc_z: DeviceBuffer<f32>,
    pub hyd_len: DeviceBuffer<f32>,
    pub fluid_rho: DeviceBuffer<f32>,
}

impl MpmParticlesDevice {
    pub fn from_host(vram: &VramBudgetManager, h: &MpmParticlesHost) -> Result<Self> {
        let count = h.len();
        let hyd_dev = if h.hyd_len.len() == count {
            vram.alloc_from_slice(&h.hyd_len)?
        } else {
            vram.alloc_zeros(count)?
        };
        let rho_dev = if h.fluid_rho.len() == count {
            vram.alloc_from_slice(&h.fluid_rho)?
        } else {
            vram.alloc_from_slice(&vec![1.204_f32; count])?
        };
        Ok(Self {
            count,
            px: vram.alloc_from_slice(&h.px)?,
            py: vram.alloc_from_slice(&h.py)?,
            pz: vram.alloc_from_slice(&h.pz)?,
            vx: vram.alloc_from_slice(&h.vx)?,
            vy: vram.alloc_from_slice(&h.vy)?,
            vz: vram.alloc_from_slice(&h.vz)?,
            c_mat: vram.alloc_from_slice(&h.c_mat)?,
            f_mat: vram.alloc_from_slice(&h.f_mat)?,
            q0_mat: vram.alloc_from_slice(&h.q0_mat)?,
            q_visc: vram.alloc_from_slice(&h.q_visc)?,
            nx0: vram.alloc_from_slice(&h.nx0)?,
            ny0: vram.alloc_from_slice(&h.ny0)?,
            nz0: vram.alloc_from_slice(&h.nz0)?,
            nx: vram.alloc_from_slice(&h.nx)?,
            ny: vram.alloc_from_slice(&h.ny)?,
            nz: vram.alloc_from_slice(&h.nz)?,
            area: vram.alloc_from_slice(&h.area)?,
            mass: vram.alloc_from_slice(&h.mass)?,
            vol0: vram.alloc_from_slice(&h.vol0)?,
            thickness: vram.alloc_from_slice(&h.thickness)?,
            damage: vram.alloc_from_slice(&h.damage)?,
            hist_h: vram.alloc_from_slice(&h.hist_h)?,
            gc_scale: vram.alloc_from_slice(&h.gc_scale)?,
            wetness: vram.alloc_from_slice(&h.wetness)?,
            fx_ext: vram.alloc_zeros(count)?,
            fy_ext: vram.alloc_zeros(count)?,
            fz_ext: vram.alloc_zeros(count)?,
            mat_id: vram.alloc_from_slice(&h.mat_id)?,
            body_id: vram.alloc_from_slice(&h.body_id)?,
            flags: vram.alloc_from_slice(&h.flags)?,
            v_presc_x: vram.alloc_from_slice(&h.v_presc_x)?,
            v_presc_y: vram.alloc_from_slice(&h.v_presc_y)?,
            v_presc_z: vram.alloc_from_slice(&h.v_presc_z)?,
            hyd_len: hyd_dev,
            fluid_rho: rho_dev,
        })
    }

    pub fn descriptor(&mut self) -> MpmParticlesGpu {
        MpmParticlesGpu {
            count: self.count as i32,
            _pad: 0,
            px: self.px.as_mut_ptr(),
            py: self.py.as_mut_ptr(),
            pz: self.pz.as_mut_ptr(),
            vx: self.vx.as_mut_ptr(),
            vy: self.vy.as_mut_ptr(),
            vz: self.vz.as_mut_ptr(),
            c_mat: self.c_mat.as_mut_ptr(),
            f_mat: self.f_mat.as_mut_ptr(),
            q0_mat: self.q0_mat.as_mut_ptr(),
            q_visc: self.q_visc.as_mut_ptr(),
            nx0: self.nx0.as_mut_ptr(),
            ny0: self.ny0.as_mut_ptr(),
            nz0: self.nz0.as_mut_ptr(),
            nx: self.nx.as_mut_ptr(),
            ny: self.ny.as_mut_ptr(),
            nz: self.nz.as_mut_ptr(),
            area: self.area.as_mut_ptr(),
            mass: self.mass.as_mut_ptr(),
            vol0: self.vol0.as_mut_ptr(),
            thickness: self.thickness.as_mut_ptr(),
            damage: self.damage.as_mut_ptr(),
            hist_h: self.hist_h.as_mut_ptr(),
            gc_scale: self.gc_scale.as_mut_ptr(),
            wetness: self.wetness.as_mut_ptr(),
            fx_ext: self.fx_ext.as_mut_ptr(),
            fy_ext: self.fy_ext.as_mut_ptr(),
            fz_ext: self.fz_ext.as_mut_ptr(),
            mat_id: self.mat_id.as_mut_ptr(),
            body_id: self.body_id.as_mut_ptr(),
            flags: self.flags.as_mut_ptr(),
            v_presc_x: self.v_presc_x.as_mut_ptr(),
            v_presc_y: self.v_presc_y.as_mut_ptr(),
            v_presc_z: self.v_presc_z.as_mut_ptr(),
            hyd_len: self.hyd_len.as_mut_ptr(),
            fluid_rho: self.fluid_rho.as_mut_ptr(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MpmGridSpec {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f32,
    pub ox: f32,
    pub oy: f32,
    pub oz: f32,
    pub has_floor: bool,
    pub floor_z: f32,
    pub floor_mu: f32,
}

pub struct MpmGridDevice {
    pub spec: MpmGridSpec,
    pub mass: DeviceBuffer<f32>,
    pub vx: DeviceBuffer<f32>,
    pub vy: DeviceBuffer<f32>,
    pub vz: DeviceBuffer<f32>,
    pub fx: DeviceBuffer<f32>,
    pub fy: DeviceBuffer<f32>,
    pub fz: DeviceBuffer<f32>,
    pub bc_w: DeviceBuffer<f32>,
    pub bc_vx: DeviceBuffer<f32>,
    pub bc_vy: DeviceBuffer<f32>,
    pub bc_vz: DeviceBuffer<f32>,
    pub d_field: DeviceBuffer<f32>,
    pub d_new: DeviceBuffer<f32>,
    pub h_field: DeviceBuffer<f32>,
    pub gc_field: DeviceBuffer<f32>,
    pub lc_field: DeviceBuffer<f32>,
    pub eta_field: DeviceBuffer<f32>,
    pub w_field: DeviceBuffer<f32>,
}

impl MpmGridDevice {
    pub fn new(vram: &VramBudgetManager, spec: MpmGridSpec, allocate_phase_field: bool) -> Result<Self> {
        let n_nodes = spec.nx * spec.ny * spec.nz;
        let n2 = 2 * n_nodes;
        let pf = if allocate_phase_field { n_nodes } else { 0 };
        Ok(Self {
            mass: vram.alloc_zeros(n2)?,
            vx: vram.alloc_zeros(n2)?,
            vy: vram.alloc_zeros(n2)?,
            vz: vram.alloc_zeros(n2)?,
            fx: vram.alloc_zeros(n2)?,
            fy: vram.alloc_zeros(n2)?,
            fz: vram.alloc_zeros(n2)?,
            bc_w: vram.alloc_zeros(n2)?,
            bc_vx: vram.alloc_zeros(n2)?,
            bc_vy: vram.alloc_zeros(n2)?,
            bc_vz: vram.alloc_zeros(n2)?,
            d_field: vram.alloc_zeros(pf)?,
            d_new: vram.alloc_zeros(pf)?,
            h_field: vram.alloc_zeros(pf)?,
            gc_field: vram.alloc_zeros(pf)?,
            lc_field: vram.alloc_zeros(pf)?,
            eta_field: vram.alloc_zeros(pf)?,
            w_field: vram.alloc_zeros(pf)?,
            spec,
        })
    }

    pub fn descriptor(&mut self) -> MpmGridGpu {
        MpmGridGpu {
            nx: self.spec.nx as i32,
            ny: self.spec.ny as i32,
            nz: self.spec.nz as i32,
            dx: self.spec.dx,
            ox: self.spec.ox,
            oy: self.spec.oy,
            oz: self.spec.oz,
            has_floor: if self.spec.has_floor { 1 } else { 0 },
            floor_z: self.spec.floor_z,
            floor_mu: self.spec.floor_mu,
            _pad: 0.0,
            _pad2: 0.0,
            mass: self.mass.as_mut_ptr(),
            vx: self.vx.as_mut_ptr(),
            vy: self.vy.as_mut_ptr(),
            vz: self.vz.as_mut_ptr(),
            fx: self.fx.as_mut_ptr(),
            fy: self.fy.as_mut_ptr(),
            fz: self.fz.as_mut_ptr(),
            bc_w: self.bc_w.as_mut_ptr(),
            bc_vx: self.bc_vx.as_mut_ptr(),
            bc_vy: self.bc_vy.as_mut_ptr(),
            bc_vz: self.bc_vz.as_mut_ptr(),
            d_field: self.d_field.as_mut_ptr(),
            d_new: self.d_new.as_mut_ptr(),
            h_field: self.h_field.as_mut_ptr(),
            gc_field: self.gc_field.as_mut_ptr(),
            lc_field: self.lc_field.as_mut_ptr(),
            eta_field: self.eta_field.as_mut_ptr(),
            w_field: self.w_field.as_mut_ptr(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ShellMeshHost {
    pub tri_v0: Vec<i32>,
    pub tri_v1: Vec<i32>,
    pub tri_v2: Vec<i32>,
    pub tri_dm_inv: Vec<f32>,
    pub tri_area0: Vec<f32>,
    pub tri_h: Vec<f32>,
    pub tri_mat_id: Vec<i32>,
    pub hinge_e0: Vec<i32>,
    pub hinge_e1: Vec<i32>,
    pub hinge_o0: Vec<i32>,
    pub hinge_o1: Vec<i32>,
    pub hinge_theta0: Vec<f32>,
    pub hinge_kb: Vec<f32>,
    pub hinge_dmax: Vec<f32>,
    pub hinge_mat_id: Vec<i32>,
}

pub struct ShellMeshDevice {
    pub num_tris: usize,
    pub num_hinges: usize,
    pub num_particles: usize,
    pub tri_v0: DeviceBuffer<i32>,
    pub tri_v1: DeviceBuffer<i32>,
    pub tri_v2: DeviceBuffer<i32>,
    pub tri_dm_inv: DeviceBuffer<f32>,
    pub tri_area0: DeviceBuffer<f32>,
    pub tri_h: DeviceBuffer<f32>,
    pub tri_mat_id: DeviceBuffer<i32>,
    pub hinge_e0: DeviceBuffer<i32>,
    pub hinge_e1: DeviceBuffer<i32>,
    pub hinge_o0: DeviceBuffer<i32>,
    pub hinge_o1: DeviceBuffer<i32>,
    pub hinge_theta0: DeviceBuffer<f32>,
    pub hinge_kb: DeviceBuffer<f32>,
    pub hinge_dmax: DeviceBuffer<f32>,
    pub hinge_mat_id: DeviceBuffer<i32>,
    pub p_fx: DeviceBuffer<f32>,
    pub p_fy: DeviceBuffer<f32>,
    pub p_fz: DeviceBuffer<f32>,
    pub hash_head: DeviceBuffer<i32>,
    pub hash_next: DeviceBuffer<i32>,
    pub hash_size: i32,
    pub cell_size: f32,
}

impl ShellMeshDevice {
    pub fn from_host(
        vram: &VramBudgetManager,
        h: &ShellMeshHost,
        num_particles: usize,
        cell_size: f32,
    ) -> Result<Self> {
        let hash_size = 16384i32;
        let hash_head = vram.alloc_from_slice(&vec![-1i32; hash_size as usize])?;
        Ok(Self {
            num_tris: h.tri_v0.len(),
            num_hinges: h.hinge_e0.len(),
            num_particles,
            tri_v0: vram.alloc_from_slice(&h.tri_v0)?,
            tri_v1: vram.alloc_from_slice(&h.tri_v1)?,
            tri_v2: vram.alloc_from_slice(&h.tri_v2)?,
            tri_dm_inv: vram.alloc_from_slice(&h.tri_dm_inv)?,
            tri_area0: vram.alloc_from_slice(&h.tri_area0)?,
            tri_h: vram.alloc_from_slice(&h.tri_h)?,
            tri_mat_id: vram.alloc_from_slice(&h.tri_mat_id)?,
            hinge_e0: vram.alloc_from_slice(&h.hinge_e0)?,
            hinge_e1: vram.alloc_from_slice(&h.hinge_e1)?,
            hinge_o0: vram.alloc_from_slice(&h.hinge_o0)?,
            hinge_o1: vram.alloc_from_slice(&h.hinge_o1)?,
            hinge_theta0: vram.alloc_from_slice(&h.hinge_theta0)?,
            hinge_kb: vram.alloc_from_slice(&h.hinge_kb)?,
            hinge_dmax: vram.alloc_from_slice(&h.hinge_dmax)?,
            hinge_mat_id: vram.alloc_from_slice(&h.hinge_mat_id)?,
            p_fx: vram.alloc_zeros(num_particles)?,
            p_fy: vram.alloc_zeros(num_particles)?,
            p_fz: vram.alloc_zeros(num_particles)?,
            hash_head,
            hash_next: vram.alloc_zeros(num_particles)?,
            hash_size,
            cell_size,
        })
    }

    pub fn descriptor(&mut self) -> ShellMeshGpu {
        ShellMeshGpu {
            num_tris: self.num_tris as i32,
            num_hinges: self.num_hinges as i32,
            tri_v0: self.tri_v0.as_mut_ptr(),
            tri_v1: self.tri_v1.as_mut_ptr(),
            tri_v2: self.tri_v2.as_mut_ptr(),
            tri_dm_inv: self.tri_dm_inv.as_mut_ptr(),
            tri_area0: self.tri_area0.as_mut_ptr(),
            tri_h: self.tri_h.as_mut_ptr(),
            tri_mat_id: self.tri_mat_id.as_mut_ptr(),
            hinge_e0: self.hinge_e0.as_mut_ptr(),
            hinge_e1: self.hinge_e1.as_mut_ptr(),
            hinge_o0: self.hinge_o0.as_mut_ptr(),
            hinge_o1: self.hinge_o1.as_mut_ptr(),
            hinge_theta0: self.hinge_theta0.as_mut_ptr(),
            hinge_kb: self.hinge_kb.as_mut_ptr(),
            hinge_dmax: self.hinge_dmax.as_mut_ptr(),
            hinge_mat_id: self.hinge_mat_id.as_mut_ptr(),
            p_fx: self.p_fx.as_mut_ptr(),
            p_fy: self.p_fy.as_mut_ptr(),
            p_fz: self.p_fz.as_mut_ptr(),
        }
    }
}
