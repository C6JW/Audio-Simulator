use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FluidConstantsGpu {
    pub rho_a: f64,
    pub rho_w: f64,
    pub gamma_a: f64,
    pub p0: f64,
    pub c_w: f64,
    pub k_w: f64,
    pub sigma: f64,
    pub mu_a: f64,
    pub mu_w: f64,
    pub gx: f32,
    pub gy: f32,
    pub gz: f32,
    pub alpha_cfs: f32,
    pub vof_beta: f32,
    pub pml_coeff: f32,
    pub wall_admittance: f32,
    pub div_damp_coeff: f32,
    pub bulk_viscosity_a: f32,
    pub prandtl_a: f32,
}

impl Default for FluidConstantsGpu {
    fn default() -> Self {
        let rho_w = 998.2_f64;
        let c_w = 1480.0_f64;
        Self {
            rho_a: 1.204,
            rho_w,
            gamma_a: 1.4,
            p0: 101_325.0,
            c_w,
            k_w: rho_w * c_w * c_w,
            sigma: 0.0728,
            mu_a: 1.81e-5,
            mu_w: 1.002e-3,
            gx: 0.0,
            gy: 0.0,
            gz: -9.80665,
            alpha_cfs: 300.0,
            vof_beta: 1.8,
            pml_coeff: 1.2,
            wall_admittance: 0.0,
            div_damp_coeff: 0.025,
            bulk_viscosity_a: 0.003,
            prandtl_a: 0.71,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FluidLevelGpu {
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub halo: i32,
    pub nxp: i32,
    pub nyp: i32,
    pub nzp: i32,
    pub dx: f32,
    pub ox: f32,
    pub oy: f32,
    pub oz: f32,
    pub level_idx: i32,
    pub refine_ratio: i32,
    pub pml_thickness: i32,
    pub pml_face_mask: u32,
    pub off_x: i32,
    pub off_y: i32,
    pub off_z: i32,

    pub p_gauge: *mut f32,
    pub p_gauge_old: *mut f32,
    pub p_hydro: *mut f32,
    pub alpha: *mut f32,
    pub alpha_tmp: *mut f32,
    pub alpha_smooth: *mut f32,
    pub kappa: *mut f32,
    pub div_u: *mut f32,
    pub src_p: *mut f32,

    pub ux: *mut f32,
    pub ux_old: *mut f32,
    pub a_slow_x: *mut f32,
    pub ib_beta_x: *mut f32,
    pub ib_beta_static_x: *mut f32,
    pub ib_vel_x: *mut f32,
    pub mflux_x: *mut f32,
    pub vflux_x: *mut f32,

    pub uy: *mut f32,
    pub uy_old: *mut f32,
    pub a_slow_y: *mut f32,
    pub ib_beta_y: *mut f32,
    pub ib_beta_static_y: *mut f32,
    pub ib_vel_y: *mut f32,
    pub mflux_y: *mut f32,
    pub vflux_y: *mut f32,

    pub uz: *mut f32,
    pub uz_old: *mut f32,
    pub a_slow_z: *mut f32,
    pub ib_beta_z: *mut f32,
    pub ib_beta_static_z: *mut f32,
    pub ib_vel_z: *mut f32,
    pub mflux_z: *mut f32,
    pub vflux_z: *mut f32,

    pub psi_ux: *mut f32,
    pub psi_uy: *mut f32,
    pub psi_uz: *mut f32,
    pub psi_px: *mut f32,
    pub psi_py: *mut f32,
    pub psi_pz: *mut f32,

    pub bnd_vflux_xlo: *mut f32,
    pub bnd_vflux_xhi: *mut f32,
    pub bnd_vflux_ylo: *mut f32,
    pub bnd_vflux_yhi: *mut f32,
    pub bnd_vflux_zlo: *mut f32,
    pub bnd_vflux_zhi: *mut f32,

    pub bnd_aflux_xlo: *mut f32,
    pub bnd_aflux_xhi: *mut f32,
    pub bnd_aflux_ylo: *mut f32,
    pub bnd_aflux_yhi: *mut f32,
    pub bnd_aflux_zlo: *mut f32,
    pub bnd_aflux_zhi: *mut f32,

    pub u_rho: *mut f32,
    pub u_mx: *mut f32,
    pub u_my: *mut f32,
    pub u_mz: *mut f32,
    pub u_ene: *mut f32,
    pub u1_rho: *mut f32,
    pub u1_mx: *mut f32,
    pub u1_my: *mut f32,
    pub u1_mz: *mut f32,
    pub u1_ene: *mut f32,
    pub cell_ib: *mut f32,
    pub cell_ib_static: *mut f32,
    pub cell_vx: *mut f32,
    pub cell_vy: *mut f32,
    pub cell_vz: *mut f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MicProbeGpu {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub level_idx: i32,
    pub is_hydrophone: i32,
    pub _pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SimReduceMetricsGpu {
    pub nan_count: u64,
    pub c_max_sq: f64,
    pub u_max_sq: f64,
    pub acoustic_energy: f64,
    pub kinetic_energy: f64,
    pub water_volume: f64,
    pub subsurface_air_volume: f64,
    pub max_abs_p: f64,
    pub min_p: f64,
    pub max_p: f64,
    pub crater_min_z: f32,
    pub _pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MpmMetricsGpu {
    pub kinetic_energy: f64,
    pub elastic_energy: f64,
    pub damage_sum: f64,
    pub nan_count: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MaterialParamsGpu {
    pub rho: f32,
    pub e1: f32,
    pub e2: f32,
    pub e3: f32,
    pub nu12: f32,
    pub nu23: f32,
    pub nu13: f32,
    pub g12: f32,
    pub g23: f32,
    pub g13: f32,
    pub eta_visc: f32,
    pub sigma_y: f32,
    pub tau_vp: f32,
    pub g_c: f32,
    pub l_c: f32,
    pub eta_frac: f32,
    pub thickness: f32,
    pub wetness: f32,
    pub contact_angle_cos: f32,
    pub friction_mu: f32,
    pub flags: u32,
    pub sls_beta: f32,
    pub sls_tau: f32,
    pub c11: f32,
    pub c22: f32,
    pub c33: f32,
    pub c12: f32,
    pub c13: f32,
    pub c23: f32,
    pub hygro_absorb_rate: f32,
    pub hygro_softening: f32,
    pub _pad_m2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MpmParticlesGpu {
    pub count: i32,
    pub _pad: i32,
    pub px: *mut f32,
    pub py: *mut f32,
    pub pz: *mut f32,
    pub vx: *mut f32,
    pub vy: *mut f32,
    pub vz: *mut f32,
    pub c_mat: *mut f32,
    pub f_mat: *mut f32,
    pub q0_mat: *mut f32,
    pub q_visc: *mut f32,
    pub nx0: *mut f32,
    pub ny0: *mut f32,
    pub nz0: *mut f32,
    pub nx: *mut f32,
    pub ny: *mut f32,
    pub nz: *mut f32,
    pub area: *mut f32,
    pub mass: *mut f32,
    pub vol0: *mut f32,
    pub thickness: *mut f32,
    pub damage: *mut f32,
    pub hist_h: *mut f32,
    pub gc_scale: *mut f32,
    pub wetness: *mut f32,
    pub fx_ext: *mut f32,
    pub fy_ext: *mut f32,
    pub fz_ext: *mut f32,
    pub mat_id: *mut i32,
    pub body_id: *mut i32,
    pub flags: *mut u32,
    pub v_presc_x: *mut f32,
    pub v_presc_y: *mut f32,
    pub v_presc_z: *mut f32,
    pub hyd_len: *mut f32,
    pub fluid_rho: *mut f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MpmGridGpu {
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub dx: f32,
    pub ox: f32,
    pub oy: f32,
    pub oz: f32,
    pub has_floor: i32,
    pub floor_z: f32,
    pub floor_mu: f32,
    pub _pad: f32,
    pub _pad2: f32,
    pub mass: *mut f32,
    pub vx: *mut f32,
    pub vy: *mut f32,
    pub vz: *mut f32,
    pub fx: *mut f32,
    pub fy: *mut f32,
    pub fz: *mut f32,
    pub bc_w: *mut f32,
    pub bc_vx: *mut f32,
    pub bc_vy: *mut f32,
    pub bc_vz: *mut f32,
    pub d_field: *mut f32,
    pub d_new: *mut f32,
    pub h_field: *mut f32,
    pub gc_field: *mut f32,
    pub lc_field: *mut f32,
    pub eta_field: *mut f32,
    pub w_field: *mut f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ShellMeshGpu {
    pub num_tris: i32,
    pub num_hinges: i32,
    pub tri_v0: *mut i32,
    pub tri_v1: *mut i32,
    pub tri_v2: *mut i32,
    pub tri_dm_inv: *mut f32,
    pub tri_area0: *mut f32,
    pub tri_h: *mut f32,
    pub tri_mat_id: *mut i32,
    pub hinge_e0: *mut i32,
    pub hinge_e1: *mut i32,
    pub hinge_o0: *mut i32,
    pub hinge_o1: *mut i32,
    pub hinge_theta0: *mut f32,
    pub hinge_kb: *mut f32,
    pub hinge_dmax: *mut f32,
    pub hinge_mat_id: *mut i32,
    pub p_fx: *mut f32,
    pub p_fy: *mut f32,
    pub p_fz: *mut f32,
}

extern "C" {
    pub fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> i32;

    pub fn sim_cuda_init_device(
        device_id: i32,
        cc_major: *mut i32,
        cc_minor: *mut i32,
        total_vram: *mut usize,
    ) -> i32;
    pub fn sim_cuda_malloc(ptr: *mut *mut c_void, bytes: usize) -> i32;
    pub fn sim_cuda_free(ptr: *mut c_void) -> i32;
    pub fn sim_cuda_memset(ptr: *mut c_void, val: i32, bytes: usize) -> i32;
    pub fn sim_cuda_memcpy_h2d(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
    pub fn sim_cuda_memcpy_d2h(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
    pub fn sim_cuda_memcpy_d2d(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
    pub fn sim_cuda_device_sync() -> i32;
    pub fn sim_cuda_get_last_error() -> i32;

    pub fn sim_cuda_stream_create(stream_out: *mut *mut c_void) -> i32;
    pub fn sim_cuda_stream_destroy(stream: *mut c_void) -> i32;
    pub fn sim_cuda_stream_sync(stream: *mut c_void) -> i32;

    pub fn sim_cuda_graph_begin_capture(stream: *mut c_void) -> i32;
    pub fn sim_cuda_graph_end_capture(stream: *mut c_void, graph_exec_out: *mut *mut c_void) -> i32;
    pub fn sim_cuda_graph_launch(graph_exec: *mut c_void, stream: *mut c_void) -> i32;
    pub fn sim_cuda_graph_destroy(graph_exec: *mut c_void) -> i32;

    pub fn sim_fluid_init_hydrostatic_root(
        lvl: FluidLevelGpu,
        c: FluidConstantsGpu,
        p_top_ref: f64,
        stream: *mut c_void,
    ) -> i32;
    pub fn sim_fluid_init_hydrostatic_child(
        child: FluidLevelGpu,
        parent: FluidLevelGpu,
        c: FluidConstantsGpu,
        stream: *mut c_void,
    ) -> i32;
    pub fn sim_fluid_add_sphere_laplace(
        lvl: FluidLevelGpu,
        cx: f32,
        cy: f32,
        cz: f32,
        radius: f32,
        delta_p: f64,
        stream: *mut c_void,
    ) -> i32;
    pub fn sim_fluid_inject_gaussian_pulse(
        lvl: FluidLevelGpu,
        cx: f32,
        cy: f32,
        cz: f32,
        sigma_r: f32,
        amp_pa: f64,
        stream: *mut c_void,
    ) -> i32;
    pub fn sim_fluid_set_point_source_ricker(
        lvl: FluidLevelGpu,
        cx: f32,
        cy: f32,
        cz: f32,
        radius: f32,
        rate_pa_per_s: f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn sim_fluid_apply_directional_source_bbox(
        lvl: FluidLevelGpu,
        c: FluidConstantsGpu,
        cx: f32,
        cy: f32,
        cz: f32,
        dir_x: f32,
        dir_y: f32,
        dir_z: f32,
        cardioid_beta: f32,
        radius: f32,
        rate_pa_per_s: f32,
        dt_ac: f32,
        clear_prev: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn sim_fluid_save_old(lvl: FluidLevelGpu, stream: *mut c_void) -> i32;
    pub fn sim_fluid_apply_outer_bc(lvl: FluidLevelGpu, stream: *mut c_void) -> i32;
    pub fn sim_fluid_acoustic_step(
        lvl: FluidLevelGpu,
        c: FluidConstantsGpu,
        dt_ac: f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn sim_fluid_flow_step(
        lvl: FluidLevelGpu,
        c: FluidConstantsGpu,
        dt_flow: f32,
        sweep_parity: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn sim_fluid_update_slow_forces_only(
        lvl: FluidLevelGpu,
        c: FluidConstantsGpu,
        stream: *mut c_void,
    ) -> i32;
    pub fn sim_fluid_amr_reset_fluxes(child: FluidLevelGpu, stream: *mut c_void) -> i32;
    pub fn sim_fluid_amr_fill_ghosts(
        child: FluidLevelGpu,
        parent: FluidLevelGpu,
        theta_time: f64,
        fill_alpha: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn sim_fluid_amr_reflux_and_restrict(
        child: FluidLevelGpu,
        parent: FluidLevelGpu,
        c: FluidConstantsGpu,
        dt_parent: f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn sim_fluid_amr_restrict_only(
        child: FluidLevelGpu,
        parent: FluidLevelGpu,
        c: FluidConstantsGpu,
        stream: *mut c_void,
    ) -> i32;
    pub fn sim_fluid_sample_probes(
        d_levels: *const FluidLevelGpu,
        num_levels: i32,
        d_probes: *const MicProbeGpu,
        num_probes: i32,
        d_out_samples: *mut f32,
        sample_idx: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn sim_fluid_reduce_metrics(
        lvl: FluidLevelGpu,
        c: FluidConstantsGpu,
        d_out: *mut SimReduceMetricsGpu,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_mpm_step(
        pts: MpmParticlesGpu,
        grid: MpmGridGpu,
        d_mats: *const MaterialParamsGpu,
        gx: f32,
        gy: f32,
        gz: f32,
        dt_s: f32,
        friction_mu: f32,
        extra_fx: *const f32,
        extra_fy: *const f32,
        extra_fz: *const f32,
        stream: *mut c_void,
    ) -> i32;
    pub fn sim_mpm_reduce_energy(
        pts: MpmParticlesGpu,
        d_mats: *const MaterialParamsGpu,
        d_out: *mut MpmMetricsGpu,
        stream: *mut c_void,
    ) -> i32;
    pub fn sim_ib_splat_level(pts: MpmParticlesGpu, lvl: FluidLevelGpu, stream: *mut c_void) -> i32;
    pub fn sim_ib_bake_static_level(pts: MpmParticlesGpu, lvl: FluidLevelGpu, stream: *mut c_void) -> i32;
    pub fn sim_ib_gather_to_solid(
        pts: MpmParticlesGpu,
        d_mats: *const MaterialParamsGpu,
        d_levels: *const FluidLevelGpu,
        num_levels: i32,
        c: FluidConstantsGpu,
        dt_ac: f32,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_shell_forces(
        shell: ShellMeshGpu,
        pts: MpmParticlesGpu,
        d_mats: *const MaterialParamsGpu,
        dt_s: f32,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_capillary_contact(
        pts: MpmParticlesGpu,
        shell: ShellMeshGpu,
        d_mats: *const MaterialParamsGpu,
        hash_head: *mut i32,
        hash_next: *mut i32,
        hash_size: i32,
        cell_size: f32,
        dt_s: f32,
        sigma_w: f32,
        mu_w: f32,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_shell_integrate(
        pts: MpmParticlesGpu,
        shell: ShellMeshGpu,
        d_mats: *const MaterialParamsGpu,
        gx: f32,
        gy: f32,
        gz: f32,
        dt_s: f32,
        has_floor: i32,
        floor_z: f32,
        floor_mu: f32,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_phase_field_step(
        pts: MpmParticlesGpu,
        grid: MpmGridGpu,
        d_mats: *const MaterialParamsGpu,
        dt_pf: f32,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_shell_substep_fused(
        shell: ShellMeshGpu,
        pts: MpmParticlesGpu,
        d_mats: *const MaterialParamsGpu,
        hash_head: *mut i32,
        hash_next: *mut i32,
        hash_size: i32,
        cell_size: f32,
        gx: f32,
        gy: f32,
        gz: f32,
        dt_s: f32,
        n_mpm: i32,
        do_contact: i32,
        has_floor: i32,
        floor_z: f32,
        floor_mu: f32,
        sigma_w: f32,
        mu_w: f32,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_mpm_set_body_prescribed_vel(
        pts: MpmParticlesGpu,
        target_body_id: i32,
        vx: f32,
        vy: f32,
        vz: f32,
        wx: f32,
        wy: f32,
        wz: f32,
        piv_x: f32,
        piv_y: f32,
        piv_z: f32,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_fluid_apply_inflow_nozzle(
        lvl: FluidLevelGpu,
        cx: f32,
        cy: f32,
        cz: f32,
        radius: f32,
        height: f32,
        vx: f32,
        vy: f32,
        vz: f32,
        relax: f32,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_fluid_spawn_droplet(
        lvl: FluidLevelGpu,
        c: FluidConstantsGpu,
        cx: f32,
        cy: f32,
        cz: f32,
        radius: f32,
        vx: f32,
        vy: f32,
        vz: f32,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_euler_init_ambient(lvl: FluidLevelGpu, c: FluidConstantsGpu, stream: *mut c_void) -> i32;

    pub fn sim_euler_add_blast_region(
        lvl: FluidLevelGpu,
        c: FluidConstantsGpu,
        cx: f32,
        cy: f32,
        cz: f32,
        radius: f32,
        length: f32,
        is_cylinder_x: i32,
        overpressure_pa: f32,
        density_kg_m3: f32,
        vx: f32,
        vy: f32,
        vz: f32,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_euler_outer_bc(lvl: FluidLevelGpu, stage: i32, stream: *mut c_void) -> i32;

    pub fn sim_euler_rk_stage(
        lvl: FluidLevelGpu,
        c: FluidConstantsGpu,
        stage: i32,
        dt: f32,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_euler_apply_projectile(
        lvl: FluidLevelGpu,
        c: FluidConstantsGpu,
        base_x_old: f32,
        center_y_old: f32,
        center_z_old: f32,
        base_x_new: f32,
        center_y_new: f32,
        center_z_new: f32,
        vx: f32,
        vy: f32,
        vz: f32,
        radius: f32,
        length: f32,
        dt: f32,
        d_force_out: *mut f64,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_euler_apply_directional_source(
        lvl: FluidLevelGpu,
        c: FluidConstantsGpu,
        cx: f32,
        cy: f32,
        cz: f32,
        dir_x: f32,
        dir_y: f32,
        dir_z: f32,
        cardioid_beta: f32,
        radius: f32,
        rate_pa_per_s: f32,
        dt_ac: f32,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_euler_apply_wind_inlet(
        lvl: FluidLevelGpu,
        c: FluidConstantsGpu,
        min_x: f32,
        min_y: f32,
        min_z: f32,
        max_x: f32,
        max_y: f32,
        max_z: f32,
        vx: f32,
        vy: f32,
        vz: f32,
        turb_amp: f32,
        t_now: f32,
        relax: f32,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_euler_amr_fill_ghosts(
        child: FluidLevelGpu,
        parent: FluidLevelGpu,
        target_stage: i32,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_euler_amr_restrict(
        child: FluidLevelGpu,
        parent: FluidLevelGpu,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_euler_shift_x(
        patch: FluidLevelGpu,
        parent: FluidLevelGpu,
        shift_cells: i32,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_euler_copy_overlap(
        dst_lvl: FluidLevelGpu,
        src_lvl: FluidLevelGpu,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_euler_ib_splat_level(
        pts: MpmParticlesGpu,
        lvl: FluidLevelGpu,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_euler_ib_gather_to_solid(
        pts: MpmParticlesGpu,
        d_mats: *const MaterialParamsGpu,
        d_levels: *const FluidLevelGpu,
        num_levels: i32,
        c: FluidConstantsGpu,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_euler_sample_probes(
        d_levels: *const FluidLevelGpu,
        num_levels: i32,
        d_probes: *const MicProbeGpu,
        num_probes: i32,
        d_out_samples: *mut f32,
        sample_idx: i32,
        c: FluidConstantsGpu,
        stream: *mut c_void,
    ) -> i32;

    pub fn sim_euler_reduce_metrics(
        lvl: FluidLevelGpu,
        c: FluidConstantsGpu,
        d_out: *mut SimReduceMetricsGpu,
        stream: *mut c_void,
    ) -> i32;
}
