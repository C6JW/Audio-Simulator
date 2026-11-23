#include "common.cuh"
#include <cooperative_groups.h>

// Kernel forward declarations
extern __global__ void k_init_hydrostatic_column(FluidLevelGpu lvl, FluidConstantsGpu c, double p_top_ref);
extern __global__ void k_init_hydrostatic_child(FluidLevelGpu child, FluidLevelGpu parent, FluidConstantsGpu c);
extern __global__ void k_save_old_state(FluidLevelGpu lvl);
extern __global__ void k_outer_bc_all(FluidLevelGpu lvl);
extern __global__ void k_outer_bc_slab_x(FluidLevelGpu lvl);
extern __global__ void k_outer_bc_slab_y(FluidLevelGpu lvl);
extern __global__ void k_outer_bc_slab_z(FluidLevelGpu lvl);
extern __global__ void k_acoustic_velocity(FluidLevelGpu lvl, FluidConstantsGpu c, float dt_ac);
extern __global__ void k_acoustic_pressure(FluidLevelGpu lvl, FluidConstantsGpu c, float dt_ac);

extern __global__ void k_vof_save_start_and_div(FluidLevelGpu lvl);
extern __global__ void k_vof_flux_x(FluidLevelGpu lvl, FluidConstantsGpu c, float dt_flow);
extern __global__ void k_vof_flux_y(FluidLevelGpu lvl, FluidConstantsGpu c, float dt_flow);
extern __global__ void k_vof_flux_z(FluidLevelGpu lvl, FluidConstantsGpu c, float dt_flow);
extern __global__ void k_vof_update_axis(FluidLevelGpu lvl, float dt_flow, int axis);
extern __global__ void k_vof_copy_tmp_and_bc(FluidLevelGpu lvl);
extern __global__ void k_vof_kapila_finalize(FluidLevelGpu lvl, FluidConstantsGpu c, float dt_flow);
extern __global__ void k_mflux_ghost_bc(FluidLevelGpu lvl);
extern __global__ void k_advect_momentum(FluidLevelGpu lvl, FluidConstantsGpu c);
extern __global__ void k_copy_vflux_to_u(FluidLevelGpu lvl);
extern __global__ void k_binomial_smooth_3d(const float* src, float* dst, int nxp, int nyp, int nzp);
extern __global__ void k_compute_curvature(FluidLevelGpu lvl);
extern __global__ void k_compute_slow_forces(FluidLevelGpu lvl, FluidConstantsGpu c);

extern __global__ void k_amr_reset_fluxes(FluidLevelGpu child);
extern __global__ void k_amr_fill_ghost_cells(FluidLevelGpu child, FluidLevelGpu parent, double theta_time, int fill_alpha);
extern __global__ void k_amr_fill_ghost_cells_slab(FluidLevelGpu child, FluidLevelGpu parent, float t_f, int fill_alpha, int axis);
extern __global__ void k_amr_fill_ghost_faces(FluidLevelGpu child, FluidLevelGpu parent, float theta_time);
extern __global__ void k_amr_fill_ghost_faces_slab(FluidLevelGpu child, FluidLevelGpu parent, float theta_time, int axis);
extern __global__ void k_amr_reflux_pressure(FluidLevelGpu child, FluidLevelGpu parent, FluidConstantsGpu c, float dt_parent);
extern __global__ void k_amr_restrict_cells(FluidLevelGpu child, FluidLevelGpu parent, FluidConstantsGpu c);
extern __global__ void k_amr_restrict_faces(FluidLevelGpu child, FluidLevelGpu parent, FluidConstantsGpu c);

extern __global__ void k_sample_probes(
    const FluidLevelGpu* levels, int num_levels,
    const MicProbeGpu* probes, int num_probes,
    float* out_samples, int sample_idx
);
extern __global__ void k_reset_metrics(SimReduceMetricsGpu* out);
extern __global__ void k_reduce_fluid_metrics(FluidLevelGpu lvl, FluidConstantsGpu c, SimReduceMetricsGpu* out);

extern __global__ void k_mpm_clear_grid(MpmGridGpu grid);
extern __global__ void k_mpm_p2g(
    MpmParticlesGpu pts, MpmGridGpu grid, const MaterialParamsGpu* mats, float dt_s,
    const float* extra_fx, const float* extra_fy, const float* extra_fz
);
extern __global__ void k_mpm_grid_pre_contact(MpmGridGpu grid, float gx, float gy, float gz, float dt_s);
extern __global__ void k_mpm_grid_contact(MpmGridGpu grid, float friction_mu);
extern __global__ void k_mpm_g2p(MpmParticlesGpu pts, MpmGridGpu grid, const MaterialParamsGpu* mats, float dt_s);
extern __global__ void k_mpm_reset_metrics(MpmMetricsGpu* out);
extern __global__ void k_mpm_reduce_energy(MpmParticlesGpu pts, const MaterialParamsGpu* mats, MpmMetricsGpu* out);

extern __global__ void k_ib_clear_fluid_level(FluidLevelGpu lvl);
extern __global__ void k_ib_copy_to_static_level(FluidLevelGpu lvl);
extern __global__ void k_ib_reset_dynamic_particles(MpmParticlesGpu pts, FluidLevelGpu lvl);
extern __global__ void k_ib_splat_solid_to_fluid(MpmParticlesGpu pts, FluidLevelGpu lvl);
extern __global__ void k_ib_normalize_dynamic_particles(MpmParticlesGpu pts, FluidLevelGpu lvl);
extern __global__ void k_ib_unmark_dynamic_particles(MpmParticlesGpu pts, FluidLevelGpu lvl);
extern __global__ void k_ib_normalize_fluid_level(FluidLevelGpu lvl);
extern __global__ void k_ib_gather_fluid_to_solid(
    MpmParticlesGpu pts, const MaterialParamsGpu* mats, const FluidLevelGpu* levels, int num_levels,
    FluidConstantsGpu c, float dt_ac
);

extern __global__ void k_shell_reset_forces_and_normals(ShellMeshGpu shell, MpmParticlesGpu pts);
extern __global__ void k_shell_triangle_membrane(
    ShellMeshGpu shell, MpmParticlesGpu pts, const MaterialParamsGpu* mats, float dt_s
);
extern __global__ void k_shell_hinge_bending(
    ShellMeshGpu shell, MpmParticlesGpu pts, const MaterialParamsGpu* mats, float dt_s
);
extern __global__ void k_shell_normalize_normals(MpmParticlesGpu pts);
extern __global__ void k_contact_build_hash(
    MpmParticlesGpu pts, int32_t* hash_head, int32_t* hash_next, int32_t hash_size, float cell_size
);
extern __global__ void k_capillary_lubrication_contact(
    MpmParticlesGpu pts, ShellMeshGpu shell, const MaterialParamsGpu* mats,
    const int32_t* hash_head, const int32_t* hash_next, int32_t hash_size,
    float cell_size, float dt_s, float sigma_w, float mu_w
);
extern __global__ void k_shell_integrate_particles(
    MpmParticlesGpu pts, ShellMeshGpu shell, const MaterialParamsGpu* mats,
    float gx, float gy, float gz, float dt_s, int32_t has_floor, float floor_z, float floor_mu
);
extern __global__ void k_pf_clear_grid(MpmGridGpu grid);
extern __global__ void k_pf_p2g(MpmParticlesGpu pts, MpmGridGpu grid, const MaterialParamsGpu* mats);
extern __global__ void k_pf_normalize(MpmGridGpu grid);
extern __global__ void k_pf_solve_jacobi(MpmGridGpu grid, const float* d_in, float* d_out, float dt_pf);
extern __global__ void k_pf_g2p(MpmParticlesGpu pts, MpmGridGpu grid, const MaterialParamsGpu* mats);
extern __global__ void k_shell_substep_fused(
    ShellMeshGpu shell, MpmParticlesGpu pts, const MaterialParamsGpu* mats,
    int32_t* hash_head, int32_t* hash_next, int32_t hash_size, float cell_size,
    float gx, float gy, float gz, float dt_s, int32_t n_mpm, int32_t do_contact,
    int32_t has_floor, float floor_z, float floor_mu, float sigma_w, float mu_w
);
extern __global__ void k_mpm_set_body_prescribed_vel(
    MpmParticlesGpu pts, int32_t target_body_id,
    float vx, float vy, float vz, float wx, float wy, float wz,
    float piv_x, float piv_y, float piv_z
);

extern __global__ void k_euler_init_ambient(FluidLevelGpu lvl, FluidConstantsGpu c);
extern __global__ void k_euler_add_blast_region(
    FluidLevelGpu lvl, FluidConstantsGpu c,
    float cx, float cy, float cz, float radius, float length, int32_t is_cylinder_x,
    float overpressure_pa, float density_kg_m3, float vx, float vy, float vz
);
extern __global__ void k_euler_outer_bc_slab_x(FluidLevelGpu lvl, int32_t stage);
extern __global__ void k_euler_outer_bc_slab_y(FluidLevelGpu lvl, int32_t stage);
extern __global__ void k_euler_outer_bc_slab_z(FluidLevelGpu lvl, int32_t stage);
extern __global__ void k_euler_rk_stage(FluidLevelGpu lvl, FluidConstantsGpu c, int32_t stage, float dt);
extern __global__ void k_euler_apply_projectile_bbox(
    FluidLevelGpu lvl, FluidConstantsGpu c,
    int i_lo, int j_lo, int k_lo, int bx, int by, int bz,
    float base_x_old, float center_y_old, float center_z_old,
    float base_x_new, float center_y_new, float center_z_new,
    float vx, float vy, float vz, float radius, float length, float dt, double* d_force_out
);
extern __global__ void k_euler_apply_directional_source_bbox(
    FluidLevelGpu lvl, FluidConstantsGpu c,
    int i_lo, int j_lo, int k_lo, int bx, int by, int bz,
    float cx, float cy, float cz, float dir_x, float dir_y, float dir_z,
    float cardioid_beta, float radius, float rate_pa_per_s, float dt_ac
);
extern __global__ void k_euler_apply_wind_inlet_bbox(
    FluidLevelGpu lvl, FluidConstantsGpu c,
    int i_lo, int j_lo, int k_lo, int bx, int by, int bz,
    float vx, float vy, float vz, float turb_amp, float t_now, float relax
);
extern __global__ void k_euler_amr_fill_ghosts(FluidLevelGpu child, FluidLevelGpu parent, int32_t target_stage);
extern __global__ void k_euler_amr_fill_ghosts_slab(FluidLevelGpu child, FluidLevelGpu parent, int32_t target_stage, int32_t axis);
extern __global__ void k_euler_amr_restrict(FluidLevelGpu child, FluidLevelGpu parent);
extern __global__ void k_euler_shift_x_stage1(FluidLevelGpu patch, int32_t shift_cells);
extern __global__ void k_euler_shift_x_stage2(FluidLevelGpu patch, FluidLevelGpu parent, int32_t shift_cells);
extern __global__ void k_euler_copy_overlap(FluidLevelGpu dst_lvl, FluidLevelGpu src_lvl);
extern __global__ void k_euler_ib_reset_dynamic(MpmParticlesGpu pts, FluidLevelGpu lvl);
extern __global__ void k_euler_ib_splat_particles(MpmParticlesGpu pts, FluidLevelGpu lvl);
extern __global__ void k_euler_ib_normalize(MpmParticlesGpu pts, FluidLevelGpu lvl);
extern __global__ void k_euler_ib_gather_to_solid(
    MpmParticlesGpu pts, const MaterialParamsGpu* mats, const FluidLevelGpu* levels,
    int32_t num_levels, FluidConstantsGpu c
);
extern __global__ void k_euler_sample_probes(
    const FluidLevelGpu* levels, int32_t num_levels,
    const MicProbeGpu* probes, int32_t num_probes,
    float* out_samples, int32_t sample_idx, FluidConstantsGpu c
);
extern __global__ void k_euler_reduce_metrics(FluidLevelGpu lvl, FluidConstantsGpu c, SimReduceMetricsGpu* out);

static __global__ void k_add_sphere_laplace(
    FluidLevelGpu lvl, float cx, float cy, float cz, float radius, double delta_p
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= lvl.nxp || j >= lvl.nyp || k >= lvl.nzp) return;

    float x = lvl.ox + ((float)(i - lvl.halo) + 0.5f) * lvl.dx;
    float y = lvl.oy + ((float)(j - lvl.halo) + 0.5f) * lvl.dx;
    float z = lvl.oz + ((float)(k - lvl.halo) + 0.5f) * lvl.dx;
    float dx = x - cx;
    float dy = y - cy;
    float dz = z - cz;
    float r2 = dx * dx + dy * dy + dz * dz;
    if (r2 <= (radius + lvl.dx) * (radius + lvl.dx)) {
        size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
        float a = lvl.alpha[cid];
        if (a > 0.0f && r2 <= radius * radius) {
            lvl.p_gauge[cid] += (float)delta_p * a;
            if (lvl.p_gauge_old) lvl.p_gauge_old[cid] = lvl.p_gauge[cid];
        }
    }
}

static __global__ void k_inject_gaussian_pulse(
    FluidLevelGpu lvl, float cx, float cy, float cz, float sigma_r, double amp_pa
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= lvl.nxp || j >= lvl.nyp || k >= lvl.nzp) return;

    float x = lvl.ox + ((float)(i - lvl.halo) + 0.5f) * lvl.dx;
    float y = lvl.oy + ((float)(j - lvl.halo) + 0.5f) * lvl.dx;
    float z = lvl.oz + ((float)(k - lvl.halo) + 0.5f) * lvl.dx;
    float dx = x - cx;
    float dy = y - cy;
    float dz = z - cz;
    double r2 = (double)(dx * dx + dy * dy + dz * dz);
    double s2 = (double)sigma_r * (double)sigma_r;
    double dp = amp_pa * exp(-0.5 * r2 / s2);
    if (fabs(dp) > 1e-12) {
        size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
        lvl.p_gauge[cid] += (float)dp;
        if (lvl.p_gauge_old) lvl.p_gauge_old[cid] = lvl.p_gauge[cid];
    }
}

static __global__ void k_apply_directional_source_bbox(
    FluidLevelGpu lvl,
    FluidConstantsGpu c,
    int i_lo, int j_lo, int k_lo,
    int bx, int by, int bz,
    float cx, float cy, float cz,
    float dir_x, float dir_y, float dir_z,
    float cardioid_beta,
    float radius,
    float rate_pa_per_s,
    float dt_ac,
    int clear_prev
) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    int tj = blockIdx.y * blockDim.y + threadIdx.y;
    int tk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ti > bx || tj > by || tk > bz) return;

    int h = lvl.halo;
    int i = i_lo + ti;
    int j = j_lo + tj;
    int k = k_lo + tk;
    float s2 = radius * radius;
    float cut2 = 9.0f * s2;

    if (ti < bx && tj < by && tk < bz && lvl.src_p != nullptr) {
        float x = lvl.ox + ((float)(i - h) + 0.5f) * lvl.dx;
        float y = lvl.oy + ((float)(j - h) + 0.5f) * lvl.dx;
        float z = lvl.oz + ((float)(k - h) + 0.5f) * lvl.dx;
        float r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz);
        size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
        float val = (r2 < cut2) ? (rate_pa_per_s * expf(-0.5f * r2 / s2)) : 0.0f;
        lvl.src_p[cid] = clear_prev ? val : (lvl.src_p[cid] + val);
    }

    if (fabsf(cardioid_beta) > 1e-6f && fabsf(rate_pa_per_s) > 0.0f && dt_ac > 0.0f) {
        float rho_a = (float)c.rho_a;
        float c_a = sqrtf((float)(c.gamma_a * c.p0 / c.rho_a));
        float vel_scale = dt_ac * cardioid_beta * rate_pa_per_s / (rho_a * c_a);

        if (fabsf(dir_x) > 1e-6f && i <= h + lvl.nx && tj < by && tk < bz) {
            float x = lvl.ox + (float)(i - h) * lvl.dx;
            float y = lvl.oy + ((float)(j - h) + 0.5f) * lvl.dx;
            float z = lvl.oz + ((float)(k - h) + 0.5f) * lvl.dx;
            float r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz);
            if (r2 < cut2) {
                size_t fx = idx_fx(i, j, k, lvl.nxp, lvl.nyp);
                float w = expf(-0.5f * r2 / s2);
                lvl.ux[fx] += (1.0f - lvl.ib_beta_x[fx]) * vel_scale * dir_x * w;
            }
        }
        if (fabsf(dir_y) > 1e-6f && ti < bx && j <= h + lvl.ny && tk < bz) {
            float x = lvl.ox + ((float)(i - h) + 0.5f) * lvl.dx;
            float y = lvl.oy + (float)(j - h) * lvl.dx;
            float z = lvl.oz + ((float)(k - h) + 0.5f) * lvl.dx;
            float r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz);
            if (r2 < cut2) {
                size_t fy = idx_fy(i, j, k, lvl.nxp, lvl.nyp);
                float w = expf(-0.5f * r2 / s2);
                lvl.uy[fy] += (1.0f - lvl.ib_beta_y[fy]) * vel_scale * dir_y * w;
            }
        }
        if (fabsf(dir_z) > 1e-6f && ti < bx && tj < by && k <= h + lvl.nz) {
            float x = lvl.ox + ((float)(i - h) + 0.5f) * lvl.dx;
            float y = lvl.oy + ((float)(j - h) + 0.5f) * lvl.dx;
            float z = lvl.oz + (float)(k - h) * lvl.dx;
            float r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz);
            if (r2 < cut2) {
                size_t fz = idx_fz(i, j, k, lvl.nxp, lvl.nyp);
                float w = expf(-0.5f * r2 / s2);
                lvl.uz[fz] += (1.0f - lvl.ib_beta_z[fz]) * vel_scale * dir_z * w;
            }
        }
    }
}

static __global__ void k_apply_point_source_ricker_bbox(
    FluidLevelGpu lvl,
    int i_lo, int j_lo, int k_lo, int bx, int by, int bz,
    float cx, float cy, float cz, float radius, float rate_pa_per_s
) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    int tj = blockIdx.y * blockDim.y + threadIdx.y;
    int tk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ti >= bx || tj >= by || tk >= bz || lvl.src_p == nullptr) return;

    int h = lvl.halo;
    int i = i_lo + ti;
    int j = j_lo + tj;
    int k = k_lo + tk;
    float x = lvl.ox + ((float)(i - h) + 0.5f) * lvl.dx;
    float y = lvl.oy + ((float)(j - h) + 0.5f) * lvl.dx;
    float z = lvl.oz + ((float)(k - h) + 0.5f) * lvl.dx;
    float dx = x - cx;
    float dy = y - cy;
    float dz = z - cz;
    float r2 = dx * dx + dy * dy + dz * dz;
    float s2 = radius * radius;
    if (r2 < 9.0f * s2) {
        size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
        lvl.src_p[cid] = rate_pa_per_s * expf(-0.5f * r2 / s2);
    }
}

static inline dim3 grid3(int nx, int ny, int nz, dim3 b) {
    return dim3(
        (unsigned)((nx + b.x - 1) / b.x),
        (unsigned)((ny + b.y - 1) / b.y),
        (unsigned)((nz + b.z - 1) / b.z)
    );
}

static __global__ void k_fluid_apply_inflow_nozzle_bbox(
    FluidLevelGpu lvl,
    int i_lo, int j_lo, int k_lo, int bx, int by, int bz,
    float cx, float cy, float cz,
    float radius, float half_height,
    float vx, float vy, float vz,
    float relax
) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    int tj = blockIdx.y * blockDim.y + threadIdx.y;
    int tk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ti > bx || tj > by || tk > bz) return;

    int h = lvl.halo;
    int i = i_lo + ti;
    int j = j_lo + tj;
    int k = k_lo + tk;
    float r_lim_a = radius + 0.5f * lvl.dx;
    float z_lim_a = half_height + 0.5f * lvl.dx;
    float r_lim_u = radius + 1.05f * lvl.dx;
    float z_lim_u = half_height + 1.05f * lvl.dx;

    if (ti < bx && tj < by && tk < bz && lvl.alpha != nullptr) {
        float x = lvl.ox + ((float)(i - h) + 0.5f) * lvl.dx;
        float y = lvl.oy + ((float)(j - h) + 0.5f) * lvl.dx;
        float z = lvl.oz + ((float)(k - h) + 0.5f) * lvl.dx;
        float r_xy = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        if (r_xy <= r_lim_a && fabsf(z - cz) <= z_lim_a) {
            float w_r = clamp_f(0.5f + (radius - r_xy) / lvl.dx, 0.0f, 1.0f);
            float w_z = clamp_f(0.5f + (half_height - fabsf(z - cz)) / lvl.dx, 0.0f, 1.0f);
            float w = w_r * w_z;
            if (w > 0.0f) {
                size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
                lvl.alpha[cid] = fmaxf(lvl.alpha[cid], w * clamp_f(relax * 1.25f, 0.0f, 1.0f));
            }
        }
    }

    if (i <= h + lvl.nx && tj < by && tk < bz && lvl.ux != nullptr) {
        float x = lvl.ox + (float)(i - h) * lvl.dx;
        float y = lvl.oy + ((float)(j - h) + 0.5f) * lvl.dx;
        float z = lvl.oz + ((float)(k - h) + 0.5f) * lvl.dx;
        float r_xy = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        if (r_xy <= r_lim_u && fabsf(z - cz) <= z_lim_u) {
            size_t fx = idx_fx(i, j, k, lvl.nxp, lvl.nyp);
            if (lvl.ib_beta_x[fx] < 0.5f) {
                lvl.ux[fx] += relax * (vx - lvl.ux[fx]);
            }
        }
    }
    if (ti < bx && j <= h + lvl.ny && tk < bz && lvl.uy != nullptr) {
        float x = lvl.ox + ((float)(i - h) + 0.5f) * lvl.dx;
        float y = lvl.oy + (float)(j - h) * lvl.dx;
        float z = lvl.oz + ((float)(k - h) + 0.5f) * lvl.dx;
        float r_xy = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        if (r_xy <= r_lim_u && fabsf(z - cz) <= z_lim_u) {
            size_t fy = idx_fy(i, j, k, lvl.nxp, lvl.nyp);
            if (lvl.ib_beta_y[fy] < 0.5f) {
                lvl.uy[fy] += relax * (vy - lvl.uy[fy]);
            }
        }
    }
    if (ti < bx && tj < by && k <= h + lvl.nz && lvl.uz != nullptr) {
        float x = lvl.ox + ((float)(i - h) + 0.5f) * lvl.dx;
        float y = lvl.oy + ((float)(j - h) + 0.5f) * lvl.dx;
        float z = lvl.oz + (float)(k - h) * lvl.dx;
        float r_xy = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        if (r_xy <= r_lim_u && fabsf(z - cz) <= z_lim_u) {
            size_t fz = idx_fz(i, j, k, lvl.nxp, lvl.nyp);
            if (lvl.ib_beta_z[fz] < 0.5f) {
                lvl.uz[fz] += relax * (vz - lvl.uz[fz]);
            }
        }
    }
}

static __global__ void k_fluid_spawn_droplet_bbox(
    FluidLevelGpu lvl, FluidConstantsGpu c,
    int i_lo, int j_lo, int k_lo, int bx, int by, int bz,
    float cx, float cy, float cz, float radius,
    float vx, float vy, float vz
) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    int tj = blockIdx.y * blockDim.y + threadIdx.y;
    int tk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ti > bx || tj > by || tk > bz) return;

    int h = lvl.halo;
    int i = i_lo + ti;
    int j = j_lo + tj;
    int k = k_lo + tk;
    float r_lim_a = radius + 0.5f * lvl.dx;
    float r_u = radius + 1.05f * lvl.dx;
    float r_air = radius + 2.05f * lvl.dx;
    double laplace_dp = 2.0 * c.sigma / fmax(1e-4, (double)radius);

    if (ti < bx && tj < by && tk < bz && lvl.alpha != nullptr) {
        float x = lvl.ox + ((float)(i - h) + 0.5f) * lvl.dx;
        float y = lvl.oy + ((float)(j - h) + 0.5f) * lvl.dx;
        float z = lvl.oz + ((float)(k - h) + 0.5f) * lvl.dx;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
        if (dist <= r_lim_a) {
            float w = clamp_f(0.5f + (radius - dist) / lvl.dx, 0.0f, 1.0f);
            size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
            float a_old = lvl.alpha[cid];
            float da = fmaxf(0.0f, w - a_old);
            if (da > 0.0f) {
                lvl.alpha[cid] = fminf(1.0f, a_old + da);
                if (lvl.p_gauge != nullptr) {
                    lvl.p_gauge[cid] += (float)laplace_dp * da;
                    if (lvl.p_gauge_old != nullptr) lvl.p_gauge_old[cid] = lvl.p_gauge[cid];
                }
            }
        }
    }

    if (i <= h + lvl.nx && tj < by && tk < bz && lvl.ux != nullptr) {
        float x = lvl.ox + (float)(i - h) * lvl.dx;
        float y = lvl.oy + ((float)(j - h) + 0.5f) * lvl.dx;
        float z = lvl.oz + ((float)(k - h) + 0.5f) * lvl.dx;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
        if (dist <= r_u) {
            size_t fx = idx_fx(i, j, k, lvl.nxp, lvl.nyp);
            if (lvl.ib_beta_x[fx] < 0.5f) lvl.ux[fx] = vx;
        } else if (dist <= r_air) {
            float w_air = clamp_f((r_air - dist) / lvl.dx, 0.0f, 1.0f);
            size_t fx = idx_fx(i, j, k, lvl.nxp, lvl.nyp);
            if (lvl.ib_beta_x[fx] < 0.5f) lvl.ux[fx] += w_air * (vx - lvl.ux[fx]);
        }
    }
    if (ti < bx && j <= h + lvl.ny && tk < bz && lvl.uy != nullptr) {
        float x = lvl.ox + ((float)(i - h) + 0.5f) * lvl.dx;
        float y = lvl.oy + (float)(j - h) * lvl.dx;
        float z = lvl.oz + ((float)(k - h) + 0.5f) * lvl.dx;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
        if (dist <= r_u) {
            size_t fy = idx_fy(i, j, k, lvl.nxp, lvl.nyp);
            if (lvl.ib_beta_y[fy] < 0.5f) lvl.uy[fy] = vy;
        } else if (dist <= r_air) {
            float w_air = clamp_f((r_air - dist) / lvl.dx, 0.0f, 1.0f);
            size_t fy = idx_fy(i, j, k, lvl.nxp, lvl.nyp);
            if (lvl.ib_beta_y[fy] < 0.5f) lvl.uy[fy] += w_air * (vy - lvl.uy[fy]);
        }
    }
    if (ti < bx && tj < by && k <= h + lvl.nz && lvl.uz != nullptr) {
        float x = lvl.ox + ((float)(i - h) + 0.5f) * lvl.dx;
        float y = lvl.oy + ((float)(j - h) + 0.5f) * lvl.dx;
        float z = lvl.oz + (float)(k - h) * lvl.dx;
        float dist = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
        if (dist <= r_u) {
            size_t fz = idx_fz(i, j, k, lvl.nxp, lvl.nyp);
            if (lvl.ib_beta_z[fz] < 0.5f) lvl.uz[fz] = vz;
        } else if (dist <= r_air) {
            float w_air = clamp_f((r_air - dist) / lvl.dx, 0.0f, 1.0f);
            size_t fz = idx_fz(i, j, k, lvl.nxp, lvl.nyp);
            if (lvl.ib_beta_z[fz] < 0.5f) lvl.uz[fz] += w_air * (vz - lvl.uz[fz]);
        }
    }
}

extern "C" {

int32_t sim_cuda_init_device(int32_t device_id, int32_t* cc_major, int32_t* cc_minor, size_t* total_vram) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return (int32_t)err;
    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) return (int32_t)err;
    if (cc_major) *cc_major = prop.major;
    if (cc_minor) *cc_minor = prop.minor;
    if (total_vram) *total_vram = prop.totalGlobalMem;
    return (int32_t)cudaFree(0); // Force runtime context initialization
}

int32_t sim_cuda_malloc(void** ptr, size_t bytes) {
    return (int32_t)cudaMalloc(ptr, bytes);
}

int32_t sim_cuda_free(void* ptr) {
    return (int32_t)cudaFree(ptr);
}

int32_t sim_cuda_memset(void* ptr, int32_t val, size_t bytes) {
    return (int32_t)cudaMemset(ptr, val, bytes);
}

int32_t sim_cuda_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    return (int32_t)cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

int32_t sim_cuda_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    return (int32_t)cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

int32_t sim_cuda_memcpy_d2d(void* dst, const void* src, size_t bytes) {
    return (int32_t)cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
}

int32_t sim_cuda_device_sync() {
    return (int32_t)cudaDeviceSynchronize();
}

int32_t sim_cuda_get_last_error() {
    return (int32_t)cudaGetLastError();
}

int32_t sim_cuda_stream_create(void** stream_out) {
    cudaStream_t s = nullptr;
    cudaError_t err = cudaStreamCreate(&s);
    *stream_out = (void*)s;
    return (int32_t)err;
}

int32_t sim_cuda_stream_destroy(void* stream) {
    return (int32_t)cudaStreamDestroy((cudaStream_t)stream);
}

int32_t sim_cuda_stream_sync(void* stream) {
    return (int32_t)cudaStreamSynchronize((cudaStream_t)stream);
}

int32_t sim_cuda_graph_begin_capture(void* stream) {
    return (int32_t)cudaStreamBeginCapture((cudaStream_t)stream, cudaStreamCaptureModeGlobal);
}

int32_t sim_cuda_graph_end_capture(void* stream, void** graph_exec_out) {
    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamEndCapture((cudaStream_t)stream, &graph);
    if (err != cudaSuccess) return (int32_t)err;
    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    *graph_exec_out = (void*)exec;
    return (int32_t)err;
}

int32_t sim_cuda_graph_launch(void* graph_exec, void* stream) {
    return (int32_t)cudaGraphLaunch((cudaGraphExec_t)graph_exec, (cudaStream_t)stream);
}

int32_t sim_cuda_graph_destroy(void* graph_exec) {
    return (int32_t)cudaGraphExecDestroy((cudaGraphExec_t)graph_exec);
}

int32_t sim_fluid_init_hydrostatic_root(FluidLevelGpu lvl, FluidConstantsGpu c, double p_top_ref, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    dim3 b2(16, 16, 1);
    dim3 g2((unsigned)((lvl.nxp + 15) / 16), (unsigned)((lvl.nyp + 15) / 16), 1);
    k_init_hydrostatic_column<<<g2, b2, 0, s>>>(lvl, c, p_top_ref);
    return (int32_t)cudaGetLastError();
}

int32_t sim_fluid_init_hydrostatic_child(FluidLevelGpu child, FluidLevelGpu parent, FluidConstantsGpu c, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    dim3 b2(16, 16, 1);
    dim3 g2((unsigned)((child.nxp + 15) / 16), (unsigned)((child.nyp + 15) / 16), 1);
    k_init_hydrostatic_child<<<g2, b2, 0, s>>>(child, parent, c);
    return (int32_t)cudaGetLastError();
}

int32_t sim_fluid_add_sphere_laplace(
    FluidLevelGpu lvl, float cx, float cy, float cz, float radius, double delta_p, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    dim3 b(8, 8, 4);
    k_add_sphere_laplace<<<grid3(lvl.nxp, lvl.nyp, lvl.nzp, b), b, 0, s>>>(lvl, cx, cy, cz, radius, delta_p);
    return (int32_t)cudaGetLastError();
}

int32_t sim_fluid_inject_gaussian_pulse(
    FluidLevelGpu lvl, float cx, float cy, float cz, float sigma_r, double amp_pa, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    dim3 b(8, 8, 4);
    k_inject_gaussian_pulse<<<grid3(lvl.nxp, lvl.nyp, lvl.nzp, b), b, 0, s>>>(lvl, cx, cy, cz, sigma_r, amp_pa);
    return (int32_t)cudaGetLastError();
}

int32_t sim_fluid_set_point_source_ricker(
    FluidLevelGpu lvl, float cx, float cy, float cz, float radius, float rate_pa_per_s, void* stream
) {
    if (lvl.src_p == nullptr) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    size_t n_cells = (size_t)lvl.nxp * (size_t)lvl.nyp * (size_t)lvl.nzp;
    cudaError_t err = cudaMemsetAsync(lvl.src_p, 0, n_cells * sizeof(float), s);
    if (err != cudaSuccess) return (int32_t)err;
    if (fabsf(rate_pa_per_s) <= 0.0f) return 0;

    float span = 3.5f * radius + lvl.dx;
    int h = lvl.halo;
    int i_lo = max(h, (int)floorf((cx - span - lvl.ox) / lvl.dx) + h);
    int i_hi = min(h + lvl.nx - 1, (int)ceilf((cx + span - lvl.ox) / lvl.dx) + h);
    int j_lo = max(h, (int)floorf((cy - span - lvl.oy) / lvl.dx) + h);
    int j_hi = min(h + lvl.ny - 1, (int)ceilf((cy + span - lvl.oy) / lvl.dx) + h);
    int k_lo = max(h, (int)floorf((cz - span - lvl.oz) / lvl.dx) + h);
    int k_hi = min(h + lvl.nz - 1, (int)ceilf((cz + span - lvl.oz) / lvl.dx) + h);
    if (i_lo > i_hi || j_lo > j_hi || k_lo > k_hi) return 0;

    int bx = i_hi - i_lo + 1;
    int by = j_hi - j_lo + 1;
    int bz = k_hi - k_lo + 1;
    dim3 b(8, 8, 4);
    k_apply_point_source_ricker_bbox<<<grid3(bx, by, bz, b), b, 0, s>>>(
        lvl, i_lo, j_lo, k_lo, bx, by, bz, cx, cy, cz, radius, rate_pa_per_s
    );
    return (int32_t)cudaGetLastError();
}

int32_t sim_fluid_apply_directional_source_bbox(
    FluidLevelGpu lvl,
    FluidConstantsGpu c,
    float cx, float cy, float cz,
    float dir_x, float dir_y, float dir_z,
    float cardioid_beta,
    float radius,
    float rate_pa_per_s,
    float dt_ac,
    int32_t clear_prev,
    void* stream
) {
    float span = 3.5f * radius + lvl.dx;
    int h = lvl.halo;
    int i_lo = max(h, (int)floorf((cx - span - lvl.ox) / lvl.dx) + h);
    int i_hi = min(h + lvl.nx - 1, (int)ceilf((cx + span - lvl.ox) / lvl.dx) + h);
    int j_lo = max(h, (int)floorf((cy - span - lvl.oy) / lvl.dx) + h);
    int j_hi = min(h + lvl.ny - 1, (int)ceilf((cy + span - lvl.oy) / lvl.dx) + h);
    int k_lo = max(h, (int)floorf((cz - span - lvl.oz) / lvl.dx) + h);
    int k_hi = min(h + lvl.nz - 1, (int)ceilf((cz + span - lvl.oz) / lvl.dx) + h);
    if (i_lo > i_hi || j_lo > j_hi || k_lo > k_hi) return 0;

    int bx = i_hi - i_lo + 1;
    int by = j_hi - j_lo + 1;
    int bz = k_hi - k_lo + 1;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 b(8, 8, 4);
    k_apply_directional_source_bbox<<<grid3(bx + 1, by + 1, bz + 1, b), b, 0, s>>>(
        lvl, c, i_lo, j_lo, k_lo, bx, by, bz,
        cx, cy, cz, dir_x, dir_y, dir_z, cardioid_beta,
        radius, rate_pa_per_s, dt_ac, clear_prev
    );
    return (int32_t)cudaGetLastError();
}

static inline void launch_outer_bc_slabs(FluidLevelGpu lvl, cudaStream_t s) {
    if (lvl.level_idx != 0) return;
    int h2 = 2 * lvl.halo;
    dim3 bx(2, 16, 8);
    k_outer_bc_slab_x<<<grid3(h2, lvl.nyp, lvl.nzp, bx), bx, 0, s>>>(lvl);
    dim3 by(16, 2, 8);
    k_outer_bc_slab_y<<<grid3(lvl.nxp, h2, lvl.nzp, by), by, 0, s>>>(lvl);
    dim3 bz(16, 8, 2);
    k_outer_bc_slab_z<<<grid3(lvl.nxp, lvl.nyp, h2, bz), bz, 0, s>>>(lvl);
}

int32_t sim_fluid_save_old(FluidLevelGpu lvl, void* stream) {
    if (lvl.p_gauge_old == nullptr && lvl.ux_old == nullptr) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 b(8, 8, 4);
    k_save_old_state<<<grid3(lvl.nxp + 1, lvl.nyp + 1, lvl.nzp + 1, b), b, 0, s>>>(lvl);
    return (int32_t)cudaGetLastError();
}

int32_t sim_fluid_apply_outer_bc(FluidLevelGpu lvl, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    launch_outer_bc_slabs(lvl, s);
    return (int32_t)cudaGetLastError();
}

int32_t sim_fluid_acoustic_step(FluidLevelGpu lvl, FluidConstantsGpu c, float dt_ac, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    dim3 b(8, 8, 4);
    k_acoustic_velocity<<<grid3(lvl.nx + 1, lvl.ny + 1, lvl.nz + 1, b), b, 0, s>>>(lvl, c, dt_ac);
    k_acoustic_pressure<<<grid3(lvl.nx, lvl.ny, lvl.nz, b), b, 0, s>>>(lvl, c, dt_ac);
    if (lvl.level_idx == 0) {
        launch_outer_bc_slabs(lvl, s);
    }
    return (int32_t)cudaGetLastError();
}

int32_t sim_fluid_flow_step(
    FluidLevelGpu lvl, FluidConstantsGpu c, float dt_flow, int32_t sweep_parity, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    dim3 b(8, 8, 4);

    k_vof_save_start_and_div<<<grid3(lvl.nxp, lvl.nyp, lvl.nzp, b), b, 0, s>>>(lvl);

    int order[3];
    if ((sweep_parity & 1) == 0) {
        order[0] = 0; order[1] = 1; order[2] = 2;
    } else {
        order[0] = 2; order[1] = 1; order[2] = 0;
    }

    for (int step = 0; step < 3; ++step) {
        int ax = order[step];
        if (ax == 0) {
            k_vof_flux_x<<<grid3(lvl.nx + 1, lvl.ny, lvl.nz, b), b, 0, s>>>(lvl, c, dt_flow);
        } else if (ax == 1) {
            k_vof_flux_y<<<grid3(lvl.nx, lvl.ny + 1, lvl.nz, b), b, 0, s>>>(lvl, c, dt_flow);
        } else {
            k_vof_flux_z<<<grid3(lvl.nx, lvl.ny, lvl.nz + 1, b), b, 0, s>>>(lvl, c, dt_flow);
        }
        k_vof_update_axis<<<grid3(lvl.nx, lvl.ny, lvl.nz, b), b, 0, s>>>(lvl, dt_flow, ax);
        k_vof_copy_tmp_and_bc<<<grid3(lvl.nxp, lvl.nyp, lvl.nzp, b), b, 0, s>>>(lvl);
    }

    // Kapila mechanical pressure-equilibrium volume-fraction relaxation (compressible bubble breathing)
    k_vof_kapila_finalize<<<grid3(lvl.nx, lvl.ny, lvl.nz, b), b, 0, s>>>(lvl, c, dt_flow);

    k_mflux_ghost_bc<<<grid3(lvl.nxp + 1, lvl.nyp + 1, lvl.nzp + 1, b), b, 0, s>>>(lvl);
    k_advect_momentum<<<grid3(lvl.nx + 1, lvl.ny + 1, lvl.nz + 1, b), b, 0, s>>>(lvl, c);
    k_copy_vflux_to_u<<<grid3(lvl.nx + 1, lvl.ny + 1, lvl.nz + 1, b), b, 0, s>>>(lvl);

    if (lvl.level_idx == 0) {
        launch_outer_bc_slabs(lvl, s);
    }

    // 2 passes of 3D binomial smoothing (alpha -> alpha_tmp -> alpha_smooth)
    k_binomial_smooth_3d<<<grid3(lvl.nxp, lvl.nyp, lvl.nzp, b), b, 0, s>>>(
        lvl.alpha, lvl.alpha_tmp, lvl.nxp, lvl.nyp, lvl.nzp
    );
    k_binomial_smooth_3d<<<grid3(lvl.nxp, lvl.nyp, lvl.nzp, b), b, 0, s>>>(
        lvl.alpha_tmp, lvl.alpha_smooth, lvl.nxp, lvl.nyp, lvl.nzp
    );
    k_compute_curvature<<<grid3(lvl.nxp - 2, lvl.nyp - 2, lvl.nzp - 2, b), b, 0, s>>>(lvl);
    k_compute_slow_forces<<<grid3(lvl.nx + 1, lvl.ny + 1, lvl.nz + 1, b), b, 0, s>>>(lvl, c);

    return (int32_t)cudaGetLastError();
}

int32_t sim_fluid_update_slow_forces_only(FluidLevelGpu lvl, FluidConstantsGpu c, void* stream) {
    if (lvl.alpha_tmp == nullptr || lvl.alpha_smooth == nullptr || lvl.kappa == nullptr || lvl.a_slow_x == nullptr) {
        return 0;
    }
    cudaStream_t s = (cudaStream_t)stream;
    dim3 b(8, 8, 4);
    k_binomial_smooth_3d<<<grid3(lvl.nxp, lvl.nyp, lvl.nzp, b), b, 0, s>>>(
        lvl.alpha, lvl.alpha_tmp, lvl.nxp, lvl.nyp, lvl.nzp
    );
    k_binomial_smooth_3d<<<grid3(lvl.nxp, lvl.nyp, lvl.nzp, b), b, 0, s>>>(
        lvl.alpha_tmp, lvl.alpha_smooth, lvl.nxp, lvl.nyp, lvl.nzp
    );
    k_compute_curvature<<<grid3(lvl.nxp - 2, lvl.nyp - 2, lvl.nzp - 2, b), b, 0, s>>>(lvl);
    k_compute_slow_forces<<<grid3(lvl.nx + 1, lvl.ny + 1, lvl.nz + 1, b), b, 0, s>>>(lvl, c);
    return (int32_t)cudaGetLastError();
}

int32_t sim_fluid_amr_reset_fluxes(FluidLevelGpu child, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    int m = max(child.nx, max(child.ny, child.nz));
    dim3 b2(16, 16, 1);
    dim3 g2((unsigned)((m + 15) / 16), (unsigned)((m + 15) / 16), 1);
    k_amr_reset_fluxes<<<g2, b2, 0, s>>>(child);
    return (int32_t)cudaGetLastError();
}

int32_t sim_fluid_amr_fill_ghosts(
    FluidLevelGpu child, FluidLevelGpu parent, double theta_time, int32_t fill_alpha, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int hc = child.halo;
    int h2 = 2 * hc;
    int hf2 = 2 * (hc + 1);
    float tf = (float)theta_time;

    // 3 disjoint boundary slabs for ghost cells (skips interior nx*ny*nz)
    dim3 bcx(2, 16, 8);
    k_amr_fill_ghost_cells_slab<<<grid3(h2, child.nyp, child.nzp, bcx), bcx, 0, s>>>(child, parent, tf, fill_alpha, 0);
    dim3 bcy(16, 2, 8);
    k_amr_fill_ghost_cells_slab<<<grid3(child.nx, h2, child.nzp, bcy), bcy, 0, s>>>(child, parent, tf, fill_alpha, 1);
    dim3 bcz(16, 8, 2);
    k_amr_fill_ghost_cells_slab<<<grid3(child.nx, child.ny, h2, bcz), bcz, 0, s>>>(child, parent, tf, fill_alpha, 2);

    // 3 disjoint boundary slabs for ghost faces (includes patch-boundary faces)
    k_amr_fill_ghost_faces_slab<<<grid3(hf2, child.nyp + 1, child.nzp + 1, bcx), bcx, 0, s>>>(child, parent, tf, 0);
    k_amr_fill_ghost_faces_slab<<<grid3(child.nx, hf2, child.nzp + 1, bcy), bcy, 0, s>>>(child, parent, tf, 1);
    k_amr_fill_ghost_faces_slab<<<grid3(child.nx, child.ny, hf2, bcz), bcz, 0, s>>>(child, parent, tf, 2);

    return (int32_t)cudaGetLastError();
}

int32_t sim_fluid_amr_reflux_and_restrict(
    FluidLevelGpu child, FluidLevelGpu parent, FluidConstantsGpu c, float dt_parent, void* stream
) {
    (void)dt_parent;
    cudaStream_t s = (cudaStream_t)stream;
    int r = child.refine_ratio;
    int nx_c = child.nx / r;
    int ny_c = child.ny / r;
    int nz_c = child.nz / r;

    dim3 b(8, 8, 4);
    k_amr_restrict_cells<<<grid3(nx_c, ny_c, nz_c, b), b, 0, s>>>(child, parent, c);
    k_amr_restrict_faces<<<grid3(nx_c + 1, ny_c + 1, nz_c + 1, b), b, 0, s>>>(child, parent, c);
    return (int32_t)cudaGetLastError();
}

int32_t sim_fluid_amr_restrict_only(
    FluidLevelGpu child, FluidLevelGpu parent, FluidConstantsGpu c, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int r = child.refine_ratio;
    int nx_c = child.nx / r;
    int ny_c = child.ny / r;
    int nz_c = child.nz / r;
    dim3 b(8, 8, 4);
    k_amr_restrict_cells<<<grid3(nx_c, ny_c, nz_c, b), b, 0, s>>>(child, parent, c);
    k_amr_restrict_faces<<<grid3(nx_c + 1, ny_c + 1, nz_c + 1, b), b, 0, s>>>(child, parent, c);
    return (int32_t)cudaGetLastError();
}

int32_t sim_fluid_sample_probes(
    const FluidLevelGpu* d_levels, int32_t num_levels,
    const MicProbeGpu* d_probes, int32_t num_probes,
    float* d_out_samples, int32_t sample_idx, void* stream
) {
    if (num_probes <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    int threads = 64;
    int blocks = (num_probes + threads - 1) / threads;
    k_sample_probes<<<blocks, threads, 0, s>>>(
        d_levels, num_levels, d_probes, num_probes, d_out_samples, sample_idx
    );
    return (int32_t)cudaGetLastError();
}

int32_t sim_fluid_reduce_metrics(
    FluidLevelGpu lvl, FluidConstantsGpu c, SimReduceMetricsGpu* d_out, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    k_reset_metrics<<<1, 1, 0, s>>>(d_out);
    k_reduce_fluid_metrics<<<128, 256, 0, s>>>(lvl, c, d_out);
    return (int32_t)cudaGetLastError();
}

int32_t sim_mpm_step(
    MpmParticlesGpu pts,
    MpmGridGpu grid,
    const MaterialParamsGpu* d_mats,
    float gx, float gy, float gz,
    float dt_s,
    float friction_mu,
    const float* extra_fx,
    const float* extra_fy,
    const float* extra_fz,
    void* stream
) {
    if (pts.count <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    size_t n_nodes = (size_t)grid.nx * (size_t)grid.ny * (size_t)grid.nz;
    size_t total_g = 2 * n_nodes;
    int t = 256;
    k_mpm_clear_grid<<<(unsigned)((total_g + t - 1) / t), t, 0, s>>>(grid);
    k_mpm_p2g<<<(unsigned)((pts.count + t - 1) / t), t, 0, s>>>(
        pts, grid, d_mats, dt_s, extra_fx, extra_fy, extra_fz
    );
    k_mpm_grid_pre_contact<<<(unsigned)((total_g + t - 1) / t), t, 0, s>>>(grid, gx, gy, gz, dt_s);
    dim3 b(8, 8, 4);
    k_mpm_grid_contact<<<grid3(grid.nx, grid.ny, grid.nz, b), b, 0, s>>>(grid, friction_mu);
    k_mpm_g2p<<<(unsigned)((pts.count + t - 1) / t), t, 0, s>>>(pts, grid, d_mats, dt_s);
    return (int32_t)cudaGetLastError();
}

int32_t sim_mpm_reduce_energy(
    MpmParticlesGpu pts,
    const MaterialParamsGpu* d_mats,
    MpmMetricsGpu* d_out,
    void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    k_mpm_reset_metrics<<<1, 1, 0, s>>>(d_out);
    if (pts.count > 0) {
        k_mpm_reduce_energy<<<64, 256, 0, s>>>(pts, d_mats, d_out);
    }
    return (int32_t)cudaGetLastError();
}

int32_t sim_ib_splat_level(
    MpmParticlesGpu pts,
    FluidLevelGpu lvl,
    void* stream
) {
    if (pts.count <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    int t = 256;
    unsigned blocks = (unsigned)((pts.count + t - 1) / t);
    k_ib_reset_dynamic_particles<<<blocks, t, 0, s>>>(pts, lvl);
    k_ib_splat_solid_to_fluid<<<blocks, t, 0, s>>>(pts, lvl);
    k_ib_normalize_dynamic_particles<<<blocks, t, 0, s>>>(pts, lvl);
    k_ib_unmark_dynamic_particles<<<blocks, t, 0, s>>>(pts, lvl);
    return (int32_t)cudaGetLastError();
}

int32_t sim_ib_bake_static_level(
    MpmParticlesGpu pts,
    FluidLevelGpu lvl,
    void* stream
) {
    if (pts.count <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 b(8, 8, 4);
    k_ib_clear_fluid_level<<<grid3(lvl.nxp + 1, lvl.nyp + 1, lvl.nzp + 1, b), b, 0, s>>>(lvl);
    int t = 256;
    k_ib_splat_solid_to_fluid<<<(unsigned)((pts.count + t - 1) / t), t, 0, s>>>(pts, lvl);
    k_ib_normalize_fluid_level<<<grid3(lvl.nxp + 1, lvl.nyp + 1, lvl.nzp + 1, b), b, 0, s>>>(lvl);
    k_ib_copy_to_static_level<<<grid3(lvl.nxp + 1, lvl.nyp + 1, lvl.nzp + 1, b), b, 0, s>>>(lvl);
    return (int32_t)cudaGetLastError();
}

int32_t sim_ib_gather_to_solid(
    MpmParticlesGpu pts,
    const MaterialParamsGpu* d_mats,
    const FluidLevelGpu* d_levels,
    int32_t num_levels,
    FluidConstantsGpu c,
    float dt_ac,
    void* stream
) {
    if (pts.count <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    int t = 256;
    k_ib_gather_fluid_to_solid<<<(unsigned)((pts.count + t - 1) / t), t, 0, s>>>(
        pts, d_mats, d_levels, num_levels, c, dt_ac
    );
    return (int32_t)cudaGetLastError();
}

int32_t sim_shell_forces(
    ShellMeshGpu shell,
    MpmParticlesGpu pts,
    const MaterialParamsGpu* d_mats,
    float dt_s,
    void* stream
) {
    if (pts.count <= 0 || shell.p_fx == nullptr) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    int t = 256;
    k_shell_reset_forces_and_normals<<<(unsigned)((pts.count + t - 1) / t), t, 0, s>>>(shell, pts);
    if (shell.num_tris > 0) {
        k_shell_triangle_membrane<<<(unsigned)((shell.num_tris + t - 1) / t), t, 0, s>>>(
            shell, pts, d_mats, dt_s
        );
        k_shell_normalize_normals<<<(unsigned)((pts.count + t - 1) / t), t, 0, s>>>(pts);
    }
    if (shell.num_hinges > 0) {
        k_shell_hinge_bending<<<(unsigned)((shell.num_hinges + t - 1) / t), t, 0, s>>>(
            shell, pts, d_mats, dt_s
        );
    }
    return (int32_t)cudaGetLastError();
}

int32_t sim_capillary_contact(
    MpmParticlesGpu pts,
    ShellMeshGpu shell,
    const MaterialParamsGpu* d_mats,
    int32_t* hash_head,
    int32_t* hash_next,
    int32_t hash_size,
    float cell_size,
    float dt_s,
    float sigma_w,
    float mu_w,
    void* stream
) {
    if (pts.count <= 0 || hash_size <= 0 || hash_head == nullptr || shell.p_fx == nullptr) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    cudaError_t err = cudaMemsetAsync(hash_head, 0xFF, (size_t)hash_size * sizeof(int32_t), s);
    if (err != cudaSuccess) return (int32_t)err;
    int t = 256;
    unsigned blocks = (unsigned)((pts.count + t - 1) / t);
    k_contact_build_hash<<<blocks, t, 0, s>>>(pts, hash_head, hash_next, hash_size, cell_size);
    k_capillary_lubrication_contact<<<blocks, t, 0, s>>>(
        pts, shell, d_mats, hash_head, hash_next, hash_size, cell_size, dt_s, sigma_w, mu_w
    );
    return (int32_t)cudaGetLastError();
}

int32_t sim_shell_integrate(
    MpmParticlesGpu pts,
    ShellMeshGpu shell,
    const MaterialParamsGpu* d_mats,
    float gx,
    float gy,
    float gz,
    float dt_s,
    int32_t has_floor,
    float floor_z,
    float floor_mu,
    void* stream
) {
    if (pts.count <= 0 || shell.p_fx == nullptr) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    int t = 256;
    k_shell_integrate_particles<<<(unsigned)((pts.count + t - 1) / t), t, 0, s>>>(
        pts, shell, d_mats, gx, gy, gz, dt_s, has_floor, floor_z, floor_mu
    );
    return (int32_t)cudaGetLastError();
}

int32_t sim_phase_field_step(
    MpmParticlesGpu pts,
    MpmGridGpu grid,
    const MaterialParamsGpu* d_mats,
    float dt_pf,
    void* stream
) {
    if (pts.count <= 0 || grid.d_field == nullptr || grid.fx == nullptr) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    size_t n_nodes = (size_t)grid.nx * (size_t)grid.ny * (size_t)grid.nz;
    int t = 256;
    k_pf_clear_grid<<<(unsigned)((n_nodes + t - 1) / t), t, 0, s>>>(grid);
    k_pf_p2g<<<(unsigned)((pts.count + t - 1) / t), t, 0, s>>>(pts, grid, d_mats);
    k_pf_normalize<<<(unsigned)((n_nodes + t - 1) / t), t, 0, s>>>(grid);

    dim3 b(8, 8, 4);
    dim3 g = grid3(grid.nx, grid.ny, grid.nz, b);
    // Ping-pong using grid.fx (scratch MPM buffer) so grid.d_field preserves d_old intact
    k_pf_solve_jacobi<<<g, b, 0, s>>>(grid, grid.d_field, grid.fx, dt_pf);
    k_pf_solve_jacobi<<<g, b, 0, s>>>(grid, grid.fx, grid.d_new, dt_pf);
    k_pf_g2p<<<(unsigned)((pts.count + t - 1) / t), t, 0, s>>>(pts, grid, d_mats);
    return (int32_t)cudaGetLastError();
}

int32_t sim_shell_substep_fused(
    ShellMeshGpu shell,
    MpmParticlesGpu pts,
    const MaterialParamsGpu* d_mats,
    int32_t* hash_head,
    int32_t* hash_next,
    int32_t hash_size,
    float cell_size,
    float gx,
    float gy,
    float gz,
    float dt_s,
    int32_t n_mpm,
    int32_t do_contact,
    int32_t has_floor,
    float floor_z,
    float floor_mu,
    float sigma_w,
    float mu_w,
    void* stream
) {
    if (pts.count <= 0 || n_mpm <= 0 || shell.p_fx == nullptr) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    int max_work = max(pts.count, max(shell.num_tris, shell.num_hinges));
    int threads = 256;
    int blocks = min(32, max(1, (max_work + threads - 1) / threads));
    void* args[] = {
        (void*)&shell, (void*)&pts, (void*)&d_mats,
        (void*)&hash_head, (void*)&hash_next, (void*)&hash_size, (void*)&cell_size,
        (void*)&gx, (void*)&gy, (void*)&gz, (void*)&dt_s, (void*)&n_mpm, (void*)&do_contact,
        (void*)&has_floor, (void*)&floor_z, (void*)&floor_mu, (void*)&sigma_w, (void*)&mu_w
    };
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)k_shell_substep_fused, dim3((unsigned)blocks, 1, 1), dim3((unsigned)threads, 1, 1),
        args, 0, s
    );
    return (int32_t)err;
}

int32_t sim_mpm_set_body_prescribed_vel(
    MpmParticlesGpu pts,
    int32_t target_body_id,
    float vx, float vy, float vz,
    float wx, float wy, float wz,
    float piv_x, float piv_y, float piv_z,
    void* stream
) {
    if (pts.count <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    int t = 256;
    k_mpm_set_body_prescribed_vel<<<(unsigned)((pts.count + t - 1) / t), t, 0, s>>>(
        pts, target_body_id, vx, vy, vz, wx, wy, wz, piv_x, piv_y, piv_z
    );
    return (int32_t)cudaGetLastError();
}

int32_t sim_fluid_apply_inflow_nozzle(
    FluidLevelGpu lvl,
    float cx, float cy, float cz,
    float radius, float height,
    float vx, float vy, float vz,
    float relax,
    void* stream
) {
    float span_xy = radius + 2.0f * lvl.dx;
    float half_h = 0.5f * height;
    float span_z = half_h + 2.0f * lvl.dx;
    int h = lvl.halo;
    int i_lo = max(h, (int)floorf((cx - span_xy - lvl.ox) / lvl.dx) + h);
    int i_hi = min(h + lvl.nx - 1, (int)ceilf((cx + span_xy - lvl.ox) / lvl.dx) + h);
    int j_lo = max(h, (int)floorf((cy - span_xy - lvl.oy) / lvl.dx) + h);
    int j_hi = min(h + lvl.ny - 1, (int)ceilf((cy + span_xy - lvl.oy) / lvl.dx) + h);
    int k_lo = max(h, (int)floorf((cz - span_z - lvl.oz) / lvl.dx) + h);
    int k_hi = min(h + lvl.nz - 1, (int)ceilf((cz + span_z - lvl.oz) / lvl.dx) + h);
    if (i_lo > i_hi || j_lo > j_hi || k_lo > k_hi) return 0;

    int bx = i_hi - i_lo + 1;
    int by = j_hi - j_lo + 1;
    int bz = k_hi - k_lo + 1;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 b(8, 8, 4);
    k_fluid_apply_inflow_nozzle_bbox<<<grid3(bx + 1, by + 1, bz + 1, b), b, 0, s>>>(
        lvl, i_lo, j_lo, k_lo, bx, by, bz, cx, cy, cz, radius, half_h, vx, vy, vz, relax
    );
    return (int32_t)cudaGetLastError();
}

int32_t sim_fluid_spawn_droplet(
    FluidLevelGpu lvl, FluidConstantsGpu c,
    float cx, float cy, float cz, float radius,
    float vx, float vy, float vz,
    void* stream
) {
    float span = radius + 2.5f * lvl.dx;
    int h = lvl.halo;
    int i_lo = max(h, (int)floorf((cx - span - lvl.ox) / lvl.dx) + h);
    int i_hi = min(h + lvl.nx - 1, (int)ceilf((cx + span - lvl.ox) / lvl.dx) + h);
    int j_lo = max(h, (int)floorf((cy - span - lvl.oy) / lvl.dx) + h);
    int j_hi = min(h + lvl.ny - 1, (int)ceilf((cy + span - lvl.oy) / lvl.dx) + h);
    int k_lo = max(h, (int)floorf((cz - span - lvl.oz) / lvl.dx) + h);
    int k_hi = min(h + lvl.nz - 1, (int)ceilf((cz + span - lvl.oz) / lvl.dx) + h);
    if (i_lo > i_hi || j_lo > j_hi || k_lo > k_hi) return 0;

    int bx = i_hi - i_lo + 1;
    int by = j_hi - j_lo + 1;
    int bz = k_hi - k_lo + 1;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 b(8, 8, 4);
    k_fluid_spawn_droplet_bbox<<<grid3(bx + 1, by + 1, bz + 1, b), b, 0, s>>>(
        lvl, c, i_lo, j_lo, k_lo, bx, by, bz, cx, cy, cz, radius, vx, vy, vz
    );
    return (int32_t)cudaGetLastError();
}

static inline void launch_euler_outer_bc_slabs(FluidLevelGpu lvl, int32_t stage, cudaStream_t s) {
    if (lvl.level_idx != 0) return;
    int h2 = 2 * lvl.halo;
    dim3 bx(2, 16, 8);
    k_euler_outer_bc_slab_x<<<grid3(h2, lvl.nyp, lvl.nzp, bx), bx, 0, s>>>(lvl, stage);
    dim3 by(16, 2, 8);
    k_euler_outer_bc_slab_y<<<grid3(lvl.nxp, h2, lvl.nzp, by), by, 0, s>>>(lvl, stage);
    dim3 bz(16, 8, 2);
    k_euler_outer_bc_slab_z<<<grid3(lvl.nxp, lvl.nyp, h2, bz), bz, 0, s>>>(lvl, stage);
}

int32_t sim_euler_init_ambient(FluidLevelGpu lvl, FluidConstantsGpu c, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    dim3 b(8, 8, 4);
    k_euler_init_ambient<<<grid3(lvl.nxp, lvl.nyp, lvl.nzp, b), b, 0, s>>>(lvl, c);
    return (int32_t)cudaGetLastError();
}

int32_t sim_euler_add_blast_region(
    FluidLevelGpu lvl, FluidConstantsGpu c,
    float cx, float cy, float cz, float radius, float length, int32_t is_cylinder_x,
    float overpressure_pa, float density_kg_m3, float vx, float vy, float vz,
    void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    dim3 b(8, 8, 4);
    k_euler_add_blast_region<<<grid3(lvl.nxp, lvl.nyp, lvl.nzp, b), b, 0, s>>>(
        lvl, c, cx, cy, cz, radius, length, is_cylinder_x, overpressure_pa, density_kg_m3, vx, vy, vz
    );
    return (int32_t)cudaGetLastError();
}

int32_t sim_euler_outer_bc(FluidLevelGpu lvl, int32_t stage, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    launch_euler_outer_bc_slabs(lvl, stage, s);
    return (int32_t)cudaGetLastError();
}

int32_t sim_euler_rk_stage(FluidLevelGpu lvl, FluidConstantsGpu c, int32_t stage, float dt, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    if (lvl.level_idx == 0) {
        launch_euler_outer_bc_slabs(lvl, stage, s);
    }
    dim3 b(8, 8, 4);
    k_euler_rk_stage<<<grid3(lvl.nx, lvl.ny, lvl.nz, b), b, 0, s>>>(lvl, c, stage, dt);
    if (stage != 0 && lvl.level_idx == 0) {
        launch_euler_outer_bc_slabs(lvl, 0, s);
    }
    return (int32_t)cudaGetLastError();
}

int32_t sim_euler_apply_projectile(
    FluidLevelGpu lvl, FluidConstantsGpu c,
    float base_x_old, float center_y_old, float center_z_old,
    float base_x_new, float center_y_new, float center_z_new,
    float vx, float vy, float vz, float radius, float length, float dt,
    double* d_force_out, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    if (d_force_out != nullptr) {
        cudaMemsetAsync(d_force_out, 0, 3 * sizeof(double), s);
    }
    float x_min = fminf(base_x_old, base_x_new) - 1.5f * lvl.dx;
    float x_max = fmaxf(base_x_old, base_x_new) + length + 1.5f * lvl.dx;
    float y_min = fminf(center_y_old, center_y_new) - radius - 1.5f * lvl.dx;
    float y_max = fmaxf(center_y_old, center_y_new) + radius + 1.5f * lvl.dx;
    float z_min = fminf(center_z_old, center_z_new) - radius - 1.5f * lvl.dx;
    float z_max = fmaxf(center_z_old, center_z_new) + radius + 1.5f * lvl.dx;

    int h = lvl.halo;
    int i_lo = max(h, (int)floorf((x_min - lvl.ox) / lvl.dx) + h);
    int i_hi = min(h + lvl.nx - 1, (int)ceilf((x_max - lvl.ox) / lvl.dx) + h);
    int j_lo = max(h, (int)floorf((y_min - lvl.oy) / lvl.dx) + h);
    int j_hi = min(h + lvl.ny - 1, (int)ceilf((y_max - lvl.oy) / lvl.dx) + h);
    int k_lo = max(h, (int)floorf((z_min - lvl.oz) / lvl.dx) + h);
    int k_hi = min(h + lvl.nz - 1, (int)ceilf((z_max - lvl.oz) / lvl.dx) + h);
    if (i_lo > i_hi || j_lo > j_hi || k_lo > k_hi) return 0;

    int bx = i_hi - i_lo + 1;
    int by = j_hi - j_lo + 1;
    int bz = k_hi - k_lo + 1;
    dim3 b(8, 8, 4);
    k_euler_apply_projectile_bbox<<<grid3(bx, by, bz, b), b, 0, s>>>(
        lvl, c, i_lo, j_lo, k_lo, bx, by, bz,
        base_x_old, center_y_old, center_z_old,
        base_x_new, center_y_new, center_z_new,
        vx, vy, vz, radius, length, dt, d_force_out
    );
    return (int32_t)cudaGetLastError();
}

int32_t sim_euler_apply_directional_source(
    FluidLevelGpu lvl, FluidConstantsGpu c,
    float cx, float cy, float cz, float dir_x, float dir_y, float dir_z,
    float cardioid_beta, float radius, float rate_pa_per_s, float dt_ac, void* stream
) {
    if (fabsf(rate_pa_per_s) <= 0.0f) return 0;
    float span = 3.5f * radius + lvl.dx;
    int h = lvl.halo;
    int i_lo = max(h, (int)floorf((cx - span - lvl.ox) / lvl.dx) + h);
    int i_hi = min(h + lvl.nx - 1, (int)ceilf((cx + span - lvl.ox) / lvl.dx) + h);
    int j_lo = max(h, (int)floorf((cy - span - lvl.oy) / lvl.dx) + h);
    int j_hi = min(h + lvl.ny - 1, (int)ceilf((cy + span - lvl.oy) / lvl.dx) + h);
    int k_lo = max(h, (int)floorf((cz - span - lvl.oz) / lvl.dx) + h);
    int k_hi = min(h + lvl.nz - 1, (int)ceilf((cz + span - lvl.oz) / lvl.dx) + h);
    if (i_lo > i_hi || j_lo > j_hi || k_lo > k_hi) return 0;

    int bx = i_hi - i_lo + 1;
    int by = j_hi - j_lo + 1;
    int bz = k_hi - k_lo + 1;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 b(8, 8, 4);
    k_euler_apply_directional_source_bbox<<<grid3(bx, by, bz, b), b, 0, s>>>(
        lvl, c, i_lo, j_lo, k_lo, bx, by, bz,
        cx, cy, cz, dir_x, dir_y, dir_z, cardioid_beta, radius, rate_pa_per_s, dt_ac
    );
    return (int32_t)cudaGetLastError();
}

int32_t sim_euler_apply_wind_inlet(
    FluidLevelGpu lvl, FluidConstantsGpu c,
    float min_x, float min_y, float min_z, float max_x, float max_y, float max_z,
    float vx, float vy, float vz, float turb_amp, float t_now, float relax, void* stream
) {
    int h = lvl.halo;
    int i_lo = max(h, (int)floorf((min_x - lvl.ox) / lvl.dx) + h);
    int i_hi = min(h + lvl.nx - 1, (int)ceilf((max_x - lvl.ox) / lvl.dx) + h);
    int j_lo = max(h, (int)floorf((min_y - lvl.oy) / lvl.dx) + h);
    int j_hi = min(h + lvl.ny - 1, (int)ceilf((max_y - lvl.oy) / lvl.dx) + h);
    int k_lo = max(h, (int)floorf((min_z - lvl.oz) / lvl.dx) + h);
    int k_hi = min(h + lvl.nz - 1, (int)ceilf((max_z - lvl.oz) / lvl.dx) + h);
    if (i_lo > i_hi || j_lo > j_hi || k_lo > k_hi) return 0;

    int bx = i_hi - i_lo + 1;
    int by = j_hi - j_lo + 1;
    int bz = k_hi - k_lo + 1;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 b(8, 8, 4);
    k_euler_apply_wind_inlet_bbox<<<grid3(bx, by, bz, b), b, 0, s>>>(
        lvl, c, i_lo, j_lo, k_lo, bx, by, bz, vx, vy, vz, turb_amp, t_now, relax
    );
    return (int32_t)cudaGetLastError();
}

int32_t sim_euler_amr_fill_ghosts(FluidLevelGpu child, FluidLevelGpu parent, int32_t target_stage, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    int h2 = 2 * child.halo;
    dim3 bx(2, 16, 8);
    k_euler_amr_fill_ghosts_slab<<<grid3(h2, child.nyp, child.nzp, bx), bx, 0, s>>>(child, parent, target_stage, 0);
    dim3 by(16, 2, 8);
    k_euler_amr_fill_ghosts_slab<<<grid3(child.nx, h2, child.nzp, by), by, 0, s>>>(child, parent, target_stage, 1);
    dim3 bz(16, 8, 2);
    k_euler_amr_fill_ghosts_slab<<<grid3(child.nx, child.ny, h2, bz), bz, 0, s>>>(child, parent, target_stage, 2);
    return (int32_t)cudaGetLastError();
}

int32_t sim_euler_amr_restrict(FluidLevelGpu child, FluidLevelGpu parent, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    int r = child.refine_ratio;
    int nx_c = child.nx / r;
    int ny_c = child.ny / r;
    int nz_c = child.nz / r;
    dim3 b(8, 8, 4);
    k_euler_amr_restrict<<<grid3(nx_c, ny_c, nz_c, b), b, 0, s>>>(child, parent);
    return (int32_t)cudaGetLastError();
}

int32_t sim_euler_shift_x(FluidLevelGpu patch, FluidLevelGpu parent, int32_t shift_cells, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    dim3 b(8, 8, 4);
    k_euler_shift_x_stage1<<<grid3(patch.nxp, patch.nyp, patch.nzp, b), b, 0, s>>>(patch, shift_cells);
    k_euler_shift_x_stage2<<<grid3(patch.nxp, patch.nyp, patch.nzp, b), b, 0, s>>>(patch, parent, shift_cells);
    return (int32_t)cudaGetLastError();
}

int32_t sim_euler_copy_overlap(FluidLevelGpu dst_lvl, FluidLevelGpu src_lvl, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    dim3 b(8, 8, 4);
    k_euler_copy_overlap<<<grid3(dst_lvl.nx, dst_lvl.ny, dst_lvl.nz, b), b, 0, s>>>(dst_lvl, src_lvl);
    return (int32_t)cudaGetLastError();
}

int32_t sim_euler_ib_splat_level(MpmParticlesGpu pts, FluidLevelGpu lvl, void* stream) {
    if (pts.count <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    int t = 256;
    unsigned blocks = (unsigned)((pts.count + t - 1) / t);
    k_euler_ib_reset_dynamic<<<blocks, t, 0, s>>>(pts, lvl);
    k_euler_ib_splat_particles<<<blocks, t, 0, s>>>(pts, lvl);
    k_euler_ib_normalize<<<blocks, t, 0, s>>>(pts, lvl);
    return (int32_t)cudaGetLastError();
}

int32_t sim_euler_ib_gather_to_solid(
    MpmParticlesGpu pts, const MaterialParamsGpu* d_mats, const FluidLevelGpu* d_levels,
    int32_t num_levels, FluidConstantsGpu c, void* stream
) {
    if (pts.count <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    int t = 256;
    k_euler_ib_gather_to_solid<<<(unsigned)((pts.count + t - 1) / t), t, 0, s>>>(
        pts, d_mats, d_levels, num_levels, c
    );
    return (int32_t)cudaGetLastError();
}

int32_t sim_euler_sample_probes(
    const FluidLevelGpu* d_levels, int32_t num_levels,
    const MicProbeGpu* d_probes, int32_t num_probes,
    float* d_out_samples, int32_t sample_idx, FluidConstantsGpu c, void* stream
) {
    if (num_probes <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    int threads = 64;
    int blocks = (num_probes + threads - 1) / threads;
    k_euler_sample_probes<<<blocks, threads, 0, s>>>(
        d_levels, num_levels, d_probes, num_probes, d_out_samples, sample_idx, c
    );
    return (int32_t)cudaGetLastError();
}

int32_t sim_euler_reduce_metrics(FluidLevelGpu lvl, FluidConstantsGpu c, SimReduceMetricsGpu* d_out, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    k_reset_metrics<<<1, 1, 0, s>>>(d_out);
    k_euler_reduce_metrics<<<128, 256, 0, s>>>(lvl, c, d_out);
    return (int32_t)cudaGetLastError();
}

} // extern "C"
