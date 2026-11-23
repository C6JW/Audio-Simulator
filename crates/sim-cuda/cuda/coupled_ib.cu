#include "common.cuh"

__device__ __forceinline__ void splat_trilinear_face(
    float* beta_arr,
    float* vel_arr,
    int sx, int sy, int sz,
    float fx, float fy, float fz,
    float weight_scale,
    float v_val
) {
    if (weight_scale <= 1e-12f) return;
    int i0 = (int)floorf(fx);
    int j0 = (int)floorf(fy);
    int k0 = (int)floorf(fz);
    if (i0 < -1 || i0 >= sx || j0 < -1 || j0 >= sy || k0 < -1 || k0 >= sz) return;

    float tx = fx - (float)i0;
    float ty = fy - (float)j0;
    float tz = fz - (float)k0;

    for (int dk = 0; dk <= 1; ++dk) {
        int kk = k0 + dk;
        if (kk < 0 || kk >= sz) continue;
        float wz = dk ? tz : (1.0f - tz);
        for (int dj = 0; dj <= 1; ++dj) {
            int jj = j0 + dj;
            if (jj < 0 || jj >= sy) continue;
            float wy = dj ? ty : (1.0f - ty);
            for (int di = 0; di <= 1; ++di) {
                int ii = i0 + di;
                if (ii < 0 || ii >= sx) continue;
                float wx = di ? tx : (1.0f - tx);
                float w = weight_scale * wx * wy * wz;
                if (w > 1e-10f) {
                    size_t fid = (size_t)ii + (size_t)sx * ((size_t)jj + (size_t)sy * (size_t)kk);
                    atomicAdd(&beta_arr[fid], w);
                    if (vel_arr != nullptr) {
                        atomicAdd(&vel_arr[fid], w * v_val);
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ void reset_trilinear_face_from_static(
    float* beta_arr,
    const float* beta_static_arr,
    float* vel_arr,
    int sx, int sy, int sz,
    float fx, float fy, float fz
) {
    int i0 = (int)floorf(fx);
    int j0 = (int)floorf(fy);
    int k0 = (int)floorf(fz);
    if (i0 < -2 || i0 >= sx + 1 || j0 < -2 || j0 >= sy + 1 || k0 < -2 || k0 >= sz + 1) return;

    for (int dk = -1; dk <= 2; ++dk) {
        int kk = k0 + dk;
        if (kk < 0 || kk >= sz) continue;
        for (int dj = -1; dj <= 2; ++dj) {
            int jj = j0 + dj;
            if (jj < 0 || jj >= sy) continue;
            for (int di = -1; di <= 2; ++di) {
                int ii = i0 + di;
                if (ii < 0 || ii >= sx) continue;
                size_t fid = (size_t)ii + (size_t)sx * ((size_t)jj + (size_t)sy * (size_t)kk);
                beta_arr[fid] = (beta_static_arr != nullptr) ? beta_static_arr[fid] : 0.0f;
                if (vel_arr != nullptr) {
                    vel_arr[fid] = 0.0f;
                }
            }
        }
    }
}

__device__ __forceinline__ void normalize_trilinear_face(
    float* beta_arr,
    float* vel_arr,
    int sx, int sy, int sz,
    float fx, float fy, float fz
) {
    int i0 = (int)floorf(fx);
    int j0 = (int)floorf(fy);
    int k0 = (int)floorf(fz);
    if (i0 < -1 || i0 >= sx || j0 < -1 || j0 >= sy || k0 < -1 || k0 >= sz) return;

    for (int dk = 0; dk <= 1; ++dk) {
        int kk = k0 + dk;
        if (kk < 0 || kk >= sz) continue;
        for (int dj = 0; dj <= 1; ++dj) {
            int jj = j0 + dj;
            if (jj < 0 || jj >= sy) continue;
            for (int di = 0; di <= 1; ++di) {
                int ii = i0 + di;
                if (ii < 0 || ii >= sx) continue;
                size_t fid = (size_t)ii + (size_t)sx * ((size_t)jj + (size_t)sy * (size_t)kk);
                float b = beta_arr[fid];
                if (b > 1e-6f) {
                    float old_b = atomicExch(&beta_arr[fid], -b);
                    if (old_b > 1e-6f && vel_arr != nullptr) {
                        vel_arr[fid] /= old_b;
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ void unmark_trilinear_face(
    float* beta_arr,
    int sx, int sy, int sz,
    float fx, float fy, float fz
) {
    int i0 = (int)floorf(fx);
    int j0 = (int)floorf(fy);
    int k0 = (int)floorf(fz);
    if (i0 < -1 || i0 >= sx || j0 < -1 || j0 >= sy || k0 < -1 || k0 >= sz) return;

    for (int dk = 0; dk <= 1; ++dk) {
        int kk = k0 + dk;
        if (kk < 0 || kk >= sz) continue;
        for (int dj = 0; dj <= 1; ++dj) {
            int jj = j0 + dj;
            if (jj < 0 || jj >= sy) continue;
            for (int di = 0; di <= 1; ++di) {
                int ii = i0 + di;
                if (ii < 0 || ii >= sx) continue;
                size_t fid = (size_t)ii + (size_t)sx * ((size_t)jj + (size_t)sy * (size_t)kk);
                float b = beta_arr[fid];
                if (b < 0.0f) {
                    beta_arr[fid] = fminf(1.0f, -b);
                }
            }
        }
    }
}

__global__ void k_ib_clear_fluid_level(FluidLevelGpu lvl) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i <= lvl.nxp && j < lvl.nyp && k < lvl.nzp) {
        size_t fx = idx_fx(i, j, k, lvl.nxp, lvl.nyp);
        lvl.ib_beta_x[fx] = (lvl.ib_beta_static_x != nullptr) ? lvl.ib_beta_static_x[fx] : 0.0f;
        if (lvl.ib_vel_x) lvl.ib_vel_x[fx] = 0.0f;
    }
    if (i < lvl.nxp && j <= lvl.nyp && k < lvl.nzp) {
        size_t fy = idx_fy(i, j, k, lvl.nxp, lvl.nyp);
        lvl.ib_beta_y[fy] = (lvl.ib_beta_static_y != nullptr) ? lvl.ib_beta_static_y[fy] : 0.0f;
        if (lvl.ib_vel_y) lvl.ib_vel_y[fy] = 0.0f;
    }
    if (i < lvl.nxp && j < lvl.nyp && k <= lvl.nzp) {
        size_t fz = idx_fz(i, j, k, lvl.nxp, lvl.nyp);
        lvl.ib_beta_z[fz] = (lvl.ib_beta_static_z != nullptr) ? lvl.ib_beta_static_z[fz] : 0.0f;
        if (lvl.ib_vel_z) lvl.ib_vel_z[fz] = 0.0f;
    }
}

__global__ void k_ib_copy_to_static_level(FluidLevelGpu lvl) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i <= lvl.nxp && j < lvl.nyp && k < lvl.nzp && lvl.ib_beta_static_x != nullptr) {
        size_t fx = idx_fx(i, j, k, lvl.nxp, lvl.nyp);
        lvl.ib_beta_static_x[fx] = lvl.ib_beta_x[fx];
    }
    if (i < lvl.nxp && j <= lvl.nyp && k < lvl.nzp && lvl.ib_beta_static_y != nullptr) {
        size_t fy = idx_fy(i, j, k, lvl.nxp, lvl.nyp);
        lvl.ib_beta_static_y[fy] = lvl.ib_beta_y[fy];
    }
    if (i < lvl.nxp && j < lvl.nyp && k <= lvl.nzp && lvl.ib_beta_static_z != nullptr) {
        size_t fz = idx_fz(i, j, k, lvl.nxp, lvl.nyp);
        lvl.ib_beta_static_z[fz] = lvl.ib_beta_z[fz];
    }
}

__global__ void k_ib_reset_dynamic_particles(MpmParticlesGpu pts, FluidLevelGpu lvl) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;

    float inv_dx = 1.0f / lvl.dx;
    float gx = (pts.px[p] - lvl.ox) * inv_dx + (float)lvl.halo;
    float gy = (pts.py[p] - lvl.oy) * inv_dx + (float)lvl.halo;
    float gz = (pts.pz[p] - lvl.oz) * inv_dx + (float)lvl.halo;

    if (gx < -2.0f || gx > (float)(lvl.nxp + 2) ||
        gy < -2.0f || gy > (float)(lvl.nyp + 2) ||
        gz < -2.0f || gz > (float)(lvl.nzp + 2)) {
        return;
    }

    reset_trilinear_face_from_static(
        lvl.ib_beta_x, lvl.ib_beta_static_x, lvl.ib_vel_x,
        lvl.nxp + 1, lvl.nyp, lvl.nzp, gx, gy - 0.5f, gz - 0.5f
    );
    reset_trilinear_face_from_static(
        lvl.ib_beta_y, lvl.ib_beta_static_y, lvl.ib_vel_y,
        lvl.nxp, lvl.nyp + 1, lvl.nzp, gx - 0.5f, gy, gz - 0.5f
    );
    reset_trilinear_face_from_static(
        lvl.ib_beta_z, lvl.ib_beta_static_z, lvl.ib_vel_z,
        lvl.nxp, lvl.nyp, lvl.nzp + 1, gx - 0.5f, gy - 0.5f, gz
    );
}

__global__ void k_ib_splat_solid_to_fluid(MpmParticlesGpu pts, FluidLevelGpu lvl) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;

    float px = pts.px[p];
    float py = pts.py[p];
    float pz = pts.pz[p];

    float inv_dx = 1.0f / lvl.dx;
    float gx = (px - lvl.ox) * inv_dx + (float)lvl.halo;
    float gy = (py - lvl.oy) * inv_dx + (float)lvl.halo;
    float gz = (pz - lvl.oz) * inv_dx + (float)lvl.halo;

    if (gx < -1.0f || gx > (float)(lvl.nxp + 1) ||
        gy < -1.0f || gy > (float)(lvl.nyp + 1) ||
        gz < -1.0f || gz > (float)(lvl.nzp + 1)) {
        return;
    }

    float vx = pts.vx[p];
    float vy = pts.vy[p];
    float vz = pts.vz[p];
    uint32_t pflag = pts.flags[p];

    float wx_scale, wy_scale, wz_scale;
    float svx = vx, svy = vy, svz = vz;
    if (pflag & 8u) {
        // Thin shell: normal-projected area blocking + normal-only velocity splat (free tangential slip)
        float intact = fmaxf(0.0f, 1.0f - pts.damage[p]);
        if (intact <= 0.05f) return;
        float base = intact * 1.8f * pts.area[p] * (inv_dx * inv_dx);
        float nxp = pts.nx[p], nyp = pts.ny[p], nzp = pts.nz[p];
        wx_scale = base * fabsf(nxp);
        wy_scale = base * fabsf(nyp);
        wz_scale = base * fabsf(nzp);
        float vn = vx * nxp + vy * nyp + vz * nzp;
        svx = vn * nxp;
        svy = vn * nyp;
        svz = vn * nzp;
    } else {
        // Bulk continuum solid: isotropic volume-fraction blocking
        float base = 2.2f * pts.vol0[p] * (inv_dx * inv_dx * inv_dx);
        wx_scale = base;
        wy_scale = base;
        wz_scale = base;
    }

    splat_trilinear_face(
        lvl.ib_beta_x, lvl.ib_vel_x, lvl.nxp + 1, lvl.nyp, lvl.nzp,
        gx, gy - 0.5f, gz - 0.5f, wx_scale, svx
    );
    splat_trilinear_face(
        lvl.ib_beta_y, lvl.ib_vel_y, lvl.nxp, lvl.nyp + 1, lvl.nzp,
        gx - 0.5f, gy, gz - 0.5f, wy_scale, svy
    );
    splat_trilinear_face(
        lvl.ib_beta_z, lvl.ib_vel_z, lvl.nxp, lvl.nyp, lvl.nzp + 1,
        gx - 0.5f, gy - 0.5f, gz, wz_scale, svz
    );
}

__global__ void k_ib_normalize_dynamic_particles(MpmParticlesGpu pts, FluidLevelGpu lvl) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;

    float inv_dx = 1.0f / lvl.dx;
    float gx = (pts.px[p] - lvl.ox) * inv_dx + (float)lvl.halo;
    float gy = (pts.py[p] - lvl.oy) * inv_dx + (float)lvl.halo;
    float gz = (pts.pz[p] - lvl.oz) * inv_dx + (float)lvl.halo;

    if (gx < -1.0f || gx > (float)(lvl.nxp + 1) ||
        gy < -1.0f || gy > (float)(lvl.nyp + 1) ||
        gz < -1.0f || gz > (float)(lvl.nzp + 1)) {
        return;
    }

    normalize_trilinear_face(lvl.ib_beta_x, lvl.ib_vel_x, lvl.nxp + 1, lvl.nyp, lvl.nzp, gx, gy - 0.5f, gz - 0.5f);
    normalize_trilinear_face(lvl.ib_beta_y, lvl.ib_vel_y, lvl.nxp, lvl.nyp + 1, lvl.nzp, gx - 0.5f, gy, gz - 0.5f);
    normalize_trilinear_face(lvl.ib_beta_z, lvl.ib_vel_z, lvl.nxp, lvl.nyp, lvl.nzp + 1, gx - 0.5f, gy - 0.5f, gz);
}

__global__ void k_ib_unmark_dynamic_particles(MpmParticlesGpu pts, FluidLevelGpu lvl) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;

    float inv_dx = 1.0f / lvl.dx;
    float gx = (pts.px[p] - lvl.ox) * inv_dx + (float)lvl.halo;
    float gy = (pts.py[p] - lvl.oy) * inv_dx + (float)lvl.halo;
    float gz = (pts.pz[p] - lvl.oz) * inv_dx + (float)lvl.halo;

    if (gx < -1.0f || gx > (float)(lvl.nxp + 1) ||
        gy < -1.0f || gy > (float)(lvl.nyp + 1) ||
        gz < -1.0f || gz > (float)(lvl.nzp + 1)) {
        return;
    }

    unmark_trilinear_face(lvl.ib_beta_x, lvl.nxp + 1, lvl.nyp, lvl.nzp, gx, gy - 0.5f, gz - 0.5f);
    unmark_trilinear_face(lvl.ib_beta_y, lvl.nxp, lvl.nyp + 1, lvl.nzp, gx - 0.5f, gy, gz - 0.5f);
    unmark_trilinear_face(lvl.ib_beta_z, lvl.nxp, lvl.nyp, lvl.nzp + 1, gx - 0.5f, gy - 0.5f, gz);
}

__global__ void k_ib_normalize_fluid_level(FluidLevelGpu lvl) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i <= lvl.nxp && j < lvl.nyp && k < lvl.nzp) {
        size_t fx = idx_fx(i, j, k, lvl.nxp, lvl.nyp);
        float b = lvl.ib_beta_x[fx];
        if (b > 1e-6f) {
            if (lvl.ib_vel_x) lvl.ib_vel_x[fx] /= b;
            lvl.ib_beta_x[fx] = fminf(1.0f, b);
        } else {
            if (lvl.ib_vel_x) lvl.ib_vel_x[fx] = 0.0f;
            lvl.ib_beta_x[fx] = 0.0f;
        }
    }
    if (i < lvl.nxp && j <= lvl.nyp && k < lvl.nzp) {
        size_t fy = idx_fy(i, j, k, lvl.nxp, lvl.nyp);
        float b = lvl.ib_beta_y[fy];
        if (b > 1e-6f) {
            if (lvl.ib_vel_y) lvl.ib_vel_y[fy] /= b;
            lvl.ib_beta_y[fy] = fminf(1.0f, b);
        } else {
            if (lvl.ib_vel_y) lvl.ib_vel_y[fy] = 0.0f;
            lvl.ib_beta_y[fy] = 0.0f;
        }
    }
    if (i < lvl.nxp && j < lvl.nyp && k <= lvl.nzp) {
        size_t fz = idx_fz(i, j, k, lvl.nxp, lvl.nyp);
        float b = lvl.ib_beta_z[fz];
        if (b > 1e-6f) {
            if (lvl.ib_vel_z) lvl.ib_vel_z[fz] /= b;
            lvl.ib_beta_z[fz] = fminf(1.0f, b);
        } else {
            if (lvl.ib_vel_z) lvl.ib_vel_z[fz] = 0.0f;
            lvl.ib_beta_z[fz] = 0.0f;
        }
    }
}

__device__ __forceinline__ void sample_fluid_state_at(
    const FluidLevelGpu& lvl, float x, float y, float z,
    float& p_ac_out, float& p_hydro_out, float& alpha_out, float& ux_out, float& uy_out, float& uz_out
) {
    float inv_dx = 1.0f / lvl.dx;
    float fx = (x - lvl.ox) * inv_dx - 0.5f + (float)lvl.halo;
    float fy = (y - lvl.oy) * inv_dx - 0.5f + (float)lvl.halo;
    float fz = (z - lvl.oz) * inv_dx - 0.5f + (float)lvl.halo;
    int i0 = max(0, min(lvl.nxp - 2, (int)floorf(fx)));
    int j0 = max(0, min(lvl.nyp - 2, (int)floorf(fy)));
    int k0 = max(0, min(lvl.nzp - 2, (int)floorf(fz)));
    float tx = clamp_f(fx - (float)i0, 0.0f, 1.0f);
    float ty = clamp_f(fy - (float)j0, 0.0f, 1.0f);
    float tz = clamp_f(fz - (float)k0, 0.0f, 1.0f);

    float pac = 0.0f;
    float ph = 0.0f;
    float al = 0.0f;
    for (int dk = 0; dk <= 1; ++dk) {
        float wz = dk ? tz : (1.0f - tz);
        for (int dj = 0; dj <= 1; ++dj) {
            float wy = dj ? ty : (1.0f - ty);
            for (int di = 0; di <= 1; ++di) {
                float wx = di ? tx : (1.0f - tx);
                float w = wx * wy * wz;
                size_t cid = idx_cell(i0 + di, j0 + dj, k0 + dk, lvl.nxp, lvl.nyp);
                pac += w * lvl.p_gauge[cid];
                ph  += w * lvl.p_hydro[cid];
                al  += w * lvl.alpha[cid];
            }
        }
    }
    p_ac_out = pac;
    p_hydro_out = ph;
    alpha_out = clamp_f(al, 0.0f, 1.0f);

    int ix = max(0, min(lvl.nxp, (int)roundf((x - lvl.ox) * inv_dx + (float)lvl.halo)));
    int iy = max(0, min(lvl.nyp, (int)roundf((y - lvl.oy) * inv_dx + (float)lvl.halo)));
    int iz = max(0, min(lvl.nzp, (int)roundf((z - lvl.oz) * inv_dx + (float)lvl.halo)));
    int cx = max(0, min(lvl.nxp - 1, (int)floorf((x - lvl.ox) * inv_dx + (float)lvl.halo)));
    int cy = max(0, min(lvl.nyp - 1, (int)floorf((y - lvl.oy) * inv_dx + (float)lvl.halo)));
    int cz = max(0, min(lvl.nzp - 1, (int)floorf((z - lvl.oz) * inv_dx + (float)lvl.halo)));
    ux_out = lvl.ux[idx_fx(ix, cy, cz, lvl.nxp, lvl.nyp)];
    uy_out = lvl.uy[idx_fy(cx, iy, cz, lvl.nxp, lvl.nyp)];
    uz_out = lvl.uz[idx_fz(cx, cy, iz, lvl.nxp, lvl.nyp)];
}

__global__ void k_ib_gather_fluid_to_solid(
    MpmParticlesGpu pts,
    const MaterialParamsGpu* mats,
    const FluidLevelGpu* levels,
    int num_levels,
    FluidConstantsGpu c,
    float dt_ac
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;

    const MaterialParamsGpu mat = mats[pts.mat_id[p]];
    if (mat.flags & 4u) return; // is_rigid_static

    uint32_t pflag = pts.flags[p];
    if (!(pflag & (4u | 8u))) {
        pts.fx_ext[p] = 0.0f;
        pts.fy_ext[p] = 0.0f;
        pts.fz_ext[p] = 0.0f;
        return;
    }

    float px = pts.px[p];
    float py = pts.py[p];
    float pz = pts.pz[p];

    // Pick finest fluid level containing particle with 2-cell margin
    int best_lid = 0;
    for (int li = 1; li < num_levels; ++li) {
        const FluidLevelGpu& l = levels[li];
        float m = 2.0f * l.dx;
        if (px >= l.ox + m && px <= l.ox + (float)l.nx * l.dx - m &&
            py >= l.oy + m && py <= l.oy + (float)l.ny * l.dx - m &&
            pz >= l.oz + m && pz <= l.oz + (float)l.nz * l.dx - m) {
            best_lid = li;
        }
    }
    const FluidLevelGpu& lvl = levels[best_lid];
    float nx = pts.nx[p];
    float ny = pts.ny[p];
    float nz = pts.nz[p];
    float h_samp = 1.25f * lvl.dx;

    float p_0, p_plus, p_minus = 0.0f;
    float ph_0, ph_plus, ph_minus = 0.0f;
    float a_0, a_plus, a_minus;
    float ux_0, uy_0, uz_0, ux_p, uy_p, uz_p;
    float ux_m = 0.0f, uy_m = 0.0f, uz_m = 0.0f;

    sample_fluid_state_at(lvl, px, py, pz, p_0, ph_0, a_0, ux_0, uy_0, uz_0);
    sample_fluid_state_at(lvl, px + h_samp * nx, py + h_samp * ny, pz + h_samp * nz, p_plus, ph_plus, a_plus, ux_p, uy_p, uz_p);
    a_minus = a_0;
    ph_minus = ph_0;

    float force_mag;
    if (pflag & 8u) {
        sample_fluid_state_at(lvl, px - h_samp * nx, py - h_samp * ny, pz - h_samp * nz, p_minus, ph_minus, a_minus, ux_m, uy_m, uz_m);
        float intact = fmaxf(0.0f, 1.0f - pts.damage[p]);
        force_mag = -intact * (p_plus - p_minus) * pts.area[p];
    } else {
        force_mag = -p_plus * pts.area[p];
    }

    float rho_local = 0.5f * (mix_density_f(a_plus, c) + mix_density_f(a_minus, c));
    pts.fluid_rho[p] = rho_local;
    float m0 = fmaxf(1e-18f, pts.mass[p]);
    float m_add = rho_local * pts.area[p] * fmaxf(0.0f, pts.hyd_len[p]);
    float m_eff = m0 + m_add;

    if (pflag & 8u) {
        // Partitioned explicit fluid-shell passivity factor:
        // accounts for radiation impedance over the acoustic transit + coupling window tau_c
        // and prevents double-counting added mass already integrated in d_shell_integrate_particle_one.
        float k_local = 0.5f * (mix_bulk_modulus_f(a_plus, p_plus + ph_plus, c) + mix_bulk_modulus_f(a_minus, p_minus + ph_minus, c));
        float z_local = sqrtf(fmaxf(1.0f, k_local * rho_local));
        float c_local = sqrtf(fmaxf(1.0f, k_local / fmaxf(1e-3f, rho_local)));
        float tau_c = fmaxf(0.0f, dt_ac) + 2.0f * h_samp / c_local;
        float w_imp = m0 / (m0 + 4.0f * z_local * pts.area[p] * tau_c + 4.0f * m_add);
        force_mag *= w_imp;
    }

    // Physical dynamic form-drag exchange from impacting fluid jets / winds outside the IB band
    // (skip when acoustic radiation impedance is already damping liquid water shells)
    if (rho_local < 50.0f || !(pflag & 8u)) {
        float ux_f = (pflag & 8u) ? 0.5f * (ux_p + ux_m) : ux_p;
        float uy_f = (pflag & 8u) ? 0.5f * (uy_p + uy_m) : uy_p;
        float uz_f = (pflag & 8u) ? 0.5f * (uz_p + uz_m) : uz_p;
        float vn_rel = (ux_f - pts.vx[p]) * nx + (uy_f - pts.vy[p]) * ny + (uz_f - pts.vz[p]) * nz;
        float f_drag_raw = 0.5f * rho_local * pts.area[p] * fabsf(vn_rel) * vn_rel;
        float f_drag_cap = 0.25f * m_eff * fabsf(vn_rel) / fmaxf(1e-7f, dt_ac);
        float f_drag = (f_drag_raw >= 0.0f) ? fminf(f_drag_raw, f_drag_cap) : -fminf(-f_drag_raw, f_drag_cap);
        force_mag += f_drag;
    }

    // Monotonic hygroscopic wetting uptake from contacting liquid water fraction
    if (mat.hygro_absorb_rate > 0.0f && dt_ac > 0.0f) {
        float a_wet = fmaxf(a_0, fmaxf(a_plus, a_minus));
        float w_cur = pts.wetness[p];
        if (a_wet > w_cur) {
            pts.wetness[p] = clamp_f(w_cur + dt_ac * mat.hygro_absorb_rate * (a_wet - w_cur), 0.0f, 1.0f);
        }
    }

    pts.fx_ext[p] = force_mag * nx;
    pts.fy_ext[p] = force_mag * ny;
    pts.fz_ext[p] = force_mag * nz;
}
