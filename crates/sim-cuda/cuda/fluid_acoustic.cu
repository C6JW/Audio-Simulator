#include "common.cuh"

__device__ __forceinline__ float compute_pml_sigma(
    int coord_int,
    int n_int,
    int pml_thick,
    uint32_t mask,
    int bit_lo,
    int bit_hi,
    float offset,
    float c_local,
    float dx,
    float pml_coeff
) {
    if (pml_thick <= 0) return 0.0f;
    float pos = (float)coord_int + offset;
    float p_f = (float)pml_thick;
    if ((mask & (1u << bit_lo)) && pos < p_f) {
        float d = (p_f - pos) / p_f;
        d = clamp_f(d, 0.0f, 1.0f);
        return pml_coeff * c_local / dx * (d * d * d);
    }
    if ((mask & (1u << bit_hi)) && pos > (float)(n_int - pml_thick)) {
        float d = (pos - (float)(n_int - pml_thick)) / p_f;
        d = clamp_f(d, 0.0f, 1.0f);
        return pml_coeff * c_local / dx * (d * d * d);
    }
    return 0.0f;
}

__global__ void k_init_hydrostatic_column(
    FluidLevelGpu lvl,
    FluidConstantsGpu c,
    double p_top_ref
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= lvl.nxp || j >= lvl.nyp) return;

    size_t top_idx = idx_cell(i, j, lvl.nzp - 1, lvl.nxp, lvl.nyp);
    double p_prev = p_top_ref;
    lvl.p_hydro[top_idx] = (float)p_prev;
    lvl.p_gauge[top_idx] = 0.0f;
    if (lvl.p_gauge_old) lvl.p_gauge_old[top_idx] = 0.0f;

    for (int k = lvl.nzp - 1; k > 0; --k) {
        size_t idx_hi = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
        size_t idx_lo = idx_cell(i, j, k - 1, lvl.nxp, lvl.nyp);
        double a_hi = (double)lvl.alpha[idx_hi];
        double a_lo = (double)lvl.alpha[idx_lo];
        double rho_f = 0.5 * (mix_density(a_hi, c) + mix_density(a_lo, c));
        double p_lo = p_prev - rho_f * (double)c.gz * (double)lvl.dx;
        p_prev = p_lo;
        lvl.p_hydro[idx_lo] = (float)p_lo;
        lvl.p_gauge[idx_lo] = 0.0f;
        if (lvl.p_gauge_old) lvl.p_gauge_old[idx_lo] = 0.0f;
    }
}

__global__ void k_init_hydrostatic_child(
    FluidLevelGpu child,
    FluidLevelGpu parent,
    FluidConstantsGpu c
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= child.nxp || j >= child.nyp) return;

    // Sample parent p_hydro at top cell center of child column
    float x = child.ox + ((float)(i - child.halo) + 0.5f) * child.dx;
    float y = child.oy + ((float)(j - child.halo) + 0.5f) * child.dx;
    float z = child.oz + ((float)(child.nzp - 1 - child.halo) + 0.5f) * child.dx;

    float fx = (x - parent.ox) / parent.dx - 0.5f + (float)parent.halo;
    float fy = (y - parent.oy) / parent.dx - 0.5f + (float)parent.halo;
    float fz = (z - parent.oz) / parent.dx - 0.5f + (float)parent.halo;

    int i0 = max(0, min(parent.nxp - 2, (int)floorf(fx)));
    int j0 = max(0, min(parent.nyp - 2, (int)floorf(fy)));
    int k0 = max(0, min(parent.nzp - 2, (int)floorf(fz)));
    float tx = clamp_f(fx - (float)i0, 0.0f, 1.0f);
    float ty = clamp_f(fy - (float)j0, 0.0f, 1.0f);
    float tz = clamp_f(fz - (float)k0, 0.0f, 1.0f);

    double p_top = 0.0;
    for (int dk = 0; dk <= 1; ++dk) {
        double wz = dk ? (double)tz : (1.0 - (double)tz);
        for (int dj = 0; dj <= 1; ++dj) {
            double wy = dj ? (double)ty : (1.0 - (double)ty);
            for (int di = 0; di <= 1; ++di) {
                double wx = di ? (double)tx : (1.0 - (double)tx);
                p_top += wx * wy * wz * (double)parent.p_hydro[idx_cell(i0 + di, j0 + dj, k0 + dk, parent.nxp, parent.nyp)];
            }
        }
    }

    size_t top_idx = idx_cell(i, j, child.nzp - 1, child.nxp, child.nyp);
    double p_prev = p_top;
    child.p_hydro[top_idx] = (float)p_prev;
    child.p_gauge[top_idx] = 0.0f;
    if (child.p_gauge_old) child.p_gauge_old[top_idx] = 0.0f;

    for (int k = child.nzp - 1; k > 0; --k) {
        size_t idx_hi = idx_cell(i, j, k, child.nxp, child.nyp);
        size_t idx_lo = idx_cell(i, j, k - 1, child.nxp, child.nyp);
        double a_hi = (double)child.alpha[idx_hi];
        double a_lo = (double)child.alpha[idx_lo];
        double rho_f = 0.5 * (mix_density(a_hi, c) + mix_density(a_lo, c));
        double p_lo = p_prev - rho_f * (double)c.gz * (double)child.dx;
        p_prev = p_lo;
        child.p_hydro[idx_lo] = (float)p_lo;
        child.p_gauge[idx_lo] = 0.0f;
        if (child.p_gauge_old) child.p_gauge_old[idx_lo] = 0.0f;
    }
}

__global__ void k_save_old_state(FluidLevelGpu lvl) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (lvl.p_gauge_old && i < lvl.nxp && j < lvl.nyp && k < lvl.nzp) {
        size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
        lvl.p_gauge_old[cid] = lvl.p_gauge[cid];
    }
    if (lvl.ux_old && i <= lvl.nxp && j < lvl.nyp && k < lvl.nzp) {
        size_t fx = idx_fx(i, j, k, lvl.nxp, lvl.nyp);
        lvl.ux_old[fx] = lvl.ux[fx];
    }
    if (lvl.uy_old && i < lvl.nxp && j <= lvl.nyp && k < lvl.nzp) {
        size_t fy = idx_fy(i, j, k, lvl.nxp, lvl.nyp);
        lvl.uy_old[fy] = lvl.uy[fy];
    }
    if (lvl.uz_old && i < lvl.nxp && j < lvl.nyp && k <= lvl.nzp) {
        size_t fz = idx_fz(i, j, k, lvl.nxp, lvl.nyp);
        lvl.uz_old[fz] = lvl.uz[fz];
    }
}

__global__ void k_outer_bc_all(FluidLevelGpu lvl) {
    if (lvl.level_idx != 0) return;
    int h = lvl.halo;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < lvl.nxp && j < lvl.nyp && k < lvl.nzp) {
        bool in_x = (i >= h && i < h + lvl.nx);
        bool in_y = (j >= h && j < h + lvl.ny);
        bool in_z = (k >= h && k < h + lvl.nz);
        if (!(in_x && in_y && in_z)) {
            int ic = min(max(i, h), h + lvl.nx - 1);
            int jc = min(max(j, h), h + lvl.ny - 1);
            int kc = min(max(k, h), h + lvl.nz - 1);
            size_t dst_c = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
            size_t src_c = idx_cell(ic, jc, kc, lvl.nxp, lvl.nyp);
            lvl.alpha[dst_c] = lvl.alpha[src_c];
            lvl.p_gauge[dst_c] = lvl.p_gauge[src_c];
        }
    }

    if (i <= lvl.nxp && j < lvl.nyp && k < lvl.nzp) {
        bool in_x = (i >= h && i <= h + lvl.nx);
        bool in_y = (j >= h && j < h + lvl.ny);
        bool in_z = (k >= h && k < h + lvl.nz);
        if (!(in_x && in_y && in_z)) {
            int ic_ux = min(max(i, h), h + lvl.nx);
            int jc = min(max(j, h), h + lvl.ny - 1);
            int kc = min(max(k, h), h + lvl.nz - 1);
            float sign = 1.0f;
            if (i < h && !(lvl.pml_face_mask & (1u << 0))) {
                ic_ux = min(h + lvl.nx, 2 * h - i);
                sign = -1.0f;
            } else if (i > h + lvl.nx && !(lvl.pml_face_mask & (1u << 1))) {
                ic_ux = max(h, 2 * (h + lvl.nx) - i);
                sign = -1.0f;
            }
            lvl.ux[idx_fx(i, j, k, lvl.nxp, lvl.nyp)] = sign * lvl.ux[idx_fx(ic_ux, jc, kc, lvl.nxp, lvl.nyp)];
        }
    }

    if (i < lvl.nxp && j <= lvl.nyp && k < lvl.nzp) {
        bool in_x = (i >= h && i < h + lvl.nx);
        bool in_y = (j >= h && j <= h + lvl.ny);
        bool in_z = (k >= h && k < h + lvl.nz);
        if (!(in_x && in_y && in_z)) {
            int ic = min(max(i, h), h + lvl.nx - 1);
            int jc_uy = min(max(j, h), h + lvl.ny);
            int kc = min(max(k, h), h + lvl.nz - 1);
            float sign = 1.0f;
            if (j < h && !(lvl.pml_face_mask & (1u << 2))) {
                jc_uy = min(h + lvl.ny, 2 * h - j);
                sign = -1.0f;
            } else if (j > h + lvl.ny && !(lvl.pml_face_mask & (1u << 3))) {
                jc_uy = max(h, 2 * (h + lvl.ny) - j);
                sign = -1.0f;
            }
            lvl.uy[idx_fy(i, j, k, lvl.nxp, lvl.nyp)] = sign * lvl.uy[idx_fy(ic, jc_uy, kc, lvl.nxp, lvl.nyp)];
        }
    }

    if (i < lvl.nxp && j < lvl.nyp && k <= lvl.nzp) {
        bool in_x = (i >= h && i < h + lvl.nx);
        bool in_y = (j >= h && j < h + lvl.ny);
        bool in_z = (k >= h && k <= h + lvl.nz);
        if (!(in_x && in_y && in_z)) {
            int ic = min(max(i, h), h + lvl.nx - 1);
            int jc = min(max(j, h), h + lvl.ny - 1);
            int kc_uz = min(max(k, h), h + lvl.nz);
            float sign = 1.0f;
            if (k < h && !(lvl.pml_face_mask & (1u << 4))) {
                kc_uz = min(h + lvl.nz, 2 * h - k);
                sign = -1.0f;
            } else if (k > h + lvl.nz && !(lvl.pml_face_mask & (1u << 5))) {
                kc_uz = max(h, 2 * (h + lvl.nz) - k);
                sign = -1.0f;
            }
            lvl.uz[idx_fz(i, j, k, lvl.nxp, lvl.nyp)] = sign * lvl.uz[idx_fz(ic, jc, kc_uz, lvl.nxp, lvl.nyp)];
        }
    }
}

__global__ void k_outer_bc_slab_x(FluidLevelGpu lvl) {
    if (lvl.level_idx != 0) return;
    int h = lvl.halo;
    int slab = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (slab >= 2 * h || j >= lvl.nyp || k >= lvl.nzp) return;

    int i = (slab < h) ? slab : (h + lvl.nx + (slab - h));
    int ic = (i < h) ? h : (h + lvl.nx - 1);
    int jc = min(max(j, h), h + lvl.ny - 1);
    int kc = min(max(k, h), h + lvl.nz - 1);

    size_t dst_c = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
    size_t src_c = idx_cell(ic, jc, kc, lvl.nxp, lvl.nyp);
    lvl.alpha[dst_c] = lvl.alpha[src_c];
    lvl.p_gauge[dst_c] = lvl.p_gauge[src_c];

    int i_ux = (slab < h) ? slab : (h + lvl.nx + 1 + (slab - h));
    int ic_ux = (i_ux < h) ? h : (h + lvl.nx);
    float sign = 1.0f;
    if (i_ux < h && !(lvl.pml_face_mask & (1u << 0))) {
        ic_ux = min(h + lvl.nx, 2 * h - i_ux);
        sign = -1.0f;
    } else if (i_ux > h + lvl.nx && !(lvl.pml_face_mask & (1u << 1))) {
        ic_ux = max(h, 2 * (h + lvl.nx) - i_ux);
        sign = -1.0f;
    }
    lvl.ux[idx_fx(i_ux, j, k, lvl.nxp, lvl.nyp)] = sign * lvl.ux[idx_fx(ic_ux, jc, kc, lvl.nxp, lvl.nyp)];

    lvl.uy[idx_fy(i, j, k, lvl.nxp, lvl.nyp)] = lvl.uy[idx_fy(ic, j, k, lvl.nxp, lvl.nyp)];
    lvl.uz[idx_fz(i, j, k, lvl.nxp, lvl.nyp)] = lvl.uz[idx_fz(ic, j, k, lvl.nxp, lvl.nyp)];
    if (j == lvl.nyp - 1) {
        lvl.uy[idx_fy(i, lvl.nyp, k, lvl.nxp, lvl.nyp)] = lvl.uy[idx_fy(ic, lvl.nyp, k, lvl.nxp, lvl.nyp)];
    }
    if (k == lvl.nzp - 1) {
        lvl.uz[idx_fz(i, j, lvl.nzp, lvl.nxp, lvl.nyp)] = lvl.uz[idx_fz(ic, j, lvl.nzp, lvl.nxp, lvl.nyp)];
    }
}

__global__ void k_outer_bc_slab_y(FluidLevelGpu lvl) {
    if (lvl.level_idx != 0) return;
    int h = lvl.halo;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int slab = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= lvl.nxp || slab >= 2 * h || k >= lvl.nzp) return;

    int j = (slab < h) ? slab : (h + lvl.ny + (slab - h));
    int ic = min(max(i, h), h + lvl.nx - 1);
    int jc = (j < h) ? h : (h + lvl.ny - 1);
    int kc = min(max(k, h), h + lvl.nz - 1);

    size_t dst_c = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
    size_t src_c = idx_cell(ic, jc, kc, lvl.nxp, lvl.nyp);
    lvl.alpha[dst_c] = lvl.alpha[src_c];
    lvl.p_gauge[dst_c] = lvl.p_gauge[src_c];

    int j_uy = (slab < h) ? slab : (h + lvl.ny + 1 + (slab - h));
    int jc_uy = (j_uy < h) ? h : (h + lvl.ny);
    float sign = 1.0f;
    if (j_uy < h && !(lvl.pml_face_mask & (1u << 2))) {
        jc_uy = min(h + lvl.ny, 2 * h - j_uy);
        sign = -1.0f;
    } else if (j_uy > h + lvl.ny && !(lvl.pml_face_mask & (1u << 3))) {
        jc_uy = max(h, 2 * (h + lvl.ny) - j_uy);
        sign = -1.0f;
    }
    lvl.uy[idx_fy(i, j_uy, k, lvl.nxp, lvl.nyp)] = sign * lvl.uy[idx_fy(ic, jc_uy, kc, lvl.nxp, lvl.nyp)];

    lvl.ux[idx_fx(i, j, k, lvl.nxp, lvl.nyp)] = lvl.ux[idx_fx(i, jc, k, lvl.nxp, lvl.nyp)];
    if (i == lvl.nxp - 1) {
        lvl.ux[idx_fx(lvl.nxp, j, k, lvl.nxp, lvl.nyp)] = lvl.ux[idx_fx(lvl.nxp, jc, k, lvl.nxp, lvl.nyp)];
    }
    lvl.uz[idx_fz(i, j, k, lvl.nxp, lvl.nyp)] = lvl.uz[idx_fz(i, jc, k, lvl.nxp, lvl.nyp)];
    if (k == lvl.nzp - 1) {
        lvl.uz[idx_fz(i, j, lvl.nzp, lvl.nxp, lvl.nyp)] = lvl.uz[idx_fz(i, jc, lvl.nzp, lvl.nxp, lvl.nyp)];
    }
}

__global__ void k_outer_bc_slab_z(FluidLevelGpu lvl) {
    if (lvl.level_idx != 0) return;
    int h = lvl.halo;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int slab = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= lvl.nxp || j >= lvl.nyp || slab >= 2 * h) return;

    int k = (slab < h) ? slab : (h + lvl.nz + (slab - h));
    int ic = min(max(i, h), h + lvl.nx - 1);
    int jc = min(max(j, h), h + lvl.ny - 1);
    int kc = (k < h) ? h : (h + lvl.nz - 1);

    size_t dst_c = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
    size_t src_c = idx_cell(ic, jc, kc, lvl.nxp, lvl.nyp);
    lvl.alpha[dst_c] = lvl.alpha[src_c];
    lvl.p_gauge[dst_c] = lvl.p_gauge[src_c];

    int k_uz = (slab < h) ? slab : (h + lvl.nz + 1 + (slab - h));
    int kc_uz = (k_uz < h) ? h : (h + lvl.nz);
    float sign = 1.0f;
    if (k_uz < h && !(lvl.pml_face_mask & (1u << 4))) {
        kc_uz = min(h + lvl.nz, 2 * h - k_uz);
        sign = -1.0f;
    } else if (k_uz > h + lvl.nz && !(lvl.pml_face_mask & (1u << 5))) {
        kc_uz = max(h, 2 * (h + lvl.nz) - k_uz);
        sign = -1.0f;
    }
    lvl.uz[idx_fz(i, j, k_uz, lvl.nxp, lvl.nyp)] = sign * lvl.uz[idx_fz(ic, jc, kc_uz, lvl.nxp, lvl.nyp)];

    lvl.ux[idx_fx(i, j, k, lvl.nxp, lvl.nyp)] = lvl.ux[idx_fx(i, j, kc, lvl.nxp, lvl.nyp)];
    if (i == lvl.nxp - 1) {
        lvl.ux[idx_fx(lvl.nxp, j, k, lvl.nxp, lvl.nyp)] = lvl.ux[idx_fx(lvl.nxp, j, kc, lvl.nxp, lvl.nyp)];
    }
    lvl.uy[idx_fy(i, j, k, lvl.nxp, lvl.nyp)] = lvl.uy[idx_fy(i, j, kc, lvl.nxp, lvl.nyp)];
    if (j == lvl.nyp - 1) {
        lvl.uy[idx_fy(i, lvl.nyp, k, lvl.nxp, lvl.nyp)] = lvl.uy[idx_fy(i, lvl.nyp, kc, lvl.nxp, lvl.nyp)];
    }
}

__device__ __forceinline__ float mix_sound_speed_f(float alpha, float p_gauge, const FluidConstantsGpu& c) {
    float a = clamp_f(alpha, 0.0f, 1.0f);
    float rho = a * (float)c.rho_w + (1.0f - a) * (float)c.rho_a;
    float k_mix = mix_bulk_modulus_f(a, p_gauge, c);
    return sqrtf(k_mix / rho);
}

__device__ __forceinline__ bool cell_is_buried_solid(const FluidLevelGpu& lvl, int i, int j, int k) {
    if (i < 0 || i >= lvl.nxp || j < 0 || j >= lvl.nyp || k < 0 || k >= lvl.nzp) return false;
    return lvl.ib_beta_x[idx_fx(i, j, k, lvl.nxp, lvl.nyp)] >= 0.99f &&
           lvl.ib_beta_x[idx_fx(i + 1, j, k, lvl.nxp, lvl.nyp)] >= 0.99f &&
           lvl.ib_beta_y[idx_fy(i, j, k, lvl.nxp, lvl.nyp)] >= 0.99f &&
           lvl.ib_beta_y[idx_fy(i, j + 1, k, lvl.nxp, lvl.nyp)] >= 0.99f &&
           lvl.ib_beta_z[idx_fz(i, j, k, lvl.nxp, lvl.nyp)] >= 0.99f &&
           lvl.ib_beta_z[idx_fz(i, j, k + 1, lvl.nxp, lvl.nyp)] >= 0.99f;
}

__device__ __forceinline__ float compute_wall_impedance_vel(
    const FluidLevelGpu& lvl,
    const FluidConstantsGpu& c,
    int iL, int jL, int kL,
    int iR, int jR, int kR,
    size_t cL, size_t cR,
    float beta_opp_L, float beta_opp_R
) {
    bool bL = cell_is_buried_solid(lvl, iL, jL, kL);
    bool bR = cell_is_buried_solid(lvl, iR, jR, kR);
    int fluid_sign = 0;
    size_t cf = cL;
    if (bL != bR) {
        fluid_sign = bR ? 1 : -1;
        cf = bR ? cL : cR;
    } else if (!bL && !bR) {
        if (beta_opp_R > beta_opp_L + 0.25f) {
            fluid_sign = 1;
            cf = cL;
        } else if (beta_opp_L > beta_opp_R + 0.25f) {
            fluid_sign = -1;
            cf = cR;
        }
    }
    if (fluid_sign == 0) return 0.0f;
    float af = lvl.alpha[cf];
    float p_ac_f = lvl.p_gauge[cf];
    float pg_f = p_ac_f + lvl.p_hydro[cf];
    float cf_loc = mix_sound_speed_f(af, pg_f, c);
    float rhof = clamp_f(af, 0.0f, 1.0f) * (float)(c.rho_w - c.rho_a) + (float)c.rho_a;
    return (float)fluid_sign * c.wall_admittance * p_ac_f / (rhof * cf_loc);
}

__device__ __forceinline__ float effective_p_ac(
    float p_ac, float p_hydro, float alpha, float div_u, const FluidConstantsGpu& c, float dx, float dt_ac
) {
    if (div_u == 0.0f) return p_ac;
    float a_clamped = clamp_f(alpha, 0.0f, 1.0f);
    float rho = a_clamped * (float)(c.rho_w - c.rho_a) + (float)c.rho_a;
    float c_loc = mix_sound_speed_f(a_clamped, p_ac + p_hydro, c);

    // Artificial grid divergence stabilization only where explicitly requested or right on the VOF interface
    bool is_interface = (a_clamped > 1e-3f && a_clamped < 0.999f);
    float coeff = is_interface ? fmaxf(c.div_damp_coeff, 0.020f) : c.div_damp_coeff;
    float p_damp = coeff * rho * c_loc * dx * div_u;

    // Physical Stokes-Kirchhoff thermoviscous acoustic attenuation in single-phase air/water:
    // dp/dt = -K div(u), rho du/dt = -grad(p - rho * nu_ac * div(u))
    if (!is_interface) {
        float nu_ac = 0.0f;
        if (a_clamped <= 1e-3f) {
            float mu_a = (float)c.mu_a;
            float pr = fmaxf(c.prandtl_a, 0.1f);
            float gamma_m1 = (float)(c.gamma_a - 1.0);
            float mu_eff = (4.0f / 3.0f) * mu_a + fmaxf(c.bulk_viscosity_a, 0.0f) + gamma_m1 * mu_a / pr;
            nu_ac = mu_eff / (float)c.rho_a;
        } else {
            nu_ac = (4.0f / 3.0f) * (float)c.mu_w / (float)c.rho_w;
        }
        // Cap effective kinematic viscosity to explicit diffusive stability limit (nu * dt / dx^2 <= 0.12)
        float nu_max = 0.12f * dx * dx / fmaxf(dt_ac, 1e-12f);
        nu_ac = fminf(nu_ac, nu_max);
        p_damp += rho * nu_ac * div_u;
    }

    return p_ac - p_damp;
}

__device__ __forceinline__ bool single_phase_4(float a0, float a1, float a2, float a3) {
    bool all_air = (a0 < 1e-4f && a1 < 1e-4f && a2 < 1e-4f && a3 < 1e-4f);
    bool all_wat = (a0 > 0.9999f && a1 > 0.9999f && a2 > 0.9999f && a3 > 0.9999f);
    return all_air || all_wat;
}

__global__ void k_acoustic_velocity(
    FluidLevelGpu lvl,
    FluidConstantsGpu c,
    float dt_ac
) {
    int h = lvl.halo;
    int i = h + blockIdx.x * blockDim.x + threadIdx.x;
    int j = h + blockIdx.y * blockDim.y + threadIdx.y;
    int k = h + blockIdx.z * blockDim.z + threadIdx.z;

    if (i > h + lvl.nx || j > h + lvl.ny || k > h + lvl.nz) return;

    float inv_dx = 1.0f / lvl.dx;
    int i_int = i - h;
    int j_int = j - h;
    int k_int = k - h;

    // Hoist center cell (i, j, k) loads once and reuse across X, Y, Z faces (saves ~40% global reads)
    bool c_valid = (i < h + lvl.nx && j < h + lvl.ny && k < h + lvl.nz);
    size_t cC = c_valid ? idx_cell(i, j, k, lvl.nxp, lvl.nyp) : 0;
    float aC = c_valid ? lvl.alpha[cC] : 0.0f;
    float pacC = c_valid ? lvl.p_gauge[cC] : 0.0f;
    float phC = c_valid ? lvl.p_hydro[cC] : 0.0f;
    float divC = c_valid ? lvl.div_u[cC] : 0.0f;
    float pEffC = c_valid ? effective_p_ac(pacC, phC, aC, divC, c, lvl.dx, dt_ac) : 0.0f;

    // 1. X-face update (fine levels skip parent-driven boundary normal faces)
    if (j < h + lvl.ny && k < h + lvl.nz && (lvl.level_idx == 0 || (i > h && i < h + lvl.nx))) {
        size_t fx = idx_fx(i, j, k, lvl.nxp, lvl.nyp);
        float beta = lvl.ib_beta_x[fx];
        float v_s = (lvl.ib_vel_x != nullptr) ? lvl.ib_vel_x[fx] : 0.0f;
        float u_val;

        if (lvl.level_idx == 0 && (i == h || i == h + lvl.nx)) {
            bool is_lo = (i == h);
            uint32_t bit = is_lo ? (1u << 0) : (1u << 1);
            if (lvl.pml_face_mask & bit) {
                size_t ci = idx_cell(is_lo ? h : (h + lvl.nx - 1), j, k, lvl.nxp, lvl.nyp);
                float a_c = is_lo ? aC : lvl.alpha[ci];
                float p_ac = is_lo ? pacC : lvl.p_gauge[ci];
                float p_g = p_ac + (is_lo ? phC : lvl.p_hydro[ci]);
                float rho_c = clamp_f(a_c, 0.0f, 1.0f) * (float)(c.rho_w - c.rho_a) + (float)c.rho_a;
                float c_c = mix_sound_speed_f(a_c, p_g, c);
                float u_bc = (is_lo ? -1.0f : 1.0f) * p_ac / (rho_c * c_c);
                u_val = (1.0f - beta) * u_bc + beta * v_s;
            } else {
                u_val = beta * v_s;
            }
        } else {
            size_t cL = idx_cell(i - 1, j, k, lvl.nxp, lvl.nyp);
            size_t cR = cC;
            float aL = lvl.alpha[cL];
            float aR = aC;
            float rho_f = 0.5f * (clamp_f(aL, 0.0f, 1.0f) + clamp_f(aR, 0.0f, 1.0f)) *
                          (float)(c.rho_w - c.rho_a) + (float)c.rho_a;
            float pacL = lvl.p_gauge[cL];
            float pacR = pacC;
            float phL = lvl.p_hydro[cL];
            float phR = phC;
            float d_L = lvl.div_u[cL];
            float pEffL = effective_p_ac(pacL, phL, aL, d_L, c, lvl.dx, dt_ac);
            float pEffR = pEffC;

            bool in_pml = false;
            if (lvl.pml_thickness > 0 && lvl.psi_px != nullptr) {
                in_pml = ((lvl.pml_face_mask & (1u << 0)) && i_int < lvl.pml_thickness) ||
                         ((lvl.pml_face_mask & (1u << 1)) && i_int > lvl.nx - lvl.pml_thickness);
            }

            float dp_ac_dx;
            bool use_4th = !in_pml && beta < 1e-4f && i_int >= 2 && i_int <= lvl.nx - 2;
            if (use_4th) {
                float bL = lvl.ib_beta_x[idx_fx(i - 1, j, k, lvl.nxp, lvl.nyp)];
                float bR = lvl.ib_beta_x[idx_fx(i + 1, j, k, lvl.nxp, lvl.nyp)];
                if (bL < 1e-4f && bR < 1e-4f) {
                    size_t cLL = idx_cell(i - 2, j, k, lvl.nxp, lvl.nyp);
                    size_t cRR = idx_cell(i + 1, j, k, lvl.nxp, lvl.nyp);
                    float aLL = lvl.alpha[cLL];
                    float aRR = lvl.alpha[cRR];
                    if (single_phase_4(aLL, aL, aR, aRR)) {
                        float pEffLL = effective_p_ac(lvl.p_gauge[cLL], lvl.p_hydro[cLL], aLL, lvl.div_u[cLL], c, lvl.dx, dt_ac);
                        float pEffRR = effective_p_ac(lvl.p_gauge[cRR], lvl.p_hydro[cRR], aRR, lvl.div_u[cRR], c, lvl.dx, dt_ac);
                        dp_ac_dx = (1.125f * (pEffR - pEffL) - (1.0f / 24.0f) * (pEffRR - pEffLL)) * inv_dx;
                    } else {
                        dp_ac_dx = (pEffR - pEffL) * inv_dx;
                    }
                } else {
                    dp_ac_dx = (pEffR - pEffL) * inv_dx;
                }
            } else {
                dp_ac_dx = (pEffR - pEffL) * inv_dx;
            }
            float dp_dx = dp_ac_dx + (phR - phL) * inv_dx;

            if (in_pml && beta < 0.5f) {
                float a_f = 0.5f * (aL + aR);
                float p_f = 0.5f * ((pacL + phL) + (pacR + phR));
                float c_loc = mix_sound_speed_f(a_f, p_f, c);
                float sig = compute_pml_sigma(
                    i_int, lvl.nx, lvl.pml_thickness, lvl.pml_face_mask, 0, 1, 0.0f,
                    c_loc, lvl.dx, c.pml_coeff
                );
                if (sig > 0.0f) {
                    float dp_ac = (pacR - pacL) * inv_dx;
                    float rate = c.alpha_cfs + sig;
                    float b = expf(-rate * dt_ac);
                    float a = -sig / rate * (1.0f - b);
                    float psi_old = lvl.psi_px[fx];
                    float psi_new = b * psi_old + a * dp_ac;
                    lvl.psi_px[fx] = psi_new;
                    dp_dx += 0.5f * (psi_old + psi_new);
                }
            }
            float a_s = (lvl.a_slow_x != nullptr) ? lvl.a_slow_x[fx] : 0.0f;
            float accel = -dp_dx / rho_f + c.gx + a_s;
            float u_next = lvl.ux[fx] + dt_ac * accel;
            float v_imp = v_s;
            if (c.wall_admittance > 0.0f && beta >= 0.5f && fabsf(v_s) < 1e-9f) {
                v_imp = compute_wall_impedance_vel(
                    lvl, c, i - 1, j, k, i, j, k, cL, cR,
                    lvl.ib_beta_x[idx_fx(i - 1, j, k, lvl.nxp, lvl.nyp)],
                    lvl.ib_beta_x[idx_fx(i + 1, j, k, lvl.nxp, lvl.nyp)]
                );
            }
            u_val = (1.0f - beta) * u_next + beta * v_imp;
        }
        lvl.ux[fx] = u_val;
    }

    // 2. Y-face update (fine levels skip parent-driven boundary normal faces)
    if (i < h + lvl.nx && k < h + lvl.nz && (lvl.level_idx == 0 || (j > h && j < h + lvl.ny))) {
        size_t fy = idx_fy(i, j, k, lvl.nxp, lvl.nyp);
        float beta = lvl.ib_beta_y[fy];
        float v_s = (lvl.ib_vel_y != nullptr) ? lvl.ib_vel_y[fy] : 0.0f;
        float v_val;

        if (lvl.level_idx == 0 && (j == h || j == h + lvl.ny)) {
            bool is_lo = (j == h);
            uint32_t bit = is_lo ? (1u << 2) : (1u << 3);
            if (lvl.pml_face_mask & bit) {
                size_t ci = idx_cell(i, is_lo ? h : (h + lvl.ny - 1), k, lvl.nxp, lvl.nyp);
                float a_c = is_lo ? aC : lvl.alpha[ci];
                float p_ac = is_lo ? pacC : lvl.p_gauge[ci];
                float p_g = p_ac + (is_lo ? phC : lvl.p_hydro[ci]);
                float rho_c = clamp_f(a_c, 0.0f, 1.0f) * (float)(c.rho_w - c.rho_a) + (float)c.rho_a;
                float c_c = mix_sound_speed_f(a_c, p_g, c);
                float v_bc = (is_lo ? -1.0f : 1.0f) * p_ac / (rho_c * c_c);
                v_val = (1.0f - beta) * v_bc + beta * v_s;
            } else {
                v_val = beta * v_s;
            }
        } else {
            size_t cL = idx_cell(i, j - 1, k, lvl.nxp, lvl.nyp);
            size_t cR = cC;
            float aL = lvl.alpha[cL];
            float aR = aC;
            float rho_f = 0.5f * (clamp_f(aL, 0.0f, 1.0f) + clamp_f(aR, 0.0f, 1.0f)) *
                          (float)(c.rho_w - c.rho_a) + (float)c.rho_a;
            float pacL = lvl.p_gauge[cL];
            float pacR = pacC;
            float phL = lvl.p_hydro[cL];
            float phR = phC;
            float d_L = lvl.div_u[cL];
            float pEffL = effective_p_ac(pacL, phL, aL, d_L, c, lvl.dx, dt_ac);
            float pEffR = pEffC;

            bool in_pml = false;
            if (lvl.pml_thickness > 0 && lvl.psi_py != nullptr) {
                in_pml = ((lvl.pml_face_mask & (1u << 2)) && j_int < lvl.pml_thickness) ||
                         ((lvl.pml_face_mask & (1u << 3)) && j_int > lvl.ny - lvl.pml_thickness);
            }

            float dp_ac_dy;
            bool use_4th = !in_pml && beta < 1e-4f && j_int >= 2 && j_int <= lvl.ny - 2;
            if (use_4th) {
                float bL = lvl.ib_beta_y[idx_fy(i, j - 1, k, lvl.nxp, lvl.nyp)];
                float bR = lvl.ib_beta_y[idx_fy(i, j + 1, k, lvl.nxp, lvl.nyp)];
                if (bL < 1e-4f && bR < 1e-4f) {
                    size_t cLL = idx_cell(i, j - 2, k, lvl.nxp, lvl.nyp);
                    size_t cRR = idx_cell(i, j + 1, k, lvl.nxp, lvl.nyp);
                    float aLL = lvl.alpha[cLL];
                    float aRR = lvl.alpha[cRR];
                    if (single_phase_4(aLL, aL, aR, aRR)) {
                        float pEffLL = effective_p_ac(lvl.p_gauge[cLL], lvl.p_hydro[cLL], aLL, lvl.div_u[cLL], c, lvl.dx, dt_ac);
                        float pEffRR = effective_p_ac(lvl.p_gauge[cRR], lvl.p_hydro[cRR], aRR, lvl.div_u[cRR], c, lvl.dx, dt_ac);
                        dp_ac_dy = (1.125f * (pEffR - pEffL) - (1.0f / 24.0f) * (pEffRR - pEffLL)) * inv_dx;
                    } else {
                        dp_ac_dy = (pEffR - pEffL) * inv_dx;
                    }
                } else {
                    dp_ac_dy = (pEffR - pEffL) * inv_dx;
                }
            } else {
                dp_ac_dy = (pEffR - pEffL) * inv_dx;
            }
            float dp_dy = dp_ac_dy + (phR - phL) * inv_dx;

            if (in_pml && beta < 0.5f) {
                float a_f = 0.5f * (aL + aR);
                float p_f = 0.5f * ((pacL + phL) + (pacR + phR));
                float c_loc = mix_sound_speed_f(a_f, p_f, c);
                float sig = compute_pml_sigma(
                    j_int, lvl.ny, lvl.pml_thickness, lvl.pml_face_mask, 2, 3, 0.0f,
                    c_loc, lvl.dx, c.pml_coeff
                );
                if (sig > 0.0f) {
                    float dp_ac = (pacR - pacL) * inv_dx;
                    float rate = c.alpha_cfs + sig;
                    float b = expf(-rate * dt_ac);
                    float a = -sig / rate * (1.0f - b);
                    float psi_old = lvl.psi_py[fy];
                    float psi_new = b * psi_old + a * dp_ac;
                    lvl.psi_py[fy] = psi_new;
                    dp_dy += 0.5f * (psi_old + psi_new);
                }
            }
            float a_s = (lvl.a_slow_y != nullptr) ? lvl.a_slow_y[fy] : 0.0f;
            float accel = -dp_dy / rho_f + c.gy + a_s;
            float v_next = lvl.uy[fy] + dt_ac * accel;
            float v_imp = v_s;
            if (c.wall_admittance > 0.0f && beta >= 0.5f && fabsf(v_s) < 1e-9f) {
                v_imp = compute_wall_impedance_vel(
                    lvl, c, i, j - 1, k, i, j, k, cL, cR,
                    lvl.ib_beta_y[idx_fy(i, j - 1, k, lvl.nxp, lvl.nyp)],
                    lvl.ib_beta_y[idx_fy(i, j + 1, k, lvl.nxp, lvl.nyp)]
                );
            }
            v_val = (1.0f - beta) * v_next + beta * v_imp;
        }
        lvl.uy[fy] = v_val;
    }

    // 3. Z-face update (fine levels skip parent-driven boundary normal faces)
    if (i < h + lvl.nx && j < h + lvl.ny && (lvl.level_idx == 0 || (k > h && k < h + lvl.nz))) {
        size_t fz = idx_fz(i, j, k, lvl.nxp, lvl.nyp);
        float beta = lvl.ib_beta_z[fz];
        float v_s = (lvl.ib_vel_z != nullptr) ? lvl.ib_vel_z[fz] : 0.0f;
        float w_val;

        if (lvl.level_idx == 0 && (k == h || k == h + lvl.nz)) {
            bool is_lo = (k == h);
            uint32_t bit = is_lo ? (1u << 4) : (1u << 5);
            if (lvl.pml_face_mask & bit) {
                size_t ci = idx_cell(i, j, is_lo ? h : (h + lvl.nz - 1), lvl.nxp, lvl.nyp);
                float a_c = is_lo ? aC : lvl.alpha[ci];
                float p_ac = is_lo ? pacC : lvl.p_gauge[ci];
                float p_g = p_ac + (is_lo ? phC : lvl.p_hydro[ci]);
                float rho_c = clamp_f(a_c, 0.0f, 1.0f) * (float)(c.rho_w - c.rho_a) + (float)c.rho_a;
                float c_c = mix_sound_speed_f(a_c, p_g, c);
                float w_bc = (is_lo ? -1.0f : 1.0f) * p_ac / (rho_c * c_c);
                w_val = (1.0f - beta) * w_bc + beta * v_s;
            } else {
                w_val = beta * v_s;
            }
        } else {
            size_t cL = idx_cell(i, j, k - 1, lvl.nxp, lvl.nyp);
            size_t cR = cC;
            float aL = lvl.alpha[cL];
            float aR = aC;
            float rho_f = 0.5f * (clamp_f(aL, 0.0f, 1.0f) + clamp_f(aR, 0.0f, 1.0f)) *
                          (float)(c.rho_w - c.rho_a) + (float)c.rho_a;
            float pacL = lvl.p_gauge[cL];
            float pacR = pacC;
            float phL = lvl.p_hydro[cL];
            float phR = phC;
            float d_L = lvl.div_u[cL];
            float pEffL = effective_p_ac(pacL, phL, aL, d_L, c, lvl.dx, dt_ac);
            float pEffR = pEffC;

            bool in_pml = false;
            if (lvl.pml_thickness > 0 && lvl.psi_pz != nullptr) {
                in_pml = ((lvl.pml_face_mask & (1u << 4)) && k_int < lvl.pml_thickness) ||
                         ((lvl.pml_face_mask & (1u << 5)) && k_int > lvl.nz - lvl.pml_thickness);
            }

            // Well-balanced vertical gradient: keep hydrostatic (phR - phL) 2nd-order so hydrostatic equilibrium cancels exactly.
            float dp_ac_dz;
            bool use_4th = !in_pml && beta < 1e-4f && k_int >= 2 && k_int <= lvl.nz - 2;
            if (use_4th) {
                float bL = lvl.ib_beta_z[idx_fz(i, j, k - 1, lvl.nxp, lvl.nyp)];
                float bR = lvl.ib_beta_z[idx_fz(i, j, k + 1, lvl.nxp, lvl.nyp)];
                if (bL < 1e-4f && bR < 1e-4f) {
                    size_t cLL = idx_cell(i, j, k - 2, lvl.nxp, lvl.nyp);
                    size_t cRR = idx_cell(i, j, k + 1, lvl.nxp, lvl.nyp);
                    float aLL = lvl.alpha[cLL];
                    float aRR = lvl.alpha[cRR];
                    if (single_phase_4(aLL, aL, aR, aRR)) {
                        float pEffLL = effective_p_ac(lvl.p_gauge[cLL], lvl.p_hydro[cLL], aLL, lvl.div_u[cLL], c, lvl.dx, dt_ac);
                        float pEffRR = effective_p_ac(lvl.p_gauge[cRR], lvl.p_hydro[cRR], aRR, lvl.div_u[cRR], c, lvl.dx, dt_ac);
                        dp_ac_dz = (1.125f * (pEffR - pEffL) - (1.0f / 24.0f) * (pEffRR - pEffLL)) * inv_dx;
                    } else {
                        dp_ac_dz = (pEffR - pEffL) * inv_dx;
                    }
                } else {
                    dp_ac_dz = (pEffR - pEffL) * inv_dx;
                }
            } else {
                dp_ac_dz = (pEffR - pEffL) * inv_dx;
            }
            float dp_hyd_dz = (phR - phL) * inv_dx;

            if (in_pml && beta < 0.5f) {
                float a_f = 0.5f * (aL + aR);
                float p_f = 0.5f * ((pacL + phL) + (pacR + phR));
                float c_loc = mix_sound_speed_f(a_f, p_f, c);
                float sig = compute_pml_sigma(
                    k_int, lvl.nz, lvl.pml_thickness, lvl.pml_face_mask, 4, 5, 0.0f,
                    c_loc, lvl.dx, c.pml_coeff
                );
                if (sig > 0.0f) {
                    float dp_ac = (pacR - pacL) * inv_dx;
                    float rate = c.alpha_cfs + sig;
                    float b = expf(-rate * dt_ac);
                    float a = -sig / rate * (1.0f - b);
                    float psi_old = lvl.psi_pz[fz];
                    float psi_new = b * psi_old + a * dp_ac;
                    lvl.psi_pz[fz] = psi_new;
                    dp_ac_dz += 0.5f * (psi_old + psi_new);
                }
            }
            float a_s = (lvl.a_slow_z != nullptr) ? lvl.a_slow_z[fz] : 0.0f;
            float buoyancy_res = (-dp_hyd_dz + rho_f * c.gz) / rho_f;
            if (fabsf(buoyancy_res) < 1e-3f) buoyancy_res = 0.0f;
            float accel = -dp_ac_dz / rho_f + buoyancy_res + a_s;
            float w_next = lvl.uz[fz] + dt_ac * accel;
            float w_imp = v_s;
            if (c.wall_admittance > 0.0f && beta >= 0.5f && fabsf(v_s) < 1e-9f) {
                w_imp = compute_wall_impedance_vel(
                    lvl, c, i, j, k - 1, i, j, k, cL, cR,
                    lvl.ib_beta_z[idx_fz(i, j, k - 1, lvl.nxp, lvl.nyp)],
                    lvl.ib_beta_z[idx_fz(i, j, k + 1, lvl.nxp, lvl.nyp)]
                );
            }
            w_val = (1.0f - beta) * w_next + beta * w_imp;
        }
        lvl.uz[fz] = w_val;
    }
}

__global__ void k_acoustic_pressure(
    FluidLevelGpu lvl,
    FluidConstantsGpu c,
    float dt_ac
) {
    int h = lvl.halo;
    int i_int = blockIdx.x * blockDim.x + threadIdx.x;
    int j_int = blockIdx.y * blockDim.y + threadIdx.y;
    int k_int = blockIdx.z * blockDim.z + threadIdx.z;
    if (i_int >= lvl.nx || j_int >= lvl.ny || k_int >= lvl.nz) return;

    int i = h + i_int;
    int j = h + j_int;
    int k = h + k_int;
    size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);

    if (cell_is_buried_solid(lvl, i, j, k)) {
        lvl.div_u[cid] = 0.0f;
        lvl.p_gauge[cid] = 0.0f;
        return;
    }

    float inv_dx = 1.0f / lvl.dx;
    float alpha_c = lvl.alpha[cid];
    float pac_old = lvl.p_gauge[cid];
    float p_g_old = pac_old + lvl.p_hydro[cid];

    bool in_pml = false;
    if (lvl.pml_thickness > 0 && lvl.psi_ux != nullptr) {
        in_pml =
            ((lvl.pml_face_mask & 3u) && (i_int < lvl.pml_thickness || i_int >= lvl.nx - lvl.pml_thickness)) ||
            ((lvl.pml_face_mask & 12u) && (j_int < lvl.pml_thickness || j_int >= lvl.ny - lvl.pml_thickness)) ||
            ((lvl.pml_face_mask & 48u) && (k_int < lvl.pml_thickness || k_int >= lvl.nz - lvl.pml_thickness));
    }

    bool single_c = (alpha_c < 1e-4f || alpha_c > 0.9999f);
    bool is_water_c = (alpha_c > 0.9999f);

    // X divergence (4th-order staggered in single-phase interior away from IB/PML)
    float u_xm = lvl.ux[idx_fx(i, j, k, lvl.nxp, lvl.nyp)];
    float u_xp = lvl.ux[idx_fx(i + 1, j, k, lvl.nxp, lvl.nyp)];
    float dux = (u_xp - u_xm) * inv_dx;
    if (!in_pml && single_c && i_int >= 1 && i_int <= lvl.nx - 2) {
        float bx0 = lvl.ib_beta_x[idx_fx(i - 1, j, k, lvl.nxp, lvl.nyp)];
        float bx1 = lvl.ib_beta_x[idx_fx(i, j, k, lvl.nxp, lvl.nyp)];
        float bx2 = lvl.ib_beta_x[idx_fx(i + 1, j, k, lvl.nxp, lvl.nyp)];
        float bx3 = lvl.ib_beta_x[idx_fx(i + 2, j, k, lvl.nxp, lvl.nyp)];
        if (bx0 < 1e-4f && bx1 < 1e-4f && bx2 < 1e-4f && bx3 < 1e-4f) {
            float aL = lvl.alpha[idx_cell(i - 1, j, k, lvl.nxp, lvl.nyp)];
            float aR = lvl.alpha[idx_cell(i + 1, j, k, lvl.nxp, lvl.nyp)];
            bool same_phase = is_water_c ? (aL > 0.9999f && aR > 0.9999f) : (aL < 1e-4f && aR < 1e-4f);
            if (same_phase) {
                float u_xmm = lvl.ux[idx_fx(i - 1, j, k, lvl.nxp, lvl.nyp)];
                float u_xpp = lvl.ux[idx_fx(i + 2, j, k, lvl.nxp, lvl.nyp)];
                dux = (1.125f * (u_xp - u_xm) - (1.0f / 24.0f) * (u_xpp - u_xmm)) * inv_dx;
            }
        }
    }

    // Y divergence
    float u_ym = lvl.uy[idx_fy(i, j, k, lvl.nxp, lvl.nyp)];
    float u_yp = lvl.uy[idx_fy(i, j + 1, k, lvl.nxp, lvl.nyp)];
    float duy = (u_yp - u_ym) * inv_dx;
    if (!in_pml && single_c && j_int >= 1 && j_int <= lvl.ny - 2) {
        float by0 = lvl.ib_beta_y[idx_fy(i, j - 1, k, lvl.nxp, lvl.nyp)];
        float by1 = lvl.ib_beta_y[idx_fy(i, j, k, lvl.nxp, lvl.nyp)];
        float by2 = lvl.ib_beta_y[idx_fy(i, j + 1, k, lvl.nxp, lvl.nyp)];
        float by3 = lvl.ib_beta_y[idx_fy(i, j + 2, k, lvl.nxp, lvl.nyp)];
        if (by0 < 1e-4f && by1 < 1e-4f && by2 < 1e-4f && by3 < 1e-4f) {
            float aL = lvl.alpha[idx_cell(i, j - 1, k, lvl.nxp, lvl.nyp)];
            float aR = lvl.alpha[idx_cell(i, j + 1, k, lvl.nxp, lvl.nyp)];
            bool same_phase = is_water_c ? (aL > 0.9999f && aR > 0.9999f) : (aL < 1e-4f && aR < 1e-4f);
            if (same_phase) {
                float u_ymm = lvl.uy[idx_fy(i, j - 1, k, lvl.nxp, lvl.nyp)];
                float u_ypp = lvl.uy[idx_fy(i, j + 2, k, lvl.nxp, lvl.nyp)];
                duy = (1.125f * (u_yp - u_ym) - (1.0f / 24.0f) * (u_ypp - u_ymm)) * inv_dx;
            }
        }
    }

    // Z divergence
    float u_zm = lvl.uz[idx_fz(i, j, k, lvl.nxp, lvl.nyp)];
    float u_zp = lvl.uz[idx_fz(i, j, k + 1, lvl.nxp, lvl.nyp)];
    float duz = (u_zp - u_zm) * inv_dx;
    if (!in_pml && single_c && k_int >= 1 && k_int <= lvl.nz - 2) {
        float bz0 = lvl.ib_beta_z[idx_fz(i, j, k - 1, lvl.nxp, lvl.nyp)];
        float bz1 = lvl.ib_beta_z[idx_fz(i, j, k, lvl.nxp, lvl.nyp)];
        float bz2 = lvl.ib_beta_z[idx_fz(i, j, k + 1, lvl.nxp, lvl.nyp)];
        float bz3 = lvl.ib_beta_z[idx_fz(i, j, k + 2, lvl.nxp, lvl.nyp)];
        if (bz0 < 1e-4f && bz1 < 1e-4f && bz2 < 1e-4f && bz3 < 1e-4f) {
            float aL = lvl.alpha[idx_cell(i, j, k - 1, lvl.nxp, lvl.nyp)];
            float aR = lvl.alpha[idx_cell(i, j, k + 1, lvl.nxp, lvl.nyp)];
            bool same_phase = is_water_c ? (aL > 0.9999f && aR > 0.9999f) : (aL < 1e-4f && aR < 1e-4f);
            if (same_phase) {
                float u_zmm = lvl.uz[idx_fz(i, j, k - 1, lvl.nxp, lvl.nyp)];
                float u_zpp = lvl.uz[idx_fz(i, j, k + 2, lvl.nxp, lvl.nyp)];
                duz = (1.125f * (u_zp - u_zm) - (1.0f / 24.0f) * (u_zpp - u_zmm)) * inv_dx;
            }
        }
    }

    if (in_pml) {
        float c_loc = mix_sound_speed_f(alpha_c, p_g_old, c);

        float sx = compute_pml_sigma(i_int, lvl.nx, lvl.pml_thickness, lvl.pml_face_mask, 0, 1, 0.5f, c_loc, lvl.dx, c.pml_coeff);
        if (sx > 0.0f) {
            float rate = c.alpha_cfs + sx;
            float b = expf(-rate * dt_ac);
            float a = -sx / rate * (1.0f - b);
            float psi_old = lvl.psi_ux[cid];
            float psi_new = b * psi_old + a * dux;
            lvl.psi_ux[cid] = psi_new;
            dux += 0.5f * (psi_old + psi_new);
        }

        float sy = compute_pml_sigma(j_int, lvl.ny, lvl.pml_thickness, lvl.pml_face_mask, 2, 3, 0.5f, c_loc, lvl.dx, c.pml_coeff);
        if (sy > 0.0f) {
            float rate = c.alpha_cfs + sy;
            float b = expf(-rate * dt_ac);
            float a = -sy / rate * (1.0f - b);
            float psi_old = lvl.psi_uy[cid];
            float psi_new = b * psi_old + a * duy;
            lvl.psi_uy[cid] = psi_new;
            duy += 0.5f * (psi_old + psi_new);
        }

        float sz = compute_pml_sigma(k_int, lvl.nz, lvl.pml_thickness, lvl.pml_face_mask, 4, 5, 0.5f, c_loc, lvl.dx, c.pml_coeff);
        if (sz > 0.0f) {
            float rate = c.alpha_cfs + sz;
            float b = expf(-rate * dt_ac);
            float a = -sz / rate * (1.0f - b);
            float psi_old = lvl.psi_uz[cid];
            float psi_new = b * psi_old + a * duz;
            lvl.psi_uz[cid] = psi_new;
            duz += 0.5f * (psi_old + psi_new);
        }
    }

    float div = dux + duy + duz;
    lvl.div_u[cid] = div;
    float k_mix = mix_bulk_modulus_f(alpha_c, p_g_old, c);
    float src = (lvl.src_p != nullptr) ? lvl.src_p[cid] : 0.0f;
    lvl.p_gauge[cid] = pac_old - dt_ac * k_mix * div + dt_ac * src;
}
