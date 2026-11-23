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
    float p_ac, float p_hydro, float alpha, float div_u, const FluidConstantsGpu& c, float dx
) {
    float coeff = (alpha > 1e-5f) ? fmaxf(c.div_damp_coeff, 0.025f) : c.div_damp_coeff;
    if (coeff <= 0.0f || div_u == 0.0f) return p_ac;
    float rho = clamp_f(alpha, 0.0f, 1.0f) * (float)(c.rho_w - c.rho_a) + (float)c.rho_a;
    float c_loc = mix_sound_speed_f(alpha, p_ac + p_hydro, c);
    return p_ac - coeff * rho * c_loc * dx * div_u;
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
                float a_c = lvl.alpha[ci];
                float p_ac = lvl.p_gauge[ci];
                float p_g = p_ac + lvl.p_hydro[ci];
                float rho_c = clamp_f(a_c, 0.0f, 1.0f) * (float)(c.rho_w - c.rho_a) + (float)c.rho_a;
                float c_c = mix_sound_speed_f(a_c, p_g, c);
                float u_bc = (is_lo ? -1.0f : 1.0f) * p_ac / (rho_c * c_c);
                u_val = (1.0f - beta) * u_bc + beta * v_s;
            } else {
                u_val = beta * v_s;
            }
        } else {
            size_t cL = idx_cell(i - 1, j, k, lvl.nxp, lvl.nyp);
            size_t cR = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
            float aL = lvl.alpha[cL];
            float aR = lvl.alpha[cR];
            float rho_f = 0.5f * (clamp_f(aL, 0.0f, 1.0f) + clamp_f(aR, 0.0f, 1.0f)) *
                          (float)(c.rho_w - c.rho_a) + (float)c.rho_a;
            float pacL = lvl.p_gauge[cL];
            float pacR = lvl.p_gauge[cR];
            float phL = lvl.p_hydro[cL];
            float phR = lvl.p_hydro[cR];
            float d_L = (c.div_damp_coeff > 0.0f || aL > 1e-5f) ? lvl.div_u[cL] : 0.0f;
            float d_R = (c.div_damp_coeff > 0.0f || aR > 1e-5f) ? lvl.div_u[cR] : 0.0f;
            float dp_dx = (effective_p_ac(pacR, phR, aR, d_R, c, lvl.dx) -
                           effective_p_ac(pacL, phL, aL, d_L, c, lvl.dx) +
                           (phR - phL)) * inv_dx;

            if (lvl.pml_thickness > 0 && lvl.psi_px != nullptr && beta < 0.5f) {
                bool in_pml = ((lvl.pml_face_mask & (1u << 0)) && i_int < lvl.pml_thickness) ||
                              ((lvl.pml_face_mask & (1u << 1)) && i_int > lvl.nx - lvl.pml_thickness);
                if (in_pml) {
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

        if (lvl.level_idx > 0) {
            if (i == h && lvl.bnd_vflux_xlo != nullptr) {
                lvl.bnd_vflux_xlo[j_int + lvl.ny * k_int] += u_val * dt_ac;
            } else if (i == h + lvl.nx && lvl.bnd_vflux_xhi != nullptr) {
                lvl.bnd_vflux_xhi[j_int + lvl.ny * k_int] += u_val * dt_ac;
            }
        }
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
                float a_c = lvl.alpha[ci];
                float p_ac = lvl.p_gauge[ci];
                float p_g = p_ac + lvl.p_hydro[ci];
                float rho_c = clamp_f(a_c, 0.0f, 1.0f) * (float)(c.rho_w - c.rho_a) + (float)c.rho_a;
                float c_c = mix_sound_speed_f(a_c, p_g, c);
                float v_bc = (is_lo ? -1.0f : 1.0f) * p_ac / (rho_c * c_c);
                v_val = (1.0f - beta) * v_bc + beta * v_s;
            } else {
                v_val = beta * v_s;
            }
        } else {
            size_t cL = idx_cell(i, j - 1, k, lvl.nxp, lvl.nyp);
            size_t cR = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
            float aL = lvl.alpha[cL];
            float aR = lvl.alpha[cR];
            float rho_f = 0.5f * (clamp_f(aL, 0.0f, 1.0f) + clamp_f(aR, 0.0f, 1.0f)) *
                          (float)(c.rho_w - c.rho_a) + (float)c.rho_a;
            float pacL = lvl.p_gauge[cL];
            float pacR = lvl.p_gauge[cR];
            float phL = lvl.p_hydro[cL];
            float phR = lvl.p_hydro[cR];
            float d_L = (c.div_damp_coeff > 0.0f || aL > 1e-5f) ? lvl.div_u[cL] : 0.0f;
            float d_R = (c.div_damp_coeff > 0.0f || aR > 1e-5f) ? lvl.div_u[cR] : 0.0f;
            float dp_dy = (effective_p_ac(pacR, phR, aR, d_R, c, lvl.dx) -
                           effective_p_ac(pacL, phL, aL, d_L, c, lvl.dx) +
                           (phR - phL)) * inv_dx;

            if (lvl.pml_thickness > 0 && lvl.psi_py != nullptr && beta < 0.5f) {
                bool in_pml = ((lvl.pml_face_mask & (1u << 2)) && j_int < lvl.pml_thickness) ||
                              ((lvl.pml_face_mask & (1u << 3)) && j_int > lvl.ny - lvl.pml_thickness);
                if (in_pml) {
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

        if (lvl.level_idx > 0) {
            if (j == h && lvl.bnd_vflux_ylo != nullptr) {
                lvl.bnd_vflux_ylo[i_int + lvl.nx * k_int] += v_val * dt_ac;
            } else if (j == h + lvl.ny && lvl.bnd_vflux_yhi != nullptr) {
                lvl.bnd_vflux_yhi[i_int + lvl.nx * k_int] += v_val * dt_ac;
            }
        }
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
                float a_c = lvl.alpha[ci];
                float p_ac = lvl.p_gauge[ci];
                float p_g = p_ac + lvl.p_hydro[ci];
                float rho_c = clamp_f(a_c, 0.0f, 1.0f) * (float)(c.rho_w - c.rho_a) + (float)c.rho_a;
                float c_c = mix_sound_speed_f(a_c, p_g, c);
                float w_bc = (is_lo ? -1.0f : 1.0f) * p_ac / (rho_c * c_c);
                w_val = (1.0f - beta) * w_bc + beta * v_s;
            } else {
                w_val = beta * v_s;
            }
        } else {
            size_t cL = idx_cell(i, j, k - 1, lvl.nxp, lvl.nyp);
            size_t cR = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
            float aL = lvl.alpha[cL];
            float aR = lvl.alpha[cR];
            float rho_f = 0.5f * (clamp_f(aL, 0.0f, 1.0f) + clamp_f(aR, 0.0f, 1.0f)) *
                          (float)(c.rho_w - c.rho_a) + (float)c.rho_a;
            float pacL = lvl.p_gauge[cL];
            float pacR = lvl.p_gauge[cR];
            float phL = lvl.p_hydro[cL];
            float phR = lvl.p_hydro[cR];
            float d_L = (c.div_damp_coeff > 0.0f || aL > 1e-5f) ? lvl.div_u[cL] : 0.0f;
            float d_R = (c.div_damp_coeff > 0.0f || aR > 1e-5f) ? lvl.div_u[cR] : 0.0f;
            // Well-balanced vertical gradient: in hydrostatic equilibrium, (phR - phL)*inv_dx / rho_f == c.gz,
            // so separating (pacR - pacL) from the hydrostatic buoyancy residual eliminates FP32 cancellation!
            float dp_ac_dz = (effective_p_ac(pacR, phR, aR, d_R, c, lvl.dx) -
                              effective_p_ac(pacL, phL, aL, d_L, c, lvl.dx)) * inv_dx;
            float dp_hyd_dz = (phR - phL) * inv_dx;

            if (lvl.pml_thickness > 0 && lvl.psi_pz != nullptr && beta < 0.5f) {
                bool in_pml = ((lvl.pml_face_mask & (1u << 4)) && k_int < lvl.pml_thickness) ||
                              ((lvl.pml_face_mask & (1u << 5)) && k_int > lvl.nz - lvl.pml_thickness);
                if (in_pml) {
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

        if (lvl.level_idx > 0) {
            if (k == h && lvl.bnd_vflux_zlo != nullptr) {
                lvl.bnd_vflux_zlo[i_int + lvl.nx * j_int] += w_val * dt_ac;
            } else if (k == h + lvl.nz && lvl.bnd_vflux_zhi != nullptr) {
                lvl.bnd_vflux_zhi[i_int + lvl.nx * j_int] += w_val * dt_ac;
            }
        }
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
    float dux = (lvl.ux[idx_fx(i + 1, j, k, lvl.nxp, lvl.nyp)] -
                 lvl.ux[idx_fx(i, j, k, lvl.nxp, lvl.nyp)]) * inv_dx;
    float duy = (lvl.uy[idx_fy(i, j + 1, k, lvl.nxp, lvl.nyp)] -
                 lvl.uy[idx_fy(i, j, k, lvl.nxp, lvl.nyp)]) * inv_dx;
    float duz = (lvl.uz[idx_fz(i, j, k + 1, lvl.nxp, lvl.nyp)] -
                 lvl.uz[idx_fz(i, j, k, lvl.nxp, lvl.nyp)]) * inv_dx;

    float alpha_c = lvl.alpha[cid];
    float pac_old = lvl.p_gauge[cid];
    float p_g_old = pac_old + lvl.p_hydro[cid];

    if (lvl.pml_thickness > 0 && lvl.psi_ux != nullptr) {
        bool in_pml =
            ((lvl.pml_face_mask & 3u) && (i_int < lvl.pml_thickness || i_int >= lvl.nx - lvl.pml_thickness)) ||
            ((lvl.pml_face_mask & 12u) && (j_int < lvl.pml_thickness || j_int >= lvl.ny - lvl.pml_thickness)) ||
            ((lvl.pml_face_mask & 48u) && (k_int < lvl.pml_thickness || k_int >= lvl.nz - lvl.pml_thickness));
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
    }

    float div = dux + duy + duz;
    lvl.div_u[cid] = div;
    float k_mix = mix_bulk_modulus_f(alpha_c, p_g_old, c);
    float src = (lvl.src_p != nullptr) ? lvl.src_p[cid] : 0.0f;
    lvl.p_gauge[cid] = pac_old - dt_ac * k_mix * div + dt_ac * src;
}
