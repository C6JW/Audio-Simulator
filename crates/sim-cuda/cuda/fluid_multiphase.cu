#include "common.cuh"

__device__ __forceinline__ float thinc_right_integral(float a_d, float cf_abs, float beta) {
    if (cf_abs <= 1e-7f) return 0.0f;
    if (a_d <= 1e-6f) return 0.0f;
    if (a_d >= 1.0f - 1e-6f) return cf_abs;
    float b2 = expf(2.0f * beta);
    float ea = expf(2.0f * beta * a_d);
    float denom = b2 - ea;
    if (denom <= 1e-7f) return cf_abs;
    float a2 = (ea - 1.0f) / denom;
    float num = a2 * b2 + 1.0f;
    float den = a2 * expf(2.0f * beta * (1.0f - cf_abs)) + 1.0f;
    return (0.5f / beta) * logf(num / den);
}

__device__ __forceinline__ void compute_youngs_normal(
    const float* alpha,
    int i, int j, int k,
    int nxp, int nyp,
    float& nx, float& ny, float& nz
) {
    // Parker-Youngs 3x3x3 centered gradient of alpha (native FP32)
    float gx = 0.0f, gy = 0.0f, gz = 0.0f;
    for (int dk = -1; dk <= 1; ++dk) {
        float wk = (dk == 0) ? 2.0f : 1.0f;
        for (int dj = -1; dj <= 1; ++dj) {
            float wj = (dj == 0) ? 2.0f : 1.0f;
            float w = wk * wj;
            float a_p = alpha[idx_cell(i + 1, j + dj, k + dk, nxp, nyp)];
            float a_m = alpha[idx_cell(i - 1, j + dj, k + dk, nxp, nyp)];
            gx += w * (a_p - a_m);
        }
    }
    for (int dk = -1; dk <= 1; ++dk) {
        float wk = (dk == 0) ? 2.0f : 1.0f;
        for (int di = -1; di <= 1; ++di) {
            float wi = (di == 0) ? 2.0f : 1.0f;
            float w = wk * wi;
            float a_p = alpha[idx_cell(i + di, j + 1, k + dk, nxp, nyp)];
            float a_m = alpha[idx_cell(i + di, j - 1, k + dk, nxp, nyp)];
            gy += w * (a_p - a_m);
        }
    }
    for (int dj = -1; dj <= 1; ++dj) {
        float wj = (dj == 0) ? 2.0f : 1.0f;
        for (int di = -1; di <= 1; ++di) {
            float wi = (di == 0) ? 2.0f : 1.0f;
            float w = wj * wi;
            float a_p = alpha[idx_cell(i + di, j + dj, k + 1, nxp, nyp)];
            float a_m = alpha[idx_cell(i + di, j + dj, k - 1, nxp, nyp)];
            gz += w * (a_p - a_m);
        }
    }
    float norm = sqrtf(gx * gx + gy * gy + gz * gz);
    if (norm > 1e-7f) {
        float inv_n = 1.0f / norm;
        nx = gx * inv_n;
        ny = gy * inv_n;
        nz = gz * inv_n;
    } else {
        nx = 0.0f;
        ny = 0.0f;
        nz = 1.0f;
    }
}

__device__ __forceinline__ float mthinc_face_flux(
    const float* alpha,
    int i_d, int j_d, int k_d,
    int nxp, int nyp,
    int axis,
    float cf,
    float beta
) {
    float a_d = clamp_f(alpha[idx_cell(i_d, j_d, k_d, nxp, nyp)], 0.0f, 1.0f);
    float cf_abs = fabsf(cf);
    if (cf_abs <= 1e-7f || a_d <= 1e-6f) return 0.0f;
    if (a_d >= 1.0f - 1e-6f) return cf;

    float nx, ny, nz;
    compute_youngs_normal(alpha, i_d, j_d, k_d, nxp, nyp, nx, ny, nz);
    float n_d = (axis == 0) ? nx : ((axis == 1) ? ny : nz);
    float abs_nd = fabsf(n_d);

    float f_thinc;
    if (cf >= 0.0f) {
        // Exiting right face of donor D: [1 - cf_abs, 1]
        if (n_d >= 0.0f) {
            f_thinc = thinc_right_integral(a_d, cf_abs, beta);
        } else {
            f_thinc = cf_abs - thinc_right_integral(1.0f - a_d, cf_abs, beta);
        }
    } else {
        // Exiting left face of donor D: [0, cf_abs]
        if (n_d >= 0.0f) {
            f_thinc = a_d - thinc_right_integral(a_d, 1.0f - cf_abs, beta);
        } else {
            f_thinc = cf_abs - ((1.0f - a_d) - thinc_right_integral(1.0f - a_d, 1.0f - cf_abs, beta));
        }
    }

    float f_abs = abs_nd * f_thinc + (1.0f - abs_nd) * (a_d * cf_abs);
    f_abs = clamp_f(f_abs, 0.0f, fminf(a_d, cf_abs));
    return (cf >= 0.0f) ? f_abs : -f_abs;
}

__global__ void k_vof_save_start_and_div(FluidLevelGpu lvl) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= lvl.nxp || j >= lvl.nyp || k >= lvl.nzp) return;

    size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
    lvl.alpha_smooth[cid] = lvl.alpha[cid]; // temporary storage for alpha_start

    int h = lvl.halo;
    if (i >= h && i < h + lvl.nx && j >= h && j < h + lvl.ny && k >= h && k < h + lvl.nz) {
        float inv_dx = 1.0f / lvl.dx;
        float dux = (lvl.ux[idx_fx(i + 1, j, k, lvl.nxp, lvl.nyp)] -
                     lvl.ux[idx_fx(i, j, k, lvl.nxp, lvl.nyp)]) * inv_dx;
        float duy = (lvl.uy[idx_fy(i, j + 1, k, lvl.nxp, lvl.nyp)] -
                     lvl.uy[idx_fy(i, j, k, lvl.nxp, lvl.nyp)]) * inv_dx;
        float duz = (lvl.uz[idx_fz(i, j, k + 1, lvl.nxp, lvl.nyp)] -
                     lvl.uz[idx_fz(i, j, k, lvl.nxp, lvl.nyp)]) * inv_dx;
        lvl.div_u[cid] = dux + duy + duz;
    } else {
        lvl.div_u[cid] = 0.0f;
    }
}

__global__ void k_vof_flux_x(FluidLevelGpu lvl, FluidConstantsGpu c, float dt_flow) {
    int h = lvl.halo;
    int i = h + blockIdx.x * blockDim.x + threadIdx.x;
    int j = h + blockIdx.y * blockDim.y + threadIdx.y;
    int k = h + blockIdx.z * blockDim.z + threadIdx.z;
    if (i > h + lvl.nx || j >= h + lvl.ny || k >= h + lvl.nz) return;

    size_t fx = idx_fx(i, j, k, lvl.nxp, lvl.nyp);
    float cf = clamp_f(lvl.ux[fx] * dt_flow / lvl.dx, -0.49f, 0.49f);
    float b_ib = lvl.ib_beta_x[fx];
    if (b_ib >= 0.30f) cf *= clamp_f(1.0f - (b_ib - 0.30f) * 4.0f, 0.0f, 1.0f);
    int i_d = (cf >= 0.0f) ? (i - 1) : i;
    float f_a = mthinc_face_flux(lvl.alpha, i_d, j, k, lvl.nxp, lvl.nyp, 0, cf, c.vof_beta);
    lvl.vflux_x[fx] = f_a;
    float rho_w = (float)c.rho_w;
    float rho_a = (float)c.rho_a;
    lvl.mflux_x[fx] = (rho_w * f_a + rho_a * (cf - f_a)) * lvl.dx;

    if (lvl.level_idx > 0) {
        int j_int = j - h, k_int = k - h;
        if (i == h && lvl.bnd_aflux_xlo != nullptr) {
            lvl.bnd_aflux_xlo[j_int + lvl.ny * k_int] += f_a * lvl.dx;
        } else if (i == h + lvl.nx && lvl.bnd_aflux_xhi != nullptr) {
            lvl.bnd_aflux_xhi[j_int + lvl.ny * k_int] += f_a * lvl.dx;
        }
    }
}

__global__ void k_vof_flux_y(FluidLevelGpu lvl, FluidConstantsGpu c, float dt_flow) {
    int h = lvl.halo;
    int i = h + blockIdx.x * blockDim.x + threadIdx.x;
    int j = h + blockIdx.y * blockDim.y + threadIdx.y;
    int k = h + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= h + lvl.nx || j > h + lvl.ny || k >= h + lvl.nz) return;

    size_t fy = idx_fy(i, j, k, lvl.nxp, lvl.nyp);
    float cf = clamp_f(lvl.uy[fy] * dt_flow / lvl.dx, -0.49f, 0.49f);
    float b_ib = lvl.ib_beta_y[fy];
    if (b_ib >= 0.30f) cf *= clamp_f(1.0f - (b_ib - 0.30f) * 4.0f, 0.0f, 1.0f);
    int j_d = (cf >= 0.0f) ? (j - 1) : j;
    float f_a = mthinc_face_flux(lvl.alpha, i, j_d, k, lvl.nxp, lvl.nyp, 1, cf, c.vof_beta);
    lvl.vflux_y[fy] = f_a;
    float rho_w = (float)c.rho_w;
    float rho_a = (float)c.rho_a;
    lvl.mflux_y[fy] = (rho_w * f_a + rho_a * (cf - f_a)) * lvl.dx;

    if (lvl.level_idx > 0) {
        int i_int = i - h, k_int = k - h;
        if (j == h && lvl.bnd_aflux_ylo != nullptr) {
            lvl.bnd_aflux_ylo[i_int + lvl.nx * k_int] += f_a * lvl.dx;
        } else if (j == h + lvl.ny && lvl.bnd_aflux_yhi != nullptr) {
            lvl.bnd_aflux_yhi[i_int + lvl.nx * k_int] += f_a * lvl.dx;
        }
    }
}

__global__ void k_vof_flux_z(FluidLevelGpu lvl, FluidConstantsGpu c, float dt_flow) {
    int h = lvl.halo;
    int i = h + blockIdx.x * blockDim.x + threadIdx.x;
    int j = h + blockIdx.y * blockDim.y + threadIdx.y;
    int k = h + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= h + lvl.nx || j >= h + lvl.ny || k > h + lvl.nz) return;

    size_t fz = idx_fz(i, j, k, lvl.nxp, lvl.nyp);
    float cf = clamp_f(lvl.uz[fz] * dt_flow / lvl.dx, -0.49f, 0.49f);
    float b_ib = lvl.ib_beta_z[fz];
    if (b_ib >= 0.30f) cf *= clamp_f(1.0f - (b_ib - 0.30f) * 4.0f, 0.0f, 1.0f);
    int k_d = (cf >= 0.0f) ? (k - 1) : k;
    float f_a = mthinc_face_flux(lvl.alpha, i, j, k_d, lvl.nxp, lvl.nyp, 2, cf, c.vof_beta);
    lvl.vflux_z[fz] = f_a;
    float rho_w = (float)c.rho_w;
    float rho_a = (float)c.rho_a;
    lvl.mflux_z[fz] = (rho_w * f_a + rho_a * (cf - f_a)) * lvl.dx;

    if (lvl.level_idx > 0) {
        int i_int = i - h, j_int = j - h;
        if (k == h && lvl.bnd_aflux_zlo != nullptr) {
            lvl.bnd_aflux_zlo[i_int + lvl.nx * j_int] += f_a * lvl.dx;
        } else if (k == h + lvl.nz && lvl.bnd_aflux_zhi != nullptr) {
            lvl.bnd_aflux_zhi[i_int + lvl.nx * j_int] += f_a * lvl.dx;
        }
    }
}

__global__ void k_vof_update_axis(FluidLevelGpu lvl, float dt_flow, int axis) {
    int h = lvl.halo;
    int i = h + blockIdx.x * blockDim.x + threadIdx.x;
    int j = h + blockIdx.y * blockDim.y + threadIdx.y;
    int k = h + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= h + lvl.nx || j >= h + lvl.ny || k >= h + lvl.nz) return;

    size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
    float a_old = lvl.alpha[cid];
    float a_start = lvl.alpha_smooth[cid];
    float d_flux = 0.0f, d_u = 0.0f;
    float dt_dx = dt_flow / lvl.dx;

    if (axis == 0) {
        size_t f_lo = idx_fx(i, j, k, lvl.nxp, lvl.nyp);
        size_t f_hi = idx_fx(i + 1, j, k, lvl.nxp, lvl.nyp);
        d_flux = lvl.vflux_x[f_hi] - lvl.vflux_x[f_lo];
        d_u = (lvl.ux[f_hi] - lvl.ux[f_lo]) * dt_dx;
    } else if (axis == 1) {
        size_t f_lo = idx_fy(i, j, k, lvl.nxp, lvl.nyp);
        size_t f_hi = idx_fy(i, j + 1, k, lvl.nxp, lvl.nyp);
        d_flux = lvl.vflux_y[f_hi] - lvl.vflux_y[f_lo];
        d_u = (lvl.uy[f_hi] - lvl.uy[f_lo]) * dt_dx;
    } else {
        size_t f_lo = idx_fz(i, j, k, lvl.nxp, lvl.nyp);
        size_t f_hi = idx_fz(i, j, k + 1, lvl.nxp, lvl.nyp);
        d_flux = lvl.vflux_z[f_hi] - lvl.vflux_z[f_lo];
        d_u = (lvl.uz[f_hi] - lvl.uz[f_lo]) * dt_dx;
    }

    float a_new = a_old - d_flux + a_start * d_u;
    lvl.alpha_tmp[cid] = clamp_f(a_new, 0.0f, 1.0f);
}

__global__ void k_vof_copy_tmp_and_bc(FluidLevelGpu lvl) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= lvl.nxp || j >= lvl.nyp || k >= lvl.nzp) return;

    int h = lvl.halo;
    if (i >= h && i < h + lvl.nx && j >= h && j < h + lvl.ny && k >= h && k < h + lvl.nz) {
        size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
        lvl.alpha[cid] = lvl.alpha_tmp[cid];
    } else if (lvl.level_idx == 0) {
        int ic = i < h ? h : (i >= h + lvl.nx ? h + lvl.nx - 1 : i);
        int jc = j < h ? h : (j >= h + lvl.ny ? h + lvl.ny - 1 : j);
        int kc = k < h ? h : (k >= h + lvl.nz ? h + lvl.nz - 1 : k);
        lvl.alpha[idx_cell(i, j, k, lvl.nxp, lvl.nyp)] =
            lvl.alpha_tmp[idx_cell(ic, jc, kc, lvl.nxp, lvl.nyp)];
    }
}

__global__ void k_vof_kapila_finalize(FluidLevelGpu lvl, FluidConstantsGpu c, float dt_flow) {
    int h = lvl.halo;
    int i = h + blockIdx.x * blockDim.x + threadIdx.x;
    int j = h + blockIdx.y * blockDim.y + threadIdx.y;
    int k = h + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= h + lvl.nx || j >= h + lvl.ny || k >= h + lvl.nz) return;

    size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
    float a_start = lvl.alpha_smooth[cid];
    // Apply Kapila compressible volume-fraction relaxation strictly on mixed air/water cells
    // so entrained bubbles breathe physically without drifting hydrostatic bulk water volume.
    if (a_start > 1e-3f && a_start < 0.999f) {
        float k_mix = mix_bulk_modulus_f(a_start, lvl.p_gauge[cid] + lvl.p_hydro[cid], c);
        float corr = a_start * (k_mix / (float)c.k_w - 1.0f) * lvl.div_u[cid] * dt_flow;
        corr = clamp_f(corr, -0.05f, 0.05f);
        lvl.alpha[cid] = clamp_f(lvl.alpha[cid] + corr, 0.0f, 1.0f);
    }
}

__global__ void k_mflux_ghost_bc(FluidLevelGpu lvl) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int h = lvl.halo;

    if (i <= lvl.nxp && j < lvl.nyp && k < lvl.nzp) {
        if (i < h || i > h + lvl.nx || j < h || j >= h + lvl.ny || k < h || k >= h + lvl.nz) {
            int ic = i < h ? h : (i > h + lvl.nx ? h + lvl.nx : i);
            int jc = j < h ? h : (j >= h + lvl.ny ? h + lvl.ny - 1 : j);
            int kc = k < h ? h : (k >= h + lvl.nz ? h + lvl.nz - 1 : k);
            lvl.mflux_x[idx_fx(i, j, k, lvl.nxp, lvl.nyp)] =
                lvl.mflux_x[idx_fx(ic, jc, kc, lvl.nxp, lvl.nyp)];
        }
    }
    if (i < lvl.nxp && j <= lvl.nyp && k < lvl.nzp) {
        if (i < h || i >= h + lvl.nx || j < h || j > h + lvl.ny || k < h || k >= h + lvl.nz) {
            int ic = i < h ? h : (i >= h + lvl.nx ? h + lvl.nx - 1 : i);
            int jc = j < h ? h : (j > h + lvl.ny ? h + lvl.ny : j);
            int kc = k < h ? h : (k >= h + lvl.nz ? h + lvl.nz - 1 : k);
            lvl.mflux_y[idx_fy(i, j, k, lvl.nxp, lvl.nyp)] =
                lvl.mflux_y[idx_fy(ic, jc, kc, lvl.nxp, lvl.nyp)];
        }
    }
    if (i < lvl.nxp && j < lvl.nyp && k <= lvl.nzp) {
        if (i < h || i >= h + lvl.nx || j < h || j >= h + lvl.ny || k < h || k > h + lvl.nz) {
            int ic = i < h ? h : (i >= h + lvl.nx ? h + lvl.nx - 1 : i);
            int jc = j < h ? h : (j >= h + lvl.ny ? h + lvl.ny - 1 : j);
            int kc = k < h ? h : (k > h + lvl.nz ? h + lvl.nz : k);
            lvl.mflux_z[idx_fz(i, j, k, lvl.nxp, lvl.nyp)] =
                lvl.mflux_z[idx_fz(ic, jc, kc, lvl.nxp, lvl.nyp)];
        }
    }
}

__device__ __forceinline__ float tvd_face_val(float um1, float u0, float up1, float up2, float mflux) {
    if (mflux >= 0.0f) {
        float slope = minmod(up1 - u0, u0 - um1);
        return u0 + 0.5f * slope;
    } else {
        float slope = minmod(up2 - up1, up1 - u0);
        return up1 - 0.5f * slope;
    }
}

__device__ __forceinline__ float bounded_momentum_advect(
    float rho_old,
    float mx_L, float mx_R,
    float my_B, float my_T,
    float mz_B, float mz_F,
    float u_xm2, float u_xm1, float u_c, float u_xp1, float u_xp2,
    float u_ym2, float u_ym1, float u_yp1, float u_yp2,
    float u_zm2, float u_zm1, float u_zp1, float u_zp2,
    float inv_dx, float rho_a_min
) {
    float rho_cons = rho_old - ((mx_R - mx_L) + (my_T - my_B) + (mz_F - mz_B)) * inv_dx;
    float u_lo = fminf(u_c, fminf(fminf(u_xm1, u_xp1), fminf(fminf(u_ym1, u_yp1), fminf(u_zm1, u_zp1))));
    float u_hi = fmaxf(u_c, fmaxf(fmaxf(u_xm1, u_xp1), fmaxf(fmaxf(u_ym1, u_yp1), fmaxf(u_zm1, u_zp1))));

    float u_adv;
    if (rho_cons >= 0.5f * rho_old && rho_cons >= 0.5f * rho_a_min) {
        float ux_L = tvd_face_val(u_xm2, u_xm1, u_c, u_xp1, mx_L);
        float ux_R = tvd_face_val(u_xm1, u_c, u_xp1, u_xp2, mx_R);
        float uy_B = tvd_face_val(u_ym2, u_ym1, u_c, u_yp1, my_B);
        float uy_T = tvd_face_val(u_ym1, u_c, u_yp1, u_yp2, my_T);
        float uz_B = tvd_face_val(u_zm2, u_zm1, u_c, u_zp1, mz_B);
        float uz_F = tvd_face_val(u_zm1, u_c, u_zp1, u_zp2, mz_F);
        float mom = rho_old * u_c -
            ((mx_R * ux_R - mx_L * ux_L) + (my_T * uy_T - my_B * uy_B) + (mz_F * uz_F - mz_B * uz_B)) * inv_dx;
        u_adv = mom / rho_cons;
    } else {
        float m_in_xL = fmaxf(0.0f, mx_L);
        float m_in_xR = fmaxf(0.0f, -mx_R);
        float m_in_yB = fmaxf(0.0f, my_B);
        float m_in_yT = fmaxf(0.0f, -my_T);
        float m_in_zB = fmaxf(0.0f, mz_B);
        float m_in_zF = fmaxf(0.0f, -mz_F);
        float m_out = fmaxf(0.0f, -mx_L) + fmaxf(0.0f, mx_R) + fmaxf(0.0f, -my_B) +
                      fmaxf(0.0f, my_T) + fmaxf(0.0f, -mz_B) + fmaxf(0.0f, mz_F);
        float rho_rem = fmaxf(0.0f, rho_old - m_out * inv_dx);
        float rho_tot = rho_rem + (m_in_xL + m_in_xR + m_in_yB + m_in_yT + m_in_zB + m_in_zF) * inv_dx;
        float num = rho_rem * u_c + inv_dx * (
            m_in_xL * u_xm1 + m_in_xR * u_xp1 +
            m_in_yB * u_ym1 + m_in_yT * u_yp1 +
            m_in_zB * u_zm1 + m_in_zF * u_zp1
        );
        u_adv = num / fmaxf(1e-7f, rho_tot);
    }
    return clamp_f(u_adv, u_lo, u_hi);
}

__global__ void k_advect_momentum(FluidLevelGpu lvl, FluidConstantsGpu c) {
    int h = lvl.halo;
    int i = h + blockIdx.x * blockDim.x + threadIdx.x;
    int j = h + blockIdx.y * blockDim.y + threadIdx.y;
    int k = h + blockIdx.z * blockDim.z + threadIdx.z;

    float inv_dx = 1.0f / lvl.dx;
    float rho_a_f = (float)c.rho_a;

    // 1. X-momentum advection (interior faces; outer and AMR boundary faces handled by acoustic/parent BC)
    int x_lo = h + 1;
    int x_hi = h + lvl.nx - 1;
    if (i >= x_lo && i <= x_hi && j < h + lvl.ny && k < h + lvl.nz) {
        size_t fx = idx_fx(i, j, k, lvl.nxp, lvl.nyp);
        float beta = lvl.ib_beta_x[fx];
        float ibv = (lvl.ib_vel_x != nullptr) ? lvl.ib_vel_x[fx] : 0.0f;
        if (beta >= 0.999f) {
            lvl.vflux_x[fx] = ibv;
        } else {
            float a_sL = lvl.alpha_smooth[idx_cell(i - 1, j, k, lvl.nxp, lvl.nyp)];
            float a_sR = lvl.alpha_smooth[idx_cell(i, j, k, lvl.nxp, lvl.nyp)];
            float rho_old = 0.5f * (mix_density_f(a_sL, c) + mix_density_f(a_sR, c));

            float mx_L = 0.5f * (lvl.mflux_x[idx_fx(i - 1, j, k, lvl.nxp, lvl.nyp)] + lvl.mflux_x[fx]);
            float mx_R = 0.5f * (lvl.mflux_x[fx] + lvl.mflux_x[idx_fx(i + 1, j, k, lvl.nxp, lvl.nyp)]);
            float my_B = 0.5f * (lvl.mflux_y[idx_fy(i - 1, j, k, lvl.nxp, lvl.nyp)] + lvl.mflux_y[idx_fy(i, j, k, lvl.nxp, lvl.nyp)]);
            float my_T = 0.5f * (lvl.mflux_y[idx_fy(i - 1, j + 1, k, lvl.nxp, lvl.nyp)] + lvl.mflux_y[idx_fy(i, j + 1, k, lvl.nxp, lvl.nyp)]);
            float mz_B = 0.5f * (lvl.mflux_z[idx_fz(i - 1, j, k, lvl.nxp, lvl.nyp)] + lvl.mflux_z[idx_fz(i, j, k, lvl.nxp, lvl.nyp)]);
            float mz_F = 0.5f * (lvl.mflux_z[idx_fz(i - 1, j, k + 1, lvl.nxp, lvl.nyp)] + lvl.mflux_z[idx_fz(i, j, k + 1, lvl.nxp, lvl.nyp)]);

            float u_c = lvl.ux[fx];
            float u_xm1 = lvl.ux[idx_fx(i - 1, j, k, lvl.nxp, lvl.nyp)];
            float u_xp1 = lvl.ux[idx_fx(i + 1, j, k, lvl.nxp, lvl.nyp)];
            float u_xm2 = (i >= 2) ? lvl.ux[idx_fx(i - 2, j, k, lvl.nxp, lvl.nyp)] : u_xm1;
            float u_xp2 = (i + 2 <= lvl.nxp) ? lvl.ux[idx_fx(i + 2, j, k, lvl.nxp, lvl.nyp)] : u_xp1;

            float u_ym1 = lvl.ux[idx_fx(i, j - 1, k, lvl.nxp, lvl.nyp)];
            float u_yp1 = lvl.ux[idx_fx(i, j + 1, k, lvl.nxp, lvl.nyp)];
            float u_ym2 = lvl.ux[idx_fx(i, j - 2, k, lvl.nxp, lvl.nyp)];
            float u_yp2 = lvl.ux[idx_fx(i, j + 2, k, lvl.nxp, lvl.nyp)];

            float u_zm1 = lvl.ux[idx_fx(i, j, k - 1, lvl.nxp, lvl.nyp)];
            float u_zp1 = lvl.ux[idx_fx(i, j, k + 1, lvl.nxp, lvl.nyp)];
            float u_zm2 = lvl.ux[idx_fx(i, j, k - 2, lvl.nxp, lvl.nyp)];
            float u_zp2 = lvl.ux[idx_fx(i, j, k + 2, lvl.nxp, lvl.nyp)];

            float u_adv = bounded_momentum_advect(
                rho_old, mx_L, mx_R, my_B, my_T, mz_B, mz_F,
                u_xm2, u_xm1, u_c, u_xp1, u_xp2,
                u_ym2, u_ym1, u_yp1, u_yp2,
                u_zm2, u_zm1, u_zp1, u_zp2,
                inv_dx, rho_a_f
            );
            lvl.vflux_x[fx] = (1.0f - beta) * u_adv + beta * ibv;
        }
    } else if (i <= h + lvl.nx && j < h + lvl.ny && k < h + lvl.nz) {
        size_t fx = idx_fx(i, j, k, lvl.nxp, lvl.nyp);
        lvl.vflux_x[fx] = lvl.ux[fx];
    }

    // 2. Y-momentum advection
    int y_lo = h + 1;
    int y_hi = h + lvl.ny - 1;
    if (i < h + lvl.nx && j >= y_lo && j <= y_hi && k < h + lvl.nz) {
        size_t fy = idx_fy(i, j, k, lvl.nxp, lvl.nyp);
        float beta = lvl.ib_beta_y[fy];
        float ibv = (lvl.ib_vel_y != nullptr) ? lvl.ib_vel_y[fy] : 0.0f;
        if (beta >= 0.999f) {
            lvl.vflux_y[fy] = ibv;
        } else {
            float a_sL = lvl.alpha_smooth[idx_cell(i, j - 1, k, lvl.nxp, lvl.nyp)];
            float a_sR = lvl.alpha_smooth[idx_cell(i, j, k, lvl.nxp, lvl.nyp)];
            float rho_old = 0.5f * (mix_density_f(a_sL, c) + mix_density_f(a_sR, c));

            float mx_L = 0.5f * (lvl.mflux_x[idx_fx(i, j - 1, k, lvl.nxp, lvl.nyp)] + lvl.mflux_x[idx_fx(i, j, k, lvl.nxp, lvl.nyp)]);
            float mx_R = 0.5f * (lvl.mflux_x[idx_fx(i + 1, j - 1, k, lvl.nxp, lvl.nyp)] + lvl.mflux_x[idx_fx(i + 1, j, k, lvl.nxp, lvl.nyp)]);
            float my_B = 0.5f * (lvl.mflux_y[idx_fy(i, j - 1, k, lvl.nxp, lvl.nyp)] + lvl.mflux_y[fy]);
            float my_T = 0.5f * (lvl.mflux_y[fy] + lvl.mflux_y[idx_fy(i, j + 1, k, lvl.nxp, lvl.nyp)]);
            float mz_B = 0.5f * (lvl.mflux_z[idx_fz(i, j - 1, k, lvl.nxp, lvl.nyp)] + lvl.mflux_z[idx_fz(i, j, k, lvl.nxp, lvl.nyp)]);
            float mz_F = 0.5f * (lvl.mflux_z[idx_fz(i, j - 1, k + 1, lvl.nxp, lvl.nyp)] + lvl.mflux_z[idx_fz(i, j, k + 1, lvl.nxp, lvl.nyp)]);

            float v_c = lvl.uy[fy];
            float v_xm1 = lvl.uy[idx_fy(i - 1, j, k, lvl.nxp, lvl.nyp)];
            float v_xp1 = lvl.uy[idx_fy(i + 1, j, k, lvl.nxp, lvl.nyp)];
            float v_xm2 = lvl.uy[idx_fy(i - 2, j, k, lvl.nxp, lvl.nyp)];
            float v_xp2 = lvl.uy[idx_fy(i + 2, j, k, lvl.nxp, lvl.nyp)];

            float v_ym1 = lvl.uy[idx_fy(i, j - 1, k, lvl.nxp, lvl.nyp)];
            float v_yp1 = lvl.uy[idx_fy(i, j + 1, k, lvl.nxp, lvl.nyp)];
            float v_ym2 = (j >= 2) ? lvl.uy[idx_fy(i, j - 2, k, lvl.nxp, lvl.nyp)] : v_ym1;
            float v_yp2 = (j + 2 <= lvl.nyp) ? lvl.uy[idx_fy(i, j + 2, k, lvl.nxp, lvl.nyp)] : v_yp1;

            float v_zm1 = lvl.uy[idx_fy(i, j, k - 1, lvl.nxp, lvl.nyp)];
            float v_zp1 = lvl.uy[idx_fy(i, j, k + 1, lvl.nxp, lvl.nyp)];
            float v_zm2 = lvl.uy[idx_fy(i, j, k - 2, lvl.nxp, lvl.nyp)];
            float v_zp2 = lvl.uy[idx_fy(i, j, k + 2, lvl.nxp, lvl.nyp)];

            float v_adv = bounded_momentum_advect(
                rho_old, mx_L, mx_R, my_B, my_T, mz_B, mz_F,
                v_xm2, v_xm1, v_c, v_xp1, v_xp2,
                v_ym2, v_ym1, v_yp1, v_yp2,
                v_zm2, v_zm1, v_zp1, v_zp2,
                inv_dx, rho_a_f
            );
            lvl.vflux_y[fy] = (1.0f - beta) * v_adv + beta * ibv;
        }
    } else if (i < h + lvl.nx && j <= h + lvl.ny && k < h + lvl.nz) {
        size_t fy = idx_fy(i, j, k, lvl.nxp, lvl.nyp);
        lvl.vflux_y[fy] = lvl.uy[fy];
    }

    // 3. Z-momentum advection
    int z_lo = h + 1;
    int z_hi = h + lvl.nz - 1;
    if (i < h + lvl.nx && j < h + lvl.ny && k >= z_lo && k <= z_hi) {
        size_t fz = idx_fz(i, j, k, lvl.nxp, lvl.nyp);
        float beta = lvl.ib_beta_z[fz];
        float ibv = (lvl.ib_vel_z != nullptr) ? lvl.ib_vel_z[fz] : 0.0f;
        if (beta >= 0.999f) {
            lvl.vflux_z[fz] = ibv;
        } else {
            float a_sL = lvl.alpha_smooth[idx_cell(i, j, k - 1, lvl.nxp, lvl.nyp)];
            float a_sR = lvl.alpha_smooth[idx_cell(i, j, k, lvl.nxp, lvl.nyp)];
            float rho_old = 0.5f * (mix_density_f(a_sL, c) + mix_density_f(a_sR, c));

            float mx_L = 0.5f * (lvl.mflux_x[idx_fx(i, j, k - 1, lvl.nxp, lvl.nyp)] + lvl.mflux_x[idx_fx(i, j, k, lvl.nxp, lvl.nyp)]);
            float mx_R = 0.5f * (lvl.mflux_x[idx_fx(i + 1, j, k - 1, lvl.nxp, lvl.nyp)] + lvl.mflux_x[idx_fx(i + 1, j, k, lvl.nxp, lvl.nyp)]);
            float my_B = 0.5f * (lvl.mflux_y[idx_fy(i, j, k - 1, lvl.nxp, lvl.nyp)] + lvl.mflux_y[idx_fy(i, j, k, lvl.nxp, lvl.nyp)]);
            float my_T = 0.5f * (lvl.mflux_y[idx_fy(i, j + 1, k - 1, lvl.nxp, lvl.nyp)] + lvl.mflux_y[idx_fy(i, j + 1, k, lvl.nxp, lvl.nyp)]);
            float mz_B = 0.5f * (lvl.mflux_z[idx_fz(i, j, k - 1, lvl.nxp, lvl.nyp)] + lvl.mflux_z[fz]);
            float mz_F = 0.5f * (lvl.mflux_z[fz] + lvl.mflux_z[idx_fz(i, j, k + 1, lvl.nxp, lvl.nyp)]);

            float w_c = lvl.uz[fz];
            float w_xm1 = lvl.uz[idx_fz(i - 1, j, k, lvl.nxp, lvl.nyp)];
            float w_xp1 = lvl.uz[idx_fz(i + 1, j, k, lvl.nxp, lvl.nyp)];
            float w_xm2 = lvl.uz[idx_fz(i - 2, j, k, lvl.nxp, lvl.nyp)];
            float w_xp2 = lvl.uz[idx_fz(i + 2, j, k, lvl.nxp, lvl.nyp)];

            float w_ym1 = lvl.uz[idx_fz(i, j - 1, k, lvl.nxp, lvl.nyp)];
            float w_yp1 = lvl.uz[idx_fz(i, j + 1, k, lvl.nxp, lvl.nyp)];
            float w_ym2 = lvl.uz[idx_fz(i, j - 2, k, lvl.nxp, lvl.nyp)];
            float w_yp2 = lvl.uz[idx_fz(i, j + 2, k, lvl.nxp, lvl.nyp)];

            float w_zm1 = lvl.uz[idx_fz(i, j, k - 1, lvl.nxp, lvl.nyp)];
            float w_zp1 = lvl.uz[idx_fz(i, j, k + 1, lvl.nxp, lvl.nyp)];
            float w_zm2 = (k >= 2) ? lvl.uz[idx_fz(i, j, k - 2, lvl.nxp, lvl.nyp)] : w_zm1;
            float w_zp2 = (k + 2 <= lvl.nzp) ? lvl.uz[idx_fz(i, j, k + 2, lvl.nxp, lvl.nyp)] : w_zp1;

            float w_adv = bounded_momentum_advect(
                rho_old, mx_L, mx_R, my_B, my_T, mz_B, mz_F,
                w_xm2, w_xm1, w_c, w_xp1, w_xp2,
                w_ym2, w_ym1, w_yp1, w_yp2,
                w_zm2, w_zm1, w_zp1, w_zp2,
                inv_dx, rho_a_f
            );
            lvl.vflux_z[fz] = (1.0f - beta) * w_adv + beta * ibv;
        }
    } else if (i < h + lvl.nx && j < h + lvl.ny && k <= h + lvl.nz) {
        size_t fz = idx_fz(i, j, k, lvl.nxp, lvl.nyp);
        lvl.vflux_z[fz] = lvl.uz[fz];
    }
}

__global__ void k_copy_vflux_to_u(FluidLevelGpu lvl) {
    int h = lvl.halo;
    int i = h + blockIdx.x * blockDim.x + threadIdx.x;
    int j = h + blockIdx.y * blockDim.y + threadIdx.y;
    int k = h + blockIdx.z * blockDim.z + threadIdx.z;

    if (i <= h + lvl.nx && j < h + lvl.ny && k < h + lvl.nz) {
        size_t fx = idx_fx(i, j, k, lvl.nxp, lvl.nyp);
        lvl.ux[fx] = lvl.vflux_x[fx];
    }
    if (i < h + lvl.nx && j <= h + lvl.ny && k < h + lvl.nz) {
        size_t fy = idx_fy(i, j, k, lvl.nxp, lvl.nyp);
        lvl.uy[fy] = lvl.vflux_y[fy];
    }
    if (i < h + lvl.nx && j < h + lvl.ny && k <= h + lvl.nz) {
        size_t fz = idx_fz(i, j, k, lvl.nxp, lvl.nyp);
        lvl.uz[fz] = lvl.vflux_z[fz];
    }
}

__global__ void k_binomial_smooth_3d(const float* src, float* dst, int nxp, int nyp, int nzp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nxp || j >= nyp || k >= nzp) return;

    float sum = 0.0f;
    for (int dk = -1; dk <= 1; ++dk) {
        int kk = min(nzp - 1, max(0, k + dk));
        float wk = (dk == 0) ? 0.5f : 0.25f;
        for (int dj = -1; dj <= 1; ++dj) {
            int jj = min(nyp - 1, max(0, j + dj));
            float wj = (dj == 0) ? 0.5f : 0.25f;
            for (int di = -1; di <= 1; ++di) {
                int ii = min(nxp - 1, max(0, i + di));
                float wi = (di == 0) ? 0.5f : 0.25f;
                sum += wi * wj * wk * src[idx_cell(ii, jj, kk, nxp, nyp)];
            }
        }
    }
    dst[idx_cell(i, j, k, nxp, nyp)] = sum;
}

__device__ __forceinline__ void vertex_unit_normal(
    const float* a_s, int i, int j, int k, int nxp, int nyp, float inv_dx,
    float& nx, float& ny, float& nz, float& valid
) {
    // Vertex (i-1/2, j-1/2, k-1/2) uses cells [i-1..i, j-1..j, k-1..k]
    float gx = 0.25f * inv_dx * (
        (a_s[idx_cell(i, j - 1, k - 1, nxp, nyp)] - a_s[idx_cell(i - 1, j - 1, k - 1, nxp, nyp)]) +
        (a_s[idx_cell(i, j, k - 1, nxp, nyp)] - a_s[idx_cell(i - 1, j, k - 1, nxp, nyp)]) +
        (a_s[idx_cell(i, j - 1, k, nxp, nyp)] - a_s[idx_cell(i - 1, j - 1, k, nxp, nyp)]) +
        (a_s[idx_cell(i, j, k, nxp, nyp)] - a_s[idx_cell(i - 1, j, k, nxp, nyp)])
    );
    float gy = 0.25f * inv_dx * (
        (a_s[idx_cell(i - 1, j, k - 1, nxp, nyp)] - a_s[idx_cell(i - 1, j - 1, k - 1, nxp, nyp)]) +
        (a_s[idx_cell(i, j, k - 1, nxp, nyp)] - a_s[idx_cell(i, j - 1, k - 1, nxp, nyp)]) +
        (a_s[idx_cell(i - 1, j, k, nxp, nyp)] - a_s[idx_cell(i - 1, j - 1, k, nxp, nyp)]) +
        (a_s[idx_cell(i, j, k, nxp, nyp)] - a_s[idx_cell(i, j - 1, k, nxp, nyp)])
    );
    float gz = 0.25f * inv_dx * (
        (a_s[idx_cell(i - 1, j - 1, k, nxp, nyp)] - a_s[idx_cell(i - 1, j - 1, k - 1, nxp, nyp)]) +
        (a_s[idx_cell(i, j - 1, k, nxp, nyp)] - a_s[idx_cell(i, j - 1, k - 1, nxp, nyp)]) +
        (a_s[idx_cell(i - 1, j, k, nxp, nyp)] - a_s[idx_cell(i - 1, j, k - 1, nxp, nyp)]) +
        (a_s[idx_cell(i, j, k, nxp, nyp)] - a_s[idx_cell(i, j, k - 1, nxp, nyp)])
    );
    float mag = sqrtf(gx * gx + gy * gy + gz * gz);
    if (mag > 1e-4f * inv_dx) {
        float inv_m = 1.0f / mag;
        nx = gx * inv_m;
        ny = gy * inv_m;
        nz = gz * inv_m;
        valid = 1.0f;
    } else {
        nx = 0.0f;
        ny = 0.0f;
        nz = 0.0f;
        valid = 0.0f;
    }
}

__global__ void k_compute_curvature(FluidLevelGpu lvl) {
    int i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int j = 1 + blockIdx.y * blockDim.y + threadIdx.y;
    int k = 1 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= lvl.nxp - 1 || j >= lvl.nyp - 1 || k >= lvl.nzp - 1) return;

    float inv_dx = 1.0f / lvl.dx;
    float nx_v[8], ny_v[8], nz_v[8], val_v[8];
    int idx = 0;
    float any_valid = 0.0f;
    for (int dk = 0; dk <= 1; ++dk) {
        for (int dj = 0; dj <= 1; ++dj) {
            for (int di = 0; di <= 1; ++di) {
                vertex_unit_normal(
                    lvl.alpha_smooth, i + di, j + dj, k + dk,
                    lvl.nxp, lvl.nyp, inv_dx,
                    nx_v[idx], ny_v[idx], nz_v[idx], val_v[idx]
                );
                any_valid += val_v[idx];
                ++idx;
            }
        }
    }

    size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
    if (any_valid < 7.5f) {
        lvl.kappa[cid] = 0.0f;
        return;
    }

    // div(n) = average of 4 differences along x, y, z; kappa = -div(n)
    float dnx = 0.25f * ((nx_v[1] - nx_v[0]) + (nx_v[3] - nx_v[2]) + (nx_v[5] - nx_v[4]) + (nx_v[7] - nx_v[6]));
    float dny = 0.25f * ((ny_v[2] - ny_v[0]) + (ny_v[3] - ny_v[1]) + (ny_v[6] - ny_v[4]) + (ny_v[7] - ny_v[5]));
    float dnz = 0.25f * ((nz_v[4] - nz_v[0]) + (nz_v[5] - nz_v[1]) + (nz_v[6] - nz_v[2]) + (nz_v[7] - nz_v[3]));
    float kap = -(dnx + dny + dnz) * inv_dx;
    float k_max = 2.0f * inv_dx;
    lvl.kappa[cid] = clamp_f(kap, -k_max, k_max);
}

__device__ __forceinline__ float mix_viscosity_f(float alpha, const FluidConstantsGpu& c) {
    float a = clamp_f(alpha, 0.0f, 1.0f);
    return 1.0f / (a / (float)c.mu_w + (1.0f - a) / (float)c.mu_a);
}

__device__ __forceinline__ float face_curvature(float kL, float kR) {
    if (kL != 0.0f && kR != 0.0f) return 0.5f * (kL + kR);
    if (kL != 0.0f) return kL;
    return kR;
}

__global__ void k_compute_slow_forces(FluidLevelGpu lvl, FluidConstantsGpu c) {
    int h = lvl.halo;
    int i = h + blockIdx.x * blockDim.x + threadIdx.x;
    int j = h + blockIdx.y * blockDim.y + threadIdx.y;
    int k = h + blockIdx.z * blockDim.z + threadIdx.z;
    float inv_dx = 1.0f / lvl.dx;
    float sigma_f = (float)c.sigma;
    float inv_rho_avg = 1.0f / (0.5f * ((float)c.rho_a + (float)c.rho_w));

    // 1. X-faces
    if (i <= h + lvl.nx && j < h + lvl.ny && k < h + lvl.nz) {
        size_t fx = idx_fx(i, j, k, lvl.nxp, lvl.nyp);
        if (lvl.ib_beta_x[fx] >= 0.999f) {
            lvl.a_slow_x[fx] = 0.0f;
        } else {
            size_t cL = idx_cell(i - 1, j, k, lvl.nxp, lvl.nyp);
            size_t cR = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
            float aL = lvl.alpha[cL];
            float aR = lvl.alpha[cR];
            float rho_f = 0.5f * (mix_density_f(aL, c) + mix_density_f(aR, c));

            // Balanced-force CSF surface tension
            float da = aR - aL;
            float f_sigma = 0.0f;
            if (fabsf(da) > 1e-6f) {
                float kf = face_curvature(lvl.kappa[cL], lvl.kappa[cR]);
                f_sigma = sigma_f * kf * (da * inv_dx);
            }

            // Viscous stress div(mu * (grad u + grad u^T))
            float uC = lvl.ux[fx];
            float mu_L = mix_viscosity_f(aL, c);
            float mu_R = mix_viscosity_f(aR, c);
            float tau_xx_R = 2.0f * mu_R * (lvl.ux[idx_fx(i + 1, j, k, lvl.nxp, lvl.nyp)] - uC) * inv_dx;
            float tau_xx_L = 2.0f * mu_L * (uC - lvl.ux[idx_fx(i - 1, j, k, lvl.nxp, lvl.nyp)]) * inv_dx;

            float mu_yb = mix_viscosity_f(0.25f * (
                lvl.alpha[idx_cell(i - 1, j - 1, k, lvl.nxp, lvl.nyp)] +
                lvl.alpha[idx_cell(i, j - 1, k, lvl.nxp, lvl.nyp)] + aL + aR), c);
            float mu_yt = mix_viscosity_f(0.25f * (
                aL + aR +
                lvl.alpha[idx_cell(i - 1, j + 1, k, lvl.nxp, lvl.nyp)] +
                lvl.alpha[idx_cell(i, j + 1, k, lvl.nxp, lvl.nyp)]), c);
            float tau_xy_T = mu_yt * (
                (lvl.ux[idx_fx(i, j + 1, k, lvl.nxp, lvl.nyp)] - uC) +
                (lvl.uy[idx_fy(i, j + 1, k, lvl.nxp, lvl.nyp)] - lvl.uy[idx_fy(i - 1, j + 1, k, lvl.nxp, lvl.nyp)])
            ) * inv_dx;
            float tau_xy_B = mu_yb * (
                (uC - lvl.ux[idx_fx(i, j - 1, k, lvl.nxp, lvl.nyp)]) +
                (lvl.uy[idx_fy(i, j, k, lvl.nxp, lvl.nyp)] - lvl.uy[idx_fy(i - 1, j, k, lvl.nxp, lvl.nyp)])
            ) * inv_dx;

            float mu_zb = mix_viscosity_f(0.25f * (
                lvl.alpha[idx_cell(i - 1, j, k - 1, lvl.nxp, lvl.nyp)] +
                lvl.alpha[idx_cell(i, j, k - 1, lvl.nxp, lvl.nyp)] + aL + aR), c);
            float mu_zf = mix_viscosity_f(0.25f * (
                aL + aR +
                lvl.alpha[idx_cell(i - 1, j, k + 1, lvl.nxp, lvl.nyp)] +
                lvl.alpha[idx_cell(i, j, k + 1, lvl.nxp, lvl.nyp)]), c);
            float tau_xz_F = mu_zf * (
                (lvl.ux[idx_fx(i, j, k + 1, lvl.nxp, lvl.nyp)] - uC) +
                (lvl.uz[idx_fz(i, j, k + 1, lvl.nxp, lvl.nyp)] - lvl.uz[idx_fz(i - 1, j, k + 1, lvl.nxp, lvl.nyp)])
            ) * inv_dx;
            float tau_xz_B = mu_zb * (
                (uC - lvl.ux[idx_fx(i, j, k - 1, lvl.nxp, lvl.nyp)]) +
                (lvl.uz[idx_fz(i, j, k, lvl.nxp, lvl.nyp)] - lvl.uz[idx_fz(i - 1, j, k, lvl.nxp, lvl.nyp)])
            ) * inv_dx;

            float f_visc = ((tau_xx_R - tau_xx_L) + (tau_xy_T - tau_xy_B) + (tau_xz_F - tau_xz_B)) * inv_dx;
            lvl.a_slow_x[fx] = f_sigma * inv_rho_avg + f_visc / rho_f;
        }
    }

    // 2. Y-faces
    if (i < h + lvl.nx && j <= h + lvl.ny && k < h + lvl.nz) {
        size_t fy = idx_fy(i, j, k, lvl.nxp, lvl.nyp);
        if (lvl.ib_beta_y[fy] >= 0.999f) {
            lvl.a_slow_y[fy] = 0.0f;
        } else {
            size_t cL = idx_cell(i, j - 1, k, lvl.nxp, lvl.nyp);
            size_t cR = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
            float aL = lvl.alpha[cL];
            float aR = lvl.alpha[cR];
            float rho_f = 0.5f * (mix_density_f(aL, c) + mix_density_f(aR, c));

            float da = aR - aL;
            float f_sigma = 0.0f;
            if (fabsf(da) > 1e-6f) {
                float kf = face_curvature(lvl.kappa[cL], lvl.kappa[cR]);
                f_sigma = sigma_f * kf * (da * inv_dx);
            }

            float vC = lvl.uy[fy];
            float mu_L = mix_viscosity_f(aL, c);
            float mu_R = mix_viscosity_f(aR, c);
            float tau_yy_R = 2.0f * mu_R * (lvl.uy[idx_fy(i, j + 1, k, lvl.nxp, lvl.nyp)] - vC) * inv_dx;
            float tau_yy_L = 2.0f * mu_L * (vC - lvl.uy[idx_fy(i, j - 1, k, lvl.nxp, lvl.nyp)]) * inv_dx;

            float mu_xl = mix_viscosity_f(0.25f * (
                lvl.alpha[idx_cell(i - 1, j - 1, k, lvl.nxp, lvl.nyp)] +
                lvl.alpha[idx_cell(i - 1, j, k, lvl.nxp, lvl.nyp)] + aL + aR), c);
            float mu_xr = mix_viscosity_f(0.25f * (
                aL + aR +
                lvl.alpha[idx_cell(i + 1, j - 1, k, lvl.nxp, lvl.nyp)] +
                lvl.alpha[idx_cell(i + 1, j, k, lvl.nxp, lvl.nyp)]), c);
            float tau_yx_R = mu_xr * (
                (lvl.uy[idx_fy(i + 1, j, k, lvl.nxp, lvl.nyp)] - vC) +
                (lvl.ux[idx_fx(i + 1, j, k, lvl.nxp, lvl.nyp)] - lvl.ux[idx_fx(i + 1, j - 1, k, lvl.nxp, lvl.nyp)])
            ) * inv_dx;
            float tau_yx_L = mu_xl * (
                (vC - lvl.uy[idx_fy(i - 1, j, k, lvl.nxp, lvl.nyp)]) +
                (lvl.ux[idx_fx(i, j, k, lvl.nxp, lvl.nyp)] - lvl.ux[idx_fx(i, j - 1, k, lvl.nxp, lvl.nyp)])
            ) * inv_dx;

            float mu_zb = mix_viscosity_f(0.25f * (
                lvl.alpha[idx_cell(i, j - 1, k - 1, lvl.nxp, lvl.nyp)] +
                lvl.alpha[idx_cell(i, j, k - 1, lvl.nxp, lvl.nyp)] + aL + aR), c);
            float mu_zf = mix_viscosity_f(0.25f * (
                aL + aR +
                lvl.alpha[idx_cell(i, j - 1, k + 1, lvl.nxp, lvl.nyp)] +
                lvl.alpha[idx_cell(i, j, k + 1, lvl.nxp, lvl.nyp)]), c);
            float tau_yz_F = mu_zf * (
                (lvl.uy[idx_fy(i, j, k + 1, lvl.nxp, lvl.nyp)] - vC) +
                (lvl.uz[idx_fz(i, j, k + 1, lvl.nxp, lvl.nyp)] - lvl.uz[idx_fz(i, j - 1, k + 1, lvl.nxp, lvl.nyp)])
            ) * inv_dx;
            float tau_yz_B = mu_zb * (
                (vC - lvl.uy[idx_fy(i, j, k - 1, lvl.nxp, lvl.nyp)]) +
                (lvl.uz[idx_fz(i, j, k, lvl.nxp, lvl.nyp)] - lvl.uz[idx_fz(i, j - 1, k, lvl.nxp, lvl.nyp)])
            ) * inv_dx;

            float f_visc = ((tau_yy_R - tau_yy_L) + (tau_yx_R - tau_yx_L) + (tau_yz_F - tau_yz_B)) * inv_dx;
            lvl.a_slow_y[fy] = f_sigma * inv_rho_avg + f_visc / rho_f;
        }
    }

    // 3. Z-faces
    if (i < h + lvl.nx && j < h + lvl.ny && k <= h + lvl.nz) {
        size_t fz = idx_fz(i, j, k, lvl.nxp, lvl.nyp);
        if (lvl.ib_beta_z[fz] >= 0.999f) {
            lvl.a_slow_z[fz] = 0.0f;
        } else {
            size_t cL = idx_cell(i, j, k - 1, lvl.nxp, lvl.nyp);
            size_t cR = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
            float aL = lvl.alpha[cL];
            float aR = lvl.alpha[cR];
            float rho_f = 0.5f * (mix_density_f(aL, c) + mix_density_f(aR, c));

            float da = aR - aL;
            float f_sigma = 0.0f;
            if (fabsf(da) > 1e-6f) {
                float kf = face_curvature(lvl.kappa[cL], lvl.kappa[cR]);
                f_sigma = sigma_f * kf * (da * inv_dx);
            }

            float wC = lvl.uz[fz];
            float mu_L = mix_viscosity_f(aL, c);
            float mu_R = mix_viscosity_f(aR, c);
            float tau_zz_R = 2.0f * mu_R * (lvl.uz[idx_fz(i, j, k + 1, lvl.nxp, lvl.nyp)] - wC) * inv_dx;
            float tau_zz_L = 2.0f * mu_L * (wC - lvl.uz[idx_fz(i, j, k - 1, lvl.nxp, lvl.nyp)]) * inv_dx;

            float mu_xl = mix_viscosity_f(0.25f * (
                lvl.alpha[idx_cell(i - 1, j, k - 1, lvl.nxp, lvl.nyp)] +
                lvl.alpha[idx_cell(i - 1, j, k, lvl.nxp, lvl.nyp)] + aL + aR), c);
            float mu_xr = mix_viscosity_f(0.25f * (
                aL + aR +
                lvl.alpha[idx_cell(i + 1, j, k - 1, lvl.nxp, lvl.nyp)] +
                lvl.alpha[idx_cell(i + 1, j, k, lvl.nxp, lvl.nyp)]), c);
            float tau_zx_R = mu_xr * (
                (lvl.uz[idx_fz(i + 1, j, k, lvl.nxp, lvl.nyp)] - wC) +
                (lvl.ux[idx_fx(i + 1, j, k, lvl.nxp, lvl.nyp)] - lvl.ux[idx_fx(i + 1, j, k - 1, lvl.nxp, lvl.nyp)])
            ) * inv_dx;
            float tau_zx_L = mu_xl * (
                (wC - lvl.uz[idx_fz(i - 1, j, k, lvl.nxp, lvl.nyp)]) +
                (lvl.ux[idx_fx(i, j, k, lvl.nxp, lvl.nyp)] - lvl.ux[idx_fx(i, j, k - 1, lvl.nxp, lvl.nyp)])
            ) * inv_dx;

            float mu_yb = mix_viscosity_f(0.25f * (
                lvl.alpha[idx_cell(i, j - 1, k - 1, lvl.nxp, lvl.nyp)] +
                lvl.alpha[idx_cell(i, j - 1, k, lvl.nxp, lvl.nyp)] + aL + aR), c);
            float mu_yt = mix_viscosity_f(0.25f * (
                aL + aR +
                lvl.alpha[idx_cell(i, j + 1, k - 1, lvl.nxp, lvl.nyp)] +
                lvl.alpha[idx_cell(i, j + 1, k, lvl.nxp, lvl.nyp)]), c);
            float tau_zy_T = mu_yt * (
                (lvl.uz[idx_fz(i, j + 1, k, lvl.nxp, lvl.nyp)] - wC) +
                (lvl.uy[idx_fy(i, j + 1, k, lvl.nxp, lvl.nyp)] - lvl.uy[idx_fy(i, j + 1, k - 1, lvl.nxp, lvl.nyp)])
            ) * inv_dx;
            float tau_zy_B = mu_yb * (
                (wC - lvl.uz[idx_fz(i, j - 1, k, lvl.nxp, lvl.nyp)]) +
                (lvl.uy[idx_fy(i, j, k, lvl.nxp, lvl.nyp)] - lvl.uy[idx_fy(i, j, k - 1, lvl.nxp, lvl.nyp)])
            ) * inv_dx;

            float f_visc = ((tau_zz_R - tau_zz_L) + (tau_zx_R - tau_zx_L) + (tau_zy_T - tau_zy_B)) * inv_dx;
            lvl.a_slow_z[fz] = f_sigma * inv_rho_avg + f_visc / rho_f;
        }
    }
}
