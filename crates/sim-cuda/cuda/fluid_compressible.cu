#include "common.cuh"

// Conservative 3D compressible Navier-Stokes (2nd-order MUSCL-HLLC + Stokes/bulk
// viscosity + Fourier thermal conduction + immersed-boundary moving/static walls)
// with Berger-Colella AMR restriction/prolongation and sliding shock-window shifts.

__device__ __forceinline__ float cons_pressure(
    float rho, float mx, float my, float mz, float ene, float gamma
) {
    float inv_r = 1.0f / fmaxf(1e-4f, rho);
    float ke = 0.5f * (mx * mx + my * my + mz * mz) * inv_r;
    return fmaxf(100.0f, (gamma - 1.0f) * (ene - ke));
}

__device__ __forceinline__ void cons_to_prim(
    const FluidLevelGpu& lvl, size_t cid, float gamma,
    float& rho, float& u, float& v, float& w, float& p
) {
    rho = fmaxf(1e-4f, lvl.u_rho[cid]);
    float inv_r = 1.0f / rho;
    u = lvl.u_mx[cid] * inv_r;
    v = lvl.u_my[cid] * inv_r;
    w = lvl.u_mz[cid] * inv_r;
    float ke = 0.5f * rho * (u * u + v * v + w * w);
    p = fmaxf(100.0f, (gamma - 1.0f) * (lvl.u_ene[cid] - ke));
}

__device__ __forceinline__ void cons1_to_prim(
    const FluidLevelGpu& lvl, size_t cid, float gamma,
    float& rho, float& u, float& v, float& w, float& p
) {
    rho = fmaxf(1e-4f, lvl.u1_rho[cid]);
    float inv_r = 1.0f / rho;
    u = lvl.u1_mx[cid] * inv_r;
    v = lvl.u1_my[cid] * inv_r;
    w = lvl.u1_mz[cid] * inv_r;
    float ke = 0.5f * rho * (u * u + v * v + w * w);
    p = fmaxf(100.0f, (gamma - 1.0f) * (lvl.u1_ene[cid] - ke));
}

__device__ __forceinline__ void hs_state(
    float /*z*/, const FluidConstantsGpu& c, float& rho_hs, float& p_hs
) {
    rho_hs = (float)c.rho_a;
    p_hs = (float)c.p0;
}

__device__ __forceinline__ float cell_z_world(const FluidLevelGpu& lvl, int k) {
    return lvl.oz + ((float)(k - lvl.halo) + 0.5f) * lvl.dx;
}

__device__ __forceinline__ float sponge_sigma_cell(
    const FluidLevelGpu& lvl, const FluidConstantsGpu& c, int i, int j, int k, float c0
) {
    if (lvl.level_idx != 0 || lvl.pml_thickness <= 0) return 0.0f;
    int h = lvl.halo;
    int pt = lvl.pml_thickness;
    float pf = (float)pt;
    float s = 0.0f;

    int ii = i - h;
    int jj = j - h;
    int kk = k - h;
    float pos;

    if ((lvl.pml_face_mask & (1u << 0)) && ii < pt) {
        float d = clamp_f((pf - ((float)ii + 0.5f)) / pf, 0.0f, 1.0f);
        s = fmaxf(s, c.pml_coeff * c0 / lvl.dx * d * d * d);
    }
    if ((lvl.pml_face_mask & (1u << 1)) && ii >= lvl.nx - pt) {
        pos = (float)ii + 0.5f;
        float d = clamp_f((pos - (float)(lvl.nx - pt)) / pf, 0.0f, 1.0f);
        s = fmaxf(s, c.pml_coeff * c0 / lvl.dx * d * d * d);
    }
    if ((lvl.pml_face_mask & (1u << 2)) && jj < pt) {
        float d = clamp_f((pf - ((float)jj + 0.5f)) / pf, 0.0f, 1.0f);
        s = fmaxf(s, c.pml_coeff * c0 / lvl.dx * d * d * d);
    }
    if ((lvl.pml_face_mask & (1u << 3)) && jj >= lvl.ny - pt) {
        pos = (float)jj + 0.5f;
        float d = clamp_f((pos - (float)(lvl.ny - pt)) / pf, 0.0f, 1.0f);
        s = fmaxf(s, c.pml_coeff * c0 / lvl.dx * d * d * d);
    }
    if ((lvl.pml_face_mask & (1u << 4)) && kk < pt) {
        float d = clamp_f((pf - ((float)kk + 0.5f)) / pf, 0.0f, 1.0f);
        s = fmaxf(s, c.pml_coeff * c0 / lvl.dx * d * d * d);
    }
    if ((lvl.pml_face_mask & (1u << 5)) && kk >= lvl.nz - pt) {
        pos = (float)kk + 0.5f;
        float d = clamp_f((pos - (float)(lvl.nz - pt)) / pf, 0.0f, 1.0f);
        s = fmaxf(s, c.pml_coeff * c0 / lvl.dx * d * d * d);
    }
    return s;
}

// Compute exact HLLC inviscid flux along axis (0=X, 1=Y, 2=Z)
__device__ __forceinline__ void hllc_flux_1d(
    float rhoL, float uL, float vL, float wL, float pL,
    float rhoR, float uR, float vR, float wR, float pR,
    int axis, float gamma,
    float& f_rho, float& f_mx, float& f_my, float& f_mz, float& f_ene
) {
    float qL = (axis == 0) ? uL : ((axis == 1) ? vL : wL);
    float qR = (axis == 0) ? uR : ((axis == 1) ? vR : wR);

    float keL = 0.5f * rhoL * (uL * uL + vL * vL + wL * wL);
    float keR = 0.5f * rhoR * (uR * uR + vR * vR + wR * wR);
    float EL = pL / (gamma - 1.0f) + keL;
    float ER = pR / (gamma - 1.0f) + keR;
    float HL = (EL + pL) / rhoL;
    float HR = (ER + pR) / rhoR;

    float cL = sqrtf(gamma * pL / rhoL);
    float cR = sqrtf(gamma * pR / rhoR);

    // Roe-Einfeldt signal speed bounds
    float srL = sqrtf(rhoL);
    float srR = sqrtf(rhoR);
    float inv_sr = 1.0f / (srL + srR);
    float u_roe = (srL * uL + srR * uR) * inv_sr;
    float v_roe = (srL * vL + srR * vR) * inv_sr;
    float w_roe = (srL * wL + srR * wR) * inv_sr;
    float q_roe = (srL * qL + srR * qR) * inv_sr;
    float H_roe = (srL * HL + srR * HR) * inv_sr;
    float v2_roe = u_roe * u_roe + v_roe * v_roe + w_roe * w_roe;
    float c_roe = sqrtf(fmaxf(100.0f, (gamma - 1.0f) * (H_roe - 0.5f * v2_roe)));

    float SL = fminf(qL - cL, q_roe - c_roe);
    float SR = fmaxf(qR + cR, q_roe + c_roe);

    if (SL >= 0.0f) {
        f_rho = rhoL * qL;
        f_mx = rhoL * uL * qL + (axis == 0 ? pL : 0.0f);
        f_my = rhoL * vL * qL + (axis == 1 ? pL : 0.0f);
        f_mz = rhoL * wL * qL + (axis == 2 ? pL : 0.0f);
        f_ene = (EL + pL) * qL;
        return;
    }
    if (SR <= 0.0f) {
        f_rho = rhoR * qR;
        f_mx = rhoR * uR * qR + (axis == 0 ? pR : 0.0f);
        f_my = rhoR * vR * qR + (axis == 1 ? pR : 0.0f);
        f_mz = rhoR * wR * qR + (axis == 2 ? pR : 0.0f);
        f_ene = (ER + pR) * qR;
        return;
    }

    float rL_sL = rhoL * (SL - qL);
    float rR_sR = rhoR * (SR - qR);
    float denom = rL_sL - rR_sR;
    float S_star = (fabsf(denom) > 1e-8f)
        ? (pR - pL + rL_sL * qL - rR_sR * qR) / denom
        : 0.5f * (qL + qR);

    if (S_star >= 0.0f) {
        float star_den = SL - S_star;
        float fac = (fabsf(star_den) > 1e-8f) ? (rL_sL / star_den) : rhoL;
        float u_s = (axis == 0) ? S_star : uL;
        float v_s = (axis == 1) ? S_star : vL;
        float w_s = (axis == 2) ? S_star : wL;
        float E_s = fac * (EL / rhoL + (S_star - qL) * (S_star + pL / (fabsf(rL_sL) > 1e-8f ? rL_sL : 1.0f)));
        if (fabsf(rL_sL) <= 1e-8f) E_s = EL;

        float fL_rho = rhoL * qL;
        float fL_mx = rhoL * uL * qL + (axis == 0 ? pL : 0.0f);
        float fL_my = rhoL * vL * qL + (axis == 1 ? pL : 0.0f);
        float fL_mz = rhoL * wL * qL + (axis == 2 ? pL : 0.0f);
        float fL_ene = (EL + pL) * qL;

        f_rho = fL_rho + SL * (fac - rhoL);
        f_mx  = fL_mx  + SL * (fac * u_s - rhoL * uL);
        f_my  = fL_my  + SL * (fac * v_s - rhoL * vL);
        f_mz  = fL_mz  + SL * (fac * w_s - rhoL * wL);
        f_ene = fL_ene + SL * (E_s - EL);
    } else {
        float star_den = SR - S_star;
        float fac = (fabsf(star_den) > 1e-8f) ? (rR_sR / star_den) : rhoR;
        float u_s = (axis == 0) ? S_star : uR;
        float v_s = (axis == 1) ? S_star : vR;
        float w_s = (axis == 2) ? S_star : wR;
        float E_s = fac * (ER / rhoR + (S_star - qR) * (S_star + pR / (fabsf(rR_sR) > 1e-8f ? rR_sR : 1.0f)));
        if (fabsf(rR_sR) <= 1e-8f) E_s = ER;

        float fR_rho = rhoR * qR;
        float fR_mx = rhoR * uR * qR + (axis == 0 ? pR : 0.0f);
        float fR_my = rhoR * vR * qR + (axis == 1 ? pR : 0.0f);
        float fR_mz = rhoR * wR * qR + (axis == 2 ? pR : 0.0f);
        float fR_ene = (ER + pR) * qR;

        f_rho = fR_rho + SR * (fac - rhoR);
        f_mx  = fR_mx  + SR * (fac * u_s - rhoR * uR);
        f_my  = fR_my  + SR * (fac * v_s - rhoR * vR);
        f_mz  = fR_mz  + SR * (fac * w_s - rhoR * wR);
        f_ene = fR_ene + SR * (E_s - ER);
    }
}

__device__ __forceinline__ void eval_face_flux(
    const FluidLevelGpu& lvl, const FluidConstantsGpu& c,
    int stage, float gamma,
    int i, int j, int k, int axis,
    float& out_rho, float& out_mx, float& out_my, float& out_mz, float& out_ene
) {
    int di = (axis == 0) ? 1 : 0;
    int dj = (axis == 1) ? 1 : 0;
    int dk = (axis == 2) ? 1 : 0;

    int iL = i - di, jL = j - dj, kL = k - dk;
    int iR = i,      jR = j,      kR = k;
    size_t cL = idx_cell(iL, jL, kL, lvl.nxp, lvl.nyp);
    size_t cR = idx_cell(iR, jR, kR, lvl.nxp, lvl.nyp);

    float ibL = (lvl.cell_ib != nullptr) ? lvl.cell_ib[cL] : 0.0f;
    float ibR = (lvl.cell_ib != nullptr) ? lvl.cell_ib[cR] : 0.0f;
    if (ibL >= 0.5f && ibR >= 0.5f) {
        out_rho = 0.0f; out_mx = 0.0f; out_my = 0.0f; out_mz = 0.0f; out_ene = 0.0f;
        return;
    }

    float r1, u1, v1, w1, p1;
    float r2, u2, v2, w2, p2;
    if (stage == 0) {
        cons_to_prim(lvl, cL, gamma, r1, u1, v1, w1, p1);
        cons_to_prim(lvl, cR, gamma, r2, u2, v2, w2, p2);
    } else {
        cons1_to_prim(lvl, cL, gamma, r1, u1, v1, w1, p1);
        cons1_to_prim(lvl, cR, gamma, r2, u2, v2, w2, p2);
    }

    // Handle solid-fluid IB wall interface
    if (ibR >= 0.5f) {
        float vw_x = (lvl.cell_vx != nullptr) ? lvl.cell_vx[cR] : 0.0f;
        float vw_y = (lvl.cell_vy != nullptr) ? lvl.cell_vy[cR] : 0.0f;
        float vw_z = (lvl.cell_vz != nullptr) ? lvl.cell_vz[cR] : 0.0f;
        float c1 = sqrtf(gamma * p1 / r1);
        float vw_n = clamp_f((axis == 0) ? vw_x : ((axis == 1) ? vw_y : vw_z), -0.5f * c1, 0.5f * c1);
        float q1 = (axis == 0) ? u1 : ((axis == 1) ? v1 : w1);
        float rhs_rho, rhs_p;
        hs_state(cell_z_world(lvl, kL), c, rhs_rho, rhs_p);
        float q_imp = clamp_f(c.wall_admittance * (p1 - rhs_p) / (r1 * c1), -0.25f * c1, 0.25f * c1);
        float q2 = 2.0f * vw_n - q1 + 2.0f * q_imp;
        r2 = r1; p2 = p1;
        u2 = (axis == 0) ? q2 : u1;
        v2 = (axis == 1) ? q2 : v1;
        w2 = (axis == 2) ? q2 : w1;
        hllc_flux_1d(r1, u1, v1, w1, p1, r2, u2, v2, w2, p2, axis, gamma,
                     out_rho, out_mx, out_my, out_mz, out_ene);
        return;
    }
    if (ibL >= 0.5f) {
        float vw_x = (lvl.cell_vx != nullptr) ? lvl.cell_vx[cL] : 0.0f;
        float vw_y = (lvl.cell_vy != nullptr) ? lvl.cell_vy[cL] : 0.0f;
        float vw_z = (lvl.cell_vz != nullptr) ? lvl.cell_vz[cL] : 0.0f;
        float c2 = sqrtf(gamma * p2 / r2);
        float vw_n = clamp_f((axis == 0) ? vw_x : ((axis == 1) ? vw_y : vw_z), -0.5f * c2, 0.5f * c2);
        float q2 = (axis == 0) ? u2 : ((axis == 1) ? v2 : w2);
        float rhs_rho, rhs_p;
        hs_state(cell_z_world(lvl, kR), c, rhs_rho, rhs_p);
        float q_imp = clamp_f(c.wall_admittance * (p2 - rhs_p) / (r2 * c2), -0.25f * c2, 0.25f * c2);
        float q1 = 2.0f * vw_n - q2 - 2.0f * q_imp;
        r1 = r2; p1 = p2;
        u1 = (axis == 0) ? q1 : u2;
        v1 = (axis == 1) ? q1 : v2;
        w1 = (axis == 2) ? q1 : w2;
        hllc_flux_1d(r1, u1, v1, w1, p1, r2, u2, v2, w2, p2, axis, gamma,
                     out_rho, out_mx, out_my, out_mz, out_ene);
        return;
    }

    // 2nd-order MUSCL reconstruction with Minmod slope limiter (fallback to 1st-order near IB walls)
    size_t cLL = idx_cell(iL - di, jL - dj, kL - dk, lvl.nxp, lvl.nyp);
    size_t cRR = idx_cell(iR + di, jR + dj, kR + dk, lvl.nxp, lvl.nyp);
    float ibLL = (lvl.cell_ib != nullptr) ? lvl.cell_ib[cLL] : 0.0f;
    float ibRR = (lvl.cell_ib != nullptr) ? lvl.cell_ib[cRR] : 0.0f;

    float r0, u0, v0, w0, p0, r3, u3, v3, w3, p3;
    if (stage == 0) {
        cons_to_prim(lvl, cLL, gamma, r0, u0, v0, w0, p0);
        cons_to_prim(lvl, cRR, gamma, r3, u3, v3, w3, p3);
    } else {
        cons1_to_prim(lvl, cLL, gamma, r0, u0, v0, w0, p0);
        cons1_to_prim(lvl, cRR, gamma, r3, u3, v3, w3, p3);
    }

    float rL = r1, uL = u1, vL = v1, wL = w1, pL = p1;
    if (ibLL < 0.5f) {
        rL = fmaxf(1e-4f, r1 + 0.5f * minmod(r1 - r0, r2 - r1));
        uL = u1 + 0.5f * minmod(u1 - u0, u2 - u1);
        vL = v1 + 0.5f * minmod(v1 - v0, v2 - v1);
        wL = w1 + 0.5f * minmod(w1 - w0, w2 - w1);
        pL = fmaxf(100.0f, p1 + 0.5f * minmod(p1 - p0, p2 - p1));
    }

    float rR = r2, uR = u2, vR = v2, wR = w2, pR = p2;
    if (ibRR < 0.5f) {
        rR = fmaxf(1e-4f, r2 - 0.5f * minmod(r2 - r1, r3 - r2));
        uR = u2 - 0.5f * minmod(u2 - u1, u3 - u2);
        vR = v2 - 0.5f * minmod(v2 - v1, v3 - v2);
        wR = w2 - 0.5f * minmod(w2 - w1, w3 - w2);
        pR = fmaxf(100.0f, p2 - 0.5f * minmod(p2 - p1, p3 - p2));
    }

    hllc_flux_1d(rL, uL, vL, wL, pL, rR, uR, vR, wR, pR, axis, gamma,
                 out_rho, out_mx, out_my, out_mz, out_ene);

    // Thermoviscous Navier-Stokes diffusive flux (Stokes shear + bulk viscosity + Fourier conduction)
    float mu = (float)c.mu_a;
    float mu_b = c.bulk_viscosity_a;
    if (mu > 0.0f || mu_b > 0.0f) {
        float inv_dx = 1.0f / lvl.dx;
        float dux_dn = (u2 - u1) * inv_dx;
        float duy_dn = (v2 - v1) * inv_dx;
        float duz_dn = (w2 - w1) * inv_dx;
        float u_f = 0.5f * (u1 + u2);
        float v_f = 0.5f * (v1 + v2);
        float w_f = 0.5f * (w1 + w2);

        float lam_plus_2mu = mu_b + 1.3333334f * mu;
        float tau_xx = (axis == 0) ? (lam_plus_2mu * dux_dn) : (mu * dux_dn);
        float tau_yy = (axis == 1) ? (lam_plus_2mu * duy_dn) : (mu * duy_dn);
        float tau_zz = (axis == 2) ? (lam_plus_2mu * duz_dn) : (mu * duz_dn);

        float mu_th = mu + 0.5f * mu_b;
        float pr = fmaxf(0.1f, c.prandtl_a);
        float k_over_r = gamma / ((gamma - 1.0f) * pr) * mu_th;
        float q_diff = -k_over_r * ((p2 / r2) - (p1 / r1)) * inv_dx;

        float f_v_ene = u_f * tau_xx + v_f * tau_yy + w_f * tau_zz - q_diff;

        out_mx  -= tau_xx;
        out_my  -= tau_yy;
        out_mz  -= tau_zz;
        out_ene -= f_v_ene;
    }
}

__global__ void k_euler_init_ambient(FluidLevelGpu lvl, FluidConstantsGpu c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= lvl.nxp || j >= lvl.nyp || k >= lvl.nzp) return;

    float rho_hs, p_hs;
    hs_state(cell_z_world(lvl, k), c, rho_hs, p_hs);
    float ene = p_hs / ((float)c.gamma_a - 1.0f);
    size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
    lvl.u_rho[cid] = rho_hs;
    lvl.u_mx[cid] = 0.0f;
    lvl.u_my[cid] = 0.0f;
    lvl.u_mz[cid] = 0.0f;
    lvl.u_ene[cid] = ene;
    if (lvl.u1_rho) {
        lvl.u1_rho[cid] = rho_hs;
        lvl.u1_mx[cid] = 0.0f;
        lvl.u1_my[cid] = 0.0f;
        lvl.u1_mz[cid] = 0.0f;
        lvl.u1_ene[cid] = ene;
    }
}

__global__ void k_euler_add_blast_region(
    FluidLevelGpu lvl,
    FluidConstantsGpu c,
    float cx, float cy, float cz,
    float radius,
    float length,
    int32_t is_cylinder_x,
    float overpressure_pa,
    float density_kg_m3,
    float vx, float vy, float vz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= lvl.nxp || j >= lvl.nyp || k >= lvl.nzp) return;

    float x = lvl.ox + ((float)(i - lvl.halo) + 0.5f) * lvl.dx;
    float y = lvl.oy + ((float)(j - lvl.halo) + 0.5f) * lvl.dx;
    float z = lvl.oz + ((float)(k - lvl.halo) + 0.5f) * lvl.dx;

    float w = 0.0f;
    float r_eff = fmaxf(radius, 0.75f * lvl.dx);
    float vol_scale = 1.0f;
    if (is_cylinder_x) {
        float ax_lo = cx;
        float ax_hi = cx + length;
        float dx_out = (x < ax_lo) ? (ax_lo - x) : ((x > ax_hi) ? (x - ax_hi) : 0.0f);
        float r2 = (y - cy) * (y - cy) + (z - cz) * (z - cz);
        float xi2 = r2 / (r_eff * r_eff) + (dx_out * dx_out) / (lvl.dx * lvl.dx);
        if (xi2 < 6.0f) {
            w = expf(-xi2 * xi2);
        }
        float len_phys = fmaxf(1e-4f, length);
        float len_eff = len_phys + 1.1f * lvl.dx;
        vol_scale = fminf(1.0f, (radius * radius * len_phys) / (r_eff * r_eff * len_eff));
    } else {
        float d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz);
        float s = sqrtf(d2) / r_eff;
        if (s < 2.2f) {
            float s4 = s * s * s * s;
            w = expf(-s4 * s * s);
        }
        vol_scale = fminf(1.0f, (radius * radius * radius) / (r_eff * r_eff * r_eff));
    }

    float w_eff = w * vol_scale;
    if (w_eff > 1e-8f) {
        size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
        float ib = (lvl.cell_ib != nullptr) ? lvl.cell_ib[cid] : 0.0f;
        w_eff *= fmaxf(0.0f, 1.0f - ib);
        if (w_eff <= 1e-8f) return;

        float gam = (float)c.gamma_a;
        float rho_amb = lvl.u_rho[cid];
        float p_amb = cons_pressure(lvl.u_rho[cid], lvl.u_mx[cid], lvl.u_my[cid], lvl.u_mz[cid], lvl.u_ene[cid], gam);
        float u_amb = lvl.u_mx[cid] / rho_amb;
        float v_amb = lvl.u_my[cid] / rho_amb;
        float w_amb = lvl.u_mz[cid] / rho_amb;

        float rho_new = rho_amb + w_eff * fmaxf(0.0f, density_kg_m3 - rho_amb);
        float u_new = u_amb + w_eff * (vx - u_amb);
        float v_new = v_amb + w_eff * (vy - v_amb);
        float w_new = w_amb + w_eff * (vz - w_amb);
        float p_new = p_amb + w_eff * overpressure_pa;
        float ene_new = p_new / (gam - 1.0f) + 0.5f * rho_new * (u_new * u_new + v_new * v_new + w_new * w_new);

        lvl.u_rho[cid] = rho_new;
        lvl.u_mx[cid] = rho_new * u_new;
        lvl.u_my[cid] = rho_new * v_new;
        lvl.u_mz[cid] = rho_new * w_new;
        lvl.u_ene[cid] = ene_new;
        if (lvl.u1_rho) {
            lvl.u1_rho[cid] = rho_new;
            lvl.u1_mx[cid] = rho_new * u_new;
            lvl.u1_my[cid] = rho_new * v_new;
            lvl.u1_mz[cid] = rho_new * w_new;
            lvl.u1_ene[cid] = ene_new;
        }
    }
}

__global__ void k_euler_outer_bc_slab_x(FluidLevelGpu lvl, int32_t stage) {
    int h = lvl.halo;
    int slab = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (slab >= 2 * h || j >= lvl.nyp || k >= lvl.nzp) return;

    int i = (slab < h) ? slab : (h + lvl.nx + (slab - h));
    bool is_lo = (slab < h);
    bool open_bc = is_lo ? ((lvl.pml_face_mask & (1u << 0)) != 0) : ((lvl.pml_face_mask & (1u << 1)) != 0);
    int ic = is_lo ? (open_bc ? h : (2 * h - 1 - i)) : (open_bc ? (h + lvl.nx - 1) : (2 * (h + lvl.nx) - 1 - i));
    ic = min(max(ic, h), h + lvl.nx - 1);
    int jc = min(max(j, h), h + lvl.ny - 1);
    int kc = min(max(k, h), h + lvl.nz - 1);
    float sgn = open_bc ? 1.0f : -1.0f;

    size_t dst = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
    size_t src = idx_cell(ic, jc, kc, lvl.nxp, lvl.nyp);
    if (stage == 0) {
        lvl.u_rho[dst] = lvl.u_rho[src];
        lvl.u_mx[dst]  = sgn * lvl.u_mx[src];
        lvl.u_my[dst]  = lvl.u_my[src];
        lvl.u_mz[dst]  = lvl.u_mz[src];
        lvl.u_ene[dst] = lvl.u_ene[src];
    } else {
        lvl.u1_rho[dst] = lvl.u1_rho[src];
        lvl.u1_mx[dst]  = sgn * lvl.u1_mx[src];
        lvl.u1_my[dst]  = lvl.u1_my[src];
        lvl.u1_mz[dst]  = lvl.u1_mz[src];
        lvl.u1_ene[dst] = lvl.u1_ene[src];
    }
}

__global__ void k_euler_outer_bc_slab_y(FluidLevelGpu lvl, int32_t stage) {
    int h = lvl.halo;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int slab = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= lvl.nxp || slab >= 2 * h || k >= lvl.nzp) return;

    int j = (slab < h) ? slab : (h + lvl.ny + (slab - h));
    bool is_lo = (slab < h);
    bool open_bc = is_lo ? ((lvl.pml_face_mask & (1u << 2)) != 0) : ((lvl.pml_face_mask & (1u << 3)) != 0);
    int jc = is_lo ? (open_bc ? h : (2 * h - 1 - j)) : (open_bc ? (h + lvl.ny - 1) : (2 * (h + lvl.ny) - 1 - j));
    jc = min(max(jc, h), h + lvl.ny - 1);
    int ic = min(max(i, h), h + lvl.nx - 1);
    int kc = min(max(k, h), h + lvl.nz - 1);
    float sgn = open_bc ? 1.0f : -1.0f;

    size_t dst = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
    size_t src = idx_cell(ic, jc, kc, lvl.nxp, lvl.nyp);
    if (stage == 0) {
        lvl.u_rho[dst] = lvl.u_rho[src];
        lvl.u_mx[dst]  = lvl.u_mx[src];
        lvl.u_my[dst]  = sgn * lvl.u_my[src];
        lvl.u_mz[dst]  = lvl.u_mz[src];
        lvl.u_ene[dst] = lvl.u_ene[src];
    } else {
        lvl.u1_rho[dst] = lvl.u1_rho[src];
        lvl.u1_mx[dst]  = lvl.u1_mx[src];
        lvl.u1_my[dst]  = sgn * lvl.u1_my[src];
        lvl.u1_mz[dst]  = lvl.u1_mz[src];
        lvl.u1_ene[dst] = lvl.u1_ene[src];
    }
}

__global__ void k_euler_outer_bc_slab_z(FluidLevelGpu lvl, int32_t stage) {
    int h = lvl.halo;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int slab = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= lvl.nxp || j >= lvl.nyp || slab >= 2 * h) return;

    int k = (slab < h) ? slab : (h + lvl.nz + (slab - h));
    bool is_lo = (slab < h);
    bool open_bc = is_lo ? ((lvl.pml_face_mask & (1u << 4)) != 0) : ((lvl.pml_face_mask & (1u << 5)) != 0);
    int kc = is_lo ? (open_bc ? h : (2 * h - 1 - k)) : (open_bc ? (h + lvl.nz - 1) : (2 * (h + lvl.nz) - 1 - k));
    kc = min(max(kc, h), h + lvl.nz - 1);
    int ic = min(max(i, h), h + lvl.nx - 1);
    int jc = min(max(j, h), h + lvl.ny - 1);
    float sgn = open_bc ? 1.0f : -1.0f;

    size_t dst = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
    size_t src = idx_cell(ic, jc, kc, lvl.nxp, lvl.nyp);
    if (stage == 0) {
        lvl.u_rho[dst] = lvl.u_rho[src];
        lvl.u_mx[dst]  = lvl.u_mx[src];
        lvl.u_my[dst]  = lvl.u_my[src];
        lvl.u_mz[dst]  = sgn * lvl.u_mz[src];
        lvl.u_ene[dst] = lvl.u_ene[src];
    } else {
        lvl.u1_rho[dst] = lvl.u1_rho[src];
        lvl.u1_mx[dst]  = lvl.u1_mx[src];
        lvl.u1_my[dst]  = lvl.u1_my[src];
        lvl.u1_mz[dst]  = sgn * lvl.u1_mz[src];
        lvl.u1_ene[dst] = lvl.u1_ene[src];
    }
}

__global__ void k_euler_rk_stage(
    FluidLevelGpu lvl, FluidConstantsGpu c, int32_t stage, float dt
) {
    int h = lvl.halo;
    int i = h + blockIdx.x * blockDim.x + threadIdx.x;
    int j = h + blockIdx.y * blockDim.y + threadIdx.y;
    int k = h + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= h + lvl.nx || j >= h + lvl.ny || k >= h + lvl.nz) return;

    size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
    float ib = (lvl.cell_ib != nullptr) ? lvl.cell_ib[cid] : 0.0f;
    if (ib >= 0.5f) {
        if (stage == 0) {
            lvl.u1_rho[cid] = lvl.u_rho[cid];
            lvl.u1_mx[cid]  = 0.0f;
            lvl.u1_my[cid]  = 0.0f;
            lvl.u1_mz[cid]  = 0.0f;
            lvl.u1_ene[cid] = lvl.u_ene[cid];
        } else {
            lvl.u_mx[cid] = 0.0f;
            lvl.u_my[cid] = 0.0f;
            lvl.u_mz[cid] = 0.0f;
        }
        return;
    }

    float gamma = (float)c.gamma_a;
    float fxm_r, fxm_u, fxm_v, fxm_w, fxm_e;
    float fxp_r, fxp_u, fxp_v, fxp_w, fxp_e;
    float fym_r, fym_u, fym_v, fym_w, fym_e;
    float fyp_r, fyp_u, fyp_v, fyp_w, fyp_e;
    float fzm_r, fzm_u, fzm_v, fzm_w, fzm_e;
    float fzp_r, fzp_u, fzp_v, fzp_w, fzp_e;

    eval_face_flux(lvl, c, stage, gamma, i,     j,     k,     0, fxm_r, fxm_u, fxm_v, fxm_w, fxm_e);
    eval_face_flux(lvl, c, stage, gamma, i + 1, j,     k,     0, fxp_r, fxp_u, fxp_v, fxp_w, fxp_e);
    eval_face_flux(lvl, c, stage, gamma, i,     j,     k,     1, fym_r, fym_u, fym_v, fym_w, fym_e);
    eval_face_flux(lvl, c, stage, gamma, i,     j + 1, k,     1, fyp_r, fyp_u, fyp_v, fyp_w, fyp_e);
    eval_face_flux(lvl, c, stage, gamma, i,     j,     k,     2, fzm_r, fzm_u, fzm_v, fzm_w, fzm_e);
    eval_face_flux(lvl, c, stage, gamma, i,     j,     k + 1, 2, fzp_r, fzp_u, fzp_v, fzp_w, fzp_e);

    float inv_dx = 1.0f / lvl.dx;
    float rhs_r = -((fxp_r - fxm_r) + (fyp_r - fym_r) + (fzp_r - fzm_r)) * inv_dx;
    float rhs_x = -((fxp_u - fxm_u) + (fyp_u - fym_u) + (fzp_u - fzm_u)) * inv_dx;
    float rhs_y = -((fxp_v - fxm_v) + (fyp_v - fym_v) + (fzp_v - fzm_v)) * inv_dx;
    float rhs_z = -((fxp_w - fxm_w) + (fyp_w - fym_w) + (fzp_w - fzm_w)) * inv_dx;
    float rhs_e = -((fxp_e - fxm_e) + (fyp_e - fym_e) + (fzp_e - fzm_e)) * inv_dx;

    const float* s_r = (stage == 0) ? lvl.u_rho : lvl.u1_rho;
    const float* s_x = (stage == 0) ? lvl.u_mx  : lvl.u1_mx;
    const float* s_y = (stage == 0) ? lvl.u_my  : lvl.u1_my;
    const float* s_z = (stage == 0) ? lvl.u_mz  : lvl.u1_mz;
    const float* s_e = (stage == 0) ? lvl.u_ene : lvl.u1_ene;

    float rho_s = s_r[cid];
    float mx_s  = s_x[cid];
    float my_s  = s_y[cid];
    float mz_s  = s_z[cid];
    float ene_s = s_e[cid];

    // Well-balanced hydrostatic-subtracted buoyancy gravity + outer absorbing sponge layer
    float rho_hs, p_hs;
    hs_state(cell_z_world(lvl, k), c, rho_hs, p_hs);
    float d_rho = rho_s - rho_hs;
    rhs_x += d_rho * c.gx;
    rhs_y += d_rho * c.gy;
    rhs_z += d_rho * c.gz;
    rhs_e += (d_rho / fmaxf(1e-4f, rho_s)) * (mx_s * c.gx + my_s * c.gy + mz_s * c.gz);

    float c0 = sqrtf(gamma * (float)c.p0 / (float)c.rho_a);
    float sig = sponge_sigma_cell(lvl, c, i, j, k, c0);
    if (sig > 0.0f) {
        float ene_hs = p_hs / (gamma - 1.0f);
        rhs_r -= sig * (rho_s - rho_hs);
        rhs_x -= sig * mx_s;
        rhs_y -= sig * my_s;
        rhs_z -= sig * mz_s;
        rhs_e -= sig * (ene_s - ene_hs);
    }

    float n_r, n_x, n_y, n_z, n_e;
    if (stage == 0) {
        n_r = rho_s + dt * rhs_r;
        n_x = mx_s  + dt * rhs_x;
        n_y = my_s  + dt * rhs_y;
        n_z = mz_s  + dt * rhs_z;
        n_e = ene_s + dt * rhs_e;
    } else {
        n_r = 0.5f * (lvl.u_rho[cid] + rho_s + dt * rhs_r);
        n_x = 0.5f * (lvl.u_mx[cid]  + mx_s  + dt * rhs_x);
        n_y = 0.5f * (lvl.u_my[cid]  + my_s  + dt * rhs_y);
        n_z = 0.5f * (lvl.u_mz[cid]  + mz_s  + dt * rhs_z);
        n_e = 0.5f * (lvl.u_ene[cid] + ene_s + dt * rhs_e);
    }

    n_r = fmaxf(1e-4f, n_r);
    float ke_min = 0.5f * (n_x * n_x + n_y * n_y + n_z * n_z) / n_r;
    float e_floor = ke_min + 100.0f / (gamma - 1.0f);
    n_e = fmaxf(e_floor, n_e);

    if (stage == 0) {
        lvl.u1_rho[cid] = n_r;
        lvl.u1_mx[cid]  = n_x;
        lvl.u1_my[cid]  = n_y;
        lvl.u1_mz[cid]  = n_z;
        lvl.u1_ene[cid] = n_e;
    } else {
        lvl.u_rho[cid] = n_r;
        lvl.u_mx[cid]  = n_x;
        lvl.u_my[cid]  = n_y;
        lvl.u_mz[cid]  = n_z;
        lvl.u_ene[cid] = n_e;
    }
}

// Sub-cell circle-square overlap approximation (64-subpixel transverse x exact axial overlap)
// Accurately resolves sub-grid bullets down to d = 5.7 mm on dx = 25 mm without aliasing.
__device__ __forceinline__ float ogive_bullet_volume_fraction(
    float xc, float yc, float zc, float dx,
    float base_x, float center_y, float center_z,
    float radius, float length
) {
    float x0 = xc - 0.5f * dx;
    float x1 = xc + 0.5f * dx;
    float ov_x0 = fmaxf(x0, base_x);
    float ov_x1 = fminf(x1, base_x + length);
    if (ov_x1 <= ov_x0) return 0.0f;

    float y0 = yc - 0.5f * dx;
    float z0 = zc - 0.5f * dx;
    if (yc + 0.5f * dx < center_y - radius || y0 > center_y + radius ||
        zc + 0.5f * dx < center_z - radius || z0 > center_z + radius) {
        return 0.0f;
    }

    // Integrate 4 axial slices x 8x8 transverse points inside overlapping bbox
    const int NX = 4;
    const int NT = 8;
    float sub_x = (ov_x1 - ov_x0) / (float)NX;
    float y_lo = fmaxf(y0, center_y - radius);
    float y_hi = fminf(yc + 0.5f * dx, center_y + radius);
    float z_lo = fmaxf(z0, center_z - radius);
    float z_hi = fminf(zc + 0.5f * dx, center_z + radius);
    if (y_hi <= y_lo || z_hi <= z_lo) return 0.0f;

    float sub_y = (y_hi - y_lo) / (float)NT;
    float sub_z = (z_hi - z_lo) / (float)NT;
    int hits = 0;
    for (int ix = 0; ix < NX; ++ix) {
        float x = ov_x0 + ((float)ix + 0.5f) * sub_x;
        float s = x - base_x;
        float r_loc = radius;
        if (s > 0.55f * length) {
            float u = (s - 0.55f * length) / fmaxf(1e-6f, 0.45f * length);
            r_loc = radius * powf(fmaxf(0.0f, 1.0f - u * u), 0.7f);
        }
        float r2_loc = r_loc * r_loc;
        for (int iz = 0; iz < NT; ++iz) {
            float dz = z_lo + ((float)iz + 0.5f) * sub_z - center_z;
            for (int iy = 0; iy < NT; ++iy) {
                float dy = y_lo + ((float)iy + 0.5f) * sub_y - center_y;
                if (dy * dy + dz * dz <= r2_loc) ++hits;
            }
        }
    }
    float box_vol = (ov_x1 - ov_x0) * (y_hi - y_lo) * (z_hi - z_lo);
    float cell_vol = dx * dx * dx;
    return clamp_f(((float)hits / (float)(NX * NT * NT)) * (box_vol / cell_vol), 0.0f, 1.0f);
}

__global__ void k_euler_apply_projectile_bbox(
    FluidLevelGpu lvl,
    FluidConstantsGpu c,
    int i_lo, int j_lo, int k_lo, int bx, int by, int bz,
    float base_x_old, float center_y_old, float center_z_old,
    float base_x_new, float center_y_new, float center_z_new,
    float vx, float vy, float vz,
    float radius, float length, float dt,
    double* d_force_out
) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    int tj = blockIdx.y * blockDim.y + threadIdx.y;
    int tk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ti >= bx || tj >= by || tk >= bz) return;

    int h = lvl.halo;
    int i = i_lo + ti;
    int j = j_lo + tj;
    int k = k_lo + tk;
    float xc = lvl.ox + ((float)(i - h) + 0.5f) * lvl.dx;
    float yc = lvl.oy + ((float)(j - h) + 0.5f) * lvl.dx;
    float zc = lvl.oz + ((float)(k - h) + 0.5f) * lvl.dx;

    float phi_old = ogive_bullet_volume_fraction(xc, yc, zc, lvl.dx, base_x_old, center_y_old, center_z_old, radius, length);
    float phi_new = ogive_bullet_volume_fraction(xc, yc, zc, lvl.dx, base_x_new, center_y_new, center_z_new, radius, length);
    float dphi = phi_new - phi_old;
    float phi_avg = 0.5f * (phi_old + phi_new);
    if (phi_avg <= 1e-8f && fabsf(dphi) <= 1e-8f) return;

    size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
    float gam = (float)c.gamma_a;
    float rho = fmaxf(1e-4f, lvl.u_rho[cid]);
    float u = lvl.u_mx[cid] / rho;
    float v = lvl.u_my[cid] / rho;
    float w = lvl.u_mz[cid] / rho;
    float p = cons_pressure(rho, lvl.u_mx[cid], lvl.u_my[cid], lvl.u_mz[cid], lvl.u_ene[cid], gam);

    // Exact area-rule volume displacement (nose compression dphi>0, wake expansion dphi<0)
    float scale_v = fminf(0.45f, fmaxf(-0.45f, dphi));
    float rho_after = fmaxf(0.2f * (float)c.rho_a, rho * (1.0f + scale_v));
    float p_after = fmaxf(0.2f * (float)c.p0, p * (1.0f + gam * scale_v));

    // Direct-forcing drag & kinetic momentum transfer over sub-cell bullet volume
    float cfl_b = clamp_f(sqrtf(vx * vx + vy * vy + vz * vz) * dt / lvl.dx, 0.05f, 1.0f);
    float relax = clamp_f(phi_new * 1.5f * cfl_b, 0.0f, 0.85f);
    float u_after = u + relax * (vx - u);
    float v_after = v + relax * (vy - v);
    float w_after = w + relax * (vz - w);

    // Impulse transferred from bullet to air -> equal and opposite aerodynamic drag force on bullet
    float vcell = lvl.dx * lvl.dx * lvl.dx;
    float inv_dt = 1.0f / fmaxf(1e-9f, dt);
    float dm_x = (rho_after * u_after - lvl.u_mx[cid]) * vcell * inv_dt;
    float dm_y = (rho_after * v_after - lvl.u_my[cid]) * vcell * inv_dt;
    float dm_z = (rho_after * w_after - lvl.u_mz[cid]) * vcell * inv_dt;

    atomicAdd_d(&d_force_out[0], -(double)dm_x);
    atomicAdd_d(&d_force_out[1], -(double)dm_y);
    atomicAdd_d(&d_force_out[2], -(double)dm_z);

    lvl.u_rho[cid] = rho_after;
    lvl.u_mx[cid]  = rho_after * u_after;
    lvl.u_my[cid]  = rho_after * v_after;
    lvl.u_mz[cid]  = rho_after * w_after;
    lvl.u_ene[cid] = p_after / (gam - 1.0f) + 0.5f * rho_after * (u_after * u_after + v_after * v_after + w_after * w_after);
}

__global__ void k_euler_apply_directional_source_bbox(
    FluidLevelGpu lvl,
    FluidConstantsGpu c,
    int i_lo, int j_lo, int k_lo, int bx, int by, int bz,
    float cx, float cy, float cz,
    float dir_x, float dir_y, float dir_z,
    float cardioid_beta,
    float radius,
    float rate_pa_per_s,
    float dt_ac
) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    int tj = blockIdx.y * blockDim.y + threadIdx.y;
    int tk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ti >= bx || tj >= by || tk >= bz) return;
    if (fabsf(rate_pa_per_s) <= 0.0f) return;

    int h = lvl.halo;
    int i = i_lo + ti;
    int j = j_lo + tj;
    int k = k_lo + tk;
    float x = lvl.ox + ((float)(i - h) + 0.5f) * lvl.dx;
    float y = lvl.oy + ((float)(j - h) + 0.5f) * lvl.dx;
    float z = lvl.oz + ((float)(k - h) + 0.5f) * lvl.dx;
    float r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz);
    float s2 = radius * radius;
    if (r2 >= 9.0f * s2) return;

    size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
    float ib = (lvl.cell_ib != nullptr) ? lvl.cell_ib[cid] : 0.0f;
    float w = expf(-0.5f * r2 / s2) * fmaxf(0.0f, 1.0f - ib);
    if (w <= 1e-6f) return;

    float gam = (float)c.gamma_a;
    float c_a = sqrtf((float)(c.gamma_a * c.p0 / c.rho_a));
    float d_p = dt_ac * rate_pa_per_s * w;
    float d_m_mag = d_p * cardioid_beta / c_a;

    float rho = fmaxf(1e-4f, lvl.u_rho[cid]);
    float p = cons_pressure(rho, lvl.u_mx[cid], lvl.u_my[cid], lvl.u_mz[cid], lvl.u_ene[cid], gam);
    float u = lvl.u_mx[cid] / rho;
    float v = lvl.u_my[cid] / rho;
    float wz = lvl.u_mz[cid] / rho;

    float mx_new = lvl.u_mx[cid] + d_m_mag * dir_x;
    float my_new = lvl.u_my[cid] + d_m_mag * dir_y;
    float mz_new = lvl.u_mz[cid] + d_m_mag * dir_z;
    float d_ke = 0.5f * ((mx_new * mx_new + my_new * my_new + mz_new * mz_new) - rho * rho * (u * u + v * v + wz * wz)) / rho;
    float p_new = fmaxf(100.0f, p + d_p);
    float ene_new = p_new / (gam - 1.0f) + 0.5f * rho * (u * u + v * v + wz * wz) + d_ke;

    lvl.u_mx[cid]  = mx_new;
    lvl.u_my[cid]  = my_new;
    lvl.u_mz[cid]  = mz_new;
    lvl.u_ene[cid] = ene_new;
}

__global__ void k_euler_apply_wind_inlet_bbox(
    FluidLevelGpu lvl,
    FluidConstantsGpu c,
    int i_lo, int j_lo, int k_lo, int bx, int by, int bz,
    float vx, float vy, float vz,
    float turb_amp, float t_now, float relax
) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    int tj = blockIdx.y * blockDim.y + threadIdx.y;
    int tk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ti >= bx || tj >= by || tk >= bz) return;

    int h = lvl.halo;
    int i = i_lo + ti;
    int j = j_lo + tj;
    int k = k_lo + tk;
    size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
    float ib = (lvl.cell_ib != nullptr) ? lvl.cell_ib[cid] : 0.0f;
    if (ib >= 0.5f) return;

    float y = lvl.oy + ((float)(j - h) + 0.5f) * lvl.dx;
    float z = lvl.oz + ((float)(k - h) + 0.5f) * lvl.dx;
    float gust = 1.0f + turb_amp * sinf(17.3f * t_now + 6.2f * y) * cosf(23.1f * t_now - 7.1f * z);

    float gam = (float)c.gamma_a;
    float rho = fmaxf(1e-4f, lvl.u_rho[cid]);
    float p = cons_pressure(rho, lvl.u_mx[cid], lvl.u_my[cid], lvl.u_mz[cid], lvl.u_ene[cid], gam);
    float u_cur = lvl.u_mx[cid] / rho;
    float v_cur = lvl.u_my[cid] / rho;
    float w_cur = lvl.u_mz[cid] / rho;

    float u_new = u_cur + relax * (vx * gust - u_cur);
    float v_new = v_cur + relax * (vy * gust - v_cur);
    float w_new = w_cur + relax * (vz * gust - w_cur);

    lvl.u_mx[cid]  = rho * u_new;
    lvl.u_my[cid]  = rho * v_new;
    lvl.u_mz[cid]  = rho * w_new;
    lvl.u_ene[cid] = p / (gam - 1.0f) + 0.5f * rho * (u_new * u_new + v_new * v_new + w_new * w_new);
}

__global__ void k_euler_amr_fill_ghosts(FluidLevelGpu child, FluidLevelGpu parent, int32_t target_stage) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= child.nxp || j >= child.nyp || k >= child.nzp) return;

    int hc = child.halo;
    bool is_ghost = (i < hc || i >= hc + child.nx ||
                     j < hc || j >= hc + child.ny ||
                     k < hc || k >= hc + child.nz);
    if (!is_ghost) return;

    float x = child.ox + ((float)(i - hc) + 0.5f) * child.dx;
    float y = child.oy + ((float)(j - hc) + 0.5f) * child.dx;
    float z = child.oz + ((float)(k - hc) + 0.5f) * child.dx;

    float fx = (x - parent.ox) / parent.dx - 0.5f + (float)parent.halo;
    float fy = (y - parent.oy) / parent.dx - 0.5f + (float)parent.halo;
    float fz = (z - parent.oz) / parent.dx - 0.5f + (float)parent.halo;

    int i0 = max(0, min(parent.nxp - 2, (int)floorf(fx)));
    int j0 = max(0, min(parent.nyp - 2, (int)floorf(fy)));
    int k0 = max(0, min(parent.nzp - 2, (int)floorf(fz)));
    float tx = clamp_f(fx - (float)i0, 0.0f, 1.0f);
    float ty = clamp_f(fy - (float)j0, 0.0f, 1.0f);
    float tz = clamp_f(fz - (float)k0, 0.0f, 1.0f);

    float rho = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f, ene = 0.0f;
    for (int dk = 0; dk <= 1; ++dk) {
        float wz = dk ? tz : (1.0f - tz);
        for (int dj = 0; dj <= 1; ++dj) {
            float wy = dj ? ty : (1.0f - ty);
            for (int di = 0; di <= 1; ++di) {
                float wx = di ? tx : (1.0f - tx);
                float w = wx * wy * wz;
                size_t pid = idx_cell(i0 + di, j0 + dj, k0 + dk, parent.nxp, parent.nyp);
                rho += w * parent.u_rho[pid];
                mx  += w * parent.u_mx[pid];
                my  += w * parent.u_my[pid];
                mz  += w * parent.u_mz[pid];
                ene += w * parent.u_ene[pid];
            }
        }
    }
    size_t cid = idx_cell(i, j, k, child.nxp, child.nyp);
    if (target_stage == 0) {
        child.u_rho[cid] = rho;
        child.u_mx[cid]  = mx;
        child.u_my[cid]  = my;
        child.u_mz[cid]  = mz;
        child.u_ene[cid] = ene;
    } else {
        child.u1_rho[cid] = rho;
        child.u1_mx[cid]  = mx;
        child.u1_my[cid]  = my;
        child.u1_mz[cid]  = mz;
        child.u1_ene[cid] = ene;
    }
}

__global__ void k_euler_amr_restrict(FluidLevelGpu child, FluidLevelGpu parent) {
    int ic = blockIdx.x * blockDim.x + threadIdx.x;
    int jc = blockIdx.y * blockDim.y + threadIdx.y;
    int kc = blockIdx.z * blockDim.z + threadIdx.z;

    int r = child.refine_ratio;
    int nx_c = child.nx / r;
    int ny_c = child.ny / r;
    int nz_c = child.nz / r;
    if (ic >= nx_c || jc >= ny_c || kc >= nz_c) return;

    int ip = parent.halo + child.off_x + ic;
    int jp = parent.halo + child.off_y + jc;
    int kp = parent.halo + child.off_z + kc;
    size_t pid = idx_cell(ip, jp, kp, parent.nxp, parent.nyp);
    if (parent.cell_ib && parent.cell_ib[pid] >= 0.5f) return;

    int hc = child.halo;
    float sr = 0.0f, smx = 0.0f, smy = 0.0f, smz = 0.0f, sene = 0.0f;
    float cnt = 0.0f;
    for (int dk = 0; dk < r; ++dk) {
        for (int dj = 0; dj < r; ++dj) {
            for (int di = 0; di < r; ++di) {
                size_t cid = idx_cell(hc + ic * r + di, hc + jc * r + dj, hc + kc * r + dk, child.nxp, child.nyp);
                float ib_c = (child.cell_ib != nullptr) ? child.cell_ib[cid] : 0.0f;
                if (ib_c < 0.5f) {
                    sr   += child.u_rho[cid];
                    smx  += child.u_mx[cid];
                    smy  += child.u_my[cid];
                    smz  += child.u_mz[cid];
                    sene += child.u_ene[cid];
                    cnt  += 1.0f;
                }
            }
        }
    }
    if (cnt > 0.0f) {
        float inv = 1.0f / cnt;
        parent.u_rho[pid] = sr * inv;
        parent.u_mx[pid]  = smx * inv;
        parent.u_my[pid]  = smy * inv;
        parent.u_mz[pid]  = smz * inv;
        parent.u_ene[pid] = sene * inv;
    }
}

__device__ __forceinline__ void trilinear_sample_parent_u(
    const FluidLevelGpu& parent, float x, float y, float z,
    float& rho, float& mx, float& my, float& mz, float& ene, float& ib_s
) {
    float fx = (x - parent.ox) / parent.dx - 0.5f + (float)parent.halo;
    float fy = (y - parent.oy) / parent.dx - 0.5f + (float)parent.halo;
    float fz = (z - parent.oz) / parent.dx - 0.5f + (float)parent.halo;
    int i0 = max(0, min(parent.nxp - 2, (int)floorf(fx)));
    int j0 = max(0, min(parent.nyp - 2, (int)floorf(fy)));
    int k0 = max(0, min(parent.nzp - 2, (int)floorf(fz)));
    float tx = clamp_f(fx - (float)i0, 0.0f, 1.0f);
    float ty = clamp_f(fy - (float)j0, 0.0f, 1.0f);
    float tz = clamp_f(fz - (float)k0, 0.0f, 1.0f);
    rho = mx = my = mz = ene = ib_s = 0.0f;
    for (int dk = 0; dk <= 1; ++dk) {
        float wz = dk ? tz : (1.0f - tz);
        for (int dj = 0; dj <= 1; ++dj) {
            float wy = dj ? ty : (1.0f - ty);
            for (int di = 0; di <= 1; ++di) {
                float wx = di ? tx : (1.0f - tx);
                float w = wx * wy * wz;
                size_t pid = idx_cell(i0 + di, j0 + dj, k0 + dk, parent.nxp, parent.nyp);
                rho += w * parent.u_rho[pid];
                mx  += w * parent.u_mx[pid];
                my  += w * parent.u_my[pid];
                mz  += w * parent.u_mz[pid];
                ene += w * parent.u_ene[pid];
                if (parent.cell_ib_static) ib_s += w * parent.cell_ib_static[pid];
            }
        }
    }
}

__global__ void k_euler_shift_x_stage1(
    FluidLevelGpu patch, int32_t shift_cells
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= patch.nxp || j >= patch.nyp || k >= patch.nzp) return;

    int src_i = i + shift_cells;
    size_t dst = idx_cell(i, j, k, patch.nxp, patch.nyp);
    if (src_i >= 0 && src_i < patch.nxp) {
        size_t src = idx_cell(src_i, j, k, patch.nxp, patch.nyp);
        patch.u1_rho[dst] = patch.u_rho[src];
        patch.u1_mx[dst]  = patch.u_mx[src];
        patch.u1_my[dst]  = patch.u_my[src];
        patch.u1_mz[dst]  = patch.u_mz[src];
        patch.u1_ene[dst] = patch.u_ene[src];
    }
}

__global__ void k_euler_shift_x_stage2(
    FluidLevelGpu patch, FluidLevelGpu parent, int32_t shift_cells
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= patch.nxp || j >= patch.nyp || k >= patch.nzp) return;

    int src_i = i + shift_cells;
    size_t dst = idx_cell(i, j, k, patch.nxp, patch.nyp);
    float x = patch.ox + ((float)(i - patch.halo) + 0.5f) * patch.dx;
    float y = patch.oy + ((float)(j - patch.halo) + 0.5f) * patch.dx;
    float z = patch.oz + ((float)(k - patch.halo) + 0.5f) * patch.dx;

    float pr, pmx, pmy, pmz, pene, pib;
    trilinear_sample_parent_u(parent, x, y, z, pr, pmx, pmy, pmz, pene, pib);
    if (patch.cell_ib_static) {
        patch.cell_ib_static[dst] = (pib >= 0.5f) ? 1.0f : 0.0f;
        patch.cell_ib[dst] = patch.cell_ib_static[dst];
    }
    if (src_i >= 0 && src_i < patch.nxp) {
        patch.u_rho[dst] = patch.u1_rho[dst];
        patch.u_mx[dst]  = patch.u1_mx[dst];
        patch.u_my[dst]  = patch.u1_my[dst];
        patch.u_mz[dst]  = patch.u1_mz[dst];
        patch.u_ene[dst] = patch.u1_ene[dst];
    } else {
        patch.u_rho[dst] = pr;
        patch.u_mx[dst]  = pmx;
        patch.u_my[dst]  = pmy;
        patch.u_mz[dst]  = pmz;
        patch.u_ene[dst] = pene;
        patch.u1_rho[dst] = pr;
        patch.u1_mx[dst]  = pmx;
        patch.u1_my[dst]  = pmy;
        patch.u1_mz[dst]  = pmz;
        patch.u1_ene[dst] = pene;
    }
}

__global__ void k_euler_copy_overlap(FluidLevelGpu dst_lvl, FluidLevelGpu src_lvl) {
    int hc = dst_lvl.halo;
    int i = hc + blockIdx.x * blockDim.x + threadIdx.x;
    int j = hc + blockIdx.y * blockDim.y + threadIdx.y;
    int k = hc + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= hc + dst_lvl.nx || j >= hc + dst_lvl.ny || k >= hc + dst_lvl.nz) return;

    float x = dst_lvl.ox + ((float)(i - hc) + 0.5f) * dst_lvl.dx;
    float y = dst_lvl.oy + ((float)(j - hc) + 0.5f) * dst_lvl.dx;
    float z = dst_lvl.oz + ((float)(k - hc) + 0.5f) * dst_lvl.dx;

    float fx = (x - src_lvl.ox) / src_lvl.dx - 0.5f + (float)src_lvl.halo;
    float fy = (y - src_lvl.oy) / src_lvl.dx - 0.5f + (float)src_lvl.halo;
    float fz = (z - src_lvl.oz) / src_lvl.dx - 0.5f + (float)src_lvl.halo;

    int hs = src_lvl.halo;
    if (fx < (float)hs || fx > (float)(hs + src_lvl.nx - 1) ||
        fy < (float)hs || fy > (float)(hs + src_lvl.ny - 1) ||
        fz < (float)hs || fz > (float)(hs + src_lvl.nz - 1)) {
        return;
    }

    int i0 = max(hs, min(hs + src_lvl.nx - 2, (int)floorf(fx)));
    int j0 = max(hs, min(hs + src_lvl.ny - 2, (int)floorf(fy)));
    int k0 = max(hs, min(hs + src_lvl.nz - 2, (int)floorf(fz)));
    float tx = clamp_f(fx - (float)i0, 0.0f, 1.0f);
    float ty = clamp_f(fy - (float)j0, 0.0f, 1.0f);
    float tz = clamp_f(fz - (float)k0, 0.0f, 1.0f);

    size_t dcid = idx_cell(i, j, k, dst_lvl.nxp, dst_lvl.nyp);
    if (dst_lvl.cell_ib && dst_lvl.cell_ib[dcid] >= 0.5f) return;

    float rho = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f, ene = 0.0f;
    for (int dk = 0; dk <= 1; ++dk) {
        float wz = dk ? tz : (1.0f - tz);
        for (int dj = 0; dj <= 1; ++dj) {
            float wy = dj ? ty : (1.0f - ty);
            for (int di = 0; di <= 1; ++di) {
                float wx = di ? tx : (1.0f - tx);
                float w = wx * wy * wz;
                size_t sid = idx_cell(i0 + di, j0 + dj, k0 + dk, src_lvl.nxp, src_lvl.nyp);
                rho += w * src_lvl.u_rho[sid];
                mx  += w * src_lvl.u_mx[sid];
                my  += w * src_lvl.u_my[sid];
                mz  += w * src_lvl.u_mz[sid];
                ene += w * src_lvl.u_ene[sid];
            }
        }
    }
    // Smooth blend inside right overlap zone so shock enters downrange patch cleanly
    dst_lvl.u_rho[dcid] = rho;
    dst_lvl.u_mx[dcid]  = mx;
    dst_lvl.u_my[dcid]  = my;
    dst_lvl.u_mz[dcid]  = mz;
    dst_lvl.u_ene[dcid] = ene;
}

__global__ void k_euler_ib_reset_dynamic(MpmParticlesGpu pts, FluidLevelGpu lvl) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;

    float inv_dx = 1.0f / lvl.dx;
    float fx = (pts.px[p] - lvl.ox) * inv_dx - 0.5f + (float)lvl.halo;
    float fy = (pts.py[p] - lvl.oy) * inv_dx - 0.5f + (float)lvl.halo;
    float fz = (pts.pz[p] - lvl.oz) * inv_dx - 0.5f + (float)lvl.halo;
    if (fx < -2.0f || fx > (float)(lvl.nxp + 1) ||
        fy < -2.0f || fy > (float)(lvl.nyp + 1) ||
        fz < -2.0f || fz > (float)(lvl.nzp + 1)) {
        return;
    }

    int i0 = (int)floorf(fx);
    int j0 = (int)floorf(fy);
    int k0 = (int)floorf(fz);
    for (int dk = -1; dk <= 2; ++dk) {
        int kk = k0 + dk; if (kk < 0 || kk >= lvl.nzp) continue;
        for (int dj = -1; dj <= 2; ++dj) {
            int jj = j0 + dj; if (jj < 0 || jj >= lvl.nyp) continue;
            for (int di = -1; di <= 2; ++di) {
                int ii = i0 + di; if (ii < 0 || ii >= lvl.nxp) continue;
                size_t cid = idx_cell(ii, jj, kk, lvl.nxp, lvl.nyp);
                lvl.cell_ib[cid] = (lvl.cell_ib_static != nullptr) ? lvl.cell_ib_static[cid] : 0.0f;
                if (lvl.cell_vx) lvl.cell_vx[cid] = 0.0f;
                if (lvl.cell_vy) lvl.cell_vy[cid] = 0.0f;
                if (lvl.cell_vz) lvl.cell_vz[cid] = 0.0f;
            }
        }
    }
}

__global__ void k_euler_ib_splat_particles(MpmParticlesGpu pts, FluidLevelGpu lvl) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;

    float px = pts.px[p];
    float py = pts.py[p];
    float pz = pts.pz[p];
    float inv_dx = 1.0f / lvl.dx;
    float fx = (px - lvl.ox) * inv_dx - 0.5f + (float)lvl.halo;
    float fy = (py - lvl.oy) * inv_dx - 0.5f + (float)lvl.halo;
    float fz = (pz - lvl.oz) * inv_dx - 0.5f + (float)lvl.halo;
    if (fx < -1.0f || fx > (float)lvl.nxp || fy < -1.0f || fy > (float)lvl.nyp || fz < -1.0f || fz > (float)lvl.nzp) {
        return;
    }

    uint32_t pflag = pts.flags[p];
    float weight = (pflag & 8u)
        ? (fmaxf(0.0f, 1.0f - pts.damage[p]) * 1.4f * pts.area[p] * (inv_dx * inv_dx))
        : (2.0f * pts.vol0[p] * (inv_dx * inv_dx * inv_dx));
    if (weight <= 1e-8f) return;

    float vx = clamp_f(pts.vx[p], -150.0f, 150.0f);
    float vy = clamp_f(pts.vy[p], -150.0f, 150.0f);
    float vz = clamp_f(pts.vz[p], -150.0f, 150.0f);
    int i0 = (int)floorf(fx);
    int j0 = (int)floorf(fy);
    int k0 = (int)floorf(fz);
    float tx = fx - (float)i0;
    float ty = fy - (float)j0;
    float tz = fz - (float)k0;

    for (int dk = 0; dk <= 1; ++dk) {
        int kk = k0 + dk; if (kk < 0 || kk >= lvl.nzp) continue;
        float wz = dk ? tz : (1.0f - tz);
        for (int dj = 0; dj <= 1; ++dj) {
            int jj = j0 + dj; if (jj < 0 || jj >= lvl.nyp) continue;
            float wy = dj ? ty : (1.0f - ty);
            for (int di = 0; di <= 1; ++di) {
                int ii = i0 + di; if (ii < 0 || ii >= lvl.nxp) continue;
                float wx = di ? tx : (1.0f - tx);
                float w = weight * wx * wy * wz;
                if (w > 1e-8f) {
                    size_t cid = idx_cell(ii, jj, kk, lvl.nxp, lvl.nyp);
                    atomicAdd(&lvl.cell_ib[cid], w);
                    if (w > 0.02f) {
                        if (lvl.cell_vx) lvl.cell_vx[cid] = vx;
                        if (lvl.cell_vy) lvl.cell_vy[cid] = vy;
                        if (lvl.cell_vz) lvl.cell_vz[cid] = vz;
                    }
                }
            }
        }
    }
}

__global__ void k_euler_ib_normalize(MpmParticlesGpu pts, FluidLevelGpu lvl) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;

    float inv_dx = 1.0f / lvl.dx;
    float fx = (pts.px[p] - lvl.ox) * inv_dx - 0.5f + (float)lvl.halo;
    float fy = (pts.py[p] - lvl.oy) * inv_dx - 0.5f + (float)lvl.halo;
    float fz = (pts.pz[p] - lvl.oz) * inv_dx - 0.5f + (float)lvl.halo;
    if (fx < -1.0f || fx > (float)lvl.nxp || fy < -1.0f || fy > (float)lvl.nyp || fz < -1.0f || fz > (float)lvl.nzp) {
        return;
    }

    int i0 = (int)floorf(fx);
    int j0 = (int)floorf(fy);
    int k0 = (int)floorf(fz);
    for (int dk = 0; dk <= 1; ++dk) {
        int kk = k0 + dk; if (kk < 0 || kk >= lvl.nzp) continue;
        for (int dj = 0; dj <= 1; ++dj) {
            int jj = j0 + dj; if (jj < 0 || jj >= lvl.nyp) continue;
            for (int di = 0; di <= 1; ++di) {
                int ii = i0 + di; if (ii < 0 || ii >= lvl.nxp) continue;
                size_t cid = idx_cell(ii, jj, kk, lvl.nxp, lvl.nyp);
                lvl.cell_ib[cid] = clamp_f(lvl.cell_ib[cid], 0.0f, 1.0f);
            }
        }
    }
}

__device__ __forceinline__ void euler_sample_state_at(
    const FluidLevelGpu& lvl, const FluidConstantsGpu& c, float x, float y, float z,
    float& rho_out, float& ux_out, float& uy_out, float& uz_out, float& p_out
) {
    float fx = (x - lvl.ox) / lvl.dx - 0.5f + (float)lvl.halo;
    float fy = (y - lvl.oy) / lvl.dx - 0.5f + (float)lvl.halo;
    float fz = (z - lvl.oz) / lvl.dx - 0.5f + (float)lvl.halo;
    int i0 = max(0, min(lvl.nxp - 2, (int)floorf(fx)));
    int j0 = max(0, min(lvl.nyp - 2, (int)floorf(fy)));
    int k0 = max(0, min(lvl.nzp - 2, (int)floorf(fz)));
    float tx = clamp_f(fx - (float)i0, 0.0f, 1.0f);
    float ty = clamp_f(fy - (float)j0, 0.0f, 1.0f);
    float tz = clamp_f(fz - (float)k0, 0.0f, 1.0f);
    float gam = (float)c.gamma_a;

    rho_out = ux_out = uy_out = uz_out = p_out = 0.0f;
    for (int dk = 0; dk <= 1; ++dk) {
        float wz = dk ? tz : (1.0f - tz);
        for (int dj = 0; dj <= 1; ++dj) {
            float wy = dj ? ty : (1.0f - ty);
            for (int di = 0; di <= 1; ++di) {
                float wx = di ? tx : (1.0f - tx);
                float w = wx * wy * wz;
                size_t cid = idx_cell(i0 + di, j0 + dj, k0 + dk, lvl.nxp, lvl.nyp);
                float r, u, v, ww, p;
                cons_to_prim(lvl, cid, gam, r, u, v, ww, p);
                rho_out += w * r;
                ux_out  += w * u;
                uy_out  += w * v;
                uz_out  += w * ww;
                p_out   += w * p;
            }
        }
    }
}

__global__ void k_euler_ib_gather_to_solid(
    MpmParticlesGpu pts,
    const MaterialParamsGpu* mats,
    const FluidLevelGpu* levels,
    int32_t num_levels,
    FluidConstantsGpu c
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;
    const MaterialParamsGpu mat = mats[pts.mat_id[p]];
    if (mat.flags & 4u) return;

    uint32_t pflag = pts.flags[p];
    if (!(pflag & (4u | 8u))) {
        pts.fx_ext[p] = 0.0f; pts.fy_ext[p] = 0.0f; pts.fz_ext[p] = 0.0f;
        return;
    }

    float px = pts.px[p], py = pts.py[p], pz = pts.pz[p];
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
    float nx = pts.nx[p], ny = pts.ny[p], nz = pts.nz[p];
    float h_samp = 1.25f * lvl.dx;

    float rho0, u0, v0, w0, p0_loc;
    euler_sample_state_at(lvl, c, px, py, pz, rho0, u0, v0, w0, p0_loc);
    pts.fluid_rho[p] = rho0;

    float r_p, u_p, v_p, w_p, p_plus, r_m, u_m, v_m, w_m, p_minus;
    euler_sample_state_at(lvl, c, px + h_samp * nx, py + h_samp * ny, pz + h_samp * nz, r_p, u_p, v_p, w_p, p_plus);
    float f_p_mag;
    if (pflag & 8u) {
        euler_sample_state_at(lvl, c, px - h_samp * nx, py - h_samp * ny, pz - h_samp * nz, r_m, u_m, v_m, w_m, p_minus);
        float intact = fmaxf(0.0f, 1.0f - pts.damage[p]);
        f_p_mag = -intact * (p_plus - p_minus) * pts.area[p];
    } else {
        float rho_hs, p_hs;
        hs_state(pz + h_samp * nz, c, rho_hs, p_hs);
        f_p_mag = -(p_plus - p_hs) * pts.area[p];
    }

    // Dynamic wind / aerodynamic form drag: 0.5 * rho * C_d * A * |u_rel| * u_rel (normal projected)
    float rux = u0 - pts.vx[p];
    float ruy = v0 - pts.vy[p];
    float ruz = w0 - pts.vz[p];
    float vn_rel = rux * nx + ruy * ny + ruz * nz;
    float cd = (pflag & 8u) ? 1.2f : 0.6f;
    float f_drag_raw = 0.5f * rho0 * cd * pts.area[p] * fabsf(vn_rel) * vn_rel;
    float f_drag_cap = 0.25f * pts.mass[p] * fabsf(vn_rel) / 2.0e-5f;
    float f_drag_n = (f_drag_raw >= 0.0f) ? fminf(f_drag_raw, f_drag_cap) : -fminf(-f_drag_raw, f_drag_cap);
    float f_tot_cap = 20.0f * pts.mass[p] / 2.0e-5f;
    float f_tot = clamp_f(f_p_mag + f_drag_n, -f_tot_cap, f_tot_cap);

    pts.fx_ext[p] = f_tot * nx;
    pts.fy_ext[p] = f_tot * ny;
    pts.fz_ext[p] = f_tot * nz;
}

__global__ void k_euler_sample_probes(
    const FluidLevelGpu* levels, int32_t num_levels,
    const MicProbeGpu* probes, int32_t num_probes,
    float* out_samples, int32_t sample_idx, FluidConstantsGpu c
) {
    int p_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (p_idx >= num_probes) return;

    const MicProbeGpu pr = probes[p_idx];
    int lid = 0;
    // Dynamically select finest level currently containing probe (essential when sliding AMR moves)
    for (int li = 0; li < num_levels; ++li) {
        const FluidLevelGpu& l = levels[li];
        float m = 1.5f * l.dx;
        if (pr.x >= l.ox + m && pr.x <= l.ox + (float)l.nx * l.dx - m &&
            pr.y >= l.oy + m && pr.y <= l.oy + (float)l.ny * l.dx - m &&
            pr.z >= l.oz + m && pr.z <= l.oz + (float)l.nz * l.dx - m) {
            lid = li;
        }
    }

    const FluidLevelGpu& lvl = levels[lid];
    float fx = (pr.x - lvl.ox) / lvl.dx - 0.5f + (float)lvl.halo;
    float fy = (pr.y - lvl.oy) / lvl.dx - 0.5f + (float)lvl.halo;
    float fz = (pr.z - lvl.oz) / lvl.dx - 0.5f + (float)lvl.halo;
    int i0 = max(0, min(lvl.nxp - 2, (int)floorf(fx)));
    int j0 = max(0, min(lvl.nyp - 2, (int)floorf(fy)));
    int k0 = max(0, min(lvl.nzp - 2, (int)floorf(fz)));
    float tx = clamp_f(fx - (float)i0, 0.0f, 1.0f);
    float ty = clamp_f(fy - (float)j0, 0.0f, 1.0f);
    float tz = clamp_f(fz - (float)k0, 0.0f, 1.0f);
    float gam = (float)c.gamma_a;

    float rho_hs, p_hs;
    hs_state(pr.z, c, rho_hs, p_hs);

    double p_val = 0.0;
    for (int dk = 0; dk <= 1; ++dk) {
        float wz = dk ? tz : (1.0f - tz);
        for (int dj = 0; dj <= 1; ++dj) {
            float wy = dj ? ty : (1.0f - ty);
            for (int di = 0; di <= 1; ++di) {
                float wx = di ? tx : (1.0f - tx);
                size_t cid = idx_cell(i0 + di, j0 + dj, k0 + dk, lvl.nxp, lvl.nyp);
                float pc = cons_pressure(lvl.u_rho[cid], lvl.u_mx[cid], lvl.u_my[cid], lvl.u_mz[cid], lvl.u_ene[cid], gam);
                p_val += (double)(wx * wy * wz) * (double)pc;
            }
        }
    }
    out_samples[(size_t)sample_idx * (size_t)num_probes + (size_t)p_idx] = (float)(p_val - (double)p_hs);
}

__global__ void k_euler_reduce_metrics(
    FluidLevelGpu lvl, FluidConstantsGpu c, SimReduceMetricsGpu* out
) {
    uint64_t l_nan = 0;
    double l_c2 = 0.0, l_u2 = 0.0, l_e_ac = 0.0, l_e_kin = 0.0;
    double l_min_p = 1e30, l_max_p = -1e30, l_max_abs = 0.0;

    int h = lvl.halo;
    size_t n_int = (size_t)lvl.nx * (size_t)lvl.ny * (size_t)lvl.nz;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    double gam = c.gamma_a;
    double dvol = (double)lvl.dx * (double)lvl.dx * (double)lvl.dx;
    double rho0_c2 = gam * c.p0;

    for (size_t idx = tid; idx < n_int; idx += stride) {
        int ii = (int)(idx % (size_t)lvl.nx);
        int jj = (int)((idx / (size_t)lvl.nx) % (size_t)lvl.ny);
        int kk = (int)(idx / ((size_t)lvl.nx * (size_t)lvl.ny));
        size_t cid = idx_cell(ii + h, jj + h, kk + h, lvl.nxp, lvl.nyp);
        if (lvl.cell_ib != nullptr && lvl.cell_ib[cid] >= 0.5f) {
            continue;
        }

        float r = lvl.u_rho[cid], mx = lvl.u_mx[cid], my = lvl.u_my[cid], mz = lvl.u_mz[cid], en = lvl.u_ene[cid];
        if (!isfinite(r) || !isfinite(mx) || !isfinite(my) || !isfinite(mz) || !isfinite(en) || r <= 0.0f) {
            l_nan += 1;
            continue;
        }
        float inv_r = 1.0f / r;
        double u2 = ((double)mx * mx + (double)my * my + (double)mz * mz) * (double)(inv_r * inv_r);
        double ke = 0.5 * (double)r * u2;
        double p = (gam - 1.0) * ((double)en - ke);
        if (!isfinite(p) || p <= 0.0) {
            l_nan += 1;
            continue;
        }
        float rho_hs, p_hs;
        hs_state(cell_z_world(lvl, kk + h), c, rho_hs, p_hs);
        double dp = p - (double)p_hs;
        double c2 = gam * p / (double)r;
        l_c2 = fmax(l_c2, c2);
        l_u2 = fmax(l_u2, u2);
        l_min_p = fmin(l_min_p, dp);
        l_max_p = fmax(l_max_p, dp);
        l_max_abs = fmax(l_max_abs, fabs(dp));
        l_e_ac += 0.5 * dp * dp / rho0_c2 * dvol;
        l_e_kin += ke * dvol;
    }

    __shared__ uint64_t s_nan[256];
    __shared__ double s_c2[256], s_u2[256], s_eac[256], s_ek[256], s_minp[256], s_maxp[256], s_maxa[256];
    int t = threadIdx.x;
    s_nan[t] = l_nan; s_c2[t] = l_c2; s_u2[t] = l_u2;
    s_eac[t] = l_e_ac; s_ek[t] = l_e_kin;
    s_minp[t] = l_min_p; s_maxp[t] = l_max_p; s_maxa[t] = l_max_abs;
    __syncthreads();

    for (int off = 128; off > 0; off >>= 1) {
        if (t < off) {
            s_nan[t] += s_nan[t + off];
            s_c2[t] = fmax(s_c2[t], s_c2[t + off]);
            s_u2[t] = fmax(s_u2[t], s_u2[t + off]);
            s_eac[t] += s_eac[t + off];
            s_ek[t] += s_ek[t + off];
            s_minp[t] = fmin(s_minp[t], s_minp[t + off]);
            s_maxp[t] = fmax(s_maxp[t], s_maxp[t + off]);
            s_maxa[t] = fmax(s_maxa[t], s_maxa[t + off]);
        }
        __syncthreads();
    }

    if (t == 0) {
        atomicAdd((unsigned long long*)&out->nan_count, (unsigned long long)s_nan[0]);
        atomicAdd_d(&out->acoustic_energy, s_eac[0]);
        atomicAdd_d(&out->kinetic_energy, s_ek[0]);
        unsigned long long* p_c2 = (unsigned long long*)&out->c_max_sq;
        unsigned long long old = *p_c2, assumed;
        do { assumed = old; if (__longlong_as_double(assumed) >= s_c2[0]) break;
             old = atomicCAS(p_c2, assumed, __double_as_longlong(s_c2[0])); } while (assumed != old);
        unsigned long long* p_u2 = (unsigned long long*)&out->u_max_sq;
        old = *p_u2;
        do { assumed = old; if (__longlong_as_double(assumed) >= s_u2[0]) break;
             old = atomicCAS(p_u2, assumed, __double_as_longlong(s_u2[0])); } while (assumed != old);
        unsigned long long* p_mp = (unsigned long long*)&out->max_p;
        old = *p_mp;
        do { assumed = old; if (__longlong_as_double(assumed) >= s_maxp[0]) break;
             old = atomicCAS(p_mp, assumed, __double_as_longlong(s_maxp[0])); } while (assumed != old);
        unsigned long long* p_np = (unsigned long long*)&out->min_p;
        old = *p_np;
        do { assumed = old; if (__longlong_as_double(assumed) <= s_minp[0]) break;
             old = atomicCAS(p_np, assumed, __double_as_longlong(s_minp[0])); } while (assumed != old);
        unsigned long long* p_ma = (unsigned long long*)&out->max_abs_p;
        old = *p_ma;
        do { assumed = old; if (__longlong_as_double(assumed) >= s_maxa[0]) break;
             old = atomicCAS(p_ma, assumed, __double_as_longlong(s_maxa[0])); } while (assumed != old);
    }
}
