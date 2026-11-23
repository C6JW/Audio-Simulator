#include "common.cuh"

// 3x3 helpers (row-major 9 floats)
__device__ __forceinline__ void mat3_mul(const float A[9], const float B[9], float C[9]) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            C[3 * r + c] = A[3 * r + 0] * B[0 + c] + A[3 * r + 1] * B[3 + c] + A[3 * r + 2] * B[6 + c];
        }
    }
}

__device__ __forceinline__ void mat3_transpose(const float A[9], float AT[9]) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            AT[3 * r + c] = A[3 * c + r];
        }
    }
}

// Symmetric 3x3 Jacobi eigensolver: C = V * diag(evals) * V^T
__device__ __forceinline__ void jacobi_sym3(const float C_in[9], float evals[3], float V[9]) {
    float A[9];
    for (int i = 0; i < 9; ++i) A[i] = C_in[i];
    for (int i = 0; i < 9; ++i) V[i] = (i % 4 == 0) ? 1.0f : 0.0f;

    const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int sweep = 0; sweep < 4; ++sweep) {
        for (int idx = 0; idx < 3; ++idx) {
            int p = pairs[idx][0];
            int q = pairs[idx][1];
            float apq = A[3 * p + q];
            if (fabsf(apq) < 1e-15f) continue;
            float app = A[3 * p + p];
            float aqq = A[3 * q + q];
            float tau = (aqq - app) / (2.0f * apq);
            float abs_tau = fabsf(tau);
            float t;
            if (abs_tau > 1e8f) {
                t = 0.5f / tau;
            } else {
                float root = sqrtf(1.0f + tau * tau);
                t = (tau >= 0.0f) ? (1.0f / (tau + root)) : (-1.0f / (-tau + root));
            }
            float c = 1.0f / sqrtf(1.0f + t * t);
            float s = t * c;

            A[3 * p + p] = app - t * apq;
            A[3 * q + q] = aqq + t * apq;
            A[3 * p + q] = 0.0f;
            A[3 * q + p] = 0.0f;
            int r = 3 - p - q;
            float arp = A[3 * r + p];
            float arq = A[3 * r + q];
            A[3 * r + p] = A[3 * p + r] = c * arp - s * arq;
            A[3 * r + q] = A[3 * q + r] = s * arp + c * arq;

            for (int k = 0; k < 3; ++k) {
                float vkp = V[3 * k + p];
                float vkq = V[3 * k + q];
                V[3 * k + p] = c * vkp - s * vkq;
                V[3 * k + q] = s * vkp + c * vkq;
            }
        }
    }
    evals[0] = A[0];
    evals[1] = A[4];
    evals[2] = A[8];
}

// Compute orthotropic (or isotropic) stress in local material frame from precomputed FP32 stiffness
__device__ __forceinline__ void eval_orthotropic_stress(
    const MaterialParamsGpu& mat,
    const float eps[9],
    float sig[9]
) {
    float e0 = eps[0];
    float e1 = eps[4];
    float e2 = eps[8];

    sig[0] = mat.c11 * e0 + mat.c12 * e1 + mat.c13 * e2;
    sig[4] = mat.c12 * e0 + mat.c22 * e1 + mat.c23 * e2;
    sig[8] = mat.c13 * e0 + mat.c23 * e1 + mat.c33 * e2;

    sig[1] = sig[3] = 2.0f * mat.g12 * eps[1];
    sig[5] = sig[7] = 2.0f * mat.g23 * eps[5];
    sig[2] = sig[6] = 2.0f * mat.g13 * eps[2];
}

__device__ __forceinline__ void bspline_weights(float fx, int n_max, int base_idx[3], float w[3]) {
    int b = (int)floorf(fx - 0.5f);
    float d = fx - (float)b;
    float z0 = 1.5f - d;
    float z1 = d - 1.0f;
    float z2 = d - 0.5f;
    w[0] = 0.5f * z0 * z0;
    w[1] = 0.75f - z1 * z1;
    w[2] = 0.5f * z2 * z2;
    for (int a = 0; a < 3; ++a) {
        int idx = b + a;
        base_idx[a] = max(0, min(n_max - 1, idx));
    }
}

__global__ void k_mpm_clear_grid(MpmGridGpu grid) {
    size_t n_nodes = (size_t)grid.nx * (size_t)grid.ny * (size_t)grid.nz;
    size_t total = 2 * n_nodes;
    size_t gid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (gid >= total) return;

    grid.mass[gid] = 0.0f;
    grid.vx[gid] = 0.0f;
    grid.vy[gid] = 0.0f;
    grid.vz[gid] = 0.0f;
    grid.bc_w[gid] = 0.0f;
    grid.bc_vx[gid] = 0.0f;
    grid.bc_vy[gid] = 0.0f;
    grid.bc_vz[gid] = 0.0f;
}

__global__ void k_mpm_p2g(
    MpmParticlesGpu pts,
    MpmGridGpu grid,
    const MaterialParamsGpu* mats,
    float dt_s,
    const float* extra_fx,
    const float* extra_fy,
    const float* extra_fz
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;

    const MaterialParamsGpu mat = mats[pts.mat_id[p]];
    if (mat.flags & 5u) return; // skip is_rigid_static (4) and is_shell (1)

    int n = pts.count;
    float px = pts.px[p];
    float py = pts.py[p];
    float pz = pts.pz[p];
    float vx = pts.vx[p];
    float vy = pts.vy[p];
    float vz = pts.vz[p];
    float mp = pts.mass[p];
    float v0 = pts.vol0[p];

    float C[9], H[9], F[9], Q0[9];
    for (int m = 0; m < 9; ++m) {
        C[m] = pts.c_mat[(size_t)m * (size_t)n + (size_t)p];
        H[m] = pts.f_mat[(size_t)m * (size_t)n + (size_t)p];
        F[m] = H[m] + ((m % 4 == 0) ? 1.0f : 0.0f);
        Q0[m] = pts.q0_mat[(size_t)m * (size_t)n + (size_t)p];
    }

    float tau[9] = {0.0f};
    if (!(mat.flags & 1u)) {
        // Bulk 3D continuum constitutive law from H = F - I (zero 1.0f cancellation)
        float FT[9], HT[9], HTH[9], EG[9], R[9], Sinv[9], V[9];
        float s_eig[3], ek_eig[3], s_inv[3];
        mat3_transpose(F, FT);
        mat3_transpose(H, HT);
        mat3_mul(HT, H, HTH);
        for (int i = 0; i < 9; ++i) {
            EG[i] = 0.5f * (H[i] + HT[i] + HTH[i]);
        }

        float mu_eig[3];
        jacobi_sym3(EG, mu_eig, V);
        for (int k = 0; k < 3; ++k) {
            float raw_lam = 1.0f + 2.0f * mu_eig[k];
            float sk = sqrtf(fmaxf(1e-4f, raw_lam));
            s_eig[k] = sk;
            ek_eig[k] = (raw_lam >= 1e-4f) ? (2.0f * mu_eig[k] / (sk + 1.0f)) : (sk - 1.0f);
            s_inv[k] = 1.0f / sk;
        }
        float eps_body[9];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                float ev = 0.0f, siv = 0.0f;
                for (int k = 0; k < 3; ++k) {
                    ev += V[3 * r + k] * ek_eig[k] * V[3 * c + k];
                    siv += V[3 * r + k] * s_inv[k] * V[3 * c + k];
                }
                eps_body[3 * r + c] = ev;
                Sinv[3 * r + c] = siv;
            }
        }
        mat3_mul(F, Sinv, R);

        // Rotate into material frame: eps_mat = Q0^T * eps_body * Q0
        float Q0T[9], tmp[9], eps_mat[9];
        mat3_transpose(Q0, Q0T);
        mat3_mul(Q0T, eps_body, tmp);
        mat3_mul(tmp, Q0, eps_mat);

        // Zener SLS internal viscoelastic branch
        float decay = (mat.sls_tau > 1e-9f) ? expf(-dt_s / mat.sls_tau) : 0.0f;
        const int sym_idx[6] = {0, 4, 8, 1, 5, 2};
        float eps_eff[9];
        for (int i = 0; i < 9; ++i) eps_eff[i] = eps_mat[i];
        if (mat.sls_beta > 0.0f) {
            float q_new[6];
            for (int s = 0; s < 6; ++s) {
                float ev = eps_mat[sym_idx[s]];
                float q_old = pts.q_visc[(size_t)s * (size_t)n + (size_t)p];
                float qn = decay * q_old + (1.0f - decay) * ev;
                pts.q_visc[(size_t)s * (size_t)n + (size_t)p] = qn;
                q_new[s] = ev + mat.sls_beta * (ev - qn);
            }
            eps_eff[0] = q_new[0];
            eps_eff[4] = q_new[1];
            eps_eff[8] = q_new[2];
            eps_eff[1] = eps_eff[3] = q_new[3];
            eps_eff[5] = eps_eff[7] = q_new[4];
            eps_eff[2] = eps_eff[6] = q_new[5];
        }

        float sig_mat[9];
        eval_orthotropic_stress(mat, eps_eff, sig_mat);

        // Rotate back to body frame: sig_body = Q0 * sig_mat * Q0^T
        float sig_body[9];
        mat3_mul(Q0, sig_mat, tmp);
        mat3_mul(tmp, Q0T, sig_body);

        // Phase-field tensile split & history update if fracture enabled (g_c > 0)
        if (mat.g_c > 0.0f) {
            float e_avg = (mat.e1 + mat.e2 + mat.e3) / 3.0f;
            float nu_avg = clamp_f(mat.nu12, 0.0f, 0.45f);
            float mu_eff = e_avg / (2.0f * (1.0f + nu_avg));
            float lam_eff = e_avg * nu_avg / ((1.0f + nu_avg) * (1.0f - 2.0f * nu_avg));
            float e_p0 = ek_eig[0];
            float e_p1 = ek_eig[1];
            float e_p2 = ek_eig[2];
            float tr_p = fmaxf(0.0f, e_p0 + e_p1 + e_p2);
            float sum_pos2 = fmaxf(0.0f, e_p0) * fmaxf(0.0f, e_p0)
                           + fmaxf(0.0f, e_p1) * fmaxf(0.0f, e_p1)
                           + fmaxf(0.0f, e_p2) * fmaxf(0.0f, e_p2);
            float psi_plus = 0.5f * lam_eff * tr_p * tr_p + mu_eff * sum_pos2;
            float gc_s = fmaxf(0.2f, pts.gc_scale[p]);
            pts.hist_h[p] = fmaxf(pts.hist_h[p], psi_plus / gc_s);

            float d = pts.damage[p];
            if (d > 1e-4f) {
                float g_d = (1.0f - d) * (1.0f - d) + 1e-5f;
                // Spectral tensile degradation in principal basis V
                float VT[9], sig_p[9];
                mat3_transpose(V, VT);
                mat3_mul(VT, sig_body, tmp);
                mat3_mul(tmp, V, sig_p);
                for (int r = 0; r < 3; ++r) {
                    if (sig_p[3 * r + r] > 0.0f) sig_p[3 * r + r] *= g_d;
                }
                sig_p[1] = sig_p[3] *= g_d;
                sig_p[5] = sig_p[7] *= g_d;
                sig_p[2] = sig_p[6] *= g_d;
                mat3_mul(V, sig_p, tmp);
                mat3_mul(tmp, VT, sig_body);
            }
        }

        // P = R * sig_body, tau = P * F^T = R * sig_body * S * R^T
        float P[9];
        mat3_mul(R, sig_body, P);
        mat3_mul(P, FT, tau);

        // Add Kelvin-Voigt viscous stress: eta_visc * (C + C^T)
        if (mat.eta_visc > 0.0f) {
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    tau[3 * r + c] += mat.eta_visc * (C[3 * r + c] + C[3 * c + r]);
                }
            }
        }
    }

    float f_ext_x = pts.fx_ext[p] + (extra_fx ? extra_fx[p] : 0.0f);
    float f_ext_y = pts.fy_ext[p] + (extra_fy ? extra_fy[p] : 0.0f);
    float f_ext_z = pts.fz_ext[p] + (extra_fz ? extra_fz[p] : 0.0f);

    float fx_p = (px - grid.ox) / grid.dx;
    float fy_p = (py - grid.oy) / grid.dx;
    float fz_p = (pz - grid.oz) / grid.dx;

    int ix[3], iy[3], iz[3];
    float wx[3], wy[3], wz[3];
    bspline_weights(fx_p, grid.nx, ix, wx);
    bspline_weights(fy_p, grid.ny, iy, wy);
    bspline_weights(fz_p, grid.nz, iz, wz);

    size_t n_nodes = (size_t)grid.nx * (size_t)grid.ny * (size_t)grid.nz;
    int field = pts.body_id[p] & 1;
    size_t field_off = (size_t)field * n_nodes;

    float stress_scale = -4.0f * v0 / (grid.dx * grid.dx);
    uint32_t pflag = pts.flags[p];
    bool is_dirichlet = (pflag & 3u) != 0;
    float bv_x = (pflag & 1u) ? 0.0f : pts.v_presc_x[p];
    float bv_y = (pflag & 1u) ? 0.0f : pts.v_presc_y[p];
    float bv_z = (pflag & 1u) ? 0.0f : pts.v_presc_z[p];

    for (int kc = 0; kc < 3; ++kc) {
        int gk = iz[kc];
        float nk_z = grid.oz + (float)gk * grid.dx;
        float dz = nk_z - pz;
        for (int jc = 0; jc < 3; ++jc) {
            int gj = iy[jc];
            float nj_y = grid.oy + (float)gj * grid.dx;
            float dy = nj_y - py;
            for (int ic = 0; ic < 3; ++ic) {
                int gi = ix[ic];
                float ni_x = grid.ox + (float)gi * grid.dx;
                float dx = ni_x - px;

                float w = wx[ic] * wy[jc] * wz[kc];
                size_t nid = field_off + (size_t)gi + (size_t)grid.nx * ((size_t)gj + (size_t)grid.ny * (size_t)gk);

                float cx_r = C[0] * dx + C[1] * dy + C[2] * dz;
                float cy_r = C[3] * dx + C[4] * dy + C[5] * dz;
                float cz_r = C[6] * dx + C[7] * dy + C[8] * dz;

                float tx_r = tau[0] * dx + tau[1] * dy + tau[2] * dz;
                float ty_r = tau[3] * dx + tau[4] * dy + tau[5] * dz;
                float tz_r = tau[6] * dx + tau[7] * dy + tau[8] * dz;

                float mvx = mp * (vx + cx_r) + dt_s * (stress_scale * tx_r + f_ext_x);
                float mvy = mp * (vy + cy_r) + dt_s * (stress_scale * ty_r + f_ext_y);
                float mvz = mp * (vz + cz_r) + dt_s * (stress_scale * tz_r + f_ext_z);

                atomicAdd(&grid.mass[nid], w * mp);
                atomicAdd(&grid.vx[nid], w * mvx);
                atomicAdd(&grid.vy[nid], w * mvy);
                atomicAdd(&grid.vz[nid], w * mvz);

                if (is_dirichlet) {
                    atomicAdd(&grid.bc_w[nid], w);
                    if (pflag & 2u) {
                        atomicAdd(&grid.bc_vx[nid], w * bv_x);
                        atomicAdd(&grid.bc_vy[nid], w * bv_y);
                        atomicAdd(&grid.bc_vz[nid], w * bv_z);
                    }
                }
            }
        }
    }
}

__global__ void k_mpm_grid_pre_contact(
    MpmGridGpu grid,
    float gx, float gy, float gz,
    float dt_s
) {
    size_t n_nodes = (size_t)grid.nx * (size_t)grid.ny * (size_t)grid.nz;
    size_t gid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (gid >= 2 * n_nodes) return;

    float m = grid.mass[gid];
    if (m <= 1e-18f) return;

    float inv_m = 1.0f / m;
    float vx = grid.vx[gid] * inv_m + dt_s * gx;
    float vy = grid.vy[gid] * inv_m + dt_s * gy;
    float vz = grid.vz[gid] * inv_m + dt_s * gz;

    float bw = grid.bc_w[gid];
    if (bw > 0.05f) {
        float inv_b = 1.0f / bw;
        vx = grid.bc_vx[gid] * inv_b;
        vy = grid.bc_vy[gid] * inv_b;
        vz = grid.bc_vz[gid] * inv_b;
    } else if (grid.has_floor) {
        size_t node = gid % n_nodes;
        int k = (int)(node / ((size_t)grid.nx * (size_t)grid.ny));
        float z = grid.oz + (float)k * grid.dx;
        if (z <= grid.floor_z && vz < 0.0f) {
            float vn = -vz;
            vz = 0.0f;
            float vt = sqrtf(vx * vx + vy * vy);
            if (vt > 1e-12f && grid.floor_mu > 0.0f) {
                float drop = fminf(vt, grid.floor_mu * vn);
                float scale = (vt - drop) / vt;
                vx *= scale;
                vy *= scale;
            }
        }
    }

    grid.vx[gid] = vx;
    grid.vy[gid] = vy;
    grid.vz[gid] = vz;
}

__global__ void k_mpm_grid_contact(MpmGridGpu grid, float friction_mu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= grid.nx || j >= grid.ny || k >= grid.nz) return;

    size_t n_nodes = (size_t)grid.nx * (size_t)grid.ny * (size_t)grid.nz;
    size_t n0 = (size_t)i + (size_t)grid.nx * ((size_t)j + (size_t)grid.ny * (size_t)k);
    size_t n1 = n_nodes + n0;

    float m0 = grid.mass[n0];
    float m1 = grid.mass[n1];
    if (m0 <= 1e-18f || m1 <= 1e-18f) return;
    if (grid.bc_w[n0] > 0.05f && grid.bc_w[n1] > 0.05f) return;

    int im = max(0, i - 1), ip = min(grid.nx - 1, i + 1);
    int jm = max(0, j - 1), jp = min(grid.ny - 1, j + 1);
    int km = max(0, k - 1), kp = min(grid.nz - 1, k + 1);

    auto m_at = [&](size_t off, int ii, int jj, int kk) -> float {
        return grid.mass[off + (size_t)ii + (size_t)grid.nx * ((size_t)jj + (size_t)grid.ny * (size_t)kk)];
    };

    float gx0 = m_at(0, ip, j, k) - m_at(0, im, j, k);
    float gy0 = m_at(0, i, jp, k) - m_at(0, i, jm, k);
    float gz0 = m_at(0, i, j, kp) - m_at(0, i, j, km);

    float gx1 = m_at(n_nodes, ip, j, k) - m_at(n_nodes, im, j, k);
    float gy1 = m_at(n_nodes, i, jp, k) - m_at(n_nodes, i, jm, k);
    float gz1 = m_at(n_nodes, i, j, kp) - m_at(n_nodes, i, j, km);

    float nx = gx0 - gx1;
    float ny = gy0 - gy1;
    float nz = gz0 - gz1;
    float nlen = sqrtf(nx * nx + ny * ny + nz * nz);
    if (nlen < 1e-18f) return;
    float inv_n = 1.0f / nlen;
    nx *= inv_n; ny *= inv_n; nz *= inv_n;

    float vx0 = grid.vx[n0], vy0 = grid.vy[n0], vz0 = grid.vz[n0];
    float vx1 = grid.vx[n1], vy1 = grid.vy[n1], vz1 = grid.vz[n1];

    float rvx = vx1 - vx0;
    float rvy = vy1 - vy0;
    float rvz = vz1 - vz0;
    float vn = rvx * nx + rvy * ny + rvz * nz;
    if (vn <= 0.0f) return; // Separating: allow clean rebound

    float m_eff = (m0 * m1) / (m0 + m1);
    float jn = m_eff * vn;
    float jx = jn * nx;
    float jy = jn * ny;
    float jz = jn * nz;

    float tvx = rvx - vn * nx;
    float tvy = rvy - vn * ny;
    float tvz = rvz - vn * nz;
    float vt = sqrtf(tvx * tvx + tvy * tvy + tvz * tvz);
    if (vt > 1e-12f && friction_mu > 0.0f) {
        float jt = fminf(m_eff * vt, friction_mu * jn);
        float s = jt / vt;
        jx += s * tvx;
        jy += s * tvy;
        jz += s * tvz;
    }

    if (grid.bc_w[n0] <= 0.05f) {
        grid.vx[n0] = vx0 + jx / m0;
        grid.vy[n0] = vy0 + jy / m0;
        grid.vz[n0] = vz0 + jz / m0;
    }
    if (grid.bc_w[n1] <= 0.05f) {
        grid.vx[n1] = vx1 - jx / m1;
        grid.vy[n1] = vy1 - jy / m1;
        grid.vz[n1] = vz1 - jz / m1;
    }
}

__global__ void k_mpm_g2p(
    MpmParticlesGpu pts,
    MpmGridGpu grid,
    const MaterialParamsGpu* mats,
    float dt_s
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;

    const MaterialParamsGpu mat = mats[pts.mat_id[p]];
    if (mat.flags & 5u) return; // skip is_rigid_static (4) and is_shell (1)

    int n = pts.count;
    float px = pts.px[p];
    float py = pts.py[p];
    float pz = pts.pz[p];

    float fx_p = (px - grid.ox) / grid.dx;
    float fy_p = (py - grid.oy) / grid.dx;
    float fz_p = (pz - grid.oz) / grid.dx;

    int ix[3], iy[3], iz[3];
    float wx[3], wy[3], wz[3];
    bspline_weights(fx_p, grid.nx, ix, wx);
    bspline_weights(fy_p, grid.ny, iy, wy);
    bspline_weights(fz_p, grid.nz, iz, wz);

    size_t n_nodes = (size_t)grid.nx * (size_t)grid.ny * (size_t)grid.nz;
    int field = pts.body_id[p] & 1;
    size_t field_off = (size_t)field * n_nodes;

    float v_new[3] = {0.0f, 0.0f, 0.0f};
    float C_new[9] = {0.0f};
    float c_scale = 4.0f / (grid.dx * grid.dx);

    for (int kc = 0; kc < 3; ++kc) {
        int gk = iz[kc];
        float dz = (grid.oz + (float)gk * grid.dx) - pz;
        for (int jc = 0; jc < 3; ++jc) {
            int gj = iy[jc];
            float dy = (grid.oy + (float)gj * grid.dx) - py;
            for (int ic = 0; ic < 3; ++ic) {
                int gi = ix[ic];
                float dx = (grid.ox + (float)gi * grid.dx) - px;
                float w = wx[ic] * wy[jc] * wz[kc];
                size_t nid = field_off + (size_t)gi + (size_t)grid.nx * ((size_t)gj + (size_t)grid.ny * (size_t)gk);

                float vx = grid.vx[nid];
                float vy = grid.vy[nid];
                float vz = grid.vz[nid];

                v_new[0] += w * vx;
                v_new[1] += w * vy;
                v_new[2] += w * vz;

                C_new[0] += w * c_scale * vx * dx;
                C_new[1] += w * c_scale * vx * dy;
                C_new[2] += w * c_scale * vx * dz;
                C_new[3] += w * c_scale * vy * dx;
                C_new[4] += w * c_scale * vy * dy;
                C_new[5] += w * c_scale * vy * dz;
                C_new[6] += w * c_scale * vz * dx;
                C_new[7] += w * c_scale * vz * dy;
                C_new[8] += w * c_scale * vz * dz;
            }
        }
    }

    uint32_t pflag = pts.flags[p];
    if (pflag & 1u) {
        // Clamped position: keep velocity zero, update F from neighbor C_new
        pts.vx[p] = 0.0f;
        pts.vy[p] = 0.0f;
        pts.vz[p] = 0.0f;
        for (int m = 0; m < 9; ++m) pts.c_mat[(size_t)m * (size_t)n + (size_t)p] = 0.0f;
    } else if (pflag & 2u) {
        float vx = pts.v_presc_x[p];
        float vy = pts.v_presc_y[p];
        float vz = pts.v_presc_z[p];
        pts.vx[p] = vx;
        pts.vy[p] = vy;
        pts.vz[p] = vz;
        pts.px[p] = px + dt_s * vx;
        pts.py[p] = py + dt_s * vy;
        pts.pz[p] = pz + dt_s * vz;
        for (int m = 0; m < 9; ++m) pts.c_mat[(size_t)m * (size_t)n + (size_t)p] = C_new[m];
    } else {
        pts.vx[p] = v_new[0];
        pts.vy[p] = v_new[1];
        pts.vz[p] = v_new[2];
        float x_next = px + dt_s * v_new[0];
        float y_next = py + dt_s * v_new[1];
        float z_next = pz + dt_s * v_new[2];
        if (grid.has_floor && z_next < grid.floor_z) {
            z_next = grid.floor_z;
            if (pts.vz[p] < 0.0f) pts.vz[p] = 0.0f;
        }
        pts.px[p] = x_next;
        pts.py[p] = y_next;
        pts.pz[p] = z_next;
        for (int m = 0; m < 9; ++m) pts.c_mat[(size_t)m * (size_t)n + (size_t)p] = C_new[m];
    }

    if (!(mat.flags & 1u) && !(pflag & 1u)) {
        // Update H_new = H_old + dt * (C_new + C_new * H_old) where F = I + H
        float H_old[9], CH[9], H_new[9], F_new[9];
        for (int m = 0; m < 9; ++m) H_old[m] = pts.f_mat[(size_t)m * (size_t)n + (size_t)p];
        mat3_mul(C_new, H_old, CH);
        for (int m = 0; m < 9; ++m) {
            H_new[m] = H_old[m] + dt_s * (C_new[m] + CH[m]);
            pts.f_mat[(size_t)m * (size_t)n + (size_t)p] = H_new[m];
            F_new[m] = H_new[m] + ((m % 4 == 0) ? 1.0f : 0.0f);
        }

        // Update deformed surface normal via Nanson cofactor cof(F_new) * n0
        if (pflag & 4u) {
            float c0x = F_new[4] * F_new[8] - F_new[7] * F_new[5];
            float c0y = F_new[7] * F_new[2] - F_new[1] * F_new[8];
            float c0z = F_new[1] * F_new[5] - F_new[4] * F_new[2];
            float c1x = F_new[6] * F_new[5] - F_new[3] * F_new[8];
            float c1y = F_new[0] * F_new[8] - F_new[6] * F_new[2];
            float c1z = F_new[3] * F_new[2] - F_new[0] * F_new[5];
            float c2x = F_new[3] * F_new[7] - F_new[6] * F_new[4];
            float c2y = F_new[6] * F_new[1] - F_new[0] * F_new[7];
            float c2z = F_new[0] * F_new[4] - F_new[3] * F_new[1];

            float n0x = pts.nx0[p], n0y = pts.ny0[p], n0z = pts.nz0[p];
            float nx = c0x * n0x + c1x * n0y + c2x * n0z;
            float ny = c0y * n0x + c1y * n0y + c2y * n0z;
            float nz = c0z * n0x + c1z * n0y + c2z * n0z;
            float len = sqrtf(nx * nx + ny * ny + nz * nz);
            if (len > 1e-12f) {
                pts.nx[p] = nx / len;
                pts.ny[p] = ny / len;
                pts.nz[p] = nz / len;
            }
        }
    }
}

__global__ void k_mpm_reset_metrics(MpmMetricsGpu* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out->kinetic_energy = 0.0;
        out->elastic_energy = 0.0;
        out->damage_sum = 0.0;
        out->nan_count = 0;
    }
}

__global__ void k_mpm_reduce_energy(
    MpmParticlesGpu pts,
    const MaterialParamsGpu* mats,
    MpmMetricsGpu* out
) {
    __shared__ double s_ek[256];
    __shared__ double s_ee[256];
    __shared__ double s_fr[256];
    __shared__ unsigned long long s_nan[256];

    int tid = threadIdx.x;
    int n = pts.count;
    double l_ek = 0.0, l_ee = 0.0, l_fr = 0.0;
    unsigned long long l_nan = 0;

    for (int p = blockIdx.x * blockDim.x + tid; p < n; p += gridDim.x * blockDim.x) {
        float px = pts.px[p], py = pts.py[p], pz = pts.pz[p];
        float vx = pts.vx[p], vy = pts.vy[p], vz = pts.vz[p];
        if (!isfinite(px) || !isfinite(py) || !isfinite(pz) ||
            !isfinite(vx) || !isfinite(vy) || !isfinite(vz)) {
            ++l_nan;
            continue;
        }
        double mp = (double)pts.mass[p];
        l_ek += 0.5 * mp * ((double)vx * vx + (double)vy * vy + (double)vz * vz);
        l_fr += (double)pts.damage[p];

        const MaterialParamsGpu mat = mats[pts.mat_id[p]];
        if (!(mat.flags & 5u)) {
            float H[9], HT[9], HTH[9], EG[9], mu_eig[3], V[9];
            for (int m = 0; m < 9; ++m) H[m] = pts.f_mat[(size_t)m * (size_t)n + (size_t)p];
            mat3_transpose(H, HT);
            mat3_mul(HT, H, HTH);
            for (int i = 0; i < 9; ++i) EG[i] = 0.5f * (H[i] + HT[i] + HTH[i]);
            jacobi_sym3(EG, mu_eig, V);
            double e2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                double raw_lam = 1.0 + 2.0 * (double)mu_eig[k];
                double sk = sqrt(fmax(1e-4, raw_lam));
                double ek = (raw_lam >= 1e-4) ? (2.0 * (double)mu_eig[k] / (sk + 1.0)) : (sk - 1.0);
                e2 += ek * ek;
            }
            double e_avg = ((double)mat.e1 + (double)mat.e2 + (double)mat.e3) / 3.0;
            l_ee += 0.5 * e_avg * e2 * (double)pts.vol0[p];
        }
    }

    s_ek[tid] = l_ek;
    s_ee[tid] = l_ee;
    s_fr[tid] = l_fr;
    s_nan[tid] = l_nan;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_ek[tid] += s_ek[tid + s];
            s_ee[tid] += s_ee[tid + s];
            s_fr[tid] += s_fr[tid + s];
            s_nan[tid] += s_nan[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd_d(&out->kinetic_energy, s_ek[0]);
        atomicAdd_d(&out->elastic_energy, s_ee[0]);
        atomicAdd_d(&out->damage_sum, s_fr[0]);
        atomicAdd((unsigned long long*)&out->nan_count, s_nan[0]);
    }
}
