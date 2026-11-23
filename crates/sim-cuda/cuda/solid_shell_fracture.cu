#include "common.cuh"
#include <cooperative_groups.h>

// Discrete Kirchhoff-Love thin-shell mechanics, grid phase-field fracture (AT2),
// and spatial-hash wet capillary bridge + Stefan squeeze-film lubrication contact.

__device__ __forceinline__ float dot3(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ __forceinline__ void cross3(const float* a, const float* b, float* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ __forceinline__ uint32_t hash_cell(int ix, int iy, int iz, int32_t hash_size) {
    uint32_t h = ((uint32_t)ix * 73856093u) ^ ((uint32_t)iy * 19349663u) ^ ((uint32_t)iz * 83492791u);
    return h & (uint32_t)(hash_size - 1);
}

__device__ __forceinline__ void shell_bspline_weights(float fx, int n_grid, int* idx, float* w) {
    if (!isfinite(fx)) {
        idx[0] = idx[1] = idx[2] = 0;
        w[0] = w[1] = w[2] = 0.0f;
        return;
    }
    float fxc = clamp_f(fx, -1.0f, (float)n_grid + 1.0f);
    int base = (int)floorf(fxc - 0.5f);
    float d = fxc - (float)base;
    idx[0] = min(max(base - 1, 0), n_grid - 1);
    idx[1] = min(max(base, 0), n_grid - 1);
    idx[2] = min(max(base + 1, 0), n_grid - 1);

    float d0 = d;
    float d1 = d - 1.0f;
    float d2 = 2.0f - d;
    w[0] = 0.5f * (1.5f - d0) * (1.5f - d0);
    w[1] = 0.75f - d1 * d1;
    w[2] = 0.5f * (1.5f - d2) * (1.5f - d2);
}

__global__ void k_shell_reset_forces_and_normals(ShellMeshGpu shell, MpmParticlesGpu pts) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;

    shell.p_fx[p] = 0.0f;
    shell.p_fy[p] = 0.0f;
    shell.p_fz[p] = 0.0f;

    if (pts.flags[p] & 8u) {
        pts.nx[p] = 0.0f;
        pts.ny[p] = 0.0f;
        pts.nz[p] = 0.0f;
    }
}

__device__ __forceinline__ void d_shell_triangle_membrane_one(
    int t,
    ShellMeshGpu& shell,
    MpmParticlesGpu& pts,
    const MaterialParamsGpu* mats,
    float dt_s
) {
    int v0 = shell.tri_v0[t];
    int v1 = shell.tri_v1[t];
    int v2 = shell.tri_v2[t];

    const MaterialParamsGpu mat = mats[shell.tri_mat_id[t]];
    if (mat.flags & 4u) return;

    float d00 = shell.tri_dm_inv[4 * t + 0];
    float d01 = shell.tri_dm_inv[4 * t + 1];
    float d10 = shell.tri_dm_inv[4 * t + 2];
    float d11 = shell.tri_dm_inv[4 * t + 3];
    float vol = shell.tri_area0[t] * shell.tri_h[t];

    float e1[3] = {pts.px[v1] - pts.px[v0], pts.py[v1] - pts.py[v0], pts.pz[v1] - pts.pz[v0]};
    float e2[3] = {pts.px[v2] - pts.px[v0], pts.py[v2] - pts.py[v0], pts.pz[v2] - pts.pz[v0]};

    float fn[3];
    cross3(e1, e2, fn);
    atomicAdd(&pts.nx[v0], fn[0]); atomicAdd(&pts.ny[v0], fn[1]); atomicAdd(&pts.nz[v0], fn[2]);
    atomicAdd(&pts.nx[v1], fn[0]); atomicAdd(&pts.ny[v1], fn[1]); atomicAdd(&pts.nz[v1], fn[2]);
    atomicAdd(&pts.nx[v2], fn[0]); atomicAdd(&pts.ny[v2], fn[1]); atomicAdd(&pts.nz[v2], fn[2]);

    float d0 = pts.damage[v0], d1 = pts.damage[v1], d2 = pts.damage[v2];
    float d_tri = 0.5f * ((d0 + d1 + d2) / 3.0f) + 0.5f * fmaxf(d0, fmaxf(d1, d2));
    if (d_tri > 0.92f) return;

    float f1[3], f2[3];
    for (int k = 0; k < 3; ++k) {
        f1[k] = d00 * e1[k] + d10 * e2[k];
        f2[k] = d01 * e1[k] + d11 * e2[k];
    }

    float E11 = 0.5f * (dot3(f1, f1) - 1.0f);
    float E22 = 0.5f * (dot3(f2, f2) - 1.0f);
    float E12 = 0.5f * dot3(f1, f2);

    float dv1[3] = {pts.vx[v1] - pts.vx[v0], pts.vy[v1] - pts.vy[v0], pts.vz[v1] - pts.vz[v0]};
    float dv2[3] = {pts.vx[v2] - pts.vx[v0], pts.vy[v2] - pts.vy[v0], pts.vz[v2] - pts.vz[v0]};
    float fd1[3], fd2[3];
    for (int k = 0; k < 3; ++k) {
        fd1[k] = d00 * dv1[k] + d10 * dv2[k];
        fd2[k] = d01 * dv1[k] + d11 * dv2[k];
    }
    float Ed11 = dot3(f1, fd1);
    float Ed22 = dot3(f2, fd2);
    float Ed12 = 0.5f * (dot3(f1, fd2) + dot3(f2, fd1));

    float w_avg = (pts.wetness[v0] + pts.wetness[v1] + pts.wetness[v2]) / 3.0f;
    float hygro_stiff = (mat.hygro_softening > 0.0f) ? fmaxf(0.15f, 1.0f - mat.hygro_softening * w_avg) : 1.0f;
    float hygro_damp  = (mat.hygro_softening > 0.0f) ? (1.0f + 2.5f * w_avg) : 1.0f;

    float nu12 = clamp_f(mat.nu12, 0.0f, 0.45f);
    float nu21 = nu12 * (mat.e2 / fmaxf(1.0f, mat.e1));
    float denom = fmaxf(1e-4f, 1.0f - nu12 * nu21);
    float Q11 = (mat.e1 * hygro_stiff) / denom;
    float Q22 = (mat.e2 * hygro_stiff) / denom;
    float Q12 = nu12 * Q22;
    float Q66 = mat.g12 * hygro_stiff;

    float S11 = Q11 * E11 + Q12 * E22;
    float S22 = Q12 * E11 + Q22 * E22;
    float S12 = 2.0f * Q66 * E12;

    if (mat.sigma_y > 0.0f) {
        float vm = sqrtf(fmaxf(0.0f, S11 * S11 - S11 * S22 + S22 * S22 + 3.0f * S12 * S12));
        if (vm > mat.sigma_y) {
            float relax = 1.0f - expf(-dt_s / fmaxf(1e-6f, mat.tau_vp));
            float scale = (mat.sigma_y + (vm - mat.sigma_y) * (1.0f - relax)) / vm;
            S11 *= scale;
            S22 *= scale;
            S12 *= scale;
        }
    }

    float tr_e = E11 + E22;
    float diff_e = E11 - E22;
    float rad_e = sqrtf(diff_e * diff_e + 4.0f * E12 * E12);
    float ep0 = 0.5f * (tr_e + rad_e);
    float ep1 = 0.5f * (tr_e - rad_e);
    float ep0p = fmaxf(0.0f, ep0);
    float ep1p = fmaxf(0.0f, ep1);
    float tr_ep = fmaxf(0.0f, tr_e);

    float e_eff = 0.5f * (mat.e1 + mat.e2) * hygro_stiff;
    float lam2 = e_eff * nu12 / ((1.0f + nu12) * (1.0f - nu12));
    float mu2 = 0.5f * e_eff / (1.0f + nu12);
    float psi_plus = 0.5f * lam2 * tr_ep * tr_ep + mu2 * (ep0p * ep0p + ep1p * ep1p);

    if (mat.g_c > 0.0f && psi_plus > 0.0f) {
        if (!(pts.flags[v0] & 3u)) atomicMax_f(&pts.hist_h[v0], psi_plus / fmaxf(0.2f, pts.gc_scale[v0]));
        if (!(pts.flags[v1] & 3u)) atomicMax_f(&pts.hist_h[v1], psi_plus / fmaxf(0.2f, pts.gc_scale[v1]));
        if (!(pts.flags[v2] & 3u)) atomicMax_f(&pts.hist_h[v2], psi_plus / fmaxf(0.2f, pts.gc_scale[v2]));
    }

    float g_d = (1.0f - d_tri) * (1.0f - d_tri) + 1e-5f;
    if (d_tri > 1e-4f) {
        float tr_s = S11 + S22;
        float diff_s = S11 - S22;
        float rad_s = sqrtf(diff_s * diff_s + 4.0f * S12 * S12);
        float s_a = 0.5f * (tr_s + rad_s);
        float s_b = 0.5f * (tr_s - rad_s);
        float c2 = 1.0f, s2 = 0.0f, cs = 0.0f;
        if (rad_s > 1e-9f) {
            float phi = 0.5f * atan2f(2.0f * S12, diff_s);
            float c = cosf(phi), s = sinf(phi);
            c2 = c * c; s2 = s * s; cs = c * s;
        }
        float sa_deg = (s_a > 0.0f) ? (s_a * g_d) : s_a;
        float sb_deg = (s_b > 0.0f) ? (s_b * g_d) : s_b;
        S11 = sa_deg * c2 + sb_deg * s2;
        S22 = sa_deg * s2 + sb_deg * c2;
        S12 = (sa_deg - sb_deg) * cs;
    }

    float eta_eff = mat.eta_visc * hygro_damp * g_d;
    S11 += eta_eff * Ed11;
    S22 += eta_eff * Ed22;
    S12 += eta_eff * Ed12;

    float fv1[3], fv2[3], fv0[3];
    for (int k = 0; k < 3; ++k) {
        float p1 = S11 * f1[k] + S12 * f2[k];
        float p2 = S12 * f1[k] + S22 * f2[k];
        fv1[k] = -vol * (d00 * p1 + d01 * p2);
        fv2[k] = -vol * (d10 * p1 + d11 * p2);
        fv0[k] = -(fv1[k] + fv2[k]);
    }

    atomicAdd(&shell.p_fx[v0], fv0[0]); atomicAdd(&shell.p_fy[v0], fv0[1]); atomicAdd(&shell.p_fz[v0], fv0[2]);
    atomicAdd(&shell.p_fx[v1], fv1[0]); atomicAdd(&shell.p_fy[v1], fv1[1]); atomicAdd(&shell.p_fz[v1], fv1[2]);
    atomicAdd(&shell.p_fx[v2], fv2[0]); atomicAdd(&shell.p_fy[v2], fv2[1]); atomicAdd(&shell.p_fz[v2], fv2[2]);
}

__global__ void k_shell_triangle_membrane(
    ShellMeshGpu shell,
    MpmParticlesGpu pts,
    const MaterialParamsGpu* mats,
    float dt_s
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= shell.num_tris) return;
    d_shell_triangle_membrane_one(t, shell, pts, mats, dt_s);
}

__device__ __forceinline__ void d_shell_hinge_bending_one(
    int h_idx,
    ShellMeshGpu& shell,
    MpmParticlesGpu& pts,
    const MaterialParamsGpu* mats,
    float dt_s
) {
    int e0 = shell.hinge_e0[h_idx];
    int e1 = shell.hinge_e1[h_idx];
    int o0 = shell.hinge_o0[h_idx];
    int o1 = shell.hinge_o1[h_idx];

    const MaterialParamsGpu mat = mats[shell.hinge_mat_id[h_idx]];
    if (mat.flags & 4u) return;

    float d_edge = 0.5f * (pts.damage[e0] + pts.damage[e1]);
    d_edge = fmaxf(d_edge, 0.25f * (pts.damage[o0] + pts.damage[o1]));
    shell.hinge_dmax[h_idx] = fmaxf(shell.hinge_dmax[h_idx], d_edge);
    if (d_edge > 0.90f) return;

    float x0[3] = {pts.px[e0], pts.py[e0], pts.pz[e0]};
    float x1[3] = {pts.px[e1], pts.py[e1], pts.pz[e1]};
    float x2[3] = {pts.px[o0], pts.py[o0], pts.pz[o0]};
    float x3[3] = {pts.px[o1], pts.py[o1], pts.pz[o1]};

    float e[3] = {x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]};
    float x20[3] = {x2[0] - x0[0], x2[1] - x0[1], x2[2] - x0[2]};
    float x30[3] = {x3[0] - x0[0], x3[1] - x0[1], x3[2] - x0[2]};

    float e_len2 = dot3(e, e);
    if (e_len2 < 1e-16f) return;
    float e_len = sqrtf(e_len2);
    float inv_e = 1.0f / e_len;
    float e_hat[3] = {e[0] * inv_e, e[1] * inv_e, e[2] * inv_e};

    float n1[3], n2[3];
    cross3(e, x20, n1);
    cross3(x30, e, n2);
    float n1_len2 = dot3(n1, n1);
    float n2_len2 = dot3(n2, n2);
    if (n1_len2 < 1e-20f || n2_len2 < 1e-20f) return;

    float n1_len = sqrtf(n1_len2);
    float n2_len = sqrtf(n2_len2);
    float n1_hat[3] = {n1[0] / n1_len, n1[1] / n1_len, n1[2] / n1_len};
    float n2_hat[3] = {n2[0] / n2_len, n2[1] / n2_len, n2[2] / n2_len};

    float c_n[3];
    cross3(n1_hat, n2_hat, c_n);
    float sin_th = dot3(c_n, e_hat);
    float cos_th = dot3(n1_hat, n2_hat);
    float theta = atan2f(sin_th, cos_th);

    float dtheta = theta - shell.hinge_theta0[h_idx];
    const float PI_F = 3.14159265359f;
    if (dtheta > PI_F) dtheta -= 2.0f * PI_F;
    if (dtheta < -PI_F) dtheta += 2.0f * PI_F;

    float c02 = dot3(x20, e_hat) * inv_e;
    float c12 = 1.0f - c02;
    float c03 = dot3(x30, e_hat) * inv_e;
    float c13 = 1.0f - c03;

    float s_o0 = -e_len / n1_len2;
    float s_o1 = -e_len / n2_len2;
    float g_o0[3] = {s_o0 * n1[0], s_o0 * n1[1], s_o0 * n1[2]};
    float g_o1[3] = {s_o1 * n2[0], s_o1 * n2[1], s_o1 * n2[2]};
    float g_e0[3], g_e1[3];
    for (int k = 0; k < 3; ++k) {
        g_e0[k] = -c12 * g_o0[k] - c13 * g_o1[k];
        g_e1[k] = -c02 * g_o0[k] - c03 * g_o1[k];
    }

    float theta_dot =
        g_e0[0] * pts.vx[e0] + g_e0[1] * pts.vy[e0] + g_e0[2] * pts.vz[e0] +
        g_e1[0] * pts.vx[e1] + g_e1[1] * pts.vy[e1] + g_e1[2] * pts.vz[e1] +
        g_o0[0] * pts.vx[o0] + g_o0[1] * pts.vy[o0] + g_o0[2] * pts.vz[o0] +
        g_o1[0] * pts.vx[o1] + g_o1[1] * pts.vy[o1] + g_o1[2] * pts.vz[o1];

    float w_avg = 0.25f * (pts.wetness[e0] + pts.wetness[e1] + pts.wetness[o0] + pts.wetness[o1]);
    float hygro_stiff = (mat.hygro_softening > 0.0f) ? fmaxf(0.15f, 1.0f - mat.hygro_softening * w_avg) : 1.0f;
    float hygro_damp  = (mat.hygro_softening > 0.0f) ? (1.0f + 2.5f * w_avg) : 1.0f;

    float h_char = fmaxf(1e-4f, 0.5f * ((n1_len * inv_e) + (n2_len * inv_e)));
    float eps_bend = 0.5f * mat.thickness * fabsf(dtheta) / h_char;
    float e_eff = 0.5f * (mat.e1 + mat.e2) * hygro_stiff;
    float psi_bend = 0.5f * e_eff * eps_bend * eps_bend;
    if (mat.g_c > 0.0f && psi_bend > 0.0f) {
        if (!(pts.flags[e0] & 3u)) atomicMax_f(&pts.hist_h[e0], psi_bend / fmaxf(0.2f, pts.gc_scale[e0]));
        if (!(pts.flags[e1] & 3u)) atomicMax_f(&pts.hist_h[e1], psi_bend / fmaxf(0.2f, pts.gc_scale[e1]));
    }

    float g_d = (1.0f - d_edge) * (1.0f - d_edge) + 1e-5f;
    float kb = shell.hinge_kb[h_idx] * g_d * hygro_stiff;

    float g2_sum = (dot3(g_e0, g_e0) / fmaxf(1e-12f, pts.mass[e0])) +
                   (dot3(g_e1, g_e1) / fmaxf(1e-12f, pts.mass[e1])) +
                   (dot3(g_o0, g_o0) / fmaxf(1e-12f, pts.mass[o0])) +
                   (dot3(g_o1, g_o1) / fmaxf(1e-12f, pts.mass[o1]));
    float cb = (mat.eta_visc / fmaxf(1e-4f, e_eff)) * kb * hygro_damp;
    if (g2_sum > 1e-12f && dt_s > 0.0f) {
        float kb_max = 0.25f / (g2_sum * dt_s * dt_s);
        kb = fminf(kb, kb_max);
        float cb_max = 0.35f / (g2_sum * dt_s);
        cb = fminf(cb, cb_max);
    }

    float torque = -(kb * dtheta + cb * theta_dot);
    atomicAdd(&shell.p_fx[e0], torque * g_e0[0]); atomicAdd(&shell.p_fy[e0], torque * g_e0[1]); atomicAdd(&shell.p_fz[e0], torque * g_e0[2]);
    atomicAdd(&shell.p_fx[e1], torque * g_e1[0]); atomicAdd(&shell.p_fy[e1], torque * g_e1[1]); atomicAdd(&shell.p_fz[e1], torque * g_e1[2]);
    atomicAdd(&shell.p_fx[o0], torque * g_o0[0]); atomicAdd(&shell.p_fy[o0], torque * g_o0[1]); atomicAdd(&shell.p_fz[o0], torque * g_o0[2]);
    atomicAdd(&shell.p_fx[o1], torque * g_o1[0]); atomicAdd(&shell.p_fy[o1], torque * g_o1[1]); atomicAdd(&shell.p_fz[o1], torque * g_o1[2]);
}

__global__ void k_shell_hinge_bending(
    ShellMeshGpu shell,
    MpmParticlesGpu pts,
    const MaterialParamsGpu* mats,
    float dt_s
) {
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (h_idx >= shell.num_hinges) return;
    d_shell_hinge_bending_one(h_idx, shell, pts, mats, dt_s);
}

__device__ __forceinline__ void d_shell_normalize_normal_one(int p, MpmParticlesGpu& pts) {
    if (!(pts.flags[p] & 8u)) return;
    float nx = pts.nx[p], ny = pts.ny[p], nz = pts.nz[p];
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    if (len > 1e-12f) {
        float inv = 1.0f / len;
        pts.nx[p] = nx * inv;
        pts.ny[p] = ny * inv;
        pts.nz[p] = nz * inv;
    } else {
        pts.nx[p] = pts.nx0[p];
        pts.ny[p] = pts.ny0[p];
        pts.nz[p] = pts.nz0[p];
    }
}

__global__ void k_shell_normalize_normals(MpmParticlesGpu pts) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;
    d_shell_normalize_normal_one(p, pts);
}

__device__ __forceinline__ void d_contact_build_hash_one(
    int p,
    MpmParticlesGpu& pts,
    int32_t* hash_head,
    int32_t* hash_next,
    int32_t hash_size,
    float cell_size
) {
    if (!(pts.flags[p] & 12u)) {
        hash_next[p] = -1;
        return;
    }
    float inv_c = 1.0f / cell_size;
    int ix = (int)floorf(pts.px[p] * inv_c);
    int iy = (int)floorf(pts.py[p] * inv_c);
    int iz = (int)floorf(pts.pz[p] * inv_c);
    uint32_t slot = hash_cell(ix, iy, iz, hash_size);
    hash_next[p] = atomicExch(&hash_head[slot], p);
}

__global__ void k_contact_build_hash(
    MpmParticlesGpu pts,
    int32_t* hash_head,
    int32_t* hash_next,
    int32_t hash_size,
    float cell_size
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;
    d_contact_build_hash_one(p, pts, hash_head, hash_next, hash_size, cell_size);
}

__device__ __forceinline__ void d_capillary_lubrication_contact_one(
    int p,
    MpmParticlesGpu& pts,
    ShellMeshGpu& shell,
    const MaterialParamsGpu* mats,
    const int32_t* hash_head,
    const int32_t* hash_next,
    int32_t hash_size,
    float cell_size,
    float dt_s,
    float sigma_w,
    float mu_w
) {
    uint32_t flag_p = pts.flags[p];
    if (!(flag_p & 12u)) return;
    if (flag_p & 3u) return; // Prescribed or clamped particles do not integrate external contact acceleration

    const MaterialParamsGpu mat_p = mats[pts.mat_id[p]];
    if (mat_p.flags & 4u) return;

    int32_t body_p = pts.body_id[p];
    float px = pts.px[p], py = pts.py[p], pz = pts.pz[p];
    float vx = pts.vx[p], vy = pts.vy[p], vz = pts.vz[p];
    float mp = pts.mass[p];
    float ap = pts.area[p];
    float wp = pts.wetness[p];
    float r_p = (flag_p & 8u) ? fmaxf(0.5f * pts.thickness[p], 0.42f * sqrtf(ap))
                              : (0.55f * cbrtf(pts.vol0[p]));

    float inv_c = 1.0f / cell_size;
    int ix = (int)floorf(px * inv_c);
    int iy = (int)floorf(py * inv_c);
    int iz = (int)floorf(pz * inv_c);

    float fx_sum = 0.0f, fy_sum = 0.0f, fz_sum = 0.0f;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                uint32_t slot = hash_cell(ix + dx, iy + dy, iz + dz, hash_size);
                int q = hash_head[slot];
                int iter = 0;
                while (q >= 0 && iter < 64) {
                    ++iter;
                    if (pts.body_id[q] != body_p) {
                        uint32_t flag_q = pts.flags[q];
                        float aq = pts.area[q];
                        float r_q = (flag_q & 8u) ? fmaxf(0.5f * pts.thickness[q], 0.42f * sqrtf(aq))
                                                  : (0.55f * cbrtf(pts.vol0[q]));
                        float d_sum = r_p + r_q;
                        float d_cap = 1.5f * d_sum;

                        float dx_pq = px - pts.px[q];
                        float dy_pq = py - pts.py[q];
                        float dz_pq = pz - pts.pz[q];
                        float dist2 = dx_pq * dx_pq + dy_pq * dy_pq + dz_pq * dz_pq;

                        if (dist2 < d_cap * d_cap) {
                            float dist = sqrtf(dist2);
                            float gap = dist - d_sum;

                            float nx = 0.0f, ny = 0.0f, nz = 1.0f;
                            if (dist > 1e-8f) {
                                float inv_d = 1.0f / dist;
                                nx = dx_pq * inv_d;
                                ny = dy_pq * inv_d;
                                nz = dz_pq * inv_d;
                            }

                            const MaterialParamsGpu mat_q = mats[pts.mat_id[q]];
                            float mq = pts.mass[q];
                            float m_eff = ((mat_q.flags & 4u) || (flag_q & 3u))
                                ? mp
                                : ((mp * mq) / fmaxf(1e-18f, mp + mq));

                            float rvx = vx - pts.vx[q];
                            float rvy = vy - pts.vy[q];
                            float rvz = vz - pts.vz[q];
                            float vn = rvx * nx + rvy * ny + rvz * nz;

                            float f_n = 0.0f;
                            if (gap < 0.0f) {
                                float e_pair = fminf(0.5f * (mat_p.e1 + mat_p.e2), 0.5f * (mat_q.e1 + mat_q.e2));
                                float k_raw = 0.05f * e_pair * sqrtf(fminf(ap, aq));
                                float k_cap = 0.04f * m_eff / fmaxf(1e-18f, dt_s * dt_s);
                                float k_c = fminf(k_raw, k_cap);
                                float f_rep = k_c * (-gap);
                                float c_n = 0.30f * sqrtf(k_c * m_eff);
                                f_n += f_rep - c_n * vn;

                                float tvx = rvx - vn * nx;
                                float tvy = rvy - vn * ny;
                                float tvz = rvz - vn * nz;
                                float vt = sqrtf(tvx * tvx + tvy * tvy + tvz * tvz);
                                float mu_f = 0.5f * (mat_p.friction_mu + mat_q.friction_mu);
                                if (vt > 1e-8f && mu_f > 0.0f) {
                                    float ft = fminf(mu_f * f_rep, 0.25f * m_eff * vt / fmaxf(1e-9f, dt_s));
                                    float s = -ft / vt;
                                    fx_sum += s * tvx;
                                    fy_sum += s * tvy;
                                    fz_sum += s * tvz;
                                }
                            }

                            float w_pair = 0.5f * (wp + pts.wetness[q]);
                            if (w_pair > 0.01f) {
                                float a_patch = sqrtf(fminf(ap, aq) / 3.14159265f);
                                float cos_t = 0.5f * (mat_p.contact_angle_cos + mat_q.contact_angle_cos);
                                if (cos_t > 0.0f) {
                                    float ramp = clamp_f(1.0f - fmaxf(0.0f, gap) / d_cap, 0.0f, 1.0f);
                                    float f_cap = w_pair * 2.0f * 3.14159265f * a_patch * sigma_w * cos_t * ramp;
                                    f_n -= fminf(f_cap, 0.25f * m_eff / fmaxf(1e-12f, dt_s));
                                }

                                float h_film = fmaxf(0.15f * d_sum, fabsf(gap) + 0.15f * d_sum);
                                float a4 = a_patch * a_patch * a_patch * a_patch;
                                float c_stefan = w_pair * 1.5f * 3.14159265f * mu_w * a4 / (h_film * h_film * h_film);
                                float f_stef = fminf(c_stefan * fabsf(vn), 0.45f * m_eff * fabsf(vn) / fmaxf(1e-9f, dt_s));
                                f_n += (vn < 0.0f) ? f_stef : -f_stef;
                            }

                            fx_sum += f_n * nx;
                            fy_sum += f_n * ny;
                            fz_sum += f_n * nz;
                        }
                    }
                    q = hash_next[q];
                }
            }
        }
    }

    shell.p_fx[p] += fx_sum;
    shell.p_fy[p] += fy_sum;
    shell.p_fz[p] += fz_sum;
}

__global__ void k_capillary_lubrication_contact(
    MpmParticlesGpu pts,
    ShellMeshGpu shell,
    const MaterialParamsGpu* mats,
    const int32_t* hash_head,
    const int32_t* hash_next,
    int32_t hash_size,
    float cell_size,
    float dt_s,
    float sigma_w,
    float mu_w
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;
    d_capillary_lubrication_contact_one(
        p, pts, shell, mats, hash_head, hash_next, hash_size, cell_size, dt_s, sigma_w, mu_w
    );
}

__device__ __forceinline__ void d_shell_integrate_particle_one(
    int p,
    MpmParticlesGpu& pts,
    const ShellMeshGpu& shell,
    const MaterialParamsGpu* mats,
    float gx, float gy, float gz,
    float dt_s,
    int32_t has_floor,
    float floor_z,
    float floor_mu
) {
    const MaterialParamsGpu mat = mats[pts.mat_id[p]];
    uint32_t pflag = pts.flags[p];
    if (mat.flags & 4u) return;
    if (!(mat.flags & 1u) && !(pflag & 2u)) return;

    if (pflag & 1u) {
        pts.vx[p] = 0.0f;
        pts.vy[p] = 0.0f;
        pts.vz[p] = 0.0f;
        return;
    }
    if (pflag & 2u) {
        float vx = pts.v_presc_x[p];
        float vy = pts.v_presc_y[p];
        float vz = pts.v_presc_z[p];
        pts.vx[p] = vx;
        pts.vy[p] = vy;
        pts.vz[p] = vz;
        pts.px[p] += dt_s * vx;
        pts.py[p] += dt_s * vy;
        pts.pz[p] += dt_s * vz;
        return;
    }

    float f_rho = fmaxf(0.0f, pts.fluid_rho[p]);
    float m_add = f_rho * pts.area[p] * fmaxf(0.0f, pts.hyd_len[p]);
    float m_eff = fmaxf(1e-18f, pts.mass[p] + m_add);
    float inv_m = 1.0f / m_eff;
    float fx = shell.p_fx[p] + pts.fx_ext[p];
    float fy = shell.p_fy[p] + pts.fy_ext[p];
    float fz = shell.p_fz[p] + pts.fz_ext[p];

    float damp = (f_rho > 50.0f) ? expf(-dt_s * 1.2f) : 1.0f;
    float vx = (pts.vx[p] + dt_s * (fx * inv_m + gx)) * damp;
    float vy = (pts.vy[p] + dt_s * (fy * inv_m + gy)) * damp;
    float vz = (pts.vz[p] + dt_s * (fz * inv_m + gz)) * damp;

    float x_next = pts.px[p] + dt_s * vx;
    float y_next = pts.py[p] + dt_s * vy;
    float z_next = pts.pz[p] + dt_s * vz;

    float z_min = floor_z + 0.5f * pts.thickness[p];
    if (has_floor && z_next < z_min) {
        z_next = z_min;
        if (vz < 0.0f) {
            float dvz = -vz;
            vz = 0.0f;
            if (floor_mu > 0.0f) {
                float vt = sqrtf(vx * vx + vy * vy);
                if (vt > 1e-8f) {
                    float drop = fminf(vt, floor_mu * dvz);
                    float s = (vt - drop) / vt;
                    vx *= s;
                    vy *= s;
                }
            }
        }
    }

    pts.vx[p] = vx;
    pts.vy[p] = vy;
    pts.vz[p] = vz;
    pts.px[p] = x_next;
    pts.py[p] = y_next;
    pts.pz[p] = z_next;
}

__global__ void k_shell_integrate_particles(
    MpmParticlesGpu pts,
    ShellMeshGpu shell,
    const MaterialParamsGpu* mats,
    float gx, float gy, float gz,
    float dt_s,
    int32_t has_floor,
    float floor_z,
    float floor_mu
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;
    d_shell_integrate_particle_one(p, pts, shell, mats, gx, gy, gz, dt_s, has_floor, floor_z, floor_mu);
}

// Single-block fused shell + contact substepper for small meshes (pts.count <= 4096)
__global__ void k_shell_substep_fused(
    ShellMeshGpu shell,
    MpmParticlesGpu pts,
    const MaterialParamsGpu* mats,
    int32_t* hash_head,
    int32_t* hash_next,
    int32_t hash_size,
    float cell_size,
    float gx, float gy, float gz,
    float dt_s,
    int32_t n_mpm,
    int32_t do_contact,
    int32_t has_floor,
    float floor_z,
    float floor_mu,
    float sigma_w,
    float mu_w
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(gridDim.x * blockDim.x);
    for (int sub = 0; sub < n_mpm; ++sub) {
        for (int p = tid; p < pts.count; p += stride) {
            shell.p_fx[p] = 0.0f;
            shell.p_fy[p] = 0.0f;
            shell.p_fz[p] = 0.0f;
            if (pts.flags[p] & 8u) {
                pts.nx[p] = 0.0f;
                pts.ny[p] = 0.0f;
                pts.nz[p] = 0.0f;
            }
            if (do_contact && hash_head != nullptr) {
                if (!(mats[pts.mat_id[p]].flags & 4u) && (pts.flags[p] & 12u)) {
                    float inv_c = 1.0f / cell_size;
                    int ix = (int)floorf(pts.px[p] * inv_c);
                    int iy = (int)floorf(pts.py[p] * inv_c);
                    int iz = (int)floorf(pts.pz[p] * inv_c);
                    hash_head[hash_cell(ix, iy, iz, hash_size)] = -1;
                }
            }
        }
        grid.sync();

        for (int t = tid; t < shell.num_tris; t += stride) {
            d_shell_triangle_membrane_one(t, shell, pts, mats, dt_s);
        }
        grid.sync();

        for (int p = tid; p < pts.count; p += stride) {
            d_shell_normalize_normal_one(p, pts);
        }
        for (int h_idx = tid; h_idx < shell.num_hinges; h_idx += stride) {
            d_shell_hinge_bending_one(h_idx, shell, pts, mats, dt_s);
        }
        if (do_contact && hash_head != nullptr && hash_size > 0) {
            for (int p = tid; p < pts.count; p += stride) {
                if (mats[pts.mat_id[p]].flags & 4u) {
                    hash_next[p] = -1;
                } else {
                    d_contact_build_hash_one(p, pts, hash_head, hash_next, hash_size, cell_size);
                }
            }
            grid.sync();
            for (int p = tid; p < pts.count; p += stride) {
                d_capillary_lubrication_contact_one(
                    p, pts, shell, mats, hash_head, hash_next, hash_size, cell_size, dt_s, sigma_w, mu_w
                );
            }
        }
        grid.sync();

        for (int p = tid; p < pts.count; p += stride) {
            d_shell_integrate_particle_one(
                p, pts, shell, mats, gx, gy, gz, dt_s, has_floor, floor_z, floor_mu
            );
        }
        grid.sync();
    }
}

__global__ void k_mpm_set_body_prescribed_vel(
    MpmParticlesGpu pts,
    int32_t target_body_id,
    float vx, float vy, float vz,
    float wx, float wy, float wz,
    float piv_x, float piv_y, float piv_z
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;
    if (pts.body_id[p] != target_body_id || !(pts.flags[p] & 2u)) return;
    float rx = pts.px[p] - piv_x;
    float ry = pts.py[p] - piv_y;
    float rz = pts.pz[p] - piv_z;
    pts.v_presc_x[p] = vx + (wy * rz - wz * ry);
    pts.v_presc_y[p] = vy + (wz * rx - wx * rz);
    pts.v_presc_z[p] = vz + (wx * ry - wy * rx);
}

// ---- Unified Phase-Field Fracture Solver (AT2 Semi-Implicit on MpmGridGpu) ----

__global__ void k_pf_clear_grid(MpmGridGpu grid) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_nodes = (size_t)grid.nx * (size_t)grid.ny * (size_t)grid.nz;
    if (idx >= n_nodes) return;

    grid.d_field[idx] = 0.0f;
    grid.d_new[idx] = 0.0f;
    grid.h_field[idx] = 0.0f;
    grid.gc_field[idx] = 0.0f;
    grid.lc_field[idx] = 0.0f;
    grid.eta_field[idx] = 0.0f;
    grid.w_field[idx] = 0.0f;
}

__global__ void k_pf_p2g(MpmParticlesGpu pts, MpmGridGpu grid, const MaterialParamsGpu* mats) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;

    const MaterialParamsGpu mat = mats[pts.mat_id[p]];
    if ((mat.flags & 4u) || mat.g_c <= 0.0f) return;

    float fx_p = (pts.px[p] - grid.ox) / grid.dx;
    float fy_p = (pts.py[p] - grid.oy) / grid.dx;
    float fz_p = (pts.pz[p] - grid.oz) / grid.dx;

    int ix[3], iy[3], iz[3];
    float wx[3], wy[3], wz[3];
    shell_bspline_weights(fx_p, grid.nx, ix, wx);
    shell_bspline_weights(fy_p, grid.ny, iy, wy);
    shell_bspline_weights(fz_p, grid.nz, iz, wz);

    float d_p = pts.damage[p];
    float h_p = pts.hist_h[p];
    float gc_p = mat.g_c;
    float lc_p = fmaxf(mat.l_c, 0.5f * grid.dx);
    float eta_p = mat.eta_frac;

    for (int kc = 0; kc < 3; ++kc) {
        for (int jc = 0; jc < 3; ++jc) {
            for (int ic = 0; ic < 3; ++ic) {
                float w = wx[ic] * wy[jc] * wz[kc];
                if (w > 1e-8f) {
                    size_t nid = (size_t)ix[ic] + (size_t)grid.nx * ((size_t)iy[jc] + (size_t)grid.ny * (size_t)iz[kc]);
                    atomicAdd(&grid.w_field[nid], w);
                    atomicAdd(&grid.d_field[nid], w * d_p);
                    atomicAdd(&grid.h_field[nid], w * h_p);
                    atomicAdd(&grid.gc_field[nid], w * gc_p);
                    atomicAdd(&grid.lc_field[nid], w * lc_p);
                    atomicAdd(&grid.eta_field[nid], w * eta_p);
                }
            }
        }
    }
}

__global__ void k_pf_normalize(MpmGridGpu grid) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_nodes = (size_t)grid.nx * (size_t)grid.ny * (size_t)grid.nz;
    if (idx >= n_nodes) return;

    float w = grid.w_field[idx];
    if (w > 1e-12f) {
        float inv_w = 1.0f / w;
        float d0 = clamp_f(grid.d_field[idx] * inv_w, 0.0f, 1.0f);
        grid.d_field[idx] = d0;
        grid.d_new[idx] = d0;
        grid.h_field[idx] *= inv_w;
        grid.gc_field[idx] *= inv_w;
        grid.lc_field[idx] *= inv_w;
        grid.eta_field[idx] *= inv_w;
    }
}

__global__ void k_pf_solve_jacobi(
    MpmGridGpu grid,
    const float* d_in,
    float* d_out,
    float dt_pf
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= grid.nx || j >= grid.ny || k >= grid.nz) return;

    size_t idx = (size_t)i + (size_t)grid.nx * ((size_t)j + (size_t)grid.ny * (size_t)k);
    if (grid.w_field[idx] <= 1e-12f) {
        d_out[idx] = 0.0f;
        return;
    }

    float d_old = grid.d_field[idx];
    float H = grid.h_field[idx];
    float Gc = grid.gc_field[idx];
    float lc = grid.lc_field[idx];
    float eta = grid.eta_field[idx];
    if (Gc <= 0.0f) {
        d_out[idx] = d_old;
        return;
    }

    float nbr_sum = 0.0f;
    int nbr_cnt = 0;
    const int off[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    for (int m = 0; m < 6; ++m) {
        int ni = i + off[m][0], nj = j + off[m][1], nk = k + off[m][2];
        if (ni >= 0 && ni < grid.nx && nj >= 0 && nj < grid.ny && nk >= 0 && nk < grid.nz) {
            size_t nid = (size_t)ni + (size_t)grid.nx * ((size_t)nj + (size_t)grid.ny * (size_t)nk);
            if (grid.w_field[nid] > 1e-12f) {
                nbr_sum += d_in[nid];
                ++nbr_cnt;
            }
        }
    }

    float eta_dt = eta / fmaxf(1e-9f, dt_pf);
    float diff_c = Gc * lc / (grid.dx * grid.dx);
    float diag = eta_dt + 2.0f * H + Gc / lc + (float)nbr_cnt * diff_c;
    float rhs = eta_dt * d_old + 2.0f * H + diff_c * nbr_sum;

    float d_next = rhs / fmaxf(1e-12f, diag);
    d_out[idx] = clamp_f(fmaxf(d_old, d_next), 0.0f, 1.0f);
}

__global__ void k_pf_g2p(MpmParticlesGpu pts, MpmGridGpu grid, const MaterialParamsGpu* mats) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pts.count) return;

    const MaterialParamsGpu mat = mats[pts.mat_id[p]];
    if ((mat.flags & 4u) || mat.g_c <= 0.0f) return;
    if (pts.flags[p] & 3u) return; // Respect clamped / prescribed grip boundaries

    float fx_p = (pts.px[p] - grid.ox) / grid.dx;
    float fy_p = (pts.py[p] - grid.oy) / grid.dx;
    float fz_p = (pts.pz[p] - grid.oz) / grid.dx;

    int ix[3], iy[3], iz[3];
    float wx[3], wy[3], wz[3];
    shell_bspline_weights(fx_p, grid.nx, ix, wx);
    shell_bspline_weights(fy_p, grid.ny, iy, wy);
    shell_bspline_weights(fz_p, grid.nz, iz, wz);

    float d_interp = 0.0f;
    for (int kc = 0; kc < 3; ++kc) {
        for (int jc = 0; jc < 3; ++jc) {
            for (int ic = 0; ic < 3; ++ic) {
                float w = wx[ic] * wy[jc] * wz[kc];
                size_t nid = (size_t)ix[ic] + (size_t)grid.nx * ((size_t)iy[jc] + (size_t)grid.ny * (size_t)iz[kc]);
                d_interp += w * grid.d_new[nid];
            }
        }
    }

    pts.damage[p] = clamp_f(fmaxf(pts.damage[p], d_interp), 0.0f, 1.0f);
}
