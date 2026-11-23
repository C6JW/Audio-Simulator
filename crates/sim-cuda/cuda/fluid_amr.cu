#include "common.cuh"

__device__ __forceinline__ double trilinear_cell_double(
    const double* arr,
    const double* p_hydro,
    int nxp, int nyp, int nzp,
    float fx, float fy, float fz
) {
    int i0 = max(0, min(nxp - 2, (int)floorf(fx)));
    int j0 = max(0, min(nyp - 2, (int)floorf(fy)));
    int k0 = max(0, min(nzp - 2, (int)floorf(fz)));
    double tx = clamp_d((double)fx - (double)i0, 0.0, 1.0);
    double ty = clamp_d((double)fy - (double)j0, 0.0, 1.0);
    double tz = clamp_d((double)fz - (double)k0, 0.0, 1.0);

    double val = 0.0;
    for (int dk = 0; dk <= 1; ++dk) {
        double wz = dk ? tz : (1.0 - tz);
        for (int dj = 0; dj <= 1; ++dj) {
            double wy = dj ? ty : (1.0 - ty);
            for (int di = 0; di <= 1; ++di) {
                double wx = di ? tx : (1.0 - tx);
                size_t id = idx_cell(i0 + di, j0 + dj, k0 + dk, nxp, nyp);
                double v = arr[id] - (p_hydro ? p_hydro[id] : 0.0);
                val += wx * wy * wz * v;
            }
        }
    }
    return val;
}

__device__ __forceinline__ float trilinear_cell_float(
    const float* arr,
    int nxp, int nyp, int nzp,
    float fx, float fy, float fz
) {
    int i0 = max(0, min(nxp - 2, (int)floorf(fx)));
    int j0 = max(0, min(nyp - 2, (int)floorf(fy)));
    int k0 = max(0, min(nzp - 2, (int)floorf(fz)));
    float tx = clamp_f(fx - (float)i0, 0.0f, 1.0f);
    float ty = clamp_f(fy - (float)j0, 0.0f, 1.0f);
    float tz = clamp_f(fz - (float)k0, 0.0f, 1.0f);

    float val = 0.0f;
    for (int dk = 0; dk <= 1; ++dk) {
        float wz = dk ? tz : (1.0f - tz);
        for (int dj = 0; dj <= 1; ++dj) {
            float wy = dj ? ty : (1.0f - ty);
            for (int di = 0; di <= 1; ++di) {
                float wx = di ? tx : (1.0f - tx);
                val += wx * wy * wz * arr[idx_cell(i0 + di, j0 + dj, k0 + dk, nxp, nyp)];
            }
        }
    }
    return val;
}

__device__ __forceinline__ float trilinear_face_float(
    const float* arr,
    int sx, int sy, int sz,
    float fx, float fy, float fz,
    int axis
) {
    int i0 = max(0, min(sx - 2, (int)floorf(fx)));
    int j0 = max(0, min(sy - 2, (int)floorf(fy)));
    int k0 = max(0, min(sz - 2, (int)floorf(fz)));
    float tx = clamp_f(fx - (float)i0, 0.0f, 1.0f);
    float ty = clamp_f(fy - (float)j0, 0.0f, 1.0f);
    float tz = clamp_f(fz - (float)k0, 0.0f, 1.0f);

    float val = 0.0f;
    for (int dk = 0; dk <= 1; ++dk) {
        float wz = dk ? tz : (1.0f - tz);
        for (int dj = 0; dj <= 1; ++dj) {
            float wy = dj ? ty : (1.0f - ty);
            for (int di = 0; di <= 1; ++di) {
                float wx = di ? tx : (1.0f - tx);
                size_t id = (size_t)(i0 + di) + (size_t)sx * ((size_t)(j0 + dj) + (size_t)sy * (size_t)(k0 + dk));
                val += wx * wy * wz * arr[id];
            }
        }
    }
    return val;
}

__global__ void k_amr_reset_fluxes(FluidLevelGpu child) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < child.ny && j < child.nz) {
        size_t idx = (size_t)i + (size_t)child.ny * (size_t)j;
        if (child.bnd_vflux_xlo) child.bnd_vflux_xlo[idx] = 0.0f;
        if (child.bnd_vflux_xhi) child.bnd_vflux_xhi[idx] = 0.0f;
        if (child.bnd_aflux_xlo) child.bnd_aflux_xlo[idx] = 0.0f;
        if (child.bnd_aflux_xhi) child.bnd_aflux_xhi[idx] = 0.0f;
    }
    if (i < child.nx && j < child.nz) {
        size_t idx = (size_t)i + (size_t)child.nx * (size_t)j;
        if (child.bnd_vflux_ylo) child.bnd_vflux_ylo[idx] = 0.0f;
        if (child.bnd_vflux_yhi) child.bnd_vflux_yhi[idx] = 0.0f;
        if (child.bnd_aflux_ylo) child.bnd_aflux_ylo[idx] = 0.0f;
        if (child.bnd_aflux_yhi) child.bnd_aflux_yhi[idx] = 0.0f;
    }
    if (i < child.nx && j < child.ny) {
        size_t idx = (size_t)i + (size_t)child.nx * (size_t)j;
        if (child.bnd_vflux_zlo) child.bnd_vflux_zlo[idx] = 0.0f;
        if (child.bnd_vflux_zhi) child.bnd_vflux_zhi[idx] = 0.0f;
        if (child.bnd_aflux_zlo) child.bnd_aflux_zlo[idx] = 0.0f;
        if (child.bnd_aflux_zhi) child.bnd_aflux_zhi[idx] = 0.0f;
    }
}

__global__ void k_amr_fill_ghost_cells(
    FluidLevelGpu child,
    FluidLevelGpu parent,
    double theta_time,
    int fill_alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= child.nxp || j >= child.nyp || k >= child.nzp) return;

    int h = child.halo;
    if (i >= h && i < h + child.nx && j >= h && j < h + child.ny && k >= h && k < h + child.nz) {
        return;
    }

    float x = child.ox + ((float)(i - h) + 0.5f) * child.dx;
    float y = child.oy + ((float)(j - h) + 0.5f) * child.dx;
    float z = child.oz + ((float)(k - h) + 0.5f) * child.dx;

    float fx = (x - parent.ox) / parent.dx - 0.5f + (float)parent.halo;
    float fy = (y - parent.oy) / parent.dx - 0.5f + (float)parent.halo;
    float fz = (z - parent.oz) / parent.dx - 0.5f + (float)parent.halo;

    float pac_old = trilinear_cell_float(parent.p_gauge_old, parent.nxp, parent.nyp, parent.nzp, fx, fy, fz);
    float pac_new = trilinear_cell_float(parent.p_gauge, parent.nxp, parent.nyp, parent.nzp, fx, fy, fz);
    float t_f = (float)theta_time;
    float pac = (1.0f - t_f) * pac_old + t_f * pac_new;

    size_t cid = idx_cell(i, j, k, child.nxp, child.nyp);
    child.p_gauge[cid] = pac;

    if (fill_alpha) {
        int ip = max(0, min(parent.nxp - 1, (int)floorf((x - parent.ox) / parent.dx) + parent.halo));
        int jp = max(0, min(parent.nyp - 1, (int)floorf((y - parent.oy) / parent.dx) + parent.halo));
        int kp = max(0, min(parent.nzp - 1, (int)floorf((z - parent.oz) / parent.dx) + parent.halo));
        child.alpha[cid] = clamp_f(parent.alpha[idx_cell(ip, jp, kp, parent.nxp, parent.nyp)], 0.0f, 1.0f);
    }
}

__global__ void k_amr_fill_ghost_faces(
    FluidLevelGpu child,
    FluidLevelGpu parent,
    float theta_time
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int h = child.halo;

    // X-face ghosts & flux-matched patch boundary faces (strictly interior faces i in (h..h+nx))
    if (i <= child.nxp && j < child.nyp && k < child.nzp) {
        bool is_interior = (i > h && i < h + child.nx && j >= h && j < h + child.ny && k >= h && k < h + child.nz);
        if (!is_interior) {
            float x = child.ox + (float)(i - h) * child.dx;
            float y = child.oy + ((float)(j - h) + 0.5f) * child.dx;
            float z = child.oz + ((float)(k - h) + 0.5f) * child.dx;
            float fx = (x - parent.ox) / parent.dx + (float)parent.halo;
            float fy = (y - parent.oy) / parent.dx - 0.5f + (float)parent.halo;
            float fz = (z - parent.oz) / parent.dx - 0.5f + (float)parent.halo;
            float u_mid = trilinear_face_float(parent.ux, parent.nxp + 1, parent.nyp, parent.nzp, fx, fy, fz, 0);
            child.ux[idx_fx(i, j, k, child.nxp, child.nyp)] = u_mid;
        }
    }

    // Y-face ghosts & flux-matched patch boundary faces
    if (i < child.nxp && j <= child.nyp && k < child.nzp) {
        bool is_interior = (i >= h && i < h + child.nx && j > h && j < h + child.ny && k >= h && k < h + child.nz);
        if (!is_interior) {
            float x = child.ox + ((float)(i - h) + 0.5f) * child.dx;
            float y = child.oy + (float)(j - h) * child.dx;
            float z = child.oz + ((float)(k - h) + 0.5f) * child.dx;
            float fx = (x - parent.ox) / parent.dx - 0.5f + (float)parent.halo;
            float fy = (y - parent.oy) / parent.dx + (float)parent.halo;
            float fz = (z - parent.oz) / parent.dx - 0.5f + (float)parent.halo;
            float v_mid = trilinear_face_float(parent.uy, parent.nxp, parent.nyp + 1, parent.nzp, fx, fy, fz, 1);
            child.uy[idx_fy(i, j, k, child.nxp, child.nyp)] = v_mid;
        }
    }

    // Z-face ghosts & flux-matched patch boundary faces
    if (i < child.nxp && j < child.nyp && k <= child.nzp) {
        bool is_interior = (i >= h && i < h + child.nx && j >= h && j < h + child.ny && k > h && k < h + child.nz);
        if (!is_interior) {
            float x = child.ox + ((float)(i - h) + 0.5f) * child.dx;
            float y = child.oy + ((float)(j - h) + 0.5f) * child.dx;
            float z = child.oz + (float)(k - h) * child.dx;
            float fx = (x - parent.ox) / parent.dx - 0.5f + (float)parent.halo;
            float fy = (y - parent.oy) / parent.dx - 0.5f + (float)parent.halo;
            float fz = (z - parent.oz) / parent.dx + (float)parent.halo;
            float w_mid = trilinear_face_float(parent.uz, parent.nxp, parent.nyp, parent.nzp + 1, fx, fy, fz, 2);
            child.uz[idx_fz(i, j, k, child.nxp, child.nyp)] = w_mid;
        }
    }
}

__global__ void k_amr_reflux_pressure(
    FluidLevelGpu child,
    FluidLevelGpu parent,
    FluidConstantsGpu c,
    float dt_parent
) {
    int r = child.refine_ratio;
    double r2_inv = 1.0 / (double)(r * r);
    int nx_c = child.nx / r;
    int ny_c = child.ny / r;
    int nz_c = child.nz / r;

    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int ph = parent.halo;

    // X-lo and X-hi outer coarse rings
    if (a < ny_c && b < nz_c) {
        // X-lo
        if (child.bnd_vflux_xlo && child.off_x > 0) {
            float fine_sum = 0.0f;
            for (int dk = 0; dk < r; ++dk) {
                for (int dj = 0; dj < r; ++dj) {
                    int jf = a * r + dj;
                    int kf = b * r + dk;
                    fine_sum += child.bnd_vflux_xlo[jf + child.ny * kf];
                }
            }
            float fine_avg = fine_sum * (float)r2_inv;
            int ic_out = ph + child.off_x - 1;
            int jc = ph + child.off_y + a;
            int kc = ph + child.off_z + b;
            size_t fx = idx_fx(ic_out + 1, jc, kc, parent.nxp, parent.nyp);
            if (parent.ib_beta_x[fx] < 0.99f) {
                float coarse_flux = parent.ux[fx] * dt_parent;
                size_t cid = idx_cell(ic_out, jc, kc, parent.nxp, parent.nyp);
                float k_mix = mix_bulk_modulus_f(parent.alpha[cid], parent.p_gauge[cid] + parent.p_hydro[cid], c);
                parent.p_gauge[cid] -= k_mix * (fine_avg - coarse_flux) / parent.dx;
            }
        }
        // X-hi
        if (child.bnd_vflux_xhi && child.off_x + nx_c < parent.nx) {
            float fine_sum = 0.0f;
            for (int dk = 0; dk < r; ++dk) {
                for (int dj = 0; dj < r; ++dj) {
                    int jf = a * r + dj;
                    int kf = b * r + dk;
                    fine_sum += child.bnd_vflux_xhi[jf + child.ny * kf];
                }
            }
            float fine_avg = fine_sum * (float)r2_inv;
            int ic_out = ph + child.off_x + nx_c;
            int jc = ph + child.off_y + a;
            int kc = ph + child.off_z + b;
            size_t fx = idx_fx(ic_out, jc, kc, parent.nxp, parent.nyp);
            if (parent.ib_beta_x[fx] < 0.99f) {
                float coarse_flux = parent.ux[fx] * dt_parent;
                size_t cid = idx_cell(ic_out, jc, kc, parent.nxp, parent.nyp);
                float k_mix = mix_bulk_modulus_f(parent.alpha[cid], parent.p_gauge[cid] + parent.p_hydro[cid], c);
                parent.p_gauge[cid] += k_mix * (fine_avg - coarse_flux) / parent.dx;
            }
        }
    }

    // Y-lo and Y-hi outer coarse rings
    if (a < nx_c && b < nz_c) {
        if (child.bnd_vflux_ylo && child.off_y > 0) {
            float fine_sum = 0.0f;
            for (int dk = 0; dk < r; ++dk) {
                for (int di = 0; di < r; ++di) {
                    int if_ = a * r + di;
                    int kf = b * r + dk;
                    fine_sum += child.bnd_vflux_ylo[if_ + child.nx * kf];
                }
            }
            float fine_avg = fine_sum * (float)r2_inv;
            int ic = ph + child.off_x + a;
            int jc_out = ph + child.off_y - 1;
            int kc = ph + child.off_z + b;
            size_t fy = idx_fy(ic, jc_out + 1, kc, parent.nxp, parent.nyp);
            if (parent.ib_beta_y[fy] < 0.99f) {
                float coarse_flux = parent.uy[fy] * dt_parent;
                size_t cid = idx_cell(ic, jc_out, kc, parent.nxp, parent.nyp);
                float k_mix = mix_bulk_modulus_f(parent.alpha[cid], parent.p_gauge[cid] + parent.p_hydro[cid], c);
                parent.p_gauge[cid] -= k_mix * (fine_avg - coarse_flux) / parent.dx;
            }
        }
        if (child.bnd_vflux_yhi && child.off_y + ny_c < parent.ny) {
            float fine_sum = 0.0f;
            for (int dk = 0; dk < r; ++dk) {
                for (int di = 0; di < r; ++di) {
                    int if_ = a * r + di;
                    int kf = b * r + dk;
                    fine_sum += child.bnd_vflux_yhi[if_ + child.nx * kf];
                }
            }
            float fine_avg = fine_sum * (float)r2_inv;
            int ic = ph + child.off_x + a;
            int jc_out = ph + child.off_y + ny_c;
            int kc = ph + child.off_z + b;
            size_t fy = idx_fy(ic, jc_out, kc, parent.nxp, parent.nyp);
            if (parent.ib_beta_y[fy] < 0.99f) {
                float coarse_flux = parent.uy[fy] * dt_parent;
                size_t cid = idx_cell(ic, jc_out, kc, parent.nxp, parent.nyp);
                float k_mix = mix_bulk_modulus_f(parent.alpha[cid], parent.p_gauge[cid] + parent.p_hydro[cid], c);
                parent.p_gauge[cid] += k_mix * (fine_avg - coarse_flux) / parent.dx;
            }
        }
    }

    // Z-lo and Z-hi outer coarse rings
    if (a < nx_c && b < ny_c) {
        if (child.bnd_vflux_zlo && child.off_z > 0) {
            float fine_sum = 0.0f;
            for (int dj = 0; dj < r; ++dj) {
                for (int di = 0; di < r; ++di) {
                    int if_ = a * r + di;
                    int jf = b * r + dj;
                    fine_sum += child.bnd_vflux_zlo[if_ + child.nx * jf];
                }
            }
            float fine_avg = fine_sum * (float)r2_inv;
            int ic = ph + child.off_x + a;
            int jc = ph + child.off_y + b;
            int kc_out = ph + child.off_z - 1;
            size_t fz = idx_fz(ic, jc, kc_out + 1, parent.nxp, parent.nyp);
            if (parent.ib_beta_z[fz] < 0.99f) {
                float coarse_flux = parent.uz[fz] * dt_parent;
                size_t cid = idx_cell(ic, jc, kc_out, parent.nxp, parent.nyp);
                float k_mix = mix_bulk_modulus_f(parent.alpha[cid], parent.p_gauge[cid] + parent.p_hydro[cid], c);
                parent.p_gauge[cid] -= k_mix * (fine_avg - coarse_flux) / parent.dx;
            }
        }
        if (child.bnd_vflux_zhi && child.off_z + nz_c < parent.nz) {
            float fine_sum = 0.0f;
            for (int dj = 0; dj < r; ++dj) {
                for (int di = 0; di < r; ++di) {
                    int if_ = a * r + di;
                    int jf = b * r + dj;
                    fine_sum += child.bnd_vflux_zhi[if_ + child.nx * jf];
                }
            }
            float fine_avg = fine_sum * (float)r2_inv;
            int ic = ph + child.off_x + a;
            int jc = ph + child.off_y + b;
            int kc_out = ph + child.off_z + nz_c;
            size_t fz = idx_fz(ic, jc, kc_out, parent.nxp, parent.nyp);
            if (parent.ib_beta_z[fz] < 0.99f) {
                float coarse_flux = parent.uz[fz] * dt_parent;
                size_t cid = idx_cell(ic, jc, kc_out, parent.nxp, parent.nyp);
                float k_mix = mix_bulk_modulus_f(parent.alpha[cid], parent.p_gauge[cid] + parent.p_hydro[cid], c);
                parent.p_gauge[cid] += k_mix * (fine_avg - coarse_flux) / parent.dx;
            }
        }
    }
}

__global__ void k_amr_restrict_cells(
    FluidLevelGpu child,
    FluidLevelGpu parent,
    FluidConstantsGpu c
) {
    int r = child.refine_ratio;
    int nx_c = child.nx / r;
    int ny_c = child.ny / r;
    int nz_c = child.nz / r;

    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int k_c = blockIdx.z * blockDim.z + threadIdx.z;
    if (a >= nx_c || b >= ny_c || k_c >= nz_c) return;

    int ic = parent.halo + child.off_x + a;
    int jc = parent.halo + child.off_y + b;
    int kc = parent.halo + child.off_z + k_c;

    float a_sum = 0.0f;
    float sum_inv_k = 0.0f;
    float sum_dp_inv_k = 0.0f;
    float inv_r3 = 1.0f / (float)(r * r * r);

    int ib = child.halo + a * r;
    int jb = child.halo + b * r;
    int kb = child.halo + k_c * r;

    for (int dk = 0; dk < r; ++dk) {
        for (int dj = 0; dj < r; ++dj) {
            for (int di = 0; di < r; ++di) {
                size_t fid = idx_cell(ib + di, jb + dj, kb + dk, child.nxp, child.nyp);
                float af = child.alpha[fid];
                float dpf = child.p_gauge[fid];
                float pf = dpf + child.p_hydro[fid];
                float kf = mix_bulk_modulus_f(af, pf, c);
                float inv_k = 1.0f / kf;
                a_sum += af;
                sum_inv_k += inv_k;
                sum_dp_inv_k += dpf * inv_k;
            }
        }
    }

    size_t cid = idx_cell(ic, jc, kc, parent.nxp, parent.nyp);
    parent.alpha[cid] = clamp_f(a_sum * inv_r3, 0.0f, 1.0f);
    parent.p_gauge[cid] = sum_dp_inv_k / sum_inv_k;
}

__global__ void k_amr_restrict_faces(
    FluidLevelGpu child,
    FluidLevelGpu parent,
    FluidConstantsGpu c
) {
    int r = child.refine_ratio;
    int nx_c = child.nx / r;
    int ny_c = child.ny / r;
    int nz_c = child.nz / r;

    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int k_c = blockIdx.z * blockDim.z + threadIdx.z;

    // Strictly interior X-faces (a in 1..nx_c)
    if (a >= 1 && a < nx_c && b < ny_c && k_c < nz_c) {
        int ic = parent.halo + child.off_x + a;
        int jc = parent.halo + child.off_y + b;
        int kc = parent.halo + child.off_z + k_c;
        int if_b = child.halo + a * r;
        int jf_b = child.halo + b * r;
        int kf_b = child.halo + k_c * r;

        double mom_sum = 0.0;
        double rho_sum = 0.0;
        for (int dk = 0; dk < r; ++dk) {
            for (int dj = 0; dj < r; ++dj) {
                int jf = jf_b + dj;
                int kf = kf_b + dk;
                double aL = (double)child.alpha[idx_cell(if_b - 1, jf, kf, child.nxp, child.nyp)];
                double aR = (double)child.alpha[idx_cell(if_b, jf, kf, child.nxp, child.nyp)];
                double rho_f = 0.5 * (mix_density(aL, c) + mix_density(aR, c));
                double uf = (double)child.ux[idx_fx(if_b, jf, kf, child.nxp, child.nyp)];
                mom_sum += rho_f * uf;
                rho_sum += rho_f;
            }
        }
        parent.ux[idx_fx(ic, jc, kc, parent.nxp, parent.nyp)] = (float)(mom_sum / rho_sum);
    }

    // Strictly interior Y-faces (b in 1..ny_c)
    if (a < nx_c && b >= 1 && b < ny_c && k_c < nz_c) {
        int ic = parent.halo + child.off_x + a;
        int jc = parent.halo + child.off_y + b;
        int kc = parent.halo + child.off_z + k_c;
        int if_b = child.halo + a * r;
        int jf_b = child.halo + b * r;
        int kf_b = child.halo + k_c * r;

        double mom_sum = 0.0;
        double rho_sum = 0.0;
        for (int dk = 0; dk < r; ++dk) {
            for (int di = 0; di < r; ++di) {
                int if_ = if_b + di;
                int kf = kf_b + dk;
                double aL = (double)child.alpha[idx_cell(if_, jf_b - 1, kf, child.nxp, child.nyp)];
                double aR = (double)child.alpha[idx_cell(if_, jf_b, kf, child.nxp, child.nyp)];
                double rho_f = 0.5 * (mix_density(aL, c) + mix_density(aR, c));
                double vf = (double)child.uy[idx_fy(if_, jf_b, kf, child.nxp, child.nyp)];
                mom_sum += rho_f * vf;
                rho_sum += rho_f;
            }
        }
        parent.uy[idx_fy(ic, jc, kc, parent.nxp, parent.nyp)] = (float)(mom_sum / rho_sum);
    }

    // Strictly interior Z-faces (k_c in 1..nz_c)
    if (a < nx_c && b < ny_c && k_c >= 1 && k_c < nz_c) {
        int ic = parent.halo + child.off_x + a;
        int jc = parent.halo + child.off_y + b;
        int kc = parent.halo + child.off_z + k_c;
        int if_b = child.halo + a * r;
        int jf_b = child.halo + b * r;
        int kf_b = child.halo + k_c * r;

        double mom_sum = 0.0;
        double rho_sum = 0.0;
        for (int dj = 0; dj < r; ++dj) {
            for (int di = 0; di < r; ++di) {
                int if_ = if_b + di;
                int jf = jf_b + dj;
                double aL = (double)child.alpha[idx_cell(if_, jf, kf_b - 1, child.nxp, child.nyp)];
                double aR = (double)child.alpha[idx_cell(if_, jf, kf_b, child.nxp, child.nyp)];
                double rho_f = 0.5 * (mix_density(aL, c) + mix_density(aR, c));
                double wf = (double)child.uz[idx_fz(if_, jf, kf_b, child.nxp, child.nyp)];
                mom_sum += rho_f * wf;
                rho_sum += rho_f;
            }
        }
        parent.uz[idx_fz(ic, jc, kc, parent.nxp, parent.nyp)] = (float)(mom_sum / rho_sum);
    }
}
