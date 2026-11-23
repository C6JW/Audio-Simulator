#include "common.cuh"

__global__ void k_sample_probes(
    const FluidLevelGpu* levels,
    int num_levels,
    const MicProbeGpu* probes,
    int num_probes,
    float* out_samples,
    int sample_idx
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_probes) return;

    MicProbeGpu pr = probes[p];
    int lid = min(max(0, pr.level_idx), num_levels - 1);
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

    float val = 0.0f;
    for (int dk = 0; dk <= 1; ++dk) {
        float wz = dk ? tz : (1.0f - tz);
        for (int dj = 0; dj <= 1; ++dj) {
            float wy = dj ? ty : (1.0f - ty);
            for (int di = 0; di <= 1; ++di) {
                float wx = di ? tx : (1.0f - tx);
                size_t id = idx_cell(i0 + di, j0 + dj, k0 + dk, lvl.nxp, lvl.nyp);
                val += wx * wy * wz * lvl.p_gauge[id];
            }
        }
    }
    out_samples[(size_t)sample_idx * (size_t)num_probes + (size_t)p] = val;
}

__global__ void k_reset_metrics(SimReduceMetricsGpu* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out->nan_count = 0;
        out->c_max_sq = 0.0;
        out->u_max_sq = 0.0;
        out->acoustic_energy = 0.0;
        out->kinetic_energy = 0.0;
        out->water_volume = 0.0;
        out->subsurface_air_volume = 0.0;
        out->max_abs_p = 0.0;
        out->min_p = 1e30;
        out->max_p = -1e30;
        out->crater_min_z = 1e30f;
    }
}

__global__ void k_reduce_fluid_metrics(
    FluidLevelGpu lvl,
    FluidConstantsGpu c,
    SimReduceMetricsGpu* out
) {
    __shared__ unsigned long long s_nan[256];
    __shared__ double s_c2[256];
    __shared__ double s_u2[256];
    __shared__ double s_ea[256];
    __shared__ double s_ek[256];
    __shared__ double s_vw[256];
    __shared__ double s_pmax[256];
    __shared__ double s_pmin[256];

    int tid = threadIdx.x;
    size_t total = (size_t)lvl.nx * (size_t)lvl.ny * (size_t)lvl.nz;
    size_t gid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)tid;
    size_t stride = (size_t)gridDim.x * (size_t)blockDim.x;

    unsigned long long l_nan = 0;
    double l_c2 = 0.0;
    double l_u2 = 0.0;
    double l_ea = 0.0;
    double l_ek = 0.0;
    double l_vw = 0.0;
    double l_pmax = -1e30;
    double l_pmin = 1e30;

    double dV = (double)lvl.dx * (double)lvl.dx * (double)lvl.dx;
    int h = lvl.halo;

    for (size_t idx = gid; idx < total; idx += stride) {
        int i_int = (int)(idx % (size_t)lvl.nx);
        size_t tmp = idx / (size_t)lvl.nx;
        int j_int = (int)(tmp % (size_t)lvl.ny);
        int k_int = (int)(tmp / (size_t)lvl.ny);
        int i = h + i_int;
        int j = h + j_int;
        int k = h + k_int;

        size_t cid = idx_cell(i, j, k, lvl.nxp, lvl.nyp);
        double p_ac = (double)lvl.p_gauge[cid];
        double p_g = p_ac + (double)lvl.p_hydro[cid];
        double a = (double)lvl.alpha[cid];

        float ux0 = lvl.ux[idx_fx(i, j, k, lvl.nxp, lvl.nyp)];
        float ux1 = lvl.ux[idx_fx(i + 1, j, k, lvl.nxp, lvl.nyp)];
        float uy0 = lvl.uy[idx_fy(i, j, k, lvl.nxp, lvl.nyp)];
        float uy1 = lvl.uy[idx_fy(i, j + 1, k, lvl.nxp, lvl.nyp)];
        float uz0 = lvl.uz[idx_fz(i, j, k, lvl.nxp, lvl.nyp)];
        float uz1 = lvl.uz[idx_fz(i, j, k + 1, lvl.nxp, lvl.nyp)];

        if (!isfinite(p_g) || !isfinite(a) || !isfinite(ux0) || !isfinite(uy0) || !isfinite(uz0)) {
            ++l_nan;
            continue;
        }

        double uc = 0.5 * ((double)ux0 + (double)ux1);
        double vc = 0.5 * ((double)uy0 + (double)uy1);
        double wc = 0.5 * ((double)uz0 + (double)uz1);
        double u2 = uc * uc + vc * vc + wc * wc;

        double rho = mix_density(a, c);
        double k_mix = mix_bulk_modulus(a, p_g, c);
        double c2 = k_mix / rho;

        bool buried_solid =
            lvl.ib_beta_x[idx_fx(i, j, k, lvl.nxp, lvl.nyp)] >= 0.99f &&
            lvl.ib_beta_x[idx_fx(i + 1, j, k, lvl.nxp, lvl.nyp)] >= 0.99f &&
            lvl.ib_beta_y[idx_fy(i, j, k, lvl.nxp, lvl.nyp)] >= 0.99f &&
            lvl.ib_beta_y[idx_fy(i, j + 1, k, lvl.nxp, lvl.nyp)] >= 0.99f &&
            lvl.ib_beta_z[idx_fz(i, j, k, lvl.nxp, lvl.nyp)] >= 0.99f &&
            lvl.ib_beta_z[idx_fz(i, j, k + 1, lvl.nxp, lvl.nyp)] >= 0.99f;

        l_c2 = fmax(l_c2, c2);
        l_u2 = fmax(l_u2, u2);
        l_vw += a * dV;
        if (!buried_solid) {
            l_ea += 0.5 * (p_ac * p_ac / k_mix) * dV;
            l_ek += 0.5 * rho * u2 * dV;
            l_pmax = fmax(l_pmax, p_ac);
            l_pmin = fmin(l_pmin, p_ac);
        }
    }

    s_nan[tid] = l_nan;
    s_c2[tid] = l_c2;
    s_u2[tid] = l_u2;
    s_ea[tid] = l_ea;
    s_ek[tid] = l_ek;
    s_vw[tid] = l_vw;
    s_pmax[tid] = l_pmax;
    s_pmin[tid] = l_pmin;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_nan[tid] += s_nan[tid + s];
            s_c2[tid] = fmax(s_c2[tid], s_c2[tid + s]);
            s_u2[tid] = fmax(s_u2[tid], s_u2[tid + s]);
            s_ea[tid] += s_ea[tid + s];
            s_ek[tid] += s_ek[tid + s];
            s_vw[tid] += s_vw[tid + s];
            s_pmax[tid] = fmax(s_pmax[tid], s_pmax[tid + s]);
            s_pmin[tid] = fmin(s_pmin[tid], s_pmin[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd((unsigned long long*)&out->nan_count, s_nan[0]);
        atomicAdd_d(&out->acoustic_energy, s_ea[0]);
        atomicAdd_d(&out->kinetic_energy, s_ek[0]);
        atomicAdd_d(&out->water_volume, s_vw[0]);

        // Double max/min via CAS on out->c_max_sq etc.
        unsigned long long* addr_c2 = (unsigned long long*)&out->c_max_sq;
        unsigned long long old = *addr_c2, assumed;
        do {
            assumed = old;
            double v = __longlong_as_double(assumed);
            if (v >= s_c2[0]) break;
            old = atomicCAS(addr_c2, assumed, __double_as_longlong(s_c2[0]));
        } while (assumed != old);

        unsigned long long* addr_u2 = (unsigned long long*)&out->u_max_sq;
        old = *addr_u2;
        do {
            assumed = old;
            double v = __longlong_as_double(assumed);
            if (v >= s_u2[0]) break;
            old = atomicCAS(addr_u2, assumed, __double_as_longlong(s_u2[0]));
        } while (assumed != old);

        unsigned long long* addr_pmax = (unsigned long long*)&out->max_p;
        old = *addr_pmax;
        do {
            assumed = old;
            double v = __longlong_as_double(assumed);
            if (v >= s_pmax[0]) break;
            old = atomicCAS(addr_pmax, assumed, __double_as_longlong(s_pmax[0]));
        } while (assumed != old);

        unsigned long long* addr_pmin = (unsigned long long*)&out->min_p;
        old = *addr_pmin;
        do {
            assumed = old;
            double v = __longlong_as_double(assumed);
            if (v <= s_pmin[0]) break;
            old = atomicCAS(addr_pmin, assumed, __double_as_longlong(s_pmin[0]));
        } while (assumed != old);
    }
}
