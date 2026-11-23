#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Physical fluid constants passed to all fluid kernels
struct FluidConstantsGpu {
    double rho_a;       // Air density (kg/m^3), e.g., 1.204
    double rho_w;       // Water density (kg/m^3), e.g., 998.2
    double gamma_a;     // Air adiabatic index, e.g., 1.4
    double p0;          // Ambient reference pressure (Pa), e.g., 101325.0
    double c_w;         // Water sound speed (m/s), e.g., 1480.0
    double k_w;         // Water bulk modulus rho_w * c_w^2 (Pa)
    double sigma;       // Surface tension (N/m), e.g., 0.0728
    double mu_a;        // Air dynamic viscosity (Pa*s), e.g., 1.81e-5
    double mu_w;        // Water dynamic viscosity (Pa*s), e.g., 1.002e-3
    float gx;           // Gravity x (m/s^2)
    float gy;           // Gravity y (m/s^2)
    float gz;           // Gravity z (m/s^2), e.g., -9.80665
    float alpha_cfs;    // CFS-ADE PML shift (s^-1), e.g., 300.0
    float vof_beta;     // MTHINC sharpness beta, e.g., 1.8
    float pml_coeff;    // Dimensionless PML sigma_max * dx / c_local, e.g., 1.2
    float wall_admittance; // Normalized acoustic admittance rho*c/Z_n on static IB walls
    float div_damp_coeff;  // Artificial divergence viscosity coefficient (default 0.025)
    float bulk_viscosity_a; // Effective bulk viscosity in air (Pa*s) for atmospheric HF attenuation
    float prandtl_a;       // Prandtl number in air (~0.71)
};

// Staggered MAC fluid level descriptor (with halo = 2 on all 6 sides)
struct FluidLevelGpu {
    int32_t nx, ny, nz;             // Interior cell dimensions
    int32_t halo;                   // Ghost thickness (= 2)
    int32_t nxp, nyp, nzp;          // Padded cell dimensions (nx + 2*halo, etc.)
    float dx;                       // Cell spacing (m)
    float ox, oy, oz;               // Origin of interior cell (0,0,0) lower corner (m)
    int32_t level_idx;              // 0 = coarsest
    int32_t refine_ratio;           // Ratio to parent (1 for level 0; 2 or 4 for fine)
    int32_t pml_thickness;          // PML layers inside interior (0 for fine levels)
    uint32_t pml_face_mask;         // Bits 0..5: x_lo, x_hi, y_lo, y_hi, z_lo, z_hi
    int32_t off_x, off_y, off_z;    // Offset in parent interior cell coordinates

    // Cell-centered arrays: size = nxp * nyp * nzp
    float* p_gauge;         // Acoustic perturbation pressure p_ac = p - (p0 + p_hydro) (Pa), FP32 centered around 0.0f
    float* p_gauge_old;     // Saved pre-step acoustic perturbation pressure for AMR time interpolation (Pa)
    float* p_hydro;         // Static background hydrostatic gauge pressure p_hydro (Pa)
    float* alpha;
    float* alpha_tmp;
    float* alpha_smooth;
    float* kappa;
    float* div_u;
    float* src_p;

    // X-face arrays: size = (nxp + 1) * nyp * nzp
    float* ux;
    float* ux_old;
    float* a_slow_x;
    float* ib_beta_x;
    float* ib_beta_static_x;
    float* ib_vel_x;
    float* mflux_x;
    float* vflux_x;

    // Y-face arrays: size = nxp * (nyp + 1) * nzp
    float* uy;
    float* uy_old;
    float* a_slow_y;
    float* ib_beta_y;
    float* ib_beta_static_y;
    float* ib_vel_y;
    float* mflux_y;
    float* vflux_y;

    // Z-face arrays: size = nxp * nyp * (nzp + 1)
    float* uz;
    float* uz_old;
    float* a_slow_z;
    float* ib_beta_z;
    float* ib_beta_static_z;
    float* ib_vel_z;
    float* mflux_z;
    float* vflux_z;

    // CFS-ADE PML auxiliary variables (Level 0 only, nullptr if pml_thickness == 0)
    float* psi_ux;
    float* psi_uy;
    float* psi_uz;
    float* psi_px;
    float* psi_py;
    float* psi_pz;

    // AMR fine-boundary flux accumulators (for level_idx > 0, nullptr on level 0)
    // Stores integrated (u_n * dt_sub) across the 6 patch faces for refluxing
    float* bnd_vflux_xlo;
    float* bnd_vflux_xhi;
    float* bnd_vflux_ylo;
    float* bnd_vflux_yhi;
    float* bnd_vflux_zlo;
    float* bnd_vflux_zhi;

    // Stores integrated VOF water volume flux (F_alpha * dx) across the 6 patch faces
    float* bnd_aflux_xlo;
    float* bnd_aflux_xhi;
    float* bnd_aflux_ylo;
    float* bnd_aflux_yhi;
    float* bnd_aflux_zlo;
    float* bnd_aflux_zhi;

    // Conservative compressible HLLC Navier-Stokes fields (allocated when alloc_compressible == true)
    float* u_rho;
    float* u_mx;
    float* u_my;
    float* u_mz;
    float* u_ene;
    float* u1_rho;
    float* u1_mx;
    float* u1_my;
    float* u1_mz;
    float* u1_ene;
    float* cell_ib;
    float* cell_ib_static;
    float* cell_vx;
    float* cell_vy;
    float* cell_vz;
};

struct MicProbeGpu {
    float x, y, z;
    int32_t level_idx;      // Preferred finest level containing this probe
    int32_t is_hydrophone;  // Diagnostic tag (0 = airborne, 1 = submerged hydrophone)
    float _pad[3];
};

struct SimReduceMetricsGpu {
    uint64_t nan_count;
    double c_max_sq;
    double u_max_sq;
    double acoustic_energy;
    double kinetic_energy;
    double water_volume;
    double subsurface_air_volume;
    double max_abs_p;
    double min_p;
    double max_p;
    float crater_min_z;
    int32_t _pad;
};

struct MpmMetricsGpu {
    double kinetic_energy;
    double elastic_energy;
    double damage_sum;
    uint64_t nan_count;
};

// Material constitutive parameters (unified across stone, wood, metal, paper, wet leaves)
struct MaterialParamsGpu {
    float rho;          // Solid density (kg/m^3)
    float e1, e2, e3;   // Young's moduli along local material axes 1, 2, 3 (Pa)
    float nu12, nu23, nu13; // Poisson ratios
    float g12, g23, g13;    // Shear moduli (Pa)
    float eta_visc;     // Kelvin-Voigt viscoelastic damping (Pa*s)
    float sigma_y;      // Perzyna von Mises yield stress (Pa; <=0 means purely elastic)
    float tau_vp;       // Perzyna viscoplastic relaxation time (s)
    float g_c;          // Fracture toughness / critical energy release rate (J/m^2; <=0 means unbreakable)
    float l_c;          // Phase-field regularization length scale (m)
    float eta_frac;     // Phase-field mobility / viscosity (Pa*s)
    float thickness;    // Shell thickness (m; 0 for bulk 3D solid)
    float wetness;      // Surface liquid film fraction [0, 1]
    float contact_angle_cos; // cos(theta_c) for capillary bridge
    float friction_mu;  // Coulomb friction coefficient
    uint32_t flags;     // Bit 0: is_shell, Bit 1: is_orthotropic, Bit 2: is_rigid_static
    float sls_beta;     // Zener Standard Linear Solid modulus ratio (viscoelastic branch)
    float sls_tau;      // Zener SLS relaxation time constant (s)
    float c11, c22, c33; // Precomputed orthotropic normal stiffness matrix entries (Pa)
    float c12, c13, c23;
    float hygro_absorb_rate; // Moisture absorption rate from VOF alpha (1/s)
    float hygro_softening;   // Fractional stiffness reduction at wetness=1 [0, 1)
    float _pad_m2;
};

// Solid MPM particle state (SoA on GPU for coalesced memory access)
struct MpmParticlesGpu {
    int32_t count;
    int32_t _pad;
    float* px;
    float* py;
    float* pz;
    float* vx;
    float* vy;
    float* vz;
    // Affine velocity matrix C (3x3, row-major 9 arrays)
    float* c_mat;   // size 9 * count (component-major: comp * count + p)
    // Deformation gradient F (3x3, row-major 9 arrays)
    float* f_mat;   // size 9 * count
    // Initial material orientation Q0 (3x3, row-major 9 arrays)
    float* q0_mat;  // size 9 * count
    // Zener SLS viscoelastic internal strain q (symmetric 6 comps: xx, yy, zz, xy, yz, xz)
    float* q_visc;  // size 6 * count
    // Rest and current surface normals + effective IB area
    float* nx0;
    float* ny0;
    float* nz0;
    float* nx;
    float* ny;
    float* nz;
    float* area;
    float* mass;
    float* vol0;
    float* thickness;
    float* damage;      // Phase-field damage d in [0, 1]
    float* hist_h;      // Tensile strain energy history H
    float* gc_scale;    // Spatial heterogeneity multiplier around 1.0 (e.g. cellulose fibers)
    float* wetness;
    float* fx_ext;      // Fluid IB feedback force on particle (N)
    float* fy_ext;
    float* fz_ext;
    int32_t* mat_id;
    int32_t* body_id;
    uint32_t* flags;    // Bit 0: clamped_pos, Bit 1: prescribed_vel, Bit 2: is_surface_ib, Bit 3: is_shell
    float* v_presc_x;   // Prescribed velocity if Bit 1 set
    float* v_presc_y;
    float* v_presc_z;
    float* hyd_len;     // Potential-flow added-mass hydraulic length scale (m)
    float* fluid_rho;   // Live local surrounding fluid density sampled by IB gather (kg/m^3)
};

// Solid background MPM Eulerian grid
struct MpmGridGpu {
    int32_t nx, ny, nz;
    float dx;
    float ox, oy, oz;
    int32_t has_floor;
    float floor_z;
    float floor_mu;
    float _pad;
    float _pad2;
    float* mass;    // size 2 * n_nodes (field 0 = even body_id, field 1 = odd body_id)
    float* vx;
    float* vy;
    float* vz;
    float* fx;
    float* fy;
    float* fz;
    float* bc_w;    // Dirichlet clamp/prescribed velocity accumulator weight (2 * n_nodes)
    float* bc_vx;
    float* bc_vy;
    float* bc_vz;
    // Phase-field grid scalars (size n_nodes)
    float* d_field;
    float* d_new;
    float* h_field;
    float* gc_field;
    float* lc_field;
    float* eta_field;
    float* w_field;
};

// Discrete Kirchhoff-Love thin-shell triangle & hinge elements coupled to MPM particles
struct ShellMeshGpu {
    int32_t num_tris;
    int32_t num_hinges;
    // Triangle vertex indices (3 * num_tris)
    int32_t* tri_v0;
    int32_t* tri_v1;
    int32_t* tri_v2;
    // Reference 2D material metric inverse D_m^-1 (4 floats per tri: d00, d01, d10, d11) and rest area A0
    float* tri_dm_inv; // size 4 * num_tris
    float* tri_area0;
    float* tri_h;
    int32_t* tri_mat_id;
    // Hinge elements: edge (e0, e1) with opposite vertices (o0, o1)
    int32_t* hinge_e0;
    int32_t* hinge_e1;
    int32_t* hinge_o0;
    int32_t* hinge_o1;
    float* hinge_theta0; // Rest dihedral angle (rad)
    float* hinge_kb;     // Bending stiffness E*h^3 / (12*(1-nu^2)) * |e|^2 / A_e
    float* hinge_dmax;   // Tracked failure flag / damage
    int32_t* hinge_mat_id;
    // Per-particle force accumulator from shell membrane + bending + contact (3 * n_particles)
    float* p_fx;
    float* p_fy;
    float* p_fz;
};

#ifdef __cplusplus
}
#endif

// ---- Inline CUDA Device Helpers ----
#ifdef __CUDACC__

__device__ __forceinline__ size_t idx_cell(int i, int j, int k, int nxp, int nyp) {
    return (size_t)i + (size_t)nxp * ((size_t)j + (size_t)nyp * (size_t)k);
}

__device__ __forceinline__ size_t idx_fx(int i, int j, int k, int nxp, int nyp) {
    return (size_t)i + (size_t)(nxp + 1) * ((size_t)j + (size_t)nyp * (size_t)k);
}

__device__ __forceinline__ size_t idx_fy(int i, int j, int k, int nxp, int nyp) {
    return (size_t)i + (size_t)nxp * ((size_t)j + (size_t)(nyp + 1) * (size_t)k);
}

__device__ __forceinline__ size_t idx_fz(int i, int j, int k, int nxp, int nyp) {
    return (size_t)i + (size_t)nxp * ((size_t)j + (size_t)nyp * (size_t)k);
}

__device__ __forceinline__ double clamp_d(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

__device__ __forceinline__ float clamp_f(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

__device__ __forceinline__ double mix_density(double alpha, const FluidConstantsGpu& c) {
    double a = clamp_d(alpha, 0.0, 1.0);
    return a * c.rho_w + (1.0 - a) * c.rho_a;
}

__device__ __forceinline__ float mix_density_f(float alpha, const FluidConstantsGpu& c) {
    float a = clamp_f(alpha, 0.0f, 1.0f);
    return a * (float)c.rho_w + (1.0f - a) * (float)c.rho_a;
}

// Wood's exact harmonic bulk modulus in native FP32
__device__ __forceinline__ float mix_bulk_modulus_f(float alpha, float p_gauge, const FluidConstantsGpu& c) {
    float a = clamp_f(alpha, 0.0f, 1.0f);
    float p0_f = (float)c.p0;
    float p_abs = clamp_f(p0_f + p_gauge, 0.5f * p0_f, 2.0f * p0_f);
    float k_a = (float)c.gamma_a * p_abs;
    if (a <= 1e-6f) return k_a;
    float inv_k = a / (float)c.k_w + (1.0f - a) / k_a;
    return 1.0f / inv_k;
}

// Wood's exact harmonic bulk modulus in gauge/absolute pressure (for telemetry reductions)
__device__ __forceinline__ double mix_bulk_modulus(double alpha, double p_gauge, const FluidConstantsGpu& c) {
    return (double)mix_bulk_modulus_f((float)alpha, (float)p_gauge, c);
}

__device__ __forceinline__ double mix_sound_speed_sq(double alpha, double p_gauge, const FluidConstantsGpu& c) {
    double rho = mix_density(alpha, c);
    double k = mix_bulk_modulus(alpha, p_gauge, c);
    return k / rho;
}

__device__ __forceinline__ float minmod(float a, float b) {
    if (a * b <= 0.0f) return 0.0f;
    return fabsf(a) < fabsf(b) ? a : b;
}

__device__ __forceinline__ double atomicAdd_d(double* addr, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(addr, val);
#else
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

__device__ __forceinline__ float atomicMin_f(float* addr, float val) {
    int* addr_i = (int*)addr;
    int old = *addr_i, assumed;
    do {
        assumed = old;
        float f_old = __int_as_float(assumed);
        if (f_old <= val) return f_old;
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ float atomicMax_f(float* addr, float val) {
    int* addr_i = (int*)addr;
    int old = *addr_i, assumed;
    do {
        assumed = old;
        float f_old = __int_as_float(assumed);
        if (f_old >= val) return f_old;
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}

#endif // __CUDACC__
