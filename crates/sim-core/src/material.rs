use serde::{Deserialize, Serialize};
use sim_cuda::{FluidConstantsGpu, MaterialParamsGpu};

use crate::units::*;

#[inline]
fn default_div_damp_coeff() -> f32 {
    0.025
}

#[inline]
fn default_bulk_viscosity_a() -> f32 {
    0.003
}

#[inline]
fn default_prandtl_a() -> f32 {
    0.71
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FluidParams {
    pub rho_a: f64,
    pub rho_w: f64,
    pub gamma_a: f64,
    pub p0: f64,
    pub c_w: f64,
    pub sigma: f64,
    pub mu_a: f64,
    pub mu_w: f64,
    pub gravity: [f32; 3],
    pub alpha_cfs: f32,
    pub vof_beta: f32,
    pub pml_coeff: f32,
    #[serde(default)]
    pub wall_absorption_alpha: f32,
    #[serde(default = "default_div_damp_coeff")]
    pub div_damp_coeff: f32,
    #[serde(default = "default_bulk_viscosity_a")]
    pub bulk_viscosity_a: f32,
    #[serde(default = "default_prandtl_a")]
    pub prandtl_a: f32,
}

impl Default for FluidParams {
    fn default() -> Self {
        Self {
            rho_a: AIR_RHO,
            rho_w: WATER_RHO,
            gamma_a: AIR_GAMMA,
            p0: AMBIENT_P0,
            c_w: WATER_C,
            sigma: WATER_SIGMA,
            mu_a: AIR_MU,
            mu_w: WATER_MU,
            gravity: [0.0, 0.0, GRAVITY_Z],
            alpha_cfs: 300.0,
            vof_beta: 1.8,
            pml_coeff: 1.2,
            wall_absorption_alpha: 0.0,
            div_damp_coeff: 0.025,
            bulk_viscosity_a: 0.003,
            prandtl_a: 0.71,
        }
    }
}

impl FluidParams {
    pub fn to_gpu(&self) -> FluidConstantsGpu {
        let alpha = self.wall_absorption_alpha.clamp(0.0, 0.99);
        let r = (1.0 - alpha).sqrt();
        let wall_admittance = if alpha > 0.0 {
            (1.0 - r) / (1.0 + r)
        } else {
            0.0
        };
        FluidConstantsGpu {
            rho_a: self.rho_a,
            rho_w: self.rho_w,
            gamma_a: self.gamma_a,
            p0: self.p0,
            c_w: self.c_w,
            k_w: self.rho_w * self.c_w * self.c_w,
            sigma: self.sigma,
            mu_a: self.mu_a,
            mu_w: self.mu_w,
            gx: self.gravity[0],
            gy: self.gravity[1],
            gz: self.gravity[2],
            alpha_cfs: self.alpha_cfs,
            vof_beta: self.vof_beta,
            pml_coeff: self.pml_coeff,
            wall_admittance,
            div_damp_coeff: self.div_damp_coeff.max(0.0),
            bulk_viscosity_a: self.bulk_viscosity_a.max(0.0),
            prandtl_a: self.prandtl_a.max(0.1),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolidMaterial {
    pub name: String,
    pub rho: f32,
    pub e1: f32,
    pub e2: f32,
    pub e3: f32,
    pub nu12: f32,
    pub nu23: f32,
    pub nu13: f32,
    pub g12: f32,
    pub g23: f32,
    pub g13: f32,
    pub eta_visc: f32,
    pub sls_beta: f32,
    pub sls_tau: f32,
    pub sigma_y: f32,
    pub tau_vp: f32,
    pub g_c: f32,
    pub l_c: f32,
    pub eta_frac: f32,
    pub thickness: f32,
    pub wetness: f32,
    pub contact_angle_cos: f32,
    pub friction_mu: f32,
    pub is_shell: bool,
    pub is_orthotropic: bool,
    pub is_rigid_static: bool,
    #[serde(default)]
    pub hygro_absorb_rate: f32,
    #[serde(default)]
    pub hygro_softening: f32,
}

impl SolidMaterial {
    pub fn isotropic(name: impl Into<String>, rho: f32, e: f32, nu: f32, eta_visc: f32) -> Self {
        let g = e / (2.0 * (1.0 + nu));
        Self {
            name: name.into(),
            rho,
            e1: e,
            e2: e,
            e3: e,
            nu12: nu,
            nu23: nu,
            nu13: nu,
            g12: g,
            g23: g,
            g13: g,
            eta_visc,
            sls_beta: 0.0,
            sls_tau: 1.0e-4,
            sigma_y: 0.0,
            tau_vp: 0.0,
            g_c: 0.0,
            l_c: 0.0,
            eta_frac: 0.0,
            thickness: 0.0,
            wetness: 0.0,
            contact_angle_cos: 0.7,
            friction_mu: 0.4,
            is_shell: false,
            is_orthotropic: false,
            is_rigid_static: false,
            hygro_absorb_rate: 0.0,
            hygro_softening: 0.0,
        }
    }

    pub fn structural_steel() -> Self {
        // Low viscoelastic damping -> high Q-factor metallic ringing
        let mut m = Self::isotropic("structural_steel", 7850.0, 200.0e9, 0.30, 120.0);
        m.sls_beta = 0.0012;
        m.sls_tau = 1.0e-4;
        m
    }

    pub fn bell_bronze() -> Self {
        let mut m = Self::isotropic("bell_bronze", 8800.0, 105.0e9, 0.34, 25.0);
        m.sls_beta = 0.0008;
        m.sls_tau = 1.0e-4;
        m
    }

    pub fn crystal_glass() -> Self {
        let mut m = Self::isotropic("crystal_glass", 2500.0, 68.0e9, 0.22, 18.0);
        m.sls_beta = 0.0006;
        m.sls_tau = 1.0e-4;
        m
    }

    pub fn concrete() -> Self {
        let mut m = Self::isotropic("concrete", 2400.0, 32.0e9, 0.20, 800.0);
        m.sls_beta = 0.04;
        m.sls_tau = 2.0e-4;
        m
    }

    pub fn oak_wood() -> Self {
        // Orthotropic hardwood: stiff along grain (e1=13 GPa), softer radially/tangentially, strong SLS wood damping
        Self {
            name: "oak_wood".to_string(),
            rho: 720.0,
            e1: 13.0e9,
            e2: 1.6e9,
            e3: 0.9e9,
            nu12: 0.35,
            nu23: 0.45,
            nu13: 0.38,
            g12: 1.1e9,
            g23: 0.45e9,
            g13: 0.85e9,
            eta_visc: 400.0,
            sls_beta: 0.14,
            sls_tau: 1.2e-4,
            sigma_y: 0.0,
            tau_vp: 0.0,
            g_c: 0.0,
            l_c: 0.0,
            eta_frac: 0.0,
            thickness: 0.0,
            wetness: 0.0,
            contact_angle_cos: 0.7,
            friction_mu: 0.45,
            is_shell: false,
            is_orthotropic: true,
            is_rigid_static: false,
            hygro_absorb_rate: 0.0,
            hygro_softening: 0.0,
        }
    }

    pub fn granite_stone() -> Self {
        let mut m = Self::isotropic("granite_stone", 2650.0, 50.0e9, 0.25, 0.0);
        m.is_rigid_static = true;
        m
    }

    pub fn cellulose_paper(thickness: f32) -> Self {
        Self {
            name: "cellulose_paper".to_string(),
            rho: 800.0,
            e1: 4.5e9,
            e2: 2.0e9,
            e3: 2.0e9,
            nu12: 0.28,
            nu23: 0.28,
            nu13: 0.28,
            g12: 1.4e9,
            g23: 0.8e9,
            g13: 0.8e9,
            eta_visc: 200.0,
            sls_beta: 0.06,
            sls_tau: 1.2e-4,
            sigma_y: 0.0,
            tau_vp: 0.0,
            g_c: 700.0,
            l_c: 2.0e-3,
            eta_frac: 40.0,
            thickness,
            wetness: 0.0,
            contact_angle_cos: 0.5,
            friction_mu: 0.35,
            is_shell: true,
            is_orthotropic: true,
            is_rigid_static: false,
            hygro_absorb_rate: 0.0,
            hygro_softening: 0.0,
        }
    }

    pub fn wet_leaf(thickness: f32, wetness: f32) -> Self {
        Self {
            name: "wet_leaf".to_string(),
            rho: 650.0,
            e1: 0.8e9,
            e2: 0.4e9,
            e3: 0.4e9,
            nu12: 0.30,
            nu23: 0.30,
            nu13: 0.30,
            g12: 0.25e9,
            g23: 0.15e9,
            g13: 0.15e9,
            eta_visc: 300.0,
            sls_beta: 0.25,
            sls_tau: 8.0e-5,
            sigma_y: 6.0e6,
            tau_vp: 2.0e-4,
            g_c: 120.0,
            l_c: 2.5e-3,
            eta_frac: 30.0,
            thickness,
            wetness,
            contact_angle_cos: 0.85,
            friction_mu: 0.25,
            is_shell: true,
            is_orthotropic: true,
            is_rigid_static: false,
            hygro_absorb_rate: 0.0,
            hygro_softening: 0.85,
        }
    }

    pub fn autumn_leaf_dry(thickness: f32) -> Self {
        Self {
            name: "autumn_leaf_dry".to_string(),
            rho: 580.0,
            e1: 1.8e9,
            e2: 1.0e9,
            e3: 1.0e9,
            nu12: 0.25,
            nu23: 0.25,
            nu13: 0.25,
            g12: 0.65e9,
            g23: 0.40e9,
            g13: 0.40e9,
            eta_visc: 60.0,
            sls_beta: 0.06,
            sls_tau: 6.0e-5,
            sigma_y: 8.0e6,
            tau_vp: 2.0e-4,
            g_c: 180.0,
            l_c: 2.5e-3,
            eta_frac: 25.0,
            thickness,
            wetness: 0.0,
            contact_angle_cos: 0.6,
            friction_mu: 0.35,
            is_shell: true,
            is_orthotropic: true,
            is_rigid_static: false,
            hygro_absorb_rate: 1.4,
            hygro_softening: 0.85,
        }
    }

    /// Maximum longitudinal elastic wave speed c_p = sqrt(M * (1 + sls_beta) / rho) for CFL sizing.
    pub fn max_wave_speed(&self) -> f32 {
        if self.is_rigid_static {
            return 0.0;
        }
        let e_max = self.e1.max(self.e2).max(self.e3) * (1.0 + self.sls_beta.max(0.0));
        let nu = self.nu12.clamp(0.0, 0.47);
        let m_mod = e_max * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
        (m_mod / self.rho.max(1.0)).sqrt()
    }

    pub fn to_gpu(&self) -> MaterialParamsGpu {
        let mut flags = 0u32;
        if self.is_shell {
            flags |= 1 << 0;
        }
        if self.is_orthotropic {
            flags |= 1 << 1;
        }
        if self.is_rigid_static {
            flags |= 1 << 2;
        }

        // Precompute exact 3x3 normal orthotropic stiffness matrix in f64
        let e_ref = ((self.e1 as f64) + (self.e2 as f64) + (self.e3 as f64)) / 3.0;
        let s11 = e_ref / (self.e1 as f64).max(1.0);
        let s22 = e_ref / (self.e2 as f64).max(1.0);
        let s33 = e_ref / (self.e3 as f64).max(1.0);
        let s12 = -(self.nu12 as f64) * s11;
        let s23 = -(self.nu23 as f64) * s22;
        let s13 = -(self.nu13 as f64) * s11;

        let c00 = s22 * s33 - s23 * s23;
        let c01 = s13 * s23 - s12 * s33;
        let c02 = s12 * s23 - s13 * s22;
        let c11 = s11 * s33 - s13 * s13;
        let c12 = s12 * s13 - s11 * s23;
        let c22 = s11 * s22 - s12 * s12;
        let det = s11 * c00 + s12 * c01 + s13 * c02;
        let scale = e_ref / det.max(1e-12);

        MaterialParamsGpu {
            rho: self.rho,
            e1: self.e1,
            e2: self.e2,
            e3: self.e3,
            nu12: self.nu12,
            nu23: self.nu23,
            nu13: self.nu13,
            g12: self.g12,
            g23: self.g23,
            g13: self.g13,
            eta_visc: self.eta_visc,
            sigma_y: self.sigma_y,
            tau_vp: self.tau_vp,
            g_c: self.g_c,
            l_c: self.l_c,
            eta_frac: self.eta_frac,
            thickness: self.thickness,
            wetness: self.wetness,
            contact_angle_cos: self.contact_angle_cos,
            friction_mu: self.friction_mu,
            flags,
            sls_beta: self.sls_beta,
            sls_tau: self.sls_tau,
            c11: (c00 * scale) as f32,
            c22: (c11 * scale) as f32,
            c33: (c22 * scale) as f32,
            c12: (c01 * scale) as f32,
            c13: (c02 * scale) as f32,
            c23: (c12 * scale) as f32,
            hygro_absorb_rate: self.hygro_absorb_rate,
            hygro_softening: self.hygro_softening,
            _pad_m2: 0.0,
        }
    }
}
