/// Fundamental SI physical constants for air, liquid water, and continuum fields.
pub const AIR_RHO: f64 = 1.204; // kg/m^3 at 20°C, 1 atm
pub const AIR_GAMMA: f64 = 1.4; // diatomic ideal gas adiabatic index
pub const AMBIENT_P0: f64 = 101_325.0; // Pa
pub const AIR_C: f64 = 343.25; // sqrt(1.4 * 101325 / 1.204) m/s
pub const AIR_MU: f64 = 1.81e-5; // Pa*s

pub const WATER_RHO: f64 = 998.2; // kg/m^3 at 20°C
pub const WATER_C: f64 = 1480.0; // m/s
pub const WATER_MU: f64 = 1.002e-3; // Pa*s
pub const WATER_SIGMA: f64 = 0.0728; // N/m surface tension at air-water interface

pub const GRAVITY_Z: f32 = -9.80665; // m/s^2
