use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sim_cuda::{MpmParticlesHost, ShellMeshHost};
use sim_io::ObjMesh;

use crate::scene::{SolidBodySpec, SolidBoundaryCondition};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GeometryShape {
    Box {
        min: [f32; 3],
        max: [f32; 3],
    },
    Sphere {
        center: [f32; 3],
        radius: f32,
    },
    HalfSpaceZ {
        z_max: f32,
    },
    CylindricalRing {
        center_xy: [f32; 2],
        r_inner: f32,
        r_outer: f32,
        z_min: f32,
        z_max: f32,
    },
    FrustumRing {
        center_xy: [f32; 2],
        r_inner_bottom: f32,
        r_inner_top: f32,
        thickness: f32,
        z_min: f32,
        z_max: f32,
    },
    AnnularDisc {
        center_xy: [f32; 2],
        r_inner: f32,
        r_outer: f32,
        z_min: f32,
        z_max: f32,
    },
    Disc {
        center_xy: [f32; 2],
        radius: f32,
        z_min: f32,
        z_max: f32,
    },
    CurvedShell {
        center: [f32; 3],
        size_x: f32,
        size_y: f32,
        nx: usize,
        ny: usize,
        dome_amp: f32,
        saddle_amp: f32,
        yaw_rad: f32,
        notch_x_len: f32,
        notch_y_pos: f32,
        notch_half_w: f32,
    },
    RevolutionShell {
        center_xy: [f32; 2],
        profile_zr: Vec<[f32; 2]>,
        n_theta: usize,
        n_meridian: usize,
        hyd_len: f32,
    },
    SquareJarShell {
        center_xy: [f32; 2],
        z_bottom: f32,
        body_half_width: f32,
        body_height: f32,
        shoulder_height: f32,
        neck_radius: f32,
        neck_height: f32,
        corner_radius: f32,
        n_perim: usize,
        n_height: usize,
        hyd_len: f32,
    },
    CanyonTerrain {
        floor_z: f32,
        valley_half_width_y: f32,
        wall_slope: f32,
        cliff_x: f32,
        cliff_height: f32,
    },
    CylindricalRingWithArch {
        center_xy: [f32; 2],
        r_inner: f32,
        r_outer: f32,
        z_min: f32,
        z_max: f32,
        arch_center_angle_rad: f32,
        arch_half_angle_rad: f32,
        arch_z_min: f32,
        arch_z_max: f32,
    },
    #[serde(skip)]
    Mesh(ObjMesh),
}

#[inline]
fn canyon_ground_z(
    x: f32,
    y: f32,
    floor_z: f32,
    valley_half_width_y: f32,
    wall_slope: f32,
    cliff_x: f32,
    cliff_height: f32,
) -> f32 {
    let dy = (y.abs() - valley_half_width_y).max(0.0);
    let wall = (dy * wall_slope).min(60.0);
    let cliff = if cliff_x > 0.0 && x > cliff_x {
        cliff_height * ((x - cliff_x) / 25.0).clamp(0.0, 1.0)
    } else {
        0.0
    };
    floor_z + wall + cliff
}

#[inline]
fn in_arch_cutout(
    dx: f32,
    dy: f32,
    z: f32,
    arch_center_angle_rad: f32,
    arch_half_angle_rad: f32,
    arch_z_min: f32,
    arch_z_max: f32,
) -> bool {
    if z < arch_z_min || z > arch_z_max {
        return false;
    }
    let arch_mid = 0.5 * (arch_z_min + arch_z_max);
    let eff_half = if z > arch_mid {
        let top_h = (arch_z_max - arch_mid).max(1e-5);
        let u = (z - arch_mid) / top_h;
        arch_half_angle_rad * (1.0 - u * u).max(0.0).sqrt()
    } else {
        arch_half_angle_rad
    };
    let ang = dy.atan2(dx);
    let mut diff = ang - arch_center_angle_rad;
    let pi = std::f32::consts::PI;
    while diff > pi {
        diff -= 2.0 * pi;
    }
    while diff < -pi {
        diff += 2.0 * pi;
    }
    diff.abs() <= eff_half
}

impl GeometryShape {
    pub fn contains(&self, p: [f32; 3]) -> bool {
        match self {
            GeometryShape::Box { min, max } => {
                p[0] >= min[0]
                    && p[0] <= max[0]
                    && p[1] >= min[1]
                    && p[1] <= max[1]
                    && p[2] >= min[2]
                    && p[2] <= max[2]
            }
            GeometryShape::Sphere { center, radius } => {
                let dx = p[0] - center[0];
                let dy = p[1] - center[1];
                let dz = p[2] - center[2];
                dx * dx + dy * dy + dz * dz <= radius * radius
            }
            GeometryShape::HalfSpaceZ { z_max } => p[2] <= *z_max,
            GeometryShape::CylindricalRing {
                center_xy,
                r_inner,
                r_outer,
                z_min,
                z_max,
            }
            | GeometryShape::AnnularDisc {
                center_xy,
                r_inner,
                r_outer,
                z_min,
                z_max,
            } => {
                if p[2] < *z_min || p[2] > *z_max {
                    return false;
                }
                let dx = p[0] - center_xy[0];
                let dy = p[1] - center_xy[1];
                let r2 = dx * dx + dy * dy;
                r2 >= r_inner * r_inner && r2 <= r_outer * r_outer
            }
            GeometryShape::FrustumRing {
                center_xy,
                r_inner_bottom,
                r_inner_top,
                thickness,
                z_min,
                z_max,
            } => {
                if p[2] < *z_min || p[2] > *z_max {
                    return false;
                }
                let dz = (z_max - z_min).max(1e-6);
                let t = ((p[2] - z_min) / dz).clamp(0.0, 1.0);
                let ri = r_inner_bottom + t * (r_inner_top - r_inner_bottom);
                let ro = ri + thickness;
                let dx = p[0] - center_xy[0];
                let dy = p[1] - center_xy[1];
                let r2 = dx * dx + dy * dy;
                r2 >= ri * ri && r2 <= ro * ro
            }
            GeometryShape::Disc {
                center_xy,
                radius,
                z_min,
                z_max,
            } => {
                if p[2] < *z_min || p[2] > *z_max {
                    return false;
                }
                let dx = p[0] - center_xy[0];
                let dy = p[1] - center_xy[1];
                dx * dx + dy * dy <= radius * radius
            }
            GeometryShape::CurvedShell { .. }
            | GeometryShape::RevolutionShell { .. }
            | GeometryShape::SquareJarShell { .. } => false,
            GeometryShape::CanyonTerrain {
                floor_z,
                valley_half_width_y,
                wall_slope,
                cliff_x,
                cliff_height,
            } => {
                p[2] <= canyon_ground_z(
                    p[0],
                    p[1],
                    *floor_z,
                    *valley_half_width_y,
                    *wall_slope,
                    *cliff_x,
                    *cliff_height,
                )
            }
            GeometryShape::CylindricalRingWithArch {
                center_xy,
                r_inner,
                r_outer,
                z_min,
                z_max,
                arch_center_angle_rad,
                arch_half_angle_rad,
                arch_z_min,
                arch_z_max,
            } => {
                if p[2] < *z_min || p[2] > *z_max {
                    return false;
                }
                let dx = p[0] - center_xy[0];
                let dy = p[1] - center_xy[1];
                let r2 = dx * dx + dy * dy;
                if r2 < r_inner * r_inner || r2 > r_outer * r_outer {
                    return false;
                }
                !in_arch_cutout(
                    dx,
                    dy,
                    p[2],
                    *arch_center_angle_rad,
                    *arch_half_angle_rad,
                    *arch_z_min,
                    *arch_z_max,
                )
            }
            GeometryShape::Mesh(m) => m.contains_point(p),
        }
    }

    /// Exact analytic signed distance lower bound / fast reject for 4x4x4 quadrature speedup.
    pub fn subcell_volume_fraction(&self, center: [f32; 3], dx: f32) -> f32 {
        let half_diag = 0.8660254 * dx;
        match self {
            GeometryShape::HalfSpaceZ { z_max } => {
                let z0 = center[2] - 0.5 * dx;
                let z1 = center[2] + 0.5 * dx;
                if z1 <= *z_max {
                    return 1.0;
                }
                if z0 >= *z_max {
                    return 0.0;
                }
                ((*z_max - z0) / dx).clamp(0.0, 1.0)
            }
            GeometryShape::Sphere { center: c, radius } => {
                let dist = ((center[0] - c[0]).powi(2)
                    + (center[1] - c[1]).powi(2)
                    + (center[2] - c[2]).powi(2))
                .sqrt();
                if dist <= radius - half_diag {
                    return 1.0;
                }
                if dist >= radius + half_diag {
                    return 0.0;
                }
                self.quadrature_4x4x4(center, dx)
            }
            GeometryShape::Box { min, max } => {
                let mut vol = 1.0_f32;
                for d in 0..3 {
                    let lo = (center[d] - 0.5 * dx).max(min[d]);
                    let hi = (center[d] + 0.5 * dx).min(max[d]);
                    if hi <= lo {
                        return 0.0;
                    }
                    vol *= (hi - lo) / dx;
                }
                vol
            }
            GeometryShape::CylindricalRing {
                center_xy,
                r_inner,
                r_outer,
                z_min,
                z_max,
            }
            | GeometryShape::AnnularDisc {
                center_xy,
                r_inner,
                r_outer,
                z_min,
                z_max,
            } => {
                if center[2] + 0.5 * dx < *z_min || center[2] - 0.5 * dx > *z_max {
                    return 0.0;
                }
                let r = ((center[0] - center_xy[0]).powi(2) + (center[1] - center_xy[1]).powi(2))
                    .sqrt();
                if r + half_diag < *r_inner || r - half_diag > *r_outer {
                    return 0.0;
                }
                if r - half_diag >= *r_inner
                    && r + half_diag <= *r_outer
                    && center[2] - 0.5 * dx >= *z_min
                    && center[2] + 0.5 * dx <= *z_max
                {
                    return 1.0;
                }
                self.quadrature_4x4x4(center, dx)
            }
            GeometryShape::FrustumRing {
                center_xy,
                r_inner_bottom,
                r_inner_top,
                thickness,
                z_min,
                z_max,
            } => {
                if center[2] + 0.5 * dx < *z_min || center[2] - 0.5 * dx > *z_max {
                    return 0.0;
                }
                let r = ((center[0] - center_xy[0]).powi(2) + (center[1] - center_xy[1]).powi(2))
                    .sqrt();
                let ri_min = r_inner_bottom.min(*r_inner_top);
                let ro_max = r_inner_bottom.max(*r_inner_top) + *thickness;
                if r + half_diag < ri_min || r - half_diag > ro_max {
                    return 0.0;
                }
                let ri_max = r_inner_bottom.max(*r_inner_top);
                let ro_min = r_inner_bottom.min(*r_inner_top) + *thickness;
                if r - half_diag >= ri_max
                    && r + half_diag <= ro_min
                    && center[2] - 0.5 * dx >= *z_min
                    && center[2] + 0.5 * dx <= *z_max
                {
                    return 1.0;
                }
                self.quadrature_4x4x4(center, dx)
            }
            GeometryShape::Disc {
                center_xy,
                radius,
                z_min,
                z_max,
            } => {
                if center[2] + 0.5 * dx < *z_min || center[2] - 0.5 * dx > *z_max {
                    return 0.0;
                }
                let r = ((center[0] - center_xy[0]).powi(2) + (center[1] - center_xy[1]).powi(2))
                    .sqrt();
                if r - half_diag > *radius {
                    return 0.0;
                }
                if r + half_diag <= *radius
                    && center[2] - 0.5 * dx >= *z_min
                    && center[2] + 0.5 * dx <= *z_max
                {
                    return 1.0;
                }
                self.quadrature_4x4x4(center, dx)
            }
            GeometryShape::CurvedShell { .. }
            | GeometryShape::RevolutionShell { .. }
            | GeometryShape::SquareJarShell { .. } => 0.0,
            GeometryShape::CanyonTerrain {
                floor_z,
                valley_half_width_y,
                wall_slope,
                cliff_x,
                cliff_height,
            } => {
                let half_dx = 0.5 * dx;
                if center[1].abs() + half_dx <= *valley_half_width_y
                    && (*cliff_x <= 0.0 || center[0] + half_dx <= *cliff_x)
                {
                    let z0 = center[2] - half_dx;
                    let z1 = center[2] + half_dx;
                    if z1 <= *floor_z {
                        return 1.0;
                    }
                    if z0 >= *floor_z {
                        return 0.0;
                    }
                    return ((*floor_z - z0) / dx).clamp(0.0, 1.0);
                }
                let gz = canyon_ground_z(
                    center[0],
                    center[1],
                    *floor_z,
                    *valley_half_width_y,
                    *wall_slope,
                    *cliff_x,
                    *cliff_height,
                );
                let margin = half_diag * (1.0 + *wall_slope + 2.0);
                if center[2] + half_dx <= gz - margin {
                    return 1.0;
                }
                if center[2] - half_dx >= gz + margin {
                    return 0.0;
                }
                self.quadrature_4x4x4(center, dx)
            }
            GeometryShape::CylindricalRingWithArch {
                center_xy,
                r_inner,
                r_outer,
                z_min,
                z_max,
                ..
            } => {
                if center[2] + 0.5 * dx < *z_min || center[2] - 0.5 * dx > *z_max {
                    return 0.0;
                }
                let r = ((center[0] - center_xy[0]).powi(2) + (center[1] - center_xy[1]).powi(2))
                    .sqrt();
                if r + half_diag < *r_inner || r - half_diag > *r_outer {
                    return 0.0;
                }
                self.quadrature_4x4x4(center, dx)
            }
            GeometryShape::Mesh(_) => self.quadrature_4x4x4(center, dx),
        }
    }

    fn quadrature_4x4x4(&self, center: [f32; 3], dx: f32) -> f32 {
        let mut count = 0u32;
        let base_x = center[0] - 0.5 * dx;
        let base_y = center[1] - 0.5 * dx;
        let base_z = center[2] - 0.5 * dx;
        for kz in 0..4 {
            let z = base_z + (kz as f32 + 0.5) * 0.25 * dx;
            for ky in 0..4 {
                let y = base_y + (ky as f32 + 0.5) * 0.25 * dx;
                for kx in 0..4 {
                    let x = base_x + (kx as f32 + 0.5) * 0.25 * dx;
                    if self.contains([x, y, z]) {
                        count += 1;
                    }
                }
            }
        }
        (count as f32) / 64.0
    }

    pub fn bounding_box(&self) -> ([f32; 3], [f32; 3]) {
        match self {
            GeometryShape::Box { min, max } => (*min, *max),
            GeometryShape::Sphere { center, radius } => (
                [center[0] - radius, center[1] - radius, center[2] - radius],
                [center[0] + radius, center[1] + radius, center[2] + radius],
            ),
            GeometryShape::HalfSpaceZ { z_max } => (
                [-1e3, -1e3, -1e3],
                [1e3, 1e3, *z_max],
            ),
            GeometryShape::CylindricalRing {
                center_xy,
                r_outer,
                z_min,
                z_max,
                ..
            }
            | GeometryShape::AnnularDisc {
                center_xy,
                r_outer,
                z_min,
                z_max,
                ..
            } => (
                [center_xy[0] - r_outer, center_xy[1] - r_outer, *z_min],
                [center_xy[0] + r_outer, center_xy[1] + r_outer, *z_max],
            ),
            GeometryShape::FrustumRing {
                center_xy,
                r_inner_bottom,
                r_inner_top,
                thickness,
                z_min,
                z_max,
            } => {
                let r_outer = r_inner_bottom.max(*r_inner_top) + *thickness;
                (
                    [center_xy[0] - r_outer, center_xy[1] - r_outer, *z_min],
                    [center_xy[0] + r_outer, center_xy[1] + r_outer, *z_max],
                )
            }
            GeometryShape::Disc {
                center_xy,
                radius,
                z_min,
                z_max,
            } => (
                [center_xy[0] - radius, center_xy[1] - radius, *z_min],
                [center_xy[0] + radius, center_xy[1] + radius, *z_max],
            ),
            GeometryShape::CurvedShell {
                center,
                size_x,
                size_y,
                dome_amp,
                saddle_amp,
                ..
            } => {
                let r = 0.5 * (size_x * size_x + size_y * size_y).sqrt() + 1e-3;
                let z_pad = dome_amp.abs() + saddle_amp.abs() + 2e-3;
                (
                    [center[0] - r, center[1] - r, center[2] - z_pad],
                    [center[0] + r, center[1] + r, center[2] + z_pad],
                )
            }
            GeometryShape::RevolutionShell {
                center_xy,
                profile_zr,
                ..
            } => {
                let mut r_max = 0.0_f32;
                let mut z_min = f32::INFINITY;
                let mut z_max = f32::NEG_INFINITY;
                for &[z, r] in profile_zr {
                    r_max = r_max.max(r);
                    z_min = z_min.min(z);
                    z_max = z_max.max(z);
                }
                if !z_min.is_finite() {
                    ([0.0; 3], [0.0; 3])
                } else {
                    (
                        [center_xy[0] - r_max, center_xy[1] - r_max, z_min],
                        [center_xy[0] + r_max, center_xy[1] + r_max, z_max],
                    )
                }
            }
            GeometryShape::SquareJarShell {
                center_xy,
                z_bottom,
                body_half_width,
                body_height,
                shoulder_height,
                neck_radius,
                neck_height,
                ..
            } => {
                let r_max = body_half_width.max(*neck_radius);
                let z_top = z_bottom + body_height + shoulder_height + neck_height;
                (
                    [center_xy[0] - r_max, center_xy[1] - r_max, *z_bottom],
                    [center_xy[0] + r_max, center_xy[1] + r_max, z_top],
                )
            }
            GeometryShape::CanyonTerrain {
                floor_z,
                cliff_height,
                ..
            } => (
                [-2e3, -2e3, -1e2],
                [2e3, 2e3, *floor_z + cliff_height.max(60.0)],
            ),
            GeometryShape::CylindricalRingWithArch {
                center_xy,
                r_outer,
                z_min,
                z_max,
                ..
            } => (
                [center_xy[0] - r_outer, center_xy[1] - r_outer, *z_min],
                [center_xy[0] + r_outer, center_xy[1] + r_outer, *z_max],
            ),
            GeometryShape::Mesh(m) => m.bounding_box(),
        }
    }

    /// Outward surface normal estimate via finite difference of indicator / analytic formula.
    pub fn surface_normal(&self, p: [f32; 3]) -> [f32; 3] {
        match self {
            GeometryShape::Sphere { center, .. } => {
                normalize3([p[0] - center[0], p[1] - center[1], p[2] - center[2]])
            }
            GeometryShape::Box { min, max } => {
                let d_x0 = (p[0] - min[0]).abs();
                let d_x1 = (max[0] - p[0]).abs();
                let d_y0 = (p[1] - min[1]).abs();
                let d_y1 = (max[1] - p[1]).abs();
                let d_z0 = (p[2] - min[2]).abs();
                let d_z1 = (max[2] - p[2]).abs();
                let m = d_x0.min(d_x1).min(d_y0).min(d_y1).min(d_z0).min(d_z1);
                if (m - d_x0).abs() < 1e-9 {
                    [-1.0, 0.0, 0.0]
                } else if (m - d_x1).abs() < 1e-9 {
                    [1.0, 0.0, 0.0]
                } else if (m - d_y0).abs() < 1e-9 {
                    [0.0, -1.0, 0.0]
                } else if (m - d_y1).abs() < 1e-9 {
                    [0.0, 1.0, 0.0]
                } else if (m - d_z0).abs() < 1e-9 {
                    [0.0, 0.0, -1.0]
                } else {
                    [0.0, 0.0, 1.0]
                }
            }
            GeometryShape::CylindricalRing {
                center_xy,
                r_inner,
                r_outer,
                z_min,
                z_max,
            }
            | GeometryShape::AnnularDisc {
                center_xy,
                r_inner,
                r_outer,
                z_min,
                z_max,
            } => {
                let dx = p[0] - center_xy[0];
                let dy = p[1] - center_xy[1];
                let r = (dx * dx + dy * dy).sqrt().max(1e-9);
                let d_in = (r - r_inner).abs();
                let d_out = (r_outer - r).abs();
                let d_zlo = (p[2] - z_min).abs();
                let d_zhi = (z_max - p[2]).abs();
                let m = d_in.min(d_out).min(d_zlo).min(d_zhi);
                if (m - d_in).abs() < 1e-9 {
                    [-dx / r, -dy / r, 0.0]
                } else if (m - d_out).abs() < 1e-9 {
                    [dx / r, dy / r, 0.0]
                } else if (m - d_zlo).abs() < 1e-9 {
                    [0.0, 0.0, -1.0]
                } else {
                    [0.0, 0.0, 1.0]
                }
            }
            GeometryShape::FrustumRing {
                center_xy,
                r_inner_bottom,
                r_inner_top,
                thickness,
                z_min,
                z_max,
            } => {
                let dx = p[0] - center_xy[0];
                let dy = p[1] - center_xy[1];
                let r = (dx * dx + dy * dy).sqrt().max(1e-9);
                let dz = (z_max - z_min).max(1e-6);
                let t = ((p[2] - z_min) / dz).clamp(0.0, 1.0);
                let slope = (r_inner_top - r_inner_bottom) / dz;
                let ri = r_inner_bottom + t * (r_inner_top - r_inner_bottom);
                let ro = ri + thickness;
                let d_in = (r - ri).abs();
                let d_out = (ro - r).abs();
                let d_zlo = (p[2] - z_min).abs();
                let d_zhi = (z_max - p[2]).abs();
                let m = d_in.min(d_out).min(d_zlo).min(d_zhi);
                if (m - d_in).abs() < 1e-9 {
                    normalize3([-dx / r, -dy / r, slope])
                } else if (m - d_out).abs() < 1e-9 {
                    normalize3([dx / r, dy / r, -slope])
                } else if (m - d_zlo).abs() < 1e-9 {
                    [0.0, 0.0, -1.0]
                } else {
                    [0.0, 0.0, 1.0]
                }
            }
            GeometryShape::CylindricalRingWithArch {
                center_xy,
                r_inner,
                r_outer,
                z_min,
                z_max,
                ..
            } => {
                let dx = p[0] - center_xy[0];
                let dy = p[1] - center_xy[1];
                let r = (dx * dx + dy * dy).sqrt().max(1e-9);
                let d_in = (r - r_inner).abs();
                let d_out = (r_outer - r).abs();
                let d_zlo = (p[2] - z_min).abs();
                let d_zhi = (z_max - p[2]).abs();
                let m = d_in.min(d_out).min(d_zlo).min(d_zhi);
                if (m - d_in).abs() < 1e-9 {
                    [-dx / r, -dy / r, 0.0]
                } else if (m - d_out).abs() < 1e-9 {
                    [dx / r, dy / r, 0.0]
                } else if (m - d_zlo).abs() < 1e-9 {
                    [0.0, 0.0, -1.0]
                } else {
                    [0.0, 0.0, 1.0]
                }
            }
            _ => [0.0, 0.0, 1.0],
        }
    }
}

#[inline]
pub fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 1e-12 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [0.0, 0.0, 1.0]
    }
}

pub fn euler_to_mat3(euler_xyz: [f32; 3]) -> [f32; 9] {
    let (sx, cx) = euler_xyz[0].sin_cos();
    let (sy, cy) = euler_xyz[1].sin_cos();
    let (sz, cz) = euler_xyz[2].sin_cos();
    // R = Rz * Ry * Rx
    [
        cz * cy,
        cz * sy * sx - sz * cx,
        cz * sy * cx + sz * sx,
        sz * cy,
        sz * sy * sx + cz * cx,
        sz * sy * cx - cz * sx,
        -sy,
        cy * sx,
        cy * cx,
    ]
}

#[inline]
fn point_in_box(p: [f32; 3], min: [f32; 3], max: [f32; 3]) -> bool {
    p[0] >= min[0]
        && p[0] <= max[0]
        && p[1] >= min[1]
        && p[1] <= max[1]
        && p[2] >= min[2]
        && p[2] <= max[2]
}

#[inline]
fn splitmix_signed(mut x: u64) -> f32 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let u = ((x >> 40) as f32) / 16_777_216.0_f32; // [0, 1)
    2.0 * u - 1.0
}

pub fn sample_bulk_solid_into(
    body: &SolidBodySpec,
    mat_id: i32,
    host: &mut MpmParticlesHost,
    q0_list: &mut Vec<[f32; 9]>,
) {
    let q0 = euler_to_mat3(body.orientation_euler_xyz);
    let dp = body.particle_spacing.max(1e-5);
    let (bmin, bmax) = body.shape.bounding_box();

    let (nx, ny, nz, sx, sy, sz) = match &body.shape {
        GeometryShape::Box { min, max } => {
            let nx = ((max[0] - min[0]) / dp).round().max(1.0) as usize;
            let ny = ((max[1] - min[1]) / dp).round().max(1.0) as usize;
            let nz = ((max[2] - min[2]) / dp).round().max(1.0) as usize;
            (
                nx,
                ny,
                nz,
                (max[0] - min[0]) / (nx as f32),
                (max[1] - min[1]) / (ny as f32),
                (max[2] - min[2]) / (nz as f32),
            )
        }
        _ => {
            let nx = ((bmax[0] - bmin[0]) / dp).ceil().max(1.0) as usize;
            let ny = ((bmax[1] - bmin[1]) / dp).ceil().max(1.0) as usize;
            let nz = ((bmax[2] - bmin[2]) / dp).ceil().max(1.0) as usize;
            (nx, ny, nz, dp, dp, dp)
        }
    };

    let vol0 = sx * sy * sz;
    let mass = body.material.rho * vol0;
    let area = (sx * sy).max(sy * sz).max(sx * sz);

    for k in 0..nz {
        let z = bmin[2] + (k as f32 + 0.5) * sz;
        for j in 0..ny {
            let y = bmin[1] + (j as f32 + 0.5) * sy;
            for i in 0..nx {
                let x = bmin[0] + (i as f32 + 0.5) * sx;
                let p = [x, y, z];
                if !body.shape.contains(p) {
                    continue;
                }

                // Surface check against 6 face neighbors
                let offsets = [
                    [sx, 0.0, 0.0],
                    [-sx, 0.0, 0.0],
                    [0.0, sy, 0.0],
                    [0.0, -sy, 0.0],
                    [0.0, 0.0, sz],
                    [0.0, 0.0, -sz],
                ];
                let mut is_surf = false;
                let mut n_est = [0.0_f32; 3];
                for o in &offsets {
                    if !body.shape.contains([p[0] + o[0], p[1] + o[1], p[2] + o[2]]) {
                        is_surf = true;
                        n_est[0] += o[0];
                        n_est[1] += o[1];
                        n_est[2] += o[2];
                    }
                }
                let nrm = if is_surf {
                    match &body.shape {
                        GeometryShape::Mesh(_) => normalize3(n_est),
                        _ => body.shape.surface_normal(p),
                    }
                } else {
                    [0.0, 0.0, 1.0]
                };

                let mut flag = 0u32;
                if is_surf || body.is_kinematic_rigid {
                    flag |= 1 << 2; // is_surface_ib
                }
                let mut v_presc = [0.0_f32; 3];
                if body.is_kinematic_rigid {
                    flag |= 1 << 1;
                    v_presc = body.initial_velocity;
                }
                for bc in &body.bcs {
                    match bc {
                        SolidBoundaryCondition::ClampedRegion { min, max } => {
                            if point_in_box(p, *min, *max) {
                                flag |= 1 << 0;
                            }
                        }
                        SolidBoundaryCondition::PrescribedVelocityRegion { min, max, velocity } => {
                            if point_in_box(p, *min, *max) {
                                flag |= 1 << 1;
                                v_presc = *velocity;
                            }
                        }
                    }
                }

                let idx = host.px.len() as u64;
                let gc_s = 1.0 + body.gc_heterogeneity_amp * splitmix_signed(idx ^ 0xA5A5_5A5A);

                host.px.push(x);
                host.py.push(y);
                host.pz.push(z);
                host.vx.push(if (flag & 1) != 0 { 0.0 } else { body.initial_velocity[0] });
                host.vy.push(if (flag & 1) != 0 { 0.0 } else { body.initial_velocity[1] });
                host.vz.push(if (flag & 1) != 0 { 0.0 } else { body.initial_velocity[2] });
                host.nx0.push(nrm[0]);
                host.ny0.push(nrm[1]);
                host.nz0.push(nrm[2]);
                host.nx.push(nrm[0]);
                host.ny.push(nrm[1]);
                host.nz.push(nrm[2]);
                host.area.push(area);
                host.mass.push(mass);
                host.vol0.push(vol0);
                host.thickness.push(0.0);
                host.damage.push(0.0);
                host.hist_h.push(0.0);
                host.gc_scale.push(gc_s);
                host.wetness.push(body.material.wetness);
                host.mat_id.push(mat_id);
                host.body_id.push(body.body_id);
                host.flags.push(flag);
                host.v_presc_x.push(v_presc[0]);
                host.v_presc_y.push(v_presc[1]);
                host.v_presc_z.push(v_presc[2]);
                host.hyd_len.push(0.0);
                host.fluid_rho.push(1.204);
                q0_list.push(q0);
            }
        }
    }
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[allow(clippy::too_many_arguments)]
fn finalize_shell_surface_into(
    body: &SolidBodySpec,
    mat_id: i32,
    world_pos: &[[f32; 3]],
    tris_raw: &[[usize; 3]],
    tri_fiber_dir: &[[f32; 3]],
    nom_cell_area: f32,
    hyd_len: f32,
    q0_rot: [f32; 9],
    host: &mut MpmParticlesHost,
    q0_list: &mut Vec<[f32; 9]>,
    shell: &mut ShellMeshHost,
) {
    let nv = world_pos.len();
    let thickness = body.material.thickness.max(1e-5);
    let base_idx = host.px.len();
    let mut vert_area = vec![0.0_f32; nv];
    let mut vert_norm = vec![[0.0_f32; 3]; nv];
    let mut edge_map: BTreeMap<(usize, usize), (Option<(usize, f32)>, Option<(usize, f32)>)> =
        BTreeMap::new();

    for (ti, &[i0, i1, i2]) in tris_raw.iter().enumerate() {
        let x0 = world_pos[i0];
        let x1 = world_pos[i1];
        let x2 = world_pos[i2];
        let e1 = sub3(x1, x0);
        let e2 = sub3(x2, x0);
        let fnrm = cross3(e1, e2);
        let fn_len = dot3(fnrm, fnrm).sqrt();
        if fn_len < 1e-14 {
            continue;
        }
        let area0 = 0.5 * fn_len;
        let n_tri = [fnrm[0] / fn_len, fnrm[1] / fn_len, fnrm[2] / fn_len];

        let fdir = tri_fiber_dir[ti];
        let proj = dot3(fdir, n_tri);
        let t1_raw = [
            fdir[0] - proj * n_tri[0],
            fdir[1] - proj * n_tri[1],
            fdir[2] - proj * n_tri[2],
        ];
        let t1 = if dot3(t1_raw, t1_raw) > 1e-10 {
            normalize3(t1_raw)
        } else {
            normalize3(e1)
        };
        let t2 = cross3(n_tri, t1);

        let dm00 = dot3(e1, t1);
        let dm01 = dot3(e2, t1);
        let dm10 = dot3(e1, t2);
        let dm11 = dot3(e2, t2);
        let det = dm00 * dm11 - dm01 * dm10;
        if det.abs() < 1e-16 {
            continue;
        }
        let inv_det = 1.0 / det;
        let d00 = dm11 * inv_det;
        let d01 = -dm01 * inv_det;
        let d10 = -dm10 * inv_det;
        let d11 = dm00 * inv_det;

        for &vi in &[i0, i1, i2] {
            vert_area[vi] += area0 / 3.0;
            vert_norm[vi][0] += fnrm[0];
            vert_norm[vi][1] += fnrm[1];
            vert_norm[vi][2] += fnrm[2];
        }

        shell.tri_v0.push((base_idx + i0) as i32);
        shell.tri_v1.push((base_idx + i1) as i32);
        shell.tri_v2.push((base_idx + i2) as i32);
        shell.tri_dm_inv.extend_from_slice(&[d00, d01, d10, d11]);
        shell.tri_area0.push(area0);
        shell.tri_h.push(thickness);
        shell.tri_mat_id.push(mat_id);

        for (a, b, opp) in [(i0, i1, i2), (i1, i2, i0), (i2, i0, i1)] {
            let key = if a < b { (a, b) } else { (b, a) };
            let entry = edge_map.entry(key).or_insert((None, None));
            if a < b {
                entry.0 = Some((opp, area0));
            } else {
                entry.1 = Some((opp, area0));
            }
        }
    }

    let e_eff = 0.5 * (body.material.e1 + body.material.e2);
    let nu = body.material.nu12.clamp(0.0, 0.45);
    let d_bend = e_eff * thickness.powi(3) / (12.0 * (1.0 - nu * nu));

    for (&(e0, e1), &(fwd, rev)) in &edge_map {
        if let (Some((o0, a1)), Some((o1, a2))) = (fwd, rev) {
            let x0 = world_pos[e0];
            let x1 = world_pos[e1];
            let x2 = world_pos[o0];
            let x3 = world_pos[o1];
            let e = sub3(x1, x0);
            let e_len2 = dot3(e, e);
            if e_len2 < 1e-14 {
                continue;
            }
            let e_len = e_len2.sqrt();
            let e_hat = [e[0] / e_len, e[1] / e_len, e[2] / e_len];
            let n1 = normalize3(cross3(e, sub3(x2, x0)));
            let n2 = normalize3(cross3(sub3(x3, x0), e));
            let theta0 = dot3(cross3(n1, n2), e_hat).atan2(dot3(n1, n2));
            let kb = d_bend * (3.0 * e_len2) / (a1 + a2).max(1e-12);

            shell.hinge_e0.push((base_idx + e0) as i32);
            shell.hinge_e1.push((base_idx + e1) as i32);
            shell.hinge_o0.push((base_idx + o0) as i32);
            shell.hinge_o1.push((base_idx + o1) as i32);
            shell.hinge_theta0.push(theta0);
            shell.hinge_kb.push(kb);
            shell.hinge_dmax.push(0.0);
            shell.hinge_mat_id.push(mat_id);
        }
    }

    for vi in 0..nv {
        let p = world_pos[vi];
        let nrm = normalize3(vert_norm[vi]);
        let area = vert_area[vi].max(0.1 * nom_cell_area);
        let vol0 = area * thickness;
        let mass = body.material.rho * vol0;

        let mut flag = (1u32 << 2) | (1u32 << 3);
        let mut v_presc = [0.0_f32; 3];
        for bc in &body.bcs {
            match bc {
                SolidBoundaryCondition::ClampedRegion { min, max } => {
                    if point_in_box(p, *min, *max) {
                        flag |= 1 << 0;
                    }
                }
                SolidBoundaryCondition::PrescribedVelocityRegion { min, max, velocity } => {
                    if point_in_box(p, *min, *max) {
                        flag |= 1 << 1;
                        v_presc = *velocity;
                    }
                }
            }
        }

        let idx = host.px.len() as u64;
        let gc_s = 1.0 + body.gc_heterogeneity_amp * splitmix_signed(idx ^ 0x9E37_79B9);

        host.px.push(p[0]);
        host.py.push(p[1]);
        host.pz.push(p[2]);
        host.vx.push(if (flag & 1) != 0 {
            0.0
        } else if (flag & 2) != 0 {
            v_presc[0]
        } else {
            body.initial_velocity[0]
        });
        host.vy.push(if (flag & 1) != 0 {
            0.0
        } else if (flag & 2) != 0 {
            v_presc[1]
        } else {
            body.initial_velocity[1]
        });
        host.vz.push(if (flag & 1) != 0 {
            0.0
        } else if (flag & 2) != 0 {
            v_presc[2]
        } else {
            body.initial_velocity[2]
        });
        host.nx0.push(nrm[0]);
        host.ny0.push(nrm[1]);
        host.nz0.push(nrm[2]);
        host.nx.push(nrm[0]);
        host.ny.push(nrm[1]);
        host.nz.push(nrm[2]);
        host.area.push(area);
        host.mass.push(mass);
        host.vol0.push(vol0);
        host.thickness.push(thickness);
        host.damage.push(0.0);
        host.hist_h.push(0.0);
        host.gc_scale.push(gc_s);
        host.wetness.push(body.material.wetness);
        host.mat_id.push(mat_id);
        host.body_id.push(body.body_id);
        host.flags.push(flag);
        host.v_presc_x.push(v_presc[0]);
        host.v_presc_y.push(v_presc[1]);
        host.v_presc_z.push(v_presc[2]);
        host.hyd_len.push(hyd_len);
        host.fluid_rho.push(1.204);
        q0_list.push(q0_rot);
    }
}

fn build_revolution_shell_mesh(
    center_xy: [f32; 2],
    profile_zr: &[[f32; 2]],
    n_theta: usize,
    n_meridian: usize,
) -> (Vec<[f32; 3]>, Vec<[usize; 3]>, Vec<[f32; 3]>, Vec<[f32; 2]>, f32) {
    let m = n_meridian.max(2);
    let n_t = n_theta.max(12);
    let mut cum_s = vec![0.0_f32; profile_zr.len()];
    for i in 1..profile_zr.len() {
        let dz = profile_zr[i][0] - profile_zr[i - 1][0];
        let dr = profile_zr[i][1] - profile_zr[i - 1][1];
        cum_s[i] = cum_s[i - 1] + (dz * dz + dr * dr).sqrt();
    }
    let total_s = (*cum_s.last().unwrap_or(&0.0)).max(1e-6);

    let mut stations = Vec::with_capacity(m + 1);
    for j in 0..=m {
        let s_target = (j as f32 / m as f32) * total_s;
        let mut seg = 0usize;
        while seg + 2 < profile_zr.len() && cum_s[seg + 1] < s_target {
            seg += 1;
        }
        let s0 = cum_s[seg];
        let s1 = cum_s[(seg + 1).min(profile_zr.len() - 1)];
        let t = if s1 > s0 + 1e-9 {
            ((s_target - s0) / (s1 - s0)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let p0 = profile_zr[seg];
        let p1 = profile_zr[(seg + 1).min(profile_zr.len() - 1)];
        stations.push([p0[0] + t * (p1[0] - p0[0]), (p0[1] + t * (p1[1] - p0[1])).max(0.0)]);
    }

    let mut world_pos = Vec::new();
    let mut uvs = Vec::new();
    let mut ring_idx: Vec<Vec<usize>> = Vec::with_capacity(m + 1);
    let mut r_avg_sum = 0.0_f32;

    for (j, &[z_j, r_j]) in stations.iter().enumerate() {
        let v_uv = j as f32 / m as f32;
        r_avg_sum += r_j;
        if r_j <= 1e-5 && (j == 0 || j == m) {
            let vid = world_pos.len();
            world_pos.push([center_xy[0], center_xy[1], z_j]);
            uvs.push([0.5, v_uv]);
            ring_idx.push(vec![vid]);
        } else {
            let mut row = Vec::with_capacity(n_t);
            for i in 0..n_t {
                let u_uv = i as f32 / n_t as f32;
                let phi = u_uv * 2.0 * std::f32::consts::PI;
                let (sp, cp) = phi.sin_cos();
                let vid = world_pos.len();
                world_pos.push([center_xy[0] + r_j * cp, center_xy[1] + r_j * sp, z_j]);
                uvs.push([u_uv, v_uv]);
                row.push(vid);
            }
            ring_idx.push(row);
        }
    }

    let mut tris = Vec::new();
    let mut tri_fdirs = Vec::new();
    let orient = |verts: &[[f32; 3]], a: usize, b: usize, c: usize, hint: [f32; 3]| -> [usize; 3] {
        let nrm = cross3(sub3(verts[b], verts[a]), sub3(verts[c], verts[a]));
        if dot3(nrm, hint) >= 0.0 {
            [a, b, c]
        } else {
            [a, c, b]
        }
    };

    for j in 0..m {
        let [z0, r0] = stations[j];
        let [z1, r1] = stations[j + 1];
        let dz = z1 - z0;
        let dr = r1 - r0;
        let row0 = &ring_idx[j];
        let row1 = &ring_idx[j + 1];

        if row0.len() == 1 && row1.len() == n_t {
            let p0 = row0[0];
            for i in 0..n_t {
                let i_next = (i + 1) % n_t;
                let phi = ((i as f32 + 0.5) / n_t as f32) * 2.0 * std::f32::consts::PI;
                let (sp, cp) = phi.sin_cos();
                let hint = [dz * cp, dz * sp, -dr];
                tris.push(orient(&world_pos, p0, row1[i], row1[i_next], hint));
                tri_fdirs.push([-sp, cp, 0.0]);
            }
        } else if row0.len() == n_t && row1.len() == 1 {
            let p1 = row1[0];
            for i in 0..n_t {
                let i_next = (i + 1) % n_t;
                let phi = ((i as f32 + 0.5) / n_t as f32) * 2.0 * std::f32::consts::PI;
                let (sp, cp) = phi.sin_cos();
                let hint = [dz * cp, dz * sp, -dr];
                tris.push(orient(&world_pos, row0[i], row0[i_next], p1, hint));
                tri_fdirs.push([-sp, cp, 0.0]);
            }
        } else if row0.len() == n_t && row1.len() == n_t {
            for i in 0..n_t {
                let i_next = (i + 1) % n_t;
                let phi = ((i as f32 + 0.5) / n_t as f32) * 2.0 * std::f32::consts::PI;
                let (sp, cp) = phi.sin_cos();
                let hint = [dz * cp, dz * sp, -dr];
                let a = row0[i];
                let b = row0[i_next];
                let c = row1[i_next];
                let d = row1[i];
                if (i + j) % 2 == 0 {
                    tris.push(orient(&world_pos, a, b, c, hint));
                    tris.push(orient(&world_pos, a, c, d, hint));
                } else {
                    tris.push(orient(&world_pos, a, b, d, hint));
                    tris.push(orient(&world_pos, b, c, d, hint));
                }
                tri_fdirs.push([-sp, cp, 0.0]);
                tri_fdirs.push([-sp, cp, 0.0]);
            }
        }
    }

    let r_mean = (r_avg_sum / ((m + 1) as f32)).max(1e-3);
    let nom_area = (2.0 * std::f32::consts::PI * r_mean / n_t as f32) * (total_s / m as f32);
    (world_pos, tris, tri_fdirs, uvs, nom_area)
}

#[allow(clippy::too_many_arguments)]
fn build_square_jar_shell_mesh(
    center_xy: [f32; 2],
    z_bottom: f32,
    body_half_width: f32,
    body_height: f32,
    shoulder_height: f32,
    neck_radius: f32,
    neck_height: f32,
    corner_radius: f32,
    n_perim: usize,
    n_height: usize,
) -> (Vec<[f32; 3]>, Vec<[usize; 3]>, Vec<[f32; 3]>, Vec<[f32; 2]>, f32) {
    let n_t = n_perim.max(24);
    let n_h = n_height.max(12);
    let w = body_half_width.max(1e-3);
    let r_neck = neck_radius.max(1e-3);
    let frac = (corner_radius / w).clamp(0.05, 0.95);
    let p_body = (2.0 / frac.powf(0.85)).clamp(2.5, 10.0);

    // Superellipse cross-section radius for angle phi with half-width scale `s` and exponent `p`.
    let super_xy = |phi: f32, s: f32, p: f32| -> (f32, f32) {
        let (sp, cp) = phi.sin_cos();
        let denom = (cp.abs().powf(p) + sp.abs().powf(p)).powf(1.0 / p).max(1e-6);
        let r = s / denom;
        (r * cp, r * sp)
    };

    // Build height/cross-section stations: (z, scale, p_exp)
    // 1. Flat bottom: pole + 3 concentric rounded-square rings
    // 2. Vertical square walls
    // 3. Smooth C1 shoulder blending rounded square -> circular neck
    // 4. Cylindrical open neck
    let total_h = (body_height + shoulder_height + neck_height).max(1e-3);
    let n_body = ((n_h as f32 * (body_height / total_h)).round() as usize).max(6);
    let n_shoulder = ((n_h as f32 * (shoulder_height / total_h)).round() as usize).max(4);
    let n_neck = ((n_h as f32 * (neck_height / total_h)).round() as usize).max(3);

    let mut stations: Vec<(f32, f32, f32)> = Vec::new();
    stations.push((z_bottom, 0.0, p_body));
    for k in 1..=3 {
        let frac_r = k as f32 / 3.0;
        stations.push((z_bottom, w * frac_r, p_body));
    }
    for k in 1..=n_body {
        let t = k as f32 / n_body as f32;
        stations.push((z_bottom + t * body_height, w, p_body));
    }
    for k in 1..=n_shoulder {
        let t = k as f32 / n_shoulder as f32;
        let u = t * t * (3.0 - 2.0 * t); // C1 smoothstep
        let z = z_bottom + body_height + t * shoulder_height;
        let s = w + u * (r_neck - w);
        let p = p_body + u * (2.0 - p_body);
        stations.push((z, s, p));
    }
    for k in 1..=n_neck {
        let t = k as f32 / n_neck as f32;
        let z = z_bottom + body_height + shoulder_height + t * neck_height;
        stations.push((z, r_neck, 2.0));
    }

    let m = stations.len() - 1;
    let mut world_pos = Vec::new();
    let mut uvs = Vec::new();
    let mut ring_idx: Vec<Vec<usize>> = Vec::with_capacity(stations.len());

    for (j, &(z_j, s_j, p_j)) in stations.iter().enumerate() {
        let v_uv = j as f32 / m as f32;
        if s_j <= 1e-6 {
            let vid = world_pos.len();
            world_pos.push([center_xy[0], center_xy[1], z_j]);
            uvs.push([0.5, v_uv]);
            ring_idx.push(vec![vid]);
        } else {
            let mut row = Vec::with_capacity(n_t);
            for i in 0..n_t {
                let u_uv = i as f32 / n_t as f32;
                let phi = u_uv * 2.0 * std::f32::consts::PI;
                let (lx, ly) = super_xy(phi, s_j, p_j);
                let vid = world_pos.len();
                world_pos.push([center_xy[0] + lx, center_xy[1] + ly, z_j]);
                uvs.push([u_uv, v_uv]);
                row.push(vid);
            }
            ring_idx.push(row);
        }
    }

    let mut tris = Vec::new();
    let mut tri_fdirs = Vec::new();
    let orient = |verts: &[[f32; 3]], a: usize, b: usize, c: usize, hint: [f32; 3]| -> [usize; 3] {
        let nrm = cross3(sub3(verts[b], verts[a]), sub3(verts[c], verts[a]));
        if dot3(nrm, hint) >= 0.0 {
            [a, b, c]
        } else {
            [a, c, b]
        }
    };

    for j in 0..m {
        let (z0, s0, _) = stations[j];
        let (z1, s1, _) = stations[j + 1];
        let dz = z1 - z0;
        let ds = s1 - s0;
        let row0 = &ring_idx[j];
        let row1 = &ring_idx[j + 1];

        if row0.len() == 1 && row1.len() == n_t {
            let p0 = row0[0];
            for i in 0..n_t {
                let i_next = (i + 1) % n_t;
                let phi = ((i as f32 + 0.5) / n_t as f32) * 2.0 * std::f32::consts::PI;
                let (sp, cp) = phi.sin_cos();
                tris.push(orient(&world_pos, p0, row1[i], row1[i_next], [0.0, 0.0, -1.0]));
                tri_fdirs.push([-sp, cp, 0.0]);
            }
        } else if row0.len() == n_t && row1.len() == n_t {
            for i in 0..n_t {
                let i_next = (i + 1) % n_t;
                let phi = ((i as f32 + 0.5) / n_t as f32) * 2.0 * std::f32::consts::PI;
                let (sp, cp) = phi.sin_cos();
                let hint = if dz.abs() < 1e-6 {
                    [0.0, 0.0, -1.0]
                } else {
                    [dz * cp, dz * sp, -ds]
                };
                let a = row0[i];
                let b = row0[i_next];
                let c = row1[i_next];
                let d = row1[i];
                if (i + j) % 2 == 0 {
                    tris.push(orient(&world_pos, a, b, c, hint));
                    tris.push(orient(&world_pos, a, c, d, hint));
                } else {
                    tris.push(orient(&world_pos, a, b, d, hint));
                    tris.push(orient(&world_pos, b, c, d, hint));
                }
                tri_fdirs.push([-sp, cp, 0.0]);
                tri_fdirs.push([-sp, cp, 0.0]);
            }
        }
    }

    let perim_est = 8.0 * w;
    let nom_area = (perim_est / n_t as f32) * (total_h / n_h as f32);
    (world_pos, tris, tri_fdirs, uvs, nom_area)
}

pub fn sample_shell_solid_into(
    body: &SolidBodySpec,
    mat_id: i32,
    host: &mut MpmParticlesHost,
    q0_list: &mut Vec<[f32; 9]>,
    shell: &mut ShellMeshHost,
) {
    match &body.shape {
        GeometryShape::CurvedShell {
            center,
            size_x,
            size_y,
            nx,
            ny,
            dome_amp,
            saddle_amp,
            yaw_rad,
            notch_x_len,
            notch_y_pos,
            notch_half_w,
        } => {
            let nx = (*nx).max(2);
            let ny = (*ny).max(2);
            let q0_base = euler_to_mat3(body.orientation_euler_xyz);
            let (sin_y, cos_y) = yaw_rad.sin_cos();
            let fiber1_dir = normalize3([
                q0_base[0] * cos_y + q0_base[1] * sin_y,
                q0_base[3] * cos_y + q0_base[4] * sin_y,
                q0_base[6] * cos_y + q0_base[7] * sin_y,
            ]);
            let q0_rot = [
                q0_base[0] * cos_y + q0_base[1] * sin_y,
                -q0_base[0] * sin_y + q0_base[1] * cos_y,
                q0_base[2],
                q0_base[3] * cos_y + q0_base[4] * sin_y,
                -q0_base[3] * sin_y + q0_base[4] * cos_y,
                q0_base[5],
                q0_base[6] * cos_y + q0_base[7] * sin_y,
                -q0_base[6] * sin_y + q0_base[7] * cos_y,
                q0_base[8],
            ];

            let nv = (nx + 1) * (ny + 1);
            let mut local_uv = Vec::with_capacity(nv);
            let mut world_pos = Vec::with_capacity(nv);
            let half_x = 0.5 * size_x;
            let half_y = 0.5 * size_y;
            for j in 0..=ny {
                let v = -half_y + (j as f32 / ny as f32) * size_y;
                let yv = v / half_y.max(1e-6);
                for i in 0..=nx {
                    let u = -half_x + (i as f32 / nx as f32) * size_x;
                    let xu = u / half_x.max(1e-6);
                    let z_loc = dome_amp * (1.0 - xu * xu) * (1.0 - yv * yv)
                        + saddle_amp * (xu * xu - yv * yv);
                    world_pos.push([
                        center[0] + cos_y * u - sin_y * v,
                        center[1] + sin_y * u + cos_y * v,
                        center[2] + z_loc,
                    ]);
                    local_uv.push([u, v]);
                }
            }

            let mut tris_raw = Vec::new();
            let mut try_push = |i0: usize, i1: usize, i2: usize| {
                let uc = (local_uv[i0][0] + local_uv[i1][0] + local_uv[i2][0]) / 3.0;
                let vc = (local_uv[i0][1] + local_uv[i1][1] + local_uv[i2][1]) / 3.0;
                if *notch_x_len > 0.0
                    && uc < -half_x + *notch_x_len
                    && (vc - *notch_y_pos).abs() < *notch_half_w
                {
                    return;
                }
                tris_raw.push([i0, i1, i2]);
            };
            for j in 0..ny {
                for i in 0..nx {
                    let a = j * (nx + 1) + i;
                    let b = a + 1;
                    let c = (j + 1) * (nx + 1) + (i + 1);
                    let d = (j + 1) * (nx + 1) + i;
                    if (i + j) % 2 == 0 {
                        try_push(a, b, c);
                        try_push(a, c, d);
                    } else {
                        try_push(a, b, d);
                        try_push(b, c, d);
                    }
                }
            }
            let tri_fdirs = vec![fiber1_dir; tris_raw.len()];
            let nom_area = (size_x / nx as f32) * (size_y / ny as f32);
            finalize_shell_surface_into(
                body,
                mat_id,
                &world_pos,
                &tris_raw,
                &tri_fdirs,
                nom_area,
                0.0,
                q0_rot,
                host,
                q0_list,
                shell,
            );
        }
        GeometryShape::RevolutionShell {
            center_xy,
            profile_zr,
            n_theta,
            n_meridian,
            hyd_len,
        } => {
            if profile_zr.len() < 2 {
                return;
            }
            let q0 = euler_to_mat3(body.orientation_euler_xyz);
            let (world_pos, tris_raw, tri_fdirs, _uvs, nom_area) =
                build_revolution_shell_mesh(*center_xy, profile_zr, *n_theta, *n_meridian);
            finalize_shell_surface_into(
                body,
                mat_id,
                &world_pos,
                &tris_raw,
                &tri_fdirs,
                nom_area,
                *hyd_len,
                q0,
                host,
                q0_list,
                shell,
            );
        }
        GeometryShape::SquareJarShell {
            center_xy,
            z_bottom,
            body_half_width,
            body_height,
            shoulder_height,
            neck_radius,
            neck_height,
            corner_radius,
            n_perim,
            n_height,
            hyd_len,
        } => {
            let q0 = euler_to_mat3(body.orientation_euler_xyz);
            let (world_pos, tris_raw, tri_fdirs, _uvs, nom_area) = build_square_jar_shell_mesh(
                *center_xy,
                *z_bottom,
                *body_half_width,
                *body_height,
                *shoulder_height,
                *neck_radius,
                *neck_height,
                *corner_radius,
                *n_perim,
                *n_height,
            );
            finalize_shell_surface_into(
                body,
                mat_id,
                &world_pos,
                &tris_raw,
                &tri_fdirs,
                nom_area,
                *hyd_len,
                q0,
                host,
                q0_list,
                shell,
            );
        }
        _ => {}
    }
}

#[inline]
fn ensure_tri_outward(
    verts: &[[f32; 3]],
    tri: [u32; 3],
    outward_hint: [f32; 3],
) -> [u32; 3] {
    let p0 = verts[tri[0] as usize];
    let p1 = verts[tri[1] as usize];
    let p2 = verts[tri[2] as usize];
    let nrm = cross3(sub3(p1, p0), sub3(p2, p0));
    if dot3(nrm, outward_hint) >= 0.0 {
        tri
    } else {
        [tri[0], tri[2], tri[1]]
    }
}

fn append_annular_frustum_mesh(
    verts: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tris: &mut Vec<[u32; 3]>,
    center_xy: [f32; 2],
    r_inner_bot: f32,
    r_inner_top: f32,
    r_outer_bot: f32,
    r_outer_top: f32,
    z_min: f32,
    z_max: f32,
    edge: f32,
    segs: usize,
) {
    let nz = (((z_max - z_min) / edge).ceil() as usize).clamp(1, 24);
    let mut add_wall = |r_bot: f32, r_top: f32, outward: bool| {
        let base = verts.len() as u32;
        for k in 0..=nz {
            let tz = k as f32 / nz as f32;
            let z = z_min + tz * (z_max - z_min);
            let rad = r_bot + tz * (r_top - r_bot);
            for i in 0..=segs {
                let tu = i as f32 / segs as f32;
                let phi = tu * 2.0 * std::f32::consts::PI;
                let (sp, cp) = phi.sin_cos();
                verts.push([center_xy[0] + rad * cp, center_xy[1] + rad * sp, z]);
                uvs.push([tu, tz]);
            }
        }
        let stride = (segs + 1) as u32;
        for k in 0..nz as u32 {
            for i in 0..segs as u32 {
                let a = base + k * stride + i;
                let b = a + 1;
                let c = a + stride + 1;
                let d = a + stride;
                let mid = [
                    0.25 * (verts[a as usize][0] + verts[b as usize][0] + verts[c as usize][0] + verts[d as usize][0]) - center_xy[0],
                    0.25 * (verts[a as usize][1] + verts[b as usize][1] + verts[c as usize][1] + verts[d as usize][1]) - center_xy[1],
                    0.0,
                ];
                let hint = if outward { mid } else { [-mid[0], -mid[1], 0.0] };
                tris.push(ensure_tri_outward(verts, [a, b, c], hint));
                tris.push(ensure_tri_outward(verts, [a, c, d], hint));
            }
        }
    };

    add_wall(r_outer_bot, r_outer_top, true);
    add_wall(r_inner_bot, r_inner_top, false);

    let mut add_annulus = |r_in: f32, r_out: f32, z: f32, hint_z: f32| {
        let base = verts.len() as u32;
        for i in 0..=segs {
            let tu = i as f32 / segs as f32;
            let phi = tu * 2.0 * std::f32::consts::PI;
            let (sp, cp) = phi.sin_cos();
            verts.push([center_xy[0] + r_in * cp, center_xy[1] + r_in * sp, z]);
            uvs.push([tu, 0.0]);
            verts.push([center_xy[0] + r_out * cp, center_xy[1] + r_out * sp, z]);
            uvs.push([tu, 1.0]);
        }
        for i in 0..segs as u32 {
            let i0 = base + 2 * i;
            let o0 = i0 + 1;
            let i1 = i0 + 2;
            let o1 = i0 + 3;
            tris.push(ensure_tri_outward(verts, [i0, o0, o1], [0.0, 0.0, hint_z]));
            tris.push(ensure_tri_outward(verts, [i0, o1, i1], [0.0, 0.0, hint_z]));
        }
    };
    add_annulus(r_inner_top, r_outer_top, z_max, 1.0);
    add_annulus(r_inner_bot, r_outer_bot, z_min, -1.0);
}

/// Generate a watertight surface triangle mesh with UVs `(u, v)` directly from a `GeometryShape`.
pub fn generate_shape_surface_mesh(
    shape: &GeometryShape,
    target_edge: f32,
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>, Vec<[f32; 2]>) {
    let edge = target_edge.max(1e-4);
    let mut verts: Vec<[f32; 3]> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();

    match shape {
        GeometryShape::Box { min, max } => {
            let lx = (max[0] - min[0]).max(1e-6);
            let ly = (max[1] - min[1]).max(1e-6);
            let lz = (max[2] - min[2]).max(1e-6);
            let sx = ((lx / edge).ceil() as usize).clamp(1, 64);
            let sy = ((ly / edge).ceil() as usize).clamp(1, 64);
            let sz = ((lz / edge).ceil() as usize).clamp(1, 64);

            let mut add_face = |p0: [f32; 3],
                                u_vec: [f32; 3],
                                v_vec: [f32; 3],
                                nu: usize,
                                nv: usize,
                                nrm_hint: [f32; 3]| {
                let base = verts.len() as u32;
                for j in 0..=nv {
                    let tv = j as f32 / nv as f32;
                    for i in 0..=nu {
                        let tu = i as f32 / nu as f32;
                        verts.push([
                            p0[0] + tu * u_vec[0] + tv * v_vec[0],
                            p0[1] + tu * u_vec[1] + tv * v_vec[1],
                            p0[2] + tu * u_vec[2] + tv * v_vec[2],
                        ]);
                        uvs.push([tu, tv]);
                    }
                }
                let stride = (nu + 1) as u32;
                for j in 0..nv as u32 {
                    for i in 0..nu as u32 {
                        let a = base + j * stride + i;
                        let b = a + 1;
                        let c = a + stride + 1;
                        let d = a + stride;
                        tris.push(ensure_tri_outward(&verts, [a, b, c], nrm_hint));
                        tris.push(ensure_tri_outward(&verts, [a, c, d], nrm_hint));
                    }
                }
            };

            add_face([min[0], min[1], min[2]], [0.0, ly, 0.0], [lx, 0.0, 0.0], sy, sx, [0.0, 0.0, -1.0]);
            add_face([min[0], min[1], max[2]], [lx, 0.0, 0.0], [0.0, ly, 0.0], sx, sy, [0.0, 0.0, 1.0]);
            add_face([min[0], min[1], min[2]], [lx, 0.0, 0.0], [0.0, 0.0, lz], sx, sz, [0.0, -1.0, 0.0]);
            add_face([min[0], max[1], min[2]], [0.0, 0.0, lz], [lx, 0.0, 0.0], sz, sx, [0.0, 1.0, 0.0]);
            add_face([min[0], min[1], min[2]], [0.0, 0.0, lz], [0.0, ly, 0.0], sz, sy, [-1.0, 0.0, 0.0]);
            add_face([max[0], min[1], min[2]], [0.0, ly, 0.0], [0.0, 0.0, lz], sy, sz, [1.0, 0.0, 0.0]);
        }
        GeometryShape::Sphere { center, radius } => {
            let n_lon = 36usize;
            let n_lat = 20usize;
            for j in 0..=n_lat {
                let v = j as f32 / n_lat as f32;
                let theta = v * std::f32::consts::PI;
                let (st, ct) = theta.sin_cos();
                for i in 0..=n_lon {
                    let u = i as f32 / n_lon as f32;
                    let phi = u * 2.0 * std::f32::consts::PI;
                    let (sp, cp) = phi.sin_cos();
                    verts.push([
                        center[0] + radius * st * cp,
                        center[1] + radius * st * sp,
                        center[2] + radius * ct,
                    ]);
                    uvs.push([u, v]);
                }
            }
            let stride = (n_lon + 1) as u32;
            for j in 0..n_lat as u32 {
                for i in 0..n_lon as u32 {
                    let a = j * stride + i;
                    let b = a + 1;
                    let c = a + stride + 1;
                    let d = a + stride;
                    let mid = [
                        0.25 * (verts[a as usize][0] + verts[b as usize][0] + verts[c as usize][0] + verts[d as usize][0]) - center[0],
                        0.25 * (verts[a as usize][1] + verts[b as usize][1] + verts[c as usize][1] + verts[d as usize][1]) - center[1],
                        0.25 * (verts[a as usize][2] + verts[b as usize][2] + verts[c as usize][2] + verts[d as usize][2]) - center[2],
                    ];
                    if j > 0 {
                        tris.push(ensure_tri_outward(&verts, [a, b, c], mid));
                    }
                    if j + 1 < n_lat as u32 {
                        tris.push(ensure_tri_outward(&verts, [a, c, d], mid));
                    }
                }
            }
        }
        GeometryShape::CylindricalRing {
            center_xy,
            r_inner,
            r_outer,
            z_min,
            z_max,
        }
        | GeometryShape::AnnularDisc {
            center_xy,
            r_inner,
            r_outer,
            z_min,
            z_max,
        } => {
            append_annular_frustum_mesh(
                &mut verts, &mut uvs, &mut tris, *center_xy, *r_inner, *r_inner, *r_outer, *r_outer,
                *z_min, *z_max, edge, 128,
            );
        }
        GeometryShape::FrustumRing {
            center_xy,
            r_inner_bottom,
            r_inner_top,
            thickness,
            z_min,
            z_max,
        } => {
            append_annular_frustum_mesh(
                &mut verts,
                &mut uvs,
                &mut tris,
                *center_xy,
                *r_inner_bottom,
                *r_inner_top,
                *r_inner_bottom + *thickness,
                *r_inner_top + *thickness,
                *z_min,
                *z_max,
                edge,
                128,
            );
        }
        GeometryShape::Disc {
            center_xy,
            radius,
            z_min,
            z_max,
        } => {
            let segs = 160usize;
            let base = verts.len() as u32;
            for i in 0..=segs {
                let tu = i as f32 / segs as f32;
                let phi = tu * 2.0 * std::f32::consts::PI;
                let (sp, cp) = phi.sin_cos();
                verts.push([center_xy[0] + radius * cp, center_xy[1] + radius * sp, *z_min]);
                uvs.push([tu, 0.0]);
                verts.push([center_xy[0] + radius * cp, center_xy[1] + radius * sp, *z_max]);
                uvs.push([tu, 1.0]);
            }
            for i in 0..segs as u32 {
                let b0 = base + 2 * i;
                let t0 = b0 + 1;
                let b1 = b0 + 2;
                let t1 = b0 + 3;
                let mid = [
                    0.5 * (verts[b0 as usize][0] + verts[b1 as usize][0]) - center_xy[0],
                    0.5 * (verts[b0 as usize][1] + verts[b1 as usize][1]) - center_xy[1],
                    0.0,
                ];
                tris.push(ensure_tri_outward(&verts, [b0, b1, t1], mid));
                tris.push(ensure_tri_outward(&verts, [b0, t1, t0], mid));
            }
            let top_c = verts.len() as u32;
            verts.push([center_xy[0], center_xy[1], *z_max]);
            uvs.push([0.5, 0.5]);
            let bot_c = verts.len() as u32;
            verts.push([center_xy[0], center_xy[1], *z_min]);
            uvs.push([0.5, 0.5]);
            for i in 0..segs as u32 {
                let t0 = base + 2 * i + 1;
                let t1 = t0 + 2;
                tris.push(ensure_tri_outward(&verts, [top_c, t0, t1], [0.0, 0.0, 1.0]));
                let b0 = base + 2 * i;
                let b1 = b0 + 2;
                tris.push(ensure_tri_outward(&verts, [bot_c, b1, b0], [0.0, 0.0, -1.0]));
            }
        }
        GeometryShape::Mesh(mesh) => {
            verts = mesh.positions.clone();
            tris = mesh
                .triangles
                .iter()
                .map(|t| [t[0] as u32, t[1] as u32, t[2] as u32])
                .collect();
            uvs = vec![[0.0, 0.0]; verts.len()];
        }
        GeometryShape::RevolutionShell {
            center_xy,
            profile_zr,
            n_theta,
            n_meridian,
            ..
        } => {
            if profile_zr.len() >= 2 {
                let (vp, tr, _, uv, _) =
                    build_revolution_shell_mesh(*center_xy, profile_zr, *n_theta, *n_meridian);
                verts = vp;
                tris = tr
                    .into_iter()
                    .map(|[a, b, c]| [a as u32, b as u32, c as u32])
                    .collect();
                uvs = uv;
            }
        }
        GeometryShape::SquareJarShell {
            center_xy,
            z_bottom,
            body_half_width,
            body_height,
            shoulder_height,
            neck_radius,
            neck_height,
            corner_radius,
            n_perim,
            n_height,
            ..
        } => {
            let (vp, tr, _, uv, _) = build_square_jar_shell_mesh(
                *center_xy,
                *z_bottom,
                *body_half_width,
                *body_height,
                *shoulder_height,
                *neck_radius,
                *neck_height,
                *corner_radius,
                *n_perim,
                *n_height,
            );
            verts = vp;
            tris = tr
                .into_iter()
                .map(|[a, b, c]| [a as u32, b as u32, c as u32])
                .collect();
            uvs = uv;
        }
        GeometryShape::CylindricalRingWithArch {
            center_xy,
            r_inner,
            r_outer,
            z_min,
            z_max,
            ..
        } => {
            append_annular_frustum_mesh(
                &mut verts, &mut uvs, &mut tris, *center_xy, *r_inner, *r_inner, *r_outer, *r_outer,
                *z_min, *z_max, edge, 128,
            );
        }
        GeometryShape::CanyonTerrain {
            floor_z,
            valley_half_width_y,
            wall_slope,
            cliff_x,
            cliff_height,
        } => {
            let nx_t = 64usize;
            let ny_t = 32usize;
            for j in 0..=ny_t {
                let v = j as f32 / ny_t as f32;
                let y = -40.0 + v * 80.0;
                for i in 0..=nx_t {
                    let u = i as f32 / nx_t as f32;
                    let x = -20.0 + u * 1120.0;
                    let z = canyon_ground_z(
                        x,
                        y,
                        *floor_z,
                        *valley_half_width_y,
                        *wall_slope,
                        *cliff_x,
                        *cliff_height,
                    );
                    verts.push([x, y, z]);
                    uvs.push([u, v]);
                }
            }
            let stride = (nx_t + 1) as u32;
            for j in 0..ny_t as u32 {
                for i in 0..nx_t as u32 {
                    let a = j * stride + i;
                    let b = a + 1;
                    let c = a + stride + 1;
                    let d = a + stride;
                    tris.push([a, b, c]);
                    tris.push([a, c, d]);
                }
            }
        }
        GeometryShape::CurvedShell { .. } | GeometryShape::HalfSpaceZ { .. } => {}
    }

    (verts, tris, uvs)
}

/// Generate matching `(u, v)` texture coordinates for the grid vertices produced by `sample_shell_solid_into`.
pub fn generate_shell_uvs(nx: usize, ny: usize) -> Vec<[f32; 2]> {
    let nx = nx.max(2);
    let ny = ny.max(2);
    let mut uvs = Vec::with_capacity((nx + 1) * (ny + 1));
    for j in 0..=ny {
        let v = j as f32 / ny as f32;
        for i in 0..=nx {
            let u = i as f32 / nx as f32;
            uvs.push([u, v]);
        }
    }
    uvs
}
