use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RenderExportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Invalid mesh or field dimensions: {0}")]
    InvalidData(String),
}

pub type Result<T> = std::result::Result<T, RenderExportError>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderManifest {
    pub scenario: String,
    pub duration_s: f64,
    #[serde(default)]
    pub pre_roll_s: f64,
    pub num_frames: usize,
    pub has_fluid_isosurface: bool,
    pub has_pressure_slice: bool,
    pub pressure_slice_z: f32,
    #[serde(default)]
    pub pslice_r_max: f32,
    #[serde(default)]
    pub pslice_max_abs_p: f32,
    #[serde(default)]
    pub water_bounds: Option<([f32; 3], [f32; 3])>,
    #[serde(default)]
    pub floor_z: Option<f32>,
    pub bodies: Vec<RenderBodyManifest>,
    pub microphones: Vec<RenderMicManifest>,
    pub point_sources: Vec<RenderSourceManifest>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderBodyManifest {
    pub name: String,
    pub body_id: i32,
    pub material_preset: String,
    pub thickness_m: f32,
    pub wetness: f32,
    pub is_static: bool,
    pub is_shell: bool,
    pub rest_mesh_file: Option<String>,
    pub clamps: Vec<([f32; 3], [f32; 3])>,
    #[serde(default)]
    pub pulls: Vec<([f32; 3], [f32; 3])>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderMicManifest {
    pub name: String,
    pub position: [f32; 3],
    pub is_hydrophone: bool,
    pub wav_path: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderSourceManifest {
    pub center: [f32; 3],
    pub t0_s: f64,
    pub f0_hz: f64,
}

/// Write a binary little-endian PLY triangle mesh with vertex `(x, y, z, u, v, damage)`.
pub fn write_ply_mesh(
    path: impl AsRef<Path>,
    verts: &[[f32; 3]],
    tris: &[[u32; 3]],
    uvs: Option<&[[f32; 2]]>,
    damage: Option<&[f32]>,
) -> Result<()> {
    if let Some(u) = uvs {
        if u.len() != verts.len() {
            return Err(RenderExportError::InvalidData(format!(
                "UV count {} != vertex count {}",
                u.len(),
                verts.len()
            )));
        }
    }
    if let Some(d) = damage {
        if d.len() != verts.len() {
            return Err(RenderExportError::InvalidData(format!(
                "Damage count {} != vertex count {}",
                d.len(),
                verts.len()
            )));
        }
    }

    let file = File::create(path)?;
    let mut w = BufWriter::with_capacity(64 * 1024, file);
    writeln!(w, "ply")?;
    writeln!(w, "format binary_little_endian 1.0")?;
    writeln!(w, "element vertex {}", verts.len())?;
    writeln!(w, "property float x")?;
    writeln!(w, "property float y")?;
    writeln!(w, "property float z")?;
    writeln!(w, "property float u")?;
    writeln!(w, "property float v")?;
    writeln!(w, "property float damage")?;
    writeln!(w, "element face {}", tris.len())?;
    writeln!(w, "property list uchar int vertex_indices")?;
    writeln!(w, "end_header")?;

    for (idx, p) in verts.iter().enumerate() {
        let uv = uvs.map(|arr| arr[idx]).unwrap_or([0.0, 0.0]);
        let dmg = damage.map(|arr| arr[idx]).unwrap_or(0.0);
        w.write_all(&p[0].to_le_bytes())?;
        w.write_all(&p[1].to_le_bytes())?;
        w.write_all(&p[2].to_le_bytes())?;
        w.write_all(&uv[0].to_le_bytes())?;
        w.write_all(&uv[1].to_le_bytes())?;
        w.write_all(&dmg.to_le_bytes())?;
    }

    for tri in tris {
        w.write_all(&[3u8])?;
        w.write_all(&(tri[0] as i32).to_le_bytes())?;
        w.write_all(&(tri[1] as i32).to_le_bytes())?;
        w.write_all(&(tri[2] as i32).to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

/// Write a 2D pressure field slice as little-endian header `(u32 nx, u32 ny, f32 ox, f32 oy, f32 dx, f32 z, f32 max_abs_p)`
/// followed by `nx * ny` row-major (`x + nx * y`) `f32` values.
pub fn write_pressure_slice_f32(
    path: impl AsRef<Path>,
    nx: usize,
    ny: usize,
    ox: f32,
    oy: f32,
    dx: f32,
    z: f32,
    max_abs_p: f32,
    values: &[f32],
) -> Result<()> {
    if values.len() != nx * ny {
        return Err(RenderExportError::InvalidData(format!(
            "Pressure slice len {} != {}x{}",
            values.len(),
            nx,
            ny
        )));
    }
    let file = File::create(path)?;
    let mut w = BufWriter::with_capacity(64 * 1024, file);
    w.write_all(&(nx as u32).to_le_bytes())?;
    w.write_all(&(ny as u32).to_le_bytes())?;
    w.write_all(&ox.to_le_bytes())?;
    w.write_all(&oy.to_le_bytes())?;
    w.write_all(&dx.to_le_bytes())?;
    w.write_all(&z.to_le_bytes())?;
    w.write_all(&max_abs_p.to_le_bytes())?;
    for &v in values {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

pub fn write_render_manifest(path: impl AsRef<Path>, manifest: &RenderManifest) -> Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut w, manifest)?;
    w.flush()?;
    Ok(())
}

#[inline]
fn emit_quad_best_diagonal(
    verts: &[[f32; 3]],
    tris: &mut Vec<[u32; 3]>,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    positive_winding: bool,
) {
    let dist2 = |u: u32, v: u32| -> f32 {
        let p = verts[u as usize];
        let q = verts[v as usize];
        let dx = p[0] - q[0];
        let dy = p[1] - q[1];
        let dz = p[2] - q[2];
        dx * dx + dy * dy + dz * dz
    };

    if dist2(a, c) <= dist2(b, d) {
        if positive_winding {
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        } else {
            tris.push([a, c, b]);
            tris.push([a, d, c]);
        }
    } else if positive_winding {
        tris.push([b, c, d]);
        tris.push([b, d, a]);
    } else {
        tris.push([b, d, c]);
        tris.push([b, a, d]);
    }
}

/// Extract a closed, watertight Dual Surface Nets isosurface from a 3D scalar volume fraction field.
/// Normals point strictly from `alpha >= iso` (water) toward `alpha < iso` (air).
pub fn extract_surface_nets_isosurface(
    alpha: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    origin: [f32; 3],
    dx: f32,
    iso: f32,
    pad_outer_air: bool,
    taubin_iters: usize,
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    if nx < 2 || ny < 2 || nz < 2 || alpha.len() < nx * ny * nz {
        return (Vec::new(), Vec::new());
    }

    let sample = |i: isize, j: isize, k: isize| -> f32 {
        if i < 0 || i >= nx as isize || j < 0 || j >= ny as isize || k < 0 || k >= nz as isize {
            if pad_outer_air {
                0.0
            } else {
                let ci = i.clamp(0, nx as isize - 1) as usize;
                let cj = j.clamp(0, ny as isize - 1) as usize;
                let ck = k.clamp(0, nz as isize - 1) as usize;
                alpha[ci + nx * (cj + ny * ck)]
            }
        } else {
            alpha[i as usize + nx * (j as usize + ny * k as usize)]
        }
    };

    let d_lo = if pad_outer_air { -1isize } else { 0isize };
    let d_hi_x = if pad_outer_air { nx as isize } else { nx as isize - 1 };
    let d_hi_y = if pad_outer_air { ny as isize } else { ny as isize - 1 };
    let d_hi_z = if pad_outer_air { nz as isize } else { nz as isize - 1 };

    let d_nx = (d_hi_x - d_lo) as usize;
    let d_ny = (d_hi_y - d_lo) as usize;
    let d_nz = (d_hi_z - d_lo) as usize;
    let mut cell_to_vert = vec![u32::MAX; d_nx * d_ny * d_nz];
    let dual_idx = |i: isize, j: isize, k: isize| -> usize {
        (i - d_lo) as usize + d_nx * ((j - d_lo) as usize + d_ny * (k - d_lo) as usize)
    };

    let mut verts: Vec<[f32; 3]> = Vec::new();
    const CUBE_EDGES: [(usize, usize); 12] = [
        (0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];

    for dk in d_lo..d_hi_z {
        for dj in d_lo..d_hi_y {
            for di in d_lo..d_hi_x {
                let mut vals = [0.0f32; 8];
                let mut mask = 0u8;
                for c in 0..8 {
                    let ox = (c & 1) as isize;
                    let oy = ((c >> 1) & 1) as isize;
                    let oz = ((c >> 2) & 1) as isize;
                    let v = sample(di + ox, dj + oy, dk + oz);
                    vals[c] = v;
                    if v >= iso {
                        mask |= 1u8 << c;
                    }
                }
                if mask == 0 || mask == 0xFF {
                    continue;
                }

                let mut sum = [0.0f32; 3];
                let mut cnt = 0.0f32;
                for &(c0, c1) in &CUBE_EDGES {
                    let b0 = (mask >> c0) & 1;
                    let b1 = (mask >> c1) & 1;
                    if b0 == b1 {
                        continue;
                    }
                    let v0 = vals[c0];
                    let v1 = vals[c1];
                    let dv = v1 - v0;
                    let t = if dv.abs() > 1e-8 {
                        ((iso - v0) / dv).clamp(0.0, 1.0)
                    } else {
                        0.5
                    };
                    let p0 = [
                        (di + (c0 & 1) as isize) as f32 + 0.5,
                        (dj + ((c0 >> 1) & 1) as isize) as f32 + 0.5,
                        (dk + ((c0 >> 2) & 1) as isize) as f32 + 0.5,
                    ];
                    let p1 = [
                        (di + (c1 & 1) as isize) as f32 + 0.5,
                        (dj + ((c1 >> 1) & 1) as isize) as f32 + 0.5,
                        (dk + ((c1 >> 2) & 1) as isize) as f32 + 0.5,
                    ];
                    sum[0] += p0[0] + t * (p1[0] - p0[0]);
                    sum[1] += p0[1] + t * (p1[1] - p0[1]);
                    sum[2] += p0[2] + t * (p1[2] - p0[2]);
                    cnt += 1.0;
                }

                if cnt > 0.0 {
                    let vid = verts.len() as u32;
                    cell_to_vert[dual_idx(di, dj, dk)] = vid;
                    verts.push([
                        origin[0] + (sum[0] / cnt) * dx,
                        origin[1] + (sum[1] / cnt) * dx,
                        origin[2] + (sum[2] / cnt) * dx,
                    ]);
                }
            }
        }
    }

    let mut tris: Vec<[u32; 3]> = Vec::with_capacity(verts.len() * 2);

    // X-edges: (i, j, k) -> (i + 1, j, k)
    let j_start = if pad_outer_air { 0isize } else { 1isize };
    let j_end = if pad_outer_air { ny as isize } else { ny as isize - 1 };
    let k_start = if pad_outer_air { 0isize } else { 1isize };
    let k_end = if pad_outer_air { nz as isize } else { nz as isize - 1 };
    let i_start = if pad_outer_air { 0isize } else { 1isize };
    let i_end = if pad_outer_air { nx as isize } else { nx as isize - 1 };

    for k in k_start..k_end {
        for j in j_start..j_end {
            for i in d_lo..d_hi_x {
                let s0 = sample(i, j, k) >= iso;
                let s1 = sample(i + 1, j, k) >= iso;
                if s0 == s1 {
                    continue;
                }
                let a = cell_to_vert[dual_idx(i, j - 1, k - 1)];
                let b = cell_to_vert[dual_idx(i, j, k - 1)];
                let c = cell_to_vert[dual_idx(i, j, k)];
                let d = cell_to_vert[dual_idx(i, j - 1, k)];
                if a != u32::MAX && b != u32::MAX && c != u32::MAX && d != u32::MAX {
                    // Normal points from water (s0=true) to air (s1=false) in +X direction
                    emit_quad_best_diagonal(&verts, &mut tris, a, b, c, d, s0 && !s1);
                }
            }
        }
    }

    // Y-edges: (i, j, k) -> (i, j + 1, k)
    for k in k_start..k_end {
        for j in d_lo..d_hi_y {
            for i in i_start..i_end {
                let s0 = sample(i, j, k) >= iso;
                let s1 = sample(i, j + 1, k) >= iso;
                if s0 == s1 {
                    continue;
                }
                let a = cell_to_vert[dual_idx(i - 1, j, k - 1)];
                let b = cell_to_vert[dual_idx(i - 1, j, k)];
                let c = cell_to_vert[dual_idx(i, j, k)];
                let d = cell_to_vert[dual_idx(i, j, k - 1)];
                if a != u32::MAX && b != u32::MAX && c != u32::MAX && d != u32::MAX {
                    // (+Z) x (+X) = +Y
                    emit_quad_best_diagonal(&verts, &mut tris, a, b, c, d, s0 && !s1);
                }
            }
        }
    }

    // Z-edges: (i, j, k) -> (i, j, k + 1)
    for k in d_lo..d_hi_z {
        for j in j_start..j_end {
            for i in i_start..i_end {
                let s0 = sample(i, j, k) >= iso;
                let s1 = sample(i, j, k + 1) >= iso;
                if s0 == s1 {
                    continue;
                }
                let a = cell_to_vert[dual_idx(i - 1, j - 1, k)];
                let b = cell_to_vert[dual_idx(i, j - 1, k)];
                let c = cell_to_vert[dual_idx(i, j, k)];
                let d = cell_to_vert[dual_idx(i - 1, j, k)];
                if a != u32::MAX && b != u32::MAX && c != u32::MAX && d != u32::MAX {
                    // (+X) x (+Y) = +Z
                    emit_quad_best_diagonal(&verts, &mut tris, a, b, c, d, s0 && !s1);
                }
            }
        }
    }

    if taubin_iters > 0 && !verts.is_empty() && !tris.is_empty() {
        taubin_smooth_mesh(
            &mut verts,
            &tris,
            origin,
            [nx as f32 * dx, ny as f32 * dx, nz as f32 * dx],
            dx,
            pad_outer_air,
            taubin_iters,
        );
    }

    (verts, tris)
}

fn taubin_smooth_mesh(
    verts: &mut Vec<[f32; 3]>,
    tris: &[[u32; 3]],
    origin: [f32; 3],
    extent: [f32; 3],
    dx: f32,
    pin_outer_walls: bool,
    iters: usize,
) {
    let n = verts.len();
    let mut pinned = vec![false; n];
    if pin_outer_walls {
        let eps = 0.25 * dx;
        for (idx, p) in verts.iter().enumerate() {
            if (p[0] - origin[0]).abs() <= eps
                || (p[0] - (origin[0] + extent[0])).abs() <= eps
                || (p[1] - origin[1]).abs() <= eps
                || (p[1] - (origin[1] + extent[1])).abs() <= eps
                || (p[2] - origin[2]).abs() <= eps
            {
                pinned[idx] = true;
            }
        }
    }

    // Collect unique undirected graph edges once so vertex degree is exact
    let mut edge_map: HashMap<u64, (u32, u32)> = HashMap::with_capacity(tris.len() * 2);
    for &[a, b, c] in tris {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            if u == v {
                continue;
            }
            let (lo, hi) = if u < v { (u, v) } else { (v, u) };
            let key = ((lo as u64) << 32) | (hi as u64);
            edge_map.entry(key).or_insert((lo, hi));
        }
    }
    let edges: Vec<(u32, u32)> = edge_map.into_values().collect();
    let mut sum = vec![[0.0f32; 3]; n];
    let mut deg = vec![0u32; n];
    let mut next = verts.clone();

    let mut apply_pass = |cur: &Vec<[f32; 3]>, dst: &mut Vec<[f32; 3]>, factor: f32| {
        sum.fill([0.0; 3]);
        deg.fill(0);
        for &(u, v) in &edges {
            let pu = cur[u as usize];
            let pv = cur[v as usize];
            sum[u as usize][0] += pv[0];
            sum[u as usize][1] += pv[1];
            sum[u as usize][2] += pv[2];
            deg[u as usize] += 1;
            sum[v as usize][0] += pu[0];
            sum[v as usize][1] += pu[1];
            sum[v as usize][2] += pu[2];
            deg[v as usize] += 1;
        }
        for i in 0..n {
            if pinned[i] || deg[i] == 0 {
                dst[i] = cur[i];
            } else {
                let d = deg[i] as f32;
                let avg = [sum[i][0] / d, sum[i][1] / d, sum[i][2] / d];
                dst[i] = [
                    cur[i][0] + factor * (avg[0] - cur[i][0]),
                    cur[i][1] + factor * (avg[1] - cur[i][1]),
                    cur[i][2] + factor * (avg[2] - cur[i][2]),
                ];
            }
        }
    };

    for _ in 0..iters {
        apply_pass(verts, &mut next, 0.45);
        apply_pass(&next, verts, -0.48);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surface_nets_sphere_outward_normals_and_watertight() {
        let nx = 32;
        let ny = 32;
        let nz = 32;
        let dx = 0.001_f32;
        let origin = [-0.016_f32, -0.016, -0.016];
        let r = 0.008_f32;
        let mut alpha = vec![0.0f32; nx * ny * nz];
        for k in 0..nz {
            let z = origin[2] + (k as f32 + 0.5) * dx;
            for j in 0..ny {
                let y = origin[1] + (j as f32 + 0.5) * dx;
                for i in 0..nx {
                    let x = origin[0] + (i as f32 + 0.5) * dx;
                    let d = (x * x + y * y + z * z).sqrt();
                    alpha[i + nx * (j + ny * k)] = ((r - d) / (0.8 * dx) + 0.5).clamp(0.0, 1.0);
                }
            }
        }

        let (verts, tris) =
            extract_surface_nets_isosurface(&alpha, nx, ny, nz, origin, dx, 0.5, true, 2);
        assert!(verts.len() > 200);
        assert!(tris.len() > 400);

        let mut outward = 0usize;
        let mut dir_edge_counts: HashMap<(u32, u32), usize> = HashMap::new();
        for &[a, b, c] in &tris {
            for (u, v) in [(a, b), (b, c), (c, a)] {
                *dir_edge_counts.entry((u, v)).or_insert(0) += 1;
            }
            let pa = verts[a as usize];
            let pb = verts[b as usize];
            let pc = verts[c as usize];
            let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
            let nrm = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let cx = (pa[0] + pb[0] + pc[0]) / 3.0;
            let cy = (pa[1] + pb[1] + pc[1]) / 3.0;
            let cz = (pa[2] + pb[2] + pc[2]) / 3.0;
            if nrm[0] * cx + nrm[1] * cy + nrm[2] * cz > 0.0 {
                outward += 1;
            }
        }

        let frac_out = outward as f64 / tris.len() as f64;
        assert!(
            frac_out > 0.99,
            "Expected >99% outward normals on Surface Nets sphere, got {:.2}%",
            frac_out * 100.0
        );
        // Every directed half-edge should match its opposite half-edge on a closed 2-manifold
        let unmatched = dir_edge_counts
            .iter()
            .filter(|(&(u, v), &cnt)| dir_edge_counts.get(&(v, u)).copied().unwrap_or(0) != cnt)
            .count();
        assert_eq!(unmatched, 0, "Expected watertight closed mesh, got {unmatched} boundary edges");
    }
}
