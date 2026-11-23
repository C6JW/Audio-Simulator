use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ObjError {
    #[error("IO error reading OBJ: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error at line {line}: {msg}")]
    Parse { line: usize, msg: String },
}

pub type Result<T> = std::result::Result<T, ObjError>;

#[derive(Clone, Debug, Default)]
pub struct ObjMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub triangles: Vec<[usize; 3]>,
}

impl ObjMesh {
    pub fn load_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::parse_reader(reader)
    }

    pub fn parse_str(data: &str) -> Result<Self> {
        Self::parse_reader(BufReader::new(data.as_bytes()))
    }

    fn parse_reader<R: BufRead>(reader: R) -> Result<Self> {
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut raw_normals: Vec<[f32; 3]> = Vec::new();
        let mut triangles: Vec<[usize; 3]> = Vec::new();
        let mut tri_norm_idx: Vec<[Option<usize>; 3]> = Vec::new();

        for (line_idx, line_res) in reader.lines().enumerate() {
            let line_no = line_idx + 1;
            let line = line_res?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let Some(tag) = parts.next() else {
                continue;
            };
            match tag {
                "v" => {
                    let coords: Vec<f32> = parts
                        .take(3)
                        .map(|s| {
                            s.parse::<f32>().map_err(|_| ObjError::Parse {
                                line: line_no,
                                msg: format!("invalid vertex coordinate `{s}`"),
                            })
                        })
                        .collect::<Result<_>>()?;
                    if coords.len() < 3 {
                        return Err(ObjError::Parse {
                            line: line_no,
                            msg: "vertex needs 3 coordinates".to_string(),
                        });
                    }
                    positions.push([coords[0], coords[1], coords[2]]);
                }
                "vn" => {
                    let coords: Vec<f32> = parts
                        .take(3)
                        .map(|s| {
                            s.parse::<f32>().map_err(|_| ObjError::Parse {
                                line: line_no,
                                msg: format!("invalid normal coordinate `{s}`"),
                            })
                        })
                        .collect::<Result<_>>()?;
                    if coords.len() < 3 {
                        return Err(ObjError::Parse {
                            line: line_no,
                            msg: "normal needs 3 coordinates".to_string(),
                        });
                    }
                    raw_normals.push([coords[0], coords[1], coords[2]]);
                }
                "f" => {
                    let mut verts = Vec::new();
                    let mut norms = Vec::new();
                    for tok in parts {
                        let mut segs = tok.split('/');
                        let v_str = segs.next().unwrap_or("");
                        let _vt_str = segs.next();
                        let vn_str = segs.next();

                        let v_raw = v_str.parse::<isize>().map_err(|_| ObjError::Parse {
                            line: line_no,
                            msg: format!("invalid face vertex index `{tok}`"),
                        })?;
                        let v_idx = resolve_obj_index(v_raw, positions.len(), line_no)?;
                        let n_idx = match vn_str {
                            Some(s) if !s.is_empty() => {
                                let n_raw = s.parse::<isize>().map_err(|_| ObjError::Parse {
                                    line: line_no,
                                    msg: format!("invalid face normal index `{tok}`"),
                                })?;
                                Some(resolve_obj_index(n_raw, raw_normals.len(), line_no)?)
                            }
                            _ => None,
                        };
                        verts.push(v_idx);
                        norms.push(n_idx);
                    }
                    if verts.len() >= 3 {
                        for i in 1..(verts.len() - 1) {
                            triangles.push([verts[0], verts[i], verts[i + 1]]);
                            tri_norm_idx.push([norms[0], norms[i], norms[i + 1]]);
                        }
                    }
                }
                _ => {}
            }
        }

        // Accumulate smooth vertex normals (or use parsed vertex normals)
        let mut normals = vec![[0.0_f32; 3]; positions.len()];
        for (ti, tri) in triangles.iter().enumerate() {
            let p0 = positions[tri[0]];
            let p1 = positions[tri[1]];
            let p2 = positions[tri[2]];
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let fnrm = cross(e1, e2);
            for c in 0..3 {
                if let Some(ni) = tri_norm_idx[ti][c] {
                    normals[tri[c]] = raw_normals[ni];
                } else {
                    for k in 0..3 {
                        normals[tri[c]][k] += fnrm[k];
                    }
                }
            }
        }
        for n in &mut normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-12 {
                n[0] /= len;
                n[1] /= len;
                n[2] /= len;
            } else {
                *n = [0.0, 0.0, 1.0];
            }
        }

        Ok(Self {
            positions,
            normals,
            triangles,
        })
    }

    pub fn bounding_box(&self) -> ([f32; 3], [f32; 3]) {
        if self.positions.is_empty() {
            return ([0.0; 3], [0.0; 3]);
        }
        let mut min_p = [f32::INFINITY; 3];
        let mut max_p = [f32::NEG_INFINITY; 3];
        for p in &self.positions {
            for d in 0..3 {
                min_p[d] = min_p[d].min(p[d]);
                max_p[d] = max_p[d].max(p[d]);
            }
        }
        (min_p, max_p)
    }

    /// Ray-triangle parity test along +Z axis for watertight closed meshes.
    pub fn contains_point(&self, p: [f32; 3]) -> bool {
        let mut crossings = 0usize;
        for tri in &self.triangles {
            let a = self.positions[tri[0]];
            let b = self.positions[tri[1]];
            let c = self.positions[tri[2]];
            if ray_z_intersects_tri(p, a, b, c) {
                crossings += 1;
            }
        }
        (crossings % 2) == 1
    }

    /// Sample interior volume points on a sub-grid spacing `spacing`.
    pub fn sample_volume_points(&self, spacing: f32) -> Vec<[f32; 3]> {
        let (bmin, bmax) = self.bounding_box();
        let mut pts = Vec::new();
        let nx = ((bmax[0] - bmin[0]) / spacing).ceil().max(1.0) as usize;
        let ny = ((bmax[1] - bmin[1]) / spacing).ceil().max(1.0) as usize;
        let nz = ((bmax[2] - bmin[2]) / spacing).ceil().max(1.0) as usize;
        for k in 0..nz {
            let z = bmin[2] + (k as f32 + 0.5) * spacing;
            for j in 0..ny {
                let y = bmin[1] + (j as f32 + 0.5) * spacing;
                for i in 0..nx {
                    let x = bmin[0] + (i as f32 + 0.5) * spacing;
                    let pt = [x, y, z];
                    if self.contains_point(pt) {
                        pts.push(pt);
                    }
                }
            }
        }
        pts
    }
}

fn resolve_obj_index(raw: isize, len: usize, line: usize) -> Result<usize> {
    let idx = if raw > 0 {
        (raw - 1) as isize
    } else if raw < 0 {
        len as isize + raw
    } else {
        return Err(ObjError::Parse {
            line,
            msg: "OBJ indices are 1-based; 0 is invalid".to_string(),
        });
    };
    if idx < 0 || (idx as usize) >= len {
        return Err(ObjError::Parse {
            line,
            msg: format!("OBJ index {raw} out of bounds for length {len}"),
        });
    }
    Ok(idx as usize)
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn ray_z_intersects_tri(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> bool {
    // Perturb query slightly away from exact edge degeneracies
    let px = p[0] + 1.37e-7;
    let py = p[1] + 2.93e-7;

    let ax = a[0] - px;
    let ay = a[1] - py;
    let bx = b[0] - px;
    let by = b[1] - py;
    let cx = c[0] - px;
    let cy = c[1] - py;

    let d0 = ax * by - ay * bx;
    let d1 = bx * cy - by * cx;
    let d2 = cx * ay - cy * ax;
    let same_sign = (d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0) || (d0 <= 0.0 && d1 <= 0.0 && d2 <= 0.0);
    if !same_sign {
        return false;
    }
    let area = d0 + d1 + d2;
    if area.abs() < 1e-18 {
        return false;
    }
    // Barycentric weights for vertex Z
    let z_hit = (d1 * a[2] + d2 * b[2] + d0 * c[2]) / area;
    z_hit > p[2]
}
