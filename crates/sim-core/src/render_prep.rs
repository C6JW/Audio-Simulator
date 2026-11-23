use std::collections::HashMap;
use std::fs;
use std::path::Path;

use sim_io::{
    extract_surface_nets_isosurface, write_ply_mesh, write_pressure_slice_f32,
    write_render_manifest, ProbeAcousticSummary, RenderBodyManifest, RenderManifest,
    RenderMicManifest, RenderSourceManifest,
};

use crate::engine::{MultiphysicsEngine, RenderExportConfig, Result};
use crate::geometry::{generate_shape_surface_mesh, GeometryShape};
use crate::scene::{SolidBodySpec, SolidBoundaryCondition};

pub struct PreparedBulkMesh {
    pub body_idx: usize,
    pub rest_verts: Vec<[f32; 3]>,
    pub tris: Vec<[u32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub bindings: Vec<Vec<(usize, f32)>>,
}

pub struct PreparedShellMesh {
    pub body_idx: usize,
    pub local_to_global: Vec<usize>,
    pub local_uvs: Vec<[f32; 2]>,
    pub local_tris: Vec<([u32; 3], [f32; 3])>,
}

pub struct PreparedRenderExporter {
    pub body_manifests: Vec<RenderBodyManifest>,
    pub bulk_meshes: Vec<PreparedBulkMesh>,
    pub shell_meshes: Vec<PreparedShellMesh>,
    pub water_bounds: Option<([f32; 3], [f32; 3])>,
    pub floor_z: Option<f32>,
    pub pslice_max_abs_p: f32,
}

pub fn classify_material_preset(body: &SolidBodySpec) -> &'static str {
    if body.is_kinematic_rigid {
        "boot_sole"
    } else if body.material.is_shell && body.material.wetness > 0.1 {
        "wet_leaf"
    } else if body.material.is_shell {
        "cellulose_paper"
    } else if body.material.is_rigid_static {
        "granite_stone"
    } else if body.material.e1 > 5.0e10 {
        "structural_steel"
    } else {
        "oak_wood"
    }
}

fn build_shepard_bindings(
    verts: &[[f32; 3]],
    p_rest: &[[f32; 3]],
    p_rest_mat: &[i32],
    mat_id: i32,
    spacing: f32,
) -> Vec<Vec<(usize, f32)>> {
    let candidates: Vec<(usize, [f32; 3])> = p_rest
        .iter()
        .enumerate()
        .filter(|&(p, _)| p_rest_mat[p] == mat_id)
        .map(|(p, &pos)| (p, pos))
        .collect();

    let h2 = (0.5 * spacing.max(1e-4)).powi(2);
    let k_max = 8usize;
    let mut out = Vec::with_capacity(verts.len());

    for &v in verts {
        if candidates.is_empty() {
            out.push(Vec::new());
            continue;
        }
        let mut best: Vec<(usize, f32)> = Vec::with_capacity(k_max + 1);
        for &(pi, pos) in &candidates {
            let dx = v[0] - pos[0];
            let dy = v[1] - pos[1];
            let dz = v[2] - pos[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if best.len() < k_max {
                best.push((pi, d2));
                if best.len() == k_max {
                    best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                }
            } else if d2 < best[k_max - 1].1 {
                best[k_max - 1] = (pi, d2);
                best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            }
        }
        if best.len() < k_max {
            best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        }
        let mut weights: Vec<(usize, f32)> =
            best.into_iter().map(|(pi, d2)| (pi, 1.0 / (d2 + h2))).collect();
        let w_sum: f32 = weights.iter().map(|x| x.1).sum();
        if w_sum > 0.0 {
            for w in &mut weights {
                w.1 /= w_sum;
            }
        }
        out.push(weights);
    }
    out
}

#[inline]
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

impl PreparedRenderExporter {
    pub fn new(engine: &MultiphysicsEngine, rcfg: &RenderExportConfig) -> Result<Self> {
        fs::create_dir_all(&rcfg.frames_dir)?;

        let mut body_manifests = Vec::with_capacity(engine.scene.solids.len());
        let mut bulk_meshes = Vec::new();
        let mut shell_meshes = Vec::new();

        for (idx, body) in engine.scene.solids.iter().enumerate() {
            let preset = classify_material_preset(body).to_string();
            let mut clamps = Vec::new();
            let mut pulls = Vec::new();
            for bc in &body.bcs {
                match bc {
                    SolidBoundaryCondition::ClampedRegion { min, max } => {
                        clamps.push((*min, *max));
                    }
                    SolidBoundaryCondition::PrescribedVelocityRegion { min, max, .. } => {
                        pulls.push((*min, *max));
                    }
                }
            }

            let mut rest_mesh_file = None;
            if body.material.is_rigid_static {
                let (verts, tris, uvs) = generate_shape_surface_mesh(&body.shape, 0.0025);
                let fname = format!("body_{idx}_static.ply");
                write_ply_mesh(
                    rcfg.frames_dir.join(&fname),
                    &verts,
                    &tris,
                    Some(&uvs),
                    None,
                )?;
                rest_mesh_file = Some(fname);
            } else if body.material.is_shell {
                if let Some(ref sh) = engine.shell_host {
                    let mat_id = idx as i32;
                    let mut g2l: HashMap<usize, u32> = HashMap::new();
                    let mut local_to_global = Vec::new();
                    let mut local_uvs = Vec::new();
                    let mut local_tris = Vec::new();

                    let mut intern = |g: usize| -> u32 {
                        if let Some(&loc) = g2l.get(&g) {
                            loc
                        } else {
                            let loc = local_to_global.len() as u32;
                            g2l.insert(g, loc);
                            local_to_global.push(g);
                            local_uvs.push(engine.p_rest_uv.get(g).copied().unwrap_or([0.0, 0.0]));
                            loc
                        }
                    };

                    for t in 0..sh.tri_v0.len() {
                        if sh.tri_mat_id[t] != mat_id {
                            continue;
                        }
                        let g0 = sh.tri_v0[t] as usize;
                        let g1 = sh.tri_v1[t] as usize;
                        let g2 = sh.tri_v2[t] as usize;
                        let l01 = dist3(engine.p_rest[g0], engine.p_rest[g1]);
                        let l12 = dist3(engine.p_rest[g1], engine.p_rest[g2]);
                        let l20 = dist3(engine.p_rest[g2], engine.p_rest[g0]);
                        let v0 = intern(g0);
                        let v1 = intern(g1);
                        let v2 = intern(g2);
                        local_tris.push(([v0, v1, v2], [l01, l12, l20]));
                    }

                    shell_meshes.push(PreparedShellMesh {
                        body_idx: idx,
                        local_to_global,
                        local_uvs,
                        local_tris,
                    });
                }
            } else {
                let (verts, tris, uvs) = generate_shape_surface_mesh(&body.shape, 0.0025);
                let bindings = build_shepard_bindings(
                    &verts,
                    &engine.p_rest,
                    &engine.p_rest_mat,
                    idx as i32,
                    body.particle_spacing,
                );
                bulk_meshes.push(PreparedBulkMesh {
                    body_idx: idx,
                    rest_verts: verts,
                    tris,
                    uvs,
                    bindings,
                });
            }

            body_manifests.push(RenderBodyManifest {
                name: body.name.clone(),
                body_id: body.body_id,
                material_preset: preset,
                thickness_m: body.material.thickness,
                wetness: body.material.wetness,
                is_static: body.material.is_rigid_static,
                is_shell: body.material.is_shell,
                rest_mesh_file,
                clamps,
                pulls,
            });
        }

        let water_bounds = if rcfg.export_fluid_isosurface && engine.levels.len() >= 2 {
            let s = &engine.levels[engine.levels.len() - 1].spec;
            Some((
                [-0.0096, -0.0096, s.oz],
                [0.0096, 0.0096, s.oz + (s.nz as f32) * s.dx],
            ))
        } else {
            None
        };

        let floor_z = engine
            .scene
            .mpm_grid
            .as_ref()
            .filter(|g| g.has_floor)
            .map(|g| g.floor_z);

        Ok(Self {
            body_manifests,
            bulk_meshes,
            shell_meshes,
            water_bounds,
            floor_z,
            pslice_max_abs_p: 0.0,
        })
    }

    pub fn export_frame(
        &mut self,
        engine: &MultiphysicsEngine,
        frame_idx: usize,
        rcfg: &RenderExportConfig,
    ) -> Result<()> {
        engine.stream.sync()?;

        if let Some(ref pts) = engine.particles {
            let px = pts.px.to_vec()?;
            let py = pts.py.to_vec()?;
            let pz = pts.pz.to_vec()?;
            let dmg = pts.damage.to_vec()?;

            for bulk in &self.bulk_meshes {
                let mut def_verts = Vec::with_capacity(bulk.rest_verts.len());
                let mut vert_dmg = Vec::with_capacity(bulk.rest_verts.len());
                for (vi, &v0) in bulk.rest_verts.iter().enumerate() {
                    let binds = &bulk.bindings[vi];
                    if binds.is_empty() {
                        def_verts.push(v0);
                        vert_dmg.push(0.0);
                        continue;
                    }
                    let mut disp = [0.0f32; 3];
                    let mut d_sum = 0.0f32;
                    for &(pi, w) in binds {
                        let r0 = engine.p_rest[pi];
                        disp[0] += w * (px[pi] - r0[0]);
                        disp[1] += w * (py[pi] - r0[1]);
                        disp[2] += w * (pz[pi] - r0[2]);
                        d_sum += w * dmg[pi];
                    }
                    def_verts.push([v0[0] + disp[0], v0[1] + disp[1], v0[2] + disp[2]]);
                    vert_dmg.push(d_sum);
                }
                let path = rcfg
                    .frames_dir
                    .join(format!("frame_{frame_idx:04}_body_{}.ply", bulk.body_idx));
                write_ply_mesh(
                    path,
                    &def_verts,
                    &bulk.tris,
                    Some(&bulk.uvs),
                    Some(&vert_dmg),
                )?;
            }

            for sh in &self.shell_meshes {
                let mut verts = Vec::with_capacity(sh.local_to_global.len());
                let mut vert_dmg = Vec::with_capacity(sh.local_to_global.len());
                for &g in &sh.local_to_global {
                    verts.push([px[g], py[g], pz[g]]);
                    vert_dmg.push(dmg[g]);
                }
                let mut active_tris = Vec::with_capacity(sh.local_tris.len());
                for &(tri, rest_lens) in &sh.local_tris {
                    let d0 = vert_dmg[tri[0] as usize];
                    let d1 = vert_dmg[tri[1] as usize];
                    let d2 = vert_dmg[tri[2] as usize];
                    let avg_d = (d0 + d1 + d2) / 3.0;
                    let p0 = verts[tri[0] as usize];
                    let p1 = verts[tri[1] as usize];
                    let p2 = verts[tri[2] as usize];
                    let s0 = dist3(p0, p1) / rest_lens[0].max(1e-6);
                    let s1 = dist3(p1, p2) / rest_lens[1].max(1e-6);
                    let s2 = dist3(p2, p0) / rest_lens[2].max(1e-6);
                    let max_stretch = s0.max(s1).max(s2);
                    // Only open gaps where fractured vertices physically separate, so crushed fragments stay visible
                    if max_stretch > 1.85 || (avg_d >= 0.90 && max_stretch > 1.18) {
                        continue;
                    }
                    active_tris.push(tri);
                }
                let path = rcfg
                    .frames_dir
                    .join(format!("frame_{frame_idx:04}_body_{}.ply", sh.body_idx));
                write_ply_mesh(
                    path,
                    &verts,
                    &active_tris,
                    Some(&sh.local_uvs),
                    Some(&vert_dmg),
                )?;
            }
        }

        if rcfg.export_fluid_isosurface && engine.levels.len() >= 2 {
            let l2 = &engine.levels[engine.levels.len() - 1];
            let l1 = &engine.levels[engine.levels.len() - 2];
            if let (Some(a2_buf), Some(a1_buf)) = (&l2.alpha, &l1.alpha) {
            let alpha_l2 = a2_buf.to_vec()?;
            let alpha_l1 = a1_buf.to_vec()?;

            let cnx = 96usize;
            let cny = 96usize;
            let cnz = l2.spec.nz;
            let dx = l2.spec.dx;
            let origin = [-0.0096_f32, -0.0096_f32, l2.spec.oz];
            let off_i = ((l2.spec.ox - origin[0]) / dx).round() as isize;
            let off_j = ((l2.spec.oy - origin[1]) / dx).round() as isize;

            let nxp1 = l1.nxp;
            let nyp1 = l1.nyp;
            let nzp1 = l1.nzp;
            let nxp2 = l2.nxp;
            let nyp2 = l2.nyp;
            let h1 = l1.halo as f32;
            let h2 = l2.halo;

            let sample_l1 = |x: f32, y: f32, z: f32| -> f32 {
                let fx = ((x - l1.spec.ox) / l1.spec.dx - 0.5 + h1).clamp(0.0, (nxp1 - 1) as f32);
                let fy = ((y - l1.spec.oy) / l1.spec.dx - 0.5 + h1).clamp(0.0, (nyp1 - 1) as f32);
                let fz = ((z - l1.spec.oz) / l1.spec.dx - 0.5 + h1).clamp(0.0, (nzp1 - 1) as f32);
                let i0 = (fx.floor() as usize).min(nxp1 - 2);
                let j0 = (fy.floor() as usize).min(nyp1 - 2);
                let k0 = (fz.floor() as usize).min(nzp1 - 2);
                let tx = (fx - i0 as f32).clamp(0.0, 1.0);
                let ty = (fy - j0 as f32).clamp(0.0, 1.0);
                let tz = (fz - k0 as f32).clamp(0.0, 1.0);
                let mut acc = 0.0f32;
                for dk in 0..=1 {
                    let wz = if dk == 1 { tz } else { 1.0 - tz };
                    for dj in 0..=1 {
                        let wy = if dj == 1 { ty } else { 1.0 - ty };
                        for di in 0..=1 {
                            let wx = if di == 1 { tx } else { 1.0 - tx };
                            let cid = (i0 + di) + nxp1 * ((j0 + dj) + nyp1 * (k0 + dk));
                            acc += wx * wy * wz * alpha_l1[cid];
                        }
                    }
                }
                acc
            };

            let mut comp_alpha = vec![0.0f32; cnx * cny * cnz];
            for k in 0..cnz {
                let z = origin[2] + (k as f32 + 0.5) * dx;
                for j in 0..cny {
                    let y = origin[1] + (j as f32 + 0.5) * dx;
                    let lj = j as isize - off_j;
                    for i in 0..cnx {
                        let x = origin[0] + (i as f32 + 0.5) * dx;
                        let li = i as isize - off_i;
                        let a_val = if li >= 0
                            && li < l2.spec.nx as isize
                            && lj >= 0
                            && lj < l2.spec.ny as isize
                        {
                            let d_edge = li
                                .min(l2.spec.nx as isize - 1 - li)
                                .min(lj)
                                .min(l2.spec.ny as isize - 1 - lj);
                            let cid2 = (li as usize + h2)
                                + nxp2 * ((lj as usize + h2) + nyp2 * (k + h2));
                            let a_l2 = alpha_l2[cid2];
                            if d_edge >= 2 {
                                a_l2
                            } else {
                                let a_l1 = sample_l1(x, y, z);
                                let w = if d_edge == 1 { 0.67 } else { 0.33 };
                                w * a_l2 + (1.0 - w) * a_l1
                            }
                        } else {
                            sample_l1(x, y, z)
                        };
                        comp_alpha[i + cnx * (j + cny * k)] = a_val;
                    }
                }
            }

            let (verts, tris) = extract_surface_nets_isosurface(
                &comp_alpha,
                cnx,
                cny,
                cnz,
                origin,
                dx,
                0.5,
                true,
                2,
            );
            let path = rcfg
                .frames_dir
                .join(format!("frame_{frame_idx:04}_water.ply"));
            write_ply_mesh(path, &verts, &tris, None, None)?;
            }
        }

        if rcfg.export_pressure_slice {
            let l0 = &engine.levels[0];
            if let Some(pg_buf) = &l0.p_gauge {
            let h0 = l0.halo;
            let nx = l0.spec.nx;
            let ny = l0.spec.ny;
            let k_rel = ((rcfg.pressure_slice_z - l0.spec.oz) / l0.spec.dx - 0.5)
                .round()
                .clamp(0.0, (l0.spec.nz.saturating_sub(1)) as f32) as usize;
            let k_idx = h0 + k_rel;
            let actual_z = l0.spec.oz + (k_rel as f32 + 0.5) * l0.spec.dx;

            let p_g = pg_buf.to_vec()?;
            let mut slice = vec![0.0f32; nx * ny];
            let mut max_abs_p = 0.0f32;
            for j in 0..ny {
                for i in 0..nx {
                    let cid = (h0 + i) + l0.nxp * ((h0 + j) + l0.nyp * k_idx);
                    let v = p_g[cid];
                    max_abs_p = max_abs_p.max(v.abs());
                    slice[i + nx * j] = v;
                }
            }
            self.pslice_max_abs_p = self.pslice_max_abs_p.max(max_abs_p);
            let path = rcfg
                .frames_dir
                .join(format!("frame_{frame_idx:04}_pslice.bin"));
            write_pressure_slice_f32(
                path,
                nx,
                ny,
                l0.spec.ox,
                l0.spec.oy,
                l0.spec.dx,
                actual_z,
                max_abs_p,
                &slice,
            )?;
            }
        }

        Ok(())
    }

    pub fn write_manifest(
        self,
        engine: &MultiphysicsEngine,
        rcfg: &RenderExportConfig,
        duration_s: f64,
        pre_roll_s: f64,
        probe_summaries: &[ProbeAcousticSummary],
        out_dir: Option<&Path>,
    ) -> Result<()> {
        let microphones = engine
            .scene
            .microphones
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let wav_path = probe_summaries
                    .get(i)
                    .and_then(|s| s.wav.as_ref().map(|w| w.path.clone()))
                    .or_else(|| {
                        out_dir.map(|d| {
                            d.join(format!("{}_{}.wav", engine.scene.name, m.name))
                                .to_string_lossy()
                                .into_owned()
                        })
                    });
                RenderMicManifest {
                    name: m.name.clone(),
                    position: m.position,
                    is_hydrophone: m.is_hydrophone,
                    wav_path,
                }
            })
            .collect();

        let mut point_sources: Vec<RenderSourceManifest> = engine
            .scene
            .point_sources
            .iter()
            .map(|s| RenderSourceManifest {
                center: s.center,
                t0_s: s.t0_s,
                f0_hz: s.f0_hz,
            })
            .collect();
        for s in &engine.scene.directional_sources {
            point_sources.push(RenderSourceManifest {
                center: s.center,
                t0_s: s.t0_s,
                f0_hz: 1000.0,
            });
        }

        let mut pslice_r_max = 0.0_f32;
        for b in &engine.scene.solids {
            match &b.shape {
                GeometryShape::FrustumRing {
                    r_inner_bottom,
                    r_inner_top,
                    ..
                } => {
                    pslice_r_max = pslice_r_max.max(r_inner_bottom.max(*r_inner_top));
                }
                GeometryShape::CylindricalRing { r_inner, .. } => {
                    pslice_r_max = pslice_r_max.max(*r_inner);
                }
                _ => {}
            }
        }

        let manifest = RenderManifest {
            scenario: engine.scene.name.clone(),
            duration_s,
            pre_roll_s,
            num_frames: rcfg.num_frames,
            has_fluid_isosurface: rcfg.export_fluid_isosurface,
            has_pressure_slice: rcfg.export_pressure_slice,
            pressure_slice_z: rcfg.pressure_slice_z,
            pslice_r_max,
            pslice_max_abs_p: self.pslice_max_abs_p,
            water_bounds: self.water_bounds,
            floor_z: self.floor_z,
            bodies: self.body_manifests,
            microphones,
            point_sources,
        };
        write_render_manifest(rcfg.frames_dir.join("manifest.json"), &manifest)?;
        Ok(())
    }
}
