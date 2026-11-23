import argparse
import json
import math
import os
import struct
import sys
import time

import bpy
import numpy as np
from mathutils import Euler, Vector


def format_duration_hms(secs):
    if not math.isfinite(secs) or secs <= 0.0:
        return "0.0s"
    if secs < 60.0:
        return f"{secs:.1f}s"
    total = int(round(secs))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-frames", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--only-frames", default="")
    return ap.parse_args(argv)


def read_ply_binary(path):
    with open(path, "rb") as f:
        n_verts = 0
        n_faces = 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex "):
                n_verts = int(line.split()[-1])
            elif line.startswith("element face "):
                n_faces = int(line.split()[-1])
            elif line == "end_header":
                break
        v_raw = np.frombuffer(f.read(n_verts * 24), dtype="<f4")
        f_raw = np.frombuffer(f.read(n_faces * 13), dtype=np.dtype([("cnt", "u1"), ("idx", "<3i4")]))
    if n_verts == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    arr = np.ascontiguousarray(v_raw.reshape((n_verts, 6)), dtype=np.float32)
    coords = np.ascontiguousarray(arr[:, 0:3], dtype=np.float32)
    uvs = np.ascontiguousarray(arr[:, 3:5], dtype=np.float32)
    damage = np.ascontiguousarray(arr[:, 5], dtype=np.float32)
    tris = np.ascontiguousarray(f_raw["idx"], dtype=np.int32) if n_faces > 0 else np.zeros((0, 3), dtype=np.int32)
    return coords, uvs, damage, tris


def read_pslice_bin(path):
    with open(path, "rb") as f:
        hdr = f.read(28)
        nx, ny, ox, oy, dx, z, max_abs_p = struct.unpack("<IIfffff", hdr)
        vals = np.frombuffer(f.read(nx * ny * 4), dtype="<f4").reshape((ny, nx)).copy()
    return nx, ny, ox, oy, dx, z, max_abs_p, vals


def append_orphan_shell_shards(coords, uvs, damage, tris, body_seed, radius=0.0021):
    n = len(coords)
    if n == 0:
        return coords, uvs, damage, tris
    used = np.zeros(n, dtype=bool)
    if len(tris) > 0:
        used[tris.ravel()] = True
    orphans = np.nonzero(~used)[0]
    if len(orphans) == 0:
        return coords, uvs, damage, tris

    # Construct deterministic jagged 4-vertex brittle leaf shards at each disconnected particle
    m = len(orphans)
    ang = (orphans.astype(np.float32) * 2.39996 + float(body_seed) * 1.7) % (2.0 * math.pi)
    tilt = 0.22 * np.sin(orphans.astype(np.float32) * 4.13 + float(body_seed))
    ca, sa = np.cos(ang), np.sin(ang)
    t1 = np.stack([ca, sa, tilt], axis=1)
    t2 = np.stack([-sa, ca, -0.5 * tilt], axis=1)
    r_var = radius * (0.75 + 0.45 * ((np.sin(orphans.astype(np.float32) * 12.9898) * 43758.5453) % 1.0))

    c_o = coords[orphans]
    v0 = c_o + (r_var[:, None] * 0.9) * t1
    v1 = c_o + (r_var[:, None] * 0.85) * t2
    v2 = c_o - (r_var[:, None] * 0.95) * t1
    v3 = c_o - (r_var[:, None] * 0.75) * t2
    # Prevent shard corners from poking cleanly below floor z = 0
    for vk in (v0, v1, v2, v3):
        vk[:, 2] = np.maximum(vk[:, 2], 0.00018)

    shard_coords = np.empty((m * 4, 3), dtype=np.float32)
    shard_coords[0::4] = v0
    shard_coords[1::4] = v1
    shard_coords[2::4] = v2
    shard_coords[3::4] = v3

    shard_uvs = np.repeat(uvs[orphans], 4, axis=0)
    shard_dmg = np.repeat(damage[orphans], 4, axis=0)

    base = n
    idx4 = base + np.arange(m, dtype=np.int32) * 4
    shard_tris = np.empty((m * 2, 3), dtype=np.int32)
    shard_tris[0::2] = np.stack([idx4, idx4 + 1, idx4 + 2], axis=1)
    shard_tris[1::2] = np.stack([idx4, idx4 + 2, idx4 + 3], axis=1)

    new_coords = np.concatenate([coords, shard_coords], axis=0)
    new_uvs = np.concatenate([uvs, shard_uvs], axis=0)
    new_dmg = np.concatenate([damage, shard_dmg], axis=0)
    new_tris = np.concatenate([tris, shard_tris], axis=0) if len(tris) > 0 else shard_tris
    return new_coords, new_uvs, new_dmg, new_tris


def populate_mesh_from_arrays(mesh, coords, tris, uvs=None, damage=None, smooth=True):
    mesh.clear_geometry()
    n_verts = len(coords)
    n_tris = len(tris)
    if n_verts == 0:
        return

    mesh.vertices.add(n_verts)
    mesh.vertices.foreach_set("co", coords.ravel())

    if n_tris > 0:
        mesh.loops.add(n_tris * 3)
        mesh.loops.foreach_set("vertex_index", tris.ravel())
        mesh.polygons.add(n_tris)
        mesh.polygons.foreach_set("loop_start", np.arange(0, n_tris * 3, 3, dtype=np.int32))
        mesh.polygons.foreach_set("loop_total", np.full(n_tris, 3, dtype=np.int32))
        if smooth:
            mesh.polygons.foreach_set("use_smooth", np.ones(n_tris, dtype=bool))

        if uvs is not None and len(uvs) == n_verts:
            if not mesh.uv_layers:
                mesh.uv_layers.new(name="UVMap")
            uv_layer = mesh.uv_layers.active.data
            loop_uvs = uvs[tris.ravel()].ravel()
            uv_layer.foreach_set("uv", loop_uvs)

    if damage is not None and len(damage) == n_verts:
        attr = mesh.attributes.get("damage")
        if attr is None:
            attr = mesh.attributes.new(name="damage", type="FLOAT", domain="POINT")
        elif len(attr.data) != n_verts:
            mesh.attributes.remove(attr)
            attr = mesh.attributes.new(name="damage", type="FLOAT", domain="POINT")
        attr.data.foreach_set("value", damage)

    mesh.update()


def configure_cycles_optix(width, height, samples, scenario):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for dev in prefs.devices:
        dev.use = (dev.type == "OPTIX")
    scene.cycles.device = "GPU"

    if scenario in ("water_drop_plop", "pouring_square_jar", "pouring_wine_glass", "sinking_bell_grotto"):
        scene.cycles.samples = max(samples, 96 if scenario != "water_drop_plop" else 112)
        scene.cycles.max_bounces = 16
        scene.cycles.diffuse_bounces = 3
        scene.cycles.glossy_bounces = 8
        scene.cycles.transmission_bounces = 16
        scene.cycles.transparent_max_bounces = 8
        scene.cycles.volume_bounces = 2
        scene.cycles.sample_clamp_indirect = 12.0
    else:
        scene.cycles.samples = samples
        scene.cycles.max_bounces = 8
        scene.cycles.diffuse_bounces = 3
        scene.cycles.glossy_bounces = 4
        scene.cycles.transmission_bounces = 6
        scene.cycles.transparent_max_bounces = 8

    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.use_denoising = True

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium Contrast"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0


def new_mat(name):
    m = bpy.data.materials.new(name=name)
    if m.node_tree is None:
        m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m, nt, bsdf, out


def add_noise_bump(nt, bsdf, scale, strength, detail=8.0, roughness=0.6, tex_coord=None):
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = detail
    noise.inputs["Roughness"].default_value = roughness
    if tex_coord is not None:
        nt.links.new(tex_coord, noise.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = 0.0005
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return noise


def make_oak_wood_material():
    m, nt, bsdf, _ = new_mat("Mat_OakWood")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.5, 22.0, 22.0)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "RINGS"
    wave.rings_direction = "X"
    wave.inputs["Scale"].default_value = 18.0
    wave.inputs["Distortion"].default_value = 6.5
    wave.inputs["Detail"].default_value = 6.0
    nt.links.new(mp.outputs["Vector"], wave.inputs["Vector"])

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 140.0
    noise.inputs["Detail"].default_value = 8.0
    nt.links.new(mp.outputs["Vector"], noise.inputs["Vector"])

    mix_f = nt.nodes.new("ShaderNodeMix")
    mix_f.data_type = "FLOAT"
    mix_f.inputs["Factor"].default_value = 0.28
    nt.links.new(wave.outputs["Fac"], mix_f.inputs["A"])
    nt.links.new(noise.outputs["Fac"], mix_f.inputs["B"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.15
    ramp.color_ramp.elements[0].color = (0.19, 0.10, 0.045, 1.0)
    ramp.color_ramp.elements[1].position = 0.85
    ramp.color_ramp.elements[1].color = (0.48, 0.30, 0.14, 1.0)
    nt.links.new(mix_f.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.34
    bsdf.inputs["Coat Weight"].default_value = 0.25
    bsdf.inputs["Coat Roughness"].default_value = 0.18

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = 0.0003
    nt.links.new(mix_f.outputs["Result"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def make_steel_material(name="Mat_Steel", polished=False, base=(0.66, 0.69, 0.73, 1.0)):
    m, nt, bsdf, _ = new_mat(name)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Base Color"].default_value = base
    if polished:
        bsdf.inputs["Roughness"].default_value = 0.055
    else:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (4.0, 60.0, 20.0)
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 40.0
        noise.inputs["Detail"].default_value = 6.0
        nt.links.new(mp.outputs["Vector"], noise.inputs["Vector"])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = 0.16
        mr.inputs["To Max"].default_value = 0.26
        nt.links.new(noise.outputs["Fac"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])
    return m


def make_granite_material():
    m, nt, bsdf, _ = new_mat("Mat_Granite")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n1 = nt.nodes.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 65.0
    n1.inputs["Detail"].default_value = 12.0
    n1.inputs["Roughness"].default_value = 0.75
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    e0 = ramp.color_ramp.elements[0]
    e1 = ramp.color_ramp.elements[1]
    e0.position = 0.28
    e0.color = (0.08, 0.085, 0.095, 1.0)
    e1.position = 0.78
    e1.color = (0.56, 0.54, 0.50, 1.0)
    e_mid = ramp.color_ramp.elements.new(0.55)
    e_mid.color = (0.28, 0.27, 0.26, 1.0)
    nt.links.new(n1.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.56
    add_noise_bump(nt, bsdf, 180.0, 0.14, detail=10.0, tex_coord=tc.outputs["Object"])
    return m


def make_paper_material():
    m, nt, bsdf, _ = new_mat("Mat_CellulosePaper")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n1 = nt.nodes.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 260.0
    n1.inputs["Detail"].default_value = 10.0
    n1.inputs["Roughness"].default_value = 0.65
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])

    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "damage"

    ramp_base = nt.nodes.new("ShaderNodeValToRGB")
    ramp_base.color_ramp.elements[0].position = 0.25
    ramp_base.color_ramp.elements[0].color = (0.87, 0.82, 0.72, 1.0)
    ramp_base.color_ramp.elements[1].position = 0.75
    ramp_base.color_ramp.elements[1].color = (0.95, 0.92, 0.85, 1.0)
    nt.links.new(n1.outputs["Fac"], ramp_base.inputs["Fac"])

    mix_dmg = nt.nodes.new("ShaderNodeMix")
    mix_dmg.data_type = "RGBA"
    mix_dmg.inputs["B"].default_value = (0.72, 0.64, 0.52, 1.0)
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.2
    mr.inputs["From Max"].default_value = 0.85
    mr.clamp = True
    nt.links.new(attr.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], mix_dmg.inputs["Factor"])
    nt.links.new(ramp_base.outputs["Color"], mix_dmg.inputs["A"])
    nt.links.new(mix_dmg.outputs["Result"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.84
    bsdf.inputs["Subsurface Weight"].default_value = 0.18
    bsdf.inputs["Subsurface Scale"].default_value = 0.001
    bsdf.inputs["Sheen Weight"].default_value = 0.15
    add_noise_bump(nt, bsdf, 420.0, 0.12, detail=8.0, tex_coord=tc.outputs["Object"])
    return m


def make_wet_leaf_material(idx):
    palettes = [
        ((0.26, 0.11, 0.025, 1.0), (0.58, 0.28, 0.05, 1.0)),
        ((0.20, 0.04, 0.02, 1.0), (0.45, 0.10, 0.04, 1.0)),
        ((0.28, 0.21, 0.04, 1.0), (0.56, 0.42, 0.08, 1.0)),
        ((0.16, 0.15, 0.045, 1.0), (0.34, 0.30, 0.09, 1.0)),
        ((0.32, 0.13, 0.02, 1.0), (0.64, 0.32, 0.06, 1.0)),
        ((0.19, 0.09, 0.03, 1.0), (0.40, 0.20, 0.07, 1.0)),
    ]
    c_lo, c_hi = palettes[idx % len(palettes)]
    m, nt, bsdf, _ = new_mat(f"Mat_WetLeaf_{idx}")
    tc = nt.nodes.new("ShaderNodeTexCoord")

    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = 14.0
    nt.links.new(tc.outputs["UV"], vor.inputs["Vector"])

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 7.5
    noise.inputs["Detail"].default_value = 10.0
    noise.inputs["Roughness"].default_value = 0.72
    nt.links.new(tc.outputs["UV"], noise.inputs["Vector"])

    mix_f = nt.nodes.new("ShaderNodeMix")
    mix_f.data_type = "FLOAT"
    mix_f.inputs["Factor"].default_value = 0.75
    nt.links.new(vor.outputs["Distance"], mix_f.inputs["A"])
    nt.links.new(noise.outputs["Fac"], mix_f.inputs["B"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.22
    ramp.color_ramp.elements[0].color = c_lo
    ramp.color_ramp.elements[1].position = 0.80
    ramp.color_ramp.elements[1].color = c_hi
    nt.links.new(mix_f.outputs["Result"], ramp.inputs["Fac"])

    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "damage"
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.25
    mr.inputs["From Max"].default_value = 0.95
    mr.clamp = True
    nt.links.new(attr.outputs["Fac"], mr.inputs["Value"])

    mix_dmg = nt.nodes.new("ShaderNodeMix")
    mix_dmg.data_type = "RGBA"
    mix_dmg.inputs["B"].default_value = (0.08, 0.04, 0.02, 1.0)
    nt.links.new(mr.outputs["Result"], mix_dmg.inputs["Factor"])
    nt.links.new(ramp.outputs["Color"], mix_dmg.inputs["A"])
    nt.links.new(mix_dmg.outputs["Result"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.34
    bsdf.inputs["Subsurface Weight"].default_value = 0.14
    bsdf.inputs["Subsurface Scale"].default_value = 0.0005
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Coat Roughness"].default_value = 0.032
    bsdf.inputs["Coat IOR"].default_value = 1.333

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.20
    bump.inputs["Distance"].default_value = 0.0004
    nt.links.new(mix_f.outputs["Result"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def make_water_material():
    m, nt, bsdf, out = new_mat("Mat_PureWater")
    bsdf.inputs["Base Color"].default_value = (0.98, 0.995, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.002
    bsdf.inputs["IOR"].default_value = 1.333
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    vol_abs = nt.nodes.new("ShaderNodeVolumeAbsorption")
    vol_abs.inputs["Color"].default_value = (0.76, 0.93, 0.99, 1.0)
    vol_abs.inputs["Density"].default_value = 28.0
    nt.links.new(vol_abs.outputs["Volume"], out.inputs["Volume"])
    return m


def make_cuvette_glass_material():
    m, nt, bsdf, _ = new_mat("Mat_GlassCuvette")
    bsdf.inputs["Base Color"].default_value = (0.98, 0.995, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.005
    bsdf.inputs["IOR"].default_value = 1.50
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    return m


def make_vessel_glass_material():
    m, nt, bsdf, _ = new_mat("Mat_VesselGlass")
    bsdf.inputs["Base Color"].default_value = (0.97, 0.99, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.015
    bsdf.inputs["IOR"].default_value = 1.52
    bsdf.inputs["Transmission Weight"].default_value = 0.98
    bsdf.inputs["Coat Weight"].default_value = 0.35
    bsdf.inputs["Coat Roughness"].default_value = 0.01
    return m


def make_bronze_material():
    m, nt, bsdf, _ = new_mat("Mat_BellBronze")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 55.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.6
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[0].color = (0.58, 0.41, 0.19, 1.0)
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = (0.80, 0.61, 0.31, 1.0)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Metallic"].default_value = 1.0
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = 0.14
    mr.inputs["To Max"].default_value = 0.28
    nt.links.new(noise.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])
    add_noise_bump(nt, bsdf, 140.0, 0.04, detail=6.0, tex_coord=tc.outputs["Object"])
    return m


def make_concrete_material():
    m, nt, bsdf, _ = new_mat("Mat_Concrete")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 6.0
    noise.inputs["Detail"].default_value = 10.0
    noise.inputs["Roughness"].default_value = 0.72
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[0].color = (0.20, 0.19, 0.17, 1.0)
    ramp.color_ramp.elements[1].position = 0.80
    ramp.color_ramp.elements[1].color = (0.42, 0.39, 0.35, 1.0)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.82
    add_noise_bump(nt, bsdf, 28.0, 0.18, detail=8.0, tex_coord=tc.outputs["Object"])
    return m


def make_boot_sole_material():
    m, nt, bsdf, _ = new_mat("Mat_BootSole")
    bsdf.inputs["Base Color"].default_value = (0.10, 0.10, 0.11, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.62
    return m


def make_dark_studio_floor_material(wet=False):
    m, nt, bsdf, _ = new_mat("Mat_StudioSurface")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 24.0
    noise.inputs["Detail"].default_value = 8.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[0].color = (0.04, 0.042, 0.048, 1.0)
    ramp.color_ramp.elements[1].position = 0.8
    ramp.color_ramp.elements[1].color = (0.09, 0.095, 0.105, 1.0)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    if wet:
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = 0.10
        mr.inputs["To Max"].default_value = 0.36
        nt.links.new(noise.outputs["Fac"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])
        bsdf.inputs["Coat Weight"].default_value = 0.65
        bsdf.inputs["Coat Roughness"].default_value = 0.05
    else:
        bsdf.inputs["Roughness"].default_value = 0.55
    add_noise_bump(nt, bsdf, 80.0, 0.06, detail=8.0, tex_coord=tc.outputs["Object"])
    return m


def make_schlieren_material():
    m, _, _, _ = new_mat("Mat_AcousticSchlieren")
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix_sh = nt.nodes.new("ShaderNodeMixShader")
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 3.2

    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "p_rgba"

    nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    nt.links.new(attr.outputs["Alpha"], mix_sh.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], mix_sh.inputs[1])
    nt.links.new(emit.outputs["Emission"], mix_sh.inputs[2])
    nt.links.new(mix_sh.outputs["Shader"], out.inputs["Surface"])
    return m


def add_area_light(name, loc, rot_deg, energy, size_x, size_y=None, color=(1.0, 1.0, 1.0)):
    ld = bpy.data.lights.new(name=name, type="AREA")
    ld.energy = energy
    ld.color = color
    ld.shape = "RECTANGLE" if size_y is not None else "SQUARE"
    ld.size = size_x
    if size_y is not None:
        ld.size_y = size_y
    ob = bpy.data.objects.new(name=name, object_data=ld)
    bpy.context.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = Euler(tuple(math.radians(a) for a in rot_deg), "XYZ")
    return ob


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_world_studio(bg_color=(0.018, 0.020, 0.024, 1.0), strength=0.35):
    world = bpy.data.worlds.new("StudioWorld")
    bpy.context.scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = bg_color
        bg.inputs["Strength"].default_value = strength


def add_box_mesh(name, bmin, bmax, mat):
    cx = 0.5 * (bmin[0] + bmax[0])
    cy = 0.5 * (bmin[1] + bmax[1])
    cz = 0.5 * (bmin[2] + bmax[2])
    sx = abs(bmax[0] - bmin[0])
    sy = abs(bmax[1] - bmin[1])
    sz = abs(bmax[2] - bmin[2])
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(cx, cy, cz))
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = (sx, sy, sz)
    bpy.ops.object.transform_apply(scale=True)
    bev = ob.modifiers.new(name="Bevel", type="BEVEL")
    bev.width = min(sx, sy, sz) * 0.08
    bev.segments = 2
    ob.data.materials.append(mat)
    return ob


def add_mic_prop(mic, scale_r=0.0022, mat_body=None, mat_cap=None):
    if mat_body is None:
        mat_body = make_steel_material("Mat_MicBody", polished=False, base=(0.14, 0.14, 0.16, 1.0))
    if mat_cap is None:
        mat_cap = make_steel_material("Mat_MicCap", polished=True, base=(0.78, 0.65, 0.35, 1.0))
    pos = Vector(mic["position"])
    # Place capsule pointing slightly inward/upward from sensor point so diaphragm touches mic.position
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=scale_r, depth=scale_r * 5.0, location=pos + Vector((0.0, 0.0, scale_r * 2.5)))
    ob = bpy.context.active_object
    ob.name = f"Mic_{mic['name']}"
    ob.data.materials.append(mat_body)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, radius=scale_r * 0.92, location=pos)
    cap = bpy.context.active_object
    cap.name = f"MicCap_{mic['name']}"
    cap.data.materials.append(mat_cap)
    return ob


def build_cuvette(water_bounds, glass_mat, floor_mat):
    bmin, bmax = water_bounds
    wx0, wy0, wz0 = bmin
    wx1, wy1, wz1 = bmax
    gap = 0.00008
    wall = 0.0007
    z_top = wz1 + 0.0032
    # Open-top optical glass cuvette: bottom slab + 4 crisp ultra-thin glass walls
    add_box_mesh("Cuvette_Bottom", (wx0 - gap - wall, wy0 - gap - wall, wz0 - gap - wall), (wx1 + gap + wall, wy1 + gap + wall, wz0 - gap), glass_mat)
    add_box_mesh("Cuvette_Wall_L", (wx0 - gap - wall, wy0 - gap - wall, wz0 - gap), (wx0 - gap, wy1 + gap + wall, z_top), glass_mat)
    add_box_mesh("Cuvette_Wall_R", (wx1 + gap, wy0 - gap - wall, wz0 - gap), (wx1 + gap + wall, wy1 + gap + wall, z_top), glass_mat)
    add_box_mesh("Cuvette_Wall_B", (wx0 - gap, wy1 + gap, wz0 - gap), (wx1 + gap, wy1 + gap + wall, z_top), glass_mat)
    add_box_mesh("Cuvette_Wall_F", (wx0 - gap, wy0 - gap - wall, wz0 - gap), (wx1 + gap, wy0 - gap, z_top), glass_mat)
    # Matte optical table underneath cuvette
    add_box_mesh("Lab_Bench", (-0.12, -0.12, wz0 - gap - wall - 0.005), (0.12, 0.12, wz0 - gap - wall - 0.00004), floor_mat)


def setup_scene_camera_and_lights(manifest):
    sc = manifest["scenario"]
    cam_data = bpy.data.cameras.new("MainCam")
    cam_obj = bpy.data.objects.new("MainCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    if sc.startswith("drop_wood") or sc.startswith("drop_steel"):
        setup_world_studio((0.012, 0.014, 0.018, 1.0), 0.14)
        bench_mat = make_dark_studio_floor_material(wet=False)
        add_box_mesh("Workbench", (-0.65, -0.60, -0.040), (0.65, 0.60, -0.022), bench_mat)
        clamp_mat = make_steel_material("Mat_ClampVice", polished=False, base=(0.24, 0.26, 0.29, 1.0))
        for body in manifest["bodies"]:
            for ci, (cmin, cmax) in enumerate(body.get("clamps", [])):
                # Solid bench vise pedestal + upper clamping block
                add_box_mesh(f"ViceBase_{ci}", (cmin[0] - 0.004, cmin[1] - 0.004, -0.022), (cmax[0] + 0.004, cmax[1] + 0.004, cmin[2]), clamp_mat)
                add_box_mesh(f"ViceJaw_{ci}", (cmin[0], cmin[1], cmin[2]), (cmax[0], cmax[1], cmax[2] + 0.004), clamp_mat)
        cam_obj.location = Vector((0.095, -0.155, 0.072))
        cam_data.lens = 48.0
        look_at(cam_obj, (0.0, 0.0, 0.004))
        add_area_light("Key", (0.14, -0.16, 0.24), (35, 18, 28), 3.2, 0.22, 0.18, (1.0, 0.96, 0.91))
        add_area_light("Fill", (-0.22, -0.10, 0.12), (55, 0, -58), 1.1, 0.30, 0.20, (0.82, 0.90, 1.0))
        add_area_light("Rim", (0.02, 0.18, 0.14), (-52, 0, 180), 2.4, 0.28, 0.12, (1.0, 0.98, 0.95))

    elif sc.startswith("whispering_gallery"):
        r_ref = float(manifest.get("pslice_r_max", 0.36) or 0.36)
        s = max(r_ref / 0.36, 1.0)
        setup_world_studio((0.010, 0.012, 0.016, 1.0), 0.14)
        floor_mat = make_dark_studio_floor_material(wet=False)
        add_box_mesh("GalleryPedestal", (-1.4 * s, -1.4 * s, -0.04 * s), (1.4 * s, 1.4 * s, -0.016 * s), floor_mat)
        cam_obj.location = Vector((0.40 * s, -0.64 * s, 0.66 * s))
        cam_data.lens = 35.0
        cam_data.clip_end = max(100.0, 40.0 * s)
        look_at(cam_obj, (0.0, 0.0, 0.03 * s))
        add_area_light("OverheadKey", (0.35 * s, -0.35 * s, 1.25 * s), (18, 18, 0), 42.0 * s * s, 0.9 * s, 0.9 * s, (1.0, 0.97, 0.93))
        add_area_light("CoolRim", (-0.65 * s, 0.55 * s, 0.80 * s), (-35, -30, 130), 22.0 * s * s, 0.8 * s, 0.5 * s, (0.80, 0.90, 1.0))
        for mic in manifest["microphones"]:
            add_mic_prop(mic, scale_r=0.0045 * s)

    elif sc == "water_drop_plop":
        setup_world_studio((0.025, 0.032, 0.042, 1.0), 0.16)
        if manifest.get("water_bounds"):
            build_cuvette(manifest["water_bounds"], make_cuvette_glass_material(), make_dark_studio_floor_material(wet=False))
        # High-contrast graduated optical backlight panel (dark edge -> bright center) so water crater & bubble rims refract sharply
        back_mat, nt, _, out = new_mat("Mat_BacklightPanel")
        nt.links.clear()
        tc = nt.nodes.new("ShaderNodeTexCoord")
        grad = nt.nodes.new("ShaderNodeTexGradient")
        grad.gradient_type = "QUADRATIC_SPHERE"
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Location"].default_value = (-1.0, -1.0, 0.0)
        mp.inputs["Scale"].default_value = (2.0, 2.0, 1.0)
        nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], grad.inputs["Vector"])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color = (0.06, 0.10, 0.16, 1.0)
        ramp.color_ramp.elements[1].position = 0.85
        ramp.color_ramp.elements[1].color = (0.95, 0.98, 1.0, 1.0)
        nt.links.new(grad.outputs["Fac"], ramp.inputs["Fac"])
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Strength"].default_value = 2.0
        nt.links.new(ramp.outputs["Color"], em.inputs["Color"])
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
        bpy.ops.mesh.primitive_plane_add(size=0.14, location=(-0.012, 0.055, -0.002))
        bp = bpy.context.active_object
        bp.name = "StudioBacklight"
        look_at(bp, (0.0, 0.0, -0.002))
        bp.data.materials.append(back_mat)

        # Clean near-frontal studio angle (~10 deg off-axis) looking straight through flat front wall without center corner seam
        cam_obj.location = Vector((0.0085, -0.044, 0.0048))
        cam_data.lens = 50.0
        cam_data.clip_start = 0.001
        look_at(cam_obj, (0.0, 0.0, -0.0022))
        cam_data.dof.use_dof = False

        add_area_light("TopSoftbox", (0.0, 0.0, 0.065), (0, 0, 0), 0.24, 0.08, 0.08, (1.0, 0.99, 0.97))
        add_area_light("SideKey", (0.055, -0.015, 0.025), (65, 0, 75), 0.18, 0.05, 0.04, (0.98, 0.99, 1.0))

    elif sc in ("pouring_square_jar", "pouring_wine_glass"):
        setup_world_studio((0.022, 0.026, 0.034, 1.0), 0.22)
        bench_mat = make_dark_studio_floor_material(wet=True)
        add_box_mesh("StudioCounter", (-0.60, -0.60, -0.020), (0.60, 0.60, -0.0015), bench_mat)
        # Soft luminous studio backdrop so refractive glass edges & pouring water stream pop cleanly
        back_mat, nt, _, out = new_mat("Mat_GlassBacklight")
        nt.links.clear()
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (0.78, 0.86, 0.96, 1.0)
        em.inputs["Strength"].default_value = 1.35
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
        bpy.ops.mesh.primitive_plane_add(size=0.90, location=(-0.12, 0.42, 0.16))
        bp = bpy.context.active_object
        bp.name = "GlassBacklight"
        look_at(bp, (0.0, 0.0, 0.09))
        bp.data.materials.append(back_mat)

        if sc == "pouring_square_jar":
            cam_obj.location = Vector((0.22, -0.34, 0.18))
            look_at(cam_obj, (0.0, 0.0, 0.085))
        else:
            cam_obj.location = Vector((0.20, -0.32, 0.18))
            look_at(cam_obj, (0.0, 0.0, 0.105))
        cam_data.lens = 45.0
        cam_data.clip_start = 0.002
        add_area_light("TopKey", (0.15, -0.18, 0.48), (24, 18, 25), 14.0, 0.35, 0.30, (1.0, 0.98, 0.94))
        add_area_light("SideRimL", (-0.36, -0.05, 0.22), (65, 0, -82), 9.0, 0.22, 0.38, (0.85, 0.93, 1.0))
        add_area_light("SideRimR", (0.36, 0.08, 0.20), (65, 0, 102), 8.5, 0.22, 0.38, (1.0, 0.96, 0.90))
        for mic in manifest["microphones"]:
            add_mic_prop(mic, scale_r=0.0032)

    elif sc == "sinking_bell_grotto":
        setup_world_studio((0.014, 0.018, 0.024, 1.0), 0.18)
        cam_obj.location = Vector((0.12, -0.175, 0.33))
        cam_data.lens = 28.0
        cam_data.clip_start = 0.002
        look_at(cam_obj, (0.0, 0.0, 0.035))
        add_area_light("GrottoShaftKey", (0.04, -0.06, 0.44), (10, 6, 0), 18.0, 0.26, 0.26, (1.0, 0.95, 0.86))
        add_area_light("GrottoFill", (-0.14, 0.12, 0.26), (-45, -35, 135), 7.0, 0.22, 0.22, (0.72, 0.88, 1.0))
        for mic in manifest["microphones"]:
            add_mic_prop(mic, scale_r=0.0035)

    elif sc == "supersonic_ear_flyby":
        setup_world_studio((0.012, 0.015, 0.022, 1.0), 0.24)
        cam_obj.location = Vector((7.8, -6.2, 5.8))
        cam_data.lens = 30.0
        cam_data.clip_end = 250.0
        look_at(cam_obj, (9.8, 0.35, 1.55))
        add_area_light("SunKey", (4.0, -8.0, 14.0), (30, 15, 18), 4500.0, 6.0, 6.0, (1.0, 0.97, 0.92))
        add_area_light("SkyFill", (14.0, 8.0, 10.0), (-35, 20, 140), 1800.0, 8.0, 8.0, (0.78, 0.88, 1.0))
        for mic in manifest["microphones"]:
            add_mic_prop(mic, scale_r=0.014)

    elif sc == "explosion_canyon":
        setup_world_studio((0.020, 0.024, 0.032, 1.0), 0.32)
        cam_obj.location = Vector((76.0, -54.0, 46.0))
        cam_data.lens = 28.0
        cam_data.clip_end = 2500.0
        look_at(cam_obj, (78.0, -2.0, 2.0))
        add_area_light("CanyonSun", (55.0, -60.0, 95.0), (32, 18, 22), 2.5e5, 35.0, 35.0, (1.0, 0.95, 0.86))
        add_area_light("CanyonFill", (110.0, 45.0, 65.0), (-35, -20, 140), 8.0e4, 40.0, 40.0, (0.75, 0.86, 1.0))
        for mic in manifest["microphones"]:
            add_mic_prop(mic, scale_r=0.22)

    elif sc.startswith("ballistics_"):
        setup_world_studio((0.018, 0.022, 0.030, 1.0), 0.30)
        cam_obj.location = Vector((160.0, -140.0, 110.0))
        cam_data.lens = 26.0
        cam_data.clip_end = 6000.0
        look_at(cam_obj, (420.0, 0.0, 2.0))
        add_area_light("RangeSun", (300.0, -200.0, 300.0), (35, 18, 20), 2.5e6, 120.0, 120.0, (1.0, 0.96, 0.88))
        for mic in manifest["microphones"]:
            add_mic_prop(mic, scale_r=0.60)

    elif sc == "rip_paper":
        setup_world_studio((0.010, 0.012, 0.016, 1.0), 0.14)
        bench_mat = make_dark_studio_floor_material(wet=False)
        add_box_mesh("StudioTable", (-0.60, -0.60, -0.052), (0.60, 0.60, -0.040), bench_mat)
        jaw_mat = make_steel_material("Mat_JawSteel", polished=False, base=(0.24, 0.26, 0.29, 1.0))
        for body in manifest["bodies"]:
            for ci, (cmin, cmax) in enumerate(body.get("clamps", [])):
                add_box_mesh(f"FixedClamp_Bot_{ci}", (0.0235, -0.026, -0.008), (0.033, 0.026, -0.0004), jaw_mat)
                add_box_mesh(f"FixedClamp_Top_{ci}", (0.0235, -0.026, 0.0004), (0.033, 0.026, 0.0045), jaw_mat)
        cam_obj.location = Vector((0.028, -0.138, 0.094))
        cam_data.lens = 40.0
        cam_data.clip_start = 0.001
        look_at(cam_obj, (-0.002, 0.0, 0.0))
        add_area_light("Key", (0.05, -0.08, 0.12), (38, 22, 20), 0.55, 0.12, 0.10, (1.0, 0.96, 0.90))
        add_area_light("Fill", (-0.10, -0.04, 0.07), (58, 0, -65), 0.22, 0.14, 0.10, (0.84, 0.92, 1.0))
        add_area_light("BackRim", (-0.02, 0.09, 0.06), (-50, 0, 180), 0.42, 0.14, 0.06, (1.0, 0.98, 0.94))

    elif sc == "drop_on_wet_leaves":
        setup_world_studio((0.010, 0.012, 0.016, 1.0), 0.14)
        fz = manifest.get("floor_z", 0.0) or 0.0
        wet_floor = make_dark_studio_floor_material(wet=True)
        add_box_mesh("WetSlateFloor", (-0.60, -0.60, fz - 0.015), (0.60, 0.60, fz), wet_floor)
        cam_obj.location = Vector((0.076, -0.114, 0.060))
        cam_data.lens = 38.0
        cam_data.clip_start = 0.001
        look_at(cam_obj, (0.0, 0.0, 0.015))
        add_area_light("WarmKey", (0.08, -0.09, 0.11), (42, 26, 30), 0.85, 0.10, 0.08, (1.0, 0.95, 0.88))
        add_area_light("SpecularRim", (-0.06, 0.08, 0.07), (-48, 0, 150), 0.72, 0.12, 0.06, (0.90, 0.96, 1.0))
        add_area_light("CoolFill", (-0.11, -0.05, 0.06), (60, 0, -60), 0.28, 0.14, 0.10, (0.80, 0.88, 1.0))

    else:
        # Generic studio auto-frame fallback using manifest water_bounds / microphones / sources
        setup_world_studio((0.015, 0.018, 0.024, 1.0), 0.22)
        pts = []
        if manifest.get("water_bounds"):
            wb = manifest["water_bounds"]
            pts.append(Vector(wb[0]))
            pts.append(Vector(wb[1]))
        for mic in manifest.get("microphones", []):
            pts.append(Vector(mic["position"]))
        for src in manifest.get("point_sources", []):
            pts.append(Vector(src["center"]))
        if pts:
            bmin = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
            bmax = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
            center = 0.5 * (bmin + bmax)
            radius = max((bmax - bmin).length * 0.65, 0.12)
        else:
            center = Vector((0.0, 0.0, 0.05))
            radius = 0.40
        cam_obj.location = center + Vector((0.95 * radius, -1.65 * radius, 0.95 * radius))
        cam_data.lens = 36.0
        cam_data.clip_start = max(0.001, radius * 0.002)
        cam_data.clip_end = max(100.0, radius * 40.0)
        look_at(cam_obj, center)
        add_area_light("AutoKey", tuple(center + Vector((1.1 * radius, -1.2 * radius, 1.8 * radius))), (32, 18, 24), 25.0 * radius * radius, 1.2 * radius, 1.0 * radius, (1.0, 0.97, 0.92))
        add_area_light("AutoFill", tuple(center + Vector((-1.5 * radius, -0.6 * radius, 1.0 * radius))), (52, 0, -58), 10.0 * radius * radius, 1.4 * radius, 1.0 * radius, (0.82, 0.90, 1.0))
        add_area_light("AutoRim", tuple(center + Vector((0.0, 1.5 * radius, 1.1 * radius))), (-48, 0, 180), 16.0 * radius * radius, 1.2 * radius, 0.7 * radius, (0.96, 0.98, 1.0))
        for mic in manifest.get("microphones", []):
            add_mic_prop(mic, scale_r=max(0.003, radius * 0.015))

    return cam_obj


def update_pull_jaw_pos(jaw_obj, coords, uvs, y_sign):
    if jaw_obj is None or len(coords) == 0 or len(uvs) != len(coords):
        return
    if y_sign > 0:
        mask = (uvs[:, 0] < 0.15) & (uvs[:, 1] > 0.56)
    else:
        mask = (uvs[:, 0] < 0.15) & (uvs[:, 1] < 0.44)
    pts = coords[mask]
    if len(pts) > 0:
        c = pts.mean(axis=0)
        jaw_obj.location = (float(c[0]) - 0.001, float(c[1]), float(c[2]))


def update_pslice_mesh(pslice_obj, nx, ny, ox, oy, dx, z, vals, r_max=0.0, p_ref=250.0):
    mesh = pslice_obj.data
    if len(mesh.vertices) != nx * ny:
        xs = ox + (np.arange(nx, dtype=np.float32) + 0.5) * dx
        ys = oy + (np.arange(ny, dtype=np.float32) + 0.5) * dx
        gx, gy = np.meshgrid(xs, ys)
        coords = np.stack([gx.ravel(), gy.ravel(), np.full(nx * ny, z, dtype=np.float32)], axis=1)
        idx = np.arange(nx * ny, dtype=np.int32).reshape((ny, nx))
        a = idx[:-1, :-1].ravel()
        b = idx[:-1, 1:].ravel()
        c = idx[1:, 1:].ravel()
        d = idx[1:, :-1].ravel()
        tris = np.concatenate([np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)], axis=0)
        populate_mesh_from_arrays(mesh, coords, tris, smooth=True)
        mesh.color_attributes.new(name="p_rgba", type="FLOAT_COLOR", domain="POINT")

    if r_max > 0.0:
        xs = ox + (np.arange(nx, dtype=np.float32) + 0.5) * dx
        ys = oy + (np.arange(ny, dtype=np.float32) + 0.5) * dx
        gx, gy = np.meshgrid(xs, ys)
        inside = ((gx * gx + gy * gy) <= (r_max * r_max)).ravel()
    else:
        inside = np.ones(ny * nx, dtype=bool)

    # Signed asinh perceptual compression adapts automatically to whispering waves, Mach cones, and blast shocks
    p_hi = max(float(p_ref), 10.0)
    knee = max(p_hi * 0.032, 1.0)
    norm_s = np.arcsinh(vals / knee) / np.arcsinh(p_hi / knee)
    norm_s = np.clip(norm_s, -1.0, 1.0)
    abs_n = np.abs(norm_s)

    rgba = np.zeros((ny * nx, 4), dtype=np.float32)
    pos = (norm_s >= 0.0).ravel()
    neg = ~pos
    # Cool cyan compression vs warm amber rarefaction
    rgba[pos, 0] = 0.05
    rgba[pos, 1] = 0.72
    rgba[pos, 2] = 1.00
    rgba[neg, 0] = 1.00
    rgba[neg, 1] = 0.55
    rgba[neg, 2] = 0.08
    alpha = np.where(inside, np.power(abs_n.ravel(), 0.85) * 0.82, 0.0)
    rgba[:, 3] = np.clip(alpha, 0.0, 0.85)

    attr = mesh.color_attributes.get("p_rgba")
    attr.data.foreach_set("color", rgba.ravel())


def main():
    args = parse_args()
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    frames_in_dir = os.path.dirname(os.path.abspath(args.manifest))
    os.makedirs(args.out_frames, exist_ok=True)

    configure_cycles_optix(args.width, args.height, args.samples, manifest["scenario"])
    setup_scene_camera_and_lights(manifest)

    # Create scene materials and objects once, then stream PLYs per frame
    mat_wood = make_oak_wood_material()
    mat_steel = make_steel_material("Mat_Steel", polished=False)
    mat_chrome = make_steel_material("Mat_Chrome", polished=True, base=(0.86, 0.88, 0.91, 1.0))
    mat_granite = make_granite_material()
    mat_concrete = make_concrete_material()
    mat_glass = make_vessel_glass_material()
    mat_bronze = make_bronze_material()
    mat_boot = make_boot_sole_material()
    mat_paper = make_paper_material()
    mat_water = make_water_material()

    body_objs = {}
    pull_jaws = []
    leaf_counter = 0

    for idx, b in enumerate(manifest["bodies"]):
        if b["name"] == "stone_ceiling":
            continue
        mesh = bpy.data.meshes.new(f"Mesh_{b['name']}")
        obj = bpy.data.objects.new(b["name"], mesh)
        bpy.context.collection.objects.link(obj)

        preset = b["material_preset"]
        if preset == "oak_wood":
            obj.data.materials.append(mat_wood)
        elif preset == "structural_steel":
            obj.data.materials.append(mat_chrome if b["name"] == "sphere" else mat_steel)
            if b.get("is_shell"):
                sol = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
                sol.thickness = max(float(b["thickness_m"]), 0.001)
                sol.offset = 0.0
        elif preset == "glass":
            obj.data.materials.append(mat_glass)
            if b.get("is_shell"):
                sol = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
                sol.thickness = max(float(b["thickness_m"]), 0.0018)
                sol.offset = 0.0
        elif preset == "bell_bronze":
            obj.data.materials.append(mat_bronze)
            if b.get("is_shell"):
                sol = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
                sol.thickness = max(float(b["thickness_m"]), 0.0022)
                sol.offset = 0.0
        elif preset == "concrete":
            obj.data.materials.append(mat_concrete)
        elif preset == "granite_stone":
            obj.data.materials.append(mat_granite)
        elif preset == "boot_sole":
            obj.data.materials.append(mat_boot)
        elif preset == "cellulose_paper":
            obj.data.materials.append(mat_paper)
            sol = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
            sol.thickness = max(float(b["thickness_m"]), 0.00028)
            sol.offset = 0.0
        elif preset == "wet_leaf":
            obj.data.materials.append(make_wet_leaf_material(leaf_counter))
            leaf_counter += 1
            sol = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
            sol.thickness = max(float(b["thickness_m"]), 0.00034)
            sol.offset = 0.0
        else:
            obj.data.materials.append(mat_steel)

        if b["is_static"] and b.get("rest_mesh_file"):
            p = os.path.join(frames_in_dir, b["rest_mesh_file"])
            coords, uvs, dmg, tris = read_ply_binary(p)
            populate_mesh_from_arrays(mesh, coords, tris, uvs, dmg, smooth=True)
        else:
            body_objs[idx] = obj

        if manifest["scenario"] == "rip_paper" and preset == "cellulose_paper":
            grip_mat = make_steel_material("Mat_PullGrip", polished=False, base=(0.24, 0.26, 0.29, 1.0))
            for yi, ys in enumerate([1.0, -1.0]):
                bpy.ops.mesh.primitive_cube_add(size=1.0)
                j = bpy.context.active_object
                j.name = f"PullGrip_{yi}"
                j.scale = (0.009, 0.014, 0.005)
                bpy.ops.object.transform_apply(scale=True)
                j.data.materials.append(grip_mat)
                pull_jaws.append((idx, j, ys))

    water_obj = None
    if manifest.get("has_fluid_isosurface"):
        wmesh = bpy.data.meshes.new("Mesh_Water")
        water_obj = bpy.data.objects.new("WaterVolume", wmesh)
        bpy.context.collection.objects.link(water_obj)
        water_obj.data.materials.append(mat_water)

    pslice_obj = None
    spark_obj = None
    pslice_r_max = float(manifest.get("pslice_r_max", 0.0) or 0.0)
    pslice_p_ref = float(manifest.get("pslice_max_abs_p", 0.0) or 0.0)
    if pslice_p_ref <= 1e-6:
        pslice_p_ref = 250.0
    if manifest.get("has_pressure_slice"):
        pmesh = bpy.data.meshes.new("Mesh_PSlice")
        pslice_obj = bpy.data.objects.new("AcousticPressureSlice", pmesh)
        bpy.context.collection.objects.link(pslice_obj)
        pslice_obj.data.materials.append(make_schlieren_material())
        if manifest.get("point_sources"):
            src = manifest["point_sources"][0]
            spark_r = max(0.007, pslice_r_max * 0.018)
            bpy.ops.mesh.primitive_uv_sphere_add(radius=spark_r, location=src["center"])
            spark_obj = bpy.context.active_object
            spark_obj.name = "SourceSpark"
            sm, snt, _, sout = new_mat("Mat_Spark")
            snt.links.clear()
            sem = snt.nodes.new("ShaderNodeEmission")
            sem.inputs["Color"].default_value = (0.35, 0.85, 1.0, 1.0)
            sem.inputs["Strength"].default_value = 0.0
            snt.links.new(sem.outputs["Emission"], sout.inputs["Surface"])
            spark_obj.data.materials.append(sm)

    num_frames = int(manifest["num_frames"])
    if args.only_frames.strip():
        frame_indices = [int(x) for x in args.only_frames.split(",") if x.strip()]
    else:
        frame_indices = list(range(0, num_frames, max(1, args.frame_step)))

    duration_s = float(manifest["duration_s"])
    t_render_start = time.monotonic()
    total_to_render = len(frame_indices)
    for done_idx, fidx in enumerate(frame_indices):
        t_frame = (fidx / max(1, num_frames - 1)) * duration_s
        for bidx, obj in body_objs.items():
            ply_p = os.path.join(frames_in_dir, f"frame_{fidx:04d}_body_{bidx}.ply")
            coords, uvs, dmg, tris = read_ply_binary(ply_p)
            if manifest["bodies"][bidx]["material_preset"] == "wet_leaf":
                coords, uvs, dmg, tris = append_orphan_shell_shards(coords, uvs, dmg, tris, bidx)
            populate_mesh_from_arrays(obj.data, coords, tris, uvs, dmg, smooth=True)
            for p_bidx, j_obj, ys in pull_jaws:
                if p_bidx == bidx:
                    update_pull_jaw_pos(j_obj, coords, uvs, ys)

        if water_obj is not None:
            wply = os.path.join(frames_in_dir, f"frame_{fidx:04d}_water.ply")
            coords, _, _, tris = read_ply_binary(wply)
            populate_mesh_from_arrays(water_obj.data, coords, tris, smooth=True)

        if pslice_obj is not None:
            pbin = os.path.join(frames_in_dir, f"frame_{fidx:04d}_pslice.bin")
            if os.path.exists(pbin):
                nx, ny, ox, oy, dx, z, _, vals = read_pslice_bin(pbin)
                update_pslice_mesh(
                    pslice_obj, nx, ny, ox, oy, dx, z, vals,
                    r_max=pslice_r_max, p_ref=pslice_p_ref,
                )
            if spark_obj is not None and manifest.get("point_sources"):
                src = manifest["point_sources"][0]
                dt_s = (t_frame - src["t0_s"]) * 1000.0
                glow = 18.0 * math.exp(-0.5 * (dt_s / 0.16) ** 2)
                sem = spark_obj.data.materials[0].node_tree.nodes.get("Emission")
                if sem:
                    sem.inputs["Strength"].default_value = glow
                spark_obj.hide_render = (glow < 0.05)

        out_png = os.path.join(args.out_frames, f"frame_{fidx:04d}.png")
        bpy.context.scene.render.filepath = out_png
        bpy.ops.render.render(write_still=True)

        done_cnt = done_idx + 1
        wall_s = time.monotonic() - t_render_start
        eta_s = (wall_s / done_cnt) * (total_to_render - done_cnt)
        print(
            f"[{manifest['scenario']}] Render frame {done_cnt}/{total_to_render} "
            f"({100.0 * done_cnt / max(1, total_to_render):5.1f}%) | "
            f"wall={format_duration_hms(wall_s)} | ETA={format_duration_hms(eta_s)}",
            flush=True,
        )

    print(
        f"Rendered {total_to_render} frames for '{manifest['scenario']}' into {args.out_frames} "
        f"in {format_duration_hms(time.monotonic() - t_render_start)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
