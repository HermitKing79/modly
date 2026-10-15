"""Render the qualified learned RGB PLY with bounded Blender Cycles CPU.

Run only through render_host.py. This script imports colour explicitly as sRGB,
verifies the imported colour and immutable geometry, then saves three views and
an editable .blend. It does not run models or reconstruct material channels.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector


E = Path(__file__).resolve().parent
ROOT = E.parents[2]
OUTPUT = ROOT.parent / "artifacts/amd-turret-learned-colour-fa3a98c"
MESH = OUTPUT / "turret-learned-colour.ply"
ORIGINAL = ROOT.parent / "artifacts/amd-turret-cpu-mesh-ad07f96"
VIEWS = ("front", "rear", "side")
PNGS = {name: OUTPUT / f"preview-{name}.png" for name in VIEWS}
BLEND = OUTPUT / "learned-colour.blend"
METADATA = OUTPUT / "render-metadata.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def aim(obj, target):
    # The pinned canonical TRELLIS renderer uses world +Z up.
    forward = (target - obj.location).normalized()
    right = forward.cross(Vector((0, 0, 1))).normalized()
    up = right.cross(forward).normalized()
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Matrix((right, up, -forward)).transposed().to_quaternion()
    return right, up


def ply_rgb(receipt):
    """Parse the qualified binary layout independently from the converter."""
    header = []
    with MESH.open("rb") as stream:
        while True:
            line = stream.readline(4097)
            assert line and len(line) <= 4096, "Invalid/unbounded PLY header"
            header.append(line.decode("ascii").strip())
            assert stream.tell() <= 65536, "Unbounded PLY header"
            if header[-1] == "end_header":
                offset = stream.tell()
                break
    semantic = [line for line in header if not line.startswith("comment ")]
    assert semantic == [
        "ply", "format binary_little_endian 1.0",
        f"element vertex {receipt['vertex_count']}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        f"element face {receipt['face_count']}",
        "property list uchar int vertex_indices", "end_header",
    ], "Unexpected coloured PLY schema"
    vertex_dtype = np.dtype([("xyz", "<f4", (3,)), ("rgb", "u1", (3,))])
    assert vertex_dtype.itemsize == 15
    records = np.memmap(MESH, mode="r", dtype=vertex_dtype, offset=offset,
                        shape=(receipt["vertex_count"],))
    assert MESH.stat().st_size == offset + 15 * receipt["vertex_count"] + 13 * receipt["face_count"]
    rgb = np.ascontiguousarray(records["rgb"])
    assert array_digest(rgb) == receipt["RGB_payload_sha256"], "PLY RGB payload differs"
    original = np.load(ORIGINAL / "vertices.npy", mmap_mode="r", allow_pickle=False)
    assert original.dtype == np.dtype("<f4") and original.shape == (receipt["vertex_count"], 3)
    assert array_digest(records["xyz"]) == array_digest(original), "PLY reordered/changed XYZ bytes"
    return rgb


began = time.monotonic()
receipt = json.loads((E / "conversion.json").read_text(encoding="utf-8"))
assert receipt == json.loads((OUTPUT / "conversion.json").read_text(encoding="utf-8"))
assert receipt["classification"] == "PASS" and receipt["geometry_unchanged"] is True
assert MESH.is_file() and not MESH.is_symlink() and OUTPUT.is_dir()
assert not any(path.exists() for path in (*PNGS.values(), BLEND, METADATA)), "Fresh render names required"
source_stat = MESH.stat()
source_sha = digest(MESH)
assert source_sha == receipt["coloured_ply"]["sha256"]
assert source_stat.st_size == receipt["coloured_ply"]["bytes"]
expected_counts = (int(receipt["vertex_count"]), int(receipt["face_count"]))
assert expected_counts == (1842968, 4012360)
source_rgb = ply_rgb(receipt)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 64
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPENIMAGEDENOISE"
scene.cycles.denoising_use_gpu = False
scene.cycles.use_adaptive_sampling = False
scene.cycles.max_bounces = 2
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.render.threads_mode = "FIXED"
scene.render.threads = 4
scene.render.resolution_x = scene.render.resolution_y = 640
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.use_compositing = False
scene.render.use_sequencer = False
scene.render.use_persistent_data = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.color_depth = "8"
scene.render.image_settings.compression = 15
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0
scene.view_settings.gamma = 1
bpy.context.preferences.filepaths.temporary_directory = str(OUTPUT / "blender-scratch")
for layer in scene.view_layers:
    layer.cycles.use_denoising = True

props = set(bpy.ops.wm.ply_import.get_rna_type().properties.keys())
assert "import_colors" in props, "Explicit sRGB PLY import required"
kwargs = {"filepath": str(MESH), "import_colors": "SRGB"}
for key, value in {"merge_verts": False, "global_scale": 1.0,
                   "use_scene_unit": False, "forward_axis": "Y", "up_axis": "Z"}.items():
    if key in props:
        kwargs[key] = value
assert bpy.ops.wm.ply_import(**kwargs) == {"FINISHED"}
objects = [obj for obj in scene.objects if obj.type == "MESH"]
assert len(objects) == 1
obj = objects[0]
obj.name = "Turret - learned RGB - unchanged geometry"
mesh = obj.data
assert not obj.modifiers
assert max(abs(obj.matrix_world[i][j] - (i == j)) for i in range(4) for j in range(4)) < 1e-7
bpy.context.view_layer.update()


def geometry_payloads():
    """Verify row order and exact float/index bytes, including signed zeros."""
    assert (len(mesh.vertices), len(mesh.polygons)) == expected_counts
    assert len(mesh.loops) == expected_counts[1] * 3
    co = np.empty(expected_counts[0] * 3, dtype="<f4")
    loops = np.empty(expected_counts[1] * 3, dtype="<i4")
    totals = np.empty(expected_counts[1], dtype="<i4")
    starts = np.empty(expected_counts[1], dtype="<i4")
    mesh.vertices.foreach_get("co", co)
    mesh.loops.foreach_get("vertex_index", loops)
    mesh.polygons.foreach_get("loop_total", totals)
    mesh.polygons.foreach_get("loop_start", starts)
    assert np.all(totals == 3), "Imported faces are not all triangles"
    assert np.array_equal(starts, np.arange(expected_counts[1], dtype=np.int32) * 3), "Face row order changed"
    payloads = {}
    for name, array in (("vertices.npy", co.reshape(-1, 3)), ("faces.npy", loops.reshape(-1, 3))):
        original = np.load(ORIGINAL / name, mmap_mode="r", allow_pickle=False)
        assert original.dtype == array.dtype and original.shape == array.shape
        actual_hash, original_hash = array_digest(array), array_digest(original)
        assert actual_hash == original_hash, "Blender changed ordered geometry bytes: " + name
        payloads["vertex_payload_sha256" if name == "vertices.npy" else "face_payload_sha256"] = actual_hash
    return payloads


assert len(mesh.color_attributes) == 1, "One imported RGB attribute required"
colour = mesh.color_attributes[0]
assert colour.domain in {"POINT", "CORNER"}
assert colour.data_type in {"BYTE_COLOR", "FLOAT_COLOR"}


def colour_payload():
    """Recover canonical sRGB bytes from Blender's actual shader attribute."""
    values = np.empty(len(colour.data) * 4, dtype=np.float32)
    colour.data.foreach_get("color_srgb", values)
    values = values.reshape(-1, 4)
    assert np.isfinite(values).all() and np.all(np.abs(values[:, 3] - 1) < 1e-6)
    quantized = np.rint(np.clip(values[:, :3], 0, 1) * np.float32(255)).astype(np.uint8)
    if colour.domain == "POINT":
        assert quantized.shape == source_rgb.shape and np.array_equal(quantized, source_rgb)
        assert array_digest(quantized) == receipt["RGB_payload_sha256"]
    else:
        loop_indices = np.empty(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loop_indices)
        assert len(quantized) == len(loop_indices)
        for start in range(0, len(loop_indices), 262144):
            end = min(start + 262144, len(loop_indices))
            assert np.array_equal(quantized[start:end], source_rgb[loop_indices[start:end]])
    return {"name": colour.name, "domain": colour.domain, "data_type": colour.data_type,
            "RGB_payload_sha256": receipt["RGB_payload_sha256"],
            "all_imported_sRGB_bytes_match": True, "alpha_is_one": True}


payload_before = geometry_payloads()
colour_before = colour_payload()
corners = [Vector(point) for point in obj.bound_box]
bbox = [[min(point[i] for point in corners) for i in range(3)],
        [max(point[i] for point in corners) for i in range(3)]]
assert all(math.isfinite(value) for row in bbox for value in row)
center = (Vector(bbox[0]) + Vector(bbox[1])) * .5
extent = max(bbox[1][i] - bbox[0][i] for i in range(3))
assert extent > 0
corners = [Vector(point) for point in itertools.product(*zip(*bbox))]

material = bpy.data.materials.new("Learned RGB preview - matte nonmetal")
material.use_nodes = True
nodes, links = material.node_tree.nodes, material.node_tree.links
shader = nodes.get("Principled BSDF")
attribute = nodes.new("ShaderNodeVertexColor")
attribute.name = "Verified learned sRGB vertex colour"
attribute.layer_name = colour.name
links.new(attribute.outputs["Color"], shader.inputs["Base Color"])
shader.inputs["Roughness"].default_value = .8
shader.inputs["Metallic"].default_value = 0
shader.inputs["Emission Color"].default_value = (0, 0, 0, 1)
shader.inputs["Emission Strength"].default_value = 0
mesh.materials.clear()
mesh.materials.append(material)


def material_summary():
    assert len(mesh.materials) == 1 and mesh.materials[0] == material
    assert attribute.layer_name == colour.name
    base_links = list(shader.inputs["Base Color"].links)
    assert len(base_links) == 1 and base_links[0].from_node == attribute
    assert base_links[0].from_socket == attribute.outputs["Color"]
    output = nodes.get("Material Output")
    assert len(output.inputs["Surface"].links) == 1
    assert output.inputs["Surface"].links[0].from_node == shader
    assert shader.inputs["Metallic"].default_value == 0
    assert abs(shader.inputs["Roughness"].default_value - .8) < 1e-6
    assert shader.inputs["Emission Strength"].default_value == 0
    return {"node_type": attribute.bl_idname, "attribute_name": attribute.layer_name,
            "linked_socket": "Principled BSDF.Base Color", "metallic": 0,
            "roughness": .8, "emission_strength": 0, "full_PBR_reconstruction": False}


material_record = material_summary()
world = bpy.data.worlds.new("Neutral opaque background")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (.12, .12, .12, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = .5
scene.world = world
light_records = []
for name, offset, energy in (("Key", (2.5, 3.2, 3), 80),
                             ("Fill", (-3, 1.7, 1), 40),
                             ("Back", (-1, 2.5, 3), 60)):
    light = bpy.data.lights.new(name, "AREA")
    light.energy, light.size = energy * extent * extent, 2.5 * extent
    light.color = (1, 1, 1)
    lamp = bpy.data.objects.new(name, light)
    scene.collection.objects.link(lamp)
    lamp.location = center + Vector(offset) * extent
    aim(lamp, center)
    light_records.append({"name": name, "normalized_energy": energy,
                          "energy": light.energy, "size": light.size, "color": list(light.color)})

directions = {"front": (1.1, -2.4, 1.2), "rear": (-1.1, 2.4, 1.2), "side": (2.6, .1, .8)}
cameras = {}
for name in VIEWS:
    data = bpy.data.cameras.new(f"Preview {name}")
    data.type = "ORTHO"
    data.clip_start, data.clip_end = .001 * extent, 20 * extent
    camera = bpy.data.objects.new(f"Preview {name}", data)
    scene.collection.objects.link(camera)
    camera.location = center + Vector(directions[name]).normalized() * (4 * extent)
    right, up = aim(camera, center)
    data.ortho_scale = 2.24 * max(max(abs((point - center).dot(right)),
                                     abs((point - center).dot(up))) for point in corners)
    cameras[name] = camera

views = []
for name in VIEWS:
    view_started = time.monotonic()
    scene.camera = cameras[name]
    assert scene.cycles.device == "CPU" and scene.render.threads == 4
    assert scene.cycles.denoiser == "OPENIMAGEDENOISE" and scene.cycles.denoising_use_gpu is False
    scene.render.filepath = str(PNGS[name])
    bpy.ops.render.render(write_still=True)
    assert PNGS[name].is_file() and PNGS[name].stat().st_size > 0
    camera = scene.camera
    views.append({"name": name, "PNG": PNGS[name].name, "camera_object": camera.name,
                  "camera_location": list(camera.location),
                  "camera_quaternion": list(camera.rotation_quaternion),
                  "ortho_scale": camera.data.ortho_scale, "seconds": time.monotonic() - view_started})
    print("CPU_LEARNED_COLOUR_RENDERED=" + name, flush=True)

assert geometry_payloads() == payload_before
assert colour_payload() == colour_before
assert material_summary() == material_record
scene.camera = cameras["front"]
scene.render.filepath = str(PNGS["front"])
# Opening the .blend gives a useful solid viewport with the imported colour.
mesh.color_attributes.active_color_index = list(mesh.color_attributes).index(colour)
mesh.color_attributes.render_color_index = mesh.color_attributes.active_color_index
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            space = area.spaces.active
            space.shading.color_type = "VERTEX"
            space.region_3d.view_perspective = "CAMERA"
            space.region_3d.view_camera_zoom = 0
assert bpy.ops.wm.save_as_mainfile(filepath=str(BLEND), check_existing=False, compress=True) == {"FINISHED"}
assert BLEND.is_file() and BLEND.stat().st_size > 0
assert scene.camera == cameras["front"] and len([o for o in scene.objects if o.type == "CAMERA"]) == 3
assert geometry_payloads() == payload_before
assert colour_payload() == colour_before
assert material_summary() == material_record
assert digest(MESH) == source_sha
assert (MESH.stat().st_size, MESH.stat().st_mtime_ns) == (source_stat.st_size, source_stat.st_mtime_ns)

metadata = {
    "classification": "PASS", "vertex_count": expected_counts[0], "face_count": expected_counts[1],
    "bbox": bbox, "source_PLY_sha256": source_sha, "geometry_unchanged": True,
    "engine": "CYCLES", "device": "CPU", "samples": 64, "threads": 4,
    "denoising": True, "denoiser": "OPENIMAGEDENOISE", "denoising_use_gpu": False,
    "transparent": False, "canonical_up_axis": "+Z", "import_settings": kwargs,
    "colour_interpretation": "PLY byte RGB is canonical image colour display sRGB",
    "imported_attribute_before_and_after": colour_before, "material": material_record,
    "view_transform": scene.view_settings.view_transform, "look": scene.view_settings.look,
    "exposure": scene.view_settings.exposure, "gamma": scene.view_settings.gamma,
    "lights": light_records, "views": views, "active_camera": scene.camera.name,
    "Blender_version": bpy.app.version_string,
    "actual_geometry_payloads_before_and_after": payload_before,
    "elapsed_seconds": time.monotonic() - began,
    "outputs": {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
                for name, path in {**PNGS, "blend": BLEND}.items()},
}
with METADATA.open("x", encoding="utf-8") as stream:
    json.dump(metadata, stream, indent=2)
    stream.write("\n")
