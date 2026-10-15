"""Render the qualified, unchanged PLY with Blender Cycles CPU. No mesh edits.

Run only after extraction qualifies, under the parent's 6 GiB / 180 s job:
blender --background --factory-startup --disable-autoexec --threads 4
        --python-exit-code 1 --python render_preview.py
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
ROOT=E.parents[2]
MESH = Path(r"C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-turret-cpu-mesh-ad07f96\turret-full.ply")
OUTPUT = ROOT.parent/'artifacts/amd-turret-preview-9380d3a'
VIEWS = ("front", "back", "side")
PNGS = {name: OUTPUT / f"preview-{name}.png" for name in VIEWS}
METADATA = OUTPUT / "render-metadata.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def aim(obj, target):
    # Pinned TRELLIS render_utils uses world +Z up; camera local -Z views mesh.
    forward = (target - obj.location).normalized()
    right = forward.cross(Vector((0, 0, 1))).normalized()
    up = right.cross(forward).normalized()
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Matrix((right, up, -forward)).transposed().to_quaternion()
    return right, up


receipt = json.loads((ROOT/'docs/compatibility/amd-turret-cpu-mesh-extraction-evidence/extraction-worker.json').read_text(encoding="utf-8"))
began=time.monotonic()
assert receipt["classification"] == "PASS", "Qualified extraction required"
assert MESH.is_file() and OUTPUT.is_dir() and not MESH.is_symlink()
assert not any(path.exists() for path in (*PNGS.values(), METADATA)), "Fresh preview names required"
source_stat = MESH.stat()
source_sha = digest(MESH)
assert source_sha == receipt["artifacts"]["turret-full.ply"]["sha256"]
assert source_stat.st_size == receipt["artifacts"]["turret-full.ply"]["bytes"]
expected_counts = (int(receipt["vertex_count"]), int(receipt["face_count"]))
assert min(expected_counts) > 0

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 8
scene.cycles.use_denoising = False
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
scene.view_settings.view_transform = "Standard"
scene.view_settings.exposure = 0
scene.view_settings.gamma = 1
bpy.context.preferences.filepaths.temporary_directory = str(OUTPUT)
for layer in scene.view_layers:
    layer.cycles.use_denoising = False

props = set(bpy.ops.wm.ply_import.get_rna_type().properties.keys())
kwargs = {"filepath": str(MESH)}
for key, value in {"merge_verts": False, "global_scale": 1.0,
                   "use_scene_unit": False, "forward_axis": "Y", "up_axis": "Z"}.items():
    if key in props:
        kwargs[key] = value
assert bpy.ops.wm.ply_import(**kwargs) == {"FINISHED"}
objects = [obj for obj in scene.objects if obj.type == "MESH"]
assert len(objects) == 1
obj = objects[0]
mesh = obj.data
assert not obj.modifiers
assert max(abs(obj.matrix_world[i][j] - (i == j)) for i in range(4) for j in range(4)) < 1e-7
bpy.context.view_layer.update()


def geometry_summary():
    counts = (len(mesh.vertices), len(mesh.polygons))
    assert counts == expected_counts, f"Import/render changed geometry counts: {counts}"
    corners = [Vector(point) for point in obj.bound_box]
    bbox = [[min(point[i] for point in corners) for i in range(3)],
            [max(point[i] for point in corners) for i in range(3)]]
    assert all(math.isfinite(value) for row in bbox for value in row)
    assert max(abs(bbox[i][j] - receipt["bbox"][i][j]) for i in range(2) for j in range(3)) < 1e-6
    return counts, bbox


counts, bbox = geometry_summary()  # Fail before rendering if any import invariant differs.
def geometry_payloads():
    # Verify the actual Blender positions and triangle winding, not only bounds.
    co=np.empty(expected_counts[0]*3,np.float32)
    loops=np.empty(expected_counts[1]*3,np.int32)
    totals=np.empty(expected_counts[1],np.int32)
    mesh.vertices.foreach_get('co',co);mesh.loops.foreach_get('vertex_index',loops)
    mesh.polygons.foreach_get('loop_total',totals)
    assert np.all(totals==3) and len(mesh.loops)==len(loops)
    for name,array in (('vertices.npy',co.reshape(-1,3)),('faces.npy',loops.reshape(-1,3))):
        original=np.load(MESH.parent/name,mmap_mode='r',allow_pickle=False)
        assert np.array_equal(array,original), 'Blender import/render altered '+name
    return {'vertex_payload_sha256':hashlib.sha256(co.tobytes()).hexdigest(),
            'face_payload_sha256':hashlib.sha256(loops.tobytes()).hexdigest()}
payload_before=geometry_payloads()
center = (Vector(bbox[0]) + Vector(bbox[1])) * .5
extent = max(bbox[1][i] - bbox[0][i] for i in range(3))
assert extent > 0
corners = [Vector(point) for point in itertools.product(*zip(*bbox))]

material = bpy.data.materials.new("Preview matte grey")
material.use_nodes = True
shader = material.node_tree.nodes.get("Principled BSDF")
shader.inputs["Base Color"].default_value = (.43, .43, .43, 1)
shader.inputs["Roughness"].default_value = .85
shader.inputs["Metallic"].default_value = 0
mesh.materials.clear()
mesh.materials.append(material)
world = bpy.data.worlds.new("Plain opaque background")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (.12, .13, .15, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = .5
scene.world = world

for name, offset, energy in (("Key", (2.5, 3.2, 3), 550),
                             ("Fill", (-3, 1.7, 1), 300),
                             ("Back", (-1, 2.5, 3), 450)):
    light = bpy.data.lights.new(name, "AREA")
    light.energy, light.size = energy * extent * extent, 2.5 * extent
    lamp = bpy.data.objects.new(name, light)
    scene.collection.objects.link(lamp)
    lamp.location = center + Vector(offset) * extent
    aim(lamp, center)

camera_data = bpy.data.cameras.new("Orthographic preview")
camera_data.type = "ORTHO"
camera = bpy.data.objects.new("Orthographic preview", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
camera_data.clip_start, camera_data.clip_end = .001 * extent, 20 * extent
directions = {"front": (1.1, -2.4, 1.2), "back": (-1.1, 2.4, 1.2), "side": (2.6, .1, .8)}
views = []
for name in VIEWS:
    view_started=time.monotonic()
    camera.location = center + Vector(directions[name]).normalized() * (4 * extent)
    right, up = aim(camera, center)
    camera_data.ortho_scale = 2.24 * max(max(abs((point - center).dot(right)),
                                            abs((point - center).dot(up))) for point in corners)
    assert scene.cycles.device == "CPU" and scene.render.threads == 4
    geometry_summary()
    scene.render.filepath = str(PNGS[name])
    bpy.ops.render.render(write_still=True)
    assert PNGS[name].is_file() and PNGS[name].stat().st_size > 0
    views.append({"name": name, "PNG": PNGS[name].name, "camera_location": list(camera.location),
                  "camera_quaternion":list(camera.rotation_quaternion),"ortho_scale": camera_data.ortho_scale,
                  "seconds":time.monotonic()-view_started})
    print("CPU_PREVIEW_RENDERED=" + name, flush=True)

counts, bbox = geometry_summary()
assert geometry_payloads()==payload_before
assert digest(MESH) == source_sha
assert (MESH.stat().st_size, MESH.stat().st_mtime_ns) == (source_stat.st_size, source_stat.st_mtime_ns)
metadata = {"vertex_count": counts[0], "face_count": counts[1], "bbox": bbox,
            "source_PLY_sha256": source_sha, "geometry_unchanged": True,
            "engine": "CYCLES", "device": "CPU", "samples": 8, "threads": 4,
            "denoising": False, "transparent": False, "canonical_up_axis": "+Z",
            "import_settings": kwargs, "views": views,"Blender_version":bpy.app.version_string,
            "actual_geometry_payloads_before_and_after":payload_before,"elapsed_seconds":time.monotonic()-began,
            "outputs":{name:{"path":str(PNGS[name]),"bytes":PNGS[name].stat().st_size,"sha256":digest(PNGS[name])} for name in VIEWS}}
with METADATA.open("x", encoding="utf-8") as stream:
    json.dump(metadata, stream, indent=2)
    stream.write("\n")
