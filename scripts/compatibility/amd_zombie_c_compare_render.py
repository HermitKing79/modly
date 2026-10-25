"""Eight untouched C previews using the immutable original cameras and CPU only."""
import sys
if sys.stdin.buffer.read(4) != b'GO\n':
    raise RuntimeError('Exact owned GO plus EOF required before render imports')

import importlib.util
from pathlib import Path

_path = Path(__file__).with_name('amd_zombie_c_compare.py')
_spec = importlib.util.spec_from_file_location('_zombie_c_render_policy', _path)
policy = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(policy)
sys.addaudithook(policy.audit_policy(policy.BASE))

import array
import hashlib
import itertools
import time
import bpy
from mathutils import Matrix, Vector
import numpy as np

BASE = policy.BASE
MESH = BASE / 'mesh/mesh.ply'
OUTPUT = BASE / 'previews'
need = policy.need
digest = policy.digest
read = policy.read


def aim(obj, target):
    forward = (target - obj.location).normalized(); right = forward.cross(Vector((0, 0, 1))).normalized()
    up = right.cross(forward).normalized()
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = Matrix((right, up, -forward)).transposed().to_quaternion()
    return right, up


inputs = read(BASE / 'comparison-inputs.json')
policy.verify_snapshot(inputs['immutable_original']); policy.verify_snapshot(inputs['decoded_inputs'])
policy.verify_snapshot(inputs['published_CPU_sources'])
original, original_mesh, camera_spec, _ = policy.original_inputs()
need(camera_spec == inputs['camera'], 'Frozen original camera target or light scale changed')
need(read(BASE / 'extract-run/host-result.json')['classification'] == 'PASS', 'Owned extraction PASS required')
metadata = read(BASE / 'mesh/mesh.json'); policy.validate_mesh(metadata, inputs['decode_binding'])
need(read(BASE / 'raw-mesh-preservation.json')['geometry_smoothed'] is False, 'Unsmoothed raw mesh required')
before = MESH.stat(); source_sha = digest(MESH)
need(source_sha == metadata['artifacts']['mesh.ply']['sha256'], 'C mesh binding differs')
policy.reserve(OUTPUT)

bpy.ops.wm.read_factory_settings(use_empty=True)
need(bpy.app.version_string == original['blender_version'], 'Original Blender rendering version required')
scene = bpy.context.scene; scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'
scene.cycles.samples = 8; scene.cycles.use_denoising = False; scene.cycles.use_adaptive_sampling = False
scene.cycles.max_bounces = 2; scene.cycles.diffuse_bounces = 1; scene.cycles.glossy_bounces = 1
scene.render.threads_mode = 'FIXED'; scene.render.threads = 4
scene.render.resolution_x = scene.render.resolution_y = 768; scene.render.resolution_percentage = 100
scene.render.film_transparent = False; scene.render.use_compositing = False; scene.render.use_sequencer = False
scene.render.use_persistent_data = True
scene.render.image_settings.file_format = 'PNG'; scene.render.image_settings.color_mode = 'RGB'
scene.render.image_settings.color_depth = '8'
scene.view_settings.view_transform = 'Standard'; scene.view_settings.exposure = 0; scene.view_settings.gamma = 1
bpy.context.preferences.filepaths.temporary_directory = str(BASE)
for layer in scene.view_layers:
    layer.cycles.use_denoising = False

kwargs = dict(filepath=str(MESH), merge_verts=False, global_scale=1.0, use_scene_unit=False, forward_axis='Y', up_axis='Z')
props = set(bpy.ops.wm.ply_import.get_rna_type().properties.keys())
need(set(kwargs) <= props, 'Original unmodified PLY import controls required')
need(bpy.ops.wm.ply_import(**kwargs) == {'FINISHED'}, 'PLY import did not complete')
objects = [obj for obj in scene.objects if obj.type == 'MESH']; need(len(objects) == 1, 'Exactly one raw mesh required')
obj = objects[0]; mesh = obj.data
need(not obj.modifiers and max(abs(obj.matrix_world[i][j] - (i == j)) for i in range(4) for j in range(4)) < 1e-7,
     'Raw mesh transform or modifiers differ')
expected = (metadata['vertices'], metadata['triangles'])
need((len(mesh.vertices), len(mesh.polygons)) == expected and len(mesh.loops) == 3 * expected[1], 'Imported C topology differs')
coordinates = array.array('f', [0]) * (3 * expected[0]); mesh.vertices.foreach_get('co', coordinates)
indices = array.array('i', [0]) * (3 * expected[1]); mesh.loops.foreach_get('vertex_index', indices)
v = np.load(BASE / 'mesh/vertices.npy', mmap_mode='r', allow_pickle=False)
f = np.load(BASE / 'mesh/faces.npy', mmap_mode='r', allow_pickle=False)
try:
    need(hashlib.sha256(coordinates.tobytes()).hexdigest() == hashlib.sha256(v.tobytes()).hexdigest()
         and hashlib.sha256(indices.tobytes()).hexdigest() == hashlib.sha256(f.tobytes()).hexdigest(), 'Imported C vertex/face payload differs')
finally:
    v._mmap.close(); f._mmap.close()
del coordinates, indices, v, f
mesh.polygons.foreach_set('use_smooth', [False] * expected[1]); bpy.context.view_layer.update()
bbox = [[min(point[i] for point in obj.bound_box) for i in range(3)],
        [max(point[i] for point in obj.bound_box) for i in range(3)]]
need(max(abs(bbox[i][j] - metadata['bounds'][('min', 'max')[i]][j]) for i in range(2) for j in range(3)) < 1e-6,
     'Raw C import bounds differ')

# Freeze framing and lighting to the original scene in world coordinates.
# Never recenter, scale or fit C independently to make the comparison look better.
center = (Vector(original_mesh['bounds']['min']) + Vector(original_mesh['bounds']['max'])) * .5
extent = camera_spec['extent']
corners = [Vector(point) for point in itertools.product(*zip(*bbox))]
material = bpy.data.materials.new('Untextured matte geometry'); material.use_nodes = True
shader = material.node_tree.nodes.get('Principled BSDF')
shader.inputs['Base Color'].default_value = (.43, .43, .43, 1); shader.inputs['Roughness'].default_value = .85
mesh.materials.clear(); mesh.materials.append(material)
world = bpy.data.worlds.new('Neutral background'); world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Color'].default_value = (.14, .16, .19, 1)
world.node_tree.nodes['Background'].inputs['Strength'].default_value = .6; scene.world = world
for name, offset, energy in (('Key', (2, -3, 3), 500), ('Fill', (-3, -1, 1.5), 300), ('Back', (-1, 3, 3), 400)):
    light = bpy.data.lights.new(name, 'AREA'); light.energy = energy * extent**2; light.size = 2.5 * extent
    lamp = bpy.data.objects.new(name, light); scene.collection.objects.link(lamp)
    lamp.location = center + Vector(offset) * extent; aim(lamp, center)
data = bpy.data.cameras.new('Orthographic geometry inspection'); data.type = 'ORTHO'
camera = bpy.data.objects.new('Orthographic geometry inspection', data)
scene.collection.objects.link(camera); scene.camera = camera
data.clip_start = .001 * extent; data.clip_end = 20 * extent

views = []; began = time.monotonic()
for old in original['views']:
    camera.location = Vector(old['camera_location']); right, up = aim(camera, center)
    data.ortho_scale = old['ortho_scale']
    need(scene.cycles.device == 'CPU' and scene.render.threads == 4 and not obj.modifiers,
         'CPU rendering safeguards changed')
    need((len(mesh.vertices), len(mesh.polygons)) == expected, 'Render altered topology')
    need(not any(poly.use_smooth for poly in mesh.polygons), 'Render smoothing denied')
    # Report clipping rather than silently changing the matched camera.
    half = data.ortho_scale / 2
    contained = all(abs((point - center).dot(right)) <= half and abs((point - center).dot(up)) <= half for point in corners)
    scene.render.filepath = str(OUTPUT / old['name'])
    bpy.ops.render.render(write_still=True)
    views.append(dict(name=old['name'], azimuth_degrees=old['azimuth_degrees'], camera_location=list(camera.location),
                      ortho_scale=data.ortho_scale, sha256=digest(OUTPUT / old['name']),
                      candidate_bounds_inside_original_frame=contained))
    print('MATCHED_CPU_GEOMETRY_VIEW=' + old['name'], flush=True)

need(digest(MESH) == source_sha and (MESH.stat().st_size, MESH.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns),
     'Rendering changed the raw C mesh')
policy.verify_snapshot(inputs['immutable_original']); policy.verify_snapshot(inputs['decoded_inputs'])
policy.verify_snapshot(inputs['published_CPU_sources'])
result = dict(classification='PASS_CPU_PREVIEWS', source_PLY_sha256=source_sha, original_PLY_sha256=policy.ORIGINAL_SHA,
              vertices=expected[0], triangles=expected[1], imported_vertex_and_face_payloads_exact=True,
              geometry_unchanged=True, modifiers=[], smooth_shading=False, engine='CYCLES', device='CPU', samples=8,
              threads=4, resolution=[768, 768], up_axis='+Z', material='temporary neutral untextured preview only',
              blender_version=bpy.app.version_string, import_settings=kwargs, views=views,
              matched_original_cameras=True, original_target=list(center), original_light_scale=extent,
              all_candidate_bounds_inside_original_frames=all(row['candidate_bounds_inside_original_frame'] for row in views),
              elapsed_seconds=time.monotonic() - began)
policy.validate_render_metadata(original, result, source_sha)
policy.save(OUTPUT / 'render-metadata.json', result)
print('MATCHED_CPU_GEOMETRY_PREVIEWS=PASS_VISUAL_UNASSESSED', flush=True)
