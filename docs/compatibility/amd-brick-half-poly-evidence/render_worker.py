"""Build the assessed index-only candidate and render one matched CPU view."""
import sys
sys.dont_write_bytecode = True
import argparse, gc, json, time, traceback
from pathlib import Path
E = Path(__file__).resolve().parent
sys.path.insert(0, str(E))
from inputs import P, MESH, SCENE, IMPORTED, REFERENCE, original, pinned, authenticate, verify_stage, plain, digest, save, require


def validate_candidate(vertices, faces, ids, colours, source_vertices, source_colours):
    """Validate every assessed row without moving or recolouring source vertices."""
    import numpy as np
    require(vertices.dtype == np.float32 and vertices.ndim == 2 and vertices.shape[1] == 3 and 0 < len(vertices) <= 1_957_891, 'Compact F32 positions')
    require(faces.dtype == np.int32 and faces.ndim == 2 and faces.shape[1] == 3 and 0 < len(faces) <= 4_173_584, 'Bounded compact I32 triangles')
    require(ids.dtype == np.int32 and ids.shape == (len(vertices),), 'Exact compact original-ID table')
    require(colours.dtype == np.uint8 and colours.shape == (len(vertices), 4), 'Compact native RGBA bytes')
    require(source_vertices.shape == (1_957_891, 3) and source_vertices.dtype == np.float32 and source_colours.shape == (1_957_891, 4) and source_colours.dtype == np.uint8, 'Authenticated original source tables')
    require((ids >= 0).all() and (ids < len(source_vertices)).all() and (ids[1:] > ids[:-1]).all(), 'Sorted unique original source IDs')
    for start in range(0, len(vertices), 32768):
        part = slice(start, start + 32768)
        require(np.isfinite(vertices[part]).all(), 'Finite candidate positions')
        require(np.array_equal(vertices[part].view(np.uint32), source_vertices[ids[part]].view(np.uint32)), 'Every candidate position retains original F32 bits')
        require(np.array_equal(colours[part], source_colours[ids[part]]) and (colours[part, 3] == 255).all(), 'Every candidate native byte retains its original source ID')
    used = np.zeros(len(vertices), dtype=np.bool_)
    for start in range(0, len(faces), 32768):
        part = faces[start:start + 32768]
        require((part >= 0).all() and (part < len(vertices)).all(), 'Candidate compact indices in range')
        used[part.ravel()] = True
    require(used.all(), 'Exact referenced-vertex compaction without unused rows')


def geometry_readback(obj, vertices, faces):
    import numpy as np
    mesh = obj.data
    require(len(mesh.vertices) == len(vertices) and len(mesh.polygons) == len(faces) and len(mesh.loops) == len(faces) * 3, 'Assessed candidate counts unchanged')
    require(not obj.modifiers and not obj.parent and len(mesh.uv_layers) == 0 and not mesh.has_custom_normals, 'No modifiers, UVs, parent or custom normals')
    require(all(obj.matrix_world[i][j] == (i == j) for i in range(4) for j in range(4)), 'Original identity world transform')
    actual = np.empty(vertices.shape, dtype=np.float32)
    mesh.vertices.foreach_get('co', actual.ravel())
    require(np.array_equal(actual.view(np.uint32), vertices.view(np.uint32)), 'All candidate position F32 bits exact')
    del actual
    actual = np.empty(faces.shape, dtype=np.int32)
    mesh.loops.foreach_get('vertex_index', actual.ravel())
    require(np.array_equal(actual, faces), 'All candidate oriented triangle indices exact')
    del actual
    starts = np.empty(len(faces), dtype=np.int32)
    totals = np.empty(len(faces), dtype=np.int32)
    smooth = np.empty(len(faces), dtype=np.bool_)
    mesh.polygons.foreach_get('loop_start', starts)
    mesh.polygons.foreach_get('loop_total', totals)
    mesh.polygons.foreach_get('use_smooth', smooth)
    require(np.array_equal(starts, np.arange(len(faces), dtype=np.int32) * 3) and (totals == 3).all() and not smooth.any(), 'Flat ordered triangles without smoothing')


def colour_readback(mesh, faces, colours):
    import numpy as np
    require(len(mesh.color_attributes) == 1, 'One candidate colour attribute')
    attribute = mesh.color_attributes[0]
    require(attribute.domain == 'CORNER' and attribute.data_type == 'BYTE_COLOR' and len(attribute.data) == faces.size, 'Native BYTE_COLOR/CORNER attribute')
    actual = np.empty((faces.size, 4), dtype=np.float32)
    attribute.data.foreach_get('color_srgb', actual.ravel())
    require(np.isfinite(actual).all() and (actual >= 0).all() and (actual <= 1).all(), 'Finite native byte fractions')
    for start in range(0, len(faces), 32768):
        expected = colours[faces[start:start + 32768].ravel()]
        block = actual[start * 3:(start + len(expected) // 3) * 3]
        rounded = np.rint(block * 255).astype(np.uint8)
        require(np.array_equal(rounded, expected) and np.max(np.abs(block - rounded.astype(np.float32) / 255)) < 1e-7, 'Every native corner byte matches its assessed original source ID')


def build_candidate(bpy, vertices, faces, colours, material, attribute_name):
    import numpy as np
    mesh = bpy.data.meshes.new('Assessed half-poly candidate')
    mesh.vertices.add(len(vertices))
    mesh.loops.add(faces.size)
    mesh.polygons.add(len(faces))
    mesh.vertices.foreach_set('co', vertices.ravel())
    mesh.loops.foreach_set('vertex_index', faces.ravel())
    mesh.polygons.foreach_set('loop_start', np.arange(len(faces), dtype=np.int32) * 3)
    mesh.polygons.foreach_set('loop_total', np.full(len(faces), 3, dtype=np.int32))
    mesh.polygons.foreach_set('use_smooth', np.zeros(len(faces), dtype=np.bool_))
    # Do not validate, weld, repair, normalise, or otherwise modify the assessed mesh.
    mesh.update(calc_edges=True)
    attribute = mesh.color_attributes.new(name=attribute_name, type='BYTE_COLOR', domain='CORNER')
    mesh.color_attributes.render_color_index = 0
    rgba = colours[faces.ravel()].astype(np.float32)
    rgba /= np.float32(255)
    # Native encoded-byte setter avoids introducing a fresh linear transfer/rounding.
    attribute.data.foreach_set('color_srgb', rgba.ravel())
    del rgba
    mesh.materials.append(material)
    obj = bpy.data.objects.new('Assessed half-poly candidate', mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def configure_studio(bpy, scratch, metadata):
    """Use the same camera, light, exposure and CPU settings as the accepted view."""
    prior = metadata['views'][0]
    settings = metadata['settings_before']
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    scene.cycles.denoising_use_gpu = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.seed = 0
    scene.cycles.use_animated_seed = False
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 8
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 4
    scene.render.resolution_x = scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.use_persistent_data = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.compression = 15
    scene.view_settings.view_transform = settings['view_transform']
    scene.view_settings.look = settings['look']
    scene.view_settings.exposure = 2
    scene.view_settings.gamma = settings['gamma']
    bpy.context.preferences.filepaths.temporary_directory = str(scratch)
    cycles = bpy.context.preferences.addons.get('cycles')
    if cycles:
        cycles.preferences.compute_device_type = 'NONE'
        for device in cycles.preferences.devices:
            device.use = device.type == 'CPU'
    for layer in scene.view_layers:
        layer.cycles.use_denoising = True
        layer.material_override = None
    world = bpy.data.worlds.new('Accepted neutral studio')
    world.use_nodes = True
    background = world.node_tree.nodes['Background']
    background.inputs['Color'].default_value = settings['world_linear_RGBA']
    background.inputs['Strength'].default_value = settings['world_strength']
    scene.world = world
    camera_data = bpy.data.cameras.new('Original positive X orthographic')
    camera_data.type = 'ORTHO'
    camera_data.ortho_scale = prior['ortho_scale']
    camera_data.clip_start, camera_data.clip_end = prior['clip']
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    scene.collection.objects.link(camera)
    camera.location = prior['camera_location']
    camera.rotation_mode = 'QUATERNION'
    camera.rotation_quaternion = prior['camera_quaternion']
    scene.camera = camera
    lamps = []
    for item in prior['actual_lights']:
        light = bpy.data.lights.new(item['name'], 'AREA')
        light.energy, light.size, light.color = item['energy'], item['size'], item['color']
        lamp = bpy.data.objects.new(light.name, light)
        scene.collection.objects.link(lamp)
        lamp.location = item['location']
        lamp.rotation_mode = 'QUATERNION'
        lamp.rotation_quaternion = item['quaternion']
        lamps.append(lamp)

    def state():
        return dict(engine=scene.render.engine, device=scene.cycles.device, samples=scene.cycles.samples,
                    denoising=scene.cycles.use_denoising, denoiser=scene.cycles.denoiser, denoising_use_gpu=scene.cycles.denoising_use_gpu,
                    adaptive_sampling=scene.cycles.use_adaptive_sampling, seed=scene.cycles.seed, animated_seed=scene.cycles.use_animated_seed,
                    bounces=[scene.cycles.max_bounces, scene.cycles.diffuse_bounces, scene.cycles.glossy_bounces, scene.cycles.transparent_max_bounces],
                    threads_mode=scene.render.threads_mode, threads=scene.render.threads,
                    resolution=[scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage],
                    film_transparent=scene.render.film_transparent, compositing=scene.render.use_compositing, sequencer=scene.render.use_sequencer, persistent_data=scene.render.use_persistent_data,
                    file_format=scene.render.image_settings.file_format, color_mode=scene.render.image_settings.color_mode, color_depth=scene.render.image_settings.color_depth,
                    exposure=scene.view_settings.exposure, view_transform=scene.view_settings.view_transform, look=scene.view_settings.look, gamma=scene.view_settings.gamma,
                    world_linear_RGBA=list(background.inputs['Color'].default_value), world_strength=background.inputs['Strength'].default_value,
                    camera_location=list(camera.location), camera_quaternion=list(camera.rotation_quaternion), ortho_scale=camera_data.ortho_scale, clip=[camera_data.clip_start, camera_data.clip_end],
                    lights=[dict(location=list(x.location), quaternion=list(x.rotation_quaternion), energy=x.data.energy, size=x.data.size, color=list(x.data.color)) for x in lamps])
    actual = state()
    require(actual['camera_location'] == prior['camera_location'] and actual['camera_quaternion'] == prior['camera_quaternion'] and actual['ortho_scale'] == prior['ortho_scale'] and actual['clip'] == prior['clip'], 'Exact accepted +X camera')
    require(actual['lights'] == [{key: item[key] for key in ('location', 'quaternion', 'energy', 'size', 'color')} for item in prior['actual_lights']], 'Exact accepted studio lights')
    require(actual['world_linear_RGBA'] == settings['world_linear_RGBA'] and actual['world_strength'] == settings['world_strength'], 'Exact accepted neutral world')
    return scene, state


def run(output):
    reader = pinned('half_poly_original_material_reader', original.MATERIAL_READER, original.PINS[original.MATERIAL_READER])
    reader.await_host()
    import bpy
    import numpy as np
    out = P/'render'
    require(Path(output) == out, 'Exact private render output')
    reader.OUT = out
    reader.initial_output(out, bpy.app.tempdir)
    legacy = reader.load_legacy()
    scratch = legacy.guard_python_writes(out)
    require(bpy.app.background and tuple(bpy.app.version) == (5, 2, 1), 'Pinned Blender5.2.1 background')
    began = time.monotonic()
    record = dict(classification='RUNNING', fresh_GLTF_imports=0, render_count=0, files={})
    try:
        watched = authenticate()
        assessed = verify_stage('assess')
        require({'vertices.npy', 'faces.npy', 'source-vertex-ids.npy', 'color-bytes.npy'} <= set(assessed['files']), 'Exact assessed candidate arrays present')
        watched += [P/'assess/result.json', *[P/'assess'/name for name in assessed['files']]]
        watched = list(dict.fromkeys(watched))
        before = {str(path): digest(path) for path in watched}
        parent = json.loads((IMPORTED/'result.json').read_text())
        require(parent['classification'] == 'PASS' and digest(plain(SCENE)) == parent['files']['inspected.blend']['sha256'], 'Exact verified saved source scene')
        require(bpy.ops.wm.open_mainfile(filepath=str(SCENE), load_ui=False, use_scripts=False) == {'FINISHED'}, 'Open saved material source without GLB import')
        objects = list(bpy.context.scene.objects)
        require(len(objects) == 1 and objects[0].type == 'MESH', 'Original sole imported mesh')
        source_obj = objects.pop()
        material_record = reader.material_readback(bpy, source_obj)
        require(material_record == parent['material'], 'Original accepted imported material')
        material = source_obj.data.materials[0]
        source_mesh = source_obj.data
        bpy.data.objects.remove(source_obj, do_unlink=True)
        bpy.data.meshes.remove(source_mesh)
        del source_obj, source_mesh, objects
        gc.collect()
        require(len(bpy.data.meshes) == 0 and not bpy.context.scene.objects, 'Full original mesh removed before candidate allocation')
        arrays = {name: np.load(P/'assess'/name, mmap_mode='r', allow_pickle=False) for name in ('vertices.npy', 'faces.npy', 'source-vertex-ids.npy', 'color-bytes.npy')}
        vertices, faces, ids, colours = [arrays[name] for name in ('vertices.npy', 'faces.npy', 'source-vertex-ids.npy', 'color-bytes.npy')]
        source_vertices = np.load(MESH/'vertices.npy', mmap_mode='r', allow_pickle=False)
        source_colours = np.load(IMPORTED/'oracle-colour-bytes.npy', mmap_mode='r', allow_pickle=False)
        validate_candidate(vertices, faces, ids, colours, source_vertices, source_colours)
        del source_vertices, source_colours
        obj = build_candidate(bpy, vertices, faces, colours, material, material_record['attribute_name'])
        geometry_readback(obj, vertices, faces)
        colour_readback(obj.data, faces, colours)
        require(reader.material_readback(bpy, obj) == material_record, 'Candidate uses the exact accepted imported shader/material')
        gc.collect()
        metadata = json.loads(original.BASELINE.read_text())
        scene, state = configure_studio(bpy, scratch, metadata)
        expected = state()
        preparation = time.monotonic() - began
        require(preparation < 100, 'Leave80seconds for the one bounded CPU frame')
        png = out/'view-pos-x-ev2.png'
        require(not png.exists(), 'Exclusive single candidate frame')
        scene.render.filepath = str(png)
        record['render_count'] = 1
        save(out/'before-render.json', dict(render_count=1, fresh_GLTF_imports=0, preparation_seconds=preparation, vertices=len(vertices), triangles=len(faces), settings=expected))
        print('HALF_POLY_ONE_CPU_FRAME_START', flush=True)
        stamp = time.monotonic()
        require(bpy.ops.render.render(write_still=True) == {'FINISHED'}, 'One matched CPU candidate frame')
        render_seconds = time.monotonic() - stamp
        require(state() == expected, 'CPU settings, camera, world and lights unchanged')
        geometry_readback(obj, vertices, faces)
        colour_readback(obj.data, faces, colours)
        require(reader.material_readback(bpy, obj) == material_record, 'Accepted shader/material unchanged after render')
        require(before == {str(path): digest(path) for path in watched}, 'Original and assessed inputs unchanged')
        record.update(classification='PASS', files={png.name: dict(bytes=plain(png).stat().st_size, sha256=digest(png))},
                      vertices=len(vertices), triangles=len(faces), original_triangles=4_173_584, target_triangles=2_086_792,
                      actual_triangle_reduction=1-len(faces)/4_173_584, assessed_result_SHA256=digest(P/'assess/result.json'),
                      saved_import_scene_SHA256=digest(SCENE), reference_preview_SHA256=digest(REFERENCE),
                      candidate_geometry_exact_before_and_after=True, native_corner_bytes_exact_before_and_after=True,
                      original_source_positions_and_native_bytes_exact=True, original_mesh_removed_before_candidate_allocation=True,
                      imported_material_unchanged=True, original_assets_unchanged=True, settings=expected,
                      preparation_seconds=preparation, render_seconds=render_seconds, elapsed_seconds=time.monotonic()-began,
                      comparison_scope='Same accepted native colour bytes, material, +X camera, studio, flat shading and +2EV; reduced assessed geometry only')
        save(out/'result.json', record)
        print('HALF_POLY_CPU_VIEW_PASS', flush=True)
    except BaseException as error:
        record.update(classification='NO-GO', error=str(error), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-began)
        save(out/'failure.json', record)
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    arguments = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
    run(arguments.output)
