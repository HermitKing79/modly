"""Inspect the existing brick PLY once, under render_host.py containment.

No model execution, mesh export, .blend save, modifiers, or geometry editing.
The host sends GO_START after Job assignment, then GO_REMAINING only after
checking the first +X PNG and its bounded resource/timing receipt. Both gates
require the exact ASCII token plus LF within 30 seconds.

blender --background --factory-startup --disable-autoexec --threads 4
  --python-exit-code 1 --python render_views.py -- --output PRIVATE_OUTPUT
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import struct
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT.parent / "artifacts/amd-brick-cpu-mesh-8473579/mesh/mesh.ply"
EXPECTED_OUTPUT = ROOT.parent / "artifacts/amd-brick-visual-inspection-b536fb6/renders"
SOURCE_SHA256 = "ec9736e2510c26303196c4e3fc2045a1a1d36401b2576ee68f102ee29b551ffd"
COUNTS = (1957891, 4173584)
RESOLUTION = 1024
SAMPLES = 32
THREADS = 4
VIEWS = (
    ("pos-x", (1, 0, 0)), ("neg-x", (-1, 0, 0)),
    ("pos-y", (0, 1, 0)), ("neg-y", (0, -1, 0)),
    ("pos-z", (0, 0, 1)), ("neg-z", (0, 0, -1)),
    ("iso-a", (1, -1, 1)), ("iso-b", (-1, 1, -1)),
)


def view_basis(direction):
    """Camera right/up/forward; +Z up, with +Y up for the two pole views."""
    length = math.sqrt(sum(value * value for value in direction))
    if not math.isfinite(length) or length <= 0:
        raise ValueError("A finite nonzero camera direction is required")
    forward = tuple(-value / length for value in direction)
    reference = (0, 1, 0) if abs(forward[2]) > .999 else (0, 0, 1)
    right = (forward[1] * reference[2] - forward[2] * reference[1],
             forward[2] * reference[0] - forward[0] * reference[2],
             forward[0] * reference[1] - forward[1] * reference[0])
    norm = math.sqrt(sum(value * value for value in right))
    right = tuple(value / norm for value in right)
    up = (right[1] * forward[2] - right[2] * forward[1],
          right[2] * forward[0] - right[0] * forward[2],
          right[0] * forward[1] - right[1] * forward[0])
    return right, up, forward


def sphere_frame(bbox):
    """One shared scale fits the bounding sphere in every square view."""
    spans = tuple(bbox[1][axis] - bbox[0][axis] for axis in range(3))
    radius = .5 * math.sqrt(sum(value * value for value in spans))
    if not math.isfinite(radius) or radius <= 0 or min(spans) < 0:
        raise ValueError("Finite nondegenerate ordered bounds are required")
    center = tuple((bbox[0][axis] + bbox[1][axis]) * .5 for axis in range(3))
    return center, radius, 2 * radius * 1.08


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, record):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def await_host(token):
    """Bounded stdin wait; a missing/extra/malformed token stops this attempt."""
    received = queue.Queue(maxsize=1)

    def read_line():
        try:
            received.put(sys.stdin.buffer.readline(64))
        except BaseException as exc:
            received.put(exc)

    print("BRICK_VISUAL_READY_" + token, flush=True)
    threading.Thread(target=read_line, daemon=True).start()
    try:
        value = received.get(timeout=30)
    except queue.Empty as exc:
        raise RuntimeError(token + " host gate timed out after 30 seconds") from exc
    if value != (token + "\n").encode("ascii"):
        raise RuntimeError(token + " host gate rejected missing or malformed token")


def image_record(path):
    with path.open("rb") as stream:
        header = stream.read(24)
    assert header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR"
    width, height = struct.unpack(">II", header[16:24])
    assert (width, height) == (RESOLUTION, RESOLUTION)
    return {"filename": path.name, "path": str(path), "bytes": path.stat().st_size,
            "sha256": digest(path), "width": width, "height": height}


def run(output):
    import bpy
    import numpy as np
    from mathutils import Matrix, Vector

    began = time.monotonic()
    assert output.resolve() == EXPECTED_OUTPUT.resolve(), "Private bounded output directory required"
    assert output.is_dir() and not output.is_symlink()
    pngs = {name: output / ("view-" + name + ".png") for name, _ in VIEWS}
    pngs["silhouette-pos-z"] = output / "silhouette-pos-z.png"
    assert not any(path.exists() for path in (*pngs.values(), output / "first-view.json",
                                            output / "render-metadata.json")), "Fresh output names required"
    record = {"classification": "RUNNING", "source_PLY_sha256_expected": SOURCE_SHA256,
              "vertex_count": COUNTS[0], "face_count": COUNTS[1], "views": [],
              "engine": "CYCLES", "device": "CPU", "samples": SAMPLES, "threads": THREADS,
              "resolution": [RESOLUTION, RESOLUTION], "denoising": True,
              "denoiser": "OPENIMAGEDENOISE", "denoising_use_gpu": False,
              "geometry_unchanged": False, "mesh_import_count": 0, "flat_shading": True,
              "Blender_version": bpy.app.version_string}
    try:
        await_host("GO_START")  # No source file has been read before Job admission.
        source_paths = (SOURCE, SOURCE.with_name("vertices.npy"), SOURCE.with_name("faces.npy"))
        assert all(path.is_file() and not path.is_symlink() for path in source_paths)
        source_stats = {path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in source_paths}
        assert digest(SOURCE) == SOURCE_SHA256, "Source PLY identity differs"

        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = SAMPLES
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.denoising_use_gpu = False
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.max_bounces = 2
        scene.cycles.diffuse_bounces = 1
        scene.cycles.glossy_bounces = 1
        scene.render.threads_mode = "FIXED"
        scene.render.threads = THREADS
        scene.render.resolution_x = scene.render.resolution_y = RESOLUTION
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
        bpy.context.preferences.filepaths.temporary_directory = str(output / "blender-scratch")
        for layer in scene.view_layers:
            layer.cycles.use_denoising = True

        props = set(bpy.ops.wm.ply_import.get_rna_type().properties.keys())
        required = {"merge_verts": False, "global_scale": 1.0, "use_scene_unit": False,
                    "forward_axis": "Y", "up_axis": "Z"}
        assert all(key in props for key in required), "Explicit unchanged PLY import settings required"
        kwargs = {"filepath": str(SOURCE), **required}
        assert bpy.ops.wm.ply_import(**kwargs) == {"FINISHED"}
        record["mesh_import_count"] = 1
        objects = [obj for obj in scene.objects if obj.type == "MESH"]
        assert len(objects) == 1
        obj, = objects
        mesh = obj.data
        assert not obj.modifiers
        assert max(abs(obj.matrix_world[i][j] - (i == j)) for i in range(4) for j in range(4)) < 1e-7
        # Shading flags only; positions, indices, winding, and object transform are untouched.
        mesh.polygons.foreach_set("use_smooth", np.zeros(COUNTS[1], dtype=np.bool_))
        bpy.context.view_layer.update()

        def geometry_payloads():
            assert not obj.modifiers and (len(mesh.vertices), len(mesh.polygons)) == COUNTS
            assert max(abs(obj.matrix_world[i][j] - (i == j)) for i in range(4) for j in range(4)) < 1e-7
            assert len(mesh.loops) == COUNTS[1] * 3
            co = np.empty(COUNTS[0] * 3, dtype="<f4")
            loops = np.empty(COUNTS[1] * 3, dtype="<i4")
            totals = np.empty(COUNTS[1], dtype="<i4")
            starts = np.empty(COUNTS[1], dtype="<i4")
            smooth = np.empty(COUNTS[1], dtype=np.bool_)
            mesh.vertices.foreach_get("co", co)
            mesh.loops.foreach_get("vertex_index", loops)
            mesh.polygons.foreach_get("loop_total", totals)
            mesh.polygons.foreach_get("loop_start", starts)
            mesh.polygons.foreach_get("use_smooth", smooth)
            assert np.all(totals == 3) and not np.any(smooth)
            assert np.array_equal(starts, np.arange(COUNTS[1], dtype=np.int32) * 3)
            payloads = {}
            for name, array in (("vertices.npy", co.reshape(-1, 3)), ("faces.npy", loops.reshape(-1, 3))):
                original = np.load(SOURCE.with_name(name), mmap_mode="r", allow_pickle=False)
                assert original.dtype == array.dtype and original.shape == array.shape
                actual = hashlib.sha256(array.view(np.uint8)).hexdigest()
                assert actual == hashlib.sha256(original.view(np.uint8)).hexdigest(), "Ordered bytes changed: " + name
                payloads[name] = actual
            return payloads

        payload_before = geometry_payloads()
        corners = [Vector(point) for point in obj.bound_box]
        bbox = [[min(point[i] for point in corners) for i in range(3)],
                [max(point[i] for point in corners) for i in range(3)]]
        assert all(math.isfinite(value) for row in bbox for value in row)
        center_tuple, radius, ortho_scale = sphere_frame(bbox)
        center = Vector(center_tuple)

        def aim(target_object, target):
            right, up, forward = view_basis(tuple(target_object.location - target))
            target_object.rotation_mode = "QUATERNION"
            target_object.rotation_quaternion = Matrix((right, up, tuple(-v for v in forward))).transposed().to_quaternion()
            return Vector(right), Vector(up), -Vector(forward)

        clay = bpy.data.materials.new("Neutral matte clay inspection override")
        clay.use_nodes = True
        shader = clay.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (.45, .45, .45, 1)
        shader.inputs["Roughness"].default_value = .9
        shader.inputs["Metallic"].default_value = 0
        shader.inputs["Emission Strength"].default_value = 0
        for layer in scene.view_layers:
            layer.material_override = clay
        world = bpy.data.worlds.new("Neutral studio world")
        world.use_nodes = True
        background = world.node_tree.nodes["Background"]
        background.inputs["Color"].default_value = (.12, .12, .12, 1)
        background.inputs["Strength"].default_value = .5
        scene.world = world
        light_records, lamps = [], []
        for name, offset, energy in (("Key", (-2.5, 3, 4), 80),
                                     ("Fill", (3, 1, 2), 40),
                                     ("Rim", (0, 3, -3), 60)):
            data = bpy.data.lights.new(name, "AREA")
            data.energy, data.size, data.color = energy * radius * radius, 2.5 * radius, (1, 1, 1)
            lamp = bpy.data.objects.new(name, data)
            scene.collection.objects.link(lamp)
            lamps.append((lamp, offset))
            light_records.append({"name": name, "camera_relative_radius_offset": offset,
                                  "energy": data.energy, "size": data.size, "color": list(data.color)})
        data = bpy.data.cameras.new("Shared orthographic inspection camera")
        data.type = "ORTHO"
        data.ortho_scale = ortho_scale
        data.clip_start, data.clip_end = .001 * radius, 20 * radius
        camera = bpy.data.objects.new(data.name, data)
        scene.collection.objects.link(camera)
        scene.camera = camera
        record.update({"bbox": bbox, "bounding_sphere_center": center_tuple, "bounding_sphere_radius": radius,
                       "shared_ortho_scale": ortho_scale, "framing_margin_fraction": .08,
                       "import_settings": kwargs, "geometry_payloads_before": payload_before,
                       "canonical_up_axis": "+Z; pole views use screen +Y up",
                       "camera_direction_convention": "Named axis is camera position relative to bounds center",
                       "lighting": "Camera-relative soft neutral three-area rig; offsets in right/up/toward-camera radius units",
                       "source_colors_used": False,
                       "lights": light_records, "clay_linear_RGB": [.45, .45, .45], "clay_roughness": .9,
                       "world_linear_RGB": [.12, .12, .12], "world_strength": .5,
                       "view_transform": "AgX", "look": "None", "exposure": 0, "gamma": 1})

        def render(name, direction):
            started = time.monotonic()
            camera.location = center + Vector(direction).normalized() * (4 * radius)
            right, up, outward = aim(camera, center)
            for lamp, offset in lamps:
                lamp.location = center + radius * (right * offset[0] + up * offset[1] + outward * offset[2])
                aim(lamp, center)
            assert scene.cycles.device == "CPU" and scene.render.threads == THREADS
            assert scene.cycles.denoising_use_gpu is False
            scene.render.filepath = str(pngs[name])
            bpy.ops.render.render(write_still=True)
            view = {"name": name, "PNG": pngs[name].name, "axis_direction": direction, "camera_location": list(camera.location),
                    "camera_quaternion": list(camera.rotation_quaternion), "camera_right": list(right),
                    "camera_up": list(up), "ortho_scale": data.ortho_scale,
                    **image_record(pngs[name]), "seconds": time.monotonic() - started,
                    "elapsed_seconds": time.monotonic() - began}
            record["views"].append(view)
            print("BRICK_VISUAL_RENDERED=" + name, flush=True)
            return view

        first = render(*VIEWS[0])
        assert geometry_payloads() == payload_before
        first_record = {**first, "classification": "FIRST_VIEW_COMPLETE", "device": "CPU",
                        "samples": SAMPLES, "threads": THREADS, "denoising_use_gpu": False,
                        "vertex_count": COUNTS[0], "face_count": COUNTS[1],
                        "source_PLY_sha256": SOURCE_SHA256, "geometry_payloads": payload_before,
                        "geometry_unchanged_after_first_view": True,
                        "elapsed_seconds": time.monotonic() - began}
        save_json(output / "first-view.json", first_record)
        await_host("GO_REMAINING")
        for name, direction in VIEWS[1:]:
            render(name, direction)

        # Render-only silhouette override: opaque black emission on a white world.
        silhouette = bpy.data.materials.new("Black emission silhouette override")
        silhouette.use_nodes = True
        nodes, links = silhouette.node_tree.nodes, silhouette.node_tree.links
        nodes.clear()
        surface = nodes.new("ShaderNodeEmission")
        surface.inputs["Color"].default_value = (0, 0, 0, 1)
        surface.inputs["Strength"].default_value = 1
        links.new(surface.outputs["Emission"], nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
        for layer in scene.view_layers:
            layer.material_override = silhouette
            layer.cycles.use_denoising = False
        background.inputs["Color"].default_value = (1, 1, 1, 1)
        background.inputs["Strength"].default_value = 1
        scene.view_settings.view_transform = "Standard"
        scene.cycles.use_denoising = False
        scene.cycles.samples = 1
        render("silhouette-pos-z", (0, 0, 1))
        record["views"][-1].update({"render_only_silhouette": True, "samples": 1, "denoising": False,
                                     "material": "black emission", "world": "white", "view_transform": "Standard"})

        payload_after = geometry_payloads()
        assert payload_after == payload_before
        assert digest(SOURCE) == SOURCE_SHA256
        assert all((path.stat().st_size, path.stat().st_mtime_ns) == source_stats[path.name] for path in source_paths)
        record.update({"classification": "PASS", "geometry_unchanged": True,
                       "source_PLY_sha256": SOURCE_SHA256, "source_stats_before_and_after": source_stats,
                       "geometry_payloads_after": payload_after, "outputs": {name: image_record(path) for name, path in pngs.items()}})
    except BaseException as exc:
        record.update({"classification": "FAIL", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - began
        save_json(output / "render-metadata.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    run(parser.parse_args(arguments).output)
