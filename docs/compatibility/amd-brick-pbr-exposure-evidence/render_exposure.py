"""Three exposure-only CPU learned-PBR +X previews, in memory only.

Only the bounded render host may launch this script. It admits the owned process
with GO_START, independently checks +X EV0, then sends GO_REMAINING for EV1/EV2.
The source PLY is imported once. No models, Torch, WSL, mesh export, .blend save,
baking, UVs, smoothing, modifiers, or geometry repair are used.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import struct
import sys
import threading
import time


sys.dont_write_bytecode = True
E = Path(__file__).resolve().parent
ROOT = E.parents[2]
SOURCE = ROOT.parent / "artifacts/amd-brick-cpu-mesh-8473579/mesh/mesh.ply"
FIELD = Path("C:/Users/wattz/Documents/ModlyArtifacts/amd-brick-pbr-qualified-a064378/field/texture-field.safetensors")
EXPECTED_OUTPUT = ROOT.parent / "artifacts/amd-brick-pbr-exposure-b0b7be1/renders"
BASELINE = ROOT / "docs/compatibility/amd-brick-visual-inspection-evidence"
NATIVE_PBR = Path("D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD/trellis2/renderers/pbr_mesh_renderer.py")
SOURCE_SHA256 = "ec9736e2510c26303196c4e3fc2045a1a1d36401b2576ee68f102ee29b551ffd"
FIELD_SHA256 = "8cd735630149e28829efe686d30f1c7c255dde76748dc3d525236fb5c2c81394"
COORDINATES_SHA256 = "690670fe58a7021c51fef9fad6457ad8e66715d13a88239adabdffa8a05d9976"
FEATURES_SHA256 = "bc562893e0b89b2b53c279ba00240e8d1edb98830c2471a82077e48b24eb0cbf"
NATIVE_PBR_SHA256 = "ab33ba6d04b2d1b7e518b90372b242cbe00ad9043be66a22e681261b04de7872"
MAPPING_RECEIPT = ROOT / "docs/compatibility/amd-brick-pbr-preview-evidence/mapping-verification.json"
MAPPING_RECEIPT_SHA256 = "84912427ddff6d470ce96215344118b057ab357f3a41ff762612a367549d0eb4"
BASELINE_SCRIPT_SHA256 = "45a4169bf294241d167858d1f92f440014992c860d0599531faad6e44e527412"
BASELINE_METADATA_SHA256 = "394b54c5633598d56c8ad38c00b4712481b2405c91e3ae5bcaf91974f974d6ee"
GEOMETRY_SHA256 = {
    "vertices.npy": "bdfcbbe8f251b33daa64c6832c4c1278b3a29600cf825a022478b0b87bb573c3",
    "faces.npy": "5fa284e1698929f4f0f6c78edc2e26632fcde6e43ad7eb57b36ae6fc9f88d883",
}
COUNTS = (1957891, 4173584)
RESOLUTION, SAMPLES, THREADS = 1024, 32, 4
DEADLINE_SECONDS = 180
CHANNELS = ("red", "green", "blue", "metallic", "roughness", "alpha")
ATTRIBUTE_NAMES = tuple("Trellis_" + channel + "_raw" for channel in CHANNELS)
LINEAR_COLOR_NAME = "Trellis_RGB_linear_native_gamma22"
def exposure_schedule():
    """The approved three frames, all using the unchanged positive-X camera."""
    return (("pos-x", (1, 0, 0), 0), ("pos-x-ev1", (1, 0, 0), 1), ("pos-x-ev2", (1, 0, 0), 2))


VIEWS = exposure_schedule()


def assert_exposure_only(baseline, actual, exposure):
    """Reject every settings difference except the scheduled display exposure."""
    if exposure not in (0, 1, 2) or baseline.get("exposure") != 0 or actual.get("exposure") != exposure:
        raise ValueError("Exact EV0/EV1/EV2 exposure schedule required")
    if "exposure" not in baseline or "exposure" not in actual or set(baseline) != set(actual):
        raise ValueError("Identical complete settings schema required")
    if any(actual[key] != value for key, value in baseline.items() if key != "exposure"):
        raise ValueError("Only exposure may differ from baseline")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_digest(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def save_json(path, record):
    # Serialize before opening so a nonfinite receipt cannot leave a partial file.
    encoded = json.dumps(record, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def read_field_header(path, rows=COUNTS[0]):
    """Validate the complete bounded, nonoverlapping three-tensor layout."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Missing safetensors header length")
        length = int.from_bytes(prefix, "little")
        if not 0 < length <= 65536 or 8 + length > size:
            raise ValueError("Bounded complete safetensors header required")
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate safetensors header key")
                result[key] = value
            return result
        header = json.loads(stream.read(length), object_pairs_hook=unique_pairs)
    if not isinstance(header, dict) or set(header) - {"__metadata__"} != {"coordinates", "features", "raw"}:
        raise ValueError("Exactly the original coordinates, features and raw tensors required")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
        raise ValueError("String-only safetensors metadata required")
    intervals = []
    for name, dtype, columns in (("coordinates", "I32", 4), ("features", "F32", 6), ("raw", "F32", 6)):
        tensor = header[name]
        if not isinstance(tensor, dict) or set(tensor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("Original tensor schema required: " + name)
        offsets = tensor["data_offsets"]
        if tensor["dtype"] != dtype or tensor["shape"] != [rows, columns]:
            raise ValueError("Original tensor rows/channels/dtype required: " + name)
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
            raise ValueError("Integer tensor offsets required")
        start, end = offsets
        if not 0 <= start < end <= size - 8 - length or end - start != rows * columns * 4:
            raise ValueError("Exact in-file tensor extent required: " + name)
        intervals.append((start, end))
    intervals.sort()
    if intervals[0][0] != 0 or any(left[1] != right[0] for left, right in zip(intervals, intervals[1:])) or intervals[-1][1] != size - 8 - length:
        raise ValueError("Tensor payload must be complete and nonoverlapping")
    return 8 + length, header


def linear_rgba(rgb):
    """Native authenticated PBR convention: clamp RGB to [0,1], then **2.2.

    This intentionally matches pbr_mesh_renderer.py:431, including its gamma
    approximation; it is not the piecewise IEC sRGB transfer. No quantization.
    """
    import numpy as np
    if rgb.dtype != np.float32 or rgb.ndim != 2 or rgb.shape[1] != 3 or not np.isfinite(rgb).all():
        raise ValueError("Finite F32[N,3] learned RGB required")
    result = np.ones((len(rgb), 4), dtype="<f4")
    result[:, :3] = np.power(np.clip(rgb, 0, 1), np.float32(2.2))
    return result


def channel_statistics(features, chunk_rows=65536):
    """Read bounded chunks; never clamp or modify the original material field."""
    import numpy as np
    if features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != 6 or not len(features) or chunk_rows <= 0:
        raise ValueError("Finite nonempty F32[N,6] material features required")
    lows, highs = np.full(6, np.inf), np.full(6, -np.inf)
    sums, outside = np.zeros(6, dtype=np.float64), np.zeros(6, dtype=np.int64)
    for start in range(0, len(features), chunk_rows):
        part = features[start:start + chunk_rows]
        if not np.isfinite(part).all():
            raise ValueError("Material feature contains nonfinite values")
        lows = np.minimum(lows, part.min(axis=0))
        highs = np.maximum(highs, part.max(axis=0))
        sums += part.sum(axis=0, dtype=np.float64)
        outside += ((part < 0) | (part > 1)).sum(axis=0)
    return {name: {"min": float(lows[i]), "max": float(highs[i]), "mean": float(sums[i] / len(features)),
                   "outside_0_1": int(outside[i])} for i, name in enumerate(CHANNELS)}


def await_host(token):
    received = queue.Queue(maxsize=1)
    def read_line():
        try:
            received.put(sys.stdin.buffer.readline(64))
        except BaseException as exc:
            received.put(exc)
    print("BRICK_PBR_READY_" + token, flush=True)
    threading.Thread(target=read_line, daemon=True).start()
    try:
        value = received.get(timeout=30)
    except queue.Empty as exc:
        raise RuntimeError(token + " host gate timed out after 30 seconds") from exc
    if value != (token + "\n").encode("ascii"):
        raise RuntimeError(token + " host gate rejected missing or malformed token")


def image_record(path):
    with Path(path).open("rb") as stream:
        header = stream.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("Complete PNG header required")
    width, height = struct.unpack(">II", header[16:24])
    if (width, height) != (RESOLUTION, RESOLUTION):
        raise ValueError("Original 1024-square resolution required")
    return {"filename": path.name, "path": str(path), "bytes": path.stat().st_size,
            "sha256": digest(path), "width": width, "height": height}


def run(output):
    import bpy
    import numpy as np
    from mathutils import Matrix, Vector

    began = time.monotonic()
    output = Path(output)
    assert output.resolve() == EXPECTED_OUTPUT.resolve(), "Exact private bounded output required"
    assert output.is_dir() and not output.is_symlink()
    pngs = {name: output / ("view-" + name + ".png") for name, _, _ in VIEWS}
    assert not any(path.exists() for path in (*pngs.values(), output / "first-view.json", output / "render-metadata.json"))
    record = {"classification": "RUNNING", "source_PLY_sha256_expected": SOURCE_SHA256,
              "material_field_sha256_expected": FIELD_SHA256, "vertex_count": COUNTS[0], "face_count": COUNTS[1],
              "views": [], "engine": "CYCLES", "device": "CPU", "samples": SAMPLES, "threads": THREADS,
              "resolution": [RESOLUTION, RESOLUTION], "denoising": True, "denoiser": "OPENIMAGEDENOISE",
              "denoising_use_gpu": False, "geometry_unchanged": False, "mesh_import_count": 0,
              "flat_shading": True, "saved_blend": False, "mesh_exported": False,
              "comparison": "Only display exposure EV0/EV1/EV2 changes; all frames use the original positive-X camera",
              "Blender_version": bpy.app.version_string}
    try:
        await_host("GO_START")
        source_paths = (SOURCE, SOURCE.with_name("vertices.npy"), SOURCE.with_name("faces.npy"), FIELD, NATIVE_PBR, MAPPING_RECEIPT,
                        BASELINE / "render_views.py", BASELINE / "render-metadata.json")
        assert all(path.is_file() and not path.is_symlink() for path in source_paths)
        source_stats = {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in source_paths}
        pinned = {SOURCE: SOURCE_SHA256, FIELD: FIELD_SHA256, NATIVE_PBR: NATIVE_PBR_SHA256, MAPPING_RECEIPT: MAPPING_RECEIPT_SHA256,
                  BASELINE / "render_views.py": BASELINE_SCRIPT_SHA256,
                  BASELINE / "render-metadata.json": BASELINE_METADATA_SHA256}
        assert all(digest(path) == expected for path, expected in pinned.items()), "Original input/reference identity differs"
        mapping = json.loads(MAPPING_RECEIPT.read_text(encoding="utf-8"))
        correspondence = mapping["full_correspondence"]
        assert mapping["classification"] == "PASS" and mapping["execution"]["inputs_unchanged"] is True
        assert all(correspondence[key] is True for key in (
            "all_coordinate_row_bytes_and_order_equal",
            "all_PLY_and_NPY_vertices_bitwise_equal_independent_original_formula",
            "all_PLY_and_NPY_face_indices_bitwise_equal"))
        assert (correspondence["vertices"], correspondence["triangles"]) == COUNTS
        assert correspondence["coordinate_payload_sha256"] == COORDINATES_SHA256
        assert correspondence["vertex_payload_sha256"] == GEOMETRY_SHA256["vertices.npy"]
        assert correspondence["face_payload_sha256"] == GEOMETRY_SHA256["faces.npy"]
        assert mapping["input_identities"]["material"]["sha256"] == FIELD_SHA256
        assert mapping["input_identities"]["mesh"]["sha256"] == SOURCE_SHA256
        assert SOURCE.stat().st_size == 77751514 and FIELD.stat().st_size == 125318120
        base, tensors = read_field_header(FIELD)
        coordinates = np.memmap(FIELD, mode="r", dtype="<i4", offset=base + tensors["coordinates"]["data_offsets"][0], shape=(COUNTS[0], 4))
        features = np.memmap(FIELD, mode="r", dtype="<f4", offset=base + tensors["features"]["data_offsets"][0], shape=(COUNTS[0], 6))
        assert array_digest(coordinates) == COORDINATES_SHA256, "Original material row order changed"
        assert array_digest(features) == FEATURES_SHA256, "Original material feature payload changed"
        statistics = channel_statistics(features)
        # Reuse the exact previously reviewed camera basis and shared framing.
        spec = importlib.util.spec_from_file_location("brick_clay_camera_contract", BASELINE / "render_views.py")
        baseline_api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline_api)
        baseline = json.loads((BASELINE / "render-metadata.json").read_text(encoding="utf-8"))
        assert baseline["classification"] == "PASS" and baseline["geometry_payloads_before"] == GEOMETRY_SHA256
        baseline_views = {view["name"]: view for view in baseline["views"]}

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

        required = {"merge_verts": False, "global_scale": 1.0, "use_scene_unit": False,
                    "forward_axis": "Y", "up_axis": "Z"}
        assert all(key in bpy.ops.wm.ply_import.get_rna_type().properties.keys() for key in required)
        import_settings = {"filepath": str(SOURCE), **required}
        assert bpy.ops.wm.ply_import(**import_settings) == {"FINISHED"}
        record["mesh_import_count"] = 1
        objects = [obj for obj in scene.objects if obj.type == "MESH"]
        assert len(objects) == 1
        obj, = objects
        mesh = obj.data
        assert (len(mesh.vertices), len(mesh.polygons)) == COUNTS
        mesh.polygons.foreach_set("use_smooth", np.zeros(COUNTS[1], dtype=np.bool_))
        bpy.context.view_layer.update()

        def geometry_payloads():
            assert not obj.modifiers and (len(mesh.vertices), len(mesh.polygons), len(mesh.loops)) == (*COUNTS, COUNTS[1] * 3)
            assert max(abs(obj.matrix_world[i][j] - (i == j)) for i in range(4) for j in range(4)) < 1e-7
            totals = np.empty(COUNTS[1], dtype="<i4")
            starts = np.empty(COUNTS[1], dtype="<i4")
            smooth = np.empty(COUNTS[1], dtype=np.bool_)
            mesh.polygons.foreach_get("loop_total", totals)
            mesh.polygons.foreach_get("loop_start", starts)
            mesh.polygons.foreach_get("use_smooth", smooth)
            assert np.all(totals == 3) and not np.any(smooth)
            assert np.array_equal(starts, np.arange(COUNTS[1], dtype=np.int32) * 3)
            del totals, starts, smooth
            payloads = {}
            for name, collection, prop, dtype, count in (("vertices.npy", mesh.vertices, "co", "<f4", COUNTS[0]),
                                                        ("faces.npy", mesh.loops, "vertex_index", "<i4", COUNTS[1])):
                actual = np.empty(count * 3, dtype=dtype)
                collection.foreach_get(prop, actual)
                original = np.load(SOURCE.with_name(name), mmap_mode="r", allow_pickle=False)
                assert original.dtype == actual.dtype and original.shape == (count, 3)
                expected = GEOMETRY_SHA256[name]
                assert array_digest(actual) == array_digest(original) == expected, "Original ordered geometry changed: " + name
                payloads[name] = expected
            return payloads

        payload_before = geometry_payloads()
        corners = [Vector(point) for point in obj.bound_box]
        bbox = [[min(point[i] for point in corners) for i in range(3)], [max(point[i] for point in corners) for i in range(3)]]
        assert bbox == baseline["bbox"]
        center_tuple, radius, ortho_scale = baseline_api.sphere_frame(bbox)
        assert radius == baseline["bounding_sphere_radius"] and ortho_scale == baseline["shared_ortho_scale"]
        center = Vector(center_tuple)

        def aim(target_object, target):
            right, up, forward = baseline_api.view_basis(tuple(target_object.location - target))
            target_object.rotation_mode = "QUATERNION"
            target_object.rotation_quaternion = Matrix((right, up, tuple(-v for v in forward))).transposed().to_quaternion()
            return Vector(right), Vector(up), -Vector(forward)

        material = bpy.data.materials.new("Genuine learned brick PBR preview")
        material.use_nodes = True
        nodes, links = material.node_tree.nodes, material.node_tree.links
        nodes.clear()
        shader = nodes.new("ShaderNodeBsdfPrincipled")
        shader.inputs["Emission Strength"].default_value = 0
        output_node = nodes.new("ShaderNodeOutputMaterial")
        links.new(shader.outputs["BSDF"], output_node.inputs["Surface"])
        mesh.materials.clear()
        mesh.materials.append(material)
        for layer in scene.view_layers:
            layer.material_override = None
        for index, name in enumerate(ATTRIBUTE_NAMES):
            assert name not in mesh.attributes
            attribute = mesh.attributes.new(name=name, type="FLOAT", domain="POINT")
            attribute.data.foreach_set("value", np.ascontiguousarray(features[:, index]))
        expected_rgba = linear_rgba(features[:, :3])
        color = mesh.color_attributes.new(name=LINEAR_COLOR_NAME, type="FLOAT_COLOR", domain="POINT")
        color.data.foreach_set("color", expected_rgba.ravel())
        rgb_node = nodes.new("ShaderNodeVertexColor")
        rgb_node.layer_name = LINEAR_COLOR_NAME
        links.new(rgb_node.outputs["Color"], shader.inputs["Base Color"])
        for index, target in ((3, "Metallic"), (4, "Roughness"), (5, "Alpha")):
            source_node = nodes.new("ShaderNodeAttribute")
            source_node.attribute_type = "GEOMETRY"
            source_node.attribute_name = ATTRIBUTE_NAMES[index]
            clamp = nodes.new("ShaderNodeClamp")
            clamp.clamp_type = "MINMAX"
            clamp.inputs["Min"].default_value = 0
            clamp.inputs["Max"].default_value = 1
            links.new(source_node.outputs["Fac"], clamp.inputs["Value"])
            links.new(clamp.outputs["Result"], shader.inputs[target])
        expected_rgba_sha = array_digest(expected_rgba)
        del expected_rgba

        def material_readback():
            assert len(mesh.materials) == 1 and mesh.materials[0] == material and material.use_nodes
            assert not shader.inputs["Normal"].is_linked and not output_node.inputs["Displacement"].is_linked
            assert shader.inputs["Emission Strength"].default_value == 0 and not shader.inputs["Emission Strength"].is_linked
            surface = output_node.inputs["Surface"].links
            assert len(surface) == 1 and surface[0].from_node == shader and surface[0].from_socket == shader.outputs["BSDF"]
            assert shader.bl_idname == "ShaderNodeBsdfPrincipled"
            attrs = {}
            actual = np.empty(COUNTS[0], dtype="<f4")
            for index, name in enumerate(ATTRIBUTE_NAMES):
                attribute = mesh.attributes[name]
                assert attribute.domain == "POINT" and attribute.data_type == "FLOAT" and len(attribute.data) == COUNTS[0]
                attribute.data.foreach_get("value", actual)
                assert array_digest(actual) == array_digest(features[:, index]), "Original F32 material channel changed: " + name
                attrs[name] = {"source_tensor": "features", "source_column": index, "data_type": "FLOAT", "domain": "POINT", "raw_unclamped": True,
                               "sha256": array_digest(actual), **statistics[CHANNELS[index]]}
            rgba = np.empty(COUNTS[0] * 4, dtype="<f4")
            assert color.domain == "POINT" and color.data_type == "FLOAT_COLOR" and len(color.data) == COUNTS[0]
            color.data.foreach_get("color", rgba)
            assert array_digest(rgba) == expected_rgba_sha, "Derived native-linear RGB attribute changed"
            edge = shader.inputs["Base Color"].links
            assert len(edge) == 1 and edge[0].from_node == rgb_node and edge[0].from_socket == rgb_node.outputs["Color"]
            assert rgb_node.layer_name == LINEAR_COLOR_NAME
            connections = {"Base Color": {"node": rgb_node.bl_idname, "attribute": LINEAR_COLOR_NAME, "output": "Color"}}
            for index, target in ((3, "Metallic"), (4, "Roughness"), (5, "Alpha")):
                edge = shader.inputs[target].links
                assert len(edge) == 1
                clamp = edge[0].from_node
                assert clamp.bl_idname == "ShaderNodeClamp" and clamp.clamp_type == "MINMAX" and edge[0].from_socket == clamp.outputs["Result"]
                assert clamp.inputs["Min"].default_value == 0 and clamp.inputs["Max"].default_value == 1
                assert not clamp.inputs["Min"].is_linked and not clamp.inputs["Max"].is_linked
                original = clamp.inputs["Value"].links
                assert len(original) == 1
                node = original[0].from_node
                assert node.bl_idname == "ShaderNodeAttribute" and node.attribute_type == "GEOMETRY" and node.attribute_name == ATTRIBUTE_NAMES[index]
                assert original[0].from_socket == node.outputs["Fac"]
                connections[target] = {"node": node.bl_idname, "attribute": node.attribute_name, "output": "Fac", "shader_clamp": [0, 1]}
            assert len(nodes) == 9 and len(links) == 8
            defaults = {}
            for socket in shader.inputs:
                if hasattr(socket, "default_value"):
                    value = socket.default_value
                    defaults[socket.name] = {"linked": socket.is_linked,
                                             "default_value": list(value) if hasattr(value, "__len__") and not isinstance(value, str) else value}
            return {"attributes": attrs, "linear_RGBA_sha256": expected_rgba_sha, "linear_color_attribute": LINEAR_COLOR_NAME,
                    "RGB_conversion": "clamp(original_RGB,0,1)**2.2; authenticated native renderer line 431; F32, no byte quantization",
                    "native_renderer_sha256": NATIVE_PBR_SHA256, "shader": shader.bl_idname,
                    "connections": connections, "Principled_inputs": defaults,
                    "alpha_linked_from_original": True, "alpha_preview": "shader clamp(original alpha,0,1)",
                    "emission_strength": 0, "normal_or_displacement_links": False,
                    "material_surface_render_method": material.surface_render_method}

        material_before = material_readback()
        world = bpy.data.worlds.new("Neutral studio world")
        world.use_nodes = True
        background = world.node_tree.nodes["Background"]
        background.inputs["Color"].default_value = (.12, .12, .12, 1)
        background.inputs["Strength"].default_value = .5
        scene.world = world
        light_records, lamps = [], []
        for name, offset, energy in (("Key", (-2.5, 3, 4), 80), ("Fill", (3, 1, 2), 40), ("Rim", (0, 3, -3), 60)):
            light = bpy.data.lights.new(name, "AREA")
            light.energy, light.size, light.color = energy * radius * radius, 2.5 * radius, (1, 1, 1)
            lamp = bpy.data.objects.new(name, light)
            scene.collection.objects.link(lamp)
            lamps.append((lamp, offset))
            light_records.append({"name": name, "camera_relative_radius_offset": list(offset), "energy": light.energy,
                                  "size": light.size, "color": list(light.color)})
        assert light_records == baseline["lights"]
        camera_data = bpy.data.cameras.new("Shared orthographic inspection camera")
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = ortho_scale
        camera_data.clip_start, camera_data.clip_end = .001 * radius, 20 * radius
        camera = bpy.data.objects.new(camera_data.name, camera_data)
        scene.collection.objects.link(camera)
        scene.camera = camera

        def settings():
            return {"engine": scene.render.engine, "device": scene.cycles.device, "samples": scene.cycles.samples,
                    "threads": scene.render.threads, "threads_mode": scene.render.threads_mode,
                    "resolution": [scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage],
                    "denoising": scene.cycles.use_denoising, "denoiser": scene.cycles.denoiser,
                    "denoising_use_gpu": scene.cycles.denoising_use_gpu, "adaptive_sampling": scene.cycles.use_adaptive_sampling,
                    "bounces": [scene.cycles.max_bounces, scene.cycles.diffuse_bounces, scene.cycles.glossy_bounces],
                    "transparent_max_bounces": scene.cycles.transparent_max_bounces,
                    "film_transparent": scene.render.film_transparent, "compositing": scene.render.use_compositing,
                    "sequencer": scene.render.use_sequencer, "persistent_data": scene.render.use_persistent_data,
                    "view_transform": scene.view_settings.view_transform,
                    "look": scene.view_settings.look, "exposure": scene.view_settings.exposure, "gamma": scene.view_settings.gamma,
                    "world_linear_RGBA": list(background.inputs["Color"].default_value), "world_strength": background.inputs["Strength"].default_value,
                    "material_override": [layer.material_override is None for layer in scene.view_layers]}

        settings_before = settings()
        record.update({"bbox": bbox, "bounding_sphere_center": list(center_tuple), "bounding_sphere_radius": radius,
                       "shared_ortho_scale": camera_data.ortho_scale, "framing_margin_fraction": .08,
                       "import_settings": import_settings, "geometry_payloads_before": payload_before,
                       "material_before": material_before, "channel_statistics": statistics,
                       "settings_before": settings_before, "lights": light_records,
                       "material_coordinates_sha256": COORDINATES_SHA256, "material_features_sha256": FEATURES_SHA256,
                       "mapping_receipt_sha256": MAPPING_RECEIPT_SHA256, "original_direct_row_correspondence_verified": True,
                       "baseline_renderer_sha256": BASELINE_SCRIPT_SHA256, "baseline_metadata_sha256": BASELINE_METADATA_SHA256,
                       "camera_direction_convention": "Named axis is camera position relative to bounds center",
                       "lighting": baseline["lighting"], "source_colors_used": False, "learned_colors_used": True})

        def render(name, direction, exposure):
            assert time.monotonic() - began < DEADLINE_SECONDS - 25, "Insufficient bounded time for another view"
            started = time.monotonic()
            camera.location = center + Vector(direction).normalized() * (4 * radius)
            right, up, outward = aim(camera, center)
            for lamp, offset in lamps:
                lamp.location = center + radius * (right * offset[0] + up * offset[1] + outward * offset[2])
                aim(lamp, center)
            prior = baseline_views["pos-x"]
            assert list(camera.location) == prior["camera_location"] and list(camera.rotation_quaternion) == prior["camera_quaternion"]
            assert list(right) == prior["camera_right"] and list(up) == prior["camera_up"] and camera_data.ortho_scale == prior["ortho_scale"]
            scene.view_settings.exposure = exposure
            settings_at_start = settings()
            assert_exposure_only(settings_before, settings_at_start, exposure)
            assert material_readback() == material_before
            presentation = {"camera_location": list(camera.location), "camera_quaternion": list(camera.rotation_quaternion),
                            "camera_right": list(right), "camera_up": list(up), "ortho_scale": camera_data.ortho_scale,
                            "camera_type": camera_data.type, "clip": [camera_data.clip_start, camera_data.clip_end],
                            "actual_lights": [{"name": lamp.name, "location": list(lamp.location), "quaternion": list(lamp.rotation_quaternion),
                                              "energy": lamp.data.energy, "size": lamp.data.size, "color": list(lamp.data.color)} for lamp, _ in lamps]}
            if record["views"]:
                assert presentation == {key: record["views"][0][key] for key in presentation}, "Camera or lights differ from EV0"
            scene.render.filepath = str(pngs[name])
            bpy.ops.render.render(write_still=True)
            settings_at_end = settings()
            assert_exposure_only(settings_before, settings_at_end, exposure)
            assert settings_at_end == settings_at_start and material_readback() == material_before
            assert list(camera.location) == presentation["camera_location"] and list(camera.rotation_quaternion) == presentation["camera_quaternion"]
            assert camera_data.ortho_scale == presentation["ortho_scale"] and camera_data.type == presentation["camera_type"]
            assert [camera_data.clip_start, camera_data.clip_end] == presentation["clip"]
            assert [{"name": lamp.name, "location": list(lamp.location), "quaternion": list(lamp.rotation_quaternion),
                     "energy": lamp.data.energy, "size": lamp.data.size, "color": list(lamp.data.color)} for lamp, _ in lamps] == presentation["actual_lights"]
            view = {"name": name, "PNG": pngs[name].name, "axis_direction": list(direction),
                    **presentation, "exposure_EV": exposure,
                    "settings_before_render": settings_at_start, "settings_after_render": settings_at_end,
                    "settings_difference_from_EV0": {"exposure": exposure} if exposure else {},
                    "camera_lights_material_identical_to_EV0": True,
                    "material_attribute_sha256": {name: value["sha256"] for name, value in material_before["attributes"].items()},
                    "linear_RGBA_sha256": expected_rgba_sha,
                    **image_record(pngs[name]), "seconds": time.monotonic() - started,
                    "elapsed_seconds": time.monotonic() - began}
            record["views"].append(view)
            print("BRICK_PBR_RENDERED=" + name, flush=True)
            return view

        first = render(*VIEWS[0])
        assert geometry_payloads() == payload_before
        assert material_readback() == material_before
        assert digest(FIELD) == FIELD_SHA256
        save_json(output / "first-view.json", {**first, "classification": "FIRST_VIEW_COMPLETE", "device": "CPU",
                  "samples": SAMPLES, "threads": THREADS, "denoising_use_gpu": False, "vertex_count": COUNTS[0], "face_count": COUNTS[1],
                  "source_PLY_sha256": SOURCE_SHA256, "geometry_payloads": payload_before, "geometry_unchanged_after_first_view": True,
                  "material_field_sha256": FIELD_SHA256, "material_verified_after_first_view": True,
                  "material_attribute_sha256": {name: value["sha256"] for name, value in material_before["attributes"].items()},
                  "linear_RGBA_sha256": expected_rgba_sha, "alpha_linked_from_original": True, "elapsed_seconds": time.monotonic() - began})
        await_host("GO_REMAINING")
        for name, direction, exposure in VIEWS[1:]:
            render(name, direction, exposure)

        scene.view_settings.exposure = 0
        payload_after = geometry_payloads()
        material_after = material_readback()
        assert payload_after == payload_before and material_after == material_before and settings() == settings_before
        assert all(digest(path) == expected for path, expected in pinned.items())
        assert all([path.stat().st_size, path.stat().st_mtime_ns] == source_stats[str(path)] for path in source_paths)
        record.update({"classification": "PASS", "geometry_unchanged": True, "material_verified": True,
                       "source_PLY_sha256": SOURCE_SHA256, "material_field_sha256": FIELD_SHA256,
                       "source_stats_before_and_after": source_stats, "geometry_payloads_after": payload_after,
                       "material_after": material_after, "settings_after": settings(),
                       "exposure_restored_to_EV0": True,
                       "outputs": {name: image_record(path) for name, path in pngs.items()}})
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
