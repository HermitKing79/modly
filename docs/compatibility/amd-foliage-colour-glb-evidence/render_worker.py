"""One exact-GO-gated Blender 5.2.1 import and four bounded CPU comparison views.

The host owns the 6 GiB/180 second Windows Job. This worker never exports,
bakes, loads learned fields, repairs imported attributes or retries a mismatch.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import stat
import struct
import threading
import time

E = Path(__file__).resolve().parent
ROOT = E.parents[2]
P = ROOT.parent / "artifacts/amd-foliage-colour-glb-cfe852b"
OUT = P / "render-qualified"
EXPORT = P / "export"
GLB = EXPORT / "foliage-learned-colour.glb"
SELECTION = ROOT.parent / "artifacts/amd-bounded-foliage-64d05aa/run"
DECISION = ROOT.parent / "artifacts/amd-foliage-export-7f1b167/run"
LEGACY = E.parent / "amd-brick-uv-bake-evidence/render_patch.py"
LEGACY_SHA = "96ad2138bb791b12557c9557348f8ad5a351df026774beea158c8f56924127e3"
INPUT_PINS = {
    "vertices.npy": (SELECTION, 38252, "93482834078c23d6eab392fbaa200e6fc472073d06b4939b98baaf2f02ef794a"),
    "faces.npy": (SELECTION, 54500, "0c9b3148b7183ea4e3544a465e8599dba92a9efcbcec3d33c438d2806cbda448"),
    "source-vertex-ids.npy": (SELECTION, 25544, "2889c0031c0e9aa7d07e2697b050fed17f1a68dd93aa4f9248c1e1679e13e2b2"),
    "source-face-ids.npy": (SELECTION, 18252, "dc0677bf75cb4206c68726ea5165f11e935578b54497ebf2fe06184d7206b9d1"),
    "linear-COLOR0.npy": (DECISION, 38252, "297d32a9cf0028004e15371a2890bbcf107771c05465ed88346c7e6fbe5d75af"),
}
METALLIC, ROUGHNESS = .00011082693761409814, .999995986777544
RGB_GATES = dict(rmse=.02, p95_abs=.05, max_abs=.15)
SILHOUETTE_IOU = .999


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_digest(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def plain(path):
    path = Path(path).absolute()
    for ancestor in (path, *path.parents):
        info = ancestor.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400,
                "Plain non-reparse path required: " + str(ancestor))
    require(path.is_dir() or (path.is_file() and path.stat().st_nlink == 1), "Plain single-link input/output required")
    return path


def identities(paths):
    return {str(path): dict(sha256=digest(plain(path)), bytes=path.stat().st_size,
                            mtime_ns=path.stat().st_mtime_ns) for path in paths}


def await_host():
    received = queue.Queue(maxsize=1)
    def read_line():
        try:
            received.put(sys.stdin.buffer.readline(64))
        except BaseException as error:
            received.put(error)
    print("FOLIAGE_COLOUR_GLB_READY_GO", flush=True)
    threading.Thread(target=read_line, daemon=True).start()
    try:
        token = received.get(timeout=30)
    except queue.Empty as error:
        raise RuntimeError("Exact binary GO timed out after 30 seconds") from error
    require(token == b"GO\n", "Exact binary GO admission rejected")


def initial_output(path, live_tempdir):
    require(path.is_absolute() and path == OUT, "Exact fixed render output required")
    plain(path)
    entries = list(path.iterdir())
    require(len(entries) <= 3 and all(x.name in ("scratch", "stdout.txt", "stderr.txt") for x in entries),
            "Only fresh host startup entries; no overwrite or retry")
    live_session = None
    for entry in entries:
        plain(entry)
        require(entry.is_dir() if entry.name == "scratch" else entry.is_file(), "Plain host startup entry required")
        if entry.name != "scratch":
            continue
        children = list(entry.iterdir())
        require(len(children) <= 1, "At most one exact empty live Blender scratch child")
        if children:
            require(isinstance(live_tempdir, str) and live_tempdir, "Live bpy.app.tempdir identity required")
            session = Path(live_tempdir)
            require(session.is_absolute() and session.parent == entry and children == [session], "Exact immediate Blender temp child required")
            plain(session)
            require(session.is_dir() and not next(session.iterdir(), None), "Empty live Blender startup session required")
            live_session = str(session)
    return dict(initial_entries=sorted(x.name for x in entries), live_session=live_session,
                plain_ancestors_verified=True, foreign_entries_allowed=False)


def load_legacy():
    require(digest(plain(LEGACY)) == LEGACY_SHA, "Pinned CPU renderer source differs")
    spec = importlib.util.spec_from_file_location("foliage_colour_pinned_renderer", LEGACY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(digest(LEGACY) == LEGACY_SHA, "Pinned CPU renderer changed during load")
    return module


def match_oriented_corners(vertices, faces, reference_vertices, reference_faces):
    """Return imported loop indices in original face/corner order, using exact F32 bits."""
    import numpy as np
    def rows(v, f):
        require(v.dtype == np.dtype("<f4") and v.ndim == 2 and v.shape[1] == 3 and np.isfinite(v).all(), "Finite F32 positions required")
        require(f.ndim == 2 and f.shape[1] == 3 and f.dtype.kind in "iu" and len(f) > 0
                and f.min() >= 0 and f.max() < len(v), "Valid oriented triangle indices required")
        result = {}
        for face_index, face in enumerate(f):
            corners = np.ascontiguousarray(v[face])
            rotations = [np.concatenate((corners[i:], corners[:i])).tobytes() for i in range(3)]
            first = min(range(3), key=lambda i: rotations[i])
            key = rotations[first]
            require(key not in result, "Duplicate oriented positions make source corner lineage ambiguous")
            result[key] = [3 * face_index + (first + i) % 3 for i in range(3)]
        return result
    actual, expected = rows(vertices, faces), rows(reference_vertices, reference_faces)
    require(actual.keys() == expected.keys(), "Exact oriented triangle position bits differ")
    mapping = np.empty(reference_faces.size, dtype="<i4")
    for key in expected:
        mapping[expected[key]] = actual[key]
    return mapping.reshape(reference_faces.shape)


def encode_srgb(linear):
    import numpy as np
    values = np.asarray(linear, dtype=np.float64)
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "Bounded finite linear RGB required")
    return np.where(values <= .0031308, 12.92 * values, 1.055 * np.power(values, 1 / 2.4) - .055)


def rgb_metrics(reference, actual):
    import numpy as np
    require(reference.shape == actual.shape and reference.size > 0, "Matching nonempty RGB arrays required")
    delta = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    require(np.isfinite(delta).all(), "Finite RGB differences required")
    result = dict(rmse=float(np.sqrt(np.mean(delta * delta))), p95_abs=float(np.percentile(delta, 95)), max_abs=float(delta.max()))
    result["pass"] = all(result[key] <= limit for key, limit in RGB_GATES.items())
    return result


def compare_images(reference, actual):
    import numpy as np
    require(reference.shape == actual.shape and reference.ndim == 3 and reference.shape[2] == 4
            and reference.dtype == actual.dtype == np.uint8, "Matching RGBA8 images required")
    ref_mask, actual_mask = reference[:, :, 3] >= 128, actual[:, :, 3] >= 128
    union = ref_mask | actual_mask
    require(union.any() and ref_mask.any() and actual_mask.any(), "Nonempty rendered silhouettes required")
    overlap = int(np.count_nonzero(ref_mask & actual_mask))
    all_rgb = rgb_metrics(reference[:, :, :3] / 255, actual[:, :, :3] / 255)
    object_rgb = rgb_metrics(reference[:, :, :3][union] / 255, actual[:, :, :3][union] / 255)
    iou = overlap / int(np.count_nonzero(union))
    return dict(all_pixels_RGB=all_rgb, silhouette_union_RGB=object_rgb, silhouette_IoU=iou,
                silhouette_pixels=[int(ref_mask.sum()), int(actual_mask.sum())],
                silhouette_exact=bool(np.array_equal(ref_mask, actual_mask)),
                RGBA_pixels_exact=bool(np.array_equal(reference, actual)), alpha_threshold=128,
                RGB_gates=RGB_GATES, silhouette_IoU_gate=SILHOUETTE_IOU,
                **{"pass": all_rgb["pass"] and object_rgb["pass"] and iou >= SILHOUETTE_IOU})


def glb_material_contract():
    payload = GLB.read_bytes()
    require(len(payload) < 1024 * 1024 and struct.unpack_from("<4sII", payload) == (b"glTF", 2, len(payload)), "Bounded complete GLB2 required")
    size, kind = struct.unpack_from("<II", payload, 12)
    require(kind == 0x4e4f534a and 20 + size <= len(payload), "First complete GLB JSON chunk required")
    document = json.loads(payload[20:20 + size])
    require(not any(document.get(key) for key in ("textures", "images", "samplers", "extensionsRequired", "extensionsUsed", "animations", "skins")), "Static core glTF without textures or extensions required")
    require(len(document["meshes"]) == len(document["materials"]) == 1, "One mesh/material required")
    primitive, = document["meshes"][0]["primitives"]
    require(set(primitive["attributes"]) == {"POSITION", "COLOR_0"} and primitive.get("mode", 4) == 4, "Only original positions and COLOR_0 triangle attributes required")
    material = document["materials"][0]
    pbr = material["pbrMetallicRoughness"]
    require(material.get("alphaMode", "OPAQUE") == "OPAQUE" and material.get("doubleSided") is True, "Opaque double-sided GLB required")
    require(pbr.get("baseColorFactor", [1, 1, 1, 1]) == [1, 1, 1, 1]
            and pbr.get("metallicFactor") == METALLIC and pbr.get("roughnessFactor") == ROUGHNESS,
            "Authenticated constant metallic/roughness and unit base factor required")
    return dict(sha256=digest(GLB), bytes=len(payload), material=material,
                attribute_names=sorted(primitive["attributes"]), numeric_accessor_check="separate independent verifier")


def load_reference():
    """Reload originals, never any imported Blender data or reconstructed colour."""
    import numpy as np
    values = {name: np.load(directory / name, allow_pickle=False) for name, (directory, _, _) in INPUT_PINS.items()}
    validate_reference(values)
    return values


def validate_reference(values):
    import numpy as np
    v, f, rgb = (values[name] for name in ("vertices.npy", "faces.npy", "linear-COLOR0.npy"))
    require(v.dtype == rgb.dtype == np.float32 and v.shape == rgb.shape == (3177, 3)
            and np.isfinite(v).all() and np.isfinite(rgb).all() and ((rgb >= 0) & (rgb <= 1)).all(), "Authentic finite 3177 F32 position/linear colour rows required")
    require(f.dtype == np.int32 and f.shape == (4531, 3) and f.min() >= 0 and f.max() < len(v), "Authentic 4531 original I32 triangles required")
    for name, shape, dtype in (("source-vertex-ids.npy", (3177,), np.int64), ("source-face-ids.npy", (4531,), np.int32)):
        require(values[name].dtype == dtype and values[name].shape == shape and len(np.unique(values[name])) == shape[0], "Exact unique original source IDs required")


def make_reference(bpy, values):
    import numpy as np
    mesh = bpy.data.meshes.new("Independent original foliage F32 reference")
    mesh.from_pydata(values["vertices.npy"].tolist(), [], values["faces.npy"].tolist())
    mesh.update()
    mesh.polygons.foreach_set("use_smooth", np.zeros(4531, dtype=np.bool_))
    color = mesh.color_attributes.new(name="Authenticated linear COLOR_0", type="FLOAT_COLOR", domain="POINT")
    rgba = np.ones((3177, 4), dtype="<f4")
    rgba[:, :3] = values["linear-COLOR0.npy"]
    color.data.foreach_set("color", rgba.ravel())
    mesh.color_attributes.render_color_index = 0
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    material = bpy.data.materials.new("Independent original linear FLOAT_COLOR constant PBR")
    material.use_nodes, material.use_backface_culling = True, False
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    shader, output, vertex_color = (nodes.new(kind) for kind in ("ShaderNodeBsdfPrincipled", "ShaderNodeOutputMaterial", "ShaderNodeVertexColor"))
    vertex_color.layer_name = color.name
    shader.inputs["Metallic"].default_value = METALLIC
    shader.inputs["Roughness"].default_value = ROUGHNESS
    shader.inputs["Alpha"].default_value = 1
    shader.inputs["Emission Strength"].default_value = 0
    links.new(vertex_color.outputs["Color"], shader.inputs["Base Color"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    mesh.materials.append(material)
    return obj


def socket_value(socket):
    value = getattr(socket, "default_value", None)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return list(value)


def emission_is_zero(strength, linear_rgb):
    import numpy as np
    colour = np.asarray(linear_rgb, dtype=np.float64)
    require(colour.shape == (3,) and np.isfinite(colour).all() and np.isfinite(strength), "Finite effective emission required")
    return bool(strength == 0 or (colour == 0).all())


def material_readback(bpy, obj):
    import numpy as np
    require(len(obj.data.materials) == 1 and len(obj.data.color_attributes) == 1, "One actual colour layer/material required")
    material = obj.data.materials[0]
    require(material.use_nodes and not material.use_backface_culling, "Actual double-sided node material required")
    nodes, links = material.node_tree.nodes, material.node_tree.links
    require(sorted(node.bl_idname for node in nodes) == sorted(("ShaderNodeBsdfPrincipled", "ShaderNodeOutputMaterial", "ShaderNodeVertexColor")), "Only vertex-colour/Principled/output nodes allowed")
    shader = next(node for node in nodes if node.bl_idname == "ShaderNodeBsdfPrincipled")
    output = next(node for node in nodes if node.bl_idname == "ShaderNodeOutputMaterial")
    color = next(node for node in nodes if node.bl_idname == "ShaderNodeVertexColor")
    attribute = obj.data.color_attributes[0]
    require(color.layer_name == attribute.name or (color.layer_name == "" and obj.data.color_attributes.render_color_index == 0), "Actual shader must resolve the actual rendered colour attribute")
    edges = {(edge.from_node, edge.from_socket.name, edge.to_node, edge.to_socket.name) for edge in links}
    require(edges == {(color, "Color", shader, "Base Color"), (shader, "BSDF", output, "Surface")}, "Actual shader wiring differs")
    for name, expected in (("Metallic", METALLIC), ("Roughness", ROUGHNESS), ("Alpha", 1)):
        socket = shader.inputs[name]
        require(not socket.is_linked and socket.default_value == float(np.float32(expected)), "Actual Principled constant differs: " + name)
    require(emission_is_zero(shader.inputs["Emission Strength"].default_value,
                             shader.inputs["Emission Color"].default_value[:3]), "Actual effective emission must be zero")
    require(not shader.inputs["Normal"].is_linked and not output.inputs["Displacement"].is_linked, "No normal/displacement input allowed")
    require(all(image.type in ("RENDER_RESULT", "COMPOSITING") for image in bpy.data.images), "No texture images allowed")
    return dict(material=material.name, double_sided=not material.use_backface_culling,
                surface_render_method=material.surface_render_method, opaque_shader_alpha=1,
                colour_node_layer_name=color.layer_name, rendered_colour_index=obj.data.color_attributes.render_color_index,
                attribute_name=attribute.name, no_texture_images=True, effective_emission_zero=True,
                nodes=[dict(name=node.name, type=node.bl_idname) for node in nodes],
                links=sorted([dict(source=edge.from_node.name, source_socket=edge.from_socket.name,
                                   destination=edge.to_node.name, destination_socket=edge.to_socket.name) for edge in links], key=lambda row: (row["source"], row["destination"])),
                Principled_inputs={socket.name: dict(linked=socket.is_linked, default=socket_value(socket)) for socket in shader.inputs})


def object_readback(bpy, obj, values):
    import numpy as np
    mesh = obj.data
    require(not obj.modifiers and not obj.parent and len(mesh.polygons) == 4531 and len(mesh.loops) == 13593
            and 3177 <= len(mesh.vertices) <= 13593 and len(mesh.uv_layers) == 0 and not mesh.has_custom_normals,
            "Unmodified flat original triangle object without UV/custom normals required")
    require(all(obj.matrix_world[i][j] == (i == j) for i in range(4) for j in range(4)), "Identity original Z-up object transform required")
    v, f = np.empty((len(mesh.vertices), 3), dtype="<f4"), np.empty((4531, 3), dtype="<i4")
    mesh.vertices.foreach_get("co", v.ravel())
    mesh.loops.foreach_get("vertex_index", f.ravel())
    totals, smooth = np.empty(4531, dtype="<i4"), np.empty(4531, dtype=np.bool_)
    mesh.polygons.foreach_get("loop_total", totals)
    mesh.polygons.foreach_get("use_smooth", smooth)
    require((totals == 3).all() and not smooth.any(), "Actual flat triangle shading required")
    mapping = match_oriented_corners(v, f, values["vertices.npy"], values["faces.npy"])
    attribute = mesh.color_attributes[0]
    require(attribute.domain in ("POINT", "CORNER") and attribute.data_type in ("FLOAT_COLOR", "BYTE_COLOR"), "Supported actual linear-readable colour layer required")
    rgba = np.empty((len(attribute.data), 4), dtype="<f4")
    attribute.data.foreach_get("color", rgba.ravel())
    require(np.isfinite(rgba).all() and ((rgba >= 0) & (rgba <= 1)).all() and (rgba[:, 3] == 1).all(), "Actual finite unit-alpha colour readback required")
    loop_rgba = rgba[f.reshape(-1)] if attribute.domain == "POINT" else rgba
    ordered = np.ascontiguousarray(loop_rgba[mapping][:, :, :3])
    reference = values["linear-COLOR0.npy"][values["faces.npy"]]
    metrics = rgb_metrics(encode_srgb(reference), encode_srgb(ordered))
    require(metrics["pass"], "Actual imported/reference encoded corner RGB exceeds predeclared gates")
    record = dict(vertices=len(v), triangles=4531, oriented_triangle_position_bits_exact=True,
                  cyclic_rotation_allowed=True, reversed_winding_allowed=False, signed_zero_normalized=False,
                  ordered_original_corner_position_sha256=array_digest(v[f.reshape(-1)][mapping]),
                  attribute_type=attribute.data_type, attribute_domain=attribute.domain,
                  actual_linear_colour_sha256=array_digest(ordered), actual_alpha_min=1, actual_alpha_max=1,
                  actual_linear_RGB_min=ordered.min(axis=(0, 1)).tolist(), actual_linear_RGB_max=ordered.max(axis=(0, 1)).tolist(),
                  linear_RGB_exact=bool(np.array_equal(reference, ordered)),
                  linear_RGB_max_abs=float(np.abs(ordered.astype(float) - reference).max()),
                  encoded_RGB_import_residual=metrics, imported_attribute_repaired=False,
                  material=material_readback(bpy, obj))
    return record, ordered


def settings(state, scene):
    result = state()
    result.update(film_transparent=scene.render.film_transparent, file_format=scene.render.image_settings.file_format,
                  colour_mode=scene.render.image_settings.color_mode, colour_depth=scene.render.image_settings.color_depth,
                  compositing=scene.render.use_compositing, sequencer=scene.render.use_sequencer,
                  persistent_data=scene.render.use_persistent_data,
                  view_layer_denoising=[layer.cycles.use_denoising for layer in scene.view_layers])
    require(result["engine"] == "CYCLES" and result["device"] == "CPU" and result["samples"] == 32
            and result["resolution"] == [640, 640, 100] and result["denoising_use_gpu"] is False,
            "Pinned bounded Cycles CPU settings required")
    return result


def main(output):
    began = time.monotonic()
    await_host()  # Before bpy access, source/model reads, module loads or output writes.
    legacy = load_legacy()
    live_bpy = sys.modules.get("bpy")
    startup = initial_output(output, getattr(getattr(live_bpy, "app", None), "tempdir", None))
    scratch = legacy.guard_python_writes(OUT)
    record = dict(classification="RUNNING", CPU_only=True, fresh_GLTF_imports=0, retry=False,
                  export=False, bake=False, neural_sampling=False, imported_attribute_repair=False,
                  startup_output_admission=startup, views=[], comparisons=[],
                  RGB_gates=RGB_GATES, silhouette_IoU_gate=SILHOUETTE_IOU)
    try:
        import bpy
        import numpy as np
        require(bpy.app.background and tuple(bpy.app.version) == (5, 2, 1), "Native Blender 5.2.1 background worker required")
        paths = [LEGACY, Path(__file__).resolve(), GLB] + list(legacy.BASELINE_HASHES)
        for name, (directory, size, sha) in INPUT_PINS.items():
            for path in (directory / name, EXPORT / name):
                require(plain(path).stat().st_size == size and digest(path) == sha, "Authenticated original/export array identity differs: " + str(path))
                paths.append(path)
        before = identities(paths)
        for path, sha in legacy.BASELINE_HASHES.items():
            require(before[str(path)]["sha256"] == sha, "Pinned prior CPU studio input differs")
        record.update(Blender_version=bpy.app.version_string, Blender_binary=bpy.app.binary_path,
                      source_files_before=before, GLB=glb_material_contract())
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.context.preferences.filepaths.temporary_directory = str(scratch)
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.cycles.device = "CPU"
        import_settings = dict(filepath=str(GLB), merge_vertices=False, import_shading="FLAT")
        record["import_settings"] = import_settings
        require(bpy.ops.import_scene.gltf(**import_settings) == {"FINISHED"}, "One fresh GLB import failed")
        record["fresh_GLTF_imports"] += 1
        objects = list(bpy.context.scene.objects)
        require(len(objects) == 1 and objects[0].type == "MESH", "Exactly one freshly imported mesh object required")
        imported = objects[0]
        values = load_reference()
        imported_before, actual_rgb = object_readback(bpy, imported, values)
        record["imported_before"] = imported_before
        for name, array in (("imported-corner-linear-COLOR0.npy", actual_rgb),
                            ("imported-corner-source-vertex-ids.npy", values["source-vertex-ids.npy"][values["faces.npy"]])):
            path = OUT / name
            with path.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
            record[name] = dict(sha256=digest(path), bytes=path.stat().st_size, shape=list(array.shape), dtype=str(array.dtype),
                               order="authenticated original face order and original oriented corner order")
        reference = make_reference(bpy, values)
        reference_before, _ = object_readback(bpy, reference, values)
        require(reference_before["linear_RGB_exact"] and reference_before["attribute_type"] == "FLOAT_COLOR"
                and reference_before["attribute_domain"] == "POINT", "Independent reference must preserve exact original F32 colours")
        record["reference_before"] = reference_before
        baseline = json.loads(legacy.BASELINE_METADATA.read_text(encoding="utf-8"))
        require(baseline["classification"] == "PASS", "Passing pinned prior CPU studio baseline required")
        scene, state, framing = legacy.setup_scene(bpy, values["vertices.npy"], baseline, scratch)
        # World illumination stays pinned; transparent film exposes rendered silhouettes.
        scene.render.film_transparent = True
        scene.render.image_settings.color_mode = "RGBA"
        base_scale = scene.camera.data.ortho_scale
        record.update(framing=framing, pinned_setup_scene_reused=True,
                      declared_output_overrides=dict(film_transparent=True, colour_mode="RGBA", purpose="silhouette measurement in the same four renders"))
        legacy.save_json(OUT / "import-checkpoint.json", {**record, "classification": "IMPORT_VERIFIED_BEFORE_RENDER"})
        for distance, multiplier in (("near", 1), ("far", 4)):
            scene.camera.data.ortho_scale = base_scale * multiplier
            expected = settings(state, scene)
            pixels = {}
            for kind, obj in (("reference", reference), ("imported", imported)):
                require(time.monotonic() - began < 155, "Insufficient time remains for bounded CPU render; no retry")
                imported.hide_render, reference.hide_render = obj is not imported, obj is not reference
                name = kind + "-" + distance
                path = OUT / (name + ".png")
                require(not path.exists(), "Render output overwrite forbidden")
                scene.render.filepath = str(path)
                started = time.monotonic()
                require(bpy.ops.render.render(write_still=True) == {"FINISHED"}, "Bounded CPU render failed")
                require(settings(state, scene) == expected, "CPU camera/lighting/render settings changed during render")
                image = legacy.png_pixels(path.read_bytes())
                require(image.shape == (640, 640, 4), "640-square RGBA8 PNG required")
                pixels[kind] = image
                require(object_readback(bpy, imported, values)[0] == imported_before
                        and object_readback(bpy, reference, values)[0] == reference_before,
                        "Actual imported/reference geometry, shader or colours changed")
                view = dict(name=name, file=str(path), sha256=digest(path), bytes=path.stat().st_size,
                            seconds=time.monotonic() - started, settings=expected,
                            projected_linear_size_relative_to_near=1 / multiplier,
                            RGB_mean=image[:, :, :3].mean(axis=(0, 1)).tolist(),
                            rendered_object=obj.name, hidden_object=reference.name if obj is imported else imported.name,
                            geometry_material_colours_unchanged=True)
                record["views"].append(view)
                print("FOLIAGE_COLOUR_GLB_RENDERED=" + name, flush=True)
            comparison = dict(distance=distance, **compare_images(pixels["reference"], pixels["imported"]))
            record["comparisons"].append(comparison)
            legacy.save_json(OUT / (distance + "-comparison.json"), comparison)
            require(comparison["pass"], "Rendered imported/reference RGB or silhouette gate failed; stopping without retry")
        require(len(record["views"]) == 4 and record["fresh_GLTF_imports"] == 1, "Exactly one fresh import and four views required")
        near, far = record["views"][0]["settings"], record["views"][2]["settings"]
        near_copy, far_copy = json.loads(json.dumps(near)), json.loads(json.dumps(far))
        require(far_copy["camera"].pop("ortho_scale") == 4 * near_copy["camera"].pop("ortho_scale")
                and near_copy == far_copy, "Only exact fourfold far orthographic scale may change")
        after = identities(paths)
        require(before == after, "Readonly sources or GLB changed")
        record.update(classification="PASS", source_files_after=after, sources_unchanged=True,
                      reference_constructed_from_original_authenticated_arrays=True,
                      import_colour_storage_scope="Blender actual attribute values; F32 GLB transport checked independently; no attribute replacement",
                      elapsed_seconds=time.monotonic() - began)
        legacy.save_json(OUT / "result.json", record)
        print("FOLIAGE_COLOUR_GLB_PASS", flush=True)
    except BaseException as error:
        record.update(classification="FAIL", error_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic() - began)
        legacy.save_json(OUT / "failure.json", record)
        raise


def script_args(argv=None):
    argv=sys.argv if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    return parser.parse_args(argv[argv.index("--") + 1:] if "--" in argv else argv[1:])


if __name__ == "__main__":main(script_args().output)
