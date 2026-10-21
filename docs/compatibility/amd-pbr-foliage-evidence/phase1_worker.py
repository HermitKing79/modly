"""One gated fresh Blender import and two CPU views of the qualified brick GLB.

This phase establishes actual Blender transport/render evidence. Its sampler
metadata and observed Blender filtering are separate facts; neither establishes
equivalence to the earlier CPU box-mip experiment or to a portable viewer.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import stat
import struct
import sys
import threading
import time

sys.dont_write_bytecode = True
E = Path(__file__).resolve().parent
ROOT = E.parents[3]
P = ROOT / "artifacts/amd-pbr-foliage-ecb15aa"
OUT = P / "brick"
CANDIDATE = ROOT / "artifacts/amd-brick-mip-safe-atlas-e8244cb/revision/run/candidate"
BAKE = CANDIDATE / "bake"
GLB = CANDIDATE / "patch.glb"
LEGACY = E.parent / "amd-brick-uv-bake-evidence/render_patch.py"
LEGACY_SHA = "96ad2138bb791b12557c9557348f8ad5a351df026774beea158c8f56924127e3"
GLB_SHA = "dc772cd55fc523c5427ab4771da27e62cd6905d5109f00b2fca47403704da857"
GLB_BYTES = 638700
SAMPLER = dict(magFilter=9729, minFilter=9987, wrapS=33071, wrapT=33071)
INPUT_PINS = {
    "vertices.npy": (28244, "144dad7717dcaf45bb5cccc1129152a5dc8265d25046c603cfaaf54425bf807c"),
    "faces.npy": (9500, "996560f4a812d4751d0d6ec494abfdd9aea122c5788ff839d4140d6d1831ee46"),
    "uv.npy": (37616, "61ae6a3c231aa2cc74f7995fda91da9ae5fe32ffe48227fd0633280e00839656"),
    "source-vertex-ids.npy": (9500, "3f87c67984ca4ca6c698d3495d56894b5787d2da98d2c7772dab07d7930ecad8"),
    "source-face-ids.npy": (3252, "06167754d3c27f8257439f6cb7dc3db4c434ce388d2cb2f17568da2b7d1149ea"),
    "base-color.png": (540248, "0048a6342df959735cb4687dc22c9a1c4f013459bb5ffb93d94d4ed0080652ad"),
    "MR.png": (17162, "307f5a0c17c974c4fadeac0a2ae16c00bcd0368a666c92313a943282cf05314f"),
    "bake.json": (7042, "370ffe817cb7aff3882048f0cf5512f464e8e8cb7df7551cecf323633feeb33c"),
}
RESULT = CANDIDATE.parent / "result.json"
RESULT_SHA = "6906247a3b6f511cf06249d4af3d2c6feee6e4b0f759adde569859cfefb1da53"
VIEW_SCALES = (("near", 1.0), ("far", 4.0))
FILTER_SCOPE = ("glTF requests LINEAR magnification, LINEAR_MIPMAP_LINEAR minification and CLAMP_TO_EDGE. "
                "Actual Blender image-node settings are recorded separately. Cycles filtering is not established "
                "as equivalent to the earlier F64 2x2-box mip experiment or a portable viewer's filtering.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plain(path):
    path = Path(path).absolute()
    for ancestor in (path, *path.parents):
        info = ancestor.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400,
                "Plain non-reparse input/output path required: " + str(ancestor))
    require(path.is_dir() or (path.is_file() and path.stat().st_nlink == 1), "Plain single-link file required")
    return path


def load_legacy():
    require(digest(plain(LEGACY)) == LEGACY_SHA, "Historical CPU renderer pin differs")
    spec = importlib.util.spec_from_file_location("pbr_foliage_phase1_pinned_renderer", LEGACY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(digest(LEGACY) == LEGACY_SHA, "Historical CPU renderer changed during load")
    return module


def source_paths(legacy):
    return ([BAKE / name for name in INPUT_PINS] + [GLB, RESULT, LEGACY]
            + list(legacy.BASELINE_HASHES) + [E / "phase1_worker.py", E / "phase1_host.py"])


def identities(paths):
    return {str(path): dict(sha256=digest(plain(path)), bytes=path.stat().st_size,
                            mtime_ns=path.stat().st_mtime_ns) for path in paths}


def verify_source_pins(legacy):
    for name, (size, sha) in INPUT_PINS.items():
        path = plain(BAKE / name)
        require(path.stat().st_size == size and digest(path) == sha, "Qualified source pin differs: " + name)
    require(plain(GLB).stat().st_size == GLB_BYTES and digest(GLB) == GLB_SHA, "Qualified GLB identity differs")
    require(digest(plain(RESULT)) == RESULT_SHA, "Qualified mip-safe outcome pin differs")
    for path, sha in legacy.BASELINE_HASHES.items():
        require(digest(plain(path)) == sha, "Prior reviewed CPU lighting/settings pin differs")
    bake = json.loads((BAKE / "bake.json").read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    require(bake["classification"] == result["classification"] == "PASS"
            and all(bake[key] is True for key in ("actual_genuine_spatial_bake", "source_samples_byte_exact", "original_geometry_unchanged"))
            and bake["triangle_count"] == 781 and bake["GLB_SHA256"] == GLB_SHA and bake["GLB_bytes"] == GLB_BYTES
            and result["GLB"]["sha256"] == GLB_SHA and result["GLB"]["bytes"] == GLB_BYTES,
            "Authentic passing candidate bake/outcome receipts required")
    for name, (size, sha) in INPUT_PINS.items():
        if name != "bake.json":
            require(bake["files"][name] == dict(bytes=size, sha256=sha), "Authentic bake file identity differs: " + name)
    return bake, result


def initial_output(path, live_tempdir):
    """Only the host's logs/scratch and exact empty live Blender temp child."""
    path = Path(path)
    require(path.is_absolute() and path == OUT, "Exact exclusive phase1 output required")
    plain(path)
    entries = list(path.iterdir())
    require(len(entries) <= 3 and all(x.name in ("scratch", "stdout.txt", "stderr.txt") for x in entries),
            "No prior outputs, foreign startup content or retry")
    live_session = None
    for entry in entries:
        plain(entry)
        require(entry.is_dir() if entry.name == "scratch" else entry.is_file(), "Plain host startup entries required")
        if entry.name != "scratch":
            continue
        children = list(entry.iterdir())
        require(len(children) <= 1, "At most one exact Blender live session child")
        if children:
            require(isinstance(live_tempdir, str) and live_tempdir, "Live bpy.app.tempdir identity required")
            session = Path(live_tempdir)
            require(session.is_absolute() and session.parent == entry and session == children[0], "Only the exact immediate live Blender temp session allowed")
            plain(session)
            require(session.is_dir() and not next(session.iterdir(), None), "Live Blender startup session must be empty")
            live_session = str(session)
    return dict(output=str(path), initial_entries=sorted(x.name for x in entries),
                live_session_directory=live_session, plain_ancestors_verified=True, foreign_entries_allowed=False)


def await_host():
    """Exact binary token; no bpy import, scene access or output writes before it."""
    received = queue.Queue(maxsize=1)
    def read_line():
        try:
            received.put(sys.stdin.buffer.readline(64))
        except BaseException as error:
            received.put(error)
    print("PBR_FOLIAGE_PHASE1_READY_GO", flush=True)
    threading.Thread(target=read_line, daemon=True).start()
    try:
        token = received.get(timeout=30)
    except queue.Empty as error:
        raise RuntimeError("Exact GO admission timed out after 30 seconds") from error
    require(token == b"GO\n", "GO admission rejects missing or malformed exact binary token")


def canonical_bits(vertices, faces):
    """Oriented position bits, permitting only face order and cyclic rotation."""
    import numpy as np
    rows = []
    for face in faces:
        corners = np.ascontiguousarray(vertices[face], dtype="<f4")
        rotations = [np.concatenate((corners[i:], corners[:i])).tobytes() for i in range(3)]
        rows.append(min(rotations))
    return b"".join(sorted(rows))


def exact_geometry_bits(obj, vertices, faces):
    import numpy as np
    actual_vertices = np.empty((len(obj.data.vertices), 3), dtype="<f4")
    actual_faces = np.empty((len(obj.data.polygons), 3), dtype="<i4")
    obj.data.vertices.foreach_get("co", actual_vertices.ravel())
    obj.data.loops.foreach_get("vertex_index", actual_faces.ravel())
    actual, expected = canonical_bits(actual_vertices, actual_faces), canonical_bits(vertices, faces)
    require(actual == expected, "Imported oriented triangle position bits differ")
    return dict(oriented_triangle_position_bits_exact=True, cyclic_rotation_allowed=True,
                reversed_winding_allowed=False, signed_zero_normalized=False,
                canonical_position_bits_sha256=hashlib.sha256(actual).hexdigest())


def container_metadata(legacy, reference, source_maps):
    """Byte/value identity and exact glTF sampler, separate from Blender settings."""
    import numpy as np
    record = legacy.glb_check(GLB, reference, source_maps)
    payload = GLB.read_bytes()
    length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20:20 + length])
    binary = payload[28 + length:]
    require(len(document["textures"]) == len(document["images"]) == 2
            and len(document["samplers"]) == 1 and document["samplers"][0] == SAMPLER,
            "Exact qualified glTF sampler metadata required")
    pbr = document["materials"][0]["pbrMetallicRoughness"]
    for name, key in (("base-color.png", "baseColorTexture"), ("MR.png", "metallicRoughnessTexture")):
        texture = document["textures"][pbr[key]["index"]]
        require(texture["sampler"] == 0, "Both textures must use the qualified sampler")
        image = document["images"][texture["source"]]
        view = document["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        encoded = binary[start:start + view["byteLength"]]
        require(encoded == (BAKE / name).read_bytes(), "Embedded PNG bytes differ from qualified bake: " + name)
        decoded = legacy.png_pixels(encoded)
        require(np.array_equal(decoded, source_maps[name]), "Full embedded PNG channel values differ: " + name)
        record["maps"][name].pop("CPU_filter_metric", None)  # Historical LOD0 prose does not describe this phase.
        record["maps"][name].update(full_encoded_PNG_bytes_exact=True, all_decoded_channels_exact=True,
                                    decoded_all_channels_sha256=legacy.array_digest(decoded))
    record.update(glTF_sampler_metadata=dict(SAMPLER), filtering_scope=FILTER_SCOPE,
                  MR_packing=dict(red="unused; no occlusion texture", green="roughness", blue="metallic"),
                  embedded_base_alpha_preserved=True, shader_alpha="opaque 1")
    return record


def imported_material(legacy, obj, source_maps):
    """Recheck actual node routing, packed full PNGs and image-node sampling."""
    import numpy as np
    result = legacy.imported_material_readback(obj, source_maps)
    material = obj.data.materials[0]
    require(material.use_nodes and not material.use_backface_culling, "Original two-sided node material required")
    textures = [node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeTexImage"]
    require(len(textures) == 2, "Exactly two imported image nodes required")
    observed = []
    for node in textures:
        require(node.extension == "EXTEND" and node.interpolation == "Linear" and node.projection == "FLAT",
                "Imported clamp/linear/flat image-node settings differ")
        packed = bytes(node.image.packed_file.data)
        names = [name for name in source_maps if packed == (BAKE / name).read_bytes()]
        require(len(names) == 1, "Actual imported packed PNG bytes must identify one authentic map")
        name = names[0]
        require(np.array_equal(legacy.png_pixels(packed), source_maps[name]), "Imported full PNG channel values differ")
        require(not node.inputs["Vector"].is_linked, "Expected implicit original active UV map without transforms")
        observed.append(dict(map=name, interpolation=node.interpolation, extension=node.extension,
                             projection=node.projection, colorspace=node.image.colorspace_settings.name,
                             vector_input="unlinked; active UV map", packed_PNG_bytes_exact=True,
                             all_decoded_channels_exact=True))
    outputs = [node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeOutputMaterial" and node.is_active_output]
    require(len(outputs) == 1 and len(outputs[0].inputs["Surface"].links) == 1
            and outputs[0].inputs["Surface"].links[0].from_node.bl_idname == "ShaderNodeBsdfPrincipled",
            "Active material output must directly use the verified Principled shader")
    result.update(actual_Blender_image_node_settings=sorted(observed, key=lambda row: row["map"]),
                  filtering_scope=FILTER_SCOPE, glTF_sampler_equivalence_claimed=False)
    return result


def settings_without_scale(settings):
    result = json.loads(json.dumps(settings))
    result["camera"].pop("ortho_scale")
    return result


def validate_two_views(views):
    require([view["name"] for view in views] == ["near", "far"], "Exactly near and far views required")
    near, far = (view["settings"] for view in views)
    require(settings_without_scale(near) == settings_without_scale(far), "Lighting, camera orientation and CPU settings must match")
    require(far["camera"]["ortho_scale"] == 4 * near["camera"]["ortho_scale"], "Far view requires exact fourfold orthographic scale")
    require(all(setting["device"] == "CPU" and setting["engine"] == "CYCLES"
                and setting["samples"] == 32 and setting["resolution"] == [640, 640, 100]
                and setting["denoising_use_gpu"] is False for setting in (near, far)), "Bounded CPU render settings required")


def main(output):
    began = time.monotonic()
    await_host()
    legacy = load_legacy()
    live_bpy = sys.modules.get("bpy")
    startup = initial_output(output, getattr(getattr(live_bpy, "app", None), "tempdir", None))
    scratch = legacy.guard_python_writes(OUT)
    record = dict(classification="RUNNING", stage="phase1", CPU_only=True, views=[],
                  startup_output_admission=startup, filtering_scope=FILTER_SCOPE, fresh_GLTF_imports=0,
                  no_export=True, no_mesh_rebuild=True, retry=False)
    try:
        import bpy
        import numpy as np
        require(bpy.app.background and tuple(bpy.app.version) == (5, 2, 1), "Approved native Blender 5.2.1 background worker required")
        bake, prior = verify_source_pins(legacy)
        paths = source_paths(legacy)
        before = identities(paths)
        vertices, faces, uv = [np.load(BAKE / name, allow_pickle=False) for name in ("vertices.npy", "faces.npy", "uv.npy")]
        require(vertices.dtype == np.float32 and vertices.shape == (2343, 3) and np.isfinite(vertices).all()
                and faces.dtype == np.int32 and faces.shape == (781, 3) and faces.min() >= 0 and faces.max() < len(vertices)
                and uv.dtype == np.float64 and uv.shape == (2343, 2) and np.isfinite(uv).all()
                and ((uv >= 0) & (uv <= 1)).all(), "Exact authentic small corner arrays required")
        source_maps = {name: legacy.png_pixels((BAKE / name).read_bytes()) for name in ("base-color.png", "MR.png")}
        require(source_maps["base-color.png"].shape == (1024, 1024, 4)
                and source_maps["MR.png"].shape == (1024, 1024, 3), "Authentic 1024-square map schemas required")
        reference = legacy.canonical_triangles(vertices, faces, uv)
        record.update(Blender_version=bpy.app.version_string, Blender_binary=bpy.app.binary_path,
                      source_files_before=before, source_bake_receipt_sha256=digest(BAKE / "bake.json"),
                      source_bake_receipt_classification=bake["classification"],
                      source_ancestry_identities_from_pinned_bake_receipt=bake["source_input_identities"],
                      source_ancestry_reloaded=False, source_vertices=2343, source_triangles=781,
                      source_UV_dtype=str(uv.dtype), GLTF_UV_dtype="float32",
                      source_UV_to_float32_maximum_error=float(np.max(np.abs(uv - uv.astype("<f4")))),
                      glb=container_metadata(legacy, reference, source_maps))
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.context.preferences.filepaths.temporary_directory = str(scratch)
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.cycles.device = "CPU"
        import_settings = dict(filepath=str(GLB), merge_vertices=False, import_shading="FLAT")
        require(bpy.ops.import_scene.gltf(**import_settings) == {"FINISHED"}, "Fresh qualified GLB import failed")
        record["fresh_GLTF_imports"] += 1
        meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        require(len(meshes) == len(bpy.context.scene.objects) == 1, "Exactly one imported patch and no source reconstruction")
        obj, = meshes
        geometry = {**legacy.mesh_readback(obj, reference), **exact_geometry_bits(obj, vertices, faces)}
        material = imported_material(legacy, obj, source_maps)
        record.update(import_settings=import_settings, imported_geometry_before=geometry, imported_material_before=material)
        baseline = json.loads(legacy.BASELINE_METADATA.read_text(encoding="utf-8"))
        require(baseline["classification"] == "PASS", "Passing prior CPU studio baseline required")
        scene, state, framing = legacy.setup_scene(bpy, vertices, baseline, scratch)
        base_scale, rig = scene.camera.data.ortho_scale, state()
        record.update(framing=framing, rig=rig)
        legacy.save_json(OUT / "import-checkpoint.json", {**record, "classification": "IMPORT_VERIFIED_BEFORE_RENDER"})
        for name, multiplier in VIEW_SCALES:
            require(time.monotonic() - began < 150, "Insufficient time remains for a bounded CPU render")
            scene.camera.data.ortho_scale = base_scale * multiplier
            expected = state()
            require(settings_without_scale(expected) == settings_without_scale(rig), "Only projected size/orthographic scale may change")
            path = OUT / (name + ".png")
            scene.render.filepath = str(path)
            started = time.monotonic()
            require(bpy.ops.render.render(write_still=True) == {"FINISHED"}, "CPU render failed")
            require(state() == expected, "CPU rig changed while rendering")
            pixels = legacy.png_pixels(path.read_bytes())
            require(pixels.shape == (640, 640, 3) and float(pixels.std(axis=(0, 1)).max()) > 1,
                    "Nonuniform 640-square RGB8 render required")
            require({**legacy.mesh_readback(obj, reference), **exact_geometry_bits(obj, vertices, faces)} == geometry,
                    "Imported oriented geometry or UVs changed during render")
            require(imported_material(legacy, obj, source_maps) == material, "Imported node material or packed maps changed during render")
            view = dict(name=name, file=str(path), sha256=digest(path), bytes=path.stat().st_size,
                        seconds=time.monotonic() - started, settings=expected,
                        RGB_mean=pixels.mean(axis=(0, 1)).tolist(), RGB_std=pixels.std(axis=(0, 1)).tolist(),
                        projected_linear_size_relative_to_near=1 / multiplier,
                        imported_geometry_and_material_unchanged=True)
            record["views"].append(view)
            legacy.save_json(OUT / (name + "-checkpoint.json"), dict(classification="RENDER_VERIFIED", **view))
            print("PBR_FOLIAGE_PHASE1_RENDERED=" + name, flush=True)
        validate_two_views(record["views"])
        after = identities(paths)
        require(before == after, "Source assets or renderer inputs changed")
        require(record["fresh_GLTF_imports"] == 1, "Exactly one fresh import required")
        record.update(classification="PASS", source_files_after=after, sources_unchanged=True,
                      same_imported_scene_for_both_views=True, elapsed_seconds=time.monotonic() - began)
        legacy.save_json(OUT / "result.json", record)
        print("PBR_FOLIAGE_PHASE1_PASS", flush=True)
    except BaseException as error:
        record.update(classification="FAIL", error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.monotonic() - began)
        legacy.save_json(OUT / "failure.json", record)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    main(args.output)
