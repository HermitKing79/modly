"""Read back the saved scene inside bounded CPU Blender; do not render or save."""
import hashlib
import json
from pathlib import Path

import bpy
import numpy as np

E = Path(__file__).resolve().parent
ROOT = E.parents[2]
OUT = ROOT.parent / "artifacts/amd-turret-learned-colour-fa3a98c"
BLEND, PLY = OUT / "learned-colour.blend", OUT / "turret-learned-colour.ply"
receipt = json.loads((E / "conversion.json").read_text(encoding="utf-8"))
assert receipt["classification"] == "PASS" and Path(bpy.data.filepath).resolve() == BLEND.resolve()
blend_stat = BLEND.stat()
scene = bpy.context.scene
objects = [obj for obj in scene.objects if obj.type == "MESH"]
assert len(objects) == 1 and len([obj for obj in bpy.data.objects if obj.type == "MESH"]) == 1
obj = objects[0]
mesh = obj.data
assert not obj.modifiers and max(abs(obj.matrix_world[i][j] - (i == j)) for i in range(4) for j in range(4)) < 1e-7
n, f = len(mesh.vertices), len(mesh.polygons)
assert (n, f, len(mesh.loops)) == (1842968, 4012360, 4012360 * 3)
co, loops = np.empty(n * 3, "<f4"), np.empty(f * 3, "<i4")
starts, totals = np.empty(f, "<i4"), np.empty(f, "<i4")
mesh.vertices.foreach_get("co", co)
mesh.loops.foreach_get("vertex_index", loops)
mesh.polygons.foreach_get("loop_start", starts)
mesh.polygons.foreach_get("loop_total", totals)
assert np.all(totals == 3) and np.array_equal(starts, np.arange(f, dtype=np.int32) * 3)
hashes = {"vertex_payload_sha256": hashlib.sha256(co).hexdigest(), "face_payload_sha256": hashlib.sha256(loops).hexdigest()}
assert all(value == receipt[key] for key, value in hashes.items()), "Saved geometry bytes differ"
with PLY.open("rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == receipt["coloured_ply"]["sha256"]
with PLY.open("rb") as stream:
    for _ in range(64):
        line = stream.readline(4097)
        assert line and len(line) <= 4096
        if line.strip() == b"end_header":
            offset = stream.tell()
            break
    else:
        raise AssertionError("Missing bounded PLY header")
records = np.memmap(PLY, mode="r", offset=offset, dtype=np.dtype([("xyz", "<f4", (3,)), ("rgb", "u1", (3,))]), shape=(n,))
rgb = np.ascontiguousarray(records["rgb"])
assert hashlib.sha256(rgb).hexdigest() == receipt["RGB_payload_sha256"]
assert len(mesh.color_attributes) == 1
colour = mesh.color_attributes[0]
assert colour.domain in {"POINT", "CORNER"} and colour.data_type in {"FLOAT_COLOR", "BYTE_COLOR"}
values = np.empty(len(colour.data) * 4, np.float32)
colour.data.foreach_get("color_srgb", values)
values = values.reshape(-1, 4)
assert np.isfinite(values).all() and np.all(np.abs(values[:, 3] - 1) < 1e-6)
quantized = np.rint(np.clip(values[:, :3], 0, 1) * np.float32(255)).astype(np.uint8)
if colour.domain == "POINT":
    assert np.array_equal(quantized, rgb)
else:
    assert len(quantized) == len(loops)
    for start in range(0, len(loops), 262144):
        end = min(start + 262144, len(loops))
        assert np.array_equal(quantized[start:end], rgb[loops[start:end]])
assert len(mesh.materials) == 1 and mesh.materials[0].use_nodes
material = mesh.materials[0]
output = next(node for node in material.node_tree.nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output)
assert len(output.inputs["Surface"].links) == 1
shader = output.inputs["Surface"].links[0].from_node
assert shader.bl_idname == "ShaderNodeBsdfPrincipled" and len(shader.inputs["Base Color"].links) == 1
link = shader.inputs["Base Color"].links[0]
assert link.from_node.bl_idname == "ShaderNodeVertexColor" and link.from_node.layer_name == colour.name
assert link.from_socket == link.from_node.outputs["Color"]
assert shader.inputs["Metallic"].default_value == 0 and abs(shader.inputs["Roughness"].default_value - .8) < 1e-6
assert shader.inputs["Emission Strength"].default_value == 0
cameras = sorted(obj.name for obj in scene.objects if obj.type == "CAMERA")
assert cameras == ["Preview front", "Preview rear", "Preview side"] and scene.camera.name == "Preview front"
assert scene.render.engine == "CYCLES" and scene.cycles.device == "CPU" and scene.cycles.samples == 64
assert scene.render.threads_mode == "FIXED" and scene.render.threads == 4 and scene.cycles.use_denoising
assert scene.cycles.denoiser == "OPENIMAGEDENOISE" and scene.cycles.denoising_use_gpu is False
assert scene.view_settings.view_transform == "AgX" and scene.view_settings.look == "None"
assert scene.view_settings.exposure == 0 and scene.view_settings.gamma == 1
assert (BLEND.stat().st_size, BLEND.stat().st_mtime_ns) == (blend_stat.st_size, blend_stat.st_mtime_ns)
record = {"classification": "PASS", "no_render": True, "saved_blend": str(BLEND), "vertex_count": n,
          "face_count": f, **hashes, "RGB_payload_sha256": receipt["RGB_payload_sha256"],
          "attribute": {"name": colour.name, "domain": colour.domain, "data_type": colour.data_type},
          "all_saved_sRGB_bytes_match": True, "material_link_verified": True, "cameras": cameras,
          "active_camera": scene.camera.name, "device": "CPU", "samples": 64,
          "denoiser": "OPENIMAGEDENOISE", "denoising_use_gpu": False, "view_transform": "AgX",
          "saved_file_unchanged": True, "Blender_version": bpy.app.version_string}
with (E / "scene-readback.json").open("x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
print("CPU_LEARNED_COLOUR_SCENE_READBACK=PASS", flush=True)
