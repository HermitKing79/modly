"""Bounded native Blender CPU proof for one original brick surface patch.

Two separately owned Blender processes are required. Each waits for exactly
b'GO\n' before importing bpy or touching its scene. Inputs are prepared by the
separate sampler; this script never imports the full mesh or material field.

blender --background --factory-startup --disable-autoexec --threads 4
  --python-exit-code 1 --python render_patch.py -- --stage export --output P/export
Use --stage compare --output P/compare in a fresh process after export passes.
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
import zlib

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
PRIVATE = ROOT / "artifacts/amd-brick-uv-bake-18c0c92/qualified"
BAKE = PRIVATE / "bake"
BASELINE_SCRIPT = ROOT / "proof/docs/compatibility/amd-brick-pbr-exposure-evidence/render_exposure.py"
BASELINE_METADATA = ROOT / "artifacts/amd-brick-pbr-exposure-b0b7be1/renders/render-metadata.json"
BASELINE_HASHES = {
    BASELINE_SCRIPT: "17af94c180944274c304cd8e034c95e21190fa21ff8434335b48a637fbde27cf",
    BASELINE_METADATA: "2e09ea2c4ba104f37133e1322759ce9de3fe5e92a5aeac798e379d8bf1685464",
}
INPUT_NAMES = ("vertices.npy", "faces.npy", "uv.npy", "direct.npy", "base-color.png", "MR.png", "bake.json")
RESOLUTION, SAMPLES, THREADS = 640, 32, 4
MAX_VERTICES, MAX_FACES = 100000, 200000
CHANNELS = ("red", "green", "blue", "metallic", "roughness", "alpha")
LINEAR_RGB = "Original_RGB_native_gamma22"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_digest(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def save_json(path, data):
    encoded = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def await_host():
    received = queue.Queue(maxsize=1)

    def read_line():
        try:
            received.put(sys.stdin.buffer.readline(64))
        except BaseException as exc:
            received.put(exc)

    print("BRICK_UV_READY_GO", flush=True)
    threading.Thread(target=read_line, daemon=True).start()
    try:
        token = received.get(timeout=30)
    except queue.Empty as exc:
        raise RuntimeError("GO host gate timed out after 30 seconds") from exc
    require(token == b"GO\n", "GO host gate rejected missing or malformed exact token")


def private_output(stage, path):
    path = Path(path)
    expected = PRIVATE / stage
    require(path.resolve() == expected.resolve(), "Exact private stage output directory required")
    require(path.is_dir() and not path.is_symlink(), "Host must create a fresh nonsymlink stage directory")
    entries = {entry.name: entry for entry in path.iterdir()}
    require(set(entries) <= {"scratch", "stdout.txt", "stderr.txt"}, "Only initial host scratch/log entries may exist; failed attempts are not retried")
    for name, entry in entries.items():
        require(not entry.is_symlink(), "Host entries must not be symlinks")
        require(entry.is_dir() if name == "scratch" else entry.is_file(), "Plain host scratch/log entries required")
        if name == "scratch":
            require(not any(entry.iterdir()), "Initial host scratch must be empty")
    for parent in (PRIVATE, path):
        require(not parent.is_symlink(), "Symlink output parents are forbidden")
    return path.resolve()


def guard_python_writes(output):
    """Python writes are restricted; the owned host also contains native writes."""
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

    def inside(value):
        if isinstance(value, int):
            return  # An existing descriptor was already admitted at open.
        target = Path(os.fsdecode(value)).resolve()
        require(target == output or output in target.parents, "Python write outside private stage: " + str(target))

    def audit(event, args):
        if event == "open":
            path, mode, flags = args
            if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & write_flags):
                inside(path)
        elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.utime"):
            inside(args[0])
        elif event in ("os.rename", "os.link", "os.symlink"):
            inside(args[0])
            inside(args[1])

    sys.addaudithook(audit)
    scratch = output / "blender-scratch"
    scratch.mkdir()
    for name in ("TMP", "TEMP", "TMPDIR"):
        os.environ[name] = str(scratch)
    import tempfile
    tempfile.tempdir = str(scratch)
    return scratch


def png_pixels(payload):
    """Independent CRC-checked decoder for the supplied/exported 8-bit PNG maps."""
    import numpy as np
    require(payload[:8] == b"\x89PNG\r\n\x1a\n", "PNG signature required")
    offset, compressed, header, ended = 8, bytearray(), None, False
    while offset < len(payload):
        require(offset + 12 <= len(payload), "Complete PNG chunk required")
        length = struct.unpack_from(">I", payload, offset)[0]
        kind = payload[offset + 4:offset + 8]
        data = payload[offset + 8:offset + 8 + length]
        require(len(data) == length and offset + length + 12 <= len(payload), "Truncated PNG chunk")
        require(zlib.crc32(kind + data) & 0xffffffff == struct.unpack_from(">I", payload, offset + 8 + length)[0], "PNG CRC mismatch")
        if kind == b"IHDR":
            require(header is None and length == 13, "One PNG IHDR required")
            header = struct.unpack(">IIBBBBB", data)
        elif kind == b"IDAT":
            compressed.extend(data)
        elif kind == b"IEND":
            ended = True
            require(offset + length + 12 == len(payload), "Trailing PNG bytes")
            break
        offset += length + 12
    require(header is not None and ended, "Complete PNG required")
    width, height, bits, kind, compression, filtering, interlace = header
    require(0 < width <= 4096 and 0 < height <= 4096, "Bounded map dimensions required")
    require(bits == 8 and kind in (2, 6) and (compression, filtering, interlace) == (0, 0, 0), "Noninterlaced RGB/RGBA8 PNG required")
    channels = 3 if kind == 2 else 4
    raw = zlib.decompress(compressed)
    stride = width * channels
    require(len(raw) == height * (stride + 1), "Exact PNG scanline bytes required")
    result = np.empty((height, stride), dtype=np.uint8)
    for y in range(height):
        line = raw[y * (stride + 1):(y + 1) * (stride + 1)]
        mode = line[0]
        require(mode <= 4, "Known PNG filter required")
        for x, value in enumerate(line[1:]):
            left = int(result[y, x - channels]) if x >= channels else 0
            up = int(result[y - 1, x]) if y else 0
            corner = int(result[y - 1, x - channels]) if y and x >= channels else 0
            if mode == 1:
                value += left
            elif mode == 2:
                value += up
            elif mode == 3:
                value += (left + up) // 2
            elif mode == 4:
                p = left + up - corner
                distances = (abs(p - left), abs(p - up), abs(p - corner))
                value += (left, up, corner)[distances.index(min(distances))]
            result[y, x] = value & 255
    return result.reshape(height, width, channels)


def canonical_triangles(vertices, faces, uv):
    """Only cyclic permutations are admitted; a reversed winding never matches."""
    import numpy as np
    rows = []
    for face in faces:
        positions = [tuple(float(x) for x in vertices[i]) for i in face]
        first = min(range(3), key=lambda i: tuple(positions[i:] + positions[:i]))
        order = [int(face[(first + i) % 3]) for i in range(3)]
        rows.append((tuple(positions[first:] + positions[:first]), tuple(tuple(float(x) for x in uv[i]) for i in order)))
    rows.sort()
    positions = np.asarray([row[0] for row in rows], dtype="<f4")
    positions[positions == 0] = 0  # +0 and -0 represent the same original position.
    uvs = np.asarray([row[1] for row in rows], dtype="<f4")
    return positions, uvs


def check_triangles(vertices, faces, uv, reference):
    import numpy as np
    positions, uvs = canonical_triangles(vertices, faces, uv)
    expected_positions, expected_uvs = reference
    require(np.array_equal(positions, expected_positions), "Original oriented triangle positions differ")
    error = float(np.max(np.abs(uvs - expected_uvs)))
    require(error <= 2e-7, "Triangle corner UVs differ beyond float32 glTF V-flip roundoff")
    return {"triangles": len(faces), "vertices_in_container": len(vertices),
            "cyclic_triangle_positions_exact": True, "reversed_winding_allowed": False,
            "canonical_positions_sha256": array_digest(positions), "canonical_uv_sha256": array_digest(uvs),
            "maximum_uv_error": error}


def glb_check(path, reference, source_maps):
    """Inspect embedded glTF bytes independently from Blender's importer."""
    import numpy as np
    payload = path.read_bytes()
    require(len(payload) >= 28 and struct.unpack_from("<4sII", payload) == (b"glTF", 2, len(payload)), "Complete glTF2 GLB required")
    offset, chunks = 12, {}
    while offset < len(payload):
        length, kind = struct.unpack_from("<II", payload, offset)
        require(kind not in chunks and offset + 8 + length <= len(payload), "Unique complete GLB chunks required")
        chunks[kind] = payload[offset + 8:offset + 8 + length]
        offset += 8 + length
    require(offset == len(payload) and set(chunks) == {0x4e4f534a, 0x004e4942}, "Exactly JSON and BIN GLB chunks required")
    document, binary = json.loads(chunks[0x4e4f534a]), chunks[0x004e4942]
    require(document.get("asset", {}).get("version") == "2.0", "glTF asset version 2.0 required")
    require(not document.get("extensionsRequired") and not document.get("extensionsUsed"), "Core glTF only; no compression or material extensions")
    require(not document.get("animations") and not document.get("skins"), "Static unskinned patch only")
    require(len(document["meshes"]) == len(document["materials"]) == 1, "One mesh and one material required")
    require(len(document["nodes"]) == len(document["scenes"]) == 1 and document.get("scene", 0) == 0 and document["scenes"][0]["nodes"] == [0], "One original patch scene node required")
    node = document["nodes"][0]
    require(node.get("mesh") == 0 and not node.get("children"), "Original patch mesh node required")
    require(node.get("translation", [0, 0, 0]) == [0, 0, 0] and node.get("rotation", [0, 0, 0, 1]) == [0, 0, 0, 1]
            and node.get("scale", [1, 1, 1]) == [1, 1, 1] and node.get("matrix", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]) == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], "No glTF node geometry transforms allowed")
    require(len(document["buffers"]) == 1 and "uri" not in document["buffers"][0], "Embedded GLB buffer required")
    primitives = document["meshes"][0]["primitives"]
    require(len(primitives) == 1, "One triangle primitive required")
    primitive = primitives[0]
    require(primitive.get("mode", 4) == 4 and primitive.get("material") == 0 and not primitive.get("targets"), "Unmodified material triangle primitive required")
    require(set(primitive["attributes"]) <= {"POSITION", "NORMAL", "TEXCOORD_0"} and "TEXCOORD_0" in primitive["attributes"], "UV material only; no vertex colours or hidden attributes")

    def accessor(index):
        entry = document["accessors"][index]
        require("sparse" not in entry and not entry.get("normalized"), "Dense unnormalized accessor required")
        view = document["bufferViews"][entry["bufferView"]]
        dtype = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[entry["componentType"]]
        columns = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[entry["type"]]
        width = np.dtype(dtype).itemsize * columns
        start = view.get("byteOffset", 0) + entry.get("byteOffset", 0)
        stride = view.get("byteStride", width)
        extent = entry.get("byteOffset", 0) + (entry["count"] - 1) * stride + width
        require(0 < entry["count"] <= MAX_FACES * 3 and view.get("buffer", 0) == 0 and stride >= width
                and extent <= view["byteLength"] and view.get("byteOffset", 0) + view["byteLength"] <= len(binary), "In-buffer accessor bounds required")
        return np.ndarray((entry["count"], columns), dtype=dtype, buffer=binary, offset=start, strides=(stride, np.dtype(dtype).itemsize)).copy()

    positions = accessor(primitive["attributes"]["POSITION"])
    require(positions.dtype == np.float32, "Unquantized F32 positions required")
    positions = positions[:, [0, 2, 1]]
    positions[:, 1] *= -1  # Standard Blender Z-up -> glTF Y-up inverse.
    uv = accessor(primitive["attributes"]["TEXCOORD_0"])
    require(uv.dtype == np.float32, "Unquantized F32 UVs required")
    uv[:, 1] = 1 - uv[:, 1]
    faces = accessor(primitive["indices"]).reshape(-1, 3)
    geometry = check_triangles(positions, faces, uv, reference)
    material = document["materials"][0]
    require(material.get("alphaMode", "OPAQUE") == "OPAQUE", "Opaque glTF material required")
    require(material.get("doubleSided", False) is True, "Original two-sided patch material required")
    require(not any(key in material for key in ("occlusionTexture", "normalTexture", "emissiveTexture", "extensions"))
            and material.get("emissiveFactor", [0, 0, 0]) == [0, 0, 0], "No occlusion, normal, emissive or extension material maps allowed")
    pbr = material["pbrMetallicRoughness"]
    require(pbr.get("baseColorFactor", [1, 1, 1, 1]) == [1, 1, 1, 1], "Unit base-colour factor required")
    require(pbr.get("metallicFactor", 1) == pbr.get("roughnessFactor", 1) == 1, "Unit metallic/roughness factors required")
    maps = {}
    for label, key in (("base-color.png", "baseColorTexture"), ("MR.png", "metallicRoughnessTexture")):
        texture_info = pbr[key]
        require(texture_info.get("texCoord", 0) == 0 and not texture_info.get("extensions"), "Original single UV chart required")
        texture = document["textures"][texture_info["index"]]
        image = document["images"][texture["source"]]
        require(image["mimeType"] == "image/png" and "uri" not in image, "Embedded PNG map required")
        view = document["bufferViews"][image["bufferView"]]
        start, length = view.get("byteOffset", 0), view["byteLength"]
        encoded = binary[start:start + length]
        actual, expected = png_pixels(encoded), source_maps[label]
        require(actual.shape[:2] == expected.shape[:2] and np.array_equal(actual[:, :, :3], expected[:, :, :3]), "Embedded map RGB pixels differ: " + label)
        sampler = document.get("samplers", [])[texture["sampler"]] if "sampler" in texture else {}
        require(sampler.get("magFilter") == 9729 and sampler.get("wrapS",10497) == sampler.get("wrapT",10497) == 33071,
                "Linear magnification and clamp-to-edge UV texture sampling required")
        maps[label] = {"embedded_png_sha256": hashlib.sha256(encoded).hexdigest(), "embedded_png_bytes": len(encoded),
                       "decoded_RGB_sha256": array_digest(actual[:, :, :3]), "decoded_source_RGB_exact": True,
                       "alpha_encoded": {"present": actual.shape[2] == 4,
                                         "min": int(actual[:, :, 3].min()) if actual.shape[2] == 4 else None,
                                         "max": int(actual[:, :, 3].max()) if actual.shape[2] == 4 else None,
                                         "sha256": array_digest(actual[:, :, 3]) if actual.shape[2] == 4 else None,
                                         "used_by_shader": False},
                       "shape": list(actual.shape), "sampler": sampler, "CPU_filter_metric": "base-level LOD0; viewer mip minification differs"}
    return {"sha256": digest(path), "bytes": len(payload), "geometry": geometry, "maps": maps,
            "alpha_mode": "OPAQUE", "double_sided": True, "no_occlusion_normal_or_emissive_maps": True,
            "no_vertex_colours": True, "standard_glTF2": True}


def create_mesh(bpy, name, vertices, faces, uv):
    import numpy as np
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    positions = np.empty_like(vertices)
    indices = np.empty_like(faces)
    mesh.vertices.foreach_get("co", positions.ravel())
    mesh.loops.foreach_get("vertex_index", indices.ravel())
    require(np.array_equal(positions, vertices) and np.array_equal(indices, faces), "Constructed patch must preserve original ordered positions and face winding")
    mesh.polygons.foreach_set("use_smooth", np.zeros(len(faces), dtype=np.bool_))
    layer = mesh.uv_layers.new(name="UVMap")
    layer.data.foreach_set("uv", np.ascontiguousarray(uv[faces.reshape(-1)]).ravel())
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def new_material(bpy, name):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.use_backface_culling = False
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    shader, output = nodes.new("ShaderNodeBsdfPrincipled"), nodes.new("ShaderNodeOutputMaterial")
    shader.inputs["Alpha"].default_value = 1
    shader.inputs["Emission Strength"].default_value = 0
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material, shader, nodes, links


def direct_material(bpy, obj, values):
    import numpy as np
    material, shader, nodes, links = new_material(bpy, "Original FLOAT POINT learned PBR")
    mesh = obj.data
    attribute_hashes = {}
    for index, channel in enumerate(CHANNELS):
        attribute = mesh.attributes.new(name="Original_" + channel, type="FLOAT", domain="POINT")
        attribute.data.foreach_set("value", np.ascontiguousarray(values[:, index]))
        actual = np.empty(len(values), dtype="<f4")
        attribute.data.foreach_get("value", actual)
        require(np.array_equal(actual, values[:, index]), "Original FLOAT/POINT readback differs")
        attribute_hashes[channel] = array_digest(actual)
    rgba = np.ones((len(values), 4), dtype="<f4")
    rgba[:, :3] = np.power(np.clip(values[:, :3], 0, 1), np.float32(2.2))
    color = mesh.color_attributes.new(name=LINEAR_RGB, type="FLOAT_COLOR", domain="POINT")
    color.data.foreach_set("color", rgba.ravel())
    actual = np.empty(rgba.size, dtype="<f4")
    color.data.foreach_get("color", actual)
    require(np.array_equal(actual.reshape(rgba.shape), rgba), "Original linear FLOAT_COLOR/POINT readback differs")
    node = nodes.new("ShaderNodeVertexColor")
    node.layer_name = LINEAR_RGB
    links.new(node.outputs["Color"], shader.inputs["Base Color"])
    for channel, socket in (("metallic", "Metallic"), ("roughness", "Roughness")):
        attribute = nodes.new("ShaderNodeAttribute")
        attribute.attribute_type, attribute.attribute_name = "GEOMETRY", "Original_" + channel
        clamp = nodes.new("ShaderNodeClamp")
        clamp.clamp_type = "MINMAX"
        clamp.inputs["Min"].default_value, clamp.inputs["Max"].default_value = 0, 1
        links.new(attribute.outputs["Fac"], clamp.inputs["Value"])
        links.new(clamp.outputs["Result"], shader.inputs[socket])
    mesh.materials.append(material)
    return {"raw_FLOAT_POINT_sha256": attribute_hashes, "linear_RGB_POINT_sha256": array_digest(rgba),
            "RGB_conversion": "clamp(original RGB,0,1)**2.2 at vertices, then triangle interpolation",
            "metallic_roughness": "triangle-interpolate original FLOAT POINT channels, shader clamp to [0,1]",
            "alpha": "opaque 1; original alpha retained in an unused FLOAT POINT attribute"}


def baked_material(bpy, obj):
    material, shader, nodes, links = new_material(bpy, "Baked standard glTF PBR")
    uv = nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    textures = {}
    for name, color_space in (("base-color.png", "sRGB"), ("MR.png", "Non-Color")):
        image = bpy.data.images.load(str(BAKE / name), check_existing=False)
        image.colorspace_settings.name = color_space
        image.alpha_mode = "STRAIGHT"
        node = nodes.new("ShaderNodeTexImage")
        node.image, node.interpolation, node.extension = image, "Linear", "EXTEND"
        links.new(uv.outputs["UV"], node.inputs["Vector"])
        textures[name] = node
    links.new(textures["base-color.png"].outputs["Color"], shader.inputs["Base Color"])
    separate = nodes.new("ShaderNodeSeparateColor")
    separate.mode = "RGB"
    links.new(textures["MR.png"].outputs["Color"], separate.inputs["Color"])
    links.new(separate.outputs["Green"], shader.inputs["Roughness"])
    links.new(separate.outputs["Blue"], shader.inputs["Metallic"])
    obj.data.materials.append(material)
    return {"base_color": {"file": str(BAKE / "base-color.png"), "colorspace": "sRGB", "socket": "Base Color"},
            "MR": {"file": str(BAKE / "MR.png"), "colorspace": "Non-Color", "roughness": "G", "metallic": "B"},
            "UV": "UVMap; standard Blender bottom-up V; PNG row flip already performed by sampler",
            "filter": "Linear", "extension": "EXTEND", "alpha": "opaque 1"}


def mesh_readback(obj, reference):
    import numpy as np
    mesh = obj.data
    require(not obj.modifiers and len(mesh.loops) == len(mesh.polygons) * 3, "Unmodified triangles required")
    require(max(abs(obj.matrix_world[i][j] - (i == j)) for i in range(4) for j in range(4)) < 1e-7, "Original Blender object transform required")
    positions = np.empty((len(mesh.vertices), 3), dtype="<f4")
    faces = np.empty((len(mesh.polygons), 3), dtype="<i4")
    uv = np.empty((len(mesh.loops), 2), dtype="<f4")
    totals = np.empty(len(mesh.polygons), dtype="<i4")
    smooth = np.empty(len(mesh.polygons), dtype=np.bool_)
    mesh.vertices.foreach_get("co", positions.ravel())
    mesh.loops.foreach_get("vertex_index", faces.ravel())
    mesh.polygons.foreach_get("loop_total", totals)
    mesh.polygons.foreach_get("use_smooth", smooth)
    require(np.all(totals == 3) and not smooth.any(), "Original flat triangle shading required")
    require(len(mesh.uv_layers) == 1, "Exactly one imported UV map required")
    mesh.uv_layers[0].data.foreach_get("uv", uv.ravel())
    # Imported glTF may duplicate vertices at flat normals; compare corner rows.
    corner_positions = positions[faces.reshape(-1)]
    corner_faces = np.arange(len(uv), dtype="<i4").reshape(-1, 3)
    result = check_triangles(corner_positions, corner_faces, uv, reference)
    result["vertices_in_container"] = len(mesh.vertices)
    result["UV_domain"] = "CORNER"
    return result


def imported_material_readback(obj, source_maps):
    import numpy as np
    require(len(obj.data.materials) == 1 and len(obj.data.color_attributes) == 0, "Imported baked material must not use vertex colours")
    material = obj.data.materials[0]
    shaders = [node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeBsdfPrincipled"]
    require(len(shaders) == 1, "One imported Principled shader required")
    shader, = shaders
    require(not shader.inputs["Alpha"].is_linked and shader.inputs["Alpha"].default_value == 1, "Imported shader alpha must remain opaque")
    require(not shader.inputs["Normal"].is_linked and shader.inputs["Emission Strength"].default_value == 0, "No added normals or emission")

    def trace(socket, channel=None):
        require(len(socket.links) == 1, "One material input edge required")
        edge = socket.links[0]
        node = edge.from_node
        if node.bl_idname == "ShaderNodeMath":
            require(node.operation == "MULTIPLY", "Only unit glTF factor multiplication is allowed")
            linked = [s for s in node.inputs[:2] if s.is_linked]
            constants = [s for s in node.inputs[:2] if not s.is_linked]
            require(len(linked) == 1 and all(s.default_value == 1 for s in constants), "Unit glTF factor required")
            return trace(linked[0], channel)
        if node.bl_idname in ("ShaderNodeSeparateColor", "ShaderNodeSeparateRGB"):
            require(edge.from_socket.name in ({"G", "Green"} if channel == "G" else {"B", "Blue"}), "Imported MR channel routing differs")
            return trace(node.inputs[0])
        require(node.bl_idname == "ShaderNodeTexImage", "Imported texture image edge required")
        require(channel is None, "MR input must route through its G/B channel separator")
        require(node.interpolation == "Linear", "Imported linear map interpolation required")
        return node

    base = trace(shader.inputs["Base Color"])
    rough, metal = trace(shader.inputs["Roughness"], "G"), trace(shader.inputs["Metallic"], "B")
    require(rough.image == metal.image and base.image != rough.image, "Independent base RGB and shared MR maps required")
    record = {}
    for name, node, space in (("base-color.png", base, "sRGB"), ("MR.png", rough, "Non-Color")):
        image = node.image
        require(image.colorspace_settings.name == space and image.packed_file is not None, "Imported embedded image colorspace differs")
        packed = bytes(image.packed_file.data)
        decoded = png_pixels(packed)
        require(np.array_equal(decoded[:, :, :3], source_maps[name][:, :, :3]), "Imported packed map bytes decode differently")
        pixels = np.empty(len(image.pixels), dtype="<f4")
        image.pixels.foreach_get(pixels)
        require(np.isfinite(pixels).all(), "Finite imported image pixel readback required")
        record[name] = {"packed_png_sha256": hashlib.sha256(packed).hexdigest(), "packed_png_bytes": len(packed),
                        "decoded_RGB_sha256": array_digest(decoded[:, :, :3]), "source_RGB_pixels_exact": True,
                        "Blender_pixels_sha256": array_digest(pixels), "colorspace": space,
                        "interpolation": node.interpolation, "extension": node.extension, "size": list(image.size)}
    return {"maps": record, "roughness": "G", "metallic": "B", "alpha": "opaque 1", "vertex_colours": False}


def setup_scene(bpy, vertices, baseline, scratch):
    import numpy as np
    from mathutils import Vector
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device, scene.cycles.samples = "CPU", SAMPLES
    scene.cycles.use_denoising, scene.cycles.denoiser = True, "OPENIMAGEDENOISE"
    scene.cycles.denoising_use_gpu, scene.cycles.use_adaptive_sampling = False, False
    scene.cycles.seed = 0
    scene.cycles.use_animated_seed = False
    scene.cycles.max_bounces, scene.cycles.diffuse_bounces, scene.cycles.glossy_bounces = 2, 1, 1
    scene.render.threads_mode, scene.render.threads = "FIXED", THREADS
    scene.render.resolution_x = scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.use_compositing = scene.render.use_sequencer = False
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode, scene.render.image_settings.color_depth = "RGB", "8"
    scene.render.image_settings.compression = 15
    settings = baseline["settings_before"]
    scene.view_settings.view_transform, scene.view_settings.look = settings["view_transform"], settings["look"]
    scene.view_settings.exposure, scene.view_settings.gamma = 2, settings["gamma"]
    bpy.context.preferences.filepaths.temporary_directory = str(scratch)
    cycles_preferences = bpy.context.preferences.addons.get("cycles")
    if cycles_preferences:
        cycles_preferences.preferences.compute_device_type = "NONE"
        for device in cycles_preferences.preferences.devices:
            device.use = device.type == "CPU"
    for layer in scene.view_layers:
        layer.cycles.use_denoising, layer.material_override = True, None
    low, high = vertices.min(axis=0).astype(float), vertices.max(axis=0).astype(float)
    center = Vector(((low + high) * .5).tolist())
    radius = float(np.linalg.norm(high - low) * .5)
    require(math.isfinite(radius) and radius > 0, "Nondegenerate patch bounds required")
    world = bpy.data.worlds.new("Prior neutral studio world")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = settings["world_linear_RGBA"]
    background.inputs["Strength"].default_value = settings["world_strength"]
    scene.world = world
    camera_data = bpy.data.cameras.new("Positive X, Z up patch inspection")
    camera_data.type, camera_data.ortho_scale = "ORTHO", 2 * radius * 1.08
    camera_data.clip_start, camera_data.clip_end = .001 * radius, 20 * radius
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    scene.collection.objects.link(camera)
    camera.location = center + Vector((4 * radius, 0, 0))
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = (.5, .5, .5, .5)
    scene.camera = camera
    lights = []
    for prior in baseline["lights"]:
        light = bpy.data.lights.new(prior["name"], "AREA")
        light.energy = prior["energy"] * (radius / baseline["bounding_sphere_radius"]) ** 2
        light.size = prior["size"] * radius / baseline["bounding_sphere_radius"]
        light.color = prior["color"]
        lamp = bpy.data.objects.new(light.name, light)
        scene.collection.objects.link(lamp)
        x, y, z = prior["camera_relative_radius_offset"]
        lamp.location = center + radius * Vector((z, x, y))
        lamp.rotation_mode = "QUATERNION"
        lamp.rotation_quaternion = (center - lamp.location).to_track_quat("-Z", "Y")
        lights.append(lamp)

    def state():
        return {"engine": scene.render.engine, "device": scene.cycles.device, "samples": scene.cycles.samples,
                "threads": scene.render.threads, "threads_mode": scene.render.threads_mode,
                "resolution": [scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage],
                "denoising": scene.cycles.use_denoising, "denoiser": scene.cycles.denoiser,
                "denoising_use_gpu": scene.cycles.denoising_use_gpu, "seed": scene.cycles.seed,
                "animated_seed": scene.cycles.use_animated_seed, "adaptive_sampling": scene.cycles.use_adaptive_sampling,
                "bounces": [scene.cycles.max_bounces, scene.cycles.diffuse_bounces, scene.cycles.glossy_bounces],
                "view_transform": scene.view_settings.view_transform, "look": scene.view_settings.look,
                "exposure": scene.view_settings.exposure, "gamma": scene.view_settings.gamma,
                "world_RGBA": list(background.inputs["Color"].default_value), "world_strength": background.inputs["Strength"].default_value,
                "camera": {"location": list(camera.location), "quaternion": list(camera.rotation_quaternion),
                           "type": camera_data.type, "ortho_scale": camera_data.ortho_scale,
                           "clip": [camera_data.clip_start, camera_data.clip_end], "right": [0, 1, 0], "up": [0, 0, 1]},
                "lights": [{"name": lamp.name, "location": list(lamp.location), "quaternion": list(lamp.rotation_quaternion),
                            "energy": lamp.data.energy, "size": lamp.data.size, "color": list(lamp.data.color)} for lamp in lights]}

    return scene, state, {"bbox": [low.tolist(), high.tolist()], "center": list(center), "radius": radius,
                          "lighting": baseline["lighting"], "light_scaling": "prior energy*(patch radius/prior radius)^2; prior size*(patch radius/prior radius)"}


def main(stage, output, save_blend=False):
    began = time.monotonic()
    output = private_output(stage, output)
    record = {"classification": "RUNNING", "stage": stage, "output": str(output), "views": []}
    try:
        await_host()
        scratch = guard_python_writes(output)
        import bpy
        import numpy as np
        require(bpy.app.background, "Native background Blender required")
        record["Blender_version"] = bpy.app.version_string
        record["Blender_binary"] = bpy.app.binary_path
        paths = [BAKE / name for name in INPUT_NAMES] + list(BASELINE_HASHES) + [Path(__file__).resolve()]
        if stage == "compare":
            paths += [PRIVATE / "export/patch.glb", PRIVATE / "export/export-checkpoint.json"]
        require(all(path.is_file() and not path.is_symlink() for path in paths), "Complete nonsymlink prepared inputs required")
        before = {str(path): {"sha256": digest(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in paths}
        for path, expected in BASELINE_HASHES.items():
            require(before[str(path)]["sha256"] == expected, "Prior reviewed settings source changed")
        baseline = json.loads(BASELINE_METADATA.read_text(encoding="utf-8"))
        require(baseline["classification"] == "PASS", "Prior preview metadata must pass")
        # The evidence uses source data, not executable hooks from bake.json.
        bake_metadata = json.loads((BAKE / "bake.json").read_text(encoding="utf-8"))
        vertices, faces, uv, values = [np.load(BAKE / name, allow_pickle=False) for name in INPUT_NAMES[:4]]
        require(vertices.dtype == np.float32 and vertices.ndim == 2 and vertices.shape[1] == 3 and 0 < len(vertices) <= MAX_VERTICES, "Small F32[N,3] patch vertices required")
        require(faces.dtype == np.int32 and faces.ndim == 2 and faces.shape[1] == 3 and 0 < len(faces) <= MAX_FACES, "Small I32[F,3] original triangle indices required")
        require(uv.dtype == values.dtype == np.float32 and uv.shape == (len(vertices), 2) and values.shape == (len(vertices), 6), "Original F32 UV/direct row shapes required")
        require(all(np.isfinite(array).all() for array in (vertices, uv, values)), "Finite prepared values required")
        require(faces.min() >= 0 and faces.max() < len(vertices) and np.all((uv >= 0) & (uv <= 1)), "Valid original indices and unit UVs required")
        source_maps = {name: png_pixels((BAKE / name).read_bytes()) for name in ("base-color.png", "MR.png")}
        require(source_maps["base-color.png"].shape[2] == 4, "Original RGBA8 base colour required; its encoded alpha is retained but shader opacity is fixed")
        require(source_maps["MR.png"].shape[2] == 3 and source_maps["MR.png"].shape[:2] == source_maps["base-color.png"].shape[:2], "Matching RGB8 MR map required")
        reference = canonical_triangles(vertices, faces, uv)
        record.update({"source_files_before": before, "patch_vertices": len(vertices), "patch_triangles": len(faces),
                       "source_payload_sha256": {name: array_digest(value) for name, value in zip(INPUT_NAMES[:4], (vertices, faces, uv, values))},
                       "source_bake_metadata_sha256": before[str(BAKE / "bake.json")]["sha256"],
                       "source_bake_classification": bake_metadata.get("classification"), "full_mesh_imported": False,
                       "source_base_color_encoded_alpha": {"min": int(source_maps["base-color.png"][:, :, 3].min()),
                                                           "max": int(source_maps["base-color.png"][:, :, 3].max()),
                                                           "sha256": array_digest(source_maps["base-color.png"][:, :, 3]), "used_by_shader": False},
                       "alpha": "opaque for both materials", "normal_maps": False, "geometry_repair": False,
                       "write_boundary": "Python audit guard restricts writes to stage output; native Blender bounded by owning host",
                       "RGB_semantic_difference": "Direct: clamp RGB**2.2 before vertex interpolation. Baked: encoded RGB texture interpolation with standard sRGB decoding; differences include gamma convention, quantization and finite texture sampling."})
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.context.preferences.filepaths.temporary_directory = str(scratch)
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.cycles.device = "CPU"
        if stage == "export":
            direct = create_mesh(bpy, "Original direct reference patch", vertices, faces, uv)
            record["direct_material"] = direct_material(bpy, direct, values)
            record["direct_geometry"] = mesh_readback(direct, reference)
            direct.hide_render = True
            baked = create_mesh(bpy, "Original patch with baked UV PBR", vertices, faces, uv)
            record["baked_material"] = baked_material(bpy, baked)
            record["baked_geometry_before"] = mesh_readback(baked, reference)
            bpy.ops.object.select_all(action="DESELECT")
            baked.select_set(True)
            bpy.context.view_layer.objects.active = baked
            glb = output / "patch.glb"
            export_settings = {"filepath": str(glb), "export_format": "GLB", "use_selection": True,
                               "export_yup": True, "export_apply": False, "export_animations": False,
                               "export_texcoords": True, "export_normals": True, "export_tangents": False,
                               "export_materials": "EXPORT", "export_image_format": "AUTO",
                               "export_extras": False, "export_cameras": False, "export_lights": False}
            require(bpy.ops.export_scene.gltf(**export_settings) == {"FINISHED"}, "GLB export failed")
            record["export_settings"] = export_settings
            record["glb"] = glb_check(glb, reference, source_maps)
            require(mesh_readback(baked, reference) == record["baked_geometry_before"], "Export mutated source patch")
            if save_blend:
                require(bpy.ops.wm.save_as_mainfile(filepath=str(output / "patch.blend"), check_existing=False) == {"FINISHED"}, "Private blend save failed")
                record["blend_sha256"] = digest(output / "patch.blend")
        else:
            glb = PRIVATE / "export/patch.glb"
            export_receipt = json.loads((PRIVATE / "export/export-checkpoint.json").read_text(encoding="utf-8"))
            require(export_receipt["classification"] == "PASS" and export_receipt["glb"]["sha256"] == digest(glb), "Passing unchanged export checkpoint required")
            require(export_receipt["source_payload_sha256"] == record["source_payload_sha256"], "Compare inputs differ from export")
            require(all(export_receipt["source_files_before"][str(BAKE / name)] == before[str(BAKE / name)] for name in INPUT_NAMES), "Compare bake inputs differ from export")
            record["glb"] = glb_check(glb, reference, source_maps)
            # This fresh factory scene imports only the exported baked GLB.
            import_settings = {"filepath": str(glb), "merge_vertices": False, "import_shading": "FLAT"}
            require(bpy.ops.import_scene.gltf(**import_settings) == {"FINISHED"}, "Fresh GLB import failed")
            meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
            require(len(meshes) == 1 and len(bpy.context.scene.objects) == 1, "Exactly one baked patch imported")
            baked, = meshes
            record["import_settings"] = import_settings
            record["imported_geometry_before"] = mesh_readback(baked, reference)
            record["imported_material_before"] = imported_material_readback(baked, source_maps)
            direct = create_mesh(bpy, "Original direct reference patch", vertices, faces, uv)
            record["direct_material"] = direct_material(bpy, direct, values)
            record["direct_geometry_before"] = mesh_readback(direct, reference)
            scene, settings, framing = setup_scene(bpy, vertices, baseline, scratch)
            record["framing"], record["settings"] = framing, settings()
            save_json(output / "import-checkpoint.json", {**record, "classification": "IMPORT_VERIFIED_BEFORE_RENDER"})
            rendered = {}
            for name, visible in (("direct", direct), ("baked", baked)):
                require(time.monotonic() - began < 150, "Insufficient time remains for another bounded render")
                direct.hide_render, baked.hide_render = visible != direct, visible != baked
                require(settings() == record["settings"], "Camera, lights or render settings changed")
                path = output / (name + ".png")
                scene.render.filepath = str(path)
                started = time.monotonic()
                require(bpy.ops.render.render(write_still=True) == {"FINISHED"}, "Render failed")
                require(settings() == record["settings"], "Camera, lights or render settings changed during render")
                image = png_pixels(path.read_bytes())
                require(image.shape == (RESOLUTION, RESOLUTION, 3), "640-square RGB8 render required")
                rendered[name] = image
                view = {"name": name, "file": str(path), "sha256": digest(path), "bytes": path.stat().st_size,
                        "seconds": time.monotonic() - started, "RGB_mean": image.mean(axis=(0, 1)).tolist(),
                        "RGB_std": image.std(axis=(0, 1)).tolist(), "RGB_min": image.min(axis=(0, 1)).tolist(),
                        "RGB_max": image.max(axis=(0, 1)).tolist(), "settings": settings()}
                record["views"].append(view)
                require(mesh_readback(direct, reference) == record["direct_geometry_before"], "Direct geometry changed during render")
                require(mesh_readback(baked, reference) == record["imported_geometry_before"], "Baked geometry changed during render")
                require(imported_material_readback(baked, source_maps) == record["imported_material_before"], "Imported maps/material changed during render")
                save_json(output / (name + "-checkpoint.json"), {"classification": "RENDER_VERIFIED", **view, "geometry_unchanged": True, "imported_maps_unchanged": True})
                print("BRICK_UV_RENDERED=" + name, flush=True)
            difference = rendered["direct"].astype(float) - rendered["baked"].astype(float)
            record["comparison_RGB8_full_frame"] = {"MAE": float(np.abs(difference).mean()), "RMSE": float(np.sqrt(np.square(difference).mean())),
                                                    "absolute_p95": float(np.percentile(np.abs(difference), 95)), "absolute_max": float(np.abs(difference).max()),
                                                    "exact_equal_pixel_fraction": float(np.all(difference == 0, axis=2).mean())}
        after = {str(path): {"sha256": digest(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in paths}
        record["source_files_after"] = after
        require(before == after, "A source or export input changed during this stage")
        record.update({"classification": "PASS", "sources_unchanged": True, "geometry_unchanged": True,
                       "elapsed_seconds": time.monotonic() - began, "retry": False})
        save_json(output / (stage + "-checkpoint.json"), record)
        print("BRICK_UV_STAGE_PASS=" + stage, flush=True)
    except BaseException as exc:
        record.update({"classification": "FAIL", "error_type": type(exc).__name__, "error": str(exc),
                       "elapsed_seconds": time.monotonic() - began, "retry": False})
        # Exclusive creation preserves any prior successful checkpoint.
        save_json(output / "failure.json", record)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("export", "compare"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-blend", action="store_true", help="Optional private export-stage project only")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    require(not args.save_blend or args.stage == "export", "Blend save is export-stage only")
    main(args.stage, args.output, args.save_blend)
