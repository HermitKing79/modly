"""Independent stdlib-only authenticated GLB versus frozen Blender OBJ gate.

Reads inputs only. --report creates a new evidence file and refuses overwrite.
No Blender, engine, project, package, network or GPU operations are invoked.
"""
import argparse
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import sys
import zlib

EXPECTED_SHA256 = "afc880ff5ec2e0c1a9a6977999fa633506b68915c73eddb2c543ee17512c1201"
POSITION_TOLERANCE_METRES = 2e-6
UV_TOLERANCE = 2e-6
NORMAL_TOLERANCE = 1e-5
INCH_METRES = .0254


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def finite(values):
    return all(math.isfinite(x) for x in values)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_glb(path):
    raw = Path(path).read_bytes()
    require(sha256(raw) == EXPECTED_SHA256, "Original GLB SHA-256 mismatch")
    require(len(raw) >= 20, "Truncated GLB")
    magic, version, total = struct.unpack_from("<4sII", raw)
    require(magic == b"glTF" and version == 2 and total == len(raw), "Invalid GLB header")
    chunks = {}
    offset = 12
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        require(kind not in chunks and offset + 8 + length <= len(raw), "Invalid GLB chunk")
        chunks[kind] = raw[offset + 8:offset + 8 + length]
        offset += 8 + length
    doc = json.loads(chunks[0x4e4f534a])
    binary = chunks[0x004e4942]
    require(len(doc["nodes"]) == 1 and doc["nodes"][0].get("mesh") == 0,
            "Expected one identity mesh node")
    node = doc["nodes"][0]
    require(not any(k in node for k in ("matrix", "translation", "rotation", "scale", "children", "skin")),
            "Unexpected GLB node transform, hierarchy or skin")
    require(len(doc["meshes"]) == 1 and len(doc["meshes"][0]["primitives"]) == 1,
            "Expected one mesh primitive")
    primitive = doc["meshes"][0]["primitives"][0]
    require(primitive.get("mode", 4) == 4, "GLB primitive is not TRIANGLES")

    def accessor(index):
        item = doc["accessors"][index]
        require("sparse" not in item and not item.get("normalized"), "Unsupported accessor packing")
        view = doc["bufferViews"][item["bufferView"]]
        require(view.get("buffer", 0) == 0, "Unexpected external buffer")
        formats = {5121: "B", 5123: "H", 5125: "I", 5126: "f"}
        sizes = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
        fmt = "<" + formats[item["componentType"]] * sizes[item["type"]]
        size = struct.calcsize(fmt)
        stride = view.get("byteStride", size)
        start = view.get("byteOffset", 0) + item.get("byteOffset", 0)
        end = start + (item["count"] - 1) * stride + size
        require(stride >= size and end <= view.get("byteOffset", 0) + view["byteLength"] and end <= len(binary),
                "Accessor outside its buffer view")
        result = [struct.unpack_from(fmt, binary, start + i * stride) for i in range(item["count"])]
        require(all(finite(x) for x in result), "Nonfinite GLB attribute")
        return result

    attrs = primitive["attributes"]
    positions = accessor(attrs["POSITION"])
    normals = accessor(attrs["NORMAL"])
    uvs = accessor(attrs["TEXCOORD_0"])
    indices = [x[0] for x in accessor(primitive["indices"])]
    require(len(positions) == len(normals) == len(uvs) == 2695, "Unexpected source attribute count")
    require(len(indices) == 3000 and set(indices) == set(range(2695)), "Unexpected source triangle/index usage")
    require(len(set(positions)) == 502, "Unexpected source unique position count")
    return dict(positions=positions, normals=normals, uvs=uvs,
                triangles=[tuple(indices[i:i + 3]) for i in range(0, len(indices), 3)],
                document=doc, binary=binary, primitive=primitive, sha256=sha256(raw))


def parse_obj_text(text):
    result = dict(positions=[], uvs=[], normals=[], triangles=[], face_materials=[], mtllibs=[])
    material = None
    allowed = {"o", "g", "s"}
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        command, _, rest = line.partition(" ")
        fields = rest.split()
        if command in ("v", "vt", "vn"):
            key, count = {"v": ("positions", 3), "vt": ("uvs", 2), "vn": ("normals", 3)}[command]
            require(len(fields) == count, f"OBJ line {line_number}: unexpected {command} components")
            values = tuple(map(float, fields))
            require(finite(values), f"OBJ line {line_number}: nonfinite attribute")
            result[key].append(values)
        elif command == "f":
            require(len(fields) == 3, f"OBJ line {line_number}: nontriangular face")
            corners = []
            for field in fields:
                indices = field.split("/")
                require(len(indices) == 3 and all(indices), f"OBJ line {line_number}: missing UV/normal")
                corner = []
                for index, key in zip(map(int, indices), ("positions", "uvs", "normals")):
                    require(index != 0, f"OBJ line {line_number}: zero index")
                    resolved = index - 1 if index > 0 else len(result[key]) + index
                    require(0 <= resolved < len(result[key]), f"OBJ line {line_number}: index outside attribute array")
                    corner.append(resolved)
                corners.append(tuple(corner))
            result["triangles"].append(tuple(corners))
            result["face_materials"].append(material)
        elif command == "mtllib":
            result["mtllibs"].append(rest.strip().strip('"'))
        elif command == "usemtl":
            material = rest.strip()
        else:
            require(command in allowed, f"OBJ line {line_number}: unsupported geometry/directive {command}")
    return result


def bounds(positions):
    mins = tuple(min(p[i] for p in positions) for i in range(3))
    maxs = tuple(max(p[i] for p in positions) for i in range(3))
    return dict(min=list(mins), max=list(maxs), dimensions=[b - a for a, b in zip(mins, maxs)])


def cyclic_key(face):
    return min(tuple(face[i:] + face[:i]) for i in range(3))


def compare_geometry(source, obj):
    require(len(obj["triangles"]) == 1000, "OBJ must contain exactly 1000 triangles")
    require(len(set(obj["positions"])) == 502, "OBJ must contain exactly 502 unique geometric positions")
    for column, key in enumerate(("positions", "uvs", "normals")):
        used = {corner[column] for face in obj["triangles"] for corner in face}
        require(used == set(range(len(obj[key]))), f"Unused or missing OBJ {key}")

    unique = list(dict.fromkeys(source["positions"]))
    source_ids = {position: i for i, position in enumerate(unique)}
    size = POSITION_TOLERANCE_METRES
    buckets = collections.defaultdict(list)
    for i, position in enumerate(unique):
        buckets[tuple(math.floor(x / size) for x in position)].append(i)
    actual_source_space = [(-p[1] * INCH_METRES, p[2] * INCH_METRES, -p[0] * INCH_METRES)
                           for p in obj["positions"]]
    obj_ids = []
    max_position_error = 0.0
    for position in actual_source_space:
        cell = tuple(math.floor(x / size) for x in position)
        matches = []
        for delta in itertools.product((-1, 0, 1), repeat=3):
            for i in buckets.get(tuple(a + b for a, b in zip(cell, delta)), ()):
                error = math.dist(position, unique[i])
                if error <= size:
                    matches.append((i, error))
        require(len(matches) == 1, "OBJ position has no unique source correspondence within 2e-6 metres")
        obj_ids.append(matches[0][0])
        max_position_error = max(max_position_error, matches[0][1])
    require(set(obj_ids) == set(range(502)), "OBJ is missing source geometric positions")

    remaining = collections.defaultdict(list)
    for face_number, face in enumerate(source["triangles"]):
        ids = tuple(source_ids[source["positions"][i]] for i in face)
        require(len(set(ids)) == 3, "Degenerate source triangle")
        remaining[cyclic_key(ids)].append((face_number, face, ids))
    max_uv_error = max_normal_error = max_raw_normal_error = 0.0
    raw_affected_corners = 0
    source_normal_lengths, obj_normal_lengths, normal_unit_dots = [], [], []
    for face_number, face in enumerate(obj["triangles"]):
        ids = tuple(obj_ids[c[0]] for c in face)
        key = cyclic_key(ids)
        candidates = remaining.get(key, [])
        if not candidates:
            reverse = cyclic_key(tuple(reversed(ids)))
            require(not remaining.get(reverse), f"OBJ face {face_number}: reversed winding")
            raise ValidationError(f"OBJ face {face_number}: extra, duplicate or missing oriented surface")
        accepted = None
        for candidate_number, (_, source_face, source_corner_ids) in enumerate(candidates):
            for shift in range(3):
                rotated_ids = source_corner_ids[shift:] + source_corner_ids[:shift]
                if ids != rotated_ids:
                    continue
                rotated_face = source_face[shift:] + source_face[:shift]
                uv_error = normal_error = 0.0
                raw_errors, source_lengths, actual_lengths, unit_dots = [], [], [], []
                for corner, source_index in zip(face, rotated_face):
                    u, vv = source["uvs"][source_index]
                    expected_uv = (u, 1.0 - vv)
                    uv_error = max(uv_error, *(abs(a - b) for a, b in zip(obj["uvs"][corner[1]], expected_uv)))
                    nx, ny, nz = obj["normals"][corner[2]]
                    actual_normal = (-ny, nz, -nx)
                    source_normal = source["normals"][source_index]
                    raw_errors.append(math.dist(actual_normal, source_normal))
                    source_length = math.sqrt(sum(x*x for x in source_normal))
                    actual_length = math.sqrt(sum(x*x for x in actual_normal))
                    require(source_length > 0 and actual_length > 0, "Zero normal direction")
                    source_unit = tuple(x / source_length for x in source_normal)
                    actual_unit = tuple(x / actual_length for x in actual_normal)
                    normal_error = max(normal_error, math.dist(actual_unit, source_unit))
                    source_lengths.append(source_length)
                    actual_lengths.append(actual_length)
                    unit_dots.append(sum(a*b for a,b in zip(source_unit, actual_unit)))
                if uv_error <= UV_TOLERANCE and normal_error <= NORMAL_TOLERANCE:
                    accepted = (candidate_number, uv_error, normal_error, raw_errors,
                                source_lengths, actual_lengths, unit_dots)
                    break
            if accepted is not None:
                break
        require(accepted is not None, f"OBJ face {face_number}: corner UV or unit-normal direction mismatch (UV <=2e-6, unit-direction Euclidean <=1e-5)")
        candidate_number, uv_error, normal_error, raw_errors, source_lengths, actual_lengths, unit_dots = accepted
        candidates.pop(candidate_number)
        max_uv_error = max(max_uv_error, uv_error)
        max_normal_error = max(max_normal_error, normal_error)
        max_raw_normal_error = max(max_raw_normal_error, *raw_errors)
        raw_affected_corners += sum(error > NORMAL_TOLERANCE for error in raw_errors)
        source_normal_lengths.extend(source_lengths)
        obj_normal_lengths.extend(actual_lengths)
        normal_unit_dots.extend(unit_dots)
    require(not any(remaining.values()), "OBJ missing source triangles")
    expected_obj = [(-z / INCH_METRES, -x / INCH_METRES, y / INCH_METRES) for x, y, z in source["positions"]]
    return dict(status="PASS", unique_positions=502, oriented_triangles=1000,
                source_attribute_vertices=len(source["positions"]), obj_position_entries=len(obj["positions"]),
                obj_uv_entries=len(obj["uvs"]), obj_normal_entries=len(obj["normals"]),
                compared_corners=3000, unused_attributes=0, reversed_faces=0,
                max_position_error_metres=max_position_error, max_uv_component_error=max_uv_error,
                normal_comparison="Unit directions compared at unchanged 1e-5 Euclidean tolerance; output vectors not altered",
                max_unit_normal_direction_error=max_normal_error,
                min_unit_normal_dot=min(normal_unit_dots),
                max_raw_normal_euclidean_error=max_raw_normal_error,
                raw_normal_vector_gate="NO-GO" if raw_affected_corners else "PASS",
                raw_normal_affected_corners=raw_affected_corners,
                source_normal_length_range=[min(source_normal_lengths), max(source_normal_lengths)],
                obj_normal_length_range=[min(obj_normal_lengths), max(obj_normal_lengths)],
                normal_qualification="Geometric normal directions preserved; raw magnitude serialization is separately reported, not claimed identical",
                source_glb_xyz_metres=bounds(source["positions"]),
                expected_obj_xyz_inches=bounds(expected_obj), actual_obj_xyz_inches=bounds(obj["positions"]),
                recovered_glb_xyz_metres=bounds(actual_source_space),
                sharp_edge_behavior="All oriented per-corner normals correspond; source discontinuities preserved within tolerance")


def decode_png(raw):
    """Decode noninterlaced eight-bit RGB/RGBA PNG without external packages."""
    require(raw.startswith(b"\x89PNG\r\n\x1a\n"), "Expected a PNG texture")
    offset, compressed, header = 8, bytearray(), None
    while offset < len(raw):
        length = struct.unpack_from(">I", raw, offset)[0]
        kind = raw[offset + 4:offset + 8]
        payload = raw[offset + 8:offset + 8 + length]
        require(len(payload) == length, "Truncated PNG")
        expected_crc = struct.unpack_from(">I", raw, offset + 8 + length)[0]
        require(zlib.crc32(kind + payload) & 0xffffffff == expected_crc, "PNG CRC mismatch")
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload)
        if kind == b"IDAT":
            compressed.extend(payload)
        offset += length + 12
        if kind == b"IEND":
            break
    require(header is not None, "PNG header missing")
    width, height, depth, color, compression, filtering, interlace = header
    require(depth == 8 and color in (2, 6) and (compression, filtering, interlace) == (0, 0, 0),
            "PNG must be eight-bit noninterlaced RGB/RGBA")
    require(0 < width <= 4096 and 0 < height <= 4096, "Unexpected texture dimensions")
    channels = 3 if color == 2 else 4
    stride = width * channels
    decoded = zlib.decompress(compressed)
    require(len(decoded) == height * (stride + 1), "PNG scanline size mismatch")
    previous = bytearray(stride)
    rgba = bytearray()
    for row_number in range(height):
        row_start = row_number * (stride + 1)
        mode = decoded[row_start]
        row = bytearray(decoded[row_start + 1:row_start + 1 + stride])
        require(mode in range(5), "Unsupported PNG filter")
        for x in range(stride):
            left = row[x - channels] if x >= channels else 0
            above = previous[x]
            upper_left = previous[x - channels] if x >= channels else 0
            if mode == 1:
                predictor = left
            elif mode == 2:
                predictor = above
            elif mode == 3:
                predictor = (left + above) // 2
            elif mode == 4:
                p = left + above - upper_left
                distances = (abs(p - left), abs(p - above), abs(p - upper_left))
                predictor = (left, above, upper_left)[distances.index(min(distances))]
            else:
                predictor = 0
            row[x] = (row[x] + predictor) & 255
        if channels == 4:
            rgba.extend(row)
        else:
            for x in range(0, stride, 3):
                rgba.extend(row[x:x + 3])
                rgba.append(255)
        previous = row
    return width, height, bytes(rgba)


def local_file(root, relative):
    require(relative and not relative.startswith("-"), "Missing path or unsupported MTL map options")
    path = (root / relative.strip('"')).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), "MTL/texture missing or escapes isolated output directory")
    return path


def compare_material(source, obj, obj_path):
    require(len(obj["mtllibs"]) == 1, "Expected exactly one MTL library")
    mtl_path = local_file(obj_path.parent, obj["mtllibs"][0])
    materials, current = {}, None
    for line in mtl_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        command, _, rest = line.partition(" ")
        if command == "newmtl":
            require(rest not in materials, "Duplicate MTL material")
            materials[rest] = current = {}
        else:
            require(current is not None and command not in current, "MTL field before material or duplicate field")
            current[command] = rest.strip()
    used = set(obj["face_materials"])
    require(len(used) == 1 and None not in used, "Expected one material on all OBJ faces")
    name = next(iter(used))
    require(set(materials) == {name}, "Missing or unused MTL material")
    material = materials[name]
    doc = source["document"]
    pbr = doc["materials"][source["primitive"]["material"]]["pbrMetallicRoughness"]
    roughness, metallic = pbr.get("roughnessFactor", 1), pbr.get("metallicFactor", 1)
    require(abs(roughness - .35) <= 1e-6 and metallic == 0, "Unexpected original neutral PBR factors")
    require("Pr" in material and "Pm" in material, "OBJ lacks exported PBR roughness/metallic extensions")
    require(abs(float(material["Pr"]) - roughness) <= 1e-6 and abs(float(material["Pm"]) - metallic) <= 1e-6,
            "MTL roughness or metallic mismatch")
    factor = pbr.get("baseColorFactor", [1, 1, 1, 1])
    require("Kd" in material, "MTL Kd omitted: exact importer base-color modulation is unqualified")
    kd = tuple(map(float, material["Kd"].split()))
    require(len(kd) == 3 and all(abs(a - b) <= 1e-6 for a, b in zip(kd, factor[:3])), "MTL base color factor mismatch")
    require(abs(float(material.get("d", "1")) - factor[3]) <= 1e-6, "MTL opacity mismatch")
    require(not any(x in material for x in ("map_Pr", "map_Pm", "map_Bump", "bump", "norm")),
            "Unexpected additional texture channel in neutral MTL")
    texture_path = local_file(obj_path.parent, material.get("map_Kd", ""))
    image = doc["images"][doc["textures"][pbr["baseColorTexture"]["index"]]["source"]]
    require(image.get("mimeType") == "image/png" and "bufferView" in image, "Unexpected original image storage")
    view = doc["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    original_png = source["binary"][start:start + view["byteLength"]]
    actual_png = texture_path.read_bytes()
    original_pixels, actual_pixels = decode_png(original_png), decode_png(actual_png)
    require(original_pixels == actual_pixels, "Neutral basecolor texture decoded pixels differ")
    return dict(status="PASS", name=name, roughness=float(material["Pr"]), metallic=float(material["Pm"]),
                base_color_factor=list(kd), opacity=float(material.get("d", "1")),
                mtl_path=str(mtl_path), mtl_sha256=sha256(mtl_path.read_bytes()),
                texture_path=str(texture_path), texture_sha256=sha256(actual_png),
                source_texture_sha256=sha256(original_png), texture_pixels_identical=True,
                texture_dimensions=list(original_pixels[:2]),
                qualification="Transport factors and decoded basecolor preserved; engine shader equivalence requires separate audit")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_source = Path(__file__).parent.parent / "amd-static-prop-uv-improve-20261031-v1/asset/experimental-static-prop.glb"
    parser.add_argument("--glb", type=Path, default=default_source)
    parser.add_argument("--obj", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = dict(status="NO-GO", source_glb=str(args.glb.resolve()), source_obj=str(args.obj.resolve()),
                  mapping="OBJ inches=(-GLB.z,-GLB.x,GLB.y)/0.0254; determinant +1; OBJ UV=(u,1-v)",
                  tolerances=dict(position_euclidean_metres=POSITION_TOLERANCE_METRES,
                                  uv_component_absolute=UV_TOLERANCE,
                                  unit_normal_direction_euclidean=NORMAL_TOLERANCE),
                  normal_semantics_correction="Normalize only during comparison. Keep raw normal-vector gate results and initial raw NO-GO evidence; no threshold or output changes")
    try:
        source = read_glb(args.glb)
        obj_bytes = args.obj.read_bytes()
        obj = parse_obj_text(obj_bytes.decode("utf-8-sig"))
        report["glb_sha256"] = source["sha256"]
        report["obj_sha256"] = sha256(obj_bytes)
        report["geometry"] = compare_geometry(source, obj)
        report["material"] = compare_material(source, obj, args.obj.resolve())
        require(sha256(args.glb.read_bytes()) == EXPECTED_SHA256, "Original GLB changed during validation")
        report["status"] = "PASS"
    except (ValidationError, ValueError, KeyError, IndexError, OSError, struct.error, zlib.error) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.report:
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
