"""Read-only, CPU NumPy proof of original brick material/mesh correspondence.

This compares all rows and all position/index bits; it never extracts a mesh,
imports a model, starts a process, or writes private arrays or images.
"""
import sys
sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np

E = Path(__file__).resolve().parent
ROOT = E.parents[2]
ARTIFACTS = ROOT.parent / "artifacts"
MESH = ARTIFACTS / "amd-brick-cpu-mesh-8473579/mesh"
GEOMETRY = ARTIFACTS / "amd-brick-neural-completion-c9447dd/neural-output/shape-decoded.safetensors"
FIELD = Path("C:/Users/wattz/Documents/ModlyArtifacts/amd-brick-pbr-qualified-a064378/field/texture-field.safetensors")
UPSTREAM = Path("D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD")
N, F, BATCH = 1_957_891, 4_173_584, 32_768
CHANNELS = ("red", "green", "blue", "metallic", "roughness", "alpha")
PINS = {
    "geometry": (GEOMETRY, "d4de05fc82315e44a739c9815a07a531555e108af05c619ea0426d0c630ef861"),
    "material": (FIELD, "8cd735630149e28829efe686d30f1c7c255dde76748dc3d525236fb5c2c81394"),
    "material_metadata": (FIELD.with_name("texture-field-metadata.json"), "4f65c6730f4a8669d9e62ba4f151f15b19f6cfd6b7e7dcd81289ad923bbdd8c2"),
    "mesh": (MESH / "mesh.ply", "ec9736e2510c26303196c4e3fc2045a1a1d36401b2576ee68f102ee29b551ffd"),
    "vertices": (MESH / "vertices.npy", "018edd85646f9b504ec52ceb311f69e0c7610e4d1ab306f41c7e73a7a77d3252"),
    "faces": (MESH / "faces.npy", "9314a921741bab5c3d87348b4ea26488ee1ab13201b33d67252d6d272997ab1d"),
    "extraction_receipt": (MESH / "mesh.json", "12810c52c808369f360f8d826eb1f58ec66224d5b520dcf129e0b5a50ff1a37a"),
    "extractor_source": (ROOT / "scripts/compatibility/amd_generation_mesh.py", "68a5e64bf7ab65c98f9b798e8d81c61c6cb629b1eaaf193637d5e52de0cae8e8"),
    "converter_source": (ROOT / "scripts/compatibility/amd_flexible_dual_grid_cpu.py", "744e44768a48603e4f6d68e8dc4ae0e1f8ce78b16c2efd73b511a2a309562b40"),
    "material_neural_source": (ROOT / "experimental/extensions/trellis2-amd/inference_texture/neural.py", "d41f6f30d10dc83d94903bb22fd0a8f5c1e928e85d22c62e8a65c8479df91987"),
    "material_writer_source": (ROOT / "scripts/compatibility/amd_texture_field_artifact.py", "77d9592662653413e9b40f81741e47e2a23739e9e5d013d48b6cefff1831d3b5"),
    "upstream_pipeline": (UPSTREAM / "trellis2/pipelines/trellis2_image_to_3d.py", "faade40274e44823e72e1659b8ced96f8815ac9e08b3abb89271077c4394982b"),
    "upstream_pbr_renderer": (UPSTREAM / "trellis2/renderers/pbr_mesh_renderer.py", "ab33ba6d04b2d1b7e518b90372b242cbe00ad9043be66a22e681261b04de7872"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def read_json(path):
    require(Path(path).stat().st_size <= 2 * 1024**2, "Bounded JSON required")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=object_pairs)


def array_digest(array):
    hasher = hashlib.sha256()
    for start in range(0, len(array), BATCH):
        hasher.update(np.ascontiguousarray(array[start:start+BATCH]).view(np.uint8))
    return hasher.hexdigest()


def equal_bits(left, right, label):
    require(left.shape == right.shape and left.dtype == right.dtype, label + " schema differs")
    for start in range(0, len(left), BATCH):
        a = np.ascontiguousarray(left[start:start+BATCH]).view(np.uint8)
        b = np.ascontiguousarray(right[start:start+BATCH]).view(np.uint8)
        require(np.array_equal(a, b), label + " row bytes/order differ")


def read_safetensors(path, schema):
    """Validate a bounded complete header, then map exactly the named tensors."""
    path = Path(path)
    require(not path.is_symlink() and path.is_file() and path.stat().st_nlink == 1,
            "Plain single-link safetensors required")
    size = path.stat().st_size
    require(8 < size <= 256 * 1024**2, "Bounded safetensors required")
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        require(2 <= length <= 65_536 and 8 + length < size, "Bounded complete header required")
        header = json.loads(handle.read(length), object_pairs_hook=object_pairs)
    require(set(header) - {"__metadata__"} == set(schema), "Tensor names differ")
    descriptors = []
    for name, (dtype, shape) in schema.items():
        row = header[name]
        require(set(row) == {"dtype", "shape", "data_offsets"}, "Tensor descriptor differs")
        require(row["dtype"] == dtype and row["shape"] == list(shape), "Tensor schema differs")
        offsets = row["data_offsets"]
        require(isinstance(offsets, list) and len(offsets) == 2
                and all(type(v) is int for v in offsets), "Tensor offsets differ")
        a, b = offsets
        npdtype = {"F32": "<f4", "I32": "<i4", "BF16": "<u2"}[dtype]
        require(0 <= a < b <= size - 8 - length
                and b - a == int(np.prod(shape)) * np.dtype(npdtype).itemsize,
                "Tensor payload size differs")
        descriptors.append((a, b, name, npdtype, shape))
    previous = 0
    for a, b, *_ in sorted(descriptors):
        require(a == previous, "Tensor payload gap/overlap")
        previous = b
    require(previous == size - 8 - length, "Trailing tensor payload")
    return {name: np.memmap(path, mode="r", offset=8+length+a, dtype=dtype, shape=shape)
            for a, b, name, dtype, shape in descriptors}, header.get("__metadata__", {})


def read_ply(path, vertices, faces):
    with Path(path).open("rb") as handle:
        lines = []
        for _ in range(20):
            line = handle.readline(512)
            require(line.endswith(b"\n"), "Complete bounded PLY header required")
            lines.append(line.decode("ascii").strip())
            if lines[-1] == "end_header":
                break
        expected = ["ply", "format binary_little_endian 1.0",
                    "comment original global vertex order; no repairs",
                    f"element vertex {vertices}", "property float x", "property float y", "property float z",
                    f"element face {faces}", "property list uchar int vertex_indices", "end_header"]
        require(lines == expected, "Original PLY header differs")
        offset = handle.tell()
    require(Path(path).stat().st_size == offset + vertices*12 + faces*13, "PLY byte length differs")
    vertex = np.memmap(path, mode="r", offset=offset, dtype="<f4", shape=(vertices, 3))
    face = np.memmap(path, mode="r", offset=offset+vertices*12,
                     dtype=[("count", "u1"), ("indices", "<i4", (3,))], shape=(faces,))
    for start in range(0, len(face), BATCH):
        require(np.all(face["count"][start:start+BATCH] == 3), "Nontriangle PLY")
    return vertex, face


def formula_vertices(coordinates, features):
    """Independent source-equivalent comparison formula, no extraction/topology."""
    x = features[:, :3]
    z = np.exp(-np.abs(x))
    sigmoid = np.where(x >= 0, np.float32(1)/(np.float32(1)+z), z/(np.float32(1)+z))
    dual = np.float32(2)*sigmoid - np.float32(.5)
    return (coordinates[:, 1:].astype(np.float32)+dual)*np.float32(1/512) - np.float32(.5)


def check_rows(geometry_coords, material_coords, geometry_features, stored_vertices, ply_vertices):
    equal_bits(geometry_coords, material_coords, "Full geometry/material coordinates")
    require(geometry_coords.shape[1:] == (4,), "Coordinate width differs")
    require(geometry_features.shape == (len(geometry_coords), 7), "Geometry feature shape differs")
    require(stored_vertices.shape == ply_vertices.shape == (len(geometry_coords), 3), "Vertex shape differs")
    for start in range(0, len(geometry_coords), BATCH):
        c = geometry_coords[start:start+BATCH]
        f = geometry_features[start:start+BATCH]
        require(np.all(c[:, 0] == 0) and np.all(c[:, 1:] >= 0) and np.all(c[:, 1:] < 512), "Coordinate grid differs")
        require(np.isfinite(f).all(), "Nonfinite source geometry")
        v = formula_vertices(c, f)
        equal_bits(v, stored_vertices[start:start+BATCH], "Formula/stored vertices")
        equal_bits(v, ply_vertices[start:start+BATCH], "Formula/PLY vertices")


def channel_stats(features):
    require(features.dtype == np.float32 and features.ndim == 2 and features.shape[1] == 6,
            "Original material F32[N,6] required")
    result = {}
    for index, name in enumerate(CHANNELS):
        low, high, total, total_squared = np.inf, -np.inf, 0.0, 0.0
        below_zero = above_one = exact_one = below_half = below_99 = 0
        for start in range(0, len(features), BATCH):
            values = features[start:start+BATCH, index]
            require(np.isfinite(values).all(), "Nonfinite material channel")
            low, high = min(low, float(values.min())), max(high, float(values.max()))
            total += float(values.sum(dtype=np.float64))
            total_squared += float(np.square(values.astype(np.float64)).sum())
            below_zero += int(np.count_nonzero(values < 0))
            above_one += int(np.count_nonzero(values > 1))
            exact_one += int(np.count_nonzero(values == 1))
            below_half += int(np.count_nonzero(values < .5))
            below_99 += int(np.count_nonzero(values < .99))
        mean = total / len(features)
        result[name] = dict(column=index, min=low, max=high, mean=mean,
                            standard_deviation=max(0, total_squared/len(features)-mean*mean)**.5,
                            negative_count=below_zero, above_one_count=above_one,
                            outside_0_1_count=below_zero+above_one, exact_one_count=exact_one,
                            below_half_count=below_half, below_0_99_count=below_99,
                            sha256=array_digest(features[:, index]))
    return result


def runtime_guard(event, args):
    if event == "import" and str(args[0]).split(".")[0] in {
        "torch", "triton", "flex_gemm", "o_voxel", "trellis2", "cupy", "bpy"
    }:
        raise RuntimeError("CPU validator forbids model/GPU/renderer imports")
    if event.startswith(("subprocess.", "socket.", "os.spawn", "os.exec")) or event == "os.system":
        raise RuntimeError("CPU validator forbids process and network execution")


def verify():
    began = time.monotonic()
    sys.addaudithook(runtime_guard)
    before = {}
    for name, (path, sha) in PINS.items():
        stat = path.stat()
        require(not path.is_symlink() and stat.st_nlink == 1 and digest(path) == sha,
                "Pinned input differs: " + name)
        before[name] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
    subdivision_counts = (4430, 21391, 97313, 438684)
    schema = {"coordinates": ("I32", (N, 4)), "features": ("F32", (N, 7))}
    for index, count in enumerate(subdivision_counts):
        schema[f"subdivision_{index}_coordinates"] = ("I32", (count, 4))
        schema[f"subdivision_{index}_logits"] = ("BF16", (count, 8))
    geometry, _ = read_safetensors(GEOMETRY, schema)
    material, material_header_metadata = read_safetensors(FIELD, {
        "coordinates": ("I32", (N, 4)), "raw": ("F32", (N, 6)), "features": ("F32", (N, 6))})
    metadata = read_json(PINS["material_metadata"][0])
    receipt = read_json(PINS["extraction_receipt"][0])
    require(metadata["channels"] == list(CHANNELS), "Original channel semantics differ")
    require(metadata["artifact_sha256"] == PINS["material"][1]
            and metadata["provenance"]["geometry_sha256"] == PINS["geometry"][1]
            and metadata["provenance"]["forward_calls"] == 1
            and metadata["provenance"]["mode"] == "production", "Genuine material provenance differs")
    require(receipt["input_sha256"] == PINS["geometry"][1]
            and receipt["vertices"] == N and receipt["triangles"] == F
            and receipt["PLY_readback_exact"] is True
            and receipt["topology_repair"] is False
            and receipt["conversion"]["global_original_indices"] is True, "Mesh provenance differs")
    vertices = np.load(PINS["vertices"][0], mmap_mode="r", allow_pickle=False)
    faces = np.load(PINS["faces"][0], mmap_mode="r", allow_pickle=False)
    require(vertices.dtype == np.float32 and faces.dtype == np.int32 and faces.shape == (F, 3), "NPY schema differs")
    ply_vertices, ply_faces = read_ply(PINS["mesh"][0], N, F)
    check_rows(geometry["coordinates"], material["coordinates"], geometry["features"], vertices, ply_vertices)
    equal_bits(faces, ply_faces["indices"], "Full NPY/PLY face index payload")
    for start in range(0, F, BATCH):
        f = faces[start:start+BATCH]
        require(np.all(f >= 0) and np.all(f < N), "Face index out of range")
    for start in range(0, N, BATCH):
        raw = material["raw"][start:start+BATCH]
        require(np.isfinite(raw).all(), "Nonfinite raw features")
        expected = raw*np.float32(.5) + np.float32(.5)
        equal_bits(expected, material["features"][start:start+BATCH], "Original unclamped material transform")
    tensor_hashes = {name: array_digest(value) for name, value in material.items()}
    for name, sha in tensor_hashes.items():
        require(sha == metadata["tensors"][name]["sha256"], "Material sidecar payload hash differs")
    coordinates_sha = array_digest(geometry["coordinates"])
    require(coordinates_sha == tensor_hashes["coordinates"] == "690670fe58a7021c51fef9fad6457ad8e66715d13a88239adabdffa8a05d9976", "Pinned full coordinate bytes differ")
    stats = {name: channel_stats(material[name]) for name in ("raw", "features")}
    for field_name, channels in stats.items():
        for channel, values in channels.items():
            previous = metadata["channel_statistics"][field_name][channel]
            require(values["min"] == previous["min"] and values["max"] == previous["max"]
                    and abs(values["mean"]-previous["mean"]) <= 1e-12, "Channel sidecar statistics differ")
    identities = {}
    for name, (path, sha) in PINS.items():
        stat = path.stat()
        require((stat.st_size, stat.st_mtime_ns, stat.st_ino) == before[name] and digest(path) == sha,
                "Input changed while validating: " + name)
        identities[name] = dict(bytes=stat.st_size, sha256=sha, full_hash_verified_before_and_after=True)
    report = {
        "schema": "modly.amd.brick-pbr-row-correspondence.v1",
        "classification": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution": {"CPU_only": True, "stdlib_and_numpy_only": True, "numpy_version": np.__version__,
                      "read_only_memmaps": True, "comparison_batch_rows": BATCH,
                      "model_GPU_Blender_imports_forbidden_by_audit_hook": True,
                      "process_network_execution_forbidden_by_audit_hook": True,
                      "geometry_generation": False, "topology_reconstruction": False,
                      "private_array_or_image_writes": False, "inputs_unchanged": True},
        "input_identities": identities,
        "full_correspondence": {
            "vertices": N, "triangles": F, "geometry_material_coordinates_dtype_shape": ["I32", [N, 4]],
            "coordinate_payload_sha256": coordinates_sha, "all_coordinate_row_bytes_and_order_equal": True,
            "all_PLY_and_NPY_vertices_bitwise_equal_independent_original_formula": True,
            "all_PLY_and_NPY_face_indices_bitwise_equal": True,
            "vertex_payload_sha256": array_digest(vertices), "face_payload_sha256": array_digest(faces),
            "formula": "x=features[:,:3]; z=exp(-abs(x)); sigmoid=where(x>=0,F32(1)/(F32(1)+z),z/(F32(1)+z)); dual=F32(2)*sigmoid-F32(.5); vertex=(coordinates[:,1:].astype(F32)+dual)*F32(1/512)-F32(.5)",
            "axis_order": "original XYZ; no permutation", "matching": "same original row index; no sort, resample or nearest-neighbour inference",
            "source_review": "amd_generation_mesh.extract passes original coordinates[:,1:] and features to rowwise converter; converter.faces maps lookup hits through order[pos] back to original global indices; write_ply preserves all vertex/index bytes; source hashes pinned above",
            "faces_recomputed": False,
        },
        "material_contract": {
            "channels": list(CHANNELS), "raw_and_features_dtype_shape": ["F32", [N, 6]],
            "full_raw_to_features_transform_bitwise_verified": "features=raw*F32(.5)+F32(.5), no clamp",
            "material_tensor_sha256": tensor_hashes, "saved_field_unchanged": True,
            "channel_statistics": stats,
            "base_color_encoding": {
                "finding": "Encoded display RGB, not already scene-linear BSDF RGB",
                "source_evidence": [
                    {"source": "upstream_pipeline", "lines": [110, 114, 1277, 1280, 3128, 3145, 3149, 3153], "finding": "RGB occupies columns 0:3; metallic 3, roughness 4, alpha 5. Decode uses *0.5+0.5. RGB is directly quantized into PIL baseColorTexture for PBRMaterial without an encoding conversion."},
                    {"source": "upstream_pbr_renderer", "lines": [431, 432, 433, 434], "finding": "Native PBR shading explicitly uses clamp(basecolor,0,1)**2.2 and clamps metallic, roughness and alpha to [0,1]."}
                ],
                "preview_policy": "If matching pinned native renderer, shader-only RGB clamp then power 2.2 to scene-linear; metallic/roughness/alpha shader-only clamp [0,1]. Retain untouched original six-channel values.",
                "exact_IEC_sRGB_transfer_claimed": False,
                "direct_row_preview_is_original_trilinear_bake": False,
            },
        },
        "conclusion": "Every original mesh vertex i has exactly the original geometry coordinate row i, which is byte-identical to material coordinate row i. Learned material feature row i therefore has a proven direct correspondence to original vertex i. All original faces and inputs remain unchanged.",
        "elapsed_seconds": time.monotonic()-began, "blockers": [],
    }
    # Release all mappings before returning the source-safe summary.
    for value in [*geometry.values(), *material.values(), vertices, faces, ply_vertices, ply_faces]:
        value._mmap.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Fresh evidence path required")
    report = verify()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"classification": report["classification"], "vertices": N,
                      "triangles": F, "elapsed_seconds": report["elapsed_seconds"]}))
