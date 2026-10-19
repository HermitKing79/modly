"""Owned, gated, CPU-only diagnosis of the unchanged original brick mesh.

The host must attach this process to its bounded Windows Job before sending
GO. No NumPy, SciPy, PIL, topology, or projection import occurs before that
gate. This worker writes only new diagnostic JSON/NPY data below its private P;
it never repairs, welds, exports, renders, starts a process, or uses a GPU.
"""
import sys
sys.dont_write_bytecode = True

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import time


E = Path(__file__).resolve().parent
ROOT = E.parents[2]
ARTIFACTS = ROOT.parent / "artifacts"
P = ARTIFACTS / "amd-brick-defect-diagnosis-1d1c1a6/analysis"
OLD_EV = ARTIFACTS / "amd-brick-pbr-exposure-b0b7be1/renders"
MESH = ARTIFACTS / "amd-brick-cpu-mesh-8473579/mesh"
N, F = 1_957_891, 4_173_584
MAPPING = ROOT / "docs/compatibility/amd-brick-pbr-preview-evidence/mapping-verification.json"
MAPPING_SHA = "84912427ddff6d470ce96215344118b057ab357f3a41ff762612a367549d0eb4"
PNG_SHA = "74d1655d53d5fe7d262756c7e23c0a695aa42b7ee3c43e075007a59fc3bf7f54"
PINS = {
    "mesh": (MESH / "mesh.ply", "ec9736e2510c26303196c4e3fc2045a1a1d36401b2576ee68f102ee29b551ffd"),
    "vertices": (MESH / "vertices.npy", "018edd85646f9b504ec52ceb311f69e0c7610e4d1ab306f41c7e73a7a77d3252"),
    "faces": (MESH / "faces.npy", "9314a921741bab5c3d87348b4ea26488ee1ab13201b33d67252d6d272997ab1d"),
    "geometry": (ARTIFACTS / "amd-brick-neural-completion-c9447dd/neural-output/shape-decoded.safetensors",
                 "d4de05fc82315e44a739c9815a07a531555e108af05c619ea0426d0c630ef861"),
    "material": (Path("C:/Users/wattz/Documents/ModlyArtifacts/amd-brick-pbr-qualified-a064378/field/texture-field.safetensors"),
                 "8cd735630149e28829efe686d30f1c7c255dde76748dc3d525236fb5c2c81394"),
    "mapping_receipt": (MAPPING, MAPPING_SHA),
    "image_ev2": (OLD_EV / "view-pos-x-ev2.png", PNG_SHA),
    "render_metadata": (OLD_EV / "render-metadata.json", "2e09ea2c4ba104f37133e1322759ce9de3fe5e92a5aeac798e379d8bf1685464"),
}
ROIS = {"A_brick_detail": (470, 360, 660, 590), "B_vines": (300, 300, 480, 500)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def read_json(path):
    require(Path(path).stat().st_size <= 2 * 1024**2, "Bounded JSON required")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=object_pairs)


def identity(path):
    path = Path(path)
    stat = path.stat()
    require(path.is_file() and not path.is_symlink() and stat.st_nlink == 1,
            "Plain single-link input required: " + str(path))
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "inode": stat.st_ino, "sha256": digest(path)}


def runtime_guard(event, args):
    if event == "import" and str(args[0]).split(".")[0] in {
        "torch", "triton", "flex_gemm", "o_voxel", "trellis2", "cupy", "bpy",
        "tensorflow", "jax", "numba", "pycuda", "moderngl",
    }:
        raise RuntimeError("CPU diagnosis forbids model/GPU/renderer imports")
    if (event.startswith("socket.") and event != "socket.gethostname") or event.startswith(("subprocess.", "os.spawn", "os.exec")) or event in {
        "os.system", "os.posix_spawn", "os.fork", "os.forkpty",
    }:
        raise RuntimeError("CPU diagnosis forbids process and network execution")
    target = None
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        mode, flags = args[1], args[2]
        if ((isinstance(mode, str) and any(c in mode for c in "wax+")) or
                (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))):
            target = args[0]
    elif event in {"os.mkdir", "os.remove", "os.rmdir"}:
        target = args[0]
    elif event in {"os.rename", "os.link", "os.symlink"}:
        raise RuntimeError("CPU diagnosis forbids file replacement/link operations")
    if target is not None:
        target = Path(os.fsdecode(target)).resolve()
        require(target == P or P in target.parents, "Writes must stay in the owned diagnostic output directory")


def material_features(path, np):
    """Validate the entire bounded material header; map only full-row features."""
    size = Path(path).stat().st_size
    require(8 < size <= 256 * 1024**2, "Bounded material required")
    with Path(path).open("rb") as stream:
        length_bytes = stream.read(8)
        require(len(length_bytes) == 8, "Complete header prefix required")
        length = struct.unpack("<Q", length_bytes)[0]
        require(2 <= length <= 65536 and 8 + length < size, "Bounded complete material header required")
        header = json.loads(stream.read(length), object_pairs_hook=object_pairs)
    schema = {"coordinates": ("I32", [N, 4], 4), "raw": ("F32", [N, 6], 4),
              "features": ("F32", [N, 6], 4)}
    require(set(header) - {"__metadata__"} == set(schema), "Original material tensor names differ")
    ranges = []
    for name, (dtype, shape, itemsize) in schema.items():
        row = header[name]
        require(set(row) == {"dtype", "shape", "data_offsets"} and row["dtype"] == dtype and
                row["shape"] == shape, "Original material tensor schema differs")
        offsets = row["data_offsets"]
        require(isinstance(offsets, list) and len(offsets) == 2 and
                all(type(x) is int for x in offsets), "Invalid material offsets")
        start, end = offsets
        require(0 <= start < end <= size - 8 - length and end - start == N * shape[1] * itemsize,
                "Material payload size differs")
        ranges.append((start, end))
    previous = 0
    for start, end in sorted(ranges):
        require(start == previous, "Material payload gap/overlap")
        previous = end
    require(previous == size - 8 - length, "Trailing material payload")
    values = np.memmap(path, mode="r", offset=8 + length + header["features"]["data_offsets"][0],
                       dtype="<f4", shape=(N, 6))
    require(not values.flags.writeable, "Material mapping must be read only")
    return values


def select_pixel_groups(rgb, np, ndimage):
    """Deterministic image-only nominations; no nomination implies a mesh hole."""
    luma = rgb.astype(np.float64) @ np.array([.2126, .7152, .0722])
    groups, selection = [], {}
    for roi_name, box in ROIS.items():
        left, top, right, bottom = box
        crop = luma[top:bottom, left:right]
        dark = crop < 30.
        labels, count = ndimage.label(dark, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        border_labels = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))).tolist())
        candidates = []
        for label in np.flatnonzero((sizes >= 2) & (sizes <= 40)):
            if label == 0 or int(label) in border_labels:
                continue
            yy, xx = np.nonzero(labels == label)
            center = np.array([xx.mean(), yy.mean()])
            # The nominated center is an actual dark pixel, nearest the centroid.
            order = np.lexsort((xx, yy, (xx - center[0]) ** 2 + (yy - center[1]) ** 2))
            i = int(order[0])
            candidates.append({"pixel": [int(xx[i]) + left, int(yy[i]) + top],
                               "area": int(sizes[label]), "label": int(label)})
        candidates.sort(key=lambda row: (-row["area"], row["pixel"][1], row["pixel"][0]))
        chosen = []
        for candidate in candidates:
            if all(sum((a - b) ** 2 for a, b in zip(candidate["pixel"], old["pixel"])) >= 18 ** 2
                   for old in chosen):
                chosen.append(candidate)
                if len(chosen) == 3:
                    break
        for candidate in chosen:
            groups.append({"roi": roi_name, "kind": "isolated_dark_image_blob", **candidate})
        # A wide dark image patch is only a broad-dark control until traced.
        distance = ndimage.distance_transform_edt(dark)
        margin = np.zeros_like(dark)
        margin[4:-4, 4:-4] = True
        broad = (distance >= 3.) & margin
        control = None
        if broad.any():
            yy, xx = np.nonzero(broad)
            order = np.lexsort((xx, yy, -distance[yy, xx]))
            i = int(order[0])
            control = {"pixel": [int(xx[i]) + left, int(yy[i]) + top],
                       "kind": "broad_dark_image_control", "dark_interior_radius_px": float(distance[yy[i], xx[i]])}
        else:
            colored = (crop >= 60.) & (crop <= 230.) & margin
            if colored.any():
                yy, xx = np.nonzero(colored)
                center = np.array([(right - left) / 2, (bottom - top) / 2])
                order = np.lexsort((xx, yy, (xx - center[0]) ** 2 + (yy - center[1]) ** 2))
                i = int(order[0])
                control = {"pixel": [int(xx[i]) + left, int(yy[i]) + top],
                           "kind": "mid_luma_image_control"}
        if control is not None:
            groups.append({"roi": roi_name, **control})
        selection[roi_name] = {"box": list(box), "dark_connected_components": int(count),
                               "eligible_isolated_blob_count": len(candidates),
                               "chosen_isolated_blob_count": len(chosen), "control_available": control is not None}
    require(len(groups) <= 8, "Image query group budget exceeded")
    queries = []
    for group_id, group in enumerate(groups):
        group["group_id"] = group_id
        for offset in ((0, 0), (-3, 0), (3, 0), (0, -3), (0, 3)):
            x, y = (group["pixel"][i] + offset[i] for i in range(2))
            queries.append({"group_id": group_id, "roi": group["roi"], "kind": group["kind"],
                            "offset_from_center": list(offset), "pixel": [x, y],
                            "image_RGB": rgb[y, x].tolist(), "image_luma": float(luma[y, x])})
    return groups, queries, selection


def main():
    print("BRICK_DIAGNOSIS_WAITING_GO", flush=True)
    require(sys.stdin.buffer.readline(64) == b"GO\n", "Owned host GO authorization required")
    began = time.monotonic()
    sys.addaudithook(runtime_guard)
    import numpy as np
    import scipy
    from scipy import ndimage
    from PIL import Image
    sys.path.insert(0, str(E))  # -I intentionally omits the script directory.
    import topology
    import projection

    def check_deadline(stage):
        require(time.monotonic() - began < 170, "Worker deadline reached before " + stage)

    require(not (P / "diagnostics.json").exists(), "Refusing to replace existing diagnostics")
    P.mkdir(parents=True, exist_ok=True)
    before = {name: identity(path) for name, (path, _) in PINS.items()}
    for name, (_, expected) in PINS.items():
        require(expected is None or before[name]["sha256"] == expected, "Pinned input differs: " + name)
    sources_before = {path.name: identity(path) for path in (Path(__file__), E / "topology.py", E / "projection.py")}
    mapping = read_json(MAPPING)
    require(mapping["classification"] == "PASS", "Authenticated row mapping must pass")
    for name in ("mesh", "vertices", "faces", "geometry", "material"):
        require(mapping["input_identities"][name]["sha256"] == before[name]["sha256"],
                "Input identity differs from exact original row mapping: " + name)
    metadata = read_json(PINS["render_metadata"][0])
    require(metadata["classification"] == "PASS" and metadata["mapping_receipt_sha256"] == MAPPING_SHA and
            metadata["source_PLY_sha256"] == PINS["mesh"][1] and
            metadata["material_field_sha256"] == PINS["material"][1] and
            metadata["original_direct_row_correspondence_verified"] is True,
            "Authenticated exposure metadata differs")
    camera = next(view for view in metadata["views"] if view["name"] == "pos-x-ev2")
    require(camera["sha256"] == PNG_SHA and camera["exposure_EV"] == 2 and
            camera["width"] == camera["height"] == 1024 and camera["camera_type"] == "ORTHO" and
            camera["camera_right"] == [0., 1., 0.] and camera["camera_up"] == [0., 0., 1.] and
            camera["axis_direction"] == [1, 0, 0] and camera["ortho_scale"] == 1.5038471221923828 and
            camera["camera_location"] == [2.797344923019409, -0.0018387436866760254, 0.0007975846529006958],
            "Exact +X EV2 camera contract differs")
    vertices = np.load(PINS["vertices"][0], mmap_mode="r", allow_pickle=False, max_header_size=4096)
    faces = np.load(PINS["faces"][0], mmap_mode="r", allow_pickle=False, max_header_size=4096)
    require(vertices.shape == (N, 3) and vertices.dtype == np.dtype("<f4") and
            faces.shape == (F, 3) and faces.dtype == np.dtype("<i4") and
            not vertices.flags.writeable and not faces.flags.writeable, "Original readonly NPY schema differs")
    features = material_features(PINS["material"][0], np)
    with Image.open(PINS["image_ev2"][0]) as image:
        require(image.format == "PNG" and image.size == (1024, 1024) and image.mode in {"RGB", "RGBA"},
                "Bounded authentic exposure PNG required")
        image.load()
        if image.mode == "RGBA":
            require(image.getextrema()[3] == (255, 255), "Opaque exposure PNG required")
        rgb = np.asarray(image.convert("RGB")).copy()
    groups, queries, selection = select_pixel_groups(rgb, np, ndimage)
    check_deadline("global topology")
    topology_result = topology.analyze(vertices, faces, P / "cache")
    check_deadline("orthographic projection")
    projected = projection.project(vertices, camera, 1024, 1024)
    bounds = projection.build_triangle_bounds(faces, projected)
    cache = topology_result["private_cache"]
    keys = np.load(cache["edge_keys"], mmap_mode="r", allow_pickle=False)
    incidence = np.load(cache["incidence"], mmap_mode="r", allow_pickle=False)
    components = np.load(cache["vertex_components"], mmap_mode="r", allow_pickle=False)
    vertex_sizes = np.bincount(components)
    face_sizes = np.bincount(components[faces[:, 0]], minlength=len(vertex_sizes))
    check_deadline("selected center rays")
    rays = projection.sample_rays(vertices, faces, projected, [q["pixel"] for q in queries], features,
                                  bounds=bounds, edge_incidence=(keys, incidence, N)) if queries else []
    for query, ray in zip(queries, rays):
        query["ray"] = ray
        if ray["hit"]:
            face = faces[ray["face_id"]]
            component = int(components[face[0]])
            require(np.all(components[face] == component), "Hit face component inconsistency")
            ray.update(face_vertex_ids=face.tolist(), component_label=component,
                       component_vertices=int(vertex_sizes[component]), component_faces=int(face_sizes[component]))
    edge_samples = []
    for roi_name, box in ROIS.items():
        for edge_class, mask in (("boundary", incidence == 1), ("nonmanifold", incidence > 2)):
            check_deadline("edge midpoint visibility")
            chosen_keys = keys[mask]
            pairs = np.column_stack((chosen_keys // np.uint64(N), chosen_keys % np.uint64(N)))
            rows = projection.visible_edge_samples(vertices, faces, projected, pairs, incidence[mask], box,
                                                      max_edges=4, bounds=bounds)
            for row in rows:
                row.update(roi=roi_name, edge_class=edge_class,
                           world_endpoints=np.asarray(vertices[row["vertices"]], dtype=np.float64).tolist(),
                           component_label=int(components[row["vertices"][0]]),
                           boundary_loop_assessment="not_assessed",
                           boundary_loop_reason="Midpoint visibility and edge incidence do not establish a degree-two closed loop, local manifoldness, interior occlusion, or repair safety.")
            edge_samples.extend(rows)
    check_deadline("final input authentication")
    after = {name: identity(path) for name, (path, _) in PINS.items()}
    sources_after = {path.name: identity(path) for path in (Path(__file__), E / "topology.py", E / "projection.py")}
    require(before == after and sources_before == sources_after, "Inputs or diagnostic sources changed during diagnosis")
    result = {
        "schema": "modly.amd.brick-defect-diagnosis.v1", "classification": "PASS_WITH_SCOPE_LIMITS",
        "created_utc": datetime.now(timezone.utc).isoformat(), "mission_base": "1d1c1a6b54050200fca870b49c1b26bb51338c6a",
        "input_identities_before": before, "input_identities_after": after, "inputs_unchanged": True,
        "source_identities_before": sources_before, "source_identities_after": sources_after,
        "camera": camera, "topology": topology_result,
        "selection": {"criteria": "PNG luma=0.2126R+0.7152G+0.0722B, dark<30; 8-connected components area2..40, crop-border components excluded; largest then raster order; chosen centers at least18px apart. Up to3 blobs per ROI plus a broad-dark interior-radius>=3px image control, falling back to a luma60..230 control nearest crop center. Each nomination has center and four axis neighbors at3px. Image candidates are not preclassified as mesh holes.",
                      "rois": selection, "groups": groups, "query_count": len(queries), "max_query_count": 40},
        "pixel_queries": queries, "edge_midpoint_samples": edge_samples,
        "edge_overlay_scope": "At most4 visible midpoint markers per incidence class per ROI. Occluded probes are retained for audit; no full edge visibility or dense full-mesh overlay is asserted.",
        "scope_limits": ["Exact center rays against opaque original triangles; no antialiasing or pixel coverage reconstruction.",
                         "Raw and per-vertex gamma22 linear RGB are material evidence, not the shaded/exposed PNG color.",
                         "Normals and depth can support a surface-orientation/occlusion hypothesis; no area-light shadow rays or relighting were performed.",
                         "Empty rays with projected-degenerate candidate ambiguities require caution; no repair eligibility follows from a dark image point or an incidence-1 edge.",
                         "No boundary loop, safe interior, repair, welding, simplification, model execution, mesh export, or mesh mutation was performed."],
        "execution": {"CPU_only": True, "readonly_original_memmaps": True,
                      "gate": "GO received before large imports, after host Job attachment",
                      "model_GPU_Blender_imports_forbidden_by_audit_hook": True,
                      "process_network_execution_forbidden_by_audit_hook": True,
                      "writes_restricted_to_owned_P": True, "numpy": np.__version__, "scipy": scipy.__version__,
                      "elapsed_seconds": round(time.monotonic() - began, 6), "worker_deadline_seconds": 170,
                      "projection_bounds_bytes": int(bounds.minimum.nbytes + bounds.maximum.nbytes),
                      "source_arrays_mutated": False},
    }
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with (P / "diagnostics.json").open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(json.dumps({"classification": result["classification"], "path": str(P / "diagnostics.json"),
                      "sha256": digest(P / "diagnostics.json"), "elapsed_seconds": result["execution"]["elapsed_seconds"],
                      "pixel_query_count": len(queries), "edge_probe_count": len(edge_samples)}), flush=True)


if __name__ == "__main__":
    main()
