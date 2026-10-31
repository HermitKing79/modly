"""Private, bounded CPU oracle for Phase 22 actual S&box screenshot sampling.

No engine, Blender, network, project mutation, or screenshot fitting is performed.
Expected geometry/UVs come only from the SHA-authenticated original GLB. Actual
screenshots are read only after all reference masks have been constructed.

Requirements already installed locally: Python 3.12, NumPy, Pillow.
Run tests: python -B test_independent_sampling_oracle.py
Run proof: python -B independent_sampling_oracle.py --metadata captures.json \
    --texture diagnostic.png --glb ORIGINAL.glb --report NEW-report.json
Generate the required fixed sentinel points before capture with --sentinel-template
NEW.json (no --metadata or --texture needed; fixed approved prop transform is used).
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np
from PIL import Image


EXPECTED_GLB_SHA256 = "afc880ff5ec2e0c1a9a6977999fa633506b68915c73eddb2c543ee17512c1201"
EVIDENCE_DIR = Path(__file__).resolve().parent
SCHEMA = "modly.phase22.sampling-captures.v2"
WIDTH, HEIGHT, TEXTURE_SIZE = 1280, 960, 1024
UV_UNCERTAINTY = .0005
GEOMETRY_EDGE_EXCLUSION_PIXELS = 2
MAX_TEXTURE_MARGIN_TEXELS = 32
EXPECTED_PROP_POSITION = np.array([0., 0., 15.892192])
EXPECTED_PROP_ROTATION_MATRIX = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
CAMERA_POSITIONS = {
    "three-quarter": [65., -75., 55.],
    "rear-three-quarter": [-65., 75., 52.],
    "side": [0., -95., 32.],
    "top": [32., -38., 110.],
    "underneath": [-60., -55., -35.],
}
CAMERA_TARGET = np.array([0., 0., 15.891])
CONTROL_NAMES = ("u_flip", "v_flip", "uv_stretch_offset", "wrong_pixels_rotate_clockwise")
GATES = {
    "projection_max_error_pixels": .05,
    "minimum_eligible_pixels_per_view": 5000,
    "minimum_faces_with_3_samples_per_view": 100,
    "minimum_faces_with_3_samples_union": 300,
    "minimum_nonblack_classes_with_100_samples_union": 6,
    "minimum_view_agreement": .99,
    "minimum_class_agreement_when_100_samples": .98,
    "minimum_face_agreement_when_20_samples": .95,
    "minimum_nonseamed_edge_pairs_union": 30,
    "minimum_nonseamed_edge_pair_agreement": .99,
    "minimum_control_reference_hamming": .20,
    "maximum_negative_control_agreement": .80,
    "minimum_positive_negative_gap": .15,
}
CLASSIFIER = {
    "black_if_max_normalized_rgb_less_than": .08,
    "high_channel_if_at_least_fraction_of_max": .65,
    "low_channel_if_at_most_fraction_of_max": .35,
    "ambiguous_actual_pixel": "mismatch; never excluded",
    "bits": "red=1, green=2, blue=4; source channels must be exactly 0 or 255",
}


class OracleError(ValueError):
    pass


def require(condition, message):
    if not bool(condition):
        raise OracleError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_array(value, shape=None, name="array"):
    array = np.asarray(value, dtype=np.float64)
    require(shape is None or array.shape == shape, f"Unexpected {name} shape: {array.shape}")
    require(np.isfinite(array).all(), f"Nonfinite {name}")
    return array


def quaternion_matrix(value):
    x, y, z, w = finite_array(value, (4,), "quaternion")
    require(abs(x*x + y*y + z*z + w*w - 1.) <= 2e-5, "Non-unit quaternion")
    # Normalize serialization roundoff, without fitting any observed image.
    x, y, z, w = np.asarray([x, y, z, w]) / np.linalg.norm([x, y, z, w])
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def read_source_glb(path):
    """Independent bounded GLB accessor reader; no Phase 21 parser is imported."""
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == EXPECTED_GLB_SHA256, "Original GLB SHA-256 mismatch")
    require(len(raw) >= 20, "Truncated GLB")
    magic, version, total = struct.unpack_from("<4sII", raw)
    require((magic, version, total) == (b"glTF", 2, len(raw)), "Invalid GLB header")
    chunks, offset = {}, 12
    while offset < len(raw):
        require(offset + 8 <= len(raw), "Truncated GLB chunk header")
        count, kind = struct.unpack_from("<II", raw, offset)
        require(kind not in chunks and offset+8+count <= len(raw), "Invalid GLB chunk")
        chunks[kind] = raw[offset+8:offset+8+count]
        offset += 8 + count
    doc = json.loads(chunks[0x4E4F534A]); binary = chunks[0x004E4942]
    require(len(doc["nodes"]) == 1 and doc["nodes"][0].get("mesh") == 0, "Unexpected source mesh node")
    require(not any(k in doc["nodes"][0] for k in ("matrix", "translation", "rotation", "scale", "children", "skin")),
            "Unexpected source node transform/hierarchy/skin")
    require(len(doc["meshes"]) == 1 and len(doc["meshes"][0]["primitives"]) == 1, "Unexpected mesh count")
    primitive = doc["meshes"][0]["primitives"][0]
    require(primitive.get("mode", 4) == 4, "Source is not a triangle list")

    def accessor(index):
        item = doc["accessors"][index]
        require("sparse" not in item and not item.get("normalized", False), "Unsupported accessor encoding")
        view = doc["bufferViews"][item["bufferView"]]
        require(view.get("buffer", 0) == 0, "Unexpected external buffer")
        formats = {5121: "B", 5123: "H", 5125: "I", 5126: "f"}
        components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
        require(item["componentType"] in formats and item["type"] in components, "Unsupported accessor type")
        fmt = "<" + formats[item["componentType"]] * components[item["type"]]
        size = struct.calcsize(fmt); stride = view.get("byteStride", size)
        start = view.get("byteOffset", 0) + item.get("byteOffset", 0)
        count = item["count"]
        require(0 < count <= 10000 and stride >= size, "Accessor exceeds proof budget")
        end = start + (count-1)*stride + size
        require(end <= len(binary) and end <= view.get("byteOffset", 0)+view["byteLength"], "Accessor out of bounds")
        return finite_array([struct.unpack_from(fmt, binary, start+i*stride) for i in range(count)])

    attrs = primitive["attributes"]
    positions = accessor(attrs["POSITION"]); uv = accessor(attrs["TEXCOORD_0"])
    raw_indices = accessor(primitive["indices"]).reshape(-1)
    require(positions.shape == (2695, 3) and uv.shape == (2695, 2), "Unexpected source attribute shape")
    require(raw_indices.shape == (3000,) and np.all(raw_indices == np.floor(raw_indices)), "Unexpected indices")
    indices = raw_indices.astype(np.int32)
    require(np.array_equal(np.unique(indices), np.arange(2695)), "Missing/unused source attributes")
    require(len(np.unique(positions, axis=0)) == 502, "Source unique position count differs")
    require(np.all((uv >= 0.) & (uv <= 1.)), "Source UVs outside approved atlas")
    return dict(positions=positions, uv=uv, faces=indices.reshape(-1, 3))


def fixed_world_positions(source_positions):
    p = finite_array(source_positions)
    engine = p[:, [1, 2, 0]] * np.array([1., -1., -1.]) / .0254
    return engine @ EXPECTED_PROP_ROTATION_MATRIX.T + EXPECTED_PROP_POSITION


def sentinel_points(world_positions):
    lo = np.min(world_positions, axis=0); hi = np.max(world_positions, axis=0)
    center = (lo+hi)/2; radius = float(np.max(hi-lo)) / 4
    output = []
    for signs in itertools.product((0, 1), repeat=3):
        output.append(dict(name="bbox-"+"".join(map(str, signs)),
                           world_position=np.where(signs, hi, lo).tolist()))
    output.append(dict(name="center", world_position=center.tolist()))
    for axis in range(3):
        for sign in (-1, 1):
            p = center.copy(); p[axis] += sign*radius
            output.append(dict(name=f"axis-{axis}-{'plus' if sign > 0 else 'minus'}", world_position=p.tolist()))
    return output


def project_points(points, camera, width, height):
    points = finite_array(points)
    d = points - finite_array(camera["position"], (3,), "camera position")
    forward = finite_array(camera["forward"], (3,), "forward")
    left = finite_array(camera["left"], (3,), "left")
    up = finite_array(camera["up"], (3,), "up")
    depth = d @ forward
    tan_half = math.tan(math.radians(float(camera["field_of_view"]))/2)
    require(0 < tan_half < 100, "Invalid horizontal FOV")
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc_x = -(d @ left) / (depth*tan_half)
        ndc_y = (d @ up) * (width/height) / (depth*tan_half)
    return np.column_stack([(ndc_x+1)*width/2, (1-ndc_y)*height/2]), depth


def projection_error(points, screen_normal, camera, width, height):
    predicted, _ = project_points(points, camera, width, height)
    actual = finite_array(screen_normal, predicted.shape, "sentinel screen coordinates") * [width, height]
    return float(np.max(np.linalg.norm(predicted-actual, axis=1)))


def validate_camera(capture, expected_sentinels):
    name = capture["name"]; c = capture["camera"]
    require(name in CAMERA_POSITIONS, f"Unexpected camera name {name}")
    require(capture["width"] == WIDTH and capture["height"] == HEIGHT, "Unexpected capture dimensions")
    p = finite_array(c["world_position"], (3,), "camera world position")
    require(np.max(np.abs(p-CAMERA_POSITIONS[name])) <= 1e-5, f"Camera {name} moved")
    r = quaternion_matrix(c["world_rotation"])
    basis = finite_array([c["basis"][k] for k in ("forward", "left", "up")], (3, 3), "camera basis")
    require(np.max(np.abs(basis-r.T)) <= 1e-5, "Captured camera basis disagrees with quaternion")
    target_direction = CAMERA_TARGET-p; target_direction /= np.linalg.norm(target_direction)
    require(np.max(np.abs(basis[0]-target_direction)) <= 1e-5, "Camera does not face fixed target")
    require(np.max(np.abs(finite_array(c["world_scale"], (3,))-1)) <= 1e-8, "Camera scale differs")
    require(c["fov_axis"] == "Horizontal" and float(c["field_of_view"]) == 42., "Unexpected FOV")
    require(c["orthographic"] is False and c["custom_projection_matrix"] is None, "Unexpected projection override")
    require(c["viewport"] == [0, 0, 1, 1] and c["custom_size"] == [WIDTH, HEIGHT], "Unexpected viewport/custom size")
    require(c["screen_rect"] == [0, 0, WIDTH, HEIGHT], "Camera API aspect differs from capture")
    require(c["enable_post_processing"] is False, "Post processing must be disabled")
    require(c["guid"] and c["enabled"] is True, "Camera identity/state unavailable")
    # Public ProjectionMatrix is recorded, never interpreted through guessed row/column conventions.
    finite_array(c["projection_matrix"], (16,), "public projection matrix")
    v = c["view"]
    require(np.max(np.abs(finite_array(v["position"], (3,))-p)) <= 1e-5, "Unexpected composed view position")
    require(np.max(np.abs(quaternion_matrix(v["rotation"])-r)) <= 1e-5, "Unexpected composed view rotation")
    require(v["field_of_view"] == c["field_of_view"] and v["z_near"] == c["z_near"] and v["z_far"] == c["z_far"],
            "Unexpected composed view lens")
    require(0 < c["z_near"] < c["z_far"], "Invalid camera clip distances")
    cam = dict(position=p, forward=basis[0], left=basis[1], up=basis[2], field_of_view=c["field_of_view"],
               z_near=c["z_near"], z_far=c["z_far"])
    sentinels = c["sentinels"]
    require(len(sentinels) == 15 and len({s["name"] for s in sentinels}) == 15, "Expected all 15 fixed sentinels")
    actual_by_name = {s["name"]: s for s in sentinels}
    points, screens = [], []
    for expected in expected_sentinels:
        require(expected["name"] in actual_by_name, "Missing fixed sentinel")
        actual = actual_by_name[expected["name"]]
        require(np.max(np.abs(finite_array(actual["world_position"], (3,))-expected["world_position"])) <= 1e-5,
                "Sentinel point differs from fixed source geometry")
        require(actual["behind"] is False, "Fixed prop sentinel unexpectedly behind camera")
        points.append(actual["world_position"]); screens.append(actual["screen_normal"])
    error = projection_error(points, screens, cam, WIDTH, HEIGHT)
    require(error <= GATES["projection_max_error_pixels"], f"Independent projection error {error:.6g}px exceeds .05px")
    return cam, dict(max_error_pixels=error, sentinel_count=15, method="analytic Source basis vs public PointToScreenNormal")


def rasterize(positions, uv, faces, camera, width, height):
    """Bounded opaque, single-sided perspective rasterizer; pixel centers x+.5/y+.5.

    These fixed proof views contain the entire prop between near/far planes. Clip
    crossings fail explicitly instead of silently dropping or incorrectly clipping.
    """
    require(0 < width <= 4096 and 0 < height <= 4096, "Raster size exceeds budget")
    positions = finite_array(positions); uv = finite_array(uv); faces = np.asarray(faces, dtype=np.int32)
    require(len(faces) <= 1000 and positions.shape[1:] == (3,) and uv.shape == (len(positions), 2), "Raster mesh exceeds budget")
    screen, z = project_points(positions, camera, width, height)
    require(np.all(z > camera["z_near"]) and np.all(z < camera["z_far"]),
            "Geometry crosses near/far clip planes; unsupported fixed proof framing")
    face_image = np.full((height, width), -1, np.int32)
    depth_image = np.full((height, width), np.inf)
    uv_image = np.full((height, width, 2), np.nan)
    edge_distance = np.zeros((height, width), np.float32)
    uv_gradient = np.full((height, width), np.inf, np.float32)
    front_faces = []
    for face_id, indices in enumerate(faces):
        world = positions[indices]
        normal = np.cross(world[1]-world[0], world[2]-world[0])
        if np.dot(normal, np.asarray(camera["position"])-world[0]) <= 0:
            continue
        front_faces.append(face_id)
        a = screen[indices]
        x0, y0 = a[0]; x1, y1 = a[1]; x2, y2 = a[2]
        denominator = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2)
        if abs(denominator) < 1e-12:
            continue
        lo = np.maximum(np.floor(a.min(axis=0)-.5).astype(int), [0, 0])
        hi = np.minimum(np.ceil(a.max(axis=0)-.5).astype(int), [width-1, height-1])
        if np.any(hi < lo):
            continue
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0]+1)+.5, np.arange(lo[1], hi[1]+1)+.5)
        l0 = ((y1-y2)*(xx-x2) + (x2-x1)*(yy-y2)) / denominator
        l1 = ((y2-y0)*(xx-x2) + (x0-x2)*(yy-y2)) / denominator
        bary = np.stack([l0, l1, 1-l0-l1], axis=-1)
        inside = np.all(bary >= -1e-10, axis=-1)
        inv_z = 1/z[indices]
        q = bary @ inv_z
        # Values outside a triangle do not participate in rasterization. Avoid
        # irrelevant zero-denominator extrapolation there without changing any
        # interior barycentric or perspective result.
        q = np.where(inside, q, 1.)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = 1/q
        region = np.s_[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        update = inside & (depth < depth_image[region]-1e-9)
        if not np.any(update):
            continue
        weighted_uv = uv[indices]*inv_z[:, None]
        numerator = bary @ weighted_uv
        interpolated = numerator/q[..., None]
        dx = np.array([(y1-y2)/denominator, (y2-y0)/denominator, 0.]); dx[2] = -dx[:2].sum()
        dy = np.array([(x2-x1)/denominator, (x0-x2)/denominator, 0.]); dy[2] = -dy[:2].sum()
        uv_dx = ((dx @ weighted_uv)*q[..., None]-numerator*(dx @ inv_z))/(q*q)[..., None]
        uv_dy = ((dy @ weighted_uv)*q[..., None]-numerator*(dy @ inv_z))/(q*q)[..., None]
        gradient = np.max(np.abs(uv_dx)+np.abs(uv_dy), axis=-1)
        opposite_lengths = np.array([np.linalg.norm(a[1]-a[2]), np.linalg.norm(a[2]-a[0]), np.linalg.norm(a[0]-a[1])])
        altitudes = abs(denominator)/opposite_lengths
        distances = np.min(bary*altitudes, axis=-1)
        face_image[region][update] = face_id; depth_image[region][update] = depth[update]
        uv_image[region][update] = interpolated[update]
        edge_distance[region][update] = distances[update]
        uv_gradient[region][update] = gradient[update]
    return dict(face=face_image, depth=depth_image, uv=uv_image, edge_distance=edge_distance,
                uv_gradient=uv_gradient, screen_vertices=screen, front_faces=front_faces)


def classify_rgb(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)[..., :3]/255.
    maximum = np.max(rgb, axis=-1)
    high = rgb >= maximum[..., None]*.65
    low = rgb <= maximum[..., None]*.35
    unambiguous = np.all(high | low, axis=-1)
    result = np.sum(high*np.array([1, 2, 4]), axis=-1).astype(np.int8)
    result[~unambiguous] = -1
    result[maximum < .08] = 0
    return result


def sample_classes(classes, uv):
    uv = finite_array(uv)
    height, width = classes.shape
    wrapped = np.mod(uv, 1.)
    x = np.floor(wrapped[..., 0]*width).astype(np.int64) % width
    y = np.floor(wrapped[..., 1]*height).astype(np.int64) % height
    return classes[y, x]


def category_margins(classes, maximum_radius=MAX_TEXTURE_MARGIN_TEXELS):
    """Largest guaranteed same-category square neighborhood, wrap addressing."""
    classes = np.asarray(classes)
    stable = classes >= 0
    margin = np.zeros(classes.shape, np.uint8)
    same_neighbor = [classes == np.roll(np.roll(classes, dy, axis=0), dx, axis=1)
                     for dx, dy in itertools.product((-1, 0, 1), repeat=2) if (dx, dy) != (0, 0)]
    for radius in range(1, maximum_radius+1):
        previous = stable
        stable = previous.copy()
        for (dx, dy), same in zip((p for p in itertools.product((-1, 0, 1), repeat=2) if p != (0, 0)), same_neighbor):
            stable &= same & np.roll(np.roll(previous, dy, axis=0), dx, axis=1)
        margin[stable] = radius
    return margin


def reference_mask(raster, classes, margins):
    """No screenshot/actual color argument exists: eligibility is reference-only."""
    face = raster["face"]; visible = face >= 0
    expected = np.full(face.shape, -1, np.int8)
    expected[visible] = sample_classes(classes, raster["uv"][visible])
    stable_face = visible.copy()
    for dx, dy in itertools.product(range(-2, 3), repeat=2):
        stable_face &= face == np.roll(np.roll(face, dy, axis=0), dx, axis=1)
    stable_face[:2] = False; stable_face[-2:] = False
    stable_face[:, :2] = False; stable_face[:, -2:] = False
    texel_margin = np.zeros(face.shape, np.uint8)
    wrapped = np.mod(raster["uv"][visible], 1.)
    tx = np.floor(wrapped[:, 0]*classes.shape[1]).astype(int) % classes.shape[1]
    ty = np.floor(wrapped[:, 1]*classes.shape[0]).astype(int) % classes.shape[0]
    texel_margin[visible] = margins[ty, tx]
    required = np.full(face.shape, MAX_TEXTURE_MARGIN_TEXELS+1, np.float64)
    required[visible] = np.ceil(UV_UNCERTAINTY*max(classes.shape)
                                + GATES["projection_max_error_pixels"]*raster["uv_gradient"][visible]*max(classes.shape)
                                + 1e-7) + 1
    required[visible] = np.maximum(required[visible], 3)
    eligible = visible & stable_face & (raster["edge_distance"] >= GEOMETRY_EDGE_EXCLUSION_PIXELS)
    eligible &= texel_margin >= required
    return expected, eligible, dict(visible_pixels=int(visible.sum()),
                                  stable_geometry_pixels=int((visible & stable_face).sum()),
                                  eligible_pixels=int(eligible.sum()),
                                  uv_uncertainty=UV_UNCERTAINTY,
                                  maximum_texture_margin_texels=MAX_TEXTURE_MARGIN_TEXELS,
                                  margin_formula="max(3,ceil(.0005*T + .05*max(|dUV/dx|+|dUV/dy|)*T + 1e-7)+1)",
                                  sampler="PointWrap SampleLevel(0)")


def control_expected(classes, uv, name):
    changed = np.array(uv, copy=True)
    if name == "u_flip":
        changed[..., 0] = 1-changed[..., 0]
    elif name == "v_flip":
        changed[..., 1] = 1-changed[..., 1]
    elif name == "uv_stretch_offset":
        changed = changed*np.array([1.2, .8])+np.array([.037, .071])
    elif name == "wrong_pixels_rotate_clockwise":
        return sample_classes(np.rot90(classes, -1), changed)
    else:
        raise OracleError(f"Unknown frozen negative control {name}")
    return sample_classes(classes, changed)


def control_results(classes, uv, actual_classes, mask, positive_expected):
    require(np.any(mask), "No pixels for negative controls")
    output = {}
    for name in CONTROL_NAMES:
        predicted = control_expected(classes, uv[mask], name)
        output[name] = dict(samples=int(mask.sum()),
                            reference_hamming_fraction=float(np.mean(predicted != positive_expected[mask])),
                            actual_agreement=float(np.mean(predicted == actual_classes[mask])))
    return output


def controls_pass(results, positive_accuracy):
    return all(v["reference_hamming_fraction"] >= GATES["minimum_control_reference_hamming"]
               and v["actual_agreement"] <= GATES["maximum_negative_control_agreement"]
               and positive_accuracy-v["actual_agreement"] >= GATES["minimum_positive_negative_gap"]
               for v in results.values())


def score_pixels(expected, actual, mask, face_image):
    correct = expected == actual
    per_class, per_face = {}, {}
    for category in range(8):
        selected = mask & (expected == category)
        count = int(selected.sum())
        per_class[str(category)] = dict(samples=count, correct=int((selected & correct).sum()),
                                        accuracy=float(correct[selected].mean()) if count else None)
    sampled_face_ids = face_image[mask]
    if sampled_face_ids.size:
        require(np.min(sampled_face_ids) >= 0, "Eligible pixel lacks a source face")
        face_counts = np.bincount(sampled_face_ids)
        face_correct = np.bincount(sampled_face_ids, weights=correct[mask].astype(np.int64), minlength=len(face_counts))
        for face in np.flatnonzero(face_counts):
            count = int(face_counts[face]); hits = int(face_correct[face])
            per_face[str(int(face))] = dict(samples=count, correct=hits, accuracy=hits/count)
    count = int(mask.sum())
    return dict(eligible_pixels=count, correct=int((correct & mask).sum()),
                accuracy=float(correct[mask].mean()) if count else 0.,
                actual_unknown_pixels=int((mask & (actual < 0)).sum()),
                tested_faces_3=sum(v["samples"] >= 3 for v in per_face.values()),
                per_class=per_class, per_face=per_face)


def view_failures(stats):
    failures = []
    if stats["eligible_pixels"] < GATES["minimum_eligible_pixels_per_view"]:
        failures.append("Fewer than 5000 eligible reference pixels")
    if stats["tested_faces_3"] < GATES["minimum_faces_with_3_samples_per_view"]:
        failures.append("Fewer than 100 source faces with at least 3 samples")
    if stats["accuracy"] < GATES["minimum_view_agreement"]:
        failures.append("View categorical agreement below 99%")
    for category, value in stats["per_class"].items():
        if value["samples"] >= 100 and value["accuracy"] < GATES["minimum_class_agreement_when_100_samples"]:
            failures.append(f"Color class {category} agreement below 98%")
    for face, value in stats["per_face"].items():
        if value["samples"] >= 20 and value["accuracy"] < GATES["minimum_face_agreement_when_20_samples"]:
            failures.append(f"Source face {face} agreement below 95%")
    return failures


def derive_topology(positions, uv, faces, require_closed=True):
    unique, ids = np.unique(positions, axis=0, return_inverse=True)
    edges = defaultdict(list)
    for face_id, corners in enumerate(faces):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            ia, ib = int(corners[a]), int(corners[b])
            pa, pb = int(ids[ia]), int(ids[ib])
            require(pa != pb, "Degenerate geometric source edge")
            if pa > pb:
                pa, pb, ia, ib = pb, pa, ib, ia
            edges[(pa, pb)].append((face_id, ia, ib))
    nonseamed, seam_count, boundary_count = [], 0, 0
    adjacency = defaultdict(set)
    for edge_id, (key, incidents) in enumerate(sorted(edges.items())):
        require(len(incidents) <= 2, "Nonmanifold source edge")
        if len(incidents) == 1:
            boundary_count += 1
            continue
        a, b = incidents
        continuous = np.max(np.abs(uv[[a[1], a[2]]]-uv[[b[1], b[2]]])) <= 2e-6
        if continuous:
            nonseamed.append(dict(edge_id=edge_id, faces=[a[0], b[0]], vertex_indices=[a[1], a[2]]))
            adjacency[a[0]].add(b[0]); adjacency[b[0]].add(a[0])
        else:
            seam_count += 1
    if require_closed:
        require(boundary_count == 0, "Source has boundary edges")
    visited, charts = set(), 0
    for face_id in range(len(faces)):
        if face_id in visited:
            continue
        charts += 1; pending = [face_id]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current); pending.extend(adjacency[current]-visited)
    return dict(edge_count=len(edges), seam_count=seam_count, boundary_count=boundary_count,
                chart_count=charts, nonseamed=nonseamed, unique_positions=len(unique))


def continuity_witnesses(topology, faces, raster, expected, actual, eligible):
    screen = raster["screen_vertices"]; height, width = expected.shape
    witnesses, tested_edges = [], set()
    for edge in topology["nonseamed"]:
        endpoints = screen[edge["vertex_indices"]]
        segment = endpoints[1]-endpoints[0]; length = np.linalg.norm(segment)
        if length < 12:
            continue
        for t in (.15, .30, .50, .70, .85):
            center = endpoints[0]+t*segment
            samples = []
            for face_id in edge["faces"]:
                centroid = screen[faces[face_id]].mean(axis=0)
                direction = centroid-center; norm = np.linalg.norm(direction)
                if norm < 1e-9:
                    break
                pixel = np.floor(center+direction/norm*3.5).astype(int)
                x, y = int(pixel[0]), int(pixel[1])
                if not (0 <= x < width and 0 <= y < height):
                    break
                offset = np.array([x+.5, y+.5])-endpoints[0]
                perpendicular_distance = abs(segment[0]*offset[1]-segment[1]*offset[0])/length
                if not (2 <= perpendicular_distance <= 6 and eligible[y, x] and raster["face"][y, x] == face_id):
                    break
                samples.append((x, y))
            if len(samples) != 2:
                continue
            (x0, y0), (x1, y1) = samples
            # Eligibility is determined entirely from source predictions on both sides.
            if expected[y0, x0] != expected[y1, x1]:
                continue
            matched = bool(actual[y0, x0] == expected[y0, x0] and actual[y1, x1] == expected[y1, x1])
            witnesses.append(dict(edge_id=edge["edge_id"], t=t, faces=edge["faces"], pixels=samples,
                                  expected_class=int(expected[y0, x0]), matched=matched))
            tested_edges.add(edge["edge_id"])
    return dict(pairs=len(witnesses), matched_pairs=sum(v["matched"] for v in witnesses),
                tested_edges=sorted(tested_edges), witnesses=witnesses)


def read_image(path, required_size=None):
    with Image.open(path) as image:
        require(image.format == "PNG", "Proof images must be original lossless PNGs")
        require(required_size is None or image.size == required_size, "Unexpected PNG dimensions")
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def resolve_input(metadata_dir, value):
    path = Path(value)
    return path if path.is_absolute() else metadata_dir/path


def read_dependency_evidence(path, expected_sha256):
    """Authenticate the actual six-document asset-tool response, retaining logs."""
    path = Path(path)
    require(digest(path) == expected_sha256, "Material dependency evidence SHA-256 mismatch")
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    require(len(lines) >= 6, "Expected the six recorded material dependency JSON documents")
    records = [json.loads(line) for line in lines[:6]]
    require(all(isinstance(record, dict) for record in records), "Unexpected dependency document type")
    require(not any(line.lstrip().startswith("{") for line in lines[6:]),
            "Additional unaccounted dependency JSON document")
    return records, [line for line in lines[6:] if line.strip()]


def validate_dependency_records(records, contract):
    """Name the actual generated texture without inventing a runtime wrapper path."""
    material_path = contract["material_resource"]
    selected = [record for record in records if record.get("Path") == material_path]
    assets = [record for record in selected if "IsCompiled" in record]
    files = [record for record in selected if "UnresolvedReferences" in record]
    references = [record for record in selected if record.get("Relation") == "References"]
    require(len(assets) == len(files) == len(references) == 1,
            "Missing/duplicated compiled material metadata, file metadata or dependency graph")
    asset, file_record, graph = assets[0], files[0], references[0]
    require(asset.get("IsCompiled") is True and asset.get("IsCompiledAndUpToDate") is True
            and asset.get("IsCompileFailed") is False, "Named diagnostic material is not compiled/current/successful")
    require(file_record.get("UnresolvedReferences") == [], "Named diagnostic material has unresolved references")
    for field, suffix in (("SourceFile", material_path), ("CompiledFile", material_path+"_c")):
        value = asset.get(field)
        require(isinstance(value, str) and value.replace("\\", "/").lower().endswith("/"+suffix.lower()),
                "Compiled material source/output file identity differs")
        require(file_record.get(field) == value, "Asset and dependency-file records refer to different material files")
    texture_path = contract["dependency_texture_resource"]
    shader_path = contract["shader_resource"]
    require(isinstance(texture_path, str) and texture_path.endswith(".generated.vtex"),
            "Actual named generated texture dependency is unavailable")
    expected = {(texture_path, "Texture"), (shader_path, "Shader")}
    dependencies = graph.get("Assets", [])
    require(graph.get("Total") == 2 and len(dependencies) == 2
            and {(item.get("Path"), item.get("Type")) for item in dependencies} == expected,
            "Actual dependency graph does not identify exactly the expected generated texture and custom shader")
    return dict(material_resource_path=material_path, shader_resource_path=shader_path,
                texture_resource_path=texture_path, is_compiled=True, is_compiled_and_up_to_date=True,
                is_compile_failed=False, unresolved_references=[], dependency_count=2)


def validate_texture_probe(probe, source_rgba, source_png_sha256, contract, dependency_records):
    """Compare all decoded source bytes with authenticated public GPU readback SHA.

    The probe copies Color32.r/g/b/a in callback order before the span expires.
    Its entire-buffer SHA establishes identity, including orientation, without
    requiring another PNG export or placing a pixel payload in public reports.
    """
    dependency_identity = validate_dependency_records(dependency_records, contract)
    require(probe["Classification"] == "READ_ONLY_PHASE22_COMPILED_SHADER_MATERIAL_TEXTURE_SNAPSHOT"
            and probe["NoSceneOrResourceMutationRequested"] is True, "Unexpected compiled texture probe")
    records = [r for r in probe["Records"] if r["MaterialPath"] == contract["material_resource"]]
    require(len(records) == 1, "Diagnostic material probe record missing/duplicated")
    record = records[0]
    runtime_path = record["TextureResourcePath"]
    require("runtime_texture_resource_path" in contract,
            "Contract must explicitly record actual runtime wrapper path, including null when unavailable")
    require(record["ShaderResourcePath"] == contract["shader_resource"]
            and runtime_path == contract["runtime_texture_resource_path"]
            and record["RuntimeTextureName"] == "g_tDiagnosticColor", "Compiled diagnostic binding identity differs")
    require(runtime_path is None or runtime_path == dependency_identity["texture_resource_path"],
            "Exposed runtime texture path differs from actual named dependency")
    require(record["MaterialValid"] is True and record["MaterialError"] is False
            and record["TextureValid"] is True and record["TextureError"] is False
            and record["IsLoaded"] is True, "Compiled material/texture unavailable or erroneous")
    height, width = source_rgba.shape[:2]
    require(record["Width"] == width and record["Height"] == height and record["Depth"] == 1 and record["Mips"] >= 1,
            "Compiled texture dimensions/mips differ")
    require(record["ImageFormat"] == "RGBA8888", "Compiled texture format differs from reviewed RGBA8888 contract")
    require(record["ImageSourceSha256"] == source_png_sha256, "Compiled texture's copied source PNG identity differs")
    expected_rgba_sha = hashlib.sha256(np.asarray(source_rgba, np.uint8).tobytes(order="C")).hexdigest()
    require(record["MipZeroRgbaByteCount"] == width*height*4, "Compiled mip-zero RGBA byte count differs")
    require(record["MipZeroRgbaSha256"] == expected_rgba_sha, "Entire compiled mip-zero RGBA SHA differs from source pixels")
    samples = record["Samples"]
    required_samples = {(column*(width-1)//4, row*(height-1)//4) for row in range(5) for column in range(5)}
    require(len(samples) == 25 and {(s["X"], s["Y"]) for s in samples} == required_samples,
            "Compiled texture row/column witness grid differs")
    for sample in samples:
        actual = [sample[k] for k in "RGBA"]
        require(actual == source_rgba[sample["Y"], sample["X"]].tolist(), "Compiled texture XY witness differs")
    return dict(decoded_rgba_identical=True, source_rgba_sha256=expected_rgba_sha,
                compiled_mip_zero_rgba_sha256=record["MipZeroRgbaSha256"], rgba_byte_count=width*height*4,
                runtime_texture_resource_path=runtime_path, runtime_wrapper_path_exposed=runtime_path is not None,
                dependency_texture_resource_path=dependency_identity["texture_resource_path"],
                dependency_identity=dependency_identity, width=width, height=height,
                mips=record["Mips"], image_format=record["ImageFormat"], flags=record["Flags"],
                material_source_sha256=record.get("MaterialSourceSha256"),
                shader_source_sha256=record.get("ShaderSourceSha256"),
                xy_witnesses_verified=25)


def authenticate_texture(texture_path, metadata, metadata_dir):
    require(digest(texture_path) == metadata["texture_sha256"], "Diagnostic PNG SHA-256 mismatch")
    image = read_image(texture_path, (TEXTURE_SIZE, TEXTURE_SIZE))
    require(np.all(image[..., 3] == 255), "Diagnostic source PNG must be opaque")
    require(np.all((image[..., :3] == 0) | (image[..., :3] == 255)), "Diagnostic source PNG is not pure binary RGB")
    classes = classify_rgb(image)
    require(np.array_equal(np.unique(classes), np.arange(8)), "Diagnostic PNG must contain all eight categories")
    contract = metadata["sampling_contract"]
    require(contract["sampler"] == "PointWrap" and contract["sample_level"] == 0
            and contract["srgb_read"] is True and contract["texture_format"] == "RGBA8888"
            and contract["opaque"] is True, "Unsupported/unresolved diagnostic sampling contract")
    require(contract["shader_resource"] and contract["material_resource"] and contract["dependency_texture_resource"],
            "Diagnostic binding identity missing")
    require("texture_resource" not in contract, "Ambiguous old texture_resource alias is not accepted in schema v2")
    dependency_reference = metadata["texture_dependencies"]
    dependency_path = resolve_input(metadata_dir, dependency_reference["path"])
    dependency_records, trailing_diagnostics = read_dependency_evidence(dependency_path, dependency_reference["sha256"])
    reference = metadata["texture_probe"]
    path = resolve_input(metadata_dir, reference["path"])
    require(digest(path) == reference["sha256"], "Compiled texture probe JSON SHA mismatch")
    probe = json.loads(path.read_text(encoding="utf-8-sig"))
    identity = validate_texture_probe(probe, image, metadata["texture_sha256"], contract, dependency_records)
    identity.update(source_sha256=digest(texture_path), probe_json_sha256=digest(path),
                    dependency_evidence_sha256=digest(dependency_path),
                    dependency_trailing_diagnostics=trailing_diagnostics,
                    source_categories=list(map(int, np.unique(classes))), sampler=contract)
    return classes, identity


def validate_prop(prop):
    require(prop["model_resource"] == "modly_tests/phase21_static_prop/experimental_static_prop.vmdl", "Wrong model resource")
    require(prop["guid"] and prop["renderer_guid"], "Missing prop/renderer identity")
    require(np.max(np.abs(finite_array(prop["world_position"], (3,))-EXPECTED_PROP_POSITION)) <= 1e-5, "Prop position differs")
    require(np.max(np.abs(quaternion_matrix(prop["world_rotation"])-EXPECTED_PROP_ROTATION_MATRIX)) <= 1e-5, "Prop rotation differs")
    require(np.max(np.abs(finite_array(prop["world_scale"], (3,))-1.)) <= 1e-8, "Prop world scale differs")
    require(prop["tint_rgba"] == [1, 1, 1, 1], "Diagnostic prop tint must be white/opaque")


def run_oracle(glb_path, texture_path, metadata_path):
    metadata_path = Path(metadata_path); metadata_dir = metadata_path.parent
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    require(metadata["schema"] == SCHEMA, "Unsupported capture metadata schema")
    require(metadata["source_glb_sha256"] == EXPECTED_GLB_SHA256, "Metadata source GLB identity mismatch")
    source = read_source_glb(glb_path); validate_prop(metadata["prop"])
    world = fixed_world_positions(source["positions"])
    topology = derive_topology(source["positions"], source["uv"], source["faces"])
    require(topology["seam_count"] == 280 and topology["chart_count"] == 26 and topology["edge_count"] == 1500,
            "Source seam/chart/topology counts differ from frozen proof")
    classes, texture_identity = authenticate_texture(texture_path, metadata, metadata_dir)
    margins = category_margins(classes)
    captures = metadata["captures"]
    require(len(captures) == 5 and {c["name"] for c in captures} == set(CAMERA_POSITIONS), "Expected exactly the five fixed views")
    fixed_sentinels = sentinel_points(world)
    prepared = []
    # Prepare every inclusion mask before opening any actual screenshot pixels.
    for capture in captures:
        camera, projection = validate_camera(capture, fixed_sentinels)
        raster = rasterize(world, source["uv"], source["faces"], camera, WIDTH, HEIGHT)
        expected, eligible, masks = reference_mask(raster, classes, margins)
        prepared.append((capture, projection, raster, expected, eligible, masks))
    views, failures = [], []
    union_visible, union_tested, union_edges = set(), set(), set()
    union_class_samples = np.zeros(8, np.int64); union_class_correct = np.zeros(8, np.int64)
    total_pairs = total_matched_pairs = 0
    aggregate_controls = {name: dict(samples=0, hamming=0., correct=0.) for name in CONTROL_NAMES}
    for capture, projection, raster, expected, eligible, masks in prepared:
        path = resolve_input(metadata_dir, capture["png_path"])
        require(digest(path) == capture["png_sha256"], "Actual engine screenshot SHA mismatch")
        rgba = read_image(path, (WIDTH, HEIGHT))
        require(np.all(rgba[..., 3] == 255), "Unexpected screenshot alpha")
        actual = classify_rgb(rgba)
        score = score_pixels(expected, actual, eligible, raster["face"])
        controls = control_results(classes, raster["uv"], actual, eligible, expected)
        continuity = continuity_witnesses(topology, source["faces"], raster, expected, actual, eligible)
        visible_faces = set(map(int, np.unique(raster["face"][raster["face"] >= 0])))
        tested_faces = {int(k) for k, v in score["per_face"].items() if v["samples"] >= 3}
        union_visible |= visible_faces; union_tested |= tested_faces; union_edges.update(continuity["tested_edges"])
        total_pairs += continuity["pairs"]; total_matched_pairs += continuity["matched_pairs"]
        for category in range(8):
            value = score["per_class"][str(category)]
            union_class_samples[category] += value["samples"]; union_class_correct[category] += value["correct"]
        for name, value in controls.items():
            target = aggregate_controls[name]; n = value["samples"]
            target["samples"] += n; target["hamming"] += value["reference_hamming_fraction"]*n
            target["correct"] += value["actual_agreement"]*n
        view_errors = view_failures(score)
        if not controls_pass(controls, score["accuracy"]):
            view_errors.append("A frozen negative control is unidentifiable or insufficiently rejected")
        failures.extend(f"{capture['name']}: {message}" for message in view_errors)
        views.append(dict(name=capture["name"], screenshot_sha256=capture["png_sha256"], projection=projection,
                          masks=masks, score=score, negative_controls=controls,
                          continuity=continuity, visible_faces=sorted(visible_faces), tested_faces_3=sorted(tested_faces),
                          untested_faces=sorted(set(range(1000))-tested_faces),
                          coverage_reasons=dict(backface_culled_faces=sorted(set(range(1000))-set(raster["front_faces"])),
                              front_but_no_visible_pixel_faces=sorted(set(raster["front_faces"])-visible_faces),
                              visible_but_insufficient_eligible_pixels_faces=sorted(visible_faces-tested_faces)),
                          failures=view_errors))
    if len(union_tested) < GATES["minimum_faces_with_3_samples_union"]:
        failures.append("Union has fewer than 300 source faces with at least 3 samples")
    if int(np.count_nonzero(union_class_samples[1:] >= 100)) < GATES["minimum_nonblack_classes_with_100_samples_union"]:
        failures.append("Union has fewer than six nonblack color classes with at least 100 samples")
    union_classes = {}
    for category, (samples, correct) in enumerate(zip(union_class_samples, union_class_correct)):
        accuracy = float(correct/samples) if samples else None
        union_classes[str(category)] = dict(samples=int(samples), correct=int(correct), accuracy=accuracy)
        if samples >= 100 and accuracy < GATES["minimum_class_agreement_when_100_samples"]:
            failures.append(f"Union class {category} agreement below 98%")
    pair_accuracy = total_matched_pairs/total_pairs if total_pairs else 0.
    if total_pairs < GATES["minimum_nonseamed_edge_pairs_union"]:
        failures.append("Fewer than 30 eligible nonseamed edge pairs across five views")
    if pair_accuracy < GATES["minimum_nonseamed_edge_pair_agreement"]:
        failures.append("Nonseamed edge-pair categorical agreement below 99%")
    total_samples = sum(v["score"]["eligible_pixels"] for v in views)
    total_correct = sum(v["score"]["correct"] for v in views)
    aggregate_controls = {name: dict(samples=v["samples"], reference_hamming_fraction=v["hamming"]/v["samples"],
                                      actual_agreement=v["correct"]/v["samples"]) for name, v in aggregate_controls.items()}
    if not controls_pass(aggregate_controls, total_correct/total_samples):
        failures.append("Union negative-control gate failed")
    return dict(status="PASS" if not failures else "NO-GO", schema="modly.phase22.sampling-report.v1",
                gates=GATES, classifier=CLASSIFIER, source_glb_sha256=EXPECTED_GLB_SHA256,
                oracle_script_sha256=digest(__file__),
                metadata_sha256=digest(metadata_path), texture_identity=texture_identity,
                source_topology={k: v for k, v in topology.items() if k != "nonseamed"},
                reference_method="Original GLB per-corner UVs; fixed proper axis transform; analytic Source projection; perspective-correct CPU raster",
                union=dict(eligible_pixels=total_samples, correct=total_correct, accuracy=total_correct/total_samples,
                           visible_faces=sorted(union_visible), tested_faces_3=sorted(union_tested),
                           never_visible_faces=sorted(set(range(1000))-union_visible),
                           untested_faces=sorted(set(range(1000))-union_tested), per_class=union_classes,
                           nonseamed_edge_pairs=total_pairs, matched_edge_pairs=total_matched_pairs,
                           edge_pair_accuracy=pair_accuracy, tested_nonseamed_edges=sorted(union_edges),
                           untested_nonseamed_edges=sorted(set(e["edge_id"] for e in topology["nonseamed"])-union_edges),
                           negative_controls=aggregate_controls),
                views=views, failures=failures,
                qualifications=[
                    "PASS covers only recorded eligible plateau pixels, tested faces and visible nonseamed edge-pairs.",
                    "Does not assert all-face/all-texel appearance, original BRDF equivalence, or user appearance approval.",
                    "280 intentional source UV seams are classified independently; no cross-seam color continuity is demanded.",
                    "Near/far clip crossings fail; no unsupported clipping or guessed engine-internal API is used.",
                    "Input screenshots, texture/readback PNGs, source mesh and reference arrays must remain private.",
                ])


def write_new_json(path, payload):
    path = Path(path)
    require(path.resolve().is_relative_to(EVIDENCE_DIR), "Output must stay inside this private Phase 22 evidence directory")
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", type=Path, default=Path(__file__).resolve().parent.parent/
                        "amd-static-prop-uv-improve-20261031-v1/asset/experimental-static-prop.glb")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--texture", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--sentinel-template", type=Path)
    args = parser.parse_args(argv)
    if args.sentinel_template:
        require(not args.metadata and not args.texture and not args.report, "Sentinel template is a separate read-only preparation mode")
        source = read_source_glb(args.glb)
        write_new_json(args.sentinel_template, dict(source_glb_sha256=EXPECTED_GLB_SHA256,
                       prop_position=EXPECTED_PROP_POSITION.tolist(),
                       prop_rotation_matrix=EXPECTED_PROP_ROTATION_MATRIX.tolist(),
                       sentinels=sentinel_points(fixed_world_positions(source["positions"]))))
        print(json.dumps(dict(status="SENTINEL_TEMPLATE_ONLY", path=str(args.sentinel_template))))
        return 0
    require(args.metadata and args.texture and args.report, "Proof requires --metadata, --texture and --report")
    require(not args.report.exists(), "Refusing to overwrite evidence report")
    try:
        report = run_oracle(args.glb, args.texture, args.metadata)
    except (OracleError, KeyError, ValueError, TypeError, IndexError, OverflowError, OSError, struct.error) as error:
        report = dict(status="NO-GO", gates=GATES, classifier=CLASSIFIER, failures=[f"Input/reference prerequisite: {error}"],
                      qualification="Actual image sampling not established; no fallback, fitting or lowered thresholds")
    write_new_json(args.report, report)
    print(json.dumps(dict(status=report["status"], report=str(args.report), failures=report["failures"])))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
