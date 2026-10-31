"""Phase 22B Stage 1: authenticate and independently recheck OLD captures only.

Writes NEW private evidence only beside this script. No engine calls, capture,
settings changes, old-file writes, installs, downloads, mask changes or fitting.
The frozen Phase 22 oracle is imported and used unchanged in memory.
"""
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np


BASE = Path(__file__).resolve().parent
OLD = BASE.parent/"amd-static-prop-sbox-uv-pbr-20261031-v1"
ORACLE_SHA = "b0aadae4d6c3ebb859aeada7f135621d219a8cbe26d80124d1296a30e3aab1ae"
METADATA_SHA = "b0372a7aa3edd03f3d19872c80050a1474215e0f38bfb3fd8a55f49453465e5b"
RESULT_SHA = "e988366234ea888028083fffd030ab2f57afdfb08a9efdf2db8a857b5cb4e718"
DETAILS_SHA = "139b795f4d0168e099e595c302226ef78257c9544e92f55834fc3d26869bd38a"


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_json(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE), "New output escaped Phase 22B evidence directory")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def import_oracle():
    path = OLD/"independent_sampling_oracle.py"
    require(sha(path) == ORACLE_SHA, "Historical frozen oracle changed")
    spec = importlib.util.spec_from_file_location("phase22_frozen_oracle", path)
    result = importlib.util.module_from_spec(spec)
    # Preserve historical evidence even when this script is invoked without -B.
    original_bytecode_policy = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(result)
    finally:
        sys.dont_write_bytecode = original_bytecode_policy
    return result


def array_sha(value):
    value = np.ascontiguousarray(value)
    header = json.dumps(dict(shape=value.shape, dtype=value.dtype.str), sort_keys=True).encode()
    return hashlib.sha256(header+b"\n"+value.tobytes()).hexdigest()


def independent_project(points, camera):
    # Deliberately independent implementation of the documented Source basis.
    points = np.atleast_2d(points); delta = points-np.asarray(camera["position"])
    basis = np.asarray([camera[k] for k in ("forward", "left", "up")])
    view = delta @ basis.T
    tangent = math.tan(math.radians(camera["field_of_view"])/2)
    return np.column_stack([640-640*view[:, 1]/(view[:, 0]*tangent),
                            480-640*view[:, 2]/(view[:, 0]*tangent)])


def independent_ray_uv(points, uv, camera, x, y):
    tangent = math.tan(math.radians(camera["field_of_view"])/2)
    basis = np.array([camera[k] for k in ("forward", "left", "up")])
    direction = np.linalg.solve(basis, [1., -(2*(x+.5)/1280-1)*tangent,
                                        (1-2*(y+.5)/960)*tangent/(1280/960)])
    b1, b2, depth = np.linalg.solve(np.column_stack([points[1]-points[0], points[2]-points[0], -direction]),
                                   camera["position"]-points[0])
    bary = np.array([1-b1-b2, b1, b2])
    require(depth > 0 and min(bary) >= -1e-10, "Witness does not intersect its stated triangle")
    return bary @ uv, bary


def independent_texture_category(rgba, uv):
    height, width = rgba.shape[:2]
    x, y = np.floor(np.mod(uv, 1.)*[width, height]).astype(int)
    color = rgba[y % height, x % width, :3]
    require(np.all((color == 0) | (color == 255)), "Expected texture is not binary RGB")
    return int((color[0] == 255)+2*(color[1] == 255)+4*(color[2] == 255)), (int(x), int(y))


def nearby_boundary(rgba, uv, category, points, corner_uv, camera, x, y):
    """Find a nearest texture-Linf category boundary; map that example to screen.

    This is not a fitted screen filter or a claim of minimum screen-space distance.
    """
    center = np.mod(uv, 1.)*1024; tx, ty = np.floor(center).astype(int)
    best = None
    for yy in range(ty-16, ty+17):
        for xx in range(tx-16, tx+17):
            c, _ = independent_texture_category(rgba, [(xx+.5)/1024, (yy+.5)/1024])
            if c == category:
                continue
            boundary = np.clip(center, [xx, yy], [xx+1, yy+1])
            distance = float(np.max(np.abs(boundary-center)))
            candidate = (distance, float(np.linalg.norm(boundary-center)), yy, xx, boundary, c)
            if best is None or candidate[:4] < best[:4]:
                best = candidate
    require(best is not None, "No nearby different-category texel found within the fixed 16-texel search")
    distance, _, yy, xx, boundary, other = best
    bary = np.linalg.solve(np.vstack([corner_uv.T, np.ones(3)]), [*(boundary/1024), 1.])
    screen = independent_project(bary @ points, camera)[0]
    return dict(texture_linf_distance_texels=distance, other_category=other,
                boundary_uv=(boundary/1024).tolist(), screen_point=screen.tolist(),
                screen_distance_pixels=float(np.linalg.norm(screen-[x+.5, y+.5])),
                maps_inside_same_triangle=bool(np.min(bary) >= -1e-8),
                qualification="One nearest texture-Linf boundary, not a fitted filter or asserted nearest screen boundary")


def prepare_baseline():
    o = import_oracle()
    metadata_path = OLD/"capture-metadata.json"; result_path = OLD/"sampling-independent-result.json"
    details_path = OLD/"sampling-failure-private-details.json"
    require(sha(metadata_path) == METADATA_SHA and sha(result_path) == RESULT_SHA and sha(details_path) == DETAILS_SHA,
            "Historical metadata/result/private witness identity changed")
    metadata = json.loads(metadata_path.read_text()); result = json.loads(result_path.read_text())
    details = json.loads(details_path.read_text())
    require(result["status"] == "NO-GO" and result["failures"] == ["top: Source face 659 agreement below 95%"],
            "Historical NO-GO differs")
    image_hashes = {}
    for capture in metadata["captures"]:
        actual_hash = sha(capture["png_path"])
        require(actual_hash == capture["png_sha256"], "Historical screenshot changed")
        image_hashes[capture["name"]] = actual_hash
    source = o.read_source_glb(OLD.parent/"amd-static-prop-uv-improve-20261031-v1/asset/experimental-static-prop.glb")
    world = o.fixed_world_positions(source["positions"])
    capture = next(c for c in metadata["captures"] if c["name"] == "top")
    camera, projection = o.validate_camera(capture, o.sentinel_points(world))
    raster = o.rasterize(world, source["uv"], source["faces"], camera, 1280, 960)
    classes, texture_identity = o.authenticate_texture(OLD/"texture/diagnostic_uv_color.png", metadata, OLD)
    rgba_texture = o.read_image(OLD/"texture/diagnostic_uv_color.png")
    expected, mask, mask_info = o.reference_mask(raster, classes, o.category_margins(classes))
    rgba_actual = o.read_image(capture["png_path"], (1280, 960)); actual = o.classify_rgb(rgba_actual)
    mismatch = mask & (actual != expected); face_mask = mask & (raster["face"] == 659)
    require(int(mask.sum()) == 132250 and int(face_mask.sum()) == 38 and int(mismatch.sum()) == 20,
            "Historical fixed top mask/face/mismatch populations differ")
    points20 = list(zip(*np.where(mismatch))); points38 = list(zip(*np.where(face_mask)))
    require(set(points20) == {(v["y"], v["x"]) for v in details["all_top_mismatches"]}, "Historical twenty positions differ")
    require(set(points38) == {(v["y"], v["x"]) for v in details["all_38_face_witnesses"]}, "Historical 38 positions differ")
    return dict(o=o, metadata=metadata, result=result, details=details, source=source, world=world,
                capture=capture, camera=camera, projection=projection, raster=raster, classes=classes,
                texture_identity=texture_identity, texture=rgba_texture, rgba=rgba_actual, actual=actual,
                expected=expected, mask=mask, face_mask=face_mask, mismatch=mismatch,
                image_hashes=image_hashes, mask_info=mask_info)


def run_stage1():
    b = prepare_baseline(); o = b["o"]
    old_records = {(v["y"], v["x"]): v for v in b["details"]["all_top_mismatches"]}
    all38 = {(v["y"], v["x"]): v for v in b["details"]["all_38_face_witnesses"]}
    records = []
    for y, x in sorted(set(old_records) | set(all38)):
        face = int(b["raster"]["face"][y, x]); indices = b["source"]["faces"][face]
        points = b["world"][indices]; uv = b["source"]["uv"][indices]
        point_uv, bary = independent_ray_uv(points, uv, b["camera"], x, y)
        category, texel = independent_texture_category(b["texture"], point_uv)
        interpolation_error = float(np.max(np.abs(point_uv-b["raster"]["uv"][y, x])))
        require(interpolation_error < 1e-11 and category == int(b["expected"][y, x]), "Independent expected category differs")
        projected_intersection = independent_project(bary @ points, b["camera"])[0]
        screen_error = float(np.max(np.abs(projected_intersection-[x+.5, y+.5])))
        require(screen_error < 1e-9, "Independent ray does not return the stored pixel center")
        actual_rgba = b["rgba"][y, x].tolist(); actual_class = int(b["actual"][y, x])
        if (y, x) in old_records:
            old = old_records[(y, x)]
            require(old["source_face"] == face and old["rgba"] == actual_rgba and old["expected"] == category
                    and old["actual"] == actual_class and old["source_texel_xy"] == list(texel), "Saved mismatch RGBA/category differs")
        if (y, x) in all38:
            old = all38[(y, x)]
            require(old["actual_rgba"] == actual_rgba and old["expected"] == category and old["actual"] == actual_class,
                    "Saved triangle-659 witness differs")
        record = dict(x=int(x), y=int(y), source_face=face, expected=category, actual=actual_class,
                      rgba=actual_rgba, source_uv=point_uv.tolist(), source_texel_xy=list(texel),
                      in_original_twenty=(y, x) in old_records, in_original_38=(y, x) in all38,
                      ray_uv_error=interpolation_error, ray_screen_error_pixels=screen_error,
                      nearby_category_boundary=nearby_boundary(b["texture"], point_uv, category, points, uv, b["camera"], x, y))
        if (y, x) in old_records and face == 659:
            categories = Counter()
            compiled_points = np.asarray(b["details"]["compiled_world_corner_positions"])
            compiled_uv = np.asarray(b["details"]["compiled_uv_corners"])
            for dy, dx in itertools.product(np.linspace(-.5, .5, 9), repeat=2):
                q, _ = independent_ray_uv(compiled_points, compiled_uv, b["camera"], x+dx, y+dy)
                c, _ = independent_texture_category(b["texture"], q); categories[c] += 1
            require(dict(categories) == {int(k): v for k, v in all38[(y, x)]["fixed_half_pixel_grid_diagnostic"]["categories"].items()},
                    "Saved half-pixel sensitivity counts differ")
            record["fixed_half_pixel_grid_categories"] = dict(categories)
        records.append(record)
    rerun = o.run_oracle(OLD.parent/"amd-static-prop-uv-improve-20261031-v1/asset/experimental-static-prop.glb",
                         OLD/"texture/diagnostic_uv_color.png", OLD/"capture-metadata.json")
    # Historical reports are JSON: normalize the in-memory witness pixel tuples
    # through the same lossless JSON representation before exact comparison.
    require(json.loads(json.dumps(rerun, allow_nan=False)) == b["result"],
            "Unchanged oracle does not reproduce historical report")
    private = dict(classification="PRIVATE_COORDINATES_RGBA_UV_BOUNDARIES_DO_NOT_PUBLISH", status="HISTORICAL_NO_GO_UNCHANGED",
                   points=records, original_twenty_positions_yx=[list(p) for p in sorted(old_records)],
                   original_38_positions_yx=[list(p) for p in sorted(all38)])
    new_json(BASE/"stage1-private-exact-witnesses.json", private)
    mismatch_records = [r for r in records if r["in_original_twenty"]]
    inside_boundary = [r["nearby_category_boundary"]["screen_distance_pixels"] for r in mismatch_records
                       if r["nearby_category_boundary"]["maps_inside_same_triangle"]]
    hashes = dict(oracle=ORACLE_SHA, metadata=METADATA_SHA, historical_result=RESULT_SHA,
                  historical_private_details=DETAILS_SHA, images=b["image_hashes"],
                  top_mask=array_sha(b["mask"]), top_expected_categories=array_sha(b["expected"]),
                  original_38_positions=array_sha(np.argwhere(b["face_mask"]).astype("<i4")),
                  original_twenty_positions=array_sha(np.argwhere(b["mismatch"]).astype("<i4")))
    summary = dict(status="STAGE1_VERIFIED_HISTORICAL_NO_GO_UNCHANGED", engine_calls=0, new_captures=0,
                   historical_no_go="top source triangle659:34/38 below95%", original_top_samples=132250,
                   original_triangle659_samples=38, original_mismatch_locations=20,
                   independently_verified_exact_mismatch_records=20, independently_verified_all_triangle659_records=38,
                   unique_reviewed_positions=len(records), historical_oracle_report_reproduced_exactly=True,
                   max_independent_uv_difference=max(r["ray_uv_error"] for r in records),
                   max_independent_screen_center_error_pixels=max(r["ray_screen_error_pixels"] for r in records),
                   top_projection_error_pixels=b["projection"]["max_error_pixels"],
                   mismatch_actual_category_counts=dict(Counter(r["actual"] for r in mismatch_records)),
                   all_twenty_have_nonbinary_rgb=all(any(c not in (0, 255) for c in r["rgba"][:3]) for r in mismatch_records),
                   nearby_boundaries_mapped_inside_same_triangle=len(inside_boundary),
                   nearby_boundary_example_screen_distance_pixels_min=min(inside_boundary),
                   nearby_boundary_example_screen_distance_pixels_max=max(inside_boundary),
                   hashes=hashes, frozen_gates=o.GATES, frozen_classifier=o.CLASSIFIER,
                   aa_state_in_old_images="UNOBSERVED: old metadata did not measure AA/MSAA/CMAA2/resolution-upscaling state",
                   interpretation="Mixed RGB and nearby categorical boundaries support an output-footprint hypothesis, not proof of any particular AA stage. No parameter was fitted to colors or boundary locations.",
                   comparison_plan="Only if root independently qualifies one public reversible camera-local control: one new top capture; reuse original mask132250,38face points,20locations, classifier, original gates and negatives. No retries/settings sweep/new masks.",
                   privacy="Exact coordinates/RGBA/UV/boundary points are in stage1-private-exact-witnesses.json; never publish it.")
    new_json(BASE/"stage1-mismatch-review.json", summary)
    with (BASE/"stage1-mismatch-review.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("# Stage 1 independent saved-capture review\n\n"
            "Historical Phase 22 remains **NO-GO**: top-view GLB triangle **659**, **34/38 (89.47%)**, below the original **95%** gate. The frozen oracle and five captured PNGs are authenticated, and the unchanged oracle reproduces the historical report exactly. No engine call, setting change or new capture was performed.\n\n"
            "All **20** saved mismatch positions/RGBA/category records and all **38** triangle-659 points were independently verified against the original PNGs, source texture, source geometry, camera basis and ray-plane calculations. These comprise **54** unique screen positions. Expected UV interpolation and pixel-center projection agree numerically with the frozen oracle. Exact coordinates, colors, UVs and mapped nearby feature boundaries remain in the new private witness JSON.\n\n"
            "All twenty mismatches contain nonbinary RGB. Nineteen are unknown under the unchanged classifier; one expected-black sample classifies as white. The four triangle-659 failures remain two expected-magenta and two expected-green mixed pixels. Nearby source-category boundaries map inside the same triangles; the saved half-pixel sensitivity findings are reproducible. This supports a sampling-footprint hypothesis but does not identify the responsible image stage.\n\n"
            "**The old images do not measure their AA state.** Post-processing=false is not evidence that AA/MSAA/CMAA2/upscaling was disabled. Public SceneCamera AA APIs are leads for separate inspection, not proof that a safe capture-relevant control has been qualified.\n\n"
            "No executable conditional-capture comparison adapter has been completed or run: a qualified local control has not been established. If root independently qualifies a reversible camera-local control and authorizes the single capture, comparison must preserve the original **132,250** top-mask pixels, original **38** face points and all original **20** mismatch locations, using the frozen classifier/gates/controls. Historical NO-GO cannot be replaced by a conditional result. No retuning or additional acceptance gates are proposed.\n")
    require(sha(OLD/"independent_sampling_oracle.py") == ORACLE_SHA, "Historical oracle changed during read-only review")
    print(json.dumps({k: summary[k] for k in ("status", "independently_verified_exact_mismatch_records", "independently_verified_all_triangle659_records", "unique_reviewed_positions", "max_independent_uv_difference")}))
    return summary


if __name__ == "__main__":
    run_stage1()
