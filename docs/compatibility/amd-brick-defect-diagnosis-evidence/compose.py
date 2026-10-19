"""Gated private PIL evidence sheets from existing PNG + accepted diagnostics.

Launch with Python -I. GO must arrive after host Job attachment. No geometry,
model, ray tracing, or image generation runs here; all markers quote existing
diagnostic coordinates. An open marker leaves its center image pixel visible.
"""
import sys
sys.dont_write_bytecode = True

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

E = Path(__file__).resolve().parent
P = E.parents[2].parent / "artifacts/amd-brick-defect-diagnosis-1d1c1a6"
PNG = E.parents[2].parent / "artifacts/amd-brick-pbr-exposure-b0b7be1/renders/view-pos-x-ev2.png"
PNG_SHA = "74d1655d53d5fe7d262756c7e23c0a695aa42b7ee3c43e075007a59fc3bf7f54"
COLORS = {"case": "#0878dd", "boundary": "#ef8b00", "nonmanifold": "#dc21ac"}
FONT = Path("C:/Windows/Fonts/segoeui.ttf")
BOLD = Path("C:/Windows/Fonts/segoeuib.ttf")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path):
    path = Path(path)
    stat = path.stat()
    require(path.is_file() and not path.is_symlink() and stat.st_nlink == 1, "Plain single-link input required")
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "inode": stat.st_ino, "sha256": digest(path)}


def runtime_guard(event, args):
    if event == "import" and str(args[0]).split(".")[0] in {
        "torch", "triton", "flex_gemm", "o_voxel", "trellis2", "cupy", "bpy", "tensorflow", "jax", "numba",
    }:
        raise RuntimeError("Evidence composer forbids model/GPU/renderer imports")
    if event.startswith(("subprocess.", "socket.", "os.spawn", "os.exec")) or event in {"os.system", "os.posix_spawn", "os.fork"}:
        raise RuntimeError("Evidence composer forbids process/network execution")
    target = None
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        mode, flags = args[1], args[2]
        if ((isinstance(mode, str) and any(c in mode for c in "wax+")) or
                (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))):
            target = args[0]
    elif event in {"os.mkdir", "os.remove", "os.rmdir"}:
        target = args[0]
    elif event in {"os.rename", "os.link", "os.symlink"}:
        raise RuntimeError("Evidence composer forbids replacement/link operations")
    if target is not None:
        target = Path(os.fsdecode(target)).resolve()
        require(target == P or P in target.parents, "Composer writes must remain under private P")


def case_rows(diagnostics):
    """Label existing groups in ROI/raster order and retain the actual ray rows."""
    rows, counts = [], {}
    for group in sorted(diagnostics["selection"]["groups"], key=lambda g: (g["roi"], g["pixel"][1], g["pixel"][0])):
        roi = group["roi"]
        counts[roi] = counts.get(roi, 0) + 1
        require(counts[roi] <= 4, "At most four image groups per ROI")
        queries = [q for q in diagnostics["pixel_queries"] if q["group_id"] == group["group_id"]]
        center = [q for q in queries if q["offset_from_center"] == [0, 0]]
        require(len(center) == 1 and center[0]["pixel"] == group["pixel"], "Group center query differs")
        neighbors = [q for q in queries if q["offset_from_center"] != [0, 0]]
        require(len(neighbors) == 4, "Four existing neighbor samples required")
        rows.append({"id": roi.split("_")[0] + str(counts[roi]), "roi": roi,
                     "group_id": group["group_id"], "pixel": group["pixel"],
                     "center_query": center[0], "neighbor_hits": sum(bool(q["ray"]["hit"]) for q in neighbors),
                     "neighbor_count": len(neighbors), "kind": group["kind"]})
    require(len(rows) <= 8, "Case marker budget exceeded")
    return rows


def edge_rows(diagnostics):
    """Round robin among ROI/class buckets; never draw any original edge line."""
    buckets = {}
    for row in diagnostics["edge_midpoint_samples"]:
        if not row["visible"]:
            continue
        require(row["edge_class"] in {"boundary", "nonmanifold"}, "Unexpected topology edge class")
        buckets.setdefault((row["roi"], row["edge_class"]), []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: (r["midpoint_xy"][1], r["midpoint_xy"][0], r["vertices"]))
    selected = []
    for rank in range(4):
        for key in sorted(buckets):
            if rank < len(buckets[key]):
                candidate = buckets[key][rank]
                # Separate open rings; raw probes remain unchanged in diagnostics.
                if any(sum((a-(b+.5))**2 for a,b in zip(candidate['midpoint_xy'], group['pixel'])) < 16**2
                       for group in diagnostics['selection']['groups']):
                    continue
                if any(sum((a-b)**2 for a,b in zip(candidate['midpoint_xy'], old['midpoint_xy'])) < 12**2
                       for old in selected):
                    continue
                selected.append(candidate)
                if len(selected) == 12:
                    return selected
    return selected


def main():
    print("BRICK_DIAGNOSIS_COMPOSER_WAITING_GO", flush=True)
    require(sys.stdin.buffer.readline(64) == b"GO\n", "Owned host GO authorization required")
    require(sys.flags.isolated == 1, "Launch composer with Python -I")
    began = time.monotonic()
    sys.addaudithook(runtime_guard)
    from PIL import Image, ImageDraw, ImageFont, ImageStat

    diagnostic_path = P / "analysis/diagnostics.json"
    require(diagnostic_path.stat().st_size <= 8 * 1024**2, "Bounded diagnostics JSON required")
    before = {"source_png": identity(PNG), "diagnostics": identity(diagnostic_path), "composer": identity(__file__)}
    require(before["source_png"]["sha256"] == PNG_SHA, "Accepted +2EV PNG differs")
    def reject_constant(value):
        raise ValueError("Finite JSON required: " + value)
    diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    require(diagnostics["schema"] == "modly.amd.brick-defect-diagnosis.v1" and
            diagnostics["classification"] == "PASS_WITH_SCOPE_LIMITS" and diagnostics["inputs_unchanged"] is True and
            diagnostics["input_identities_before"]["image_ev2"]["sha256"] == PNG_SHA,
            "Accepted unchanged original diagnostics required")
    with Image.open(PNG) as source:
        require(source.format == "PNG" and source.size == (1024, 1024) and source.mode in {"RGB", "RGBA"},
                "Authenticated 1024-square PNG required")
        source.load()
        if source.mode == "RGBA":
            require(source.getextrema()[3] == (255, 255), "Opaque source PNG required")
        original = source.convert("RGB")
    cases, edges = case_rows(diagnostics), edge_rows(diagnostics)
    font_cache = {}
    def font(size, bold=False):
        key = size, bold
        if key not in font_cache:
            font_cache[key] = ImageFont.truetype(str(BOLD if bold else FONT), size)
        return font_cache[key]

    def text(draw, xy, value, size=18, bold=False, color="#202830"):
        draw.text(xy, str(value), font=font(size, bold), fill=color)

    def wrapped(draw, xy, value, width, size=17, line=23, color="#28333d"):
        x, y = xy
        words, current = str(value).split(), ""
        for word in words:
            proposed = (current + " " + word).strip()
            if current and draw.textlength(proposed, font=font(size)) > width:
                text(draw, (x, y), current, size, color=color)
                current, y = word, y + line
            else:
                current = proposed
        if current:
            text(draw, (x, y), current, size, color=color)
            y += line
        return y

    def markers(canvas, *, origin=(0, 0), scale=1., region=None, roi_name=None):
        draw = ImageDraw.Draw(canvas)
        ox, oy = origin
        def screen(point):
            return ox + point[0] * scale, oy + point[1] * scale
        def inside(point):
            return region is None or (region[0] <= point[0] < region[2] and region[1] <= point[1] < region[3])
        drawn_cases, drawn_edges = [], []
        for row in edges:
            if roi_name is not None and row["roi"] != roi_name:
                continue
            point = row["midpoint_xy"]
            if not inside(point):
                continue
            x, y = screen(point)
            radius = 4 if scale == 1 else 6
            draw.ellipse((x-radius-2, y-radius-2, x+radius+2, y+radius+2), outline="white", width=3)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=COLORS[row["edge_class"]], width=2)
            drawn_edges.append({"roi": row["roi"], "edge_class": row["edge_class"],
                                "midpoint_xy": point, "vertices": row["vertices"], "incidence": row["incidence"]})
        for row in cases:
            if roi_name is not None and row["roi"] != roi_name:
                continue
            if not inside(row["pixel"]):
                continue
            # Integer queried pixels refer to centers; edge samples are already subpixel.
            point = [v + .5 for v in row["pixel"]]
            x, y = screen(point)
            radius = 7 if scale == 1 else 9
            draw.ellipse((x-radius-2, y-radius-2, x+radius+2, y+radius+2), outline="white", width=4)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=COLORS["case"], width=2)
            label_font = font(17 if scale == 1 else 22, True)
            label_x, label_y = max(3, x + radius + 4), max(0, y - radius - 10)
            bbox = draw.textbbox((label_x, label_y), row["id"], font=label_font)
            # Keep labels within their image panel, away from sidebar or canvas edge.
            if bbox[2] + 4 >= canvas.width:
                label_x = x - radius - (bbox[2] - bbox[0]) - 6
                bbox = draw.textbbox((label_x, label_y), row["id"], font=label_font)
            draw.rounded_rectangle((bbox[0]-3, bbox[1]-2, bbox[2]+3, bbox[3]+2), radius=3, fill="white", outline=COLORS["case"])
            draw.text((label_x, label_y), row["id"], font=label_font, fill=COLORS["case"])
            drawn_cases.append({"id": row["id"], "pixel": row["pixel"], "sample_xy": point})
        return {"case_markers": drawn_cases, "case_marker_count": len(drawn_cases),
                "edge_midpoint_markers": drawn_edges, "edge_midpoint_marker_count": len(drawn_edges)}

    def case_lines(row):
        query, ray = row["center_query"], row["center_query"]["ray"]
        lines = [f"{row['id']}  pixel {tuple(row['pixel'])}  |  image luma {query['image_luma']:.1f}"]
        if ray["hit"]:
            rgb = ", ".join(f"{v:.3f}" for v in ray["raw_rgb"])
            lines += [f"Nearest hit: original face {ray['face_id']}  |  depth {ray['depth']:.4f}",
                      f"Raw RGB ({rgb})  |  |n.camera| {ray['abs_normal_cos_camera']:.3f}",
                      f"Edge distance {ray['nearest_edge_distance_px']:.3f}px; incidences {ray['face_edge_incidence_opposite_vertex']}"]
        else:
            lines += ["No accepted triangle on this exact center ray.",
                      "This center result is not pixel coverage proof."]
        lines.append(f"Neighbor rays hit {row['neighbor_hits']}/{row['neighbor_count']}; ambiguities {len(ray['ambiguities'])}")
        return lines

    outputs = []
    def save(path, image, marker_metadata, **extra):
        require(time.monotonic() - began < 50, "Composer deadline reached")
        with path.open("xb") as stream:
            image.save(stream, format="PNG")
        with Image.open(path) as check:
            check.load()
            require(check.format == "PNG" and check.mode == "RGB" and check.size == image.size,
                    "Composed PNG readback differs")
            stddev = ImageStat.Stat(check).stddev
            require(max(stddev) > 1, "Composed image is unexpectedly flat")
        outputs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path),
                        "dimensions": list(image.size), "readback_verified": True, "channel_stddev": stddev,
                        **marker_metadata, **extra})

    sheet = Image.new("RGB", (1696, 1170), "#f1f4f7")
    draw = ImageDraw.Draw(sheet)
    text(draw, (24, 14), "Original +2 EV view / sparse diagnostic samples", 29, True)
    text(draw, (24, 54), "Original mesh and existing image. No repair, relighting, or new render.", 19)
    sheet.paste(original, (24, 96))
    mark = markers(sheet, origin=(24, 96))
    draw = ImageDraw.Draw(sheet)
    x = 1072
    text(draw, (x, 96), "What the markers report", 23, True)
    text(draw, (x, 132), "Blue / white: image case centers A1...B4", 17)
    text(draw, (x, 158), "Orange: visible boundary-edge midpoint", 17, color=COLORS["boundary"])
    text(draw, (x, 184), "Magenta: visible nonmanifold-edge midpoint", 17, color=COLORS["nonmanifold"])
    y = 221
    for row in cases:
        lines = case_lines(row)
        for index, value in enumerate(lines):
            text(draw, (x, y), value, 15, bold=index == 0)
            y += 18
        y += 8
    y = max(y + 8, 1032)
    wrapped(draw, (x, y), "Nearest hit can be a rear surface seen through an opening. Center rays and visible midpoint markers alone do not establish shadow cause, a closed hole, or repair safety.", 590, 16, 21)
    text(draw, (24, 1130), f"{len(cases)} image case centers; {len(edges)} selected visible edge midpoints. Full edge lines are not shown.", 18)
    save(P / "diagnosis-overlay.png", sheet, mark, kind="full_view_plus_evidence_sidebar",
         case_evidence_rows=[{"id": row["id"], "lines": case_lines(row)} for row in cases])

    for roi_name, roi in sorted(diagnostics["selection"]["rois"].items()):
        require(roi_name in {"A_brick_detail", "B_vines"}, "Only the two accepted regions may be composed")
        box = roi["box"]
        scale, margin, gap = 3, 24, 32
        crop = original.crop(tuple(box))
        width, height = crop.width * scale, crop.height * scale
        panel = Image.new("RGB", (margin*2 + width*2 + gap, 104 + height + 160), "#f1f4f7")
        pd = ImageDraw.Draw(panel)
        title = "A / brick detail" if roi_name.startswith("A_") else "B / vines"
        text(pd, (margin, 12), title + " / original crop and sparse annotations", 25, True)
        text(pd, (margin, 49), f"Source box {tuple(box)}; 3x nearest-neighbor enlargement", 18)
        text(pd, (margin, 77), "Existing +2 EV crop", 17, True)
        text(pd, (margin+width+gap, 77), "Same crop + diagnostic markers", 17, True)
        enlarged = crop.resize((width, height), Image.Resampling.NEAREST)
        panel.paste(enlarged, (margin, 104))
        # Annotate on a bounded crop canvas so labels cannot leak into another panel.
        annotated = enlarged.copy()
        crop_mark = markers(annotated, origin=(-box[0]*scale, -box[1]*scale), scale=scale, region=box, roi_name=roi_name)
        panel.paste(annotated, (margin+width+gap, 104))
        pd = ImageDraw.Draw(panel)
        y = 116 + height
        for row in cases:
            if row["roi"] == roi_name:
                ray = row["center_query"]["ray"]
                status = f"nearest original face {ray['face_id']}" if ray["hit"] else "no accepted center-ray triangle"
                text(pd, (margin, y), f"{row['id']} {tuple(row['pixel'])}: {status}; neighbor hits {row['neighbor_hits']}/4", 17)
                y += 22
        wrapped(pd, (margin, y+3), "A rear-surface hit does not rule out a nearer opening. No shadow-cause, closed-loop, or repair-safety proof is claimed.", panel.width-margin*2, 17, 22)
        save(P / ("diagnosis-region-a.png" if roi_name.startswith("A_") else "diagnosis-region-b.png"),
             panel, crop_mark, kind="unannotated_and_annotated_crop", source_box=box, enlargement=scale)
    after = {"source_png": identity(PNG), "diagnostics": identity(diagnostic_path), "composer": identity(__file__)}
    require(before == after, "Source image, diagnostics, or composer changed")
    require(len(outputs) <= 3, "Private PNG output budget exceeded")
    record = {"schema": "modly.amd.brick-defect-diagnosis-overlay.v1", "classification": "PASS",
              "created_utc": datetime.now(timezone.utc).isoformat(), "inputs_before": before, "inputs_after": after,
              "inputs_unchanged": True, "outputs": outputs,
              "execution": {"CPU_only": True, "PIL_only_composition": True, "isolated_python": True,
                            "GO_gate_before_PIL_import": True, "writes_restricted_to_private_P": True,
                            "elapsed_seconds": round(time.monotonic()-began, 6)},
              "scope": "Existing image pixels plus sparse open markers and factual ray labels; no repair, relighting, dense edge overlay, whole-edge visibility, or causal interpretation."}
    with (P / "overlay-metadata.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"classification": "PASS", "outputs": len(outputs),
                      "metadata": str(P / "overlay-metadata.json"), "metadata_sha256": digest(P / "overlay-metadata.json")}), flush=True)


if __name__ == "__main__":
    main()
