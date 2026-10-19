"""Bounded, read-only CPU geometry on original vertex and face identities.

No file I/O, coordinate welding, mesh repair, patch construction, or export.
Ray arithmetic follows amd-brick-defect-diagnosis-evidence/projection.py:
PNG center rays, both windings, float64 barycentrics, inclusive depth clip.
Projection and conservative bounds must come from that authenticated helper.
"""
import numpy as np


def _array(value, columns=None, *, integer=False):
    value = np.asarray(value)
    if value.dtype.kind not in ("iu" if integer else "iuf"):
        raise ValueError("Expected numeric integer data" if integer else "Expected numeric data")
    if value.ndim == 0 or (columns is not None and (value.ndim != 2 or value.shape[1] != columns)):
        raise ValueError("Unexpected array shape")
    for start in range(0, len(value), 131072):
        if not np.isfinite(value[start:start + 131072]).all():
            raise ValueError("Nonfinite numeric data")
    return value


def screen_components(boundary_pairs, full_boundary_vertex_labels, nm_pairs):
    """Screen every supplied full-graph label without tracing or relabelling.

    Component counts include only vertices incident to a boundary edge; unused
    original vertices keep degree zero and cannot qualify as cycles. Count each
    unique undirected NM edge once per boundary component touched by its ends.
    """
    pairs = _array(boundary_pairs, 2, integer=True)
    labels = _array(full_boundary_vertex_labels, integer=True)
    nm = _array(nm_pairs, 2, integer=True)
    if labels.ndim != 1 or (labels < 0).any() or (len(labels) and labels.max() >= len(labels)):
        raise ValueError("Expected original-vertex full-graph component labels")
    for edges in (pairs, nm):
        if len(edges) and (edges.min() < 0 or edges.max() >= len(labels)):
            raise ValueError("Edge index outside original vertex labels")
    labels, pairs, nm = (a.astype(np.intp, copy=False) for a in (labels, pairs, nm))
    if len(pairs) and (labels[pairs[:, 0]] != labels[pairs[:, 1]]).any():
        raise ValueError("Boundary edge crosses supplied full-graph labels")
    count = int(labels.max()) + 1 if len(labels) else 0
    degree = np.bincount(pairs.ravel(), minlength=len(labels))
    vertex_counts = np.bincount(labels[degree > 0], minlength=count)
    edge_counts = np.bincount(labels[pairs[:, 0]], minlength=count)
    endpoints = np.bincount(labels[degree == 1], minlength=count)
    branches = np.bincount(labels[degree > 2], minlength=count)
    nm = np.unique(np.sort(nm, axis=1), axis=0)
    left, right = nm[:, 0], nm[:, 1]
    left_active = degree[left] > 0
    right_active = (degree[right] > 0) & (~left_active | (labels[left] != labels[right]))
    nm_touches = np.bincount(np.concatenate((labels[left[left_active]], labels[right[right_active]])),
                             minlength=count)
    closed = (vertex_counts >= 3) & (edge_counts == vertex_counts) & (endpoints == 0) & (branches == 0)
    return {"degree": degree, "vertex_counts": vertex_counts, "edge_counts": edge_counts,
            "endpoints": endpoints, "branches": branches, "nm_touches": nm_touches,
            "closed_cycle_mask": closed, "isolated_cycle_mask": closed & (nm_touches == 0)}


def trace_component(boundary_pairs, seededge):
    """Return the entire connected boundary multigraph; retain every input edge.

    At most 600000 edge rows are accepted. Degree counts use original IDs and
    count both ends of self-edges. ``ordered_loop`` omits the repeated closing
    vertex and starts at the smallest original ID, taking its smaller neighbor.
    """
    pairs = _array(boundary_pairs, 2, integer=True)
    seed = _array(seededge, integer=True)
    if len(pairs) > 600000 or seed.shape != (2,) or (pairs < 0).any() or (seed < 0).any():
        raise ValueError("Invalid boundary graph, seed, or 600000-edge budget")
    seed_key = tuple(sorted(int(v) for v in seed))
    adjacency, found = {}, False
    for edge_id, pair in enumerate(pairs):
        a, b = (int(v) for v in pair)
        adjacency.setdefault(a, []).append((b, edge_id))
        adjacency.setdefault(b, []).append((a, edge_id))
        found |= tuple(sorted((a, b))) == seed_key
    if not found:
        raise ValueError("Seed edge is absent from the original boundary pairs")
    pending, visited, edge_ids = [seed_key[0]], set(), set()
    while pending:
        vertex = pending.pop()
        if vertex in visited:
            continue
        visited.add(vertex)
        for neighbor, edge_id in adjacency[vertex]:
            edge_ids.add(edge_id)
            if neighbor not in visited:
                pending.append(neighbor)
    vertices, edge_ids = sorted(visited), sorted(edge_ids)
    degrees = [len(adjacency[v]) for v in vertices]
    histogram = {}
    for degree in degrees:
        histogram[str(degree)] = histogram.get(str(degree), 0) + 1
    histogram = dict(sorted(histogram.items(), key=lambda item: int(item[0])))
    closed = len(vertices) >= 3 and len(edge_ids) == len(vertices) and all(d == 2 for d in degrees)
    ordered = None
    if closed:
        ordered, previous, current = [vertices[0]], None, vertices[0]
        while len(ordered) < len(vertices):
            neighbors = sorted(v for v, _ in adjacency[current] if v != previous)
            following = neighbors[0]
            ordered.append(following)
            previous, current = current, following
        if len(set(ordered)) != len(vertices) or ordered[0] not in [v for v, _ in adjacency[current]]:
            raise ValueError("Degree-two component did not produce a simple original-ID cycle")
    return {"edge_ids": edge_ids, "edges": pairs[edge_ids].tolist(), "vertices": vertices,
            "edge_count": len(edge_ids), "vertex_count": len(vertices),
            "degree_histogram": histogram, "closed_simple": bool(closed), "ordered_loop": ordered}


def multihit(vertices, faces, projection, pixels, bounds, maxhits=128, *, pixel_centers=True):
    """Return every accepted original-face hit sorted by (depth, face ID).

    At most 121 rays and 128 accepted hits per ray; exceeding either refuses the
    whole call, never truncates. Supplied bounds preserve all original face rows.
    Explicit subpixel samples use ``pixel_centers=False`` as in the source helper.
    """
    vertices, faces = _array(vertices, 3), _array(faces, 3, integer=True)
    points = _array(pixels, 2)
    screen, depths = _array(projection.screen, 2), _array(projection.depth)
    lower, upper = _array(bounds.minimum, 2), _array(bounds.maximum, 2)
    outward = _array(projection.outward)
    clip = _array(projection.clip)
    if (len(points) > 121 or not isinstance(maxhits, (int, np.integer)) or not 1 <= maxhits <= 128 or
            len(screen) != len(vertices) or depths.shape != (len(vertices),) or
            lower.shape != (len(faces), 2) or upper.shape != lower.shape or
            outward.shape != (3,) or not np.isclose(np.linalg.norm(outward), 1., atol=1e-7, rtol=0.) or
            clip.shape != (2,) or not 0 < clip[0] < clip[1] or (lower > upper).any()):
        raise ValueError("Invalid projection, bounds, ray budget, or hit budget")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("Face index outside original vertices")
    if pixel_centers and (not np.equal(points, np.floor(points)).all() or (points < 0).any() or
                          (points[:, 0] >= projection.width).any() or
                          (points[:, 1] >= projection.height).any()):
        raise ValueError("Default pixels must be integer PNG indices")
    records = []
    for point in points:
        sample = point + .5 if pixel_centers else point.copy()
        candidate = np.flatnonzero((lower[:, 0] <= sample[0] + 1e-8) &
                                  (upper[:, 0] >= sample[0] - 1e-8) &
                                  (lower[:, 1] <= sample[1] + 1e-8) &
                                  (upper[:, 1] >= sample[1] - 1e-8))
        accepted, clipped_count, degenerate_count = [], 0, 0
        # The following barycentric and clip operations match projection.py.
        for start in range(0, len(candidate), 131072):
            ids = candidate[start:start + 131072]
            tri = screen[faces[ids]]
            x0, y0 = tri[:, 0, 0], tri[:, 0, 1]
            x1, y1 = tri[:, 1, 0], tri[:, 1, 1]
            x2, y2 = tri[:, 2, 0], tri[:, 2, 1]
            product_a, product_b = (y1-y2)*(x0-x2), (x2-x1)*(y0-y2)
            denominator = product_a + product_b
            threshold = 32*np.finfo(np.float64).eps*np.maximum(
                np.abs(product_a)+np.abs(product_b), np.finfo(np.float64).tiny)
            usable = np.abs(denominator) > threshold
            degenerate_count += int((~usable).sum())
            if not usable.any():
                continue
            ids = ids[usable]
            x0, y0, x1, y1, x2, y2, denominator = (
                a[usable] for a in (x0, y0, x1, y1, x2, y2, denominator))
            b0 = ((y1-y2)*(sample[0]-x2)+(x2-x1)*(sample[1]-y2))/denominator
            b1 = ((y2-y0)*(sample[0]-x2)+(x0-x2)*(sample[1]-y2))/denominator
            bary = np.column_stack((b0, b1, 1.-b0-b1))
            inside = (bary >= -1e-9).all(axis=1)
            ids, bary = ids[inside], bary[inside]
            depth = np.einsum("ij,ij->i", bary, depths[faces[ids]])
            clipped = (depth < clip[0]) | (depth > clip[1])
            clipped_count += int(clipped.sum())
            for face_id, distance, weights in zip(ids[~clipped], depth[~clipped], bary[~clipped]):
                accepted.append((float(distance), int(face_id), weights))
                if len(accepted) > maxhits:
                    raise ValueError("Accepted-hit budget exceeded; refusing truncation")
        accepted.sort(key=lambda row: (row[0], row[1]))
        ambiguities, hits = [], []
        if clipped_count:
            ambiguities.append("intersections_outside_clip")
        if degenerate_count:
            ambiguities.append("projected_degenerate_candidates")
        for depth, face_id, bary in accepted:
            face = faces[face_id]
            tri = np.asarray(vertices[face], dtype=np.float64)
            normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
            norm = np.linalg.norm(normal)
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("Accepted projected hit has invalid world normal")
            normal /= norm
            cosine = float(np.dot(normal, outward))
            ties = [i for d, i, _ in accepted if abs(d-depth) <= 1e-9]
            flags = []
            if len(ties) > 1:
                flags.append("coincident_depth_hits")
            if np.min(np.abs(bary)) <= 1e-9:
                flags.append("on_triangle_edge")
            hits.append({"face_id": face_id, "original_vertex_ids": face.tolist(), "depth": depth,
                         "barycentric": bary.tolist(), "position": (bary @ tri).tolist(),
                         "normal": normal.tolist(), "normal_cos_camera": cosine,
                         "abs_normal_cos_camera": abs(cosine), "tied_depth_face_ids": ties,
                         "ambiguities": flags})
            ambiguities.extend(flag for flag in flags if flag not in ambiguities)
        records.append({"pixel": point.tolist(), "sample_xy": sample.tolist(), "hit": bool(hits),
                        "empty_ray": not hits, "hits": hits, "intersection_count": len(hits),
                        "candidate_count": len(candidate), "clipped_intersection_count": clipped_count,
                        "projected_degenerate_candidate_count": degenerate_count,
                        "ambiguities": ambiguities, "truncated": False})
    return records


def loop_metrics(vertices, orderedloop):
    """Measure <=512 distinct original IDs; intersections are projected segments.

    A nonplanar projection cannot establish 3D segment intersection. Report the
    2D result separately, and leave ``self_intersection_3d`` unknown in that case.
    ``planar_within_tolerance`` is numerical qualification, not exact coplanarity.
    """
    vertices, loop = _array(vertices, 3), _array(orderedloop, integer=True)
    if loop.ndim != 1 or len(loop) < 3 or len(np.unique(loop)) != len(loop):
        raise ValueError("Expected at least three distinct original loop vertex IDs")
    if loop.min() < 0 or loop.max() >= len(vertices):
        raise ValueError("Loop ID outside original vertices")
    if len(loop) > 512:
        return {"supported": False, "reason": "loop_vertex_budget_exceeded", "vertex_count": len(loop)}
    points = np.asarray(vertices[loop], dtype=np.float64)
    lower, upper, center = points.min(axis=0), points.max(axis=0), points.mean(axis=0)
    centered = points-center
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    normal, u = vh[-1].copy(), vh[0].copy()
    normal *= 1 if normal[np.argmax(np.abs(normal))] >= 0 else -1
    u *= 1 if u[np.argmax(np.abs(u))] >= 0 else -1
    v = np.cross(normal, u)
    xy = np.column_stack((centered @ u, centered @ v))
    deviation = np.abs(centered @ normal)
    diagonal = float(np.linalg.norm(upper-lower))
    tolerance = max(diagonal, np.finfo(float).tiny)*1e-9
    planar = bool(deviation.max() <= tolerance)
    rank = int((singular > max(singular[0], np.finfo(float).tiny)*1e-12).sum())
    segments = np.roll(points, -1, axis=0)-points
    lengths = np.linalg.norm(segments, axis=1)
    length_tol = 64*np.finfo(float).eps*max(diagonal, np.finfo(float).tiny)
    area_tol = length_tol*max(diagonal, np.finfo(float).tiny)
    intersections = []

    def orient(a, b, c):
        ab, ac = b-a, c-a
        return ab[0]*ac[1]-ab[1]*ac[0]

    for i in range(len(loop)):
        a, b = xy[i], xy[(i+1) % len(loop)]
        for j in range(i+2, len(loop)):
            if i == 0 and j == len(loop)-1:
                continue  # Adjacent segments share their original endpoint.
            c, d = xy[j], xy[(j+1) % len(loop)]
            if (np.maximum(np.minimum(a, b), np.minimum(c, d)) >
                    np.minimum(np.maximum(a, b), np.maximum(c, d))+length_tol).any():
                continue
            p, q, r, s = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
            straddles_ab = min(p, q) <= area_tol and max(p, q) >= -area_tol
            straddles_cd = min(r, s) <= area_tol and max(r, s) >= -area_tol
            if straddles_ab and straddles_cd:
                intersections.append([i, j])
    area = .5*abs(float(np.sum(xy[:, 0]*np.roll(xy[:, 1], -1)-
                               xy[:, 1]*np.roll(xy[:, 0], -1))))
    return {"supported": True, "vertex_count": len(loop), "perimeter": float(lengths.sum()),
            "bounds": [lower.tolist(), upper.tolist()], "plane_center": center.tolist(),
            "plane_normal": normal.tolist(), "svd_singular_values": singular.tolist(), "svd_rank": rank,
            "max_plane_deviation": float(deviation.max()),
            "rms_plane_deviation": float(np.sqrt(np.mean(deviation**2))),
            "planarity_tolerance": tolerance, "planar_within_tolerance": planar,
            "projected_shoelace_area": area, "zero_length_segment_count": int((lengths <= length_tol).sum()),
            "projected_self_intersection": bool(intersections),
            "projected_intersecting_segment_pairs": intersections,
            "self_intersection_3d": bool(intersections) if planar and rank >= 2 else None,
            "self_intersection_3d_qualification": ("conditional_on_planarity_tolerance" if planar and rank >= 2
                                                       else "not_proven_for_nonplanar_loop" if not planar
                                                       else "not_proven_for_degenerate_loop")}
