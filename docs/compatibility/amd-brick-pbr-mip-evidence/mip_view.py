"""Bounded +X orthographic UV/footprint diagnostic, without lighting or repair."""
import numpy as np

MAX_VERTICES, MAX_FACES, MAX_FRAME, MAX_WORK = 100000, 6000, 640, 16000000


def project(vertices, faces, uv, object_height, frame=640):
    """Raster unchanged double-sided triangles; return frontmost UVs and LODs.

    +Y is screen right, +Z is screen up, and larger X is nearer the +X camera.
    LOD is log2(max screen-axis UV derivative length * 1024), limited below by
    zero for magnification. Only <=1e-6 numerical excess at the LOD5 endpoint is
    rounded down and explicitly counted; genuinely unavailable levels reject.
    This is an unlit base-colour filter diagnostic, not a PBR renderer.
    """
    if (not isinstance(vertices, np.ndarray) or vertices.dtype.kind != 'f'
            or vertices.ndim != 2 or vertices.shape[1] != 3
            or not 3 <= len(vertices) <= MAX_VERTICES or not np.isfinite(vertices).all()):
        raise ValueError('Bounded finite floating vertices[N,3] required')
    if (not isinstance(faces, np.ndarray) or faces.dtype != np.int32
            or faces.ndim != 2 or faces.shape[1] != 3
            or not 1 <= len(faces) <= MAX_FACES or faces.min() < 0 or faces.max() >= len(vertices)
            or (faces[:, 0] == faces[:, 1]).any() or (faces[:, 1] == faces[:, 2]).any()
            or (faces[:, 2] == faces[:, 0]).any()):
        raise ValueError('Bounded original I32 triangle faces required')
    if (not isinstance(uv, np.ndarray) or uv.dtype.kind != 'f' or uv.shape != (len(vertices), 2)
            or not np.isfinite(uv).all() or (uv < 0).any() or (uv > 1).any()):
        raise ValueError('Finite matching unit UV coordinates required')
    if (type(frame) is not int or not 16 <= frame <= MAX_FRAME
            or type(object_height) not in (int, float) or not np.isfinite(object_height)
            or not 0 < object_height <= frame - 2):
        raise ValueError('Bounded frame and unclipped positive projected height required')
    v, tex = vertices.astype(np.float64), uv.astype(np.float64)
    low, high = v.min(0), v.max(0)
    if high[2] <= low[2] or high[1] <= low[1]:
        raise ValueError('Nonempty Y/Z projection extent required')
    scale = float(object_height) / (high[2] - low[2])
    width = (high[1] - low[1]) * scale
    if width > frame - 2:
        raise ValueError('Projected width would clip the unchanged patch')
    center = (low + high) * .5
    screen = np.column_stack((frame * .5 + (v[:, 1] - center[1]) * scale,
                              frame * .5 - (v[:, 2] - center[2]) * scale))
    plans, work, edge_on, endpoint_rounding = [], 0, 0, 0
    for face_id, corners in enumerate(faces):
        xyz, xy = v[corners], screen[corners]
        if np.linalg.norm(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])) <= 1e-24:
            raise ValueError('Degenerate original triangle cannot be rendered as valid geometry')
        matrix = np.column_stack((xy[1] - xy[0], xy[2] - xy[0]))
        if abs(np.linalg.det(matrix)) <= 1e-12:
            edge_on += 1
            continue  # Original edge-on face has no screen area; no replacement.
        inverse = np.linalg.inv(matrix)
        tuv = tex[corners]
        gradient = np.column_stack((tuv[1] - tuv[0], tuv[2] - tuv[0])) @ inverse
        singular = np.linalg.svd(gradient, compute_uv=False)
        if not np.isfinite(gradient).all() or singular[-1] <= 0:
            raise ValueError('Finite nonzero UV derivative required')
        rho = 1024 * max(np.linalg.norm(gradient[:, 0]), np.linalg.norm(gradient[:, 1]))
        lod = float(np.log2(max(1., rho)))
        if not np.isfinite(lod) or lod > 5 + 1e-6:
            raise ValueError('Projected footprint requires unavailable mip level above5')
        if lod > 5:
            endpoint_rounding += 1
            lod = 5.  # Explicitly reported numerical endpoint rounding only.
        lower = np.maximum(0, np.ceil(xy.min(0) - .5).astype(np.int32))
        upper = np.minimum(frame - 1, np.floor(xy.max(0) - .5).astype(np.int32))
        if (upper < lower).any():
            continue
        work += int(np.prod(upper - lower + 1, dtype=np.int64))
        if work > MAX_WORK:
            raise ValueError('Projected bounding-box work budget exceeded before raster allocation')
        plans.append((face_id, corners, xy[0], inverse, lower, upper, lod,
                      float(rho), float(singular[0] / singular[-1])))
    depth = np.full((frame, frame), -np.inf)
    face_image = np.full((frame, frame), -1, np.int32)
    uv_image = np.zeros((frame, frame, 2), np.float64)
    lods, rhos, anisotropy = {}, {}, {}
    for face_id, corners, origin, inverse, lower, upper, lod, rho, ratio in plans:
        x, y = np.meshgrid(np.arange(lower[0], upper[0] + 1), np.arange(lower[1], upper[1] + 1))
        x, y = x.ravel(), y.ravel()
        weights12 = (np.column_stack((x + .5, y + .5)) - origin) @ inverse.T
        bary = np.column_stack((1 - weights12.sum(1), weights12))
        inside = (bary >= -1e-12).all(1)
        x, y, bary = x[inside], y[inside], bary[inside]
        z = bary @ v[corners, 0]
        nearer = z > depth[y, x]  # Stable original face order breaks exact ties.
        x, y, bary, z = x[nearer], y[nearer], bary[nearer], z[nearer]
        depth[y, x], face_image[y, x] = z, face_id
        uv_image[y, x] = bary @ tex[corners]
        lods[face_id], rhos[face_id], anisotropy[face_id] = lod, rho, ratio
    mask = face_image >= 0
    if not mask.any():
        raise ValueError('No original patch fragments visible at the declared projection')
    y, x = np.nonzero(mask)
    face_ids = face_image[mask]
    visible = np.unique(face_ids)
    query_uv = uv_image[mask]
    if (query_uv < -1e-12).any() or (query_uv > 1 + 1e-12).any():
        raise ValueError('Raster UV escaped the original unit atlas')
    return dict(query_uv=np.clip(query_uv, 0, 1), query_lod=np.array([lods[int(f)] for f in face_ids]),
                query_faces=face_ids, pixels_yx=np.column_stack((y, x)).astype(np.int32), mask=mask,
                metadata=dict(view='+X; right+Y/up+Z; largestX nearest', unlit=True,
                              actual_projected_height_pixels=float(object_height),
                              actual_projected_width_pixels=float(width), frame_pixels=frame,
                              fragment_count=int(mask.sum()), visible_faces=len(visible),
                              edge_on_projected_faces=edge_on, bbox_work_pixels=work,
                              LOD_min=min(lods[int(f)] for f in visible), LOD_max=max(lods[int(f)] for f in visible),
                              rho_max=max(rhos[int(f)] for f in visible),
                              anisotropy_max=max(anisotropy[int(f)] for f in visible),
                              numerical_LOD5_endpoint_roundings=endpoint_rounding,
                              source_geometry_modified=False, hidden_face_culling='frontmost original depth only'))
