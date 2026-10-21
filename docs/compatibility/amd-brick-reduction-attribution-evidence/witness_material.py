"""Explain saved colour/pixel failures without a new field query or full raster.

The owned caller authenticates the old evidence and original/candidate arrays
before calling run. Exact local geometry, colours and images are private output.
Only the returned scalar/identity summary is suitable for publication.
"""
import sys
sys.dont_write_bytecode = True
import hashlib, json, os
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[2]
POLD = ROOT.parent/'artifacts/amd-brick-half-poly-5705f9d-corrected'
PNEW = ROOT.parent/'artifacts/amd-brick-reduction-attribution-da37443'
BATCH = 32768
DEPTH_REPRODUCTION_TOLERANCE = 2e-12
RGB_REPRODUCTION_TOLERANCE = 2e-15
MATERIAL_FACTORS = dict(metallic=.00011082693761409814,
                        roughness=.999995986777544, alpha=1.)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def decode(np, byte_rgb):
    """The exact previous native-byte-to-F32 scene-linear expression."""
    need(byte_rgb.dtype == np.uint8 and byte_rgb.ndim == 2 and byte_rgb.shape[1] == 3,
         'RGB native byte rows')
    encoded = byte_rgb.astype(np.float64)/255
    return np.where(encoded <= .04045, encoded/12.92,
                    ((encoded+.055)/1.055)**2.4).astype('<f4')


def encode(np, linear):
    linear = np.clip(np.asarray(linear, dtype=np.float64), 0, 1)
    return np.where(linear <= .0031308, 12.92*linear,
                    1.055*np.power(linear, 1/2.4)-.055)


def select_material_query(np, source, predicted, region_rows):
    """Max channel over the two saved cohorts; ties use query then channel."""
    need(source.ndim == predicted.ndim == 2 and source.shape[0] == predicted.shape[0]
         and source.shape[1] == 6 and predicted.shape[1] == 3
         and np.isfinite(source).all() and np.isfinite(predicted).all(),
         'Finite matching saved source/prediction populations')
    start = 0
    candidates = []
    ranges = {}
    for name, row in region_rows.items():
        count = row['samples']
        need(type(count) is int and count == 128, 'Exactly128 saved samples per region')
        ranges[name] = (start, start+count)
        if name in ('high_variation', 'octant_2'):
            candidates.extend(range(start, start+count))
        start += count
    need(start == len(source) and len(candidates) == 256,
         'Both complete frozen high-variation/octant2 populations')
    queries = np.asarray(sorted(candidates), dtype=np.int32)
    error = np.abs(predicted[queries]-np.clip(source[queries, :3].astype(np.float64), 0, 1))
    at = int(np.argmax(error))
    query, channel = int(queries[at//3]), at % 3
    region = next(name for name, (first, last) in ranges.items() if first <= query < last)
    return dict(query=query, channel=channel, region=region,
                region_query=query-ranges[region][0], maximum=float(error.ravel()[at]),
                selection='Maximum saved absolute encoded-RGB channel in the union of high_variation and octant_2; ties lowest query row then RGB channel')


def basis(np, camera):
    need(camera['camera_type'] == 'ORTHO', 'Authenticated orthographic camera')
    location, right, up, axis = [np.asarray(camera[key], dtype=np.float64) for key in
                                ('camera_location', 'camera_right', 'camera_up', 'axis_direction')]
    need(all(value.shape == (3,) and np.isfinite(value).all()
             for value in (location, right, up, axis)), 'Finite frozen camera basis')
    outward = np.cross(right, up)
    need(np.linalg.norm(axis) > 0 and np.isclose(np.linalg.norm(right), 1, rtol=0, atol=1e-7)
         and np.isclose(np.linalg.norm(up), 1, rtol=0, atol=1e-7)
         and np.isclose(np.dot(right, up), 0, rtol=0, atol=1e-7)
         and np.allclose(outward, axis/np.linalg.norm(axis), rtol=0, atol=1e-7),
         'Original orthonormal camera basis')
    scale, clip = float(camera['ortho_scale']), np.asarray(camera['clip'], dtype=np.float64)
    need(np.isfinite(scale) and scale > 0 and clip.shape == (2,)
         and np.isfinite(clip).all() and 0 < clip[0] < clip[1], 'Positive original frame/clip')
    return location, right, up, outward, scale, clip


def local_hit(np, vertices, faces, camera, row, column, size=640):
    """Stream bboxes for one existing pixel; do not expand or raster other pixels.

    Opaque two-sided barycentrics,1e-9 edge tolerance and exact-depth/lowest-row
    tie handling match the frozen raster. No projected whole-vertex population
    or image-sized geometry buffer is created.
    """
    need(vertices.ndim == faces.ndim == 2 and vertices.shape[1] == faces.shape[1] == 3
         and vertices.dtype == np.float32 and faces.dtype == np.int32
         and 0 < len(vertices) <= 1_957_891 and 0 < len(faces) <= 4_173_584,
         'Bounded original or compact F32/I32 geometry')
    need(type(size) is int and 0 < size <= 640 and type(row) is int
         and type(column) is int and 0 <= row < size and 0 <= column < size,
         'One existing bounded pixel centre')
    location, right, up, outward, scale, clip = basis(np, camera)
    best = None
    bbox_count = inside_count = 0
    for first in range(0, len(faces), BATCH):
        ids = faces[first:first+BATCH]
        need((ids >= 0).all() and (ids < len(vertices)).all(), 'Local scan index range')
        relative = vertices[ids].astype(np.float64).reshape(-1, 3)-location
        xy = np.empty((len(relative), 2), dtype=np.float64)
        xy[:, 0] = size*(.5+(relative @ right)/scale)
        xy[:, 1] = size*(.5-(relative @ up)/scale)
        depth = -(relative @ outward)
        xy, depth = xy.reshape(-1, 3, 2), depth.reshape(-1, 3)
        need(np.isfinite(xy).all() and np.isfinite(depth).all(), 'Finite local projected corners')
        lower = np.ceil(xy.min(axis=1)-.5)
        upper = np.floor(xy.max(axis=1)-.5)
        eligible = ((lower[:, 0] <= column) & (column <= upper[:, 0])
                    & (lower[:, 1] <= row) & (row <= upper[:, 1]))
        local = np.flatnonzero(eligible)
        bbox_count += len(local)
        if not len(local):
            continue
        p = xy[local]
        x0, y0 = p[:, 0, 0], p[:, 0, 1]
        x1, y1 = p[:, 1, 0], p[:, 1, 1]
        x2, y2 = p[:, 2, 0], p[:, 2, 1]
        denominator = (y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        nonflat = denominator != 0
        local, p = local[nonflat], p[nonflat]
        x0, y0, x1, y1, x2, y2, denominator = [value[nonflat] for value in
                                                             (x0, y0, x1, y1, x2, y2, denominator)]
        if not len(local):
            continue
        b0 = ((y1-y2)*(column+.5-x2)+(x2-x1)*(row+.5-y2))/denominator
        b1 = ((y2-y0)*(column+.5-x2)+(x0-x2)*(row+.5-y2))/denominator
        b2 = 1-b0-b1
        inside = ((b0 >= -1e-9) & (b1 >= -1e-9) & (b2 >= -1e-9)
                  & (b0 <= 1+1e-9) & (b1 <= 1+1e-9) & (b2 <= 1+1e-9))
        local = local[inside]
        if not len(local):
            continue
        b0, b1, b2, denominator = [value[inside] for value in (b0, b1, b2, denominator)]
        inside_count += len(local)
        z = depth[local]
        z = b0*z[:, 0]+b1*z[:, 1]+b2*z[:, 2]
        need(np.isfinite(z).all() and (z >= clip[0]).all() and (z <= clip[1]).all(),
             'Original clipped depth contract at the chosen pixel')
        at = int(np.argmin(z))
        face = int(first+local[at])
        value = float(z[at])
        if best is None or value < best['depth'] or (value == best['depth'] and face < best['face']):
            best = dict(face=face, depth=value,
                        barycentric=[float(b0[at]), float(b1[at]), float(b2[at])],
                        projected_signed_determinant=float(denominator[at]))
    return dict(hit=best, triangle_rows_scanned=len(faces),
                bbox_covering_triangle_rows=bbox_count, containing_triangle_rows=inside_count,
                maximum_face_batch=BATCH, full_raster_performed=False)


def triangle_key(np, faces, vertex_count):
    """Cyclic oriented index key; no coordinate welding or ancestry inference."""
    need(faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype == np.int32
         and type(vertex_count) is int and 0 < vertex_count <= 1_957_891
         and (faces >= 0).all() and (faces < vertex_count).all(), 'Bounded original index triples')
    at = np.argmin(faces, axis=1)
    row = np.arange(len(faces))
    a, b, c = [faces[row, (at+offset) % 3].astype(np.uint64) for offset in (0, 1, 2)]
    return (a*np.uint64(vertex_count)+b)*np.uint64(vertex_count)+c


def source_membership(np, source_faces, triples, vertex_count):
    keys = {}
    result = {}
    for name, values in triples.items():
        triple = np.asarray(values, dtype=np.int32).reshape(1, 3)
        keys[name] = (triangle_key(np, triple, vertex_count)[0],
                      triangle_key(np, triple[:, [0, 2, 1]], vertex_count)[0])
        result[name] = dict(oriented_match_count=0, reverse_match_count=0,
                            oriented_original_face_ids=[], reverse_original_face_ids=[])
    for first in range(0, len(source_faces), BATCH):
        key = triangle_key(np, source_faces[first:first+BATCH], vertex_count)
        for name, (forward, reverse) in keys.items():
            for label, target in (('oriented', forward), ('reverse', reverse)):
                rows = np.flatnonzero(key == target)+first
                result[name][label+'_match_count'] += len(rows)
                ids = result[name][label+'_original_face_ids']
                ids.extend(int(value) for value in rows[:max(0, 16-len(ids))])
    for value in result.values():
        value['no_oriented_original_triple_match'] = value['oriented_match_count'] == 0
        value['face_id_list_limit'] = 16
        value['collapse_ancestry_inferred'] = False
    return result


def local_adjacency(np, data, cv, cf, ids, triples):
    """Reuse bounded endpoint-star scans and authenticate cached edge counts."""
    import common as c
    from topology_witness import local_faces, incidence, face_records
    v, f = data['vertices'], data['faces']
    old_keys = np.load(c.original.CACHE/'edge_keys.npy', mmap_mode='r', allow_pickle=False)
    old_inc = np.load(c.original.CACHE/'incidence.npy', mmap_mode='r', allow_pickle=False)
    new_keys = np.load(POLD/'assess/cache/edge_keys.npy', mmap_mode='r', allow_pickle=False)
    new_inc = np.load(POLD/'assess/cache/incidence.npy', mmap_mode='r', allow_pickle=False)
    source_endpoints = sorted({int(value) for triple in triples.values() for value in triple})
    compact = {}
    for value in source_endpoints:
        at = int(np.searchsorted(ids, value))
        if at < len(ids) and int(ids[at]) == value:
            compact[value] = at
    original_star = local_faces(f, source_endpoints)
    candidate_star = local_faces(cf, list(compact.values()))
    result = {}
    for name, values in triples.items():
        values = list(map(int, values))
        original_rows = original_star[np.isin(f[original_star], values).any(1)]
        retained = [compact[value] for value in values if value in compact]
        candidate_rows = candidate_star[np.isin(cf[candidate_star], retained).any(1)]
        row = dict(endpoint_star_collection_limit=32768,
                   original_endpoint_star_count=len(original_rows), candidate_endpoint_star_count=len(candidate_rows),
                   original_endpoint_star_face_ids=original_rows[:256].tolist(),
                   candidate_endpoint_star_face_ids=candidate_rows[:256].tolist(),
                   endpoint_star_face_id_list_limit=256,
                   endpoint_star_ID_lists_truncated=bool(len(original_rows) > 256 or len(candidate_rows) > 256), edges=[])
        for first, second in zip(values, values[1:]+values[:1]):
            old = original_star[(f[original_star] == first).any(1) & (f[original_star] == second).any(1)]
            old_count = incidence(old_keys, old_inc, first, second, len(v))
            new = np.empty(0, dtype=np.int32)
            new_count = 0
            if first in compact and second in compact:
                a, b = compact[first], compact[second]
                new = candidate_star[(cf[candidate_star] == a).any(1) & (cf[candidate_star] == b).any(1)]
                new_count = incidence(new_keys, new_inc, a, b, len(cv))
            need(len(old) == old_count and len(new) == new_count,
                 'Local witness edge adjacency reproduces authenticated source/candidate incidence')
            row['edges'].append(dict(source_endpoints=[first, second],
                                     original_cached_incidence=old_count, candidate_cached_incidence=new_count,
                                     original_incident_face_count=len(old), candidate_incident_face_count=len(new),
                                     original_incident_faces=face_records(v, f, old[:16]),
                                     candidate_incident_faces=face_records(cv, cf, new[:16], ids),
                                     incident_face_record_limit=16,
                                     incident_records_truncated=bool(len(old) > 16 or len(new) > 16)))
        result[name] = row
    return result


def face_detail(np, vertices, face, source_ids, native_bytes, barycentric, camera=None):
    xyz = vertices[face].astype(np.float64)
    vector = np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0])
    twice_area = float(np.linalg.norm(vector))
    need(np.isfinite(xyz).all() and twice_area > 0, 'Finite nondegenerate witness face')
    normal = vector/twice_area
    bary = np.asarray(barycentric, dtype=np.float64)
    need(bary.shape == (3,) and np.isfinite(bary).all(), 'Existing witness barycentrics')
    decoded = decode(np, native_bytes[:, :3])
    linear = np.einsum('ni,nij->nj', bary.reshape(1, 3), decoded.astype(np.float64)[None])[0]
    result = dict(source_vertex_ids=[int(value) for value in source_ids],
                  vertex_world_F32=vertices[face].tolist(),
                  flat_world_normal=normal.tolist(), area=twice_area*.5,
                  native_corner_RGBA_bytes=native_bytes.tolist(),
                  decoded_corner_linear_RGB_F32=decoded.tolist(),
                  barycentric=bary.tolist(), interpolated_linear_RGB_F64=linear.tolist(),
                  interpolated_encoded_RGB_F64=encode(np, linear).tolist(),
                  material_factors=MATERIAL_FACTORS.copy(),
                  material_semantics='Original accepted native-byte oracle; F32 IEC decode followed by F64 linear interpolation; no original raw material rows or new source-field query')
    if camera is not None:
        result['normal_dot_camera_outward'] = float(np.dot(normal, basis(np, camera)[3]))
        result['opaque_two_sided'] = True
    return result


def _saved(np, directory, name, dtype, shape):
    value = np.load(directory/name, mmap_mode='r', allow_pickle=False)
    need(value.dtype == dtype and value.shape == shape, 'Exact saved array schema: '+name)
    return value


def _mask_crop(np, out, source_mask, candidate_mask, foliage, row, column):
    """Private diagnostic of existing buffers; this is not a shaded render."""
    from PIL import Image, ImageDraw
    left, top = max(0, min(576, column-32)), max(0, min(576, row-32))
    crop = (slice(top, top+64), slice(left, left+64))
    source = np.repeat((source_mask[crop].astype(np.uint8)*210)[:, :, None], 3, axis=2)
    source[foliage[crop]] = [30, 180, 70]
    candidate = np.repeat((candidate_mask[crop].astype(np.uint8)*210)[:, :, None], 3, axis=2)
    sheet = Image.new('RGB', (444, 264), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 5), 'Diagnostic saved masks; not a Blender render', fill='black')
    for x, pixels, label in ((8, source, 'Original: green=visible foliage'),
                              (236, candidate, 'Rejected candidate silhouette')):
        image = Image.fromarray(pixels).resize((192, 192), Image.Resampling.NEAREST)
        sheet.paste(image, (x, 42))
        draw.text((x, 25), label, fill='black')
        px, py = x+(column-left)*3+1, 42+(row-top)*3+1
        draw.line((px-5, py, px+5, py), fill='red', width=1)
        draw.line((px, py-5, px, py+5), fill='red', width=1)
    draw.text((8, 245), 'Red cross=one frozen failed pixel; nearest-neighbour display x3', fill='black')
    path = out/'projection-witness-mask-crop.png'
    with path.open('xb') as stream:
        sheet.save(stream, format='PNG')
        stream.flush()
        os.fsync(stream.fileno())
    with Image.open(path) as image:
        image.load()
        need(image.format == 'PNG' and image.mode == 'RGB' and image.size == sheet.size,
             'Private diagnostic crop reopening')
    return path


def run(np, data, cv, cf, ids, rgba, out):
    """Called only by the separately reviewed owned CPU worker, never standalone."""
    out = Path(out)
    need(out.is_absolute() and PNEW in out.parents and out.is_dir(), 'New private attribution output only')
    source_vertices, source_faces = data['vertices'], data['faces']
    need(cv.dtype == np.float32 and cf.dtype == ids.dtype == np.int32
         and rgba.dtype == np.uint8 and len(cv) == len(ids) == len(rgba)
         and rgba.shape == (len(ids), 4), 'Authenticated compact candidate tables')
    need(source_vertices.shape == (1_957_891, 3) and source_faces.shape == (4_173_584, 3)
         and len(cv) == 975633 and len(cf) == 2083776, 'Exact original and rejected candidate sizes')
    assessment = json.loads((POLD/'assess/result.json').read_text())
    need(assessment['classification'] == 'NO-GO'
         and list(assessment['colour_regions']) == list(data['selected']),
         'Original failed assessment and frozen region order')
    directory = POLD/'assess'
    chosen = _saved(np, directory, 'query-faces.npy', np.int32, (2048,))
    bary = _saved(np, directory, 'query-barycentric.npy', np.float64, (2048, 3))
    world = _saved(np, directory, 'query-world.npy', np.float32, (2048, 3))
    source = _saved(np, directory, 'query-source.npy', np.float32, (2048, 6))
    predicted = _saved(np, directory, 'query-predicted.npy', np.float64, (2048, 3))
    selected = select_material_query(np, source, predicted, assessment['colour_regions'])
    q = selected['query']
    face_id = int(chosen[q])
    need(0 <= face_id < len(cf), 'Saved candidate face ID')
    face = cf[face_id]
    source_ids = ids[face]
    need(np.array_equal(cv[face].view(np.uint32), source_vertices[source_ids].view(np.uint32))
         and np.array_equal(rgba[face], data['native_bytes'][source_ids]),
         'Exact accepted native-byte oracle and original position/source IDs')
    detail = face_detail(np, cv, face, source_ids, rgba[face], bary[q])
    reproduced = np.asarray(detail['interpolated_encoded_RGB_F64'], dtype=np.float64)
    rgb_reproduction = float(np.abs(reproduced-predicted[q]).max())
    need(rgb_reproduction <= RGB_REPRODUCTION_TOLERANCE, 'Reproduce saved encoded RGB from native bytes and barycentrics')
    reproduced_world = np.einsum('ni,nij->nj', bary[q:q+1], cv[cf[chosen[q:q+1]]].astype(np.float64)).astype(np.float32)[0]
    need(np.array_equal(reproduced_world.view(np.uint32), world[q].view(np.uint32)), 'Reproduce the existing saved query world F32 bits')
    error = reproduced-np.clip(source[q, :3].astype(np.float64), 0, 1)
    material = dict(selection=selected, candidate_face_id=face_id, face=detail,
                    saved_query_world_F32=world[q].tolist(), query_world_F32_bits_reproduced=True,
                    saved_source_field_six_channels=source[q].tolist(),
                    saved_predicted_encoded_RGB_F64=predicted[q].tolist(),
                    encoded_RGB_signed_error=error.tolist(),
                    encoded_RGB_reproduction_max_abs=rgb_reproduction,
                    encoded_RGB_reproduction_tolerance=RGB_REPRODUCTION_TOLERANCE,
                    field_was_resampled=False, new_query_locations_generated=False,
                    causal_scope='This reproduces the saved candidate face-interior residual and explains how retained endpoint bytes interpolate. No paired original-surface material evaluation exists at this same point; the total residual cannot be attributed solely to collapse rather than pre-existing material approximation.')
    print('SAVED_MATERIAL_WITNESS_REPRODUCED', flush=True)
    buffers = {}
    selected_pixel = None
    for view in ('pos-y', 'pos-z'):
        source_mask = _saved(np, directory, view+'-source-mask.npy', np.bool_, (640, 640))
        candidate_mask = _saved(np, directory, view+'-candidate-mask.npy', np.bool_, (640, 640))
        source_depth = _saved(np, directory, view+'-source-depth.npy', np.float64, (640, 640))
        candidate_depth = _saved(np, directory, view+'-candidate-depth.npy', np.float64, (640, 640))
        leaf = _saved(np, directory, view+'-foliage-visible.npy', np.bool_, (640, 640))
        tolerance = assessment['projections'][view]['foliage_depth_tolerance']
        need(np.isfinite(source_depth).all() and np.isfinite(candidate_depth).all()
             and not (leaf & ~source_mask).any(), 'Finite saved depths and visible source foliage')
        failed = leaf & (~candidate_mask | (np.abs(candidate_depth-source_depth) > tolerance))
        rows = np.flatnonzero(failed.ravel())
        need(len(rows) == assessment['projections'][view]['foliage_required_pixels']
             - assessment['projections'][view]['foliage_retained_pixels'], 'Saved failed foliage-pixel population')
        buffers[view] = (source_mask, candidate_mask, source_depth, candidate_depth, leaf, tolerance, len(rows))
        if len(rows) and selected_pixel is None:
            selected_pixel = (view, int(rows[0]//640), int(rows[0] % 640))
    need(selected_pixel is not None, 'At least one saved failed Y/Z foliage pixel')
    view, row, column = selected_pixel
    source_mask, candidate_mask, source_depth, candidate_depth, leaf, tolerance, failed_count = buffers[view]
    camera = json.loads((POLD/'baseline/views.json').read_text())[view]
    original_scan = local_hit(np, source_vertices, source_faces, camera, row, column)
    candidate_scan = local_hit(np, cv, cf, camera, row, column)
    old_hit, new_hit = original_scan['hit'], candidate_scan['hit']
    need(old_hit is not None and bool(source_mask[row, column])
         and abs(old_hit['depth']-float(source_depth[row, column])) <= DEPTH_REPRODUCTION_TOLERANCE,
         'Local original nearest-hit depth matches the existing saved pixel')
    need(old_hit['face'] in set(int(value) for value in data['foliage']),
         'Original nearest hit belongs to the authenticated foliage ROI')
    need((new_hit is not None) == bool(candidate_mask[row, column]), 'Local candidate hit/miss matches the saved pixel mask')
    if new_hit is not None:
        need(abs(new_hit['depth']-float(candidate_depth[row, column])) <= DEPTH_REPRODUCTION_TOLERANCE,
             'Local candidate nearest-hit depth matches the existing saved pixel')
    old_face = source_faces[old_hit['face']]
    old_hit['face_detail'] = face_detail(np, source_vertices, old_face, old_face,
                                        data['native_bytes'][old_face], old_hit['barycentric'], camera)
    triples = dict(material_candidate=source_ids, pixel_original=old_face)
    if new_hit is not None:
        new_face = cf[new_hit['face']]
        new_ids = ids[new_face]
        need(np.array_equal(rgba[new_face], data['native_bytes'][new_ids]), 'Candidate pixel-hit native bytes retain exact original IDs')
        new_hit['face_detail'] = face_detail(np, cv, new_face, new_ids, rgba[new_face],
                                            new_hit['barycentric'], camera)
        triples['pixel_candidate'] = new_ids
    membership = source_membership(np, source_faces, triples, len(source_vertices))
    adjacency = local_adjacency(np, data, cv, cf, ids, triples)
    old_triangle = membership['material_candidate']
    interpolation_class = ('unchanged_original_oriented_triangle_interpolation' if old_triangle['oriented_match_count'] else
                           'same_original_triangle_corner_set_with_reverse_winding' if old_triangle['reverse_match_count'] else
                           'new_candidate_source_ID_triple_interpolation_unpaired_with_original_surface')
    material['interpolation_classification'] = interpolation_class
    material['interpretation'] = ('The exact native-colour triangle already existed; this saved point reveals pre-existing face-interior approximation, not newly introduced colour interpolation.'
                                  if old_triangle['oriented_match_count'] or old_triangle['reverse_match_count'] else
                                  'The candidate joins a new source-ID triple. Its retained endpoint bytes reproduce this residual; without a paired original-surface evaluation at the same point, the whole residual cannot be assigned to collapse.')
    material['face']['original_source_triple_membership'] = membership['material_candidate']
    material['face']['local_adjacency'] = adjacency['material_candidate']
    old_hit['face_detail']['original_source_triple_membership'] = membership['pixel_original']
    old_hit['face_detail']['local_adjacency'] = adjacency['pixel_original']
    if new_hit is not None:
        new_hit['face_detail']['original_source_triple_membership'] = membership['pixel_candidate']
        new_hit['face_detail']['local_adjacency'] = adjacency['pixel_candidate']
    delta = None if new_hit is None else new_hit['depth']-old_hit['depth']
    reason = ('missing_candidate_coverage' if new_hit is None else
              'candidate_foreground_first_hit_occludes_original_foliage_depth' if delta < -tolerance else
              'candidate_first_hit_is_behind_original_foliage_depth')
    projection = dict(selection='Lexicographically first (view name,row,column) saved failed foliage pixel across pos-y and pos-z',
                      view=view, pixel_row=row, pixel_column=column, pixel_centre=[column+.5, row+.5],
                      camera=camera, saved_source_depth=float(source_depth[row, column]),
                      saved_candidate_depth=float(candidate_depth[row, column]),
                      candidate_mask=bool(candidate_mask[row, column]),
                      original_visible_foliage=True, failure_reason=reason,
                      signed_candidate_minus_original_depth=delta,
                      absolute_depth_difference=None if new_hit is None else abs(new_hit['depth']-old_hit['depth']),
                      frozen_depth_tolerance=float(tolerance),
                      depth_reproduction_tolerance=DEPTH_REPRODUCTION_TOLERANCE,
                      original_local_scan=original_scan, candidate_local_scan=candidate_scan,
                      all_other_pixels_not_rasterized=True, new_source_field_queries=0,
                      limitation='One existing opaque two-sided pixel-centre witness. Proximity to a face edge and barycentric tolerance are pixel-centre limitations, not evidence of a raster bug. Local triangle membership/normal/depth is not collapse ancestry, a global orientation guarantee, continuous coverage proof or a shaded render.')
    save(out/'material-witness.json', material)
    save(out/'projection-witness.json', projection)
    crop_path = _mask_crop(np, out, source_mask, candidate_mask, leaf, row, column)
    print('SAVED_PROJECTION_PIXEL_WITNESS_REPRODUCED', flush=True)
    names = ('material-witness.json', 'projection-witness.json', crop_path.name)
    return dict(classification='PASS', candidate_acceptance_remains='NO-GO',
                material=dict(region=selected['region'], saved_query_row=q,
                              candidate_face_id=face_id, maximum_error_channel=('R', 'G', 'B')[selected['channel']],
                              maximum_absolute_encoded_RGB_error=selected['maximum'],
                              saved_prediction_reproduction_max_abs=rgb_reproduction,
                              original_source_oriented_triple_match_count=membership['material_candidate']['oriented_match_count'],
                              original_source_reverse_triple_match_count=membership['material_candidate']['reverse_match_count'],
                              original_native_colour_source_ID_association_exact=True,
                              interpolation_classification=interpolation_class,
                              same_point_original_surface_material_baseline_available=False,
                              total_residual_not_attributed_solely_to_collapse=True),
                projection=dict(view=view, original_face_id=old_hit['face'],
                                candidate_face_id=None if new_hit is None else new_hit['face'],
                                failure_reason=reason, failed_saved_foliage_pixels_in_view=failed_count,
                                source_and_candidate_local_hits_match_saved_masks_and_depths=True,
                                original_nearest_hit_is_authenticated_foliage=True,
                                absolute_depth_difference=projection['absolute_depth_difference'],
                                frozen_depth_tolerance=float(tolerance),
                                source_triangle_rows_scanned=len(source_faces), candidate_triangle_rows_scanned=len(cf)),
                source_field_queries=0, full_raster_calls=0, Blender_calls=0,
                no_new_candidate_or_quality_locations=True,
                private_geometry_material_arrays_and_diagnostic_image_not_in_public_summary=True,
                files={name:dict(bytes=(out/name).stat().st_size, sha256=digest(out/name)) for name in names})
