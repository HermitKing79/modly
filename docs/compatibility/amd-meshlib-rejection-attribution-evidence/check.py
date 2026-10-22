"""Independent saved-data witness verification; no native or analysis imports."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from collections import Counter, defaultdict
import math
import struct
import time
import traceback
E = Path(__file__).resolve().parent


def need(ok, message):
    if not ok: raise ValueError(message)


def _same(actual, expected, where='record'):
    """Exact typed JSON equality, including Float64 bits and signed zero."""
    need(type(actual) is type(expected), where + ': type')
    if isinstance(expected, dict):
        need(actual.keys() == expected.keys(), where + ': keys')
        for key in expected: _same(actual[key], expected[key], where + '.' + key)
    elif isinstance(expected, list):
        need(len(actual) == len(expected), where + ': length')
        for i, value in enumerate(expected): _same(actual[i], value, where + '[' + str(i) + ']')
    elif isinstance(expected, float):
        need(math.isfinite(actual) and math.isfinite(expected)
             and struct.pack('<d', actual) == struct.pack('<d', expected), where + ': Float64 bits')
    else: need(actual == expected, where + ': value')
    return True


LINK_SCOPE = 'Descriptive triangle-link multigraph only; not native chain pairing or hasRepeatedVerts().'
INTERPRETATION = 'Saved local incidence and actual native adjacency observations only; no insertion trace or scheduling is available. No repair, duplication, welding, reordering or fallback is supported.'
CONSTRAINT = 'MeshBuilder documents acceptance of a single chain/ring, no triangles, OR two open chains. Descriptive link components are not its internal chains/triangle-pairs, and do not implement duplicationNeeded or hasRepeatedVerts.'
PREFIX = 'Prefix means original input FaceIds strictly below this witness, including any earlier rejected faces; it is not observed native insertion state.'
EDGE_CAVEAT = 'Boundary/manifold/nonmanifold labels here mean only undirected source-endpoint incidence1/2/>2; they do not certify an oriented sheet partition.'


def link_evidence(faces, vertex, face_ids):
    """Reconstruct components by disjoint sets and multigraph degree counters."""
    parent, degrees, directed, undirected = {}, Counter(), Counter(), Counter()
    usage, ambiguous = [], []
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for fid in face_ids:
        tri = [int(x) for x in faces[fid]]
        if tri.count(vertex) != 1:
            ambiguous.append(fid); continue
        at = tri.index(vertex)
        a, b = tri[(at + 1) % 3], tri[(at + 2) % 3]
        for x in (a, b): parent.setdefault(x, x); degrees[x] += 1
        ra, rb = find(a), find(b)
        parent[max(ra, rb)] = min(ra, rb)
        directed[(a, b)] += 1; undirected[tuple(sorted((a, b)))] += 1
        usage.append(dict(face_id=fid, directed_link_vertices=[a, b]))
    members = defaultdict(list)
    for node in sorted(parent): members[find(node)].append(node)
    components = []
    for nodes in sorted(members.values(), key=lambda x: x[0]):
        ds = [degrees[x] for x in nodes]; endpoints = ds.count(1)
        components.append(dict(vertices=nodes, node_count=len(nodes), edge_occurrences=sum(ds) // 2,
                               endpoints=endpoints, open_chain=endpoints == 2 and all(x in (1, 2) for x in ds),
                               closed_chain=bool(ds) and all(x == 2 for x in ds), branching=max(ds) > 2))
    return dict(component_count=len(components), open_chain_components=sum(x['open_chain'] for x in components),
                closed_chain_components=sum(x['closed_chain'] for x in components),
                branching_components=sum(x['branching'] for x in components), maximum_link_degree=max(degrees.values(), default=0),
                repeated_undirected_link_edges=sum(x > 1 for x in undirected.values()),
                co_directed_link_edge_excess=sum(x - 1 for x in directed.values()),
                ambiguous_repeated_central_vertex_faces=ambiguous, components=components, usage=usage, scope=LINK_SCOPE)


def check_arrays(np, original, saved, failed_ids, public, private):
    """Independently recompute every recorded witness and complete context."""
    need(set(original) == {'vertices', 'faces', 'source_vertex_ids', 'source_face_ids'}, 'Original array set')
    expected_saved = {'vertices_readback', 'faces_readback', 'valid_face_ids', 'valid_vertex_ids', 'failed_face_ids',
                      'vertices_after_native', 'faces_after_native', 'source_vertex_ids', 'source_face_ids',
                      'native_undirected_edge_ids', 'native_halfedge_ids', 'native_left_faces', 'native_right_faces'}
    need(set(saved) == expected_saved, 'Thirteen saved arrays')
    v, f = original['vertices'], original['faces']; n, nf = len(v), len(f)
    need(0 < n <= 4000 and 0 < nf <= 5000 and v.dtype == np.float32 and v.shape == (n, 3)
         and np.isfinite(v).all(), 'Bounded finite original positions')
    need(f.dtype == np.int32 and f.shape == (nf, 3) and np.all((f >= 0) & (f < n)), 'Original face range')
    sv, sf = original['source_vertex_ids'], original['source_face_ids']
    for ids, count in ((sv, n), (sf, nf)):
        need(ids.dtype.kind in 'iu' and ids.shape == (count,) and np.all(ids >= 0)
             and len(np.unique(ids)) == count, 'One-to-one original source IDs')
    failed = np.asarray(failed_ids)
    need(failed.dtype == np.int32 and failed.ndim == 1 and 0 < len(failed) <= 8
         and np.array_equal(failed, np.unique(failed)) and np.all((failed >= 0) & (failed < nf)), 'Failed face range/order')
    need(saved['failed_face_ids'].dtype == np.int32 and np.array_equal(saved['failed_face_ids'], failed), 'Saved failure partition')
    failed_set = set(int(x) for x in failed)
    actual_ids = saved['valid_face_ids']; rows = saved['faces_readback']; valid_vertices = saved['valid_vertex_ids']
    kept_ids = [i for i in range(nf) if i not in failed_set]; kept = set(kept_ids)
    need(actual_ids.dtype == np.int32 and actual_ids.shape == (len(kept_ids),)
         and actual_ids.tolist() == kept_ids, 'All noncontiguous retained FaceIds')
    need(rows.dtype == np.int32 and rows.shape == (len(kept_ids), 3), 'Native face row schema')
    face_rows = f.tolist(); native_rows = rows.tolist()
    for row, fid in zip(native_rows, kept_ids):
        tri = face_rows[fid]
        need(any(row == tri[shift:] + tri[:shift] for shift in range(3)), 'Retained original winding')
    need(valid_vertices.dtype == np.int32 and valid_vertices.ndim == 1
         and np.array_equal(valid_vertices, np.unique(rows)), 'Valid native vertices equal surviving references')
    for name, source in [('vertices_after_native', v), ('faces_after_native', f), ('source_vertex_ids', sv), ('source_face_ids', sf)]:
        value = saved[name]
        need(value.dtype == source.dtype and value.shape == source.shape and value.tobytes() == source.tobytes(), 'Unchanged saved source bytes')
    positions = saved['vertices_readback']
    need(positions.dtype in (np.dtype('float32'), np.dtype('float64')) and positions.shape == v.shape
         and positions.tobytes() == v.astype(positions.dtype).tobytes(), 'Every allocated position bit')

    # Independent source incidence index, with stable original face/corner ordering.
    incidence, stars = defaultdict(list), defaultdict(set)
    for fid, tri in enumerate(face_rows):
        for vertex in tri: stars[vertex].add(fid)
        for corner in range(3):
            a, b = tri[corner], tri[(corner + 1) % 3]
            incidence[(min(a, b), max(a, b))].append((fid, corner, a, b))
    # Explicit Float64 arithmetic is identical to the declared measurement definition;
    # graph construction and all record comparisons are independently implemented.
    p = v.astype(np.float64)
    cross = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    length = np.linalg.norm(cross, axis=1)
    normals = [cross[i] / length[i] if length[i] > 0 else None for i in range(nf)]
    need(np.isfinite(cross).all() and np.isfinite(length).all(), 'Finite measured Float64 geometry')
    native = defaultdict(lambda: defaultdict(list)); owner = {}; seen_halfedges = set()
    arrays = [saved[name] for name in ('native_undirected_edge_ids', 'native_halfedge_ids', 'native_left_faces', 'native_right_faces')]
    for a in arrays: need(a.dtype == np.int32 and a.shape == rows.shape, 'Aligned native metadata')
    for ri, tri in enumerate(native_rows):
        fid = kept_ids[ri]
        for corner in range(3):
            a, b = tri[corner], tri[(corner + 1) % 3]; pair = (min(a, b), max(a, b))
            eid, hid, left, right = (int(array[ri, corner]) for array in arrays)
            need(eid >= 0 and hid >= 0 and hid not in seen_halfedges and left == fid
                 and (right == -1 or right in kept), 'Native halfedge identity/face ownership')
            seen_halfedges.add(hid)
            need(eid not in owner or owner[eid] == pair, 'Native edge owns one original endpoint pair')
            owner[eid] = pair
            native[pair][eid].append(dict(face_id=fid, source_face_id=int(sf[fid]), oriented_local_vertices=[a, b],
                oriented_source_vertices=[int(sv[a]), int(sv[b])], halfedge_id=hid, left_face_id=left, right_face_id=right))
    for groups in native.values():
        for group in groups.values():
            need(len(group) in (1, 2), 'At most two actual native face sides')
            if len(group) == 1: need(group[0]['right_face_id'] == -1, 'Native singleton boundary')
            else:
                a, b = group
                need(a['face_id'] != b['face_id'] and a['oriented_local_vertices'] == b['oriented_local_vertices'][::-1]
                     and a['right_face_id'] == b['face_id'] and b['right_face_id'] == a['face_id'], 'Native opposite reciprocal pairing')
    context = set()
    def geometry(fid):
        return dict(local_face_id=fid, source_face_id=int(sf[fid]), local_vertex_ids=face_rows[fid],
                    source_vertex_ids=[int(sv[x]) for x in face_rows[fid]], coordinates=p[f[fid]].tolist(),
                    area=float(length[fid] / 2), unit_normal=None if normals[fid] is None else normals[fid].tolist())
    def edge(pair, witness=None, oriented=None):
        uses = incidence[pair]; context.update(item[0] for item in uses); usage = []
        for fid, corner, a, b in uses:
            dot = None if witness is None or normals[witness] is None or normals[fid] is None else float(np.dot(normals[witness], normals[fid]))
            usage.append(dict(face_id=fid, source_face_id=int(sf[fid]), corner=corner, oriented_local_vertices=[a, b],
                              oriented_source_vertices=[int(sv[a]), int(sv[b])], direction=1 if (a, b) == pair else -1 if a != b else 0,
                              retained=fid in kept, before_witness=None if witness is None else fid < witness, normal_dot_with_witness=dot))
        counts = Counter(row['direction'] for row in usage)
        return dict(local_endpoint_ids=list(pair), source_endpoint_ids=[int(sv[x]) for x in pair], original_incidence=len(uses),
                    original_class={1: 'boundary', 2: 'manifold'}.get(len(uses), 'nonmanifold'), original_usage=usage,
                    co_directed_conflict=max(counts.values()) > 1,
                    earlier_same_direction_face_ids=[fid for fid, _, a, b in uses if witness is not None and fid < witness and (a, b) == oriented],
                    retained_face_ids=sorted({fid for fid, _, _, _ in uses if fid in kept}),
                    native_groups=[dict(native_edge_id=eid, occurrences=group) for eid, group in sorted(native.get(pair, {}).items())])
    witness_public, witness_private = [], []
    for index, fid in enumerate(sorted(failed_set)):
        tri = face_rows[fid]; label = 'W' + str(index + 1)
        er = [edge(tuple(sorted((tri[k], tri[(k + 1) % 3]))), fid, (tri[k], tri[(k + 1) % 3])) for k in range(3)]
        vr = []
        for vertex in tri:
            ids = sorted(stars[vertex]); pre = [x for x in ids if x < fid]; retained = [x for x in ids if x in kept]
            context.update(ids)
            vr.append(dict(local_vertex_id=vertex, source_vertex_id=int(sv[vertex]), original_face_ids=ids, prefix_face_ids=pre,
                           retained_face_ids=retained, links={name: link_evidence(face_rows, vertex, pop)
                               for name, pop in (('original', ids), ('prefix', pre), ('retained', retained))}))
        wp = dict(label=label, distinct_vertices=len(set(tri)) == 3, finite_positions=True, positive_area=bool(length[fid] > 0),
                  original_edge_class_counts={name: sum(e['original_class'] == name for e in er) for name in ('boundary', 'manifold', 'nonmanifold')},
                  co_directed_original_edge_conflicts=sum(e['co_directed_conflict'] for e in er),
                  earlier_same_direction_edge_competitors=len({x for e in er for x in e['earlier_same_direction_face_ids']}),
                  earlier_edge_incident_faces=len({x['face_id'] for e in er for x in e['original_usage'] if x['face_id'] < fid}),
                  earlier_vertex_incident_faces=len({x for star in vr for x in star['prefix_face_ids']}), prefix_is_native_insertion_state=False)
        for scope in ('original', 'prefix', 'retained'):
            wp[scope + '_vertices_with_multiple_link_components'] = sum(star['links'][scope]['component_count'] > 1 for star in vr)
            wp[scope + '_vertices_with_branching_links'] = sum(star['links'][scope]['branching_components'] > 0 for star in vr)
        witness_public.append(wp)
        row = geometry(fid); row['original_local_face_id'] = row.pop('local_face_id'); row.update(label=label, edge_stars=er, vertex_stars=vr)
        witness_private.append(row)
    splits = []
    for pair in sorted(native):
        if len(native[pair]) > 1:
            row = edge(pair); row['label'] = 'S' + str(len(splits) + 1); splits.append(row)
    split_public = [dict(label=e['label'], original_incidence=e['original_incidence'], retained_incidence=len(e['retained_face_ids']),
                        native_edge_count=len(e['native_groups']), native_singleton_edges=sum(len(g['occurrences']) == 1 for g in e['native_groups']),
                        native_paired_edges=sum(len(g['occurrences']) == 2 for g in e['native_groups']),
                        original_co_directed_conflict=e['co_directed_conflict']) for e in splits]
    lost = []
    for vertex in sorted(set(stars) - set(int(x) for x in valid_vertices)):
        ids = sorted(stars[vertex]); retained = [x for x in ids if x in kept]; context.update(ids)
        lost.append(dict(local_vertex_id=vertex, source_vertex_id=int(sv[vertex]), coordinates=p[vertex].tolist(), original_face_ids=ids,
                         retained_face_ids=retained, removed_original_face_ids=[x for x in ids if x not in kept],
                         links=dict(original=link_evidence(face_rows, vertex, ids), retained=link_evidence(face_rows, vertex, retained))))
    expected_public = dict(classification='PASS', conclusion='CAUSE_UNRESOLVED_PARK', failed_face_count=len(failed), witnesses=witness_public,
                           lost_valid_vertex_count=len(lost), split_endpoint_pair_count=len(splits), split_pairs=split_public,
                           native_rejection_cause_proven=False, minimal_nondestructive_fix_supported=False, native_constructor_calls=0, remesh_calls=0,
                           interpretation=INTERPRETATION, constraint_caveat=CONSTRAINT, prefix_caveat=PREFIX, edge_class_caveat=EDGE_CAVEAT)
    expected_private = dict(witnesses=witness_private, split_endpoint_pairs=splits, lost_valid_vertices=lost,
                            context_faces=[geometry(fid) for fid in sorted(context)], prefix_definition=PREFIX, link_definition=CONSTRAINT)
    _same(public, expected_public, 'public'); _same(private, expected_private, 'private')
    return dict(classification='PASS', verification_completed=True, conclusion='CAUSE_UNRESOLVED_PARK',
                witnesses_verified=len(failed), split_pairs_verified=len(splits), lost_vertices_verified=len(lost),
                context_faces_verified=len(context), native_constructor_calls=0, remesh_calls=0,
                native_rejection_cause_proven=False, minimal_nondestructive_fix_supported=False,
                scope='Exact saved witnesses, source geometry/IDs, original/prefix/retained link multigraphs and observed native pairings; no native insertion trace or causal proof.')


def main():
    sys.path.insert(0, str(E))
    import common as c
    c.begin('verify'); began = time.monotonic(); out = c.P / 'verify'
    result = dict(classification='BLOCKED', verification_completed=False, native_constructor_calls=0, remesh_calls=0)
    try:
        c.authenticate()
        source = c.P / 'analysis/result.json'; record = c.read_json(source)
        host = c.read_json(c.P / 'analysis-host.json')
        need(record.get('classification') == 'PASS' and record.get('analysis_completed') is True, 'Completed saved-data analysis')
        need(host.get('classification') == 'PASS' and host.get('worker_receipt_sha256') == c.digest(source), 'Owned analysis receipt binding')
        extra = {'analysis_completed', 'elapsed_seconds', 'previous_constructor_result_SHA256', 'previous_independent_readback_SHA256', 'previous_private_witness_SHA256', 'files'}
        need(extra <= record.keys() and type(record['elapsed_seconds']) in (int, float) and 0 <= record['elapsed_seconds'] < 150, 'Analysis completion metadata')
        for key, path in [('previous_constructor_result_SHA256', c.PRIOR_P / 'import/result.json'),
                          ('previous_independent_readback_SHA256', c.PRIOR_P / 'verify/result.json'),
                          ('previous_private_witness_SHA256', c.PRIOR_P / 'verify/brick-rejected-source-memberships.json')]:
            need(record[key] == c.digest(path), 'Exact predecessor readback binding')
        need(set(record['files']) == {'witnesses.json'}, 'Single private witness manifest')
        path = c.P / 'analysis/witnesses.json'; c.plain(path); item = record['files']['witnesses.json']
        need(set(item) == {'bytes', 'sha256'} and type(item['bytes']) is int and 0 < item['bytes'] <= 2 * 1024 * 1024
             and path.stat().st_size == item['bytes'] and c.digest(path) == item['sha256'], 'Private witness bytes/hash')
        private = c.read_json(path)
        import numpy as np
        original, saved, failed = c.load_inputs(np)
        need(len(original['vertices']) == 457 and len(original['faces']) == 781 and len(failed) == 4, 'Exact original brick failure population')
        public = {k: value for k, value in record.items() if k not in extra}
        result.update(check_arrays(np, original, saved, failed, public, private))
        prior = c.read_json(c.PRIOR_P / 'verify/brick-rejected-source-memberships.json')
        need(prior['missing_native_face_ids'] == failed.tolist()
             and prior['missing_source_face_ids'] == original['source_face_ids'][failed].tolist(), 'Preserved independent rejection membership')
        need(time.monotonic() - began < 150, '150-second bounded independent verification')
        result.update(analysis_result_SHA256=c.digest(source), private_witness_SHA256=c.digest(path),
                      source_constraints_SHA256=c.digest(E / 'source-constraints.json'), elapsed_seconds=time.monotonic() - began)
    except BaseException as error:
        result.update(classification='BLOCKED', verification_completed=False, error=str(error), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - began)
    c.save(out / 'result.json', result)


if __name__ == '__main__': main()
