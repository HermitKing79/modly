"""Bounded observations of preserved constructor output; never imports MeshLib."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import time
import traceback
E = Path(__file__).resolve().parent


def require(ok, message):
    if not ok: raise ValueError(message)


def _incidence(f):
    edges, stars = {}, {}
    for face, tri in enumerate(f):
        for vertex in set(int(x) for x in tri): stars.setdefault(vertex, []).append(face)
        for corner in range(3):
            a, b = int(tri[corner]), int(tri[(corner + 1) % 3])
            edges.setdefault(tuple(sorted((a, b))), []).append((face, corner, a, b))
    return edges, stars


def _link(f, vertex, face_ids):
    """A descriptive oriented link multigraph, NOT MeshBuilder.VertInfo."""
    adjacency, usage, directed, ambiguous = {}, [], {}, []
    for face in face_ids:
        tri = [int(x) for x in f[face]]
        if tri.count(vertex) != 1:
            ambiguous.append(face)
            continue
        corner = tri.index(vertex); a, b = tri[(corner + 1) % 3], tri[(corner + 2) % 3]
        adjacency.setdefault(a, []).append(b); adjacency.setdefault(b, []).append(a)
        usage.append(dict(face_id=face, directed_link_vertices=[a, b]))
        directed[(a, b)] = directed.get((a, b), 0) + 1
    remaining = set(adjacency); components = []
    while remaining:
        todo = [min(remaining)]; reached = set()
        while todo:
            node = todo.pop()
            if node in reached: continue
            reached.add(node); todo.extend(adjacency[node])
        remaining.difference_update(reached)
        degree = [len(adjacency[x]) for x in reached]
        endpoints = sum(x == 1 for x in degree)
        closed = bool(degree) and all(x == 2 for x in degree)
        opened = endpoints == 2 and all(x in (1, 2) for x in degree)
        components.append(dict(vertices=sorted(reached), node_count=len(reached),
                               edge_occurrences=sum(degree) // 2, endpoints=endpoints,
                               open_chain=opened, closed_chain=closed,
                               branching=any(x > 2 for x in degree)))
    undirected = {}
    for item in usage:
        key = tuple(sorted(item['directed_link_vertices']))
        undirected[key] = undirected.get(key, 0) + 1
    return dict(component_count=len(components), open_chain_components=sum(x['open_chain'] for x in components),
                closed_chain_components=sum(x['closed_chain'] for x in components),
                branching_components=sum(x['branching'] for x in components),
                maximum_link_degree=max((len(x) for x in adjacency.values()), default=0),
                repeated_undirected_link_edges=sum(x > 1 for x in undirected.values()),
                co_directed_link_edge_excess=sum(max(x - 1, 0) for x in directed.values()),
                ambiguous_repeated_central_vertex_faces=ambiguous,
                components=components, usage=usage,
                scope='Descriptive triangle-link multigraph only; not native chain pairing or hasRepeatedVerts().')


def analyze_arrays(np, original, saved, failed_ids):
    """Return public scalar/flag observations and private exact local witnesses."""
    v, f = original['vertices'], original['faces']; N, F = len(v), len(f)
    require(0 < N <= 4000 and 0 < F <= 5000 and v.dtype == np.float32 and v.shape == (N, 3)
            and np.isfinite(v).all(), 'Bounded finite original F32 positions')
    require(f.dtype == np.int32 and f.shape == (F, 3) and (f >= 0).all() and (f < N).all(), 'Bounded original triangles')
    failed = np.asarray(failed_ids)
    require(failed.dtype == np.int32 and failed.ndim == 1 and 0 < len(failed) <= 8
            and np.array_equal(failed, np.unique(failed)) and (failed >= 0).all() and (failed < F).all(), 'Sorted bounded original failed FaceIds')
    require(np.array_equal(failed, saved['failed_face_ids']), 'The exact saved failed-face partition')
    vf, rf = saved['valid_face_ids'], saved['faces_readback']
    require(vf.dtype == np.int32 and vf.ndim == 1 and np.array_equal(vf, np.setdiff1d(np.arange(F, dtype=np.int32), failed)), 'Every retained original FaceId')
    require(rf.dtype == np.int32 and rf.shape == (len(vf), 3), 'Retained original face rows')
    require(all(any(np.array_equal(rf[row], np.roll(f[int(face)], -shift)) for shift in range(3))
                for row, face in enumerate(vf)), 'Retained oriented source triples')
    for name, source in [('vertices_after_native', v), ('faces_after_native', f),
                         ('source_vertex_ids', original['source_vertex_ids']), ('source_face_ids', original['source_face_ids'])]:
        value = saved[name]
        require(value.dtype == source.dtype and value.shape == source.shape and value.tobytes() == source.tobytes(), 'Preserved original heap/source-ID bytes')
    sv, sf = original['source_vertex_ids'], original['source_face_ids']
    require(sv.dtype.kind in 'iu' and sv.shape == (N,) and len(np.unique(sv)) == N and (sv >= 0).all(), 'Unique source vertex IDs')
    require(sf.dtype.kind in 'iu' and sf.shape == (F,) and len(np.unique(sf)) == F and (sf >= 0).all(), 'Unique source face IDs')
    rv = saved['vertices_readback']; require(rv.dtype in (np.dtype('float32'), np.dtype('float64')) and rv.shape == v.shape, 'Allocated position schema')
    require(rv.tobytes() == v.astype(rv.dtype).tobytes(), 'All allocated position bits are unchanged F32 or exact promotion')
    require(saved['valid_vertex_ids'].dtype == np.int32 and np.array_equal(saved['valid_vertex_ids'], np.unique(rf)), 'Actual valid vertices equal retained references')
    edges, stars = _incidence(f); retained = set(int(x) for x in vf)
    p = v.astype(np.float64); cross = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    twice_area = np.linalg.norm(cross, axis=1)
    normals = [cross[i] / twice_area[i] if twice_area[i] > 0 else None for i in range(F)]
    native, owners = {}, {}
    eid, hid, left, right = (saved[name] for name in ['native_undirected_edge_ids', 'native_halfedge_ids', 'native_left_faces', 'native_right_faces'])
    for value in (eid, hid, left, right): require(value.dtype == np.int32 and value.shape == rf.shape, 'Aligned actual native edge metadata')
    require((eid >= 0).all() and (hid >= 0).all() and len(np.unique(hid)) == 3 * len(rf), 'Unique actual native face-left halfedges')
    require(np.array_equal(left, np.repeat(vf[:, None], 3, axis=1)), 'Actual native left original FaceIds')
    require(all(int(x) == -1 or int(x) in retained for x in right.flat), 'Actual surviving right FaceIds or raw boundary')
    for row, tri in enumerate(rf):
        face = int(vf[row])
        for corner in range(3):
            a, b = int(tri[corner]), int(tri[(corner + 1) % 3]); pair = tuple(sorted((a, b))); edge = int(eid[row, corner])
            require(edge not in owners or owners[edge] == pair, 'Native edge endpoint ownership')
            owners[edge] = pair
            native.setdefault(pair, {}).setdefault(edge, []).append(dict(face_id=face, source_face_id=int(sf[face]),
                oriented_local_vertices=[a, b], oriented_source_vertices=[int(sv[a]), int(sv[b])],
                halfedge_id=int(hid[row, corner]), left_face_id=int(left[row, corner]), right_face_id=int(right[row, corner])))
    for groups in native.values():
        for group in groups.values():
            require(len(group) in (1, 2), 'Observed native edge has at most two face sides')
            if len(group) == 1: require(group[0]['right_face_id'] == -1, 'Actual native boundary')
            else:
                a, b = group
                require(a['face_id'] != b['face_id'] and a['oriented_local_vertices'] == list(reversed(b['oriented_local_vertices']))
                        and a['right_face_id'] == b['face_id'] and b['right_face_id'] == a['face_id'], 'Reciprocal actual native pairing')
    context_faces = set()

    def edge_record(pair, witness=None, oriented=None):
        usage = edges[pair]; context_faces.update(x[0] for x in usage)
        rows = []
        for face, corner, a, b in usage:
            dot = None
            if witness is not None and normals[witness] is not None and normals[face] is not None:
                dot = float(np.dot(normals[witness], normals[face]))
            rows.append(dict(face_id=face, source_face_id=int(sf[face]), corner=corner,
                             oriented_local_vertices=[a, b], oriented_source_vertices=[int(sv[a]), int(sv[b])],
                             direction=1 if (a, b) == pair else -1 if a != b else 0,
                             retained=face in retained, before_witness=(face < witness if witness is not None else None),
                             normal_dot_with_witness=dot))
        same = [x[0] for x in usage if witness is not None and x[0] < witness and (x[2], x[3]) == oriented]
        directions = [x['direction'] for x in rows]
        return dict(local_endpoint_ids=list(pair), source_endpoint_ids=[int(sv[x]) for x in pair],
                    original_incidence=len(usage), original_class='boundary' if len(usage) == 1 else 'manifold' if len(usage) == 2 else 'nonmanifold',
                    original_usage=rows, co_directed_conflict=any(directions.count(x) > 1 for x in set(directions)),
                    earlier_same_direction_face_ids=same,
                    retained_face_ids=sorted({x[0] for x in usage if x[0] in retained}),
                    native_groups=[dict(native_edge_id=key, occurrences=value) for key, value in sorted(native.get(pair, {}).items())])

    public_rows, private_rows = [], []
    for index, face_raw in enumerate(failed):
        face = int(face_raw); tri = [int(x) for x in f[face]]; label = 'W' + str(index + 1)
        edge_rows = [edge_record(tuple(sorted((tri[k], tri[(k + 1) % 3]))), face, (tri[k], tri[(k + 1) % 3])) for k in range(3)]
        vertex_rows = []
        for vertex in tri:
            full = stars[vertex]; prefix = [x for x in full if x < face]; kept = [x for x in full if x in retained]
            context_faces.update(full)
            vertex_rows.append(dict(local_vertex_id=vertex, source_vertex_id=int(sv[vertex]), original_face_ids=full,
                                    prefix_face_ids=prefix, retained_face_ids=kept,
                                    links={name: _link(f, vertex, ids) for name, ids in [('original', full), ('prefix', prefix), ('retained', kept)]}))
        classes = {name: sum(x['original_class'] == name for x in edge_rows) for name in ['boundary', 'manifold', 'nonmanifold']}
        earlier_same = {x for edge in edge_rows for x in edge['earlier_same_direction_face_ids']}
        earlier_edge = {x['face_id'] for edge in edge_rows for x in edge['original_usage'] if x['face_id'] < face}
        earlier_vertex = {x for vertex in vertex_rows for x in vertex['prefix_face_ids']}
        pub = dict(label=label, distinct_vertices=len(set(tri)) == 3, finite_positions=True,
                   positive_area=bool(twice_area[face] > 0), original_edge_class_counts=classes,
                   co_directed_original_edge_conflicts=sum(x['co_directed_conflict'] for x in edge_rows),
                   earlier_same_direction_edge_competitors=len(earlier_same), earlier_edge_incident_faces=len(earlier_edge),
                   earlier_vertex_incident_faces=len(earlier_vertex), prefix_is_native_insertion_state=False)
        for scope in ['original', 'prefix', 'retained']:
            pub[scope + '_vertices_with_multiple_link_components'] = sum(x['links'][scope]['component_count'] > 1 for x in vertex_rows)
            pub[scope + '_vertices_with_branching_links'] = sum(x['links'][scope]['branching_components'] > 0 for x in vertex_rows)
        public_rows.append(pub)
        private_rows.append(dict(label=label, original_local_face_id=face, source_face_id=int(sf[face]), local_vertex_ids=tri,
                                 source_vertex_ids=[int(sv[x]) for x in tri], coordinates=p[tri].tolist(), area=float(twice_area[face] / 2),
                                 unit_normal=normals[face].tolist() if normals[face] is not None else None,
                                 edge_stars=edge_rows, vertex_stars=vertex_rows))
    splits = [edge_record(pair) for pair, groups in sorted(native.items()) if len(groups) > 1]
    for index, edge in enumerate(splits): edge['label'] = 'S' + str(index + 1)
    public_splits = [dict(label=edge['label'], original_incidence=edge['original_incidence'],
                          retained_incidence=len(edge['retained_face_ids']), native_edge_count=len(edge['native_groups']),
                          native_singleton_edges=sum(len(group['occurrences']) == 1 for group in edge['native_groups']),
                          native_paired_edges=sum(len(group['occurrences']) == 2 for group in edge['native_groups']),
                          original_co_directed_conflict=edge['co_directed_conflict']) for edge in splits]
    lost = np.setdiff1d(np.unique(f), saved['valid_vertex_ids']); lost_rows = []
    for raw in lost:
        vertex = int(raw); full = stars[vertex]; kept = [x for x in full if x in retained]; context_faces.update(full)
        lost_rows.append(dict(local_vertex_id=vertex, source_vertex_id=int(sv[vertex]), coordinates=p[vertex].tolist(),
                              original_face_ids=full, retained_face_ids=kept,
                              removed_original_face_ids=[x for x in full if x not in retained],
                              links=dict(original=_link(f, vertex, full), retained=_link(f, vertex, kept))))
    table = [dict(local_face_id=face, source_face_id=int(sf[face]), local_vertex_ids=f[face].tolist(),
                  source_vertex_ids=[int(sv[x]) for x in f[face]], coordinates=p[f[face]].tolist(),
                  area=float(twice_area[face] / 2), unit_normal=normals[face].tolist() if normals[face] is not None else None)
             for face in sorted(context_faces)]
    public = dict(classification='PASS', conclusion='CAUSE_UNRESOLVED_PARK', failed_face_count=len(failed), witnesses=public_rows,
                  lost_valid_vertex_count=len(lost), split_endpoint_pair_count=len(splits), split_pairs=public_splits, native_rejection_cause_proven=False,
                  minimal_nondestructive_fix_supported=False, native_constructor_calls=0, remesh_calls=0,
                  interpretation='Saved local incidence and actual native adjacency observations only; no insertion trace or scheduling is available. No repair, duplication, welding, reordering or fallback is supported.',
                  constraint_caveat='MeshBuilder documents acceptance of a single chain/ring, no triangles, OR two open chains. Descriptive link components are not its internal chains/triangle-pairs, and do not implement duplicationNeeded or hasRepeatedVerts.',
                  prefix_caveat='Prefix means original input FaceIds strictly below this witness, including any earlier rejected faces; it is not observed native insertion state.',
                  edge_class_caveat='Boundary/manifold/nonmanifold labels here mean only undirected source-endpoint incidence1/2/>2; they do not certify an oriented sheet partition.')
    private = dict(witnesses=private_rows, split_endpoint_pairs=splits, lost_valid_vertices=lost_rows, context_faces=table,
                   prefix_definition=public['prefix_caveat'], link_definition=public['constraint_caveat'])
    return public, private


def main():
    sys.path.insert(0, str(E))
    import common as c
    c.begin('analysis'); began = time.monotonic(); out = c.P / 'analysis'
    result = dict(classification='BLOCKED', analysis_completed=False, native_constructor_calls=0, remesh_calls=0)
    try:
        c.authenticate()
        import numpy as np
        original, saved, failed = c.load_inputs(np)
        require(len(original['vertices']) == 457 and len(original['faces']) == 781 and len(failed) == 4, 'The sole authenticated brick failure population')
        public, private = analyze_arrays(np, original, saved, failed)
        prior = c.read_json(c.PRIOR_P / 'verify/brick-rejected-source-memberships.json')
        require(prior['missing_native_face_ids'] == failed.tolist()
                and prior['missing_source_face_ids'] == original['source_face_ids'][failed].tolist(), 'The exact preserved rejection witness')
        require(time.monotonic() - began < 150, '150-second bounded saved-data analysis')
        path = out / 'witnesses.json'; c.save(path, private)
        result.update(public, analysis_completed=True, elapsed_seconds=time.monotonic() - began,
                      previous_constructor_result_SHA256=c.digest(c.PRIOR_P / 'import/result.json'),
                      previous_independent_readback_SHA256=c.digest(c.PRIOR_P / 'verify/result.json'),
                      previous_private_witness_SHA256=c.digest(c.PRIOR_P / 'verify/brick-rejected-source-memberships.json'),
                      files={'witnesses.json': dict(bytes=path.stat().st_size, sha256=c.digest(path))})
    except BaseException as error:
        result.update(error=str(error), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - began)
    c.save(out / 'result.json', result)


if __name__ == '__main__': main()
