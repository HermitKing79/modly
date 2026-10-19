"""Read-only local boundary qualification; never constructs or writes a patch."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import time

E = Path(__file__).resolve().parent
ROOT = E.parents[2]
PREV = ROOT / 'docs/compatibility/amd-brick-defect-diagnosis-evidence'
P = ROOT.parent / 'artifacts/amd-brick-boundary-feasibility-677d741/recovery'
OLD = ROOT.parent / 'artifacts/amd-brick-defect-diagnosis-1d1c1a6/analysis/diagnostics.json'
OLD_SHA = '9a860a852afe7654f1cc1edd4754e01b6ec6c1912847c8b211d2562deca92ed5'
PINS = {'diagnose.py': '8fa7c617324ba636856b48f47a995ee64cd2a85f3c26d86d8760ec638b163529',
        'projection.py': '0ed20e6d03424404f3b3aebc64cae2825a0d1c8d241fd494ce50e742471eb840',
        'topology.py': '9d6e29e83b35c60f1f9b49192a290069bf867b51493da3cc5ec017525017d561'}

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def geometry_prerequisites(closed, nm_count, metrics):
    """Necessary loop checks only; never proves collision freedom or intent."""
    return bool(closed and nm_count == 0 and metrics and metrics.get('supported') and
                metrics.get('svd_rank', 0) >= 2 and metrics.get('planar_within_tolerance') and
                metrics.get('zero_length_segment_count') == 0 and
                metrics.get('projected_self_intersection') is False)

def main():
    print('BOUNDARY_QUALIFICATION_WAITING_GO', flush=True)
    if sys.stdin.buffer.readline(64) != b'GO\n':
        raise ValueError('Exact owned-host GO required')
    began = time.monotonic()
    for name, sha in PINS.items():
        if digest(PREV / name) != sha:
            raise ValueError('Changed original CPU diagnostic source: ' + name)
    if digest(OLD) != OLD_SHA:
        raise ValueError('Changed authenticated prior diagnostic')
    sys.path.insert(0, str(PREV))
    import diagnose as auth
    auth.P = P  # Only the new private diagnostic output is writable.
    sys.addaudithook(auth.runtime_guard)
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    import topology
    import projection
    sys.path.insert(0, str(E))
    from local_geometry import trace_component, multihit, loop_metrics, screen_components
    require = auth.require
    before = {name: auth.identity(path) for name, (path, _) in auth.PINS.items()}
    for name, (_, sha) in auth.PINS.items():
        require(before[name]['sha256'] == sha, 'Changed original input: ' + name)
    source_before = {str(path): auth.identity(path) for path in
                     (Path(__file__), E/'local_geometry.py', *(PREV/name for name in PINS), OLD)}
    prior = auth.read_json(OLD)
    require(prior['inputs_unchanged'] and prior['input_identities_before'] == before,
            'Original input identity differs from authenticated prior diagnosis')
    v = np.load(auth.PINS['vertices'][0], mmap_mode='r', allow_pickle=False)
    f = np.load(auth.PINS['faces'][0], mmap_mode='r', allow_pickle=False)
    require(v.shape == (1957891,3) and f.shape == (4173584,3) and
            not v.flags.writeable and not f.flags.writeable, 'Original readonly arrays required')
    material = auth.material_features(auth.PINS['material'][0], np)
    topo = topology.analyze(v, f, P/'cache')
    require(topo['edges']['boundary_incidence_1'] == 267652 and
            topo['edges']['nonmanifold_incidence_gt_2'] == 387068,
            'Original topology identity differs')
    keys = np.load(topo['private_cache']['edge_keys'], mmap_mode='r', allow_pickle=False)
    incidence = np.load(topo['private_cache']['incidence'], mmap_mode='r', allow_pickle=False)
    boundary_keys = keys[incidence == 1]
    pairs = np.column_stack((boundary_keys // len(v), boundary_keys % len(v))).astype(np.int32)
    nm_keys = keys[incidence > 2]
    nm_pairs = np.column_stack((nm_keys // len(v), nm_keys % len(v))).astype(np.int32)
    # Full original boundary graph: ROI selection never cuts chains into loops.
    rows = np.concatenate((pairs[:,0], pairs[:,1]))
    cols = np.concatenate((pairs[:,1], pairs[:,0]))
    graph = csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(len(v),len(v)))
    _, labels = connected_components(graph, directed=False)
    del graph, rows, cols
    pair_labels = labels[pairs[:,0]]
    screen = screen_components(pairs, labels, nm_pairs)
    projected = projection.project(v, prior['camera'], 1024, 1024)
    bounds = projection.build_triangle_bounds(f, projected)
    edge_xy = projected.screen[pairs]
    edge_lo, edge_hi = edge_xy.min(axis=1), edge_xy.max(axis=1)
    all_components, membership, cases = {}, {}, []

    def deadline():
        require(time.monotonic()-began < 165, 'Read-only analysis budget exhausted')

    def face_row(fid):
        tri = f[fid]
        xyz = np.asarray(v[tri], dtype=np.float64)
        normal = np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0])
        normal /= np.linalg.norm(normal)
        edges = tri[[[1,2],[2,0],[0,1]]]
        counts = projection.edge_counts(edges, (keys, incidence, len(v)))
        return dict(face_id=int(fid), original_vertex_ids=tri.tolist(),
                    edge_pairs_opposite_vertex=edges.tolist(), edge_incidences=counts.tolist(),
                    normal=normal.tolist(), normal_cos_camera=float(normal @ projected.outward),
                    projected_vertices=projected.screen[tri].tolist(),
                    vertex_depths=projected.depth[tri].tolist(),
                    original_learned_RGB=np.asarray(material[tri,:3]).tolist())

    def component(label):
        key = str(int(label))
        if key in all_components:
            return all_components[key]
        deadline()
        global_ids = np.flatnonzero(pair_labels == label)
        subpairs = pairs[global_ids]
        traced = trace_component(subpairs, subpairs[0])
        require(traced['edge_count'] == len(subpairs), 'Incomplete full-component trace')
        ids = np.asarray(traced['vertices'], dtype=np.int32)
        xyz = np.asarray(v[ids], dtype=np.float64)
        marked = np.zeros(len(v), dtype=np.bool_); marked[ids] = True
        touching_nm = np.flatnonzero(marked[nm_pairs].any(axis=1))
        degree_ids, degree_counts = np.unique(subpairs, return_counts=True)
        metrics = loop_metrics(v, traced['ordered_loop']) if traced['closed_simple'] else None
        summary = dict(component_label=int(label), edge_count=traced['edge_count'],
                       vertex_count=traced['vertex_count'], degree_histogram=traced['degree_histogram'],
                       closed_simple_graph_cycle=traced['closed_simple'],
                       bounds=[xyz.min(axis=0).tolist(),xyz.max(axis=0).tolist()],
                       projected_bounds=[projected.screen[ids].min(axis=0).tolist(),projected.screen[ids].max(axis=0).tolist()],
                       depth_range=[float(projected.depth[ids].min()),float(projected.depth[ids].max())],
                       boundary_length=float(np.linalg.norm(np.asarray(v[subpairs[:,1]],dtype=np.float64)-v[subpairs[:,0]],axis=1).sum()),
                       endpoint_vertex_ids=degree_ids[degree_counts==1].tolist(),
                       branching_vertex_ids=degree_ids[degree_counts>2].tolist(),
                       nonmanifold_edges_touching_component_vertices=len(touching_nm),
                       nonmanifold_edge_witnesses=nm_pairs[touching_nm[:16]].tolist(),
                       ordered_loop=traced['ordered_loop'], loop_metrics=metrics)
        membership[key] = dict(original_vertex_ids=traced['vertices'], original_boundary_edges=traced['edges'],
                               touching_nonmanifold_edge_keys=[int(x) for x in nm_keys[touching_nm]])
        summary['isolated_graph_candidate'] = bool(traced['closed_simple'] and len(touching_nm)==0)
        require(summary['isolated_graph_candidate'] == bool(screen['isolated_cycle_mask'][label]) and
                traced['closed_simple'] == bool(screen['closed_cycle_mask'][label]) and
                len(touching_nm) == int(screen['nm_touches'][label]),
                'Exact graph screen differs from full original component trace')
        summary['necessary_geometry_checks_passed'] = geometry_prerequisites(traced['closed_simple'], len(touching_nm), metrics)
        summary['repair_eligible'] = False
        summary['repair_status'] = 'BLOCKED: no closed boundary' if not traced['closed_simple'] else \
            'BLOCKED: component vertices touch nonmanifold junctions' if len(touching_nm) else \
            'UNPROVEN: spatial isolation, vertex links, surface collision, intent and material interpolation need review'
        all_components[key] = summary
        return summary

    def analyze_case(name, center, expected_face):
        deadline()
        x,y = center
        offsets = {(a,b) for a in range(-6,7,2) for b in range(-6,7,2)}
        offsets.update({(-3,0),(3,0),(0,-3),(0,3),(-12,-12),(-12,12),(12,-12),(12,12)})
        pixels = [[x+a,y+b] for a,b in sorted(offsets,key=lambda q:(q[1],q[0]))]
        rays = multihit(v,f,projected,pixels,bounds)
        for ray in rays:
            for hit in ray['hits']:
                tri = np.asarray(hit['original_vertex_ids'])
                hit['edge_incidences_opposite_vertex'] = projection.edge_counts(tri[[[1,2],[2,0],[0,1]]],(keys,incidence,len(v))).tolist()
                hit['original_learned_RGB'] = (np.asarray(hit['barycentric']) @ np.asarray(material[tri,:3],dtype=np.float64)).tolist()
        center_ray = next(ray for ray in rays if ray['pixel'] == [float(x),float(y)])
        require(center_ray['hits'] and center_ray['hits'][0]['face_id'] == expected_face,
                'Center ray differs from original diagnostic')
        tri = f[expected_face]
        marked = np.zeros(len(v),dtype=np.bool_);marked[tri]=True
        local_faces = np.flatnonzero(marked[f].any(axis=1))
        require(len(local_faces)<=4096,'Local face-star inspection bound')
        # Segment bounding boxes conservatively nominate every crossing boundary,
        # across all depth layers, including those with a midpoint outside the ROI.
        window=[x-12,y-12,x+12,y+12]
        selected = np.flatnonzero((edge_hi[:,0]>=window[0]) & (edge_lo[:,0]<=window[2]) &
                                  (edge_hi[:,1]>=window[1]) & (edge_lo[:,1]<=window[3]))
        face_edges = tri[[[0,1],[1,2],[2,0]]]
        face_counts = projection.edge_counts(face_edges,(keys,incidence,len(v)))
        direct_labels = [labels[edge[0]] for edge,count in zip(face_edges,face_counts) if count==1]
        selected_labels = np.unique(np.concatenate((pair_labels[selected], np.asarray(direct_labels,dtype=labels.dtype))))
        trace_labels = np.unique(np.concatenate((selected_labels[screen['isolated_cycle_mask'][selected_labels]],
                                                np.asarray(direct_labels,dtype=labels.dtype))))
        screening_rows = [dict(component_label=int(label),
                               vertices=int(screen['vertex_counts'][label]),
                               edges=int(screen['edge_counts'][label]),
                               endpoints=int(screen['endpoints'][label]),
                               branching_vertices=int(screen['branches'][label]),
                               nonmanifold_edge_touches=int(screen['nm_touches'][label]),
                               closed_graph_cycle=bool(screen['closed_cycle_mask'][label]),
                               isolated_graph_cycle=bool(screen['isolated_cycle_mask'][label]))
                          for label in selected_labels]
        # Exact necessary-test rejections do not consume a detailed-trace slot.
        # Every potentially admissible cycle and every direct-hit boundary is
        # retained; exceeding the original256 trace cap still stops unqualified.
        save(P/(name+'-selection.json'), dict(all_window_components=len(selected_labels),
             window_boundary_edges=len(selected),direct_component_labels=[int(i) for i in direct_labels],
             requested_complete_traces=len(trace_labels),trace_limit=256,
             discarded_potentially_admissible_cycles=0,all_graph_screening=screening_rows))
        direct_summaries = [component(label) for label in np.unique(np.asarray(direct_labels,dtype=labels.dtype))]
        save(P/(name+'-direct-evidence.json'),dict(case=name,pixel=center,
             direct_components=direct_summaries,
             hit_face_and_full_vertex_star=[face_row(i) for i in local_faces],rays=rays,
             scope='Durable direct-hit evidence before the unchanged complete-trace budget check; not a repair approval.'))
        require(len(trace_labels)<=256,'Potentially admissible/direct boundary trace budget exceeded')
        summaries=[component(label) for label in trace_labels]
        direct=[]
        for edge,count in zip(face_edges,face_counts):
            if count==1:
                direct.append(dict(edge=edge.tolist(),component_label=int(labels[edge[0]])))
        return dict(case=name,pixel=center,search_window=window,
                    search_scope='All original boundary segment bounding boxes intersecting the fixed24x24px window at any depth; every window component screened using full global graph counts. Direct-hit components and every potentially admissible cycle fully traced; no crop closure or discarded eligible cycle.',
                    local_boundary_edge_count=len(selected),component_labels=selected_labels.tolist(),
                    all_graph_screening=screening_rows, complete_trace_component_labels=trace_labels.tolist(),
                    isolated_graph_candidate_labels=[s['component_label'] for s in summaries if s['isolated_graph_candidate']],
                    surviving_necessary_geometry_candidate_labels=[s['component_label'] for s in summaries if s['necessary_geometry_checks_passed']],
                    direct_hit_boundary_edges=direct,hit_face_and_full_vertex_star=[face_row(i) for i in local_faces],
                    rays=rays,center_ray=center_ray,
                    closed_graph_cycles=int(screen['closed_cycle_mask'][selected_labels].sum()),
                    repair_eligible=False,
                    limits=['Ray samples cannot establish continuous coverage, lighting or intent.',
                            'A rear-hit boundary is not automatically the nearer missing-sheet boundary.',
                            'No patch or geometry was constructed. Self-intersection tests beyond boundary projection are not claimed.'])

    cases.append(analyze_case('A1',[534,372],2369396))
    # A3 is explicitly secondary. No automatic repair approval follows graph tests.
    if not cases[0]['surviving_necessary_geometry_candidate_labels']:
        cases.append(analyze_case('A3',[538,499],3356903))
    deadline()
    after = {name: auth.identity(path) for name,(path,_) in auth.PINS.items()}
    source_after={str(path):auth.identity(path) for path in (Path(__file__),E/'local_geometry.py',*(PREV/name for name in PINS),OLD)}
    require(before==after and source_before==source_after,'Original data or source changed')
    save(P/'boundary-membership.json',membership)
    result=dict(schema='modly.brick.local-boundary-qualification.v1',classification='MEASURED_PENDING_INDEPENDENT_REVIEW',
                base='677d741eba18baa667f726aa38ce2c4f52720a8d',CPU_only=True,
                input_identities_before=before,input_identities_after=after,originals_unchanged=True,
                source_identities_before=source_before,source_identities_after=source_after,
                topology=topo,components=all_components,cases=cases,
                membership_path=str(P/'boundary-membership.json'),membership_SHA256=digest(P/'boundary-membership.json'),
                resource_policy='Original6GiB/180s,12GiB live RAM/2GiB destination disk; no new model/renderer/WSL execution',
                spatial_intersection_scope='Boundary polygon self-intersection measured only for supported closed cycles; no full triangle-triangle collision certification or patch surface exists.',
                material_scope='Original row mappings authenticated; any hypothetical new face would introduce new interpolation not authenticated as original learned surface.',
                elapsed_seconds=time.monotonic()-began)
    save(P/'qualification.json',result)
    print(json.dumps(dict(status='MEASURED',path=str(P/'qualification.json'),sha256=digest(P/'qualification.json'),
                          seconds=result['elapsed_seconds'],cases=[c['case'] for c in cases])),flush=True)

if __name__=='__main__':
    main()
