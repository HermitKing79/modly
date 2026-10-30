"""Read-only saved-box forensics. Inert by default; CPU child has no neural route."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent / 'artifacts/amd-closed-solid-forensics-20261030-v1'
RAW = ROOT.parent / 'artifacts/amd-closed-solid-raw-recovery-v1'
EXECUTION = 'd9d733e447a25a638649b24a4bcbcaea51154fa1'
SOURCE_SHA = 'ea14fe7aa5ae513ff33b0f2da27b5d1c8018b7da0588790ee04805f4fa37f975'
PLY_SHA = '1317ee3e2f778487e89c3fd3effd5b59e44577f1d3ba13eed221eef8de8361b3'
UPSTREAM_SHA = '112bd474d580f279ac196129f7989df6fdafa7ed7e08c350f2b386553e73e17d'
OFFSETS = (((0,0,0),(0,0,1),(0,1,1),(0,1,0)),
           ((0,0,0),(1,0,0),(1,0,1),(0,0,1)),
           ((0,0,0),(0,1,0),(1,1,0),(1,0,0)))


def edge_table(polygons):
    """Independent lexicographic pair sort; no flattened vertex-key encoding."""
    import numpy as np
    a = np.concatenate([polygons[:, i] for i in range(polygons.shape[1])])
    b = np.concatenate([polygons[:, (i+1) % polygons.shape[1]] for i in range(polygons.shape[1])])
    direction = np.where(a < b, 1, -1).astype('i2')
    pairs = np.column_stack((np.minimum(a,b), np.maximum(a,b)))
    order = np.lexsort((pairs[:,1], pairs[:,0]))
    pairs = pairs[order]
    starts = np.r_[0, np.flatnonzero(np.any(pairs[1:] != pairs[:-1], axis=1)) + 1]
    counts = np.diff(np.r_[starts, len(pairs)]).astype('i4')
    return pairs[starts], counts, np.add.reduceat(direction[order], starts), order


def recover_quads(triangles):
    import numpy as np
    if len(triangles) % 2:raise ValueError('Paired triangles required')
    a,b = triangles[::2],triangles[1::2]
    one = (b[:,0] == a[:,0]) & (b[:,1] == a[:,2])
    two = (b[:,0] == a[:,2]) & (b[:,1] == a[:,1])
    if not np.all(one ^ two):raise ValueError('Original paired split pattern differs')
    return np.column_stack((a[:,0],a[:,1],np.where(one,a[:,2],b[:,2]),np.where(one,b[:,2],a[:,2])))


def tuple_quads(coords, flags):
    """Tuple dictionary oracle, independent of converter integer keys/searchsorted."""
    import numpy as np
    rows = coords.tolist()
    lookup = {tuple(c):i for i,c in enumerate(rows)}
    if len(lookup) != len(rows):raise ValueError('Duplicate coordinate')
    anchors, axes = np.nonzero(flags)
    quads = np.empty((len(anchors),4),np.int32)
    valid = np.zeros(len(anchors),bool);missing=[]
    for i,(anchor,axis) in enumerate(zip(anchors.tolist(),axes.tolist())):
        c=rows[anchor]
        corners=[tuple(c[k]+delta[k] for k in range(3)) for delta in OFFSETS[axis]]
        ids=[lookup.get(p,-1) for p in corners]
        if min(ids) >= 0:quads[i]=ids;valid[i]=True
        else:missing.append(dict(anchor=anchor,axis=axis,corners=corners,missing=[p for p,j in zip(corners,ids) if j<0]))
    return quads[valid],anchors[valid],axes[valid],missing


def parity_graph(n, constraints):
    """Analyze XOR constraints, never change faces or export an orientation variant."""
    parents=list(range(n));parity=[0]*n;sizes=[1]*n;bad=0
    def find(i):
        if parents[i]!=i:
            old=parents[i];parents[i],x=find(old);parity[i]^=x
        return parents[i],parity[i]
    for a,b,w in constraints:
        ra,pa=find(a);rb,pb=find(b)
        if ra==rb:bad+=int((pa^pb)!=w);continue
        if sizes[ra]<sizes[rb]:ra,rb=rb,ra
        parents[rb]=ra;parity[rb]=pa^pb^w;sizes[ra]+=sizes[rb]
    return dict(nodes=n,constraints=len(constraints),components=len({find(i)[0] for i in range(n)}),
                contradictory_constraints=bad,geometry_modified=False)


def rotations():
    import numpy as np
    result=[]
    for p in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1),repeat=3):
            m=np.zeros((3,3),np.int32);m[np.arange(3),p]=signs
            if round(np.linalg.det(m))==1:result.append(m)
    return result


def batch_seams(counts, order, anchors):
    import numpy as np
    starts=np.r_[0,np.cumsum(counts)[:-1]];n=len(anchors)
    seam=np.zeros(len(counts),bool)
    for start in range(0,len(counts),65536):
        end=min(start+65536,len(counts));rows=np.arange(start,end)
        base=anchors[order[starts[rows]]%n]//32768
        for k in range(1,int(counts[rows].max())):
            eligible=rows[counts[rows]>k]
            other=anchors[order[starts[eligible]+k]%n]//32768
            seam[eligible]|=other!=base[eligible-start]
    return seam


def authenticated():
    import numpy as np
    import amd_closed_solid_recovery_cpu as r
    import amd_generation_mesh as mesh
    with r.configured(EXECUTION) as cpu:
        reader=cpu.decode_reader()
        inputs=reader.read(RAW/'benchmark-inputs.json')
        reader.verify_snapshot(inputs['decoded_inputs']);reader.verify_snapshot(inputs['published_sources'])
        contract,binding,_=reader.receipt_inputs(EXECUTION)
        if binding!=inputs['binding'] or binding['source_sha256']!=SOURCE_SHA:raise ValueError('Original binding differs')
        _,full=reader.authenticate(EXECUTION)
        metadata=reader.read(RAW/'mesh/mesh.json');reader.validate_mesh(metadata,binding)
        paths=list(p for p in RAW.rglob('*') if p.is_file())
        paths += [Path(p) for p in inputs['decoded_inputs']]
        paths += [ROOT/'docs/compatibility/amd-turret-cpu-mesh-extraction-evidence/microsoft-flexible-dual-grid.py.txt']
        snapshot=reader.snapshot(dict.fromkeys(paths),allow_empty=True)
        if reader.digest(RAW/'mesh/mesh.ply')!=PLY_SHA:raise ValueError('Pinned original PLY differs')
        if reader.digest(paths[-1])!=UPSTREAM_SHA:raise ValueError('Pinned Microsoft source differs')
        arrays,children=mesh.read_neural(reader.SOURCE,SOURCE_SHA,contract=mesh._contract_api().parse(contract))
        hashes={name:hashlib.sha256(memoryview(a)).hexdigest() for name,a in arrays.items()}
        if hashes!=binding['tensor_payload_sha256']:raise ValueError('Ten payload hashes differ')
        v=np.load(RAW/'mesh/vertices.npy',mmap_mode='r',allow_pickle=False)
        f=np.load(RAW/'mesh/faces.npy',mmap_mode='r',allow_pickle=False)
        pv,pf=mesh.converter.read_ply(RAW/'mesh/mesh.ply')
        try:
            if not np.array_equal(v,pv) or not np.array_equal(f,pf):raise ValueError('PLY/NPY differs')
        finally:pv._mmap.close();pf._mmap.close()
        summary=dict(classification='PASS_TEN_TENSORS_PLY_ARRAYS_PROVENANCE',binding=binding,full_neural_authentication=full,
                     mesh=metadata,subdivisions=children,PLY_NPY_exact=True,model_imported=False)
        return arrays,v,f,summary,snapshot,reader


def orientation_patches(quads, edges, counts, signs, order, anchors, axes, coords):
    import numpy as np
    starts=np.r_[0,np.cumsum(counts)[:-1]];n=len(quads)
    two=np.flatnonzero(counts==2);first=order[starts[two]]%n;second=order[starts[two]+1]%n
    rhs=(np.abs(signs[two])==2).astype('i1')
    conflicts=np.flatnonzero(rhs);patches=[]
    # Six edge-order-distributed seeds; bounded crop, no global flip propagation.
    seeds=conflicts[np.linspace(0,len(conflicts)-1,min(6,len(conflicts)),dtype=int)]
    centers=coords[anchors]
    for seed in seeds:
        center=centers[first[seed]]
        chosen=np.flatnonzero(np.all(np.abs(centers-center)<=8,axis=1))
        before_cap=len(chosen)
        if len(chosen)>8192:
            delta=centers[chosen].astype('i8')-center
            chosen=chosen[np.lexsort((chosen,(delta*delta).sum(axis=1)))[:8192]]
        local=np.full(n,-1,np.int32);local[chosen]=np.arange(len(chosen))
        mask=(local[first]>=0)&(local[second]>=0)
        constraints=list(zip(local[first[mask]].tolist(),local[second[mask]].tolist(),rhs[mask].tolist()))
        row=parity_graph(len(chosen),constraints)
        row.update(grid_center=center.tolist(),grid_half_width=8,quads_before_cap=before_cap,quad_cap=8192,original_conflicts=int(rhs[mask].sum()),
                   open_constraint_ends=int(((local[first]>=0)^(local[second]>=0)).sum()),
                   nonmanifold_incident_quads=int(len(np.intersect1d(chosen,order[np.repeat(counts>2,counts)]%n))),
                   scope='Only two-quad edges inside this crop; >2 excluded, artificial crop boundary is not a source defect')
        patches.append(row)
    return patches


def spatial_and_witnesses(v, triangles, coords, flags, quads, anchors, axes, edges, counts, signs, order, missing):
    import numpy as np
    starts=np.r_[0,np.cumsum(counts)[:-1]];n=len(quads)
    lo=v.min(axis=0);hi=v.max(axis=0)
    mid=v[edges].mean(axis=1);unit=(mid-lo)/(hi-lo)
    # Geometric bins, not semantic claims that every point lies on a planar face.
    distances=np.concatenate((unit,1-unit),axis=1);side=np.argmin(distances,axis=1)
    labels=['X-min','Y-min','Z-min','X-max','Y-max','Z-max']
    kinds={'boundary':counts==1,'nonmanifold':counts>2,'winding':(counts==2)&(abs(signs)==2)}
    witnesses=[];stats={};points={}
    for kind,mask in kinds.items():
        ids=np.flatnonzero(mask);points[kind]=mid[ids]
        stats[kind]=dict(count=len(ids),nearest_bounds_face={label:int((side[ids]==i).sum()) for i,label in enumerate(labels)},
                         bounds=[mid[ids].min(axis=0).tolist(),mid[ids].max(axis=0).tolist()])
        # Up to three evenly spaced representatives per nearest-bounds face.
        for face in range(6):
            group=ids[side[ids]==face]
            selected=group[np.linspace(0,len(group)-1,min(3,len(group)),dtype=int)] if len(group) else []
            for edge_id in selected:
                occurrences=order[starts[edge_id]:starts[edge_id]+counts[edge_id]]
                qids=occurrences%n;corners=quads[qids]
                incident=[]
                for qid,occ in zip(qids,occurrences):
                    k=int(occ//n);q=quads[qid]
                    p=v[triangles[2*qid:2*qid+2]].astype('f8');normals=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
                    normals/=np.linalg.norm(normals,axis=1)[:,None]
                    incident.append(dict(quad_row=int(qid),anchor_row=int(anchors[qid]),anchor_grid=coords[anchors[qid]].tolist(),axis=int(axes[qid]),
                                         original_vertex_indices=q.tolist(),source_grid=coords[q].tolist(),triangle_rows=[int(2*qid),int(2*qid+1)],
                                         triangle_unit_normals=normals.tolist(),edge_direction=[int(q[k]),int(q[(k+1)%4])]))
                witnesses.append(dict(kind=kind,edge=edges[edge_id].tolist(),position=mid[edge_id].tolist(),nearest_bounds_face=labels[face],
                                      positions=v[edges[edge_id]].tolist(),grid=coords[edges[edge_id]].tolist(),incidence=int(counts[edge_id]),
                                      signed_sum=int(signs[edge_id]),incident_quads=incident))
    # Compare seam rate against ALL perimeter edges with two/one/>2 occurrences.
    seam=batch_seams(counts,order,anchors)
    stats['batch_seams']=dict(definition='Incident quad anchors lie in different original 32768-row conversion batches',
                              all_perimeter_edges=int(seam.sum()),all_perimeter_total=len(edges),
                              defects={k:dict(at_seam=int((m&seam).sum()),total=int(m.sum())) for k,m in kinds.items()})
    # For each boundary edge enumerate the four lattice-square candidates.
    lookup={tuple(c):i for i,c in enumerate(coords.tolist())};missing_map={(r['anchor'],r['axis']) for r in missing}
    reason={};boundary_details=[]
    for eid in np.flatnonzero(counts==1):
        edge=coords[edges[eid]];delta=np.abs(edge[1]-edge[0]);direction=int(np.argmax(delta));low=edge.min(axis=0)
        candidates=[]
        for axis in range(3):
            if axis==direction:continue
            other=next(k for k in range(3) if k not in (axis,direction))
            for shift in (-1,0):
                c=low.copy();c[other]+=shift;anchor=lookup.get(tuple(c),-1)
                state=('missing_anchor' if anchor<0 else 'flag_off' if not flags[anchor,axis] else 'flagged_missing_corner' if (anchor,axis) in missing_map else 'valid')
                reason[state]=reason.get(state,0)+1
                candidates.append(dict(axis=axis,anchor_grid=c.tolist(),state=state))
        if len(boundary_details)<24:boundary_details.append(dict(edge=edges[eid].tolist(),candidates=candidates))
    return stats,witnesses,points,dict(candidate_states=reason,scope='Four possible incident lattice quads per boundary edge; states may overlap across edges',examples=boundary_details)


def alignment(v, f):
    import numpy as np
    from scipy.spatial import cKDTree
    import amd_closed_solid_benchmark as b
    def sample(v,f):
        area=np.empty(len(f),np.float64)
        for start in range(0,len(f),32768):
            t=v[f[start:start+32768]].astype('f8');area[start:start+len(t)]=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)/2
        cumulative=np.cumsum(area);ids=np.searchsorted(cumulative,(np.arange(4096)+.5)/4096*cumulative[-1])
        t=v[f[ids]].astype('f8');uv=np.random.default_rng(b.SEED).random((4096,2));r=np.sqrt(uv[:,0])
        return (t*np.column_stack((1-r,r*(1-uv[:,1]),r*uv[:,1]))[:,:,None]).sum(axis=1)
    gv,gf=b.geometry();truth=sample(np.asarray(gv,'f4'),np.asarray(gf,'i4'));source=sample(v,f)
    tree=cKDTree(truth);diag=np.linalg.norm(b.DIMENSIONS);records=[]
    dims=v.max(axis=0)-v.min(axis=0)
    for m in rotations():
        p=source@m.T;a=tree.query(p,workers=1)[0]/diag;c=cKDTree(p).query(truth,workers=1)[0]/diag
        records.append(dict(matrix=m.tolist(),dimensions=(np.abs(m)@dims).tolist(),
                            bidirectional_RMS=float(np.sqrt((np.mean(a*a)+np.mean(c*c))/2)),
                            raw_to_GT_p95=float(np.quantile(a,.95)),GT_to_raw_p95=float(np.quantile(c,.95))))
    records.sort(key=lambda r:r['bidirectional_RMS'])
    return dict(method='4096 deterministic area-stratified points each way; sampled point-cloud distance / original GT diagonal, not Hausdorff',
                rotations=records,reflection_allowed=False,translation_allowed=False,scaling_allowed=False,
                geometry_changed=False,symmetry='Chamfered box cannot identify signs uniquely; sample scores do not prove an exporter axis bug')


def analyze():
    import numpy as np
    import amd_zombie_cpu_mesh as owner
    import amd_flexible_dual_grid_cpu as converter
    began=time.monotonic()
    def save(name,row):owner.save(BASE/name,row)
    def phase(name):
        if time.monotonic()-began>175:raise TimeoutError('175-second worker bound')
        with (BASE/'progress.jsonl').open('a',encoding='utf8') as stream:
            stream.write(json.dumps(dict(phase=name,utc=datetime.now(timezone.utc).isoformat(),elapsed=time.monotonic()-began))+'\n');stream.flush();os.fsync(stream.fileno())
    arrays={};v=f=None
    try:
        phase('authenticate');arrays,v,f,auth,snapshot,reader=authenticated()
        save('preserved-before.json',snapshot);save('authentication.json',auth)
        coords=arrays['coordinates'][:,1:];dual,flags,weight=converter.transform_features(arrays['features'])
        expected_v=converter.vertices(coords,dual)
        if not np.array_equal(v,expected_v):raise ValueError('Every original position must match neural transform')
        del dual,expected_v
        phase('independent-triangle-edges');te,tc,ts,to=edge_table(f)
        original_keys=np.load(RAW/'topology/edge_keys.npy',mmap_mode='r');original_counts=np.load(RAW/'topology/incidence.npy',mmap_mode='r')
        if not np.array_equal(te[:,0].astype('u8')*len(v)+te[:,1],original_keys) or not np.array_equal(tc,original_counts):raise ValueError('Independent edge sets/counts disagree')
        original_keys._mmap.close();original_counts._mmap.close()
        counts=dict(vertices=len(v),triangles=len(f),unique_edges=len(te),boundary=int((tc==1).sum()),nonmanifold=int((tc>2).sum()),
                    winding=int(((tc==2)&(abs(ts)==2)).sum()),max_incidence=int(tc.max()),unused=len(v)-len(np.unique(f)),
                    complete_edge_ID_and_incidence_match=True,method='Lexicographic two-column directed pairs; independent of original flattened-key bincount')
        old=reader.read(RAW/'raw-quality.json')
        if (counts['boundary'],counts['nonmanifold'],counts['winding'],counts['unused'])!=(old['edges']['boundary_incidence_1'],old['edges']['nonmanifold_incidence_gt_2'],old['two_face_winding_conflicts'],old['components']['unused_vertices']):raise ValueError('Independent counts disagree')
        save('independent-topology.json',counts);del to
        phase('tuple-coordinate-quad-correspondence');quads,anchors,axes,missing=tuple_quads(coords,flags)
        if not np.array_equal(quads,recover_quads(f)):raise ValueError('Direct source quad order/indices differ')
        q=weight[quads];choose=q[:,0]*q[:,2]>q[:,1]*q[:,3]
        split1=(f[1::2,0]==f[::2,0])
        if not np.array_equal(choose,split1):raise ValueError('Source diagonal decisions differ')
        qe,qc,qs,qo=edge_table(quads)
        for mask,other in ((qc==1,tc==1),(qc>2,tc>2),((qc==2)&(abs(qs)==2),(tc==2)&(abs(ts)==2))):
            if not np.array_equal(qe[mask],te[other]):raise ValueError('Direct quad/triangle defect edge set differs')
        diagonal=(np.abs(coords[te[:,0]]-coords[te[:,1]])!=0).sum(axis=1)==2
        if not np.all((tc[diagonal]==2)&(ts[diagonal]==0)):raise ValueError('Triangulation diagonal unexpectedly defective')
        counter=dict(flagged_quads=int(flags.sum()),valid_quads=len(quads),missing_neighbour_quads=len(missing),
                     out_of_grid_quads=sum(any(any(x<0 or x>=512 for x in p) for p in r['corners']) for r in missing),
                     split_1=int(choose.sum()),split_2=int((~choose).sum()),global_original_indices=True,batch_rows=32768)
        if counter!=auth['mesh']['conversion']:raise ValueError('All conversion counters must match')
        starts=np.r_[0,np.cumsum(qc)[:-1]];two=np.flatnonzero(qc==2)
        qa=qo[starts[two]]%len(quads);qb=qo[starts[two]+1]%len(quads);conflict=abs(qs[two])==2
        pairs=np.sort(np.column_stack((axes[qa],axes[qb])),axis=1)
        by_axis={str((a,b)):dict(all_two_quad_edges=int(((pairs==[a,b]).all(axis=1)).sum()),
                                 winding_conflicts=int((((pairs==[a,b]).all(axis=1))&conflict).sum())) for a in range(3) for b in range(a,3)}
        save('quad-attribution.json',dict(conversion=counter,all_original_quad_rows_and_indices_exact=True,all_diagonal_choices_exact=True,
             all_vertex_positions_exact=True,all_defect_edge_ID_sets_identical=True,triangulation_diagonal_edges=int(diagonal.sum()),
             defective_diagonals=0,two_quad_axis_pairs=by_axis,missing_flagged_quads=missing))
        del te,tc,ts,diagonal,q
        phase('spatial-defect-witnesses');stats,witnesses,points,reasons=spatial_and_witnesses(v,f,coords,flags,quads,anchors,axes,qe,qc,qs,qo,missing)
        save('spatial-distribution.json',stats);save('defect-witnesses.json',witnesses);save('boundary-candidate-causes.json',reasons)
        phase('bounded-orientation-constraints');save('orientation-patches.json',orientation_patches(quads,qe,qc,qs,qo,anchors,axes,coords))
        phase('proper-axis-rotation-diagnostic');save('axis-rotations.json',alignment(v,f))
        from PIL import Image,ImageDraw
        sheet=Image.new('RGB',(1500,540),'#20242b');draw=ImageDraw.Draw(sheet)
        colors={'winding':'#72a9ff','boundary':'#ffd34e','nonmanifold':'#ff4b55'}
        for k,(a,b) in enumerate(((0,1),(0,2),(1,2))):
            for kind,cloud in points.items():
                for p in cloud:draw.point((int(k*500+250+p[a]*440),int(270-p[b]*440)),fill=colors[kind])
            draw.text((k*500+12,14),f'Original axes {a}/{b}; orthographic defect locations',fill='white')
        draw.text((12,515),'Blue winding | yellow boundary | red >2 | source positions; no mesh changes',fill='white')
        sheet.save(BASE/'defect-locations.png')
        phase('source-preservation');reader.verify_snapshot(snapshot)
        save('preservation-after.json',dict(all_source_hashes_lengths_mtimes_unchanged=True,files=len(snapshot),geometry_modified=False,
             neural_execution=False,WSL=False,checkpoint_loaded=False,elapsed_seconds=time.monotonic()-began))
    finally:
        for a in [*arrays.values(),v,f]:
            if a is not None:a._mmap.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--worker',choices=('analyze',));a=p.parse_args(argv)
    if a.worker:
        if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Exact owned GO plus EOF required')
        import amd_closed_solid_benchmark as b
        sys.addaudithook(b.cpu_guard(BASE));sys.path.insert(0,'D:/AI/Trellis2AMD/venv/Lib/site-packages')
        analyze();return 0
    if a.run:
        import amd_zombie_cpu_mesh as owner
        owner.BASE=BASE;owner.__file__=str(Path(__file__).resolve())
        return owner.supervise('analyze')
    print('DRY_RUN: saved-artifact CPU forensics only; no WSL, neural or repair route');return 0

if __name__=='__main__':raise SystemExit(main())
