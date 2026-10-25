"""Two explicitly approved face reversals; bounded regional orientation analysis only."""
import argparse
import ast
from collections import deque
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT.parent/'artifacts/amd-zombie-winding-prototype-20261025-v1'
OLD=ROOT.parent/'artifacts/amd-zombie-orphan-cleanup-20261025-v1'
BASELINE_SHA='377a15dc4b5a27936f7e62b311dd03e8337685344b0f8204b8f831252176b1a5'
SELECTED=[334008,334009]
sys.path.insert(0,str(Path(__file__).parent))
import amd_zombie_orphan_cleanup as prior
topology=prior.topology
inherited=prior.inherited


def need(ok,why):
    if not ok:raise ValueError(why)


def reverse_rows(np,faces,rows):
    need(bool(rows) and len(set(rows))==len(rows) and min(rows)>=0 and max(rows)<len(faces),'Distinct valid authorized rows required')
    result=faces.copy();result[rows]=faces[rows][:,[0,2,1]]
    return result


def validate_edit(np,v0,f0,v1,f1,rows):
    need(v0.dtype==v1.dtype==np.float32 and f0.dtype==f1.dtype==np.int32,'Exact F32/I32 schema required')
    need(v0.shape==v1.shape and v0.tobytes()==v1.tobytes(),'Vertex payload changed')
    expected=reverse_rows(np,f0,rows)
    need(f1.shape==f0.shape and f1.tobytes()==expected.tobytes(),'Only the exact authorized corner permutations are permitted')
    need(np.array_equal(np.sort(f0,axis=1),np.sort(f1,axis=1)),'Per-face coordinate multisets changed')
    return hashlib.sha256(v0[np.sort(f0,axis=1)].tobytes()).hexdigest()


def solve_constraints(face_ids,constraints,fixed):
    """Solve x[a] XOR x[b] = parity; fixed constraints are x[a] = parity.

    Non-tree contradictions carry an odd-XOR fundamental-cycle witness.
    Returns symbolic face ID sets, never modifies or constructs a regional mesh.
    """
    ids=sorted(face_ids);need(len(set(ids))==len(ids),'Distinct face IDs required')
    graph={i:[] for i in ids};pins={i:[] for i in ids}
    for a,b,p,e in constraints:
        need(a in graph and b in graph and a!=b and p in (0,1),'Invalid internal orientation constraint')
        graph[a].append((b,p,e));graph[b].append((a,p,e))
    for a,p,e in fixed:
        need(a in graph and p in (0,1),'Invalid exterior orientation constraint');pins[a].append((p,e))
    labels={};parent={};components=[]
    for start in ids:
        if start in labels:continue
        labels[start]=0;parent[start]=None;queue=deque([start]);members=[]
        while queue:
            a=queue.popleft();members.append(a)
            for b,p,e in graph[a]:
                if b not in labels:
                    labels[b]=labels[a]^p;parent[b]=(a,p,e);queue.append(b)
        members.sort();member_set=set(members)
        edges=[row for row in constraints if row[0] in member_set]
        boundary=[(a,p,e) for a in members for p,e in pins[a]]
        def cycle(a,b,p,e):
            # Path a->LCA plus LCA->b plus the inconsistent closing edge.
            ancestors={a};x=a
            while parent[x] is not None:x=parent[x][0];ancestors.add(x)
            right=[];x=b
            while x not in ancestors:
                up,q,k=parent[x];right.append((x,up,q,k));x=up
            lca=x;left=[];x=a
            while x!=lca:
                up,q,k=parent[x];left.append((x,up,q,k));x=up
            path=left+[(up,x,q,k) for x,up,q,k in reversed(right)]+[(b,a,p,e)]
            xor=0
            for _,_,q,_ in path:xor^=q
            need(xor==1,'Contradiction witness must have odd XOR')
            return dict(face_walk=[path[0][0]]+[r[1] for r in path],edge_ids=[r[3] for r in path],parities=[r[2] for r in path],xor=xor)
        contradictions=[cycle(a,b,p,e) for a,b,p,e in edges if labels[a]^labels[b]!=p]
        requirements=[(labels[a]^p,a,e) for a,p,e in boundary]
        gauges=sorted({r[0] for r in requirements});consistent=not contradictions
        boundary_ok=consistent and len(gauges)<=1
        free=min((sum(labels[a]^g for a in members),g) for g in (0,1))[1]
        fixed_gauge=gauges[0] if gauges else free
        witness=[]
        if len(gauges)>1:
            first=next(r for r in requirements if r[0]==0);second=next(r for r in requirements if r[0]==1)
            witness=[dict(required_gauge=g,face_id=a,edge_id=e) for g,a,e in (first,second)]
        components.append(dict(face_ids=members,internal_constraint_count=len(edges),crossing_constraint_count=len(boundary),
            internal_conflicts_before=sum(p for _,_,p,_ in edges),crossing_conflicts_before=sum(p for _,p,_ in boundary),
            internally_consistent=consistent,contradictory_cycles=contradictions,
            fixed_exterior_consistent=boundary_ok,fixed_boundary_contradictions=witness,
            minimum_free_flip_face_ids=[a for a in members if labels[a]^free] if consistent else [],
            safe_fixed_exterior_flip_face_ids=[a for a in members if labels[a]^fixed_gauge] if boundary_ok else []))
    return dict(components=components,labels={str(k):v for k,v in labels.items()},analysis_only=True,mesh_constructed=False)


def baseline():
    need(inherited.digest(inherited.plain(OLD/'variant/mesh.ply'))==BASELINE_SHA,'Approved orphan baseline identity differs')
    manifest=inherited.read(OLD/'variant/manifest.json');accept=inherited.read(OLD/'variant/acceptance.json')
    need(accept['classification']=='PASS_APPROVED_ORPHAN_ONLY_VARIANT' and accept['variant_PLY_sha256']==BASELINE_SHA,'Accepted baseline required')
    for name,row in manifest['artifacts'].items():
        p=inherited.plain(OLD/'variant'/name)
        need(p.stat().st_size==row['bytes'] and inherited.digest(p)==row['sha256'],'Baseline closure changed')
    need(accept['initial_manifest_sha256']==inherited.digest(OLD/'variant/manifest.json') and
         accept['independent_verification_sha256']==inherited.digest(OLD/'independent-verification.json') and
         accept['silhouette_verification_sha256']==inherited.digest(OLD/'silhouette-verification.json'),'Baseline acceptance receipts changed')


def no_new_optimum(face_ids,constraints,fixed):
    """Exact minimum residual conflicts in an internally orientable bounded region.

    With a consistent orientation label L, let y=x XOR L. Every internal
    orientation constraint becomes y[a]=y[b]. Original good edges and good
    exterior constraints get capacity INF; existing bad edges get capacity 1.
    A minimum s-t cut therefore minimizes residual bad edges while forbidding
    new conflicts. INF is a combinatorial bound, not a resource safety gate.
    """
    solved=solve_constraints(face_ids,constraints,fixed)
    need(all(c['internally_consistent'] for c in solved['components']),'Internally consistent constraints required for exact cut reduction')
    ids=sorted(face_ids);index={a:i for i,a in enumerate(ids)};labels={int(k):v for k,v in solved['labels'].items()}
    count=len(ids);source=count;sink=count+1;graph=[{} for _ in range(count+2)]
    baseline_bad=sum(p for _,_,p,_ in constraints)+sum(p for _,p,_ in fixed);inf=baseline_bad+1
    def arc(a,b,capacity):
        graph[a][b]=graph[a].get(b,0)+capacity;graph[b].setdefault(a,0)
    for a,b,p,e in constraints:
        capacity=inf if p==0 else 1;arc(index[a],index[b],capacity);arc(index[b],index[a],capacity)
    for a,p,e in fixed:
        target=labels[a]^p;capacity=inf if p==0 else 1
        if target==0:arc(source,index[a],capacity)
        else:arc(index[a],sink,capacity)
    original=[dict(row) for row in graph];flow=0;augmentations=0
    while True:
        parent={source:None};queue=deque([source])
        while queue and sink not in parent:
            a=queue.popleft()
            for b,capacity in graph[a].items():
                if capacity>0 and b not in parent:parent[b]=a;queue.append(b)
        if sink not in parent:break
        value=inf;x=sink
        while x!=source:value=min(value,graph[parent[x]][x]);x=parent[x]
        x=sink
        while x!=source:
            a=parent[x];graph[a][x]-=value;graph[x][a]+=value;x=a
        flow+=value;augmentations+=1
        need(flow<=baseline_bad,'Baseline x=0 must bound the optimum')
    reachable=set(parent);cut=sum(capacity for a in reachable for b,capacity in original[a].items() if b not in reachable)
    need(cut==flow,'Max-flow/min-cut certificate differs')
    flips={a:(int(index[a] not in reachable)^labels[a]) for a in ids}
    constraints_after=[(p,flips[a]^flips[b]^p,e) for a,b,p,e in constraints]+[(p,flips[a]^p,e) for a,p,e in fixed]
    added=[e for p,q,e in constraints_after if p==0 and q==1];removed=[e for p,q,e in constraints_after if p==1 and q==0]
    residual=sum(q for p,q,e in constraints_after)
    need(not added and residual==flow,'No-new-conflict optimality certificate failed')
    return dict(analysis_only=True,mesh_constructed=False,flip_face_ids=[a for a in ids if flips[a]],
        original_conflicts=baseline_bad,minimum_residual_conflicts=residual,conflicts_removed=len(removed),
        new_conflicts=len(added),removed_conflict_edge_ids=removed,max_flow=flow,cut_capacity=cut,
        augmentations=augmentations,hard_constraint_capacity=inf,
        source_side_face_ids=[a for a in ids if index[a] in reachable],
        limitation='Optimum applies only to this internally orientable region, fixed exterior and two-face constraints. It proves no outward normals, manifoldness or production readiness.')


def edit():
    import numpy as np
    import amd_flexible_dual_grid_cpu as fdg
    baseline();contract,authentication=inherited.authenticate()
    spec=inherited.read(OLD/'future-prototype-spec.json')
    need(spec['face_ids']==SELECTED and spec['analysis_only'] and not spec['applied'],'Approved proposal differs')
    v,f=fdg.read_ply(OLD/'variant/mesh.ply')
    try:
        ff=reverse_rows(np,f,SELECTED);canonical=validate_edit(np,v,f,v,ff,SELECTED)
        out=BASE/'variant';out.mkdir(exist_ok=False)
        fdg.write_ply(out/'mesh.ply',v,ff)
        prior.save_npy(out/'vertices.npy',v);prior.save_npy(out/'faces.npy',ff)
        inherited.save(out/'manifest.json',dict(classification='CREATED_PENDING_INDEPENDENT_VERIFICATION',
            baseline_PLY_sha256=BASELINE_SHA,source_neural_sha256=inherited.SOURCE_SHA,run_contract_sha256=contract.sha256,
            changed_face_rows=SELECTED,permutation=[0,2,1],vertices=len(v),triangles=len(ff),
            canonical_per_face_coordinate_stream_sha256=canonical,
            artifacts={p.name:dict(bytes=p.stat().st_size,sha256=inherited.digest(p)) for p in out.iterdir()}))
        inherited.save(BASE/'source-authentication.json',authentication)
    finally:v._mmap.close();f._mmap.close()


def verify():
    import numpy as np
    baseline();checker=inherited.load('_winding_independent_ply',ROOT/'docs/compatibility/amd-brick-cpu-mesh-evidence/verify_mesh.py')
    opened=[]
    try:
        v0,p0,_=checker.ply_arrays(np,OLD/'variant/mesh.ply',325807,673604,opened,need)
        v1,p1,_=checker.ply_arrays(np,BASE/'variant/mesh.ply',325807,673604,opened,need)
        need(np.all(p0['count']==3) and np.all(p1['count']==3),'Triangle schema required');f0=p0['idx'];f1=p1['idx']
        canonical=validate_edit(np,v0,f0,v1,f1,SELECTED)
        for name,a in [('vertices.npy',v1),('faces.npy',f1)]:
            saved=np.load(BASE/'variant'/name,allow_pickle=False);need(saved.tobytes()==a.tobytes(),'PLY/NPY independent readback differs')
        e0,c0,i0,w0=topology.incidence(np,f0);e1,c1,i1,w1=topology.incidence(np,f1)
        need(np.array_equal(e0,e1) and np.array_equal(c0,c1),'Complete edge identity/incidence changed')
        b0=(c0==2)&(abs(w0)==2);b1=(c1==2)&(abs(w1)==2)
        removed=np.flatnonzero(b0&~b1);added=np.flatnonzero(~b0&b1)
        need((int(b0.sum()),int(b1.sum()),len(removed),len(added))==(147484,147480,4,0),'Unexpected global winding change')
        prediction=prior.proposal_from_incidence(np,f0,SELECTED,len(v0),e0,c0,w0)
        need(prediction['before_conflicts']==4 and prediction['predicted_after_conflicts']==0,'Local prediction differs')
        affected=np.flatnonzero(w0!=w1)
        need(np.array_equal(affected,removed),'Signed incidence changed outside four approved perimeter edges')
        vl0,fl0=topology.component_labels(np,f0,len(v0));vl1,fl1=topology.component_labels(np,f1,len(v1))
        need(np.array_equal(vl0,vl1) and np.array_equal(fl0,fl1) and len(np.unique(fl1))==10,'Component identity changed')
        need(int((c1==1).sum())==19512 and int((c1>2).sum())==31449,'Boundary/nonmanifold counts changed')
        normals=[]
        for row in SELECTED:
            a=v0[f0[row]].astype(np.float64);b=v1[f1[row]].astype(np.float64)
            n0=np.cross(a[1]-a[0],a[2]-a[0]);n0/=np.linalg.norm(n0)
            n1=np.cross(b[1]-b[0],b[2]-b[0]);n1/=np.linalg.norm(n1)
            need(np.allclose(n0,-n1,rtol=0,atol=1e-14),'Expected normal inversion differs')
            normals.append(dict(face_id=row,before=n0.tolist(),after=n1.tolist(),dot=float(n0@n1)))
        inherited.save(BASE/'independent-verification.json',dict(classification='PASS_EXACT_TWO_FACE_WINDING_EDIT',
            baseline_PLY_sha256=BASELINE_SHA,variant_PLY_sha256=inherited.digest(BASE/'variant/mesh.ply'),
            vertices=325807,triangles=673604,changed_face_rows=SELECTED,permutation=[0,2,1],
            all_other_face_rows_bit_identical=True,all_vertex_payloads_bit_identical=True,
            canonical_per_face_coordinate_stream_sha256=canonical,per_face_coordinate_multisets_bit_identical=True,
            bounds_bit_identical=True,bounds=[v1.min(axis=0).tolist(),v1.max(axis=0).tolist()],
            component_labels_bit_identical=True,surface_components=10,boundary_edges=19512,nonmanifold_edges=31449,
            winding_conflicts_before=147484,winding_conflicts_after=147480,new_conflicts=0,
            removed_conflict_edge_ids=removed.tolist(),removed_conflict_edges=e0[removed].tolist(),
            all_other_edge_signed_sums_unchanged=True,local_prediction=prediction,face_normals=normals))
    finally:
        for a in opened:
            m=getattr(a,'_mmap',None)
            if m is not None and not m.closed:m.close()


def silhouette_check():
    import numpy as np
    from PIL import Image
    meta=inherited.read(BASE/'silhouettes/render-metadata.json');pairs=[]
    for row in meta['pairs']:
        data=[]
        for kind in ('source','variant'):
            p=BASE/'silhouettes'/row[kind];need(inherited.digest(p)==row[kind+'_sha256'],'Render changed')
            with Image.open(p) as image:data.append(np.array(image.convert('RGB')))
        need(np.array_equal(*data),'Unexpected silhouette pixel change')
        pairs.append(dict(azimuth_degrees=row['azimuth_degrees'],different_RGB_pixels=0,mask_differences=0,
            RGB_sha256=hashlib.sha256(data[0].tobytes()).hexdigest()))
    need([r['azimuth_degrees'] for r in pairs]==list(range(0,360,45)),'Eight original cameras required')
    inherited.save(BASE/'silhouette-verification.json',dict(classification='PASS_EIGHT_EXACT_SILHOUETTES',pairs=pairs))


def regional():
    import numpy as np
    from PIL import Image
    proof=inherited.read(BASE/'independent-verification.json')
    need(proof['classification']=='PASS_EXACT_TWO_FACE_WINDING_EDIT','Phase A verified first')
    need(inherited.read(BASE/'silhouette-verification.json')['classification']=='PASS_EIGHT_EXACT_SILHOUETTES','Phase A silhouettes first')
    v=np.load(BASE/'variant/vertices.npy',mmap_mode='r',allow_pickle=False)
    f=np.load(BASE/'variant/faces.npy',mmap_mode='r',allow_pickle=False)
    try:
        before=hashlib.sha256(f.tobytes()).hexdigest();points=v[f]
        # Previously accepted ordinary posterior upper-trunk box. Include complete triangles only.
        lo=np.array([-.045,.02,.17]);hi=np.array([.045,.12,.23])
        roi=np.all((points>=lo)&(points<=hi),axis=(1,2));box_count=int(roi.sum());clearances=[]
        for row in inherited.read(BASE/'silhouettes/render-metadata.json')['pairs']:
            with Image.open(BASE/'silhouettes'/row['source']) as image:mask=np.array(image.convert('L'))<128
            boundary=mask&~(np.roll(mask,1,0)&np.roll(mask,-1,0)&np.roll(mask,1,1)&np.roll(mask,-1,1))
            ids=np.flatnonzero(roi);p=points[ids]
            samples=np.concatenate((p,p.mean(axis=1)[:,None,:],(p+np.roll(p,-1,axis=1))*.5),axis=1)
            forward=np.array(row['target'])-np.array(row['camera_location']);forward/=np.linalg.norm(forward)
            right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right);up=np.cross(right,forward)
            d=samples-np.array(row['target']);xy=np.stack((384+(d@right)*768/row['ortho_scale'],384-(d@up)*768/row['ortho_scale']),axis=2)
            pix=np.rint(xy).astype(int);valid=np.all((pix>=0)&(pix<768),axis=(1,2));valid[valid]&=np.all((mask&~boundary)[pix[valid,:,1],pix[valid,:,0]],axis=1)
            roi[ids[~valid]]=False;clearances.append(dict(angle=row['azimuth_degrees'],excluded_at_this_view=int((~valid).sum())))
        ids=np.flatnonzero(roi);need(len(ids)>2 and all(roi[SELECTED]),'Region must include prototype and ordinary multi-face surface')
        edges,counts,inverse,sums=topology.incidence(np,f);n=len(f)
        order=np.argsort(inverse,kind='stable');starts=np.r_[0,np.cumsum(counts)]
        two=np.flatnonzero(counts==2);a=order[starts[two]]%n;b=order[starts[two]+1]%n
        internal=roi[a]&roi[b];crossing=roi[a]^roi[b]
        constraints=[(int(x),int(y),int(abs(sums[e])==2),int(e)) for x,y,e in zip(a[internal],b[internal],two[internal])]
        fixed=[(int(x if roi[x] else y),int(abs(sums[e])==2),int(e)) for x,y,e in zip(a[crossing],b[crossing],two[crossing])]
        solved=solve_constraints(ids.tolist(),constraints,fixed)
        safe=sorted(x for c in solved['components'] for x in c['safe_fixed_exterior_flip_face_ids'])
        # Symbolic signs only; no modified regional face array exists.
        flips=np.zeros(n,bool);flips[safe]=True
        parity=(abs(sums[two])==2);predicted=parity^flips[a]^flips[b]
        removed=two[parity&~predicted];added=two[~parity&predicted]
        need(len(added)==0,'Conservative fixed-exterior feasibility unexpectedly introduces a conflict')
        need(hashlib.sha256(f.tobytes()).hexdigest()==before,'Analysis modified faces')
        touched=np.unique(inverse.reshape(3,n)[:,ids]);inc_hist={str(int(c)):int((counts[touched]==c).sum()) for c in np.unique(counts[touched])}
        comps=solved['components'];consistent=[c for c in comps if c['internally_consistent']];fixed_ok=[c for c in comps if c['fixed_exterior_consistent']]
        prior.save_npy(BASE/'region-face-ids.npy',ids.astype(np.int32))
        prior.save_npy(BASE/'symbolic-safe-flip-face-ids.npy',np.array(safe,np.int32))
        inherited.save(BASE/'regional-constraints.json',dict(constraints=constraints,fixed_exterior_constraints=fixed,**solved))
        result=dict(classification='REGIONAL_ANALYSIS_COMPLETE',analysis_only=True,additional_mesh_modified=False,
            variant_PLY_sha256=proof['variant_PLY_sha256'],region_bounds=[lo.tolist(),hi.tolist()],box_complete_triangles=box_count,
            selected_faces=len(ids),selected_vertices=len(np.unique(f[ids])),exclusion_by_view=clearances,
            silhouette_exclusion='All triangle corners, edge midpoints and centroid must project inside non-contour silhouette pixels at all eight original cameras; raster test, not semantic segmentation',
            feature_exclusion='Accepted ordinary central posterior upper trunk box: excludes front wounds, clothing, scalp, face, hands and feet by position; inspect regional context image',
            two_face_internal_constraints=len(constraints),two_face_crossing_constraints=len(fixed),
            internal_conflicts_before=sum(p for _,_,p,_ in constraints),crossing_conflicts_before=sum(p for _,p,_ in fixed),
            adjacency_components=len(comps),internally_consistent_components=len(consistent),internally_contradictory_components=len(comps)-len(consistent),
            contradictory_fundamental_cycle_witnesses=sum(len(c['contradictory_cycles']) for c in comps),
            internally_consistent_faces=sum(len(c['face_ids']) for c in consistent),
            internal_conflicts_resolvable_if_exterior_unconstrained=sum(c['internal_conflicts_before'] for c in consistent),
            fixed_exterior_consistent_components=len(fixed_ok),fixed_boundary_incompatible_components=sum(c['internally_consistent'] and not c['fixed_exterior_consistent'] for c in comps),
            fixed_exterior_consistent_faces=sum(len(c['face_ids']) for c in fixed_ok),
            safe_fixed_exterior_symbolic_flips=len(safe),safe_conflicts_resolvable=len(removed),safe_new_conflicts=len(added),
            safe_predicted_global_conflicts=147480-len(removed),
            touched_edge_incidence_histogram=inc_hist,touched_boundary_edges=int((counts[touched]==1).sum()),
            touched_nonmanifold_edges=int((counts[touched]>2).sum()),
            boundary_edges_unresolved=19512,nonmanifold_edges_unresolved=31449,
            orientation_does_not_change_incidence_or_solve_nonmanifold_fans=True,
            no_regional_repaired_mesh_constructed=True,
            limitation='Contradiction witnesses are fundamental cycles under this spanning forest, not a maximum satisfiable subset or minimal repair count. Internal feasibility ignores exterior; safe prediction fixes every exterior face. No global propagation.')
        inherited.save(BASE/'regional-analysis.json',result);print(json.dumps(result),flush=True)
    finally:v._mmap.close();f._mmap.close()


def compose():
    from PIL import Image,ImageDraw,ImageFont
    from numpy import array,array_equal
    proof=inherited.read(BASE/'independent-verification.json');sil=inherited.read(BASE/'silhouette-verification.json')
    local=inherited.read(BASE/'local-render/render-metadata.json');region=inherited.read(BASE/'regional-analysis.json')
    need(sil['classification']=='PASS_EIGHT_EXACT_SILHOUETTES','Exact silhouettes required')
    need(local['all_imported_vertex_face_payloads_exact'],'Render payload verification required')
    pixel_changes=[]
    for mode in ('neutral','front-back','culled','regional'):
        for kind in ('source','variant'):need(inherited.digest(BASE/'local-render'/f'{kind}-{mode}.png')==local['hashes'][f'{kind}-{mode}.png'],'Local image hash changed')
        with Image.open(BASE/'local-render'/f'source-{mode}.png') as a,Image.open(BASE/'local-render'/f'variant-{mode}.png') as b:
            aa=array(a.convert('RGB'));bb=array(b.convert('RGB'));count=int((aa!=bb).any(axis=2).sum())
            pixel_changes.append(dict(mode=mode,changed_RGB_pixels=count))
    out=BASE/'inspection-sheets';out.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    image=Image.new('RGB',(1536,1616),'#24272c');draw=ImageDraw.Draw(image)
    for i,mode in enumerate(('neutral','front-back','culled','regional')):
        for j,kind in enumerate(('source','variant')):
            with Image.open(BASE/'local-render'/f'{kind}-{mode}.png') as view:image.paste(view.convert('RGB').resize((384,384)),(j*384+(i%2)*768,(i//2)*808+40))
            draw.text((j*384+(i%2)*768+8,(i//2)*808+8),f'{kind.upper()} / {mode}',font=font,fill='white')
    draw.text((12,1260),'Front/back: cyan = camera-facing, red = backface. Culled: backfaces transparent.',font=font,fill='white')
    draw.text((12,1300),'Regional overlay: blue = analysis region; gold = approved two-face edit. No regional repair.',font=font,fill='white')
    image=image.crop((0,0,1536,1340));image.save(out/'local-comparison.png')
    image=Image.new('RGB',(1536,840),'#24272c');draw=ImageDraw.Draw(image)
    for i,row in enumerate(sil['pairs']):
        angle=row['azimuth_degrees'];x=(i%4)*384;y=(i//4)*420
        with Image.open(BASE/'silhouettes'/f'variant-{angle:03d}.png') as view:image.paste(view.convert('RGB').resize((384,384)),(x,y+32))
        draw.text((x+6,y+6),f'{angle} deg: 0 changed pixels',font=font,fill='white')
    image.save(out/'eight-exact-silhouettes.png')
    inherited.save(BASE/'variant/acceptance.json',dict(classification='PASS_TWO_FACE_PROTOTYPE',
        variant_PLY_sha256=proof['variant_PLY_sha256'],baseline_PLY_sha256=BASELINE_SHA,
        independent_verification_sha256=inherited.digest(BASE/'independent-verification.json'),
        silhouette_verification_sha256=inherited.digest(BASE/'silhouette-verification.json'),
        local_render_sha256=inherited.digest(BASE/'local-render/render-metadata.json'),
        winding_conflicts=147480,new_conflicts=0,changed_face_rows=SELECTED,
        regional_repair_performed=False))
    inherited.save(BASE/'presentation-verification.json',dict(classification='PASS',pixel_changes=pixel_changes,
        local_normals=proof['face_normals'],backface_witnesses=local['backface_witnesses'],
        sheets={p.name:inherited.digest(p) for p in out.glob('*.png')}))


def feasibility():
    """Append an exact partial-repair bound and clearer local comparisons; no mesh writes."""
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    data=inherited.read(BASE/'regional-constraints.json');ids=[int(k) for k in data['labels']]
    result=no_new_optimum(ids,data['constraints'],data['fixed_exterior_constraints'])
    gauges=[];labels={int(k):v for k,v in data['labels'].items()}
    for c in data['components']:
        members=set(c['face_ids']);boundary=[r for r in data['fixed_exterior_constraints'] if r[0] in members]
        options=[]
        for g in (0,1):
            old_new=[(p,labels[a]^g^p) for a,p,e in boundary]
            options.append(dict(gauge=g,flipped_faces=sum(labels[a]^g for a in members),
                crossing_conflicts_after=sum(q for p,q in old_new),new_crossing_conflicts=sum(p==0 and q==1 for p,q in old_new),
                removed_crossing_conflicts=sum(p==1 and q==0 for p,q in old_new)))
        gauges.append(dict(faces=len(members),internal_conflicts=c['internal_conflicts_before'],
            crossing_conflicts=c['crossing_conflicts_before'],options=options))
    result['coherent_component_gauge_options']=gauges
    v=np.load(BASE/'variant/vertices.npy',mmap_mode='r',allow_pickle=False)
    f=np.load(BASE/'variant/faces.npy',mmap_mode='r',allow_pickle=False)
    try:
        e,c,inv,w=topology.incidence(np,f);order=np.argsort(inv,kind='stable');starts=np.r_[0,np.cumsum(c)]
        two=np.flatnonzero(c==2);a=order[starts[two]]%len(f);b=order[starts[two]+1]%len(f)
        flips=np.zeros(len(f),bool);flips[result['flip_face_ids']]=True
        before=(abs(w[two])==2);after=before^flips[a]^flips[b]
        need(int((before&~after).sum())==result['conflicts_removed'] and not np.any(~before&after),'Full-mesh symbolic certificate differs')
        result['full_mesh_symbolic_verification']=True;result['predicted_global_conflicts']=147480-result['conflicts_removed']
        # Camera-facing is only a local visual direction witness, never global outwardness.
        points=v[f[ids]].astype(np.float64);normals=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])
        normals/=np.linalg.norm(normals,axis=1)[:,None];to_camera=np.array([0.,2.,.3]);to_camera/=np.linalg.norm(to_camera)
        dots=normals@to_camera;predicted=dots*np.where(flips[ids],-1,1)
        result['posterior_camera_facing_before']=int((dots>0).sum());result['posterior_camera_facing_symbolic_after']=int((predicted>0).sum())
        result['posterior_facing_to_backfacing']=int(((dots>0)&(predicted<0)).sum());result['posterior_backfacing_to_facing']=int(((dots<0)&(predicted>0)).sum())
        result['camera_direction_metric_not_global_outwardness']=True
        result['variant_PLY_sha256']=inherited.digest(BASE/'variant/mesh.ply')
    finally:v._mmap.close();f._mmap.close()
    inherited.save(BASE/'regional-optimum.json',result)
    out=BASE/'refined-sheets';out.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    image=Image.new('RGB',(960,1580),'#24272c');draw=ImageDraw.Draw(image)
    for i,mode in enumerate(('neutral','front-back','culled')):
        for j,kind in enumerate(('source','variant')):
            with Image.open(BASE/'local-render'/f'{kind}-{mode}.png') as view:
                picture=view.convert('RGB').crop((304,304,464,464)).resize((480,480),Image.Resampling.NEAREST)
            image.paste(picture,(j*480,i*520+40));draw.text((j*480+8,i*520+10),f'{kind.upper()} / {mode}',font=font,fill='white')
    image.save(out/'local-winding-detail.png')
    image=Image.new('RGB',(1536,900),'#24272c');draw=ImageDraw.Draw(image)
    for i,mode in enumerate(('neutral','front-back','culled','regional')):
        for j,kind in enumerate(('source','variant')):
            x=j*384+(i%2)*768;y=(i//2)*424
            with Image.open(BASE/'local-render'/f'{kind}-{mode}.png') as view:image.paste(view.convert('RGB').resize((384,384)),(x,y+32))
            draw.text((x+6,y+6),f'{kind.upper()} / {mode}',font=font,fill='white')
    draw.text((10,858),'Cyan front / red back. Only two face windings changed; region is analysis-only.',font=font,fill='white')
    image.save(out/'local-comparison.png')
    inherited.save(out/'manifest.json',dict(presentation_only=True,images={p.name:inherited.digest(p) for p in out.glob('*.png')}))
    print(json.dumps({k:v for k,v in result.items() if k not in ('flip_face_ids','removed_conflict_edge_ids','source_side_face_ids','coherent_component_gauge_options')}),flush=True)


def visibility_bound():
    """One symbolic comparison: preserve existing rear-facing presentation witnesses.

    This is an explicit alternative optimization constraint, not a new safety
    threshold or a claim to know globally outward orientation on an open mesh.
    """
    import numpy as np
    data=inherited.read(BASE/'regional-constraints.json');ids=sorted(int(k) for k in data['labels'])
    v=np.load(BASE/'variant/vertices.npy',mmap_mode='r',allow_pickle=False)
    f=np.load(BASE/'variant/faces.npy',mmap_mode='r',allow_pickle=False)
    try:
        points=v[f[ids]].astype(np.float64);normals=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])
        to_camera=np.array([0.,2.,.3]);dots=normals@to_camera
        protected=[ids[i] for i in np.flatnonzero(dots>0)]
        extra=[(a,0,-1-i) for i,a in enumerate(protected)]
        result=no_new_optimum(ids,data['constraints'],data['fixed_exterior_constraints']+extra)
        flips=set(result['flip_face_ids']);need(not flips.intersection(protected),'Facing witness reversed despite fixed presentation constraint')
        result.update(presentation_only_constraint='Keep all currently posterior-camera-facing selected faces fixed; this is not global outwardness',
            fixed_posterior_camera_facing_faces=len(protected),predicted_global_conflicts=147480-result['conflicts_removed'],
            comparison_to_unconstrained_bound=inherited.read(BASE/'regional-optimum.json')['conflicts_removed'])
        inherited.save(BASE/'regional-visibility-bound.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('flip_face_ids','removed_conflict_edge_ids','source_side_face_ids')}),flush=True)
    finally:v._mmap.close();f._mmap.close()


def owner(stage):
    module=inherited.load('_zombie_winding_owned_cpu',Path(inherited.__file__))
    tree=ast.parse(inspect.getsource(module.supervise));changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='amd_zombie_mesh_render.py':node.value='amd_zombie_winding_render.py';changed+=1
    need(changed==1,'Preserved owner route differs');module.__file__=__file__;module.BASE=BASE
    exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),module.__dict__)
    if stage in ('silhouettes','local-render'):module.BASE=BASE/('render-'+stage);module.BASE.mkdir(exist_ok=False)
    return module


def worker(stage):
    if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO/EOF required')
    def audit(event,args):
        if event=='import' and str(args[0]).split('.')[0].lower() in {'torch','triton','trellis2','flex_gemm','o_voxel','cupy','cumesh','nvdiffrast'}:raise RuntimeError('Model/GPU import denied')
        if event in ('subprocess.Popen','os.system','socket.connect','socket.bind'):raise RuntimeError('Process/network denied')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and (any(c in str(args[1]) for c in 'wax+') or args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):
            inherited.plain(os.fsdecode(args[0])).relative_to(BASE)
        if event=='os.mkdir':inherited.plain(args[0]).relative_to(BASE)
        if event in ('os.remove','os.rename','os.rmdir','os.link','os.symlink','os.truncate'):raise RuntimeError('Destructive mutation denied')
    sys.addaudithook(audit);sys.path.append('D:/AI/Trellis2AMD/venv/Lib/site-packages')
    {'edit':edit,'verify':verify,'silhouette-check':silhouette_check,'regional':regional,'compose':compose,'feasibility':feasibility,'visibility-bound':visibility_bound}[stage]()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('edit','verify','silhouettes','silhouette-check','regional','local-render','compose','feasibility','visibility-bound'))
    parser.add_argument('--worker',choices=('edit','verify','silhouette-check','regional','compose','feasibility','visibility-bound'))
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.stage:
        module=owner(args.stage)
        if args.stage in ('silhouettes','local-render'):
            os.environ['MODLY_ZOMBIE_WINDING_MODE']=args.stage;raise SystemExit(module.supervise('render'))
        raise SystemExit(module.supervise(args.stage))
    else:print('DRY_RUN: explicit approved two-face CPU stage required; no neural or regional repair entrypoint')
