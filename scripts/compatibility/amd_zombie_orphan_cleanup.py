"""Approved copy-only orphan compaction plus analysis-only local winding diagnosis."""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT.parent/'artifacts/amd-zombie-orphan-cleanup-20261025-v1'
OLD=ROOT.parent/'artifacts/amd-zombie-cpu-mesh-20261025-v1'
DIAG=ROOT.parent/'artifacts/amd-zombie-topology-diagnosis-20261025-v1'
sys.path.insert(0,str(Path(__file__).parent))
import amd_zombie_topology as topology
inherited=topology.inherited


def compact(np,vertices,faces,approved_ids):
    if vertices.dtype!=np.float32 or vertices.ndim!=2 or vertices.shape[1]!=3 or faces.dtype!=np.int32 or faces.ndim!=2 or faces.shape[1]!=3:
        raise ValueError('Exact F32 vertices and I32 triangles required')
    if not len(faces) or faces.min()<0 or faces.max()>=len(vertices):raise ValueError('Valid nonempty faces required')
    keep=np.unique(faces);removed=np.setdiff1d(np.arange(len(vertices),dtype=np.int32),keep)
    if approved_ids.ndim!=1 or not np.array_equal(np.sort(approved_ids),removed):raise ValueError('Only the exact approved orphan IDs may be removed')
    new_to_old=keep.astype(np.int32);old_to_new=np.full(len(vertices),-1,np.int32)
    old_to_new[new_to_old]=np.arange(len(keep),dtype=np.int32)
    vv=np.ascontiguousarray(vertices[new_to_old]);ff=np.ascontiguousarray(old_to_new[faces])
    surface_equal(np,vertices,faces,vv,ff)
    return vv,ff,new_to_old,old_to_new


def surface_equal(np,v0,f0,v1,f1):
    if f0.shape!=f1.shape:raise ValueError('Triangle count/order changed')
    a=hashlib.sha256();b=hashlib.sha256()
    for start in range(0,len(f0),32768):
        before=np.ascontiguousarray(v0[f0[start:start+32768]])
        after=np.ascontiguousarray(v1[f1[start:start+32768]])
        if not np.array_equal(before.view(np.uint32),after.view(np.uint32)):raise ValueError('Triangle geometry/corner order changed')
        a.update(before.tobytes());b.update(after.tobytes())
    if a.digest()!=b.digest():raise ValueError('Triangle stream hash differs')
    return a.hexdigest()


def winding_proposal(np,faces,selected,n):
    edges,counts,inverse,sums=topology.incidence(np,faces)
    return proposal_from_incidence(np,faces,selected,n,edges,counts,sums)


def proposal_from_incidence(np,faces,selected,n,edges,counts,sums):
    keys=edges[:,0].astype(np.int64)*n+edges[:,1];local={}
    for face in faces[selected]:
        for a,b in zip(face,np.roll(face,-1)):
            key=int(min(a,b))*n+int(max(a,b));edge=int(np.searchsorted(keys,key))
            if edge>=len(keys) or keys[edge]!=key:raise ValueError('Missing face edge')
            local[edge]=local.get(edge,0)+(1 if a<b else -1)
    touched=sorted(local);predicted={i:int(sums[i])-2*local[i] for i in touched}
    return dict(all_touched_edges_two_face=bool(np.all(counts[touched]==2)),
        before_conflicts=int(sum(counts[i]==2 and abs(int(sums[i]))==2 for i in touched)),
        predicted_after_conflicts=int(sum(counts[i]==2 and abs(predicted[i])==2 for i in touched)),
        touched_edges=edges[touched].tolist(),incidences=counts[touched].tolist(),
        before_signed_sums=sums[touched].tolist(),predicted_signed_sums=[predicted[i] for i in touched],
        faces_reversed_or_repaired=False)


def save_npy(path,array):
    import numpy as np
    with path.open('xb') as stream:np.save(stream,array,allow_pickle=False)


def open_original(np):
    import amd_flexible_dual_grid_cpu as fdg
    if inherited.digest(inherited.plain(OLD/'mesh/mesh.ply'))!=topology.PLY_SHA:raise ValueError('Original mesh identity differs')
    metadata=inherited.read(OLD/'mesh/mesh.json')
    for name,row in metadata['artifacts'].items():
        p=inherited.plain(OLD/'mesh'/name)
        if p.stat().st_size!=row['bytes'] or inherited.digest(p)!=row['sha256']:raise ValueError('Original artifact hash differs')
    return fdg.read_ply(OLD/'mesh/mesh.ply')


def cleanup():
    import numpy as np
    import amd_flexible_dual_grid_cpu as fdg
    contract,authentication=inherited.authenticate()
    v,f=open_original(np)
    with np.load(DIAG/'topology.npz',allow_pickle=False) as data:approved=data['unused_vertex_ids']
    try:
        if len(v)!=326232 or len(f)!=673604 or len(approved)!=425:raise ValueError('Approved original counts differ')
        vv,ff,new_to_old,old_to_new=compact(np,v,f,approved)
        if (len(vv),len(ff))!=(325807,673604):raise ValueError('Required variant counts differ')
        if not np.array_equal(v.min(axis=0).view(np.uint32),vv.min(axis=0).view(np.uint32)) or not np.array_equal(v.max(axis=0).view(np.uint32),vv.max(axis=0).view(np.uint32)):
            raise ValueError('Bounds changed; stop')
        triangle_sha=surface_equal(np,v,f,vv,ff)
        output=BASE/'variant';output.mkdir(exist_ok=False)
        fdg.write_ply(output/'mesh.ply',vv,ff)
        for name,array in [('vertices.npy',vv),('faces.npy',ff),('new-to-original.npy',new_to_old),('original-to-new.npy',old_to_new),('removed-original-ids.npy',approved)]:save_npy(output/name,array)
        inherited.save(output/'manifest.json',dict(classification='CREATED_PENDING_INDEPENDENT_VERIFICATION',
            source_PLY_sha256=topology.PLY_SHA,source_neural_sha256=inherited.SOURCE_SHA,run_contract_sha256=contract.sha256,
            removed_unreferenced_vertices=425,vertices=len(vv),triangles=len(ff),triangle_stream_sha256=triangle_sha,
            bounds=dict(min=vv.min(axis=0).tolist(),max=vv.max(axis=0).tolist()),
            surface_changed=False,structural_repair_performed=False,
            artifacts={p.name:dict(bytes=p.stat().st_size,sha256=inherited.digest(p)) for p in output.iterdir() if p.is_file()}))
        inherited.save(BASE/'cleanup-source-authentication.json',authentication)
    finally:
        for array in (v,f):array._mmap.close()


def verify():
    import numpy as np
    output=BASE/'variant';manifest=inherited.read(output/'manifest.json')
    for name,row in manifest['artifacts'].items():
        p=inherited.plain(output/name)
        if inherited.digest(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise ValueError('Variant artifact changed')
    v0,f0=open_original(np)
    checker=inherited.load('_independent_orphan_ply_reader',ROOT/'docs/compatibility/amd-brick-cpu-mesh-evidence/verify_mesh.py')
    opened=[]
    def need(ok,why):
        if not ok:raise ValueError(why)
    try:
        vv,pf,header=checker.ply_arrays(np,output/'mesh.ply',325807,673604,opened,need)
        ff=pf['idx'];need(np.all(pf['count']==3),'PLY triangle schema')
        new_to_old=np.load(output/'new-to-original.npy',allow_pickle=False)
        old_to_new=np.load(output/'original-to-new.npy',allow_pickle=False)
        removed=np.load(output/'removed-original-ids.npy',allow_pickle=False)
        with np.load(DIAG/'topology.npz',allow_pickle=False) as data:
            need(np.array_equal(removed,data['unused_vertex_ids']),'Exact approved IDs')
            original_labels=data['face_component'].copy()
        used=np.unique(f0)
        need(len(removed)==425 and len(new_to_old)==325807 and np.array_equal(used,new_to_old),'Retained IDs/order')
        need(np.array_equal(np.flatnonzero(old_to_new<0),removed),'Complete reverse map')
        need(np.array_equal(old_to_new[new_to_old],np.arange(len(new_to_old))),'Reversible maps')
        need(np.array_equal(new_to_old[ff],f0),'Original triangle IDs/order/winding restored by mapping')
        need(np.array_equal(vv.view(np.uint32),v0[new_to_old].view(np.uint32)),'Exact retained vertex payloads')
        sha=surface_equal(np,v0,f0,vv,ff);need(sha==manifest['triangle_stream_sha256'],'Triangle stream SHA')
        need(np.array_equal(vv.min(axis=0).view(np.uint32),v0.min(axis=0).view(np.uint32)) and np.array_equal(vv.max(axis=0).view(np.uint32),v0.max(axis=0).view(np.uint32)),'Exact bounds')
        e0,c0,_,w0=topology.incidence(np,f0);e1,c1,_,w1=topology.incidence(np,ff)
        need(np.array_equal(new_to_old[e1],e0) and np.array_equal(c0,c1) and np.array_equal(w0,w1),'Every edge identity, incidence and winding unchanged')
        vl,fl=topology.component_labels(np,ff,len(vv))
        need(np.array_equal(fl,original_labels) and len(np.unique(fl))==10 and np.all(vl>=0),'Exact surface components, zero unused variant IDs')
        for name,array in [('vertices.npy',vv),('faces.npy',ff)]:
            saved=np.load(output/name,mmap_mode='r',allow_pickle=False)
            try:need(np.array_equal(saved.view(np.uint32),array.view(np.uint32)),'Independent PLY/NPY payload agreement')
            finally:saved._mmap.close()
        inherited.save(BASE/'independent-verification.json',dict(classification='PASS_SURFACE_IDENTICAL_ORPHAN_VARIANT',
            source_PLY_sha256=topology.PLY_SHA,variant_PLY_sha256=inherited.digest(output/'mesh.ply'),
            vertices=325807,triangles=673604,removed_exactly_approved_ids=425,triangle_stream_sha256=sha,
            triangle_coordinates_corner_order_and_face_order_bit_exact=True,all_index_mappings_verified=True,
            all_edge_ID_sets_incidence_and_winding_unchanged=True,boundary_edges=int((c1==1).sum()),
            nonmanifold_edges=int((c1>2).sum()),winding_conflicts=int(((c1==2)&(abs(w1)==2)).sum()),
            surface_components=10,unused_variant_vertices=0,bounds=manifest['bounds'],independent_PLY_NPY_readback=True,
            silhouettes_pending=True,structural_repair_performed=False))
    finally:
        for a in (v0,f0,*opened):
            m=getattr(a,'_mmap',None)
            if m is not None and not m.closed:m.close()


def silhouette_check():
    import numpy as np
    from PIL import Image
    metadata=inherited.read(BASE/'silhouettes/render-metadata.json');rows=[]
    for row in metadata['pairs']:
        for name in ('source','variant'):
            if inherited.digest(BASE/'silhouettes'/row[name])!=row[name+'_sha256']:raise ValueError('Silhouette image changed')
        with Image.open(BASE/'silhouettes'/row['source']) as a,Image.open(BASE/'silhouettes'/row['variant']) as b:
            aa=np.asarray(a.convert('RGB'));bb=np.asarray(b.convert('RGB'))
            if not np.array_equal(aa,bb):raise ValueError('Silhouette pixels changed; stop for review')
            rows.append(dict(azimuth_degrees=row['azimuth_degrees'],pixel_differences=0,mask_differences=0,
                source_pixel_sha256=hashlib.sha256(aa.tobytes()).hexdigest(),variant_pixel_sha256=hashlib.sha256(bb.tobytes()).hexdigest()))
    if len(rows)!=8 or [r['azimuth_degrees'] for r in rows]!=list(range(0,360,45)):raise ValueError('All original eight cameras required')
    inherited.save(BASE/'silhouette-verification.json',dict(classification='PASS_EIGHT_BIT_EXACT_SILHOUETTE_PAIRS',pairs=rows,
        source_PLY_sha256=topology.PLY_SHA,variant_PLY_sha256=inherited.read(BASE/'independent-verification.json')['variant_PLY_sha256']))


def vertex_fan(np,faces,vertex,incident=None):
    if incident is None:incident=np.flatnonzero(np.any(faces==vertex,axis=1))
    graph={};degree={}
    for face in faces[incident].tolist():
        other=[int(x) for x in face if x!=vertex]
        if len(other)!=2:raise ValueError('Degenerate fan triangle')
        a,b=other;graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
        degree[a]=degree.get(a,0)+1;degree[b]=degree.get(b,0)+1
    seen=set();components=0
    for start in graph:
        if start in seen:continue
        components+=1;stack=[start];seen.add(start)
        while stack:
            for node in graph[stack.pop()]:
                if node not in seen:seen.add(node);stack.append(node)
    return dict(vertex_id=int(vertex),incident_face_ids=incident.tolist(),link_components=components,
        link_degree_histogram={str(k):list(degree.values()).count(k) for k in sorted(set(degree.values()))},
        closed_single_fan=components==1 and bool(degree) and all(d==2 for d in degree.values()))


def patch_analysis():
    import numpy as np
    from PIL import Image
    if inherited.read(BASE/'silhouette-verification.json')['classification']!='PASS_EIGHT_BIT_EXACT_SILHOUETTE_PAIRS':raise ValueError('Phase A must finish before Phase B')
    v=np.load(BASE/'variant/vertices.npy',mmap_mode='r',allow_pickle=False)
    f=np.load(BASE/'variant/faces.npy',mmap_mode='r',allow_pickle=False)
    mapping=np.load(BASE/'variant/new-to-original.npy',allow_pickle=False)
    try:
        edges,counts,inverse,sums=topology.incidence(np,f);n=len(v)
        quads=topology.recover_quads(np,f);points=v[quads]
        roi=np.all((points[:,:,0]>=-.045)&(points[:,:,0]<=.045)&(points[:,:,1]>=.02)&(points[:,:,1]<=.12)&(points[:,:,2]>=.17)&(points[:,:,2]<=.23),axis=1)
        qe=np.stack([quads[:,[i,(i+1)%4]] for i in range(4)],axis=1)
        keys=edges[:,0].astype(np.int64)*n+edges[:,1]
        query=qe.min(axis=2).astype(np.int64)*n+qe.max(axis=2)
        positions=np.searchsorted(keys,query)
        if not np.array_equal(keys[positions],query):raise ValueError('Source quad edge lookup differs')
        two=np.all(counts[positions]==2,axis=1);bad=(np.abs(sums[positions])==2).sum(axis=1)
        candidates=np.flatnonzero(roi&two&(bad==4))
        selection_rule='Complete source quad in posterior upper trunk, all four perimeter edges two-face, all four currently winding-conflicted'
        if not len(candidates):
            candidates=np.flatnonzero(roi&two);selection_rule+='; no four-conflict quad exists, diagnose best two-face candidate only'
        if not len(candidates):raise ValueError('No ordinary-region two-face source quad candidate; stop for review')
        sample=np.concatenate((points[candidates],points[candidates].mean(axis=1)[:,None,:],(points[candidates]+np.roll(points[candidates],-1,axis=1))*.5),axis=1)
        clears=np.full(len(candidates),np.inf);inside_all=np.ones(len(candidates),bool);per_view=[]
        render=inherited.read(BASE/'silhouettes/render-metadata.json')
        for row in render['pairs']:
            with Image.open(BASE/'silhouettes'/row['source']) as image:mask=np.asarray(image.convert('L'))<128
            boundary=mask&~(np.roll(mask,1,0)&np.roll(mask,-1,0)&np.roll(mask,1,1)&np.roll(mask,-1,1))
            contour=np.column_stack(np.nonzero(boundary))[:,::-1]
            forward=np.array(row['target'])-np.array(row['camera_location']);forward/=np.linalg.norm(forward)
            right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right);up=np.cross(right,forward)
            d=sample-np.array(row['target']);xy=np.stack((384+(d@right)*768/row['ortho_scale'],384-(d@up)*768/row['ortho_scale']),axis=2)
            pix=np.rint(xy).astype(int);inside=np.all((pix[:,:,0]>=0)&(pix[:,:,0]<768)&(pix[:,:,1]>=0)&(pix[:,:,1]<768),axis=1)
            for i in range(len(candidates)):
                if inside[i]:inside[i]=bool(np.all(mask[pix[i,:,1],pix[i,:,0]]&~boundary[pix[i,:,1],pix[i,:,0]]))
            distance=np.empty(len(candidates))
            for i in range(len(candidates)):
                distance[i]=float(np.sqrt(((xy[i,:,None,:]-contour[None,:,:])**2).sum(axis=2).min()))
            clears=np.minimum(clears,distance);inside_all&=inside
            per_view.append((row['azimuth_degrees'],distance))
        flat=f.ravel();order=np.argsort(flat,kind='stable');num=np.bincount(flat,minlength=n);starts=np.r_[0,np.cumsum(num)]
        fan_cache={}
        def fan(vertex):
            vertex=int(vertex)
            if vertex not in fan_cache:
                ids=np.unique(order[starts[vertex]:starts[vertex+1]]//3)
                fan_cache[vertex]=vertex_fan(np,f,vertex,ids)
            return fan_cache[vertex]
        ranking=sorted(range(len(candidates)),key=lambda i:(not inside_all[i],-clears[i],int(candidates[i])))
        chosen=ranking[0]
        for index in ranking:
            if inside_all[index] and all(fan(k)['closed_single_fan'] for k in quads[candidates[index]]):chosen=index;break
        quad_id=int(candidates[chosen]);quad=quads[quad_id];face_ids=[2*quad_id,2*quad_id+1]
        fans=[fan(k) for k in quad];incident=sorted({x for row in fans for x in row['incident_face_ids']})
        prediction=proposal_from_incidence(np,f,face_ids,n,edges,counts,sums)
        go=bool(inside_all[chosen] and all(row['closed_single_fan'] for row in fans) and prediction['all_touched_edges_two_face'] and prediction['before_conflicts']==4 and prediction['predicted_after_conflicts']==0)
        original_ids=mapping[quad]
        import amd_generation_mesh as mesh
        contract=mesh._contract_api().load(inherited.SOURCE.parent/'run-contract.json')
        arrays,children=mesh.read_neural(inherited.SOURCE,inherited.SOURCE_SHA,contract=contract)
        try:
            coords=arrays['coordinates'][original_ids,1:];delta=tuple((coords[1]-coords[0]).tolist())
            axis={(0,0,1):0,(1,0,0):1,(0,1,0):2}[delta]
            features=arrays['features'][original_ids].copy();dual,flags,weights=mesh.converter.transform_features(features)
            if not flags[0,axis]:raise ValueError('Source quad activation differs')
            source=dict(source_neural_sha256=inherited.SOURCE_SHA,source_quad_ordinal=quad_id,
                original_vertex_ids=original_ids.tolist(),source_grid_coordinates=coords.tolist(),
                activated_axis=axis,anchor_activation_logit=float(features[0,3+axis]),
                original_weighted_diagonal='02' if weights[0]*weights[2]>weights[1]*weights[3] else '13')
        finally:
            for a in arrays.values():a._mmap.close()
        result=dict(classification='GO_FUTURE_TWO_FACE_WINDING_PROTOTYPE' if go else 'NO_GO_WINDING_PROTOTYPE_REQUIRES_MORE_ANALYSIS',
            selection_rule=selection_rule,eligible_quads_considered=len(candidates),
            region_bounds=[[-.045,.02,.17],[.045,.12,.23]],region='posterior central upper trunk; ordinary source skin away from named front wounds/face/hands/feet/hem',
            candidate_face_ids=face_ids,candidate_variant_vertex_ids=quad.tolist(),source=source,
            centroid=points[quad_id].mean(axis=0).tolist(),bounds=[points[quad_id].min(axis=0).tolist(),points[quad_id].max(axis=0).tolist()],
            all_projected_vertices_midpoints_and_centroid_inside_noncontour_pixels=bool(inside_all[chosen]),
            minimum_silhouette_clearance_pixels=float(clears[chosen]),
            silhouette_clearance_by_angle={str(angle):float(distance[chosen]) for angle,distance in per_view},
            silhouette_metric_is_projected_raster_clearance_not_geodesic_or_visibility=True,
            vertex_fans=fans,incident_face_ids=incident,symbolic_prediction=prediction,
            predicted_global_winding_conflicts=147484-prediction['before_conflicts']+prediction['predicted_after_conflicts'],
            expected_boundary_edges=19512,expected_nonmanifold_edges=31449,
            all_exterior_faces_fixed_in_prediction=True,structural_repair_performed=False,
            hypothetical_repaired_face_array_created=False,
            prototype_scope='Future copy only: reverse corner order on exactly these two existing triangles; keep face row order, positions and all other faces unchanged')
        inherited.save(BASE/'local-patch.json',result)
        print(json.dumps({k:result[k] for k in ('classification','eligible_quads_considered','candidate_face_ids','centroid','minimum_silhouette_clearance_pixels','predicted_global_winding_conflicts')}),flush=True)
    finally:v._mmap.close();f._mmap.close()


def compose():
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    verified=inherited.read(BASE/'independent-verification.json');sil=inherited.read(BASE/'silhouette-verification.json')
    patch=inherited.read(BASE/'local-patch.json');render=inherited.read(BASE/'patch-render/render-metadata.json')
    if sil['classification']!='PASS_EIGHT_BIT_EXACT_SILHOUETTE_PAIRS':raise ValueError('Phase A visual proof required')
    for row in render['views']:
        if inherited.digest(BASE/'patch-render'/row['name'])!=row['sha256']:raise ValueError('Patch display changed')
    output=BASE/'inspection-sheets';output.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    image=Image.new('RGB',(1280,1434),'#25282e');draw=ImageDraw.Draw(image)
    draw.text((12,12),'Source / orphan-only variant: eight exact silhouette pairs, zero pixel differences',fill='white',font=font)
    for index,row in enumerate(sil['pairs']):
        angle=row['azimuth_degrees']
        for offset,name in enumerate(('source','variant')):
            i=2*index+offset;x=(i%4)*320;y=(i//4)*346+50
            with Image.open(BASE/'silhouettes'/f'{name}-{angle:03d}.png') as view:image.paste(view.convert('RGB').resize((320,320),Image.Resampling.LANCZOS),(x,y))
            draw.text((x+8,y+323),f'{name.upper()} {angle} deg',fill='white',font=small)
    image.save(output/'eight-silhouette-pairs.png')
    image=Image.new('RGB',(1536,592),'#25282e');draw=ImageDraw.Draw(image)
    draw.text((12,12),'Unrepaired upper-back candidate: gold = two triangles, blue = complete incident vertex fans',fill='white',font=font)
    for index,row in enumerate(render['views']):
        with Image.open(BASE/'patch-render'/row['name']) as view:picture=view.convert('RGB')
        if index==0:
            p=np.array(patch['centroid'])-np.array(row['target'])
            forward=np.array(row['target'])-np.array(row['camera_location']);forward/=np.linalg.norm(forward)
            right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right);up=np.cross(right,forward)
            x=384+float(p@right)*768/row['ortho_scale'];y=384-float(p@up)*768/row['ortho_scale']
            mark=ImageDraw.Draw(picture);mark.ellipse((x-14,y-14,x+14,y+14),outline='#ffcc33',width=3)
            mark.text((x+20,y-10),'candidate location',font=small,fill='#ffcc33')
        image.paste(picture.resize((512,512),Image.Resampling.LANCZOS),(index*512,50))
        draw.text((index*512+8,565),row['name'].replace('.png',''),fill='white',font=small)
    image.save(output/'local-candidate-context.png')
    declaration=dict(classification=patch['classification'],analysis_only=True,applied=False,
        baseline_variant_PLY_sha256=verified['variant_PLY_sha256'],source_PLY_sha256=topology.PLY_SHA,
        source_quad_ordinal=patch['source']['source_quad_ordinal'],face_ids=patch['candidate_face_ids'],
        operation='Future separate copy only: swap corner slots 1 and 2 on each specified face (permutation [0,2,1]); keep face rows and all vertices unchanged',
        expected_global_winding_conflicts=patch['predicted_global_winding_conflicts'],
        expected_boundary_edges=19512,expected_nonmanifold_edges=31449,expected_surface_components=10,
        expected_vertices=325807,expected_triangles=673604,
        complete_local_closed_face_fans=all(row['closed_single_fan'] for row in patch['vertex_fans']),
        selected_faces_visible_in_closeup=all(row['selected_face_is_first_hit'] for row in render['closeup_raycast_witness']),
        no_sheet_split_needed_for_this_locally_manifold_quad=True,
        acceptance='Only these two corner-order rows may differ; every face retains its exact coordinate multiset and row ID, all other face rows/vertices/maps/bounds/components/incidences/silhouettes unchanged; local five edge signed sums all zero',
        global_repair_or_automatic_propagation_authorized=False)
    inherited.save(BASE/'future-prototype-spec.json',declaration)
    inherited.save(BASE/'variant/acceptance.json',dict(classification='PASS_APPROVED_ORPHAN_ONLY_VARIANT',
        independent_verification_sha256=inherited.digest(BASE/'independent-verification.json'),
        silhouette_verification_sha256=inherited.digest(BASE/'silhouette-verification.json'),
        initial_manifest_sha256=inherited.digest(BASE/'variant/manifest.json'),
        source_PLY_sha256=topology.PLY_SHA,variant_PLY_sha256=verified['variant_PLY_sha256'],vertices=325807,triangles=673604,
        triangle_stream_sha256=verified['triangle_stream_sha256'],eight_view_silhouette_pixel_differences=0,
        removed_only_unreferenced_vertices=425,structural_repair_performed=False))
    inherited.save(output/'sheets.json',dict(classification='PASS_CLEANUP_AND_UNREPAIRED_PATCH_DISPLAY',
        annotations_are_display_only=True,assets={p.name:dict(bytes=p.stat().st_size,sha256=inherited.digest(p)) for p in output.glob('*.png')}))


def owner(stage):
    module=inherited.load('_zombie_orphan_owned_cpu',Path(inherited.__file__))
    tree=ast.parse(inspect.getsource(module.supervise));changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='amd_zombie_mesh_render.py':node.value='amd_zombie_orphan_render.py';changed+=1
    if changed!=1:raise ValueError('Preserved owner render route differs')
    module.__file__=__file__;module.BASE=BASE
    exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),module.__dict__)
    if stage in ('silhouettes','patch-render'):
        module.BASE=BASE/('render-'+stage);module.BASE.mkdir(exist_ok=False)
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
    if stage=='cleanup':cleanup()
    elif stage=='verify':verify()
    elif stage=='silhouette-check':silhouette_check()
    elif stage=='patch':patch_analysis()
    elif stage=='compose':compose()
    else:raise ValueError('Unknown bounded stage')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('cleanup','verify','silhouettes','silhouette-check','patch','patch-render','compose'))
    parser.add_argument('--worker',choices=('cleanup','verify','silhouette-check','patch','compose'))
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.stage:
        module=owner(args.stage)
        if args.stage in ('silhouettes','patch-render'):
            os.environ['MODLY_ZOMBIE_ORPHAN_MODE']=args.stage;raise SystemExit(module.supervise('render'))
        raise SystemExit(module.supervise(args.stage))
    else:print('DRY_RUN: explicit approved copy-only cleanup or analysis stage required')
