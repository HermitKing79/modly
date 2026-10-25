"""Read-only source wound geometry, layered visibility and conservative role audit."""
import argparse,ast,hashlib,inspect,json,os,sys
from collections import deque
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT.parent/'artifacts/amd-zombie-wound-source-audit-20261025-v1'
OLD=ROOT.parent/'artifacts/amd-zombie-orphan-cleanup-20261025-v1'
EVIDENCE=ROOT.parent/'artifacts/amd-zombie-retopology-feasibility-20261025-v1'
BASELINE_SHA='377a15dc4b5a27936f7e62b311dd03e8337685344b0f8204b8f831252176b1a5'
# Manually traced from the authenticated 512px full-source frontal reference.
# This visual outline is an inspection constraint, not a watertight 3D rim.
WOUND_OUTLINE=[(215,70),(263,87),(287,105),(336,128),(349,154),(333,190),(321,216),(300,239),(282,261),(252,276),(209,276),(178,264),(153,245),(138,222),(131,199),(129,162),(150,150),(174,130),(190,108),(201,85)]
sys.path.insert(0,str(Path(__file__).parent))
import amd_zombie_winding as previous
inherited=previous.inherited;prior=previous.prior;topology=previous.topology;need=previous.need


def intersections(np,origin,direction,triangles):
    """F64 two-sided Moller-Trumbore; return all forward hits, no normal-sign test."""
    a=triangles[:,0];e1=triangles[:,1]-a;e2=triangles[:,2]-a
    p=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.einsum('ij,ij->i',e1,p)
    good=abs(det)>1e-15;inv=np.zeros_like(det);inv[good]=1/det[good]
    tvec=origin-a;u=np.einsum('ij,ij->i',tvec,p)*inv;q=np.cross(tvec,e1)
    v=q@direction*inv;t=np.einsum('ij,ij->i',e2,q)*inv
    valid=good&(u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7)&(t>=0)
    ids=np.flatnonzero(valid);order=np.argsort(t[ids],kind='stable')
    return ids[order],t[ids][order]


def depth_groups(depths,epsilon):
    groups=[]
    for i,t in enumerate(depths):
        if not groups or t-depths[groups[-1][0]]>epsilon:groups.append([i])
        else:groups[-1].append(i)
    return groups


def role(*,visible_views,branch,skin,cavity,rim):
    if not visible_views or branch or sum((skin,cavity,rim))!=1:return 0
    if skin:return 1 if visible_views>=2 else 0
    return 3 if cavity else 2


def inside_polygon(point,polygon):
    x,y=point;inside=False
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:inside=not inside
    return inside


def authenticate():
    import numpy as np
    previous.baseline();contract,auth=inherited.authenticate()
    need(inherited.digest(EVIDENCE/'artifact-inventory.json')=='396c604b378389ff11a3aabb00d475af252cf6dbabab2be1853265067a216768','Prior evidence seal differs')
    seal=inherited.read(EVIDENCE/'artifact-inventory.json')
    for name,row in seal['artifacts'].items():
        p=inherited.plain(EVIDENCE/name);need(p.stat().st_size==row['bytes'] and inherited.digest(p)==row['sha256'],'Prior sealed evidence changed')
    settings=inherited.read(EVIDENCE/'experiment-settings.json');region=settings['regions']['chest-wound']
    need(settings['baseline_PLY_sha256']==BASELINE_SHA,'Prior reference differs')
    views=[r for r in inherited.read(EVIDENCE/'renders/render-metadata.json')['frames'] if r['region']=='chest-wound' and r['kind']=='reference' and r['mode']=='neutral']
    need([r['angle'] for r in views]==[-35,0,35],'Three original matched cameras required')
    for row in views:need(inherited.digest(EVIDENCE/'renders'/row['name'])==row['sha256'],'Prior reference frame changed')
    folder=EVIDENCE/'reference/chest-wound'
    for name,sha in region['reference_arrays'].items():need(inherited.digest(folder/name)==sha,'Prior crop arrays changed')
    need(inherited.digest(folder/'mesh.ply')==region['reference_PLY_sha256'],'Prior crop PLY changed')
    witness=inherited.read(EVIDENCE/'view-measurements.json')['chest-wound']['0']['reference']['wound_depth_witness']
    inherited.save(BASE/'inputs.json',dict(baseline_PLY_sha256=BASELINE_SHA,neural_sha256=inherited.SOURCE_SHA,
        contract_sha256=contract.sha256,region=region,views=views,wound_depth_witness=witness,
        prior_seal_verified_files=len(seal['artifacts']),layer_cap=16,
        layer_epsilon=8*float(np.finfo(np.float32).eps),
        limit_note='BVH continuation groups sub-epsilon hits and is capped; exact F64 deep-ray intersections audit ties and layer coverage separately',
        no_geometry_written=True))
    inherited.save(BASE/'source-authentication.json',auth)


def bfs(graph,seeds,allowed):
    distance=[-1]*len(graph);parent=[-1]*len(graph);queue=deque()
    for i in seeds:
        if allowed[i]:distance[i]=0;queue.append(int(i))
    while queue:
        a=queue.popleft()
        for b in graph[a]:
            if allowed[b] and distance[b]<0:distance[b]=distance[a]+1;parent[b]=a;queue.append(b)
    return distance,parent


def shortest_rim_path(graph,points,seeds,targets,rim,allowed):
    """Shortest geometric adjacency witness touching a rim seed; not a repair."""
    import heapq,math
    distance={};parents={};queue=[]
    for seed in seeds:
        seed=int(seed)
        if allowed[seed]:
            state=(seed,int(seed in rim));distance[state]=0.;parents[state]=None;heapq.heappush(queue,(0.,*state))
    while queue:
        cost,a,touched=heapq.heappop(queue);state=(a,touched)
        if cost!=distance.get(state):continue
        if a in targets and touched:
            path=[];node=state
            while node is not None:path.append(node[0]);node=parents[node]
            return dict(path=list(reversed(path)),centroid_path_length=cost)
        for b in graph[a]:
            if not allowed[b]:continue
            next_state=(b,int(touched or b in rim));value=cost+math.dist(points[a],points[b])
            if value<distance.get(next_state,float('inf')):
                distance[next_state]=value;parents[next_state]=state;heapq.heappush(queue,(value,*next_state))
    return None


def features():
    import numpy as np
    inputs=inherited.read(BASE/'inputs.json');r=inputs['region'];h=r['chart_grid_spacing_requested']
    v=np.load(OLD/'variant/vertices.npy',mmap_mode='r',allow_pickle=False);f=np.load(OLD/'variant/faces.npy',mmap_mode='r',allow_pickle=False)
    try:
        ids=np.load(BASE/'role-context-face-ids.npy',allow_pickle=False);labels=np.load(BASE/'face-role-labels.npy',allow_pickle=False)
        deep=np.load(BASE/'deep-source-face-ids.npy',allow_pickle=False);unique=np.unique(deep)
        p=v[f[ids]].astype(np.float64);cent=p.mean(axis=1);mapping=np.full(len(f),-1,np.int32);mapping[ids]=np.arange(len(ids),dtype=np.int32)
        e,c,inv,s=topology.incidence(np,f);order=np.argsort(inv,kind='stable');starts=np.r_[0,np.cumsum(c)];two=np.flatnonzero(c==2)
        a=order[starts[two]]%len(f);b=order[starts[two]+1]%len(f);both=(mapping[a]>=0)&(mapping[b]>=0)
        graph=[[] for _ in ids]
        for x,y in zip(mapping[a[both]].tolist(),mapping[b[both]].tolist()):graph[x].append(y);graph[y].append(x)
        branch=np.any(c[inv.reshape(3,len(f))[:,ids]]>2,axis=0)
        deep_rows=mapping[unique];deepest_cut=float(np.quantile(cent[deep_rows,1],.75))
        targets={int(k) for k in deep_rows if cent[k,1]>=deepest_cut and labels[ids[k]]==3}
        witness=shortest_rim_path(graph,cent.tolist(),np.flatnonzero(labels[ids]==1).tolist(),targets,set(np.flatnonzero(labels[ids]==2).tolist()),(~branch).tolist())
        records=inherited.read(BASE/'deep-intersections.json')['rows'];near_hist={};gaps=[]
        for row in records:
            ys=row['group_world_y'];near=sum(y<=r['hi'][1] for y in ys);near_hist[str(near)]=near_hist.get(str(near),0)+1
            gaps.extend(b-a for a,b in zip(ys,ys[1:]))
        summary=dict(classification='PRELIMINARY_DEPTH_ONLY_TRANSITION_REQUIRES_VISUAL_VALIDATION',
            analysis_only=True,geometry_changed=False,reconstruction_started=False,
            deep_ray_near_front_crop_layer_histogram=near_hist,
            positive_layer_y_gap_summary=dict(minimum=float(min(gaps)),median=float(np.median(gaps)),maximum=float(max(gaps))),
            target_selection='Deepest quartile by centroid Y among observed saved deep faces; target must be probable cavity',
            deepest_quartile_centroid_y=deepest_cut,qualifying_targets=len(targets),
            source_layer_events_not_automatically_semantic_sheets=True)
        if witness:
            rows=np.array(witness['path'],np.int32);source=ids[rows];points=p[rows]
            lo=points.min(axis=(0,1))-2*h;hi=points.max(axis=(0,1))+2*h
            sector=np.flatnonzero(np.any(np.all((p>=lo)&(p<=hi),axis=2),axis=1))
            source_sector=ids[sector];covered=np.isin(deep,source_sector)
            prior.save_npy(BASE/'proposed-sector-source-face-ids.npy',source_sector)
            summary['transition_witness']=dict(source_face_ids=source.tolist(),role_codes=labels[source].tolist(),
                centroid_path_length=witness['centroid_path_length'],source_triangles_on_path=len(rows),
                source_path_bounds=[points.min(axis=(0,1)).tolist(),points.max(axis=(0,1)).tolist()],
                all_path_faces_avoid_global_nonmanifold_edge_incidence=bool(not branch[rows].any()),
                explicit_probable_rim_faces_on_path=source[labels[source]==2].tolist(),
                start_face_id=int(source[0]),target_face_id=int(source[-1]),
                source_face_positions_and_corners_unchanged=True)
            summary['proposed_sector']=dict(source_face_count=len(source_sector),source_vertex_count=len(np.unique(f[source_sector])),
                bounds=[lo.tolist(),hi.tolist()],saved_deep_samples_in_sector=int(covered.sum()),
                directly_nonmanifold_incident_faces=int(branch[sector].sum()),
                source_role_counts={str(k):int((labels[source_sector]==k).sum()) for k in range(4)},
                immutable_reference_face_id_list_sha256=inherited.digest(BASE/'proposed-sector-source-face-ids.npy'),
                note='A future test domain, not an extracted or reconstructed mesh. Uncertain faces remain reference-only until explicitly resolved.')
        inherited.save(BASE/'source-sufficiency.json',summary);print(json.dumps(summary),flush=True)
    finally:v._mmap.close();f._mmap.close()


def analyse():
    import numpy as np
    inputs=inherited.read(BASE/'inputs.json');eps=inputs['layer_epsilon'];r=inputs['region'];witness=inputs['wound_depth_witness']
    v=np.load(OLD/'variant/vertices.npy',mmap_mode='r',allow_pickle=False);f=np.load(OLD/'variant/faces.npy',mmap_mode='r',allow_pickle=False)
    try:
        ids=np.load(EVIDENCE/'reference/chest-wound/original-face-ids.npy',allow_pickle=False)
        layers=np.load(BASE/'layer-face-ids.npy',mmap_mode='r',allow_pickle=False);depths=np.load(BASE/'layer-depths.npy',mmap_mode='r',allow_pickle=False)
        origins=np.load(BASE/'ray-origins.npy',allow_pickle=False);directions=np.load(BASE/'ray-directions.npy',allow_pickle=False)
        old_face=np.load(EVIDENCE/'renders/chest-wound-+00-reference-face-ids.npy',allow_pickle=False)
        old_depth=np.load(EVIDENCE/'renders/chest-wound-+00-reference-depth.npy',allow_pickle=False)
        front=1;points=origins[front]+directions[front]*old_depth[:,:,None]
        roi=(origins[front,:,:,0]>=.03)&(origins[front,:,:,0]<=.10)&(origins[front,:,:,2]>=.15)&(origins[front,:,:,2]<=.23)
        deep=roi&(old_face>=0)&(points[:,:,1]>witness['deep_y_threshold']);pixels=np.argwhere(deep)
        need(len(pixels)==988,'Original 988 deep-wound witnesses must reproduce')
        # Independent all-intersection audit, using all full-source triangles whose
        # XZ bounding boxes overlap the deep-ray window, not just the old crop.
        p=v[f];low=p.min(axis=1);high=p.max(axis=1)
        pool=np.flatnonzero((high[:,0]>=.03)&(low[:,0]<=.10)&(high[:,2]>=.15)&(low[:,2]<=.23))
        triangles=p[pool].astype(np.float64);plow=triangles.min(axis=1);phigh=triangles.max(axis=1)
        records=[];source_found=0;nearer=0;first_equivalent=0;multi=0;tie_events=0;layer_mismatches=0
        for j,i in pixels:
            origin=origins[front,j,i];direction=directions[front];target=int(ids[old_face[j,i]])
            candidates=np.flatnonzero((plow[:,0]-1e-8<=origin[0])&(phigh[:,0]+1e-8>=origin[0])&(plow[:,2]-1e-8<=origin[2])&(phigh[:,2]+1e-8>=origin[2]))
            local,t=intersections(np,origin,direction,triangles[candidates]);hit_ids=pool[candidates[local]];groups=depth_groups(t.tolist(),eps)
            matches=np.flatnonzero(hit_ids==target);found=bool(len(matches));source_found+=found
            old_t=float(old_depth[j,i]);is_nearer=bool(len(t) and t[0]<old_t-4*eps);nearer+=is_nearer
            equivalent=bool(len(t) and abs(float(t[0])-old_t)<=4*eps);first_equivalent+=equivalent
            multi+=len(groups)>1;tie_events+=sum(len(group)>1 for group in groups)
            group_depth=[float(t[g[0]]) for g in groups];bvh_depth=depths[front,j,i];bvh_depth=bvh_depth[np.isfinite(bvh_depth)]
            # Continuation may merge thin/tied events; record disagreements rather than invent surfaces.
            agrees=len(group_depth)==len(bvh_depth) and all(abs(a-b)<4*eps for a,b in zip(group_depth,bvh_depth))
            layer_mismatches+=not agrees
            records.append(dict(pixel=[int(j),int(i)],saved_source_face_id=target,saved_depth=old_t,
                saved_triangle_intersection_found=found,full_first_depth_equivalent=equivalent,nearer_full_source_occluder=is_nearer,
                exact_hit_face_ids=hit_ids.tolist(),exact_hit_depths=t.tolist(),depth_groups=groups,
                BVH_group_depths_agree=bool(agrees),group_world_y=[float(origin[1]+direction[1]*x) for x in group_depth]))
        need(source_found==988,'Every saved deep hit must be present in original source triangles')
        prior.save_npy(BASE/'deep-pixels.npy',pixels.astype(np.int32));prior.save_npy(BASE/'deep-source-face-ids.npy',ids[old_face[deep]].astype(np.int32))
        inherited.save(BASE/'deep-intersections.json',dict(rows=records,epsilon=eps,all_source_triangles_considered_in_XZ_window=len(pool)))
        # Face visibility, unsigned shape and depth. Original winding is never a role predicate.
        bits=np.zeros(len(f),np.uint8);cavity_seed=np.zeros(len(f),bool)
        for a in range(3):
            first=layers[a,:,:,0];valid=first>=0;bits[np.unique(first[valid])]|=1<<a
            pts=origins[a]+directions[a]*depths[a,:,:,0,None]
            support=valid&(pts[:,:,0]>=.03)&(pts[:,:,0]<=.10)&(pts[:,:,2]>=.15)&(pts[:,:,2]<=.23)&(pts[:,:,1]>witness['deep_y_threshold'])&(pts[:,:,1]<=r['hi'][1])
            cavity_seed[first[support]]=True
        visibility=((bits&1)>0).astype(np.int8)+((bits&2)>0).astype(np.int8)+((bits&4)>0).astype(np.int8)
        e,c,inverse,sums=topology.incidence(np,f);incident_counts=c[inverse.reshape(3,len(f))]
        branch=np.any(incident_counts>2,axis=0);opening=np.any(incident_counts==1,axis=0)
        order=np.argsort(inverse,kind='stable');starts=np.r_[0,np.cumsum(c)];two=np.flatnonzero(c==2)
        aa=order[starts[two]]%len(f);bb=order[starts[two]+1]%len(f)
        map_to_local=np.full(len(f),-1,np.int32);map_to_local[ids]=np.arange(len(ids),dtype=np.int32)
        both=(map_to_local[aa]>=0)&(map_to_local[bb]>=0);a=map_to_local[aa[both]];b=map_to_local[bb[both]]
        graph=[[] for _ in ids]
        for x,y in zip(a.tolist(),b.tolist()):graph[x].append(y);graph[y].append(x)
        cent=p[ids].mean(axis=1).astype(np.float64);cross=np.cross(p[ids,1]-p[ids,0],p[ids,2]-p[ids,0]);abs_y=abs(cross[:,1])/np.linalg.norm(cross,axis=1)
        in_roi=(cent[:,0]>=.03)&(cent[:,0]<=.10)&(cent[:,2]>=.15)&(cent[:,2]<=.23)
        h=r['chart_grid_spacing_requested'];cut=witness['deep_y_threshold'];skin_y=witness['skin_depth_reference_y']
        with np.load(ROOT.parent/'artifacts/amd-zombie-topology-diagnosis-20261025-v1/topology.npz',allow_pickle=False) as topo:components=topo['face_component'][ids].copy()
        skin=(~in_roi)&(cent[:,1]<=cut)&(abs_y>=np.cos(np.deg2rad(35)))&(components==0)
        cavity=cavity_seed[ids]&(cent[:,1]>cut)
        adjacent_deep=np.zeros(len(ids),bool)
        adjacent_deep[a[cavity[b]]]=True;adjacent_deep[b[cavity[a]]]=True
        rim=in_roi&(cent[:,1]>=skin_y-2*h)&(cent[:,1]<=cut)&((opening[ids]&(abs_y<np.cos(np.deg2rad(35))))|adjacent_deep)
        labels=np.array([role(visible_views=int(visibility[g]),branch=bool(branch[g]),skin=bool(skin[k]),cavity=bool(cavity[k]),rim=bool(rim[k])) for k,g in enumerate(ids)],np.uint8)
        all_labels=np.full(len(f),4,np.uint8)
        # Also mark farther body sheets in the local window uncertain, never as cavity automatically.
        local_window=(cent[:,0]>=r['lo'][0])&(cent[:,0]<=r['hi'][0])
        context=np.flatnonzero((low[:,0]>=r['lo'][0])&(high[:,0]<=r['hi'][0])&(low[:,2]>=r['lo'][2])&(high[:,2]<=r['hi'][2])&(low[:,1]>=r['lo'][1])&(high[:,1]<=.12))
        all_labels[context]=0;all_labels[ids]=labels
        prior.save_npy(BASE/'face-role-labels.npy',all_labels);prior.save_npy(BASE/'role-context-face-ids.npy',ids)
        skin_rows=np.flatnonzero(labels==1);distance,parent=bfs(graph,skin_rows,[True]*len(ids));clean_distance,clean_parent=bfs(graph,skin_rows,(~branch[ids]).tolist())
        deep_ids=ids[old_face[deep]];local_deep=map_to_local[deep_ids]
        role_hist={str(k):int((all_labels[deep_ids]==k).sum()) for k in range(4)}
        deep_unique=np.unique(local_deep);candidate_rim=np.flatnonzero(labels==2);candidate_cavity=np.flatnonzero(labels==3)
        role_stats={}
        for code,name in [(1,'high_confidence_exterior_candidate'),(2,'probable_rim'),(3,'probable_cavity'),(0,'uncertain')]:
            rows=np.flatnonzero(labels==code);members=set(rows.tolist());seen=set();sizes=[]
            for start in rows:
                start=int(start)
                if start in seen:continue
                queue=[start];seen.add(start);size=0
                while queue:
                    x=queue.pop();size+=1
                    for y in graph[x]:
                        if y in members and y not in seen:seen.add(y);queue.append(y)
                sizes.append(size)
            role_stats[name]=dict(faces=len(rows),two_face_adjacency_components=len(sizes),largest_component_faces=max(sizes,default=0),
                global_source_component_ids=sorted(np.unique(components[rows]).tolist()))
        prior.save_npy(BASE/'skin-connectivity-hop-distance.npy',np.array(distance,np.int32));prior.save_npy(BASE/'unbranched-skin-connectivity-hop-distance.npy',np.array(clean_distance,np.int32))
        void=roi&(old_face<0);fullhit=layers[front,:,:,0]>=0
        fullpoints=origins[front]+directions[front]*depths[front,:,:,0,None]
        summary=dict(classification='READ_ONLY_SOURCE_GEOMETRY_AUDIT_COMPLETE',baseline_PLY_sha256=BASELINE_SHA,
            saved_deep_samples=988,saved_deep_sample_triangle_intersections_found=source_found,full_first_hit_depth_equivalent=first_equivalent,
            nearer_full_source_occluders=nearer,deep_rays_with_multiple_depth_groups=multi,shared_depth_tie_events=tie_events,
            deep_rays_with_BVH_vs_exact_group_disagreement=layer_mismatches,
            exact_layer_group_histogram={str(k):sum(len(row['depth_groups'])==k for row in records) for k in sorted({len(row['depth_groups']) for row in records})},
            source_face_roles=role_stats,deep_sample_role_histogram=role_hist,
            deep_samples_on_nonmanifold_incident_faces=int(branch[deep_ids].sum()),
            deep_samples_with_two_face_path_to_skin=int(sum(distance[x]>=0 for x in local_deep)),
            deep_samples_with_nonmanifold_free_two_face_path_to_skin=int(sum(clean_distance[x]>=0 for x in local_deep)),
            unique_deep_faces=len(deep_unique),unique_deep_faces_with_clean_skin_path=int(sum(clean_distance[x]>=0 for x in deep_unique)),
            prior_crop_void_pixels_inside_ROI=int(void.sum()),prior_crop_voids_hitting_full_source=int((void&fullhit).sum()),
            prior_crop_voids_hitting_beyond_prior_front_crop=int((void&fullhit&(fullpoints[:,:,1]>r['hi'][1])).sum()),
            role_rules=dict(skin='Two or more first-hit views; outside wound ROI; shallow, main-component, unsigned normal within 35deg of Y; no incident >2-face edge',
                cavity='Observed first-hit deep inside wound ROI, centroid deeper than measured skin threshold; no incident >2-face edge',
                rim='Visible shallow transition within wound ROI, adjoining deep candidate or steep genuine opening edge; no incident >2-face edge',
                uncertain='Unseen, conflicting labels, incident nonmanifold ownership, or unproven farther body surfaces; no forced assignment'),
            thresholds_are_analysis_rules_not_resource_gates=True,winding_used_for_exterior_classification=False,
            source_geometry_changed=False,replacement_mesh_created=False,
            limitations=['Role colors are conservative geometric candidates, not semantic ground truth','Two-face paths are local to the unchanged prior crop, not global absence proofs','Layer continuation cannot resolve sub-epsilon sheets; exact deep-ray ties retain ambiguity'])
        inherited.save(BASE/'analysis.json',summary);print(json.dumps(summary),flush=True)
    finally:v._mmap.close();f._mmap.close()


def owner(stage):
    module=inherited.load('_zombie_wound_owned_cpu',Path(inherited.__file__));tree=ast.parse(inspect.getsource(module.supervise));changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='amd_zombie_mesh_render.py':node.value='amd_zombie_wound_source_blender.py';changed+=1
    need(changed==1,'Qualified owner render route differs');module.__file__=__file__;module.BASE=BASE
    exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),module.__dict__)
    if stage in ('rays','render'):module.BASE=BASE/('owned-'+stage);module.BASE.mkdir(exist_ok=False)
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
    {'authenticate':authenticate,'analyse':analyse,'features':features,'compose':compose,'landmarks':landmarks,'semantic':semantic}[stage]()


def landmarks():
    from PIL import Image,ImageDraw,ImageFont
    out=BASE/'landmarks';out.mkdir(exist_ok=False)
    with Image.open(BASE/'renders/full-reference-+00-neutral.png') as view:image=view.convert('RGB')
    draw=ImageDraw.Draw(image);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
    for x in range(0,512,32):draw.line((x,0,x,511),fill='#3355bb',width=1);draw.text((x+1,0),str(x),font=font,fill='#ffee44')
    for y in range(0,512,32):draw.line((0,y,511,y),fill='#3355bb',width=1);draw.text((0,y+1),str(y),font=font,fill='#ffee44')
    image.save(out/'frontal-pixel-grid.png')


def semantic():
    """Validate depth proxies against visible anatomy, keeping unsupported roles uncertain."""
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    inputs=inherited.read(BASE/'inputs.json');r=inputs['region'];rows=inputs['views'];front=rows[1]
    out=BASE/'semantic-inspection';out.mkdir(exist_ok=False)
    v=np.load(OLD/'variant/vertices.npy',mmap_mode='r',allow_pickle=False);f=np.load(OLD/'variant/faces.npy',mmap_mode='r',allow_pickle=False)
    try:
        ids=np.load(BASE/'role-context-face-ids.npy',allow_pickle=False);cent=v[f[ids]].mean(axis=1).astype(np.float64)
        direction=np.load(BASE/'ray-directions.npy',allow_pickle=False)[1];right=np.cross(direction,[0.,0.,1.]);right/=np.linalg.norm(right);up=np.cross(right,direction)
        def project(points):
            d=points-np.array(front['target']);return np.stack((256+(d@right)*512/front['ortho_scale'],256-(d@up)*512/front['ortho_scale']),axis=1)
        xy=project(cent);inside=np.array([inside_polygon(p,WOUND_OUTLINE) for p in xy])
        internal_polygons=[[(222,94),(249,109),(261,142),(274,173),(286,204),(278,226),(259,230),(243,205),(231,183),(226,151)],
            [(154,214),(178,212),(205,224),(233,227),(231,246),(213,260),(191,257),(171,248),(157,233)]]
        skin_polygons=[[(76,134),(110,134),(110,211),(76,211)],[(120,300),(190,300),(190,345),(120,345)],[(380,185),(420,185),(420,245),(380,245)]]
        layers=np.load(BASE/'layer-face-ids.npy',allow_pickle=False);depth=np.load(BASE/'layer-depths.npy',allow_pickle=False);origins=np.load(BASE/'ray-origins.npy',allow_pickle=False);directions=np.load(BASE/'ray-directions.npy',allow_pickle=False)
        bits=np.zeros(len(f),np.uint8)
        for a in range(3):
            hits=layers[a,:,:,0];bits[np.unique(hits[hits>=0])]|=1<<a
        visibility=((bits&1)>0).astype(np.int8)+((bits&2)>0).astype(np.int8)+((bits&4)>0).astype(np.int8)
        e,c,inv,s=topology.incidence(np,f);incident=c[inv.reshape(3,len(f))[:,ids]];branch=np.any(incident>2,axis=0);opening=np.any(incident==1,axis=0)
        band=np.full(len(xy),np.inf)
        for a,b in zip(WOUND_OUTLINE,WOUND_OUTLINE[1:]+WOUND_OUTLINE[:1]):
            a=np.array(a);b=np.array(b);edge=b-a;t=np.clip((xy-a)@edge/(edge@edge),0,1);band=np.minimum(band,np.linalg.norm(xy-(a+t[:,None]*edge),axis=1))
        one_edge_pixels=r['median_source_edge_length']*512/front['ortho_scale']
        skin=np.array([any(inside_polygon(p,poly) for poly in skin_polygons) for p in xy])&~inside
        inner=np.array([any(inside_polygon(p,poly) for poly in internal_polygons) for p in xy])&inside
        rim=(band<=one_edge_pixels)&opening
        # Frontal visibility and multiple views are required for hand-identified
        # interior candidates. They may still be folded rim/flaps, not cavity walls.
        labels=np.zeros(len(ids),np.uint8)
        for k,g in enumerate(ids):
            if branch[k] or not(bits[g]&2):continue
            if skin[k] and visibility[g]>=2:labels[k]=1
            elif rim[k]:labels[k]=2
            elif inner[k] and visibility[g]>=2:labels[k]=3
        all_labels=np.full(len(f),4,np.uint8);all_labels[ids]=labels;prior.save_npy(out/'face-role-labels.npy',all_labels)
        deep_pixels=np.load(BASE/'deep-pixels.npy',allow_pickle=False);deep_ids=np.load(BASE/'deep-source-face-ids.npy',allow_pickle=False)
        deep_inside=np.array([inside_polygon((4*int(i)+2,4*int(j)+2),WOUND_OUTLINE) for j,i in deep_pixels])
        pix_wound=np.array([[inside_polygon((4*i+2,4*j+2),WOUND_OUTLINE) for i in range(128)] for j in range(128)])
        full_face=layers[1,:,:,0];valid=full_face>=0;yy=origins[1,:,:,1]+directions[1,1]*depth[1,:,:,0]
        local=valid&(yy<=r['hi'][1]);far=valid&~local
        prelim=inherited.read(BASE/'source-sufficiency.json');path=prelim.get('transition_witness',{}).get('source_face_ids',[])
        path_cent=v[f[path]].mean(axis=1) if path else np.empty((0,3));path_xy=project(path_cent)
        path_inside=sum(inside_polygon(p,WOUND_OUTLINE) for p in path_xy)
        # Record exact geometric support without equating image-seeded identity
        # to a verified, separated cavity wall or a continuous source rim.
        label_stats={str(k):int((labels==k).sum()) for k in range(4)}
        records=dict(classification='NO_GO_RELIABLE_COMPLETE_WOUND_SHEET_SEPARATION',
            baseline_PLY_sha256=BASELINE_SHA,source_geometry_modified=False,replacement_geometry_created=False,
            reference_frontal_image_sha256=inherited.digest(BASE/'renders/full-reference-+00-neutral.png'),
            visual_outline_pixels=WOUND_OUTLINE,manual_exterior_seed_polygons=skin_polygons,manual_internal_candidate_polygons=internal_polygons,
            semantic_labels_are_visual_candidates_not_anatomical_ground_truth=True,
            source_triangle_role_counts=label_stats,
            old_depth_samples=988,old_depth_samples_inside_visible_wound_outline=int(deep_inside.sum()),
            old_depth_samples_outside_visible_wound_outline=int((~deep_inside).sum()),
            all_988_source_intersections_remain_verified=True,
            corrected_deep_sample_role_histogram={str(k):int((all_labels[deep_ids]==k).sum()) for k in range(4)},
            visible_wound_outline_ray_samples=int(pix_wound.sum()),near_front_crop_source_hits=int((pix_wound&local).sum()),
            farther_torso_source_hits=int((pix_wound&far).sum()),no_full_source_hit_samples=int((pix_wound&~valid).sum()),
            preliminary_path_face_count=len(path),preliminary_path_faces_inside_visible_wound_outline=path_inside,
            preliminary_depth_only_GO_rejected=True,
            supersedes_semantic_interpretation_in=['analysis.json','source-sufficiency.json','future-multisheet-proposal.json'],
            supported_features=['Exterior torso surfaces are represented','Ragged opening edges and visible internal strips/flaps are represented by triangles','Farther torso sheets exist behind many openings'],
            unresolved_features=['No complete reliable three-dimensional rim/cavity ownership has been established','Visible internal strips may be folded exterior rim/flaps rather than distinct cavity walls','Open gaps have distant-body or no-hit evidence, not a source-supported local cavity floor','Nonmanifold incidences and sparse/occluded face support remain uncertain'],
            decision='Do not authorize multi-sheet reconstruction from these labels. Improve upstream wound geometric coherence and explicit cavity-wall definition before another automated reconstruction attempt; do not launch generation now.',
            followup_executed=False)
        inherited.save(BASE/'semantic-validation.json',records)
        with Image.open(BASE/'renders/full-reference-+00-neutral.png') as view:picture=view.convert('RGB')
        mark=ImageDraw.Draw(picture);mark.line(WOUND_OUTLINE+[WOUND_OUTLINE[0]],fill='#ffff33',width=2)
        for poly in skin_polygons:mark.line(poly+[poly[0]],fill='#3388ff',width=2)
        for poly in internal_polygons:mark.line(poly+[poly[0]],fill='#22dd99',width=2)
        picture.save(out/'visual-landmark-constraints.png')
        palette=np.array([[185,75,195],[45,135,235],[250,150,20],[30,185,115],[85,85,85]],np.uint8)
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
        image=Image.new('RGB',(1536,1655),'#24272c');draw=ImageDraw.Draw(image)
        for a,row in enumerate(rows):
            first=layers[a,:,:,0];good=first>=0;rgb=np.full((128,128,3),15,np.uint8);rgb[good]=palette[all_labels[first[good]]]
            for col,kind in enumerate(('saved-crop','full-neutral','candidate-roles')):
                if kind=='candidate-roles':pic=Image.fromarray(rgb).resize((512,512),Image.Resampling.NEAREST)
                else:
                    p=EVIDENCE/'renders'/row['name'] if kind=='saved-crop' else BASE/'renders'/f'full-reference-{row["angle"]:+03d}-neutral.png'
                    with Image.open(p) as im:pic=im.convert('RGB')
                image.paste(pic,(col*512,a*542+30));draw.text((col*512+8,a*542+6),f'{row["angle"]:+} deg / {kind}',font=font,fill='white')
        draw.text((8,1630),'Blue supported exterior / orange probable rim / green probable internal surface (wall vs flap unresolved) / purple uncertain',font=small,fill='white')
        image.save(out/'validated-source-role-inspection.png')
        draw_image=Image.new('RGB',(1024,600),'#24272c');draw=ImageDraw.Draw(draw_image)
        draw_image.paste(picture,(0,30));rgb=np.full((128,128,3),20,np.uint8)
        rgb[pix_wound&local]=(30,185,115);rgb[pix_wound&far]=(235,150,30);rgb[pix_wound&~valid]=(220,45,90)
        draw_image.paste(Image.fromarray(rgb).resize((512,512),Image.Resampling.NEAREST),(512,30));draw.text((8,5),'Visible wound constraint / near surface, farther torso and no-hit evidence',font=font,fill='white')
        draw.text((8,552),'Green near source / orange farther torso / red no full-source hit. This does not invent or close a cavity.',font=small,fill='white')
        draw_image.save(out/'validated-wound-support.png')
        inherited.save(out/'manifest.json',dict(classification=records['classification'],proposed_reconstruction_authorized=False,
            images={p.name:inherited.digest(p) for p in out.glob('*.png')},
            face_labels_sha256=inherited.digest(out/'face-role-labels.npy')))
        print(json.dumps(records),flush=True)
    finally:v._mmap.close();f._mmap.close()


def compose():
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    inputs=inherited.read(BASE/'inputs.json');analysis=inherited.read(BASE/'analysis.json');features=inherited.read(BASE/'source-sufficiency.json')
    render=inherited.read(BASE/'renders/render-metadata.json');need(render['all_source_vertex_and_corner_payloads_exact'] and
        not render['geometry_repaired_or_modified'] and render['device']=='CPU','Unchanged full-source render required')
    need(len(render['frames'])==6,'Three matched neutral/color pairs required')
    for row in render['frames']:need(inherited.digest(BASE/'renders'/row['name'])==row['sha256'],'New render changed')
    for row in inputs['views']:need(inherited.digest(EVIDENCE/'renders'/row['name'])==row['sha256'],'Original crop render changed')
    out=BASE/'inspection-sheets';out.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    image=Image.new('RGB',(1536,1655),'#24272c');draw=ImageDraw.Draw(image)
    for j,row in enumerate(inputs['views']):
        for i,kind in enumerate(('saved-crop','neutral','roles')):
            path=EVIDENCE/'renders'/row['name'] if kind=='saved-crop' else BASE/'renders'/f'full-reference-{row["angle"]:+03d}-{kind}.png'
            with Image.open(path) as view:image.paste(view.convert('RGB'),(i*512,j*542+30))
            draw.text((i*512+8,j*542+6),f'{kind.upper()} / {row["angle"]:+} deg',fill='white',font=font)
    draw.text((10,1630),'Blue exterior candidate / orange probable rim / green probable cavity / purple uncertain / gray outside audit',fill='white',font=small)
    image.save(out/'matched-source-roles.png')
    layers=np.load(BASE/'layer-face-ids.npy',allow_pickle=False);depth=np.load(BASE/'layer-depths.npy',allow_pickle=False)
    labels=np.load(BASE/'face-role-labels.npy',allow_pickle=False);origins=np.load(BASE/'ray-origins.npy',allow_pickle=False);directions=np.load(BASE/'ray-directions.npy',allow_pickle=False)
    palette=np.array([[180,72,190],[45,140,235],[250,145,20],[25,190,110],[85,85,85]],np.uint8)
    image=Image.new('RGB',(1536,1655),'#24272c');draw=ImageDraw.Draw(image)
    for j,row in enumerate(inputs['views']):
        counts=(layers[j]>=0).sum(axis=2);color=np.zeros((128,128,3),np.uint8)
        for count,rgb in [(0,(10,10,10)),(1,(40,110,230)),(2,(30,190,110)),(3,(235,185,35)),(4,(220,80,160))]:
            color[counts==count]=rgb
        color[counts>4]=(200,75,220)
        first=layers[j,:,:,0];valid=first>=0;role_color=np.full_like(color,10);role_color[valid]=palette[labels[first[valid]]]
        yy=origins[j,:,:,1]+directions[j,1]*depth[j,:,:,0]
        depth_color=np.full_like(color,10);cut=inputs['wound_depth_witness']['deep_y_threshold'];hi=inputs['region']['hi'][1]
        depth_color[valid&(yy<=cut)]=(45,140,235);depth_color[valid&(yy>cut)&(yy<=hi)]=(25,190,110);depth_color[valid&(yy>hi)]=(230,145,30)
        for i,(name,picture) in enumerate([('Layer count: 1 blue / 2 green / 3 yellow / 4+ magenta',color),('First-hit depth: shallow blue / near-deep green / far orange',depth_color),('Candidate role of first full-source hit',role_color)]):
            image.paste(Image.fromarray(picture).resize((512,512),Image.Resampling.NEAREST),(i*512,j*542+30))
            draw.text((i*512+5,j*542+6),f'{row["angle"]:+} {name}',fill='white',font=small)
    draw.text((10,1630),'Ray events are not automatically physical sheets; numerical ties/thin layers remain limited by recorded epsilon.',fill='white',font=small)
    image.save(out/'layer-and-depth-inspection.png')
    deep_pixels=np.load(BASE/'deep-pixels.npy',allow_pickle=False);old_face=np.load(EVIDENCE/'renders/chest-wound-+00-reference-face-ids.npy',allow_pickle=False)
    roi=(origins[1,:,:,0]>=.03)&(origins[1,:,:,0]<=.10)&(origins[1,:,:,2]>=.15)&(origins[1,:,:,2]<=.23)
    void=roi&(old_face<0);valid=layers[1,:,:,0]>=0;yy=origins[1,:,:,1]+directions[1,1]*depth[1,:,:,0]
    colors=np.full((128,128,3),25,np.uint8);colors[void&valid]=(240,150,30);colors[void&~valid]=(230,50,90)
    deep_ids=np.load(BASE/'deep-source-face-ids.npy',allow_pickle=False)
    for (j,i),face in zip(deep_pixels,deep_ids):colors[j,i]=(25,190,110) if labels[face]==3 else (180,72,190)
    image=Image.new('RGB',(1024,600),'#24272c');draw=ImageDraw.Draw(image)
    with Image.open(BASE/'renders/full-reference-+00-neutral.png') as view:image.paste(view.convert('RGB'),(0,30))
    image.paste(Image.fromarray(colors).resize((512,512),Image.Resampling.NEAREST),(512,30))
    draw.text((10,5),'Unchanged full source / saved deep and void witnesses',fill='white',font=font)
    draw.text((10,550),'Green probable cavity / purple uncertain deep / orange farther body behind crop void / red no full-source hit',fill='white',font=small)
    image.save(out/'deep-and-opening-witnesses.png')
    if features.get('transition_witness'):
        v=np.load(OLD/'variant/vertices.npy',mmap_mode='r',allow_pickle=False);f=np.load(OLD/'variant/faces.npy',mmap_mode='r',allow_pickle=False)
        try:
            path=features['transition_witness'];cent=v[f[path['source_face_ids']]].mean(axis=1);bounds=np.array(features['proposed_sector']['bounds'])
            corners=np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
            image=Image.new('RGB',(1536,590),'#24272c');draw=ImageDraw.Draw(image)
            for a,row in enumerate(inputs['views']):
                with Image.open(BASE/'renders'/f'full-reference-{row["angle"]:+03d}-neutral.png') as view:picture=view.convert('RGB')
                mark=ImageDraw.Draw(picture);forward=directions[a];right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right);up=np.cross(right,forward)
                def project(points):
                    d=points-np.array(row['target']);return np.stack((256+(d@right)*512/row['ortho_scale'],256-(d@up)*512/row['ortho_scale']),axis=1)
                xy=project(cent);box=project(corners);mark.line([tuple(p) for p in xy],fill='#ffff33',width=2)
                mark.rectangle((float(box[:,0].min()),float(box[:,1].min()),float(box[:,0].max()),float(box[:,1].max())),outline='#ffbb44',width=2)
                for point,text in [(xy[0],'skin'),(xy[-1],'deep wall')]:mark.ellipse((point[0]-4,point[1]-4,point[0]+4,point[1]+4),fill='#ffff33');mark.text(tuple(point),text,fill='#ffff33',font=small)
                image.paste(picture,(a*512,30));draw.text((a*512+6,5),f'{row["angle"]:+} deg: source path / proposed sector',fill='white',font=font)
            draw.text((10,552),'Yellow = Xray source-adjacency path (not all segments visible); orange box = proposed future test domain only.',fill='white',font=small)
            image.save(out/'source-transition-and-proposed-sector.png')
        finally:v._mmap.close();f._mmap.close()
    sector=features.get('proposed_sector')
    inherited.save(BASE/'future-multisheet-proposal.json',dict(classification='PRELIMINARY_ONLY_REQUIRES_VISUAL_WOUND_VALIDATION',
        proposed_only=True,authorized=False,executed=False,baseline_PLY_sha256=BASELINE_SHA,sector=sector,
        source_transition=features.get('transition_witness'),
        operation_authorized=False,reason='A depth-proxy adjacency path must be rejected if visual anatomy does not support its skin/rim/cavity interpretation. Run semantic validation; no reconstruction proposal follows from this preliminary record.',
        no_full_wound_or_character_authorization=True,no_upstream_generation_authorization=True))
    inherited.save(BASE/'inspection-verification.json',dict(classification='PASS_READ_ONLY_COLOR_AND_LAYER_INSPECTION',source_payloads_exact=True,
        six_new_frame_hashes_verified=True,three_prior_crop_frame_hashes_verified=True,no_geometry_exported_or_reconstructed=True,
        uncertain_regions_not_forced=True,sheets={p.name:inherited.digest(p) for p in out.glob('*.png')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('authenticate','rays','analyse','features','render','compose','landmarks','semantic'));parser.add_argument('--worker',choices=('authenticate','analyse','features','compose','landmarks','semantic'));args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.stage:
        module=owner(args.stage)
        if args.stage in ('rays','render'):os.environ['MODLY_ZOMBIE_WOUND_SOURCE_MODE']=args.stage;raise SystemExit(module.supervise('render'))
        raise SystemExit(module.supervise(args.stage))
    else:print('DRY_RUN: explicit read-only source-audit stage required')
