"""Read-only saved-mesh topology diagnosis; no extraction or repair entrypoint."""
import argparse
import ast
import csv
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT.parent/'artifacts/amd-zombie-topology-diagnosis-20261025-v1'
ORIGINAL=ROOT.parent/'artifacts/amd-zombie-cpu-mesh-20261025-v1'
PLY=ORIGINAL/'mesh/mesh.ply'
PLY_SHA='5301f7dc3e22abf6bfd7c1253a4d6253935ad3054658e2eddbb6ac024e6fc12c'
sys.path.insert(0,str(Path(__file__).parent))
import amd_zombie_cpu_mesh as inherited


def incidence(np,faces):
    directed=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    edges,inverse,counts=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
    winding=np.bincount(inverse,weights=np.where(directed[:,0]<directed[:,1],1,-1),minlength=len(edges)).astype(np.int16)
    return edges,counts,inverse,winding


def recover_quads(np,faces):
    a,b=faces.reshape(-1,2,3).transpose(1,0,2)
    one=(b[:,0]==a[:,0])&(b[:,1]==a[:,2]);two=(b[:,0]==a[:,2])&(b[:,1]==a[:,1])
    if not np.all(one^two):raise ValueError('Original paired split pattern differs')
    return np.column_stack((a[:,0],a[:,1],np.where(one,a[:,2],b[:,2]),np.where(one,b[:,2],a[:,2])))


def quad_incidence(np,quads):
    pairs=np.concatenate([quads[:,[i,(i+1)%4]] for i in range(4)])
    return np.unique(np.sort(pairs,axis=1),axis=0,return_counts=True)


def source_quads(np,coords,flags):
    # Independent dictionary lookup, not the converter's sorted-key algorithm.
    offsets=(((0,0,0),(0,0,1),(0,1,1),(0,1,0)),
             ((0,0,0),(1,0,0),(1,0,1),(0,0,1)),
             ((0,0,0),(0,1,0),(1,1,0),(1,0,0)))
    rows=coords.tolist();lookup={tuple(c):i for i,c in enumerate(rows)};quads=[];missing=0
    for anchor,axis in np.argwhere(flags).tolist():
        base=rows[anchor];corners=[tuple(base[j]+offset[j] for j in range(3)) for offset in offsets[axis]]
        if all(c in lookup for c in corners):quads.append([lookup[c] for c in corners])
        else:missing+=1
    return np.asarray(quads,np.int32).reshape(-1,4),missing


def attribute_source_quads():
    import numpy as np
    import amd_generation_mesh as mesh
    contract=mesh._contract_api().load(inherited.SOURCE.parent/'run-contract.json')
    if contract.sha256!=inherited.CONTRACT_SHA:raise ValueError('Original run contract differs')
    arrays,children=mesh.read_neural(inherited.SOURCE,inherited.SOURCE_SHA,contract=contract)
    f=np.load(ORIGINAL/'mesh/faces.npy',mmap_mode='r',allow_pickle=False)
    try:
        quads,missing=source_quads(np,arrays['coordinates'][:,1:],arrays['features'][:,3:6]>0)
        if not np.array_equal(quads,recover_quads(np,f)):raise ValueError('Direct learned quads differ from saved triangle pairs')
        edges,counts=quad_incidence(np,quads)
        with np.load(BASE/'topology.npz',allow_pickle=False) as data:
            if not np.array_equal(edges[counts==1],data['boundary_edges']) or not np.array_equal(edges[counts>2],data['nonmanifold_edges']):
                raise ValueError('Direct learned quad defects differ from PLY measurement')
        installed=Path('D:/AI/Trellis2AMD/venv/Lib/site-packages/o_voxel/convert/flexible_dual_grid.py')
        installed_sha=inherited.digest(installed)
        if installed_sha!='2538315098531d5306e2cb5bf84942335e114c23fd1192dbd1920800c1c9b7a2':raise ValueError('Qualified installed converter source changed')
        inherited.save(BASE/'source-quad-attribution.json',dict(classification='PASS_DIRECT_LEARNED_QUAD_ATTRIBUTION',
            neural_source_sha256=inherited.SOURCE_SHA,PLY_sha256=PLY_SHA,valid_source_quads=len(quads),missing_corner_flags=missing,
            all_direct_quad_rows_and_order_match_original_mesh=True,
            direct_quad_boundary_edges=int((counts==1).sum()),direct_quad_nonmanifold_edges=int((counts>2).sum()),
            direct_quad_boundary_and_nonmanifold_ID_sets_match_PLY=True,
            installed_source_read_only_sha256=installed_sha,installed_module_imported=False,
            CPU_converter_sha256=inherited.digest(ROOT/'scripts/compatibility/amd_flexible_dual_grid_cpu.py'),
            mesh_reextracted_rebuilt_or_changed=False))
    finally:
        f._mmap.close()
        for a in arrays.values():a._mmap.close()


def component_labels(np,faces,n):
    parents=list(range(n));sizes=[1]*n
    def find(x):
        while parents[x]!=x:parents[x]=parents[parents[x]];x=parents[x]
        return x
    for block in range(0,len(faces),32768):
        for a,b,c in faces[block:block+32768].tolist():
            for x,y in ((a,b),(a,c)):
                x,y=find(x),find(y)
                if x!=y:
                    if sizes[x]<sizes[y]:x,y=y,x
                    parents[y]=x;sizes[x]+=sizes[y]
    face_roots=np.fromiter((find(int(x)) for x in faces[:,0]),dtype=np.int32,count=len(faces))
    keys,counts=np.unique(face_roots,return_counts=True)
    rank=sorted(zip(keys.tolist(),counts.tolist()),key=lambda row:(-row[1],row[0]))
    lookup={key:i for i,(key,count) in enumerate(rank)}
    fl=np.fromiter((lookup[int(x)] for x in face_roots),dtype=np.int16,count=len(faces))
    vl=np.full(n,-1,np.int16);used=np.unique(faces)
    vl[used]=np.fromiter((lookup[find(int(x))] for x in used),dtype=np.int16,count=len(used))
    return vl,fl


def region_labels(np,points):
    # Explicit geometric viewing bins only; these are NOT semantic segmentation.
    x=np.abs(points[:,0]);z=points[:,2]
    return np.select((z>=.43,z>=.30,x>=.40,x>=.145,z>=.16,z>=-.035,z>=-.19,z>=-.45),
        ('crown-scalp','head-neck','hands','arms','chest-shoulders','abdomen','shorts-hips','legs'),default='feet')


def edge_groups(np,edges,vertices):
    parents={int(v):int(v) for v in np.unique(edges)}
    def find(x):
        while parents[x]!=x:parents[x]=parents[parents[x]];x=parents[x]
        return x
    for a,b in edges.tolist():
        a,b=find(a),find(b)
        if a!=b:parents[b]=a
    roots=np.fromiter((find(int(x)) for x in edges[:,0]),dtype=np.int32,count=len(edges))
    keys,counts=np.unique(roots,return_counts=True);order=np.argsort(-counts,kind='stable')
    labels=np.empty(len(edges),np.int32);records=[]
    degree=np.bincount(edges.ravel(),minlength=len(vertices))
    for group,index in enumerate(order):
        selected=np.flatnonzero(roots==keys[index]);labels[selected]=group
        ids=np.unique(edges[selected]);points=vertices[ids];delta=vertices[edges[selected,1]]-vertices[edges[selected,0]]
        records.append(dict(id=group,edges=len(selected),vertices=len(ids),bounds=[points.min(axis=0).tolist(),points.max(axis=0).tolist()],
            centroid=points.mean(axis=0).tolist(),length=float(np.linalg.norm(delta,axis=1).sum()),
            closed_degree_two_loop=bool(np.all(degree[ids]==2)),maximum_vertex_degree=int(degree[ids].max())))
    return labels,records


def measurements(np,vertices,edges):
    midpoint=vertices[edges].mean(axis=1);length=np.linalg.norm(vertices[edges[:,1]]-vertices[edges[:,0]],axis=1)
    bins=region_labels(np,midpoint)
    rows={str(name):dict(edges=int((bins==name).sum()),length=float(length[bins==name].sum())) for name in np.unique(bins)}
    return midpoint,length,rows


def analyze():
    import numpy as np
    import amd_generation_mesh as mesh
    began=time.monotonic()
    contract,authentication=inherited.authenticate()
    metadata=inherited.read(ORIGINAL/'mesh/mesh.json')
    if inherited.digest(inherited.plain(PLY))!=PLY_SHA:raise ValueError('Original PLY differs')
    for name,row in metadata['artifacts'].items():
        path=inherited.plain(ORIGINAL/'mesh'/name)
        if path.stat().st_size!=row['bytes'] or inherited.digest(path)!=row['sha256']:raise ValueError('Original mesh arrays differ')
    v,f=mesh.converter.read_ply(PLY)
    saved_v=np.load(ORIGINAL/'mesh/vertices.npy',mmap_mode='r',allow_pickle=False)
    saved_f=np.load(ORIGINAL/'mesh/faces.npy',mmap_mode='r',allow_pickle=False)
    if not np.array_equal(v,saved_v) or not np.array_equal(f,saved_f):raise ValueError('PLY/arrays differ')
    arrays,children=mesh.read_neural(inherited.SOURCE,inherited.SOURCE_SHA,contract=contract)
    try:
        edges,counts,inverse,winding=incidence(np,f)
        boundary=edges[counts==1];excess=edges[counts>2]
        if len(boundary)!=19512 or len(excess)!=31449:raise ValueError('Independent edge counts disagree with retained analysis')
        quads=recover_quads(np,f);qe,qc=quad_incidence(np,quads)
        quad_boundary=qe[qc==1];quad_excess=qe[qc>2]
        source_coords=arrays['coordinates'][:,1:]
        grid_delta=np.abs(source_coords[edges[:,0]]-source_coords[edges[:,1]])
        diagonal=(grid_delta!=0).sum(axis=1)==2
        diagonal_ok=bool(np.all(counts[diagonal]==2))
        same_boundary=np.array_equal(quad_boundary,boundary);same_excess=np.array_equal(quad_excess,excess)
        if not same_boundary or not same_excess or not diagonal_ok:raise ValueError('Quad/triangle attribution differs; stop for review')
        vl,fl=component_labels(np,f,len(v));fragments=[];main=v[vl==0]
        for component in np.unique(fl):
            ids=np.flatnonzero(vl==component);points=v[ids];gap=None
            if component:
                minimum=float('inf')
                for start in range(0,len(points),8):
                    distance=((points[start:start+8,None,:]-main[None,:,:])**2).sum(axis=2)
                    minimum=min(minimum,float(distance.min()))
                gap=float(np.sqrt(minimum))
            fragments.append(dict(id=int(component),triangles=int((fl==component).sum()),vertices=len(ids),
                bounds=[points.min(axis=0).tolist(),points.max(axis=0).tolist()],centroid=points.mean(axis=0).tolist(),
                minimum_vertex_to_main_vertex_gap=gap,gap_is_not_surface_distance=True,
                boundary_edges=int((vl[boundary[:,0]]==component).sum()),nonmanifold_edges=int((vl[excess[:,0]]==component).sum())))
        bm,bl,br=measurements(np,v,boundary);nm,nl,nr=measurements(np,v,excess)
        bg,bgr=edge_groups(np,boundary,v);ng,ngr=edge_groups(np,excess,v)
        regions={}
        for name,low,high in (
            ('front-chest-wound-candidate',[.015,-.12,.13],[.12,-.015,.25]),
            ('front-abdominal-wound-candidate',[-.09,-.12,-.02],[.05,-.015,.14]),
            ('clothing-hem-region',[-.15,-.12,-.21],[.15,.12,-.10]),
            ('crown-projection-region',[-.10,-.12,.46],[.10,.12,.51])):
            regions[name]=dict(bounds=[low,high],boundary_edges=int(np.all((bm>=low)&(bm<=high),axis=1).sum()),
                nonmanifold_edges=int(np.all((nm>=low)&(nm<=high),axis=1).sum()),intent='UNCERTAIN; geometric candidate ROI, not a semantic label')
        with (BASE/'defect-edges.csv').open('x',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(['type','vertex_a','vertex_b','incident_faces','component','network_group','x','y','z','length'])
            for kind,es,ms,ls,groups,cs in [('boundary',boundary,bm,bl,bg,counts[counts==1]),('nonmanifold',excess,nm,nl,ng,counts[counts>2])]:
                for edge,point,length,group,count in zip(es,ms,ls,groups,cs):writer.writerow([kind,int(edge[0]),int(edge[1]),int(count),int(vl[edge[0]]),int(group),*map(float,point),float(length)])
        with (BASE/'topology.npz').open('xb') as stream:
            np.savez(stream,boundary_edges=boundary.astype(np.int32),nonmanifold_edges=excess.astype(np.int32),
                boundary_group=bg,nonmanifold_group=ng,vertex_component=vl,face_component=fl,
                unused_vertex_ids=np.flatnonzero(vl<0).astype(np.int32),nonmanifold_face_ids=np.unique(np.tile(np.arange(len(f)),3)[np.isin(inverse,np.flatnonzero(counts>2))]).astype(np.int32))
        result=dict(classification='PASS_READONLY_TOPOLOGY_DIAGNOSIS',PLY_sha256=PLY_SHA,neural_authentication=authentication,
            vertices=len(v),triangles=len(f),unique_edges=len(edges),boundary_edges=len(boundary),nonmanifold_edges=len(excess),
            incidence_histogram={str(int(n)):int((counts==n).sum()) for n in np.unique(counts)},
            two_face_edges_with_same_direction=int(((counts==2)&(np.abs(winding)==2)).sum()),
            boundary_regions=br,nonmanifold_regions=nr,candidate_ROIs=regions,
            region_definitions='Geometric bins: crown z>=.43; head z>=.30; hands abs(x)>=.40; arms abs(x)>=.145; chest z>=.16; abdomen z>=-.035; shorts z>=-.19; legs z>=-.45; feet otherwise; first match wins.',
            fragments=fragments,unused_vertices=int((vl<0).sum()),
            boundary_networks=len(bgr),boundary_degree_two_loops=sum(r['closed_degree_two_loop'] for r in bgr),
            nonmanifold_networks=len(ngr),largest_boundary_networks=bgr[:20],largest_nonmanifold_networks=ngr[:20],
            converter_attribution=dict(quad_boundary_matches_all_triangle_boundaries=same_boundary,
                quad_excess_incidence_matches_all_triangle_nonmanifold_edges=same_excess,
                internal_diagonals=len(edges[diagonal]),all_internal_diagonals_incidence_two=diagonal_ok,
                quad_incidence_histogram={str(int(n)):int((qc==n).sum()) for n in np.unique(qc)},
                converter_changed=False,mesh_rebuilt_or_repaired=False),
            elapsed_seconds=time.monotonic()-began)
        inherited.save(BASE/'topology-analysis.json',result)
        inherited.save(BASE/'edge-networks.json',dict(boundary=bgr,nonmanifold=ngr))
        print(json.dumps({k:result[k] for k in ('classification','boundary_edges','nonmanifold_edges','two_face_edges_with_same_direction','boundary_networks','nonmanifold_networks','elapsed_seconds')}),flush=True)
    finally:
        for a in (v,f,saved_v,saved_f,*arrays.values()):
            mapping=getattr(a,'_mmap',None)
            if mapping is not None and not mapping.closed:mapping.close()


def owner(stage):
    # Private copy of the accepted owner. Only its render entry filename changes;
    # all resource limits, GO/EOF and failure-path cleanup stay byte-for-byte.
    module=inherited.load('_zombie_topology_owned_cpu',Path(inherited.__file__))
    source=inspect.getsource(module.supervise)
    tree=ast.parse(source);changed=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='amd_zombie_mesh_render.py':node.value='amd_zombie_topology_render.py';changed.append(node)
    if len(changed)!=1:raise ValueError('Exact preserved owner render route required')
    module.__file__=__file__;module.BASE=BASE
    exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),module.__dict__)
    if stage in ('compare','overlay','details'):
        module.BASE=BASE/('render-'+stage);module.BASE.mkdir(exist_ok=False)
    return module


def compose():
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    output=BASE/'inspection-sheets';output.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    records={}
    for mode in ('compare','overlay','details'):
        meta=inherited.read(BASE/mode/'render-metadata.json')
        if meta['classification']!='PASS_UNCHANGED_GEOMETRY_CPU_DIAGNOSTICS' or not meta['original_vertex_and_face_payloads_exact']:
            raise ValueError('Complete unchanged geometry render receipt required')
        for row in meta['views']:
            if inherited.digest(BASE/mode/row['name'])!=row['sha256']:raise ValueError('Rendered view changed')
        records[mode]=meta
    compared=records['compare']['views']
    for i in range(0,len(compared),2):
        a,b=compared[i:i+2]
        if (a['camera_location'],a['target'],a['ortho_scale'])!=(b['camera_location'],b['target'],b['ortho_scale']):raise ValueError('Flat/smooth camera mismatch')
    def sheet(name,items,columns=4,size=384,subtitle=''):
        rows=(len(items)+columns-1)//columns
        image=Image.new('RGB',(columns*size,rows*(size+30)+50),'#25282e');draw=ImageDraw.Draw(image)
        draw.text((10,12),subtitle,fill='white',font=font)
        for index,(path,label) in enumerate(items):
            with Image.open(path) as view:thumb=view.convert('RGB').resize((size,size),Image.Resampling.LANCZOS)
            x=(index%columns)*size;y=(index//columns)*(size+30)+50;image.paste(thumb,(x,y));draw.text((x+8,y+size+5),label,fill='white',font=small)
        image.save(output/name)
    sheet('flat-smooth-eight-angles.png',[(BASE/'compare'/f'{style}-{angle:03d}.png',f'{style.upper()} {angle} deg') for angle in range(0,360,45) for style in ('flat','smooth')],
        subtitle='Identical vertex/face payloads, cameras and neutral lights; normals only differ')
    sheet('topology-eight-angles.png',[(BASE/'overlay'/f'topology-{angle:03d}.png',f'{angle} deg') for angle in range(0,360,45)],
        subtitle='Cyan: boundary edges (1 face). Magenta: nonmanifold edges (>2 faces).')
    sheet('front-flat-smooth-topology.png',[(BASE/'compare/flat-000.png','NEUTRAL FLAT'),(BASE/'compare/smooth-000.png','NORMALS ONLY: SMOOTH'),(BASE/'overlay/topology-000.png','TOPOLOGY OVERLAY')],3,512,
        'Unchanged original geometry; smooth shading does not repair topology')
    sheet('old-neutral-lighting-comparison.png',[(ORIGINAL/'previews/view-000.png','PREVIOUS LIGHTING'),(BASE/'compare/flat-000.png','NEUTRAL FLAT: SAME CAMERA')],2,512,
        'Presentation-only lighting comparison; no vertex or face changes')
    for page,features in [('crown-hair-details',['crown-front','crown-back','hair-back','head-front']),
                          ('wound-hand-details',['chest-wound','abdomen','hand-xplus','hand-xminus']),
                          ('clothing-details',['clothing-hem'])]:
        sheet(page+'.png',[(BASE/'details'/f'{feature}-{style}.png',feature+' '+style) for feature in features for style in ('smooth','topology')],2,512,
            'Cyan boundary / magenta >2 faces. Intent remains uncertain near wounds and clothing.')
    analysis=inherited.read(BASE/'topology-analysis.json')
    view=compared[0]
    forward=np.array(view['target'])-np.array(view['camera_location']);forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right);up=np.cross(right,forward)
    center=np.array(view['target']);scale=view['ortho_scale']
    def project(p):
        d=np.asarray(p)-center
        return (384+float(d@right)*768/scale,384-float(d@up)*768/scale)
    v=np.load(ORIGINAL/'mesh/vertices.npy',mmap_mode='r',allow_pickle=False)
    with Image.open(BASE/'compare/flat-000.png') as source:
        dark=Image.new('RGB',source.size,'#333333');ghost=Image.blend(dark,source.convert('RGB'),.45)
    palette=['#ffba20','#20dd55','#ff5520','#af55ff','#3399ff','#ffdd20','#20ddcc','#ff5599','#aadd22']
    locations=Image.new('RGB',(1100,850),'#25282e');locations.paste(ghost,(160,60));draw=ImageDraw.Draw(locations)
    draw.text((10,10),'Detached fragments C1-C9: coloured markers are X-ray projections, not enlarged geometry',font=font,fill='white')
    last_left=last_right=70
    for row in sorted(analysis['fragments'][1:],key=lambda r:-r['centroid'][2]):
        x,y=project(row['centroid']);x+=160;y+=60;color=palette[row['id']-1]
        draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2)
        left=row['id']%2==0
        label_y=max(70+y*.80,last_left+28 if left else last_right+28)
        if left:last_left=label_y;label_x=10;line_x=150
        else:last_right=label_y;label_x=940;line_x=935
        draw.line((x,y,line_x,label_y+8),fill=color,width=1)
        draw.text((label_x,label_y),f'C{row["id"]}: {row["triangles"]} tris',font=small,fill=color)
    draw.text((10,820),'No detached crown component. Crown projection is part of C0, the main surface.',font=font,fill='white')
    locations.save(output/'fragment-locations-xray.png')
    with np.load(BASE/'topology.npz',allow_pickle=False) as data:unused=data['unused_vertex_ids']
    points=ghost.copy();draw=ImageDraw.Draw(points)
    for point in v[unused]:
        x,y=project(point);draw.ellipse((x-1.5,y-1.5,x+1.5,y+1.5),fill='#4488ff')
    draw.rectangle((0,0,768,52),fill='#25282e');draw.text((12,10),'425 unreferenced vertices: blue X-ray markers',font=font,fill='white')
    draw.text((12,32),'No surface faces use these IDs; they are not the visible crown protrusion.',font=small,fill='white')
    points.save(output/'unreferenced-vertex-locations.png');v._mmap.close()
    with Image.open(BASE/'overlay/topology-000.png') as source:regions=source.convert('RGB')
    draw=ImageDraw.Draw(regions)
    for number,(name,row) in enumerate(analysis['candidate_ROIs'].items(),1):
        import itertools
        corners=[project(p) for p in itertools.product(*zip(*row['bounds']))]
        x0=min(p[0] for p in corners);y0=min(p[1] for p in corners);x1=max(p[0] for p in corners);y1=max(p[1] for p in corners)
        draw.rectangle((x0,y0,x1,y1),outline='#ffcc66',width=2);draw.text((x1+2,y0),f'ROI {number} ?',fill='#ffcc66',font=small)
    regions.save(output/'uncertain-feature-regions-front.png')
    inherited.save(output/'sheets.json',dict(classification='PASS_AUTHENTIC_DIAGNOSTIC_SHEETS',
        source_PLY_sha256=PLY_SHA,matched_flat_smooth_cameras=True,
        fragment_and_orphan_markers_are_unoccluded_Xray_annotations=True,
        component_crown_distinction='Crown projection belongs to main C0; no detached component has z>=.43',
        assets={p.name:dict(bytes=p.stat().st_size,sha256=inherited.digest(p)) for p in output.glob('*.png')}))


def worker(stage):
    if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO plus EOF required')
    def audit(event,args):
        if event=='import' and str(args[0]).split('.')[0].lower() in {'torch','triton','trellis2','flex_gemm','o_voxel','cupy','cumesh','nvdiffrast'}:raise RuntimeError('CPU-only model/GPU import denial')
        if event in ('subprocess.Popen','os.system','socket.connect','socket.bind'):raise RuntimeError('CPU-only process/network denial')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and (any(c in str(args[1]) for c in 'wax+') or args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):
            inherited.plain(os.fsdecode(args[0])).relative_to(BASE)
        if event=='os.mkdir':inherited.plain(args[0]).relative_to(BASE)
        if event in ('os.remove','os.rename','os.rmdir','os.link','os.symlink','os.truncate'):raise RuntimeError('Destructive mutation denied')
    sys.addaudithook(audit)
    sys.path.append('D:/AI/Trellis2AMD/venv/Lib/site-packages')
    if stage=='analyze':analyze()
    elif stage=='source-quads':attribute_source_quads()
    elif stage=='compose':compose()
    else:raise ValueError('Unknown read-only stage')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('analyze','source-quads','compare','overlay','details','compose'))
    parser.add_argument('--worker',choices=('analyze','source-quads','compose'))
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.stage:
        module=owner(args.stage)
        if args.stage in ('compare','overlay','details'):
            os.environ['MODLY_ZOMBIE_TOPOLOGY_MODE']=args.stage
            raise SystemExit(module.supervise('render'))
        raise SystemExit(module.supervise(args.stage))
    else:print('DRY_RUN: read-only topology analysis and CPU presentation comparisons only')
