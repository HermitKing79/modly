"""Bounded two-crop retopology feasibility; immutable orphan reference only."""
import argparse,ast,hashlib,inspect,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT.parent/'artifacts/amd-zombie-retopology-feasibility-20261025-v1'
OLD=ROOT.parent/'artifacts/amd-zombie-orphan-cleanup-20261025-v1'
BASELINE_SHA='377a15dc4b5a27936f7e62b311dd03e8337685344b0f8204b8f831252176b1a5'
REGIONS={
 'posterior':dict(lo=[-.045,.02,.17],hi=[.045,.12,.23],outside_y=1),
 'chest-wound':dict(lo=[-.005,-.12,.125],hi=[.115,.055,.255],outside_y=-1)}
sys.path.insert(0,str(Path(__file__).parent))
import amd_zombie_winding as previous
prior=previous.prior;topology=previous.topology;inherited=previous.inherited
need=previous.need


def crop_boundary(np,whole,crop,n):
    edges,counts,_,_=topology.incidence(np,whole);ce,cc,_,_=topology.incidence(np,crop)
    keys=edges[:,0].astype(np.int64)*n+edges[:,1];query=ce[:,0].astype(np.int64)*n+ce[:,1]
    positions=np.searchsorted(keys,query);need(np.array_equal(keys[positions],query),'Crop source edge identity differs')
    global_counts=counts[positions];boundary=cc==1
    return dict(genuine_source_boundary_edges=int((boundary&(global_counts==1)).sum()),
        artificial_crop_boundary_edges=int((boundary&(global_counts>1)).sum()),
        artificial_cuts_on_source_nonmanifold_edges=int((boundary&(global_counts>2)).sum()),
        included_global_nonmanifold_edge_IDs=positions[global_counts>2].tolist(),
        global_boundary_edge_IDs=positions[global_counts==1].tolist(),
        artificial_boundary_original_vertex_pairs=ce[boundary&(global_counts>1)].tolist())


def split_fans(np,v,f):
    """Separate disconnected vertex links on a new chart, without moving triangles."""
    if not len(f):return v.copy(),f.copy(),0
    edges,counts,inverse,_=topology.incidence(np,f);order=np.argsort(inverse,kind='stable');starts=np.r_[0,np.cumsum(counts)]
    parent=np.arange(3*len(f))
    def root(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=int(parent[a])
        return a
    def union(a,b):
        a=root(a);b=root(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for e in np.flatnonzero(counts==2):
        a,b=(order[starts[e]:starts[e]+2]%len(f)).tolist()
        for vertex in edges[e]:
            ca=3*a+int(np.flatnonzero(f[a]==vertex)[0]);cb=3*b+int(np.flatnonzero(f[b]==vertex)[0]);union(ca,cb)
    roots=np.array([root(i) for i in range(len(parent))]);unique,reverse=np.unique(roots,return_inverse=True)
    vv=np.ascontiguousarray(v[f.ravel()[unique]],dtype=np.float32);ff=reverse.reshape(-1,3).astype(np.int32)
    need(v[f].tobytes()==vv[ff].tobytes(),'Fan separation changed chart triangles')
    return vv,ff,len(vv)-len(np.unique(f))


def chart(np,points,valid,outside_y,limit):
    need(points.ndim==3 and points.shape[2]==3 and points.shape[:2]==valid.shape and
        points.shape[0]%2==1 and points.shape[1]%2==1 and outside_y in (-1,1) and limit>0,'Odd dense chart grid required')
    rows=(len(points)-1)//2;cols=(points.shape[1]-1)//2;faces=[];missing=0;jumps=0
    nodes=points[::2,::2].copy().reshape(-1,3)
    for j in range(rows):
        for i in range(cols):
            p=points[2*j:2*j+3,2*i:2*i+3];good=valid[2*j:2*j+3,2*i:2*i+3]
            if not good.all():missing+=1;continue
            if max(float(np.linalg.norm(np.diff(p,axis=0),axis=2).max()),float(np.linalg.norm(np.diff(p,axis=1),axis=2).max()))>limit:
                jumps+=1;continue
            a=j*(cols+1)+i;b=a+1;c=a+cols+2;d=a+cols+1
            pair=[[a,b,c],[a,c,d]]
            if outside_y==1:pair=[[x,z,y] for x,y,z in pair]
            faces.extend(pair)
    ff=np.array(faces,np.int32).reshape(-1,3)
    used=np.unique(ff);mapping=np.full(len(nodes),-1,np.int32);mapping[used]=np.arange(len(used),dtype=np.int32)
    v=np.ascontiguousarray(nodes[used],dtype=np.float32);f=np.ascontiguousarray(mapping[ff])
    vv,ff,duplicates=split_fans(np,v,f)
    return vv,ff,dict(grid_cells=rows*cols,accepted_cells=len(faces)//2,missing_witness_cells=missing,
        depth_discontinuity_cells=jumps,duplicated_chart_vertices_for_disconnected_fans=duplicates,
        all_nine_dense_samples_required=True,neighbor_3D_distance_limit=limit,
        unsupported_cells_omitted_not_filled=True,hidden_or_overlapping_sheets_not_reconstructed=True)


def prepare():
    import numpy as np
    import amd_flexible_dual_grid_cpu as fdg
    previous.baseline();contract,auth=inherited.authenticate()
    need(inherited.digest(OLD/'variant/mesh.ply')==BASELINE_SHA,'Wrong reference')
    v,f=fdg.read_ply(OLD/'variant/mesh.ply')
    try:
        output=BASE/'reference';output.mkdir(exist_ok=False);settings={}
        for name,region in REGIONS.items():
            inside=np.all((v>=region['lo'])&(v<=region['hi']),axis=1);ids=np.flatnonzero(np.all(inside[f],axis=1))
            need(len(ids)>2,'Nonempty multi-face crop required');vertices=np.unique(f[ids]);back=np.full(len(v),-1,np.int32);back[vertices]=np.arange(len(vertices),dtype=np.int32)
            vv=np.ascontiguousarray(v[vertices]);ff=np.ascontiguousarray(back[f[ids]])
            need(v[f[ids]].tobytes()==vv[ff].tobytes(),'Reference crop triangle payload changed')
            folder=output/name;folder.mkdir(exist_ok=False)
            fdg.write_ply(folder/'mesh.ply',vv,ff)
            for filename,a in [('vertices.npy',vv),('faces.npy',ff),('original-vertex-ids.npy',vertices.astype(np.int32)),('original-face-ids.npy',ids.astype(np.int32))]:prior.save_npy(folder/filename,a)
            edges,counts,inv,w=topology.incidence(np,ff);lengths=np.linalg.norm(vv[edges[:,0]].astype(np.float64)-vv[edges[:,1]],axis=1)
            median=float(np.median(lengths));boundary=crop_boundary(np,f,f[ids],len(v))
            settings[name]=dict(**region,source_face_count=len(ff),source_vertices=len(vv),median_source_edge_length=median,
                voxel_size=2*median,chart_grid_spacing_requested=2*median,
                boundary_attribution=boundary,reference_PLY_sha256=inherited.digest(folder/'mesh.ply'),
                reference_arrays={p.name:inherited.digest(p) for p in folder.glob('*.npy')})
        inherited.save(BASE/'experiment-settings.json',dict(baseline_PLY_sha256=BASELINE_SHA,neural_sha256=inherited.SOURCE_SHA,
            run_contract_sha256=contract.sha256,regions=settings,
            method_settings='One fixed voxel size and first-hit chart spacing per crop, twice measured median reference edge; no iterative parameter search',
            full_character_rebuilt=False,source_modified=False))
        inherited.save(BASE/'source-authentication.json',auth)
    finally:v._mmap.close();f._mmap.close()


def metrics(np,v,f):
    if not len(f):return dict(vertices=len(v),triangles=0,classification='EMPTY')
    edges,counts,inverse,sums=topology.incidence(np,f);vl,fl=topology.component_labels(np,f,len(v))
    p=v[f].astype(np.float64);cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);areas=np.linalg.norm(cross,axis=1)*.5
    flat=f.ravel();order=np.argsort(flat,kind='stable');num=np.bincount(flat,minlength=len(v));starts=np.r_[0,np.cumsum(num)]
    bad_links=0
    for vertex in np.flatnonzero(num):
        incident=np.unique(order[starts[vertex]:starts[vertex+1]]//3);row=prior.vertex_fan(np,f,int(vertex),incident)
        # A manifold vertex link is a single cycle or a single open path.
        hist={int(k):val for k,val in row['link_degree_histogram'].items()}
        path=row['link_components']==1 and hist.get(1,0)==2 and set(hist)<={1,2}
        if not (row['closed_single_fan'] or path):bad_links+=1
    return dict(classification='MEASURED',vertices=len(v),triangles=len(f),components=len(np.unique(fl)),
        component_triangle_counts=sorted(np.bincount(fl).tolist(),reverse=True),boundary_edges=int((counts==1).sum()),
        nonmanifold_edges=int((counts>2).sum()),winding_conflicts=int(((counts==2)&(abs(sums)==2)).sum()),
        nonmanifold_vertex_links=bad_links,zero_area_triangles=int((areas==0).sum()),
        surface_area=float(areas.sum()),bounds=[v.min(axis=0).tolist(),v.max(axis=0).tolist()])


def depth_p95(np,a,b,common):
    # Match the recorder's F32 difference followed by explicit F64 interpolation.
    return float(np.quantile(abs(a[common]-b[common]).astype(np.float64),.95))


def verify():
    import numpy as np
    import amd_flexible_dual_grid_cpu as fdg
    previous.baseline();settings=inherited.read(BASE/'experiment-settings.json');build=inherited.read(BASE/'processing.json');result={}
    baseline_v=np.load(OLD/'variant/vertices.npy',mmap_mode='r',allow_pickle=False);baseline_f=np.load(OLD/'variant/faces.npy',mmap_mode='r',allow_pickle=False)
    try:
        for region in REGIONS:
            result[region]={}
            for kind in ('reference','voxel','chart'):
                folder=BASE/('reference' if kind=='reference' else 'candidates')/region
                if kind!='reference':folder=folder/kind
                if not (folder/'mesh.ply').exists():
                    result[region][kind]=dict(classification='METHOD_FAILED_OR_EMPTY',reason=build[region][kind].get('error','Empty candidate'));continue
                v,f=fdg.read_ply(folder/'mesh.ply')
                try:
                    for name,a in [('vertices.npy',v),('faces.npy',f)]:need(np.load(folder/name,allow_pickle=False).tobytes()==a.tobytes(),'Independent PLY/NPY readback differs')
                    if kind=='reference':
                        ids=np.load(folder/'original-face-ids.npy',allow_pickle=False);vertices=np.load(folder/'original-vertex-ids.npy',allow_pickle=False)
                        need(v.tobytes()==baseline_v[vertices].tobytes() and np.array_equal(vertices[f],baseline_f[ids]),'Reference crop changed original geometry')
                    row=metrics(np,v,f);row['PLY_sha256']=inherited.digest(folder/'mesh.ply');result[region][kind]=row
                    if kind=='chart':
                        p=v[f].astype(np.float64);normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);need(np.all(normal[:,1]*REGIONS[region]['outside_y']>0),'Chart facing differs')
                        need(row['nonmanifold_edges']==row['winding_conflicts']==row['nonmanifold_vertex_links']==0,'Chart sheet reconstruction remains nonmanifold or incoherent')
                finally:v._mmap.close();f._mmap.close()
        inherited.save(BASE/'independent-mesh-verification.json',dict(classification='PASS_READBACK_AND_REFERENCE_IDENTITY',regions=result,
            original_reference_unchanged=True,all_components_counted_no_largest_component_filter=True))
    finally:baseline_v._mmap.close();baseline_f._mmap.close()


def owner(stage):
    module=inherited.load('_zombie_retopology_owned_cpu',Path(inherited.__file__));tree=ast.parse(inspect.getsource(module.supervise));changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='amd_zombie_mesh_render.py':node.value='amd_zombie_retopology_blender.py';changed+=1
    need(changed==1,'Qualified owner route differs');module.__file__=__file__;module.BASE=BASE
    exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),module.__dict__)
    if stage in ('capabilities','build','render','compose'):
        module.BASE=BASE/('owned-'+stage);module.BASE.mkdir(exist_ok=False)
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
    {'prepare':prepare,'verify':verify,'compose':compose}[stage]()


def compose():
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    settings=inherited.read(BASE/'experiment-settings.json');geometry=inherited.read(BASE/'independent-mesh-verification.json')
    views=inherited.read(BASE/'view-measurements.json');render=inherited.read(BASE/'renders/render-metadata.json')
    need(render['classification']=='PASS_MATCHED_CPU_PATCH_PRESENTATION' and render['all_imported_payloads_exact'] and
        render['device']=='CPU' and not render['smooth_shading'] and not render['modifiers_applied_during_render'],'Matched unchanged CPU inputs required')
    need(len(render['frames'])==54 and len({r['name'] for r in render['frames']})==54,'All two-region three-method three-view three-mode frames required')
    for row in render['frames']:need(inherited.digest(BASE/'renders'/row['name'])==row['sha256'],'Frame changed')
    out=BASE/'inspection-sheets-verified';out.mkdir(exist_ok=False);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    measurements={}
    for region,r in settings['regions'].items():
        frame_by={(x['angle'],x['kind'],x['mode']):x for x in render['frames'] if x['region']==region}
        image=Image.new('RGB',(1152,1230),'#24272c');draw=ImageDraw.Draw(image)
        for j,angle in enumerate((-35,0,35)):
            for i,kind in enumerate(('reference','voxel','chart')):
                row=frame_by[angle,kind,'neutral']
                with Image.open(BASE/'renders'/row['name']) as view:image.paste(view.convert('RGB').resize((384,384)),(i*384,j*410+26))
                draw.text((i*384+8,j*410+5),f'{kind.upper()} / {angle:+} deg',font=small,fill='white')
        image.save(out/f'{region}-matched-views.png')
        image=Image.new('RGB',(1152,850),'#24272c');draw=ImageDraw.Draw(image)
        for j,mode in enumerate(('front-back','culled')):
            for i,kind in enumerate(('reference','voxel','chart')):
                row=frame_by[0,kind,mode]
                with Image.open(BASE/'renders'/row['name']) as view:image.paste(view.convert('RGB').resize((384,384)),(i*384,j*410+26))
                draw.text((i*384+8,j*410+5),f'{kind.upper()} / {mode}',font=small,fill='white')
        draw.text((8,827),'Cyan = front, red = back; bottom row hides backfaces. No source smoothing.',font=small,fill='white')
        image.save(out/f'{region}-facing.png')
        # Recompute all sampled coverage/depth counts independently from saved arrays.
        for angle in (-35,0,35):
            prefix=BASE/'renders'/f'{region}-{angle:+03d}'
            ref_depth=np.load(str(prefix)+'-reference-depth.npy',allow_pickle=False);ref_face=np.load(str(prefix)+'-reference-face-ids.npy',allow_pickle=False);ref_valid=ref_face>=0
            for kind in ('reference','voxel','chart'):
                d=np.load(str(prefix)+f'-{kind}-depth.npy',allow_pickle=False);face=np.load(str(prefix)+f'-{kind}-face-ids.npy',allow_pickle=False)
                facing=np.load(str(prefix)+f'-{kind}-facing.npy',allow_pickle=False);valid=face>=0;common=valid&ref_valid;union=valid|ref_valid;row=views[region][str(angle)][kind]
                need(np.array_equal(np.isfinite(d),valid),'Saved ray hit/depth mismatch')
                need(int(valid.sum())==row['candidate_hit_pixels'] and int(common.sum())==row['shared_hit_pixels'] and
                    int((ref_valid&~valid).sum())==row['reference_hit_pixels_lost'] and int((~ref_valid&valid).sum())==row['new_hit_pixels_where_reference_void'] and
                    int((valid&facing).sum())==row['camera_facing_first_hits'],'Independent raster counts differ')
                if common.any():need(abs(depth_p95(np,ref_depth,d,common)-row['absolute_depth_difference']['p95'])<1e-12,'Independent depth quantile differs')
        ref=geometry['regions'][region]['reference'];measurements[region]=dict(reference_boundary_attribution=r['boundary_attribution'],methods={})
        for kind in ('voxel','chart'):
            row=geometry['regions'][region][kind];measurements[region]['methods'][kind]=dict(
                triangle_reduction_fraction=1-row['triangles']/ref['triangles'],geometry=row,views=views[region])
        # Artificial crop margin is a measurement domain, not a claimed source opening.
        row=frame_by[0,'reference','neutral'];scale=row['ortho_scale'];target=np.array(row['target']);location=np.array(row['camera_location'])
        forward=target-location;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward)
        ii,jj=np.meshgrid(np.arange(128),np.arange(128));world=location[None,None,:]+right[None,None,:]*((ii+.5)/128-.5)[:,:,None]*scale+up[None,None,:]*(.5-(jj+.5)/128)[:,:,None]*scale
        h=r['chart_grid_spacing_requested'];core=(world[:,:,0]>=r['lo'][0]+2*h)&(world[:,:,0]<=r['hi'][0]-2*h)&(world[:,:,2]>=r['lo'][2]+2*h)&(world[:,:,2]<=r['hi'][2]-2*h)
        prefix=BASE/'renders'/f'{region}-+00';ref_face=np.load(str(prefix)+'-reference-face-ids.npy',allow_pickle=False);ref_valid=ref_face>=0
        core_rows={}
        for kind in ('reference','voxel','chart'):
            face=np.load(str(prefix)+f'-{kind}-face-ids.npy',allow_pickle=False);valid=face>=0
            core_rows[kind]=dict(reference_core_hit_pixels=int((core&ref_valid).sum()),
                reference_core_hits_retained=int((core&ref_valid&valid).sum()),reference_core_hits_lost=int((core&ref_valid&~valid).sum()),
                new_core_hit_pixels_where_reference_void=int((core&~ref_valid&valid).sum()))
        measurements[region]['straight_view_core_excluding_two_grid_steps_at_crop_margin']=core_rows
        # Explicit boundary annotations: source attribution is exact; chart inner cuts are unclassified support seams.
        image=Image.new('RGB',(1536,566),'#24272c');draw=ImageDraw.Draw(image)
        for i,kind in enumerate(('reference','voxel','chart')):
            folder=BASE/'reference'/region if kind=='reference' else BASE/'candidates'/region/kind
            v=np.load(folder/'vertices.npy',allow_pickle=False);f=np.load(folder/'faces.npy',allow_pickle=False)
            edges,counts,_,_=topology.incidence(np,f);boundary=edges[counts==1]
            with Image.open(BASE/'renders'/frame_by[0,kind,'neutral']['name']) as view:picture=view.convert('RGB')
            mark=ImageDraw.Draw(picture);classes={};artificial=set()
            if kind=='reference':
                mapping=np.load(folder/'original-vertex-ids.npy',allow_pickle=False)
                artificial={tuple(pair) for pair in r['boundary_attribution']['artificial_boundary_original_vertex_pairs']}
            for edge in boundary:
                p=v[edge].astype(np.float64)
                if kind=='reference':tag='crop' if tuple(mapping[edge]) in artificial else 'source-open'
                else:
                    margin=(p[:,0]<=r['lo'][0]+h)|(p[:,0]>=r['hi'][0]-h)|(p[:,2]<=r['lo'][2]+h)|(p[:,2]>=r['hi'][2]-h)
                    tag='crop-margin' if margin.all() else 'unclassified-support-seam'
                classes[tag]=classes.get(tag,0)+1;color='#ffcc33' if tag.startswith('crop') else ('#00eeee' if tag=='source-open' else '#ff66cc')
                d=p-target;xy=np.stack((256+(d@right)*512/scale,256-(d@up)*512/scale),axis=1)
                mark.line(tuple(xy.ravel()),fill=color,width=1)
            image.paste(picture,(i*512,30));draw.text((i*512+8,5),kind.upper(),fill='white',font=font)
            draw.text((i*512+8,543),str(classes),fill='white',font=small)
            measurements[region]['methods'].setdefault(kind,{})['boundary_classes']=classes
        image.save(out/f'{region}-boundary-attribution.png')
    inherited.save(BASE/'comparison-verification.json',dict(classification='PASS_SAVED_GEOMETRY_AND_RASTER_READBACK',
        baseline_PLY_sha256=BASELINE_SHA,all_54_frame_hashes_verified=True,all_saved_ray_counts_and_depth_quantiles_verified=True,
        source_reference_payloads_unchanged=True,measurements=measurements,
        boundary_annotations_are_Xray_projection=True,chart_inner_boundaries_not_semantically_classified=True,
        sheet_images={p.name:inherited.digest(p) for p in out.glob('*.png')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('capabilities','prepare','build','verify','render','compose'))
    parser.add_argument('--worker',choices=('prepare','verify','compose'));args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.stage:
        module=owner(args.stage)
        if args.stage in ('capabilities','build','render'):
            os.environ['MODLY_ZOMBIE_RETOPOLOGY_MODE']=args.stage;raise SystemExit(module.supervise('render'))
        raise SystemExit(module.supervise(args.stage))
    else:print('DRY_RUN: explicit two-crop CPU feasibility stage required')
