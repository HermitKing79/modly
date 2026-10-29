"""Fixed benchmark CPU preparation/tail. No default execution or neural entry."""
import argparse,hashlib,io,json,os,shutil,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import amd_closed_solid_benchmark as b

def owned(stage,base,*,reference=False):
    owner=b.load('_closed_solid_cpu_owner',Path(__file__).with_name('amd_zombie_cpu_mesh.py'))
    owner.BASE=base;owner.__file__=str(Path(__file__).resolve())
    limits=owner.admission(12*1024**3,2*1024**3)
    if (owner.CAP,owner.DEADLINE,limits['required_RAM_bytes'],limits['required_disk_bytes'])!=(6*1024**3,180,12*1024**3,2*1024**3):raise ValueError('Qualified CPU limits differ')
    launch=subprocess.Popen
    def route(command,**kwargs):
        if stage=='render':
            expected=[str(b.BLENDER),'--background','--factory-startup','--disable-autoexec','--threads','4','--python-exit-code','1','--python',str(Path(__file__).with_name('amd_zombie_mesh_render.py'))]
            if command!=expected:raise ValueError('Foreign render route')
            command=command[:-1]+[str(Path(__file__).with_name('amd_closed_solid_render.py')),'--','--reference' if reference else '--generated']
        else:
            expected=[sys._base_executable,'-B','-I','-S',str(Path(__file__).resolve()),'--worker',stage]
            if command!=expected:raise ValueError('Foreign CPU route')
        return launch(command,**kwargs)
    owner.subprocess=SimpleNamespace(Popen=route,PIPE=subprocess.PIPE)
    return owner.supervise(stage)

def qualify():
    import numpy as np
    from PIL import Image
    import amd_flexible_dual_grid_cpu as mesh
    from amd_brick_four_edge_forensics import triangle_relation
    import amd_generation_occupancy as occ
    v,f=mesh.read_ply(b.PREP/'source/mesh.ply');ev,ef=b.geometry()
    if not np.array_equal(v,np.asarray(ev,dtype='f4')) or not np.array_equal(f,np.asarray(ef,dtype='i4')):raise ValueError('Independent GT readback differs')
    for name,array in [('vertices',v),('faces',f)]:
        if not np.array_equal(np.load(b.PREP/'source'/f'{name}.npy',allow_pickle=False),array):raise ValueError('GT NPY/PLY differs')
    counts=b.small_topology(v.tolist(),f.tolist())
    if counts['boundary'] or counts['nonmanifold'] or counts['winding'] or counts['components']!=1 or counts['euler']!=2 or not counts['convex'] or counts['signed_volume']<=0:raise ValueError('Closed convex ground truth required')
    ratios=[[float(x).as_integer_ratio() for x in row] for row in v];scale=max(d for row in ratios for _,d in row)
    exact=[tuple(n*(scale//d) for n,d in row) for row in ratios];pairs=0
    for i in range(len(f)):
        for j in range(i):
            relation=triangle_relation([exact[k] for k in f[i]],[exact[k] for k in f[j]]);pairs+=1
            if relation['kind'] not in ('disjoint','shared_edge_only','shared_vertex_only'):raise ValueError('GT triangles intersect beyond their shared simplex')
    source=b.PREP/'reference.png';blob=source.read_bytes();future=b.CASE/'occupancy/source-image.png'
    image,contract=occ.image_contract(future,blob)
    if image.size!=(1024,1024) or image.getchannel('A').getextrema()!=(0,255):raise ValueError('Real semantic alpha required')
    alpha=np.array(image.getchannel('A'));coords=np.argwhere(alpha>204);lo=coords.min(axis=0);hi=coords.max(axis=0)
    center=[float((lo[1]+hi[1])/2),float((lo[0]+hi[0])/2)];size=int(max(hi-lo));rect=[int(round(center[0]-size//2)),int(round(center[1]-size//2)),int(round(center[0]+size//2)),int(round(center[1]+size//2))]
    if not (lo>0).all() or not (hi<1023).all():raise ValueError('Source framing clips object')
    cropped=image.crop(rect);a=np.array(cropped).astype('f4')/255
    manual=Image.fromarray((a[:,:,:3]*a[:,:,3:4]*255).astype('u1'))
    actual=occ.original.preprocess_cpu(image)
    if manual.size!=actual.size or manual.tobytes()!=actual.tobytes():raise ValueError('Frozen camera crop differs from qualified preprocessing')
    conditioned=actual.resize((512,512),Image.Resampling.LANCZOS);conditioned.save(b.PREP/'conditioning-512.png')
    cropped.getchannel('A').resize((512,512),Image.Resampling.LANCZOS).save(b.PREP/'silhouette-512.png')
    prepared=b.PREP/'prepared';prepared.mkdir(exist_ok=False)
    with (prepared/'source-image.png').open('xb') as stream:stream.write(blob)
    b.save(prepared/'input-contract.json',contract)
    b.save(b.PREP/'qualification.json',dict(classification='PASS_REFERENCE',topology=counts,all_triangle_pairs_checked=pairs,
        nontrivial_self_intersections=0,PLY_NPY_exact=True,ground_truth_not_neural_output=True,
        source_image_sha256=b.digest(source),input_contract_sha256=b.digest(prepared/'input-contract.json'),source_pixel_sha256=contract['pixel_sha256'],
        crop_rectangle=rect,crop_size=list(actual.size),crop_pixel_sha256=contract['crop_pixel_sha256'],
        conditioning_512_pixel_sha256=hashlib.sha256(conditioned.tobytes()).hexdigest(),conditioning_512_sha256=b.digest(b.PREP/'conditioning-512.png'),
        foreground_bbox_half_open=[int(lo[1]),int(lo[0]),int(hi[1]+1),int(hi[0]+1)],
        camera=b.CAMERA,target=b.TARGET,ortho_scale=b.ORTHO,automatic_candidate_refitting=False,
        seed=b.SEED,neural_attempt_reserved=False,model_imported=False,background='black premultiplied RGB; unchanged qualified method'))
    v._mmap.close();f._mmap.close()

def prepare():
    from amd_zombie_d_handoff import plain
    for path in (b.PREP,b.CASE,b.REFERENCE,b.PRIVATE,b.RAW):plain(path)
    if b.PREP.exists() or b.CASE.exists() or b.REFERENCE.exists():raise FileExistsError('Reference preparation is exclusive; no retry')
    if b.digest(b.BLENDER)!=b.BLENDER_SHA:raise ValueError('Qualified Blender differs')
    b.PREP.mkdir(exist_ok=False)
    source_paths=[Path(__file__),Path(__file__).with_name('amd_closed_solid_benchmark.py'),Path(__file__).with_name('amd_closed_solid_render.py'),
        Path(__file__).with_name('amd_zombie_cpu_mesh.py'),Path(__file__).with_name('amd_zombie_d_handoff.py'),
        Path(__file__).with_name('amd_flexible_dual_grid_cpu.py'),Path(__file__).with_name('amd_generation_occupancy.py'),
        Path(__file__).with_name('amd_object_occupancy.py'),Path(__file__).with_name('amd_brick_four_edge_forensics.py'),
        b.ROOT/'experimental/extensions/trellis2-amd/alpha_input.py',b.BLENDER,
        Path('D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD/trellis2/pipelines/trellis2_image_to_3d.py')]
    sources={str(p.resolve()):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=b.digest(p)) for p in source_paths}
    b.save(b.PREP/'sources-before.json',sources)
    archive=b.PREP/'executed-source';archive.mkdir(exist_ok=False)
    for name in ('amd_closed_solid_benchmark.py','amd_closed_solid_cpu.py','amd_closed_solid_render.py'):
        with Path(__file__).with_name(name).open('rb') as source,(archive/name).open('xb') as out:shutil.copyfileobj(source,out,65536)
    for stage in ('render','qualify'):
        if owned(stage,b.PREP,reference=True):raise RuntimeError('CPU preparation failed; preserve evidence, no retry')
    for p,row in sources.items():
        path=Path(p)
        if path.stat().st_size!=row['bytes'] or path.stat().st_mtime_ns!=row['mtime_ns'] or b.digest(path)!=row['sha256']:raise ValueError('Preparation source changed')
    b.save(b.PREP/'sources-after.json',dict(unchanged=True,files=len(sources)))
    q=json.loads((b.PREP/'qualification.json').read_text());contract=json.loads((b.PREP/'prepared/input-contract.json').read_text())
    target=b.CASE/'occupancy';target.mkdir(parents=True,exist_ok=False)
    for name in ('source-image.png','input-contract.json'):
        with (b.PREP/'prepared'/name).open('rb') as source,(target/name).open('xb') as out:shutil.copyfileobj(source,out,65536)
        if b.digest(target/name)!=b.digest(b.PREP/'prepared'/name):raise ValueError('Prepared input copy differs')
    names=['source/mesh.ply','source/vertices.npy','source/faces.npy','reference.png','conditioning-512.png','silhouette-512.png','render.json','qualification.json']
    files={str((b.PREP/name).relative_to(b.ROOT.parent).as_posix()):dict(bytes=(b.PREP/name).stat().st_size,sha256=b.digest(b.PREP/name)) for name in names}
    evidence={str((b.PREP/name).relative_to(b.ROOT.parent).as_posix()):dict(bytes=(b.PREP/name).stat().st_size,sha256=b.digest(b.PREP/name)) for name in b.PREPARATION_EVIDENCE}
    b.save(b.REFERENCE,dict(schema='modly.amd.closed-solid-reference.v1',source_path=(target/'source-image.png').relative_to(b.ROOT).as_posix(),
        input_contract_path=(target/'input-contract.json').relative_to(b.ROOT).as_posix(),source_sha256=q['source_image_sha256'],source_bytes=(target/'source-image.png').stat().st_size,
        input_contract_sha256=q['input_contract_sha256'],source_pixel_sha256=contract['pixel_sha256'],encoded_worker_input_png_sha256=contract['input_png_sha256'],
        dimensions=list(b.DIMENSIONS),chamfer=b.CHAMFER,seed=b.SEED,resolution=512,candidate_count=1,reference_views=1,
        intended_geometry='one connected closed convex chamfered rectangular box; no openings or separate parts',
        conditioning_512_pixel_sha256=q['conditioning_512_pixel_sha256'],crop_rectangle=q['crop_rectangle'],files=files,preparation_evidence=evidence,
        neural_execution_hold=True,automatic_retries=0,ground_truth_may_replace_neural_output=False))
    print(json.dumps(q),flush=True);return 0

def decode_reader():
    """Read-only existing ten-tensor verifier, bound to this exact input only."""
    import amd_closed_solid_handoff as h
    row=h.authority();reader=b.load('_closed_solid_ten_tensor_reader',Path(__file__).with_name('amd_zombie_d_compare.py'))
    reader.D_ROOT=b.CASE;reader.DECODE=b.CASE/'decode';reader.OCCUPANCY=b.CASE/'occupancy';reader.SHAPE=b.CASE/'shape'
    reader.SOURCE=reader.DECODE/'run/guest/shape-decoded.safetensors';reader.BASE=b.RAW;reader.REFERENCE=b.REFERENCE;reader.IMAGE_SHA=row['source_sha256']
    def reference(ref,contract,input_sha):
        if (ref!=h.authority() or contract['source_path']!=ref['source_path'] or contract['source_sha256']!=ref['source_sha256']
            or contract['pixel_sha256']!=ref['source_pixel_sha256'] or contract['input_png_sha256']!=ref['encoded_worker_input_png_sha256']
            or input_sha!=ref['input_contract_sha256']):raise ValueError('Exact new box image provenance required')
    reader.validate_reference=reference
    return reader

def topology_and_shape():
    import numpy as np
    from scipy.spatial import cKDTree
    helper=b.load('_closed_solid_topology',b.ROOT/'docs/compatibility/amd-brick-defect-diagnosis-evidence/topology.py')
    v=np.load(b.RAW/'mesh/vertices.npy',mmap_mode='r',allow_pickle=False);f=np.load(b.RAW/'mesh/faces.npy',mmap_mode='r',allow_pickle=False)
    result=helper.analyze(v,f,b.RAW/'topology');n=len(v)
    keys=np.load(b.RAW/'topology/edge_keys.npy',mmap_mode='r');counts=np.load(b.RAW/'topology/incidence.npy',mmap_mode='r');winding=np.zeros(len(keys),np.int16)
    for start in range(0,len(f),32768):
        block=f[start:start+32768]
        for a,c in ((0,1),(1,2),(2,0)):
            code=np.minimum(block[:,a],block[:,c]).astype('u8')*n+np.maximum(block[:,a],block[:,c]).astype('u8');pos=np.searchsorted(keys,code)
            if not np.array_equal(keys[pos],code):raise ValueError('Original edge correspondence differs')
            np.add.at(winding,pos,np.where(block[:,a]<block[:,c],1,-1).astype('i2'))
    conflicts=int(((counts==2)&(np.abs(winding)==2)).sum())
    ordered=np.sort(f,axis=1);codes=(ordered[:,0].astype('u8')*n+ordered[:,1].astype('u8'))*n+ordered[:,2].astype('u8')
    duplicate=len(f)-len(np.unique(codes));del ordered,codes
    result['degeneracy'].update(duplicate_faces_assessed=True,duplicate_faces_reason='Complete independent sorted index-triple count, ignoring winding',duplicate_index_triangles=duplicate)
    def sample(vertices,faces):
        areas=np.empty(len(faces),np.float64)
        for start in range(0,len(faces),32768):
            tri=vertices[faces[start:start+32768]].astype('f8');areas[start:start+len(tri)]=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5
        cumulative=np.cumsum(areas)
        if not np.isfinite(cumulative).all() or cumulative[-1]<=0:raise ValueError('Finite nonempty surface required')
        ids=np.searchsorted(cumulative,(np.arange(4096)+.5)/4096*cumulative[-1]);tri=vertices[faces[ids]].astype('f8')
        uv=np.random.default_rng(b.SEED).random((4096,2));r=np.sqrt(uv[:,0]);weights=np.column_stack((1-r,r*(1-uv[:,1]),r*uv[:,1]))
        return (tri*weights[:,:,None]).sum(axis=1)
    gv,gf=b.geometry();gv=np.asarray(gv,'f4');gf=np.asarray(gf,'i4');original=sample(gv,gf);candidate=sample(v,f);diag=float(np.linalg.norm(np.asarray(b.DIMENSIONS)))
    shape={}
    for label,a,c in [('GT_to_raw',original,candidate),('raw_to_GT',candidate,original)]:
        d=cKDTree(c).query(a,workers=1)[0]/diag
        shape[label]=dict(samples=len(d),RMS=float(np.sqrt(np.mean(d*d))),p95=float(np.quantile(d,.95)),maximum=float(d.max()))
    result.update(two_face_winding_conflicts=conflicts,duplicate_index_triangles_ignoring_winding=duplicate,
        raw_bounds=[v.min(axis=0).tolist(),v.max(axis=0).tolist()],ground_truth_dimensions=list(b.DIMENSIONS),
        raw_axis_aligned_dimensions=(v.max(axis=0)-v.min(axis=0)).astype('f8').tolist(),
        raw_axis_aligned_dimension_ratios=((v.max(axis=0)-v.min(axis=0))/(v.max(axis=0)-v.min(axis=0)).max()).astype('f8').tolist(),
        ground_truth_dimension_ratios=list(b.DIMENSIONS),euler_referenced_vertices=n-result['components']['unused_vertices']-len(keys)+len(f),
        shape=shape,shape_method='4096 area-stratified deterministic surface-point samples each way; point-cloud distance, not triangle/Hausdorff distance',
        registration='NONE: no rotation, translation, scaling, fitting or candidate normalisation; values are pose-sensitive',
        self_intersections='UNASSESSED for full AI mesh; GT all946pairs verified separately',
        raw_array_sha256={name:b.digest(b.RAW/'mesh'/name) for name in ('vertices.npy','faces.npy')},
        texture_ready=False,automatic_geometry_promotion=False,geometry_repair=False)
    b.save(b.RAW/'raw-quality.json',result)
    for array in (v,f,keys,counts):array._mmap.close()

def compare_view():
    import numpy as np
    from PIL import Image,ImageDraw
    ref=json.loads((b.PREP/'render.json').read_bytes());raw=json.loads((b.RAW/'view/render.json').read_bytes());a=json.loads((b.RAW/'benchmark-inputs.json').read_bytes())
    reader=decode_reader();mesh=json.loads((b.RAW/'mesh/mesh.json').read_bytes());reader.validate_mesh(mesh,a['binding'])
    for key in ('camera','target','ortho_scale','resolution','lights','device','engine','threads','samples','material_linear_RGB','roughness','view_transform','look','camera_matrix_world','blender_version'):
        if ref[key]!=raw[key]:raise ValueError('Matched camera/lighting differs: '+key)
    if raw['ground_truth'] is not False or raw['source_PLY_sha256']!=mesh['artifacts']['mesh.ply']['sha256'] or not raw['geometry_unchanged']:raise ValueError('Raw AI render provenance differs')
    unchanged=reader.snapshot((b.RAW/'view/render.json',b.RAW/'view/raw-reference-camera.png'))
    if b.digest(b.RAW/'view/raw-reference-camera.png')!=raw['image_sha256']:raise ValueError('Raw preview PNG differs from render receipt')
    q=json.loads((b.PREP/'qualification.json').read_bytes());rectangle=q['crop_rectangle'];out=b.RAW/'comparison';out.mkdir(exist_ok=False)
    with Image.open(b.RAW/'view/raw-reference-camera.png') as image:
        frame=np.array(image.getchannel('A'))>204;clipped=bool(frame[0].any() or frame[-1].any() or frame[:,0].any() or frame[:,-1].any())
        crop=image.crop(rectangle);array=np.array(crop).astype('f4')/255
        rgb=Image.fromarray((array[:,:,:3]*array[:,:,3:4]*255).astype('u1')).resize((512,512),Image.Resampling.LANCZOS)
        mask=np.array(crop.getchannel('A').resize((512,512),Image.Resampling.LANCZOS))>204
        outside_crop=int(frame.sum()-frame[rectangle[1]:rectangle[3],rectangle[0]:rectangle[2]].sum())
    with Image.open(b.PREP/'silhouette-512.png') as image:truth=np.array(image)>204
    union=int((truth|mask).sum());intersection=int((truth&mask).sum())
    if not union:raise ValueError('Empty matched silhouettes')
    rgb.save(out/'raw-conditioning-camera-512.png')
    overlay=np.zeros((512,512,3),np.uint8);overlay[truth&mask]=[180,180,180];overlay[truth&~mask]=[80,160,255];overlay[mask&~truth]=[255,100,70]
    Image.fromarray(overlay).save(out/'silhouette-overlay.png')
    sheet=Image.new('RGB',(1536,540),'#20242b');draw=ImageDraw.Draw(sheet)
    with Image.open(b.PREP/'conditioning-512.png') as image:sheet.paste(image.convert('RGB'),(0,0))
    sheet.paste(rgb,(512,0));sheet.paste(Image.fromarray(overlay),(1024,0))
    for x,label in [(8,'Procedural GT: not AI output'),(520,'Untouched AI raw geometry'),(1032,'Grey overlap | blue GT only | red AI only')]:draw.text((x,518),label,fill='white')
    sheet.save(out/'comparison.png')
    b.save(out/'comparison.json',dict(classification='MATCHED_RAW_EVIDENCE_COMPLETE_NOT_GEOMETRY_APPROVAL',silhouette_IoU=intersection/union,
        GT_foreground_pixels=int(truth.sum()),raw_foreground_pixels=int(mask.sum()),intersection_pixels=intersection,union_pixels=union,
        fixed_crop_rectangle=rectangle,automatic_candidate_refitting=False,registration='NONE; single-image output pose/scale ambiguity may affect comparison',
        raw_clipped_in_full_frame=clipped,raw_pixels_outside_frozen_crop=outside_crop,source_image_sha256=a['binding']['source_image_sha256'],
        neural_output_sha256=a['binding']['source_sha256'],raw_PLY_sha256=mesh['artifacts']['mesh.ply']['sha256'],GT_PLY_sha256=ref['source_PLY_sha256'],
        silhouette_overlay_sha256=b.digest(out/'silhouette-overlay.png'),comparison_sha256=b.digest(out/'comparison.png'),postprocessing=False,texture_ready=False))
    reader.verify_snapshot(unchanged)

def future(commit):
    import amd_closed_solid_handoff as h
    h.verify_publication(commit);h.authority()
    if b.RAW.exists():raise FileExistsError('Raw evidence already consumed; no retry')
    b.RAW.mkdir(exist_ok=False);reader=decode_reader();document,binding,unchanged=reader.receipt_inputs(commit)
    b.save(b.RAW/'benchmark-inputs.json',dict(harness_commit=commit,binding=binding,decoded_inputs=unchanged,
        published_sources=reader.snapshot(b.ROOT/name for name in h.CRITICAL),reference_files=h.authority()['files']))
    for stage in ('extract','analyze','render','compare'):
        inputs=json.loads((b.RAW/'benchmark-inputs.json').read_bytes());reader.verify_snapshot(inputs['decoded_inputs']);reader.verify_snapshot(inputs['published_sources']);h.authority()
        if stage!='extract':reader.validate_mesh(reader.read(b.RAW/'mesh/mesh.json'),binding)
        if owned(stage,b.RAW):return 1
    reader.verify_snapshot(unchanged);h.authority();b.save(b.RAW/'final.json',dict(classification='RAW_BENCHMARK_EVIDENCE_COMPLETE_REVIEW_REQUIRED',
        raw_geometry_unchanged=True,automatic_repair=False,postprocessing=False,texture_ready=False))
    return 0

def worker(stage):
    if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO plus EOF required')
    base=b.PREP if stage=='qualify' else b.RAW
    sys.addaudithook(b.cpu_guard(base));sys.path.insert(0,'D:/AI/Trellis2AMD/venv/Lib/site-packages')
    if stage=='qualify':return qualify()
    inputs=json.loads((b.RAW/'benchmark-inputs.json').read_bytes());reader=decode_reader();reader.verify_snapshot(inputs['decoded_inputs']);reader.verify_snapshot(inputs['published_sources'])
    mesh_snapshot=None
    if stage!='extract':
        reader.validate_mesh(reader.read(b.RAW/'mesh/mesh.json'),inputs['binding'])
        mesh_snapshot=reader.snapshot([b.RAW/'mesh'/n for n in ('mesh.json','mesh.ply','vertices.npy','faces.npy')])
    if stage=='extract':
        import amd_generation_mesh as mesh
        contract,binding=reader.authenticate(inputs['harness_commit'])
        b.save(b.RAW/'authentication-extract.json',binding)
        result=mesh.extract(reader.SOURCE,binding['source_sha256'],b.RAW/'mesh',contract=contract);reader.validate_mesh(result,binding)
    elif stage=='analyze':topology_and_shape()
    elif stage=='compare':compare_view()
    else:raise ValueError('Unknown fixed CPU phase')
    reader.verify_snapshot(inputs['decoded_inputs']);reader.verify_snapshot(inputs['published_sources'])
    if mesh_snapshot:reader.verify_snapshot(mesh_snapshot);reader.validate_mesh(reader.read(b.RAW/'mesh/mesh.json'),inputs['binding'])

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-reference',action='store_true');parser.add_argument('--worker',choices=('qualify','extract','analyze','compare'))
    args=parser.parse_args(argv)
    if args.worker:return worker(args.worker)
    if args.prepare_reference:return prepare()
    print('DRY_RUN: no reference, model, WSL, marker or future CPU work');return 0
if __name__=='__main__':raise SystemExit(main())
