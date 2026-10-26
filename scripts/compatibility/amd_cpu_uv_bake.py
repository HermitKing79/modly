"""Bounded CPU UV/baking proof. Blender fallback is distinct from Modly native ops."""
import argparse,hashlib,importlib.util,json,os,re,struct,subprocess,sys,zlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT.parent/'artifacts/amd-cpu-uv-bake-proof-20261026-v1'
SOURCE=ROOT.parent/'artifacts/amd-zombie-d-comparison-9ac0364f-v1/mesh/mesh.ply'
SOURCE_SHA='2f8a1f9057c88f588b171c66f75af88b98fa47f1cea7494d6ee55517ce9e8205'
sys.path.insert(0,str(Path(__file__).parent))
import amd_zombie_cpu_mesh as cpu
import amd_zombie_d_compare as legacy_policy
read=cpu.read;save=cpu.save;digest=cpu.digest
_base_audit=legacy_policy.audit_policy(BASE)


def geometry_hash(np,v,f):
    # Triangle order / cyclic corner rotation / seam duplicates may differ in GLB.
    # Winding, triangle membership and every coordinate must still agree exactly.
    p=np.array(v[f],dtype=np.float32,copy=True);p[p==0]=0
    original=p.view(np.uint32).reshape(-1,9);best=original.copy()
    for order in ([1,2,0],[2,0,1]):
        candidate=p[:,order].copy().view(np.uint32).reshape(-1,9)
        equal=np.ones(len(best),bool);less=np.zeros(len(best),bool)
        for i in range(9):less|=equal&(candidate[:,i]<best[:,i]);equal&=candidate[:,i]==best[:,i]
        best[less]=candidate[less]
    keys=np.ascontiguousarray(best).view(np.dtype((np.void,36))).ravel();keys.sort()
    return hashlib.sha256(keys.tobytes()).hexdigest()


def uv_metrics(np,uv):
    if uv.ndim!=3 or uv.shape[1:]!=(3,2) or not np.isfinite(uv).all():raise ValueError('Finite per-corner triangle UVs required')
    if uv.min()<0 or uv.max()>1:raise ValueError('UVs outside [0,1]')
    d1=uv[:,1].astype(np.float64)-uv[:,0];d2=uv[:,2].astype(np.float64)-uv[:,0]
    area=np.abs(d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0])*.5
    result=dict(triangles=len(uv),zero_area_triangles=int((area==0).sum()),sum_area=float(area.sum()),min=float(uv.min()),max=float(uv.max()))
    if result['zero_area_triangles']:raise ValueError('Zero-area UV triangle')
    return result


def srgb(np,x):return np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,0)**(1/2.4)-.055)


def write_png(np,path,linear_rgba):
    data=np.clip(linear_rgba,0,1).copy();data[...,:3]=srgb(np,data[...,:3])
    pixels=np.rint(data[::-1]*255).astype(np.uint8);h,w=pixels.shape[:2]
    def chunk(kind,payload):return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload)&0xffffffff)
    scan=b''.join(b'\0'+row.tobytes() for row in pixels)
    encoded=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>2I5B',w,h,8,6,0,0,0))+chunk(b'sRGB',b'\0')+chunk(b'IDAT',zlib.compress(scan,6))+chunk(b'IEND',b'')
    with Path(path).open('xb') as f:f.write(encoded)


def raster_reference(np,uv,colors,size):
    """Independent CPU barycentric reference at pixel centres, strict interiors.

    Returns coverage multiplicity and first-hit colour. Shared-edge samples are
    excluded rather than misclassified as overlapping interiors.
    """
    counts=np.zeros((size,size),np.uint16);expected=np.zeros((size,size,3),np.float32)
    scaled=uv.astype(np.float64)*size-.5
    lo=np.maximum(0,np.ceil(scaled.min(axis=1)).astype(np.int32));hi=np.minimum(size-1,np.floor(scaled.max(axis=1)).astype(np.int32))
    relevant=np.flatnonzero(np.all(hi>=lo,axis=1))
    for i in relevant:
        x,y=np.meshgrid(np.arange(lo[i,0],hi[i,0]+1),np.arange(lo[i,1],hi[i,1]+1))
        p=scaled[i];e1=p[1]-p[0];e2=p[2]-p[0];det=e1[0]*e2[1]-e1[1]*e2[0]
        if det==0:raise ValueError('Degenerate reference triangle')
        dx=x-p[0,0];dy=y-p[0,1]
        b=(dx*e2[1]-dy*e2[0])/det;c=(e1[0]*dy-e1[1]*dx)/det;a=1-b-c
        inside=(a>1e-6)&(b>1e-6)&(c>1e-6)
        xx=x[inside];yy=y[inside]
        first=counts[yy,xx]==0
        rgb=a[inside,None]*colors[i,0]+b[inside,None]*colors[i,1]+c[inside,None]*colors[i,2]
        expected[yy[first],xx[first]]=rgb[first]
        counts[yy,xx]+=1
    return counts,expected


def glb_arrays(np,path):
    from PIL import Image
    import io
    if path.stat().st_size>512*1024**2:raise ValueError('Bounded GLB required')
    data=path.read_bytes();magic,version,total=struct.unpack_from('<4sII',data)
    if (magic,version,total)!=(b'glTF',2,len(data)):raise ValueError('GLB header')
    length,kind=struct.unpack_from('<I4s',data,12)
    if kind!=b'JSON':raise ValueError('GLB JSON chunk')
    doc=json.loads(data[20:20+length]);start=20+length;blen,bkind=struct.unpack_from('<I4s',data,start)
    if bkind!=b'BIN\0':raise ValueError('Embedded BIN required')
    blob=memoryview(data)[start+8:start+8+blen]
    def accessor(index):
        a=doc['accessors'][index];view=doc['bufferViews'][a['bufferView']]
        if a.get('sparse') or a.get('normalized'):raise ValueError('Unexpected sparse/normalized accessor')
        dtype={5126:'<f4',5125:'<u4',5123:'<u2',5121:'u1'}[a['componentType']]
        width={'VEC3':3,'VEC2':2,'SCALAR':1}[a['type']];dt=np.dtype(dtype)
        stride=view.get('byteStride',width*dt.itemsize);offset=view.get('byteOffset',0)+a.get('byteOffset',0)
        if offset+(a['count']-1)*stride+width*dt.itemsize>len(blob):raise ValueError('Accessor bounds')
        return np.ndarray((a['count'],width),dt,buffer=blob,offset=offset,strides=(stride,dt.itemsize)).copy()
    if len(doc['meshes'])!=1 or len(doc['meshes'][0]['primitives'])!=1:raise ValueError('One mesh/primitive required')
    primitive=doc['meshes'][0]['primitives'][0]
    if primitive.get('mode',4)!=4:raise ValueError('Triangles only')
    v=accessor(primitive['attributes']['POSITION']);uv=accessor(primitive['attributes']['TEXCOORD_0']);f=accessor(primitive['indices']).reshape(-1,3).astype(np.int32)
    # Exporter uses Y-up. Source and Blender inspection copies use Z-up.
    v=np.ascontiguousarray(v[:,[0,2,1]]);v[:,1]*=-1
    for node in doc.get('nodes',[]):
        if 'mesh' in node and any(k in node for k in ('matrix','translation','rotation','scale')):raise ValueError('Unexpected mesh-node transform')
    if not np.isfinite(v).all() or f.min()<0 or f.max()>=len(v):raise ValueError('Invalid exported geometry')
    material=doc['materials'][primitive['material']]
    if material.get('alphaMode','OPAQUE')!='OPAQUE':raise ValueError('Diagnostic colour must not cut geometry using alpha')
    texture=material['pbrMetallicRoughness']['baseColorTexture']
    if texture.get('texCoord',0)!=0:raise ValueError('UV set mismatch')
    image=doc['images'][doc['textures'][texture['index']]['source']]
    if image.get('mimeType')!='image/png' or 'uri' in image:raise ValueError('Embedded lossless PNG required')
    view=doc['bufferViews'][image['bufferView']];encoded=bytes(blob[view.get('byteOffset',0):view.get('byteOffset',0)+view['byteLength']])
    with Image.open(io.BytesIO(encoded)) as im:pixels=np.array(im.convert('RGBA'))
    return v,f,uv,pixels,dict(material_references_valid=True,embedded_PNG_bytes=len(encoded),export_vertices=len(v),triangles=len(f),opaque_material=True)


def audit_policy(event,args):
    # Only temporary files created by this owned proof may be removed by glTF I/O.
    if event in ('os.remove','os.rmdir'):
        cpu.plain(os.fsdecode(args[0])).relative_to(BASE/'temp');return
    _base_audit(event,args)


def verify(stage):
    import numpy as np
    from PIL import Image
    names=['plane','cube'] if stage=='small-verify' else ['zombie']
    result=[]
    for name in names:
        folder=BASE/name;metadata=read(folder/'bake.json')
        v=np.load(folder/'vertices.npy',allow_pickle=False);f=np.load(folder/'faces.npy',allow_pickle=False)
        uv=np.load(folder/'loop-uv.npy',allow_pickle=False);colors=np.load(folder/'loop-colors.npy',allow_pickle=False)
        baked=np.load(folder/'linear-bake.npy',allow_pickle=False)
        if not np.isfinite(baked).all():raise ValueError('Non-finite bake')
        stats=uv_metrics(np,uv);counts,expected=raster_reference(np,uv,colors,metadata['resolution'])
        unique=counts==1;errors=np.abs(baked[...,:3]-expected)
        maximum=float(errors[unique].max()) if unique.any() else None
        rms=float(np.sqrt(np.mean(errors[unique].astype(np.float64)**2))) if unique.any() else None
        covered=int((counts>0).sum());overlap=int((counts>1).sum())
        if not unique.any() or maximum>2e-3:raise ValueError('Independent linear-colour reference differs: '+str(maximum))
        if name!='zombie' and overlap:raise ValueError('Synthetic UV interiors overlap')
        with Image.open(folder/'diagnostic.png') as im:png=np.array(im.convert('RGBA'))
        encoded=np.rint(np.clip(srgb(np,baked[::-1,:,:3]),0,1)*255).astype(np.uint8)
        if not np.array_equal(png[...,:3],encoded):raise ValueError('Saved image differs from explicit sRGB encoding')
        gv,gf,guv,gpng,gmeta=glb_arrays(np,folder/'textured.glb')
        if len(gf)!=len(f) or geometry_hash(np,v,f)!=geometry_hash(np,gv,gf):raise ValueError('GLB surface geometry changed')
        uv_metrics(np,guv[gf])
        if not np.array_equal(png[...,:3],gpng[...,:3]):raise ValueError('Embedded texture pixels changed')
        row=dict(name=name,classification='PASS_CPU_TEXTURED_GLB',baker='BLENDER_CYCLES_CPU_FALLBACK_NOT_MODLY_NATIVE',
                 uv=stats,strict_interior_covered_texels=covered,sampled_overlapping_interior_texels=overlap,
                 sampled_unique_texels=int(unique.sum()),linear_colour_max_abs_error=maximum,linear_colour_RMSE=rms,
                 texture_resolution=metadata['resolution'],opaque_alpha_texels=int((baked[...,3]>.5).sum()),
                 alpha_is_not_UV_coverage=True,strict_interior_coverage_fraction=covered/metadata['resolution']**2,
                 UV_overlap_quality='FAIL_SAMPLED_OVERLAP' if overlap else 'NO_OVERLAP_AT_TESTED_TEXEL_CENTRES_CONTINUOUS_TEST_NOT_PERFORMED',
                 surface_geometry_exact=True,PNG_pixels_exact=True,GLB_sha256=digest(folder/'textured.glb'),
                 diagnostic_colours_only=True,**gmeta)
        save(folder/'independent-verification.json',row);result.append(row)
    save(BASE/(stage+'.json'),dict(classification='PASS',results=result))


def failed_colour_error(stderr):
    matches=re.findall(r'^ValueError: Independent linear-colour reference differs: ([0-9.eE+\-]+)$',stderr,re.MULTILINE)
    if len(matches)!=1:raise ValueError('An explicit retained colour failure is required')
    value=float(matches[0])
    if not .002<value<=1:raise ValueError('Invalid colour-failure value')
    return value


def preview_record(name,sha256,location,target,scale):
    """JSON-safe camera values, including Blender's NumPy Float32 bounds."""
    return dict(name=name,sha256=sha256,camera=[float(v) for v in location],
                target=[float(v) for v in target],ortho_scale=float(scale))


def preview_closeout():
    """Label already saved images; never rerender or upgrade the rejected bake."""
    from PIL import Image,ImageDraw
    status=read(BASE/'owned-preview/render-run/host-result.json')
    stderr=(BASE/'owned-preview/render-run/stderr.txt').read_text(encoding='utf-8')
    if status['classification']!='FAIL' or 'TypeError: Object of type float32 is not JSON serializable' not in stderr:
        raise ValueError('Retained failed preview metadata evidence required')
    folder=BASE/'previews';records=[]
    sheet=Image.new('RGB',(1536,660),'#202020');draw=ImageDraw.Draw(sheet)
    draw.text((12,8),'DIAGNOSTIC ONLY - Zombie D colour accuracy FAILED. Neutral original / rejected texture.',fill='white')
    for i,name in enumerate(('front','back','hand-minus','hand-plus','face','wounds')):
        x=(i%3)*512;y=36+(i//3)*312
        draw.text((x+8,y),name+' | Original neutral / diagnostic output',fill='white')
        for j,label in enumerate(('original','diagnostic-texture')):
            path=folder/(label+'-'+name+'.png')
            with Image.open(path) as image:
                if image.size!=(640,640) or image.format!='PNG':raise ValueError('Unexpected completed preview')
                image.load();sheet.paste(image.convert('RGB').resize((256,256)),(x+j*256,y+22))
            records.append(dict(name=path.name,sha256=digest(path)))
    target=folder/'comparison-sheet.png'
    with target.open('xb') as stream:sheet.save(stream,format='PNG')
    save(folder/'closeout.json',dict(classification='COMPLETED_IMAGES_FROM_FAILED_METADATA_STAGE',
      original_stage_remains='FAIL',accepted_Zombie_D_texture=False,images=records,
      rendering_repeated=False,baking_repeated=False,original_partial_metadata_preserved=True,
      camera_claim='Same camera/lighting loop in amd_cpu_uv_bake_blender.py; failed metadata is not a completed receipt',
      comparison_sheet_sha256=digest(target)))


def readback_only():
    """Close out the retained rejected candidate, without rebaking or retesting colour."""
    import numpy as np
    from PIL import Image
    failed=read(BASE/'owned-zombie-verify/zombie-verify-run/host-result.json')
    if failed['classification']!='FAIL':raise ValueError('Retained failed verification required')
    stderr=BASE/'owned-zombie-verify/zombie-verify-run/stderr.txt'
    if stderr.stat().st_size>65536:raise ValueError('Bounded error evidence required')
    logged_error=failed_colour_error(stderr.read_text(encoding='utf-8'))
    folder=BASE/'zombie';v=np.load(folder/'vertices.npy',allow_pickle=False);f=np.load(folder/'faces.npy',allow_pickle=False)
    uv=np.load(folder/'loop-uv.npy',allow_pickle=False)
    gv,gf,guv,gpng,gmeta=glb_arrays(np,folder/'textured.glb')
    if geometry_hash(np,v,f)!=geometry_hash(np,gv,gf) or len(f)!=len(gf):raise ValueError('Container surface differs')
    uv_metrics(np,guv[gf])
    with Image.open(folder/'diagnostic.png') as image:png=np.array(image.convert('RGBA'))
    if not np.array_equal(png[...,:3],gpng[...,:3]):raise ValueError('Embedded texture pixels differ')
    d1=uv[:,1].astype(np.float64)-uv[:,0];d2=uv[:,2].astype(np.float64)-uv[:,0]
    areas=np.abs(d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0])*.5*2048**2
    save(BASE/'rejected-candidate-readback.json',dict(classification='PASS_CONTAINER_READBACK_REJECTED_COLOUR',
      overall_Zombie_D_texture_verdict='FAIL_COLOUR_ACCURACY',colour_verification_repeated=False,
      UV_or_bake_or_export_repeated=False,retained_failed_test_max_error=logged_error,
      fixed_acceptance_max_error=.002,surface_geometry_exact=True,embedded_RGB_pixels_exact=True,
      UV_sum_area=float(areas.sum()/2048**2),mean_texel_area_per_triangle=float(areas.mean()),
      triangles_below_one_texel_area=int((areas<1).sum()),triangles=len(f),
      GLB_sha256=digest(folder/'textured.glb'),**{k:value for k,value in gmeta.items() if k!='triangles'}))


def supervise(stage):
    owner=cpu.load('_uv_bake_owned_cpu',Path(cpu.__file__))
    control=BASE/('owned-'+stage);control.mkdir(exist_ok=False)
    owner.BASE=control;owner.__file__=str(Path(__file__).resolve());launch=subprocess.Popen
    is_blender=stage in ('synthetic','zombie-uv','zombie-bake','preview')
    def routed(command,**kwargs):
        if is_blender:
            expected=[str(cpu.BLENDER),'--background','--factory-startup','--disable-autoexec','--threads','4','--python-exit-code','1',
                      '--python',str(Path(__file__).with_name('amd_zombie_mesh_render.py'))]
            if command!=expected:raise ValueError('Changed inherited Blender route')
            command=command[:-1]+[str(Path(__file__).with_name('amd_cpu_uv_bake_blender.py'))]
        elif command!=[sys._base_executable,'-B','-I','-S',str(Path(__file__).resolve()),'--worker',stage]:raise ValueError('Changed CPU route')
        kwargs['env']=dict(kwargs['env'],MODLY_UV_STAGE=stage,TEMP=str(BASE/'temp'),TMP=str(BASE/'temp'))
        return launch(command,**kwargs)
    owner.subprocess=SimpleNamespace(Popen=routed,PIPE=subprocess.PIPE)
    return owner.supervise('render' if is_blender else stage)


def worker(stage):
    if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Exact owned GO and EOF required')
    sys.addaudithook(audit_policy);sys.path.append('D:/AI/Trellis2AMD/venv/Lib/site-packages')
    if stage=='readback-only':readback_only()
    elif stage=='preview-closeout':preview_closeout()
    else:verify(stage)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    choices=('synthetic','small-verify','zombie-uv','zombie-bake','zombie-verify','readback-only','preview','preview-closeout')
    parser.add_argument('--stage',choices=choices);parser.add_argument('--worker',choices=('small-verify','zombie-verify','readback-only','preview-closeout'))
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.stage:raise SystemExit(supervise(args.stage))
    else:print('DRY_RUN: explicit CPU proof stage required; no native compilation, model or WSL entrypoint')
