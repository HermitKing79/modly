"""One integer-translation atlas, original CPU spatial sampler and native padding."""
import sys,json,hashlib
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
OLD=ROOT/'docs/compatibility/amd-brick-pbr-mip-evidence'
SEAM=ROOT/'docs/compatibility/amd-brick-seam-uv-bake-evidence'
sys.path.insert(0,str(OLD))
from assets import PRIOR,GLB,decode_glb,digest,save,require,align_uv,authenticate
P=ROOT.parent/'artifacts/amd-brick-mip-safe-atlas-e8244cb'

def prepare_candidate(out):
    import numpy as np
    from PIL import Image
    from footprint import validate_isolation
    from repack_glb import repack_glb
    phase=json.loads((P/'feasibility/run/result.json').read_text())
    review=json.loads((E/'independent-feasibility-outcome.json').read_text())
    require(review['classification']=='GO' and review['result_SHA256']==digest(P/'feasibility/run/result.json'),'Independent full phase1 GO')
    auth,identities,watched,receipts=authenticate()
    # Historical implementation identities are checked before importing them.
    sys.path.insert(0,str(SEAM));import common
    common.originals()
    import uv_layout
    from bake_math import isolate_padding
    from sampler import SparseSampler
    b=PRIOR/'bake';q=PRIOR/'qualify';data={n:np.load(b/(n+'.npy'),allow_pickle=False) for n in ('vertices','faces','uv','source-vertex-ids','source-face-ids','face-island-ids','direct','query-uv','query-faces','continuous-values','seam-pairing')}
    v,f,u,fi=data['vertices'],data['faces'],data['uv'],data['face-island-ids']
    rects=np.array(phase['ownership_rectangles'],np.int32);shifts=np.array(phase['virtual_integer_translations'],np.int32)
    require(np.array_equal(f,np.arange(len(v)).reshape(-1,3)),'Original sequential corner layout required')
    uv=u.astype(float)+np.repeat(shifts[fi],3,axis=0)/1024
    proof=validate_isolation(uv[f],fi,rects)
    ids,bary=uv_layout._rasterize(uv,f,1024)
    oldids=np.load(q/'raster-ids.npy',allow_pickle=False);oldbary=np.load(q/'barycentrics.npy',allow_pickle=False)
    counts=np.bincount(ids[ids>=0],minlength=781);oldcounts=np.bincount(oldids[oldids>=0],minlength=781)
    require(np.array_equal(counts,oldcounts) and counts.min()>=74,'Every original triangle texel coverage retained')
    for island,(dx,dy) in enumerate(shifts):
        ys,xs=np.nonzero((oldids>=0)&(fi[np.maximum(oldids,0)]==island))
        require(np.array_equal(ids[ys+dy,xs+dx],oldids[ys,xs]) and np.array_equal(bary[ys+dy,xs+dx],oldbary[ys,xs]),'Integer translation raster/barycentric identity')
    overlap=uv_layout.first_overlap(uv,f)
    require(overlap is None,'No positive UV overlap')
    # Prove exact original oriented triangle/material source identity.
    ov=np.load(auth.PINS['vertices'][0],mmap_mode='r',allow_pickle=False);of=np.load(auth.PINS['faces'][0],mmap_mode='r',allow_pickle=False)
    require(np.array_equal(v.view(np.uint32),ov[data['source-vertex-ids']].view(np.uint32)) and np.array_equal(data['source-vertex-ids'][f],of[data['source-face-ids']]),'Master triangle geometry identity')
    features=auth.material_features(auth.PINS['material'][0],np);coords=common.tensor(auth.PINS['material'][0],'coordinates',np)
    require(np.array_equal(coords,common.tensor(auth.PINS['geometry'][0],'coordinates',np)) and np.array_equal(features[data['source-vertex-ids']],data['direct']),'Authentic six-channel source rows/coords')
    sampler=SparseSampler(coords,features);values=np.zeros((1024,1024,6),np.float32);pixels=np.argwhere(ids>=0)
    for start in range(0,len(pixels),32768):
        yy,xx=pixels[start:start+32768].T
        world=(v[f[ids[yy,xx]]].astype(float)*bary[yy,xx,:,None]).sum(1).astype(np.float32)
        result=sampler.sample_world(world);require(not result.zero_support.any(),'Genuine eight-neighbour spatial support')
        values[yy,xx]=result.values
    oldvalues=np.load(b/'sampled-field.npy',mmap_mode='r',allow_pickle=False)
    for island,(dx,dy) in enumerate(shifts):
        ys,xs=np.nonzero((oldids>=0)&(fi[np.maximum(oldids,0)]==island))
        require(np.array_equal(values[ys+dy,xs+dx].view(np.uint32),oldvalues[ys,xs].view(np.uint32)),'Rebaked covered genuine F32 values byte-exact')
    padded=isolate_padding(values,ids,fi,rects)
    target=out/'candidate';target.mkdir();nb=target/'bake';nq=target/'qualify';nb.mkdir();nq.mkdir()
    for name,array in {'raster-ids':ids,'barycentrics':bary}.items():common.np_save(nq/(name+'.npy'),array)
    data['query-uv']=data['query-uv']+shifts[fi[data['query-faces']]]/1024
    data.update(uv=uv,**{'island-rects':rects,'ownership':padded['ownership'],'coverage':ids>=0,'sampled-field':values,'reference-field':padded['reference']})
    for name,array in data.items():common.np_save(nb/(name+'.npy'),array)
    for name,array in [('base-color.png',padded['base_color_rgba']),('MR.png',padded['metallic_roughness_rgb'])]:
        Image.fromarray(array[::-1]).save(nb/name)
        with Image.open(nb/name) as image:require(np.array_equal(np.array(image)[::-1],array),'Encoded PNG readback')
    original=decode_glb(GLB.read_bytes());actualold,error=align_uv(v,f,u,original)
    payload=repack_glb(GLB.read_bytes(),v,f,actualold,uv.astype(np.float32),(nb/'base-color.png').read_bytes(),(nb/'MR.png').read_bytes())
    with (target/'patch.glb').open('xb') as stream:stream.write(payload)
    glb=decode_glb(payload);actualnew,error=align_uv(v,f,uv,glb);validate_isolation(actualnew[f],fi,rects)
    for key,name in [('base','base-color.png'),('mr','MR.png')]:require(glb['embedded'][key]['sha256']==digest(nb/name),'Actual GLB embedded genuine maps')
    save(nb/'bake.json',dict(classification='PASS',actual_genuine_spatial_bake=True,source_samples_byte_exact=True,
        original_geometry_unchanged=True,source_input_identities=identities,triangle_count=781,covered_texels=int(len(pixels)),minimum_face_coverage=int(counts.min()),
        UV_density_unchanged=True,integer_translation_only=True,support_proof=proof,GLB_SHA256=digest(target/'patch.glb'),GLB_bytes=len(payload),
        files=common.manifest(nb,[n+'.npy' for n in data]+['base-color.png','MR.png'])))
    return target
