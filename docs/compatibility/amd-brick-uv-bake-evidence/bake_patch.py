"""One unchanged original brick patch, covered-texel source-semantic CPU bake.

No import or execution of Torch, the original GPU baker, or a neural model.
The native closest-point step is identity on these exact original triangles.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,importlib.util,json,struct,time,os
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-brick-uv-bake-18c0c92/qualified'
PREV=ROOT/'docs/compatibility/amd-brick-defect-diagnosis-evidence'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def save(path,value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def tensor(path,name,np):
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        if not 0<n<65536:raise ValueError('Bounded authenticated tensor header')
        h=json.loads(f.read(n))[name]
    return np.memmap(path,mode='r',dtype={'I32':'<i4','F32':'<f4'}[h['dtype']],
                     offset=8+n+h['data_offsets'][0],shape=tuple(h['shape']))


def main():
    print('BRICK_BAKE_WAITING_GO',flush=True)
    if sys.stdin.buffer.readline(64)!=b'GO\n':raise ValueError('Owned host binary GO required')
    started=time.monotonic()
    if hashlib.sha256((PREV/'diagnose.py').read_bytes()).hexdigest()!='8fa7c617324ba636856b48f47a995ee64cd2a85f3c26d86d8760ec638b163529':raise ValueError('Input verifier pin')
    if hashlib.sha256((PREV/'projection.py').read_bytes()).hexdigest()!='0ed20e6d03424404f3b3aebc64cae2825a0d1c8d241fd494ce50e742471eb840':raise ValueError('Projection pin')
    auth=load('source_input_auth',PREV/'diagnose.py');auth.P=P
    sys.addaudithook(auth.runtime_guard)
    import numpy as np
    import cv2
    cv2.ocl.setUseOpenCL(False);cv2.setNumThreads(4)
    from PIL import Image,ImageDraw
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    sys.path.insert(0,str(E))
    from sampler import SparseSampler
    from patch_math import chart,rasterize,bilinear,error,srgb_decode,srgb_encode,source_padding,assert_injective
    projection=load('source_projection',PREV/'projection.py')
    if auth.digest(PREV/'projection.py')!='0ed20e6d03424404f3b3aebc64cae2825a0d1c8d241fd494ce50e742471eb840':raise ValueError('Projection pin')
    before={k:auth.identity(path) for k,(path,_) in auth.PINS.items()}
    for k,(_,sha) in auth.PINS.items():auth.require(before[k]['sha256']==sha,'Original input pin: '+k)
    sem=json.loads((E/'source-semantics.json').read_text())
    # Source files are independently pinned in the source audit and the host marker.
    v=np.load(auth.PINS['vertices'][0],mmap_mode='r',allow_pickle=False)
    f=np.load(auth.PINS['faces'][0],mmap_mode='r',allow_pickle=False)
    field=auth.PINS['material'][0];features=auth.material_features(field,np)
    coords=tensor(field,'coordinates',np);gc=tensor(auth.PINS['geometry'][0],'coordinates',np)
    auth.require(np.array_equal(coords,gc),'Complete geometry/material coordinate ordering differs')
    mapping=auth.read_json(auth.PINS['mapping_receipt'][0])
    auth.require(mapping['classification']=='PASS','Authenticated extraction correspondence required')
    camera=next(q for q in auth.read_json(auth.PINS['render_metadata'][0])['views'] if q['name']=='pos-x')
    projected=projection.project(v,camera,1024,1024);screen=projected.screen
    # Fixed nomination in an exposed middle brick, separate from A1/A3 and leaves.
    roi=np.array([[560,540],[600,580]],float);center=np.array([580.5,554.5])
    candidates=[]
    for start in range(0,len(f),65536):
        tri=f[start:start+65536];xy=screen[tri]
        ok=((xy>=roi[0]).all((1,2)) & (xy<=roi[1]).all((1,2)))
        ids=np.flatnonzero(ok)+start
        if len(ids):
            xyz=v[f[ids]].astype(np.float64);normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
            length=np.linalg.norm(normal,axis=1);cos=normal[:,0]/np.maximum(length,1e-30)
            candidates.extend(ids[cos<-.9].tolist())  # Observed original visible brick sheet has negative-X winding.
    candidates=np.asarray(candidates,np.int32)
    auth.require(20<=len(candidates)<=10000,'Bounded source-front patch nomination failed')
    tri=f[candidates];xy=screen[tri];a=xy[:,1]-xy[:,0];b=xy[:,2]-xy[:,0];r=center-xy[:,0]
    det=a[:,0]*b[:,1]-a[:,1]*b[:,0]
    w1=(r[:,0]*b[:,1]-r[:,1]*b[:,0])/det;w2=(a[:,0]*r[:,1]-a[:,1]*r[:,0])/det
    weights=np.stack((1-w1-w2,w1,w2),1);inside=weights.min(1)>=-1e-9
    auth.require(inside.any(),'Original center ray must hit nominated brick')
    depth=(v[tri,:,][...,0]*weights).sum(1)
    seed=int(np.flatnonzero(inside)[np.argmax(depth[inside])]);seed_x=depth[seed]
    true_ray=projection.sample_rays(v,f,projected,[[580,554]])[0]
    auth.require(true_ray['hit'] and true_ray['face_id']==int(candidates[seed]),'Nominated patch must be the true nearest original surface, not an occluded rear sheet')
    near=np.abs(v[tri,:,][...,0].mean(1)-seed_x)<.012
    near[seed]=True;local=np.flatnonzero(near);subtri=tri[local]
    # Connected original edge component, not a clipped/remeshed triangle surface.
    edges={};adj=[]
    for i,t in enumerate(subtri):
        for u,z in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):edges.setdefault(tuple(sorted((int(u),int(z)))),[]).append(i)
    for group in edges.values():
        for i in group:
            for j in group:
                if i!=j:adj.append((i,j))
    auth.require(adj,'Connected original faces required')
    rows,cols=np.asarray(adj).T
    _,labels=connected_components(csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(local),len(local))),directed=False)
    seedlocal=int(np.flatnonzero(local==seed)[0]);selected=candidates[local[labels==labels[seedlocal]]]
    auth.require(20<=len(selected)<=6000,'Selected patch must remain small but representative')
    original_faces=f[selected];original_ids,inv=np.unique(original_faces,return_inverse=True)
    vertices=np.array(v[original_ids],copy=True);faces=inv.reshape(-1,3).astype(np.int32)
    direct=np.array(features[original_ids],copy=True)
    auth.require(np.array_equal(vertices[faces],v[f[selected]]),'Original triangle positions/winding differ')
    auth.require(float(np.ptp(direct[:,:3],axis=0).max())>.05,'Insufficient learned colour variation')
    uv=chart(vertices,faces,1024,16);injectivity=assert_injective(uv,faces);ids,bary=rasterize(uv,faces,1024)
    covered=ids>=0;counts=np.bincount(ids[covered],minlength=len(faces))
    auth.require((counts>0).all(),'Every selected triangle must have covered texel samples')
    sampler=SparseSampler(coords,features)
    values=np.zeros((1024,1024,6),np.float32);positions=np.argwhere(covered)
    support_min=8;denom_min=1.;zero=0;world_all=[]
    for start in range(0,len(positions),32768):
        pos=positions[start:start+32768];yy,xx=pos.T;face=faces[ids[yy,xx]]
        world=(vertices[face].astype(np.float64)*bary[yy,xx,:,None]).sum(1).astype(np.float32)
        result=sampler.sample_world(world);values[yy,xx]=result.values
        zero+=int(result.zero_support.sum());support_min=min(support_min,int(result.positive_weight_neighbour_count.min()))
        denom_min=min(denom_min,float(result.raw_weight_sum.min()))
    auth.require(zero==0,'Unresolved sparse support at covered texels')
    # Exact source padding: native byte packing first, then CPU Telea inpaint.
    base,mr,original_base,original_mr=source_padding(values,covered)
    out=P/'bake'
    for name,array in [('vertices',vertices),('faces',faces),('uv',uv),('direct',direct),('source-face-ids',selected),('source-vertex-ids',original_ids),('sampled-field',values),('coverage',covered)]:
        with (out/(name+'.npy')).open('xb') as stream:np.save(stream,array,allow_pickle=False)
        auth.require(np.array_equal(np.load(out/(name+'.npy'),allow_pickle=False),array),'NPY byte/value readback')
    Image.fromarray(base[::-1]).save(out/'base-color.png')
    Image.fromarray(mr[::-1]).save(out/'MR.png')
    for name,expected in [('base-color.png',base),('MR.png',mr)]:
        with Image.open(out/name) as im:auth.require(np.array_equal(np.asarray(im)[::-1],expected),'PNG exact pixel readback')
    rng=np.random.default_rng(20261019)
    xyz=vertices[faces].astype(np.float64);areas=np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)/2
    query_faces=np.concatenate((np.arange(len(faces)),rng.choice(len(faces),10000,p=areas/areas.sum())))
    rand=rng.random((10000,2));root=np.sqrt(rand[:,0]);wb=np.column_stack((1-root,root*(1-rand[:,1]),root*rand[:,1]))
    wb=np.vstack((np.full((len(faces),3),1/3),wb))
    qworld=(xyz[query_faces]*wb[:,:,None]).sum(1).astype(np.float32)
    quv=(uv[faces[query_faces]].astype(np.float64)*wb[:,:,None]).sum(1)
    holdout=sampler.sample_world(qworld)
    auth.require(not holdout.zero_support.any(),'Unsupported independent surface query/face centroid')
    expected=holdout.values
    baked_rgb=srgb_encode(bilinear(srgb_decode(base[:,:,:3].astype(float)/255),quv))
    baked_scalar=np.column_stack((bilinear(mr.astype(float)/255,quv)[:,2],bilinear(mr.astype(float)/255,quv)[:,1],bilinear(base.astype(float)/255,quv)[:,3]))
    rgb_error=error(baked_rgb,np.clip(expected[:,:3],0,1));scalar_error=error(baked_scalar,np.clip(expected[:,3:],0,1))
    direct_at=(direct[faces[query_faces]].astype(float)*wb[:,:,None]).sum(1)
    direct_difference=error(direct_at[:,:3],expected[:,:3])
    quant_values=np.concatenate((original_base[covered,:3],original_mr[covered,2,None],original_mr[covered,1,None],original_base[covered,3,None]),axis=1).astype(float)/255
    quant_error=error(quant_values,np.clip(values[covered],0,1))
    # Independent F64 scalar oracle uses a small complete spatial window from
    # ORIGINAL coordinate tuples, not the sampler's sorted table or neighbour rows.
    test=qworld[::max(1,len(qworld)//64)][:64];q=((test+.5)*512).astype(np.float32)
    low=np.floor(q-.5).astype(int).min(0);high=np.floor(q-.5).astype(int).max(0)+1
    eligible=np.flatnonzero(((coords[:,1:]>=low)&(coords[:,1:]<=high)).all(1))
    auth.require(len(eligible)<20000,'Independent local reference must be bounded')
    table={tuple(map(int,coords[i,1:])):np.asarray(features[i],float) for i in eligible}
    oracle=[]
    for point in q:
        b=np.floor(point-.5).astype(int);total=np.zeros(6);den=0.
        for n in range(8):
            c=b+np.array([n&1,(n>>1)&1,(n>>2)&1]);value=table.get(tuple(c))
            if value is not None and (c>=0).all() and (c<512).all():
                weight=float(np.prod(1-np.abs(point.astype(float)-c-.5)));total+=weight*value;den+=weight
        oracle.append(total/max(den,1e-12))
    oracle_error=error(sampler.sample_grid(q).values,np.asarray(oracle))
    passed=(rgb_error['rmse']<=.02 and rgb_error['p95_absolute']<=.05 and rgb_error['max_absolute']<=.15 and scalar_error['rmse']<=.02 and oracle_error['max_absolute']<=2e-6 and quant_error['max_absolute']<1/255+1e-7)
    image=Image.open(auth.PINS['image_ev2'][0]).copy();draw=ImageDraw.Draw(image);draw.rectangle(tuple(roi.flatten()),outline='cyan',width=3)
    image.save(out/'selection.png')
    pixels=screen[original_ids]
    record=dict(classification='PASS' if passed else 'NO-GO',CPU_only=True,selected_original_face_count=len(faces),selected_original_vertex_count=len(vertices),original_faces_and_vertices_bitwise_preserved=True,
                nomination_roi=roi.tolist(),nomination_center_pixel=center.tolist(),A1_A3_excluded=True,projected_bounds=[pixels.min(0).tolist(),pixels.max(0).tolist()],
                bounds=[vertices.min(0).tolist(),vertices.max(0).tolist()],learned_channel_ranges=[direct.min(0).tolist(),direct.max(0).tolist()],
                UV_chart='single YZ planar chart, mirrored U for observed negative-X original winding; no face reversal; reject analytic positive-area overlaps',UV_injectivity=injectivity,true_original_center_ray=true_ray,texture_size=[1024,1024],covered_texels=int(covered.sum()),coverage_fraction=float(covered.mean()),
                every_face_sampled=True,texture_seams_internal=0,artificial_patch_boundaries=True,minimum_found_positive_weight_neighbours=support_min,minimum_native_weight_denominator=denom_min,unsupported_covered_texels=zero,
                source_spatial_RGB_vs_baked_GLTF_filter_error=rgb_error,source_scalars_vs_baked_filter_error=scalar_error,direct_vertex_RGB_vs_source_spatial=direct_difference,
                quantization_error_against_clipped_field=quant_error,independent_F64_oracle_error=oracle_error,independent_reference_rows=len(eligible),
                test_samples=len(query_faces),every_face_centroid_included=True,source_vs_bake_predeclared_RGB_limits=dict(rmse=.02,p95_absolute=.05,max_absolute=.15),
                source_packing='clip(encoded_sample*255,0,255).astype(uint8), truncation; Telea padding only outside coverage',
                RGB_encoding='Native encoded field sampled before quantization; PNG sRGB; GLTF bilinear filtering in linear light',scalar_encoding='MR Non-Color R=0,G=roughness,B=metallic; base alpha retained; OPAQUE',
                no_native_GPU_bitwise_parity_claim=True,closest_point_projection='identity: queried points on exact original source triangles, no remesh',
                elapsed_seconds=time.monotonic()-started,input_identities_before=before,
                files={x.name:dict(bytes=x.stat().st_size,sha256=auth.digest(x)) for x in out.iterdir() if x.is_file()})
    after={k:auth.identity(path) for k,(path,_) in auth.PINS.items()}
    auth.require(before==after,'Original source changed during CPU bake')
    record['inputs_unchanged']=True;save(out/'bake.json',record)
    print(json.dumps(record),flush=True)
    if not passed:raise ValueError('Predeclared bake fidelity gates failed; no retry')


if __name__=='__main__':main()
