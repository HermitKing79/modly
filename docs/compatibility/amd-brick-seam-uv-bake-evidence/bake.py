"""One genuine CPU spatial bake after independently authenticated UV admission."""
import sys,time
from pathlib import Path
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from common import *

def main():
    auth=begin();started=time.monotonic();before=identity_inputs(auth)
    import numpy as np
    from PIL import Image
    math=originals()
    from sampler import SparseSampler
    from bake_math import isolate_padding,footprint_owner,seam_queries
    prior=verify(P/'qualify','uv.json');src=P/'qualify';out=P/'bake'
    review=__import__('json').loads((E/'independent-uv-review.json').read_text())
    if review['classification']!='GO' or review['qualified_uv_receipt_sha256']!=digest(src/'uv.json'):raise ValueError('Independent exact UV review required')
    arrays={n:np.load(src/(n+'.npy'),allow_pickle=False) for n in ('vertices','faces','uv','direct','source-face-ids','source-vertex-ids','face-island-ids','island-rects','raster-ids','barycentrics')}
    v,f,uv=arrays['vertices'],arrays['faces'],arrays['uv'];ids,bary=arrays['raster-ids'],arrays['barycentrics'];covered=ids>=0
    original_v=np.load(auth.PINS['vertices'][0],mmap_mode='r',allow_pickle=False);original_f=np.load(auth.PINS['faces'][0],mmap_mode='r',allow_pickle=False)
    if not np.array_equal(v.view(np.uint32),original_v[arrays['source-vertex-ids']].view(np.uint32)) or not np.array_equal(arrays['source-vertex-ids'][f],original_f[arrays['source-face-ids']]):raise ValueError('Original mesh correspondence changed')
    field=auth.PINS['material'][0];features=auth.material_features(field,np);coords=tensor(field,'coordinates',np)
    if not np.array_equal(coords,tensor(auth.PINS['geometry'][0],'coordinates',np)) or not np.array_equal(arrays['direct'],features[arrays['source-vertex-ids']]):raise ValueError('Original global material rows/coordinates changed')
    sampler=SparseSampler(coords,features)
    values=np.zeros((1024,1024,6),np.float32);pixels=np.argwhere(covered);positive_min=8;denominator_min=1.
    for start in range(0,len(pixels),32768):
        yy,xx=pixels[start:start+32768].T
        world=(v[f[ids[yy,xx]]].astype(np.float64)*bary[yy,xx,:,None]).sum(1).astype(np.float32)
        sample=sampler.sample_world(world)
        if sample.zero_support.any():raise ValueError('Unsupported genuine covered texel')
        values[yy,xx]=sample.values;positive_min=min(positive_min,int(sample.positive_weight_neighbour_count.min()));denominator_min=min(denominator_min,float(sample.raw_weight_sum.min()))
    padded=isolate_padding(values,ids,arrays['face-island-ids'],arrays['island-rects'])
    base,mr=padded['base_color_rgba'],padded['metallic_roughness_rgb']
    # Genuine files, then independent PNG readback before computing fidelity.
    Image.fromarray(base[::-1]).save(out/'base-color.png');Image.fromarray(mr[::-1]).save(out/'MR.png')
    for name,expected in [('base-color.png',base),('MR.png',mr)]:
        with Image.open(out/name) as image:image.load();actual=np.array(image)[::-1]
        if not np.array_equal(actual,expected):raise ValueError('PNG persistence/readback differs')
        if name=='base-color.png':base=actual
        else:mr=actual
    linear_rgb=math.srgb_decode(base[:,:,:3].astype(float)/255)
    scalar_maps=np.stack((mr[:,:,2],mr[:,:,1],base[:,:,3]),-1).astype(float)/255
    xyz=v[f].astype(np.float64);areas=np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)/2
    rng=np.random.default_rng(20261019);random=rng.random((10000,2));root=np.sqrt(random[:,0])
    wf=np.column_stack((1-root,root*(1-random[:,1]),root*random[:,1]))
    query_faces=np.concatenate((np.arange(len(f)),rng.choice(len(f),10000,p=areas/areas.sum()))).astype(np.int32)
    weights=np.vstack((np.full((len(f),3),1/3),wf))
    seams=seam_queries(arrays['source-vertex-ids'][f],arrays['face-island-ids'])
    surface_count=len(query_faces);query_faces=np.concatenate((query_faces,seams['query_faces']));weights=np.vstack((weights,seams['bary']))
    world=(xyz[query_faces]*weights[:,:,None]).sum(1).astype(np.float32)
    quv=(uv[f[query_faces]].astype(np.float64)*weights[:,:,None]).sum(1)
    expected_islands=arrays['face-island-ids'][query_faces]
    if not footprint_owner(quv,expected_islands,padded['ownership']).all():raise ValueError('Texture filtering crosses unowned/foreign island padding')
    expected=np.empty((len(world),6),np.float32)
    for start in range(0,len(world),32768):
        sample=sampler.sample_world(world[start:start+32768])
        if sample.zero_support.any():raise ValueError('Unsupported continuous source holdout or seam query')
        expected[start:start+32768]=sample.values
    baked=np.column_stack((math.srgb_encode(math.bilinear(linear_rgb,quv)),math.bilinear(scalar_maps,quv)))
    def metrics(actual,target):
        return dict(RGB=math.error(actual[:,:3],target[:,:3]),scalars=math.error(actual[:,3:],target[:,3:]))
    target=np.clip(expected,0,1)
    cohorts={'surface':np.arange(surface_count),'seam_edge':np.flatnonzero(seams['cohort']==0)+surface_count,'seam_near':np.flatnonzero(seams['cohort']==1)+surface_count}
    errors={name:metrics(baked[index],target[index]) for name,index in cohorts.items() if len(index)}
    pair=seams['pairing']+surface_count
    if len(pair):
        if not np.allclose(world[pair[:,0]],world[pair[:,1]],rtol=0,atol=1e-7) or not np.allclose(expected[pair[:,0]],expected[pair[:,1]],rtol=0,atol=2e-6):raise ValueError('Source seam correspondence changed')
        errors['seam_gap']=metrics(baked[pair[:,0]]-baked[pair[:,1]],np.zeros((len(pair),6)))
    packed_rgb=padded['original_base_color_rgba'][covered,:3]
    packed=np.column_stack((packed_rgb,padded['original_metallic_roughness_rgb'][covered,2],padded['original_metallic_roughness_rgb'][covered,1],padded['original_base_color_rgba'][covered,3])).astype(float)/255
    quantization=math.error(packed,np.clip(values[covered],0,1))
    # Independent local dictionary oracle from original coordinate tuples only.
    test=world[::max(1,len(world)//64)][:64];q=((test+.5)*512).astype(np.float32)
    # Independently match original coordinate triples, not the sampler's sorted
    # keys/indices. At most 64*8 requested cells enter the scalar dictionary.
    required=np.concatenate([np.floor(q-.5).astype(np.int32)+np.array([i&1,(i>>1)&1,(i>>2)&1],np.int32) for i in range(8)])
    triple=np.dtype([('x','<i4'),('y','<i4'),('z','<i4')])
    original_triples=np.ascontiguousarray(coords[:,1:]).view(triple).ravel()
    requested_triples=np.ascontiguousarray(required).view(triple).ravel()
    eligible=np.flatnonzero(np.isin(original_triples,requested_triples))
    if len(eligible)>512:raise ValueError('Independent scalar oracle exceeds 64*8 genuine rows')
    table={tuple(map(int,coords[i,1:])):np.asarray(features[i],float) for i in eligible};oracle=[]
    for point in q:
        first=np.floor(point-.5).astype(int);total=np.zeros(6);den=0.
        for i in range(8):
            cell=first+np.array([i&1,(i>>1)&1,(i>>2)&1]);value=table.get(tuple(cell))
            if value is not None and (cell>=0).all() and (cell<512).all():
                w=float(np.prod(1-np.abs(point.astype(float)-cell-.5)));total+=w*value;den+=w
        oracle.append(total/max(den,1e-12))
    oracle_error=math.error(sampler.sample_grid(q).values,np.asarray(oracle))
    quality=all(e['RGB']['rmse']<=.02 and e['RGB']['p95_absolute']<=.05 and e['RGB']['max_absolute']<=.15 and e['scalars']['rmse']<=.02 for e in errors.values())
    quality=quality and quantization['max_absolute']<1/255+1e-7 and oracle_error['max_absolute']<=2e-6
    # All original values remain unchanged. Derived clipping is confined to PNG/shader.
    data={n:arrays[n] for n in ('vertices','faces','uv','direct','source-face-ids','source-vertex-ids','face-island-ids','island-rects')}
    data.update({'sampled-field':values,'reference-field':padded['reference'],'coverage':covered,'ownership':padded['ownership'],
                 'query-world':world,'query-uv':quv,'query-faces':query_faces,'continuous-values':expected,'baked-values':baked,'seam-pairing':pair})
    for name,array in data.items():np_save(out/(name+'.npy'),array)
    if identity_inputs(auth)!=before:raise ValueError('Original input identities changed during bake')
    record=dict(classification='PASS' if quality else 'NO-GO',CPU_only=True,inputs_unchanged=True,actual_genuine_spatial_bake=True,
        original_triangle_count=len(f),UV_corner_vertices=len(v),texture_size=[1024,1024],covered_texels=int(covered.sum()),UV_island_count=len(arrays['island-rects']),
        minimum_positive_weight_neighbours=positive_min,minimum_weight_sum=denominator_min,unsupported_queries=0,
        quality_errors=errors,quantization_error=quantization,independent_F64_oracle=oracle_error,independent_oracle_rows=len(eligible),
        query_counts={name:len(idx) for name,idx in cohorts.items()},paired_seam_queries=len(pair),all_filter_footprints_in_correct_island=True,
        predeclared_limits=dict(RGB_RMSE=.02,RGB_p95=.05,RGB_max=.15,scalar_RMSE=.02,oracle_max=2e-6,quantization_max='1/255+1e-7'),
        clipping_counts=((values[covered]<0)|(values[covered]>1)).sum(0).tolist(),sampled_channel_ranges=[values[covered].min(0).tolist(),values[covered].max(0).tolist()],
        padding='Native quantize-then-Telea operation independently within each disjoint owned island rectangle; covered bytes unchanged; all four LOD0 filtering pixels owned by same island',
        reference_scope='Covered texels exact pre-quantization F32 source samples; derived PNG padding held fixed outside coverage. Same-resolution reference is not a continuous field shader.',
        colour_space='Native encoded RGB PNG with exact glTF sRGB decode for filtering; MR Non-Color G=roughness/B=metallic; alpha retained but OPAQUE material',
        UV_orientation='All NPY/raster/reference arrays V-up; only PNG image rows flipped',
        original_geometry_and_source_rows_unchanged=True,qualified_uv_receipt_sha256=digest(src/'uv.json'),
        input_identities_before=before,elapsed_seconds=time.monotonic()-started,
        files=manifest(out,[n+'.npy' for n in data]+['base-color.png','MR.png']))
    save(out/'bake.json',record);print('GENUINE_CPU_BAKE_'+record['classification']+' '+str(errors),flush=True)
    if not quality:raise ValueError('Actual material transfer failed unchanged accuracy gates; no retry')

if __name__=='__main__':main()
