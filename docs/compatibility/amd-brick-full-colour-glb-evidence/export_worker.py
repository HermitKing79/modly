"""One CPU full-model material measurement, never a GLB/Blender export."""
import sys
sys.dont_write_bytecode=True
import json,time
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from inputs import P,SAMPLER,METRICS,pinned,load,load_auth,require,digest,array,save


def main():
    require(sys.stdin.buffer.readline(64)==b'GO\n','Exact owned binary GO')
    auth=load_auth();auth.P=P;sys.addaudithook(auth.runtime_guard)
    import numpy as np
    from strata import scan,select
    from decision import evaluate
    began=time.monotonic();data,identities,watched=load(np)
    before={str(path):digest(path) for path in watched}
    m=pinned('full_original_IEC_metrics',METRICS,'7e02dfa43cbf2167e7df44efa0ba1536bb59fdb39ebb0e812bbee61b2a971e20')
    s=pinned('full_original_eight_neighbour',SAMPLER,'0a330eca2c991d3b7751d3cbbbe94edac317b34b28ce7216d9d3d374ef20b0c6')
    v,f,features=data['vertices'],data['faces'],data['features']
    geometry=scan(v,f,features[:,:3]);selection=select(geometry,data['brick'],data['foliage'],seed=20261022)
    qf,bary=selection['query_face'],selection['bary'];require(0<len(qf)<=65536,'Complete frozen bounded sample')
    # Freeze every region and sample before the first source-spatial query.
    out=P/'suitability'
    kinds=np.tile(np.arange(6,dtype=np.int8),len(qf)//6)
    array(out/'query-faces.npy',qf);array(out/'barycentrics.npy',bary);array(out/'query-kinds.npy',kinds)
    cohorts=selection['cohorts']
    saved_cohorts={}
    for name,row in cohorts.items():
        ids=row['selected_face_ids'];array(out/(name+'-selected-face-ids.npy'),np.asarray(ids,dtype=np.int32))
        saved_cohorts[name]={key:value for key,value in row.items() if key!='selected_face_ids'}
    require(len(cohorts)==16 and cohorts['brick']['selected_count']==781,'All declared regions and complete original brick cohort')
    save(out/'frozen-sampling.json',dict(seed=20261022,draw_order=selection['draw_order'],query_count=len(qf),cohorts=saved_cohorts,thresholds=selection['thresholds'],geometry_edge_counts=geometry['edge_counts'],original_bounds=geometry['bounds'],observed_material_errors_before_freeze=False))
    require(time.monotonic()-began<140,'Leave bounded sampling/verification time')
    world=(v[f[qf]].astype(np.float64)*bary[:,:,None]).sum(1).astype(np.float32)
    sampler=s.SparseSampler(data['coordinates'],features)
    reference=sampler.sample_world(world)
    predicted,linear=m.color_prediction(features[:,:3],f,qf,bary)
    result=evaluate(predicted,reference.values,qf,kinds,cohorts,reference.zero_support)
    for name,value in dict(world=world,source_spatial=reference.values,vertex_RGB=predicted,positive_neighbours=reference.positive_weight_neighbour_count,raw_weight_sum=reference.raw_weight_sum).items():array(out/(name+'.npy'),value)
    if result['material_suitability']=='PASS':array(out/'qualified-full-linear-COLOR0.npy',linear)
    require(before=={str(path):digest(path) for path in watched},'Every authenticated input unchanged')
    files={path.name:dict(bytes=path.stat().st_size,sha256=digest(path)) for path in out.glob('*.npy')}
    save(out/'result.json',dict(classification='PASS',measurement_scope='Completed CPU phase-one measurement only; not pipeline/export PASS',material_suitability=result['material_suitability'],source_vertices=len(v),source_triangles=len(f),query_count=len(qf),regions=saved_cohorts,measurements=result,zero_support=int(reference.zero_support.sum()),minimum_positive_neighbours=int(reference.positive_weight_neighbour_count.min()),minimum_raw_weight_sum=float(reference.raw_weight_sum.min()),files=files,original_source_identities=identities,inputs_unchanged=True,no_GLB_export_or_Blender_execution=True,elapsed_seconds=time.monotonic()-began))
    print('FULL_COLOUR_MATERIAL_'+result['material_suitability'],flush=True)


if __name__=='__main__':main()
