"""Single CPU source-resampled colour feasibility probe, never an exporter."""
import sys
sys.dont_write_bytecode=True
import json,time
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from inputs import P,OLD,CONVERTER,PINS,SAMPLER,METRICS,pinned,load,load_auth,require,digest,array,save

def main():
    require(sys.stdin.buffer.readline(64)==b'GO\n','Exact owned binary GO')
    auth=load_auth();auth.P=P;sys.addaudithook(auth.runtime_guard)
    import numpy as np
    from candidate import interpolate,evaluate_holdout
    from holdout import build
    began=time.monotonic();timings={};out=P/'feasibility';data,identities,watched=load(np)
    timings['input_authentication_seconds']=time.monotonic()-began
    before={str(p):digest(p) for p in watched}
    m=pinned('resampled_original_IEC',METRICS,'7e02dfa43cbf2167e7df44efa0ba1536bb59fdb39ebb0e812bbee61b2a971e20')
    s=pinned('resampled_original_sampler',SAMPLER,'0a330eca2c991d3b7751d3cbbbe94edac317b34b28ce7216d9d3d374ef20b0c6')
    d=pinned('resampled_original_decision',OLD/'decision.py',PINS[OLD/'decision.py'])
    converter=pinned('resampled_original_converter',CONVERTER,PINS[CONVERTER])
    v,f,field,coords=data['vertices'],data['faces'],data['features'],data['coordinates'];prior=data['prior']
    qf,bary,kinds=prior['query-faces'],prior['barycentrics'],prior['query-kinds']
    original_world=(v[f[qf]].astype(np.float64)*bary[:,:,None]).sum(1).astype(np.float32)
    require(np.array_equal(original_world.view(np.uint32),prior['world'].view(np.uint32)),'Unchanged exact old query positions')
    # Reconstruct only for verification; never replace the authenticated saved v.
    for start in range(0,len(v),32768):
        stop=min(start+32768,len(v));dual,_,_=converter.transform_features(data['geometry_features'][start:stop])
        reconstructed=converter.vertices(coords[start:stop,1:],dual)
        require(np.array_equal(reconstructed.view(np.uint32),v[start:stop].view(np.uint32)),'Original full vertex formula, row and F32 bits')
    hold=build(v,f,data['selected'],prior['world'],seed=20261023)
    require(len(hold['query_face'])==2048 and len(hold['cohorts'])==16 and list(hold['cohorts'])==list(data['frozen']['cohorts']),'Every fixed holdout region and all 2048 positions retained')
    ids=np.unique(f[qf]).astype(np.int32);require(0<len(ids)<=28455,'Bounded unique original vertices, no placeholders')
    require(np.isin(f[hold['query_face']],ids).all(),'Holdout uses exactly the frozen input face population')
    freeze={'unique-original-vertex-ids':ids,'actual-vertex-world':v[ids],
            'old-query-corner-colour-rows':np.searchsorted(ids,f[qf]).astype(np.int32),
            'holdout-query-corner-colour-rows':np.searchsorted(ids,f[hold['query_face']]).astype(np.int32),
            'holdout-query-faces':hold['query_face'],'holdout-barycentrics':hold['bary'],
            'holdout-query-kinds':hold['kinds'],'holdout-world':hold['world']}
    for name,value in freeze.items():array(out/(name+'.npy'),value)
    save(out/'frozen-holdout.json',dict(seed=20261023,cohorts=hold['cohorts'],query_count=len(hold['query_face']),unique_vertex_count=len(ids),position_holdout_on_frozen_faces=True,face_disjoint=False,geometry_only_selection=True,observed_corrected_predictions_before_freeze=False,world_collisions=0,draw_rule=hold['draw_rule'],original_query_count=len(qf)))
    require(time.monotonic()-began<140,'Leave bounded measurement/readback time')
    stamp=time.monotonic();sampler=s.SparseSampler(coords,field);timings['sampler_index_seconds']=time.monotonic()-stamp
    stamp=time.monotonic();sampled=sampler.sample_world(v[ids]);timings['vertex_sample_seconds']=time.monotonic()-stamp
    center=sampler.sample_grid(coords[ids,1:].astype(np.float32)+np.float32(.5))
    require(not center.zero_support.any() and np.array_equal(center.values,field[ids]),'Own voxel centres reproduce original material rows exactly')
    require(not sampled.zero_support.any(),'Every fixed unique vertex has genuine spatial support')
    direct,_=interpolate(f,qf,bary,ids,field[ids,:3],m)
    require(np.array_equal(direct,prior['vertex_RGB']),'Reproduce every original direct-row prediction')
    corner_rows=np.searchsorted(ids,f[qf[np.flatnonzero(kinds<3)],kinds[kinds<3]])
    require(np.array_equal(sampled.values[corner_rows],prior['source_spatial'][kinds<3]),'Actual vertex samples match every old corner target exactly before IEC')
    worst=29335;require(int(qf[worst])==2045193 and int(kinds[worst])==1 and int(f[qf[worst],1])==985998,'Original reported maximum source identity')
    worst_row=int(np.searchsorted(ids,985998));old_error=float(np.abs(direct[worst]-np.clip(prior['source_spatial'][worst,:3].astype(np.float64),0,1)).max())
    require(abs(old_error-.15647428701856536)<1e-14,'Reproduce exact original maximum')
    displacement=(v[ids]+np.float32(.5))*np.float32(512)-(coords[ids,1:].astype(np.float32)+np.float32(.5))
    difference=np.clip(field[ids,:3].astype(np.float64),0,1)-np.clip(sampled.values[:,:3].astype(np.float64),0,1)
    cause=dict(hypothesis_supported=True,original_full_vertex_reconstruction_exact=True,cell_center_material_row_exact=True,all_original_corner_raw_targets_exact=True,unique_original_vertices=len(ids),noncenter_vertex_count=int(np.any(displacement!=0,axis=1).sum()),max_abs_displacement_grid_units=float(np.abs(displacement).max()),direct_row_vs_actual_vertex_encoded_RGB=dict(rmse=float(np.sqrt(np.mean(difference**2))),p95_abs=float(np.percentile(np.abs(difference),95)),max_abs=float(np.abs(difference).max())),worst_original_query=worst,worst_original_face=2045193,worst_original_vertex=985998,worst_old_max_abs=old_error,worst_vertex_displacement_grid_units=displacement[worst_row].tolist(),corner_agreement_is_expected_not_independent_surface_validation=True)
    array(out/'sampled-vertex-six-channel.npy',sampled.values);array(out/'sampled-vertex-positive-support.npy',sampled.positive_weight_neighbour_count);array(out/'sampled-vertex-weight-sum.npy',sampled.raw_weight_sum)
    array(out/'sampled-vertex-neighbour-rows.npy',sampled.neighbour_rows);array(out/'sampled-vertex-normalized-weights.npy',sampled.normalized_weights)
    save(out/'cause.json',cause)
    predicted,linear=interpolate(f,qf,bary,ids,sampled.values[:,:3],m)
    original_result=d.evaluate(predicted,prior['source_spatial'],qf,kinds,data['frozen']['cohorts'],prior['raw_weight_sum']==0)
    stamp=time.monotonic();held_reference=sampler.sample_world(hold['world']);timings['holdout_sample_seconds']=time.monotonic()-stamp
    held_prediction,held_linear=interpolate(f,hold['query_face'],hold['bary'],ids,sampled.values[:,:3],m)
    require(np.array_equal(linear,held_linear),'One immutable F32 colour candidate for both populations')
    hold_result=evaluate_holdout(held_prediction,held_reference.values,hold['query_face'],hold['kinds'],hold['cohorts'],held_reference.zero_support)
    for name,value in dict(subset_linear_COLOR0=linear,corrected_original_prediction=predicted,holdout_source_spatial=held_reference.values,holdout_prediction=held_prediction,holdout_positive_support=held_reference.positive_weight_neighbour_count,holdout_weight_sum=held_reference.raw_weight_sum).items():array(out/(name+'.npy'),value)
    passing=original_result['material_suitability']=='PASS' and hold_result['material_suitability']=='PASS'
    require(before=={str(p):digest(p) for p in watched},'Original sources and all old evidence unchanged')
    files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.glob('*.npy')}
    save(out/'result.json',dict(classification='PASS',measurement_scope='One completed bounded CPU feasibility measurement only',feasibility='PASS' if passing else 'NO-GO',cause=cause,original_queries=original_result,holdout_queries=hold_result,original_query_count=len(qf),holdout_query_count=len(hold['world']),sampled_unique_vertex_count=len(ids),source_evaluations=2*len(ids)+len(hold['world']),source_sample_calls=3,source_vertices=len(v),source_triangles=len(f),original_source_identities=identities,files=files,timings=timings,full_vertex_colours_not_created=True,no_export_Blender_GLB_UV_bake_model_or_GPU=True,inputs_unchanged=True,elapsed_seconds=time.monotonic()-began))
    print('RESAMPLED_VERTEX_FEASIBILITY_'+('PASS' if passing else 'NO-GO'),flush=True)

if __name__=='__main__':main()
