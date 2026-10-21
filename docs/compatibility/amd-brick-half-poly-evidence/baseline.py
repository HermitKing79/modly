"""Read-only authentication and deterministic plan; no simplifier invocation."""
import sys
sys.dont_write_bytecode=True
import json,time,traceback
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from inputs import *

def main():
    begin();import numpy as np
    from reduction_contract import scan,oriented_keys,TARGET_TRIANGLES,ERROR,FLAGS,GATES
    started=time.monotonic();out=P/'baseline'
    try:
        data=load(np);v,f=data['vertices'],data['faces'];original_scan=scan(v,f)
        source_keys=oriented_keys(f,len(v));source_keys.sort();duplicates=int((source_keys[1:]==source_keys[:-1]).sum());del source_keys
        require(duplicates==0,'Original has no inherited duplicate oriented triangles')
        labels=data['labels'];face_counts=np.bincount(labels[f[:,0]],minlength=int(labels.max())+1)
        expected=np.flatnonzero(face_counts).astype(np.int32);require(len(expected)==207,'Original207 components')
        keys=np.load(CACHE/'edge_keys.npy',mmap_mode='r',allow_pickle=False);inc=np.load(CACHE/'incidence.npy',mmap_mode='r',allow_pickle=False)
        require(keys.shape==inc.shape==(6167053,) and (inc>2).sum()==387068 and (inc==1).sum()==267652,'Historical topology identity/counts')
        bad=keys[inc>2];locked=np.zeros(len(v),np.bool_);locked[bad//len(v)]=True;locked[bad%len(v)]=True
        protected=0
        for s in range(0,len(f),32768):protected+=int((locked[f[s:s+32768]].sum(1)>=2).sum())
        meta=json.loads(BASELINE.read_text());views=meta['views'];frozen={name:next(x for x in views if x['name']==name) for name in ('pos-x','pos-y','pos-z','iso-a')}
        for camera in frozen.values():
            camera['camera_type']='ORTHO';camera['clip']=[float(np.float32(.001*meta['bounding_sphere_radius'])),float(np.float32(20*meta['bounding_sphere_radius']))]
        write_array(out/'expected-components.npy',expected)
        # The already frozen holdouts avoid adapting quality probes to output.
        h=data['resampled']
        for src,name in [('holdout-query-faces','query-faces'),('holdout-barycentrics','query-barycentrics'),('holdout-world','query-world'),('holdout_source_spatial','query-source')]:write_array(out/(name+'.npy'),h[src])
        save(out/'views.json',frozen);save(out/'cohorts.json',data['holdout_frozen']['cohorts'])
        plan=dict(method='installed meshoptimizer0.22 stable index-only simplify',target_triangles=TARGET_TRIANGLES,error=ERROR,flags=FLAGS,
                  positions=dict(path=str(MESH/'vertices.npy'),sha256=digest(MESH/'vertices.npy'),shape=[1957891,3],dtype='<f4'),
                  indices=dict(path=str(MESH/'faces.npy'),sha256=digest(MESH/'faces.npy'),shape=[4173584,3],dtype='<i4'),
                  implementation=dict(path=str(MOD/'meshopt_simplifier.js'),sha256=PINS[MOD/'meshopt_simplifier.js']),
                  output=str(P/'reduce'),WASM_max_pages=32768,V8_heap_MiB=1536,
                  no_weld_prune_repair_or_attribute_replacement=True,gates=GATES)
        save(out/'plan.json',plan)
        names=['expected-components.npy','query-faces.npy','query-barycentrics.npy','query-world.npy','query-source.npy','views.json','cohorts.json','plan.json']
        save(out/'result.json',dict(classification='PASS',source=original_scan,original_face_components=207,original_duplicate_oriented_triangles=duplicates,
             original_boundary_edges=267652,original_nonmanifold_edges=387068,Blender_protected_triangle_lower_bound=protected,
             Blender_bound_is_not_a_meshoptimizer_output_prediction=True,no_candidate_calls=True,views=list(frozen),
             private_plan_SHA256=digest(out/'plan.json'),files=manifest(out,names),elapsed_seconds=time.monotonic()-started))
        print('HALF_POLY_BASELINE_PASS',flush=True)
    except BaseException as error:
        save(out/'failure.json',dict(classification='BLOCKED',error=str(error),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
