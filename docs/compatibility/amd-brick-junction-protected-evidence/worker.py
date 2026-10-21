"""Original-patch feasibility only. No simplifier/Blender/native invocation."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
import common as c
import json,time,traceback,os

def main():
    c.begin();began=time.monotonic();out=c.P/'diagnose'
    try:
        import numpy as np
        import patch_contract as pc
        c.authenticate();v,f,rgba,keys,inc=c.load(np)
        chosen,seed,edge_faces,rings=pc.select(f,len(v))
        source_faces=np.asarray(f[chosen],np.int32);ids=np.unique(source_faces).astype(np.int32)
        vertices=np.asarray(v[ids],np.float32);faces=np.searchsorted(ids,source_faces).astype(np.int32)
        outer=np.searchsorted(chosen,rings[-1]).astype(np.int32)
        summary,arrays=pc.classify(vertices,faces,ids,source_faces,outer,keys,inc,len(v))
        for name,a in dict(vertices=vertices,faces=faces,source_vertex_ids=ids,source_face_ids=chosen,source_faces=source_faces,original_native_RGBA=np.asarray(rgba[ids],np.uint8),**arrays).items():
            with (out/(name+'.npy')).open('xb') as stream:np.save(stream,a,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
            b=np.load(out/(name+'.npy'),mmap_mode='r',allow_pickle=False);c.require(a.dtype==b.dtype and np.array_equal(a,b),'Exact readback: '+name)
        c.require(np.array_equal(ids[faces],source_faces) and np.array_equal(vertices.view(np.uint32),v[ids].view(np.uint32)),'Exact original positions/corner order/source correspondence')
        details=dict(seed_source_edge=list(pc.SEED),complete_original_incident_face_ids=edge_faces.tolist(),seed_endpoint_star_face_ids=seed.tolist(),ring_added_face_ids=[a.tolist() for a in rings],original_face_ids=chosen.tolist(),outer_ring_source_face_ids=rings[-1].tolist(),summary=summary)
        c.save(out/'selection.json',details)
        verdict='FEASIBILITY_ONLY' if summary['eligible_interior_reduction_admission'] else 'BLOCKED'
        reason='Separate API/candidate/sample/view/material qualification still required; no simplification authorised by a positive count alone' if verdict=='FEASIBILITY_ONLY' else 'No sufficient strictly eligible interior after complete junction/interface/boundary and one-ring protection'
        c.save(out/'result.json',dict(classification=verdict,reason=reason,summary=summary,seed_edge=list(pc.SEED),ring_face_counts=[len(a) for a in rings],seed_vertex_star_faces=len(seed),complete_original_edge_faces=edge_faces.tolist(),actual_simplification_calls=0,input_is_accepted_original_not_rejected=True,files={p.name:dict(bytes=p.stat().st_size,sha256=c.digest(p)) for p in sorted(out.iterdir()) if p.suffix in {'.json','.npy'}},elapsed_seconds=time.monotonic()-began))
    except BaseException as error:
        c.save(out/'failure.json',dict(classification='BLOCKED',error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-began));raise

if __name__=='__main__':main()
