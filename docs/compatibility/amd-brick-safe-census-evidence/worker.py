"""One bounded original full-mesh census; never a simplification."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
import common as c
import time,json,traceback,os,gc

def main():
    c.begin();started=time.monotonic();out=c.P/'diagnose';stage='authentication';progress=[];summary={}
    def check():
        if time.monotonic()-started>150:raise TimeoutError('Incomplete census:150-second work budget; preserve30seconds cleanup headroom')
    def checkpoint(name):
        nonlocal stage
        check();stage=name;progress.append(dict(stage=stage,elapsed_seconds=time.monotonic()-started));c.save(out/('progress-%02d.json'%len(progress)),progress[-1]);print(name,flush=True)
    def write(name,a):
        check()
        with (out/(name+'.npy')).open('xb') as f:__import__('numpy').save(f,a,allow_pickle=False);f.flush();os.fsync(f.fileno())
    try:
        import numpy as np
        import census_core as core
        c.authenticate();v,f,rgba,keys,inc=c.load(np);N=len(v)
        labels=np.load(c.plain(c.original.CACHE/'vertex_components.npy'),mmap_mode='r',allow_pickle=False)
        c.require(v.shape==(1957891,3) and f.shape==(4173584,3) and labels.shape==(N,) and labels.dtype==np.int32,'Full original source contracts')
        for s in range(0,N,32768):check();c.require(np.isfinite(v[s:s+32768]).all(),'Finite original source positions')
        checkpoint('directed_source_occurrences')
        order,offsets,used,face_degree,consistent,conflict=core.occurrences(f,keys,inc,N,check)
        summary.update(original_vertices=N,referenced_vertices=int(used.sum()),unused_vertices=int((~used).sum()),original_triangles=len(f),original_edges=len(keys),original_nonmanifold_edges=int((inc>2).sum()),original_boundary_edges=int((inc==1).sum()),consistent_two_face_edges=int(consistent.sum()),orientation_conflict_edges=len(conflict))
        checkpoint('oriented_sheets_and_local_fans')
        sheets,multifan,multisheet,fanstats=core.fan_and_sheets(f,order,offsets,consistent,used,check);summary.update(fanstats)
        checkpoint('strict_source_protection')
        masks=core.protection(v,f,keys,inc,order,offsets,used,multifan,multisheet,conflict,check)
        summary.update({name:int(mask.sum()) for name,mask in masks.items()});masks['multiple_fan_vertices']=multifan;masks['multiple_sheet_vertices']=multisheet
        checkpoint('full_simplicial_link_certification')
        eligible,linkstats,_=core.full_links(f,keys,order,offsets,consistent,masks['protected_faces'],masks['locked_vertices'],N,check);summary.update(linkstats)
        masks['certified_edge_touch_faces']=np.zeros(len(f),np.bool_)
        if len(eligible):
            masks['certified_edge_touch_faces'][order[offsets[eligible]]//3]=True;masks['certified_edge_touch_faces'][order[offsets[eligible]+1]//3]=True
        summary['certified_edge_face_population']=int(masks['certified_edge_touch_faces'].sum())
        checkpoint('component_and_spatial_distributions')
        distribution,regions=core.distributions(v,f,labels,masks,keys,eligible,check)
        c.require(distribution['original_face_bearing_vertex_components']==207 and distribution['unused_source_vertices']==5749,'Original component and unused-row correspondence')
        checkpoint('durable_census_indices')
        for name,a in dict(edge_occurrence_order=order,edge_offsets=offsets,consistent_edges=consistent,global_sheet_labels=sheets,eligible_edge_ids=eligible,face_degree=face_degree,face_octants=regions,**masks).items():write(name,a)
        c.save(out/'distributions.json',distribution)
        result=dict(classification='CENSUS_COMPLETE' if len(eligible) else 'BLOCKED',census_completed=True,reason='One positive-only patch still requires separate qualified extraction; no reduction authorised' if len(eligible) else 'No global edge survives the unchanged junction/interface/one-ring and full-link requirements',summary=summary,progress=progress,source='accepted original only',actual_simplification_calls=0,topology_caches_reused_not_overwritten=True,elapsed_seconds=time.monotonic()-started,projected_peak_admission_bytes=5*1024**3,projected_peak_is_not_measured=True,files={p.name:dict(bytes=p.stat().st_size,sha256=c.digest(p)) for p in sorted(out.iterdir()) if p.suffix in {'.json','.npy'}})
        c.save(out/'result.json',result)
    except BaseException as error:
        c.save(out/'result.json',dict(classification='BLOCKED',census_completed=False,stage=stage,error=str(error),traceback=traceback.format_exc(),summary=summary,progress=progress,elapsed_seconds=time.monotonic()-started,actual_simplification_calls=0));print('CENSUS_INCOMPLETE '+str(error),flush=True)

if __name__=='__main__':main()
