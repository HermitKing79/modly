"""Observe every frozen complete-ring context; never simplify or choose a seed."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
import common as c
import json,time,traceback,os

def main():
    c.begin();began=time.monotonic();out=c.P/'context'
    record=dict(classification='BLOCKED',actual_simplification_calls=0,census_reruns=0,
                baseline_requalified=False,observations_completed=False,source='accepted original only',observations=[],selected_depth=None,diagram=None)
    def check():
        if time.monotonic()-began>150:raise TimeoutError('150-second context work budget; leave30seconds for receipts/cleanup')
    def manifest():
        return {p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=c.digest(p))
                for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in {'.json','.npy','.png'} and p!=out/'result.json'}
    try:
        # Production numerical imports and all pinned NumPy modules follow GO.
        import numpy as np
        import qualification as q
        import progress_contract as progress
        c.authenticate();check();choice=c.frozen_choice()
        old_result=c.RECOVERY/'patch/result.json';old=json.loads(c.plain(old_result).read_text())
        old_selection=json.loads(c.plain(c.RECOVERY/'patch/selection.json').read_text())
        c.require(c.digest(old_result)=='f1a3dbe764f73757153edacbe2e782c99426870798d2b9f79c2985b81a8a7865'
                  and old['classification']=='BLOCKED' and old['summary']['original_patch_faces']==44,
                  'Authenticated prior44-face BLOCKED observation')
        c.require(choice['seed_edge_id']==1889720 and choice['seed_source_vertex_ids']==[612841,612852]
                  and old_selection['seed_edge_face_ids']==[1270262,1270266], 'One unchanged frozen original seed')
        baseline=np.load(c.plain(c.RECOVERY/'patch/source_face_ids.npy'),mmap_mode='r',allow_pickle=False)
        baseline_outer=np.asarray(old_selection['ring_added_face_ids'][-1],np.int32)
        c.require(baseline.dtype==np.int32 and len(baseline)==44 and len(baseline_outer)==16,
                  'Exact saved baseline and complete third outer ring; no baseline qualification')
        record.update(frozen_choice=choice,baseline=dict(depth=3,labels=['baseline'],result_SHA256=c.digest(old_result),summary=old['summary']))
        v,f,rgba,keys,inc=c.load(np);check()
        c.require(list(divmod(int(keys[choice['seed_edge_id']]),len(v)))==choice['seed_source_vertex_ids'],
                  'Frozen source edge key unchanged')
        labels=np.load(c.plain(c.original.CACHE/'vertex_components.npy'),mmap_mode='r',allow_pickle=False)
        names={'order':'edge_occurrence_order','offsets':'edge_offsets','consistent':'consistent_edges',
               'protected_faces':'protected_faces','protected_vertices':'protected_vertices','locked_vertices':'locked_vertices',
               'global_sheet_labels':'global_sheet_labels','eligible_edge_ids':'eligible_edge_ids','face_degree':'face_degree'}
        context={name:np.load(c.plain(c.CENSUS/'diagnose'/(filename+'.npy')),mmap_mode='r',allow_pickle=False)
                 for name,filename in names.items()}
        prior=c.pin('guard_context_original_classifier',c.ROOT/'docs/compatibility/amd-brick-junction-protected-evidence/patch_contract.py',
                    '336e49bd7836f4a66c02ade831d6b37b66f290cb42e3aa30c72e7a009599d60c')
        context.update(check=check,classify=prior.classify)
        built=progress.build_contexts(f,keys,context['order'],context['offsets'],len(v),baseline,baseline_outer,check)
        record['baseline']['labels']=built['baseline_labels']
        record['progression']={name:built[name] for name in ('baseline_depth','baseline_labels','ring_records','terminal')}
        c.save(out/'progression.json',record['progression'])
        depths=[s['depth'] for s in built['snapshots']]
        c.require(depths==sorted(set(depths)) and all(depth>3 for depth in depths),'Distinct new complete-ring depths only')
        for snapshot in built['snapshots']:
            check();depth=int(snapshot['depth']);sub=out/('depth-%04d'%depth);sub.mkdir()
            face_ids=snapshot['source_face_ids'];outer=snapshot['outer_face_ids']
            c.require(0<len(face_ids)<=6000 and np.isin(baseline,face_ids).all()
                      and len(face_ids)>len(baseline) and np.isin(outer,face_ids).all(),'Whole original expanded context within hard cap')
            selection=dict(source_face_ids=face_ids,ring_added_face_ids=[outer])
            summary,arrays,local=q.qualify(v,f,rgba,keys,inc,labels,context,choice,selection)
            screening=progress.screen(summary)
            row=dict(depth=depth,labels=snapshot['labels'],relative_directory=sub.name,summary=summary,
                     local_classifier_summary=local,screening=screening)
            arrays['outer_face_ids']=np.asarray(outer,np.int32)
            files={}
            for name,a in arrays.items():
                check();path=sub/(name+'.npy')
                with path.open('xb') as stream:np.save(stream,a,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
                saved=np.load(path,mmap_mode='r',allow_pickle=False)
                c.require(a.shape==saved.shape and a.dtype==saved.dtype and np.array_equal(a,saved),'Exact private context readback: '+name)
                files[path.name]=dict(bytes=path.stat().st_size,sha256=c.digest(path));del saved
            c.save(sub/'result.json',dict(row,files=files))
            record['observations'].append(row);del arrays
        c.save(out/'observations.json',dict(baseline=record['baseline'],observations=record['observations']))
        selected=progress.select_first(record['observations']);record['selected_depth']=selected
        if selected is not None:
            chosen=next(row for row in record['observations'] if row['depth']==selected)
            selected_dir=out/chosen['relative_directory']
            arrays={name:np.load(c.plain(selected_dir/(name+'.npy')),mmap_mode='r',allow_pickle=False)
                    for name in ('vertices','faces','protected_faces','eligible_faces')}
            renderer=c.pin('guard_context_original_private_diagram',c.OLD/'patch_worker.py',
                           '136be90778c80c1a2b8369aad64aea38a83c4ce93e976fd4b108e2f2e1258fdb')
            renderer.diagram(np,out,arrays,chosen['summary']);record['diagram']='patch-protection.png'
        record.update(classification='PASS' if selected is not None else 'BLOCKED',
                      observations_completed=True,
                      reason='Earliest complete-ring context meets all frozen feasibility screens; no future reduction or quality guarantee' if selected is not None else
                             'No frozen context within6000faces meets all screens; stop and recommend a clean target-mesh approach, not another context or seed search',
                      source_correspondence_exact=True,no_reduction_field_query_renderer_or_export=True,
                      original_census_SHA256=c.digest(c.CENSUS/'diagnose/result.json'),
                      qualified_readback_SHA256=c.digest(c.RECOVERY/'verify/result.json'),
                      old_patch_SHA256=c.digest(old_result),files=manifest(),elapsed_seconds=time.monotonic()-began)
        check();c.save(out/'result.json',record)
    except BaseException as error:
        record.update(classification='BLOCKED',observations_completed=False,error=str(error),traceback=traceback.format_exc(),files=manifest(),elapsed_seconds=time.monotonic()-began)
        c.save(out/'result.json',record);print('GUARD_CONTEXT_BLOCKED '+str(error),flush=True)

if __name__=='__main__':main()
