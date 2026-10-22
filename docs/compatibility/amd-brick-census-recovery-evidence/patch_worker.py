"""One reviewed positive-only original patch qualification; never simplify."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
import common as c
import json,time,traceback,os

def diagram(np,out,arrays,summary):
    from PIL import Image,ImageDraw
    v,f=arrays['vertices'],arrays['faces'];axes=np.argsort(-np.ptp(v.astype(np.float64),axis=0),kind='stable')[:2]
    points=v[:,axes].astype(np.float64);low=points.min(0);span=float(np.ptp(points,axis=0).max());c.require(span>0,'Nonzero patch diagram span')
    xy=(points-low)*(700/span);image=Image.new('RGB',(820,820),'white');draw=ImageDraw.Draw(image)
    for mask,colour in ((np.ones(len(f),np.bool_),'#dddddd'),(arrays['protected_faces'],'#777777'),(arrays['eligible_faces'],'#228b22')):
        for tri in f[mask]:
            line=[(45+float(x),760-float(y)) for x,y in xy[tri]];draw.line(line+[line[0]],fill=colour,width=1)
    draw.text((12,8),'Original patch diagnostic; grey=protected, green=eligible, pale=context',fill='black')
    draw.text((12,26),'World '+('XYZ'[int(axes[0])])+' right / '+('XYZ'[int(axes[1])])+' up; no shaded visibility or occlusion claim',fill='black')
    draw.text((12,44),'Source seed edge row%d; original faces%d; no simplification'%(summary['seed_edge_id'],len(f)),fill='black')
    path=out/'patch-protection.png'
    with path.open('xb') as stream:image.save(stream,format='PNG');stream.flush();os.fsync(stream.fileno())
    with Image.open(path) as reopened:reopened.load();c.require(reopened.format=='PNG' and reopened.mode=='RGB' and reopened.size==image.size,'Private diagnostic PNG readback')

def main():
    c.begin();began=time.monotonic();out=c.P/'patch';record=dict(classification='BLOCKED',actual_simplification_calls=0,source='accepted original only')
    def check():
        if time.monotonic()-began>150:raise TimeoutError('150-second patch work budget; leave30seconds for receipts/cleanup')
    def write(name,a):
        check()
        with (out/(name+'.npy')).open('xb') as stream:np.save(stream,a,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
        saved=np.load(out/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        c.require(a.shape==saved.shape and a.dtype==saved.dtype and np.array_equal(a,saved),'Exact private patch readback: '+name)
    try:
        import numpy as np
        import patch_contract as pc
        c.authenticate();verified=json.loads(c.plain(c.P/'verify/result.json').read_text());host=json.loads(c.plain(c.P/'verify-host.json').read_text())
        c.require(c.digest(c.P/'verify/result.json')=='cd11a416d1d3a54187530e9990b838373af1d73bae2cc5ee2d375c7e6e7ba221'
                  and verified['classification']=='PASS' and host['classification']=='PASS'
                  and host['worker_receipt_sha256']==c.digest(c.P/'verify/result.json'),'Exact owned corrected census verification PASS')
        parent=c.census_receipt();v,f,rgba,keys,inc=c.load(np);check()
        labels=np.load(c.plain(c.original.CACHE/'vertex_components.npy'),mmap_mode='r',allow_pickle=False)
        names={'order':'edge_occurrence_order','offsets':'edge_offsets','consistent':'consistent_edges','protected_faces':'protected_faces',
               'protected_vertices':'protected_vertices','locked_vertices':'locked_vertices','global_sheet_labels':'global_sheet_labels',
               'eligible_edge_ids':'eligible_edge_ids','face_degree':'face_degree'}
        context={name:np.load(c.plain(c.CENSUS/'diagnose'/(filename+'.npy')),mmap_mode='r',allow_pickle=False) for name,filename in names.items()}
        c.require(len(context['eligible_edge_ids'])==verified['eligible_edges']>0,'Verified nonempty population')
        prior=c.pin('recovery_original_local_patch_classifier',c.ROOT/'docs/compatibility/amd-brick-junction-protected-evidence/patch_contract.py','336e49bd7836f4a66c02ade831d6b37b66f290cb42e3aa30c72e7a009599d60c')
        context.update(check=check,classify=prior.classify)
        choice=pc.choose_component(f,keys,context['order'],context['offsets'],context['eligible_edge_ids'],check)
        record['choice']=dict(eligible_face_components=choice['eligible_component_count'],selected_component_faces=len(choice['component_face_ids']),
                              selected_component_min_source_face=int(choice['component_face_ids'].min()),selected_component_edges=len(choice['component_edge_ids']),
                              lower_median_rank=(len(choice['component_edge_ids'])-1)//2,seed_edge_id=choice['seed_edge_id'],
                              seed_source_vertex_ids=list(divmod(int(keys[choice['seed_edge_id']]),len(v))))
        c.save(out/'frozen-choice.json',record['choice']);check()
        selection=pc.extract(f,keys,context['order'],context['offsets'],choice['seed_edge_id'],len(v),check)
        summary,arrays,local=pc.qualify(v,f,rgba,keys,inc,labels,context,choice,selection)
        for name,a in arrays.items():write(name,a)
        details=dict(choice=record['choice'],seed_face_ids=selection['seed_face_ids'].tolist(),seed_edge_face_ids=selection['seed_edge_face_ids'].tolist(),
                     ring_added_face_ids=[a.tolist() for a in selection['ring_added_face_ids']],local_classifier_summary=local,
                     complete_star_certificate='Unique selected original faces plus equality to verified global face_degree proves the stored original endpoint stars complete; full links use original source face rows, not cropped-only adjacency.')
        c.save(out/'selection.json',details)
        if summary['qualified']:diagram(np,out,arrays,summary)
        record.update(classification='PASS' if summary['qualified'] else 'BLOCKED',summary=summary,
                      reason='One initial protected topological patch opportunity; no achievable reduction/geometry/material guarantee' if summary['qualified'] else 'Selected fixed patch has no strict eligible interior or no non-touching opportunity; no alternative selection',
                      verified_census_SHA256=c.CENSUS_SHA,qualified_readback_SHA256=c.digest(c.P/'verify/result.json'),
                      source_correspondence_exact=True,no_reduction_field_query_renderer_or_export=True,
                      projection_known_buffers_bytes=2*1024**3,projection_total_admission_bytes=5*1024**3,projection_is_not_measured=True,
                      files={p.name:dict(bytes=p.stat().st_size,sha256=c.digest(p)) for p in sorted(out.iterdir()) if p.suffix in {'.json','.npy','.png'}},elapsed_seconds=time.monotonic()-began)
        c.save(out/'result.json',record)
    except BaseException as error:
        record.update(classification='BLOCKED',error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-began)
        record['files']={p.name:dict(bytes=p.stat().st_size,sha256=c.digest(p)) for p in sorted(out.iterdir()) if p.suffix in {'.json','.npy','.png'}}
        c.save(out/'result.json',record);print('ONE_PATCH_BLOCKED '+str(error),flush=True)

if __name__=='__main__':main()
