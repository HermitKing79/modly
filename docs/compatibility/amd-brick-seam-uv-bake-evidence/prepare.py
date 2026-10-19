"""Deterministically replay the exact pinned selector, then record its overlap.

The published selector consumes the genuine owned-host GO and original input
readers. A line trace stops it immediately before its first UV call; no reader,
stdin token, selection arithmetic, learned field or native operation is mocked.
"""
import sys,ast,time
from pathlib import Path
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from common import *

class SelectionRecovered(Exception):pass

def main():
    started=time.monotonic()
    # Loading these pinned modules imports no scientific library before GO.
    original=load('published_selector',OLD/'bake_patch.py',PINS['bake_patch.py']);original.P=P
    tree=ast.parse((OLD/'bake_patch.py').read_text())
    sites=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='uv' for t in n.targets) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='chart']
    if len(sites)!=1 or sys.gettrace() is not None:raise ValueError('Unique pinned capture site and no existing trace required')
    captured={}
    def trace(frame,event,arg):
        if frame.f_code is not original.main.__code__:return None
        if event=='line' and frame.f_lineno==sites[0]:
            if captured:raise ValueError('Duplicate selector capture')
            data=frame.f_locals
            for k in ('selected','original_ids','vertices','faces','direct'):captured[k]=data[k].copy()
            captured.update(before=data['before'],auth=data['auth'],true_ray=data['true_ray'],roi=data['roi'].tolist(),projected_bounds=[data['screen'][data['original_ids']].min(0).tolist(),data['screen'][data['original_ids']].max(0).tolist()])
            raise SelectionRecovered()
        return trace
    try:
        sys.settrace(trace)
        try:original.main()
        except SelectionRecovered:pass
    finally:sys.settrace(None)
    if not captured or sys.gettrace() is not None:raise ValueError('Authenticated selector capture did not settle')
    import numpy as np
    math=originals();v,f=captured['vertices'],captured['faces'];uv=math.chart(v,f)
    witness={}
    def overlap_trace(frame,event,arg):
        if frame.f_code is not math.assert_injective.__code__:return None
        if event=='exception' and isinstance(arg[1],ValueError) and str(arg[1])=='Analytic positive-area UV overlap':
            q=frame.f_locals
            witness.update(face_i=int(q['i']),face_j=int(q['j']),UV_intersection_area=float(q['area']),UV_intersection_polygon=np.asarray(q['poly']).tolist())
        return overlap_trace
    try:
        sys.settrace(overlap_trace)
        try:math.assert_injective(uv,f)
        except ValueError as exc:
            if str(exc)!='Analytic positive-area UV overlap':raise
    finally:sys.settrace(None)
    if not witness:raise ValueError('Original overlap failure did not reproduce')
    witness['original_triangle_ids']=[int(captured['selected'][witness['face_i']]),int(captured['selected'][witness['face_j']])]
    auth=captured['auth']
    if identity_inputs(auth)!=captured['before']:raise ValueError('Original inputs changed during replay')
    out=P/'prepare';data={'vertices':v,'faces':f,'direct':captured['direct'],'source-face-ids':captured['selected'],'source-vertex-ids':captured['original_ids'],'planar-uv':uv}
    for name,array in data.items():np_save(out/(name+'.npy'),array)
    record=dict(classification='PASS',operation='deterministic replay of pinned published selector; no earlier saved patch ID list exists',selector_sha256=PINS['bake_patch.py'],capture_line=sites[0],capture_count=1,
                trace_removed=True,actual_texture_sampling_calls=0,original_triangle_count=len(f),original_vertex_count=len(v),source_row_order_preserved=True,overlap=witness,
                nomination_roi=captured['roi'],projected_bounds=captured['projected_bounds'],true_nearest_original_ray=captured['true_ray'],
                inputs_unchanged=True,input_identities=captured['before'],elapsed_seconds=time.monotonic()-started,files=manifest(out,[n+'.npy' for n in data]))
    save(out/'patch.json',record);print('PATCH_REPLAY_PASS triangles='+str(len(f))+' overlap='+str(witness['UV_intersection_area']),flush=True)

if __name__=='__main__':main()
