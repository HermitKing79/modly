"""Subset colour mapping and separately gated interior position holdout."""
import sys
sys.dont_write_bytecode=True
import numpy as np

def need(ok,text):
    if not ok:raise ValueError(text)

def interpolate(faces,qfaces,bary,ids,encoded,metrics):
    need(ids.dtype==np.int32 and ids.ndim==1 and 0<len(ids)<=28455 and np.all(ids[1:]>ids[:-1]),'Sorted unique original vertex IDs')
    need(faces.dtype==np.int32 and faces.ndim==2 and faces.shape[1]==3 and qfaces.dtype==np.int32 and qfaces.ndim==1,'Original I32 triangle/query structures')
    need(0<len(qfaces)<=65536 and qfaces.min()>=0 and qfaces.max()<len(faces),'Bounded original face IDs')
    need(encoded.dtype==np.float32 and encoded.shape==(len(ids),3),'Sampled F32 RGB at original unique vertices')
    original=faces[qfaces];local=np.searchsorted(ids,original)
    need(np.all(local<len(ids)) and np.array_equal(ids[np.minimum(local,len(ids)-1)],original),'Every original triangle corner has its own sampled colour row')
    return metrics.color_prediction(encoded,local.astype(np.int32),np.arange(len(qfaces),dtype=np.int32),bary)

def evaluate_holdout(predicted,raw,qfaces,kinds,groups,zero):
    n=len(raw)
    need(raw.dtype==np.float32 and raw.shape==(n,6) and 0<n<=2048,'Bounded original F32 six-channel holdout')
    need(predicted.dtype==np.float64 and predicted.shape==(n,3) and np.isfinite(predicted).all() and np.isfinite(raw).all() and ((predicted>=0)&(predicted<=1)).all(),'Finite holdout and encoded prediction')
    need(qfaces.dtype==np.int32 and qfaces.shape==(n,) and np.all(qfaces>=0),'Original holdout face IDs')
    need(kinds.dtype==np.int8 and kinds.shape==(n,) and np.all((kinds==4)|(kinds==5)),'Only true holdout interiors')
    need(zero.dtype==np.bool_ and zero.shape==(n,),'Explicit holdout support')
    target=np.clip(raw.astype(np.float64),0,1);covered=np.zeros(n,np.bool_)
    factors=dict(metallic=.00011082693761409814,roughness=.999995986777544,alpha=1.)
    def stats(error):
        return dict(rmse=float(np.sqrt(np.mean(error**2))),p95_abs=float(np.percentile(np.abs(error),95)),max_abs=float(np.abs(error).max()))
    def measured(index):
        need(len(index)>0,'No empty required interior cohort')
        rgb=stats(predicted[index]-target[index,:3]);rgb['pass']=rgb['rmse']<=.02 and rgb['p95_abs']<=.05 and rgb['max_abs']<=.15
        scalars={}
        for channel,(name,value) in enumerate(factors.items()):
            row=stats(value-target[index,channel+3]);row['pass']=row['rmse']<=.02;scalars[name]=row
        return dict(samples=len(index),RGB=rgb,scalars=scalars,zero_support=int(zero[index].sum()),pass_gate=bool(rgb['pass'] and all(s['pass'] for s in scalars.values()) and not zero[index].any()))
    result={}
    for name,g in groups.items():
        a,b=g['query_start'],g['query_stop'];need(type(a) is int and type(b) is int and 0<=a<b<=n and not covered[a:b].any(),'Disjoint complete holdout ranges')
        covered[a:b]=True;index=np.arange(a,b)
        row={part:measured(which) for part,which in dict(all=index,interior0=index[kinds[index]==4],interior1=index[kinds[index]==5]).items()}
        row['pass']=all(x['pass_gate'] for x in row.values());result[name]=row
    need(result and covered.all(),'Every holdout query is required')
    failures=[n for n,r in result.items() if not r['pass']]
    return dict(material_suitability='NO-GO' if failures else 'PASS',regions=result,failing_regions=failures,RGB_limits=dict(rmse=.02,p95_abs=.05,max_abs=.15),scalar_RMSE_limit=.02,fixed_factors=factors,zero_support=int(zero.sum()),no_colour_or_factor_fit=True)
