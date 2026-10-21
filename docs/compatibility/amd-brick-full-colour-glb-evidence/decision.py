"""Unchanged per-region RGB gates and fixed scalar factors, without fitting."""
import sys
sys.dont_write_bytecode=True
import numpy as np
RGB_LIMITS=dict(rmse=.02,p95_abs=.05,max_abs=.15)
FACTORS=dict(metallic=.00011082693761409814,roughness=.999995986777544,alpha=1.)


def evaluate(predicted,raw,query_faces,kinds,groups,zero_support):
    n=len(raw)
    def need(ok,text):
        if not ok:raise ValueError(text)
    need(isinstance(raw,np.ndarray) and raw.dtype==np.float32 and raw.shape==(n,6) and 0<n<=65536,'Bounded F32 source samples')
    need(isinstance(predicted,np.ndarray) and predicted.dtype==np.float64 and predicted.shape==(n,3),'Encoded F64 predictions')
    need(np.isfinite(raw).all() and np.isfinite(predicted).all() and ((predicted>=0)&(predicted<=1)).all(),'Finite original data and unit encoded prediction')
    need(query_faces.dtype==np.int32 and query_faces.shape==(n,) and query_faces.min()>=0,'Original face IDs')
    need(kinds.dtype==np.int8 and kinds.shape==(n,) and ((kinds>=0)&(kinds<6)).all(),'Declared corner/centroid/interior kinds')
    need(zero_support.dtype==np.bool_ and zero_support.shape==(n,),'Explicit reference support')
    target=np.clip(raw.astype(np.float64),0,1);covered=np.zeros(n,np.bool_)
    def stats(error):
        return dict(rmse=float(np.sqrt(np.mean(error**2))),p95_abs=float(np.percentile(np.abs(error),95)),max_abs=float(np.abs(error).max()))
    def measured(index):
        if not len(index):return dict(samples=0,classification='BLOCKED_EMPTY_COHORT',pass_gate=False)
        error=predicted[index]-target[index,:3];rgb=stats(error)
        rgb['pass']=all(rgb[key]<=value for key,value in RGB_LIMITS.items())
        row,col=np.unravel_index(np.argmax(np.abs(error)),error.shape);worst=int(index[row])
        scalars={}
        for channel,(name,value) in enumerate(FACTORS.items()):
            scalars[name]=stats(value-target[index,channel+3]);scalars[name]['pass']=scalars[name]['rmse']<=.02
        valid=rgb['pass'] and all(x['pass'] for x in scalars.values()) and not zero_support[index].any()
        return dict(samples=len(index),RGB=rgb,scalars=scalars,zero_support=int(zero_support[index].sum()),worst_RGB=dict(query_index=worst,original_face_id=int(query_faces[worst]),kind=int(kinds[worst]),channel=int(col),absolute_error=rgb['max_abs']),pass_gate=bool(valid))
    regions={}
    for name,group in groups.items():
        start,stop=group['query_start'],group['query_stop']
        need(type(start) is int and type(stop) is int and 0<=start<=stop<=n and not covered[start:stop].any(),'Disjoint declared region ranges')
        covered[start:stop]=True;index=np.arange(start,stop)
        parts={'all':index,'corners':index[kinds[index]<3],'centroids':index[kinds[index]==3],'interiors':index[kinds[index]>=4]}
        result={part:measured(ids) for part,ids in parts.items()}
        result['pass']=all(row['pass_gate'] for row in result.values())
        regions[name]=result
    need(covered.all() and regions,'Every query belongs to a declared region')
    failures=[name for name,row in regions.items() if not row['pass']]
    return dict(material_suitability='PASS' if not failures else 'BLOCKED',regions=regions,failing_regions=failures,zero_support=int(zero_support.sum()),RGB_limits=RGB_LIMITS,scalar_RMSE_limit=.02,fixed_factors=FACTORS,no_colour_or_factor_fit=True)
