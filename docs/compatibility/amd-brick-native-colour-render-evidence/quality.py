"""Original112 source-quality rows with native byte decoding, no new limits."""
import sys
sys.dont_write_bytecode=True
def decode(byte_rgb):
    import numpy as np
    if byte_rgb.dtype!=np.uint8 or byte_rgb.ndim!=2 or byte_rgb.shape[1]!=3:raise ValueError('RGB byte matrix')
    v=byte_rgb.astype(np.float64)/255
    return np.where(v<=.04045,v/12.92,((v+.055)/1.055)**2.4).astype('<f4')
def cohorts(data):
    import numpy as np
    groups={};old=data['prior'];kinds=old['query-kinds']
    for name,g in data['frozen']['cohorts'].items():
        index=np.arange(g['query_start'],g['query_stop']);groups.update({name+':'+part:x for part,x in dict(all=index,corners=index[kinds[index]<3],centroids=index[kinds[index]==3],interiors=index[kinds[index]>=4]).items()})
    held={};kinds=data['resampled']['holdout-query-kinds']
    for name,g in data['holdout_frozen']['cohorts'].items():
        index=np.arange(g['query_start'],g['query_stop']);held.update({name+':'+part:x for part,x in dict(all=index,interior0=index[kinds[index]==4],interior1=index[kinds[index]==5]).items()})
    if len(groups)!=64 or len(held)!=48:raise ValueError('Every original112 cohort')
    return groups,held
def evaluate(data,colours,metrics):
    groups,held=cohorts(data);old=data['prior'];h=data['resampled']
    first=metrics.evaluate_queries(colours,data['faces'],old['query-faces'],old['barycentrics'],old['source_spatial'],groups)
    second=metrics.evaluate_queries(colours,data['faces'],h['holdout-query-faces'],h['holdout-barycentrics'],h['holdout_source_spatial'],held)
    return first,second
