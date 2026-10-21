"""Fresh independent full-file verification and unchanged saved-query gates."""
import sys
sys.dont_write_bytecode=True
import json,time,traceback
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from inputs import ROOT,P,pinned,load,load_auth,plain,digest,require,save,source_hashes

def main():
    require(sys.stdin.buffer.readline(64)==b'GO\n','Exact owned binary GO')
    auth=load_auth();auth.P=P;sys.addaudithook(auth.runtime_guard)
    import numpy as np
    from verify_stream import verify
    began=time.monotonic();out=P/'verify';stage='independent_input_authentication'
    try:
        data,identities,watched=load(np);directory=P/'export';receipt=plain(directory/'result.json');r=json.loads(receipt.read_text())
        require(r['classification']=='PASS' and r['source_rows_sampled']==1957891 and r['zero_support']==0 and len(r['sampling_batches'])==60,'Complete actual resampling before independent export verification')
        require(r['sampling_batches'][-1]['rows']==24579,'Authentic last batch')
        for name,row in r['files'].items():
            require(Path(name).name==name,'Single exact output basename');p=plain(directory/name)
            require(p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],'Original export artifact identity: '+name);watched.append(p)
        watched.append(receipt);before={str(p):digest(p) for p in watched}
        colours=np.load(plain(directory/'linear-COLOR0.npy'),mmap_mode='r',allow_pickle=False)
        counts=np.load(plain(directory/'positive-support.npy'),mmap_mode='r',allow_pickle=False);weights=np.load(plain(directory/'raw-weight-sum.npy'),mmap_mode='r',allow_pickle=False)
        require(colours.dtype==np.float32 and colours.shape==(1957891,3) and counts.dtype==np.uint8 and counts.shape==weights.shape==(1957891,) and weights.dtype==np.float32,'Full persisted original-row colour/support tables')
        for start in range(0,len(colours),32768):require(np.all((counts[start:start+32768]>=1)&(counts[start:start+32768]<=8)) and np.isfinite(weights[start:start+32768]).all() and np.all(weights[start:start+32768]>0),'Complete persisted positive source support')
        stage='independent_entire_GLB';stamp=time.monotonic();source=source_hashes(identities,digest(directory/'linear-COLOR0.npy'))
        require(source==r['source_hashes'],'Original complete source lineage')
        parsed=verify(plain(directory/'brick-source-sampled-colour.glb'),data['vertices'],data['faces'],colours,source);read_seconds=time.monotonic()-stamp
        require(parsed['linear_COLOR0'].shape==(1957891,3),'Entire original vertex colour table from GLB')
        stage='saved_material_queries';metrics=pinned('full_GLB_original_independent_metrics',ROOT/'docs/compatibility/amd-foliage-colour-glb-evidence/verify_glb.py','82ff500153f29a7e1412a88c8ce0ab580278616b23d64a5708f5b2a4dda16ce9')
        old=data['prior'];held=data['resampled'];qf,kinds=old['query-faces'],old['query-kinds'];groups={}
        for name,g in data['frozen']['cohorts'].items():
            index=np.arange(g['query_start'],g['query_stop']);groups.update({name+':'+part:x for part,x in dict(all=index,corners=index[kinds[index]<3],centroids=index[kinds[index]==3],interiors=index[kinds[index]>=4]).items()})
        require(len(groups)==64 and len(qf)==56910 and np.all(old['raw_weight_sum']>0),'All original fixed source targets')
        first=metrics.evaluate_queries(parsed['linear_COLOR0'],data['faces'],qf,old['barycentrics'],old['source_spatial'],groups,held['corrected_original_prediction'])
        first.update(scope='All 56,910 original saved queries; no new source sampling or visual-equivalence claim',query_count=len(qf))
        groups={};kinds=held['holdout-query-kinds']
        for name,g in data['holdout_frozen']['cohorts'].items():
            index=np.arange(g['query_start'],g['query_stop']);groups.update({name+':'+part:x for part,x in dict(all=index,interior0=index[kinds[index]==4],interior1=index[kinds[index]==5]).items()})
        require(len(groups)==48 and len(held['holdout-query-faces'])==2048 and np.all(held['holdout_weight_sum']>0),'All frozen interior holdout targets')
        second=metrics.evaluate_queries(parsed['linear_COLOR0'],data['faces'],held['holdout-query-faces'],held['holdout-barycentrics'],held['holdout_source_spatial'],groups,held['holdout_prediction'])
        second.update(scope='All 2,048 frozen interior position holdouts; no new query population or visual-equivalence claim',query_count=len(held['holdout-query-faces']))
        require(first['all_gates_pass'] and second['all_gates_pass'],'Every unchanged original/holdout material gate')
        require(before=={str(p):digest(p) for p in watched},'Original sources/proofs and exported files unchanged')
        save(out/'result.json',dict(classification='PASS',scope='Fresh CPU independent complete GLB transport and all112 material gates',container=parsed['report'],original_queries=first,holdout_queries=second,GLB_SHA256=digest(directory/'brick-source-sampled-colour.glb'),GLB_bytes=(directory/'brick-source-sampled-colour.glb').stat().st_size,full_F32_colour_SHA256=digest(directory/'linear-COLOR0.npy'),source_hashes=source,all_watched_sources_unchanged=True,no_new_sampling_or_candidate_or_Blender=True,GLB_readback_seconds=read_seconds,elapsed_seconds=time.monotonic()-began))
        print('INDEPENDENT_FULL_SOURCE_SAMPLED_GLB_PASS',flush=True)
    except BaseException as error:
        if not (out/'result.json').exists():save(out/'result.json',dict(classification='NO-GO',stage=stage,error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-began,retry=False))
        raise

if __name__=='__main__':main()
