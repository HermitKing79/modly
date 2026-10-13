"""CPU stdlib verification of the three saved results and bounded F32 containers."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

evidence=Path(__file__).parent
commit='aeb5173bda5b414ce99274ec4764799968d4f56c'
def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in ('torch','triton','trellis2','cumesh','flex_gemm','o_voxel'):
        raise RuntimeError('Independent saved-result validation prohibits AI imports')
sys.addaudithook(audit)
def unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON field')
        result[key]=value
    return result
def digest(blob):return hashlib.sha256(blob).hexdigest()
def container(stage,name,key,shape,limit,row):
    path=evidence/stage/'run/scratch'/name
    with path.open('rb') as f:blob=f.read(limit+1)
    assert len(blob)<=limit and len(blob)==row['file_bytes'] and digest(blob)==row['file_sha256']
    header_length=struct.unpack('<Q',blob[:8])[0];assert 0<header_length<=65536
    header=json.loads(blob[8:8+header_length],object_pairs_hook=unique)
    payload=blob[8+header_length:]
    assert header=={'__metadata__':row['provenance'],key:{'dtype':'F32','shape':shape,'data_offsets':[0,4*math.prod(shape)]}}
    assert len(payload)==4*math.prod(shape) and digest(payload)==row['output_sha256']
    assert all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',payload))
    target=evidence/name
    with target.open('xb') as f:f.write(blob)
    assert target.read_bytes()==blob
    return dict(path=str(target.resolve()),bytes=len(blob),sha256=digest(blob),payload_sha256=digest(payload),shape=shape,dtype='F32',all_finite=True,byte_identical_to_owned_scratch=True)

results={};summary={}
for stage in ('dino','sampling','decoder'):
    result=json.loads((evidence/stage/'run/results.json').read_text());results[stage]=result
    w=result['worker'];p=result['process']
    assert result['classification']=='PASS' and result['observer_healthy'] and result['checkpoint_unchanged']
    assert w['status']=='PASS' and w['stage']=='complete' and p['returncode']==0 and not p['timed_out']
    assert p['worker_exit_verified'] and p['job_attached_before_preparation'] and p['job_cleanup']['owned_tree_exit_verified']
    assert not p['job_cleanup']['forced_job_cleanup'] and p['harness_commit']==commit
    assert w['release']['allocated_bytes']==w['release']['reserved_bytes']==0
    assert w['release']['peak_allocated_bytes']<=12*1024**3 and w['release']['peak_reserved_bytes']<=12*1024**3
    windows=[json.loads(line,object_pairs_hook=unique) for line in (evidence/stage/'run/windows-periodic.jsonl').read_text().splitlines()]
    assert windows and all(r['healthy'] for r in windows)
    gap=max((b['monotonic_seconds']-a['monotonic_seconds'] for a,b in zip(windows,windows[1:])),default=0)
    assert gap<=10
    summary[stage]=dict(supervised_seconds=p['elapsed_seconds'],cpu_loading_seconds=w['cpu_load']['elapsed_seconds'],
        synchronized_compute_seconds=w['conditioning_seconds'] if stage=='dino' else w['forward_seconds'] if stage=='sampling' else w['forward']['elapsed_seconds'],
        peak_allocated_bytes=w['release']['peak_allocated_bytes'],peak_reserved_bytes=w['release']['peak_reserved_bytes'],
        available_before_placement_bytes=w['before_placement']['available_bytes'],available_after_release_bytes=w['release']['available_bytes'],
        allocated_reserved_after_release=[0,0],windows_periodic_samples=len(windows),maximum_windows_gap_seconds=gap,process_exit_status=0,owned_tree_exit_verified=True)

dino=results['dino']['worker'];sampling=results['sampling']['worker'];decoder=results['decoder']['worker']
assert dino['load_calls']==dino['conditioning_calls']==1 and dino['workspace_clear_calls']==1
assert dino['calls']['extractor']==dino['calls']['features']==1 and all(dino['calls']['layer.'+str(i)]==1 for i in range(24))
assert sampling['sampling']['sequences']==1 and sampling['sampling']['started']==sampling['sampling']['completed']==22
assert len(sampling['sampling']['steps'])==12 and all(step['all_finite'] and step['synchronized'] for step in sampling['sampling']['steps'])
assert sampling['sampling']['positive_calls']==12 and sampling['sampling']['negative_calls']==10
assert len(sampling['sampling']['models'])==22 and all(v['frequency']==1 for v in sampling['sampling']['models'])
assert decoder['load_calls']==decoder['forward_calls']==decoder['root_decoder_calls']==1 and decoder['workspace_clear_calls']==1
pins=json.loads((evidence/'input-contract.json').read_text());feature=dino['feature_checkpoint'];latent=sampling['latent_checkpoint']
assert feature['provenance']['input_pixel_sha256']==pins['pixel_sha256'] and feature['provenance']['input_png_sha256']==pins['input_png_sha256']
assert feature['provenance']['harness_commit']==latent['provenance']['harness_commit']==commit
assert latent['provenance']['feature_file_sha256']==feature['file_sha256'] and json.loads(latent['provenance']['feature_provenance'])==feature['provenance']
assert latent['file_sha256']==decoder['saved_latent']['file_sha256']
old=json.loads((evidence.parents[0]/'amd-sparse-structure-sampling-evidence/run/results.json').read_text())['worker']
assert sampling['input']['x']['sha256']==old['input']['x']['sha256']==latent['provenance']['noise_sha256']
assert latent['provenance']['seed']=='20261013' and json.loads(latent['provenance']['params'])==json.loads(old['latent_checkpoint']['provenance']['params'])
artifacts=dict(conditioning=container('dino','conditioning.safetensors','cond',[1,1029,1024],4231168,feature),
               latent=container('sampling','sparse-structure-latent.safetensors','sparse_latent',[1,8,16,16,16],262144,latent))
for stage in ('dino','sampling','decoder'):
    scratch=evidence/stage/'run/scratch'
    rows=[dict(path=p.relative_to(scratch).as_posix(),bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in sorted(scratch.rglob('*')) if p.is_file()]
    with (evidence/stage/'scratch-manifest.json').open('x') as f:json.dump(dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),entries=rows),f,indent=2)
row=dict(classification='PASS',strict_containers_verified=True,unchanged_noise_and_sampler=True,exact_forwards=dict(dino=1,sampling=22,decoder=1),
    completed_steps=12,artifacts=artifacts,stages=summary,torch_imported='torch' in sys.modules,additional_gpu_execution=False)
assert not row['torch_imported']
with (evidence/'independent-sequence-analysis.json').open('x') as f:json.dump(row,f,indent=2)
print(json.dumps(row))
