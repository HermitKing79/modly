"""Independent saved-result/container validation; stdlib only, no model/GPU work."""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

root=Path(__file__).resolve().parents[3];evidence=Path(__file__).parent
def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in ('torch','trellis2','triton','safetensors','numpy'):
        raise RuntimeError('Independent result validation uses stdlib only')
sys.addaudithook(audit)
def read(name):return json.loads((evidence/name).read_text())
def unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate container key: '+key)
        result[key]=value
    return result
r=read('run/results.json');w=r['worker'];p=r['process'];artifact=w['latent_checkpoint']
assert r['classification']=='PASS' and r['observer_healthy'] and r['checkpoint_unchanged']
assert w['status']=='PASS' and w['stage']=='complete' and w['load_calls']==1 and w['forward_calls']==22
assert w['sampling']['sequences']==1 and w['sampling']['started']==w['sampling']['completed']==22
assert (w['sampling']['positive_calls'],w['sampling']['negative_calls'])==(12,10)
assert w['transfers']==dict(x=1,t=22,cond=1,neg_cond=1,frequency=22,output=1)
assert w['workspace_clear_calls']==1 and w['release']['allocated_bytes']==w['release']['reserved_bytes']==0
assert w['release']['model_collected'] and w['release']['owned_tensors_collected']
assert p['returncode']==0 and not p['timed_out'] and p['worker_exit_verified']
assert p['job_cleanup']['owned_tree_exit_verified'] and not p['job_cleanup']['forced_job_cleanup']
assert p['stage_timeouts_seconds']==dict(cpu=180,forward_release=240,overall=480)
for key in ('cpu_observer_error','windows_periodic_error','diagnostic_journal_error','windows_journal_error'):
    assert p[key] is None
module_keys={'<root>',*[f'blocks.{i}' for i in range(30)],*[f'blocks.{i}.{a}' for i in range(30) for a in ('self_attn','cross_attn')]}
assert w['calls']==dict.fromkeys(module_keys,22)
models=w['sampling']['models'];assert len(models)==22
assert [m['role'] for m in models]==['positive','negative']*10+['positive']*2
for i,m in enumerate(models):assert m['index']==i+1 and m['frequency']==1 and m['modules']==dict.fromkeys(module_keys,1)
steps=w['sampling']['steps'];assert len(steps)==12 and [s['forwards'] for s in steps]==[2]*10+[1]*2
assert all(s['step']==i+1 and s['synchronized'] and s['all_finite'] for i,s in enumerate(steps))
trace=[json.loads(line) for line in (evidence/'run/trace.jsonl').read_text().splitlines()]
worker_trace=[row for row in trace if row['origin']=='worker'];assert len(worker_trace)==90<=192
assert [row['step'] for row in worker_trace if row['name']=='euler-step-complete']==steps
windows=[json.loads(line) for line in (evidence/'run/windows-periodic.jsonl').read_text().splitlines()]
assert len(windows)==p['windows_periodic_samples']==3 and all(row['healthy'] for row in windows)
gaps=[b['monotonic_seconds']-a['monotonic_seconds'] for a,b in zip(windows,windows[1:])];assert max(gaps)<=10
source=Path(artifact['path']);assert source.resolve()==evidence/'run/scratch/sparse-structure-latent.safetensors'
with source.open('rb') as f:blob=f.read(262145)
assert len(blob)==artifact['file_bytes']==134688<=262144
file_sha=hashlib.sha256(blob).hexdigest();assert file_sha==artifact['file_sha256']
header_size=struct.unpack('<Q',blob[:8])[0];assert 0<header_size<=65536 and 8+header_size+131072==len(blob)
header=json.loads(blob[8:8+header_size],object_pairs_hook=unique)
assert set(header)=={'__metadata__','sparse_latent'}
assert header['sparse_latent']==dict(dtype='F32',shape=[1,8,16,16,16],data_offsets=[0,131072])
metadata=header['__metadata__'];assert metadata==artifact['provenance']
payload=blob[8+header_size:];payload_sha=hashlib.sha256(payload).hexdigest()
assert payload_sha==metadata['output_sha256']==artifact['output_sha256']==w['output']['sha256']
values=struct.unpack('<32768f',payload);assert all(math.isfinite(v) for v in values)
mean=math.fsum(values)/len(values);std=math.sqrt(math.fsum((v-mean)**2 for v in values)/len(values))
assert min(values)==w['output']['minimum'] and max(values)==w['output']['maximum']
assert abs(mean-w['output']['mean'])<1e-6 and abs(std-w['output']['stddev'])<1e-6
assert list(values[:8])==w['output']['representative_values']
assert metadata['harness_commit']=='98d1d29f750236580c7853091dfd949a077b531a'
assert metadata['sigma_min']=='1e-05' and metadata['seed']=='20261013'
assert json.loads(metadata['params'])==dict(steps=12,rescale_t=5.,guidance_strength=7.5,guidance_rescale=.7,guidance_interval=[.6,1.],dino_lock=0.)
preflight=read('resume-preflight.json')
assert metadata['model_checkpoint_sha256']==preflight['checkpoint_sha256'] and metadata['model_config_sha256']==preflight['config_sha256']
assert metadata['feature_file_sha256']==preflight['feature']['file_sha256'] and json.loads(metadata['feature_provenance'])==preflight['feature']['provenance']
assert json.loads(metadata['sampler_sources'])==preflight['sampler_sources']['sampler_sources']
assert metadata['noise_sha256']==w['input']['x']['sha256'] and json.loads(metadata['runtime'])==w['runtime']
target=evidence/'sparse-structure-latent.safetensors'
with target.open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
assert hashlib.sha256(target.read_bytes()).hexdigest()==file_sha
process_rows=[row for row in trace if row['origin']=='supervisor' and row['event']=='process']
children=[entry for row in process_rows for entry in row['processes'] if entry['pid']!=p['pid']]
gpu_start=next(row for row in worker_trace if row['event']=='stage' and row['name']=='GPU-metadata')['monotonic_seconds']
gpu_end=worker_trace[-1]['monotonic_seconds']
summary=dict(classification='PASS',independent_validation='stdlib header, raw F32 values, payload/file hashes and source provenance',
    original_path=str(source.resolve()),published_copy_path=str(target.resolve()),file_bytes=len(blob),file_sha256=file_sha,
    payload_bytes=len(payload),payload_sha256=payload_sha,shape=[1,8,16,16,16],dtype='F32',all_finite=True,
    provenance=metadata,completed_steps=12,completed_forwards=22,positive_forwards=12,negative_forwards=10,
    cpu_loading_seconds=w['cpu_load']['elapsed_seconds'],synchronized_sampling_seconds=w['forward_seconds'],
    gpu_stage_seconds=gpu_end-gpu_start,overall_worker_seconds=p['elapsed_seconds'],
    peak_allocated_bytes=w['after_forward']['peak_allocated_bytes'],peak_reserved_bytes=w['after_forward']['peak_reserved_bytes'],
    release=w['release'],process_exit_status=p['returncode'],owned_tree_exit_verified=True,forced_cleanup=False,
    windows_periodic_samples=len(windows),maximum_windows_gap_seconds=max(gaps),worker_trace_events=len(worker_trace),
    process_activity_samples=len(process_rows),observed_child_processes=children,
    first_step_seconds=steps[0]['elapsed_seconds'],remaining_step_seconds=[s['elapsed_seconds'] for s in steps[1:]],
    independent_statistics=dict(minimum=min(values),maximum=max(values),mean=mean,stddev=std),
    torch_imported='torch' in sys.modules,additional_model_execution=False)
with (evidence/'resume-result-analysis.json').open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2)
print(json.dumps({k:v for k,v in summary.items() if k not in ('provenance','release','observed_child_processes','remaining_step_seconds')}))
