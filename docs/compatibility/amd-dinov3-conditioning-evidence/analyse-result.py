"""Read-only evidence checks after the one attempt; no AI/GPU imports."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

p=Path(__file__).parent;root=p.parents[2]
sys.path.insert(0,str(root/'scripts/compatibility'))
import amd_dinov3_contract as contract
import amd_decoder_contract as strict

def save(name,value):
    with (p/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)

r=json.loads((p/'run/results.json').read_text());w=r['worker'];proc=r['process']
assert r['classification']=='PARTIAL' and r['observer_healthy'] and r['checkpoint_unchanged']
assert w['load_calls']==w['conditioning_calls']==1 and w['transfer_calls']=={'input':1,'output':1}
assert w['normalization_copies']=={'mean':1,'std':1}
contract.validate_counts(w['calls']);contract.validate_output(w['output'])
assert w['output']['requires_grad'] is False and w['output']['grad_enabled'] is False and w['output']['is_inference_tensor'] is True
assert w['cpu_load']['status']=='PASS' and w['cpu_load']['strict'] and w['cpu_load']['assign']
assert w['cpu_load']['missing_keys']==w['cpu_load']['unexpected_keys']==[] and not w['cpu_load']['gpu_initialized']
assert w['cpu_load']['loader_diagnostics']['entries']==415
before=json.loads((p/'run/checkpoint-before.json').read_text())
strict.validate_tensors(before['tensors'],w['cpu_load']['tensors'],'cpu')
assert before==json.loads((p/'run/checkpoint-after.json').read_text())
assert w['placed_buffers']=={'rope_embeddings.inv_freq':{'shape':[16],'dtype':'F32','device':'cuda:0','is_meta':False}}
assert w['gpu_operation_counts']['aten.conv2d.default']==1
assert w['gpu_operation_counts']['aten.scaled_dot_product_attention.default']==24
release=w['release']
assert release['allocated_bytes']==release['reserved_bytes']==79691776
assert release['model_collected'] and release['output_collected'] and release['rope_cache_entries_after']==0
assert w['status']=='FAIL' and w['stage']=='release' and w['error_type']=='ValueError'
try:strict.validate_release(release)
except ValueError:pass
else:raise AssertionError('Cleanup failure was incorrectly accepted')
assert proc['returncode']==1 and not proc['timed_out'] and proc['worker_exit_verified']
assert proc['job_cleanup']['owned_tree_exit_verified'] and proc['job_cleanup']['active_after_cleanup']==0
assert not proc['job_cleanup']['forced_job_cleanup']
assert all(proc[key] for key in ('cpu_observer_stopped','windows_periodic_stopped','output_readers_completed',
                               'diagnostic_journal_stopped','windows_journal_stopped'))
assert not any(proc.get(key) for key in ('error','cpu_observer_error','windows_periodic_error',
                                      'diagnostic_journal_error','windows_journal_error'))
trace=[json.loads(line) for line in (p/'run/trace.jsonl').read_text().splitlines()]
worker=[row for row in trace if row['origin']=='worker'];cpu=[row for row in trace if row['event']=='process']
assert [row['seq'] for row in worker]==list(range(1,len(worker)+1))
assert next(row for row in worker if row['name']=='gpu')['gpu']==w['gpu']
stack=[];operations=[]
for row in worker:
    if row['event']=='enter':stack.append(row)
    elif row['event']=='return':
        entry=stack.pop();assert entry['name']==row['name'] and row['completion']=='host-return-only'
        operations.append(dict(name=row['name'],entry=entry,returned=row))
    assert row['event']!='error'
assert not stack
windows=[json.loads(line) for line in (p/'run/windows-periodic.jsonl').read_text().splitlines()]
assert len(windows)==proc['windows_periodic_samples']==7 and all(row['healthy'] for row in windows)
assert next(row for row in worker if row['name']=='synchronize:after-conditioning' and row['event']=='return')
files=[]
for path in sorted((p/'run/scratch').rglob('*')):
    if path.is_file():
        value=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
        info=path.stat();files.append(dict(path=path.relative_to(p/'run/scratch').as_posix(),bytes=info.st_size,
                                         sha256=value.hexdigest(),mtime_ns=info.st_mtime_ns))
save('scratch-side-effects.json',dict(files=files,file_count=len(files),total_bytes=sum(row['bytes'] for row in files),
                                    policy='inert hashes only; no historical seed, cache execution or deletion'))
source=p/'run/scratch/input.png';assert hashlib.sha256(source.read_bytes()).hexdigest()==w['input']['png_sha256']
with (p/'input.png').open('xb') as stream:stream.write(source.read_bytes())
embedded=next(op for op in operations if op['name']=='module:embeddings')
start=embedded['entry']['monotonic_seconds'];end=embedded['returned']['monotonic_seconds']
active=[row for row in cpu if start<=row['monotonic_seconds']<=end]
observed_names=sorted({v['executable'] for row in cpu for v in row['processes']})
save('attempt-interpretation.json',dict(classification='PARTIAL',complete_conditioning_proved=True,cleanup_proved=False,
    unresolved_allocated_reserved_bytes=79691776,model_and_output_collected=True,coordinate_cache_cleared=True,
    owned_tree_exit_verified=True,observer_samples=7,trace_worker_rows=len(worker),operations=operations,
    embedding_interval_cpu_delta_seconds=sum(v['cpu_delta_seconds'] for row in active for v in row['processes']),
    observed_process_executables=observed_names,explicit_compiler_child_observed=False,
    compilation_interpretation='Fresh COMGR/MIOpen products and host CPU activity are consistent with first-use compilation; direct compiler attribution is not recorded.',
    cleanup_interpretation='Runtime workspace versus a surviving allocation is unresolved; no zero-memory threshold was waived.'))
a=json.loads((p/'inventory-before.json').read_text());b=json.loads((p/'inventory-after.json').read_text())
assert a==b
metadata={name:(p/(name+'-before.json.gz')).read_bytes()==(p/(name+'-after.json.gz')).read_bytes()
          for name in ('venv-metadata','model-cache-metadata')}
assert all(metadata.values())
git=json.loads((p/'git-after.json').read_text());assert all(git['preserved'].values())
assert git['historical_local_refs_unchanged'] and git['historical_remote_refs_unchanged']
save('preservation.json',dict(inventory_unchanged=True,metadata_byte_identical=metadata,historical_files=len(git['historical_files']),
    historical_files_unchanged=True,historical_local_refs_unchanged=True,historical_remote_refs_unchanged=True,
    amd_source_unchanged=True,main=git['local_refs']['main'],scratch_files_preserved=len(files)))
print(json.dumps({'validated_result':'PARTIAL','conditioning_seconds':w['conditioning_seconds'],
    'unresolved_release_bytes':79691776,'worker_exit_verified':True,'inventory_unchanged':True,
    'scratch_file_count':len(files),'scratch_bytes':sum(row['bytes'] for row in files)}))
