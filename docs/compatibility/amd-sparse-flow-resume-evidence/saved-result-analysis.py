"""Passive verification of one completed forward and its unresolved transfer guard."""
import gzip
import hashlib
import json
from pathlib import Path
import stat
import sys
root=Path(__file__).resolve().parents[3];ev=Path(__file__).parent
sys.path.insert(0,str(root/'scripts/compatibility'))
import amd_forward_diagnostics as diagnostics
import amd_sparse_flow_contract as contract
result=json.loads((ev/'run/results.json').read_text());w=result['worker'];p=result['process']
assert result['classification']=='PARTIAL' and result['observer_healthy'] and result['checkpoint_unchanged']
assert w['status']=='PASS' and w['stage']=='complete' and w['load_calls']==w['forward_calls']==w['workspace_clear_calls']==1
assert w['transfers']=={'x':1,'t':1,'cond':1,'output':1,'frequency':0}
assert w['gpu_operation_counts']['aten.to.dtype_layout']==1
contract.validate_output(w['output'])
assert w['release']['allocated_bytes']==w['release']['reserved_bytes']==0
assert w['release']['model_collected'] and w['release']['owned_tensors_collected']
expected={'<root>':1,**{f'blocks.{i}':1 for i in range(30)},**{f'blocks.{i}.{a}':1 for i in range(30) for a in ('self_attn','cross_attn')}}
assert w['calls']==expected
gate=w['cpu_load'];assert gate['payload_entries']==640 and gate['reversible_widening_count']==220 and gate['strict'] and gate['assign']
assert len(gate['loader_diagnostics']['payload_sha256'])==640 and len(gate['loader_diagnostics']['runtime_fp32_widened'])==220
assert p['returncode']==0 and not p['timed_out'] and p['gpu_authorized']
assert p['job_attached_before_preparation'] and p['job_cleanup']['owned_tree_exit_verified'] and not p['job_cleanup']['forced_job_cleanup']
assert p['worker_exit_verified'] and p['output_readers_completed']
for name in ('cpu_observer','windows_periodic','diagnostic_journal','windows_journal'):
    assert p[name+'_stopped'] and not p.get(name+'_error')
assert p['stage_timeouts_seconds']=={'cpu':180,'forward_release':240,'overall':480}
assert w['budget']['allocator_limit_bytes']==12*1024**3
assert w['before_placement']['available_bytes']>=13*1024**3
assert w['release']['peak_allocated_bytes']<=12*1024**3 and w['release']['peak_reserved_bytes']<=12*1024**3
rows=[json.loads(line) for line in (ev/'run/trace.jsonl').read_text().splitlines()]
for row in rows:diagnostics.validate_record(row)
worker=[r for r in rows if r['origin']=='worker'];assert [r['seq'] for r in worker]==list(range(1,235))
for name in ['module:<root>','synchronize:after-forward','synchronize:before-workspace-clear','blas:clear-workspaces']:
    assert [r['event'] for r in worker if r['name']==name]==['enter','return']
root_return=next(r for r in worker if r['name']=='module:<root>' and r['event']=='return')
sync_return=next(r for r in worker if r['name']=='synchronize:after-forward' and r['event']=='return')
assert sync_return['seq']>root_return['seq'] and sync_return['monotonic_seconds']>=root_return['monotonic_seconds']
windows=[json.loads(line) for line in (ev/'run/windows-periodic.jsonl').read_text().splitlines()]
assert len(windows)==p['windows_periodic_samples']==2 and all(r['healthy'] and r['query_seconds']<=4.5 for r in windows)
max_gap=max(b['monotonic_seconds']-a['monotonic_seconds'] for a,b in zip(windows,windows[1:]));assert max_gap<=10
before=json.loads((ev/'inventory-before.json').read_text());after=json.loads((ev/'inventory-after.json').read_text());assert before==after
differences=[];raw={}
for stem in ('venv-metadata','model-cache-metadata'):
    paths=[ev/f'{stem}-{phase}.json.gz' for phase in ('before','after')]
    raw[stem]=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    a,b=[json.loads(gzip.decompress(path.read_bytes())) for path in paths];assert set(a)==set(b)
    for tree in a:
        left={r[0]:r for r in a[tree]};right={r[0]:r for r in b[tree]};assert set(left)==set(right)
        for path in left:
            if left[path]!=right[path]:
                x,y=left[path],right[path]
                assert x[0]==y[0] and x[2:]==y[2:] and stat.S_ISDIR(x[3]) and stat.S_ISDIR(y[3])
                differences.append({'tree':tree,'path':path,'before':x,'after':y,'reason':'directory st_size only; existing normalization policy'})
(ev/'metadata-reconciliation.json').open('x').write(json.dumps(dict(inventory_equal=True,raw_sha256=raw,meaningful_changes=0,differences=differences),indent=2))
git=json.loads((ev/'git-after.json').read_text());assert all(git['preserved'].values()) and git['historical_local_refs_unchanged'] and git['historical_remote_refs_unchanged']
activities=[r for r in rows if r['event']=='process']
summary=dict(classification='PARTIAL',model_computation='PASS',unresolved='frequency transfer opcode/accounting guard',
    checkpoint_loads=1,forward_calls=1,all640_payloads_strict=True,fp32_widenings=220,output=w['output'],
    forward_seconds=w['forward_seconds'],cpu_load_seconds=gate['elapsed_seconds'],placement_seconds=w['placement_seconds'],worker_seconds=p['elapsed_seconds'],
    release=w['release'],observer_healthy=True,windows_samples=2,max_windows_gap_seconds=max_gap,
    worker_trace_rows=len(worker),cpu_activity_samples=len(activities),observed_executables=sorted({q['executable'] for r in activities for q in r['processes']}),
    no_forced_cleanup=True,exit_code=0,installed_environments_preserved=True,historical_evidence_files=len(git['historical_files']),
    raw_directory_size_changes=len(differences),raw_results_unchanged=True)
(ev/'saved-result-analysis.json').open('x').write(json.dumps(summary,indent=2))
scratch=ev/'run/scratch';files=[]
for path in sorted(scratch.rglob('*')):
    if path.is_file():files.append(dict(path=path.relative_to(scratch).as_posix(),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(ev/'scratch-side-effects.json').open('x').write(json.dumps(dict(files=files,count=len(files),bytes=sum(r['bytes'] for r in files),executed_during_analysis=False),indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ('output','release')}))
