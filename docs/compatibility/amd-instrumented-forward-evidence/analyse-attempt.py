"""Summarize stopped evidence as inert data; reproduce only a pure Python error."""
import hashlib
import json
from pathlib import Path
import sys

p=Path(__file__).parent;root=p.parents[2]
sys.path.insert(0,str(root/'scripts/compatibility'))
from amd_forward_diagnostics import Trace
from amd_forward_cache_reuse import inspect_comgr

result=json.loads((p/'run/results.json').read_text());worker=result['worker']
assert result['classification']=='PARTIAL' and worker['root_decoder_calls']==worker['forward_calls']==0
assert worker['transfer_calls']=={'input':0,'output':0} and worker['stage']=='GPU-metadata'
lines=[]
try:Trace(lines.append).emit('state','gpu',**worker['gpu'])
except TypeError as exc:reproduction={'error_type':type(exc).__name__,'error':str(exc),'emitted_lines':len(lines)}
else:raise AssertionError('Expected diagnostic argument collision')
assert reproduction['error']==worker['error'] and not lines
plan=json.loads((p/'cache-plan.json').read_text());selected={r['path']:r for r in plan['selected']}
scratch=p/'run/scratch';rows=[]
for path in sorted(scratch.rglob('*')):
    if path.is_file():
        name=path.relative_to(scratch).as_posix();data=path.read_bytes();sha=hashlib.sha256(data).hexdigest()
        row={'path':name,'bytes':len(data),'sha256':sha,'mtime_ns':path.stat().st_mtime_ns,
             'copied_seed':name in selected,'same_seed_bytes':sha==selected[name]['sha256'] if name in selected else None}
        if name.startswith('amd_comgr_cache_dir/llvmcache-'):row['inert_container_inspection']=inspect_comgr(data)
        rows.append(row)
with (p/'scratch-side-effects.json').open('x',encoding='utf-8') as stream:json.dump({'file_count':len(rows),'total_bytes':sum(r['bytes'] for r in rows),'files':rows},stream,indent=2)
interpretation={'classification':'PARTIAL','harness_commit':result['process']['harness_commit'],
    'failed_operation':"trace.emit('state','gpu',**record['gpu'])",'python_only_reproduction':reproduction,
    'native_computation_failure_demonstrated':False,'timeout_cause_tested':False,
    'placement_reached':False,'forward_calls':0,'input_output_transfers':worker['transfer_calls'],
    'gpu_memory_measurements_available':False,'allocator_release_measurements_available':False,
    'observer_healthy':result['observer_healthy'],'job_cleanup':result['process']['job_cleanup'],
    'no_retry_or_postfailure_gpu_query':True,'torch_imported_by_analysis':'torch' in sys.modules,
    'scratch_files':len(rows),'new_files':sum(not r['copied_seed'] for r in rows),
    'seed_files_byte_identical':all(r['same_seed_bytes'] for r in rows if r['copied_seed'])}
with (p/'attempt-interpretation.json').open('x',encoding='utf-8') as stream:json.dump(interpretation,stream,indent=2)
print(json.dumps(interpretation))
