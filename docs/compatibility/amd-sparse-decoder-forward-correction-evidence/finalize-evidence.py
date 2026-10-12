"""Interpret immutable run logs and preservation snapshots without GPU calls."""
import hashlib
import json
from pathlib import Path

p=Path(__file__).parent
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,value):
    with (p/name).open('x',encoding='utf-8') as stream: json.dump(value,stream,indent=2)
result=json.loads((p/'run/results.json').read_text())
cpu=json.loads((p/'run/cpu-gate.json').read_text())['gate_one']
lines=(p/'run/worker.stdout.txt').read_text().splitlines()
stages=[line.split('=',1)[1] for line in lines if line.startswith('FORWARD_STAGE=')]
assert result['classification']=='PARTIAL' and result['process']['timed_out'] is True
assert result['process']['worker_exit_verified'] is True and result['observer_healthy'] is True
assert cpu['status']=='PASS' and len(cpu['tensors'])==74 and cpu['missing_keys']==[] and cpu['unexpected_keys']==[]
assert stages.count('single-forward')==1 and not any(line.startswith('FORWARD_RESULT=') for line in lines)
save('attempt-interpretation.json',{
    'classification':'PARTIAL','harness_commit':json.loads((p/'attempt-started.json').read_text())['harness_commit'],
    'failure_boundary':'30-second authorized GPU-stage deadline; last durable stage single-forward',
    'supervisor_error':result['process']['error'],'process_exit':result['process']['returncode'],
    'forced_tree_cleanup':True,'worker_exit_verified':True,'observer_healthy':True,
    'guard_exception_captured':False,'native_exception_captured':False,'stderr_bytes':(p/'run/worker.stderr.txt').stat().st_size,
    'cpu_gate':{'strict':True,'entries':74,'checkpoint_load_seconds':cpu['elapsed_seconds'],'gpu_initialized':False,
                'memory_before':cpu['memory_before'],'memory_after':cpu['memory_after']},
    'stages':stages,'control_flow_inference':'Reaching single-forward requires model placement, input copy, synchronize and input metadata checks to have completed',
    'root_forward_call_count':None,'output':None,'gpu_memory_measurements':None,'worker_release_measurements':None,
    'missing_final_result':'Raw results.json retains its default NOT ATTEMPTED worker placeholder when no final message arrives; durable stages and CPU gate establish that an attempt occurred',
    'retry_performed':False})
scratch=p/'run/scratch'
files=[{'path':file.relative_to(scratch).as_posix(),'bytes':file.stat().st_size,'sha256':digest(file)}
       for file in sorted(scratch.rglob('*')) if file.is_file()]
save('scratch-side-effects.json',{'location':str(scratch.resolve()),'files':files,'file_count':len(files),
                                'total_bytes':sum(row['bytes'] for row in files),'published_binary_caches':False})
pairs={}
for stem,suffix in (('inventory','.json'),('venv-metadata','.json.gz'),('model-cache-metadata','.json.gz')):
    before=p/(stem+'-before'+suffix);after=p/(stem+'-after'+suffix)
    pairs[stem]={'before_sha256':digest(before),'after_sha256':digest(after),'byte_identical':before.read_bytes()==after.read_bytes()}
assert all(row['byte_identical'] for row in pairs.values())
git=json.loads((p/'git-after-attempt.json').read_text())
save('preservation.json',{'pairs':pairs,'historical_local_refs_unchanged':git['historical_local_refs_unchanged'],
    'historical_remote_refs_unchanged':git['historical_remote_refs_unchanged'],**git['preserved_fields'],
    'checkpoint_before_after_byte_identical':(p/'run/checkpoint-before.json').read_bytes()==(p/'run/checkpoint-after.json').read_bytes()})
print(json.dumps({'classification':'PARTIAL','preserved':True,'scratch_files':len(files),'scratch_bytes':sum(row['bytes'] for row in files)}))
