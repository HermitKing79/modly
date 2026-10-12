"""CPU-only verification/copy of this completed attempt; never runs a worker."""
import hashlib
import importlib.abc
import json
from pathlib import Path
import struct
import sys

class NoAI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'trellis2', 'transformers'}:
            raise RuntimeError('AI imports prohibited in saved-evidence analysis: '+fullname)

sys.meta_path.insert(0, NoAI())
def audit(event, args):
    if event.startswith('socket.') or event in {'subprocess.Popen', 'os.system'}:
        raise RuntimeError('Network/subprocess prohibited in saved-evidence analysis')
sys.addaudithook(audit)

evidence=Path(__file__).resolve().parent
root=evidence.parents[2]
sys.path.insert(0, str(root/'scripts/compatibility'))
import amd_feature_checkpoint as feature
import amd_dinov3_allocation as allocation
import numpy as np
from safetensors.numpy import load

def digest(blob): return hashlib.sha256(blob).hexdigest()
def read(name): return json.loads((evidence/name).read_text())
def exclusive_copy(source, target):
    blob=source.read_bytes()
    with target.open('xb') as stream: stream.write(blob)
    assert target.read_bytes()==blob
    return digest(blob)

results=read('run/results.json'); worker=results['worker']; process=results['process']
assert results['classification']=='PARTIAL' and results['observer_healthy'] and results['checkpoint_unchanged']
assert worker['load_calls']==1 and worker['conditioning_calls']==1
assert worker['calls'] and all(count==1 for count in worker['calls'].values())
assert worker['output']['shape']==[1,1029,1024] and worker['output']['all_finite']
assert process['returncode']==1 and not process['timed_out'] and process['worker_exit_verified']
assert process['job_cleanup']['owned_tree_exit_verified'] and process['job_cleanup']['active_after_cleanup']==0
assert not process['job_cleanup']['forced_job_cleanup']

saved=worker['feature_checkpoint']; provenance={key:saved['provenance'][key] for key in feature.FIELDS}
scratch=evidence/'run/scratch'; path=scratch/'conditioning.safetensors'
verified=feature.verify(path,scratch,provenance,worker['output']['sha256'])
assert verified['file_sha256']==saved['file_sha256'] and verified['file_bytes']==saved['file_bytes']
blob=path.read_bytes(); tensor=load(blob)['cond']
assert feature.array_hash(tensor)==worker['output']['sha256']
header_bytes=struct.unpack('<Q',blob[:8])[0]
assert exclusive_copy(path,evidence/'conditioning.safetensors')==saved['file_sha256']
assert exclusive_copy(scratch/'input.png',evidence/'input.png')==worker['input']['png_sha256']

snapshot_path=scratch/'allocator-release.json'; snapshot_blob=snapshot_path.read_bytes()
diagnosis=worker['allocation_diagnosis']; snapshot=json.loads(snapshot_blob)
assert digest(snapshot_blob)==diagnosis['snapshot_sha256'] and len(snapshot_blob)==diagnosis['snapshot_bytes']
summary=allocation.summarize(snapshot['snapshot'],worker['release']['allocated_bytes'],
                             worker['release']['reserved_bytes'],snapshot['observed_python_owners'])
assert summary==snapshot['summary']==diagnosis['summary'] and diagnosis['history_disabled']
assert exclusive_copy(snapshot_path,evidence/'allocator-release.json')==diagnosis['snapshot_sha256']

windows=[json.loads(line) for line in (evidence/'run/windows-periodic.jsonl').read_text().splitlines()]
assert len(windows)==process['windows_periodic_samples'] and all(row['healthy'] for row in windows)
gaps=[b['monotonic_seconds']-a['monotonic_seconds'] for a,b in zip(windows,windows[1:])]
assert max(gaps,default=0)<=10
for prefix in ('venv-metadata','model-cache-metadata'):
    assert (evidence/(prefix+'-before.json.gz')).read_bytes()==(evidence/(prefix+'-after.json.gz')).read_bytes()
before=read('inventory-before.json'); after=read('inventory-after.json')
assert {k:v for k,v in before.items() if k!='phase'}=={k:v for k,v in after.items() if k!='phase'}
git=read('git-after.json')
assert all(git['preserved'].values()) and git['historical_local_refs_unchanged'] and git['historical_remote_refs_unchanged']

manifest=[]
for file in sorted(scratch.rglob('*')):
    if file.is_file():
        data=file.read_bytes()
        manifest.append({'path':file.relative_to(scratch).as_posix(),'bytes':len(data),'sha256':digest(data)})
with (evidence/'scratch-side-effects.json').open('x',encoding='utf-8') as stream:
    json.dump({'files':manifest,'file_count':len(manifest),'total_bytes':sum(row['bytes'] for row in manifest),
               'executed_during_analysis':False},stream,indent=2)
record={'classification':'PARTIAL','conditioning_verified':True,'single_load_and_forward':True,
        'feature_cpu_readback':verified,'header_bytes':header_bytes,
        'recomputed_statistics':{'minimum':float(tensor.min()),'maximum':float(tensor.max()),
          'mean_float64':float(tensor.mean(dtype=np.float64)),'stddev_float64':float(tensor.std(dtype=np.float64)),
          'count':int(tensor.size),'all_finite':bool(np.isfinite(tensor).all())},
        'snapshot_reconciled':True,'active_allocated_bytes':summary['active_allocated_bytes'],
        'active_blocks':len(summary['active_blocks']),'observed_surviving_python_owners':summary['observed_python_owner_count'],
        'observed_tensor_records':diagnosis['observed_tensor_records'],'workspace_settings':snapshot['workspace_settings'],
        'native_cpp_stacks_supported':False,'workspace_ownership_proved':False,
        'worker_exit_verified':True,'windows_samples':len(windows),'maximum_windows_gap_seconds':max(gaps,default=0),
        'inventory_unchanged':True,'metadata_byte_identical':True,'historical_files':len(git['historical_files']),
        'historical_files_unchanged':True,'historical_branches_unchanged':True,'amd_source_unchanged':True,
        'main':git['local_refs']['main'],'scratch_files_preserved':len(manifest),
        'ai_imports':[name for name in sys.modules if name.split('.')[0] in {'torch','triton','trellis2','transformers'}]}
assert not record['ai_imports']
with (evidence/'saved-evidence-analysis.json').open('x',encoding='utf-8') as stream: json.dump(record,stream,indent=2)
print(json.dumps(record,indent=2))
