"""Passive validation of one completed attempt; no GPU/AI imports or execution."""
import hashlib
import gzip
import json
from pathlib import Path
import sys
import stat

evidence=Path(__file__).resolve().parent;root=evidence.parents[2]
sys.path.insert(0,str(root/'scripts/compatibility'))
import amd_blaslt_teardown as candidate
import amd_forward_diagnostics as diagnostics
def read(name):return json.loads((evidence/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
r=read('run/results.json');w=r['worker'];p=r['process']
assert candidate.classify(w,True,r['observer_healthy'],r['checkpoint_unchanged'])=='PASS'
assert r['classification']=='PASS' and p['returncode']==0 and not p['timed_out']
assert p['job_attached_before_preparation'] and p['worker_exit_verified']
assert p['job_cleanup']['owned_tree_exit_verified'] and p['job_cleanup']['active_after_cleanup']==0
assert not p['job_cleanup']['forced_job_cleanup']
assert all(p[name] is None for name in ('cpu_observer_error','windows_periodic_error','diagnostic_journal_error','windows_journal_error'))
rows=[json.loads(line,object_pairs_hook=diagnostics.unique_fields) for line in (evidence/'run/trace.jsonl').read_text().splitlines()]
for row in rows:diagnostics.validate_record(row)
clear=[row for row in rows if row['name']=='blas:clear-workspaces']
assert [row['event'] for row in clear]==['enter','return']
assert w['release']['allocated_bytes']==w['release']['reserved_bytes']==0
assert w['release']['model_collected'] and w['release']['output_collected'] and w['release']['rope_cache_entries_after']==0
assert w['allocation_diagnosis']['history_disabled'] and not w['allocation_diagnosis']['snapshot_captured']
assert w['allocation_diagnosis']['observed_python_owner_count']==0
scratch=evidence/'run/scratch'
assert not (scratch/'allocator-release.json').exists() and not list(scratch.rglob('*.safetensors'))
assert 'feature_checkpoint' not in w
features=root/'docs/compatibility/amd-dinov3-allocation-evidence/conditioning.safetensors'
assert sha(features)=='3f444d42191df07415a8877a3d8d3706cf1b129bedbbb2dfcc0221d5571be56a'
assert sha(scratch/'input.png')==w['input']['png_sha256']=='9abc7b8f89e5b001e863fc3ef2fbd6813b2e83f63c0a6b0a45203c2ef1edfd8d'
before=read('inventory-before.json');after=read('inventory-after.json');assert before==after
metadata=[]
for name in ('venv-metadata','model-cache-metadata'):
    old_blob=(evidence/(name+'-before.json.gz')).read_bytes();new_blob=(evidence/(name+'-after.json.gz')).read_bytes()
    old=json.loads(gzip.decompress(old_blob));new=json.loads(gzip.decompress(new_blob));assert set(old)==set(new)
    changes={}
    for tree in old:
        a={row[0]:row for row in old[tree]};b={row[0]:row for row in new[tree]};assert set(a)==set(b)
        changed=[]
        for path in sorted(a):
            if a[path]!=b[path]:
                # Existing inventory policy excludes directory allocation size.
                # No other field or file entry is permitted to differ.
                assert stat.S_ISDIR(a[path][3]) and stat.S_ISDIR(b[path][3])
                assert a[path][0]==b[path][0] and a[path][2:]==b[path][2:] and a[path][1]!=b[path][1]
                changed.append({'path':path,'before':a[path],'after':b[path],'changed_field':'directory st_size only'})
        changes[tree]=changed
    metadata.append({'name':name,'byte_identical':old_blob==new_blob,'before_sha256':hashlib.sha256(old_blob).hexdigest(),
        'after_sha256':hashlib.sha256(new_blob).hexdigest(),'all_file_metadata_unchanged':True,
        'all_path_mtime_mode_metadata_unchanged':True,'directory_size_changes':changes,
        'difference_count':sum(len(rows) for rows in changes.values())})
with (evidence/'metadata-reconciliation.json').open('x',encoding='utf-8') as stream:json.dump(metadata,stream,indent=2)
git=read('git-after.json')
assert all(git['preserved'].values()) and git['historical_local_refs_unchanged'] and git['historical_remote_refs_unchanged']
windows=[json.loads(line) for line in (evidence/'run/windows-periodic.jsonl').read_text().splitlines()]
assert len(windows)==p['windows_periodic_samples'] and all(row['healthy'] for row in windows)
gaps=[b['monotonic_seconds']-a['monotonic_seconds'] for a,b in zip(windows,windows[1:])];assert max(gaps,default=0)<=10
files=[]
for file in sorted(scratch.rglob('*')):
    if file.is_file():files.append({'path':file.relative_to(scratch).as_posix(),'bytes':file.stat().st_size,'sha256':sha(file)})
with (evidence/'scratch-side-effects.json').open('x',encoding='utf-8') as stream:
    json.dump({'file_count':len(files),'total_bytes':sum(file['bytes'] for file in files),'files':files,'executed_during_analysis':False},stream,indent=2)
result={'classification':'PASS','strict_loads':w['load_calls'],'conditioning_calls':w['conditioning_calls'],
    'workspace_clear_calls':w['workspace_clear_calls'],'clear_trace_events':[row['event'] for row in clear],
    'output':w['output'],'release':w['release'],'allocation_diagnosis':w['allocation_diagnosis'],
    'snapshot_captured':False,'new_feature_checkpoint_written':False,'saved_feature_preserved_sha256':sha(features),
    'worker_returncode':p['returncode'],'owned_tree_exit_verified':True,'windows_samples':len(windows),
    'maximum_windows_gap_seconds':max(gaps,default=0),'inventories_unchanged':True,
    'compressed_metadata_byte_identical':{row['name']:row['byte_identical'] for row in metadata},
    'raw_metadata_difference_counts':{row['name']:row['difference_count'] for row in metadata},
    'all_file_metadata_unchanged':True,'directory_size_only_differences_reconciled':True,
    'historical_files_preserved':len(git['historical_files']),'historical_branches_preserved':True,
    'amd_source_preserved':True,'main':git['local_refs']['main'],'scratch_file_count':len(files),
    'scratch_bytes':sum(file['bytes'] for file in files),'post_exit_vram_measured':False,
    'ai_imports':[name for name in sys.modules if name.split('.')[0] in ('torch','triton','trellis2','transformers')]}
assert not result['ai_imports']
with (evidence/'saved-result-analysis.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result))
