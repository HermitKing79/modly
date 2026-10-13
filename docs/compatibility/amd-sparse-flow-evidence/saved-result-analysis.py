"""Passive saved-evidence checks; no model/framework imports or GPU queries."""
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
assert result['classification']=='PARTIAL' and result['observer_healthy'] is True and result['checkpoint_unchanged'] is True
assert w['error']=='Insufficient host strict-load margin' and w['load_calls']==w['forward_calls']==0
assert w['calls']==w['gpu_operation_counts']=={} and all(v==0 for v in w['transfers'].values())
assert p['gpu_authorized'] is False and p['returncode']==1 and p['timed_out'] is False
assert p['worker_exit_verified'] and p['job_cleanup']['owned_tree_exit_verified'] and not p['job_cleanup']['forced_job_cleanup']
assert p['job_attached_before_preparation'] and p['output_readers_completed'] and p['diagnostic_journal_stopped'] and p['windows_journal_stopped']
assert p['stage_timeouts_seconds']=={'cpu':180,'forward_release':240,'overall':480}
rows=[json.loads(line) for line in (ev/'run/trace.jsonl').read_text().splitlines()]
for row in rows:diagnostics.validate_record(row)
assert [(row['event'],row['name']) for row in rows]==[('stage','runtime-import'),('state','runtime')]
assert not (ev/'run/cpu-gate.json').exists() and (ev/'run/windows-periodic.jsonl').stat().st_size==0
assert contract.classify(w,False,True,True)=='PARTIAL'
before=json.loads((ev/'inventory-before.json').read_text());after=json.loads((ev/'inventory-after.json').read_text())
assert before==after
differences=[];raw={}
for stem in ('venv-metadata','model-cache-metadata'):
    paths=[ev/f'{stem}-{phase}.json.gz' for phase in ('before','after')]
    raw[stem]=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    a,b=[json.loads(gzip.decompress(path.read_bytes())) for path in paths]
    assert set(a)==set(b)
    for tree in a:
        left={r[0]:r for r in a[tree]};right={r[0]:r for r in b[tree]}
        assert set(left)==set(right)
        for path in left:
            if left[path]!=right[path]:
                x,y=left[path],right[path]
                assert x[0]==y[0] and x[2:]==y[2:] and stat.S_ISDIR(x[3]) and stat.S_ISDIR(y[3])
                differences.append({'tree':tree,'path':path,'before':x,'after':y,'reason':'directory st_size only; existing normalization policy'})
reconciliation=dict(inventory_equal=True,raw_sha256=raw,meaningful_changes=0,directory_size_changes=len(differences),differences=differences)
(ev/'metadata-reconciliation.json').open('x').write(json.dumps(reconciliation,indent=2))
git=json.loads((ev/'git-after.json').read_text())
assert all(git['preserved'].values()) and git['historical_local_refs_unchanged'] and git['historical_remote_refs_unchanged']
summary=dict(classification='PARTIAL',blocker=w['error'],checkpoint_loads=0,forward_calls=0,gpu_authorized=False,
    no_gpu_initialization_evidence='runtime/binding initialize=false; _lazy_init forbidden thereafter; no authorization/dispatch GPU ops',
    worker_exit_code=1,worker_seconds=p['elapsed_seconds'],no_timeout=True,owned_tree_exit_verified=True,forced_cleanup=False,
    observer_healthy=True,checkpoint_unchanged=True,inventory_equal=True,historical_evidence_files=len(git['historical_files']),
    meaningful_metadata_changes=0,raw_directory_size_changes=len(differences),saved_trace_valid=True,
    installed_environments_preserved=True,source_edits_preserved=True,prior_branches_preserved=True)
(ev/'saved-result-analysis.json').open('x').write(json.dumps(summary,indent=2));print(json.dumps(summary))
