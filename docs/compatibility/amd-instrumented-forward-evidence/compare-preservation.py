"""Compare captured passive inventories; write only new workspace evidence."""
import hashlib
import json
from pathlib import Path

p=Path(__file__).parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
rows={}
for stem,suffix in (('inventory','.json'),('venv-metadata','.json.gz'),('model-cache-metadata','.json.gz')):
    before=p/(stem+'-before'+suffix);after=p/(stem+'-after'+suffix)
    rows[stem]={'before_sha256':sha(before),'after_sha256':sha(after),'byte_identical':before.read_bytes()==after.read_bytes()}
assert all(row['byte_identical'] for row in rows.values())
git=json.loads((p/'git-after.json').read_text())
record={'one_authorized_attempt_no_retry':True,'inventory_comparison':rows,
        'historical_files_and_amd_source':git['preserved'],
        'historical_local_refs_unchanged':git['historical_local_refs_unchanged'],
        'historical_remote_refs_unchanged':git['historical_remote_refs_unchanged'],
        'historical_cache_artifacts_executed_or_modified':False}
with (p/'preservation.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record))
