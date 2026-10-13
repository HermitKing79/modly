"""Structural comparison of saved passive snapshots; no framework imports."""
import gzip
import hashlib
import json
from pathlib import Path
import stat

evidence=Path(__file__).parent
before=json.loads((evidence/'inventory-before.json').read_text())
after=json.loads((evidence/'inventory-after.json').read_text());assert before==after
rows={}
for stem in ('venv-metadata','model-cache-metadata'):
    a=(evidence/(stem+'-before.json.gz')).read_bytes();b=(evidence/(stem+'-after.json.gz')).read_bytes()
    old=json.loads(gzip.decompress(a));new=json.loads(gzip.decompress(b));assert set(old)==set(new)
    changes={}
    for tree in old:
        aa={r[0]:r for r in old[tree]};bb={r[0]:r for r in new[tree]};assert set(aa)==set(bb)
        changes[tree]=[]
        for name in aa:
            if aa[name]!=bb[name]:
                assert stat.S_ISDIR(aa[name][3]) and stat.S_ISDIR(bb[name][3])
                assert aa[name][2:]==bb[name][2:] and aa[name][1]!=bb[name][1]
                changes[tree].append(dict(path=name,before=aa[name],after=bb[name],field='directory allocation size only'))
    rows[stem]=dict(before_sha256=hashlib.sha256(a).hexdigest(),after_sha256=hashlib.sha256(b).hexdigest(),
        byte_identical=a==b,structurally_equal=old==new,all_regular_files_unchanged=True,directory_size_changes=changes)
git=json.loads((evidence/'git-after.json').read_text())
assert all(git['preserved'].values()) and git['historical_local_refs_unchanged'] and git['historical_remote_refs_unchanged']
old_git=json.loads((evidence/'git-before.json').read_text())
row=dict(inventories_equal=True,raw_metadata=rows,historical_files_unchanged=len(old_git['historical_files']),
    source_and_git_preserved=git['preserved'],historical_local_refs_unchanged=True,historical_remote_refs_unchanged=True,
    no_sampling_worker=True,attempt_marker_exists=(evidence/'attempt-started.json').exists(),
    run_directory_exists=(evidence/'run').exists())
assert not row['attempt_marker_exists'] and not row['run_directory_exists']
with (evidence/'preservation-summary.json').open('x',encoding='utf-8') as f:json.dump(row,f,indent=2)
print(json.dumps(row))
