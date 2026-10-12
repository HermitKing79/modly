"""Passive preservation checks; no AI imports, checkpoint loading or GPU calls."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[3];evidence=Path(__file__).parent
source=Path('D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD')
branch='codex/amd-forward-timeout-investigation'
def git(*args,cwd=root):return subprocess.run(['git',*args],cwd=cwd,capture_output=True,check=True,timeout=15).stdout.decode().strip()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
local=dict(line.split(' ',1) for line in git('for-each-ref','--format=%(refname:short) %(objectname)','refs/heads').splitlines())
remote={line.split()[1].removeprefix('refs/heads/'):line.split()[0] for line in git('ls-remote','--heads','origin').splitlines()}
old={}
for directory in ('amd-sparse-decoder-forward-evidence','amd-sparse-decoder-forward-correction-evidence'):
    for path in sorted((root/'docs/compatibility'/directory).rglob('*')):
        if path.is_file():old[path.relative_to(root).as_posix()]=digest(path)
for name in ('amd-sparse-decoder-forward.md','amd-sparse-decoder-forward-guard-correction.md'):
    old['docs/compatibility/'+name]=digest(root/'docs/compatibility'/name)
record={'head':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),'origin':git('remote','get-url','origin'),
        'local_refs':local,'remote_refs':remote,'historical_files':old,'amd_source_head':git('rev-parse','HEAD',cwd=source),
        'amd_source_status':git('status','--short','--untracked-files=all',cwd=source),
        'amd_modified_files':{name:digest(source/name) for name in ('nodes.py','nodes.py.before-profiler-fix')}}
if sys.argv[1]=='after':
    before=json.loads((evidence/'git-before.json').read_text())
    record['preserved']={key:record[key]==before[key] for key in ('historical_files','amd_source_head','amd_source_status','amd_modified_files')}
    record['historical_local_refs_unchanged']=all(local.get(key)==value for key,value in before['local_refs'].items() if key!=branch)
    record['historical_remote_refs_unchanged']=all(remote.get(key)==value for key,value in before['remote_refs'].items() if key!=branch)
    assert all(record['preserved'].values()) and record['historical_local_refs_unchanged'] and record['historical_remote_refs_unchanged']
with (evidence/f'git-{sys.argv[1]}.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
print(json.dumps({key:value for key,value in record.items() if key not in ('local_refs','remote_refs','historical_files','amd_modified_files')}))
