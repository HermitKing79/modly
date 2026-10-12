"""Read-only Git/history/source checks for this correction; no AI imports."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[3]
evidence=Path(__file__).parent
branch='codex/amd-sparse-decoder-forward-guard-correction'
base='0f82092515849041ee7df313fef5d2bad2a7ab98'
source=Path('D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD')
def git(*args, cwd=root):
    return subprocess.run(['git',*args],cwd=cwd,capture_output=True,check=True,timeout=15).stdout.decode().strip()
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
local=dict(line.split(' ',1) for line in git('for-each-ref','--format=%(refname:short) %(objectname)','refs/heads').splitlines())
remote={line.split()[1].removeprefix('refs/heads/'):line.split()[0] for line in git('ls-remote','--heads','origin').splitlines()}
first=root/'docs/compatibility/amd-sparse-decoder-forward-evidence'
old={path.relative_to(root).as_posix():digest(path) for path in sorted(first.rglob('*')) if path.is_file()}
old['docs/compatibility/amd-sparse-decoder-forward.md']=digest(root/'docs/compatibility/amd-sparse-decoder-forward.md')
tracked=git('ls-files','docs/compatibility/amd-sparse-decoder-forward-evidence','docs/compatibility/amd-sparse-decoder-forward.md').splitlines()
for name in tracked:
    committed=subprocess.run(['git','show',base+':'+name],cwd=root,capture_output=True,check=True,timeout=5).stdout
    if (root/name).read_bytes().replace(b'\r\n',b'\n')!=committed.replace(b'\r\n',b'\n'):
        raise RuntimeError('First attempt evidence changed: '+name)
result={'branch':git('branch','--show-current'),'head':git('rev-parse','HEAD'),'origin':git('remote','get-url','origin'),
        'local_refs':local,'remote_refs':remote,'first_attempt_files':old,'first_attempt_matches_base':True,
        'amd_source_head':git('rev-parse','HEAD',cwd=source),'amd_source_status':git('status','--short','--untracked-files=all',cwd=source),
        'amd_modified_files':{name:digest(source/name) for name in ('nodes.py','nodes.py.before-profiler-fix')}}
if sys.argv[1]!='before':
    before=json.loads((evidence/'git-before.json').read_text())
    fields=('first_attempt_files','amd_source_head','amd_source_status','amd_modified_files')
    result['preserved_fields']={key:result[key]==before[key] for key in fields}
    result['historical_local_refs_unchanged']=all(local.get(key)==value for key,value in before['local_refs'].items() if key!=branch)
    result['historical_remote_refs_unchanged']=all(remote.get(key)==value for key,value in before['remote_refs'].items() if key!=branch)
    if not all(result['preserved_fields'].values()) or not result['historical_local_refs_unchanged'] or not result['historical_remote_refs_unchanged']:
        raise RuntimeError('Preservation mismatch')
with (evidence/f'git-{sys.argv[1]}.json').open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
print(json.dumps({key:value for key,value in result.items() if key not in ('local_refs','remote_refs','first_attempt_files','amd_modified_files')}))
