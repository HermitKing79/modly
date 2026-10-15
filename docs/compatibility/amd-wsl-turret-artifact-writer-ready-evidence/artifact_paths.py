"""Only this proof's exact artifact parent and two fixed leaves."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

E=Path(__file__).resolve().parent
ROOT=E.parents[2]
PARENT=ROOT.parent/'artifacts'
LEAVES={'neural':'amd-wsl-turret-writer-ready-4870856','fixture':'cpu-writer-acceptance-4870856'}
POWERSHELL=Path('C:/Users/wattz/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe')

def leaf(kind):return PARENT/LEAVES[kind]

def host_command(leaf_name,existing=False,prepare=False):
    # File-script execution is disabled on this host. Supply the exact audited
    # command block without changing Windows execution-policy configuration.
    source=(E/'artifact_parent.ps1').read_text(encoding='utf-8')
    command=[str(POWERSHELL),'-NoProfile','-NonInteractive','-Command',
             '& {\n'+source+'\n} -LeafName \''+leaf_name.replace("'","''")+"'"+
             (' -AllowExistingLeaf' if existing else '')+(' -Prepare' if prepare else '')]
    return command

def check_host(kind,existing=False,prepare=False):
    if sys.platform!='win32':raise RuntimeError('Windows directory authority required')
    command=host_command(LEAVES[kind],existing,prepare)
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=10)
    if result.returncode:raise RuntimeError('Trusted artifact path check: '+result.stderr)
    record=json.loads(result.stdout)
    if record['status']!='PASS' or Path(record['parent'])!=PARENT or Path(record['leaf'])!=leaf(kind):
        raise RuntimeError('Host artifact path differs')
    return record

def check_guest(kind):
    if sys.platform!='linux':raise RuntimeError('Linux mapping check required')
    expected=Path('/mnt/c/Users/wattz/Documents/ChatGPT/modly/artifacts')
    if PARENT!=expected or PARENT.resolve(strict=True)!=expected:raise RuntimeError('Foreign WSL artifact parent')
    for path in [PARENT,*PARENT.parents]:
        if path.is_symlink():raise RuntimeError('WSL artifact ancestor symlink')
    info=PARENT.lstat()
    if not stat.S_ISDIR(info.st_mode) or not os.access(PARENT,os.W_OK|os.X_OK):raise RuntimeError('WSL parent permission/type mismatch')
    if os.path.lexists(leaf(kind)):raise RuntimeError('Exclusive guest artifact leaf already exists')
    fs=os.statvfs(PARENT)
    if fs.f_bavail*fs.f_frsize<2*1024**3:raise RuntimeError('Guest artifact disk floor refused')
    return {'parent':str(PARENT),'resolved':str(PARENT.resolve()),'leaf':str(leaf(kind)),
            'uid':info.st_uid,'gid':info.st_gid,'mode':info.st_mode,'WSL_projection_only':True,
            'Windows_owner_authority':'artifact_parent.ps1','writable':True,'leaf_absent':True}
