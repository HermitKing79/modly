"""The unchanged proven CPU supervisor with experiment-local paths/authentication."""
import sys
sys.dont_write_bytecode=True
import importlib.util,subprocess
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
sys.path.insert(0,str(E))
from admission import P,authenticate,require
BASE='7f1b1675913dcdd91175caa8169383236f8ed7eb'
OLD=ROOT/'docs/compatibility/amd-brick-pbr-mip-evidence'


def main():
    reused=[]
    for folder,names in [('amd-brick-pbr-mip-evidence',('run_host.py','assets.py')),
        ('amd-connected-foliage-uv-evidence',('inputs.py',)),
        ('amd-brick-defect-diagnosis-evidence',('diagnose.py',)),
        ('amd-brick-seam-uv-bake-evidence',('common.py',)),
        ('amd-brick-uv-bake-evidence',('sampler.py','patch_math.py'))]:
        reused += [ROOT/'docs/compatibility'/folder/name for name in names]
    for path in reused:
        require(path.read_bytes()==subprocess.check_output(['git','show',BASE+':'+path.relative_to(ROOT).as_posix()],cwd=ROOT),'Historical source identity: '+str(path))
    sys.path.insert(0,str(OLD))
    spec=importlib.util.spec_from_file_location('decision_original_owned_cpu_host',OLD/'run_host.py')
    host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
    host.E=E;host.P=P;host.BASE=BASE;host.BRANCH='codex/amd-foliage-export-decision';host.helper.PRIVATE=P
    def authenticated():
        auth,identities,watched,receipts=authenticate()
        watched+=reused+[E/'cost.json',E/'independent-source-contract.json']
        return auth,identities,list(dict.fromkeys(watched)),receipts
    host.authenticate=authenticated
    return host.main()


if __name__=='__main__':raise SystemExit(main())
