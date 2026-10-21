"""Experiment-local paths; the original CPU Job supervisor remains unchanged."""
import sys
sys.dont_write_bytecode = True
import importlib.util, subprocess
from pathlib import Path

E=Path(__file__).resolve().parent; ROOT=E.parents[2]
OLD=ROOT/'docs/compatibility/amd-brick-pbr-mip-evidence'
P=ROOT.parent/'artifacts/amd-bounded-foliage-64d05aa'
BASE='64d05aa0d40073778f316e788d5ebe595b44495f'


def main():
    reused=[OLD/name for name in ('run_host.py','assets.py')]
    for folder,names in [('amd-pbr-foliage-evidence',('foliage_select.py','foliage_layout.py')),
        ('amd-brick-defect-diagnosis-evidence',('diagnose.py','projection.py')),
        ('amd-brick-seam-uv-bake-evidence',('common.py','uv_layout.py')),
        ('amd-brick-mip-safe-atlas-evidence',('footprint.py',))]:
        reused += [ROOT/'docs/compatibility'/folder/name for name in names]
    for source in reused:
        expected=subprocess.check_output(['git','show',BASE+':'+source.relative_to(ROOT).as_posix()],cwd=ROOT)
        if source.read_bytes()!=expected: raise ValueError('Historical source changed: '+str(source))
    sys.path.insert(0,str(OLD))
    spec=importlib.util.spec_from_file_location('bounded_original_owned_host',OLD/'run_host.py')
    host=importlib.util.module_from_spec(spec); spec.loader.exec_module(host)
    host.E=E; host.P=P; host.BASE=BASE; host.BRANCH='codex/amd-bounded-foliage-uv'; host.helper.PRIVATE=P
    original=host.authenticate
    def authenticated():
        auth,identities,watched,receipts=original()
        watched += reused + [E/'nomination.json']
        return auth,identities,list(dict.fromkeys(watched)),receipts
    host.authenticate=authenticated
    return host.main()


if __name__=='__main__': raise SystemExit(main())
