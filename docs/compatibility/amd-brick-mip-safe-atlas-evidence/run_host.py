"""Experiment-local adapter to the frozen, qualified CPU owned-Job supervisor."""
import sys,os,importlib.util,hashlib,argparse
from pathlib import Path
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
OLD=ROOT/'docs/compatibility/amd-brick-pbr-mip-evidence'
P=ROOT.parent/'artifacts/amd-brick-mip-safe-atlas-e8244cb'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['feasibility','revision']);ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
    original=OLD/'run_host.py'
    # Tracked identity is bound by the exact authenticated starting commit.
    import subprocess
    reused=[OLD/name for name in ('run_host.py','assets.py','mip_math.py','mip_view.py')]
    if a.stage=='revision':
        for folder,names in [('amd-brick-seam-uv-bake-evidence',('common.py','uv_layout.py','bake_math.py')),
                             ('amd-brick-uv-bake-evidence',('sampler.py','patch_math.py','source-semantics.json'))]:
            reused.extend(ROOT/'docs/compatibility'/folder/name for name in names)
    for source in reused:
        expected=subprocess.check_output(['git','show','e8244cb0f5e044c6749c278968da720c9f029942:'+source.relative_to(ROOT).as_posix()],cwd=ROOT)
        if source.read_bytes()!=expected:raise ValueError('Frozen CPU source changed: '+source.name)
    sys.path.insert(0,str(OLD))
    s=importlib.util.spec_from_file_location('frozen_mip_host',original);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    m.E=E;m.P=P/a.stage;m.BASE='e8244cb0f5e044c6749c278968da720c9f029942';m.BRANCH='codex/amd-brick-mip-safe-atlas'
    old_authenticate=m.authenticate
    def authenticate_watched():
        auth,identities,watched,receipts=old_authenticate()
        return auth,identities,list(dict.fromkeys(watched+reused)),receipts
    m.authenticate=authenticate_watched
    old_plain=m.plain
    def phase_plain(path):
        if Path(path)==E/'independent-candidate-review.json':path=E/('independent-'+a.stage+'-review.json')
        return old_plain(path)
    m.plain=phase_plain
    m.helper.PRIVATE=m.P
    os.environ['MODLY_ATLAS_STAGE']=a.stage
    sys.argv=[str(__file__)]+(['--preflight'] if a.preflight else [])
    return m.main()

if __name__=='__main__':raise SystemExit(main())
