"""Fixed CPU foliage admission via the unchanged proven owned-Job supervisor."""
import sys,importlib.util,subprocess,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
OLD=ROOT/'docs/compatibility/amd-brick-pbr-mip-evidence'
P=ROOT.parent/'artifacts/amd-pbr-foliage-ecb15aa'

def main():
    reused=[OLD/name for name in ('run_host.py','assets.py')]
    for folder,names in [('amd-brick-defect-diagnosis-evidence',('projection.py',)),('amd-brick-seam-uv-bake-evidence',('uv_layout.py','common.py')),('amd-brick-mip-safe-atlas-evidence',('footprint.py',))]:
        reused.extend(ROOT/'docs/compatibility'/folder/name for name in names)
    for source in reused:
        expected=subprocess.check_output(['git','show','ecb15aa0a2855b9c4f9d3a45aea4192cd92a3900:'+source.relative_to(ROOT).as_posix()],cwd=ROOT)
        if source.read_bytes()!=expected:raise ValueError('Frozen CPU source changed: '+str(source))
    # Actual Blender success and independent outcome review precede any foliage run.
    result=json.loads((P/'brick/result.json').read_text());review=json.loads((E/'independent-brick-outcome.json').read_text())
    if result['classification']!='PASS' or review['classification']!='PASS':raise ValueError('Independent real Blender import PASS required')
    if review['result_SHA256']!=hashlib.sha256((P/'brick/result.json').read_bytes()).hexdigest():raise ValueError('Exact independent brick result binding')
    if review['host_result_SHA256']!=hashlib.sha256((P/'brick-host.json').read_bytes()).hexdigest():raise ValueError('Exact independent brick containment binding')
    sys.path.insert(0,str(OLD))
    s=importlib.util.spec_from_file_location('frozen_foliage_owned_host',OLD/'run_host.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    m.E=E;m.P=P/'foliage';m.BASE='ecb15aa0a2855b9c4f9d3a45aea4192cd92a3900';m.BRANCH='codex/amd-portable-pbr-foliage';m.helper.PRIVATE=m.P
    original=m.authenticate
    def authenticated():
        auth,identities,watched,receipts=original()
        watched += reused+[P/'brick/result.json',P/'brick-host.json',P/'brick/near.png',P/'brick/far.png',E/'independent-brick-outcome.json']
        return auth,identities,list(dict.fromkeys(watched)),receipts
    m.authenticate=authenticated;oldplain=m.plain
    def mapped(path):
        if Path(path)==E/'independent-candidate-review.json':path=E/'independent-foliage-review.json'
        return oldplain(path)
    m.plain=mapped
    # The shared host resolves its worker by this exact filename.
    return m.main()

if __name__=='__main__':raise SystemExit(main())
