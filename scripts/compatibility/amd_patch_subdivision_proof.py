"""Phase17 bounded CPU architecture gate; no model, reconstruction or MIP route."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import amd_cgal_proof as owner
import amd_closure_proof as previous
PRIOR=previous.BASE
BASE=PRIOR.parent/'amd-patch-subdivision-20261031-v1'
STAGES=('authenticate','backend','source-authenticate','cell-audit','preserve')

def source_authenticate():
    import hashlib
    import zipfile
    root=BASE.parent/'amd-cgal-global-proof-20261030-v1'
    archive=root/'downloads/CGAL-6.2.1-library.zip'
    pin=dict(bytes=10925461,sha256='db49ffa9d2c5a3386e0725983bf3478e3fd2ade43553f8fc3900e18554decd49')
    checked=previous.check(archive,pin);headers={}
    names=['CGAL-6.2.1/include/CGAL/Polygonal_surface_reconstruction.h',
           'CGAL-6.2.1/include/CGAL/Polygonal_surface_reconstruction/internal/hypothesis.h']
    with zipfile.ZipFile(archive) as z:
        for name in names:
            if z.namelist().count(name)!=1:raise ValueError('Ambiguous source archive member')
            data=z.read(name);identity=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            headers[name]=previous.check(root/'isolated-native'/name,identity)
    owner.save(BASE/'source-authentication.json',dict(classification='PASS_SOURCE_ONLY',archive=checked,headers=headers,native_backend_admission=False))

def authenticate():
    checked={}
    for name,row in json.loads((PRIOR/'evidence-manifest.json').read_text()).items():
        checked[name]=previous.check(PRIOR/name,row)
    old=json.loads((PRIOR/'authentication.json').read_text())
    for name,row in old['Phase14_files'].items():previous.check(previous.PRIOR/name,row)
    for name,row in old['original_snapshot'].items():previous.check(Path(name),row)
    protected=json.loads((PRIOR/'phase15-authentication.json').read_text())['protected_records']
    for name,row in protected.items():
        s=Path(name).stat()
        if (s.st_size,s.st_mtime_ns,s.st_ino)!=(row['bytes'],row['mtime_ns'],row['file_id']):raise ValueError('Protected identity differs')
    for receipt in ('diagnostic-review.json','refinement-review.json'):
        pin=json.loads((PRIOR/receipt).read_text())['bindings']
        exe=PRIOR/('build/Release/official_sample.exe' if receipt.startswith('diagnostic') else 'certify-build/Release/official_sample.exe')
        if owner.sha(exe)!=pin['executable']:raise ValueError('Qualified native binary changed')
    owner.save(BASE/'authentication.json',dict(classification='PASS',Phase16_files=checked,Phase14_files=old['Phase14_files'],original_snapshot=old['original_snapshot'],protected_records=protected,criteria_sha256=owner.sha(owner.ROOT/'docs/compatibility/amd-ai-hard-surface-criteria.json')))

def worker(stage):
    if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO and EOF required')
    from amd_zombie_d_compare import audit_policy
    sys.addaudithook(audit_policy(BASE))
    if owner.sha(owner.ROOT/'docs/compatibility/amd-ai-hard-surface-criteria.json')!='397bf6c2dc8908e8a3b1a02ba169d6ec6ab5bc2aaf11ee96a4cb84e81591aad0':raise ValueError('Frozen criteria differs')
    if stage=='authenticate':authenticate();return
    if stage=='source-authenticate':source_authenticate();return
    if stage=='backend':
        import amd_mesh_aware_input as data
        data.run.BASE=BASE;data.verify_backend();return
    if stage=='cell-audit':
        review=json.loads((BASE/'cell-review.json').read_text())
        for key,path in {'entry':Path(__file__),'audit':Path(__file__).with_name('amd_patch_cell_audit.py'),'graph':Path(__file__).with_name('amd_closure_graph.py'),'plan':BASE/'plan.json','source_authentication':BASE/'source-authentication.json'}.items():
            if review['classification']!='APPROVED' or owner.sha(path)!=review['bindings'][key]:raise ValueError('Reviewed audit changed')
        auth=json.loads((BASE/'authentication.json').read_text())
        for name in ('diagnostic/actual-polygons.jsonl','diagnostic/actual-fans.jsonl','diagnostic/owned-session/arrangement/prepared-planes.csv','complete-certificates.jsonl','certified-closure-screen.json','fidelity-impossibility.json'):
            previous.check(PRIOR/name,previous.old.manifest_identity(auth['Phase16_files'],name))
        for name in ('augmented/train.ply','validation-points.bin'):
            previous.check(previous.PRIOR/name,previous.old.manifest_identity(auth['Phase14_files'],name))
        import amd_patch_cell_audit
        amd_patch_cell_audit.run();return
    if stage=='preserve':
        a=json.loads((BASE/'authentication.json').read_text())
        for name,row in a['Phase16_files'].items():previous.check(PRIOR/name,row)
        for name,row in a['Phase14_files'].items():previous.check(previous.PRIOR/name,row)
        for name,row in a['original_snapshot'].items():previous.check(Path(name),row)
        for name,row in a['protected_records'].items():
            s=Path(name).stat()
            if (s.st_size,s.st_mtime_ns,s.st_ino)!=(row['bytes'],row['mtime_ns'],row['file_id']):raise ValueError('Protected identity changed')
        owner.save(BASE/'preservation-after.json',dict(classification='PASS',Phase16_files=len(a['Phase16_files']),Phase14_files=len(a['Phase14_files']),original_sources=len(a['original_snapshot']),protected_records=len(a['protected_records'])));return

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=STAGES);p.add_argument('--worker',choices=STAGES);a=p.parse_args()
    if a.worker:worker(a.worker);return 0
    if not a.stage:print('INERT: bounded architecture audit only; no solve or export');return 0
    owner.BASE=BASE;owner.__file__=__file__
    return owner.supervise(a.stage)

if __name__=='__main__':raise SystemExit(main())
