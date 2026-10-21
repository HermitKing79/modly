"""The unchanged original asset and prior complete source/importer audits."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
import hashlib,importlib.util
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-brick-full-glb-acceptance-a258c4e'
OLD=ROOT/'docs/compatibility/amd-brick-full-glb-blender-evidence'
def pinned(name,path,sha):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Pinned source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
original=pinned('amended_original_inputs',OLD/'inputs.py','f9d121c2bf749f864efb5d0ad76bbae2477ff85d00405e1f1545f90ce3cfbc2f')
plain,digest,require,save,load_auth=original.plain,original.digest,original.require,original.save,original.load_auth
GLB,COLOURS,MESH,BLENDER,ADDON=original.GLB,original.COLOURS,original.MESH,original.BLENDER,original.ADDON
AUDIT=OLD/'importer-audit.json';AUDIT_SHA='266d2a9fb761e392ad2e2019c29b87f4c915627eb184e376277390447c470c72'
PRIOR_PROOF=ROOT.parent/'artifacts/amd-brick-full-glb-blender-df195c4/preflight/result.json'
BASELINE=ROOT.parent/'artifacts/amd-brick-pbr-exposure-b0b7be1/renders/render-metadata.json'
REFERENCE=BASELINE.parent/'view-pos-x-ev2.png'
MATERIAL_READER=ROOT/'docs/compatibility/amd-foliage-colour-glb-evidence/render_worker.py'
PINS={AUDIT:AUDIT_SHA,PRIOR_PROOF:'a182aff5060704ff78d76b9249696691e85872d30cde89af0143c2ff72422911',OLD/'independent-outcome.json':'be1b56d169b178d1e37b74b360a66cc5fed20f80520ecf1dda734d0b8455bc89',BASELINE:'2e09ea2c4ba104f37133e1322759ce9de3fe5e92a5aeac798e379d8bf1685464',REFERENCE:'74d1655d53d5fe7d262756c7e23c0a695aa42b7ee3c43e075007a59fc3bf7f54',MATERIAL_READER:'a4754f1c8643c17e368edf121e870d43960bb1459a1fd29e2d8b8c0e8098f03e'}
def authenticate():
    auth,ids,watched,receipts=original.authenticate()
    for path,sha in PINS.items():require(digest(plain(path))==sha,'Original source/proof identity: '+str(path))
    audit=json.loads(AUDIT.read_text())
    for name,sha in audit['installed_source_SHA256'].items():
        path=plain(ADDON/name);require(digest(path)==sha,'Installed importer changed');watched.append(path)
    proof=json.loads(PRIOR_PROOF.read_text())
    require(proof['counts']['referenced_vertices']==1952142 and proof['counts']['unreferenced_vertices']==5749 and proof['original_geometry_and_F32_COLOUR0_entire_file_exact'],'Exact prior membership/full-file proof')
    return auth,ids,list(dict.fromkeys(watched+list(PINS))),receipts
