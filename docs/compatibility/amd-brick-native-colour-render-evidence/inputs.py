"""Authenticate unchanged saved import, original queries and failed history."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
import hashlib,importlib.util
E=Path(__file__).resolve().parent;ROOT=E.parents[2];P=ROOT.parent/'artifacts/amd-brick-native-colour-render-4c38a7a'
OLD=ROOT/'docs/compatibility/amd-brick-full-glb-acceptance-evidence'
def pinned(name,path,sha):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Pinned source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
original=pinned('native_original_inputs',OLD/'inputs.py','77bc4bcdec9003225e52f1acfd75f1031e9c0f6a7bdb15d9ae78135c04d29cd8')
plain,digest,require,save,load_auth=original.plain,original.digest,original.require,original.save,original.load_auth
GLB,COLOURS,MESH,BLENDER,ADDON,AUDIT,BASELINE,REFERENCE,MATERIAL_READER=original.GLB,original.COLOURS,original.MESH,original.BLENDER,original.ADDON,original.AUDIT,original.BASELINE,original.REFERENCE,original.MATERIAL_READER
IMPORTED=ROOT.parent/'artifacts/amd-brick-full-glb-acceptance-a258c4e/import'
SCENE=IMPORTED/'inspected.blend'
PINS={**original.PINS,SCENE:'67d9a9439417eebfa8ab294d955e8056f413fea28866f6006f8298a105e19588',IMPORTED/'result.json':'13ab7159c0db783ffd22404420c19d02d51b678ec5e87e18cf53565ba68ab3a7',IMPORTED.parent/'import-host.json':'19c23db0a47c7e3c003013b70dab7b5fc67cc2e1ae8042fb2e621b45f1b6b370',IMPORTED.parent/'verify/failure.json':'eb6c26254d55ca14b5d35dc923ae7146c7569332b63fed6dec1ca779f98db0f6',IMPORTED.parent/'verify-host.json':'b4a52a4bd3b6c66ddbd9363b8ba8be52d1bbf588fe12b235f36699f5ff6dd476',OLD/'outcome.json':'37aa1af0d347f94f96cb356a5a786152f8bd371d68b3f9964e3bd9db0dec773a',OLD/'independent-outcome.json':'c3e709c0d7cd80dc992cef4fa5fcd410923607851724d2f683cc63a16e9a91fd',OLD/'common.py':'3560a3b9db9776b5098dfca479c63d1dcc8a2e09dcedf5bc12074342a4f6b876'}
FULL=ROOT/'docs/compatibility/amd-brick-full-sampled-glb-evidence'
PINS[OLD/'contract.py']='4c2eb28ee14707b90edac7d9ab9882473f3dfcd9e194ddc5d2f7b14c7d68edf5'
full=pinned('native_full_original_queries',FULL/'inputs.py','f20e771b322788a1c4fbde2031906dd4e81dcd2c55c772f232a9b33106c3b07c')
def authenticate():
    auth,ids,watched,receipts=original.authenticate()
    for path,sha in PINS.items():require(digest(plain(path))==sha,'Saved input/history identity: '+str(path))
    r=json.loads((IMPORTED/'result.json').read_text());require(r['classification']=='PASS' and r['fresh_GLTF_imports']==1,'Original single actual import')
    require(set(r['files'])=={'actual-vertices.npy','actual-faces.npy','actual-colour-bytes.npy','oracle-colour-bytes.npy','inspected.blend'},'Exact original import manifest')
    for name,row in r['files'].items():
        path=plain(IMPORTED/name);require(path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],'Saved imported artifact identity');watched.append(path)
    require(json.loads((IMPORTED.parent/'verify/failure.json').read_text())['classification']=='NO-GO','Preserve original failed verifier')
    return auth,ids,list(dict.fromkeys(watched+list(PINS)+[FULL/'inputs.py'])),receipts
def load(np):
    data,ids,watched=full.load(np);_,_,more,_=authenticate()
    return data,ids,list(dict.fromkeys(watched+more))
