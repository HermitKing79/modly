"""Exact original full geometry/material and prior semantic nomination identities."""
import sys
sys.dont_write_bytecode=True
import importlib.util,json
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-brick-full-colour-glb-1012a6c'
SOURCE=ROOT/'docs/compatibility/amd-foliage-colour-glb-evidence/inputs.py'


def pinned(name,path,sha):
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Pinned original source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


original=pinned('full_colour_original_inputs',SOURCE,'27d2e8215b58d7a92933647c3748eac31f00e5464833863ff3d065f05e308b26')
plain,digest,require,save,array,load_auth=original.plain,original.digest,original.require,original.save,original.array,original.load_auth
BRICK=ROOT.parent/'artifacts/amd-brick-seam-uv-bake-87f3755/qualify/source-face-ids.npy'
FOLIAGE=ROOT.parent/'artifacts/amd-bounded-foliage-64d05aa/run/source-face-ids.npy'
SAMPLER=ROOT/'docs/compatibility/amd-brick-uv-bake-evidence/sampler.py'
METRICS=ROOT/'docs/compatibility/amd-foliage-export-decision-evidence/metrics.py'
HOST=ROOT/'docs/compatibility/amd-foliage-colour-glb-evidence/run_host.py'
PINS={BRICK:'06167754d3c27f8257439f6cb7dc3db4c434ce388d2cb2f17568da2b7d1149ea',FOLIAGE:'dc0677bf75cb4206c68726ea5165f11e935578b54497ebf2fe06184d7206b9d1',SAMPLER:'0a330eca2c991d3b7751d3cbbbe94edac317b34b28ce7216d9d3d374ef20b0c6',METRICS:'7e02dfa43cbf2167e7df44efa0ba1536bb59fdb39ebb0e812bbee61b2a971e20',HOST:'ce49d185ec23ca89a034eb6dda41d654cf97ca508501736c76029cf28397c475'}


def authenticate():
    auth,identities,watched,receipts=original.authenticate()
    for path,sha in PINS.items():require(digest(plain(path))==sha,'Source/nomination pin changed: '+str(path))
    watched += [SOURCE,*PINS]
    return auth,identities,list(dict.fromkeys(watched)),receipts


def load(np):
    auth,identities,watched,_=authenticate()
    mapping=json.loads(auth.PINS['mapping_receipt'][0].read_text())
    require(mapping['classification']=='PASS','Original complete row mapping')
    for name in ('mesh','vertices','faces','geometry','material'):
        require(mapping['input_identities'][name]['sha256']==identities[name]['sha256'],'Complete original lineage: '+name)
    v=np.load(auth.PINS['vertices'][0],mmap_mode='r',allow_pickle=False);f=np.load(auth.PINS['faces'][0],mmap_mode='r',allow_pickle=False)
    require(v.dtype==np.float32 and v.shape==(1957891,3) and f.dtype==np.int32 and f.shape==(4173584,3),'Complete original full-model arrays')
    features=auth.material_features(auth.PINS['material'][0],np)
    common=pinned('full_original_tensor_reader',ROOT/'docs/compatibility/amd-brick-seam-uv-bake-evidence/common.py','e239c7e031bfa5a072d0c02b4804d96b73ed8ca7dc55145be46882ca0b02e96b')
    coordinates=common.tensor(auth.PINS['material'][0],'coordinates',np);geometry_coordinates=common.tensor(auth.PINS['geometry'][0],'coordinates',np)
    require(coordinates.shape==geometry_coordinates.shape==(1957891,4) and coordinates.dtype==geometry_coordinates.dtype==np.int32,'Original sparse coordinate structure')
    for start in range(0,len(v),32768):require(np.array_equal(coordinates[start:start+32768],geometry_coordinates[start:start+32768]),'Full material/geometry row order')
    brick=np.load(BRICK,allow_pickle=False);foliage=np.load(FOLIAGE,allow_pickle=False)
    require(brick.dtype==foliage.dtype==np.int32 and brick.shape==(781,) and foliage.shape==(4531,),'Authenticated original region face IDs')
    for ids in (brick,foliage):require(len(np.unique(ids))==len(ids) and ids.min()>=0 and ids.max()<len(f),'Unique original region bounds')
    return dict(vertices=v,faces=f,features=features,coordinates=coordinates,brick=brick,foliage=foliage),identities,watched
