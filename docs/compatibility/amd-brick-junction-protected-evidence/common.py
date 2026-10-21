"""Read-only original inputs; previous rejected geometry is preservation-only."""
import sys
sys.dont_write_bytecode=True
import hashlib,importlib.util,json
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-brick-junction-protected-4a03342'
OLD=ROOT/'docs/compatibility/amd-brick-reduction-attribution-evidence'

def pin(name,path,sha):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Changed pinned source: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

previous=pin('junction_previous_common',OLD/'common.py','0ce5f2b8382a062d3679649ea3a1fb3404039a3c56e483e4fe01951809da9c5e')
plain,digest,require,save=previous.plain,previous.digest,previous.require,previous.save
original=previous.original

def authenticate():
    watched=previous.authenticate()+[OLD/'common.py']
    for path,sha in {
        OLD/'outcome.json':'0484547176da902b0372841339750d30241a421b13669ce494ff4c007d6c857f',
        OLD/'independent-outcome.json':'1a937482da0227fed0ade07b200e532fc889158f85ea1482ef8caa9051b60b44',
        ROOT/'docs/compatibility/amd-brick-reduction-defect-attribution.md':'3bb483bc63b990e42e775c00a15ffef06d8efa88fd424e6ac54d8e94dad171fa',
    }.items():require(digest(plain(path))==sha,'Previous proof identity');watched.append(path)
    return list(dict.fromkeys(watched))

def begin():
    previous.P=P;previous.begin()

def load(np):
    # Do not call the previous load(): that would select rejected candidate arrays.
    v=np.load(plain(original.MESH/'vertices.npy'),mmap_mode='r',allow_pickle=False)
    f=np.load(plain(original.MESH/'faces.npy'),mmap_mode='r',allow_pickle=False)
    rgba=np.load(plain(original.IMPORTED/'oracle-colour-bytes.npy'),mmap_mode='r',allow_pickle=False)
    keys=np.load(plain(original.CACHE/'edge_keys.npy'),mmap_mode='r',allow_pickle=False)
    inc=np.load(plain(original.CACHE/'incidence.npy'),mmap_mode='r',allow_pickle=False)
    require(v.shape==(1957891,3) and v.dtype==np.float32 and f.shape==(4173584,3) and f.dtype==np.int32 and rgba.shape==(1957891,4) and rgba.dtype==np.uint8,'Original mesh and accepted native source colours only')
    require(keys.shape==inc.shape==(6167053,),'Original global edge caches')
    return v,f,rgba,keys,inc
