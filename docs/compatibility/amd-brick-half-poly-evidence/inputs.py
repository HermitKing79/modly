"""Authenticated original scene/mesh/native colour and one private reduction."""
import sys
sys.dont_write_bytecode=True
import json,hashlib,importlib.util,os
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-brick-half-poly-5705f9d-corrected'
OLD=ROOT/'docs/compatibility/amd-brick-native-colour-render-evidence'

def pinned(name,path,sha):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Pinned source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

original=pinned('half_original_native_inputs',OLD/'inputs.py','a25e1589b891e4b308242de64c879a2eae93e6c8e3589659b3f383f498290564')
plain,digest,require,save,load_auth=original.plain,original.digest,original.require,original.save,original.load_auth
GLB,MESH,SCENE,IMPORTED,BLENDER=original.GLB,original.MESH,original.SCENE,original.IMPORTED,original.BLENDER
REFERENCE=ROOT.parent/'artifacts/amd-brick-native-colour-render-4c38a7a/render/view-pos-x-ev2.png'
BASELINE=ROOT/'docs/compatibility/amd-brick-visual-inspection-evidence/render-metadata.json'
MOD=Path('C:/Users/wattz/AppData/Roaming/Modly/builtin-extensions/mesh-optimizer/node_modules/meshoptimizer')
NODE=Path('C:/Users/wattz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')
CACHE=ROOT.parent/'artifacts/amd-brick-defect-diagnosis-1d1c1a6/analysis/cache'
PINS={OLD/'independent-outcome.json':'9fd602299e86f0e865b0c58bc22f800de95f349697cdc4ac4db1cee723884f77',
      OLD/'outcome.json':'0790c05ef040df16039b1525b8dc28999828e3710bc44d0738cd1799ceb60011',
      OLD/'quality.py':'6b619aef3176e2d86ac1b71fbfa3369a4c215cee14ca3c59b7f6aeaf481f726a',
      REFERENCE:'cb9b6df90fe9ec1d9dbf7b5b38878776c6b975cb8f6042a1a8d2f6e86e8ad52b',
      BASELINE:'394b54c5633598d56c8ad38c00b4712481b2405c91e3ae5bcaf91974f974d6ee',
      CACHE/'edge_keys.npy':'528da7b562cfca6d9428182f91a174ffea908f55f6bb3670232d914238d5e489',
      CACHE/'incidence.npy':'f83daf0682cc1606b1591a60ebc0d13796a8436eb3e14a1d4a4ce81d0dc1d413',
      CACHE/'vertex_components.npy':'7c3a259aad204bad583d79630374b80a6425bde6f99829be3711e33f985d8100',
      MOD/'package.json':'559c5fdb441f735c53c84ad7c3d0025420315f891190326853fd423aae138fe6',
      MOD/'meshopt_simplifier.js':'5a10a6d6a2c2ab113cfa81be2d31aaf89e46842eedaa1245d266b16e2976dc6b',
      MOD/'README.md':'fb6e8186210e13172e79cc426bc628cf2a9a28ab6ef2c362c7fd5bfa4f8fe38a',
      NODE:'3602f2bb1a10f2cbab4c36886218a33c1ab3db87290e73b033c46c77147d0237'}
PINS[ROOT/'docs/compatibility/amd-brick-defect-diagnosis-evidence/topology.py']='9d6e29e83b35c60f1f9b49192a290069bf867b51493da3cc5ec017525017d561'
PINS[ROOT/'docs/compatibility/amd-brick-uv-bake-evidence/sampler.py']='0a330eca2c991d3b7751d3cbbbe94edac317b34b28ce7216d9d3d374ef20b0c6'

def authenticate():
    _,_,watched,_=original.authenticate()
    for path,sha in PINS.items():require(digest(plain(path))==sha,'Original proof/runtime identity changed: '+str(path))
    prior=json.loads((OLD/'outcome.json').read_text());require(prior['import_qualification']==prior['classification']=='PASS' and prior['render_count']==1,'Accepted original native-colour preview')
    require(json.loads((MOD/'package.json').read_text())['version']=='0.22.0','Already installed meshoptimizer0.22')
    return list(dict.fromkeys(watched+[OLD/'inputs.py',*PINS]))

def begin():
    require(sys.stdin.buffer.readline(64)==b'GO\n','Owned exact binary GO')
    auth=load_auth();auth.P=P;sys.addaudithook(auth.runtime_guard)

def load(np):
    data,ids,watched=original.load(np);authenticate()
    data['native_bytes']=np.load(IMPORTED/'oracle-colour-bytes.npy',mmap_mode='r',allow_pickle=False)
    data['labels']=np.load(CACHE/'vertex_components.npy',mmap_mode='r',allow_pickle=False)
    require(data['labels'].shape==(1957891,) and data['labels'].dtype==np.int32,'Original component table')
    return data

def write_array(path,a):
    import numpy as np
    with path.open('xb') as f:np.save(f,a,allow_pickle=False);f.flush();os.fsync(f.fileno())
    b=np.load(path,mmap_mode='r',allow_pickle=False);require(a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b),'Independent NPY readback')

def manifest(directory,names):return {n:dict(bytes=(directory/n).stat().st_size,sha256=digest(directory/n)) for n in names}

def verify_stage(stage):
    out=plain(P/stage);r=json.loads((out/'result.json').read_text());require(r['classification']=='PASS','Previous actual stage PASS: '+stage)
    for n,row in r.get('files',{}).items():
        require(Path(n).name==n,'Stage safe basename');path=plain(out/n)
        require(path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],'Stage output changed: '+n)
    return r
