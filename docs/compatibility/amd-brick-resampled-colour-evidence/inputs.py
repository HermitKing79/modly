"""Pin the entire preceding measurement and original source closure, read only."""
import sys
sys.dont_write_bytecode=True
import hashlib,importlib.util,json
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-brick-resampled-colour-ad619e5'
OLD=ROOT/'docs/compatibility/amd-brick-full-colour-glb-evidence'
PRIOR=ROOT.parent/'artifacts/amd-brick-full-colour-glb-1012a6c/suitability'

def pinned(name,path,sha):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Pinned source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

original=pinned('resampled_original_inputs',OLD/'inputs.py','a7ddabfdf13f24564377bd78cabccfd69679d51579e74a4ec5c35bd20073779f')
plain,digest,require,save,array,load_auth=original.plain,original.digest,original.require,original.save,original.array,original.load_auth
HOST=original.HOST;SAMPLER=original.SAMPLER;METRICS=original.METRICS
CONVERTER=ROOT/'scripts/compatibility/amd_flexible_dual_grid_cpu.py'
COMMON=ROOT/'docs/compatibility/amd-brick-seam-uv-bake-evidence/common.py'
PINS={OLD/'inputs.py':'a7ddabfdf13f24564377bd78cabccfd69679d51579e74a4ec5c35bd20073779f',OLD/'decision.py':'3c6353b4328879255fae30d9ab115a0b82da3656cf19f847bd618e772e545f3f',OLD/'independent-outcome.json':'7ccaa89eb692441823e438481044d36c0f9aa12bda355e3a59dc57df723485d6',PRIOR/'result.json':'11ac45c5132cea9f638c94fbd4a597eaf6f6747520b5d591217a34fba27c6eb0',PRIOR/'frozen-sampling.json':'202c940094796c50b9d1823c4f718fe88d9b116fad136f0774b8652e81c7f69d',CONVERTER:'744e44768a48603e4f6d68e8dc4ae0e1f8ce78b16c2efd73b511a2a309562b40',COMMON:'e239c7e031bfa5a072d0c02b4804d96b73ed8ca7dc55145be46882ca0b02e96b'}
PINS[PRIOR.parent/'suitability-host.json']='21bfa91ef4263b538838a8c92e2902ffec26896419b2a986243fe90856aa2476'

def authenticate():
    auth,identities,watched,receipts=original.authenticate()
    for path,sha in PINS.items():require(digest(plain(path))==sha,'Original source/result pin changed: '+str(path))
    r=json.loads((PRIOR/'result.json').read_text());require(r['query_count']==56910 and r['material_suitability']=='BLOCKED','Exact completed preceding measurement')
    for name,row in r['files'].items():
        require(Path(name).name==name and name.endswith('.npy'),'Bounded previous array basename')
        path=plain(PRIOR/name);require(path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],'Previous frozen array: '+name);watched.append(path)
    require(len(r['files'])==24,'Every previous array retained')
    return auth,identities,list(dict.fromkeys(watched+list(PINS))),receipts

def load(np):
    data,identities,watched=original.load(np);auth,_,all_watched,_=authenticate()
    reader=pinned('resampled_original_reader',COMMON,PINS[COMMON]);data['geometry_features']=reader.tensor(auth.PINS['geometry'][0],'features',np)
    data['prior']={name:np.load(plain(PRIOR/(name+'.npy')),allow_pickle=False) for name in ('query-faces','barycentrics','query-kinds','world','source_spatial','vertex_RGB','positive_neighbours','raw_weight_sum')}
    q=data['prior'];require(q['query-faces'].dtype==np.int32 and q['query-faces'].shape==(56910,) and q['barycentrics'].dtype==np.float64 and q['barycentrics'].shape==(56910,3),'Exact previous query structures')
    require(q['world'].dtype==np.float32 and q['world'].shape==(56910,3) and q['source_spatial'].dtype==np.float32 and q['source_spatial'].shape==(56910,6),'Exact previous world/reference structures')
    require(q['query-kinds'].dtype==np.int8 and q['query-kinds'].shape==(56910,),'Original query categories')
    data['frozen']=json.loads((PRIOR/'frozen-sampling.json').read_text())
    data['selected']={n:np.load(plain(PRIOR/(n+'-selected-face-ids.npy')),allow_pickle=False) for n in data['frozen']['cohorts']}
    return data,identities,list(dict.fromkeys(watched+all_watched))
