"""Read-only original inputs and the exact preceding successful colour proof."""
import sys
sys.dont_write_bytecode=True
import json,shutil
from pathlib import Path
import hashlib,importlib.util
E=Path(__file__).resolve().parent;ROOT=E.parents[2];P=ROOT.parent/'artifacts/amd-brick-full-sampled-glb-3db265c'
OLD=ROOT/'docs/compatibility/amd-brick-resampled-colour-evidence';PREVIOUS=ROOT.parent/'artifacts/amd-brick-resampled-colour-ad619e5/feasibility'

def pinned(name,path,sha):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Pinned source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

original=pinned('full_sampled_previous_inputs',OLD/'inputs.py','495449db3a99dd61496d5a7c1bddd1fc6ed1118acd1da826556984edcad1af40')
plain,digest,require,save,array,load_auth=original.plain,original.digest,original.require,original.save,original.array,original.load_auth
HOST,SAMPLER,METRICS,CONVERTER=original.HOST,original.SAMPLER,original.METRICS,original.CONVERTER
PINS={OLD/'inputs.py':'495449db3a99dd61496d5a7c1bddd1fc6ed1118acd1da826556984edcad1af40',OLD/'candidate.py':'eaaa72f23659f2d68da849f56857af2211964eb887f7aed53960932b1b66cf32',OLD/'independent-outcome.json':'a58518d2cd8d5825efd36ede169b1e299f2a7bca31d22b795e210522a4b2f228',PREVIOUS/'result.json':'82dadec1650f63858145a8f3dd27fe0da9ffbb36299cbfc4b1c72552c75b516c',PREVIOUS/'frozen-holdout.json':'e5fb855a26dd686aeff0539b224221f0092ae828e7fc25a0ff03d586a499b998',PREVIOUS.parent/'feasibility-host.json':'11f12e7fcecd512277766c375d256998efb451eed237bdf86e0b3a3ccc01fd74'}

def authenticate():
    auth,identities,watched,receipts=original.authenticate()
    for path,sha in PINS.items():require(digest(plain(path))==sha,'Successful feasibility/source identity: '+str(path))
    r=json.loads((PREVIOUS/'result.json').read_text());require(r['feasibility']=='PASS' and r['original_query_count']==56910 and r['holdout_query_count']==2048,'Exact accepted preceding populations')
    for name,row in r['files'].items():
        require(Path(name).name==name and name.endswith('.npy'),'Bounded saved proof basename')
        path=plain(PREVIOUS/name);require(path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],'Previous successful proof array: '+name);watched.append(path)
    require(len(r['files'])==19,'All previous proof arrays')
    require(shutil.disk_usage(P).free>=2*1024**3+256*1024**2,'Retain2GiB disk floor plus256MiB complete-output/staging allowance')
    return auth,identities,list(dict.fromkeys(watched+list(PINS))),receipts

def load(np):
    data,identities,watched=original.load(np);_,_,more,_=authenticate()
    data['resampled']={n:np.load(plain(PREVIOUS/(n+'.npy')),allow_pickle=False) for n in ('unique-original-vertex-ids','subset_linear_COLOR0','corrected_original_prediction','holdout-query-faces','holdout-barycentrics','holdout-query-kinds','holdout-world','holdout_source_spatial','holdout_prediction','holdout_weight_sum')}
    data['holdout_frozen']=json.loads((PREVIOUS/'frozen-holdout.json').read_text())
    return data,identities,list(dict.fromkeys(watched+more))

def source_hashes(identities,colour_sha):
    return dict(sourceVerticesNPY_SHA256=identities['vertices']['sha256'],sourceFacesNPY_SHA256=identities['faces']['sha256'],sourceGeometry_SHA256=identities['geometry']['sha256'],sourceMaterial_SHA256=identities['material']['sha256'],linearCOLOR0NPY_SHA256=colour_sha)
