"""Read-only exact saved selection and qualified decision inputs."""
import sys
sys.dont_write_bytecode=True
import hashlib,importlib.util,json
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-foliage-colour-glb-cfe852b'
DECISION=ROOT.parent/'artifacts/amd-foliage-export-7f1b167/run'
SELECTION=ROOT.parent/'artifacts/amd-bounded-foliage-64d05aa/run'
SOURCE=ROOT/'docs/compatibility/amd-foliage-export-decision-evidence/admission.py'
if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!='eabc7e916a19fce3a7ba3a30c002b2f68b945426e9fab0623b9d074e1e5589ab':raise ValueError('Original selection admission source changed')
spec=importlib.util.spec_from_file_location('colour_original_inputs',SOURCE);original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
plain,digest,require,save,array,load_auth=original.plain,original.digest,original.require,original.save,original.array,original.load_auth


def authenticate():
    auth,identities,watched,receipts=original.authenticate()
    result=plain(DECISION/'result.json')
    require(digest(result)=='4bc3e9add407742a78897e4cfb07dbfb059b72deab7d5d33cc5e9cddb4b904c1','Exact qualified no-UV result')
    decision=json.loads(result.read_text())
    require(decision['vertex_colour_candidate']=='PASS' and decision['total_queries']==58951,'Previously qualified complete population')
    for name,row in decision['files'].items():
        require(Path(name).name==name,'Bounded decision basename')
        path=plain(DECISION/name)
        require(path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],'Original decision array identity: '+name)
        watched.append(path)
    watched += [result,SOURCE]
    return auth,identities,list(dict.fromkeys(watched)),receipts


def load(np):
    data,identities,watched=original.load_selection(np)
    _,_,all_watched,_=authenticate()
    linear=np.load(plain(DECISION/'linear-COLOR0.npy'),allow_pickle=False)
    require(linear.dtype==np.float32 and linear.shape==(3177,3) and np.isfinite(linear).all() and ((linear>=0)&(linear<=1)).all(),'Saved qualified F32 linear colours')
    data['linear-COLOR0']=linear
    return data,identities,list(dict.fromkeys(watched+all_watched))
