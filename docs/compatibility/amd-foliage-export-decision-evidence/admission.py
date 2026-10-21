"""Pinned reuse of the exact saved selection and prior815-chart evidence."""
import sys
sys.dont_write_bytecode=True
import importlib.util,json
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-foliage-export-7f1b167'
OLD=ROOT/'docs/compatibility/amd-connected-foliage-uv-evidence'
path=OLD/'inputs.py'
import hashlib
if hashlib.sha256(path.read_bytes()).hexdigest()!='77078fbe3a8cf83f84642d604a93dea87ac229c8a9068e0b31d5369553c6b5d4':raise ValueError('Qualified input adapter changed')
spec=importlib.util.spec_from_file_location('decision_original_inputs',path)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
digest,plain,save,require,array,load_auth=original.digest,original.plain,original.save,original.require,original.array,original.load_auth
C=ROOT.parent/'artifacts/amd-connected-foliage-bad7ba4/capacity/run'


def authenticate():
    auth,identities,watched,receipts=original.authenticate()
    require(digest(plain(C/'result.json'))=='26f3150964f61aa39f623a236b42456da35d94978e8a50049b1d81447cd6cf6c','Exact saved disk capacity result')
    report=json.loads((C/'result.json').read_text())
    require(report['candidate_chart_report']['chart_count']==815,'Original815-chart capacity identity')
    for name,row in report['files'].items():
        require(Path(name).name==name,'Single saved disk basename')
        item=plain(C/name);require(item.stat().st_size==row['bytes'] and digest(item)==row['sha256'],'Saved disk data identity')
        watched.append(item)
    watched += [path,C/'result.json']
    return auth,identities,list(dict.fromkeys(watched)),receipts


def load_selection(np):
    values,identities,watched=original.load_selection(np)
    _,_,all_watched,_=authenticate()
    return values,identities,list(dict.fromkeys(watched+all_watched))


def pinned(name,path,sha):
    require(digest(plain(path))==sha,'Pinned source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
