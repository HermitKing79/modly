"""Fresh recovery outputs; immutable original census and failed readback."""
import sys
sys.dont_write_bytecode=True
import hashlib,importlib.util,json
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-brick-census-recovery-523ac20'
OLD=ROOT/'docs/compatibility/amd-brick-safe-census-evidence'
CENSUS=ROOT.parent/'artifacts/amd-brick-safe-census-e262ad7'
CENSUS_SHA='69c2133a0ae6c04bd5aabc8c924a5587d13980cb2e5e0c81abd59eb946b64c4b'

def pin(name,path,sha):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Changed preserved source: '+str(path))
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

previous=pin('recovery_previous_common',OLD/'common.py','d48ac1e63f7d089fdc7dd95f57470a6b8c3381437e2b3b9cae0493715a55baaa')
plain,digest,require,save=previous.plain,previous.digest,previous.require,previous.save
original=previous.original

def census_receipt():
    from summary_contract import sealed_json
    path=plain(CENSUS/'diagnose/result.json');require(path.stat().st_size<=1024**2,'Bounded producer receipt')
    return sealed_json(path.read_bytes(),CENSUS_SHA)

def authenticate():
    watched=previous.authenticate()+[OLD/'common.py']
    pins={
        OLD/'independent-candidate-review.json':'f7bf2223e0b25455afd2fac84fd5794aa25de75dae3d1f24ceb8eea69e750dff',
        OLD/'outcome.json':'1bbfbfb65ae5e83b738a47c8a747eeec036fd74e4af608a175b58dba9bfc826b',
        OLD/'independent-outcome.json':'62446f9b68f0a348931723b81370b1c6758ec3953aa222b792401ea3faf010eb',
        CENSUS/'diagnose/result.json':CENSUS_SHA,
        CENSUS/'diagnose-host.json':'706c26d6cae649f2b8da7cbd60e45ef9c468e126318835300c74738c9a079b98',
        CENSUS/'verify/result.json':'1e278c38d9fc9b4642f2eae7323ee1860991e1772410f0fb9f48fc2b2f770dd7',
        CENSUS/'verify-host.json':'e2bd8454e0e34e784cc67106154cbd6b2535a4a6edce024dec0efc0ed19bda3b',
    }
    for path,sha in pins.items():require(digest(plain(path))==sha,'Preserved census/failed-readback identity');watched.append(path)
    review=json.loads((OLD/'independent-candidate-review.json').read_text())
    for name,key in (('census-plan.md','plan_SHA256'),('focused-tests.txt','focused_tests_SHA256')):
        p=plain(OLD/name);require(digest(p)==review[key],'Unchanged historical plan/test receipt');watched.append(p)
    for name,sha in review['source_SHA256'].items():
        require(Path(name).name==name,'Original source basename');p=plain(OLD/name);require(digest(p)==sha,'Unchanged historical source');watched.append(p)
    parent=census_receipt();require(parent['census_completed'] is True and parent['actual_simplification_calls']==0,'Original-only complete census')
    for name,row in parent['files'].items():
        require(Path(name).name==name,'Original census basename');p=plain(CENSUS/'diagnose'/name)
        require(p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],'Every original census artifact identity');watched.append(p)
    return list(dict.fromkeys(watched))

def begin():previous.P=P;previous.begin()
def load(np):return previous.load(np)
