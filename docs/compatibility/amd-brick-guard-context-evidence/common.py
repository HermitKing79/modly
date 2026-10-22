"""One frozen original seed; immutable verified receipts; private diagnostic writes only."""
import sys
sys.dont_write_bytecode=True
import hashlib,importlib.util,json
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-brick-guard-context-94456bf'
OLD=ROOT/'docs/compatibility/amd-brick-census-recovery-evidence'
RECOVERY=ROOT.parent/'artifacts/amd-brick-census-recovery-523ac20'
CENSUS=ROOT.parent/'artifacts/amd-brick-safe-census-e262ad7'

def pin(name,path,sha):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Changed preserved source: '+str(path))
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

pin('summary_contract',OLD/'summary_contract.py','3b055290bb49892c33f21094c71a8092990fe6c58da695cca8fe54a886aa24fb')
previous=pin('context_recovery_common',OLD/'common.py','1fcd5d8aa601b14c8baccd621fca85c7ee85c850fc5957658b1666383b84abb6')
plain,digest,require,save=previous.plain,previous.digest,previous.require,previous.save
original=previous.original

def pinnedpc():
    # Imports NumPy; called only after the owned worker's binary GO.
    name='context_preserved_patch_contract'
    if name not in sys.modules:pin(name,OLD/'patch_contract.py','94080df52f5f61084ac304e14d3c499ba02bd805d0f72fbe3592673bba8e34f7')
    return sys.modules[name]

def frozen_choice():
    path=plain(RECOVERY/'patch/frozen-choice.json')
    require(digest(path)=='f3acd67beef51587be47b39b36592b088edd6a246ac58b3ec8c9989975a2c92a','Original frozen seed receipt')
    choice=json.loads(path.read_text())
    require(choice==dict(eligible_face_components=9431,selected_component_faces=24396,selected_component_min_source_face=289566,selected_component_edges=35127,lower_median_rank=17563,seed_edge_id=1889720,seed_source_vertex_ids=[612841,612852]),'Exact original seed; no component/seed search')
    return choice

def authenticate():
    watched=previous.authenticate()+[OLD/'common.py',OLD/'summary_contract.py']
    pins={
        OLD/'independent-readback-outcome.json':'c6cacaa43046e95860910995f14e01155db034136615078db095eadf85ec0c41',
        OLD/'independent-patch-outcome.json':'8468c65832a9553a4da276c16378ce44748e7fdb3378406ffeb846cd300d9bd5',
        OLD/'independent-patch-review.json':'78904c9fc3ec1b1ded614b0f17ecfe5e64871d187d388f5d66af547aedd93865',
        OLD/'independent-verify-review.json':'4cc54c7cc069fe153e111b130ff90e49c9aaadad73a179a8fc886bce035accad',
        ROOT/'docs/compatibility/amd-brick-census-recovery-patch.md':'2c399b04f67d31791816f4c6c850b06a7c4e7fc2d1f8202cca474c73bfc65801',
        RECOVERY/'verify/result.json':'cd11a416d1d3a54187530e9990b838373af1d73bae2cc5ee2d375c7e6e7ba221',
        RECOVERY/'verify-host.json':'d02b6d8e4a6211bf2977ecdf7eaf57e437ecd37853f66b678204c852e289a365',
        RECOVERY/'patch/result.json':'f1a3dbe764f73757153edacbe2e782c99426870798d2b9f79c2985b81a8a7865',
        RECOVERY/'patch-host.json':'2f0becee8407ac2c0bbb822846f0f403bcbb719f0faa9946eb9a648148cc8547',
        RECOVERY/'patch/frozen-choice.json':'f3acd67beef51587be47b39b36592b088edd6a246ac58b3ec8c9989975a2c92a',
        RECOVERY/'patch/selection.json':'b6f3a4d0e556989c8848a547d75927e30f82b3390e89790a510a8f20696f19a9',
    }
    for path,sha in pins.items():require(digest(plain(path))==sha,'Historical qualification identity');watched.append(path)
    for phase in ('verify','patch'):
        host=json.loads((RECOVERY/(phase+'-host.json')).read_text());result=json.loads((RECOVERY/phase/'result.json').read_text())
        require(host['sources_unchanged'] and host['returncode']==0 and host['job_cleanup']['owned_tree_exit_verified'] and not host['job_cleanup']['forced_job_cleanup'],'Historical natural owned release')
        require(result['classification']==('PASS' if phase=='verify' else 'BLOCKED') and host['worker_receipt_sha256']==digest(RECOVERY/phase/'result.json'),'Historical verdict preserved')
        for name,row in result.get('files',{}).items():
            require(Path(name).name==name,'Historical output basename');p=plain(RECOVERY/phase/name)
            require(p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],'Historical private output identity');watched.append(p)
    review=json.loads((OLD/'independent-patch-review.json').read_text())
    for name,sha in review['source_SHA256'].items():
        require(Path(name).name==name,'Preserved source basename');p=plain(OLD/name);require(digest(p)==sha,'Unchanged historical source');watched.append(p)
    for name,key in [('recovery-plan.md','plan_SHA256'),('patch-tests.txt','focused_tests_SHA256')]:
        p=plain(OLD/name);require(digest(p)==review[key],'Historical plan/test identity');watched.append(p)
    verification_review=json.loads((OLD/'independent-verify-review.json').read_text())
    p=plain(OLD/'focused-tests.txt');require(digest(p)==verification_review['focused_tests_SHA256'],'Historical independent-readback test identity');watched.append(p)
    frozen_choice()
    return list(dict.fromkeys(watched))

def begin():previous.P=P;previous.begin()
def load(np):return previous.load(np)
