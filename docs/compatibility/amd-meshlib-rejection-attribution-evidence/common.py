"""Saved-array admission only; all MeshLib imports are prohibited."""
import sys
sys.dont_write_bytecode=True
import importlib.util,os
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-meshlib-rejection-attribution-b0d6fcf'
PRIOR_E=ROOT/'docs/compatibility/amd-meshlib-patch-constructor-evidence'
PRIOR_P=ROOT.parent/'artifacts/amd-meshlib-patch-constructor-bbdb6ac'
PRIOR_COMMON_SHA='650a7b05ec705227d0c19f09f387ff3c344eeb53ce2e1b5085e317f0543150cb'
import hashlib
def initial_digest(path):
    path=Path(path).absolute()
    for node in (path,*path.parents):
        info=node.lstat()
        if node.is_symlink() or getattr(info,'st_file_attributes',0)&0x400:raise ValueError('Plain bootstrap source path required')
    if path.stat().st_nlink!=1:raise ValueError('Single-link bootstrap source required')
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if initial_digest(PRIOR_E/'common.py')!=PRIOR_COMMON_SHA:raise ValueError('Pinned historical read-only admission helper')
spec=importlib.util.spec_from_file_location('historical_patch_admission',PRIOR_E/'common.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
plain,digest,save,read_json,require=prior.plain,prior.digest,prior.save,prior.read_json,prior.require
SDK,SDK_SHA,INPUT_MANIFEST,GUARD,GUARD_SHA=prior.SDK,prior.SDK_SHA,prior.INPUT_MANIFEST,prior.GUARD,prior.GUARD_SHA
PINS={
    PRIOR_E/'common.py':PRIOR_COMMON_SHA,
    PRIOR_E/'independent-outcome.json':'74d81a183dd00ec39f568e23425be8b8187520083df35bdbcef489d89db03a94',
    PRIOR_P/'import/result.json':'6a47a4df80fc32610bbb2ff2072d15c38f635c621de92d8dab684d766e55ed1f',
    PRIOR_P/'verify/result.json':'433d37573237a001b0f6ca5f1f2c1bfe77dfbf533fc6396c34566669ef630aab',
    PRIOR_P/'verify/brick-rejected-source-memberships.json':'d8bbf36aa4d3083a2d6fb7fd875f683f8749be2c52f58d537f8128640d0fef80',
    PRIOR_P/'import-host.json':'68b0a00f6e0740b1ec72d620cacb29f018ffeed9d7eaa0722fc02d1602650a1d',
    PRIOR_P/'verify-host.json':'c92cd2d04bf94e4fc80e55dbb5c93f95a05a66aa1ea0d9a6d5c3908452cd7d67',
}

def authenticate():
    watched=prior.authenticate()+list(PINS)
    for path,sha in PINS.items():require(digest(path)==sha,'Exact saved rejection evidence: '+str(path))
    outcome=read_json(PRIOR_E/'independent-outcome.json')
    require(outcome['brick_input_transport']=='NO-GO' and outcome['foliage_input_transport']=='NOT RUN','Historical NO-GO retained')
    for name,sha in outcome['source_SHA256'].items():
        path=PRIOR_E/name;require(digest(path)==sha,'Unchanged original constructor/checker source');watched.append(path)
    result=read_json(PRIOR_P/'import/result.json')
    require(result['constructor_calls']==1 and [x['name'] for x in result['patches']]==['brick'],'Sole existing constructor only')
    npys=0
    for rel,item in result['files'].items():
        require(type(rel)is str and '\\' not in rel and ':' not in rel and rel.startswith('brick/') and all(x not in {'.','..'} for x in rel.split('/')),'Canonical existing brick artifact')
        path=plain(PRIOR_P/'import'/rel)
        require(path.is_relative_to(PRIOR_P/'import/brick') and path.stat().st_size==item['bytes'] and item['bytes']<=2*1024**2 and digest(path)==item['sha256'],'Existing bounded partial readback identity')
        watched.append(path);npys+=path.suffix=='.npy'
    require(npys==13,'Exactly thirteen existing partial readbacks')
    return list(dict.fromkeys(watched))

def load_inputs(np):
    patch=prior.load_plan()['patches'][0];original={}
    for name,item in patch['inputfiles'].items():
        a=np.load(plain(item['path']),mmap_mode='r',allow_pickle=False);schema=patch['array_schemas'][name]
        require(a.dtype==np.dtype(schema['dtype']) and list(a.shape)==schema['shape'],'Original immutable brick schema');original[name]=a
    record=read_json(PRIOR_P/'import/brick/result.json')
    saved={Path(name).stem:np.load(plain(PRIOR_P/'import/brick'/name),mmap_mode='r',allow_pickle=False) for name in record['files'] if name.endswith('.npy')}
    failed=saved['failed_face_ids'];require(len(saved)==13 and len(failed)==4,'Exact saved rejection population')
    return original,saved,failed

def begin(stage='analysis'):
    require(stage in {'analysis','verify'},'Read-only analysis/verification phases only')
    require(sys.stdin.buffer.readline(64)==b'GO\n','Owned binary GO before numerical imports')
    require(digest(GUARD)==GUARD_SHA,'Unchanged CPU audit guard')
    spec=importlib.util.spec_from_file_location('saved_attribution_cpu_guard',GUARD)
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard);guard.P=P/stage
    sys.addaudithook(guard.runtime_guard)
    def extra(event,args):
        if event=='import' and str(args[0]).split('.')[0]=='meshlib':raise RuntimeError('No MeshLib/native geometry import')
        if event in {'os.chmod','os.chown'}:raise RuntimeError('No ownership/permission changes')
    sys.addaudithook(extra)
