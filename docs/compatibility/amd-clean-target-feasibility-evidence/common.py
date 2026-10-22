"""Bounded exact patch import qualification; no remesher or model loader."""
import sys
sys.dont_write_bytecode=True
import hashlib,importlib.util,json,os
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-clean-target-feasibility-d10c18e'
PLAN_SHA='aeb98fd9411ab718d7fd439f6345ad2e10927f55371a4d3c6bce4f22b39204bb'
GUARD=ROOT/'docs/compatibility/amd-brick-defect-diagnosis-evidence/diagnose.py'
GUARD_SHA='8fa7c617324ba636856b48f47a995ee64cd2a85f3c26d86d8760ec638b163529'
def require(ok,message):
    if not ok:raise ValueError(message)
def plain(path):
    path=Path(path).absolute()
    for q in (path,*path.parents):
        s=q.lstat();require(not q.is_symlink() and not getattr(s,'st_file_attributes',0)&0x400,'Plain non-reparse path required')
    require(not path.is_file() or path.stat().st_nlink==1,'Single-link file required');return path
def digest(path):
    with plain(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
def read_json(path):
    path=plain(path);require(path.stat().st_size<=2*1024**2,'Bounded JSON')
    def pairs(rows):
        value={}
        for key,item in rows:require(key not in value,'Unique JSON keys');value[key]=item
        return value
    return json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=pairs,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Finite JSON')))
def load_plan():
    path=plain(P/'input-manifest.json');require(digest(path)==PLAN_SHA,'Exact independently authenticated patch manifest')
    plan=read_json(path);require([x['name'] for x in plan['patches']]==['brick','foliage'],'Only the two frozen patches')
    for patch,(vertices,faces) in zip(plan['patches'],[(457,781),(3177,4531)]):
        require((patch['vertex_count'],patch['triangle_count'])==(vertices,faces),'Exact accepted patch counts')
        require(set(patch['inputfiles'])=={'vertices','faces','source_vertex_ids','source_face_ids'},'Four exact correspondence arrays')
        for item in patch['inputfiles'].values():
            p=plain(item['path']);require(p.is_relative_to(ROOT.parent/'artifacts') and p.stat().st_size==item['bytes'] and p.stat().st_size<=1024**2 and digest(p)==item['sha256'],'Exact bounded original patch file')
    return plan
def authenticate():
    plan=load_plan();watched=[P/'input-manifest.json',GUARD,E/'probe-contract.json',E/'patch-authentication.json',E/'installed-method-audit.json']
    require(digest(GUARD)==GUARD_SHA,'Unchanged CPU audit guard')
    require(digest(E/'probe-contract.json')=='056de8c75c7c09f1627ba9221b5174dc4b28b8e42c6a80a21bc40a620ac8da30','Frozen mission gates')
    for patch in plan['patches']:watched.extend(Path(row['path']) for row in patch['inputfiles'].values())
    libraries=read_json(E/'constructor-library-pins.json')
    require(libraries['CPU_import_admitted'] is True,'Static CPU dependency review required')
    for row in libraries['files'].values():
        p=plain(row['path']);require(p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],'Audited native/source library identity');watched.append(p)
    return list(dict.fromkeys(watched+[E/'constructor-library-pins.json']))
def begin():
    require(sys.stdin.buffer.readline(64)==b'GO\n','Owned binary GO before numerical/native imports')
    require(digest(GUARD)==GUARD_SHA,'Pinned original CPU guard')
    spec=importlib.util.spec_from_file_location('clean_target_cpu_guard',GUARD);guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard);guard.P=P
    sys.addaudithook(guard.runtime_guard)
    def extra(event,args):
        if event=='import' and str(args[0]).startswith(('meshlib.mrcudapy','meshlib.mrviewerpy')):raise RuntimeError('No GPU or viewer imports')
        if event in {'os.remove','os.rmdir','os.rename','os.link','os.symlink','os.chmod','os.chown'}:raise RuntimeError('No existing-file mutation or linking')
    sys.addaudithook(extra)
