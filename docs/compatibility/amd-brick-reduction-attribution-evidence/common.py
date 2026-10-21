"""Pins and read-only saved evidence; no simplifier or renderer imports."""
import sys
sys.dont_write_bytecode=True
import hashlib,importlib.util,json,os
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-brick-reduction-attribution-da37443'
OLD=ROOT/'docs/compatibility/amd-brick-half-poly-evidence'
PREVIOUS=ROOT.parent/'artifacts/amd-brick-half-poly-5705f9d-corrected'

def pin(name,path,sha):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Changed pinned source: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

original=pin('attribution_original_inputs',OLD/'inputs.py','a95b01eca1725684aec6233288dff297247e274c7b62641217604ba53cff3b5f')
plain,digest,require,save=original.plain,original.digest,original.require,original.save

def authenticate():
    watched=original.authenticate()
    watched.append(OLD/'inputs.py')
    contract=plain(OLD/'reduction_contract.py')
    require(digest(contract)=='bdf0fa90da2a4b93020dca1ec73423ec5b8b22b48f951460bd4e9d23a5d8babc','Original bounded oriented-key arithmetic');watched.append(contract)
    for name,sha in {
        'assessment-metrics.json':'e0bd6e0499154f98f3e2660e521ad05a102e91423fcecf9d0e9c7c0cc7a0b4ac',
        'independent-outcome.json':'ada53c177c36c91250368da799f3e5a62d2626e291bc1ff61ff65cc85411a30a',
        'outcome.json':'00a87b4d2636370b6861cc6f7d19aa8ead55afd73446d0f41e671c21a4cbbd5a',
    }.items():
        path=plain(OLD/name);require(digest(path)==sha,'Prior evidence identity');watched.append(path)
    receipt=plain(PREVIOUS/'assess/result.json')
    require(digest(receipt)=='b9eec3be7f05d1d1ba140e5e3931e36d0e47b2388c6df552d985a0e8f96d36f1','Actual prior assessment receipt')
    r=json.loads(receipt.read_text());require(r['classification']=='NO-GO','Preserve rejected status')
    watched.append(receipt)
    for name,row in r['files'].items():
        require(Path(name).name==name,'Safe recorded basename')
        path=plain(PREVIOUS/'assess'/name)
        require(path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],'Saved candidate data identity: '+name);watched.append(path)
    require(digest(PREVIOUS/'assess/candidate.ply')=='5024daaaa716a1bbf00d316f89ff47e5e6d30653a370b15a9aff7cb4fbf8fc99','Exact rejected PLY')
    for relative,sha in {
        'assess/cache/edge_keys.npy':'c499b2704b5a221d56aef478d575d92524b8903fc3ac930e241c958ea3c97490',
        'assess/cache/incidence.npy':'a3b7462ce82805a9bbb9247f215e250bd2ce49001a7b4ec58e9d6179fa59489b',
        'baseline/views.json':'86442d9a5b94c1d072723a5b22fde157f5a3f39e138f6f21b57bbad97f039a25',
    }.items():
        path=plain(PREVIOUS/relative);require(digest(path)==sha,'Saved graph/camera pinned by BEFORE preservation');watched.append(path)
    return list(dict.fromkeys(watched))

def begin():
    require(sys.stdin.buffer.readline(64)==b'GO\n','Owned binary GO required')
    auth=original.load_auth();auth.P=P;sys.addaudithook(auth.runtime_guard)
    def owned_writes(event,args):
        if event=='open':
            name,mode,flags=args
            if isinstance(name,(str,bytes,os.PathLike)) and ((isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))):
                path=Path(name).resolve()
                require(path.is_relative_to(P.resolve()),'Writes confined to new private diagnostics')
        if event in {'os.remove','os.rmdir','os.rename','os.link','os.symlink','os.chmod','os.chown'}:raise RuntimeError('Read-only investigation forbids mutation of existing files')
    sys.addaudithook(owned_writes)

def load(np):
    data=original.load(np)
    out=PREVIOUS/'assess'
    arrays=[np.load(out/(x+'.npy'),mmap_mode='r',allow_pickle=False) for x in ('vertices','faces','source-vertex-ids','color-bytes')]
    cv,cf,ids,rgba=arrays
    require(cv.shape==(975633,3) and cf.shape==(2083776,3) and ids.shape==(975633,) and rgba.shape==(975633,4),'Exact saved candidate contracts')
    require(ids.dtype==cf.dtype==np.int32 and cv.dtype==np.float32 and rgba.dtype==np.uint8,'Exact candidate dtypes')
    for start in range(0,len(ids),32768):
        s=slice(start,start+32768)
        require(np.array_equal(cv[s].view(np.uint32),data['vertices'][ids[s]].view(np.uint32)) and np.array_equal(rgba[s],data['native_bytes'][ids[s]]),'Exact original position/colour source IDs')
    return data,*arrays
