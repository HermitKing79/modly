"""Pinned reuse and exclusive private receipts for this one CPU patch proof."""
import sys,hashlib,importlib.util,json,os,struct
from pathlib import Path
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-brick-seam-uv-bake-87f3755'
OLD=ROOT/'docs/compatibility/amd-brick-uv-bake-evidence'
PINS={'bake_patch.py':'c0b53da02346b7bd60eaa2fa07b33a1b6573cd8f5e43944d0e9faff20fc1a174',
      'sampler.py':'0a330eca2c991d3b7751d3cbbbe94edac317b34b28ce7216d9d3d374ef20b0c6',
      'patch_math.py':'bad966911d5b428c4cbcc9fd22b50868fcd8a90df79e5b258bdaeb69626c4364',
      'render_patch.py':'96ad2138bb791b12557c9557348f8ad5a351df026774beea158c8f56924127e3',
      'source-semantics.json':'094198de62098b413a503f9dae2f81c6de4f9ddc1ebc313d98e49f7f739ee79f'}
AUTH=ROOT/'docs/compatibility/amd-brick-defect-diagnosis-evidence/diagnose.py'
AUTH_SHA='8fa7c617324ba636856b48f47a995ee64cd2a85f3c26d86d8760ec638b163529'

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load(name,path,sha):
    if digest(path)!=sha:raise ValueError('Pinned source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def originals():
    for name,sha in PINS.items():
        if digest(OLD/name)!=sha:raise ValueError('Historical implementation changed: '+name)
    sys.path.insert(0,str(OLD))
    return load('original_patch_math',OLD/'patch_math.py',PINS['patch_math.py'])

def begin():
    if sys.stdin.buffer.readline(64)!=b'GO\n':raise ValueError('Exact owned-host binary GO required')
    auth=load('original_readonly_auth',AUTH,AUTH_SHA);auth.P=P;sys.addaudithook(auth.runtime_guard)
    return auth

def identity_inputs(auth):
    r={k:auth.identity(path) for k,(path,_) in auth.PINS.items()}
    if any(r[k]['sha256']!=sha for k,(_,sha) in auth.PINS.items()):raise ValueError('Original input identity changed')
    return r

def save(path,value):
    text=json.dumps(value,indent=2,allow_nan=False)+'\n'
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:f.write(text);f.flush();os.fsync(f.fileno())

def np_save(path,array):
    import numpy as np
    with Path(path).open('xb') as f:np.save(f,array,allow_pickle=False);f.flush();os.fsync(f.fileno())
    if not np.array_equal(np.load(path,allow_pickle=False),array):raise ValueError('Exact NPY readback failed')

def manifest(directory,names):
    return {n:dict(bytes=(directory/n).stat().st_size,sha256=digest(directory/n)) for n in names}

def verify(directory,receipt):
    r=json.loads((directory/receipt).read_text())
    if r['classification']!='PASS':raise ValueError('Prior qualified stage required: '+receipt)
    for n,row in r['files'].items():
        p=directory/n
        if p.resolve().parent!=directory.resolve() or p.is_symlink() or not p.is_file() or p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:raise ValueError('Producer artifact changed: '+n)
    return r

def tensor(path,name,np):
    with path.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]
        if not 0<length<65536:raise ValueError('Bounded authenticated tensor header')
        t=json.loads(f.read(length))[name]
    return np.memmap(path,mode='r',offset=8+length+t['data_offsets'][0],dtype={'F32':'<f4','I32':'<i4'}[t['dtype']],shape=tuple(t['shape']))
