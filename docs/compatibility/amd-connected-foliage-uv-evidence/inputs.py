"""Exact private selection and unchanged original geometry/material admission."""
import sys
sys.dont_write_bytecode=True
import hashlib,importlib.util,json,os
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-connected-foliage-bad7ba4'
S=ROOT.parent/'artifacts/amd-bounded-foliage-64d05aa/run'
OLD=ROOT/'docs/compatibility/amd-brick-pbr-mip-evidence'
sys.path.insert(0,str(OLD))
from assets import authenticate as original_authenticate,load_auth,digest,plain,save,require


def selection_receipt():
    path=plain(S/'result.json')
    require(digest(path)=='8ed7861ef47aeb5e7468eddff1df886286c36c49542a4f61ea48d628e4faede0','Exact saved selection result')
    receipt=json.loads(path.read_text())
    require(receipt['selection']['selected_faces']==4531 and receipt['selection']['vertices']==3177,'Only frozen 4531-face selection')
    for name,row in receipt['files'].items():
        require(Path(name).name==name,'Single saved basename')
        item=plain(S/name)
        require(item.stat().st_size==row['bytes'] and digest(item)==row['sha256'],'Saved selection identity: '+name)
    return receipt


def authenticate():
    auth,identities,watched,receipts=original_authenticate()
    receipt=selection_receipt()
    watched += [S/'result.json',S/'selection.json',S/'selection-admission.json']+[S/name for name in receipt['files']]
    return auth,identities,list(dict.fromkeys(watched)),receipts


def load_selection(np):
    auth,identities,watched,_=authenticate()
    values={n:np.load(plain(S/(n+'.npy')),allow_pickle=False) for n in ('vertices','faces','source-face-ids','source-vertex-ids','direct')}
    v,f,sf,sv,direct=[values[n] for n in ('vertices','faces','source-face-ids','source-vertex-ids','direct')]
    require(v.shape==(3177,3) and v.dtype==np.float32 and f.shape==(4531,3) and f.dtype==np.int32,'Actual saved geometry schema')
    require(sf.shape==(4531,) and sf.dtype==np.int32 and sv.shape==(3177,) and sv.dtype==np.int64,'Actual original source-ID schemas')
    original_v=np.load(auth.PINS['vertices'][0],mmap_mode='r',allow_pickle=False)
    original_f=np.load(auth.PINS['faces'][0],mmap_mode='r',allow_pickle=False)
    require(np.array_equal(sv[f],original_f[sf]),'Exact triangle source IDs/order/winding/multiplicity')
    require(v.tobytes()==original_v[sv].tobytes(),'Exact original vertex positions')
    require(np.isfinite(v).all() and np.isfinite(direct).all(),'Finite original positions/materials')
    material=auth.material_features(auth.PINS['material'][0],np)
    require(direct.shape==(3177,6) and direct.dtype==np.float32 and direct.tobytes()==material[sv].tobytes(),'Exact saved learned material rows')
    common_path=ROOT/'docs/compatibility/amd-brick-seam-uv-bake-evidence/common.py'
    require(digest(common_path)=='e239c7e031bfa5a072d0c02b4804d96b73ed8ca7dc55145be46882ca0b02e96b','Original tensor reader identity')
    spec=importlib.util.spec_from_file_location('connected_original_common',common_path)
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    require(np.array_equal(common.tensor(auth.PINS['geometry'][0],'coordinates',np),common.tensor(auth.PINS['material'][0],'coordinates',np)),'Full genuine geometry/material coordinate-row correspondence')
    return values,identities,watched


def array(path,value):
    import numpy as np
    with Path(path).open('xb') as stream:
        np.save(stream,value,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
    read=np.load(path,allow_pickle=False)
    require(read.shape==value.shape and read.dtype==value.dtype and read.tobytes()==value.tobytes(),'Byte-exact saved array readback')
