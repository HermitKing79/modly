"""Read-only source-ray check after a cheap patch-nomination rejection."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,importlib.util,os
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-brick-uv-bake-18c0c92'
PREV=ROOT/'docs/compatibility/amd-brick-defect-diagnosis-evidence'

def load(name,path,sha):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Pinned original source required')
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def main():
    if sys.stdin.buffer.readline(64)!=b'GO\n':raise ValueError('Owned host GO required')
    auth=load('original_auth',PREV/'diagnose.py','8fa7c617324ba636856b48f47a995ee64cd2a85f3c26d86d8760ec638b163529');auth.P=P
    sys.addaudithook(auth.runtime_guard)
    import numpy as np
    projection=load('original_projection',PREV/'projection.py','0ed20e6d03424404f3b3aebc64cae2825a0d1c8d241fd494ce50e742471eb840')
    for path,sha in auth.PINS.values():auth.require(auth.digest(path)==sha,'Unchanged original inputs required')
    v=np.load(auth.PINS['vertices'][0],mmap_mode='r');f=np.load(auth.PINS['faces'][0],mmap_mode='r')
    camera=next(q for q in auth.read_json(auth.PINS['render_metadata'][0])['views'] if q['name']=='pos-x')
    projected=projection.project(v,camera,1024,1024)
    pixels=[[580+dx,558+dy] for dy in (-4,0,4) for dx in (-4,0,4)]
    rays=projection.sample_rays(v,f,projected,pixels)
    record=dict(CPU_only=True,read_only=True,texture_bake=False,renderer=False,original_mesh_sha256=auth.PINS['mesh'][1],rays=rays)
    with (P/'nomination/nearest-original-rays.json').open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps([dict(pixel=r['pixel'],hit=r['hit'],face=r['face_id'],cosX=r.get('normal_cos_camera'),normal=r.get('normal')) for r in rays]),flush=True)

if __name__=='__main__':main()
