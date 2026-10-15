"""Production tensor-file I/O and one bounded independent host receipt."""
import hashlib
import json
from math import prod
import os
from pathlib import Path
import time
from artifact_paths import E,leaf,check_host

MAX_OUTPUT=256*1024**2
MAX_HEADER=128*1024

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def durable_json(path,record):
    path=Path(path)
    if os.path.lexists(path):raise FileExistsError(str(path))
    pending=path.with_name(path.name+'.publishing')
    with pending.open('x',encoding='utf-8') as f:
        json.dump(record,f,indent=2);f.flush();os.fsync(f.fileno())
    os.rename(pending,path)  # Both paths are within the same fresh owned root.

def marker(kind):return E/('neural-attempt.marker' if kind=='neural' else 'writer-acceptance.marker')

def write_output(tensors,torch,provenance,summary,kind,receipt_dir):
    from safetensors.torch import save_file,load_file
    from safetensors import safe_open
    target=leaf(kind);core=dict(summary)
    target.mkdir(exist_ok=False)  # Parent is already attested; never create ancestors.
    path=target/'turret-neural-output.safetensors'
    save_file(tensors,str(path),metadata={'provenance':json.dumps(provenance,sort_keys=True),'summary':json.dumps(core,sort_keys=True)})
    if path.stat().st_size>MAX_OUTPUT+MAX_HEADER:raise RuntimeError('Artifact file size exceeded')
    with path.open('rb') as f:
        header=int.from_bytes(f.read(8),'little');os.fsync(f.fileno())
    if not 0<header<=MAX_HEADER:raise RuntimeError('Unbounded output header')
    with safe_open(str(path),framework='pt',device='cpu') as f:
        if f.metadata()!={'provenance':json.dumps(provenance,sort_keys=True),'summary':json.dumps(core,sort_keys=True)}:
            raise RuntimeError('Artifact metadata readback failed')
    restored=load_file(str(path),device='cpu')
    if set(restored)!=set(tensors) or any(not torch.equal(tensors[k],restored[k]) for k in tensors):
        raise RuntimeError('Tensor artifact readback failed')
    raw={k:hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest() for k,t in tensors.items()}
    if any(raw[k]!=hashlib.sha256(restored[k].view(torch.uint8).numpy().tobytes()).hexdigest() for k in tensors):
        raise RuntimeError('Raw tensor byte readback failed')
    restored=None
    summary=dict(core,path=str(path),bytes=path.stat().st_size,sha256=digest(path),tensor_payload_sha256=raw,verified=True)
    durable_json(target/'provenance.json',{'provenance':provenance,'output':summary})
    ready={'kind':kind,'summary':summary,'core_summary':core,'provenance':provenance,'marker_sha256':digest(marker(kind))}
    durable_json(receipt_dir/'artifact-ready.json',ready)
    deadline=time.monotonic()+10
    ack=receipt_dir/'host-artifact-verified.json'
    while not ack.exists():
        if time.monotonic()>=deadline:raise TimeoutError('Independent artifact verification deadline')
        time.sleep(.025)
    verified=json.loads(ack.read_text())
    if verified.get('status')!='PASS' or verified.get('sha256')!=summary['sha256'] or verified.get('bytes')!=summary['bytes']:
        raise RuntimeError('Independent host artifact verification refused')
    summary['independently_verified_before_release']=True
    return summary

def expected_schema(kind):
    n=1842968 if kind=='neural' else 2
    schema={'features':('F32',[n,7]),'coordinates':('I32',[n,4])}
    for i,rows in enumerate([3743,17015,79169,378891] if kind=='neural' else [2]*4):
        schema[f'subdivision_{i}_logits']=('BF16',[rows,8])
        schema[f'subdivision_{i}_coordinates']=('I32',[rows,4])
    return schema

def verify_on_host(receipt_dir,kind):
    check_host(kind,existing=True)
    ready_path=receipt_dir/'artifact-ready.json'
    if ready_path.stat().st_size>MAX_HEADER:raise RuntimeError('Unbounded artifact receipt')
    ready=json.loads(ready_path.read_text())
    if ready['kind']!=kind or ready['marker_sha256']!=digest(marker(kind)):raise RuntimeError('Artifact attempt identity differs')
    provenance=ready['provenance']
    if kind=='neural':
        if (provenance.get('checkpoint_sha256')!='e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581'
            or provenance.get('resolution')!=512 or provenance.get('forward_calls')!=1
            or provenance.get('convolutions')!=40 or provenance.get('mesh_extraction') is not False):
            raise RuntimeError('Genuine artifact provenance differs')
    elif provenance.get('fixture_kind')!='CPU_WRITER_ACCEPTANCE' or provenance.get('neural_generation') is not False:
        raise RuntimeError('CPU fixture provenance differs')
    summary=ready['summary'];path=leaf(kind)/'turret-neural-output.safetensors'
    guest='/mnt/c/Users/wattz/Documents/ChatGPT/modly/artifacts/'+leaf(kind).name+'/turret-neural-output.safetensors'
    if summary['path']!=guest or not 0<summary['bytes']<=MAX_OUTPUT+MAX_HEADER:raise RuntimeError('Artifact receipt path/size differs')
    before=path.stat()
    if before.st_size!=summary['bytes'] or digest(path)!=summary['sha256']:raise RuntimeError('Independent artifact hash differs')
    with path.open('rb') as f:
        length=int.from_bytes(f.read(8),'little')
        if not 0<length<=MAX_HEADER:raise RuntimeError('Unbounded safetensors header')
        header=json.loads(f.read(length));base=8+length;payload=before.st_size-base
        meta=header.pop('__metadata__')
        if meta!={'provenance':json.dumps(ready['provenance'],sort_keys=True),'summary':json.dumps(ready['core_summary'],sort_keys=True)}:
            raise RuntimeError('Independent metadata differs')
        schema=expected_schema(kind)
        if set(header)!=set(schema) or set(summary['tensor_payload_sha256'])!=set(schema):raise RuntimeError('Independent tensor closure differs')
        ranges=[]
        for key,(dtype,shape) in schema.items():
            row=header[key];start,end=row['data_offsets'];size=prod(shape)*{'F32':4,'I32':4,'BF16':2}[dtype]
            if (row['dtype']!=dtype or row['shape']!=shape or type(start)is not int or type(end)is not int
                or not 0<=start<end<=payload or end-start!=size):raise RuntimeError('Tensor schema/offset differs: '+key)
            ranges.append((start,end));f.seek(base+start);h=hashlib.sha256();remaining=size
            while remaining:
                chunk=f.read(min(remaining,1024**2))
                if not chunk:raise RuntimeError('Truncated tensor payload')
                h.update(chunk);remaining-=len(chunk)
            if h.hexdigest()!=summary['tensor_payload_sha256'][key]:raise RuntimeError('Independent tensor hash differs: '+key)
    ranges.sort()
    if ranges[0][0]!=0 or ranges[-1][1]!=payload or any(a[1]!=b[0] for a,b in zip(ranges,ranges[1:])):
        raise RuntimeError('Payload has gaps/overlaps')
    document=json.loads((path.parent/'provenance.json').read_text())
    if document!={'provenance':ready['provenance'],'output':summary}:raise RuntimeError('Independent provenance differs')
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Artifact changed while reading')
    return {'status':'PASS','kind':kind,'Windows_path':str(path),'bytes':after.st_size,'sha256':summary['sha256'],
            'tensor_payload_sha256':summary['tensor_payload_sha256'],'metadata_verified':True,'payload_bytes':payload,
            'provenance_json_path':str(path.parent/'provenance.json'),'provenance_json_bytes':(path.parent/'provenance.json').stat().st_size,
            'provenance_json_sha256':digest(path.parent/'provenance.json'),'marker_sha256':ready['marker_sha256'],'independent_of_worker':True}

class HostReceipt:
    def __init__(self,output,kind,base):
        self.directory=output/'guest';self.kind=kind;self.base=base
        self.future=None;self.started=None;self.done=False
    def poll(self):
        if self.done:return
        if self.future is None and (self.directory/'artifact-ready.json').exists():
            self.started=time.monotonic()
            self.future=self.base.in_thread(lambda:verify_on_host(self.directory,self.kind))
        if self.future is not None:
            if time.monotonic()-self.started>=9:raise TimeoutError('Host artifact verification deadline')
            if self.future.done():
                durable_json(self.directory/'host-artifact-verified.json',self.future.result())
                self.done=True
