"""Dynamic ten-tensor output with bounded stdlib readback before GPU release.

The Windows supervisor must independently verify its copied artifact. This
guest readback does not represent that later host verification.
"""
import hashlib
import json
from math import prod
import os
from pathlib import Path
from run_contract import digest, support_counts, artifact_binding

MAX_OUTPUT = 256*1024**2
MAX_HEADER = 128*1024
FILENAME = 'shape-decoded.safetensors'


def output_metadata(provenance,summary,contract=None):
    result={'provenance':json.dumps(provenance,sort_keys=True),
            'summary':json.dumps(summary,sort_keys=True)}
    if contract is not None and contract.document()['schema']=='modly.amd.current-image-neural-run.v1':
        from run_contract import _current_api
        result['product_context']=json.dumps(_current_api().binding_document(contract)['product_context'],sort_keys=True)
    return result


def durable_json(path, record):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())


def expected_schema(summary,contract=None):
    n=summary['rows']; stages=summary['subdivisions']
    support_counts(n,[s['parents'] for s in stages],contract)
    schema={'features':('F32',[n,7]),'coordinates':('I32',[n,4])}
    for i,stage in enumerate(stages):
        schema[f'subdivision_{i}_logits']=('BF16',[stage['parents'],8])
        schema[f'subdivision_{i}_coordinates']=('I32',[stage['parents'],4])
    return schema


def verify_file(path, provenance, summary, tensor_hashes,contract=None):
    """Independent of Torch/safetensors readers; exact schema, metadata and bytes."""
    artifact_binding(provenance,contract)
    schema=expected_schema(summary,contract)
    path=Path(path);before=path.stat()
    if path.is_symlink() or not path.is_file() or not 8<before.st_size<=MAX_OUTPUT+MAX_HEADER:
        raise ValueError('Bounded regular neural artifact required')
    with path.open('rb') as stream:
        length=int.from_bytes(stream.read(8),'little')
        if not 0<length<=MAX_HEADER:raise ValueError('Unbounded output header')
        header=json.loads(stream.read(length));payload=before.st_size-8-length
        if header.pop('__metadata__')!=output_metadata(provenance,summary,contract):
            raise ValueError('Readback metadata differs')
        if set(header)!=set(schema) or set(tensor_hashes)!=set(schema):
            raise ValueError('Readback tensor closure differs')
        ranges=[]
        for key,(dtype,shape) in schema.items():
            row=header[key];start,end=row['data_offsets'];size=prod(shape)*{'F32':4,'I32':4,'BF16':2}[dtype]
            if (row['dtype']!=dtype or row['shape']!=shape or type(start)is not int or type(end)is not int
                    or not 0<=start<end<=payload or end-start!=size):
                raise ValueError('Readback tensor layout differs: '+key)
            ranges.append((start,end));stream.seek(8+length+start);remaining=size;h=hashlib.sha256()
            while remaining:
                block=stream.read(min(remaining,1024**2))
                if not block:raise ValueError('Truncated output tensor')
                h.update(block);remaining-=len(block)
            if h.hexdigest()!=tensor_hashes[key]:raise ValueError('Readback tensor payload differs: '+key)
        ranges.sort()
        if (not 0<payload<=MAX_OUTPUT or ranges[0][0]!=0 or ranges[-1][1]!=payload
                or any(a[1]!=b[0] for a,b in zip(ranges,ranges[1:]))):
            raise ValueError('Readback payload has gaps, overlaps or excess size')
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Output changed during readback')
    if contract is not None and contract.document()['schema']=='modly.amd.current-image-neural-run.v1':
        # Independent CPU reader verifies finite values and all four original
        # coordinate chains, including the current input payload identity.
        from amd_generation_mesh import read_neural
        arrays,_=read_neural(path,digest(path),contract=contract)
        try: pass
        finally:
            for array in arrays.values(): array._mmap.close()
    return {'verified':True,'bytes':after.st_size,'sha256':digest(path),'payload_bytes':payload,
            'tensor_payload_sha256':dict(tensor_hashes),'reader':'bounded stdlib tensor header/payload reader'}


def write_output(tensors, torch, provenance, summary, contract):
    from safetensors.torch import save_file,load_file
    from artifact_paths import attest_output
    artifact_binding(provenance,contract)
    expected_schema(summary,contract)
    if (provenance.get('run_contract_sha256')!=contract.sha256
            or provenance.get('source_job_sha256')!=contract.source_job_sha256
            or summary['subdivisions'][0]['parents']!=contract.initial_count):
        raise ValueError('Output run provenance differs')
    target=attest_output(contract);path=target/FILENAME
    if path.exists():raise FileExistsError(str(path))
    save_file(tensors,str(path),metadata=output_metadata(provenance,summary,contract))
    with path.open('r+b') as stream:os.fsync(stream.fileno())
    restored=load_file(str(path),device='cpu')
    hashes={k:hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest() for k,t in tensors.items()}
    if (set(restored)!=set(tensors) or any(not torch.equal(restored[k],v) or restored[k].dtype!=v.dtype for k,v in tensors.items())):
        raise ValueError('Safetensors output readback differs')
    del restored
    verified=verify_file(path,provenance,summary,hashes,contract)
    result=dict(summary,path=str(path),**{k:v for k,v in verified.items() if k!='payload_bytes'},
                readback_verified_before_release=True,host_verification_pending=True)
    durable_json(target/'provenance.json',{'provenance':provenance,'output':result,
                                         'run_contract':contract.document()})
    return result
