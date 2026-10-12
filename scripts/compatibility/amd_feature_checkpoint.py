"""One bounded CPU-only conditioning artifact; no model or GPU framework imports."""
import hashlib
import json
import os
from pathlib import Path
import struct

SHAPE=(1,1029,1024)
TENSOR_BYTES=4214784
MAX_BYTES=TENSOR_BYTES+16384
FIELDS={'component','checkpoint_sha256','config_sha256','input_pixel_sha256','input_png_sha256',
        'extractor_source_sha256','pipeline_source_sha256','harness_commit','runtime'}


def metadata(provenance,output_sha256):
    if set(provenance)!=FIELDS or provenance['component']!='DINOv3ViTModel':
        raise ValueError('Feature checkpoint requires complete DINOv3 provenance')
    for key,value in {**provenance,'output_sha256':output_sha256}.items():
        if type(value) is not str or not value or len(value)>2048:raise ValueError('Invalid feature provenance: '+key)
        if key.endswith('sha256') or key=='harness_commit':
            if len(value)!=(40 if key=='harness_commit' else 64) or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Invalid provenance hash: '+key)
    return {**provenance,'output_sha256':output_sha256,'format':'modly-amd-conditioning-v1',
            'shape':'[1,1029,1024]','dtype':'F32','conditioning_width':'1024','resolution':'512','alpha_only':'true'}


def owned_path(path,scratch):
    path=Path(path);scratch=Path(scratch).resolve(strict=True)
    if path.name!='conditioning.safetensors' or path.is_symlink() or path.resolve()!=scratch/'conditioning.safetensors':
        raise ValueError('Feature artifact must be the exact owned scratch/conditioning.safetensors path')
    return path


def array_hash(array):
    import numpy as np
    if (type(array) is not np.ndarray or array.shape!=SHAPE or array.dtype!=np.dtype('<f4')
            or not array.flags.c_contiguous or array.nbytes!=TENSOR_BYTES or not np.isfinite(array).all()):
        raise ValueError('Feature checkpoint requires contiguous finite CPU float32 [1,1029,1024]')
    return hashlib.sha256(array.tobytes()).hexdigest()


def unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate safetensors metadata field')
        result[key]=value
    return result


def verify(path,scratch,provenance,output_sha256):
    path=owned_path(path,scratch)
    with path.open('rb') as stream:blob=stream.read(MAX_BYTES+1)
    if not TENSOR_BYTES+8<len(blob)<=MAX_BYTES:raise ValueError('Feature file exceeds its exact payload/header bound')
    count=struct.unpack('<Q',blob[:8])[0]
    if not 0<count<=16384 or 8+count+TENSOR_BYTES!=len(blob):raise ValueError('Invalid feature container length')
    header=json.loads(blob[8:8+count],object_pairs_hook=unique)
    if (set(header)!={'__metadata__','cond'} or header['__metadata__']!=metadata(provenance,output_sha256)
            or header['cond']!={'dtype':'F32','shape':list(SHAPE),'data_offsets':[0,TENSOR_BYTES]}):
        raise ValueError('Feature tensor/provenance header mismatch')
    from safetensors.numpy import load
    tensors=load(blob)
    if set(tensors)!={'cond'} or array_hash(tensors['cond'])!=output_sha256:
        raise ValueError('Feature payload integrity validation failed')
    return dict(verified=True,shape=list(SHAPE),dtype='F32',tensor_bytes=TENSOR_BYTES,
                file_bytes=len(blob),file_sha256=hashlib.sha256(blob).hexdigest(),output_sha256=output_sha256,
                provenance=header['__metadata__'],path=str(path.resolve()))


def write(path,scratch,array,provenance):
    path=owned_path(path,scratch);output_sha256=array_hash(array)
    from safetensors.numpy import save
    blob=save({'cond':array},metadata=metadata(provenance,output_sha256))
    if len(blob)>MAX_BYTES:raise ValueError('Serialized feature artifact exceeds bound')
    # Exclusive creation preserves any existing or partially written artifact.
    with path.open('xb') as stream:
        stream.write(blob);stream.flush();os.fsync(stream.fileno())
    return verify(path,scratch,provenance,output_sha256)
