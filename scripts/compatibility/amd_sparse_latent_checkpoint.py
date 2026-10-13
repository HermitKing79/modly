"""Exclusive bounded CPU safetensors latent, using the proven feature container checks."""
import hashlib
import json
import os
from pathlib import Path
import struct
from amd_feature_checkpoint import unique
import amd_sparse_sampling_contract as policy

SHAPE=(1,8,16,16,16);TENSOR_BYTES=131072;MAX_BYTES=256*1024
FIELDS={'component','model_checkpoint_sha256','model_config_sha256','model_source_sha256',
        'sampler_sources','params','seed','sigma_min','noise_sha256','feature_file_sha256','feature_provenance','harness_commit','runtime'}

def metadata(provenance,output_sha256):
    if (set(provenance)!=FIELDS or provenance['component']!='SparseStructureFlowModel'
        or provenance['seed']!=str(policy.SEED)
        or provenance['sigma_min']!=str(policy.SIGMA_MIN)
        or json.loads(provenance['params'])!=json.loads(json.dumps(policy.PARAMS))):
        raise ValueError('Latent provenance does not describe the exact approved sampler')
    result={**provenance,'output_sha256':output_sha256,'format':'modly-amd-sparse-structure-latent-v1',
            'shape':'[1,8,16,16,16]','dtype':'F32'}
    for key,value in result.items():
        if type(value) is not str or not value or len(value)>16384:raise ValueError('Invalid latent provenance: '+key)
        if key.endswith('sha256') or key=='harness_commit':
            if len(value)!=(40 if key=='harness_commit' else 64) or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Invalid latent provenance hash: '+key)
    return result

def owned_path(path,scratch):
    path=Path(path);scratch=Path(scratch).resolve(strict=True)
    if path.name!='sparse-structure-latent.safetensors' or path.is_symlink() or path.resolve()!=scratch/path.name:
        raise ValueError('Latent must be the exact exclusive owned scratch artifact')
    return path

def array_hash(array):
    import numpy as np
    if (type(array) is not np.ndarray or array.shape!=SHAPE or array.dtype!=np.dtype('<f4')
        or not array.flags.c_contiguous or array.nbytes!=TENSOR_BYTES or not np.isfinite(array).all()):
        raise ValueError('Latent requires contiguous finite CPU F32[1,8,16,16,16]')
    return hashlib.sha256(array.tobytes()).hexdigest()

def verify(path,scratch,provenance,output_sha256):
    path=owned_path(path,scratch)
    with path.open('rb') as stream:blob=stream.read(MAX_BYTES+1)
    if not TENSOR_BYTES+8<len(blob)<=MAX_BYTES:raise ValueError('Latent file exceeds exact payload/header limit')
    count=struct.unpack('<Q',blob[:8])[0]
    if not 0<count<=65536 or 8+count+TENSOR_BYTES!=len(blob):raise ValueError('Invalid bounded latent container length')
    header=json.loads(blob[8:8+count],object_pairs_hook=unique)
    if (set(header)!={'__metadata__','sparse_latent'} or header['__metadata__']!=metadata(provenance,output_sha256)
        or header['sparse_latent']!={'dtype':'F32','shape':list(SHAPE),'data_offsets':[0,TENSOR_BYTES]}):
        raise ValueError('Latent header/provenance mismatch')
    from safetensors.numpy import load
    tensors=load(blob)
    if set(tensors)!={'sparse_latent'} or array_hash(tensors['sparse_latent'])!=output_sha256:
        raise ValueError('Latent payload integrity failed')
    return dict(verified=True,shape=list(SHAPE),dtype='F32',tensor_bytes=TENSOR_BYTES,file_bytes=len(blob),
        file_sha256=hashlib.sha256(blob).hexdigest(),output_sha256=output_sha256,provenance=header['__metadata__'],path=str(path.resolve()))

def write(path,scratch,array,provenance):
    path=owned_path(path,scratch);output_sha256=array_hash(array)
    from safetensors.numpy import save
    blob=save({'sparse_latent':array},metadata=metadata(provenance,output_sha256))
    if len(blob)>MAX_BYTES:raise ValueError('Serialized latent exceeds256KiB')
    with path.open('xb') as stream:stream.write(blob);stream.flush();os.fsync(stream.fileno())
    return verify(path,scratch,provenance,output_sha256)
