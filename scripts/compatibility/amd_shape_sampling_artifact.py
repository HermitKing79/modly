"""Bounded CPU-only numeric artifacts; no GPU/model/library initialization."""
import hashlib,json,os
from pathlib import Path
import numpy as np
from amd_shape_slat_contract import validate_coords
MAX_TOKENS=3743
def validate(x):
    if (type(x) is not np.ndarray or x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=32
        or not 0<x.shape[0]<=MAX_TOKENS or not x.flags.c_contiguous or not np.isfinite(x).all()):raise ValueError('Requires bounded finite contiguous F32 [N,32]')
def denormalize(x,mean,std):
    validate(x)
    if (type(mean) is not np.ndarray or type(std) is not np.ndarray or mean.shape!=(32,) or std.shape!=(32,)
        or mean.dtype!=np.float32 or std.dtype!=np.float32 or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0)):
        raise ValueError('Normalization requires exact32 finite F32 means/positive stds')
    result=np.ascontiguousarray(x*std[None]+mean[None]);validate(result);return result
def write(scratch,raw,normalized,coords,provenance,*,noise=None):
    scratch=Path(scratch).resolve(strict=True);validate(raw);validate(normalized)
    if raw.shape!=normalized.shape:raise ValueError('Latent layouts differ')
    validate_coords(coords,len(raw))
    from amd_forward_diagnostics import validate_json
    validate_json(provenance)
    rows={}
    arrays=[('shape-latent-raw.npy',raw),('shape-latent-denormalized.npy',normalized),('shape-latent-coordinates.npy',coords)]
    if noise is not None:
        validate(noise)
        if noise.shape!=raw.shape:raise ValueError('Noise layout differs')
        arrays.append(('shape-initial-noise.npy',noise))
    for name,array in arrays:
        path=scratch/name
        if path.is_symlink():raise ValueError('Artifact symlink prohibited')
        with path.open('xb') as f:np.save(f,array,allow_pickle=False);f.flush();os.fsync(f.fileno())
        read=np.load(path,allow_pickle=False)
        if not np.array_equal(read,array) or read.dtype!=array.dtype:raise ValueError('Latent artifact readback differs')
        blob=path.read_bytes()
        if len(blob)>MAX_TOKENS*32*4+1024:raise ValueError('Artifact exceeds bound')
        rows[name]=dict(path=str(path),file_bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),
            payload_sha256=hashlib.sha256(array.tobytes()).hexdigest(),shape=list(array.shape),dtype=str(array.dtype),verified=True)
    blob=json.dumps(provenance,indent=2,allow_nan=False).encode('utf-8')
    if len(blob)>65536:raise ValueError('Provenance exceeds bound')
    path=scratch/'shape-latent-provenance.json'
    if path.is_symlink():raise ValueError('Provenance symlink prohibited')
    with path.open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
    if json.loads(path.read_text())!=provenance:raise ValueError('Provenance readback differs')
    rows[path.name]=dict(path=str(path),file_bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),verified=True)
    return rows
