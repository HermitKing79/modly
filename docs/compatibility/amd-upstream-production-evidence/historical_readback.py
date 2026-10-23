"""Historical-only five-file readback; never issues a production observation.

This authenticates the published 3743-row archive, not a fresh product request
or the current installed checkpoints. It launches no worker or model.
"""
import copy
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
BASE='411055a6218d64de187e74b3587822fb5d2b2883'
ARCHIVE=ROOT/'docs/compatibility/amd-shape-slat-512-sampling-evidence'
sys.path[:0]=[str(ROOT/'api'),str(ROOT/'scripts/compatibility')]
class NoAI(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'torch','triton','trellis2','flex_gemm','transformers'}:
            raise AssertionError('AI/native import forbidden in historical readback: '+name)
sys.meta_path.insert(0,NoAI())

def sha(blob):return hashlib.sha256(blob).hexdigest()

def published_json(path):
    from services.amd_completed_result import _opened,_json
    with _opened(path,8*1024**2) as (stream,_):blob=stream.read(8*1024**2+1)
    saved=subprocess.check_output(['git','show',BASE+':'+path.relative_to(ROOT).as_posix()],cwd=ROOT,timeout=15)
    if blob!=saved:raise ValueError('Historical published JSON changed')
    return _json(blob),sha(blob)

def verify():
    import numpy as np
    from services.amd_completed_result import _opened,_json
    import amd_shape_sampling_contract as policy
    from amd_generation_shape import require_released
    from amd_shape_sampling_artifact import validate,denormalize
    from amd_shape_slat_contract import validate_coords
    result,result_sha=published_json(ARCHIVE/'run/results.json')
    pins,pins_sha=published_json(ARCHIVE/'source-contract.json')
    worker=require_released(result,'shape')
    # The original classifier remains the sampler/count/release check. It is
    # not treated as a payload reader or independent supervisor authority.
    if policy.classify(worker,True,True,True)!='PASS':raise ValueError('Historical sampler receipt failed')
    count=policy.TOKEN_COUNT
    spec=importlib.util.spec_from_file_location('_historical_shape_npy_reader',
        ROOT/'experimental/extensions/trellis2-amd/inference_decoder/run_contract.py')
    reader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reader;spec.loader.exec_module(reader)
    arrays={};summary={}
    expected={'shape-latent-raw.npy':('float32',(count,32)),
              'shape-latent-denormalized.npy':('float32',(count,32)),
              'shape-latent-coordinates.npy':('int32',(count,4)),
              'shape-initial-noise.npy':('float32',(count,32))}
    if set(worker['artifacts'])!=set(expected)|{'shape-latent-provenance.json'}:
        raise ValueError('Original five-artifact closure differs')
    for name,(dtype,shape) in expected.items():
        row=worker['artifacts'][name];path=ARCHIVE/name
        # These are the preserved archive copies, not current output paths.
        with _opened(path,2*1024**2) as (_,info):
            if info.st_size!=row['file_bytes']:raise ValueError('Historical file size differs')
        array=reader.read_npy(path,row['sha256'],shape,dtype)
        payload=sha(array.tobytes())
        if payload!=row['payload_sha256']:raise ValueError('Historical payload differs')
        if name=='shape-latent-coordinates.npy':validate_coords(array,count)
        else:validate(array)
        arrays[name]=array
        summary[name]=dict(sha256=row['sha256'],payload_sha256=payload,bytes=row['file_bytes'],shape=list(shape),dtype=dtype)
    for name,key in [('shape-latent-raw.npy',worker['output']['sha256']),
                     ('shape-initial-noise.npy',worker['input']['x']['sha256']),
                     ('shape-latent-coordinates.npy',worker['input']['coords']['sha256'])]:
        if summary[name]['payload_sha256']!=key:raise ValueError('Saved input/output relationship differs')
    norm=pins['normalization']
    if not np.array_equal(denormalize(arrays['shape-latent-raw.npy'],np.asarray(norm['mean'],dtype='f4'),
                                    np.asarray(norm['std'],dtype='f4')),arrays['shape-latent-denormalized.npy']):
        raise ValueError('Original F32 denormalization differs')
    name='shape-latent-provenance.json';row=worker['artifacts'][name]
    with _opened(ARCHIVE/name,65536) as (stream,info):blob=stream.read(65537)
    if info.st_size!=row['file_bytes'] or sha(blob)!=row['sha256']:raise ValueError('Historical provenance bytes differ')
    actual=_json(blob)
    sequence=copy.deepcopy(worker['sampling'])
    sequence['models_sha256']=sha(json.dumps(sequence.pop('models'),sort_keys=True).encode())
    expected_provenance=dict(harness_commit=result['process']['harness_commit'],checkpoint=worker['provenance'],
        input=worker['input'],parameters=policy.PARAMS,sigma_min=policy.SIGMA_MIN,seed=policy.SEED,
        sampler_sources=pins['sampler_sources'],normalization=norm,sequence=sequence,raw_output=worker['output'])
    expected_provenance=json.loads(json.dumps(expected_provenance))
    if actual!=expected_provenance:raise ValueError('Original provenance construction differs')
    summary[name]=dict(sha256=row['sha256'],bytes=row['file_bytes'])
    return dict(classification='HISTORICAL_ONLY_READBACK_PASS',rows=count,artifacts=summary,
        published_result_sha256=result_sha,published_source_contract_sha256=pins_sha,
        all_four_payloads_finite=True,original_coordinate_validation=True,
        canonical_F32_denormalization_exact=True,original_provenance_reconstruction_exact=True,
        sampler_forwards=21,sampler_steps=12,noise_matches_saved_input_hash=True,
        noise_rng_independently_reproduced=False,current_checkpoint_bytes_verified=False,
        fresh_product_job=False,supervisor_authority_established=False,
        original_historical_verdict=result['classification'])

if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
