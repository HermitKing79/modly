"""One strict decoder forward on the committed latent; reuse existing supervisor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).parent))
import amd_sparse_decoder_forward as base
import amd_sparse_occupancy_contract as contract
import amd_sparse_occupancy_data as data
import amd_blaslt_teardown as teardown

EVIDENCE=base.ROOT/'docs/compatibility/amd-sparse-occupancy-evidence'
CACHE_EVIDENCE=base.EVIDENCE
PIPELINE=base.SOURCE/'trellis2/pipelines/trellis2_image_to_3d.py'
PIPELINE_SHA='faade40274e44823e72e1659b8ced96f8815ac9e08b3abb89271077c4394982b'

def admission():
    import psutil
    available=psutil.virtual_memory().available;required=4*1024**3
    if available<required:raise RuntimeError('Decoder CPU admission requires4GiB available RAM')
    return dict(available_bytes=available,required_bytes=required,admitted=True)

def cache_plan():
    plan=json.loads((CACHE_EVIDENCE/'cache-plan.json').read_text())
    baseline=json.loads((base.ROOT/'docs/compatibility/amd-forward-timeout-investigation-evidence/inventory-before.json').read_text())
    if base.cache_reuse.attest_runtime(base.cache_reuse.SITE,baseline)!=plan['runtime_libraries']:raise RuntimeError('Native runtime cache attestation changed')
    base.cache_reuse.verify_copies(CACHE_EVIDENCE/'prepared-cache',plan['selected'])
    return plan

def private_cache(scratch):
    plan=cache_plan();staging=scratch.parent/'cache-staging'
    base.cache_reuse.copy_entries(CACHE_EVIDENCE/'prepared-cache',staging,plan['selected'])
    for name in ('amd_comgr_cache_dir','miopen_custom_cache_dir'):
        source=staging/name;target=scratch/name
        source.resolve().relative_to(EVIDENCE.resolve());target.resolve().relative_to(EVIDENCE.resolve())
        source.rename(target)
    staging.rmdir();base.cache_reuse.verify_copies(scratch,plan['selected'])
    return dict(policy='fresh verified eligible copies; historical files are never live cache paths',copied_entries=len(plan['selected']))

def identity():
    current=base.previous.identity();old=json.loads(base.OLD.read_text())['worker']['identity']
    if current['checkpoint_sha256']!=old['checkpoint_sha256']:raise RuntimeError('Previously proven decoder checkpoint changed')
    contract.fixture(current['config']);base.source_identity();teardown.verify_binding_evidence()
    if base.previous.digest(PIPELINE)!=PIPELINE_SHA:raise RuntimeError('Pinned occupancy threshold source changed')
    data.read_input();return current

def output_summary(array):
    import numpy as np
    return dict(all_finite=bool(np.isfinite(array).all()),count=array.size,minimum=float(array.min()),maximum=float(array.max()),mean=float(array.mean()),stddev=float(array.std()))

def provenance(record):
    return dict(input=record['saved_latent'],decoder_checkpoint_sha256=record['checkpoint_identity']['checkpoint_sha256'],
        decoder_config_sha256=record['checkpoint_identity']['config_sha256'],decoder_source_sha256=record['checkpoint_identity']['source_module_sha256'],
        support_sources=record['verified_support_source'],pipeline_source_sha256=PIPELINE_SHA,
        runtime=record['runtime'],harness_commit=os.environ['MODLY_OCCUPANCY_HARNESS_COMMIT'],
        threshold='decoded > 0',conversion='none; preserve native64cubed',coordinate_order=['batch','D','H','W'])

def profile():
    files=list(base.COMMITTED_FILES)+['scripts/compatibility/'+n+'.py' for n in ('amd_sparse_occupancy','amd_sparse_occupancy_contract','amd_sparse_occupancy_data','amd_sparse_latent_checkpoint','amd_sparse_sampling_contract','amd_sparse_flow_contract','amd_feature_checkpoint','amd_blaslt_teardown','amd_dinov3_conditioning','amd_dinov3_local','amd_dinov3_contract')]
    files += [p.relative_to(base.ROOT).as_posix() for p in (data.INPUT,data.INPUT.parent/'resume-result-analysis.json',teardown.PINS,teardown.EVIDENCE/'binary-review.json')]
    return SimpleNamespace(kind='occupancy',evidence=EVIDENCE,cpu_seconds=contract.CPU_SECONDS,gpu_seconds=contract.GPU_SECONDS,
        overall_seconds=contract.OVERALL_SECONDS,identity=identity,source_identity=lambda:(base.source_identity(),cache_plan()),
        validate_config=contract.fixture,classify=contract.classify,prepare_private_cache=private_cache,worker_script=Path(__file__).resolve(),
        authorize=b'EXECUTE_ONE_DECODER_FORWARD\n',committed_files=files)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--output',type=Path,required=True);parser.add_argument('--harness-commit')
    args=parser.parse_args();teardown.verify_workspace()
    if args.worker:
        base.EVIDENCE=EVIDENCE
        raise SystemExit(base.worker(args.output,occupancy=sys.modules[__name__]))
    expected=identity();ram=admission();plan=cache_plan()
    observer=base.windows_observation(__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    if not base.healthy_observation(observer):raise RuntimeError('Windows observer unhealthy before marker')
    base.verify_committed_files(args.harness_commit,profile().committed_files)
    with (EVIDENCE/'pre-marker-admission.json').open('x') as f:json.dump(dict(ram=ram,observer=observer,checkpoint_sha256=expected['checkpoint_sha256'],input_sha256=data.INPUT_SHA,eligible_cache_files=len(plan['selected']),workspace_dir=os.environ['WORKSPACE_DIR']),f,indent=2)
    os.environ['MODLY_OCCUPANCY_HARNESS_COMMIT']=args.harness_commit
    raise SystemExit(base.supervise(args.output,args.harness_commit,profile=profile()))
