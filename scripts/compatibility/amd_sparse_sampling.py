"""One published, contained sparse sampling attempt using the existing forward worker."""
import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import amd_sparse_flow_forward as forward
import amd_sparse_sampling_contract as policy
from amd_sparse_sampling_control import SamplingControl

EVIDENCE=forward.local.ROOT/'docs/compatibility/amd-sparse-structure-sampling-evidence'
PINS=EVIDENCE/'source-contract.json'
AUTHORIZE=b'EXECUTE_ONE_12_STEP_SPARSE_SAMPLING_SEQUENCE\n'
NAMES=('__init__.py','base.py','flow_euler.py','classifier_free_guidance_mixin.py','guidance_interval_mixin.py')

def sources():
    return {str(forward.local.SOURCE/'trellis2/pipelines/samplers'/n):forward.local.digest(forward.local.SOURCE/'trellis2/pipelines/samplers'/n) for n in NAMES}

def dependencies():
    site=forward.local.VENV/'Lib/site-packages'
    wanted={'torch','numpy','safetensors','tqdm','easydict'}
    versions={d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions(path=[str(site)]) if d.metadata['Name'].lower() in wanted}
    return {'versions':versions,'pure_entry_hashes':{n:forward.local.digest(site/(n+'/__init__.py')) for n in ('tqdm','easydict')}}

def verify_sampler():
    pins=json.loads(PINS.read_text())
    if sources()!=pins['sampler_sources'] or dependencies()!=pins['dependencies']:
        raise RuntimeError('Pinned sampler source/dependency identity changed')
    if forward.local.digest(forward.local.PREFIX.parent.parent/'pipeline.json')!=pins['previous_pipeline_config_sha256']:
        raise RuntimeError('Audited pipeline configuration changed')
    return pins

def import_sampler():
    verify_sampler();sys.path.insert(0,str(forward.local.SOURCE))
    cls=importlib.import_module('trellis2.pipelines.samplers.flow_euler').FlowEulerGuidanceIntervalSampler
    if Path(inspect.getfile(cls)).resolve()!=forward.local.SOURCE/'trellis2/pipelines/samplers/flow_euler.py':
        raise RuntimeError('Sampler provenance mismatch')
    return cls

def preflight():
    verify_sampler()
    expected=forward.verify_preflight()
    return expected

def admission():
    import psutil
    from datetime import datetime,timezone
    available=psutil.virtual_memory().available
    row=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),available_bytes=available,
             required_bytes=policy.REQUIRED_RAM,admitted=available>=policy.REQUIRED_RAM,torch_imported='torch' in sys.modules)
    if not row['admitted']:raise RuntimeError('Sampling RAM admission failed: '+json.dumps(row))
    return row

def profile():
    p=forward.profile();p.evidence=EVIDENCE;p.kind='sparse-sampling';p.worker_script=Path(__file__).resolve()
    p.maximum_forwards=22;p.maximum_steps=12;p.maximum_sequences=1
    p.authorize=AUTHORIZE;p.classify=policy.classify;p.identity=preflight
    p.source_identity=lambda:(forward.local.verify_sources(),verify_sampler(),forward.teardown.verify_binding_evidence())
    p.committed_files+=['scripts/compatibility/'+n+'.py' for n in ('amd_sparse_sampling','amd_sparse_sampling_contract',
        'amd_sparse_sampling_control','amd_sparse_latent_checkpoint')]
    p.committed_files.append(PINS.relative_to(forward.local.ROOT).as_posix())
    return p

def factory(torch):
    return SamplingControl(torch,import_sampler(),device='cuda:0')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--harness-commit')
    args=parser.parse_args();forward.teardown.verify_workspace();forward.local.EVIDENCE=EVIDENCE
    if args.worker:
        raise SystemExit(forward.worker(args.output,sampling_factory=factory))
    preflight();row=admission()
    with (EVIDENCE/'pre-execution-admission.json').open('x') as stream:json.dump(row,stream,indent=2)
    os.environ['MODLY_SAMPLING_HARNESS_COMMIT']=args.harness_commit or ''
    raise SystemExit(forward.supervisor.supervise(args.output,args.harness_commit,profile=profile()))
