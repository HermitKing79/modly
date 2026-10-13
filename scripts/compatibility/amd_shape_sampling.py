"""One canonical sparse shape sequence; reuse the proven model worker/supervisor."""
import argparse,importlib,importlib.metadata,inspect,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_shape_slat_forward as forward
import amd_shape_sampling_contract as policy
from amd_shape_sampling_control import ShapeSamplingControl
EVIDENCE=forward.local.ROOT/'docs/compatibility/amd-shape-slat-512-sampling-evidence'
PINS=EVIDENCE/'source-contract.json'
AUTHORIZE=b'EXECUTE_ONE_CANONICAL_12_STEP_21_FORWARD_SHAPE_SEQUENCE\n'
NAMES=('__init__.py','base.py','flow_euler.py','classifier_free_guidance_mixin.py','guidance_interval_mixin.py')
COORDS=forward.local.ROOT/'docs/compatibility/amd-shape-slat-512-forward-evidence/coordinates32.npy'
PIPELINE=forward.local.PREFIX.parent.parent/'pipeline.json'
def sources():return {str(forward.local.SOURCE/'trellis2/pipelines/samplers'/n):forward.local.digest(forward.local.SOURCE/'trellis2/pipelines/samplers'/n) for n in NAMES}
def dependencies():
    site=forward.local.VENV/'Lib/site-packages';wanted={'torch','numpy','safetensors','tqdm','easydict'}
    return dict(versions={d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions(path=[str(site)]) if d.metadata['Name'].lower() in wanted},
        pure_entry_hashes={n:forward.local.digest(site/(n+'/__init__.py')) for n in ('tqdm','easydict')})
def verify_sampler():
    import numpy as np
    pins=json.loads(PINS.read_text());pipeline=json.loads(PIPELINE.read_text())['args']
    if (pins['sampler_sources']!=sources() or pins['dependencies']!=dependencies() or pins['pipeline_sha256']!=forward.local.digest(PIPELINE)
        or pins['normalization']!=pipeline['shape_slat_normalization'] or pins['coords_sha256']!=forward.local.digest(COORDS)):raise RuntimeError('Pinned sampler/pipeline/coordinates changed')
    expected=dict(steps=12,rescale_t=3.0,guidance_strength=7.5,guidance_rescale=.5,guidance_interval=[.6,1.0])
    configured=pipeline['shape_slat_sampler']
    if configured['name']!='FlowEulerGuidanceIntervalSampler' or configured['args']!={'sigma_min':policy.SIGMA_MIN} or configured['params']!=expected:raise RuntimeError('Canonical shape sampler parameters differ')
    coords=np.load(COORDS,allow_pickle=False);forward.contract.validate_coords(coords,policy.TOKEN_COUNT)
    if not np.array_equal(coords,forward.local.verify_inputs()[2]):raise RuntimeError('Canonical saved coordinates differ')
    policy.require_count(21,12,9);return pins
def import_sampler():
    verify_sampler();sys.path.insert(0,str(forward.local.SOURCE))
    cls=importlib.import_module('trellis2.pipelines.samplers.flow_euler').FlowEulerGuidanceIntervalSampler
    if Path(inspect.getfile(cls)).resolve()!=forward.local.SOURCE/'trellis2/pipelines/samplers/flow_euler.py':raise RuntimeError('Sampler provenance differs')
    return cls
def preflight():verify_sampler();return forward.verify_preflight()
def admission():
    import psutil
    from datetime import datetime,timezone
    available=psutil.virtual_memory().available
    row=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),available_bytes=available,required_bytes=policy.REQUIRED_RAM,admitted=available>=policy.REQUIRED_RAM,torch_imported='torch' in sys.modules)
    if not row['admitted']:raise RuntimeError('Host RAM admission failed: '+json.dumps(row))
    return row
def profile():
    p=forward.profile();p.evidence=EVIDENCE;p.kind='shape-sampling';p.worker_script=Path(__file__).resolve()
    p.maximum_forwards=21;p.maximum_steps=12;p.maximum_sequences=1;p.authorize=AUTHORIZE;p.classify=policy.classify;p.identity=preflight
    p.source_identity=lambda:(forward.local.verify_sources(),verify_sampler(),forward.teardown.verify_binding_evidence())
    p.committed_files+=['scripts/compatibility/'+n+'.py' for n in ('amd_shape_sampling','amd_shape_sampling_contract','amd_shape_sampling_control','amd_shape_sampling_artifact')]
    p.committed_files.extend([PINS.relative_to(forward.local.ROOT).as_posix(),COORDS.relative_to(forward.local.ROOT).as_posix()])
    p.committed_files.append((forward.local.PINS.parent/'run/results.json').relative_to(forward.local.ROOT).as_posix())
    return p
def factory(torch):return ShapeSamplingControl(torch,import_sampler(),device='cuda:0')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--output',type=Path,required=True);parser.add_argument('--harness-commit')
    args=parser.parse_args();forward.teardown.verify_workspace();forward.local.EVIDENCE=EVIDENCE
    if args.worker:raise SystemExit(forward.worker(args.output,sampling_factory=factory))
    preflight();row=admission()
    with (EVIDENCE/'pre-execution-admission.json').open('x') as f:json.dump(row,f,indent=2)
    os.environ['MODLY_SAMPLING_HARNESS_COMMIT']=args.harness_commit or ''
    raise SystemExit(forward.supervisor.supervise(args.output,args.harness_commit,profile=profile()))
