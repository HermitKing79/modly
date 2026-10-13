"""One image-only continuation of the existing three contained proof workers."""
import argparse
import ast
from collections import deque
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MethodType,SimpleNamespace

sys.path.insert(0,str(Path(__file__).parent))
import amd_blaslt_teardown as teardown
import amd_dinov3_conditioning as dino
import amd_sparse_sampling as sampling
import amd_sparse_occupancy as occupancy
import amd_feature_checkpoint as features
import amd_sparse_latent_checkpoint as latents
from alpha_input import decode_alpha_image,preprocess_alpha_image

ROOT=dino.local.ROOT
EVIDENCE=ROOT/'docs/compatibility/amd-object-conditioned-occupancy-evidence'
CONTRACT=EVIDENCE/'input-contract.json'
SOURCE=EVIDENCE/'image-selection/0f168a4b1b6e96c72e9627c97a212c27a4572250ff58e25703b9d0c2bc74191a.webp'
SOURCE_SHA='a8d419ac16852b73fbc7b92c6857ba5d4be2f2dd8056bf44af707e6af4d62207'
STAGES=('dino','sampling','decoder')
LIMITS={'dino':(120,180,360),'sampling':(180,240,480),'decoder':(120,180,360)}
should_save_features=dino.should_save_features


def digest_bytes(blob):return hashlib.sha256(blob).hexdigest()

def read_source():
    with SOURCE.open('rb') as f:return f.read(2*1024**2+1)

def selected_image():
    blob=read_source()
    if len(blob)>2*1024**2 or digest_bytes(blob)!=SOURCE_SHA:
        raise ValueError('Selected official image bytes differ from the pinned source')
    image,alpha=decode_alpha_image(blob)
    if image.size!=(1042,1042):raise ValueError('Selected source resolution changed')
    row=json.loads(CONTRACT.read_text())
    if row['source_sha256']!=SOURCE_SHA or digest_bytes(image.tobytes())!=row['pixel_sha256']:
        raise ValueError('Selected decoded image/provenance changed')
    return image,row

def preprocess_cpu(image):
    """Execute only the identical real CPU preprocessing method, without Torch."""
    import numpy as np
    from PIL import Image
    dino.local.verify_sources()
    path=dino.local.SOURCE/'trellis2/pipelines/trellis2_image_to_3d.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Trellis2ImageTo3DPipeline')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='preprocess_image')
    namespace={'np':np,'Image':Image}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    shell=SimpleNamespace(rembg_model=None)
    shell.preprocess_image=MethodType(namespace['preprocess_image'],shell)
    return preprocess_alpha_image(shell,image)

def image_factory(path):
    image,row=selected_image()
    path=Path(path);path.resolve().relative_to(EVIDENCE/'dino/run/scratch')
    with path.open('xb') as stream:image.save(stream,format='PNG')
    if dino.local.digest(path)!=row['input_png_sha256']:
        raise ValueError('Actual worker input PNG differs from CPU preflight')
    return image

def successful_stage(stage):
    result=json.loads((EVIDENCE/stage/'run/results.json').read_text())
    worker=result['worker'];process=result['process']
    if (result['classification']!='PASS' or result['observer_healthy'] is not True
        or result['checkpoint_unchanged'] is not True or process.get('returncode')!=0
        or process.get('worker_exit_verified') is not True
        or process.get('job_cleanup',{}).get('owned_tree_exit_verified') is not True
        or worker['release']['allocated_bytes']!=0 or worker['release']['reserved_bytes']!=0):
        raise ValueError('Preceding owned stage did not pass all safety/cleanup gates: '+stage)
    return worker

def feature_reference():
    worker=successful_stage('dino');row=worker['feature_checkpoint']
    image,pins=selected_image();del image
    expected={k:row['provenance'][k] for k in features.FIELDS}
    if (expected['input_pixel_sha256']!=pins['pixel_sha256']
        or expected['input_png_sha256']!=pins['input_png_sha256']
        or expected['harness_commit']!=os.environ['MODLY_OBJECT_HARNESS_COMMIT']
        or expected['checkpoint_sha256']!=json.loads(dino.local.PINS.read_text())['checkpoint_sha256']):
        raise ValueError('Generated conditioning provenance differs from this approved sequence')
    path=EVIDENCE/'dino/run/scratch/conditioning.safetensors'
    restored=features.verify(path,path.parent,expected,row['output_sha256'])
    if restored!=row:raise ValueError('Generated conditioning artifact/reference differs')
    return restored

def latent_reference():
    worker=successful_stage('sampling');row=worker['latent_checkpoint']
    feature=feature_reference();provenance={k:row['provenance'][k] for k in latents.FIELDS}
    if (provenance['feature_file_sha256']!=feature['file_sha256']
        or json.loads(provenance['feature_provenance'])!=feature['provenance']
        or provenance['harness_commit']!=os.environ['MODLY_OBJECT_HARNESS_COMMIT']):
        raise ValueError('Generated latent provenance breaks the approved feature chain')
    path=EVIDENCE/'sampling/run/scratch/sparse-structure-latent.safetensors'
    restored=latents.verify(path,path.parent,provenance,row['output_sha256'])
    if restored!=row:raise ValueError('Generated latent artifact/reference differs')
    return restored

def prepare_profile(stage):
    """Retain each original profile and alter only image/artifact routing."""
    if stage=='dino':
        p=teardown.profile();dino.local.fixture=image_factory
    elif stage=='sampling':
        p=sampling.profile();row=feature_reference()
        local=sampling.forward.local
        local.FEATURE=Path(row['path']);local.FEATURE_SHA=row['file_sha256']
        local.verify_feature=partial(local.verify_feature,expected=row)
        sampling.forward.verify_preflight=partial(sampling.forward.verify_preflight,feature_expected=row)
        local.EVIDENCE=EVIDENCE/stage
    elif stage=='decoder':
        p=occupancy.profile();row=latent_reference()
        occupancy.data.INPUT=Path(row['path']);occupancy.data.INPUT_SHA=row['file_sha256']
        occupancy.data.read_input=partial(occupancy.data.read_input,occupancy.data.INPUT,expected=row)
        occupancy.EVIDENCE=EVIDENCE/stage;occupancy.base.EVIDENCE=EVIDENCE/stage
    else:raise ValueError('Unknown stage')
    p.evidence=EVIDENCE/stage;p.worker_script=Path(__file__).resolve()
    p.committed_files+=['scripts/compatibility/amd_object_occupancy.py','scripts/compatibility/amd_feature_checkpoint.py',CONTRACT.relative_to(ROOT).as_posix(),SOURCE.relative_to(ROOT).as_posix()]
    if (p.cpu_seconds,p.gpu_seconds,p.overall_seconds)!=LIMITS[stage]:raise ValueError('Stage deadlines changed')
    return p

def run_stages(run):
    for stage in STAGES:
        if not run(stage):return False
    return True

def diagnostic_mask(mask):
    """Exactly the predeclared six-connected per-component span criterion."""
    import numpy as np
    if mask.shape!=(64,64,64) or mask.dtype!=bool:raise ValueError('Native binary mask required')
    remaining=set(int(i) for i in np.flatnonzero(mask));components=[]
    while remaining:
        start=remaining.pop();queue=deque([start]);low=[64]*3;high=[-1]*3;count=0
        while queue:
            value=queue.popleft();point=(value//4096,(value//64)%64,value%64);count+=1
            for axis in range(3):low[axis]=min(low[axis],point[axis]);high[axis]=max(high[axis],point[axis])
            for axis,stride in enumerate((4096,64,1)):
                for sign in (-1,1):
                    if 0<=point[axis]+sign<64:
                        neighbor=value+sign*stride
                        if neighbor in remaining:remaining.remove(neighbor);queue.append(neighbor)
        components.append(dict(count=count,minimum=low,maximum=high,spans=[high[i]-low[i]+1 for i in range(3)]))
    count=int(mask.sum());passes=count<mask.size and any(min(c['spans'])>=8 for c in components)
    return dict(classification='PASS' if passes else 'FAIL',occupied=count,nonfull=count<mask.size,connectivity=6,
                components=sorted(components,key=lambda c:c['count'],reverse=True),shape_accuracy_proven=False)

def preflight(commit):
    teardown.verify_workspace();selected_image();dino.verify_preflight();sampling.preflight();occupancy.identity()
    teardown.verify_binding_evidence();occupancy.cache_plan()
    ram=sampling.admission()
    from datetime import datetime,timezone
    observer=occupancy.base.windows_observation(datetime.now(timezone.utc).isoformat())
    if not occupancy.base.healthy_observation(observer):raise RuntimeError('Unhealthy Windows observer before sequence marker')
    files=list(set(teardown.profile().committed_files+sampling.profile().committed_files+occupancy.profile().committed_files+
        ['scripts/compatibility/amd_object_occupancy.py',CONTRACT.relative_to(ROOT).as_posix(),SOURCE.relative_to(ROOT).as_posix()]))
    occupancy.base.verify_committed_files(commit,files)
    with (EVIDENCE/'pre-marker-admission.json').open('x') as stream:json.dump(dict(ram=ram,observer=observer,workspace=os.environ['WORKSPACE_DIR'],torch_imported='torch' in sys.modules),stream,indent=2)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--output',type=Path);parser.add_argument('--harness-commit')
    args=parser.parse_args();teardown.verify_workspace()
    commit=args.harness_commit or os.environ.get('MODLY_OBJECT_HARNESS_COMMIT')
    if args.worker:
        path=args.output.resolve();stage=next((s for s in STAGES if path==EVIDENCE/s/'run/scratch'),None)
        if stage is None:raise ValueError('Worker output is not an exact owned stage scratch')
        prepare_profile(stage)
        if stage=='dino':return dino.worker(path,allocation_evidence=EVIDENCE/stage,workspace_teardown=True,save_features=True)
        if stage=='sampling':return sampling.forward.worker(path,sampling_factory=sampling.factory)
        return occupancy.base.worker(path,occupancy=occupancy)
    os.environ['MODLY_OBJECT_HARNESS_COMMIT']=commit or ''
    os.environ['MODLY_SAMPLING_HARNESS_COMMIT']=commit or ''
    os.environ['MODLY_OCCUPANCY_HARNESS_COMMIT']=commit or ''
    preflight(commit)
    with (EVIDENCE/'sequence-attempt-started.json').open('x') as stream:
        json.dump(dict(harness_commit=commit,maximum_sequences=1,stages=list(STAGES),forwards=dict(dino=1,sampling=22,decoder=1)),stream);stream.flush();os.fsync(stream.fileno())
    def run(stage):
        sampling.admission() if stage!='decoder' else occupancy.admission()
        profile=prepare_profile(stage)
        return occupancy.base.supervise(EVIDENCE/stage/'run',commit,profile=profile)==0
    complete=run_stages(run)
    result=dict(classification='UNKNOWN',stages_complete=complete)
    if complete:
        import numpy as np
        worker=successful_stage('decoder');art=worker['occupancy_artifacts']
        scratch=EVIDENCE/'decoder/run/scratch'
        logits=np.load(scratch/'occupancy-logits.npy',allow_pickle=False,max_header_size=4096)
        mask=np.load(scratch/'occupancy-mask.npy',allow_pickle=False,max_header_size=4096)
        if not np.isfinite(logits).all() or not np.array_equal(mask,logits[0,0]>0):raise ValueError('Independent occupancy readback failed')
        result.update(diagnostic=diagnostic_mask(mask));result['classification']=result['diagnostic']['classification']
        if mask.any():result['preview']=occupancy.data.write_preview(EVIDENCE,mask)
        result['artifacts']=art;feature_reference();latent_reference()
    with (EVIDENCE/'sequence-result.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(dict(classification=result['classification'],stages_complete=complete)),flush=True)
    return 0 if complete else 1

if __name__=='__main__':raise SystemExit(main())
