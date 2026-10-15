"""Fresh-lineage texture sampling adapter; CPU preparation precedes one owned worker.

No model/GPU work occurs while importing this module or preparing persistence.
The original texture worker retains its strict loader, sampler and containment.
"""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import amd_generation_shape as shape
import amd_texture_sampling as original
import amd_texture_sampling_local as immutable
import amd_texture_sampling_contract as policy
import amd_texture_sampling_artifact as persistence

APPROVED_ROOT=shape.APPROVED_ROOT
MAX_TOKENS=4430
SCHEMA='modly.amd.generation-texture-inputs.v1'
CONTEXT_ENV='MODLY_GENERATION_TEXTURE_CONTEXT'


def validate_count(value):
    if type(value) is not int or not 0<value<=MAX_TOKENS:
        raise ValueError('Fresh texture support must remain in1..4430')
    return value


def context(shape_evidence,evidence,artifact_root,harness_commit=None):
    from amd_generation_occupancy import plain_path
    source,target=plain_path(shape_evidence),plain_path(evidence)
    approved=(APPROVED_ROOT/'new-image').resolve()
    source.relative_to(approved);target.relative_to(approved)
    if (source.name!='shape' or target.name!='texture' or source==target
            or target in source.parents or source in target.parents):
        raise ValueError('Distinct new-image shape and texture evidence required')
    root=persistence.approved_root(Path(artifact_root))
    if harness_commit is not None and (type(harness_commit) is not str or not re.fullmatch('[0-9a-f]{40}',harness_commit)):
        raise ValueError('Exact published texture harness commit required')
    return dict(shape_evidence=str(source),evidence=str(target),artifact_root=str(root),harness_commit=harness_commit)


def worker_context(value,output):
    from amd_feature_checkpoint import unique
    if type(value) is not str or len(value)>65536:raise ValueError('Bounded owned texture context required')
    parsed=json.loads(value,object_pairs_hook=unique)
    if type(parsed) is not dict or set(parsed)!={'shape_evidence','evidence','artifact_root','harness_commit'}:
        raise ValueError('Exact owned texture context required')
    result=context(**parsed)
    if not result['harness_commit'] or Path(output).resolve()!=Path(result['artifact_root']):
        raise ValueError('Exact owned texture worker context/output required')
    return result


def input_bundle(shape_evidence):
    """Reopen genuine shape outputs; preserve the source's F32 roundoff semantics."""
    import numpy as np
    from amd_generation_occupancy import plain_path
    evidence=plain_path(shape_evidence,existing=True)
    evidence.relative_to((APPROVED_ROOT/'new-image').resolve())
    result=shape._json(evidence/'run/results.json')
    worker=shape.require_released(result)
    if shape.classify(worker,True,True,True)!='PASS':
        raise ValueError('Fresh shape sequence acceptance is incomplete')
    prefix=shape.input_bundle(evidence.parent)
    if worker['provenance']['conditioning']!=prefix['feature']:
        raise ValueError('Shape and texture DINO conditioning lineage differs')
    from amd_shape_sampling_artifact import denormalize
    arrays={};rows={}
    scratch=evidence/'run/scratch'
    for name in ('shape-latent-raw.npy','shape-latent-denormalized.npy','shape-latent-coordinates.npy'):
        row=worker['artifacts'][name];path=plain_path(row['path'],existing=True)
        if path!=scratch/name or path.stat().st_size!=row['file_bytes'] or shape._digest(path)!=row['sha256']:
            raise ValueError('Fresh shape artifact file identity differs: '+name)
        with path.open('rb') as stream:
            array=np.load(stream,allow_pickle=False,max_header_size=4096)
            if stream.read(1):raise ValueError('Trailing shape artifact bytes')
        if (array.shape!=tuple(row['shape']) or str(array.dtype)!=row['dtype'] or not array.flags.c_contiguous
                or not np.isfinite(array).all() or hashlib.sha256(array.tobytes()).hexdigest()!=row['payload_sha256']):
            raise ValueError('Fresh shape artifact payload differs: '+name)
        arrays[name]=array
        rows[name]=dict(bytes=row['file_bytes'],sha256=row['sha256'],payload_sha256=row['payload_sha256'],shape=row['shape'],dtype=row['dtype'])
    raw,denormalized,coords=(arrays[k] for k in ('shape-latent-raw.npy','shape-latent-denormalized.npy','shape-latent-coordinates.npy'))
    n=validate_count(len(coords));policy.validate_coords(coords,n)
    if not np.array_equal(coords,prefix['coords']):raise ValueError('Shape coordinates differ from native occupancy support')
    provenance_path=scratch/'shape-latent-provenance.json';provenance=shape._json(provenance_path)
    if (shape._digest(provenance_path)!=worker['artifacts'][provenance_path.name]['sha256']
            or provenance['generation_inputs']!=worker['generation_inputs']
            or provenance['input']!=worker['input'] or provenance['checkpoint']!=worker['provenance']
            or provenance['raw_output']!=worker['output'] or provenance['normalization']!=policy.SHAPE_NORMALIZATION):
        raise ValueError('Fresh shape provenance differs from accepted worker')
    mean,std=policy.normalization_arrays(policy.SHAPE_NORMALIZATION,policy.SHAPE_NORMALIZATION)
    if not np.array_equal(denormalize(raw,mean,std,maximum_tokens=MAX_TOKENS),denormalized):
        raise ValueError('Original shape denormalization does not reproduce the saved bytes')
    # The real pipeline re-normalizes its saved denormalized shape; the inverse
    # need not equal raw byte-for-byte because F32 rounding is legitimate.
    normalized=policy.normalize_shape(denormalized,maximum_tokens=MAX_TOKENS)
    lineage=dict(schema=SCHEMA,token_count=n,input_contract_sha256=prefix['lineage']['input_contract_sha256'],
        feature_file_sha256=prefix['feature']['file_sha256'],shape_raw_payload_sha256=rows['shape-latent-raw.npy']['payload_sha256'],
        shape_denormalized_payload_sha256=rows['shape-latent-denormalized.npy']['payload_sha256'],
        shape_coordinates_payload_sha256=rows['shape-latent-coordinates.npy']['payload_sha256'],
        shape_provenance_sha256=shape._digest(provenance_path),shape_results_sha256=shape._digest(evidence/'run/results.json'),
        shape_condition_sha256=hashlib.sha256(normalized.tobytes()).hexdigest(),shape_harness_commit=provenance['harness_commit'])
    return dict(feature=prefix['feature'],normalized=normalized,coords=coords,saved_inputs=rows,
                shape_provenance_sha256=lineage['shape_provenance_sha256'],lineage=lineage)


def classify(record,process_ok,healthy,preserved):
    try:
        lineage=record['generation_inputs'];n=validate_count(record['token_count'])
        if (lineage['schema']!=SCHEMA or lineage['token_count']!=n
                or lineage['shape_coordinates_payload_sha256']!=record['input']['coords']['sha256']
                or lineage['shape_condition_sha256']!=record['input']['shape_cond']['sha256']
                or record['artifacts']['metadata']['provenance']['generation_inputs']!=lineage):
            raise ValueError('Fresh texture lineage differs')
        feature=lineage['feature_file_sha256']
    except (KeyError,ValueError,TypeError):return 'PARTIAL'
    return policy.classify(record,process_ok,healthy,preserved,token_count=n,maximum_tokens=MAX_TOKENS,feature_sha256=feature)


class Runtime:
    def __init__(self,config):
        self.config=context(**config);self.shape_evidence=Path(self.config['shape_evidence'])
        self.evidence=Path(self.config['evidence']);self.artifact_root=Path(self.config['artifact_root']);self._bundle=None
        names=('ROOT','SOURCE','VENV','loading','digest','manifest','PREFIX','AUDIT','PINS','SOURCE_AUDIT',
               'verify_sources','verify_sampler','import_class','import_sampler')
        self.local=SimpleNamespace(**{key:getattr(immutable,key) for key in names})
        self.local.EVIDENCE=self.evidence;self.local.COORDS=self.shape_evidence/'run/scratch/shape-latent-coordinates.npy'
        self.local.FEATURE=self.shape_evidence.parent/'dino/run/scratch/conditioning.safetensors'
        self.local.verify_inputs=self.verify_inputs
        self.local.load_cpu=partial(immutable.load_cpu,identity_verifier=self.identity)
        self.artifact=SimpleNamespace(approved_root=persistence.approved_root,write=self.write_artifact,verify=self.verify_artifact)
        self.readable_tensor_paths=(self.shape_evidence.parent/'sampling/run/scratch/sparse-structure-latent.safetensors',)

    @property
    def bundle(self):
        if self._bundle is None:
            self._bundle=input_bundle(self.shape_evidence)
            if Path(self._bundle['feature']['path']).resolve()!=self.local.FEATURE:
                raise ValueError('Fresh feature is not the exact owned DINO artifact')
        return self._bundle

    def current(self):
        import numpy as np
        current=input_bundle(self.shape_evidence);expected=self.bundle
        if (current['lineage']!=expected['lineage'] or current['feature']!=expected['feature']
                or not np.array_equal(current['coords'],expected['coords']) or not np.array_equal(current['normalized'],expected['normalized'])):
            raise ValueError('Fresh shape inputs changed after admission')
        return current

    def verify_inputs(self):
        current=self.current();return current['feature'],current['normalized'],current['coords']

    def identity(self,*,hash_weights=True):
        return immutable.identity(hash_weights=hash_weights,input_verifier=self.current)

    def verify_artifact(self,path,root,expected_metadata=None):
        return persistence.verify(path,root,expected_metadata,token_count=len(self.bundle['coords']),maximum_tokens=MAX_TOKENS)

    def write_artifact(self,root,raw,denormalized,coords,provenance,*,noise):
        if provenance.get('generation_inputs')!=self.bundle['lineage']:
            raise ValueError('Texture writer requires this fresh shape lineage')
        return persistence.write(root,raw,denormalized,coords,provenance,noise=noise,
                                 token_count=len(self.bundle['coords']),maximum_tokens=MAX_TOKENS)

    def denormalize_texture(self,values):
        return policy.denormalize_texture(values,maximum_tokens=MAX_TOKENS)

    def validate_inputs(self,rows,feature,normalized,coords):
        import numpy as np
        current=self.current()
        if feature!=current['feature'] or not np.array_equal(coords,current['coords']) or not np.array_equal(normalized,current['normalized']):
            raise ValueError('Texture worker received different fresh conditioning')
        for key,expected in [('coords',current['lineage']['shape_coordinates_payload_sha256']),
                              ('cond',feature['output_sha256']),('shape_cond',current['lineage']['shape_condition_sha256'])]:
            if rows[key]['sha256']!=expected:raise ValueError('Texture input payload differs: '+key)
        return current['lineage'].copy()

    def factory(self,torch):
        from amd_texture_sampling_control import TextureSamplingControl
        return TextureSamplingControl(torch,immutable.import_sampler(),token_count=len(self.bundle['coords']),maximum_tokens=MAX_TOKENS)

    def verify_preflight(self):
        immutable.verify_sources();original.teardown.verify_workspace();original.teardown.verify_binding_evidence()
        if shape._json(immutable.PINS)!=immutable.verify_sampler():raise ValueError('Immutable texture sampler pins differ')
        prepared=shape._json(self.evidence/'texture-input-contract.json')
        if (prepared['schema']!=SCHEMA or prepared['lineage']!=self.bundle['lineage']
                or prepared['artifact_root']!=str(self.artifact_root) or prepared['shape_evidence']!=str(self.shape_evidence)):
            raise ValueError('CPU-prepared texture inputs differ')
        probe=shape._json(self.evidence/'persistence-preflight.json')
        expected={k:v for k,v in probe.items() if k not in ('synthetic','torch_imported')}
        if (probe.get('synthetic') is not True or probe.get('torch_imported') is not False
                or self.verify_artifact(Path(probe['artifact_path']),self.artifact_root/'persistence-probe',probe['metadata'])!=expected):
            raise ValueError('Fresh CPU production persistence proof differs')
        persistence.approved_root(self.artifact_root/'latent')
        return self.identity()

    def profile(self):
        p=original.profile();p.evidence=self.evidence;p.worker_script=Path(__file__).resolve();p.scratch_directory=self.artifact_root
        p.identity=self.verify_preflight;p.source_identity=lambda:(immutable.verify_sources(),immutable.verify_sampler(),original.teardown.verify_binding_evidence())
        p.validate_config=lambda c:policy.validate_header(c,self.identity(hash_weights=False)['tensors']);p.classify=classify
        old=original.existing.local
        excluded={path.relative_to(ROOT).as_posix() for path in
            (old.FEATURE,old.TURRET/'occupancy-logits.npy',old.TURRET/'occupancy-mask.npy',old.TURRET/'dino/run/results.json',
             immutable.SHAPE_RAW,immutable.SHAPE_DENORMALIZED,immutable.COORDS,immutable.SHAPE_PROVENANCE,
             immutable.EVIDENCE/'cpu-qualification.json',immutable.EVIDENCE/'persistence-preflight.json')}
        p.committed_files=[name for name in p.committed_files if name not in excluded]
        # The prefix reader executes the same complete source closure as the
        # shape adapter; generated run receipts remain per-run inputs.
        prefix=shape.Runtime(shape.context(self.shape_evidence.parent,self.shape_evidence,self.config['harness_commit']))
        p.committed_files+=prefix.profile().committed_files
        p.committed_files+=['scripts/compatibility/'+n+'.py' for n in ('amd_generation_texture','test_amd_generation_texture','amd_generation_shape')]
        p.committed_files += [(self.evidence/name).relative_to(ROOT).as_posix() for name in ('texture-input-contract.json','persistence-preflight.json')]
        p.committed_files.append(shape.HISTORICAL_RESULTS.relative_to(ROOT).as_posix())
        from amd_generation_occupancy import required_files
        p.committed_files+=required_files(self.shape_evidence.parent);p.committed_files=sorted(set(p.committed_files))
        return p


def prepare(shape_evidence,evidence,artifact_root):
    config=context(shape_evidence,evidence,artifact_root);runtime=Runtime(config)
    if runtime.evidence.exists() or any(runtime.artifact_root.iterdir()):
        raise FileExistsError('Texture CPU preparation requires fresh evidence and an empty explicit artifact root')
    bundle=runtime.bundle
    runtime.evidence.mkdir(parents=True,exist_ok=False)
    probe=persistence.persistence_probe(runtime.artifact_root,token_count=len(bundle['coords']),maximum_tokens=MAX_TOKENS)
    if probe['torch_imported'] is not False:raise RuntimeError('CPU persistence preparation must not import Torch')
    (runtime.artifact_root/'latent').mkdir(exist_ok=False)
    with (runtime.evidence/'persistence-preflight.json').open('x') as stream:json.dump(probe,stream,indent=2)
    record=dict(schema=SCHEMA,shape_evidence=str(runtime.shape_evidence),artifact_root=str(runtime.artifact_root),
                lineage=bundle['lineage'],scope='CPU input and actual production artifact persistence only',model_inference=False)
    with (runtime.evidence/'texture-input-contract.json').open('x') as stream:json.dump(record,stream,indent=2)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape-evidence',type=Path);parser.add_argument('--evidence',type=Path)
    parser.add_argument('--artifact-root',type=Path);parser.add_argument('--harness-commit')
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--worker',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.worker:
        if args.output is None:raise ValueError('Owned texture worker requires an explicit output')
        config=worker_context(os.environ[CONTEXT_ENV],args.output);runtime=Runtime(config)
        os.environ['MODLY_SAMPLING_HARNESS_COMMIT']=config['harness_commit']
        return original.worker(args.output,runtime=runtime)
    if not all((args.shape_evidence,args.evidence,args.artifact_root)):
        parser.error('--shape-evidence/--evidence/--artifact-root required')
    if args.prepare_only:
        print(json.dumps(prepare(args.shape_evidence,args.evidence,args.artifact_root)));return 0
    config=context(args.shape_evidence,args.evidence,args.artifact_root,args.harness_commit)
    if not config['harness_commit']:raise ValueError('Published texture harness commit required')
    runtime=Runtime(config)
    if (runtime.evidence/'run').exists() or any((runtime.artifact_root/'latent').iterdir()):
        raise FileExistsError('One fresh texture attempt requires an absent run and empty latent output')
    runtime.verify_preflight();profile=runtime.profile()
    original.supervisor.verify_committed_files(args.harness_commit,profile.committed_files)
    import psutil
    available=psutil.virtual_memory().available
    with (runtime.evidence/'pre-execution-admission.json').open('x') as stream:
        json.dump(dict(available_host_RAM_bytes=available,required_bytes=policy.REQUIRED_RAM,
                       admitted=available>=policy.REQUIRED_RAM,torch_imported='torch' in sys.modules),stream,indent=2)
    if available<policy.REQUIRED_RAM:raise RuntimeError('12GiB available host RAM required')
    os.environ[CONTEXT_ENV]=json.dumps(config,sort_keys=True);os.environ['MODLY_SAMPLING_HARNESS_COMMIT']=args.harness_commit
    os.environ.update(original.TUNABLE_ENV)
    rc=original.supervisor.supervise(runtime.evidence/'run',args.harness_commit,profile=profile)
    result=shape._json(runtime.evidence/'run/results.json')
    post=dict(classification=result['classification'],worker_exit_verified=result['process'].get('worker_exit_verified'),verified=False)
    try:
        if result['worker'].get('artifacts'):
            saved=result['worker']['artifacts']
            post['artifact']=runtime.verify_artifact(Path(saved['artifact_path']),runtime.artifact_root/'latent',saved['metadata'])
            post['verified']=post['artifact']==saved
        if result['classification']=='PASS':
            shape.require_released(result)
            if not post['verified']:raise RuntimeError('Post-exit texture artifact identity differs')
    except BaseException as exc:post.update(classification='UNKNOWN',error=repr(exc));rc=1
    with (runtime.evidence/'post-exit-artifact-verification.json').open('x') as stream:json.dump(post,stream,indent=2)
    print(json.dumps(post))
    return rc


if __name__=='__main__':raise SystemExit(main())
