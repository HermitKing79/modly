"""One zombie Shape-SLat sequence from separate authenticated completed roots.

Experiment-specific routing around the original Windows worker; no new neural
implementation, historical receipt rewriting or product dispatch.
"""
import argparse
import copy
from functools import partial
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amd_generation_shape as generation
import amd_shape_sampling as historical
import amd_shape_sampling_artifact as artifact
import amd_zombie_occupancy as prior

ROOT = prior.ROOT
PRIVATE = ROOT / 'artifacts/amd-zombie-shape-0d15dd9'
EVIDENCE = PRIVATE / 'shape'
PUBLIC = ROOT / 'docs/compatibility/amd-zombie-shape-evidence'
CPU_SUMMARY = PUBLIC / 'cpu-admission-summary.json'
OCCUPANCY_COMMIT = 'e6b6330ba2f9e51abfb74c726729e0be4ade7cac'
FEATURE = prior.zombie.EVIDENCE / 'run/scratch/conditioning.safetensors'
SPARSE = prior.EVIDENCE / 'sampling/run/scratch/sparse-structure-latent.safetensors'
ENV_COMMIT = 'MODLY_ZOMBIE_SHAPE_COMMIT'
MAX_TOKENS = generation.MAX_TOKENS
# Freeze the historical static profile before any per-run routing.
_PROFILE = historical.profile()
HISTORY = ['docs/compatibility/amd-zombie-occupancy-evidence/' + n for n in
           ('sampling-outcome-review.json', 'decoder-outcome-review.json')]
CLOSURE = sorted(set(_PROFILE.committed_files + prior.CLOSURE + HISTORY + [
    'scripts/compatibility/amd_generation_shape.py',
    'scripts/compatibility/amd_zombie_shape.py',
    'scripts/compatibility/test_amd_zombie_shape.py',
    'scripts/compatibility/test_amd_zombie_shape_acceptance.py',
    CPU_SUMMARY.relative_to(ROOT).as_posix(),
    generation.HISTORICAL_RESULTS.relative_to(ROOT).as_posix()]))


def valid_commit(commit):
    prior.valid_commit(commit)
    if commit == OCCUPANCY_COMMIT:
        raise ValueError('A distinct new Shape-SLat execution commit is required')


def derive_support(logits, mask):
    pooled, coords = historical.forward.contract.reduce_occupancy(logits, mask)
    generation.validate_count(len(coords))
    return pooled, coords


def input_bundle():
    """Original independent roots remain original; only input lineage is joined."""
    feature = prior.saved_features()
    occupancy = prior.saved_occupancy(OCCUPANCY_COMMIT)
    latent = occupancy['provenance']['input']
    receipt_hashes = {}
    for stage, expected_artifact in (('sampling', latent), ('decoder', occupancy)):
        review_path = ROOT / ('docs/compatibility/amd-zombie-occupancy-evidence/' + stage + '-outcome-review.json')
        review = prior.read_json(review_path)
        if (review['classification'] != 'PASS' or review['blockers']
                or review['execution_harness_commit'] != OCCUPANCY_COMMIT
                or review['historical_dino_execution_commit'] != prior.DINO_COMMIT):
            raise ValueError('Original independent outcome review failed')
        for name, expected in review['private_run_evidence_sha256'].items():
            if prior.digest(prior.read(prior.EVIDENCE / stage / 'run' / name)) != expected:
                raise ValueError('Original reviewed stage receipt changed: ' + name)
        independent_path = prior.EVIDENCE / (stage + '-independent-readback.json')
        independent = prior.read_json(independent_path)
        if (prior.digest(prior.read(independent_path)) != review['separate_adapter_post_exit_receipt_sha256']
                or independent['harness_commit'] != OCCUPANCY_COMMIT or independent['verified'] is not True
                or independent['artifact'] != expected_artifact
                or independent['results_sha256'] != review['private_run_evidence_sha256']['results.json']
                or prior.digest(prior.read(prior.EVIDENCE / stage / 'attempt-started.json')) != review['exclusive_attempt_marker_sha256']):
            raise ValueError('Original completed-stage readback or marker changed')
        receipt_hashes[stage] = dict(review_sha256=prior.digest(prior.read(review_path)),
            independent_sha256=prior.digest(prior.read(independent_path)),
            results_sha256=independent['results_sha256'])
    files = occupancy['files']
    import numpy as np
    logits = np.load(io.BytesIO(prior.read(Path(files['occupancy-logits.npy']['path']), 2 * 1024**2)), allow_pickle=False)
    mask = np.load(io.BytesIO(prior.read(Path(files['occupancy-mask.npy']['path']), 512 * 1024)), allow_pickle=False)
    pooled, coords = derive_support(logits, mask)
    if (feature['file_sha256'] != prior.FEATURE_SHA or Path(feature['path']) != FEATURE
            or feature['provenance']['input_png_sha256'] != prior.zombie.SOURCE_SHA
            or Path(latent['path']) != SPARSE):
        raise ValueError('Only the exact saved zombie input roots are admitted')
    origin = dict(source_image_sha256=prior.zombie.SOURCE_SHA, dino_execution_commit=prior.DINO_COMMIT,
        occupancy_execution_commit=OCCUPANCY_COMMIT, feature_file_sha256=feature['file_sha256'],
        sparse_latent_file_sha256=latent['file_sha256'], reviewed_stage_receipts=receipt_hashes,
        occupancy_files={name: row['file_sha256'] for name, row in files.items()})
    lineage = dict(schema=generation.SCHEMA, token_count=len(coords),
        input_contract_sha256=prior.digest(json.dumps(origin, sort_keys=True).encode()),
        coords_payload_sha256=prior.digest(coords.tobytes()), feature_file_sha256=feature['file_sha256'],
        occupancy_provenance_sha256=prior.digest(prior.read(prior.EVIDENCE / 'decoder/run/scratch/occupancy-provenance.json')),
        occupancy_harness_commit=OCCUPANCY_COMMIT, feature_payload_sha256=feature['output_sha256'],
        sparse_latent_file_sha256=latent['file_sha256'], separate_completed_roots=origin)
    return dict(feature=feature, latent=latent, occupancy=occupancy, coords=coords, pooled=pooled,
        lineage=lineage, logits_sha256=files['occupancy-logits.npy']['file_sha256'],
        mask_sha256=files['occupancy-mask.npy']['file_sha256'])


def prepared_contract():
    blob = prior.read(PRIVATE / 'cpu-admission.json')
    if prior.digest(blob) != prior.read_json(CPU_SUMMARY)['private_cpu_admission_sha256']:
        raise ValueError('Committed CPU admission digest differs')
    return json.loads(blob, object_pairs_hook=prior.zombie.features.unique)


class Runtime(generation.Runtime):
    def __init__(self):
        # Constructor does no input I/O. Original worker receives the real owned
        # Job handshake before calling verify_preflight or private input readers.
        self.forward, self.historical = historical.forward, historical
        self.evidence, self.entry, self.authorize = EVIDENCE, self, historical.AUTHORIZE
        self.artifact = SimpleNamespace(denormalize=partial(artifact.denormalize, maximum_tokens=MAX_TOKENS),
            write=partial(artifact.write, maximum_tokens=MAX_TOKENS))
        self._bundle = None
        original = self.forward.local
        self.local = SimpleNamespace(**{key: getattr(original, key) for key in
            ('ROOT', 'SOURCE', 'VENV', 'loading', 'digest', 'manifest', 'PREFIX', 'AUDIT', 'PINS',
             'rope_frequencies', 'import_class', 'load_cpu')})
        self.local.EVIDENCE, self.local.FEATURE = EVIDENCE, FEATURE
        self.local.verify_inputs = self.verify_inputs
        # Two exact immutable proof tensors are hashed by the historical source
        # closure. They are readable proof data, never replacement neural inputs.
        proofs = tuple(ROOT / name for name in prior.CLOSURE if Path(name).suffix == '.safetensors')
        self.readable_tensor_paths = (SPARSE, *proofs)

    @property
    def bundle(self):
        if self._bundle is None:
            self._bundle = input_bundle()
            self.local.LOGITS_SHA, self.local.MASK_SHA = self._bundle['logits_sha256'], self._bundle['mask_sha256']
        return self._bundle

    def verify_inputs(self):
        expected, fresh = self.bundle, input_bundle()
        import numpy as np
        if (any(fresh[k] != expected[k] for k in ('feature', 'latent', 'occupancy', 'lineage'))
                or not np.array_equal(fresh['coords'], expected['coords'])):
            raise ValueError('Authenticated inputs changed after preparation')
        return fresh['feature'], fresh['pooled'], fresh['coords']

    def verify_preflight(self):
        current = super().verify_preflight()
        if current != prepared_contract()['identity']:
            raise ValueError('Exact CPU-prepared model/input identity changed')
        return current

    def profile(self):
        p = copy.copy(_PROFILE)
        p.evidence, p.worker_script = EVIDENCE, Path(__file__).resolve()
        p.identity, p.classify = self.verify_preflight, generation.classify
        p.source_identity = lambda: (self.forward.local.verify_sources(), self.verify_sampler(),
                                    self.forward.teardown.verify_binding_evidence())
        p.committed_files = list(CLOSURE)
        return p


def tensor_audit(runtime):
    """Exercise the original worker's exact tensor-read boundary on CPU first."""
    allowed = {runtime.local.PREFIX.with_suffix('.safetensors').resolve(), FEATURE.resolve(),
               *(p.resolve() for p in runtime.readable_tensor_paths)}
    def audit(event, args):
        if event == 'import' and str(args[0]).split('.')[0] in ('torch', 'trellis2', 'triton', 'flex_gemm'):
            raise RuntimeError('CPU preparation cannot import neural libraries')
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0]))
            if path.suffix.lower() in ('.safetensors', '.pt', '.pth', '.bin', '.gguf', '.ckpt') and path.resolve() not in allowed:
                raise ValueError('Original worker tensor-read admission rejected: ' + str(path))
    sys.addaudithook(audit)


def prepare():
    import importlib.metadata
    import psutil
    runtime = Runtime()
    tensor_audit(runtime)
    identity = generation.Runtime.verify_preflight(runtime)
    ram = historical.admission()
    versions = {n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'safetensors', 'psutil')}
    if (Path(sys.executable) != runtime.local.VENV / 'Scripts/python.exe'
            or sys.version.split()[0] != '3.12.10'
            or runtime.local.digest(Path(sys.executable)) != '0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14'
            or versions != dict(torch='2.13.0+rocm10.0.0', numpy='2.5.3', safetensors='0.8.0', psutil='7.2.2')
            or 'torch' in sys.modules):
        raise RuntimeError('Exact installed CPU runtime identity required')
    free = psutil.disk_usage(str(PRIVATE)).free
    if free < 2 * 1024**3:
        raise RuntimeError('Actual artifact destination below original 2 GiB floor')
    n = identity['token_count']
    inputs = runtime.bundle
    artifacts = dict(saved_conditioning_bytes=inputs['feature']['file_bytes'],
        saved_sparse_latent_bytes=inputs['latent']['file_bytes'],
        saved_occupancy_bytes=sum(v['file_bytes'] for v in inputs['occupancy']['files'].values()),
        pooled_coordinate_payload_bytes=n * 16, output_numeric_file_bytes=3 * (128 + n * 128) + 128 + n * 16,
        maximum_output_provenance_bytes=65536, strict_cpu_model_margin_bytes=identity['tensor_data_bytes'] * 4 + 1024**3)
    row = dict(identity=identity, ram=ram, dependencies=versions, destination_free_bytes=free,
        artifact_memory=artifacts, no_torch_import=True, weights_loaded=False,
        original_worker_tensor_audit_exercised=True, maximum_neural_sequences=1)
    prior.save(PRIVATE / 'cpu-admission.json', row)
    EVIDENCE.mkdir(exist_ok=False)
    summary = dict(private_cpu_admission_sha256=prior.digest(prior.read(PRIVATE / 'cpu-admission.json')),
        token_count=n, maximum_tokens=MAX_TOKENS, coords_payload_sha256=identity['generation_inputs']['coords_payload_sha256'],
        source_image_sha256=prior.zombie.SOURCE_SHA, feature_file_sha256=prior.FEATURE_SHA,
        dino_execution_commit=prior.DINO_COMMIT, occupancy_execution_commit=OCCUPANCY_COMMIT,
        checkpoint_sha256=identity['checkpoint_sha256'], config_sha256=identity['config_sha256'],
        header_sha256=identity['header_sha256'], tensor_count=identity['tensor_count'], parameter_bytes=identity['tensor_data_bytes'],
        resource_qualification=identity['resource_qualification'], artifact_memory=artifacts, ram=ram,
        dependencies=versions, destination_free_bytes=free, original_worker_tensor_audit_exercised=True,
        no_torch_import=True, weights_loaded=False)
    prior.save(CPU_SUMMARY, summary)
    return summary


def verify_arrays(rows, scratch, coords, normalization, provenance_expected):
    """Independent byte/numeric readback; no trust in a claimed worker PASS."""
    import numpy as np
    scratch = prior.zombie.plain_path(scratch, existing=True)
    n = generation.validate_count(len(coords))
    historical.forward.contract.validate_coords(coords, n)
    names = {'shape-latent-raw.npy', 'shape-latent-denormalized.npy', 'shape-latent-coordinates.npy',
             'shape-initial-noise.npy', 'shape-latent-provenance.json'}
    if set(rows) != names:
        raise ValueError('The exact original five-artifact closure is required')
    arrays, verified = {}, {}
    for name in sorted(names):
        row = rows[name]
        path = prior.zombie.plain_path(row['path'], existing=True)
        if path != scratch / name or str(path) != row['path'] or row['verified'] is not True:
            raise ValueError('Artifact is not the exact owned output: ' + name)
        limit = 65536 if name.endswith('.json') else 128 + n * 128
        blob = prior.read(path, limit)
        if len(blob) != row['file_bytes'] or prior.digest(blob) != row['sha256']:
            raise ValueError('Artifact file identity differs: ' + name)
        if name.endswith('.json'):
            if json.loads(blob, object_pairs_hook=prior.zombie.features.unique) != provenance_expected:
                raise ValueError('Complete original provenance differs')
        else:
            array = np.load(io.BytesIO(blob), allow_pickle=False, max_header_size=4096)
            is_coords = name == 'shape-latent-coordinates.npy'
            shape, dtype = ((n, 4), np.dtype('<i4')) if is_coords else ((n, 32), np.dtype('<f4'))
            if (array.shape != shape or array.dtype != dtype or not np.isfinite(array).all()
                    or list(array.shape) != row['shape'] or str(array.dtype) != row['dtype']
                    or len(blob) != 128 + array.nbytes or prior.digest(array.tobytes()) != row['payload_sha256']
                    or not is_coords and not array.flags.c_contiguous):
                raise ValueError('Saved array shape/precision/finiteness/identity differs: ' + name)
            arrays[name] = array
        verified[name] = dict(file_bytes=len(blob), sha256=prior.digest(blob))
    if not np.array_equal(arrays['shape-latent-coordinates.npy'], coords):
        raise ValueError('Output coordinates changed from authentic pooled input')
    mean, std = (np.array(normalization[k], dtype=np.float32) for k in ('mean', 'std'))
    if (mean.shape != (32,) or std.shape != (32,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError('Original F32 normalisation required')
    expected = np.ascontiguousarray(arrays['shape-latent-raw.npy'] * std[None] + mean[None])
    if expected.tobytes() != arrays['shape-latent-denormalized.npy'].tobytes():
        raise ValueError('CPU F32 denormalisation differs')
    return dict(verified=True, token_count=n, files=verified, exact_cpu_f32_denormalization=True,
        raw_payload_sha256=prior.digest(arrays['shape-latent-raw.npy'].tobytes()),
        noise_payload_sha256=prior.digest(arrays['shape-initial-noise.npy'].tobytes()),
        coords_payload_sha256=prior.digest(coords.tobytes()))


def verify_saved(commit):
    valid_commit(commit)
    runtime = Runtime()
    result = prior.read_json(EVIDENCE / 'run/results.json')
    worker = generation.require_released(result)
    process = result['process']
    prepared = prepared_contract()['identity']
    command = [str(runtime.local.VENV / 'Scripts/python.exe'), '-B', '-I', str(Path(__file__).resolve()),
               '--worker', '--output', str(EVIDENCE / 'run/scratch')]
    if (generation.classify(worker, True, True, True) != 'PASS' or process['command'] != command
            or process['harness_commit'] != commit
            or process.get('job_close_error')
            or process['stage_timeouts_seconds'] != dict(cpu=180, forward_release=240, overall=480)
            or process['job_cleanup']['active_before_cleanup'] != 0 or process['job_cleanup']['active_after_cleanup'] != 0):
        raise ValueError('Original sequence, supervision or natural settlement failed')
    committed = prior.read_json(EVIDENCE / 'run/committed-harness.json')
    if (committed['commit'] != commit or set(committed['verified_file_sha256']) != set(CLOSURE)
            or any(prior.digest(prior.read(ROOT / name)) != sha for name, sha in committed['verified_file_sha256'].items())
            or prior.read_json(EVIDENCE / 'run/checkpoint-before.json') != prepared
            or prior.read_json(EVIDENCE / 'run/checkpoint-after.json') != prepared):
        raise ValueError('Execution source or checkpoint identity differs')
    gate = prior.read_json(EVIDENCE / 'run/cpu-gate.json')
    historical.forward.contract.authorize_placement(gate['gate_one'], prepared, prepared['checkpoint_sha256'], gate['checkpoint_sha256'])
    if gate != dict(event='cpu-ready', gate_one=worker['cpu_load'], checkpoint_sha256=prepared['checkpoint_sha256']):
        raise ValueError('Actual strict CPU gate differs')
    expected_runtime = dict(python=sys.version, executable=str(runtime.local.VENV / 'Scripts/python.exe'),
        torch='2.13.0+rocm10.0.0', torch_path=str(runtime.local.VENV / 'Lib/site-packages/torch/__init__.py'),
        hip='7.15.26333', cuda=None, git_version='6596a0dea537c1c78520c73cc6678214781a9d06',
        dependencies=dict(numpy='2.5.3', safetensors='0.8.0'))
    required_ram = prepared['tensor_data_bytes'] * 4 + 1024**3
    if (worker['runtime'] != expected_runtime or worker['budget'] != dict(allocator_limit_bytes=12*1024**3,
            cpu_seconds=180, gpu_seconds=240, overall_seconds=480, required_ram_bytes=required_ram)
            or worker['before_placement']['available_bytes'] < 13*1024**3
            or worker['before_placement']['allocated_bytes'] != 0 or worker['before_placement']['reserved_bytes'] != 0
            or worker['ram_before_strict_load']['required_bytes'] != required_ram
            or worker['ram_before_strict_load']['available_bytes'] < required_ram
            or worker['sampling']['minimum_required_free_vram_bytes'] != generation.LIVE_FREE_RESERVE
            or worker['sampling']['minimum_observed_free_vram_bytes'] < generation.LIVE_FREE_RESERVE):
        raise ValueError('Actual runtime or original RAM/VRAM admission differs')
    feature, _, coords = runtime.verify_inputs()
    lineage = dict(runtime.bundle['lineage'], resource_qualification=generation.resource_qualification(len(coords)))
    checkpoint = dict(seed=historical.policy.SEED, conditioning=feature,
        checkpoint_sha256=prepared['checkpoint_sha256'], config_sha256=prepared['config_sha256'],
        source_sha256=runtime.local.digest(runtime.local.SOURCE / 'trellis2/models/structured_latent_flow.py'),
        occupancy_logits_sha256=runtime.bundle['logits_sha256'], occupancy_mask_sha256=runtime.bundle['mask_sha256'],
        pooling='max_pool3d((decoded>0).float(),2,2,0)>0.5; argwhere[:,[0,2,3,4]].int()',
        sparse_attention='explicit existing PyTorch SDPA branch; no FlashAttention fallback',
        sparse_convolution='none; no convolution used', timestep_schedule=historical.policy.times())
    if (worker['generation_inputs'] != lineage or worker['provenance'] != checkpoint
            or worker['input']['cond']['sha256'] != feature['output_sha256']
            or worker['input']['coords']['sha256'] != lineage['coords_payload_sha256']):
        raise ValueError('Genuine conditioning/coordinate/model lineage differs')
    import numpy as np
    layouts = dict(x=([len(coords), 32], 'torch.float32'), coords=([len(coords), 4], 'torch.int32'),
        cond=([1, 1029, 1024], 'torch.float32'), neg_cond=([1, 1029, 1024], 'torch.float32'),
        batch_map=([len(coords)], 'torch.int64'), seqlen=([1], 'torch.int64'))
    if set(worker['input']) != set(layouts):
        raise ValueError('Exact original CPU input closure required')
    for name, (shape, dtype) in layouts.items():
        row = worker['input'][name]
        sha = row.get('sha256')
        if (type(sha) is not str or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)
                or row != dict(shape=shape, dtype=dtype, device='cpu', sha256=sha)):
            raise ValueError('Exact original CPU input schema differs: ' + name)
    for name, expected in (('neg_cond', np.zeros((1, 1029, 1024), dtype=np.float32)),
                           ('batch_map', np.zeros(len(coords), dtype=np.int64)),
                           ('seqlen', np.array([len(coords)], dtype=np.int64))):
        if worker['input'][name]['sha256'] != prior.digest(expected.tobytes()):
            raise ValueError('Original zero/unconditioned sparse input differs: ' + name)
    sequence = dict(worker['sampling'])
    sequence['models_sha256'] = prior.digest(json.dumps(sequence.pop('models'), sort_keys=True).encode())
    normalization = runtime.verify_sampler()['normalization']
    provenance = json.loads(json.dumps(dict(harness_commit=commit, checkpoint=checkpoint, input=worker['input'],
        parameters=historical.policy.PARAMS, sigma_min=historical.policy.SIGMA_MIN, seed=historical.policy.SEED,
        sampler_sources=runtime.sources(), normalization=normalization, sequence=sequence,
        raw_output=worker['output'], generation_inputs=lineage)))
    readback = verify_arrays(worker['artifacts'], EVIDENCE / 'run/scratch', coords, normalization, provenance)
    if (readback['raw_payload_sha256'] != worker['output']['sha256']
            or readback['noise_payload_sha256'] != worker['input']['x']['sha256']):
        raise ValueError('Saved raw/noise payload differs from actual computation')
    return dict(classification='PASS', harness_commit=commit, results_sha256=prior.digest(prior.read(EVIDENCE / 'run/results.json')),
        readback=readback, scope='Original separate-root provenance; actual post-exit CPU readback, no neural execution')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--harness-commit')
    args = parser.parse_args()
    historical.forward.teardown.verify_workspace()
    if sum((args.prepare, args.worker, args.verify)) > 1:
        raise ValueError('Choose exactly one operation')
    if args.prepare:
        row = prepare()
        print(json.dumps({k: row[k] for k in ('token_count', 'checkpoint_sha256', 'ram', 'private_cpu_admission_sha256')}))
        return 0
    if args.worker:
        if args.harness_commit or prior.zombie.plain_path(args.output, existing=True) != EVIDENCE / 'run/scratch':
            raise ValueError('Exact owned Shape-SLat worker output required')
        commit = os.environ.get(ENV_COMMIT)
        valid_commit(commit)
        os.environ['MODLY_SAMPLING_HARNESS_COMMIT'] = commit
        runtime = Runtime()
        return runtime.forward.worker(args.output, sampling_factory=runtime.factory, runtime=runtime)
    valid_commit(args.harness_commit)
    if args.verify:
        row = verify_saved(args.harness_commit)
        prior.save(PRIVATE / 'independent-readback.json', row)
        print(json.dumps(row))
        return 0
    runtime = Runtime()
    runtime.verify_preflight()
    historical.admission()
    import psutil
    if psutil.disk_usage(str(PRIVATE)).free < 2*1024**3:
        raise RuntimeError('Artifact destination below original disk floor')
    os.environ[ENV_COMMIT] = args.harness_commit
    os.environ['MODLY_SAMPLING_HARNESS_COMMIT'] = args.harness_commit
    result = runtime.forward.supervisor.supervise(EVIDENCE / 'run', args.harness_commit, profile=runtime.profile())
    if result:
        return result
    row = verify_saved(args.harness_commit)
    prior.save(PRIVATE / 'host-readback.json', row)
    print(json.dumps(dict(shape_slat='PASS', token_count=row['readback']['token_count'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
