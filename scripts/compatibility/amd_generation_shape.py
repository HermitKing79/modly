"""One fresh, bounded shape sequence from verified per-run native occupancy.

This adapter does not launch the application. The existing Windows supervisor
owns its sole model worker and retains its original observer, Job and budgets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from functools import partial
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
APPROVED_ROOT = ROOT / 'docs/compatibility/amd-repeatable-generation-v1-evidence'
CONTEXT_ENV = 'MODLY_GENERATION_SHAPE_CONTEXT'
MAX_TOKENS = 4430
HISTORICAL_RESULTS = ROOT / 'docs/compatibility/amd-shape-slat-512-sampling-evidence/run/results.json'
SCHEMA = 'modly.amd.generation-shape-inputs.v1'
LIVE_FREE_RESERVE = 3 * 1024**3


from amd_shape_sampling_control import ShapeSamplingControl


class QualifiedShapeControl(ShapeSamplingControl):
    """New adapter's additional live 2 GiB margin plus 1 GiB opaque reserve."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.minimum_observed_free_vram_bytes = None

    def block_memory(self, row):
        available = row.get('available_bytes')
        if type(available) is not int or available < LIVE_FREE_RESERVE:
            raise RuntimeError('Qualified shape block has less than 3 GiB free VRAM')
        self.minimum_observed_free_vram_bytes = available if self.minimum_observed_free_vram_bytes is None else min(self.minimum_observed_free_vram_bytes, available)
        super().block_memory(row)

    def report(self):
        return {**super().report(), 'minimum_observed_free_vram_bytes': self.minimum_observed_free_vram_bytes,
                'minimum_required_free_vram_bytes': LIVE_FREE_RESERVE}


def validate_count(value):
    if type(value) is not int or not 0 < value <= MAX_TOKENS:
        raise ValueError('Fresh shape token count must be in the measured input range 1..4430')
    return value


def coordinates(logits, mask):
    from amd_shape_slat_contract import reduce_occupancy
    _, coords = reduce_occupancy(logits, mask)
    validate_count(len(coords))
    return coords


def context(occupancy_evidence, evidence, harness_commit):
    from amd_generation_occupancy import plain_path
    occupancy_evidence, evidence = plain_path(occupancy_evidence), plain_path(evidence)
    approved = (APPROVED_ROOT / 'new-image').resolve()
    try:
        occupancy_evidence.relative_to(approved)
        evidence.relative_to(approved)
    except ValueError as exc:
        raise ValueError('Only new-image evidence under the repeatable-generation root is permitted') from exc
    if evidence == occupancy_evidence or evidence in occupancy_evidence.parents or evidence.name != 'shape':
        raise ValueError('Use a separate owned shape evidence directory')
    if type(harness_commit) is not str or re.fullmatch('[0-9a-f]{40}', harness_commit) is None:
        raise ValueError('An exact published harness commit is required')
    return dict(occupancy_evidence=str(occupancy_evidence), evidence=str(evidence), harness_commit=harness_commit)


def require_released(result, stage=None):
    """Reject claimed success when the source worker/owned tree was not released."""
    try:
        worker, process = result['worker'], result['process']
        if (result['classification'] != 'PASS' or result['observer_healthy'] is not True
                or result['checkpoint_unchanged'] is not True or worker['status'] != 'PASS'
                or worker['stage'] != 'complete' or type(process['returncode']) is not int or process['returncode'] != 0
                or process['timed_out'] is not False):
            raise ValueError('Upstream stage did not pass')
        for key in ('worker_exit_verified', 'gpu_authorized', 'output_readers_completed', 'cpu_observer_stopped',
                    'windows_periodic_stopped', 'diagnostic_journal_stopped', 'windows_journal_stopped',
                    'job_attached_before_preparation'):
            if process[key] is not True:
                raise ValueError('Upstream ownership/observer evidence incomplete: ' + key)
        for key in ('forced_cleanup', 'cleanup_unverified', 'error', 'cpu_observer_error', 'windows_periodic_error',
                    'diagnostic_journal_error', 'windows_journal_error', 'job_cleanup_error', 'postcheck_error'):
            if process.get(key):
                raise ValueError('Upstream cleanup or observer failed: ' + key)
        cleanup = process['job_cleanup']
        if cleanup['owned_tree_exit_verified'] is not True or cleanup['forced_job_cleanup'] is not False:
            raise ValueError('Upstream owned tree did not exit naturally')
        from amd_decoder_contract import validate_release
        validate_release(worker['release'])
        if stage == 'dino':
            if worker['release'].get('output_collected') is not True or worker['release'].get('rope_cache_entries_after') != 0:
                raise ValueError('Upstream DINO output/cache was not released')
        elif worker['release'].get('owned_tensors_collected') is not True:
            # Decoder records its input/output references individually.
            if worker['release'].get('input_collected') is not True or worker['release'].get('output_collected') is not True:
                raise ValueError('Upstream tensors were not released')
    except (KeyError, TypeError) as exc:
        raise ValueError('Incomplete upstream PASS/release evidence') from exc
    return worker


def classify(record, process_ok, healthy, preserved):
    from amd_shape_sampling_contract import classify as historical_classify
    try:
        count = validate_count(record['token_count'])
        lineage = record['generation_inputs']
        if lineage['schema'] != SCHEMA or lineage['token_count'] != count:
            raise ValueError('Fresh input contract/count differs')
        for key in ('input_contract_sha256', 'coords_payload_sha256', 'feature_file_sha256', 'occupancy_provenance_sha256'):
            if type(lineage[key]) is not str or re.fullmatch('[0-9a-f]{64}', lineage[key]) is None:
                raise ValueError('Fresh input digest is missing')
        if lineage['coords_payload_sha256'] != record['input']['coords']['sha256']:
            raise ValueError('Fresh coordinates differ from the verified input')
        if count > 3743:
            if lineage['resource_qualification'] != resource_qualification(count):
                raise ValueError('Measured input resource qualification differs')
            samples = record['sampling']
            if (samples['minimum_required_free_vram_bytes'] != LIVE_FREE_RESERVE
                    or type(samples['minimum_observed_free_vram_bytes']) is not int
                    or samples['minimum_observed_free_vram_bytes'] < LIVE_FREE_RESERVE):
                raise ValueError('Qualified live free-VRAM reserve is not evidenced')
    except (KeyError, ValueError, TypeError):
        return 'PARTIAL'
    return historical_classify(record, process_ok, healthy, preserved, token_count=count, maximum_tokens=MAX_TOKENS)


def _json(path, limit=16 * 1024 * 1024):
    from amd_feature_checkpoint import unique
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Invalid bounded evidence file: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique)


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resource_qualification(token_count):
    """CPU admission basis for the measured input; live guards remain decisive."""
    import math
    import amd_shape_sampling_contract as policy
    count = validate_count(token_count)
    baseline = _json(HISTORICAL_RESULTS)
    worker = require_released(baseline)
    if worker['token_count'] != 3743 or policy.classify(worker, True, True, True) != 'PASS':
        raise ValueError('Historical full shape sequence is not a qualified resource baseline')
    peak = worker['after_forward']
    ratio = count / 3743
    estimate = math.ceil(peak['peak_reserved_bytes'] * max(1.0, ratio) ** 2)
    self_scores = 12 * count * count * 4
    cross_scores = 12 * count * 1029 * 4
    # Deliberately count two dense score/softmax buffers for both attention
    # paths, QKV, two FFN activations, four residuals, RoPE and sampler history.
    # SDPA may avoid materializing scores; this remains an estimate, not a bound.
    visible = 2 * self_scores + 2 * cross_scores + count * 1536 * 3 * 4
    visible += 2 * count * 8192 * 4 + 4 * count * 1536 * 4 + count * 64 * 8 + 28 * count * 32 * 4
    if estimate >= policy.MEMORY_LIMIT or visible >= policy.MEMORY_LIMIT:
        raise ValueError('Estimated candidate buffers exceed the unchanged allocator budget')
    return dict(token_count=count, maximum_token_count=MAX_TOKENS, historical_token_count=3743,
        token_ratio=ratio, historical_peak_allocated_bytes=peak['peak_allocated_bytes'],
        historical_peak_reserved_bytes=peak['peak_reserved_bytes'], historical_forward_seconds=worker['forward_seconds'],
        historical_results_sha256=_digest(HISTORICAL_RESULTS), whole_peak_quadratic_estimate_bytes=estimate,
        self_attention_scores_f32_bytes=self_scores, cross_attention_scores_f32_bytes=cross_scores,
        conservative_visible_buffers_bytes=visible, estimate_is_guarantee=False,
        basis='whole historical reserved peak scaled by max(1,N/3743)^2; visible F32 buffer accounting; inference only',
        allocator_limit_bytes=policy.MEMORY_LIMIT, minimum_free_vram_bytes=policy.MEMORY_LIMIT + 1024**3,
        vram_reserve_bytes=1024**3, required_ram_bytes=policy.REQUIRED_RAM,
        preplacement_reserve_policy='13 GiB free before placement minus 12 GiB allocator cap; historical policy unchanged',
        minimum_live_free_vram_bytes=LIVE_FREE_RESERVE, live_free_margin_bytes=2*1024**3,
        opaque_workspace_reserve_bytes=1024**3,
        cpu_seconds=policy.CPU_SECONDS, gpu_seconds=policy.GPU_SECONDS, overall_seconds=policy.OVERALL_SECONDS,
        live_memory_observer_and_allocator_guards_required=True)


def input_bundle(occupancy_evidence):
    """Revalidate original receipts and exact fresh artifacts, never old turret inputs."""
    import numpy as np
    from amd_generation_occupancy import verify_sequence
    evidence = Path(occupancy_evidence).resolve(strict=True)
    sequence = verify_sequence(evidence)
    for stage in ('dino', 'sampling', 'decoder'):
        require_released(_json(evidence / stage / 'run/results.json'), stage)
    feature, latent, occupancy = sequence['feature'], sequence['latent'], sequence['occupancy']
    from amd_feature_checkpoint import FIELDS, verify
    feature_path = Path(feature['path']).resolve(strict=True)
    feature_path.relative_to(evidence)
    fresh_feature = verify(feature_path, feature_path.parent, {key: feature['provenance'][key] for key in FIELDS}, feature['output_sha256'])
    if fresh_feature != feature:
        raise ValueError('DINO feature identity differs from the verified sequence')
    arrays = {}
    for name in ('occupancy-logits.npy', 'occupancy-mask.npy'):
        row = occupancy['files'][name]
        path = Path(row['path']).resolve(strict=True)
        path.relative_to(evidence)
        if path.is_symlink() or path.stat().st_size != row['file_bytes'] or _digest(path) != row['file_sha256']:
            raise ValueError('Fresh native occupancy file identity differs')
        array = np.load(path, allow_pickle=False, max_header_size=4096)
        if list(array.shape) != row['shape'] or str(array.dtype) != row['dtype'] or hashlib.sha256(array.tobytes()).hexdigest() != row['payload_sha256']:
            raise ValueError('Fresh native occupancy payload differs')
        arrays[name] = array
    coords = coordinates(arrays['occupancy-logits.npy'], arrays['occupancy-mask.npy'])
    provenance_path = Path(occupancy['files']['occupancy-logits.npy']['path']).parent / 'occupancy-provenance.json'
    if _json(provenance_path) != occupancy:
        raise ValueError('Occupancy provenance/receipt mismatch')
    lineage = dict(schema=SCHEMA, token_count=len(coords), input_contract_sha256=sequence['input_contract_sha256'],
                   coords_payload_sha256=hashlib.sha256(coords.tobytes()).hexdigest(), feature_file_sha256=feature['file_sha256'],
                   occupancy_provenance_sha256=_digest(provenance_path), occupancy_harness_commit=sequence['harness_commit'],
                   feature_payload_sha256=feature['output_sha256'], sparse_latent_file_sha256=latent['file_sha256'])
    return dict(feature=feature, latent=latent, occupancy=occupancy, coords=coords, lineage=lineage,
                logits_sha256=occupancy['files']['occupancy-logits.npy']['file_sha256'],
                mask_sha256=occupancy['files']['occupancy-mask.npy']['file_sha256'])


def validate_input_rows(rows, feature, coords, lineage):
    count = validate_count(len(coords))
    digest = hashlib.sha256(coords.tobytes()).hexdigest()
    if lineage['token_count'] != count or lineage['coords_payload_sha256'] != digest:
        raise ValueError('Verified per-run coordinate identity changed')
    for name, shape, dtype in (('x', [count, 32], 'torch.float32'), ('coords', [count, 4], 'torch.int32'),
                                ('cond', [1, 1029, 1024], 'torch.float32')):
        row = rows[name]
        if any(row.get(key) != value for key, value in dict(shape=shape, dtype=dtype, device='cpu').items()):
            raise ValueError('Fresh CPU input layout differs: ' + name)
        if type(row.get('sha256')) is not str or re.fullmatch('[0-9a-f]{64}', row['sha256']) is None:
            raise ValueError('Fresh CPU input hash missing: ' + name)
    if rows['coords']['sha256'] != digest or rows['cond']['sha256'] != feature['output_sha256']:
        raise ValueError('Fresh feature/coordinate values differ')
    return lineage.copy()


def worker_context(value, output):
    parsed = json.loads(value)
    if type(parsed) is not dict or set(parsed) != {'occupancy_evidence', 'evidence', 'harness_commit'}:
        raise ValueError('Exact per-run shape context is required')
    result = context(**parsed)
    if Path(output).resolve() != Path(result['evidence']) / 'run/scratch':
        raise ValueError('Worker output must be the exact owned shape run scratch')
    return result


class Runtime:
    """Bind fresh inputs while preserving genuine factories and all worker guards."""
    def __init__(self, config):
        import amd_shape_slat_forward as forward
        import amd_shape_sampling as historical
        self.forward, self.historical = forward, historical
        self.config = context(**config)
        self.occupancy_evidence = Path(config['occupancy_evidence'])
        self.evidence = Path(config['evidence'])
        self.entry = self
        self.authorize = historical.AUTHORIZE
        import amd_shape_sampling_artifact as artifact
        self.artifact = SimpleNamespace(denormalize=partial(artifact.denormalize, maximum_tokens=MAX_TOKENS),
                                       write=partial(artifact.write, maximum_tokens=MAX_TOKENS))
        # Worker construction must not inspect inputs before the original owned
        # Job handshake. Input validation is deferred to verify_preflight().
        self._bundle = None
        original = forward.local
        self.local = SimpleNamespace(**{key: getattr(original, key) for key in
            ('ROOT', 'SOURCE', 'VENV', 'loading', 'digest', 'manifest', 'PREFIX', 'AUDIT', 'PINS',
             'rope_frequencies', 'import_class', 'load_cpu')})
        self.local.EVIDENCE = self.evidence
        self.local.FEATURE = self.occupancy_evidence / 'dino/run/scratch/conditioning.safetensors'
        self.local.verify_inputs = self.verify_inputs
        latent_path = self.occupancy_evidence / 'sampling/run/scratch/sparse-structure-latent.safetensors'
        self.readable_tensor_paths = (latent_path,)

    @property
    def bundle(self):
        if self._bundle is None:
            self._bundle = input_bundle(self.occupancy_evidence)
            if Path(self._bundle['feature']['path']).resolve() != self.local.FEATURE:
                raise ValueError('Verified feature is not the exact owned DINO scratch artifact')
            if Path(self._bundle['latent']['path']).resolve() != self.readable_tensor_paths[0]:
                raise ValueError('Verified latent is not the exact owned sparse sampling artifact')
            self.local.LOGITS_SHA = self._bundle['logits_sha256']
            self.local.MASK_SHA = self._bundle['mask_sha256']
        return self._bundle

    def sources(self):
        return self.historical.sources()

    def verify_sampler(self):
        """Reuse source/config pins; fresh coordinates have their own input receipt."""
        import amd_shape_sampling_contract as policy
        pins = _json(self.historical.PINS)
        pipeline = _json(self.historical.PIPELINE)['args']
        if (pins['sampler_sources'] != self.sources() or pins['dependencies'] != self.historical.dependencies()
                or pins['pipeline_sha256'] != _digest(self.historical.PIPELINE)
                or pins['normalization'] != pipeline['shape_slat_normalization']):
            raise ValueError('Immutable sampler/pipeline source contract changed')
        expected = dict(steps=12, rescale_t=3.0, guidance_strength=7.5, guidance_rescale=.5, guidance_interval=[.6, 1.0])
        configured = pipeline['shape_slat_sampler']
        if (configured['name'] != 'FlowEulerGuidanceIntervalSampler'
                or configured['args'] != {'sigma_min': policy.SIGMA_MIN} or configured['params'] != expected):
            raise ValueError('Canonical shape sampler parameters differ')
        policy.require_count(21, 12, 9)
        return pins

    def verify_inputs(self):
        import numpy as np
        expected = self.bundle
        fresh = input_bundle(self.occupancy_evidence)
        if (fresh['lineage'] != expected['lineage'] or fresh['feature'] != expected['feature']
                or fresh['latent'] != expected['latent'] or fresh['occupancy'] != expected['occupancy']
                or not np.array_equal(fresh['coords'], expected['coords'])):
            raise ValueError('Per-run shape input changed after admission')
        from amd_shape_slat_contract import reduce_occupancy
        files = fresh['occupancy']['files']
        logits = np.load(files['occupancy-logits.npy']['path'], allow_pickle=False, max_header_size=4096)
        mask = np.load(files['occupancy-mask.npy']['path'], allow_pickle=False, max_header_size=4096)
        pooled, coords = reduce_occupancy(logits, mask)
        return fresh['feature'], pooled, coords

    def verify_preflight(self):
        """Historical CPU proof qualifies architecture, never this image's inputs."""
        original = self.forward.local
        original.verify_sources()
        self.forward.teardown.verify_workspace()
        self.forward.teardown.verify_binding_evidence()
        self.verify_sampler()
        probe = _json(self.forward.PROBE)
        row = probe.get('worker', {})
        if (probe.get('returncode') != 0 or probe.get('worker_exit_verified') is not True
                or probe.get('job_cleanup', {}).get('forced_job_cleanup') is not False
                or row.get('status') != 'PASS' or row.get('weights_loaded') is not False
                or row.get('gpu_initialized') is not False or row.get('strict_header_match') is not True
                or row.get('pooling_torch_equal') is not True or row.get('cpu_sparse_layout_verified') is not True
                or row.get('synthetic', {}).get('strict_positive') is not True
                or len(row.get('synthetic', {}).get('negative', [])) != 4):
            raise ValueError('Historical architecture/strict-load proof incomplete')
        from services.amd_model_metadata import json_metadata, safetensors_metadata
        config = json_metadata(original.PREFIX.with_suffix('.json'))
        current = safetensors_metadata(original.PREFIX.with_suffix('.safetensors'))
        audited = _json(original.AUDIT)
        for key in ('file_bytes', 'mtime_ns', 'header_bytes', 'header_sha256', 'tensor_count',
                    'parameter_count', 'tensors', 'tensor_data_bytes'):
            if current[key] != audited[key]:
                raise ValueError('Audited shape checkpoint changed: ' + key)
        if (config != audited['config']
                or _digest(original.SOURCE / 'trellis2/models/structured_latent_flow.py') != audited['source_module_sha256']):
            raise ValueError('Audited shape configuration/source changed')
        self.forward.contract.validate_header(config, current['tensors'])
        checkpoint_sha = _digest(original.PREFIX.with_suffix('.safetensors'))
        if checkpoint_sha != row['checkpoint_sha256']:
            raise ValueError('Architecture proof checkpoint identity changed')
        feature, _, coords = self.verify_inputs()
        return dict(current, config=config, config_sha256=_digest(original.PREFIX.with_suffix('.json')),
                    checkpoint_sha256=checkpoint_sha, token_count=validate_count(len(coords)), feature=feature,
                    generation_inputs=self.bundle['lineage'], resource_qualification=resource_qualification(len(coords)))

    def validate_inputs(self, rows, feature, coords):
        current_feature, _, current_coords = self.verify_inputs()
        if feature != current_feature:
            raise ValueError('Worker feature differs from admitted fresh feature')
        import numpy as np
        if not np.array_equal(coords, current_coords):
            raise ValueError('Worker coordinates differ from admitted fresh support')
        result = validate_input_rows(rows, feature, coords, self.bundle['lineage'])
        result['resource_qualification'] = resource_qualification(len(coords))
        return result

    def factory(self, torch):
        self.verify_sampler()
        import importlib
        import inspect
        sys.path.insert(0, str(self.local.SOURCE))
        cls = importlib.import_module('trellis2.pipelines.samplers.flow_euler').FlowEulerGuidanceIntervalSampler
        if Path(inspect.getfile(cls)).resolve() != self.local.SOURCE / 'trellis2/pipelines/samplers/flow_euler.py':
            raise ValueError('Sampler provenance differs')
        return QualifiedShapeControl(torch, cls, device='cuda:0', token_count=len(self.bundle['coords']), maximum_tokens=MAX_TOKENS)

    def profile(self):
        p = self.forward.profile()
        p.evidence, p.kind = self.evidence, 'shape-sampling'
        p.worker_script = Path(__file__).resolve()
        p.maximum_forwards, p.maximum_steps, p.maximum_sequences = 21, 12, 1
        p.authorize, p.classify, p.identity = self.authorize, classify, self.verify_preflight
        p.source_identity = lambda: (self.forward.local.verify_sources(), self.verify_sampler(),
                                     self.forward.teardown.verify_binding_evidence())
        old = self.forward.local
        old_inputs = {path.relative_to(ROOT).as_posix() for path in
                      (old.FEATURE, old.TURRET / 'occupancy-logits.npy', old.TURRET / 'occupancy-mask.npy',
                       old.TURRET / 'dino/run/results.json')}
        p.committed_files = [name for name in p.committed_files if name not in old_inputs]
        p.committed_files += ['scripts/compatibility/' + name + '.py' for name in
            ('amd_generation_shape', 'amd_generation_occupancy', 'amd_shape_sampling', 'amd_shape_sampling_control',
             'amd_shape_sampling_contract', 'amd_shape_sampling_artifact', 'amd_sparse_latent_checkpoint',
             'amd_object_occupancy', 'amd_sparse_occupancy', 'amd_sparse_occupancy_data', 'amd_sparse_occupancy_contract',
             'amd_sparse_sampling', 'amd_sparse_sampling_contract', 'test_amd_generation_shape')]
        p.committed_files.append(self.historical.PINS.relative_to(ROOT).as_posix())
        p.committed_files.append(HISTORICAL_RESULTS.relative_to(ROOT).as_posix())
        from amd_generation_occupancy import required_files
        p.committed_files += required_files(self.occupancy_evidence)
        p.committed_files = sorted(set(p.committed_files))
        return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--occupancy-evidence', type=Path)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--harness-commit')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.output is None or not os.environ.get(CONTEXT_ENV):
            raise ValueError('Owned worker requires explicit per-run context and output')
        config = worker_context(os.environ[CONTEXT_ENV], args.output)
        runtime = Runtime(config)
        os.environ['MODLY_SAMPLING_HARNESS_COMMIT'] = config['harness_commit']
        return runtime.forward.worker(args.output, sampling_factory=runtime.factory, runtime=runtime)
    if args.occupancy_evidence is None or args.evidence is None:
        parser.error('--occupancy-evidence and --evidence are required')
    config = context(args.occupancy_evidence, args.evidence, args.harness_commit)
    evidence = Path(config['evidence'])
    if evidence.exists():
        raise ValueError('Shape evidence already exists; automatic repeat/overwrite is prohibited')
    runtime = Runtime(config)
    identity = runtime.verify_preflight()
    admission = runtime.historical.admission()
    profile = runtime.profile()
    runtime.forward.supervisor.verify_committed_files(args.harness_commit, profile.committed_files)
    evidence.mkdir(parents=True, exist_ok=False)
    with (evidence / 'pre-execution-admission.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(context=config, ram=admission, generation_inputs=runtime.bundle['lineage'],
                       resource_qualification=identity['resource_qualification']), stream, indent=2)
    os.environ[CONTEXT_ENV] = json.dumps(config, sort_keys=True)
    os.environ['MODLY_SAMPLING_HARNESS_COMMIT'] = args.harness_commit
    return runtime.forward.supervisor.supervise(evidence / 'run', args.harness_commit, profile=profile)


if __name__ == '__main__':
    raise SystemExit(main())
