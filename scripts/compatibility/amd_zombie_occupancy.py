"""Saved zombie features -> one sparse sample -> conditional native occupancy.

Standalone research routing only. No DINO invocation or product authority.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amd_zombie_dinov3 as zombie
import amd_sparse_sampling as sampling
import amd_sparse_occupancy as occupancy
import amd_sparse_latent_checkpoint as latents

ROOT = zombie.ROOT
EVIDENCE = ROOT / 'artifacts/amd-zombie-occupancy-c570c2e'
DINO_COMMIT = '93269412602a338acd9fbde41798901ddb5f707d'
FEATURE_SHA = 'e5659b1842509e73ec496ef90bf95fae86c93ddf5ee73fba8015822982a3cb60'
STAGES = ('sampling', 'decoder')
LIMITS = {'sampling': (180, 240, 480), 'decoder': (120, 180, 360)}
NOISE_SHA = 'f76e94b25a4798a5da0e9de56667b5dda38e75bf12ccbc373521fbcd7af1a7b3'
# Freeze original static proof dependencies BEFORE replacing private artifacts.
_PROFILES = {'sampling': sampling.profile(), 'decoder': occupancy.profile()}
_FEATURE_READER = sampling.forward.local.verify_feature
_FLOW_PREFLIGHT = sampling.forward.verify_preflight
_LATENT_READER = occupancy.data.read_input
STATIC = ['scripts/compatibility/amd_zombie_occupancy.py',
          'scripts/compatibility/test_amd_zombie_occupancy.py',
          'scripts/compatibility/test_amd_zombie_occupancy_acceptance.py',
          'scripts/compatibility/amd_zombie_voxel_preview.py',
          'scripts/compatibility/test_amd_zombie_voxel_preview.py']
HISTORY = ['docs/compatibility/amd-zombie-dinov3-evidence/' + name for name in (
    'result.json', 'outcome-review.json', 'private-evidence-hashes.json', 'candidate-source-manifest.json')]
CPU_SUMMARY = ROOT / 'docs/compatibility/amd-zombie-occupancy-evidence/cpu-admission-summary.json'
CLOSURE = list(dict.fromkeys([*_PROFILES['sampling'].committed_files,
    *_PROFILES['decoder'].committed_files, *zombie.profile().committed_files, *STATIC, *HISTORY,
    CPU_SUMMARY.relative_to(ROOT).as_posix()]))


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def read(path, limit=8 * 1024**2):
    return zombie.bounded(path, limit)


def read_json(path, limit=8 * 1024**2):
    return json.loads(read(path, limit), object_pairs_hook=zombie.features.unique)


def save(path, value):
    zombie.save(path, value)


def valid_commit(commit):
    if (type(commit) is not str or len(commit) != 40 or commit == DINO_COMMIT
            or any(c not in '0123456789abcdef' for c in commit)):
        raise ValueError('A distinct exact new execution commit is required')


class OwnedInput:
    """Replay only the genuine first host line after validating attachment.

    Routing may read private CPU inputs only after this real receipt. The
    unmodified worker then consumes that same receipt once; its second read
    still waits on the original host's binary GPU authorization.
    """
    def __init__(self, stream):
        self.stream = stream
        self.first = stream.readline()
        if self.first != 'OWNED_JOB_ATTACHED\n':
            raise ValueError('Real owned Job attachment receipt required before input preparation')

    def readline(self, *args, **kwargs):
        if self.first is not None:
            first, self.first = self.first, None
            return first
        return self.stream.readline(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def prepared_contract():
    blob = read(EVIDENCE / 'cpu-admission.json')
    if digest(blob) != read_json(CPU_SUMMARY)['private_cpu_admission_sha256']:
        raise ValueError('Host-prepared CPU admission identity changed')
    return json.loads(blob, object_pairs_hook=zombie.features.unique)


def saved_features():
    """Authenticate real historical success, never create a new DINO receipt."""
    review = read_json(ROOT / HISTORY[1])
    run = zombie.EVIDENCE / 'run'
    for name in ('run/results.json', 'run/committed-harness.json', 'host-readback.json'):
        path = zombie.EVIDENCE / name
        if digest(read(path)) != review['private_evidence_sha256'][name]:
            raise ValueError('Historical genuine DINO receipt changed: ' + name)
    committed = read_json(run / 'committed-harness.json')
    if committed['commit'] != DINO_COMMIT:
        raise ValueError('Original DINO execution identity changed')
    public = read_json(ROOT / HISTORY[0])
    independent = ROOT.parent / 'artifacts/amd-zombie-dinov3-b8afdd5/independent-readback.json'
    if (digest(read(independent)) != public['independent_receipt_sha256']
            or public['harness_commit'] != DINO_COMMIT or public['file_sha256'] != FEATURE_SHA):
        raise ValueError('Original independent DINO success receipt changed')
    row = zombie.validate_saved(read_json(run / 'results.json'), DINO_COMMIT)
    if row['file_sha256'] != FEATURE_SHA or row['provenance']['harness_commit'] != DINO_COMMIT:
        raise ValueError('Only the saved genuine zombie features are admitted')
    return row


def bind_feature(row):
    local = sampling.forward.local
    local.FEATURE = Path(row['path']); local.FEATURE_SHA = row['file_sha256']
    local.verify_feature = partial(_FEATURE_READER, expected=row)
    sampling.forward.verify_preflight = partial(_FLOW_PREFLIGHT, feature_expected=row)
    local.EVIDENCE = EVIDENCE / 'sampling'


def configure(stage, commit):
    valid_commit(commit)
    if stage not in STAGES:
        raise ValueError('Only sparse sampling and occupancy decoding are authorised')
    profile = copy.copy(_PROFILES[stage]); profile.committed_files = list(CLOSURE)
    if stage == 'sampling':
        row = saved_features(); bind_feature(row)
        os.environ['MODLY_SAMPLING_HARNESS_COMMIT'] = commit
    else:
        row = saved_sampling(commit)
        occupancy.data.INPUT = Path(row['path']); occupancy.data.INPUT_SHA = row['file_sha256']
        occupancy.data.read_input = partial(_LATENT_READER, occupancy.data.INPUT, expected=row)
        occupancy.EVIDENCE = EVIDENCE / stage; occupancy.base.EVIDENCE = EVIDENCE / stage
        os.environ['MODLY_OCCUPANCY_HARNESS_COMMIT'] = commit
    profile.evidence = EVIDENCE / stage; profile.worker_script = Path(__file__).resolve()
    if (profile.cpu_seconds, profile.gpu_seconds, profile.overall_seconds) != LIMITS[stage]:
        raise ValueError('Original stage deadline contract changed')
    return profile


def validate_latent_provenance(provenance, feature, commit, identity, runtime):
    valid_commit(commit)
    if (provenance.get('harness_commit') != commit or commit == feature['provenance']['harness_commit']
            or feature['file_sha256'] != FEATURE_SHA
            or feature['provenance']['harness_commit'] != DINO_COMMIT
            or provenance.get('feature_file_sha256') != FEATURE_SHA
            or json.loads(provenance.get('feature_provenance', '{}')) != feature['provenance']
            or provenance.get('model_checkpoint_sha256') != identity['checkpoint_sha256']
            or provenance.get('model_config_sha256') != identity['config_sha256']
            or provenance.get('model_source_sha256') != zombie.dino.local.digest(
                sampling.forward.local.SOURCE / 'trellis2/models/sparse_structure_flow.py')
            or json.loads(provenance.get('sampler_sources', '{}')) != sampling.sources()
            or json.loads(provenance.get('runtime', '{}')) != runtime):
        raise ValueError('Fresh latent must preserve original DINO and new sampler lineage')
    core = {key: provenance[key] for key in latents.FIELDS}
    if latents.metadata(core, provenance['output_sha256']) != provenance:
        raise ValueError('Original sampler parameter/container semantics changed')


def stage_record(stage, commit):
    valid_commit(commit)
    result = read_json(EVIDENCE / stage / 'run/results.json')
    worker = result['worker']; process = result['process']
    classify = sampling.policy.classify if stage == 'sampling' else occupancy.contract.classify
    expected_command = [str(sampling.forward.local.VENV / 'Scripts/python.exe'), '-B', '-I',
        str(Path(__file__).resolve()), '--worker', '--output', str(EVIDENCE / stage / 'run/scratch')]
    cpu, gpu, overall = LIMITS[stage]
    if (classify(worker, True, True, True) != 'PASS' or result.get('classification') != 'PASS'
            or result.get('observer_healthy') is not True or result.get('checkpoint_unchanged') is not True
            or process.get('returncode') != 0 or process.get('harness_commit') != commit
            or process.get('command') != expected_command
            or process.get('stage_timeouts_seconds') != dict(cpu=cpu, forward_release=gpu, overall=overall)
            or process.get('timed_out') is not False or process.get('gpu_authorized') is not True
            or process.get('worker_exit_verified') is not True
            or any(process.get(k) for k in ('error', 'forced_cleanup', 'cleanup_unverified', 'job_cleanup_error',
                'job_close_error', 'postcheck_error', 'cpu_observer_error', 'diagnostic_journal_error',
                'windows_periodic_error', 'windows_journal_error'))
            or any(process.get(k) is not True for k in ('job_attached_before_preparation', 'output_readers_completed',
                'cpu_observer_stopped', 'diagnostic_journal_stopped', 'windows_periodic_stopped', 'windows_journal_stopped'))
            or process.get('job_cleanup', {}).get('owned_tree_exit_verified') is not True
            or process['job_cleanup'].get('forced_job_cleanup') is not False
            or process['job_cleanup'].get('active_before_cleanup') != 0
            or process['job_cleanup'].get('active_after_cleanup') != 0):
        raise ValueError('Original numerical, admission, observer or natural Job settlement gate failed')
    run = EVIDENCE / stage / 'run'
    committed = read_json(run / 'committed-harness.json')
    prepared = prepared_contract()
    expected = prepared['sampling_identity' if stage == 'sampling' else 'decoder_identity']
    if (committed['commit'] != commit or read_json(run / 'checkpoint-before.json') != expected
            or read_json(run / 'checkpoint-after.json') != expected
            or set(committed['verified_file_sha256']) != set(CLOSURE)
            or any(digest(read(ROOT / name)) != sha for name, sha in committed['verified_file_sha256'].items())):
        raise ValueError('Executed source or checkpoint preservation differs')
    runtime = worker['runtime']
    fixed = dict(python=sys.version, executable=str(sampling.forward.local.VENV / 'Scripts/python.exe'),
        torch='2.13.0+rocm10.0.0', torch_path=str(sampling.forward.local.VENV / 'Lib/site-packages/torch/__init__.py'),
        hip='7.15.26333', cuda=None)
    if stage == 'sampling':
        fixed.update(git_version='6596a0dea537c1c78520c73cc6678214781a9d06',
                     dependencies=dict(safetensors='0.8.0', numpy='2.5.3'))
    budget = dict(allocator_limit_bytes=12 * 1024**3, cpu_seconds=cpu, gpu_seconds=gpu, overall_seconds=overall)
    if stage == 'decoder':
        budget['minimum_available_bytes'] = 13 * 1024**3
    memory = worker['before_placement']
    ram = worker['ram_before_strict_load' if stage == 'sampling' else 'ram_before_load']
    required = sampling.policy.REQUIRED_RAM if stage == 'sampling' else 4 * 1024**3
    gate = read_json(run / 'cpu-gate.json')
    authorize = sampling.policy.authorize_placement if stage == 'sampling' else occupancy.base.loading_contract.authorize_placement
    authorize(gate.get('gate_one', {}), expected, expected['checkpoint_sha256'], gate.get('checkpoint_sha256'))
    if (runtime != fixed or worker['budget'] != budget or memory['allocated_bytes'] != 0
            or memory['reserved_bytes'] != 0 or memory['available_bytes'] < 13 * 1024**3
            or ram['required_bytes'] != required or ram['available_bytes'] < required
            or gate != dict(event='cpu-ready', gate_one=worker['cpu_load'], checkpoint_sha256=expected['checkpoint_sha256'])):
        raise ValueError('Actual runtime, CPU-ready, RAM or original GPU memory admission differs')
    if worker['release']['allocated_bytes'] != 0 or worker['release']['reserved_bytes'] != 0:
        raise ValueError('Owned Torch allocations were not completely released')
    return result


def saved_sampling(commit):
    result = stage_record('sampling', commit); worker = result['worker']
    feature = saved_features(); identity = read_json(EVIDENCE / 'sampling/run/checkpoint-before.json')
    row = worker['latent_checkpoint']
    validate_latent_provenance(row['provenance'], feature, commit, identity, worker['runtime'])
    if (worker['feature'] != feature or worker['input']['cond']['sha256'] != feature['output_sha256']
            or worker['input']['x']['sha256'] != NOISE_SHA or row['provenance']['noise_sha256'] != NOISE_SHA):
        raise ValueError('Actual noise or original conditioning payload changed')
    path = zombie.plain_path(EVIDENCE / 'sampling/run/scratch/sparse-structure-latent.safetensors', existing=True)
    core = {key: row['provenance'][key] for key in latents.FIELDS}
    restored = latents.verify(path, path.parent, core, worker['output']['sha256'])
    if restored != row:
        raise ValueError('Fresh latent independent container/tensor readback failed')
    return restored


def saved_occupancy(commit):
    import numpy as np
    result = stage_record('decoder', commit); worker = result['worker']
    latent = saved_sampling(commit); artifact = worker['occupancy_artifacts']
    provenance = artifact['provenance']; identity = worker['checkpoint_identity']
    if (worker['saved_latent'] != latent or provenance.get('input') != latent
            or provenance.get('harness_commit') != commit
            or provenance.get('decoder_checkpoint_sha256') != identity['checkpoint_sha256']
            or provenance.get('decoder_config_sha256') != identity['config_sha256']
            or provenance.get('decoder_source_sha256') != identity['source_module_sha256']
            or provenance.get('support_sources') != worker['verified_support_source']
            or provenance.get('pipeline_source_sha256') != occupancy.PIPELINE_SHA
            or provenance.get('runtime') != worker['runtime']
            or provenance.get('threshold') != 'decoded > 0'
            or provenance.get('conversion') != 'none; preserve native64cubed'
            or provenance.get('coordinate_order') != ['batch', 'D', 'H', 'W']):
        raise ValueError('Decoder provenance does not link the authentic new latent and original sources')
    scratch = EVIDENCE / 'decoder/run/scratch'
    if read_json(scratch / 'occupancy-provenance.json') != artifact:
        raise ValueError('Saved native occupancy sidecar differs')
    layouts = {'occupancy-logits.npy': ((1, 1, 64, 64, 64), np.dtype('<f4'), 2 * 1024**2),
        'occupancy-mask.npy': ((64, 64, 64), np.dtype('bool'), 512 * 1024),
        'occupancy-coordinates.npy': (None, np.dtype('<i4'), 5 * 1024**2)}
    if set(artifact['files']) != set(layouts):
        raise ValueError('Native occupancy artifact closure changed')
    arrays = {}
    for name, (shape, dtype, limit) in layouts.items():
        path = zombie.plain_path(scratch / name, existing=True); row = artifact['files'][name]; blob = read(path, limit)
        array = np.load(io.BytesIO(blob), allow_pickle=False, max_header_size=4096)
        if (str(path) != row['path'] or len(blob) != row['file_bytes'] or digest(blob) != row['file_sha256']
                or array.dtype != dtype or shape is not None and array.shape != shape
                or list(array.shape) != row['shape'] or not np.isfinite(array).all()
                or digest(array.tobytes()) != row['payload_sha256']):
            raise ValueError('Native array byte/numerical readback differs: ' + name)
        arrays[name] = array
    mask = arrays['occupancy-mask.npy']; xyz = np.argwhere(mask).astype('<i4')
    coordinates = np.concatenate((np.zeros((len(xyz), 1), dtype='<i4'), xyz), axis=1)
    if (not np.array_equal(mask, arrays['occupancy-logits.npy'][0, 0] > 0)
            or not np.array_equal(coordinates, arrays['occupancy-coordinates.npy'])
            or digest(arrays['occupancy-logits.npy'].tobytes()) != worker['forward']['output_sha256']
            or len(xyz) != artifact['occupied_count'] or occupancy.data.components(mask) != artifact['components']):
        raise ValueError('Original threshold, coordinate ordering or connectivity differs')
    return artifact


def decoder_preflight_without_new_input():
    current = occupancy.base.previous.identity()
    original = read_json(occupancy.base.OLD)['worker']['identity']
    if current['checkpoint_sha256'] != original['checkpoint_sha256']:
        raise ValueError('Original decoder checkpoint changed')
    occupancy.contract.fixture(current['config']); occupancy.base.source_identity()
    occupancy.teardown.verify_binding_evidence(); occupancy.cache_plan()
    if zombie.dino.local.digest(occupancy.PIPELINE) != occupancy.PIPELINE_SHA:
        raise ValueError('Original occupancy threshold source changed')
    return current


def prepare():
    occupancy.teardown.verify_workspace()
    feature = saved_features()
    original_dino = zombie.dino.local.identity()
    if original_dino['checkpoint_sha256'] != feature['provenance']['checkpoint_sha256']:
        raise ValueError('Original conditioning checkpoint provenance changed')
    bind_feature(feature)  # CPU-only artifact routing; no invented execution identity.
    sample = sampling.preflight(); decoder = decoder_preflight_without_new_input()
    ram = sampling.admission(); decoder_ram = occupancy.admission()
    import psutil
    free = psutil.disk_usage(str(ROOT)).free
    if free < 2 * 1024**3 or 'torch' in sys.modules:
        raise RuntimeError('Destination disk floor or CPU-only preparation failed')
    row = dict(feature=feature, original_dino_checkpoint_sha256=original_dino['checkpoint_sha256'],
        sampling_identity=sample, decoder_identity=decoder, sampler_ram=ram, decoder_ram=decoder_ram,
        destination_free_bytes=free, sampling_params=sampling.policy.PARAMS, seed=sampling.policy.SEED,
        sigma_min=sampling.policy.SIGMA_MIN, deadlines=LIMITS, gpu_memory_limit=12 * 1024**3,
        initial_gpu_free_required=13 * 1024**3, no_torch_import=True, weights_loaded=False,
        scope='CPU source/input/model metadata preparation only; no neural stage receipt')
    zombie.plain_path(EVIDENCE).mkdir(parents=True, exist_ok=False)
    for stage in STAGES:
        (EVIDENCE / stage).mkdir()
    save(EVIDENCE / 'cpu-admission.json', row)
    return row


def verify_stage(stage, commit):
    artifact = saved_sampling(commit) if stage == 'sampling' else saved_occupancy(commit)
    result_path = EVIDENCE / stage / 'run/results.json'
    save(EVIDENCE / (stage + '-independent-readback.json'), dict(stage=stage, harness_commit=commit,
        verified=True, results_sha256=digest(read(result_path)), artifact=artifact,
        scope='Separate post-exit CPU process; original bounded artifact reader; no neural imports'))
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', action='store_true')
    parser.add_argument('--stage', choices=STAGES); parser.add_argument('--verify', choices=STAGES)
    parser.add_argument('--preview', action='store_true'); parser.add_argument('--output', type=Path)
    parser.add_argument('--harness-commit')
    args = parser.parse_args(); occupancy.teardown.verify_workspace()
    if args.prepare:
        row = prepare(); print(json.dumps(dict(prepared=True, feature_sha256=FEATURE_SHA,
            sampling_checkpoint_sha256=row['sampling_identity']['checkpoint_sha256'],
            decoder_checkpoint_sha256=row['decoder_identity']['checkpoint_sha256'], ram=row['sampler_ram']))); return 0
    if args.worker:
        output = zombie.plain_path(args.output, existing=True)
        stage = next((name for name in STAGES if output == EVIDENCE / name / 'run/scratch'), None)
        if stage is None or args.stage or args.verify or args.preview:
            raise ValueError('Exact single-stage worker scratch required')
        commit = os.environ.get('MODLY_ZOMBIE_OCCUPANCY_COMMIT', '')
        original_stdin = sys.stdin
        attached_input = OwnedInput(original_stdin)
        configure(stage, commit)
        sys.stdin = attached_input
        try:
            return sampling.forward.worker(output, sampling_factory=sampling.factory) if stage == 'sampling' else occupancy.base.worker(output, occupancy=occupancy)
        finally:
            sys.stdin = original_stdin
    valid_commit(args.harness_commit)
    if args.verify:
        row = verify_stage(args.verify, args.harness_commit)
        print(json.dumps(dict(stage=args.verify, independent_readback='PASS', artifact=row.get('file_sha256', 'native NPY arrays')))); return 0
    if args.preview:
        artifact = saved_occupancy(args.harness_commit)
        receipt = read_json(EVIDENCE / 'decoder-independent-readback.json')
        if receipt['artifact'] != artifact or receipt['harness_commit'] != args.harness_commit:
            raise ValueError('Independent decoder readback required before preview')
        import numpy as np
        from amd_zombie_voxel_preview import produce
        mask = np.load(EVIDENCE / 'decoder/run/scratch/occupancy-mask.npy', allow_pickle=False, max_header_size=4096)
        directory = EVIDENCE / 'preview'; zombie.plain_path(directory).mkdir(exist_ok=False)
        row = produce(directory, mask); save(EVIDENCE / 'preview-result.json', row)
        print(json.dumps(row)); return 0
    if args.stage is None:
        raise ValueError('Select one authorised stage')
    prepared = prepared_contract()
    if prepared['feature'] != saved_features():
        raise ValueError('Prepared original feature identity changed')
    if args.stage == 'decoder':
        receipt = read_json(EVIDENCE / 'sampling-independent-readback.json')
        if (receipt['artifact'] != saved_sampling(args.harness_commit)
                or receipt['harness_commit'] != args.harness_commit
                or receipt['results_sha256'] != digest(read(EVIDENCE / 'sampling/run/results.json'))):
            raise ValueError('Independent sampler settlement and latent readback required first')
    profile = configure(args.stage, args.harness_commit)
    sampling.admission() if args.stage == 'sampling' else occupancy.admission()
    os.environ['MODLY_ZOMBIE_OCCUPANCY_COMMIT'] = args.harness_commit
    code = occupancy.base.supervise(EVIDENCE / args.stage / 'run', args.harness_commit, profile=profile)
    if code:
        return code
    row = saved_sampling(args.harness_commit) if args.stage == 'sampling' else saved_occupancy(args.harness_commit)
    save(EVIDENCE / (args.stage + '-host-readback.json'), row)
    print(json.dumps(dict(stage=args.stage, host_readback='PASS'))); return 0


if __name__ == '__main__':
    raise SystemExit(main())
