"""CPU-only independent readers for the one completed brick texture sample."""
import datetime
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
OCCUPANCY = ROOT / 'docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/occupancy'
TEXTURE = OCCUPANCY / 'texture'
ARTIFACT_ROOT = Path('C:/Users/wattz/Documents/ModlyArtifacts/amd-brick-material-stage1-df41688')
COMMIT = '6875a1e7b1f27e3a79d4b5ce2ddd939dccbd4a69'
EXPECTED_SHA = '4c74343c961e74bf3cfde4c24bd2a21b23063e17f0468b7f3049311fa9dc4289'
READABLE_TENSORS = {
    (ARTIFACT_ROOT / 'latent/texture-latent.safetensors').resolve(),
    (OCCUPANCY / 'dino/run/scratch/conditioning.safetensors').resolve(),
    (OCCUPANCY / 'sampling/run/scratch/sparse-structure-latent.safetensors').resolve(),
}


def guard(event, args):
    if event == 'import' and str(args[0]).split('.')[0] in {
        'torch', 'triton', 'trellis2', 'flex_gemm', 'o_voxel', 'cumesh'
    }:
        raise RuntimeError('Independent review may not import runtime/model packages')
    if event.startswith('socket.'):
        raise RuntimeError('Independent review is offline')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        import os
        path = Path(os.fsdecode(args[0]))
        if path.suffix.lower() in {'.safetensors', '.pt', '.pth', '.bin', '.gguf', '.ckpt'}:
            if path.resolve() not in READABLE_TENSORS:
                raise RuntimeError('Independent review may not open model weight payloads')


sys.addaudithook(guard)
sys.path.insert(0, str(ROOT / 'scripts/compatibility'))
import numpy as np
import amd_generation_texture as adapter
import amd_texture_sampling_artifact as production
import amd_texture_sampling_contract as policy


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def load(path):
    return adapter.shape._json(path)


def unique(pairs):
    row = {}
    for key, value in pairs:
        if key in row:
            raise ValueError('Duplicate public-format JSON key')
        row[key] = value
    return row


def main():
    result = load(TEXTURE / 'run/results.json')
    worker = adapter.shape.require_released(result)
    process = result['process']
    post = load(TEXTURE / 'post-exit-artifact-verification.json')
    attempt = load(TEXTURE / 'attempt-started.json')
    published = load(EVIDENCE / 'published-harness.json')
    before = load(TEXTURE / 'run/checkpoint-before.json')
    after = load(TEXTURE / 'run/checkpoint-after.json')
    committed_receipt = load(TEXTURE / 'run/committed-harness.json')
    assert committed_receipt['commit'] == COMMIT
    committed = committed_receipt['verified_file_sha256']
    assert result['classification'] == 'PASS'
    assert adapter.classify(worker, True, result['observer_healthy'], result['checkpoint_unchanged']) == 'PASS'
    assert before == after
    assert process['harness_commit'] == attempt['harness_commit'] == published['harness_commit'] == COMMIT
    assert published['classification'] == 'PASS' and published['remote_branch_commit_verified'] is True
    assert (attempt['maximum_sequences'], attempt['maximum_steps'], attempt['maximum_forwards']) == (1, 12, 12)
    assert len(committed) == published['committed_source_files'] == 67
    assert committed == published['verified_file_sha256']
    runtime = adapter.Runtime(adapter.context(OCCUPANCY / 'shape', TEXTURE, ARTIFACT_ROOT, COMMIT))
    bundle = runtime.bundle
    assert worker['token_count'] == len(bundle['coords']) == 4430
    assert worker['generation_inputs'] == bundle['lineage']
    assert adapter.original.supervisor.verify_committed_files(COMMIT, runtime.profile().committed_files) == committed

    saved = worker['artifacts']
    path = Path(saved['artifact_path'])
    assert path == ARTIFACT_ROOT / 'latent/texture-latent.safetensors'
    verified = production.verify(path, path.parent, saved['metadata'], token_count=4430, maximum_tokens=4430)
    assert verified == saved == worker['live_artifact_verification'] == post['artifact']
    assert post['classification'] == 'PASS' and post['verified'] is True and post['worker_exit_verified'] is True

    # A separate public-format reader does not call the production parser.
    blob = path.read_bytes()
    assert len(blob) == 1814784 and digest(blob) == EXPECTED_SHA == saved['metadata']['artifact_sha256']
    header_bytes = struct.unpack('<Q', blob[:8])[0]
    assert 0 < header_bytes <= 65536
    header = json.loads(blob[8:8 + header_bytes], object_pairs_hook=unique)
    assert set(header) == {'__metadata__', 'raw', 'denormalized', 'coords', 'noise'}
    arrays, tensor_rows, intervals = {}, {}, []
    for name in ('raw', 'denormalized', 'coords', 'noise'):
        row = header[name]
        shape = [4430, 4 if name == 'coords' else 32]
        dtype = 'I32' if name == 'coords' else 'F32'
        assert row['shape'] == shape and row['dtype'] == dtype
        start, end = row['data_offsets']
        assert type(start) is int and type(end) is int and 0 <= start < end <= 4430 * 400
        data = blob[8 + header_bytes + start:8 + header_bytes + end]
        assert len(data) == int(np.prod(shape)) * 4
        array = np.frombuffer(data, dtype='<i4' if name == 'coords' else '<f4').reshape(shape)
        assert np.isfinite(array).all()
        assert digest(data) == saved['metadata']['tensors'][name]['sha256']
        arrays[name] = array
        tensor_rows[name] = dict(dtype=dtype, shape=shape, tensor_bytes=len(data), sha256=digest(data), all_finite=True)
        intervals.append((start, end))
    intervals.sort()
    assert intervals[0][0] == 0 and intervals[-1][1] == 4430 * 400
    assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
    assert len(blob) == 8 + header_bytes + 4430 * 400
    public_provenance = json.loads(header['__metadata__']['provenance'], object_pairs_hook=unique)
    provenance = saved['metadata']['provenance']
    assert public_provenance == provenance
    assert provenance['harness_commit'] == COMMIT and provenance['generation_inputs'] == bundle['lineage']
    assert provenance['checkpoint'] == worker['provenance'] and provenance['input'] == worker['input']
    assert provenance['parameters'] == json.loads(json.dumps(policy.PARAMS))
    assert provenance['shape_normalization'] == policy.SHAPE_NORMALIZATION
    assert provenance['texture_normalization'] == policy.TEXTURE_NORMALIZATION
    assert provenance['seed'] == policy.SEED and provenance['sigma_min'] == policy.SIGMA_MIN
    assert provenance['checkpoint']['checkpoint_sha256'] == before['checkpoint_sha256'] == adapter.immutable.CHECKPOINT_SHA
    assert np.array_equal(arrays['coords'], bundle['coords'])
    assert tensor_rows['raw']['sha256'] == worker['output']['sha256']
    assert tensor_rows['coords']['sha256'] == worker['input']['coords']['sha256']
    assert tensor_rows['noise']['sha256'] == worker['input']['x']['sha256']
    mean = np.asarray(policy.TEXTURE_NORMALIZATION['mean'], dtype=np.float32)
    std = np.asarray(policy.TEXTURE_NORMALIZATION['std'], dtype=np.float32)
    assert np.array_equal(arrays['raw'] * std[None] + mean[None], arrays['denormalized'])
    assert worker['input']['shape_cond']['sha256'] == bundle['lineage']['shape_condition_sha256']
    assert worker['input']['cond']['sha256'] == bundle['feature']['output_sha256']
    assert worker['provenance']['conditioning'] == bundle['feature']
    sidecar = Path(saved['metadata_path'])
    assert load(sidecar) == saved['metadata']
    assert all(value == 12 for value in worker['calls'].values()) and len(worker['calls']) == 91
    assert worker['cpu_load']['payload_entries'] == 640 and worker['cpu_load']['reversible_widening_count'] == 220
    assert worker['gpu_operation_counts'].get('aten._scaled_dot_product_attention_math.default', 0) > 0 or any('scaled_dot_product' in key for key in worker['gpu_operation_counts'])

    steps = worker['sampling']['steps']
    report = dict(
        schema='modly.amd.brick-texture-sampling-outcome-review.v1', classification='PASS',
        reviewer='/root/brick_texture_sampling_review', reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='One genuine texture-SLat sample and persisted latent/provenance only; no texture decoder, PBR fields, baking, export or visual-material proof.',
        harness_commit=COMMIT, attempts_observed=1, published_67_file_closure_verified=True,
        raw_receipt_sha256={name: digest((TEXTURE / name).read_bytes()) for name in ('run/results.json', 'post-exit-artifact-verification.json', 'attempt-started.json')},
        production_reader_verified=True, separate_public_format_reader_verified=True,
        artifact=dict(path=str(path), artifact_bytes=len(blob), artifact_sha256=digest(blob), header_bytes=header_bytes,
                      metadata_bytes=sidecar.stat().st_size, metadata_sha256=digest(sidecar.read_bytes()), tensors=tensor_rows),
        lineage=bundle['lineage'], original_coordinate_order_byte_exact=True,
        texture_denormalization_byte_exact=True, stored_noise_matches_recorded_initial_noise=True,
        noise_regenerated=False, checkpoint_unchanged=True, checkpoint_sha256=before['checkpoint_sha256'],
        sequence=dict(sequences=worker['sampling']['sequences'], steps=len(steps), started=worker['sampling']['started'],
                      completed=worker['sampling']['completed'], positive_forwards=worker['sampling']['positive_calls'],
                      negative_forwards=worker['sampling']['negative_calls'], concat_calls=worker['sampling']['concat_calls'],
                      genuine_modules=len(worker['calls']), calls_per_module=12,
                      parameters=provenance['parameters'], sigma_min=policy.SIGMA_MIN, seed=provenance['seed'],
                      steps_observed=steps),
        strict_cpu_gate=dict(strict=worker['cpu_load']['strict'], assign=worker['cpu_load']['assign'],
                             missing_keys=worker['cpu_load']['missing_keys'], unexpected_keys=worker['cpu_load']['unexpected_keys'],
                             payload_entries=640, reversible_widenings=220),
        timing_seconds=dict(strict_cpu_load=worker['cpu_load']['elapsed_seconds'], placement=worker['placement_seconds'],
                            sampling=worker['forward_seconds'], owned_worker=process['elapsed_seconds']),
        gpu_memory_before_placement=worker['before_placement'], gpu_memory_after_forward=worker['after_forward'],
        minimum_observed_step_free_vram_bytes=min(row['memory']['available_bytes'] for row in steps),
        host_peak_bytes=worker['release']['cpu_memory'], budget=worker['budget'], transfers=worker['transfers'],
        actual_tunable_op_import=worker['tunable_import'], actual_tunable_op_before_forward=worker['tunable_before_forward'],
        observed_sdpa_operation_counts={key: value for key, value in worker['gpu_operation_counts'].items() if 'scaled_dot_product' in key},
        observer_healthy=result['observer_healthy'], periodic_windows_samples=process['windows_periodic_samples'],
        natural_owned_job_exit=process['job_cleanup'], worker_exit_verified=process['worker_exit_verified'],
        owned_release=worker['release'],
        reviewer_boundaries=dict(torch_imported='torch' in sys.modules, model_weights_opened=False, GPU_executed=False,
                                 WSL_executed=False, network_used=False, tensor_outputs_published=False, photos_published=False),
    )
    assert report['reviewer_boundaries']['torch_imported'] is False
    destination = EVIDENCE / 'independent-texture-sampling-outcome.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(classification='PASS', report=str(destination), artifact_sha256=EXPECTED_SHA,
                          tokens=4430, steps=12, positive_forwards=12, negative_forwards=0,
                          torch_imported=False)))


if __name__ == '__main__':
    main()
