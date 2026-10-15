"""CPU-only independent post-exit texture artifact and result inspection."""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).parent
PERMANENT = Path('C:/Users/wattz/Documents/ModlyArtifacts/amd-turret-texture-4f845cd/latent')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run():
    result = read(EVIDENCE / 'run/results.json')
    worker, process = result['worker'], result['process']
    require(result['classification'] == 'PASS' and result['observer_healthy'] is True
            and result['checkpoint_unchanged'] is True, 'Host classification/health/identity')
    require(worker['status'] == 'PASS' and worker['stage'] == 'complete'
            and worker['load_calls'] == 1 and worker['forward_calls'] == 12, 'Worker completion/count')
    require(process['returncode'] == 0 and process['timed_out'] is False
            and process['gpu_authorized'] is True and process['job_attached_before_preparation'] is True
            and process['worker_exit_verified'] is True, 'Worker natural exit')
    cleanup = process['job_cleanup']
    require(cleanup['owned_tree_exit_verified'] is True and cleanup['forced_job_cleanup'] is False
            and cleanup['active_before_cleanup'] == cleanup['active_after_cleanup'] == 0, 'Owned Job natural exit')
    for name in ('cpu_observer', 'windows_periodic', 'diagnostic_journal', 'windows_journal'):
        require(process[name + '_stopped'] is True and not process.get(name + '_error'), 'Observer/journal: ' + name)
    require(process['output_readers_completed'] is True and not process.get('error'), 'Output/host error')
    require(0 <= worker['cpu_load']['elapsed_seconds'] <= 180
            and 0 <= worker['forward_seconds'] <= 240
            and 0 <= process['elapsed_seconds'] <= 480, 'Stage deadlines')
    require(worker['cpu_load']['strict'] is True and worker['cpu_load']['assign'] is True
            and worker['cpu_load']['payload_entries'] == 640
            and worker['cpu_load']['reversible_widening_count'] == 220
            and worker['cpu_load']['missing_keys'] == worker['cpu_load']['unexpected_keys'] == [], 'Strict load')
    for phase, initialized in [('tunable_import', False), ('tunable_before_forward', True)]:
        row = worker[phase]
        require(row['environment'] == {key: '0' for key in ('PYTORCH_TUNABLEOP_ENABLED',
                'PYTORCH_TUNABLEOP_TUNING', 'PYTORCH_TUNABLEOP_RECORD_UNTUNED')}
                and set(row['flags']) == {'is_enabled', 'tuning_is_enabled', 'record_untuned_is_enabled'}
                and all(type(v) is bool and v is False for v in row['flags'].values())
                and row['gpu_initialized'] is initialized, 'Actual TunableOp state: ' + phase)
    release = worker['release']
    require(release['allocated_bytes'] == release['reserved_bytes'] == 0
            and release['model_collected'] is True and release['owned_tensors_collected'] is True
            and worker['workspace_clear_calls'] == 1 and worker['workspace_binding']['verified'] is True, 'Release')
    require(worker['gpu']['architecture'].split(':')[0] == 'gfx1201'
            and 'RX 9070 XT' in worker['gpu']['name']
            and worker['runtime']['torch'] == '2.13.0+rocm10.0.0'
            and worker['runtime']['hip'] == '7.15.26333' and worker['runtime']['cuda'] is None, 'GPU/runtime provenance')
    budget = worker['budget']
    require(budget['allocator_limit_bytes'] == 12 * 1024**3
            and budget['free_reserve_bytes'] == 2 * 1024**3
            and budget['opaque_workspace_headroom_bytes'] == 1024**3
            and budget['effective_allocator_limit_bytes'] == min(12 * 1024**3,
                worker['before_placement']['available_bytes'] - 3 * 1024**3), 'Memory allowance')
    sampled = worker['sampling']
    require({key: sampled[key] for key in ('sequences', 'started', 'completed', 'positive_calls',
            'negative_calls', 'concat_calls')} == dict(sequences=1, started=12, completed=12,
                positive_calls=12, negative_calls=0, concat_calls=12), 'Sequence exact counts')
    modules = {'<root>': 1, **{f'blocks.{i}': 1 for i in range(30)},
               **{f'blocks.{i}.{name}': 1 for i in range(30) for name in ('self_attn', 'cross_attn')}}
    require(worker['calls'] == {key: 12 for key in modules}, 'All91 genuine module calls')
    seq = np.linspace(1, 0, 13)
    times = (3 * seq / (1 + 2 * seq)).tolist()
    require(len(sampled['models']) == len(sampled['steps']) == 12, 'Step/model records')
    for index, (model, step) in enumerate(zip(sampled['models'], sampled['steps'])):
        require(model == dict(index=index + 1, step=index + 1, role='positive', t=times[index],
                next_t=times[index + 1], modules=modules, frequency=1, cache_reused=index > 0), 'Source model ticket')
        require(step['step'] == step['forwards'] + index == index + 1
                and step['total_forwards'] == index + 1
                and step['t'] == times[index] and step['next_t'] == times[index + 1]
                and all(step[key] is True for key in ('all_finite', 'coords_unchanged', 'synchronized'))
                and 0 <= step['elapsed_seconds'] <= 240, 'Source step')
        mem = step['memory']
        require(all(0 <= mem[key] <= budget['effective_allocator_limit_bytes'] for key in
                ('allocated_bytes', 'reserved_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes'))
                and mem['available_bytes'] >= 3 * 1024**3, 'Step memory reserve/cap')
    transfers = {key: 1 for key in ('x', 'coords', 'batch_map', 'seqlen', 'cond', 'shape_cond',
                 'output', 'output_coords', 'rope_frequency')}
    transfers.update(t=12, frequency=12)
    require(worker['transfers'] == transfers, 'Transfer exact counts')
    for path in [EVIDENCE / 'run/windows-before.json', EVIDENCE / 'run/windows-cpu-gate.json',
                 EVIDENCE / 'run/windows-after-exit.json']:
        row = read(path)
        require(row['returncode'] == 0 and row['timed_out'] is False and not row['stderr']
                and not row['observation']['events']
                and all(c['Status'] == 'OK' for c in row['observation']['controllers']), 'Windows health: ' + path.name)
    periodic = [json.loads(line) for line in (EVIDENCE / 'run/windows-periodic.jsonl').read_text().splitlines()]
    require(len(periodic) == process['windows_periodic_samples'] >= 1
            and all(row['healthy'] is True for row in periodic), 'Periodic Windows health')
    pre = read(EVIDENCE / 'independent-preexecution-review.json')
    require(all(digest((ROOT / path).read_bytes()) == value for path, value in pre['source_sha256'].items()),
            'Executed sources differ from reviewed final candidate')
    committed = read(EVIDENCE / 'run/committed-harness.json')
    attempt = read(EVIDENCE / 'attempt-started.json')
    require(committed['commit'] == process['harness_commit'] == attempt['harness_commit']
            and attempt['maximum_forwards'] == attempt['maximum_steps'] == 12
            and attempt['maximum_sequences'] == 1, 'Committed one-attempt identity')
    # Production verifier is reopened in this independent CPU-only process.
    path = ROOT / 'scripts/compatibility/amd_texture_sampling_artifact.py'
    spec = importlib.util.spec_from_file_location('independent_texture_artifact_reader', path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    checked = reader.verify(PERMANENT / 'texture-latent.safetensors', PERMANENT, worker['artifacts']['metadata'])
    require(checked == worker['artifacts'], 'Independent production-reader result')
    # Independent format reader: decode the public header and payload directly,
    # with no use of the production container parser or a model framework.
    blob = (PERMANENT / 'texture-latent.safetensors').read_bytes()
    header_size = struct.unpack('<Q', blob[:8])[0]
    require(0 < header_size <= 65536 and header_size % 8 == 0, 'Manual header bound')
    header = json.loads(blob[8:8 + header_size])
    layouts = {'raw': ('F32', [3743, 32], '<f4'), 'denormalized': ('F32', [3743, 32], '<f4'),
               'coords': ('I32', [3743, 4], '<i4'), 'noise': ('F32', [3743, 32], '<f4')}
    require(set(header) == {'__metadata__', *layouts}, 'Manual tensor set')
    arrays, tensor_hashes, cursor = {}, {}, 0
    for name, (dtype, shape, numpy_dtype) in layouts.items():
        descriptor = header[name]
        size = shape[0] * shape[1] * 4
        require(descriptor == {'dtype': dtype, 'shape': shape, 'data_offsets': [cursor, cursor + size]},
                'Manual layout: ' + name)
        data = blob[8 + header_size + cursor:8 + header_size + cursor + size]
        require(len(data) == size, 'Manual payload truncation')
        arrays[name] = np.frombuffer(data, dtype=numpy_dtype).reshape(shape)
        require(np.isfinite(arrays[name]).all(), 'Manual nonfinite tensor: ' + name)
        tensor_hashes[name] = digest(data)
        require(tensor_hashes[name] == checked['metadata']['tensors'][name]['sha256'], 'Manual tensor digest')
        cursor += size
    require(8 + header_size + cursor == len(blob) == checked['metadata']['artifact_bytes']
            and digest(blob) == checked['metadata']['artifact_sha256'], 'Full-file SHA/length')
    original_coords = np.load(ROOT / 'docs/compatibility/amd-shape-slat-512-sampling-evidence/shape-latent-coordinates.npy', allow_pickle=False)
    require(arrays['coords'].dtype == original_coords.dtype and arrays['coords'].tobytes() == original_coords.tobytes(),
            'Original coordinate exact order')
    provenance = checked['metadata']['provenance']
    require(provenance == json.loads(header['__metadata__']['provenance'])
            and provenance['seed'] == 20261015 and provenance['harness_commit'] == committed['commit']
            and provenance['sampling'] == sampled and provenance['input'] == worker['input']
            and provenance['parameters'] == {'steps': 12, 'rescale_t': 3., 'guidance_strength': 1.,
                'guidance_rescale': 0., 'guidance_interval': [.6, .9], 'dino_lock': 0.}
            and provenance['sigma_min'] == .00001, 'Manual canonical provenance')
    require(tensor_hashes['raw'] == worker['output']['sha256']
            and tensor_hashes['noise'] == worker['input']['x']['sha256']
            and tensor_hashes['coords'] == worker['input']['coords']['sha256'], 'Original input/output hashes')
    stats = read(EVIDENCE / 'source-contract.json')['normalization']
    require(provenance['texture_normalization'] == stats, 'Texture normalization identity')
    denormalized = np.ascontiguousarray(arrays['raw'] * np.array(stats['std'], np.float32)[None]
                                      + np.array(stats['mean'], np.float32)[None])
    require(denormalized.tobytes() == arrays['denormalized'].tobytes(), 'Exact texture denormalization')
    post = read(EVIDENCE / 'post-exit-artifact-verification.json')
    require(post['classification'] == 'PASS' and post['worker_exit_verified'] is True
            and post['verified'] is True and post['artifact'] == checked, 'Host post-exit verification')
    preservation = read(EVIDENCE / 'preservation-after.json')
    require(preservation['status'] == 'PASS' and preservation['unexpected_deltas'] == [], 'Preservation postcheck')
    require('torch' not in sys.modules, 'Review must remain free of Torch')
    report = {'reviewer': '/root/texture_preservation_review', 'classification': 'PASS',
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'harness_commit': committed['commit'], 'attempts_observed': 1,
        'artifact': {'path': str(PERMANENT / 'texture-latent.safetensors'), 'bytes': len(blob),
                     'sha256': digest(blob), 'tensor_payload_sha256': tensor_hashes,
                     'raw_and_denormalized_finite': True, 'texture_denormalization_byte_exact': True,
                     'original_coordinate_order_byte_exact': True, 'noise_matches_recorded_initial_noise': True,
                     'seed': provenance['seed']},
        'independent_readers': ['Production bounded reader reopened after worker exit',
                               'Separate manual public-format header/payload decoder with NumPy'],
        'sequence': {'steps': 12, 'positive_forwards': 12, 'negative_forwards': 0,
                     'genuine_modules': len(modules), 'calls_per_module': 12, 'concat_calls': 12},
        'timing_seconds': {'strict_CPU_load': worker['cpu_load']['elapsed_seconds'],
                          'genuine_sampling': worker['forward_seconds'], 'whole_owned_worker': process['elapsed_seconds']},
        'peaks_bytes': {'allocated': worker['after_forward']['peak_allocated_bytes'],
                        'reserved': worker['after_forward']['peak_reserved_bytes'],
                        'host_working_set': release['cpu_memory']['peak_working_set_bytes'],
                        'host_private_commit': release['cpu_memory']['peak_commit_bytes']},
        'actual_TunableOp_all_false_twice': True, 'natural_job_and_worker_exit': True,
        'windows_health_and_periodic_observer_pass': True, 'owned_GPU_allocated_reserved_after_release': [0, 0],
        'original_artifacts_sources_backup_selected_Windows_metadata_preserved': True,
        'preservation_report': 'preservation-after.json', 'torch_imported_by_reviewer': False,
        'new_GPU_model_WSL_install_actions': 0,
        'limitations': 'This verifies genuine texture latents and their saved provenance, not visual fidelity, a texture decoder, six-channel material fields, UVs, baking, emissive glow or a textured GLB. No additional inference or retry was performed.'}
    with (EVIDENCE / 'independent-outcome-review.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({'classification': report['classification'], 'artifact_sha256': digest(blob),
                      'coordinate_order_exact': True, 'denormalization_exact': True,
                      'torch_imported': 'torch' in sys.modules}))


if __name__ == '__main__':
    run()
