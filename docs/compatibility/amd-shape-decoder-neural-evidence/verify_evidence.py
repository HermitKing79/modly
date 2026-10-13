"""Passive readback of audit, final candidate, checkpoint and historical cache hashes."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
EV = Path(__file__).parent
sys.path[:0] = [str(ROOT/'scripts/compatibility'), str(ROOT/'api')]
import amd_shape_decoder_preflight as audit


def deny_runtime(event, args):
    if event == 'import' and str(args[0]).split('.')[0] in ('torch','triton','trellis2','flex_gemm','cumesh','o_voxel'):
        raise RuntimeError('Final evidence verification is passive')


sys.addaudithook(deny_runtime)
before = json.loads((EV/'preflight.json').read_text(encoding='utf-8'))
probe = json.loads((EV/'cpu-probe-4/results.json').read_text(encoding='utf-8'))
assert audit.metadata_passed(probe)
assert probe['worker']['harness_sha256'] == audit.digest(ROOT/'scripts/compatibility/amd_shape_decoder_preflight.py')
assert probe['worker']['precision'] == before['cpu_probe']['precision']
for field in ('source_sha256', 'native_source_and_binary_sha256'):
    for path, expected in before[field].items(): assert audit.digest(path) == expected, path
cache_files = 0
for row in before['cache_metadata']:
    meta = Path(row['path']); group = meta.parent/('__grp__'+meta.name)
    assert audit.digest(group) == row['group_sha256']
    paths = audit.cache_group_paths(meta, json.loads(group.read_text(encoding='utf-8'))['child_paths'])
    for name, path in paths.items():
        assert audit.digest(path) == row['files'][name]['sha256'], str(path)
        cache_files += 1
for row in before['inputs'].values(): assert audit.digest(row['path']) == row['sha256']
assert audit.digest(audit.PREFIX.with_suffix('.safetensors')) == before['checkpoint']['checkpoint_sha256']
assert audit.digest(audit.PREFIX.with_suffix('.json')) == before['checkpoint']['config_sha256']
assert audit.digest(before['tuning']['path']) == before['tuning']['sha256']
assert not (EV/'attempt-started.json').exists() and not (EV/'run').exists()
assert not any(name in sys.modules for name in ('torch','triton','flex_gemm','cumesh','o_voxel'))
row = dict(classification='UNKNOWN', gpu_attempted=False, real_weights_loaded=False, neural_forward_calls=0,
           candidate_sha256=probe['worker']['harness_sha256'], final_cpu_probe_seconds=probe['elapsed_seconds'],
           cpu_manifest_verified=True, checkpoint_hash_unchanged=True, config_hash_unchanged=True,
           amd_source_files_verified=len(before['source_sha256']), native_files_verified=len(before['native_source_and_binary_sha256']),
           historical_cache_products_verified=cache_files, tuning_cache_unchanged=True, original_inputs_unchanged=True,
           cache_current_source_identity_verified=False, native_execution_policy_verified=False,
           no_attempt_marker_or_run=True, passive_no_runtime_imports=True)
with (EV/'final-evidence-verification.json').open('x', encoding='utf-8') as stream: json.dump(row, stream, indent=2)
print(json.dumps(row))
