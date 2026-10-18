"""Owned CPU acceptance of this production worker's actual prepare_inputs route.

The original supervisor retains Windows Job ownership, binary GO, live Windows
health observations, five-second Job grace, and named-distro termination. Its
GPU result contract remains unused; this adapter issues a separate CPU verdict.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCRIPTS = ROOT / 'scripts/compatibility'
WORKER = HERE / 'worker.py'
PRIVATE_ATTESTATION = ROOT.parent / 'artifacts/amd-wsl-input-ownership-48ad081/windows-input-attestation.json'
MODE = 'cpu-input-acceptance'
OUTPUT = HERE / MODE
CPU_MARKER = HERE / 'CPU-input-acceptance-attempt-started.json'
MANIFEST = HERE / 'candidate-identities.json'
HISTORICAL_MANIFEST = ROOT / 'docs/compatibility/amd-brick-pbr-gpu-attempt-evidence/candidate-identities.json'
HISTORICAL_MANIFEST_SHA256 = '724ee58f93a001db52f4d35d8c7919593dd90ac2b1d36bdb73a41f544969b80e'
INPUT_EVIDENCE = ROOT / 'docs/compatibility/amd-wsl-input-ownership-evidence'
INPUT_MANIFEST = INPUT_EVIDENCE / 'candidate-identities.json'
INPUT_MANIFEST_SHA256 = 'c3b2b10f5741a5f71957b02f5c0401ab95007336cc6f58f4d6128172fbcff839'
INPUT_POLICY = INPUT_EVIDENCE / 'readonly_input.py'
INPUT_POLICY_SHA256 = 'fe41b6410c8897e4062366457de06c795261c6d19aea059d9f7f9851e5d2d98e'
WINDOWS_ATTEST = INPUT_EVIDENCE / 'windows_attest.py'
WINDOWS_ATTEST_SHA256 = '30ca2eea0aab7365961f602e7f4860a6cfed2ac40d95802e0064aaa101a9a712'
PRIVATE_ATTESTATION_SHA256 = '0a0bd0cffb12709341276fad3ab8f08fa9b2ed57ef02549b1d40db9c9f37b8cd'
PROFILE_SHA256 = 'cbad376b38c69943defe59e0273817dad9f8cb044ea9b600768eb62f7e51e5c0'
RUN_CONTRACT_SHA256 = '82bf47920f667a3a445d5e70297a0057fae78a073d7ae9c903f0827319a8dfdd'
# Isolate reads from all pre-existing bytecode caches. -B forbids cache writes;
# the selected location is deliberately never created by this harness.
sys.pycache_prefix = str(HERE / '_unused_host_bytecode')
sys.dont_write_bytecode = True
DISTRO = 'Modly-AMD-Ubuntu-24.04'
HOST_SECONDS = 120.0
GUEST_SECONDS = 90.0
PINNED = {
    'amd_wsl_rocdxg_smoke.py': 'c10dc0afc08be765bfcfb1ddba6acd1aba22ebdb214da0c82bbdb9e12ae58cc2',
    'amd_bounded_gpu_proof.py': '8910884b4d4bd433df654a8137586eb43711e37e97acfcf8462392ef4258dff4',
    'amd_forward_owned_job.py': '19632bab033e8caaafb2bbfcddf002024ff1a57105224a30c4fd6a2242cccb64',
    'amd_texture_sampling_artifact.py': '5eaf35c08cf2e96f46dfe8bb32fa835c079ef34ce5a322a0afd411ee7cf9474c',
}
TUNABLE_ENV = {
    'PYTORCH_TUNABLEOP_ENABLED': '0',
    'PYTORCH_TUNABLEOP_TUNING': '0',
    'PYTORCH_TUNABLEOP_RECORD_UNTUNED': '0',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != PINNED[path.name]:
        raise RuntimeError('Pinned helper changed before source compilation')
    # Compile precisely the checked source bytes, never existing bytecode.
    sys.modules[name] = module
    exec(compile(blob, str(path), 'exec'), module.__dict__)
    return module


def guest_path(path):
    path = Path(path).resolve(strict=False)
    if path != PRIVATE_ATTESTATION.resolve(strict=False):
        path.relative_to(ROOT)
    if path.drive.lower() != 'c:':
        raise ValueError('Only this C: proof checkout is admitted')
    return '/mnt/c/' + path.as_posix()[3:]


def checked_source(path, expected=None, maximum=2*1024**2):
    path = Path(path)
    if not path.is_absolute() or path != path.resolve(strict=True):
        raise ValueError('Absolute canonical source required')
    for component in (path, *path.parents):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
            raise ValueError('Aliased source path prohibited')
    before = path.lstat()
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_nlink)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= maximum:
        raise ValueError('Bounded plain source required')
    with path.open('rb') as stream:
        if identity(__import__('os').fstat(stream.fileno())) != identity(before):
            raise ValueError('Source identity changed before read')
        blob = stream.read(maximum+1)
        if identity(__import__('os').fstat(stream.fileno())) != identity(before):
            raise ValueError('Source identity changed during read')
    if len(blob) != before.st_size or identity(path.lstat()) != identity(before):
        raise ValueError('Source identity changed after read')
    if expected is not None and hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError('Pinned source identity differs')
    return blob


def revalidate_qualified_inputs():
    if __import__('os').name != 'nt':
        raise RuntimeError('Qualified input attestation must be revalidated on Windows')
    checked_source(INPUT_MANIFEST, INPUT_MANIFEST_SHA256)
    checked_source(INPUT_POLICY, INPUT_POLICY_SHA256)
    blob = checked_source(WINDOWS_ATTEST, WINDOWS_ATTEST_SHA256)
    spec = importlib.util.spec_from_file_location('_production_cpu_windows_attest', WINDOWS_ATTEST)
    module = importlib.util.module_from_spec(spec)
    exec(compile(blob, str(WINDOWS_ATTEST), 'exec', dont_inherit=True), module.__dict__)
    result = module.verify(PRIVATE_ATTESTATION, PRIVATE_ATTESTATION_SHA256)
    if result.get('passed') is not True or result.get('files') != 5:
        raise ValueError('Fresh qualified Windows input check failed')
    return result


def build_worker_command(identities):
    command = ['wsl.exe', '-d', DISTRO, '-u', 'modly', '--exec', '/usr/bin/env',
               *(key + '=' + value for key, value in TUNABLE_ENV.items()),
               '/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', '90s',
               '/opt/modly-wsl/venv/bin/python', '-I', '-B', guest_path(WORKER),
               '--cpu-input-acceptance', '--worker', guest_path(OUTPUT / 'guest'),
               '--expected-worker-sha256', identities['worker.py'],
               '--expected-host-sha256', identities['cpu_acceptance.py'],
               '--expected-manifest-sha256', identities['manifest']]
    return command


def preflight():
    if OUTPUT.exists() or CPU_MARKER.exists():
        raise RuntimeError('CPU mode already reserved; existing evidence is never overwritten')
    if not sys.dont_write_bytecode or Path(sys.pycache_prefix).exists():
        raise RuntimeError('CPU imports must bypass preexisting bytecode')
    manifest_blob = checked_source(MANIFEST)
    manifest = json.loads(manifest_blob)
    for path, sha in ((HISTORICAL_MANIFEST, HISTORICAL_MANIFEST_SHA256),
                      (INPUT_MANIFEST, INPUT_MANIFEST_SHA256)):
        preserved = json.loads(checked_source(path, sha))
        if type(manifest) is not dict or any(manifest.get(name) != pin for name, pin in preserved.items()):
            raise ValueError('Preserved source closure incomplete')
        if manifest.get(path.relative_to(ROOT).as_posix()) != sha:
            raise ValueError('Preserved manifest absent from source closure')
    for filename in ('host.py', 'worker.py', 'cpu_acceptance.py', 'test_host.py', 'test_worker.py'):
        if (HERE / filename).relative_to(ROOT).as_posix() not in manifest:
            raise ValueError('Production execution source closure incomplete')
    for name, sha in manifest.items():
        if (type(name) is not str or Path(name).is_absolute() or '..' in Path(name).parts
                or '\\' in name or ':' in name):
            raise ValueError('Canonical repository-relative source required')
        # The preserved closure includes the exact 3,231,402-byte native wheel.
        checked_source(ROOT / name, sha, maximum=4*1024**2)
    identities = {'cpu_acceptance.py': digest(HERE / 'cpu_acceptance.py'), 'worker.py': digest(WORKER),
                  'manifest': hashlib.sha256(manifest_blob).hexdigest(),
                  'qualified_manifest': INPUT_MANIFEST_SHA256,
                  'attestation': PRIVATE_ATTESTATION_SHA256, 'readonly_input.py': INPUT_POLICY_SHA256}
    rows = {}
    for filename, expected in PINNED.items():
        path = (SCRIPTS / filename).resolve(strict=True)
        if path.parent != SCRIPTS.resolve(strict=True) or digest(path) != expected:
            raise RuntimeError('Pinned original helper/reader changed: ' + filename)
        rows[filename] = {'path': str(path), 'sha256': expected}
    attestation = revalidate_qualified_inputs()
    command = build_worker_command(identities)
    record = {'schema': 'modly.amd.wsl-input-cpu-preflight.v1', 'classification': 'PASS',
              'mode': MODE, 'distro': DISTRO, 'identity': identities, 'pinned_originals': rows,
              'worker_command': command, 'output': str(OUTPUT),
              'CPU_marker': str(CPU_MARKER), 'input_attestation': attestation,
              'input_route': 'actual production worker prepare_inputs',
              'GPU_authorized': False, 'model_authorized': False,
              'guest_seconds': GUEST_SECONDS, 'host_seconds': HOST_SECONDS,
              'original_supervisor_SCRIPT': str(SCRIPTS / 'amd_wsl_rocdxg_smoke.py')}
    return record


def cpu_verdict(raw_exit, raw, expected_manifest_sha256):
    worker = raw.get('worker', {})
    expected = 'CPU_PREPARED'
    contained = (raw_exit == 1 and raw.get('classification') == 'UNKNOWN'
                 and raw.get('worker_returncode') == 0
                 and raw.get('observer_healthy') is True
                 and raw.get('windows_job_cleanup', {}).get('owned_tree_exit_verified') is True
                 and raw.get('windows_job_cleanup', {}).get('forced_job_cleanup') is False
                 and raw.get('windows_job_grace', {}).get('empty') is True
                 and raw.get('distro_termination', {}).get('returncode') == 0
                 and raw.get('elapsed_seconds', HOST_SECONDS) < HOST_SECONDS
                 and not any('error' in key for key in raw))
    security = worker.get('acceptance_security_results', {})
    security_tests = security.get('tests')
    passed_security = (security.get('all_passed') is True and isinstance(security_tests, list)
                       and bool(security_tests) and all(row.get('passed') is True for row in security_tests))
    route = worker.get('input_route', {})
    runtime = worker.get('runtime', {})
    route_expected = dict(production_prepare_function='prepare_inputs', generic_array_revalidation=True,
        policy_path=guest_path(INPUT_POLICY), policy_sha256=INPUT_POLICY_SHA256,
        manifest_path=guest_path(INPUT_MANIFEST), manifest_sha256=INPUT_MANIFEST_SHA256,
        attestation_path=guest_path(PRIVATE_ATTESTATION), attestation_sha256=PRIVATE_ATTESTATION_SHA256,
        historical_readers_modified=False, writer_capability=False,
        supplied_profile_verified=True, plan_closure_verified=True)
    metadata = worker.get('prepared_metadata', {})
    passed = (contained and worker.get('classification') == expected
              and worker.get('gate_authorized') is True
              and worker.get('forbidden_import_attempts') == []
              and worker.get('forbidden_modules') == []
              and worker.get('CPU_only') is True
              and all(route.get(key) == value for key, value in route_expected.items())
              and all(runtime.get(key) == 1000 for key in ('uid', 'euid', 'gid', 'egid'))
              and type(expected_manifest_sha256) is str and len(expected_manifest_sha256) == 64
              and worker.get('source_manifest_sha256') == expected_manifest_sha256
              and type(worker.get('candidate_identities')) is dict and bool(worker['candidate_identities'])
              and worker.get('authenticated_profile_sha256') == PROFILE_SHA256
              and metadata.get('profile_sha256') == PROFILE_SHA256
              and metadata.get('run_contract_sha256') == RUN_CONTRACT_SHA256 and passed_security
              and worker.get('prepared_arrays_released') is True
              and type(worker.get('prepared_array_objects_observed')) is int
              and worker['prepared_array_objects_observed'] > 0
              and not any('error' in key for key in worker))
    return {'schema': 'modly.amd.wsl-input-cpu-verdict.v1',
            'classification': 'PASS' if passed else 'UNKNOWN', 'mode': MODE,
            'CPU_only': True, 'GPU_authorized': False, 'model_authorized': False,
            'owned_containment_verified': contained,
            'expected_source_manifest_sha256': expected_manifest_sha256,
            'authenticated_profile_sha256': worker.get('authenticated_profile_sha256'),
            'worker_classification': worker.get('classification'),
            'production_supervisor_exit': raw_exit,
            'production_supervisor_classification': raw.get('classification'),
            'production_GPU_contract_unused': True}


def host():
    record = preflight()
    # Resolve and verify all imports before any source-dependent operation. SCRIPT
    # stays at the real supervisor so sibling helper lookup can never relocate.
    originals = {name[:-3]: load('input_cpu_' + name[:-3], SCRIPTS / name)
                 for name in ('amd_wsl_rocdxg_smoke.py', 'amd_bounded_gpu_proof.py',
                              'amd_forward_owned_job.py')}
    base = originals['amd_wsl_rocdxg_smoke']
    if base.SCRIPT != (SCRIPTS / 'amd_wsl_rocdxg_smoke.py').resolve(strict=True):
        raise RuntimeError('Original supervisor source anchor differs')

    def helper(name):
        if name not in ('amd_bounded_gpu_proof', 'amd_forward_owned_job'):
            raise RuntimeError('Unexpected supervisor helper request')
        return originals[name]

    def command(original_script, output):
        if (original_script != base.guest_path(base.SCRIPT)
                or output != guest_path(OUTPUT / 'guest')):
            raise RuntimeError('Supervisor requested a foreign worker/output')
        return record['worker_command']

    def deadlines(now, began, go):
        if now - began >= HOST_SECONDS:
            raise TimeoutError('120-second CPU host deadline')
        if go is not None and now - go >= GUEST_SECONDS:
            raise TimeoutError('90-second CPU authorized worker deadline')

    def reserve(output, marker):
        if output != OUTPUT.resolve() or marker != CPU_MARKER:
            raise RuntimeError('CPU evidence reservation escaped its mode')
        revalidate_qualified_inputs()
        base.save_json(marker, dict(record, started_utc=base.utc_now(), scope='CPU input only'))
        output.mkdir(exist_ok=False)
        base.save_json(output / 'helper-preflight.json', record)

    with patch.object(base, 'load_sibling', side_effect=helper), \
            patch.object(base, 'worker_command', side_effect=command), \
            patch.object(base, 'check_deadlines', side_effect=deadlines), \
            patch.object(base, 'reserve_attempt', side_effect=reserve), \
            patch.object(base, 'ATTEMPT_MARKER', CPU_MARKER):
        raw_exit = base.supervise(OUTPUT)
    raw = json.loads((OUTPUT / 'results.json').read_text(encoding='utf-8'))
    verdict = cpu_verdict(raw_exit, raw, record['identity']['manifest'])
    base.save_json(OUTPUT / 'CPU-verdict.json', verdict)
    print(json.dumps(verdict), flush=True)
    return 0 if verdict['classification'] == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(), indent=2))
        return 0
    return host()


if __name__ == '__main__':
    raise SystemExit(main())
