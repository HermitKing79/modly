"""Owned, bounded WSL CPU input diagnosis and read-only acceptance adapter.

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
MODES = ('diagnose', 'diagnose-confirmation', 'acceptance', 'independent')
DIAGNOSIS_MODES = ('diagnose', 'diagnose-confirmation')
# Isolate reads from all pre-existing bytecode caches. -B forbids cache writes;
# the selected location is deliberately never created by this harness.
sys.pycache_prefix = str(HERE / '_unused_host_bytecode')
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


def checked_local_input(path, name):
    path = Path(path)
    if not path.is_absolute() or path != path.resolve(strict=True):
        raise ValueError(name + ' must be an absolute canonical path')
    if name == 'attestation':
        if path != PRIVATE_ATTESTATION.resolve(strict=True):
            raise ValueError('Only the exact private Windows attestation is admitted')
    else:
        path.relative_to(HERE)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or getattr(info, 'st_file_attributes', 0) & 1024 or path.is_symlink()):
        raise ValueError(name + ' must be a plain evidence file')
    return path


def build_worker_command(mode, identities, manifest=None, attestation=None):
    if mode not in MODES:
        raise ValueError('Unapproved CPU mode')
    command = ['wsl.exe', '-d', DISTRO, '-u', 'modly', '--exec', '/usr/bin/env',
               *(key + '=' + value for key, value in TUNABLE_ENV.items()),
               '/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', '90s',
               '/opt/modly-wsl/venv/bin/python', '-I', '-B', guest_path(WORKER),
               '--worker', guest_path(HERE / mode / 'guest'), '--mode', mode,
               '--expected-worker-sha256', identities['worker.py'],
               '--expected-host-sha256', identities['host.py']]
    if mode not in DIAGNOSIS_MODES:
        if manifest is None or attestation is None:
            raise ValueError('Full CPU acceptance requires manifest and Windows attestation')
        command.extend(['--manifest', guest_path(manifest),
                        '--attestation', guest_path(attestation),
                        '--expected-manifest-sha256', identities['manifest'],
                        '--expected-attestation-sha256', identities['attestation'],
                        '--expected-policy-sha256', identities['readonly_input.py']])
    return command


def preflight(mode, manifest=None, attestation=None):
    if mode not in MODES:
        raise ValueError('Unapproved CPU mode')
    if (HERE / mode).exists() or (HERE / ('CPU-' + mode + '-attempt-started.json')).exists():
        raise RuntimeError('CPU mode already reserved; existing evidence is never overwritten')
    identities = {'host.py': digest(HERE / 'host.py'), 'worker.py': digest(WORKER)}
    rows = {}
    for filename, expected in PINNED.items():
        path = (SCRIPTS / filename).resolve(strict=True)
        if path.parent != SCRIPTS.resolve(strict=True) or digest(path) != expected:
            raise RuntimeError('Pinned original helper/reader changed: ' + filename)
        rows[filename] = {'path': str(path), 'sha256': expected}
    if mode not in DIAGNOSIS_MODES:
        manifest = checked_local_input(manifest, 'manifest')
        attestation = checked_local_input(attestation, 'attestation')
        identities.update({'manifest': digest(manifest), 'attestation': digest(attestation),
                           'readonly_input.py': digest(HERE / 'readonly_input.py')})
    elif manifest is not None or attestation is not None:
        raise ValueError('The stdlib diagnosis does not accept preparation inputs')
    command = build_worker_command(mode, identities, manifest, attestation)
    record = {'schema': 'modly.amd.wsl-input-cpu-preflight.v1', 'classification': 'PASS',
              'mode': mode, 'distro': DISTRO, 'identity': identities, 'pinned_originals': rows,
              'worker_command': command, 'output': str(HERE / mode),
              'CPU_marker': str(HERE / ('CPU-' + mode + '-attempt-started.json')),
              'GPU_authorized': False, 'model_authorized': False,
              'guest_seconds': GUEST_SECONDS, 'host_seconds': HOST_SECONDS,
              'original_supervisor_SCRIPT': str(SCRIPTS / 'amd_wsl_rocdxg_smoke.py')}
    return record


def cpu_verdict(mode, raw_exit, raw):
    worker = raw.get('worker', {})
    expected = 'CPU_DIAGNOSED' if mode in DIAGNOSIS_MODES else 'CPU_PREPARED'
    contained = (raw_exit == 1 and raw.get('classification') == 'UNKNOWN'
                 and raw.get('worker_returncode') == 0
                 and raw.get('observer_healthy') is True
                 and raw.get('windows_job_cleanup', {}).get('owned_tree_exit_verified') is True
                 and raw.get('windows_job_cleanup', {}).get('forced_job_cleanup') is False
                 and raw.get('windows_job_grace', {}).get('empty') is True
                 and raw.get('distro_termination', {}).get('returncode') == 0
                 and raw.get('elapsed_seconds', HOST_SECONDS) < HOST_SECONDS
                 and not any('error' in key for key in raw))
    passed = (contained and worker.get('classification') == expected
              and worker.get('gate_authorized') is True
              and worker.get('forbidden_import_attempts') == []
              and worker.get('forbidden_modules') == []
              and worker.get('CPU_only') is True)
    return {'schema': 'modly.amd.wsl-input-cpu-verdict.v1',
            'classification': 'PASS' if passed else 'UNKNOWN', 'mode': mode,
            'CPU_only': True, 'GPU_authorized': False, 'model_authorized': False,
            'owned_containment_verified': contained,
            'worker_classification': worker.get('classification'),
            'production_supervisor_exit': raw_exit,
            'production_supervisor_classification': raw.get('classification'),
            'production_GPU_contract_unused': True}


def host(mode, manifest=None, attestation=None):
    record = preflight(mode, manifest, attestation)
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
                or output != guest_path(HERE / mode / 'guest')):
            raise RuntimeError('Supervisor requested a foreign worker/output')
        return record['worker_command']

    def deadlines(now, began, go):
        if now - began >= HOST_SECONDS:
            raise TimeoutError('120-second CPU host deadline')
        if go is not None and now - go >= GUEST_SECONDS:
            raise TimeoutError('90-second CPU authorized worker deadline')

    def reserve(output, marker):
        if output != (HERE / mode).resolve() or marker != HERE / ('CPU-' + mode + '-attempt-started.json'):
            raise RuntimeError('CPU evidence reservation escaped its mode')
        base.save_json(marker, dict(record, started_utc=base.utc_now(), scope='CPU input only'))
        output.mkdir(exist_ok=False)
        base.save_json(output / 'helper-preflight.json', record)

    with patch.object(base, 'load_sibling', side_effect=helper), \
            patch.object(base, 'worker_command', side_effect=command), \
            patch.object(base, 'check_deadlines', side_effect=deadlines), \
            patch.object(base, 'reserve_attempt', side_effect=reserve), \
            patch.object(base, 'ATTEMPT_MARKER', HERE / ('CPU-' + mode + '-attempt-started.json')):
        raw_exit = base.supervise(HERE / mode)
    raw = json.loads((HERE / mode / 'results.json').read_text(encoding='utf-8'))
    verdict = cpu_verdict(mode, raw_exit, raw)
    base.save_json(HERE / mode / 'CPU-verdict.json', verdict)
    print(json.dumps(verdict), flush=True)
    return 0 if verdict['classification'] == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--attestation', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(args.mode, args.manifest, args.attestation), indent=2))
        return 0
    return host(args.mode, args.manifest, args.attestation)


if __name__ == '__main__':
    raise SystemExit(main())
