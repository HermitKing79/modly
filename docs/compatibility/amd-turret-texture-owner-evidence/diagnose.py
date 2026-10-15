"""Read-only UID/DrvFS diagnosis in the actual contained WSL identity."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

E = Path(__file__).resolve().parent
ROOT = E.parents[2]
sys.path.insert(0, str(ROOT/'scripts/compatibility'))
import amd_wsl_rocdxg_smoke as base
FIELD = Path('/mnt/c/Users/wattz/Documents/ModlyArtifacts/amd-turret-texture-bootstrap-473b7bd/field')


def stat_row(p):
    s = p.lstat()
    return {'path': str(p), 'resolved': str(p.resolve(strict=True)), 'uid': s.st_uid,
        'gid': s.st_gid, 'mode': oct(s.st_mode), 'device': s.st_dev,
        'device_major': os.major(s.st_dev), 'device_minor': os.minor(s.st_dev),
        'inode': s.st_ino, 'links': s.st_nlink, 'bytes': s.st_size,
        'is_symlink': p.is_symlink(), 'mtime_ns': s.st_mtime_ns}


def guest(output):
    output.mkdir(exist_ok=False)
    result = {'classification': 'CPU_UNKNOWN', 'forbidden_imports': []}
    try:
        base.await_go(sys.stdin.buffer)
        def deny(event, args):
            if event == 'import' and str(args[0]).split('.')[0] in {'torch','triton','flex_gemm','trellis2','numpy','o_voxel','cumesh'}:
                result['forbidden_imports'].append(str(args[0]))
                raise RuntimeError('Read-only identity probe forbids AI/native imports')
        sys.addaudithook(deny)
        from amd_wsl_native_import_policy import ImportPolicy
        policy = ImportPolicy(output)
        policy.install_audit_hook()
        result['identity'] = {'uid': os.getuid(), 'euid': os.geteuid(), 'gid': os.getgid(),
            'egid': os.getegid(), 'groups': os.getgroups(), 'python': sys.version,
            'executable': sys.executable, 'prefix': sys.prefix,
            'executable_resolved': str(Path(sys.executable).resolve()),
            'executable_sha256': hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()}
        result['field_and_ancestors'] = [stat_row(p) for p in (FIELD, *FIELD.parents)]
        result['trusted_scratch'] = [stat_row(Path(p)) for p in (
            '/home/modly/turret-texture-bootstrap-473b7bd-cpu',
            '/home/modly/native-decoder-preparation-6dd94a85')]
        result['mountinfo'] = Path('/proc/self/mountinfo').read_text()
        result['mounts'] = Path('/proc/mounts').read_text()
        result['wsl_configuration'] = {p: Path(p).read_text() if Path(p).is_file() else None
            for p in ('/etc/wsl.conf', '/etc/fstab')}
        probe = FIELD.parent/'probe/texture-field.safetensors'
        result['existing_probe'] = dict(stat_row(probe), sha256=hashlib.sha256(probe.read_bytes()).hexdigest())
        result['field_empty'] = not any(FIELD.iterdir())
        result['policy'] = policy.snapshot()
        if result['forbidden_imports'] or result['policy']['rejected_calls']:
            raise RuntimeError('Probe crossed read-only boundary')
        result['classification'] = 'CPU_DIAG_PASS'
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        base.save_json(output/'results.json', result)
    return 0 if result['classification'] == 'CPU_DIAG_PASS' else 1


def host(output):
    if output.resolve() != (E/'diagnosis-run').resolve():
        raise RuntimeError('Exclusive diagnosis path required')
    helpers = {n: base.load_sibling(n) for n in ('amd_bounded_gpu_proof','amd_forward_owned_job')}
    with patch.object(base,'SCRIPT',Path(__file__).resolve()), \
            patch.object(base,'ATTEMPT_MARKER',E/'diagnosis-attempt.marker'), \
            patch.object(base,'load_sibling',side_effect=lambda n:helpers[n]):
        raw_exit = base.supervise(output)
    raw = json.loads((output/'results.json').read_text())
    passed = (raw_exit == 1 and raw.get('worker_returncode') == 0
        and raw.get('worker',{}).get('classification') == 'CPU_DIAG_PASS'
        and raw.get('observer_healthy') is True
        and raw.get('windows_job_cleanup',{}).get('owned_tree_exit_verified') is True
        and raw['windows_job_cleanup'].get('forced_job_cleanup') is False
        and raw.get('distro_termination',{}).get('returncode') == 0
        and not any('error' in k for k in raw))
    base.save_json(output/'diagnosis-verdict.json', {'status':'PASS' if passed else 'UNKNOWN','GPU_work':False})
    print('READ_ONLY_DIAGNOSIS='+('PASS' if passed else 'UNKNOWN'))
    return 0 if passed else 1


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    raise SystemExit(guest(a.worker) if a.worker is not None else host(a.output))
