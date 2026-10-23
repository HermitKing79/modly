"""User launcher for the isolated image-only preview; no installation/fallback."""
import argparse
from collections import deque
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid

SCHEMA = 'modly.amd.image-test-preview.v1'
BASE = '63332ca5d415de8eaaff15fdaf193bc7d9935a72'
GiB = 1024**3
CAP = 6 * GiB
REQUIRED_FILES = {
    'launch_preview.py', 'runtime_probe.py', 'owned_job.py',
    'runtime/Modly.exe', 'runtime/resources/app/package.json',
    'runtime/resources/app/out/main/preview-bootstrap.cjs',
    'runtime/resources/app/out/main/index.js', 'runtime/resources/app/out/preload/index.js',
    'runtime/resources/app/out/renderer/app.js', 'api/preview_api.py',
    'api/routers/amd_image_test.py', 'api/services/amd_image_test.py',
    'api/routers/__init__.py', 'api/services/__init__.py',
    'api/services/amd_runtime.py', 'api/services/amd_stage_contract.py', 'api/services/amd_completed_result.py',
    'profile/settings.json',
}


def runtime_tree_sha256(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob('*'), key=lambda p: p.relative_to(root).as_posix()):
        plain(path)
        if path.is_file():
            need(path.stat().st_nlink == 1, 'Single-link runtime file required')
            digest.update(path.relative_to(root).as_posix().encode() + b'\0' + sha(path).encode() + b'\n')
    return digest.hexdigest()


def ensure_directory(path):
    # Inspect existing ancestors before mkdir can follow an unexpected junction.
    parent = Path(path)
    while not parent.exists():
        parent = parent.parent
    plain(parent)
    Path(path).mkdir(parents=True, exist_ok=True)
    return plain(path)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def plain(path):
    path = Path(path).absolute()
    for node in (path, *path.parents):
        need(not node.is_symlink() and not getattr(node.lstat(), 'st_file_attributes', 0) & 0x400,
             'Plain non-reparse path required: ' + str(path))
    return path


def validate_config(filename):
    filename = plain(filename)
    need(filename.is_file() and filename.stat().st_size <= 256*1024, 'Bounded preview configuration required')
    c = json.loads(filename.read_bytes())
    root = filename.parent
    need(c['schema'] == SCHEMA and c['source_commit'] == BASE, 'Qualified preview schema/source required')
    need(Path(c['root']) == root, 'Confined preview root required')
    for key, relative in [('executable', 'runtime/Modly.exe'), ('profile', 'profile'), ('api_dir', 'api')]:
        need(Path(c[key]) == root/relative, 'Confined ' + key + ' required')
        plain(c[key])
    need(type(c['files']) is dict and REQUIRED_FILES <= c['files'].keys() and len(c['files']) <= 256,
         'Complete bounded immutable preview closure required')
    for name, expected in c['files'].items():
        relative = PurePosixPath(name)
        need(type(name) is str and not relative.is_absolute() and '\\' not in name and ':' not in name
             and all(part not in ('', '.', '..') for part in name.split('/')), 'Safe relative file required')
        path = plain(root/name)
        need(path.is_file() and path.stat().st_nlink == 1 and sha(path) == expected,
             'Changed preview file: ' + name)
    for kind in ('api', 'cpu'):
        path = plain(c[kind+'_python'])
        need(path.is_file() and path.name.lower().endswith('python.exe')
             and sha(path) == c[kind+'_probe']['python_sha256'], 'Changed configured ' + kind + ' runtime')
    need(runtime_tree_sha256(root/'runtime') == c['runtime_tree_sha256'], 'Changed copied runtime tree')
    return c


def environment(profile, nonce, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONUSERBASE', 'VIRTUAL_ENV', 'CONDA_PREFIX',
                 'PIP_TARGET', 'PIP_PREFIX', 'ELECTRON_RUN_AS_NODE', 'NODE_OPTIONS',
                 'CHROME_LOG_FILE', 'ELECTRON_LOG_FILE', 'ELECTRON_ENABLE_LOGGING', 'ELECTRON_ENABLE_STACK_DUMPING'):
        env.pop(name, None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='', SELECTED_MODEL_ID='',
               MODLY_AMD_PREVIEW_NONCE=nonce)
    for key, folder in [('TEMP','temp'), ('TMP','temp'), ('HF_HOME','cache/huggingface'),
                        ('XDG_CACHE_HOME','cache'), ('TORCH_HOME','cache/torch'), ('PIP_CACHE_DIR','cache/pip')]:
        env[key] = str(profile/folder)
    return env


def verify_runtime(executable, script, kind, expected, env):
    r = subprocess.run([executable, '-B', '-I', str(script), kind], stdin=subprocess.DEVNULL,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, env=env,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    need(r.returncode == 0 and len(r.stdout) <= 16384, 'Configured ' + kind + ' CPU runtime probe failed')
    need(json.loads(r.stdout) == expected, 'Runtime probe changed: ' + kind)


def check_port(port=8765):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            server.bind(('127.0.0.1', port))
    except OSError as error:
        raise ValueError('API port is already in use; close its application yourself. Nothing was terminated.') from error


def resources(root):
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total', 'page_available',
                                                 'virtual_total', 'virtual_available', 'extended')]
    memory = Memory(); memory.length = ctypes.sizeof(memory)
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    if not api.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(available_ram_bytes=memory.available, disk_free_bytes=shutil.disk_usage(root).free)


def admit(measured):
    need(measured['available_ram_bytes'] >= 12*GiB and measured['disk_free_bytes'] >= 2*GiB,
         'Preview requires 12 GiB available RAM and 2 GiB free destination disk: ' + str(measured))


def save(path, value):
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def run(filename, test_timeout=None):
    c = validate_config(filename)
    root, profile = Path(c['root']), Path(c['profile'])
    check_port()
    initial = resources(root); admit(initial)
    nonce = uuid.uuid4().hex
    env = environment(profile, nonce)
    for folder in ('temp', 'cache/huggingface', 'cache/torch', 'cache/pip', 'logs', 'launches'):
        ensure_directory(profile/folder)
    env['MODLY_AMD_IMAGE_TEST_PYTHON'] = c['cpu_python']
    for kind in ('api', 'cpu'):
        verify_runtime(c[kind+'_python'], root/'runtime_probe.py', kind, c[kind+'_probe'], env)
    # Check again after runtime probes, immediately before this preview launch.
    check_port(); admit(resources(root))
    session = profile/'launches'/(time.strftime('%Y%m%d-%H%M%S') + '-' + nonce[:12])
    session.mkdir(exist_ok=False)
    env['MODLY_AMD_PREVIEW_ATTACH_FILE'] = str(session/'attached.json')
    history = deque(maxlen=3600)
    record = dict(status='BLOCKED', source_commit=BASE, initial_resources=initial, app_launches=0,
                  resource_sample_count=0, minimum_available_ram_bytes=initial['available_ram_bytes'],
                  minimum_free_disk_bytes=initial['disk_free_bytes'],
                  generation_enabled=False, process_and_Job_cap_bytes=CAP,
                  user_session_has_deadline=test_timeout is not None)
    job = child = None
    attached = False
    began = time.monotonic()
    print('Opening Experimental AMD Image Test Preview. Close the Modly preview window to finish.\nSession logs: '+str(session), flush=True)
    try:
        spec = importlib.util.spec_from_file_location('preview_owned_job', root/'owned_job.py')
        owned = importlib.util.module_from_spec(spec); spec.loader.exec_module(owned)
        job = owned.OwnedJob(); limits = owned.ExtendedLimits()
        limits.Basic.Flags = 0x2300; limits.ProcessMemory = limits.JobMemory = CAP
        if not job.api.SetInformationJobObject(job.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        with (session/'stdout.txt').open('xb') as stdout, (session/'stderr.txt').open('xb') as stderr:
            child = subprocess.Popen([c['executable'], '--user-data-dir='+str(profile), '--disable-gpu'],
                cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
            record['app_launches'] = 1
            job.attach(child); attached = True
            # The bootstrap waits for this fresh, exclusive acknowledgement before main/renderer/API startup.
            save(session/'attached.json', dict(nonce=nonce, attached=True))
            save(session/'running.json', dict(pid=child.pid, nonce=nonce, bootstrap_released_after_Job_attachment=True))
            while child.poll() is None:
                elapsed = time.monotonic()-began
                if test_timeout is not None:
                    need(elapsed < test_timeout, 'Explicit UI smoke deadline reached')
                sample = resources(root); sample['elapsed_seconds'] = elapsed
                history.append(sample)
                record['resource_sample_count'] += 1
                record['minimum_available_ram_bytes'] = min(record['minimum_available_ram_bytes'], sample['available_ram_bytes'])
                record['minimum_free_disk_bytes'] = min(record['minimum_free_disk_bytes'], sample['disk_free_bytes'])
                admit(sample)
                time.sleep(1)
            record['app_returncode'] = child.returncode
            grace = time.monotonic()
            while job.active() and time.monotonic()-grace < 5:
                time.sleep(.05)
            record['owned_processes_before_cleanup'] = job.active()
            need(child.returncode == 0 and job.active() == 0, 'Preview did not close normally with an empty owned Job')
            record['status'] = 'NORMAL_APP_CLOSE'
    except BaseException as error:
        record.update(error=str(error), traceback=traceback.format_exc())
        print('Preview stopped: '+str(error), file=sys.stderr)
    finally:
        if child is not None and not attached and child.poll() is None:
            child.kill()  # Only the exact newly created preview, still blocked by its bootstrap.
        if job is not None:
            try:
                peak = owned.ExtendedLimits()
                if not job.api.QueryInformationJobObject(job.handle, 9, ctypes.byref(peak), ctypes.sizeof(peak), None):
                    raise ctypes.WinError(ctypes.get_last_error())
                record.update(peak_process_bytes=peak.PeakProcessMemory, peak_Job_bytes=peak.PeakJobMemory)
                record['cleanup'] = job.finish()
                need(not record['cleanup']['forced_job_cleanup'], 'Owned preview required forced cleanup')
            except BaseException as error:
                record.update(status='BLOCKED', cleanup_error=str(error))
        if child is not None:
            try: record['final_returncode'] = child.wait(timeout=5)
            except BaseException as error: record.update(status='BLOCKED', reap_error=str(error))
        record['elapsed_seconds'] = time.monotonic()-began
        record['samples'] = list(history)
        record['retained_sample_limit'] = 3600
        save(session/'result.json', record)
    print('Preview closed. Result: '+record['status'], flush=True)
    return 0 if record['status'] == 'NORMAL_APP_CLOSE' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--test-timeout', type=int, choices=range(30, 331), help='Explicit bounded developer UI smoke only; normal launch has no deadline')
    args = parser.parse_args()
    try:
        raise SystemExit(run(args.config, args.test_timeout))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
