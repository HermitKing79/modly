"""One fixed private image-composition process, 1 GiB / 60 seconds."""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

E = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('preview_host', E / 'render_host.py')
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)
sys.path.insert(0, str(host.ROOT / 'scripts/compatibility'))
from amd_forward_owned_job import OwnedJob, ExtendedLimits


def main():
    if not host.resources()['admitted']:
        raise RuntimeError('Original CPU resource floors not met')
    record_path = host.PRIVATE / 'composition-host.json'
    if record_path.exists():
        raise FileExistsError('No image-composition retry')
    command = [sys.executable, '-I', '-B', str(E / 'compose_exposure.py')]
    record = dict(CPU_only=True, command=command, memory_cap_bytes=1024**3,
                  deadline_seconds=60, resource_samples=[])
    job = OwnedJob()
    limits = ExtendedLimits()
    limits.Basic.Flags = 0x2100
    limits.ProcessMemory = 1024**3
    if not job.api.SetInformationJobObject(job.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        job.close()
        raise ctypes.WinError()
    child = None
    attached = False
    began = time.monotonic()
    try:
        with (host.PRIVATE / 'composition.stdout.txt').open('xb') as out, \
             (host.PRIVATE / 'composition.stderr.txt').open('xb') as err:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                     env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            job.attach(child)
            attached = True
            child.stdin.write(b'GO\n')
            child.stdin.flush()
            child.stdin.close()
            while child.poll() is None:
                if time.monotonic() - began >= 60:
                    raise TimeoutError('60-second CPU image-composition deadline')
                resources = host.resources()
                record['resource_samples'].append(resources)
                if not resources['admitted']:
                    raise RuntimeError('Live original RAM/disk floor failed')
                time.sleep(.1)
            record['returncode'] = child.returncode
            record['process_memory'] = host.process_memory(child)
            grace = time.monotonic()
            while job.active() and time.monotonic() - grace < 5:
                time.sleep(.025)
    except BaseException as exc:
        record.update(error=str(exc), traceback=traceback.format_exc())
        if child is not None and not attached and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
    finally:
        if child is not None and child.stdin is not None and not child.stdin.closed:
            child.stdin.close()
        peak = ExtendedLimits()
        if job.api.QueryInformationJobObject(job.handle, 9, ctypes.byref(peak), ctypes.sizeof(peak), None):
            record['peak_committed_bytes'] = peak.PeakProcessMemory
        else:
            record['measurement_error'] = str(ctypes.WinError())
        record['job_cleanup'] = job.finish()
        if child is not None:
            child.wait(timeout=5)
        record['elapsed_seconds'] = time.monotonic() - began
        record['classification'] = 'PASS' if record.get('returncode') == 0 and \
            not any(key in record for key in ('error', 'measurement_error')) and \
            record['job_cleanup']['owned_tree_exit_verified'] and \
            not record['job_cleanup']['forced_job_cleanup'] else 'NO-GO'
        host.save(record_path, record)
    print(json.dumps(record), flush=True)
    return 0 if record['classification'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
