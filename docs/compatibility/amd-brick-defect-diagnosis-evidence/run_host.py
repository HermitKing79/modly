"""One owned, read-only CPU diagnostic process: 6 GiB / 180 seconds."""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
E = Path(__file__).resolve().parent
ROOT = E.parents[2]
P = ROOT.parent / 'artifacts/amd-brick-defect-diagnosis-1d1c1a6/analysis'
HELPER = ROOT / 'docs/compatibility/amd-brick-pbr-exposure-evidence/render_host.py'
HELPER_SHA = '2d98a925a0b97aee29d21d259e40eeb290ab44386ab035e15a3463c5df0c1418'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA
spec = importlib.util.spec_from_file_location('original_CPU_containment_helpers', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.PRIVATE = P  # Same resource function, sampling the actual new output volume.
sys.path.insert(0, str(ROOT / 'scripts/compatibility'))
from amd_forward_owned_job import OwnedJob, ExtendedLimits


def main():
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != '1d1c1a6b54050200fca870b49c1b26bb51338c6a':
        raise ValueError('Exact approved base required')
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() != 'codex/amd-brick-geometry-defect-diagnosis':
        raise ValueError('Isolated diagnosis branch required')
    first = helper.resources()
    if not first['admitted']:
        raise RuntimeError('CPU resource admission: ' + json.dumps(first))
    helper.plain(P)
    helper.plain(E / 'diagnose.py')
    marker = P / 'analysis-attempt.json'
    command = [sys.executable, '-I', '-B', str(E / 'diagnose.py')]
    helper.save(marker, dict(CPU_only=True, command=command, memory_cap_bytes=6 * 1024**3,
                            deadline_seconds=180, initial_resources=first,
                            source_sha256={n: helper.digest(E / n) for n in
                                           ('diagnose.py', 'topology.py', 'projection.py', 'run_host.py')}))
    record = dict(CPU_only=True, command=command, initial_resources=first,
                  memory_cap_bytes=6 * 1024**3, deadline_seconds=180, resource_samples=[])
    job = OwnedJob()
    limits = ExtendedLimits()
    limits.Basic.Flags = 0x2100
    limits.ProcessMemory = 6 * 1024**3
    if not job.api.SetInformationJobObject(job.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        job.close()
        raise ctypes.WinError()
    child, attached = None, False
    began, last = time.monotonic(), -1
    try:
        with (P / 'analysis.stdout.txt').open('xb') as out, (P / 'analysis.stderr.txt').open('xb') as err:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                     env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                                              OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4'))
            job.attach(child)
            attached = True
            child.stdin.write(b'GO\n')
            child.stdin.flush()
            child.stdin.close()
            while child.poll() is None:
                elapsed = time.monotonic() - began
                if elapsed >= 180:
                    raise TimeoutError('180-second CPU diagnosis deadline')
                if elapsed - last >= 1:
                    sample = helper.resources()
                    sample['elapsed_seconds'] = elapsed
                    record['resource_samples'].append(sample)
                    last = elapsed
                    if not sample['admitted']:
                        raise RuntimeError('Live CPU resource floor: ' + json.dumps(sample))
                time.sleep(.1)
            record['returncode'] = child.returncode
            record['process_memory'] = helper.process_memory(child)
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
            record['peak_process_committed_bytes'] = peak.PeakProcessMemory
        else:
            record['measurement_error'] = str(ctypes.WinError())
        record['job_cleanup'] = job.finish()
        if child is not None:
            child.wait(timeout=5)
        record['elapsed_seconds'] = time.monotonic() - began
        record['classification'] = 'PASS' if record.get('returncode') == 0 and \
            not any(k in record for k in ('error', 'measurement_error')) and \
            not record['job_cleanup']['forced_job_cleanup'] and \
            record['job_cleanup']['owned_tree_exit_verified'] else 'BLOCKED'
        helper.save(P / 'host-result.json', record)
    print(json.dumps(record), flush=True)
    return 0 if record['classification'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
