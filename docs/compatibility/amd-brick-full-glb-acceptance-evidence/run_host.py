"""Four CPU-only acceptance stages using the qualified owned supervision rules."""
import sys
sys.dont_write_bytecode = True
import argparse, ctypes, importlib.util, json, os, subprocess, time, traceback
from pathlib import Path
E = Path(__file__).resolve().parent
sys.path.insert(0, str(E))
from inputs import ROOT, P, ADDON, AUDIT, authenticate, plain, digest, save, require
BASE = 'a258c4eb97c239e3ec766648e71ca59b51ce1de9'
BRANCH = 'codex/amd-brick-full-glb-acceptance'
HELPER = ROOT/'docs/compatibility/amd-brick-pbr-exposure-evidence/render_host.py'
BLENDER = Path('C:/Program Files/Blender Foundation/Blender 5.2/blender.exe')
CAP = 6*1024**3
DEADLINE = 180
FOLDERS = {'import': 'import', 'verify': 'verify', 'render': 'render', 'inspect': 'inspect'}
PREDECESSORS = {'import': (), 'verify': ('import',), 'render': ('import', 'verify'), 'inspect': ('import', 'verify', 'render')}


def worker_command(stage, output):
    if stage in ('import', 'render'):
        return [str(BLENDER), '--background', '--factory-startup', '--disable-autoexec', '--threads', '4', '--python-exit-code', '1', '--python', str(E/(stage+'_worker.py')), '--', '--output', str(output)]
    scripts = {'verify': 'verify_import.py', 'inspect': 'inspect_worker.py'}
    if stage not in scripts:
        raise ValueError('Unknown acceptance stage')
    return ['D:/AI/Trellis2AMD/venv/Scripts/python.exe', '-I', '-B', str(E/scripts[stage])]


def main(stage, preflight=False):
    require(stage in FOLDERS, 'Only the four reviewed acceptance stages are authorised')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == BASE, 'Exact starting HEAD')
    require(subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == BRANCH, 'Isolated branch')
    require(digest(plain(HELPER)) == '2d98a925a0b97aee29d21d259e40eeb290ab44386ab035e15a3463c5df0c1418', 'Original CPU resource helper')
    spec = importlib.util.spec_from_file_location('Blender_acceptance_CPU_monitor', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.PRIVATE = P
    initial = helper.resources()
    require(initial['admitted'], 'Pre-marker resource floor: '+json.dumps(initial))
    _, _, watched, _ = authenticate()
    sources = list(E.glob('*.py'))
    plan = plain(E/'qualification-plan.md')
    tests = plain(E/'focused-tests.txt')
    audit_path = plain(AUDIT)
    review = plain(E/'independent-candidate-review.json')
    r = json.loads(review.read_text())
    require(r['classification'] == 'GO' and digest(plan) == r['qualification_plan_SHA256'], 'Reviewed acceptance candidate and criteria')
    require(r['focused_tests_SHA256'] == digest(tests) and r['importer_audit_SHA256'] == digest(audit_path), 'Reviewed focused tests and importer audit')
    require(set(r['candidate_source_SHA256']) == {x.name for x in sources}, 'Every candidate Python source must be reviewed')
    for name, sha in r['candidate_source_SHA256'].items():
        require(digest(plain(E/name)) == sha, 'Reviewed source changed: '+name)
    watched += sources+[plan, tests, audit_path, review, HELPER, ROOT/'scripts/compatibility/amd_forward_owned_job.py']
    for relative, sha in json.loads(audit_path.read_text())['installed_source_SHA256'].items():
        path = plain(ADDON/relative)
        require(path.resolve().is_relative_to(ADDON.resolve()), 'Installed importer source only')
        require(digest(path) == sha, 'Installed importer source changed: '+relative)
        watched.append(path)
    for source in sources:
        compile(source.read_bytes(), str(source), 'exec')
    for previous in PREDECESSORS[stage]:
        folder = plain(P/FOLDERS[previous])
        result = plain(folder/'result.json')
        require(json.loads(result.read_text())['classification'] == 'PASS', 'Previous stage PASS required: '+previous)
        receipt = plain(P/(FOLDERS[previous]+'-host.json'))
        prior_host = json.loads(receipt.read_text())
        require(prior_host['classification'] == 'PASS' and prior_host['worker_receipt_sha256'] == digest(result), 'Previous owned execution and exact receipt must PASS: '+previous)
        watched.append(receipt)
        watched.extend(path for path in folder.iterdir() if path.is_file())
    if stage == 'render':
        measured = json.loads((P/'import-host.json').read_text())
        require(measured['peak_process_committed_bytes'] <= CAP-1024**3
                and measured['peak_job_committed_bytes'] <= CAP-1024**3,
                'Measured import must leave at least1GiB below unchanged6GiB cap before render')
    if stage in ('import', 'render'):
        require(digest(plain(BLENDER)) == '284f4041f98e113f3dc10654a7193ffaaa9bfdfec8b87fa116620a48b5f6d4cb', 'Installed Blender binary identity')
        watched.append(BLENDER)
    out = P/FOLDERS[stage]
    command = worker_command(stage, out)
    marker = P/(FOLDERS[stage]+'-attempt.json')
    host_result = P/(FOLDERS[stage]+'-host.json')
    require(not any(x.exists() for x in (out, marker, host_result)), 'Exclusive stage; no overwrite/retry')
    paths = list(dict.fromkeys(watched))
    before = {str(x): dict(sha256=digest(plain(x)), bytes=x.stat().st_size, mtime_ns=x.stat().st_mtime_ns) for x in paths}
    if preflight:
        print(json.dumps(dict(classification='PASS', stage=stage, no_marker_reserved=True, resources=initial, command=command, watched_files=len(paths))))
        return 0
    final = helper.resources()
    require(final['admitted'], 'Final pre-marker resource floor: '+json.dumps(final))
    sys.path.insert(0, str(ROOT/'scripts/compatibility'))
    from amd_forward_owned_job import OwnedJob, ExtendedLimits
    out.mkdir(exist_ok=False)
    scratch = out/'scratch'
    scratch.mkdir(exist_ok=False)
    save(marker, dict(CPU_only=True, stage=stage, base=BASE, command=command, source_before=before, memory_cap_bytes=CAP, deadline_seconds=DEADLINE, retry=False, resources=final))
    job = OwnedJob()
    limits = ExtendedLimits()
    limits.Basic.Flags = 0x2300
    limits.ProcessMemory = limits.JobMemory = CAP
    if not job.api.SetInformationJobObject(job.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        job.close()
        raise ctypes.WinError()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TEMP=str(scratch), TMP=str(scratch), TMPDIR=str(scratch), BLENDER_USER_CONFIG=str(scratch), BLENDER_USER_SCRIPTS=str(scratch), BLENDER_USER_EXTENSIONS=str(scratch), CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='', ONEAPI_DEVICE_SELECTOR='*:cpu', OPENCV_OPENCL_RUNTIME='disabled', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    child = None
    began = time.monotonic()
    last = -1
    record = dict(CPU_only=True, stage=stage, command=command, resource_samples=[], memory_cap_bytes=CAP, deadline_seconds=DEADLINE, initial_resources=final)
    try:
        with (out/'stdout.txt').open('xb') as stdout, (out/'stderr.txt').open('xb') as stderr:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, env=env)
            job.attach(child)
            record['attached_before_GO'] = True
            child.stdin.write(b'GO\n')
            child.stdin.flush()
            child.stdin.close()
            record['exact_binary_GO_sent'] = True
            while child.poll() is None:
                elapsed = time.monotonic()-began
                if elapsed >= DEADLINE:
                    raise TimeoutError('Unchanged 180-second CPU deadline')
                if elapsed-last >= 1:
                    sample = helper.resources()
                    sample['elapsed_seconds'] = elapsed
                    record['resource_samples'].append(sample)
                    last = elapsed
                    require(sample['admitted'], 'Live resource floor: '+json.dumps(sample))
                time.sleep(.1)
            record['returncode'] = child.returncode
            require(time.monotonic()-began <= DEADLINE, 'Exact CPU deadline')
            grace = time.monotonic()
            while job.active() and time.monotonic()-grace < 5 and time.monotonic()-began < DEADLINE:
                time.sleep(.025)
            record['owned_execution_seconds'] = time.monotonic()-began
            require(job.active() == 0 and record['owned_execution_seconds'] <= DEADLINE, 'Owned process tree must settle naturally inside the deadline')
            if child.returncode == 0:
                result = plain(out/'result.json')
                require(result.stat().st_size < 1024**2, 'Bounded worker receipt')
                receipt = json.loads(result.read_text())
                require(receipt['classification'] == 'PASS', 'Actual worker PASS required')
                record['worker_receipt_sha256'] = digest(result)
    except BaseException as error:
        record.update(error=str(error), traceback=traceback.format_exc())
        if child is not None and not record.get('attached_before_GO') and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
    finally:
        if child is not None and child.stdin is not None and not child.stdin.closed:
            child.stdin.close()
        peak = ExtendedLimits()
        if job.api.QueryInformationJobObject(job.handle, 9, ctypes.byref(peak), ctypes.sizeof(peak), None):
            record.update(peak_process_committed_bytes=peak.PeakProcessMemory, peak_job_committed_bytes=peak.PeakJobMemory)
        else:
            record['measurement_error'] = str(ctypes.WinError())
        try:
            record['job_cleanup'] = job.finish()
            if child is not None:
                child.wait(timeout=5)
        except BaseException as error:
            record['cleanup_error'] = str(error)
        try:
            after = {str(x): dict(sha256=digest(x), bytes=x.stat().st_size, mtime_ns=x.stat().st_mtime_ns) for x in paths}
            record['sources_unchanged'] = before == after
        except BaseException as error:
            record['preservation_error'] = str(error)
        record['elapsed_seconds'] = time.monotonic()-began
        cleanup = record.get('job_cleanup', {})
        clean = record.get('returncode') == 0 and 'worker_receipt_sha256' in record and record.get('sources_unchanged') and not any(k in record for k in ('error', 'measurement_error', 'cleanup_error', 'preservation_error')) and cleanup.get('owned_tree_exit_verified') and not cleanup.get('forced_job_cleanup', True) and record.get('peak_process_committed_bytes', CAP+1) <= CAP and record.get('peak_job_committed_bytes', CAP+1) <= CAP
        record['classification'] = 'PASS' if clean else 'BLOCKED'
        save(host_result, record)
    print(json.dumps({k: v for k, v in record.items() if k != 'resource_samples'}), flush=True)
    return 0 if record['classification'] == 'PASS' else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=tuple(FOLDERS))
    ap.add_argument('--preflight', action='store_true')
    args = ap.parse_args()
    raise SystemExit(main(args.stage, args.preflight))
