"""One bounded CPU qualification: two native controls and ONE synthetic chain.

No retry or alternate case is dispatched after any failed gate. No GLB is read
or generated. The WSL broker, not distro termination, must settle Linux children.
"""
import ctypes
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time
import traceback
import zlib

E = Path(__file__).resolve().parent
ROOT = E.parents[2]
P = ROOT.parent / 'artifacts/amd-job-stage-orchestrator-c5f7ccd'
OUT = P / 'cpu-once'
FIXTURE = E / 'cpu_stage_fixture.py'


def save(path, row):
    path = Path(path)
    staging = path.with_name(path.name + '.writing')
    with staging.open('x', encoding='utf-8') as stream:
        json.dump(row, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.link(staging, path); staging.unlink()


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def image_bytes():
    def chunk(kind, data): return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    pixels = b''.join(b'\0' + b''.join(bytes((40, 80, 100, 255 if 2 <= x <= 5 and 2 <= y <= 5 else 0)) for x in range(8)) for y in range(8))
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 8, 8, 8, 6, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b'')


def worker():
    if sys.stdin.buffer.readline(32) != b'GO\n': raise ValueError('Outer Job admission required')
    sys.path.insert(0, str(ROOT / 'api'))
    from services.amd_completed_result import CompletionContext, FilePin
    from services.amd_stage_process import OwnedStageRunner, StageCommand, _guest_path
    from services.amd_stage_orchestrator import StagePlan, StageDefinition, OutputSpec, StageOrchestrator, StageCancelled
    params = {'image_mode': 'alpha-only', 'resolution': 512}
    image = image_bytes()
    def context(name): return CompletionContext.new(name, image, params, model_id='test-only/CPU-lifecycle')
    controls = OUT / 'windows-controls'; controls.mkdir()
    def command(name, mode, identity):
        cwd = controls / name; cwd.mkdir()
        return StageCommand(name, 'windows', (sys.executable, '-B', '-I', '-S', str(FIXTURE), mode), cwd, identity.to_dict(), 25)
    cancelled, sentinel_event = threading.Event(), threading.Event()
    active_events = [cancelled, sentinel_event]
    outer_abort = threading.Event()
    def outer_cancel_watch():
        marker = OUT / 'host-cancel.json'
        while True:
            if marker.exists():
                row = json.loads(marker.read_text())
                if row != {'action': 'CANCEL', 'attempt_sha256': sha(OUT / 'attempt.json')}: raise RuntimeError('Foreign outer cancellation')
                outer_abort.set()
                for active in active_events: active.set()
                return
            time.sleep(.025)
    threading.Thread(target=outer_cancel_watch, daemon=True).start()
    target = command('cancel-target', 'wait', context('control-target'))
    sentinel = command('unrelated-sentinel', 'sentinel', context('control-sentinel'))
    target_runner = OwnedStageRunner(OUT / 'target-process')
    sentinel_runner = OwnedStageRunner(OUT / 'sentinel-process')
    with ThreadPoolExecutor(max_workers=2) as pool:
        target_future = pool.submit(target_runner.run, target, cancelled)
        sentinel_future = pool.submit(sentinel_runner.run, sentinel, sentinel_event)
        until = time.monotonic() + 15
        while not all((command.cwd / 'ready.json').exists() for command in (target, sentinel)):
            if time.monotonic() >= until or target_future.done() or sentinel_future.done():
                raise RuntimeError('Windows controls failed before admitted cancellation')
            time.sleep(.025)
        cancelled.set()
        a = target_future.result(timeout=30)
        if sentinel_future.done() or sentinel_event.is_set(): raise RuntimeError('Independent sentinel did not survive target cancellation')
        save(sentinel.cwd / 'sentinel-release.json', {'context': sentinel.context, 'stage_id': sentinel.stage_id})
        b = sentinel_future.result(timeout=30)
    save(OUT / 'windows-controls.json', {'target': a, 'sentinel': b})
    for row in (a, b):
        if row['windows_settled'] is not True or row['natural'] is not True or row['forced'] is not False:
            raise RuntimeError('Windows native tree settlement failed')
    if a['terminal'] != 'cancelled' or b['terminal'] != 'succeeded' or sentinel_event.is_set():
        raise RuntimeError('Cancellation was not isolated to its product job')
    # Exactly one actual StageOrchestrator run; it must stop at WSL cancellation.
    identity = context('one-synthetic-chain')
    runs = OUT / 'runs'; runs.mkdir()
    pins = {'worker': FilePin(FIXTURE, sha(FIXTURE))}
    stages = (
        StageDefinition('control_windows', 'windows', (sys.executable, '-B', '-I', '-S', str(FIXTURE), 'success'), pins, (OutputSpec('output.json', 'json', {}),), 25),
        StageDefinition('control_wsl', 'wsl', ('/usr/bin/python3', '-B', '-I', '-S', _guest_path(FIXTURE), 'wait'), pins, (OutputSpec('output.json', 'json', {}),), 45),
    )
    plan = StagePlan(identity, 'cpu_lifecycle_test', params, stages)
    event, finished = threading.Event(), threading.Event()
    active_events.append(event)
    if outer_abort.is_set(): event.set()
    def cancel_wsl():
        ready = runs / identity.run_id / 'stages/01-control_wsl/ready.json'
        until = time.monotonic() + 55
        while not finished.is_set():
            if ready.exists(): event.set(); return
            if time.monotonic() >= until: return
            time.sleep(.025)
    watcher = threading.Thread(target=cancel_wsl)
    watcher.start()
    chain_cancelled = False
    try:
        StageOrchestrator(runs, OwnedStageRunner(OUT / 'chain-processes')).run(plan, image, params, event)
    except StageCancelled:
        chain_cancelled = True
    finally:
        finished.set(); watcher.join(timeout=2)
    if not chain_cancelled: raise RuntimeError('One synthetic chain did not terminate by clean cancellation')
    if outer_abort.is_set(): raise RuntimeError('Outer resource/deadline cancellation prevents qualification')
    result = json.loads((runs / identity.run_id / 'result.json').read_text())
    wsl = json.loads((runs / identity.run_id / 'stages/01-control_wsl/process-settlement.json').read_text())
    if not (wsl['terminal'] == 'cancelled' and wsl['windows_settled'] and wsl['linux_settled'] and wsl['natural'] and not wsl['forced']):
        raise RuntimeError('Linux session/descendant settlement was not independently acknowledged')
    if list(OUT.rglob('*.glb')) or any(n.split('.')[0] in ('torch', 'triton', 'flex_gemm') for n in sys.modules):
        raise RuntimeError('Forbidden generated asset or model runtime')
    save(OUT / 'result.json', {'classification': 'PASS', 'CPU_only': True, 'synthetic_only': True,
                             'actual_orchestrated_chains': 1, 'windows_controls': 'PASS',
                             'chain_cancelled': result, 'WSL': wsl, 'fresh_GPU_inference': 'NOT RUN'})


def host():
    sys.path.insert(0, str(ROOT / 'api'))
    from services.amd_stage_process import _OwnedJob, _host_health
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, timeout=15).strip() != 'c5f7ccd3cdb3de371405c09856a356ebc41a6ba3':
        raise RuntimeError('Approved base plus reviewed working candidate required')
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True, timeout=15).strip() != 'codex/amd-job-stage-orchestrator-v1':
        raise RuntimeError('Isolated product branch required')
    initial = _host_health(P)
    watched = [ROOT / 'api/services' / name for name in ('amd_stage_contract.py', 'amd_stage_orchestrator.py', 'amd_stage_process.py', 'amd_stage_bootstrap.py', 'amd_wsl_session.py')]
    watched += [FIXTURE, Path(__file__)]
    pins = {str(path): sha(path) for path in watched}
    OUT.mkdir(exist_ok=False)
    save(OUT / 'attempt.json', {'CPU_only': True, 'max_chains': 1, 'source_pins': pins, 'initial_resources': initial,
                              'deadline_seconds': 180, 'process_and_job_bytes': 6 * 1024**3,
                              'scenarios': ['native Windows cancelled target + isolated natural sentinel', 'one Windows-success/WSL-cancelled synthetic chain']})
    job = _OwnedJob(); child = None; attached = False
    result = {'classification': 'BLOCKED', 'initial_resources': initial, 'samples': []}
    started = time.monotonic()
    try:
        with (OUT / 'stdout.txt').open('xb') as stdout, (OUT / 'stderr.txt').open('xb') as stderr:
            child = subprocess.Popen([sys.executable, '-B', '-I', '-S', str(Path(__file__).resolve()), '--worker'],
                                     stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                     env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES=''))
            job.attach(child); attached = True
            child.stdin.write(b'GO\n'); child.stdin.close()
            while child.poll() is None:
                if time.monotonic() - started >= 180: raise TimeoutError('Outer CPU qualification deadline')
                result['samples'].append({'elapsed_s': time.monotonic() - started, **_host_health(P)})
                result['windows'] = job.snapshot()
                time.sleep(.1)
            result['returncode'] = child.returncode
            until = time.monotonic() + 5
            while job.snapshot()['active'] and time.monotonic() < until: time.sleep(.025)
            result['windows'] = job.snapshot()
            if result['windows']['active'] != 0: raise RuntimeError('Outer native tree survived normal worker exit')
            if child.returncode == 0:
                receipt = json.loads((OUT / 'result.json').read_text())
                if receipt['classification'] != 'PASS': raise RuntimeError('CPU worker validation failed')
                result['classification'] = 'PASS'
    except BaseException:
        result['error'] = traceback.format_exc()
        if child is not None and attached and child.poll() is None:
            try:
                save(OUT / 'host-cancel.json', {'action': 'CANCEL', 'attempt_sha256': sha(OUT / 'attempt.json')})
                until = time.monotonic() + 8
                while child.poll() is None and time.monotonic() < until: time.sleep(.025)
            except BaseException: result['cooperative_outer_cancellation_error'] = traceback.format_exc()
        if child is not None and not attached and child.poll() is None: child.kill(); child.wait(timeout=5)
    finally:
        if job.snapshot()['active']:
            result['forced_outer_cleanup'] = True; job.terminate()
            result['linux_settlement'] = 'UNVERIFIED; outer Windows Job termination cannot prove Linux settlement'
            until = time.monotonic() + 5
            while job.snapshot()['active'] and time.monotonic() < until: time.sleep(.025)
        result['windows'] = job.snapshot(); job.close()
        if child is not None: child.wait(timeout=5)
        result.update(elapsed_s=time.monotonic() - started, sources_unchanged=all(sha(path) == expected for path, expected in pins.items()))
        if result.get('forced_outer_cleanup') or not result['sources_unchanged']: result['classification'] = 'BLOCKED'
        save(OUT / 'host-result.json', result)
    print(json.dumps({'classification': result['classification'], 'elapsed_s': result['elapsed_s'], 'windows': result['windows']}))
    return 0 if result['classification'] == 'PASS' else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']: worker()
    elif not sys.argv[1:]: raise SystemExit(host())
    else: raise SystemExit('Unsupported CPU qualification invocation')
