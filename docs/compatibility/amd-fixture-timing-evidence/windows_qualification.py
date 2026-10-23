"""One source-pinned, private Windows CPU target/sentinel pair; no retry.

Uses the separately pinned, timing-instrumented fixture; no production changes. Importing this file launches nothing.
--preflight verifies static admission without reserving or launching an attempt.
"""
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
import uuid
import zlib

E = Path(__file__).resolve().parent
ROOT = E.parents[2]
P = ROOT.parent / 'artifacts/amd-fixture-timing-0c031be'
OUT = P / 'cpu-once'
FIXTURE = E / 'diagnostic_fixture.py'
BASE, BRANCH = '0c031befef0483f4ac3bed05fa35c3546166f255', 'codex/amd-fixture-readiness-timing'
PINS = ('api/services/amd_stage_process.py', 'api/services/amd_stage_bootstrap.py',
        'api/services/amd_completed_result.py',
        'docs/compatibility/amd-job-stage-orchestrator-evidence/cpu_stage_fixture.py',
        'docs/compatibility/amd-retired-process-evidence/windows_qualification.py',
        'docs/compatibility/amd-fixture-timing-evidence/diagnostic_fixture.py',
        'docs/compatibility/amd-fixture-timing-evidence/diagnostic_process.py',
        'docs/compatibility/amd-fixture-timing-evidence/diagnostic_bootstrap.py',
        'docs/compatibility/amd-fixture-timing-evidence/windows_qualification.py')
_lock = threading.Lock()
_PRELAUNCH_PHASE = 'not-entered'
# Independently selected private diagnostics, never the consumed attempt leaf.
ERROR_ROOT = ROOT.parent / 'artifacts/amd-evidence-writer-d784e4b/diagnostics'

def sha(path):
    path = Path(path)
    stream = _publication_call('sha.open-read', lambda: path.open('rb'), None, path)
    try:
        value = _publication_call('sha.read-digest',
            lambda: hashlib.file_digest(stream, 'sha256').hexdigest(), None, path)
    except BaseException as original:
        _close_after_error(stream, original)
        raise
    _publication_call('sha.close', stream.close, None, path)
    return value

def _publication_call(operation, callback, staging, final):
    try:
        return callback()
    except BaseException as error:
        # Attach attribution to the original object; never wrap or replace it.
        try:
            error.modly_publication = dict(operation=operation, phase=operation,
                staging=None if staging is None else str(staging), final=str(final))
        except BaseException:
            pass
        raise


def _close_after_error(stream, original):
    try:
        stream.close()
    except BaseException as secondary:
        try:
            original.modly_publication['close_error'] = dict(type=type(secondary).__name__,
                message=str(secondary), errno=getattr(secondary, 'errno', None),
                winerror=getattr(secondary, 'winerror', None))
        except BaseException:
            pass


def save(path, row):
    path = Path(path); staging = path.with_name(path.name + '.writing')
    stream = _publication_call('staging.open-exclusive',
        lambda: staging.open('x', encoding='utf-8'), staging, path)
    try:
        def write_json():
            json.dump(row, stream, indent=2, allow_nan=False); stream.write('\n')
        _publication_call('json.write', write_json, staging, path)
        _publication_call('flush', stream.flush, staging, path)
        _publication_call('os.fsync', lambda: os.fsync(stream.fileno()), staging, path)
    except BaseException as original:
        _close_after_error(stream, original)
        raise
    _publication_call('close', stream.close, staging, path)
    _publication_call('os.link', lambda: os.link(staging, path), staging, path)
    _publication_call('staging.unlink', staging.unlink, staging, path)


def _path_snapshot(path):
    if path is None:
        return dict(path=None, exists=None)
    try:
        info = os.lstat(path)
        return dict(path=str(path), exists=True, device=info.st_dev, inode=info.st_ino,
                    mode=info.st_mode)
    except FileNotFoundError:
        return dict(path=str(path), exists=False)
    except BaseException as error:
        return dict(path=str(path), exists=None, state_error=dict(type=type(error).__name__,
            message=str(error), errno=getattr(error, 'errno', None), winerror=getattr(error, 'winerror', None)))


def _write_error_row(row):
    location = ERROR_ROOT.resolve()
    if location == OUT.resolve() or OUT.resolve() in location.parents:
        raise ValueError('Error location must be independent of attempted output')
    location.mkdir(parents=True, exist_ok=True)
    path = location / ('prelaunch-' + uuid.uuid4().hex + '.json')
    # Ordinary exclusive write avoids recursively depending on failed publication.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return path


def report_prelaunch_failure(error):
    """Best-effort diagnostic delivery; callers always re-raise their original."""
    row = None
    try:
        publication = getattr(error, 'modly_publication', {})
        writing = publication.get('staging')
        staged_path = writing if writing is not None else OUT / 'attempt.json.writing'
        final_path = publication.get('final') if writing is not None else OUT / 'attempt.json'
        row = dict(schema='modly.prelaunch-error.v1', type=type(error).__name__, message=str(error),
            errno=getattr(error, 'errno', None), winerror=getattr(error, 'winerror', None),
            traceback=''.join(traceback.format_exception(type(error), error, error.__traceback__)),
            prelaunch_phase=_PRELAUNCH_PHASE, publication_phase=publication.get('phase'),
            filesystem_operation=publication.get('operation'), publication=publication,
            staging=_path_snapshot(staged_path), final=_path_snapshot(final_path),
            operation_paths=dict(staging=_path_snapshot(publication.get('staging')),
                                 final=_path_snapshot(publication.get('final'))))
        return _write_error_row(row)
    except BaseException as diagnostic_error:
        # No diagnostic failure may replace or conceal the primary exception.
        try:
            if row is None:
                row = dict(type=type(error).__name__, message=str(error),
                    traceback=''.join(traceback.format_exception(type(error), error, error.__traceback__)))
            row['diagnostic_delivery_error'] = dict(type=type(diagnostic_error).__name__,
                message=str(diagnostic_error), errno=getattr(diagnostic_error, 'errno', None),
                winerror=getattr(diagnostic_error, 'winerror', None))
            print(json.dumps(row, indent=2, allow_nan=False), file=sys.stderr, flush=True)
        except BaseException:
            # If stderr itself is broken, host_entry still raises the original.
            pass
        return None


def host_entry():
    try:
        return host()
    except Exception as error:
        report_prelaunch_failure(error)
        raise

def preflight_entry():
    global _PRELAUNCH_PHASE
    _PRELAUNCH_PHASE = 'host.preflight'
    try:
        return preflight()
    except Exception as error:
        report_prelaunch_failure(error)
        raise


def read(path):
    path = Path(path)
    data = _publication_call('Path.read_bytes', path.read_bytes, None, path)
    if len(data) > 16 * 1024**2: raise ValueError('Private JSON exceeded bounded size')
    return json.loads(data)

def trace(who, event, **fields):
    import datetime
    row = dict(event=event, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), monotonic_s=time.monotonic(), **fields)
    with _lock, (OUT / (who + '-trace.jsonl')).open('ab', buffering=0) as stream:
        stream.write((json.dumps(row, allow_nan=False) + '\n').encode()); os.fsync(stream.fileno())

def source_pins(): return {name: sha(ROOT / name) for name in PINS}

def preflight():
    if sys.platform != 'win32': raise RuntimeError('Windows-only diagnostic')
    admission = read(E / 'admission.json'); pins = source_pins()
    ref = 'refs/heads/' + BRANCH
    head_path, ref_path = ROOT / '.git/HEAD', ROOT / '.git' / ref
    if _publication_call('Path.read_text', head_path.read_text, None, head_path).strip() != 'ref: ' + ref: raise RuntimeError('Reviewed isolated branch required')
    if _publication_call('Path.read_text', ref_path.read_text, None, ref_path).strip() != BASE: raise RuntimeError('Reviewed base required')
    if admission.get('base_commit') != BASE or admission.get('branch') != BRANCH: raise RuntimeError('Exact base/branch admission required')
    if admission.get('source_pins') != pins: raise RuntimeError('Exact reviewed source pin map required')
    executable = admission.get('python_executable')
    if not isinstance(executable, str) or not Path(executable).is_absolute(): raise RuntimeError('Reviewed original API interpreter path required')
    current = _publication_call('Path.resolve', lambda: Path(sys.executable).resolve(), None, sys.executable)
    expected = _publication_call('Path.resolve', lambda: Path(executable).resolve(), None, executable)
    if current != expected: raise RuntimeError('Original API interpreter required')
    if sha(sys.executable) != admission.get('python_executable_sha256'): raise RuntimeError('API interpreter hash differs')
    if _publication_call('Path.exists', OUT.exists, None, OUT) or not _publication_call('Path.is_dir', P.is_dir, None, P): raise RuntimeError('Fresh exclusive attempt and existing private parent required')
    return admission, pins

def image_bytes():
    def chunk(kind, data): return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    pixels = b''.join(b'\0' + b''.join(bytes((40, 80, 100, 255 if 2 <= x <= 5 and 2 <= y <= 5 else 0)) for x in range(8)) for y in range(8))
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 8, 8, 8, 6, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b'')

def natural(row, terminal):
    return (isinstance(row, dict) and row.get('terminal') == terminal and row.get('windows_settled') is True
            and row.get('natural') is True and row.get('forced') is False and not row.get('diagnostic_errors'))


def control_pair_pass():
    # Raw observation provenance can exceed the unchanged bounded JSON reader.
    # Authenticate its complete streamed bytes instead of loading it in the host.
    status = read(OUT / 'control-verdict.json'); raw = OUT / 'windows-controls.json'
    return status == {'schema': 'modly.windows-control-verdict.v1', 'classification': 'PASS',
                      'bytes': raw.stat().st_size, 'sha256': sha(raw)}

def worker():
    save(OUT / 'worker-ready.json', {'pid': os.getpid()})
    if sys.stdin.buffer.readline(32) != b'GO\n': raise ValueError('Outer owned Job admission required')
    sys.path.insert(0, str(ROOT / 'api'))
    from services.amd_completed_result import CompletionContext
    sys.path.insert(0, str(E))
    from diagnostic_process import OwnedStageRunner, StageCommand, _error_detail
    attempt = read(OUT / 'attempt.json')
    if (attempt['source_pins'] != source_pins() or attempt['admission_sha256'] != sha(E / 'admission.json')
            or attempt['python_executable_sha256'] != sha(sys.executable)): raise RuntimeError('Worker admission changed')
    worker_timing = dict(anchor=dict(before_ns=time.monotonic_ns(), utc_ns=time.time_ns(), after_ns=time.monotonic_ns()), last_poll=None)
    events = {name: threading.Event() for name in ('target', 'sentinel')}
    done = {name: threading.Event() for name in events}; states = {}; commands = {}; runners = {}; threads = {}
    stop = threading.Event(); abort = threading.Event(); errors = []
    pair = {'classification': 'BLOCKED', 'CPU_only': True, 'synthetic_only': True, 'actual_control_pairs': 1,
            'source_pins': attempt['source_pins'], 'pre_cancel': None, 'during_cancel': [], 'post_target_settle': None}
    def watch():
        try:
            while not stop.wait(.025):
                marker = OUT / 'host-cancel.json'
                if marker.exists() or time.monotonic() >= attempt['active_end']:
                    if marker.exists() and read(marker) != {'action': 'CANCEL', 'attempt_sha256': sha(OUT / 'attempt.json')}: raise RuntimeError('Foreign host cancellation')
                    abort.set()
                    for event in events.values(): event.set()
                    trace('worker', 'outer-cancellation-accepted'); return
        except BaseException as error:
            errors.append(_error_detail(error, 'worker.outer-watch')); abort.set()
            for event in events.values(): event.set()
    def run(name):
        try:
            row = runners[name].run(commands[name], events[name])
            states[name] = {'receipt': row, 'finished_monotonic_s': time.monotonic()}
            save(OUT / (name + '-receipt.json'), row)
        except BaseException as error:
            states.setdefault(name, {})['exception'] = _error_detail(error, 'worker.runner.' + name)
        finally: done[name].set()
    def live(name, after=0):
        command = commands[name]; ready = read(command.cwd / 'ready.json'); now = time.monotonic()
        if ready.get('context') != command.context or ready.get('stage_id') != command.stage_id or type(ready.get('pid')) is not int: raise RuntimeError('Fixture readiness identity/context differs')
        observed = runners[name].last_windows_observation
        if not isinstance(observed, dict) or not after <= observed.get('monotonic_s', -1) <= now or now - observed['monotonic_s'] > .5: return None
        identities = [row for row in observed.get('process_identities', []) if row.get('pid') == ready['pid']
                      and row.get('job_membership_verified') is True and type(row.get('creation_filetime')) is int
                      and row['creation_filetime'] > 0 and row.get('observed_alive', {}).get('alive') is True]
        if len(identities) != 1 or ready['pid'] not in observed.get('enumerated_pids', []) or not observed.get('job_owner'): return None
        return {'ready': ready, 'observation': observed, 'actual_fixture_identity': identities[0],
                'source_context': {'source_pins': attempt['source_pins'], 'context': command.context, 'stage_id': command.stage_id}}
    def check():
        if abort.is_set() or time.monotonic() >= attempt['active_end']: raise RuntimeError('Outer cancellation/deadline prevents qualification')
    threading.Thread(target=watch, daemon=True).start()
    try:
        controls = OUT / 'windows-controls'; controls.mkdir()
        for name, stage, mode in (('target', 'cancel-target', 'wait'), ('sentinel', 'independent-sentinel', 'sentinel')):
            context = CompletionContext.new('control-' + name, image_bytes(), {'image_mode': 'alpha-only', 'resolution': 512}, model_id='test-only/CPU-lifecycle')
            cwd = controls / stage; cwd.mkdir()
            commands[name] = StageCommand(stage, 'windows', (sys.executable, '-B', '-I', '-S', str(FIXTURE), mode), cwd, context.to_dict(), 25)
            runners[name] = OwnedStageRunner(OUT / (name + '-process'))
            threads[name] = threading.Thread(target=run, args=(name,), daemon=True)
        trace('worker', 'pair-admitted', contexts={name: cmd.context for name, cmd in commands.items()})
        for thread in threads.values(): thread.start()
        worker_timing['readiness_begin_ns'] = time.monotonic_ns()
        until = min(time.monotonic() + 15, attempt['active_end'])
        while True:
            check()
            if any(event.is_set() for event in done.values()): raise RuntimeError('A control exited before admitted cancellation')
            poll_begin_ns = time.monotonic_ns()
            ready_states = {name: (cmd.cwd / 'ready.json').exists() for name, cmd in commands.items()}
            worker_timing['last_poll'] = dict(begin_ns=poll_begin_ns, end_ns=time.monotonic_ns(), states=ready_states)
            if all(ready_states.values()):
                a, b = live('target'), live('sentinel')
                if a and b:
                    if a['observation']['job_owner'] == b['observation']['job_owner']: raise RuntimeError('Distinct owned Jobs required')
                    worker_timing['authenticated_ready_ns'] = time.monotonic_ns()
                    pair['pre_cancel'] = {'target': a, 'sentinel': b}; trace('worker', 'pre-cancel-live', proof=pair['pre_cancel']); break
            if time.monotonic() >= until: raise TimeoutError('Original fixtures not authenticated ready')
            time.sleep(.01)
        events['target'].set()
        cancelled_at = time.monotonic(); pair['cancel_monotonic_s'] = cancelled_at
        trace('worker', 'target-cancel-requested', cancel_monotonic_s=cancelled_at)
        until = min(time.monotonic() + 12, attempt['active_end']); last = None
        while not done['target'].is_set():
            check()
            if done['sentinel'].is_set() or events['sentinel'].is_set(): raise RuntimeError('Sentinel ended during target cancellation')
            row = live('sentinel', cancelled_at)
            if row and row['observation']['monotonic_s'] != last:
                pair['during_cancel'].append(row); last = row['observation']['monotonic_s']; trace('worker', 'sentinel-live-during-cancel', proof=row)
            if time.monotonic() >= until: raise TimeoutError('Target cancellation settlement deadline')
            time.sleep(.01)
        a = states['target'].get('receipt')
        if not natural(a, 'cancelled') or states['target'].get('exception'): raise RuntimeError('Target natural cancellation failed; release forbidden')
        settled_at = states['target']['finished_monotonic_s']; pair['target_settled_monotonic_s'] = settled_at
        pair['during_cancel'] = [row for row in pair['during_cancel'] if cancelled_at <= row['observation']['monotonic_s'] <= settled_at]
        row = live('sentinel', cancelled_at)
        if row and row['observation']['monotonic_s'] <= settled_at and row['observation']['monotonic_s'] != last: pair['during_cancel'].append(row)
        if not pair['during_cancel']: raise RuntimeError('Fresh sentinel observation during target cancellation missing')
        until = min(time.monotonic() + 1, attempt['active_end'])
        while True:
            check()
            if done['sentinel'].is_set() or events['sentinel'].is_set(): raise RuntimeError('Sentinel did not survive target settlement')
            row = live('sentinel', settled_at)
            if row:
                if row['observation']['job_owner'] != pair['pre_cancel']['sentinel']['observation']['job_owner']: raise RuntimeError('Sentinel Job owner changed')
                pair['post_target_settle'] = row; trace('worker', 'sentinel-live-after-target-settle', proof=row); break
            if time.monotonic() >= until: raise TimeoutError('Fresh post-settlement sentinel liveness missing')
            time.sleep(.01)
        cmd = commands['sentinel']; save(cmd.cwd / 'sentinel-release.json', {'context': cmd.context, 'stage_id': cmd.stage_id})
        trace('worker', 'sentinel-released-after-natural-target-settlement'); until = min(time.monotonic() + 12, attempt['active_end'])
        while not done['sentinel'].wait(.025):
            check()
            if time.monotonic() >= until: raise TimeoutError('Released sentinel settlement deadline')
        if not natural(states['sentinel'].get('receipt'), 'succeeded') or states['sentinel'].get('exception') or abort.is_set() or events['sentinel'].is_set() or errors: raise RuntimeError('Sentinel natural success failed')
        if list(OUT.rglob('*.glb')) or any(n.split('.')[0] in ('torch', 'triton', 'flex_gemm') for n in sys.modules): raise RuntimeError('Forbidden generated asset or model runtime')
        pair['classification'] = 'PASS'
    except BaseException as error:
        worker_timing['error_observed_ns'] = time.monotonic_ns()
        pair['error'] = _error_detail(error, 'worker.controls')
        try: trace('worker', 'primary-error', error=pair['error'])
        except BaseException as secondary: errors.append(_error_detail(secondary, 'worker.error-trace'))
    finally:
        if pair['classification'] != 'PASS':
            for event in events.values(): event.set()
        until = min(time.monotonic() + 8, attempt['grace_end'])
        for thread in threads.values():
            if thread.ident is not None: thread.join(timeout=max(0, until - time.monotonic()))
        stop.set(); pair.update(timing=worker_timing, runner_states=states, target=states.get('target', {}).get('receipt'), sentinel=states.get('sentinel', {}).get('receipt'), observer_errors=errors)
        pair['runner_threads_alive'] = [name for name, thread in threads.items() if thread.is_alive()]
        if errors or pair['runner_threads_alive']: pair['classification'] = 'BLOCKED'
        save(OUT / 'timing-before-controls.json', dict(schema='modly.fixture-timing.worker.v1', timing=worker_timing))
        raw_begin_ns = time.monotonic_ns()
        save(OUT / 'windows-controls.json', pair)
        raw_end_ns = time.monotonic_ns()
        save(OUT / 'timing-after-controls.json', dict(schema='modly.fixture-timing.worker.v1', timing=worker_timing,
             raw_serialization_begin_ns=raw_begin_ns, raw_serialization_end_ns=raw_end_ns,
             raw_serialization_elapsed_ns=raw_end_ns-raw_begin_ns))
        raw = OUT / 'windows-controls.json'
        save(OUT / 'control-verdict.json', {'schema': 'modly.windows-control-verdict.v1',
             'classification': pair['classification'], 'bytes': raw.stat().st_size, 'sha256': sha(raw)})
    return 0 if pair['classification'] == 'PASS' else 1

def host():
    global _PRELAUNCH_PHASE
    _PRELAUNCH_PHASE = 'host.preflight'
    began = time.monotonic(); admission, pins = preflight()
    sys.path.insert(0, str(ROOT / 'api'))
    sys.path.insert(0, str(E))
    from diagnostic_process import _OwnedJob, _host_health, _error_detail, _timing_record
    _PRELAUNCH_PHASE = 'host.initial-resources'
    initial = _host_health(P)
    if time.monotonic() - began > 5 or pins != source_pins(): raise RuntimeError('Fresh prelaunch pins/admission budget failed')
    _PRELAUNCH_PHASE = 'host.reserve-output'
    _publication_call('output.mkdir-exclusive', lambda: OUT.mkdir(exist_ok=False), None, OUT)
    attempt = dict(CPU_only=True, maximum_control_pairs=1, base_commit=BASE, branch=BRANCH, source_pins=pins,
                   admission_sha256=sha(E / 'admission.json'), python_executable_sha256=admission['python_executable_sha256'],
                   initial_resources=initial, deadline_seconds=90, process_and_job_bytes=6 * 1024**3,
                   ram_floor_bytes=12 * 1024**3, disk_floor_bytes=2 * 1024**3, stage_timeout_s=25,
                   active_end=began + 72, grace_end=began + 80, force_end=began + 85, hard_end=began + 90)
    _PRELAUNCH_PHASE = 'host.attempt-publication'
    save(OUT / 'attempt.json', attempt)
    _PRELAUNCH_PHASE = 'host.execution'
    finished = threading.Event()
    def hard_stop():
        if not finished.wait(max(0, attempt['hard_end'] - time.monotonic())): os._exit(124)
    threading.Thread(target=hard_stop, daemon=True).start()  # Own kill-on-close Job enforces final containment.
    result = dict(classification='BLOCKED', initial_resources=initial, samples=[], errors=[], cleanup_errors=[], windows=None)
    result['timing'] = dict(anchor=dict(before_ns=time.monotonic_ns(), utc_ns=time.time_ns(), after_ns=time.monotonic_ns()), outer_snapshot={})
    job = child = None; attached = gate_open = False; phase = 'host.job-create'
    def error(exc, cleanup=False):
        detail = _error_detail(exc, phase)
        detail.update(stage_id='windows-controls-supervisor', during_shutdown=cleanup,
                      supervisor_job_owner=getattr(job, 'job_owner', None))
        result['cleanup_errors' if cleanup else 'errors'].append(detail)
        try: trace('host', 'cleanup-error' if cleanup else 'primary-error', error=detail)
        except BaseException as secondary: result['cleanup_errors'].append(_error_detail(secondary, 'host.error-trace'))
    def snapshot():
        snapshot_begin_ns = time.monotonic_ns()
        timing_phase = 'cleanup.snapshot' if '.cleanup.' in phase else 'normal.snapshot'
        try:
            job.phase = phase; result['windows'] = job.snapshot(); return result['windows']
        finally:
            _timing_record(result['timing']['outer_snapshot'], timing_phase, snapshot_begin_ns, time.monotonic_ns())
    try:
        job = _OwnedJob()
        with (OUT / 'worker.stdout.txt').open('xb') as stdout, (OUT / 'worker.stderr.txt').open('xb') as stderr:
            phase = 'host.Popen'
            child = subprocess.Popen([sys.executable, '-B', '-I', '-S', str(Path(__file__).resolve()), '--worker'], stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES=''), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            phase = 'host.attach'; job.phase = phase; job.attach(child); attached = True
            phase = 'host.runtime-ready'; until = min(time.monotonic() + 5, attempt['active_end'])
            while not (OUT / 'worker-ready.json').exists():
                if child.poll() is not None or time.monotonic() >= until: raise RuntimeError('Actual worker runtime readiness missing')
                time.sleep(.025)
            runtime_pid = read(OUT / 'worker-ready.json')['pid']; observed = snapshot()
            identities = [row for row in observed['process_identities'] if row['pid'] == runtime_pid and row.get('job_membership_verified') is True
                          and row.get('creation_filetime', 0) > 0 and row.get('observed_alive', {}).get('alive') is True]
            if runtime_pid not in observed['enumerated_pids'] or len(identities) != 1: raise RuntimeError('Actual worker runtime is not authenticated in outer Job; GO forbidden')
            trace('host', 'runtime-owned-before-GO', worker_ready=read(OUT / 'worker-ready.json'), observation=observed)
            phase = 'host.GO'; result['go_dispatch_begin_ns'] = time.monotonic_ns()
            child.stdin.write(b'GO\n'); child.stdin.close(); gate_open = True
            result['go_dispatch_return_ns'] = time.monotonic_ns()
            while child.poll() is None:
                phase = 'host.observe'
                if time.monotonic() >= attempt['active_end']: raise TimeoutError('Outer active deadline; reserved cleanup begins')
                result['samples'].append({'monotonic_s': time.monotonic(), **_host_health(P)}); snapshot(); time.sleep(.05)
            result['returncode'] = child.returncode; phase = 'host.normal-settle'; until = min(time.monotonic() + 1, attempt['active_end'])
            while snapshot()['active'] and time.monotonic() < until: time.sleep(.025)
            if snapshot()['active'] or child.returncode != 0 or not control_pair_pass(): raise RuntimeError('Raw Windows controls did not qualify')
            result['classification'] = 'PASS'
    except BaseException as exc: error(exc)
    finally:
        try:
            if child is not None and child.stdin is not None and not child.stdin.closed: child.stdin.close()
            phase = 'host.cleanup.cooperative-cancel'
            if child is not None and child.poll() is None and gate_open:
                save(OUT / 'host-cancel.json', {'action': 'CANCEL', 'attempt_sha256': sha(OUT / 'attempt.json')})
                trace('host', 'owned-cooperative-cancel'); until = min(time.monotonic() + 8, attempt['grace_end'])
                while child.poll() is None and time.monotonic() < until: time.sleep(.025)
        except BaseException as exc: error(exc, True)
        if job is not None:
            active = True
            try: phase = 'host.cleanup.observe'; active = snapshot()['active'] != 0
            except BaseException as exc: error(exc, True)
            if active:
                result['forced_outer_cleanup'] = True
                try:
                    phase = 'host.cleanup.terminate-owned-job'; job.phase = phase; job.terminate(); trace('host', 'owned-job-force', job_owner=job.job_owner)
                    until = min(time.monotonic() + 5, attempt['force_end'])
                    while time.monotonic() < until and snapshot()['active']: time.sleep(.025)
                except BaseException as exc: error(exc, True)
            try: phase = 'host.cleanup.final-observe'; snapshot()
            except BaseException as exc: error(exc, True)
            try: phase = 'host.cleanup.close-owned-job'; job.phase = phase; job.close()
            except BaseException as exc: error(exc, True)
        try:
            phase = 'host.cleanup.child-wait'
            if child is not None:
                if child.stdin is not None and not child.stdin.closed: child.stdin.close()
                child.wait(timeout=max(.001, min(2, attempt['hard_end'] - time.monotonic() - 2))); result['returncode'] = child.returncode
        except BaseException as exc: error(exc, True)
        try:
            phase = 'host.cleanup.sources-resources'; result['sources_unchanged'] = source_pins() == pins
            result['admission_unchanged'] = sha(E / 'admission.json') == attempt['admission_sha256']
            result['interpreter_unchanged'] = sha(sys.executable) == attempt['python_executable_sha256']; result['final_resources'] = _host_health(P)
        except BaseException as exc: error(exc, True)
        result['elapsed_s'] = time.monotonic() - began
        if (result['errors'] or result['cleanup_errors'] or result.get('forced_outer_cleanup') or not result.get('sources_unchanged') or not result.get('admission_unchanged') or not result.get('interpreter_unchanged')
                or result['elapsed_s'] >= 90 or not isinstance(result['windows'], dict) or result['windows'].get('active') != 0): result['classification'] = 'BLOCKED'
        save(OUT / 'host-result.json', result)
        trace('host', 'finished', classification=result['classification'], elapsed_s=result['elapsed_s'])
        print(json.dumps({key: result.get(key) for key in ('classification', 'elapsed_s', 'initial_resources', 'final_resources')})); finished.set()
    return 0 if result['classification'] == 'PASS' else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']: raise SystemExit(worker())
    elif sys.argv[1:] == ['--preflight']:
        preflight_entry(); print(json.dumps({'classification': 'STATIC_PREFLIGHT_PASS', 'maximum_control_pairs': 1}))
    elif not sys.argv[1:]:
        raise SystemExit(host_entry())
    else: raise SystemExit('Unsupported Windows diagnostic invocation')
