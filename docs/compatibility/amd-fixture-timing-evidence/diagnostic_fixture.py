"""Separate, synthetic-only timing copy of the historical CPU lifecycle fixture.

Source: cpu_stage_fixture.py at 0c031befef0483f4ac3bed05fa35c3546166f255,
SHA-256 cb808b5993c859a0ed62120109550059d267b1df2c47191d53a9753535c4b3f1.
The historical file and production supervision are not changed. This copy uses
integer monotonic_ns for the parent readiness timer instead of float monotonic;
the budget remains exactly five seconds, >= is inclusive, Popen is excluded,
and the original short-circuit readiness-before-timeout order is retained.

Run only in a fresh private root scratch directory under the authorized runner.
Timing sidecars are per-process and bounded. A completed publication checkpoint
can survive forced cleanup; an absent checkpoint/exit remains unknown. Sidecar
writes are deliberately separate from the fixture's exclusive/fsync/hardlink
protocol and never recursively instrument themselves. PID/PPID and self-reported
context are diagnostic labels; external Job/creation authentication is required.
"""
import time

# First executable measurement after importing the clock. UTC is a sequential
# anchor, not a simultaneous calibration or an interpreter/process birth time.
SCRIPT_ENTRY_MONOTONIC_NS = time.monotonic_ns()
SCRIPT_ENTRY_UTC_NS = time.time_ns()

import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib
import queue
import threading
import traceback


SOURCE_COMMIT = '0c031befef0483f4ac3bed05fa35c3546166f255'
SOURCE_PATH = 'docs/compatibility/amd-job-stage-orchestrator-evidence/cpu_stage_fixture.py'
SOURCE_SHA256 = 'cb808b5993c859a0ed62120109550059d267b1df2c47191d53a9753535c4b3f1'
EVENT_LIMIT = 96


class TimingTrace:
    def __init__(self, mode, base):
        self.base, self.mode = base, mode
        self.pid, self.ppid = os.getpid(), os.getppid()
        self.context = self.stage_id = None
        self.events, self.dropped_events = [], 0
        self.last_poll = self.readiness_timer = self.timeout_decision_ns = None
        self.poll_count = 0
        self.outcome, self.exception = 'running', None
        self.completed_checkpoint_count = 0
        self.completed_checkpoint_total_ns = self.completed_checkpoint_max_ns = 0
        self.current_checkpoint_begin_ns = None
        self.record('script-entry', at_ns=SCRIPT_ENTRY_MONOTONIC_NS)

    def record(self, event, at_ns=None, **details):
        row = dict(event=event, monotonic_ns=(time.monotonic_ns() if at_ns is None else at_ns),
                   **details)
        if len(self.events) < EVENT_LIMIT:
            self.events.append(row)
        else:
            self.dropped_events += 1

    def descendants_ready(self):
        names = ('child-ready.json', 'grandchild-ready.json')
        self.poll_count += 1
        poll = dict(begin_ns=time.monotonic_ns(),
                    states={name: None for name in names},
                    observed_ns={name: None for name in names})
        self.last_poll = poll
        # Preserve all(generator)'s original short-circuit filesystem accesses.
        for name in names:
            exists = (self.base / name).exists()
            poll['states'][name] = exists
            poll['observed_ns'][name] = time.monotonic_ns()
            if not exists:
                return False
        return True

    def snapshot(self, checkpoint):
        return dict(schema='modly.amd.fixture-timing.process.v1',
                    source=dict(commit=SOURCE_COMMIT, path=SOURCE_PATH, sha256=SOURCE_SHA256),
                    process=dict(pid=self.pid, ppid=self.ppid, mode=self.mode),
                    context=self.context, stage_id=self.stage_id,
                    clock_anchor=dict(monotonic_ns=SCRIPT_ENTRY_MONOTONIC_NS,
                                      utc_ns=SCRIPT_ENTRY_UTC_NS,
                                      reads='monotonic_ns then time_ns; sequential'),
                    checkpoint=checkpoint, event_limit=EVENT_LIMIT,
                    dropped_events=self.dropped_events, events=self.events,
                    readiness_timer=self.readiness_timer, poll_count=self.poll_count,
                    last_poll=self.last_poll, timeout_decision_ns=self.timeout_decision_ns,
                    persistence_overhead=dict(
                        completed_checkpoint_count=self.completed_checkpoint_count,
                        completed_total_ns=self.completed_checkpoint_total_ns,
                        completed_max_ns=self.completed_checkpoint_max_ns,
                        current_checkpoint_begin_ns=self.current_checkpoint_begin_ns,
                        current_checkpoint_elapsed_ns=None,
                        current_checkpoint_limit='tail not self-observable in this snapshot'),
                    outcome=self.outcome, exception=self.exception)

    def persist(self, checkpoint):
        # Own sidecars only. Fresh staging is exclusive; replacement is confined
        # to this process's summary. No readiness save() call or instrumentation.
        started_ns = time.monotonic_ns()
        self.current_checkpoint_begin_ns = started_ns
        path = self.base / ('fixture-timing-' + self.mode + '-' + str(self.pid) + '.json')
        staging = path.with_name(path.name + '.timing-writing')
        with staging.open('x', encoding='utf-8') as stream:
            json.dump(self.snapshot(checkpoint), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
        elapsed_ns = time.monotonic_ns() - started_ns
        self.completed_checkpoint_count += 1
        self.completed_checkpoint_total_ns += elapsed_ns
        self.completed_checkpoint_max_ns = max(self.completed_checkpoint_max_ns, elapsed_ns)


def save(path, row, trace=None):
    path = Path(path)
    staging = path.with_name(path.name + '.writing')
    if trace is not None: trace.record('staged-write-begin', name=path.name)
    with staging.open('x', encoding='utf-8') as stream:
        json.dump(row, stream)
        if trace is not None: trace.record('flush-begin', name=path.name)
        stream.flush()
        if trace is not None: trace.record('flush-end', name=path.name)
        if trace is not None: trace.record('fsync-begin', name=path.name)
        os.fsync(stream.fileno())
        if trace is not None: trace.record('fsync-end', name=path.name)
    if trace is not None: trace.record('hardlink-publication-begin', name=path.name)
    os.link(staging, path)
    if trace is not None: trace.record('hardlink-publication-end', name=path.name)
    staging.unlink()
    if trace is not None: trace.persist('published:' + path.name)


def fixture_main(mode, trace):
    context = json.loads(os.environ['MODLY_COMPLETION_CONTEXT_JSON'])
    stage = os.environ['MODLY_STAGE_ID']
    trace.context, trace.stage_id = context, stage
    base = trace.base
    script = str(Path(__file__).resolve())

    def stopped():
        if not (base / 'stop.json').exists(): return False
        row = json.loads((base / 'stop.json').read_text())
        if row != {'context': context, 'stage_id': stage}: raise RuntimeError('Foreign stop token')
        return True

    if mode == 'grandchild':
        save(base / 'grandchild-ready.json', {'pid': os.getpid(), 'context': context}, trace)
        deadline = time.monotonic() + 30
        while not stopped():
            if time.monotonic() >= deadline: raise TimeoutError('Fixture child deadline')
            time.sleep(.02)
        return
    if mode in ('child', 'daemonizer'):
        trace.record('popen-begin', role='grandchild')
        child = subprocess.Popen([sys.executable, '-B', '-I', '-S', script, 'grandchild'],
                                 start_new_session=mode == 'daemonizer', close_fds=True)
        trace.record('popen-end', role='grandchild', child_pid=child.pid)
        save(base / 'child-ready.json', {'pid': os.getpid(), 'grandchild': child.pid, 'context': context}, trace)
        if mode == 'child':
            if child.wait(timeout=32) != 0: raise RuntimeError('Grandchild failed')
        return
    if mode not in ('success', 'wait', 'sentinel'): raise ValueError('Exact fixture mode required')
    child_mode = 'daemonizer' if sys.platform == 'linux' else 'child'
    trace.record('popen-begin', role=child_mode)
    child = subprocess.Popen([sys.executable, '-B', '-I', '-S', script, child_mode], close_fds=True)
    trace.record('popen-end', role=child_mode, child_pid=child.pid)
    origin_ns = time.monotonic_ns()
    until_ns = origin_ns + 5_000_000_000
    trace.readiness_timer = dict(origin_ns=origin_ns, deadline_ns=until_ns, budget_ns=5_000_000_000)
    trace.record('readiness-timer', at_ns=origin_ns, deadline_ns=until_ns)
    while not trace.descendants_ready():
        decision_ns = time.monotonic_ns()
        if decision_ns >= until_ns:
            trace.timeout_decision_ns = decision_ns
            trace.record('readiness-timeout-decision', at_ns=decision_ns)
            raise TimeoutError('Fixture descendants did not start')
        time.sleep(.02)
    trace.record('readiness-observed')
    descendants = json.loads((base / 'child-ready.json').read_text())
    save(base / 'ready.json', {'pid': os.getpid(), 'context': context, 'stage_id': stage,
                             'descendants': descendants, 'synthetic_only': True}, trace)
    cancelled = False
    if mode == 'wait':
        message = json.loads(sys.stdin.buffer.readline(65537))
        if message != {'action': 'CANCEL', 'stage_id': stage, 'nonce': context['nonce']}:
            raise ValueError('Fixture requires exact cancellation')
        cancelled = True
    elif mode == 'sentinel':
        controls = queue.Queue()
        def read_cancel():
            try: controls.put(json.loads(sys.stdin.buffer.raw.readline(65537)))
            except Exception: controls.put(None)
        threading.Thread(target=read_cancel, daemon=True).start()
        until = time.monotonic() + 20
        while not (base / 'sentinel-release.json').exists():
            if time.monotonic() >= until: raise TimeoutError('Sentinel release deadline')
            try:
                message = controls.get_nowait()
                if message != {'action': 'CANCEL', 'stage_id': stage, 'nonce': context['nonce']}: raise RuntimeError('Foreign sentinel cancellation')
                cancelled = True
                break
            except queue.Empty: pass
            time.sleep(.02)
        if not cancelled and json.loads((base / 'sentinel-release.json').read_text()) != {'context': context, 'stage_id': stage}:
            raise RuntimeError('Foreign sentinel release')
    else:
        time.sleep(.25)
    save(base / 'stop.json', {'context': context, 'stage_id': stage})
    if child.wait(timeout=4) != 0: raise RuntimeError('Fixture child failed')
    # On Linux the daemonized grandchild is reaped by the broker, not this parent.
    if cancelled:
        save(os.environ['MODLY_STAGE_CANCEL_ACK'], dict(terminal='cancelled', context=context,
                                                       stage_id=stage, nonce=context['nonce']))
    if not cancelled and (base / 'stage-request.json').exists():
        request = json.loads((base / 'stage-request.json').read_text())
        output = base / 'output.json'
        save(output, {'product_context': context, 'synthetic_only': True, 'stage_id': stage})
        save(base / 'stage-result.json', dict(schema='modly.amd.stage-receipt.v1', mode='cpu_lifecycle_test',
            product_context=context, stage_id=stage, status='PASS', predecessor_sha256=request['predecessor_sha256'],
            inputs=request['inputs'], source_sha256=request['source_sha256'],
            outputs=[dict(relative_path='output.json', sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                          bytes=output.stat().st_size, kind='json', schema={})]))


def main(mode):
    trace = TimingTrace(mode, Path.cwd())
    active_exception = None
    try:
        fixture_main(mode, trace)
        trace.outcome = 'returned'
    except BaseException as error:
        active_exception = error
        trace.outcome = 'exception'
        trace.exception = dict(type=type(error).__name__, message=str(error)[:2048])
        trace.record('fixture-exception', exception_type=type(error).__name__)
        raise
    finally:
        # This is the last controlled Python-main boundary, not an authenticated
        # OS exit instant. Forced cleanup may prevent this event from existing.
        trace.record('process-exit', boundary='main-finally-before-interpreter-exit',
                     outcome=trace.outcome)
        try:
            trace.persist('finally')
        except BaseException as persistence_error:
            if active_exception is None:
                raise
            # Keep the fixture exception and its traceback as the raised error.
            # Persistence failure is explicit and fail-closed, never hidden.
            print('DIAGNOSTIC_TIMING_PERSISTENCE_FAILED; timing evidence incomplete', file=sys.stderr)
            traceback.print_exception(type(persistence_error), persistence_error,
                                      persistence_error.__traceback__, file=sys.stderr)
            if hasattr(active_exception, 'add_note'):
                active_exception.add_note('Diagnostic timing persistence failed; inspect stderr; evidence incomplete')


if __name__ == '__main__': main(sys.argv[1])
