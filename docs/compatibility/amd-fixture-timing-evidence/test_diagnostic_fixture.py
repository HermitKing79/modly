"""Mocked diagnostic contracts: no process, Job, WSL or GPU is launched.

The in-memory filesystem retains the actual exclusive-write/fsync/hardlink
publication sequence. All time and subprocess boundaries are replaced.
"""
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
CONTEXT = {'nonce': 'synthetic-nonce', 'run': 'synthetic-run'}
STAGE = 'synthetic-stage'


class Clock:
    def __init__(self):
        self.now = 0
        self.on_sleep = None

    def monotonic_ns(self): return self.now
    def monotonic(self): return self.now / 1_000_000_000
    def time_ns(self): return 1_790_000_000_000_000_000 + self.now

    def sleep(self, seconds):
        if self.on_sleep is not None:
            callback, self.on_sleep = self.on_sleep, None
            callback()
        else:
            self.now += round(seconds * 1_000_000_000)


class MemoryStream(io.StringIO):
    def __init__(self, fs, name):
        super().__init__()
        self.fs, self.name = fs, name

    def fileno(self): return 71

    def flush(self):
        self.fs.operations.append(('flush', self.name))
        self.fs.clock.now += self.fs.flush_cost

    def close(self):
        if not self.closed:
            self.fs.files[self.name] = self.getvalue()
            self.fs.operations.append(('close', self.name))
        super().close()


class MemoryFS:
    def __init__(self, clock):
        self.clock, self.files, self.operations = clock, {}, []
        self.flush_cost = self.fsync_cost = self.link_cost = 0
        self.fail_link = None
        self.fail_replace = False
        self.on_link = None
        owner = self

        class MemoryPath:
            def __init__(self, value): self.value = str(value)
            @classmethod
            def cwd(cls): return cls('/scratch')
            @property
            def name(self): return PurePosixPath(self.value).name
            def __str__(self): return self.value
            def __truediv__(self, name): return MemoryPath(self.value + '/' + name)
            def with_name(self, name):
                return MemoryPath(str(PurePosixPath(self.value).with_name(name)))
            def resolve(self): return self
            def exists(self): return self.value in owner.files
            def read_text(self, *args, **kwargs): return owner.files[self.value]
            def read_bytes(self): return self.read_text().encode()
            def stat(self):
                return type('Stat', (), {'st_size': len(self.read_bytes())})()
            def open(self, mode, **kwargs):
                if mode == 'x' and self.value in owner.files:
                    raise FileExistsError(self.value)
                if mode not in ('x', 'w'): raise AssertionError(mode)
                owner.operations.append(('open-' + mode, self.value))
                owner.files[self.value] = ''
                return MemoryStream(owner, self.value)
            def unlink(self):
                owner.operations.append(('unlink', self.value))
                del owner.files[self.value]

        self.Path = MemoryPath

    def fsync(self, fd):
        if fd != 71: raise AssertionError('Unexpected real descriptor')
        self.operations.append(('fsync', fd))
        self.clock.now += self.fsync_cost

    def link(self, source, target):
        source, target = str(source), str(target)
        self.operations.append(('link', target))
        if self.fail_link == target: raise OSError('synthetic hardlink failure')
        if target in self.files: raise FileExistsError(target)
        self.clock.now += self.link_cost
        self.files[target] = self.files[source]
        if self.on_link is not None: self.on_link(target)

    def replace(self, source, target):
        source, target = str(source), str(target)
        self.operations.append(('replace', target))
        if self.fail_replace: raise OSError('synthetic timing persistence failure')
        self.files[target] = self.files.pop(source)

    def publish(self, name, row): self.files['/scratch/' + name] = json.dumps(row)
    def row(self, name): return json.loads(self.files['/scratch/' + name])


class DiagnosticContracts(unittest.TestCase):
    def setUp(self):
        path = HERE / 'diagnostic_fixture.py'
        self.assertTrue(path.exists(), 'The separate diagnostic fixture is not implemented')
        spec = importlib.util.spec_from_file_location('mocked_diagnostic_fixture', path)
        self.fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.fixture)
        self.clock = Clock()
        self.fs = MemoryFS(self.clock)
        self.spawn_delay = 0
        self.spawn_calls = []
        self.wait_calls = []
        self.child_returncode = 0

    def publish_ready(self):
        self.fs.publish('child-ready.json', {'pid': 202, 'grandchild': 303, 'context': CONTEXT})
        self.fs.publish('grandchild-ready.json', {'pid': 303, 'context': CONTEXT})

    def run_mode(self, mode, message=None):
        owner = self

        class Child:
            pid = 202
            def wait(self, timeout):
                owner.wait_calls.append(timeout)
                return owner.child_returncode

        def spawn(argv, **kwargs):
            self.spawn_calls.append((argv, kwargs))
            self.clock.now += self.spawn_delay
            return Child()

        class InputBuffer(io.BytesIO):
            @property
            def raw(self): return self

        class InlineThread:
            def __init__(self, target, daemon): self.target = target
            def start(self): self.target()

        stdin = type('Input', (), {'buffer': InputBuffer(
            b'' if message is None else json.dumps(message).encode() + b'\n')})()
        env = {'MODLY_COMPLETION_CONTEXT_JSON': json.dumps(CONTEXT),
               'MODLY_STAGE_ID': STAGE, 'MODLY_STAGE_CANCEL_ACK': '/scratch/cancel-ack.json'}
        with patch.object(self.fixture, 'Path', self.fs.Path), \
             patch.dict(os.environ, env), \
             patch.object(self.fixture.subprocess, 'Popen', side_effect=spawn), \
             patch.object(self.fixture.time, 'monotonic_ns', self.clock.monotonic_ns), \
             patch.object(self.fixture.time, 'monotonic', self.clock.monotonic), \
             patch.object(self.fixture.time, 'time_ns', self.clock.time_ns), \
             patch.object(self.fixture.time, 'sleep', self.clock.sleep), \
             patch.object(self.fixture.os, 'fsync', self.fs.fsync), \
             patch.object(self.fixture.os, 'link', self.fs.link), \
             patch.object(self.fixture.os, 'replace', self.fs.replace), \
             patch.object(self.fixture.os, 'getpid', return_value=101), \
             patch.object(self.fixture.os, 'getppid', return_value=99), \
             patch.object(self.fixture.threading, 'Thread', InlineThread), \
             patch.object(self.fixture.sys, 'stdin', stdin), \
             patch.object(self.fixture.sys, 'platform', 'win32'), \
             patch.object(self.fixture, 'SCRIPT_ENTRY_MONOTONIC_NS', 0), \
             patch.object(self.fixture, 'SCRIPT_ENTRY_UTC_NS', 1_790_000_000_000_000_000):
            self.fixture.main(mode)

    def summary(self, mode='success'):
        return self.fs.row('fixture-timing-' + mode + '-101.json')

    def events(self, summary): return {row['event']: row for row in summary['events']}

    def test_timeout_is_post_popen_five_seconds_and_durable_before_reraise(self):
        self.spawn_delay = 8_000_000_000
        self.clock.on_sleep = lambda: setattr(self.clock, 'now', 13_000_000_000)
        with self.assertRaisesRegex(TimeoutError, 'Fixture descendants did not start'):
            self.run_mode('success')
        report = self.summary()
        self.assertEqual(report['readiness_timer'], {
            'origin_ns': 8_000_000_000, 'deadline_ns': 13_000_000_000,
            'budget_ns': 5_000_000_000})
        self.assertEqual(report['timeout_decision_ns'], 13_000_000_000)
        self.assertEqual(report['last_poll']['states'], {'child-ready.json': False,
                                                        'grandchild-ready.json': None})
        self.assertEqual(report['last_poll']['observed_ns']['child-ready.json'], 13_000_000_000)
        self.assertIsNone(report['last_poll']['observed_ns']['grandchild-ready.json'])
        self.assertEqual(report['outcome'], 'exception')
        self.assertEqual(report['exception']['type'], 'TimeoutError')
        self.assertNotIn('/scratch/ready.json', self.fs.files)
        self.assertEqual(self.wait_calls, [])
        self.assertIn('process-exit', self.events(report))

    def test_only_child_marker_still_times_out_with_both_observed_states(self):
        self.fs.publish('child-ready.json', {'pid': 202, 'grandchild': 303, 'context': CONTEXT})
        self.clock.on_sleep = lambda: setattr(self.clock, 'now', 5_000_000_000)
        with self.assertRaises(TimeoutError): self.run_mode('success')
        self.assertEqual(self.summary()['last_poll']['states'], {
            'child-ready.json': True, 'grandchild-ready.json': False})

    def test_publication_seen_after_sleep_is_accepted_even_after_deadline(self):
        def overshoot():
            self.clock.now = 6_000_000_000
            self.publish_ready()
        self.clock.on_sleep = overshoot
        self.run_mode('success')
        report = self.summary()
        self.assertIsNone(report['timeout_decision_ns'])
        self.assertEqual(report['last_poll']['states'], {
            'child-ready.json': True, 'grandchild-ready.json': True})
        self.assertEqual(report['readiness_timer']['deadline_ns'], 5_000_000_000)
        self.assertEqual(self.fs.row('ready.json')['context'], CONTEXT)
        self.assertEqual(self.fs.row('stop.json'), {'context': CONTEXT, 'stage_id': STAGE})
        self.assertEqual(self.wait_calls, [4])

    def test_event_log_is_bounded_and_polling_never_writes_reports(self):
        with self.assertRaises(TimeoutError): self.run_mode('success')
        report_writes = [op for op in self.fs.operations if op[0] == 'replace']
        self.assertEqual(len(report_writes), 1)
        report = self.summary()
        self.assertLessEqual(len(report['events']), report['event_limit'])
        self.assertEqual(report['checkpoint'], 'finally')
        self.assertEqual(report['process']['pid'], 101)
        self.assertEqual(report['process']['ppid'], 99)
        self.assertEqual(report['context'], CONTEXT)
        self.assertEqual(report['stage_id'], STAGE)
        self.assertEqual(report['clock_anchor']['utc_ns'], 1_790_000_000_000_000_000)
        self.assertEqual(report['poll_count'], 251)

    def test_overflow_is_counted_without_unbounded_memory_or_invented_events(self):
        trace = self.fixture.TimingTrace('child', self.fs.Path('/scratch'))
        for number in range(200): trace.record('synthetic-overflow', number=number)
        self.assertEqual(len(trace.events), 96)
        self.assertEqual(trace.dropped_events, 105)
        self.assertEqual(trace.events[-1]['number'], 94)

    def test_missing_readiness_evidence_remains_null_rather_than_fabricated(self):
        self.run_mode('child')
        report = self.summary('child')
        self.assertIsNone(report['readiness_timer'])
        self.assertIsNone(report['timeout_decision_ns'])
        self.assertIsNone(report['last_poll'])
        self.assertNotIn('/scratch/grandchild-ready.json', self.fs.files)
        self.assertEqual(report['poll_count'], 0)

    def test_grandchild_checkpoint_survives_forced_cleanup_after_publication(self):
        # Simulate a killed descendant by examining its publication checkpoint
        # before the fake sleep raises; no real process is ever terminated.
        checkpoints = []
        def killed():
            checkpoints.append(self.summary('grandchild'))
            raise SystemExit('synthetic forced cleanup')
        self.clock.on_sleep = killed
        with self.assertRaises(SystemExit): self.run_mode('grandchild')
        checkpoint = checkpoints[0]
        self.assertEqual(checkpoint['checkpoint'], 'published:grandchild-ready.json')
        self.assertEqual(checkpoint['outcome'], 'running')
        self.assertIn('hardlink-publication-end', self.events(checkpoint))
        self.assertNotIn('process-exit', self.events(checkpoint))
        self.assertEqual(self.fs.row('grandchild-ready.json'), {'pid': 101, 'context': CONTEXT})

    def test_child_publication_retains_original_exclusive_fsync_link_protocol(self):
        self.fs.flush_cost, self.fs.fsync_cost, self.fs.link_cost = 2, 3, 5
        self.run_mode('child')
        report = self.summary('child')
        events = self.events(report)
        self.assertEqual(self.wait_calls, [32])
        self.assertEqual(self.spawn_calls[0][0][-1], 'grandchild')
        self.assertEqual(self.spawn_calls[0][1], {'start_new_session': False, 'close_fds': True})
        self.assertEqual(self.fs.row('child-ready.json'), {
            'pid': 101, 'grandchild': 202, 'context': CONTEXT})
        self.assertEqual(events['staged-write-begin']['monotonic_ns'], 0)
        self.assertEqual(events['flush-begin']['monotonic_ns'], 0)
        self.assertEqual(events['flush-end']['monotonic_ns'], 2)
        self.assertEqual(events['fsync-begin']['monotonic_ns'], 2)
        self.assertEqual(events['fsync-end']['monotonic_ns'], 5)
        self.assertIn('hardlink-publication-begin', events)
        self.assertEqual(events['hardlink-publication-begin']['monotonic_ns'], 5)
        self.assertEqual(events['hardlink-publication-end']['monotonic_ns'], 10)
        self.assertEqual([op[0] for op in self.fs.operations[:6]],
                         ['open-x', 'flush', 'fsync', 'close', 'link', 'unlink'])
        self.assertEqual(len([op for op in self.fs.operations if op[0] == 'replace']), 2)

    def test_final_summary_reports_prior_checkpoint_cost_and_current_tail_unknown(self):
        self.fs.flush_cost, self.fs.fsync_cost, self.fs.link_cost = 2, 3, 5
        self.run_mode('child')
        self.assertIn('persistence_overhead', self.summary('child'))
        overhead = self.summary('child')['persistence_overhead']
        self.assertEqual(overhead['completed_checkpoint_count'], 1)
        self.assertEqual(overhead['completed_total_ns'], 5)
        self.assertEqual(overhead['completed_max_ns'], 5)
        self.assertEqual(overhead['current_checkpoint_begin_ns'], 15)
        self.assertIsNone(overhead['current_checkpoint_elapsed_ns'])
        self.assertEqual(overhead['current_checkpoint_limit'], 'tail not self-observable in this snapshot')

    def test_cancel_ack_requires_exact_context_nonce_and_stage(self):
        self.publish_ready()
        self.run_mode('wait', {'action': 'CANCEL', 'stage_id': STAGE,
                               'nonce': 'synthetic-nonce'})
        self.assertEqual(self.fs.row('cancel-ack.json'), {
            'terminal': 'cancelled', 'context': CONTEXT, 'stage_id': STAGE,
            'nonce': 'synthetic-nonce'})

    def test_foreign_cancel_is_rejected_and_final_error_evidence_remains(self):
        self.publish_ready()
        with self.assertRaisesRegex(ValueError, 'Fixture requires exact cancellation'):
            self.run_mode('wait', {'action': 'CANCEL', 'stage_id': STAGE, 'nonce': 'foreign'})
        self.assertEqual(self.summary('wait')['exception']['type'], 'ValueError')
        self.assertNotIn('/scratch/cancel-ack.json', self.fs.files)
        self.assertNotIn('/scratch/stop.json', self.fs.files)

    def test_sentinel_release_keeps_exact_context_validation(self):
        self.publish_ready()
        self.fs.publish('sentinel-release.json', {'context': CONTEXT, 'stage_id': STAGE})
        self.run_mode('sentinel')
        self.assertEqual(self.summary('sentinel')['outcome'], 'returned')
        self.assertNotIn('/scratch/cancel-ack.json', self.fs.files)

    def test_foreign_sentinel_release_is_rejected(self):
        self.publish_ready()
        self.fs.publish('sentinel-release.json', {'context': {'nonce': 'foreign'}, 'stage_id': STAGE})
        with self.assertRaisesRegex(RuntimeError, 'Foreign sentinel release'):
            self.run_mode('sentinel')
        self.assertNotIn('/scratch/stop.json', self.fs.files)

    def test_sentinel_cancel_preserves_exact_nonce_protocol(self):
        self.publish_ready()
        self.run_mode('sentinel', {'action': 'CANCEL', 'stage_id': STAGE,
                                   'nonce': 'synthetic-nonce'})
        self.assertEqual(self.fs.row('cancel-ack.json'), {
            'terminal': 'cancelled', 'context': CONTEXT, 'stage_id': STAGE,
            'nonce': 'synthetic-nonce'})

    def test_sentinel_rejects_foreign_nonce_before_stop_and_ack(self):
        self.publish_ready()
        with self.assertRaisesRegex(RuntimeError, 'Foreign sentinel cancellation'):
            self.run_mode('sentinel', {'action': 'CANCEL', 'stage_id': STAGE, 'nonce': 'foreign'})
        self.assertNotIn('/scratch/stop.json', self.fs.files)
        self.assertNotIn('/scratch/cancel-ack.json', self.fs.files)

    def test_readiness_hardlink_failure_is_never_relabelled_as_publication(self):
        self.fs.fail_link = '/scratch/child-ready.json'
        with self.assertRaisesRegex(OSError, 'synthetic hardlink failure'): self.run_mode('child')
        report = self.summary('child')
        self.assertNotIn('hardlink-publication-end', self.events(report))
        self.assertEqual(report['exception']['type'], 'OSError')
        self.assertNotIn('/scratch/child-ready.json', self.fs.files)

    def test_timing_persistence_error_is_not_suppressed(self):
        self.fs.fail_replace = True
        self.publish_ready()
        errors = io.StringIO()
        with patch.object(self.fixture.sys, 'stderr', errors):
            with self.assertRaisesRegex(OSError, 'synthetic timing persistence failure'):
                self.run_mode('success')
        self.assertIn('DIAGNOSTIC_TIMING_PERSISTENCE_FAILED', errors.getvalue())

    def test_final_persistence_failure_preserves_original_timeout_and_reports_both(self):
        self.fs.fail_replace = True
        self.clock.on_sleep = lambda: setattr(self.clock, 'now', 5_000_000_000)
        errors = io.StringIO()
        with patch.object(self.fixture.sys, 'stderr', errors):
            with self.assertRaisesRegex(TimeoutError, 'Fixture descendants did not start'):
                self.run_mode('success')
        self.assertIn('DIAGNOSTIC_TIMING_PERSISTENCE_FAILED', errors.getvalue())
        self.assertIn('synthetic timing persistence failure', errors.getvalue())
        self.assertNotIn('/scratch/fixture-timing-success-101.json', self.fs.files)


if __name__ == '__main__': unittest.main()
