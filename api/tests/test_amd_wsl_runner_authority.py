"""Actual bound runner with simulated OS boundaries; never native WSL proof."""
import copy
import io
import json
from pathlib import Path
import queue
import sys
import threading
import unittest
from unittest.mock import patch

from services import amd_stage_process as stage
from tests import test_amd_upstream_job as upstream_fixture


class WSLRunnerIntegrationTests(unittest.TestCase):
    """A runner must reject missing facts even if mutable diagnostics say PASS."""
    def prepare(self):
        fixture = upstream_fixture.UpstreamJobTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.job = fixture.job()
        self.ticket = self.job.issue('conditioning')
        self.runner = stage.OwnedStageRunner(fixture.root / 'wsl-runner')
        self.command = stage.StageCommand('conditioning', 'wsl',
            ('/usr/bin/python3', '-B', '-I', '-S', stage._guest_path(fixture.sources['policy'].path)),
            fixture.root, fixture.context.to_dict(), timeout_s=25)
        self.client_path = fixture.root / 'System32/wsl.exe'
        self.client_path.parent.mkdir()
        self.client_path.write_bytes(b'unit-test-only WSL client pin; never executable')
        self.clock = 100.0
        self.variant = None

    def setUp(self):
        from services import amd_windows_authority
        self.module = amd_windows_authority
        self.prepare()
        test = self

        def monotonic():
            self.clock += .001
            return self.clock

        class Pipe:
            def __init__(self):
                self.lines = queue.Queue()
                self.previous = None
                self.closed = threading.Event()

            def readline(self, limit):
                if self.previous is not None:
                    self.previous.set()
                line, self.previous = self.lines.get(timeout=2)
                return line

            def close(self):
                self.closed.set()

            def feed(self, line):
                consumed = threading.Event()
                self.lines.put((line, consumed))
                if not consumed.wait(2):
                    raise RuntimeError('Simulated private-pipe reader did not consume line')

            def finish(self):
                if not self.closed.is_set():
                    self.lines.put((b'', threading.Event()))
                    if not self.closed.wait(2):
                        raise RuntimeError('Simulated private pipe did not reach EOF')

        class Job:
            def __init__(self):
                self.job_owner, self.handle = 'unit-only-WSL-job', 17
                self.proc = None
                self.terminated = False
                test.native_job = self

            def attach(self, proc):
                self.proc = proc

            def snapshot(self):
                proc = self.proc
                pids = [101] + ([303] if proc.go else [])
                rows = []
                for pid in pids:
                    if pid == 303 and test.variant == 'missing-native-client':
                        continue
                    image = sys.executable if pid == 101 else str(test.client_path)
                    if pid == 303 and test.variant == 'wrong-native-client-image':
                        image = str(test.fixture.root / 'foreign-client.exe')
                    alive = proc.returncode is None if pid == 101 else not proc.client_exited
                    identity = dict(pid=pid, creation_filetime=1000 + pid, image=image,
                                    job_owner=self.job_owner, job_handle=self.handle)
                    if pid == 303 and test.variant == 'reused-native-client' and proc.terminal_emitted:
                        identity['creation_filetime'] += 1
                    rows.append(dict(identity, job_membership_verified=True,
                        identity_source='native-image-query', observed_alive={'alive': alive},
                        image_authentication=dict(identity, api='QueryFullProcessImageNameW',
                                                  succeeded=True, observed_utc='unit-only'),
                        exit_observation=dict(api='GetExitCodeProcess', succeeded=True,
                                              exit_code=259 if alive else 0)))
                    if pid == 303 and test.variant == 'reused-client-before-start':
                        # The cached expected image must not hide this different
                        # current incarnation of the same numeric PID.
                        reused = dict(identity, creation_filetime=2302,
                                      image=str(test.fixture.root / 'foreign-current-client.exe'))
                        rows.append(dict(reused, job_membership_verified=True,
                            identity_source='native-image-query', observed_alive={'alive': alive},
                            image_authentication=dict(reused, api='QueryFullProcessImageNameW',
                                                      succeeded=True, observed_utc='unit-only-current'),
                            exit_observation=dict(api='GetExitCodeProcess', succeeded=True,
                                                  exit_code=259 if alive else 0)))
                alive = proc.returncode is None or test.variant == 'forced-native-cleanup' and not self.terminated
                total = len(pids) + (test.variant in ('unobserved-native-descendant', 'reused-client-before-start'))
                enumerated = pids if alive else []
                if test.variant == 'stale-client-before-start' and proc.go and alive:
                    # A retained row is not a current Job-membership observation.
                    # Its old image-query and alive flags intentionally remain.
                    enumerated = [101]
                return dict(active=len(enumerated), total_processes=total,
                    peak_process_bytes=1024, peak_job_bytes=2048, process_identities=rows,
                    job_owner=self.job_owner, enumerated_pids=enumerated)

            def terminate(self):
                self.terminated = True
                self.proc.returncode = 1
                self.proc.client_exited = True
                self.proc.stdout.finish()

            def close(self):
                pass

        class Input(io.BytesIO):
            def write(self, blob):
                value = json.loads(blob)
                proc = test.process
                proc.sent_controls.append((value['action'], proc.terminal_emitted))
                if value['action'] == 'GO':
                    proc.go = True
                elif value['action'] == 'START':
                    if test.variant == 'exec-before-start-ack':
                        proc.exec_ack()
                    if test.variant != 'missing-start-ack':
                        extra = {'nonce': '0' * 32} if test.variant == 'wrong-nonce' else {}
                        proc.emit(event='control-received', action='START', **proc.linux, **extra)
                        if test.variant == 'duplicate-start-ack':
                            proc.emit(event='control-received', action='START', **proc.linux)
                    if test.variant not in ('missing-exec', 'exec-before-start-ack'):
                        proc.exec_ack()
                elif value['action'] == 'PING' and not proc.terminal_emitted:
                    proc.emit(event='control-received', action='PING', **proc.linux)
                elif value['action'] == 'CANCEL':
                    proc.returncode = 1
                    proc.client_exited = True
                    proc.stdout.finish()
                return super().write(blob)

        class Process:
            def __init__(self, argv, **kwargs):
                test.process = self
                self.spec = json.loads(Path(argv[-1]).read_bytes())
                self.stdout, self.stdin = Pipe(), Input()
                self.pid, self.returncode, self.go = 101, None, False
                self.client_exited = self.terminal_emitted = False
                self.polls = self.post_terminal_polls = 0
                self.sent_controls = []
                self.linux = dict(uid=1000, boot_id='11111111-2222-4333-8444-555555555555',
                    broker=dict(pid=401, start_ticks=9001, ppid=1, session=400, pgrp=400,
                                uid=1000, state='S', rss_pages=1),
                    leader=dict(pid=402, start_ticks=9002, ppid=401, session=402, pgrp=402,
                                uid=1000, state='R', rss_pages=1),
                    memory=dict(rlimit_as_bytes=6442450944, aggregate_rss_limit_bytes=6442450944,
                                aggregate_rss_peak_bytes=2048, atomic_cgroup_limit=False))
                test.addCleanup(self.stdout.finish)

            def emit(self, **value):
                row = dict(stage_id=self.spec['stage_id'], nonce=self.spec['nonce'], context=self.spec['context'])
                row.update(value)
                self.stdout.feed((json.dumps(row) + '\n').encode())

            def exec_ack(self):
                value = dict(event='leader-executed', **copy.deepcopy(self.linux),
                    executable='/usr/bin/python3.12', argv=list(self.spec['argv']),
                    rlimit_as_bytes=[6442450944, 6442450944])
                if test.variant == 'wrong-native-commandline':
                    value['argv'][-1] = '/tmp/foreign-policy.py'
                if test.variant == 'missing-native-commandline':
                    del value['argv']
                if test.variant == 'wrong-guest-executable':
                    value['executable'] = '/usr/bin/foreign-python'
                if test.variant == 'missing-native-executable':
                    del value['executable']
                if test.variant == 'wrong-exec-rlimit':
                    value['rlimit_as_bytes'] = [5368709120, 5368709120]
                if test.variant == 'missing-native-rlimit':
                    del value['rlimit_as_bytes']
                self.emit(**value)

            def terminal(self):
                value = dict(event='terminal', **copy.deepcopy(self.linux), terminal='succeeded',
                    returncode=0, natural=True, forced=False, cooperative_cancelled=False,
                    linux_settled=True, reaped_all=True, owned_remaining=[], errors=[],
                    process_identities=[dict(self.linux['leader'])],
                    reaped=[dict(pid=402, status=0, timestamp_ns=1234)])
                if test.variant == 'foreign-boot':
                    value['boot_id'] = 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee'
                elif test.variant == 'foreign-leader':
                    value['leader']['start_ticks'] += 1
                elif test.variant == 'foreign-broker':
                    value['broker']['pid'] = 999
                elif test.variant == 'missing-reap':
                    value['reaped'] = []
                elif test.variant == 'failed-leader-reap':
                    value['reaped'][0]['status'] = 256
                elif test.variant == 'owned-remains':
                    value['owned_remaining'] = [dict(self.linux['leader'])]
                elif test.variant == 'missing-linux-settlement':
                    value['linux_settled'] = False
                elif test.variant == 'forced-linux-cleanup':
                    value['forced'] = True
                elif test.variant == 'broker-errors':
                    value['errors'] = ['unit-only broker failure']
                elif test.variant == 'foreign-observed-identity':
                    value['process_identities'][0]['start_ticks'] += 1
                elif test.variant == 'linux-memory-over-cap':
                    value['memory']['aggregate_rss_peak_bytes'] = 6442450945
                return value

            def poll(self):
                if not self.go or self.returncode is not None:
                    return self.returncode
                self.polls += 1
                if self.polls == 1:
                    if test.variant == 'forged-diagnostic':
                        path = Path(self.spec['stdout_file']).with_name('protocol.jsonl')
                        events = [dict(event='control-received', action='GO'),
                            dict(event='ready', **self.linux),
                            dict(event='control-received', action='START', **self.linux),
                            dict(event='leader-executed', **self.linux, executable='/usr/bin/python3.12',
                                 argv=list(self.spec['argv']), rlimit_as_bytes=[6442450944, 6442450944]),
                            self.terminal(), dict(event='linux-client-exited', pid=303, returncode=0)]
                        with path.open('ab') as stream:
                            for event in events:
                                value = dict(stage_id=self.spec['stage_id'], nonce=self.spec['nonce'],
                                             context=self.spec['context'], **event)
                                stream.write((json.dumps(value) + '\n').encode())
                        self.returncode, self.client_exited = 0, True
                        self.stdout.finish()
                    else:
                        self.emit(event='control-received', action='GO')
                        self.emit(event='ready', **self.linux)
                        if test.variant == 'duplicate-ready':
                            self.emit(event='ready', **self.linux)
                        test.clock += .6  # A legitimate pre-terminal PING is due.
                    return self.returncode
                if self.polls == 2:
                    test.fixture.emit('conditioning')
                    self.emit(**self.terminal())
                    self.terminal_emitted = True
                    test.clock += .6  # A heartbeat would be due if not suppressed.
                    return None
                self.post_terminal_polls += 1
                if test.variant == 'stuck-after-terminal':
                    test.clock += 2
                    return None
                if test.variant == 'cancel-after-terminal':
                    test.fixture.cancel.set()
                    return None
                if test.variant == 'duplicate-terminal':
                    self.emit(**self.terminal())
                if test.variant == 'event-after-terminal':
                    self.emit(event='control-received', action='PING', **self.linux)
                if test.variant != 'missing-client-exit':
                    self.emit(event='linux-client-exited', pid=303,
                              returncode=7 if test.variant == 'nonzero-client' else 0)
                self.client_exited = True
                self.returncode = 7 if test.variant == 'nonzero-bootstrap' else 0
                if test.variant != 'missing-eof':
                    self.stdout.finish()
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode, self.client_exited = 1, True
                self.stdout.finish()

        for item in (patch.object(stage, '_OwnedJob', Job),
                     patch.object(stage, '_require_windows', lambda: None),
                     patch.object(stage, '_host_health', lambda _: dict(available_ram_bytes=13*1024**3,
                                                                       free_disk_bytes=3*1024**3)),
                     patch.object(stage.subprocess, 'Popen', Process),
                     patch.object(stage.time, 'monotonic', side_effect=monotonic),
                     patch.object(stage.time, 'sleep', lambda _: None),
                     patch.object(self.module, '_wsl_client_path', lambda: self.client_path, create=True)):
            item.start()
            self.addCleanup(item.stop)

    def run_once(self, variant=None):
        self.assertTrue(callable(getattr(self.job, 'bind_wsl', None)),
                        'Host-owned WSL invocation binding is missing')
        self.variant = variant
        token = self.job.bind_wsl(self.ticket, self.runner, self.command)
        return self.runner.run_bound(token)

    def refused(self, variant):
        result, authority = self.run_once(variant)
        self.assertIsNone(authority, (variant, result))
        self.assertTrue(self.job._blocked)
        self.assertFalse(any(action == 'PING' and after for action, after in self.process.sent_controls))
        return result

    def test_actual_bound_wsl_flow_mints_only_registered_settled_authority(self):
        result, authority = self.run_once()
        self.assertIsNotNone(authority, result.get('error') or result.get('diagnostic_errors'))
        self.assertEqual(result['terminal'], 'succeeded')
        self.assertEqual(result['bootstrap_returncode'], 0)
        self.assertEqual(result['linux_client_returncode'], 0)
        self.assertIs(result['linux_settled'], True)
        self.assertEqual(result['windows']['active'], 0)
        self.assertEqual(result['windows']['total_processes'], 2)
        self.assertEqual(self.process.post_terminal_polls, 1)
        self.assertTrue(self.process.stdout.closed.is_set())
        self.assertEqual(self.process.sent_controls, [('GO', False), ('START', False), ('PING', False)])
        self.assertIs(self.process.spec['upstream_ticket_bound'], True)
        self.assertEqual(self.process.spec['nonce'], self.ticket.nonce)
        self.assertNotEqual(self.ticket.nonce, self.fixture.context.to_dict()['nonce'])
        self.assertEqual(self.process.spec['context'], self.fixture.context.to_dict())
        # Mutable returned dictionaries cannot supply the authority's facts.
        result['terminal'], result['linux_settled'] = 'forged diagnostic', False
        result['windows']['active'] = 99
        pin = self.job.accept_wsl(self.ticket, self.runner, authority)
        document = json.loads(pin.path.read_bytes())
        self.assertEqual(document['qualification'], 'WSL_CPU_SUPERVISED')
        self.assertIs(document['genuine_execution_qualified'], False)
        evidence = document['wsl_supervision']['evidence']
        self.assertEqual(evidence['control_provenance'], 'private-bootstrap-stdout-pipe')
        self.assertIs(evidence['protocol_eof'], True)

    def test_stale_native_client_cannot_receive_start_or_mint_authority(self):
        result, authority = self.run_once('stale-client-before-start')
        self.assertFalse(any(action == 'START' for action, _ in self.process.sent_controls),
                         'START released the guest using a cached client absent from the current Job enumeration')
        self.assertIsNone(authority, result)
        self.assertTrue(self.job._blocked)

    def test_reused_native_client_pid_cannot_receive_start_or_mint_authority(self):
        result, authority = self.run_once('reused-client-before-start')
        self.assertFalse(any(action == 'START' for action, _ in self.process.sent_controls),
                         'START released the guest despite ambiguous CreationFILETIMEs for the client PID')
        self.assertIsNone(authority, result)
        self.assertTrue(self.job._blocked)

    def test_missing_or_unordered_start_and_exec_are_refused(self):
        for variant in ('missing-start-ack', 'duplicate-start-ack', 'exec-before-start-ack', 'missing-exec',
                        'wrong-nonce', 'duplicate-ready', 'wrong-native-commandline',
                        'missing-native-commandline', 'wrong-guest-executable', 'missing-native-executable',
                        'wrong-exec-rlimit', 'missing-native-rlimit'):
            with self.subTest(variant=variant):
                if self.variant is not None:
                    self.prepare()
                self.refused(variant)

    def test_foreign_linux_identity_or_unsettled_worker_is_refused(self):
        for variant in ('foreign-boot', 'foreign-leader', 'foreign-broker', 'missing-reap',
                        'failed-leader-reap', 'owned-remains', 'missing-linux-settlement',
                        'forced-linux-cleanup', 'broker-errors', 'foreign-observed-identity',
                        'linux-memory-over-cap'):
            with self.subTest(variant=variant):
                if self.variant is not None:
                    self.prepare()
                self.refused(variant)

    def test_windows_client_identity_and_actual_exit_codes_are_required(self):
        for variant in ('missing-native-client', 'wrong-native-client-image', 'reused-native-client',
                        'unobserved-native-descendant', 'nonzero-client', 'nonzero-bootstrap',
                        'missing-client-exit', 'forced-native-cleanup'):
            with self.subTest(variant=variant):
                if self.variant is not None:
                    self.prepare()
                self.refused(variant)

    def test_worker_written_diagnostic_success_is_not_private_pipe_authority(self):
        self.refused('forged-diagnostic')

    def test_terminal_wait_deadline_and_cancellation_remain_live(self):
        for variant in ('stuck-after-terminal', 'cancel-after-terminal'):
            with self.subTest(variant=variant):
                if self.variant is not None:
                    self.prepare()
                self.refused(variant)

    def test_duplicate_or_unexpected_private_protocol_after_terminal_is_refused(self):
        for variant in ('duplicate-terminal', 'event-after-terminal'):
            with self.subTest(variant=variant):
                if self.variant is not None:
                    self.prepare()
                self.refused(variant)

    def test_actual_bootstrap_exit_without_private_pipe_eof_is_refused(self):
        with patch.object(threading.Thread, 'join', lambda *args, **kwargs: None):
            result = self.refused('missing-eof')
        self.process.stdout.finish()
        self.assertIn('EOF', result['error']['message'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
