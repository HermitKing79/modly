"""Private-registry unit simulations only; no native execution qualification."""
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
from tests.test_amd_upstream_job import UpstreamJobTests

MODULE = Path(__file__).resolve().parents[1] / 'services/amd_windows_authority.py'


class WindowsAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.is_file(), 'Private Windows invocation registry is missing')
        from services import amd_windows_authority
        self.module = amd_windows_authority
        self.prepare()
        self.addCleanup(patch.stopall)
        patch.object(stage, '_OwnedJob', side_effect=AssertionError('No native Job in registry unit checks')).start()
        patch.object(stage.subprocess, 'Popen', side_effect=AssertionError('No process in registry unit checks')).start()

    def prepare(self):
        fixture = UpstreamJobTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.job = fixture.job()
        self.ticket = self.job.issue('conditioning')
        self.runner = stage.OwnedStageRunner(fixture.root / 'runner')
        self.command = stage.StageCommand('conditioning', 'windows',
            (sys.executable, '-B', '-I', '-S', str(fixture.sources['policy'].path)),
            fixture.root, fixture.context.to_dict(), timeout_s=25)

    def bind(self):
        token = self.job.bind_windows(self.ticket, self.runner, self.command)
        return token, self.runner._bindings[token]

    def simulated_seal(self):
        """Explicitly call the private seal: tests must never claim native PASS."""
        token, invocation = self.bind()
        invocation.start(self.runner)
        self.fixture.emit('conditioning')
        path = self.runner.evidence_dir / 'conditioning/receipt.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'unit_test_simulation_only': True}))
        invocation.seal(dict(terminal='succeeded', returncode=0, bootstrap_returncode=0,
            natural=True, settled=True, forced=False, diagnostic_errors=0,
            receipt_path=str(path), unit_test_simulation_only=True))
        return token, invocation, path

    def test_binding_uses_ticket_control_nonce_and_detached_original_context(self):
        token, invocation = self.bind()
        self.assertIs(type(token), object)
        command, cancel = invocation.start(self.runner)
        self.assertIs(cancel, self.fixture.cancel)
        self.assertIsNot(command, self.command)
        self.assertEqual(command.context, self.fixture.context.to_dict())
        self.assertEqual(command.control_nonce, self.ticket.nonce)
        self.assertNotEqual(command.control_nonce, command.context['nonce'])
        with self.assertRaises(ValueError):
            invocation.start(self.runner)
        self.assertTrue(self.job._blocked)

    def test_legacy_observation_cannot_accept_a_windows_bound_ticket(self):
        self.bind()
        self.fixture.emit('conditioning')
        with self.assertRaises(ValueError):
            self.job.accept(self.ticket, self.fixture.observe(self.ticket))

    def test_private_simulated_seal_is_single_use_with_explicit_cpu_scope(self):
        _, invocation, path = self.simulated_seal()
        authority = invocation.authority
        receipt_pin = self.job.accept_windows(self.ticket, self.runner, authority)
        document = json.loads(receipt_pin.path.read_bytes())
        self.assertEqual(document['qualification'], 'WINDOWS_CPU_SUPERVISED')
        self.assertIs(document['genuine_execution_qualified'], False)
        self.assertEqual(document['runner_receipt']['path'], str(path))
        self.assertTrue(document['windows_supervision']['evidence']['unit_test_simulation_only'])
        with self.assertRaises(ValueError):
            self.job.accept_windows(self.ticket, self.runner, authority)

    def test_unregistered_copied_and_json_authority_do_not_accept(self):
        for kind in ('object', 'copied', 'json'):
            with self.subTest(kind=kind):
                if kind != 'object':
                    self.prepare()
                _, invocation, _ = self.simulated_seal()
                value = {'object': object(), 'copied': copy.copy(invocation.authority),
                         'json': {'terminal': 'succeeded', 'natural': True}}[kind]
                with self.assertRaises(ValueError):
                    self.job.accept_windows(self.ticket, self.runner, value)

    def test_cross_runner_acceptance_refused(self):
        _, invocation, _ = self.simulated_seal()
        other = stage.OwnedStageRunner(self.fixture.root / 'other-runner')
        with self.assertRaises(ValueError):
            self.job.accept_windows(self.ticket, other, invocation.authority)

    def test_copied_ticket_cannot_consume_rightful_registered_authority(self):
        _, invocation, _ = self.simulated_seal()
        authority = invocation.authority
        with self.assertRaises(ValueError):
            self.module.consume(authority, self.job, copy.copy(self.ticket), self.runner)
        pin = self.job.accept_windows(self.ticket, self.runner, authority)
        self.assertEqual(json.loads(pin.path.read_bytes())['qualification'], 'WINDOWS_CPU_SUPERVISED')

    def test_changed_original_command_is_not_an_invocation(self):
        _, invocation = self.bind()
        self.command.context['request_id'] = 'changed'
        with self.assertRaises(ValueError):
            invocation.start(self.runner)
        self.assertIsNone(invocation.authority)

    def test_changed_runner_directory_is_not_the_bound_invocation(self):
        _, invocation = self.bind()
        self.runner.evidence_dir = self.fixture.root / 'changed-runner'
        with self.assertRaises(ValueError):
            invocation.start(self.runner)

    def test_changed_interpreter_selection_is_not_the_bound_invocation(self):
        _, invocation = self.bind()
        with patch.object(sys, 'executable', str(self.fixture.root / 'different-python.exe')):
            with self.assertRaises(ValueError):
                invocation.start(self.runner)

    def test_changed_entry_source_before_start_blocks_permanently(self):
        _, invocation = self.bind()
        source = self.fixture.sources['policy'].path
        original = source.read_bytes()
        source.write_bytes(original + b'changed')
        with self.assertRaises(ValueError):
            invocation.start(self.runner)
        source.write_bytes(original)
        with self.assertRaises(ValueError):
            invocation.start(self.runner)

    def test_changed_runner_receipt_cannot_be_accepted(self):
        _, invocation, path = self.simulated_seal()
        path.write_bytes(b'changed after settlement')
        with self.assertRaises(ValueError):
            self.job.accept_windows(self.ticket, self.runner, invocation.authority)

    def test_cancellation_after_seal_is_sticky_even_after_event_clears(self):
        _, invocation, _ = self.simulated_seal()
        self.fixture.cancel.set()
        with self.assertRaises((ValueError, InterruptedError)):
            self.job.accept_windows(self.ticket, self.runner, invocation.authority)
        self.fixture.cancel.clear()
        with self.assertRaises((ValueError, InterruptedError)):
            invocation.recheck()

    def test_expiry_after_binding_and_before_acceptance_is_sticky(self):
        with patch.object(self.module.time, 'monotonic', return_value=100):
            _, invocation = self.bind()
        with patch.object(self.module.time, 'monotonic', return_value=126):
            with self.assertRaises(ValueError):
                invocation.start(self.runner)
        self.prepare()
        with patch.object(self.module.time, 'monotonic', return_value=100):
            _, invocation, _ = self.simulated_seal()
        with patch.object(self.module.time, 'monotonic', return_value=126):
            with self.assertRaises(ValueError):
                self.job.accept_windows(self.ticket, self.runner, invocation.authority)

    def test_wsl_foreign_context_entrypoint_and_missing_cancel_refused(self):
        for kind in ('wsl', 'context', 'entrypoint', 'cancel'):
            with self.subTest(kind=kind):
                if kind != 'wsl':
                    self.prepare()
                if kind == 'wsl':
                    self.command = stage.StageCommand('conditioning', 'wsl', self.command.argv,
                        self.command.cwd, self.command.context, timeout_s=25)
                elif kind == 'context':
                    self.command.context['nonce'] = 'foreign'
                elif kind == 'entrypoint':
                    path = self.fixture.root / 'unregistered.py'; path.write_text('# unregistered')
                    self.command = stage.StageCommand('conditioning', 'windows',
                        (*self.command.argv[:4], str(path)), self.command.cwd, self.command.context, timeout_s=25)
                else:
                    self.job._cancel_event = None
                with self.assertRaises(ValueError):
                    self.bind()

    def test_failure_cannot_leave_usable_authority(self):
        _, invocation, _ = self.simulated_seal()
        authority = invocation.authority
        invocation.fail()
        self.assertIsNone(invocation.authority)
        with self.assertRaises(ValueError):
            self.job.accept_windows(self.ticket, self.runner, authority)

    def test_windows_stages_never_upgrade_final_shape_to_neural(self):
        _, invocation, _ = self.simulated_seal()
        self.job.accept_windows(self.ticket, self.runner, invocation.authority)
        for stage_id in ('occupancy', 'shape'):
            ticket = self.job.issue(stage_id)
            self.fixture.emit(stage_id)
            self.job.accept(ticket, self.fixture.observe(ticket))
        complete = json.loads(self.job.complete().receipt_path.read_bytes())
        self.assertEqual(complete['qualification'], 'CPU_CONTRACT_ONLY')
        self.assertIs(complete['genuine_execution_qualified'], False)


class WindowsRunnerIntegrationTests(unittest.TestCase):
    """Run the actual runner with simulated OS boundaries; no native verdict."""
    prepare = WindowsAuthorityTests.prepare
    bind = WindowsAuthorityTests.bind

    def setUp(self):
        from services import amd_windows_authority
        self.module = amd_windows_authority
        self.prepare()
        test = self
        self.variant = None
        self.clock = 100.0

        def monotonic():
            self.clock += .001
            return self.clock

        class SimulatedProtocol:
            """Feed only the runner's pipe; await reader consumption deterministically."""
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
                    raise RuntimeError('Simulated protocol reader did not consume line')

            def finish(self):
                if not self.closed.is_set():
                    self.lines.put((b'', threading.Event()))
                    if not self.closed.wait(2):
                        raise RuntimeError('Simulated protocol did not reach EOF')

        class SimulatedJob:
            def __init__(self):
                self.job_owner = 'unit-test-only-native-job'
                self.handle = 17
                self.proc = None
                self.phase = 'unit'
                self.terminated = False
                test.native_job = self

            def attach(self, proc):
                self.proc = proc

            def snapshot(self):
                proc = self.proc
                pids = [101] + ([202] if proc.go else [])
                rows = []
                for pid in pids:
                    if test.variant == 'missing-child' and pid == 202:
                        continue
                    identity = dict(pid=pid, creation_filetime=1000+pid, image=sys.executable,
                                    job_owner=self.job_owner, job_handle=self.handle)
                    rows.append(dict(identity, job_membership_verified=True, identity_source='native-image-query',
                        observed_alive={'alive': proc.returncode is None},
                        image_authentication=dict(identity, api='QueryFullProcessImageNameW', succeeded=True,
                                                  observed_utc='unit-test-only'),
                        exit_observation=dict(api='GetExitCodeProcess', succeeded=True,
                                              exit_code=259 if proc.returncode is None else 0)))
                alive = proc.returncode is None or (test.variant == 'post-terminal/forced-cleanup' and not self.terminated)
                total = len(pids) + (1 if test.variant == 'post-terminal/incomplete-ownership' else 0)
                return dict(active=len(pids) if alive else 0, total_processes=total,
                    peak_process_bytes=1024, peak_job_bytes=2048, process_identities=rows,
                    job_owner=self.job_owner, enumerated_pids=pids if alive else [])

            def terminate(self):
                self.terminated = True
                self.proc.returncode = 1

            def close(self):
                if test.variant == 'close-error':
                    raise OSError('unit-test-only close failure')

        class SimulatedInput(io.BytesIO):
            def write(self, blob):
                value = json.loads(blob)
                test.process.sent_controls.append((value['action'], test.process.terminal_emitted))
                if value['action'] == 'GO':
                    test.process.go = True
                if value['action'] == 'CANCEL':
                    test.process.returncode = 0
                    test.process.stdout.finish()
                if value['action'] == 'PING' and not test.process.terminal_emitted:
                    test.process.emit(event='control-received', action='PING')
                return super().write(blob)

        class SimulatedProcess:
            def __init__(self, argv, **kwargs):
                test.process = self
                self.spec = json.loads(Path(argv[-1]).read_bytes())
                self.stdout = SimulatedProtocol()
                self.pid = 101
                self.returncode = None
                self.go = False
                self.polls = 0
                self.stdin = SimulatedInput()
                self.sent_controls = []
                self.terminal_emitted = False
                self.post_terminal_polls = 0

            def emit(self, **value):
                row = dict(stage_id=self.spec['stage_id'], nonce=self.spec['nonce'],
                           context=self.spec['context'], **value)
                self.stdout.feed((json.dumps(row)+'\n').encode())

            def poll(self):
                if not self.go or self.returncode is not None:
                    return self.returncode
                self.polls += 1
                if test.variant and test.variant.startswith('post-terminal/'):
                    return self.post_terminal_poll()
                if self.polls == 1:
                    if test.variant == 'forged-mirror':
                        events = [dict(event='control-received', action='GO'),
                                  dict(event='stage-started', pid=202),
                                  dict(event='child-exited', child_pid=202, returncode=0,
                                       forced=False, cancellation_sent=False),
                                  dict(event='terminal', terminal='succeeded', returncode=0,
                                       natural=True, forced=False, cooperative_cancelled=False)]
                        diagnostic = Path(self.spec['stdout_file']).with_name('protocol.jsonl')
                        with diagnostic.open('ab') as stream:
                            for event in events:
                                row = dict(stage_id=self.spec['stage_id'], nonce=self.spec['nonce'],
                                           context=self.spec['context'], **event)
                                stream.write((json.dumps(row)+'\n').encode())
                        self.returncode = 0
                        self.stdout.finish()
                    elif test.variant == 'terminal-only':
                        self.emit(event='terminal', terminal='succeeded', returncode=0,
                                  natural=True, forced=False, cooperative_cancelled=False)
                        self.returncode = 0
                        self.stdout.finish()
                    else:
                        self.emit(event='control-received', action='GO')
                        self.emit(event='stage-started', pid=202)
                        if test.variant == 'protocol-error':
                            self.emit(event='control-error', operation='unit-only')
                else:
                    test.fixture.emit('conditioning')
                    self.emit(event='child-exited', child_pid=202, returncode=0,
                              forced=test.variant == 'forced', cancellation_sent=False)
                    self.emit(event='terminal', terminal='succeeded', returncode=0,
                              natural=True, forced=False, cooperative_cancelled=False)
                    self.returncode = 0
                    self.stdout.finish()
                    if test.variant == 'late':
                        test.clock = 126.0
                    if test.variant == 'cancel-on-exit':
                        test.fixture.cancel.set()
                return self.returncode

            def post_terminal_poll(self):
                """Terminal pipe message and actual process exit are separate facts."""
                variant = test.variant.split('/', 1)[1]
                if self.polls == 1:
                    self.emit(event='control-received', action='GO')
                    self.emit(event='stage-started', pid=202)
                    test.clock += .6  # Ordinary pre-terminal heartbeat is due.
                    return None
                if self.polls == 2:
                    test.fixture.emit('conditioning')
                    self.emit(event='child-exited', child_pid=202, returncode=0,
                              forced=False, cancellation_sent=False)
                    self.emit(event='terminal', terminal='succeeded', returncode=0,
                              natural=True, forced=False, cooperative_cancelled=False)
                    self.terminal_emitted = True
                    test.clock += .6  # A heartbeat is due while poll still says None.
                    return None
                self.post_terminal_polls += 1
                if variant == 'never-exits':
                    test.clock += 2
                    return None
                if variant == 'cancel-before-exit' and self.post_terminal_polls == 1:
                    test.fixture.cancel.set()
                    return None
                if variant == 'unexpected-event':
                    self.emit(event='control-received', action='PING')
                if variant == 'duplicate-terminal':
                    self.emit(event='terminal', terminal='succeeded', returncode=0,
                              natural=True, forced=False, cooperative_cancelled=False)
                if variant == 'changed-source':
                    path = test.fixture.sources['policy'].path
                    path.write_bytes(path.read_bytes() + b'changed during terminal wait')
                if variant == 'missing-source':
                    test.fixture.sources['policy'].path.unlink()
                self.returncode = 7 if variant == 'nonzero-bootstrap' else 0
                if variant != 'missing-eof':
                    self.stdout.finish()
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = 1
                self.stdout.finish()

        for item in (patch.object(stage, '_OwnedJob', SimulatedJob),
                     patch.object(stage, '_require_windows', lambda: None),
                     patch.object(stage, '_host_health', lambda _: dict(available_ram_bytes=13*1024**3,
                                                                       free_disk_bytes=3*1024**3)),
                     patch.object(stage.subprocess, 'Popen', SimulatedProcess),
                     patch.object(stage.time, 'monotonic', side_effect=monotonic),
                     patch.object(stage.time, 'sleep', lambda _: None)):
            item.start()
            self.addCleanup(item.stop)

    def run_once(self, variant=None):
        self.variant = variant
        token, _ = self.bind()
        return self.runner.run_bound(token)

    def test_actual_runner_flow_returns_registered_authority_after_local_settlement(self):
        result, authority = self.run_once()
        self.assertIsNotNone(authority)
        self.assertEqual(result['terminal'], 'succeeded')
        # Mutable diagnostics are not the authority's source of truth.
        result['terminal'] = 'changed diagnostic'
        result['windows']['active'] = 99
        pin = self.job.accept_windows(self.ticket, self.runner, authority)
        self.assertEqual(json.loads(pin.path.read_bytes())['qualification'], 'WINDOWS_CPU_SUPERVISED')

    def test_terminal_only_mutable_success_cannot_mint_authority(self):
        _, authority = self.run_once('terminal-only')
        self.assertIsNone(authority)
        self.assertTrue(self.job._blocked)

    def test_complete_forged_protocol_file_is_not_the_private_bootstrap_pipe(self):
        _, authority = self.run_once('forged-mirror')
        self.assertIsNone(authority)
        self.assertTrue(self.job._blocked)

    def test_protocol_error_cannot_mint_authority(self):
        _, authority = self.run_once('protocol-error')
        self.assertIsNone(authority)

    def test_missing_native_child_membership_cannot_mint_authority(self):
        _, authority = self.run_once('missing-child')
        self.assertIsNone(authority)

    def test_forced_child_exit_cannot_mint_authority(self):
        _, authority = self.run_once('forced')
        self.assertIsNone(authority)

    def test_close_error_cannot_mint_authority(self):
        _, authority = self.run_once('close-error')
        self.assertIsNone(authority)

    def test_late_successful_actual_exit_cannot_mint_authority(self):
        _, authority = self.run_once('late')
        self.assertIsNone(authority)

    def test_cancellation_at_actual_exit_is_sticky(self):
        _, authority = self.run_once('cancel-on-exit')
        self.assertIsNone(authority)
        self.fixture.cancel.clear()
        with self.assertRaises(ValueError):
            self.job.accept_windows(self.ticket, self.runner, object())


class TerminalHeartbeatIntegrationTests(unittest.TestCase):
    """Actual bound runner, mocked OS; terminal receipt never substitutes for exit."""
    prepare = WindowsAuthorityTests.prepare
    bind = WindowsAuthorityTests.bind
    setUp = WindowsRunnerIntegrationTests.setUp
    run_once = WindowsRunnerIntegrationTests.run_once

    def refused(self, variant):
        result, authority = self.run_once('post-terminal/' + variant)
        self.assertIsNone(authority)
        self.assertTrue(self.job._blocked)
        self.assertFalse(any(action == 'PING' and after for action, after in self.process.sent_controls))
        return result

    def test_terminal_before_actual_exit_suppresses_due_ping_and_seals_only_after_exit(self):
        result, authority = self.run_once('post-terminal/success')
        self.assertIsNotNone(authority, result.get('error'))
        self.assertEqual(result['terminal'], 'succeeded')
        self.assertEqual(result['bootstrap_returncode'], 0)
        self.assertEqual(self.process.post_terminal_polls, 1)
        self.assertTrue(self.process.stdout.closed.is_set())
        self.assertEqual(self.process.sent_controls, [('GO', False), ('PING', False)])
        self.assertIs(self.process.spec['upstream_ticket_bound'], True)
        self.assertEqual(self.process.spec['nonce'], self.ticket.nonce)
        self.assertEqual(self.process.spec['context'], self.fixture.context.to_dict())
        self.assertEqual(self.process.spec['memory_limit'], 6 * 1024**3)
        self.assertEqual(self.process.spec['timeout_s'], 25)
        self.assertEqual(result['windows']['active'], 0)
        self.assertFalse(result['forced'])
        pin = self.job.accept_windows(self.ticket, self.runner, authority)
        document = json.loads(pin.path.read_bytes())
        self.assertEqual(document['qualification'], 'WINDOWS_CPU_SUPERVISED')
        self.assertIs(document['genuine_execution_qualified'], False)
        self.assertEqual(document['windows_supervision']['evidence']['control_provenance'],
                         'private-bootstrap-stdout-pipe')
        self.assertIs(document['windows_supervision']['evidence']['protocol_eof'], True)
        self.assertEqual(document['windows_supervision']['binding']['resource_policy'],
                         dict(windows_process_bytes=6*1024**3, windows_job_bytes=6*1024**3,
                              available_ram_floor_bytes=12*1024**3,
                              free_disk_floor_bytes=2*1024**3, timeout_s=25))

    def test_terminal_success_with_bootstrap_still_running_expires(self):
        result = self.refused('never-exits')
        self.assertIn('deadline', result['error']['message'].lower())

    def test_cancellation_after_terminal_before_actual_exit_is_sticky(self):
        self.refused('cancel-before-exit')
        self.fixture.cancel.clear()
        with self.assertRaises(ValueError):
            self.job.accept_windows(self.ticket, self.runner, object())

    def test_unexpected_private_event_after_terminal_is_refused(self):
        result = self.refused('unexpected-event')
        self.assertIn('after terminal', result['error']['message'])

    def test_duplicate_private_terminal_during_exit_wait_is_refused(self):
        result = self.refused('duplicate-terminal')
        self.assertIn('after terminal', result['error']['message'])

    def test_nonzero_bootstrap_exit_after_successful_worker_terminal_is_refused(self):
        result = self.refused('nonzero-bootstrap')
        self.assertEqual(result['bootstrap_returncode'], 7)

    def test_actual_exit_without_private_pipe_eof_is_refused(self):
        # Keep the pipe genuinely open, but mock join's timeout to avoid four
        # seconds of wall time. Actual done/alive/EOF guards still execute.
        with patch.object(threading.Thread, 'join', lambda *args, **kwargs: None):
            result = self.refused('missing-eof')
        self.process.stdout.finish()
        self.assertIn('EOF', result['error']['message'])

    def test_unobserved_owned_identity_after_terminal_cannot_seal(self):
        self.refused('incomplete-ownership')

    def test_forced_cleanup_after_terminal_cannot_seal(self):
        result = self.refused('forced-cleanup')
        self.assertTrue(result['forced'])

    def test_changed_pinned_source_during_terminal_wait_cannot_seal(self):
        self.refused('changed-source')

    def test_missing_pinned_source_during_terminal_wait_cannot_seal(self):
        self.refused('missing-source')

    def test_post_terminal_wait_authority_still_expires_before_acceptance(self):
        _, authority = self.run_once('post-terminal/success')
        self.assertIsNotNone(authority)
        self.clock = 126
        with self.assertRaises(ValueError):
            self.job.accept_windows(self.ticket, self.runner, authority)

    def test_post_terminal_wait_authority_cannot_be_replayed(self):
        _, authority = self.run_once('post-terminal/success')
        self.assertIsNotNone(authority)
        self.job.accept_windows(self.ticket, self.runner, authority)
        with self.assertRaises(ValueError):
            self.job.accept_windows(self.ticket, self.runner, authority)


if __name__ == '__main__':
    # Imported historical TestCase is a fixture helper, never a second suite.
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                               for cls in (WindowsAuthorityTests, WindowsRunnerIntegrationTests,
                                           TerminalHeartbeatIntegrationTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
