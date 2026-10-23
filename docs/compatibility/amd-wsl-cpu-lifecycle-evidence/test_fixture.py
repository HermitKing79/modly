"""Mocked Linux fixture contracts; no WSL, process, pidfd or resource call."""
import ast
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name('fixture.py')
CONTEXT = dict(job_id='synthetic-job', request_id='request', run_id='run', nonce='stage-nonce',
               input_sha256='a'*64, params_sha256='b'*64, created_ns=123,
               mode='generated', model_id='test-only/no-neural-computation')
FACTS = dict(pid=41, uid=1000, euid=1000, user='modly', executable='/usr/bin/python3.12',
             rlimit_as=[6*1024**3, 6*1024**3], boot_id='unit-test-boot-id',
             identity=dict(pid=41, ppid=40, pgrp=41, session=41, start_ticks=321,
                           rss_pages=100, state='S', uid=1000))


class PendingInput:
    def __init__(self):
        self.closed = threading.Event()

    def readline(self, size):
        self.closed.wait(2)
        return b''


class LinuxFixtureTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SOURCE.is_file(), 'Linux CPU fixture is missing')
        spec = importlib.util.spec_from_file_location('wsl_cpu_fixture', SOURCE)
        self.fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.fixture)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.leaf = Path(temp.name).resolve()
        self.ack = self.leaf / 'cancel-ack.json'
        self.binding = dict(context=copy.deepcopy(CONTEXT), stage_id='wsl-target', nonce=CONTEXT['nonce'])
        self.host_after_alive = None
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, MODLY_COMPLETION_CONTEXT_JSON=json.dumps(CONTEXT),
                   MODLY_STAGE_ID=self.binding['stage_id'], MODLY_STAGE_CANCEL_ACK=str(self.ack)).start()
        patch.object(self.fixture, 'collect_runtime', side_effect=lambda: copy.deepcopy(FACTS)).start()
        patch.object(self.fixture.time, 'monotonic', return_value=1.0).start()
        def exclusive_link(source, destination):
            # Known sandbox-denied hardlink syscall only. Fixture control and
            # file readback stay real; actual Linux publication is NOT qualified.
            with Path(destination).open('xb') as stream:
                stream.write(Path(source).read_bytes())
            if Path(destination).name == 'alive-proof.json' and self.host_after_alive:
                self.host_after_alive()
        patch.object(self.fixture.os, 'link', side_effect=exclusive_link).start()

    def control(self, **changes):
        row = dict(action='CANCEL', stage_id=self.binding['stage_id'], nonce=CONTEXT['nonce'])
        row.update(changes)
        return io.BytesIO((json.dumps(row)+'\n').encode())

    def pending(self):
        stream = PendingInput()
        self.addCleanup(stream.closed.set)
        return stream

    def probe(self, **changes):
        row = dict(self.binding, action='PROBE', challenge='independent-host-challenge')
        row.update(changes)
        return row

    def release(self, **changes):
        row = dict(self.binding, action='RELEASE', challenge='independent-host-challenge')
        row.update(changes)
        return row

    def write(self, name, value):
        (self.leaf / name).write_text(json.dumps(value), encoding='utf-8')

    def test_target_readiness_and_exact_cooperative_ack_precede_completed_exit(self):
        self.assertEqual(self.fixture.main(['target', str(self.leaf)], stdin=self.control()), 0)
        ready = json.loads((self.leaf/'ready.json').read_bytes())
        self.assertEqual(ready, dict(self.binding, mode='target', monotonic_s=1.0, **FACTS))
        expected = dict(ready, terminal='cancelled')
        self.assertEqual(json.loads(self.ack.read_bytes()), expected)
        self.assertEqual(json.loads((self.leaf/'completed.json').read_bytes()), expected)
        self.assertFalse((self.leaf/'success.json').exists())

    def test_sentinel_answers_fresh_exact_probe_before_independent_success_release(self):
        self.write('host-challenge.json', self.probe())
        self.host_after_alive = lambda: self.write('release.json', self.release())
        self.assertEqual(self.fixture.main(['sentinel', str(self.leaf)], stdin=self.pending()), 0)
        alive = json.loads((self.leaf/'alive-proof.json').read_bytes())
        self.assertEqual(alive, dict(self.binding, mode='sentinel', monotonic_s=1.0,
                                    action='PROBE', challenge='independent-host-challenge', **FACTS))
        success = json.loads((self.leaf/'success.json').read_bytes())
        self.assertEqual(success, dict(alive, action='RELEASE', terminal='succeeded'))
        self.assertFalse(self.ack.exists())

    def test_sentinel_cancel_cleanup_is_cancelled_instead_of_success(self):
        self.assertEqual(self.fixture.main(['sentinel', str(self.leaf)], stdin=self.control()), 0)
        self.assertEqual(json.loads(self.ack.read_bytes())['terminal'], 'cancelled')
        self.assertFalse((self.leaf/'success.json').exists())

    def test_foreign_extra_duplicate_partial_controls_and_eof_are_refused(self):
        controls = [self.control(nonce='foreign'), self.control(stage_id='foreign'),
                    self.control(action='RELEASE'), self.control(extra=True), io.BytesIO(b''),
                    io.BytesIO(b'{"action":"CANCEL"}'),
                    io.BytesIO(b'{"action":"CANCEL","action":"CANCEL","stage_id":"wsl-target","nonce":"stage-nonce"}\n')]
        for index, stream in enumerate(controls):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as name:
                with self.assertRaises(ValueError):
                    self.fixture.main(['target', str(Path(name).resolve())], stdin=stream)
                self.assertFalse(self.ack.exists())

    def test_wrong_uid_user_session_interpreter_or_inherited_limit_cannot_publish_ready(self):
        for key, value in [('uid', 0), ('euid', 0), ('user', 'root'),
                           ('executable', '/foreign/python3'), ('rlimit_as', [1, 1]),
                           ('identity', dict(FACTS['identity'], session=99)),
                           ('identity', dict(FACTS['identity'], start_ticks=0))]:
            with self.subTest(key=key, value=value):
                facts = copy.deepcopy(FACTS); facts[key] = value
                with self.assertRaises(ValueError):
                    self.fixture.validate_runtime(facts, '/usr/bin/python3.12')
        self.fixture.validate_runtime(copy.deepcopy(FACTS), '/usr/bin/python3.12')

    def test_foreign_probe_and_reused_context_nonce_are_refused(self):
        for changes in [dict(nonce='foreign'), dict(stage_id='foreign'),
                        dict(context=dict(CONTEXT, request_id='foreign')), dict(extra=True),
                        dict(challenge=CONTEXT['nonce'])]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.fixture.validate_probe(self.probe(**changes), self.binding)

    def test_foreign_release_and_changed_challenge_are_refused(self):
        for changes in [dict(nonce='foreign'), dict(stage_id='foreign'), dict(action='CANCEL'),
                        dict(context=dict(CONTEXT, request_id='foreign')), dict(extra=True),
                        dict(challenge='another-challenge')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.fixture.validate_release(self.release(**changes), self.binding,
                                              'independent-host-challenge')

    def test_release_before_probe_response_fails_closed(self):
        self.write('release.json', self.release())
        with self.assertRaises(ValueError):
            self.fixture.main(['sentinel', str(self.leaf)], stdin=self.pending())
        self.assertFalse((self.leaf/'success.json').exists())

    def test_changed_probe_after_alive_response_is_not_accepted_as_release(self):
        self.write('host-challenge.json', self.probe())
        self.host_after_alive = lambda: self.write('host-challenge.json', self.probe(challenge='replacement'))
        with self.assertRaises(ValueError):
            self.fixture.main(['sentinel', str(self.leaf)], stdin=self.pending())
        self.assertFalse((self.leaf/'success.json').exists())

    def test_missing_durable_cancel_ack_cannot_be_completed_successfully(self):
        real_publish = self.fixture.publish
        def missing_ack(path, value):
            if Path(path) != self.ack:
                return real_publish(path, value)
        with patch.object(self.fixture, 'publish', side_effect=missing_ack):
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.fixture.main(['target', str(self.leaf)], stdin=self.control())
        self.assertFalse((self.leaf/'completed.json').exists())

    def test_duplicate_invocation_cannot_reuse_nonce_and_existing_publications(self):
        self.fixture.main(['target', str(self.leaf)], stdin=self.control())
        previous = self.ack.read_bytes()
        with self.assertRaises(FileExistsError):
            self.fixture.main(['target', str(self.leaf)], stdin=self.control())
        self.assertEqual(self.ack.read_bytes(), previous)

    def test_internal_24_second_ceiling_does_not_create_a_readiness_subdeadline(self):
        clock = [0.0]
        def advance():
            clock[0] += 2
            return clock[0]
        with patch.object(self.fixture.time, 'monotonic', side_effect=advance):
            with self.assertRaises(TimeoutError):
                self.fixture.main(['sentinel', str(self.leaf)], stdin=self.pending())
        self.assertFalse((self.leaf/'success.json').exists())
        self.assertFalse(self.ack.exists())

    def test_fixture_source_has_no_child_or_grandchild_process_creation(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        imports = {alias.name.split('.')[0] for node in ast.walk(tree)
                   if isinstance(node, ast.Import) for alias in node.names}
        self.assertFalse(imports & {'subprocess', 'multiprocessing'})
        forbidden = {'fork', 'forkpty', 'Popen', 'system', 'spawn', 'spawnl', 'spawnv',
                     'posix_spawn', 'execv', 'execve', 'execvpe', 'execl', 'execlp'}
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Attribute, ast.Name))}
        self.assertFalse(calls & forbidden)


if __name__ == '__main__':
    unittest.main(verbosity=2)
