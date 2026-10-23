"""Focused CPU checks; fake controls/link syscall, no worker or Job launch.

The sandbox denies os.link. Its exclusive publication effect is substituted
here; these tests do not qualify the real outside-sandbox link primitive.
"""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch


FIXTURE = Path(__file__).with_name('fixture.py')
CONTEXT = dict(job_id='job-target', request_id='request-target', run_id='run-target',
               nonce='context-nonce', input_sha256='a' * 64, params_sha256='b' * 64,
               created_ns=123456789, mode='generated', model_id='trellis2-amd')


class PendingInput:
    """Keep the sentinel's control channel open until test cleanup."""
    def __init__(self):
        self.closed = threading.Event()

    def readline(self, limit):
        self.closed.wait(5)
        return b''


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(FIXTURE.is_file(), 'Missing stdlib control fixture')
        spec = importlib.util.spec_from_file_location('minimal_windows_fixture', FIXTURE)
        self.fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.fixture)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.leaf = Path(temp.name).resolve()
        self.ack = self.leaf / 'cancel-ack.json'
        self.env = dict(MODLY_COMPLETION_CONTEXT_JSON=json.dumps(CONTEXT),
                        MODLY_STAGE_ID='target-stage', MODLY_STAGE_CANCEL_ACK=str(self.ack))
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, self.env).start()
        patch.object(self.fixture.os, 'getpid', return_value=4242).start()
        patch.object(self.fixture.time, 'monotonic', return_value=12.5).start()
        def exclusive_link(source, destination):
            with Path(destination).open('xb') as stream:
                stream.write(Path(source).read_bytes())
        patch.object(self.fixture.os, 'link', side_effect=exclusive_link).start()

    def args(self, role='target'):
        return ['--role', role, '--private-leaf', str(self.leaf),
                '--release-nonce', 'independent-release', '--job-nonce', 'host-stage-launch']

    def cancel(self, **changes):
        value = dict(action='CANCEL', stage_id='target-stage', nonce='context-nonce')
        value.update(changes)
        return io.BytesIO((json.dumps(value) + '\n').encode())

    def release(self):
        return dict(action='RELEASE', context=copy.deepcopy(CONTEXT), stage_id='target-stage',
                    pid=4242, release_nonce='independent-release', job_nonce='host-stage-launch')

    def write_release(self, value):
        (self.leaf / 'release.json').write_text(json.dumps(value), encoding='utf-8')

    def test_target_readiness_and_original_bootstrap_cancel_ack_are_context_bound(self):
        self.assertEqual(self.fixture.main(self.args(), stdin=self.cancel()), 0)
        ready = json.loads((self.leaf / 'ready.json').read_bytes())
        expected = dict(context=CONTEXT, stage_id='target-stage', role='target', pid=4242,
                        nonce='context-nonce', release_nonce='independent-release',
                        job_nonce='host-stage-launch', monotonic_s=12.5)
        self.assertEqual(ready, expected)
        self.assertEqual(json.loads(self.ack.read_bytes()), dict(expected, terminal='cancelled'))
        self.assertFalse((self.leaf / 'released.json').exists())

    def test_sentinel_accepts_independent_exact_release_and_has_no_cancel_ack(self):
        self.write_release(self.release())
        stream = PendingInput()
        self.addCleanup(stream.closed.set)
        self.assertEqual(self.fixture.main(self.args('sentinel'), stdin=stream), 0)
        ready = json.loads((self.leaf / 'ready.json').read_bytes())
        self.assertEqual(ready['role'], 'sentinel')
        self.assertEqual(ready['context'], CONTEXT)
        self.assertEqual(json.loads((self.leaf / 'released.json').read_bytes()),
                         dict(ready, terminal='succeeded', action='RELEASE'))
        self.assertFalse(self.ack.exists())

    def test_sentinel_cooperates_with_real_cancel_only_as_cancelled_cleanup(self):
        self.assertEqual(self.fixture.main(self.args('sentinel'), stdin=self.cancel()), 0)
        self.assertEqual(json.loads(self.ack.read_bytes())['terminal'], 'cancelled')
        self.assertEqual(json.loads(self.ack.read_bytes())['role'], 'sentinel')
        self.assertFalse((self.leaf / 'released.json').exists())

    def test_foreign_extra_malformed_controls_and_eof_fail_closed(self):
        messages = [self.cancel(stage_id='other'), self.cancel(nonce='foreign'),
                    self.cancel(action='RELEASE'), self.cancel(extra=True), io.BytesIO(b''),
                    io.BytesIO(b'{"action":"CANCEL"}'), io.BytesIO(b'x\n'),
                    io.BytesIO(b'{"action":"CANCEL","action":"CANCEL","stage_id":"target-stage","nonce":"context-nonce"}\n')]
        for index, stream in enumerate(messages):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as name:
                leaf = Path(name).resolve()
                args = self.args(); args[3] = str(leaf)
                with self.assertRaises(ValueError):
                    self.fixture.main(args, stdin=stream)
                self.assertFalse(self.ack.exists())
                self.assertFalse((leaf / 'released.json').exists())

    def test_release_refuses_foreign_context_stage_pid_and_each_host_nonce(self):
        mutations = [dict(context=dict(CONTEXT, request_id='foreign')),
                     dict(stage_id='foreign'), dict(pid=4343),
                     dict(release_nonce='context-nonce'), dict(job_nonce='foreign'),
                     dict(action='CANCEL'), dict(extra=True)]
        for changed in mutations:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as name:
                leaf = Path(name).resolve()
                args = self.args('sentinel'); args[3] = str(leaf)
                (leaf / 'release.json').write_text(json.dumps(dict(self.release(), **changed)))
                stream = PendingInput()
                try:
                    with self.assertRaises(ValueError):
                        self.fixture.main(args, stdin=stream)
                finally:
                    stream.closed.set()
                self.assertFalse((leaf / 'released.json').exists())
                self.assertFalse(self.ack.exists())

    def test_context_nonce_cannot_be_reused_as_host_release_token(self):
        args = self.args(); args[5] = CONTEXT['nonce']
        with self.assertRaises(ValueError):
            self.fixture.main(args, stdin=self.cancel())
        self.assertFalse((self.leaf / 'ready.json').exists())

    def test_existing_publication_is_never_overwritten(self):
        path = self.leaf / 'ready.json'; path.write_bytes(b'preserved')
        with self.assertRaises(FileExistsError):
            self.fixture.main(self.args(), stdin=self.cancel())
        self.assertEqual(path.read_bytes(), b'preserved')
        self.assertFalse(self.ack.exists())

    def test_target_never_accepts_sentinel_release_instead_of_cancellation(self):
        self.write_release(self.release())
        with self.assertRaises(ValueError):
            self.fixture.main(self.args(), stdin=io.BytesIO(b''))
        self.assertFalse((self.leaf / 'released.json').exists())
        self.assertFalse(self.ack.exists())

    def test_release_waits_for_publisher_staging_unlink_then_requires_single_link(self):
        path = self.leaf / 'release.json'
        staging = self.leaf / 'release.json.writing'
        self.write_release(self.release())
        staging.write_bytes(path.read_bytes())
        original_stat = Path.stat
        def linked_stat(entry, *args, **kwargs):
            result = original_stat(entry, *args, **kwargs)
            if entry == path:
                values = {name: getattr(result, name) for name in dir(result) if name.startswith('st_')}
                values['st_nlink'] = 2
                return SimpleNamespace(**values)
            return result
        with patch.object(Path, 'stat', linked_stat):
            self.assertFalse(self.fixture.release_requested(path, self.release()))
        staging.unlink()
        self.assertTrue(self.fixture.release_requested(path, self.release()))
        with patch.object(Path, 'stat', linked_stat):
            with self.assertRaises(ValueError):
                self.fixture.release_requested(path, self.release())

    def test_failed_publication_preserves_pending_evidence_and_original_link_error(self):
        path = self.leaf / 'ready.json'
        original_error = PermissionError('original link failure')
        with patch.object(self.fixture.os, 'link', side_effect=original_error), \
             patch.object(Path, 'unlink', side_effect=RuntimeError('must not replace original')):
            with self.assertRaises(PermissionError) as raised:
                self.fixture.publish(path, {'original': 'evidence'})
        self.assertIs(raised.exception, original_error)
        self.assertFalse(path.exists())
        pending = self.leaf / '.ready.json.4242.pending'
        self.assertEqual(json.loads(pending.read_bytes()), {'original': 'evidence'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
