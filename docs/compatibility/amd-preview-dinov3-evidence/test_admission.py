"""Synthetic CPU refusal checks against existing host admission and authority.

No model, GPU, native Job, worker, or WSL execution is permitted. The reused
fixture contains synthetic bytes; a successful bind qualifies no inference.
"""
import importlib.abc
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'api'))


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
                'torch', 'triton', 'trellis2', 'transformers', 'flex_gemm',
                'cumesh', 'nvdiffrast', 'o_voxel', 'rembg', 'onnxruntime'}:
            raise AssertionError('CPU refusal check attempted model/native import: ' + fullname)


GUARD = NoModels()
sys.meta_path.insert(0, GUARD)
from services import amd_stage_process as stage
from services import amd_windows_authority as authority
from tests import test_amd_upstream_job as fixtures


class ExistingAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.UpstreamJobTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.job = self.fixture.job()
        self.ticket = self.job.issue('conditioning')
        self.runner = stage.OwnedStageRunner(self.fixture.root / 'runner')
        self.launch = patch.object(stage.subprocess, 'Popen',
            side_effect=AssertionError('CPU refusal proof must not launch a process')).start()
        self.native = patch.object(stage, '_OwnedJob',
            side_effect=AssertionError('CPU refusal proof must not construct a native Job')).start()
        self.addCleanup(patch.stopall)
        self.addCleanup(self.assert_no_launch)

    def assert_no_launch(self):
        self.launch.assert_not_called()
        self.native.assert_not_called()

    def command(self, argv=None):
        return stage.StageCommand('conditioning', 'windows', argv or (
            sys.executable, '-B', '-I', '-S', str(self.fixture.sources['policy'].path)),
            self.fixture.root, self.fixture.context.to_dict(), timeout_s=180)

    def bind(self, command=None):
        # Real UpstreamJob delegates to the original amd_windows_authority.bind.
        return self.job.bind_windows(self.ticket, self.runner, command or self.command())

    def test_exact_legacy_amd_worker_command_is_rejected_before_launch(self):
        command = self.command((
            'D:/AI/Trellis2AMD/venv/Scripts/python.exe', '-B', '-I',
            str(ROOT / 'scripts/compatibility/amd_dinov3_conditioning.py'), '--worker'))
        with self.assertRaisesRegex(ValueError, 'original interpreter and isolated stdlib'):
            self.bind(command)
        self.assertEqual(self.runner._bindings, {})
        self.assertTrue(self.job._blocked)

    def test_api_interpreter_without_stdlib_isolation_is_rejected(self):
        command = self.command((sys.executable, '-B', '-I',
            str(self.fixture.sources['policy'].path), '--worker'))
        with self.assertRaisesRegex(ValueError, 'original interpreter and isolated stdlib'):
            self.bind(command)
        self.assertEqual(self.runner._bindings, {})

    def test_existing_stdlib_command_binds_without_launch_or_authority(self):
        token = self.bind()
        self.assertIs(type(token), object)
        self.assertIn(token, self.runner._bindings)
        self.assertIsNone(self.runner._bindings[token].authority)
        self.assertEqual(self.job._accepted, [])
        self.assertEqual(self.job._pending.stage_id, 'conditioning')

    def test_worker_authored_pass_cannot_supply_runner_authority(self):
        self.bind()
        worker_pass = dict(status='PASS', terminal='succeeded', returncode=0,
            natural=True, settled=True, forced=False)
        with self.assertRaisesRegex(ValueError, 'Opaque registered Windows authority'):
            self.job.accept_windows(self.ticket, self.runner, worker_pass)
        self.assertEqual(self.job._accepted, [])
        self.assertTrue(self.job._blocked)
        with self.assertRaisesRegex(ValueError, 'Opaque registered Windows authority'):
            authority.consume(worker_pass, self.job, self.ticket, self.runner)

    def test_cancelled_ticket_refuses_binding_and_remains_blocked(self):
        self.fixture.cancel.set()
        with self.assertRaises(InterruptedError):
            self.bind()
        self.assertEqual(self.runner._bindings, {})
        self.fixture.cancel.clear()
        with self.assertRaisesRegex(ValueError, 'permanently blocked'):
            self.bind()
        self.assertEqual(self.job._accepted, [])

    def test_legacy_360_second_budget_cannot_raise_outer_180_second_cap(self):
        with self.assertRaisesRegex(ValueError, 'within 180 seconds'):
            stage.StageCommand('conditioning', 'windows', self.command().argv,
                self.fixture.root, self.fixture.context.to_dict(), timeout_s=360)
        self.assertEqual(stage.MEMORY_LIMIT, 6 * 1024**3)
        self.assertEqual(self.command().timeout_s, 180)


if __name__ == '__main__':
    unittest.main(verbosity=2)
