"""Private registry simulations only; no native/WSL execution qualification."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from services import amd_stage_process as stage
from services import amd_windows_authority as authority
from tests.test_amd_upstream_job import UpstreamJobTests


class WslAuthorityTests(unittest.TestCase):
    def setUp(self):
        from services.amd_upstream_job import UpstreamJob
        self.assertTrue(hasattr(UpstreamJob, 'bind_wsl'), 'WSL registry binding is missing')
        self.addCleanup(patch.stopall)
        patch.object(stage, '_OwnedJob', side_effect=AssertionError('No native Job in registry unit checks')).start()
        patch.object(stage.subprocess, 'Popen', side_effect=AssertionError('No process in registry unit checks')).start()
        self.prepare()

    def prepare(self):
        fixture = UpstreamJobTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.job = fixture.job()
        self.ticket = self.job.issue('conditioning')
        self.runner = stage.OwnedStageRunner(fixture.root / 'runner')
        self.client = fixture.root / 'wsl.exe'
        self.client.write_bytes(b'unit-test OS client pin, never executed')
        patch.object(authority, '_wsl_client_path', return_value=self.client).start()
        guest = '/mnt/c/' + fixture.sources['policy'].path.as_posix()[3:]
        self.command = stage.StageCommand('conditioning', 'wsl',
            ('/usr/bin/python3', '-B', '-I', '-S', guest), fixture.root,
            fixture.context.to_dict(), timeout_s=25)

    def bind(self):
        token = self.job.bind_wsl(self.ticket, self.runner, self.command)
        return self.runner._bindings[token]

    def evidence(self):
        path = self.runner.evidence_dir / 'conditioning/receipt.json'
        path.parent.mkdir(parents=True)
        path.write_text('{"unit_test_simulation_only":true}')
        return dict(terminal='succeeded', returncode=0, bootstrap_returncode=0,
            natural=True, settled=True, forced=False, diagnostic_errors=0,
            backendkind='wsl', linux_client_returncode=0, linux_settled=True,
            receipt_path=str(path), unit_test_simulation_only=True)

    def seal(self):
        invocation = self.bind()
        invocation.start(self.runner)
        self.fixture.emit('conditioning')
        invocation.seal(self.evidence())
        return invocation

    def test_binding_detaches_context_and_uses_ticket_control_nonce(self):
        invocation = self.bind()
        command, cancel = invocation.start(self.runner)
        self.assertEqual(invocation.kind, 'wsl')
        self.assertIs(cancel, self.fixture.cancel)
        self.assertIsNot(command, self.command)
        self.assertEqual(command.context, self.fixture.context.to_dict())
        self.assertEqual(command.control_nonce, self.ticket.nonce)
        self.assertNotEqual(command.control_nonce, command.context['nonce'])

    def test_simulated_accept_is_wsl_cpu_only_and_one_use(self):
        invocation = self.seal()
        token = invocation.authority
        pin = self.job.accept_wsl(self.ticket, self.runner, token)
        result = json.loads(pin.path.read_bytes())
        self.assertEqual(result['qualification'], 'WSL_CPU_SUPERVISED')
        self.assertIs(result['genuine_execution_qualified'], False)
        policy = result['wsl_supervision']['binding']['linux_policy']
        self.assertEqual(policy['distro'], 'Modly-AMD-Ubuntu-24.04')
        self.assertEqual(policy['uid'], 1000)
        self.assertEqual(policy['executable'], '/usr/bin/python3')
        self.assertEqual(policy['real_executable'], '/usr/bin/python3.12')
        self.assertEqual(policy['rlimit_as_bytes'], [6*1024**3]*2)
        with self.assertRaises(ValueError):
            authority.consume(token, self.job, self.ticket, self.runner, backendkind='wsl')

    def test_wrong_backend_and_copied_ticket_do_not_burn_rightful_capability(self):
        invocation = self.seal()
        token = invocation.authority
        with self.assertRaises(ValueError):
            authority.consume(token, self.job, self.ticket, self.runner)
        with self.assertRaises(ValueError):
            authority.consume(token, self.job, copy.copy(self.ticket), self.runner, backendkind='wsl')
        self.job.accept_wsl(self.ticket, self.runner, token)

    def test_guest_alias_unpinned_source_wrong_executable_and_backend_refused(self):
        for variant in ('alias', 'unregistered', 'executable', 'backend'):
            with self.subTest(variant=variant):
                if variant != 'alias':
                    self.prepare()
                argv = list(self.command.argv)
                if variant == 'alias':
                    argv[4] = argv[4].replace('/host-selected', '/../host-selected')
                elif variant == 'unregistered':
                    argv[4] = '/mnt/c/foreign.py'
                elif variant == 'executable':
                    argv[0] = sys.executable
                self.command = stage.StageCommand('conditioning', 'windows' if variant == 'backend' else 'wsl',
                    tuple(argv), self.command.cwd, self.command.context, timeout_s=25)
                with self.assertRaises(ValueError):
                    self.bind()

    def test_missing_or_untyped_wsl_settlement_refused(self):
        for key, value in (('backendkind', 'windows'), ('linux_settled', 1), ('linux_client_returncode', False)):
            with self.subTest(key=key):
                if key != 'backendkind':
                    self.prepare()
                invocation = self.bind()
                invocation.start(self.runner)
                evidence = self.evidence()
                evidence[key] = value
                with self.assertRaises(ValueError):
                    invocation.seal(evidence)
                self.assertIsNone(invocation.authority)

    def test_client_mutation_after_seal_refuses_acceptance(self):
        invocation = self.seal()
        self.client.write_bytes(b'changed OS client')
        with self.assertRaises(ValueError):
            self.job.accept_wsl(self.ticket, self.runner, invocation.authority)

    def test_changed_original_policy_after_bind_refuses_start(self):
        invocation = self.bind()
        with patch.object(stage, 'WSL_REAL_EXECUTABLE', '/usr/bin/foreign'):
            with self.assertRaises(ValueError):
                invocation.start(self.runner)

    def test_cancelled_authority_is_sticky(self):
        invocation = self.seal()
        self.fixture.cancel.set()
        with self.assertRaises((ValueError, InterruptedError)):
            self.job.accept_wsl(self.ticket, self.runner, invocation.authority)
        self.fixture.cancel.clear()
        with self.assertRaises(ValueError):
            invocation.recheck()

    def test_expired_authority_blocks_before_acceptance(self):
        with patch.object(authority.time, 'monotonic', return_value=100):
            invocation = self.seal()
        with patch.object(authority.time, 'monotonic', return_value=126):
            with self.assertRaises(ValueError):
                self.job.accept_wsl(self.ticket, self.runner, invocation.authority)
        self.assertIsNone(invocation.authority)

    def test_os_link_exception_is_exact_path_and_original_count_stable(self):
        other = self.fixture.root / 'foreign-wsl.exe'
        other.write_bytes(self.client.read_bytes())
        with self.assertRaises(ValueError):
            authority._wsl_client_snapshot(other)
        info = self.client.stat()
        fields = ('st_mode', 'st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        double_link = SimpleNamespace(**{key: getattr(info, key) for key in fields}, st_nlink=2)
        original_stat = Path.stat
        with patch.object(Path, 'stat', lambda path, *args, **kwargs:
                double_link if path == self.client else original_stat(path, *args, **kwargs)), \
             patch.object(authority.os, 'fstat', return_value=double_link):
            snapshot = authority._wsl_client_snapshot(self.client)
        self.assertEqual(snapshot.identity[-1], 2)
        with self.assertRaises(ValueError):
            snapshot.recheck()
        # Ordinary source/output readers retain the original single-link rule.
        from services.amd_upstream_job import _snapshot
        with patch.object(Path, 'stat', lambda path, *args, **kwargs:
                double_link if path == self.client else original_stat(path, *args, **kwargs)):
            with self.assertRaises(ValueError):
                _snapshot(self.client, 64*1024**2)

    def test_backend_specific_accept_wrapper_refuses_other_authority(self):
        invocation = self.seal()
        with self.assertRaises(ValueError):
            self.job.accept_windows(self.ticket, self.runner, invocation.authority)
        self.assertTrue(self.job._blocked)

    def test_wsl_stage_cannot_upgrade_final_neural_qualification(self):
        invocation = self.seal()
        self.job.accept_wsl(self.ticket, self.runner, invocation.authority)
        for name in ('occupancy', 'shape'):
            ticket = self.job.issue(name)
            self.fixture.emit(name)
            self.job.accept(ticket, self.fixture.observe(ticket))
        result = json.loads(self.job.complete().receipt_path.read_bytes())
        self.assertEqual(result['qualification'], 'CPU_CONTRACT_ONLY')
        self.assertIs(result['genuine_execution_qualified'], False)
