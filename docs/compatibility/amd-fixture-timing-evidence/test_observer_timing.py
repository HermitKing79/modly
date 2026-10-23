"""CPU-only scalar instrumentation contracts; never creates a Job/process."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

E = Path(__file__).resolve().parent
ROOT = E.parents[2]
spec = importlib.util.spec_from_file_location('timing_process_contract', E / 'diagnostic_process.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
bootstrap_spec = importlib.util.spec_from_file_location('timing_bootstrap_contract', E / 'diagnostic_bootstrap.py')
bootstrap = importlib.util.module_from_spec(bootstrap_spec)
bootstrap_spec.loader.exec_module(bootstrap)


class ObserverTimingTests(unittest.TestCase):
    def test_scalar_elapsed_count_total_max_and_phase(self):
        stats = {}
        module._timing_record(stats, 'normal.observe', 100, 137)
        module._timing_record(stats, 'normal.observe', 200, 211)
        self.assertEqual(stats['normal.observe'], dict(count=2, total_ns=48, max_ns=37,
                         first_begin_ns=100, last_end_ns=211))

    def test_cleanup_bucket_is_not_startup(self):
        stats = {}
        module._timing_record(stats, 'cleanup.observe', 1, 5)
        self.assertNotIn('normal.observe', stats)
        self.assertEqual(stats['cleanup.observe']['count'], 1)

    def test_missing_samples_remain_absent(self):
        self.assertEqual(module._timing_summary({}), {})

    def test_native_job_and_command_contract_unchanged(self):
        old = ast.parse((ROOT / 'api/services/amd_stage_process.py').read_text())
        new = ast.parse((E / 'diagnostic_process.py').read_text())
        for name in ('_OwnedJob', 'StageCommand', '_host_health', '_linux_ready', '_linux_settled'):
            a = next(x for x in old.body if getattr(x, 'name', None) == name)
            b = next(x for x in new.body if getattr(x, 'name', None) == name)
            self.assertEqual(ast.dump(a), ast.dump(b), name)

    def test_observer_record_does_not_do_io_or_enumeration(self):
        tree = ast.parse((E / 'diagnostic_process.py').read_text())
        fn = next(n for n in tree.body if getattr(n, 'name', None) == '_timing_record')
        calls = [ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)]
        self.assertEqual(calls, ['stats.setdefault', 'max', 'dict'])


class BootstrapTimingTests(unittest.TestCase):
    def setUp(self):
        bootstrap._EMIT_STATS = dict(count=0, total_ns=0, max_ns=0)
        self.spec = dict(stage_id='synthetic', nonce='test', context={'nonce': 'test'})

    def test_emit_records_actual_flush_scope(self):
        with patch.object(bootstrap.time, 'monotonic_ns', side_effect=[100, 138]), patch('builtins.print') as output:
            bootstrap.emit(self.spec, event='control-received', action='GO')
        self.assertEqual(bootstrap._EMIT_STATS, dict(count=1, total_ns=38, max_ns=38))
        self.assertTrue(output.call_args.kwargs['flush'])

    def test_failed_emit_is_measured_and_not_suppressed(self):
        with patch.object(bootstrap.time, 'monotonic_ns', side_effect=[100, 109]), patch('builtins.print', side_effect=BrokenPipeError('mock')):
            with self.assertRaises(BrokenPipeError): bootstrap.emit(self.spec, event='mock')
        self.assertEqual(bootstrap._EMIT_STATS['total_ns'], 9)

    def test_original_control_authentication_and_native_stage_unchanged(self):
        old = ast.parse((ROOT / 'api/services/amd_stage_bootstrap.py').read_text())
        new = ast.parse((E / 'diagnostic_bootstrap.py').read_text())
        for name in ('valid_control', 'read_control', 'native_stage'):
            a = next(n for n in old.body if getattr(n, 'name', None) == name)
            b = next(n for n in new.body if getattr(n, 'name', None) == name)
            self.assertEqual(ast.dump(a), ast.dump(b), name)


if __name__ == '__main__':
    unittest.main()
