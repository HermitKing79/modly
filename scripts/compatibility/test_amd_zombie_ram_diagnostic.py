"""Synthetic-only tests: never import historical adapters or query real RAM."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest import mock

SCRATCH = Path(__file__).resolve().parents[3] / 'artifacts/amd-zombie-standalone-preparation-20261024-v1/test-scratch'
SCRATCH.mkdir(exist_ok=True)
tempfile.tempdir = str(SCRATCH)

SCRIPT = Path(__file__).with_name('amd_zombie_ram_diagnostic.py')
spec = importlib.util.spec_from_file_location('zombie_ram_diagnostic_test_target', SCRIPT)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class Gates(unittest.TestCase):
    def test_recorded_failure_is_not_overridden_by_ram_recovery(self):
        values = iter([22 * 1024**3 - 1] + [24 * 1024**3] * 10)
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            with diagnostic.Phases(Path(directory), lambda: {'available_bytes': next(values)},
                                   handshake=lambda event: None) as phase:
                with self.assertRaises(diagnostic.DiagnosticStop):
                    phase.run('imports', lambda: calls.append('must not run'))
        self.assertEqual(calls, [])

    def test_22_passes_without_24(self):
        self.assertTrue(diagnostic.admitted([22 * 1024**3] * 3))
        self.assertFalse(diagnostic.admitted([22 * 1024**3] * 2))
        self.assertFalse(diagnostic.admitted([22 * 1024**3, 22 * 1024**3 - 1, 24 * 1024**3]))

    def test_phase_gate_stops_before_operation(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            with diagnostic.Phases(Path(directory), lambda: {'available_bytes': 22 * 1024**3 - 1},
                                   handshake=lambda event: None) as phase:
                with self.assertRaises(diagnostic.DiagnosticStop):
                    phase.run('imports', lambda: calls.append('forbidden'))
            self.assertEqual(calls, [])
            records = [json.loads(line) for line in (Path(directory) / 'phases.jsonl').read_text().splitlines()]
            self.assertEqual(records[-1]['phase'], 'imports')
            self.assertEqual(records[-1]['edge'], 'gate-rejection')

    def test_after_gate_stops_next_phase(self):
        value = {'available_bytes': 23 * 1024**3}
        with tempfile.TemporaryDirectory() as directory:
            with diagnostic.Phases(Path(directory), lambda: dict(value), handshake=lambda event: None) as phase:
                with self.assertRaises(diagnostic.DiagnosticStop):
                    phase.run('constructor', lambda: value.update(available_bytes=22 * 1024**3 - 1))
            records = [json.loads(line) for line in (Path(directory) / 'phases.jsonl').read_text().splitlines()]
            self.assertEqual(records[-2]['edge'], 'after')
            self.assertEqual(records[-2]['available_delta_bytes'], -1024**3 - 1)


class Boundaries(unittest.TestCase):
    def test_neural_import_roots(self):
        for name in ('torch', 'torch.nn', 'triton', 'trellis2.models', 'safetensors.torch', 'flex_gemm'):
            self.assertTrue(diagnostic.forbidden_import(name), name)
        for name in ('numpy', 'safetensors.numpy', 'shape_decoder_loader', 'json'):
            self.assertFalse(diagnostic.forbidden_import(name), name)

    def test_only_bounded_git_reads(self):
        root = Path(tempfile.gettempdir()).resolve()
        for args in (['git', 'show', diagnostic.BASE + ':scripts/compatibility/example.py'],
                     ['git', 'rev-parse', 'HEAD']):
            diagnostic.validate_spawn('git', args, root, root)
        for args in (['wsl.exe'], ['git', 'checkout', 'main'], ['git', 'show', '--output=file'],
                     ['python', '-c', 'pass'], 'git rev-parse HEAD'):
            with self.assertRaises(diagnostic.DiagnosticStop):
                diagnostic.validate_spawn(args[0], args, root, root)

    def test_generation_calls_denied(self):
        for name in ('main', 'supervise', 'reserve_attempt', 'load_cpu', '_factory', 'worker_command'):
            self.assertTrue(diagnostic.forbidden_call(name, '/repo/scripts/compatibility/anything.py', '/repo'))
        self.assertFalse(diagnostic.forbidden_call('source_inputs', '/repo/scripts/compatibility/anything.py', '/repo'))

    def test_extraction_does_not_execute_module(self):
        source = "raise RuntimeError('module entry must not run')\ndef checkpoint_record(path):\n    return digest(path)\n"
        function = diagnostic.extract_checkpoint(source, {'digest': lambda path: 'streamed'})
        self.assertEqual(function('synthetic'), 'streamed')
        with self.assertRaises(diagnostic.DiagnosticStop):
            diagnostic.extract_checkpoint('pass', {})

    def test_write_scope_and_reparse_denial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'owned'
            root.mkdir()
            diagnostic.validate_write(root / 'new.json', root)
            with self.assertRaises(diagnostic.DiagnosticStop):
                diagnostic.validate_write(Path(directory) / 'historical.json', root)


class Sequence(unittest.TestCase):
    def test_only_cpu_backend_methods_called_in_order(self):
        calls = []
        class Backend:
            def __getattr__(self, name):
                if name not in diagnostic.CPU_STEPS:
                    raise AssertionError(name)
                return lambda: calls.append(name)
        class Phase:
            def run(self, name, operation):
                operation()
        diagnostic.prepare(Backend(), Phase())
        self.assertEqual(calls, list(diagnostic.CPU_STEPS))
        self.assertNotIn('supervise', calls)

    def test_exception_never_reaches_next_stage(self):
        calls = []
        class Backend:
            def __getattr__(self, name):
                def operation():
                    calls.append(name)
                    if name == 'construct_runtime':
                        raise diagnostic.DiagnosticStop('synthetic failure')
                return operation
        class Phase:
            def run(self, name, operation):
                operation()
        with self.assertRaises(diagnostic.DiagnosticStop):
            diagnostic.prepare(Backend(), Phase())
        self.assertEqual(calls, ['imports', 'construct_runtime'])

    def test_missing_observer_permit_stops(self):
        with mock.patch.object(sys, 'stdin') as stdin, mock.patch('builtins.print'):
            stdin.readline.return_value = ''
            with self.assertRaises(diagnostic.DiagnosticStop):
                diagnostic.Phases.permit({'phase': 'imports', 'edge': 'before'})

    def test_streaming_decline_stops_before_next_read(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'synthetic.bin'
            file.write_bytes(b'x' * (2 * 1024**2))
            calls = []
            def guard():
                calls.append(1)
                if len(calls) == 2:
                    raise diagnostic.DiagnosticStop('mock RAM decline')
            with self.assertRaises(diagnostic.DiagnosticStop):
                diagnostic.stream_digest(file, guard)
            self.assertEqual(len(calls), 2)

    def test_actual_guards_in_synthetic_subprocess(self):
        for probe in ('import', 'spawn', 'write', 'call', 'socket'):
            result = subprocess.run([sys.executable, '-B', '-I', '-S', str(Path(__file__).resolve()),
                                     '--guard-probe', probe], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('BLOCKED', result.stdout)

    def test_launcher_contains_no_generation_entrypoint(self):
        source = SCRIPT.with_name('Invoke-ZombieRamDiagnostic.ps1').read_text()
        self.assertIn("if (-not $Run)", source)
        self.assertIn("'-B -I -S", source)
        self.assertIn('KILL_ON_JOB_CLOSE', source)
        self.assertIn("Check-Resources $boundary", source)
        self.assertNotIn('amd_zombie_decode.py', source)
        self.assertNotIn('wsl.exe', source.lower())


def guard_probe(kind):
    # Each test installs permanent hooks in a fresh synthetic interpreter. The
    # mocked OS boundary ensures even a broken guard cannot spawn an executable.
    with tempfile.TemporaryDirectory() as directory:
        evidence = Path(directory)
        with mock.patch('subprocess._winapi.CreateProcess', side_effect=AssertionError('OS spawn reached')):
            diagnostic.install_guards(evidence, SCRIPT.parents[2], lambda: None)
            try:
                if kind == 'import':
                    __import__('torch')
                elif kind == 'spawn':
                    subprocess.Popen(['wsl.exe'])
                elif kind == 'write':
                    (evidence.parent / 'never-created.json').write_text('denied')
                elif kind == 'call':
                    code = compile('def load_cpu():\n    raise AssertionError("body reached")\n', str(SCRIPT.parent/'synthetic_guard_test.py'), 'exec')
                    namespace = {}; exec(code, namespace); namespace['load_cpu']()
                else:
                    import socket
                    socket.socket()
            except diagnostic.DiagnosticStop:
                print('BLOCKED', kind)
                # os._exit bypasses temporary cleanup blocked by the permanent
                # audit hook. The parent retains this unique synthetic directory.
                sys.stdout.flush()
                os._exit(0)
            os._exit(3)


if __name__ == '__main__':
    import os
    if len(sys.argv) == 3 and sys.argv[1] == '--guard-probe':
        guard_probe(sys.argv[2])
    else:
        unittest.main()
