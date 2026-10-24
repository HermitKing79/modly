"""Focused CPU-only zombie decode routing tests; never launch WSL or a model."""
import copy
import ast
from contextlib import ExitStack
import hashlib
import importlib.abc
import json
from pathlib import Path, PureWindowsPath
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class NoNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'triton', 'trellis2',
                'flex_gemm', 'transformers', 'cumesh', 'o_voxel', 'nvdiffrast'}:
            raise AssertionError('CPU-only zombie tests forbid neural/native import: ' + fullname)


sys.meta_path.insert(0, NoNeuralImports())
import numpy as np
import amd_generation_decode as historical
ORIGINAL_INPUT_FUNCTION = historical.shape_inputs
import amd_zombie_shape as shape
try:
    import amd_zombie_decode as subject
except ModuleNotFoundError as exc:
    if exc.name != 'amd_zombie_decode':
        raise
    subject = None

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT.parent / 'artifacts/amd-zombie-decode-0468f86'
SHAPE_COMMIT = '35a5b825e3e9128326897a2ce35c87b237fdfa8b'
TEST_COMMIT = '1' * 40
EXPECTED_INPUTS = {
    'shape-latent-denormalized.npy': '18778e295e7d9405ca2e80b8fde64dacc790696c361f6418436d2e498149b2b3',
    'coords.npy': 'c432b04ec70c6200fadf8f5f62ce995514e24bf4194b154ac30a76148a7bb4a2',
}


class AdapterExists(unittest.TestCase):
    def test_zombie_adapter_exists(self):
        self.assertIsNotNone(subject, 'Explicitly zombie-bound standalone decode adapter is required')


class ProbePathTests(unittest.TestCase):
    def helper(self):
        # Execute the real pure helper without installing the probe's write-denial
        # audit hook in this test process. No duplicate conversion implementation.
        tree = ast.parse((ROOT / 'docs/compatibility/amd-zombie-decode-evidence/wsl_cpu_probe.py').read_text())
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in ('require', 'installed_source_path')]
        self.assertEqual({node.name for node in selected}, {'require', 'installed_source_path'})
        scope = {'Path': Path, 'PureWindowsPath': PureWindowsPath}
        exec(compile(ast.Module(body=selected, type_ignores=[]), '<actual probe helpers>', 'exec'), scope)
        return scope['installed_source_path']

    def test_actual_installed_windows_paths_convert_without_regex_ambiguity(self):
        convert = self.helper()
        for value in ('D:/AI/model/source.py', r'D:\AI\model\source.py', 'C:/Users/modly/source.py'):
            with self.subTest(path=value):
                self.assertEqual(convert(value).as_posix(), '/mnt/' + value[0].lower() + '/' + value[3:].replace('\\', '/'))

    def test_foreign_relative_traversal_and_unc_source_paths_are_refused(self):
        convert = self.helper()
        for value in ('E:/foreign.py', 'D:relative.py', '../source.py', 'D:/AI/../source.py',
                      r'\\server\share\source.py', r'D:\AI\..\source.py', '/mnt/d/source.py'):
            with self.subTest(path=value), self.assertRaises(ValueError):
                convert(value)


@unittest.skipIf(subject is None, 'Adapter not implemented yet')
class ZombieDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Authenticate the actual originals once. No temporary neural artifacts.
        cls.original_input_function = staticmethod(ORIGINAL_INPUT_FUNCTION)
        cls.actual_bundle = subject.source_inputs(shape.EVIDENCE)

    def setUp(self):
        PRIVATE.mkdir(parents=True, exist_ok=True)
        scratch = PRIVATE / 'test-scratch'
        scratch.mkdir(exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.temporary = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix='cpu-routing-', dir=scratch)))

    def runtime(self):
        # Source authentication has its own real positive and mutation tests.
        # Runtime tests reuse that authenticated bundle without repeated hashing.
        self.stack.enter_context(patch.object(subject, 'source_inputs', return_value=self.actual_bundle))
        return subject.make_runtime(TEST_COMMIT)

    def test_only_exact_saved_source_root_is_admitted_before_private_reads(self):
        with patch.object(shape, 'verify_saved', side_effect=AssertionError('private read before path admission')) as read:
            for path in (self.temporary, shape.EVIDENCE.parent, shape.EVIDENCE / 'run',
                         historical.APPROVED_ROOT / 'shape', shape.EVIDENCE / '..'):
                with self.subTest(path=str(path)), self.assertRaises(ValueError):
                    subject.source_inputs(path)
            read.assert_not_called()

    def test_actual_original_input_paths_hashes_count_order_and_c_layout_are_preserved(self):
        bundle = self.actual_bundle
        self.assertEqual(bundle['source_job'], shape.EVIDENCE / 'run/results.json')
        self.assertEqual(bundle['source_job_sha256'], '913ad907c55f1f510cce00d1d7573d518883190628d040506a6bd78578dd54bf')
        self.assertEqual(bundle['count'], 1000)
        self.assertEqual({key: row['sha256'] for key, row in bundle['inputs'].items()}, EXPECTED_INPUTS)
        self.assertEqual(bundle['inputs']['coords.npy']['path'], shape.EVIDENCE / 'run/scratch/shape-latent-coordinates.npy')
        self.assertEqual(bundle['inputs']['shape-latent-denormalized.npy']['path'], shape.EVIDENCE / 'run/scratch/shape-latent-denormalized.npy')
        saved = np.load(bundle['inputs']['coords.npy']['path'], allow_pickle=False)
        self.assertTrue(saved.flags.c_contiguous)
        self.assertFalse(saved.flags.f_contiguous)
        np.testing.assert_array_equal(bundle['coords'], saved)
        self.assertEqual(hashlib.sha256(saved.tobytes()).hexdigest(), 'b871475ff86493ca9d023aa7736262a898193e3a1b6833a2b6716f427cf2a603')

    def test_old_or_malformed_execution_commits_are_refused(self):
        for value in (None, '', '1' * 39, 'g' * 40, TEST_COMMIT.upper().replace('1', 'A'),
                      SHAPE_COMMIT, shape.OCCUPANCY_COMMIT, shape.prior.DINO_COMMIT):
            with self.subTest(value=value), self.assertRaises(ValueError):
                subject.valid_commit(value)
        subject.valid_commit(TEST_COMMIT)

    def test_actual_old_result_and_incomplete_five_artifact_closure_are_refused(self):
        original_reader = shape.prior.read_json
        target = shape.EVIDENCE / 'run/results.json'
        baseline = original_reader(target)
        def refuse(mutator):
            altered = copy.deepcopy(baseline)
            mutator(altered)
            def reader(path, *args, **kwargs):
                return altered if Path(path) == target else original_reader(path, *args, **kwargs)
            with patch.object(shape.prior, 'read_json', side_effect=reader):
                with self.assertRaises(ValueError):
                    subject.source_inputs(shape.EVIDENCE)
        refuse(lambda row: row['worker']['release'].update(allocated_bytes=1))
        refuse(lambda row: row['worker']['artifacts'].pop('shape-initial-noise.npy'))

    def test_changed_old_independent_outcome_review_and_marker_are_refused(self):
        # Mutate only JSON returned to the reader. No original file is edited.
        original_read_text = Path.read_text
        review = shape.PUBLIC / 'outcome-review.json'
        marker = shape.EVIDENCE / 'attempt-started.json'
        mutations = ((review, lambda row: row.update(classification='PARTIAL')),
                     (review, lambda row: row['private_run_evidence_sha256'].update({'results.json': '0' * 64})),
                     (marker, lambda row: row.update(harness_commit=TEST_COMMIT)))
        for target, mutate in mutations:
            def altered(path, *args, **kwargs):
                result = original_read_text(path, *args, **kwargs)
                if Path(path) == target:
                    row = json.loads(result)
                    mutate(row)
                    return json.dumps(row)
                return result
            with self.subTest(path=target.name, mutate=mutate), patch.object(Path, 'read_text', altered):
                with self.assertRaises(ValueError):
                    subject.source_inputs(shape.EVIDENCE)

    def test_private_host_clone_preserves_shared_historical_route_and_generic_caps(self):
        runtime = self.runtime()
        self.assertIs(historical.shape_inputs, self.original_input_function)
        self.assertIsNot(type(runtime), historical.Runtime)
        self.assertIs(type(runtime).__init__.__globals__['shape_inputs'], subject.source_inputs)
        self.assertEqual(runtime.evidence, historical.APPROVED_ROOT / 'zombie-0468f86/decode')
        self.assertEqual(runtime.contract.initial_count, 1000)
        self.assertEqual(dict(runtime.contract.input_sha256), EXPECTED_INPUTS)
        self.assertEqual(runtime.contract.maximum_final_rows, 1842968)
        self.assertEqual(runtime.contract.maximum_intermediate_rows, 1000000)
        self.assertTrue(runtime.contract.large_convolution_tiling)
        self.assertIsNone(runtime.contract.final_support_profile)
        self.assertNotIn('final_support_profile', runtime.contract.document())
        self.assertFalse(runtime.evidence.exists())

    def test_worker_command_uses_qualified_named_wsl_and_original_deadlines_without_launch(self):
        runtime = self.runtime()
        entry = historical.guest_path(historical.GUEST_ENTRY)
        mirror = historical.guest_path(runtime.evidence / 'run/guest')
        command = runtime.worker_command(entry, mirror)
        self.assertEqual(command[:6], ['wsl.exe', '-d', 'Modly-AMD-Ubuntu-24.04', '-u', 'modly', '--exec'])
        self.assertIn('/opt/modly-wsl/venv/bin/python', command)
        self.assertIn('/usr/bin/timeout', command)
        self.assertIn('--signal=TERM', command)
        self.assertIn('--kill-after=5s', command)
        self.assertIn('450s', command)
        self.assertIn(runtime.contract.sha256, command)
        supervisor = runtime.supervisor()
        self.assertEqual((supervisor.NATIVE_DEADLINE, supervisor.CPU_DEADLINE,
                          supervisor.GPU_DEADLINE, supervisor.HOST_DEADLINE), (60., 120., 240., 480.))
        with self.assertRaises(ValueError):
            runtime.worker_command('/foreign/decode.py', mirror)
        self.assertFalse(runtime.evidence.exists())

    def test_existing_guest_bootstrap_inherits_go_eof_stdin_and_copies_after_child_exit(self):
        mirror = self.temporary / 'mirror'
        mirror.mkdir()
        contract = SimpleNamespace(sha256='a' * 64, output_directory=self.temporary / 'owned/geometry',
                                   host_evidence_directory=mirror)
        calls = []
        paths = SimpleNamespace(attest_parent=lambda row: calls.append('parent'),
            attest_output=lambda row: calls.append('output') or contract.output_directory)
        def run(command, **kwargs):
            self.assertNotIn('stdin', kwargs)
            self.assertEqual(command[1:3], ['-B', '-I'])
            calls.append('child-exited')
            return SimpleNamespace(returncode=0)
        with historical.module_binding('run_contract', SimpleNamespace(load=lambda path: contract)), \
                historical.module_binding('artifact_paths', paths), \
                patch('subprocess.run', side_effect=run), patch('os.getuid', return_value=1000, create=True), \
                patch.object(historical, 'copy_files', side_effect=lambda *args: calls.append('copy') or {}):
            self.assertEqual(historical.guest_bootstrap(str(historical.GUEST_ENTRY), 'contract.json',
                contract.sha256, historical.COPY_LIMITS), 0)
        self.assertEqual(calls, ['parent', 'child-exited', 'output', 'copy'])

    def test_original_model_decoder_and_generic_contract_source_bytes_are_unchanged(self):
        expected = {
            'scripts/compatibility/amd_generation_decode.py': '8b9d66bba9390579f2c3b1d16136baa652dd0964e52ede4f95534b03a095b228',
            'experimental/extensions/trellis2-amd/inference_decoder/run_contract.py': 'b5897c709bb0d005476e854ebc2f1a33c874b934d9b16f0fd80de1a65ef7327d',
            'experimental/extensions/trellis2-amd/inference_decoder/decode.py': '122df9e57c756b44da3f73aeb8cae43dc2745ce78e46bbf2adc4a05b1e56cb19',
            'experimental/extensions/trellis2-amd/shape_decoder_loader.py': '8fa385af60aff19ddeabc305d01885abd4c4bf3b7a8a0d89d87b9397666bd571',
        }
        for name, expected_sha in expected.items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected_sha)


if __name__ == '__main__':
    unittest.main()
