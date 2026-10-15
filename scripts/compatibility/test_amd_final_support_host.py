"""Focused host profile tests: CPU reads only, no model/native or WSL entry."""
from contextlib import ExitStack
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class NoAI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
                'torch', 'triton', 'trellis2', 'flex_gemm', 'transformers',
                'cumesh', 'o_voxel', 'nvdiffrast'}:
            raise AssertionError('Unexpected model/native import: '+fullname)


sys.meta_path.insert(0, NoAI())
import amd_generation_decode as subject


PROFILE = 'brick-1957891-v1'
SOURCE_SHA = 'b92aade7aca2e832fbf425ec9d4e54d40aa7ffa4b4afdea54c38f71f61b8e62f'
INPUT_SHA = {
    'shape-latent-denormalized.npy': 'e6a6eec09029db43b7b2356ee71a943ad2f225c45d48a03bfaa519a031b8c808',
    'coords.npy': '17a8ddc94edaa9da5b02834157eef9f8d4f7f99bfa8e2f4d2b5806f1b1cfe93c'}
MEASUREMENT_SHA = '03d66c89933a5c400c6d825173149c0fa75a8c747d4d5d7ac738ae83571a2b34'


class StopBeforeWeights(RuntimeError):
    pass


class ReaderReached(RuntimeError):
    pass


class FinalSupportHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shape = subject.ROOT/'docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/occupancy/shape'
        # Reopen the genuine source once; all constructor cases use this bundle.
        cls.bundle = subject.shape_inputs(cls.shape)
        if cls.bundle['count'] != 4430 or cls.bundle['source_job_sha256'] != SOURCE_SHA:
            raise AssertionError('Saved genuine brick shape identity differs')
        if {k:v['sha256'] for k,v in cls.bundle['inputs'].items()} != INPUT_SHA:
            raise AssertionError('Saved genuine brick input hashes differ')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='_final-support-host-', dir=subject.APPROVED_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evidence = self.root/'decode'
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(subject, 'shape_inputs', return_value=self.bundle))

    def runtime(self, **kwargs):
        return subject.Runtime(self.shape, self.evidence, 'a'*40,
                               'run-final-support-host-test', **kwargs)

    def profile_runtime(self):
        return self.runtime(tile_large_convolutions=True, final_support_profile=PROFILE)

    def test_genuine_brick_constructor_binds_profile_maximum_and_canonical_hash(self):
        runtime = self.profile_runtime()
        contract = runtime.contract
        self.assertEqual(contract.final_support_profile, PROFILE)
        self.assertEqual(contract.maximum_final_rows, 1957891)
        self.assertEqual(contract.source_job_sha256, SOURCE_SHA)
        self.assertEqual(dict(contract.input_sha256), INPUT_SHA)
        self.assertTrue(contract.large_convolution_tiling)
        self.assertEqual(contract.initial_count, 4430)
        encoded = json.dumps(contract.document(), sort_keys=True, separators=(',', ':')).encode()
        self.assertEqual(contract.sha256, hashlib.sha256(encoded).hexdigest())
        old = self.runtime(tile_large_convolutions=True).contract
        self.assertNotEqual(contract.sha256, old.sha256)
        self.assertFalse(self.evidence.exists())

    def test_no_profile_preserves_the_original_final_support_limit(self):
        runtime = self.runtime()
        self.assertEqual(runtime.contract.maximum_final_rows, 1842968)
        self.assertIsNone(runtime.contract.final_support_profile)
        self.assertNotIn('final_support_profile', runtime.contract.document())
        self.assertFalse(runtime.contract.large_convolution_tiling)
        self.assertFalse(self.evidence.exists())

    def test_foreign_source_hash_count_profile_or_disabled_tiling_rejects_before_marker(self):
        cases = [('source_job', dict(self.bundle, source_job_sha256='0'*64)),
                 ('count', dict(self.bundle, count=4429))]
        for name in INPUT_SHA:
            inputs = {k:dict(v) for k,v in self.bundle['inputs'].items()}
            inputs[name]['sha256'] = '0'*64
            cases.append((name, dict(self.bundle, inputs=inputs)))
        for name, bundle in cases:
            with self.subTest(name=name), patch.object(subject, 'shape_inputs', return_value=bundle):
                with self.assertRaises(ValueError): self.profile_runtime()
                self.assertFalse(self.evidence.exists())
        for options in ({'tile_large_convolutions':False, 'final_support_profile':PROFILE},
                        {'tile_large_convolutions':True, 'final_support_profile':'brick-foreign'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.runtime(**options)
            self.assertFalse(self.evidence.exists())

    def preflight_stubs(self, runtime, verifier):
        stack = ExitStack()
        stack.enter_context(patch.dict('os.environ', {'WORKSPACE_DIR':str(subject.ROOT)}))
        stack.enter_context(patch.object(subject.subprocess, 'check_output', return_value=runtime.commit+'\n'))
        stack.enter_context(patch.object(runtime, 'host_admission', return_value={'admitted':True}))
        stack.enter_context(patch.object(verifier, 'verify_committed_files', side_effect=StopBeforeWeights('closure reached')))
        stack.enter_context(patch.object(runtime, 'reserve_attempt', side_effect=AssertionError('marker forbidden')))
        return stack

    def test_changed_pinned_measurement_rejects_before_committed_closure_and_marker(self):
        import amd_sparse_decoder_forward as verifier
        runtime = self.profile_runtime()
        measurement = subject.ROOT/runtime.contract_module.BRICK_MEASUREMENT
        real_digest = subject.digest
        with self.preflight_stubs(runtime, verifier), patch.object(subject, 'digest',
                side_effect=lambda path:'0'*64 if Path(path)==measurement else real_digest(path)):
            with self.assertRaisesRegex(ValueError, 'qualification evidence changed'):
                runtime.preflight()
            verifier.verify_committed_files.assert_not_called()
            runtime.reserve_attempt.assert_not_called()
        self.assertFalse(self.evidence.exists())

    def test_actual_pinned_measurement_enters_committed_closure_before_weights_or_marker(self):
        import amd_sparse_decoder_forward as verifier
        runtime = self.profile_runtime()
        measured = runtime.contract_module.BRICK_MEASUREMENT
        self.assertEqual(subject.digest(subject.ROOT/measured), MEASUREMENT_SHA)
        with self.preflight_stubs(runtime, verifier):
            with self.assertRaises(StopBeforeWeights): runtime.preflight()
            call = verifier.verify_committed_files.call_args
            self.assertEqual(call.args[0], runtime.commit)
            self.assertIn(measured, call.args[1])
            runtime.reserve_attempt.assert_not_called()
        self.assertFalse(self.evidence.exists())

    def readback_fixture(self, runtime):
        target = runtime.evidence/'run/guest'
        target.mkdir(parents=True)
        saved = dict(runtime.bundle['summary'], input_sha256=dict(runtime.contract.input_sha256))
        provenance = {'run_contract_sha256':runtime.contract.sha256,
            'source_job_sha256':SOURCE_SHA, 'checkpoint_sha256':subject.CHECKPOINT_SHA,
            'input_file_sha256':dict(runtime.contract.input_sha256), 'inputs':saved,
            'final_support_profile':PROFILE}
        artifact = target/'shape-decoded.safetensors'
        artifact.write_bytes(b'CPU reader routing sentinel; never treated as genuine output')
        output = {'bytes':artifact.stat().st_size, 'sha256':subject.digest(artifact),
                  'path':str(runtime.contract.output_directory/artifact.name)}
        worker = {'classification':'QUALIFIED', 'saved_inputs':saved, 'output':output}
        documents = {'results.json':worker, 'run-contract.json':runtime.contract.document(),
            'provenance.json':{'run_contract':runtime.contract.document(),
                               'output':output, 'provenance':provenance}}
        for name, value in documents.items():
            (target/name).write_text(json.dumps(value), encoding='utf8')
        return artifact, worker

    def test_host_readback_forwards_the_authenticated_profile_contract_to_cpu_reader(self):
        import amd_generation_mesh as mesh
        runtime = self.profile_runtime()
        artifact, worker = self.readback_fixture(runtime)
        with patch.object(runtime.guest, 'validate_qualified') as validator, \
                patch.object(mesh, 'read_neural', side_effect=ReaderReached('CPU boundary reached')) as reader:
            with self.assertRaises(ReaderReached): runtime.verify_outputs()
            validator.assert_called_once_with(worker, runtime.contract)
            reader.assert_called_once_with(artifact, worker['output']['sha256'], contract=runtime.contract)

    def test_profile_provenance_is_checked_before_cpu_output_reader(self):
        import amd_generation_mesh as mesh
        runtime = self.profile_runtime()
        self.readback_fixture(runtime)
        sidecar = runtime.evidence/'run/guest/provenance.json'
        row = subject.read_json(sidecar)
        row['provenance']['final_support_profile'] = 'foreign-profile'
        sidecar.write_text(json.dumps(row), encoding='utf8')
        with patch.object(runtime.guest, 'validate_qualified'), patch.object(mesh, 'read_neural') as reader:
            with self.assertRaisesRegex(ValueError, 'artifact provenance differs'):
                runtime.verify_outputs()
            reader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
