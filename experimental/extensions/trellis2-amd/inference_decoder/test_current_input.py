"""CPU-only current-image input authentication; no model/native/GPU imports."""
import copy
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / 'api'), str(ROOT / 'scripts/compatibility')]


class NoAI(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'torch', 'triton', 'trellis2', 'flex_gemm', 'transformers'}:
            raise AssertionError('Current-input qualification attempted AI import: ' + name)


def base_document():
    return {'schema': 'modly.amd.neural-run.v1', 'initial_count': 2,
            'input_paths': {'shape-latent-denormalized.npy': '/home/modly/source/shape.npy',
                            'coords.npy': '/home/modly/source/coords.npy'},
            'input_sha256': {'shape-latent-denormalized.npy': 'a'*64, 'coords.npy': 'b'*64},
            'source_job_sha256': 'c'*64,
            'output_directory': '/home/modly/run-current/geometry',
            'scratch_directory': '/home/modly/run-current/scratch',
            'repo_root': '/mnt/c/current-proof', 'maximum_final_rows': 1842968,
            'maximum_intermediate_rows': 1000000,
            'native_stage': '/home/modly/native-decoder-preparation-6dd94a85',
            'large_convolution_tiling': True}


def context_document():
    return {'job_id': 'current-job', 'request_id': 'current-request', 'run_id': 'current-run',
            'nonce': 'current-nonce', 'input_sha256': 'd'*64, 'params_sha256': 'e'*64,
            'created_ns': 1, 'mode': 'generated', 'model_id': 'trellis2-amd'}


class CurrentInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = NoAI()
        sys.meta_path.insert(0, cls.guard)

    @classmethod
    def tearDownClass(cls):
        sys.meta_path.remove(cls.guard)

    def setUp(self):
        self.assertTrue((HERE / 'current_input.py').is_file(), 'Separate current-image contract is missing')
        import run_contract
        self.run = run_contract
        self.current = self.run._current_api()

    def make(self, base=None, **binding):
        defaults = dict(product_context=context_document(), image_input_contract_sha256='f'*64,
                        initial_features_payload_sha256='1'*64, initial_coords_payload_sha256='2'*64)
        defaults.update(binding)
        return self.current.create(base_document() if base is None else base, **defaults)

    def provenance(self, contract):
        policy = self.current.binding_document(contract)['identity_policy']
        manifest = json.loads((ROOT / 'docs/compatibility/amd-wsl-neural-decoder-preparation-evidence/decoder-contract-readback.json').read_text())
        source = {'/mnt/' + name[0].lower() + '/' + name[3:].replace('\\', '/'): row['expected_sha256']
                  for name, row in manifest['source_checks'].items()}
        runtime = dict(policy['runtime'])
        runtime.pop('python_prefix')
        runtime['python'] = '3.12.10'
        runtime['gpu_initialized_before'] = False
        return dict(run_contract_sha256=contract.sha256, source_job_sha256=contract.source_job_sha256,
                    input_file_sha256=dict(contract.input_sha256), source=source,
                    checkpoint_sha256=policy['checkpoint_sha256'],
                    native_binary_sha256=policy['native_binary_sha256'],
                    algorithm='explicit_gemm', BF16_policy='216 body tensors after original strict F16/F32 verification',
                    runtime=runtime, **self.current.metadata(contract))

    def test_separate_wrapper_dispatch_preserves_generic_bounds_and_base(self):
        value = self.make()
        document = value.document()
        self.assertEqual(set(document), {'schema', 'base', 'current_input'})
        self.assertEqual(document['schema'], 'modly.amd.current-image-neural-run.v1')
        self.assertEqual(document['base'], base_document())
        self.assertEqual(self.run.parse(document).sha256, value.sha256)
        self.assertTrue(self.current.is_current(value))
        self.assertEqual(self.run.final_limit(value), 1842968)
        self.assertIsNone(value.final_support_profile)
        self.run.support_counts(16, (2, 2, 4, 8), value)
        with self.assertRaises(ValueError):
            self.run.support_counts(1842969, (2, 2, 4, 8), value)

    def test_nested_caller_and_returned_documents_cannot_mutate_binding(self):
        context = context_document()
        value = self.make(product_context=context)
        original = value.sha256
        context['job_id'] = 'different-job'
        document = value.document()
        document['current_input']['product_context']['job_id'] = 'other-job'
        document['current_input']['identity_policy']['runtime']['torch'] = 'foreign'
        document['base']['input_sha256']['coords.npy'] = '0'*64
        self.assertEqual(value.sha256, original)
        with self.assertRaises((TypeError, AttributeError)):
            value.current_input['product_context']['job_id'] = 'mutated'

    def test_rejects_unbound_profile_tiling_context_and_policy_expansion(self):
        document = self.make().document()
        mutations = [('base.initial_count', 4431), ('base.maximum_final_rows', 1957891),
                     ('base.large_convolution_tiling', False), ('base.final_support_profile', 'brick-1957891-v1'),
                     ('current_input.product_context.mode', 'replayed'),
                     ('current_input.product_context.model_id', 'other'),
                     ('current_input.product_context.created_ns', True),
                     ('current_input.identity_policy.native_binary_sha256', '0'*64),
                     ('current_input.identity_policy.runtime.torch', 'foreign'),
                     ('current_input.initial_coords_payload_sha256', 'invalid'),
                     ('current_input.gpu_allowed', True), ('worker_pass', True)]
        for path, replacement in mutations:
            changed = copy.deepcopy(document)
            keys = path.split('.')
            node = changed
            for key in keys[:-1]:
                node = node[key]
            node[keys[-1]] = replacement
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.run.parse(changed)

    def test_dynamic_reader_module_dispatch_does_not_need_sibling_on_path(self):
        document = self.make().document()
        spec = importlib.util.spec_from_file_location('_current_input_external_reader', HERE / 'run_contract.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        original = sys.path[:]
        try:
            sys.path[:] = [p for p in sys.path if p != str(HERE)]
            spec.loader.exec_module(module)
            value = module.parse(document)
            self.assertEqual(module.final_limit(value), 1842968)
            self.assertEqual(value.document(), document)
        finally:
            sys.path[:] = original
            sys.modules.pop(spec.name, None)

    def test_artifact_binding_rejects_cross_request_or_policy_provenance(self):
        value = self.make()
        valid = self.provenance(value)
        self.run.artifact_binding(valid, value)
        mutations = [('run_contract_sha256', '0'*64), ('source_job_sha256', '0'*64),
                     ('input_file_sha256.coords.npy', '0'*64), ('checkpoint_sha256', '0'*64),
                     ('native_binary_sha256', '0'*64), ('algorithm', 'implicit_gemm'),
                     ('BF16_policy', 'arbitrary precision'), ('runtime.torch', 'foreign'),
                     ('product_context.job_id', 'different-job'),
                     ('current_input.product_context.nonce', 'different-nonce'),
                     ('current_input.initial_features_payload_sha256', '0'*64)]
        for path, replacement in mutations:
            changed = copy.deepcopy(valid)
            if path == 'input_file_sha256.coords.npy':
                changed['input_file_sha256']['coords.npy'] = replacement
            else:
                keys = path.split('.')
                node = changed
                for key in keys[:-1]:
                    node = node[key]
                node[keys[-1]] = replacement
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.run.artifact_binding(changed, value)
        changed = copy.deepcopy(valid)
        changed['source'][next(iter(changed['source']))] = '0'*64
        with self.assertRaises(ValueError):
            self.run.artifact_binding(changed, value)
        with self.assertRaises(ValueError):
            self.run.artifact_binding({}, value)

    def test_current_contract_cannot_authorize_gpu_and_legacy_stays_available(self):
        self.run.require_execution_qualified(self.run.parse(base_document()))
        with self.assertRaisesRegex(ValueError, 'CPU-qualified only'):
            self.run.require_execution_qualified(self.make())

    def test_current_input_loader_authenticates_both_actual_payloads(self):
        import numpy as np
        features = np.arange(64, dtype=np.float32).reshape(2, 32)
        coords = np.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int32)
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            local = {}
            base = base_document()
            for name, array in zip(self.run.INPUT_NAMES, (features, coords)):
                path = Path(directory) / name
                np.save(path, array, allow_pickle=False)
                local[base['input_paths'][name]] = path
                base['input_sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            value = self.make(base, initial_features_payload_sha256=hashlib.sha256(features.tobytes()).hexdigest(),
                              initial_coords_payload_sha256=hashlib.sha256(coords.tobytes()).hexdigest())
            genuine_reader = self.run.read_npy
            def routed(path, expected, shape, dtype):
                return genuine_reader(local[str(path)], expected, shape, dtype)
            with patch.object(self.run, 'read_npy', side_effect=routed):
                a, c, summary = self.run.load_inputs(value)
                np.testing.assert_array_equal(a, features)
                np.testing.assert_array_equal(c, coords)
                self.assertEqual(summary['current_input'], self.current.binding_document(value))
                for field in ('initial_features_payload_sha256', 'initial_coords_payload_sha256'):
                    changed = value.document()
                    changed['current_input'][field] = '0'*64
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        self.run.load_inputs(self.run.parse(changed))

    def test_chain_rejects_valid_but_different_initial_coordinate_order(self):
        import numpy as np
        initial = np.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int32)
        stages = []
        for i in range(4):
            coords = initial.copy()
            coords[:, 1:] *= 2**i
            logits = np.full((2, 8), -1, dtype=np.float32)
            logits[:, 0] = 1
            stages.append((coords, logits))
        final = initial.copy()
        final[:, 1:] *= 16
        value = self.make(initial_coords_payload_sha256=hashlib.sha256(initial.tobytes()).hexdigest())
        self.assertTrue(self.run.validate_chain(stages, final, 2, contract=value))
        reordered = [(coords[::-1].copy(), logits[::-1].copy()) for coords, logits in stages]
        with self.assertRaises(ValueError):
            self.run.validate_chain(reordered, final[::-1].copy(), 2, contract=value)
        with self.assertRaises(ValueError):
            self.run.validate_chain(stages, final, 1, contract=value)

    def test_current_learned_mask_counts_are_dynamic_and_oversize_stops_before_allocation(self):
        import admission
        current = self.make()
        events = []
        facade = SimpleNamespace(bool='bool', int32='int32', equal=lambda left, right: left is right)
        memory = lambda: dict(allocated_bytes=0, reserved_bytes=0, free_bytes=13*admission.GIB)
        def exercise(parents, children):
            hooks = admission.AdmissionHooks(None, facade, contract=current,
                                             tile_large_convolutions=True, emit=events.append, memory=memory)
            hooks.sequence = [('next', None, None, 'conv2', 3)]
            hooks.up_states = {3: {'completed': 0}}
            coords = SimpleNamespace(dtype='int32', device='cpu')
            class Mask:
                shape = (parents, 8)
                dtype = 'bool'
                device = 'cpu'
                def sum(self):
                    return SimpleNamespace(item=lambda: children)
            sub = SimpleNamespace(feats=Mask(), coords=coords)
            x = SimpleNamespace(feats=SimpleNamespace(shape=(parents, 512), device='cpu'), coords=coords,
                                spatial_shape=(256, 256, 256), get_spatial_cache=lambda key: None)
            block = SimpleNamespace(out_channels=64, channels=128, _forward=lambda: None)
            module = SimpleNamespace(forward=lambda: None)
            with patch.object(hooks, '_sparse', return_value=(parents, 512)):
                hooks._up_pre(('blocks.3.4.updown', module, block, 3), module, (x, sub))
            return hooks
        hooks = exercise(7, 9)
        self.assertEqual(hooks.up_states[3]['children'], 9)
        self.assertTrue(events[-1]['admitted'])
        events.clear()
        with self.assertRaises(admission.AdmissionStop):
            exercise(300000, 1842969)
        self.assertEqual([row['event'] for row in events], ['c2s-count-before-admission'])
        self.assertEqual(events[0]['children'], 1842969)
        self.assertFalse(events[0]['bounded_fourth_support'])

    def test_current_convolution_pre_hook_refuses_memory_before_native_map_entry(self):
        import admission
        events = []
        facade = SimpleNamespace(bfloat16='BF16')
        hooks = admission.AdmissionHooks(None, facade, contract=self.make(), tile_large_convolutions=True,
            emit=events.append, memory=lambda: dict(allocated_bytes=12*admission.GIB,
                reserved_bytes=12*admission.GIB, free_bytes=13*admission.GIB))
        weight = SimpleNamespace(shape=(64, 3, 3, 3, 64), dtype='BF16', device='cuda:0')
        module = SimpleNamespace(weight=weight, bias=None, kernel_size=(3, 3, 3),
                                 dilation=(1, 1, 1), stride=(1, 1, 1))
        block = SimpleNamespace()
        site = ('blocks.3.4.conv2', module, block, 'conv2', 3)
        hooks.sequence = [site]*40
        hooks.convolution_completed = 39
        hooks.up_states = {3: {'completed': 2}}
        x = SimpleNamespace(feats=SimpleNamespace(device='cuda:0'), spatial_shape=(512, 512, 512),
                            get_spatial_cache=lambda key: None)
        with patch.object(hooks, '_sparse', return_value=(9, 64)):
            with self.assertRaises(admission.AdmissionStop):
                hooks._conv_pre(site, module, (x,))
        self.assertIsNone(hooks.active_conv)
        self.assertEqual(events[-1]['site'], 'blocks.3.4.conv2')
        self.assertFalse(events[-1]['admitted'])

    def test_current_outer_final_and_large_admission_precedes_full_output_helper(self):
        import admission
        import chunked_final
        import chunked_large
        facade = SimpleNamespace(bfloat16='BF16', device=lambda *args: 'cuda:0')
        current = self.make()
        for kind in ('final', 'large'):
            with self.subTest(kind=kind):
                events = []
                hooks = admission.AdmissionHooks(None, facade, contract=current, tile_large_convolutions=True,
                    emit=events.append, memory=lambda: dict(allocated_bytes=12*admission.GIB,
                        reserved_bytes=12*admission.GIB, free_bytes=13*admission.GIB))
                width = 64 if kind == 'final' else 128
                feats = SimpleNamespace(ndim=2, shape=(9, width), dtype='BF16', device='cuda:0')
                weight = SimpleNamespace(shape=(width, 3, 3, 3, width))
                module = SimpleNamespace(weight=weight, bias=None)
                coords, neighbor = object(), object()
                x = SimpleNamespace(feats=feats, coords=coords, spatial_shape=(5, 5, 5))
                site = chunked_final.SITE if kind == 'final' else admission.LARGE_TILED_SITES[0]
                hooks.active_conv = (site, module, x)
                hooks.up_states = {3: {'children': 9}}
                hooks.final_neighbor_map = hooks.large_neighbor_map = neighbor
                hooks.large_neighbor_coords = coords
                # Exercise the real allocation boundary with scalar-only state;
                # no constructor/source or real learned-geometry claim is made.
                owner = chunked_final.FinalConvolutionPatch if kind == 'final' else chunked_large.LargeSupportConvolutionPatch
                scoped = object.__new__(owner)
                scoped.hooks, scoped.torch, scoped.module = hooks, facade, module
                scoped.active = True if kind == 'final' else hooks.active_conv
                scoped.calls, scoped.records, scoped.invoked = 0, [], False
                scoped.tile_rows = admission.TILE_ROWS
                hooks.large_conv_plan = admission.large_chunked_conv_plan(9, 128, 128, 32768, 125)
                implementation = chunked_final if kind == 'final' else chunked_large
                with patch.object(implementation, 'row_chunked_explicit', side_effect=AssertionError('allocation reached')) as allocate:
                    with self.assertRaises(admission.AdmissionStop):
                        scoped.forward(feats, {'neighbor_map': neighbor}, weight, None)
                    allocate.assert_not_called()
                self.assertFalse(events[-1]['admitted'])
                self.assertIn('full-output-and-first-tile', events[-1]['site'])
                self.assertEqual(scoped.calls, 0)
                self.assertEqual(scoped.records, [])


if __name__ == '__main__':
    unittest.main()
