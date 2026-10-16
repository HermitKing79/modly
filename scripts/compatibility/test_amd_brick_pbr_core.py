"""CPU qualification of the opt-in F16 brick core; never import a native model."""
import ast
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace, FunctionType, CodeType
from fractions import Fraction
import weakref
import gc

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'experimental/extensions/trellis2-amd/inference_texture'
sys.path.insert(0, str(PACKAGE.parent))


class CoreAvailability(unittest.TestCase):
    def test_authenticated_f16_core_is_present(self):
        for name in ('tiling.py', 'admission.py', 'runtime.py', 'neural.py'):
            self.assertTrue((PACKAGE / name).is_file(), 'missing opt-in F16 core: ' + name)


def authentic_profile():
    from inference_texture import profile
    contract = ROOT.parent / 'artifacts/amd-brick-neural-completion-c9447dd/neural-output/run-contract.json'
    return profile.authenticate(profile.NAME, json.loads(contract.read_bytes()))


PINNED_EXPLICIT = ROOT / 'docs/compatibility/amd-wsl-native-decoder-preparation-evidence/pinned-linux-submanifold_conv3d.py.txt'


def pinned_nodes():
    blob = PINNED_EXPLICIT.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == '67641d13e06ca135a4f90eb71a01f7d8d32448bb5aa7c31b858c69368811fb6b'
    return ast.parse(blob)


def original_cache_class():
    node = next(n for n in pinned_nodes().body if isinstance(n, ast.ClassDef)
                and n.name == 'SubMConv3dNeighborCache')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    namespace = {'__name__': 'flex_gemm.ops.spconv.submanifold_conv3d'}
    exec(compile(tree, str(PINNED_EXPLICIT), 'exec'), namespace)
    return namespace['SubMConv3dNeighborCache']


_TORCH = None
CPU_GUARD = dict(import_gpu_entries_blocked=0, post_import_gpu_entries_blocked=0,
                 gpu_initialized_after_import=None, gpu_initialized_after_tests=None)
NUMERICAL_OBSERVATIONS = []


def cpu_torch():
    global _TORCH
    if _TORCH is not None:
        return _TORCH
    os.environ.update(CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')
    # Windows platform probing may launch cmd /c ver when WMI is unavailable.
    # Supply the same OS facts from Python's direct API before forbidding children.
    if sys.platform == 'win32':
        import platform
        version = sys.getwindowsversion()
        platform._uname_cache = platform.uname_result('Windows', 'CPU-test', str(version.major),
            '.'.join(map(str, (version.major, version.minor, version.build))),
            os.environ.get('PROCESSOR_ARCHITECTURE', 'AMD64'))
    def audit(event, args):
        if event == 'import' and str(args[0]).split('.')[0] in {
                'flex_gemm', 'trellis2', 'triton', 'o_voxel', 'cumesh'}:
            raise RuntimeError('CPU core tests forbid native/model imports')
        if (event.startswith('socket.') and event != 'socket.gethostname') or event in ('subprocess.Popen', 'os.system'):
            raise RuntimeError('CPU core tests forbid network and child execution')
    sys.addaudithook(audit)
    previous_profile = sys.getprofile()
    def import_guard(frame, event, arg):
        if ((event == 'call' and frame.f_globals.get('__name__') == 'torch.cuda'
             and frame.f_code.co_name in ('init', '_lazy_init', 'set_device', 'device_count'))
            or (event == 'c_call' and getattr(arg, '__name__', '') in {
                '_cuda_init', '_cuda_setDevice', '_cuda_getDeviceCount',
                '_cuda_getDeviceProperties', '_cuda_getCurrentStream', '_cuda_synchronize'})):
            CPU_GUARD['import_gpu_entries_blocked'] += 1
            raise RuntimeError('CPU tests block GPU entry during Torch import')
    sys.setprofile(import_guard)
    try:
        import torch
    finally:
        sys.setprofile(previous_profile)
    CPU_GUARD['gpu_initialized_after_import'] = torch.cuda.is_initialized()
    if torch.cuda.is_initialized():
        raise RuntimeError('Torch imported with a GPU already initialized')
    def denied(*_args, **_kwargs):
        CPU_GUARD['post_import_gpu_entries_blocked'] += 1
        raise RuntimeError('CPU tests block GPU entry')
    for name in ('init', '_lazy_init', 'synchronize', 'current_device', 'current_stream',
                 'device_count', 'get_device_properties', 'mem_get_info', 'set_device'):
        setattr(torch.cuda, name, denied)
    for name in dir(torch._C):
        if name.startswith(('_cuda_', '_hip_')) and callable(getattr(torch._C, name)):
            setattr(torch._C, name, denied)
    _TORCH = torch
    return torch


def original_explicit(torch):
    pinned_nodes()  # Authenticate original source before compiling code only.
    algorithm = SimpleNamespace(EXPLICIT_GEMM='explicit_gemm')
    namespace = dict(torch=torch, Algorithm=algorithm,
                     spconv=SimpleNamespace(ALGORITHM='explicit_gemm', Algorithm=algorithm),
                     SubMConv3dNeighborCache=original_cache_class())
    return FunctionType(original_code(PINNED_EXPLICIT, 'SubMConv3dFunction',
                                     '_sparse_submanifold_conv_forward'), namespace, argdefs=(None,))


class F16Numerics(unittest.TestCase):
    def test_f16_matches_pinned_untiled_with_cross_tile_missing_neighbors_bias_and_tail(self):
        from inference_texture import tiling
        torch = cpu_torch()
        torch.manual_seed(1016)
        n, ci, co, tile = 11, 4, 5, 4
        feats = torch.randn(n, ci, dtype=torch.float16)
        weight = torch.randn(co, 3, 3, 3, ci, dtype=torch.float16).contiguous()
        neighbors = torch.randint(0, n, (n, 27), dtype=torch.int64)
        neighbors[:, 0] = 0xffffffff
        neighbors[0, 1], neighbors[10, 2], neighbors[4, 3] = 10, 0, 8
        neighbors = neighbors.to(torch.uint32).contiguous()
        original = original_explicit(torch)
        cache = original_cache_class()(neighbor_map=neighbors)
        for bias in (None, torch.randn(co, dtype=torch.float16)):
            observations = []
            actual = tiling.row_chunked_explicit(torch, feats, neighbors, weight, bias, tile,
                lambda phase, start, end: observations.append((phase, start, end)))
            expected = original(feats, cache, weight, bias)
            torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
            NUMERICAL_OBSERVATIONS.append(dict(dtype=str(actual.dtype), bias=bias is not None,
                rtol=0.002, atol=0.002, maximum_absolute_error=float((actual.float()-expected.float()).abs().max())))
            self.assertEqual(observations, [('entry', 0, 4), ('return', 0, 4),
                ('entry', 4, 8), ('return', 4, 8), ('entry', 8, 11), ('return', 8, 11)])
        self.assertFalse(torch.cuda.is_initialized())
        with self.assertRaisesRegex(RuntimeError, 'block GPU entry'):
            torch.cuda.init()

    def test_f16_cancellation_releases_live_output_and_tile_temporaries(self):
        from inference_texture import tiling
        torch = cpu_torch()
        feats = torch.zeros((7, 2), dtype=torch.float16)
        weight = torch.ones((3, 3, 3, 3, 2), dtype=torch.float16)
        neighbors = torch.zeros((7, 27), dtype=torch.uint32)
        output_refs = []
        original_empty = torch.empty
        def capture(*args, **kwargs):
            value = original_empty(*args, **kwargs)
            output_refs.append(weakref.ref(value))
            return value
        torch.empty = capture
        calls = []
        def cancel(phase, start, end):
            calls.append((phase, start, end))
            if phase == 'entry' and start == 3:
                raise KeyboardInterrupt('synthetic cancellation between tiles')
        try:
            try:
                tiling.row_chunked_explicit(torch, feats, neighbors, weight, None, 3, cancel)
            except KeyboardInterrupt as exc:
                exc.__traceback__ = None
        finally:
            torch.empty = original_empty
        gc.collect()
        self.assertEqual(calls, [('entry', 0, 3), ('return', 0, 3), ('entry', 3, 6)])
        self.assertTrue(output_refs and all(ref() is None for ref in output_refs))


class CoreContracts(unittest.TestCase):
    def test_original_neighbor_cache_class_is_accepted_and_duck_type_is_refused(self):
        from inference_texture import admission
        self.assertTrue(hasattr(admission, 'qualify_cache_type'), 'missing genuine cache-class qualification')
        cache_type = original_cache_class()
        self.assertIs(admission.qualify_cache_type(cache_type), cache_type)
        with self.assertRaises(admission.AdmissionStop):
            admission.qualify_cache_type(type('ForeignCache', (), {'__getitem__': lambda s, k: None}))

    def test_complete_byte_plans_keep_original_limits_and_exact_coverage(self):
        from inference_texture import admission
        p = authentic_profile()
        plans = admission.memory_plans(p)
        self.assertEqual(len(plans['convolution_plans']), 40)
        self.assertEqual(len(plans['c2s_plans']), 8)
        large = [x for x in plans['convolution_plans'] if x['site'] in admission.LARGE_TILED_SITES]
        self.assertEqual([x['rows'] for x in large], [438684]*6)
        self.assertEqual(sum(x['chunks'] for x in large), 84)
        self.assertEqual(plans['convolution_plans'][-1]['chunks'], 60)
        self.assertEqual(plans['convolution_plans'][-1]['last_chunk_rows'], 24579)
        self.assertEqual(plans['output_payload_bytes'], 125305024)
        self.assertEqual((admission.ALLOCATOR_LIMIT, admission.INITIAL_FREE_REQUIRED,
                          admission.FREE_RESERVE, admission.OPAQUE_HEADROOM),
                         (12*1024**3, 13*1024**3, 2*1024**3, 1024**3))
        with self.assertRaises(ValueError):
            admission.memory_plans('brick-1957891-v1')

    def test_admission_includes_retained_reserved_and_refuses_before_allocation(self):
        from inference_texture import admission
        hooks = admission.AdmissionHooks.__new__(admission.AdmissionHooks)
        hooks.rows = 438684
        events = []
        hooks.emit = events.append
        hooks.memory = lambda: dict(allocated_bytes=1, reserved_bytes=11*1024**3,
                                    free_bytes=30*1024**3, total_bytes=32*1024**3)
        with self.assertRaises(admission.AdmissionStop):
            hooks._admit('retained-test', {'visible_bytes': 4})
        self.assertFalse(events[-1]['admitted'])
        self.assertEqual(events[-1]['padded_bytes'], 5)


class MetaTensor:
    """Shape/device metadata only; never allocates or initializes a GPU."""
    def __init__(self, shape, dtype, device):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.ndim = len(self.shape)
    def is_contiguous(self):
        return True


class TiledLifecycle(unittest.TestCase):
    def fixture(self):
        from inference_texture import admission, tiling
        torch = cpu_torch()
        p = authentic_profile()
        original = original_explicit(torch)
        owner = type('SubMConv3dFunction', (), {'_sparse_submanifold_conv_forward': staticmethod(original)})
        cache_type = admission.qualify_cache_type(original_cache_class())
        device = torch.device('cuda', 0)
        coords = MetaTensor((438684, 4), torch.int32, device)
        neighbor = MetaTensor((438684, 27), torch.uint32, device)
        cache = cache_type(neighbor_map=neighbor)
        modules = {}
        for i, name in enumerate(admission.LARGE_TILED_SITES):
            module = torch.nn.Identity()
            module.weight = MetaTensor((512 if i == 5 else 128, 3, 3, 3, 128), torch.float16, device)
            module.bias = None
            modules[name] = module
        admissions = []
        hooks = SimpleNamespace(profile=p, cache_type=cache_type, active_conv=None,
            spconv=original.__globals__['spconv'],
            convolution_completed=33, up_states={3: {'completed': 2}},
            large_neighbor_map=neighbor, large_neighbor_coords=coords,
            large_conv_plan=None, final_neighbor_map=None, final_conv_plan=None,
            _admit=lambda name, plan: admissions.append((name, dict(plan))), memory=lambda: {})
        instance = tiling.LargeSupportConvolutionPatch(owner, modules, hooks, torch,
                                                       lambda *a, **k: None, profile=p)
        return SimpleNamespace(admission=admission, tiling=tiling, torch=torch, p=p, owner=owner,
            modules=modules, hooks=hooks, instance=instance, cache=cache, coords=coords,
            neighbor=neighbor, admissions=admissions, device=device)

    def enter(self, f, index):
        site = f.admission.LARGE_TILED_SITES[index]
        module = f.modules[site]
        x = SimpleNamespace(feats=MetaTensor((438684, 128), f.torch.float16, f.device),
                            coords=f.coords, spatial_shape=(256, 256, 256))
        f.hooks.active_conv = (site, module, x)
        co = 512 if index == 5 else 128
        extra = 438684*(32*128 if site.endswith('.conv') else 16*128+4*co)
        f.hooks.large_conv_plan = f.admission.large_chunked_conv_plan(438684, 128, co, 32768,
            256**3, index > 0, extra, profile=f.p)
        f.instance._enter(site, module, (x,))
        return site, module, x

    @staticmethod
    def metadata_helper(torch, feats, neighbor, weight, bias, tile, observe):
        for start in range(0, feats.shape[0], tile):
            end = min(feats.shape[0], start+tile)
            observe('entry', start, end)
            observe('return', start, end)
        return MetaTensor((feats.shape[0], weight.shape[0]), feats.dtype, feats.device)

    def test_changed_retained_feature_allowance_is_refused_before_helper(self):
        f = self.fixture()
        _, module, x = self.enter(f, 0)
        f.hooks.large_conv_plan['visible_bytes'] -= f.hooks.large_conv_plan['block_extra_bytes']
        f.hooks.large_conv_plan['block_extra_bytes'] = 0
        try:
            with patch.object(f.tiling, 'row_chunked_explicit') as helper:
                with self.assertRaises(f.admission.AdmissionStop):
                    f.instance.forward(x.feats, f.cache, module.weight, module.bias)
                helper.assert_not_called()
        finally:
            f.instance.remove()

    def test_full_six_site_coverage_global_cache_identity_and_allocations_are_simulated(self):
        f = self.fixture()
        descriptor = f.instance.original_descriptor
        f.instance.install()
        try:
            with patch.object(f.tiling, 'row_chunked_explicit', self.metadata_helper):
                for index in range(6):
                    site, module, x = self.enter(f, index)
                    if index == 1:
                        foreign = type(f.cache)(neighbor_map=f.neighbor)
                        with self.assertRaises(f.admission.AdmissionStop):
                            f.instance.forward(x.feats, foreign, module.weight, module.bias)
                    result = f.instance.forward(x.feats, f.cache, module.weight, module.bias)
                    f.hooks.convolution_completed += 1
                    f.hooks.active_conv = None
                    f.instance._exit(site, module, (x,), result)
                    self.assertIs(f.owner.__dict__[f.instance.name], descriptor)
            f.instance.assert_complete()
            self.assertEqual(f.instance.chunks, 84)
            self.assertEqual(f.instance.rows_completed, 6*438684)
            self.assertEqual(len(f.admissions), 6*(1+14))
            self.assertIn('output_bytes', f.admissions[0][1])
            self.assertNotIn('output_bytes', f.admissions[1][1])
            self.assertNotIn('neighbor_map_bytes', f.admissions[0][1])
            self.assertTrue(f.admissions[1][1]['block_extra_bytes'] > 0)
        finally:
            f.instance.remove()
        self.assertIsNone(f.instance.active)
        self.assertIsNone(f.instance.neighbor_cache)
        self.assertIsNone(f.instance.hooks)
        self.assertFalse(f.instance.modules or f.instance.handles)

    def test_cancel_restores_descriptor_and_clears_map_cache_references(self):
        f = self.fixture()
        descriptor = f.instance.original_descriptor
        site, module, x = self.enter(f, 0)
        def cancel(name, plan):
            if name.endswith('-tile-32768'):
                raise KeyboardInterrupt('synthetic control-path cancellation')
        f.hooks._admit = cancel
        try:
            with patch.object(f.tiling, 'row_chunked_explicit', self.metadata_helper):
                with self.assertRaises(KeyboardInterrupt):
                    f.instance.forward(x.feats, f.cache, module.weight, module.bias)
        finally:
            f.instance._exit(site, module, (x,), None)
            f.instance.remove()
        self.assertIs(f.owner.__dict__[f.instance.name], descriptor)
        self.assertIsNone(f.instance.neighbor_cache)
        self.assertIsNone(f.instance.active)
        self.assertEqual(f.instance.records[0]['chunks'], 1)
        self.assertFalse(f.instance.records[0]['completed'])

    def test_final_sixty_tiles_and_real_module_failure_hook_restore_are_simulated(self):
        f = self.fixture()
        f.instance.remove()
        module = f.torch.nn.Identity()
        module.weight = MetaTensor((64, 3, 3, 3, 64), f.torch.float16, f.device)
        module.bias = None
        coords = MetaTensor((1957891, 4), f.torch.int32, f.device)
        neighbor = MetaTensor((1957891, 27), f.torch.uint32, f.device)
        cache = type(f.cache)(neighbor_map=neighbor)
        x = SimpleNamespace(feats=MetaTensor((1957891, 64), f.torch.float16, f.device),
                            coords=coords, spatial_shape=(512,)*3)
        f.hooks.final_neighbor_map = neighbor
        f.hooks.final_conv_plan = f.admission.chunked_conv_plan(1957891, 32768, 512**3, profile=f.p)
        f.hooks.active_conv = (f.admission.FINAL_SITE, module, x)
        f.hooks.convolution_completed = 39
        instance = f.tiling.FinalConvolutionPatch(f.owner, module, f.hooks, f.torch,
                                                 lambda *a, **k: None, profile=f.p).install()
        descriptor = instance.original_descriptor
        def forward(value):
            result = getattr(f.owner, instance.name)(value.feats, cache, module.weight, module.bias)
            f.hooks.convolution_completed += 1
            return result
        module.forward = forward
        try:
            with patch.object(f.tiling, 'row_chunked_explicit', self.metadata_helper):
                module(x)
            instance.assert_complete()
            self.assertEqual((instance.chunks, instance.rows_completed), (60, 1957891))
            self.assertEqual(f.admissions[-1][0], 'blocks.3.4.conv2-tile-1933312')
            self.assertIs(f.owner.__dict__[instance.name], descriptor)
        finally:
            instance.remove()
        # A fresh genuine nn.Module forward hook restores on ordinary failure.
        f.hooks.convolution_completed = 39
        instance = f.tiling.FinalConvolutionPatch(f.owner, module, f.hooks, f.torch,
                                                 lambda *a, **k: None, profile=f.p).install()
        def failure(_value):
            raise RuntimeError('synthetic original-forward failure')
        module.forward = failure
        try:
            with self.assertRaisesRegex(RuntimeError, 'original-forward failure'):
                module(x)
            self.assertIs(f.owner.__dict__[instance.name], descriptor)
            self.assertIsNone(instance.active)
        finally:
            instance.remove()


class NativeProvenance(unittest.TestCase):
    def test_five_original_maps_order_coordinates_stream_and_restoration_are_simulated(self):
        from inference_texture import admission, runtime
        torch = cpu_torch()
        p = authentic_profile()
        device = torch.device('cuda', 0)
        binding_calls = []
        def binding(keys, vals, coordinates, *scalars):
            binding_calls.append(coordinates)
            return MetaTensor((coordinates.shape[0], 27), torch.uint32, device)
        names = (runtime.ALLOWED_NATIVE, *(f'forbidden_{i}' for i in range(17)))
        cuda = SimpleNamespace(**{name: binding for name in names})
        config = SimpleNamespace(FLEX_GEMM_ALGO='explicit_gemm', FLEX_GEMM_HASHMAP_RATIO=2.0)
        spconv = SimpleNamespace(ALGORITHM='explicit_gemm', HASHMAP_RATIO=2.0)
        fake_torch = SimpleNamespace(uint32=torch.uint32, uint64=torch.uint64, int32=torch.int32,
                                     cuda=SimpleNamespace(current_stream=lambda _: SimpleNamespace(cuda_stream=0)))
        hooks = SimpleNamespace(profile=p, active_conv=None, large_neighbor_map=None,
                                large_neighbor_coords=None, final_neighbor_map=None)
        boundary = runtime.NativeBoundary(cuda, names, hooks, fake_torch, config, spconv,
                                           lambda *a, **k: None, profile=p)
        old_profile = sys.getprofile()
        boundary.install()
        boundary.phase = 'neural'
        try:
            for i, expected in enumerate(runtime.expected_native_records(p)):
                n = expected['rows']
                coords = MetaTensor((n, 4), torch.int32, device)
                x = SimpleNamespace(coords=coords, feats=MetaTensor((n, 128), torch.float16, device),
                                    spatial_shape=(32*2**i,)*3)
                hooks.active_conv = (expected['site'], object(), x)
                keys = MetaTensor((2*n,), torch.uint32, device)
                vals = MetaTensor((2*n,), torch.uint32, device)
                args = (keys, vals, coords, *x.spatial_shape, 3, 3, 3, 1, 1, 1)
                if i == 0:
                    with self.assertRaises(RuntimeError):
                        boundary.call(names[1], args, {})
                    foreign = (keys, vals, MetaTensor((n, 4), torch.int32, device), *args[3:])
                    with self.assertRaises(RuntimeError):
                        boundary.call(names[0], foreign, {})
                    self.assertFalse(binding_calls)
                result = boundary.call(names[0], args, {})
                if i == 3:
                    self.assertIs(hooks.large_neighbor_map, result)
                    self.assertIs(hooks.large_neighbor_coords, coords)
            boundary.assert_complete()
            self.assertEqual(len(binding_calls), 5)
            with self.assertRaises(admission.AdmissionStop):
                boundary.call(names[0], args, {})
        finally:
            boundary.remove()
        self.assertIs(sys.getprofile(), old_profile)
        self.assertTrue(all(getattr(cuda, name) is binding for name in names))
        self.assertIsNone(boundary.hooks)


def receipt_fixture(profile, mode):
    """Handwritten TEST-ONLY receipt. Never serialized or called a model result."""
    from inference_texture import admission, runtime, tiling
    p = profile
    record = dict(pbr_profile=p.document(), profile_sha256=p.sha256,
        run_contract_sha256=p.run_contract_sha256, decoder_cpu='PASS', native_import='PASS',
        forward_calls=1, convolutions_completed=40, final_convolution_chunks=60,
        final_convolution_rows=p.final_rows, large_convolution_chunks=84, native_map_calls=5,
        native_map_records=runtime.expected_native_records(p),
        memory_release={'allocated': 0, 'reserved': 0}, workspace_clear_calls=1,
        runtime_dtype_counts={'F16': 208, 'F32': 76}, runtime_dtype_changes={}, buffers_count=0,
        decoder_diagnostics=dict(strict=True, roundtrip_verified=True, entries=284,
                                 dtype_counts={'F16': 208, 'F32': 76}, missing_keys=[], unexpected_keys=[]),
        policy_final={'rejected_calls': 0},
        large_convolution_records=[tiling._expected_record(s, p) for s in admission.LARGE_TILED_SITES],
        final_convolution_record=tiling._expected_record(admission.FINAL_SITE, p, final=True),
        subdivision_operations=[dict(site=f'blocks.{i}.{admission.COUNTS[i]}.updown', call=call,
                                      parents=p.parent_rows[i], children=p.child_rows[i])
                                for i in range(4) for call in (1, 2)],
        output=dict(verified=True, independently_verified_before_release=True,
                    metadata={'provenance': {'mode': mode, 'pbr_profile_document': json.dumps(
                        p.document(), sort_keys=True, separators=(',', ':')), 'pbr_profile_sha256': p.sha256}}),
        output_validation=dict(rows=p.final_rows, raw_shape=[p.final_rows, 6],
                features_shape=[p.final_rows, 6], coordinates_shape=[p.final_rows, 4], all_finite=True,
                geometry_coordinate_order_exact=True,
                original_transform='output * F32(0.5) + F32(0.5), CPU elementwise; no clamp', clamped=False),
        guide_order_validation=dict(verified=True, calls=[2, 2, 2, 2], final_geometry_order_exact=True))
    for key in ('large_convolution_patch_restored', 'final_convolution_patch_restored',
                'native_bindings_restored', 'convolution_hooks_removed', 'GPU_input_output_tensors_collected',
                'CPU_model_collected', 'CPU_parameters_collected', 'CPU_component_collected', 'constructed_on_meta'):
        record[key] = True
    return record


class ResultAcceptance(unittest.TestCase):
    def test_synthetic_field_cannot_qualify_as_genuine_neural_output(self):
        from inference_texture import admission, runtime
        p = authentic_profile()
        with self.assertRaises(admission.AdmissionStop):
            runtime.validate_result(receipt_fixture(p, 'synthetic'), profile=p)

    def test_test_only_receipt_acceptance_requires_all_cleanup_lineage_and_profile_fields(self):
        from inference_texture import admission, runtime
        p = authentic_profile()
        record = receipt_fixture(p, 'production')
        self.assertTrue(runtime.validate_result(record, profile=p))
        for mutation in ('native_map_calls', 'final_convolution_chunks', 'large_convolution_chunks',
                         'final_convolution_patch_restored', 'CPU_parameters_collected', 'profile_sha256'):
            changed = copy.deepcopy(record)
            changed[mutation] = None
            with self.assertRaises(admission.AdmissionStop, msg=mutation):
                runtime.validate_result(changed, profile=p)
        changed = copy.deepcopy(record)
        changed['subdivision_operations'][7]['children'] -= 1
        with self.assertRaises(admission.AdmissionStop):
            runtime.validate_result(changed, profile=p)


class NeuralComposition(unittest.TestCase):
    def test_foreign_injected_parent_forward_is_refused(self):
        from inference_texture import neural, admission
        with self.assertRaises(admission.AdmissionStop):
            neural._validate_dependencies(object(), SimpleNamespace(decoder_forward=lambda *a, **k: None))

    def test_authentic_inputs_low_initial_memory_refuses_before_model_placement_or_writer(self):
        from inference_texture import inputs, profile, neural, runtime, admission
        torch = cpu_torch()
        p = authentic_profile()
        prepared = inputs.prepare(p)
        parameters = {name: SimpleNamespace(shape=row['shape'],
            dtype=torch.float16 if row['dtype'] == 'F16' else torch.float32,
            device=torch.device('cpu'), requires_grad=False)
            for name, row in runtime.loader.schema(device='cpu').items()}
        model = SimpleNamespace(dtype=torch.float16, pred_subdiv=False, out_channels=6,
            low_vram=True, training=False, named_buffers=lambda: [],
            named_parameters=lambda: list(parameters.items()), parameters=lambda: iter(parameters.values()))
        fake_torch = SimpleNamespace(float16=torch.float16,
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                     cudnn=SimpleNamespace(allow_tf32=False)),
            cuda=SimpleNamespace(mem_get_info=lambda _: (12*1024**3, 16*1024**3),
                                 memory_allocated=lambda _: 0, memory_reserved=lambda _: 0))
        record = dict(decoder_cpu='PASS', native_import='PASS', runtime_dtype_counts={'F16':208,'F32':76},
                      runtime_dtype_changes={}, checkpoint={'freshly_streamed_sha256':profile.CHECKPOINT_SHA256})
        writer_calls = []
        with self.assertRaisesRegex(admission.AdmissionStop, '13GiB'):
            neural.run_neural(model, fake_torch, SimpleNamespace(), record, None, time.monotonic(),
                profile=p, prepared=prepared, write_output=lambda *a, **k: writer_calls.append(1),
                dependencies=SimpleNamespace())
        self.assertFalse(writer_calls)
        self.assertEqual(record['convolutions_completed'], 0)
        self.assertEqual(record['native_map_calls'], 0)
        self.assertTrue(record['GPU_input_output_tensors_collected'])
        self.assertTrue(record['final_convolution_patch_restored'])


def original_code(path, *names):
    code = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True)
    for name in names:
        code = [value for value in code.co_consts if isinstance(value, CodeType) and value.co_name == name][-1]
    return code


class ExecutableSourceIdentity(unittest.TestCase):
    """Source-code fixtures only: never import/construct a Trellis or Flex model."""
    def fixture(self):
        torch = cpu_torch()
        model_path = Path('D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD/trellis2/models/sc_vaes/sparse_unet_vae.py')
        sparse_path = model_path.parents[2] / 'modules/sparse/basic.py'
        model_globals = dict(__name__='trellis2.models.sc_vaes.sparse_unet_vae',
                             torch=torch, F=torch.nn.functional)
        chunked = FunctionType(original_code(model_path, 'chunked_apply'), model_globals)
        model_globals['chunked_apply'] = chunked
        forward = FunctionType(original_code(model_path, 'SparseUnetVaeDecoder', 'forward'),
                               model_globals, argdefs=(None, False))
        model_type = type('SparseUnetVaeDecoder', (), dict(__module__=model_globals['__name__'], forward=forward))
        config = SimpleNamespace(CONV='flex_gemm', DEBUG=False)
        sparse_globals = dict(__name__='trellis2.modules.sparse.basic', torch=torch,
                              config=config, Fraction=Fraction)
        initializer = FunctionType(original_code(sparse_path, 'SparseTensor', '__init__'), sparse_globals)
        sparse_type = type('SparseTensor', (), dict(__module__=sparse_globals['__name__'], __init__=initializer,
                                                   SparseTensorData=None))
        explicit = original_explicit(torch)
        explicit.__globals__['torch'] = torch
        owner = type('SubMConv3dFunction', (), dict(__module__='flex_gemm.ops.spconv.submanifold_conv3d',
                      _sparse_submanifold_conv_forward=staticmethod(explicit)))
        dependencies = SimpleNamespace(torch=torch, decoder_forward=forward, sparse=SimpleNamespace(SparseTensor=sparse_type),
            sparse_config=config, spconv=explicit.__globals__['spconv'], owner=owner)
        return SimpleNamespace(torch=torch, model=object.__new__(model_type), deps=dependencies,
            model_path=model_path, sparse_path=sparse_path, forward=forward, initializer=initializer)

    def test_original_executable_sources_pass_without_native_import_or_model_constructor(self):
        from inference_texture import neural
        f = self.fixture()
        self.assertIsNone(neural._validate_dependencies(f.model, f.deps))

    def test_changed_forward_with_genuine_filename_is_refused(self):
        from inference_texture import neural, admission
        f = self.fixture()
        namespace = dict(f.forward.__globals__)
        exec(compile('def forward(self, x, guide_subs=None, return_subs=False):\n    return x\n',
                     str(f.model_path), 'exec'), namespace)
        type(f.model).forward = f.deps.decoder_forward = namespace['forward']
        with self.assertRaises(admission.AdmissionStop):
            neural._validate_dependencies(f.model, f.deps)

    def test_changed_sparse_initializer_with_genuine_filename_is_refused(self):
        from inference_texture import neural, admission
        f = self.fixture()
        namespace = dict(f.initializer.__globals__)
        exec(compile('def __init__(self, *args, **kwargs):\n    self.data = {}\n',
                     str(f.sparse_path), 'exec'), namespace)
        f.deps.sparse.SparseTensor.__init__ = namespace['__init__']
        with self.assertRaises(admission.AdmissionStop):
            neural._validate_dependencies(f.model, f.deps)

    def test_exact_forward_code_with_changed_torch_global_is_refused(self):
        from inference_texture import neural, admission
        f = self.fixture()
        changed = FunctionType(f.forward.__code__, dict(f.forward.__globals__, torch=object()),
                               argdefs=(None, False))
        type(f.model).forward = f.deps.decoder_forward = changed
        with self.assertRaises(admission.AdmissionStop):
            neural._validate_dependencies(f.model, f.deps)

    def test_changed_nested_finalizer_is_refused_when_outer_bytecode_is_unchanged(self):
        from inference_texture import neural, admission
        f = self.fixture()
        constants = list(f.forward.__code__.co_consts)
        index = next(i for i, value in enumerate(constants)
                     if isinstance(value, CodeType) and value.co_name == 'fused_finalize')
        namespace = {}
        exec(compile('def fused_finalize(t):\n    return t\n', str(f.model_path), 'exec'), namespace)
        constants[index] = namespace['fused_finalize'].__code__
        code = f.forward.__code__.replace(co_consts=tuple(constants))
        self.assertEqual(code.co_code, f.forward.__code__.co_code)
        changed = FunctionType(code, f.forward.__globals__, argdefs=(None, False))
        type(f.model).forward = f.deps.decoder_forward = changed
        with self.assertRaises(admission.AdmissionStop):
            neural._validate_dependencies(f.model, f.deps)

    def test_changed_finalization_helper_global_is_refused(self):
        from inference_texture import neural, admission
        f = self.fixture()
        namespace = dict(f.forward.__globals__)
        exec(compile('def chunked_apply(module, x, chunk_size):\n    return x\n',
                     str(f.model_path), 'exec'), namespace)
        f.forward.__globals__['chunked_apply'] = namespace['chunked_apply']
        with self.assertRaises(admission.AdmissionStop):
            neural._validate_dependencies(f.model, f.deps)

    def test_changed_descriptor_with_genuine_filename_is_refused(self):
        from inference_texture import tiling, admission
        f = TiledLifecycle().fixture()
        f.instance.remove()
        namespace = dict(getattr(f.owner, '_sparse_submanifold_conv_forward').__globals__)
        exec(compile('def _sparse_submanifold_conv_forward(feats, neighbor_cache, weight, bias=None):\n    return feats\n',
                     str(PINNED_EXPLICIT), 'exec'), namespace)
        f.owner._sparse_submanifold_conv_forward = staticmethod(namespace['_sparse_submanifold_conv_forward'])
        with self.assertRaises(admission.AdmissionStop):
            tiling.LargeSupportConvolutionPatch(f.owner, f.modules, f.hooks, f.torch,
                                                lambda *a, **k: None, profile=f.p)


if __name__ == '__main__':
    result = unittest.main(verbosity=2, exit=False).result
    if _TORCH is not None:
        CPU_GUARD['gpu_initialized_after_tests'] = _TORCH.cuda.is_initialized()
    versions = {}
    for name in ('torch', 'triton-windows', 'triton'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    receipt = dict(schema='modly.amd.brick-pbr-core-cpu-tests.v1',
        classification='PASS_CPU_CORE_TESTS' if result.wasSuccessful() else 'FAIL_CPU_CORE_TESTS',
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        python=sys.version, python_executable=sys.executable, versions=versions,
        torch_runtime_version=str(_TORCH.__version__) if _TORCH is not None else None,
        cpu_guard=CPU_GUARD, numerical_observations=NUMERICAL_OBSERVATIONS,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_source_sha256='707d73d7331a4896bffeb851c2aa144175841cd4752a4497760d9f6a44ea2d39',
        native_imported=False, model_imported=False, weights_loaded=False, WSL_executed=False,
        fixture_scope='Actual bounded F16 CPU helper arithmetic; cache class compiled from original AST. Native calls, GPU device metadata and full tile lifecycle are explicitly simulated. Handwritten acceptance receipts are TEST-ONLY, never model outputs.',
        initial_incomplete_runs='Two Torch import attempts failed at local test guard: socket.gethostname, then blocked platform cmd /c ver probing. No native Flex/model/GPU operation ran. Fixed test-only platform cache from sys.getwindowsversion.')
    destination = ROOT / 'docs/compatibility/amd-brick-guided-pbr-cpu-evidence/core-cpu-tests.json'
    destination.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
