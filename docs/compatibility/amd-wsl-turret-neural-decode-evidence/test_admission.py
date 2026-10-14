"""CPU metadata tests: no Torch import, model loading, or native execution."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).parent))
try:
    import admission as a
except ModuleNotFoundError:
    a = None


class Tensor:
    def __init__(self, shape, dtype='bfloat16', value=None):
        self.shape, self.dtype, self.value = shape, dtype, value
        self.device = NS(type='cuda', index=0)
        self.sums = 0

    def is_contiguous(self):
        return True

    def sum(self):
        self.sums += 1
        return NS(item=lambda: self.value)


class Module:
    def __init__(self, callback):
        self.pre, self.post, self.callback, self.calls = [], [], callback, 0

    def register_forward_pre_hook(self, hook):
        self.pre.append(hook)
        return NS(remove=lambda: self.pre.remove(hook))

    def register_forward_hook(self, hook):
        self.post.append(hook)
        return NS(remove=lambda: self.post.remove(hook))

    def __call__(self, *args):
        for hook in tuple(self.pre):
            hook(self, args)
        self.calls += 1
        out = self.callback(*args)
        for hook in tuple(self.post):
            hook(self, args, out)
        return out


class SparseConv3d(Module):
    def __init__(self, ci, co):
        super().__init__(lambda x: sparse(x.feats.shape[0], co, coords=x.coords, cache=x.cache))
        self.in_channels, self.out_channels = ci, co
        self.kernel_size, self.dilation, self.stride = (3, 3, 3), (1, 1, 1), (1, 1, 1)
        self.weight, self.bias = Tensor((co, 3, 3, 3, ci)), Tensor((co,))


class SparseChannel2Spatial(Module):
    def __init__(self):
        super().__init__(lambda x, sub: sparse(sub.feats.value, x.feats.shape[1] // 8))
        self.factor = 2


def sparse(n, c, coords=None, cache=None):
    return NS(feats=Tensor((n, c)), coords=coords or Tensor((n, 4), 'int32', 'coords'),
              shape=(1, c), spatial_shape=(64, 64, 64), cache=cache,
              get_spatial_cache=lambda key: cache)


def decoder():
    blocks, named = [], []
    widths = [1024, 512, 256, 128, 64]
    for i, count in enumerate([4, 16, 8, 4, 0]):
        stage = []
        for j in range(count):
            conv = SparseConv3d(widths[i], widths[i])
            block = NS(conv=conv, mlp=[NS(out_features=widths[i] * 4)])
            stage.append(block)
            named.append((f'blocks.{i}.{j}.conv', conv))
        if i < 4:
            up = NS(channels=widths[i], out_channels=widths[i+1], pred_subdiv=True,
                    conv1=SparseConv3d(widths[i], widths[i+1] * 8),
                    conv2=SparseConv3d(widths[i+1], widths[i+1]), updown=SparseChannel2Spatial())
            stage.append(up)
            named.extend((f'blocks.{i}.{count}.{key}', getattr(up, key))
                         for key in ('conv1', 'updown', 'conv2'))
        blocks.append(stage)
    return NS(blocks=blocks, model_channels=widths, num_blocks=[4, 16, 8, 4, 0],
              pred_subdiv=True, named_modules=lambda: iter(named))


TORCH = NS(bfloat16='bfloat16', int32='int32', uint32='uint32', bool='bool',
           equal=lambda x, y: x.shape == y.shape and x.value == y.value)
GIB = 1024**3


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(a, 'admission implementation is missing')
        self.model = decoder()
        self.mem = dict(free_bytes=16*GIB, allocated_bytes=GIB, reserved_bytes=GIB)
        self.guard = a.AdmissionHooks(self.model, TORCH, memory=lambda: dict(self.mem)).install()

    def tearDown(self):
        if hasattr(self, 'guard'):
            self.guard.remove()

    def test_explicit_gemm_counts_simultaneous_gather_and_uint64_hash(self):
        p = a.conv_plan(10, 8, 4, 10)
        self.assertEqual(p['visible_bytes'], 14550)
        self.assertEqual(p['im2col_bytes'], 4320)
        self.assertEqual(p['gathered_bytes'], 4320)
        self.assertEqual(a.conv_plan(10, 8, 4, 2**32)['visible_bytes'], 14630)
        self.assertEqual(a.conv_plan(10, 8, 4, 10, cached=True)['visible_bytes'], 13310)

    def test_row_and_volume_overflow_fail_closed(self):
        for args in ((0, 8, 4, 10), (1_000_001, 8, 4, 10), (10, 8, 4, 2**64)):
            with self.assertRaises(a.AdmissionStop):
                a.conv_plan(*args)
        for children in (0, 81, 1_000_001):
            with self.assertRaises(a.AdmissionStop):
                a.c2s_plan(10, 64, 32, children, 8)
        self.assertEqual(a.c2s_plan(10, 64, 32, 80, 8)['visible_bytes'], 28160)

    def test_malformed_feature_weight_and_worst_rows_gate_callback(self):
        conv = self.model.blocks[0][0].conv
        x = sparse(10, 1024)
        x.feats.dtype = 'float32'
        with self.assertRaises(a.AdmissionStop):
            conv(x)
        conv.weight.dtype = 'float32'
        with self.assertRaises(a.AdmissionStop):
            conv(sparse(10, 1024))
        conv.weight.dtype = 'bfloat16'
        with self.assertRaises(a.AdmissionStop):
            conv(sparse(1_000_000, 1024))
        self.assertEqual(conv.calls, 0)

    def test_negative_extra_cannot_reduce_memory_bound(self):
        with self.assertRaises(a.AdmissionStop):
            a.conv_plan(10, 8, 4, 10, block_extra=-14550)

    def test_rejection_blocks_original_callback_with_partial_status(self):
        self.mem['free_bytes'] = 2*GIB
        conv = self.model.blocks[0][0].conv
        with self.assertRaisesRegex(a.AdmissionStop, 'PARTIAL'):
            conv(sparse(3743, 1024))
        self.assertEqual(conv.calls, 0)
        self.assertIsNone(self.guard.active_conv)

    def test_allocator_headroom_and_reserved_bytes_gate_callback(self):
        self.mem['reserved_bytes'] = 11*GIB
        conv = self.model.blocks[0][0].conv
        with self.assertRaises(a.AdmissionStop):
            conv(sparse(3743, 1024))
        self.assertEqual(conv.calls, 0)

    def test_active_native_confinement_and_post_counter(self):
        conv = self.model.blocks[0][0].conv
        original = conv.callback
        seen = []
        conv.callback = lambda x: (seen.append(self.guard.active_conv), original(x))[1]
        x = sparse(3743, 1024)
        conv(x)
        self.assertEqual(seen, [('blocks.0.0.conv', conv, x)])
        self.assertIsNone(self.guard.active_conv)
        self.assertEqual(self.guard.convolution_completed, 1)

    def test_conv_order_and_bad_cache_gate_actual_callback(self):
        second = self.model.blocks[0][1].conv
        with self.assertRaises(a.AdmissionStop):
            second(sparse(10, 1024))
        self.assertEqual(second.calls, 0)
        first = self.model.blocks[0][0].conv
        for shape, dtype in (((10, 26), 'uint32'), ((10, 27), 'int32')):
            with self.assertRaises(a.AdmissionStop):
                first(sparse(10, 1024, cache=NS(neighbor_map=Tensor(shape, dtype))))
        self.assertEqual(first.calls, 0)
        first(sparse(10, 1024, cache=NS(neighbor_map=Tensor((10, 27), 'uint32'))))
        self.assertEqual(first.calls, 1)

    def reach_up(self):
        x = sparse(10, 1024)
        for block in self.model.blocks[0][:-1]:
            x = block.conv(x)
        return x, self.model.blocks[0][-1], self.model.blocks[0][-1].conv1(x)

    def test_both_c2s_calls_sum_authentic_mask_and_gate_second(self):
        x, up, h = self.reach_up()
        sub = sparse(10, 8, coords=x.coords)
        sub.feats = Tensor((10, 8), 'bool', 40)
        up.updown(h, sub)
        self.mem['free_bytes'] = 2*GIB
        with self.assertRaises(a.AdmissionStop):
            up.updown(x, sub)
        self.assertEqual(sub.feats.sums, 2)
        self.assertEqual(up.updown.calls, 1)
        self.assertEqual(len(self.guard.subdivision_records), 1)

    def test_changed_mask_or_coords_gate_second_and_conv2(self):
        x, up, h = self.reach_up()
        sub = sparse(10, 8, coords=x.coords)
        sub.feats = Tensor((10, 8), 'bool', 40)
        out = up.updown(h, sub)
        with self.assertRaises(a.AdmissionStop):
            up.conv2(out)
        changed = sparse(10, 8, coords=x.coords)
        changed.feats = Tensor((10, 8), 'bool', 40)
        with self.assertRaises(a.AdmissionStop):
            up.updown(x, changed)
        sub.coords = Tensor((10, 4), 'int32', 'different')
        with self.assertRaises(a.AdmissionStop):
            up.updown(x, sub)
        self.assertEqual(up.updown.calls, 1)

    def test_real_mask_zero_and_overflow_reject_before_expansion(self):
        x, up, h = self.reach_up()
        sub = sparse(10, 8, coords=x.coords)
        sub.feats = Tensor((10, 8), 'bool', 0)
        for value in (0, 81):
            sub.feats.value = value
            with self.assertRaises(a.AdmissionStop):
                up.updown(h, sub)
        self.assertEqual(sub.feats.sums, 2)
        self.assertEqual(up.updown.calls, 0)

    def test_original_all_40_and_eight_c2s_calls_complete_without_rewrite(self):
        x = sparse(10, 1024)
        for stage in self.model.blocks[:4]:
            for block in stage[:-1]:
                x = block.conv(x)
            up = stage[-1]
            sub = sparse(x.feats.shape[0], 8, coords=x.coords)
            sub.feats = Tensor((x.feats.shape[0], 8), 'bool', x.feats.shape[0]*2)
            h = up.updown(up.conv1(x), sub)
            up.updown(x, sub)
            x = up.conv2(h)
        self.guard.assert_complete()
        self.assertEqual(self.guard.convolution_completed, 40)
        self.assertEqual(len(self.guard.subdivision_records), 8)
        self.assertEqual(self.guard.rows, 160)

    def test_removal_restores_original_callbacks_and_other_hooks(self):
        conv = self.model.blocks[0][0].conv
        original = conv.callback
        seen = []
        extra = conv.register_forward_pre_hook(lambda m, args: seen.append(1))
        self.guard.remove()
        self.assertIs(conv.callback, original)
        conv(sparse(10, 1024))
        self.assertEqual(seen, [1])
        self.assertEqual(conv.calls, 1)
        extra.remove()


if __name__ == '__main__':
    unittest.main()
