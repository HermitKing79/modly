"""CPU-only cache admission tests; the GPU boundary is a scalar-only fake."""
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('reclaim_admission', HERE/'admission.py')
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)

C4 = dict(allocated_bytes=1261897216, reserved_bytes=3546284032,
          free_bytes=11385540608, total_bytes=16974905344)
RECLAIMED = dict(allocated_bytes=1261897216, reserved_bytes=1280000000,
                 free_bytes=13651929088, total_bytes=16974905344)
PLAN = dict(visible_bytes=7555454532)


class CudaBoundary:
    def __init__(self, failure=None):
        self.calls = 0
        self.failure = failure

    def empty_cache(self):
        self.calls += 1
        if self.calls > 1:
            raise AssertionError('a second cache release was attempted')
        if self.failure:
            raise self.failure


class ReclaimTests(unittest.TestCase):
    def hooks(self, samples, enabled=True, failure=None):
        self.assertIn('reclaim_unused_cache', inspect.signature(a.AdmissionHooks).parameters,
                      'the bounded opt-in cache release is missing')
        cuda = CudaBoundary(failure)
        values = iter(samples)
        events = []
        hooks = a.AdmissionHooks(None, SimpleNamespace(cuda=cuda), emit=events.append,
            memory=lambda: dict(next(values)), reclaim_unused_cache=enabled)
        return hooks, cuda, events

    def test_c4_remeasurement_can_admit_without_changing_the_plan(self):
        hooks, cuda, events = self.hooks([C4, RECLAIMED])
        self.assertEqual(hooks._admit('blocks.2.8.conv2', PLAN), RECLAIMED)
        self.assertEqual(cuda.calls, 1)
        self.assertEqual([e['event'] for e in events],
                         ['admission', 'cache-reclaim-entry', 'cache-reclaim-result', 'admission'])
        self.assertFalse(events[0]['admitted'])
        self.assertTrue(events[-1]['admitted'])
        self.assertEqual(events[-1]['padded_bytes'], 9444318165)
        self.assertEqual(events[-1]['plan'], PLAN)
        self.assertEqual(hooks.cache_reclaims[0]['before'], C4)
        self.assertEqual(hooks.cache_reclaims[0]['after'], RECLAIMED)
        self.assertTrue(hooks.cache_reclaims[0]['admitted'])

    def test_still_insufficient_memory_stops_after_one_release(self):
        after = dict(RECLAIMED, free_bytes=11591801812)
        hooks, cuda, events = self.hooks([C4, after])
        with self.assertRaises(a.AdmissionStop): hooks._admit('bounded', PLAN)
        self.assertEqual(cuda.calls, 1)
        self.assertFalse(events[-1]['admitted'])
        self.assertFalse(hooks.cache_reclaims[0]['admitted'])

    def test_original_allocator_limit_still_rejects_after_release(self):
        after = dict(RECLAIMED, reserved_bytes=2366841900)
        hooks, cuda, _ = self.hooks([C4, after])
        with self.assertRaises(a.AdmissionStop): hooks._admit('bounded', PLAN)
        self.assertEqual(cuda.calls, 1)

    def test_default_remains_no_release(self):
        cuda = CudaBoundary()
        hooks = a.AdmissionHooks(None, SimpleNamespace(cuda=cuda), memory=lambda: dict(C4))
        with self.assertRaises(a.AdmissionStop): hooks._admit('legacy', PLAN)
        self.assertEqual(cuda.calls, 0)

    def test_already_admitted_and_no_unused_cache_do_not_release(self):
        for before, plan, should_pass in ((C4, dict(visible_bytes=1), True),
                (dict(C4, allocated_bytes=C4['reserved_bytes']), PLAN, False)):
            with self.subTest(should_pass=should_pass):
                hooks, cuda, _ = self.hooks([before])
                if should_pass: self.assertEqual(hooks._admit('bounded', plan), before)
                else:
                    with self.assertRaises(a.AdmissionStop): hooks._admit('bounded', plan)
                self.assertEqual(cuda.calls, 0)
                self.assertEqual(hooks.cache_reclaims, [])

    def test_live_allocation_change_is_rejected_even_if_limits_would_pass(self):
        for allocated in (C4['allocated_bytes']-1, C4['allocated_bytes']+1):
            with self.subTest(allocated=allocated):
                hooks, cuda, events = self.hooks([C4, dict(RECLAIMED, allocated_bytes=allocated)])
                with self.assertRaisesRegex(a.AdmissionStop, 'live allocation'):
                    hooks._admit('bounded', PLAN)
                self.assertEqual(cuda.calls, 1)
                self.assertFalse(hooks.cache_reclaims[0]['admitted'])
                self.assertEqual(events[-1]['event'], 'cache-reclaim-result')

    def test_increased_reserved_memory_is_rejected(self):
        hooks, cuda, _ = self.hooks([C4, dict(RECLAIMED, reserved_bytes=C4['reserved_bytes']+1)])
        with self.assertRaisesRegex(a.AdmissionStop, 'reserved'):
            hooks._admit('bounded', PLAN)
        self.assertEqual(cuda.calls, 1)

    def test_release_error_records_failure_and_never_retries(self):
        hooks, cuda, events = self.hooks([C4], failure=RuntimeError('release failed'))
        with self.assertRaisesRegex(a.AdmissionStop, 'cache release failed'):
            hooks._admit('bounded', PLAN)
        self.assertEqual(cuda.calls, 1)
        self.assertEqual(events[-1]['event'], 'cache-reclaim-result')
        self.assertEqual(hooks.cache_reclaims[0]['error'], 'RuntimeError: release failed')
        self.assertFalse(hooks.cache_reclaims[0]['admitted'])

    def test_remeasurement_error_does_not_release_again(self):
        hooks, cuda, events = self.hooks([C4])
        with self.assertRaisesRegex(a.AdmissionStop, 'cache release failed'):
            hooks._admit('bounded', PLAN)
        self.assertEqual(cuda.calls, 1)
        self.assertEqual(events[-1]['event'], 'cache-reclaim-result')
        self.assertFalse(hooks.cache_reclaims[0]['admitted'])

    def test_conv39_required_live_buffers_fail_even_with_zero_unused_cache(self):
        plan = a.conv_plan(438684, 128, 512, 12156928, True, 1796849664)
        self.assertEqual(plan['visible_bytes'], 8511785652)
        # Measured model parameter payload plus original x and distinct h.
        lower_bound = 948598492 + 112303104 + 112303104
        self.assertEqual(lower_bound, 1173204700)
        memory = dict(allocated_bytes=lower_bound, reserved_bytes=lower_bound,
                      free_bytes=16*a.GIB, total_bytes=16*a.GIB)
        hooks, cuda, events = self.hooks([memory])
        with self.assertRaises(a.AdmissionStop): hooks._admit('blocks.3.4.conv1', plan)
        self.assertEqual(cuda.calls, 0)
        self.assertEqual(events[-1]['padded_bytes'], 10639732065)
        self.assertFalse(events[-1]['admitted'])


if __name__ == '__main__':
    unittest.main()
