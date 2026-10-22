"""Pure contract tests, frozen before the one owned native micro-probe."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('counter_contract', Path(__file__).with_name('counter_probe.py'))
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)

def state(values, ok=None):
    if ok is None: ok = len(values) == 1
    getter = {'ok': True, 'value': values[0] if values else 0} if ok else {'ok': False, 'error_type': 'RuntimeError', 'error': 'Expected exactly one value.'}
    return {'length': len(values), 'values': values, 'getter': getter}

def bound():
    return {'initial_sentinel': -12345, 'mutated_sentinel': -23456,
            'holder_state': state([-23456]), 'settings_before': -12345, 'settings_after': -23456}

def output():
    return {'before': -23456, 'after': state([0]), 'settings_after': 0,
            'failed_faces': [], 'valid_faces': [0], 'valid_vertices': [0, 1, 2],
            'vertices': [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
            'faces': [[0, 1, 2]], 'constructor_calls': 1}

class CounterContract(unittest.TestCase):
    def reject(self, fn, value):
        with self.assertRaises(ValueError): fn(value)

    def test_single_storage(self):
        self.assertTrue(contract.validate_state(state([7])))
    def test_empty_getter_rejected(self):
        self.assertTrue(contract.validate_state(state([])))
    def test_multiple_getter_rejected(self):
        self.assertTrue(contract.validate_state(state([7, 8])))
    def test_getter_type_is_observation(self):
        row = state([]); row['getter']['error_type'] = 'ValueError'
        self.assertTrue(contract.validate_state(row))
    def test_wrong_scalar(self):
        row = state([7]); row['getter']['value'] = 8
        self.reject(contract.validate_state, row)
    def test_single_getter_must_succeed(self):
        self.reject(contract.validate_state, state([7], False))
    def test_empty_cannot_read_default(self):
        self.reject(contract.validate_state, state([], True))
    def test_multiple_cannot_read_first(self):
        self.reject(contract.validate_state, state([7, 8], True))
    def test_storage_length(self):
        row = state([7]); row['length'] = 0
        self.reject(contract.validate_state, row)
    def test_boolean_not_native_int(self):
        for key in ('length',):
            row = state([7]); row[key] = True
            self.reject(contract.validate_state, row)
        self.reject(contract.validate_state, state([True]))
    def test_storage_bound(self):
        self.reject(contract.validate_state, state(list(range(9))))
    def test_no_empty_error_description(self):
        row = state([]); row['getter']['error'] = ''
        self.reject(contract.validate_state, row)
    def test_live_bound_alias(self):
        self.assertTrue(contract.validate_bound(bound()))
    def test_detached_python_default_rejected(self):
        row = bound(); row['settings_after'] = -12345
        self.reject(contract.validate_bound, row)
    def test_bound_holder_scalar_coherence(self):
        row = bound(); row['holder_state'] = state([0])
        self.reject(contract.validate_bound, row)
    def test_bound_real_integer_required(self):
        row = bound(); row['settings_after'] = -23456.0
        self.reject(contract.validate_bound, row)
    def test_observable_native_write(self):
        self.assertTrue(contract.validate_output(output()))
    def test_oriented_cyclic_triangle(self):
        row = output(); row['faces'] = [[1, 2, 0]]
        self.assertTrue(contract.validate_output(row))
    def test_unchanged_sentinel_not_success(self):
        row = output(); row['after'] = state([-23456]); row['settings_after'] = -23456
        self.reject(contract.validate_output, row)
    def test_unbound_post_setting_not_success(self):
        row = output(); row['settings_after'] = -23456
        self.reject(contract.validate_output, row)
    def test_nonzero_before_required(self):
        row = output(); row['before'] = 0
        self.reject(contract.validate_output, row)
    def test_failed_region(self):
        row = output(); row['failed_faces'] = [0]
        self.reject(contract.validate_output, row)
    def test_no_second_or_missing_constructor(self):
        for count in (0, 2, True, 1.0):
            row = output(); row['constructor_calls'] = count
            self.reject(contract.validate_output, row)
    def test_dense_valid_ids(self):
        for key, value in [('valid_faces', []), ('valid_vertices', [0, 2]), ('valid_vertices', [0, 1, 3])]:
            row = output(); row[key] = value
            self.reject(contract.validate_output, row)
    def test_no_winding_reversal(self):
        row = output(); row['faces'] = [[0, 2, 1]]
        self.reject(contract.validate_output, row)
    def test_no_extra_or_duplicate_triangle(self):
        row = output(); row['faces'] *= 2
        self.reject(contract.validate_output, row)
    def test_no_vertex_motion(self):
        row = output(); row['vertices'][1][0] = 1.001
        self.reject(contract.validate_output, row)
    def test_no_nonfinite(self):
        for bad in (float('nan'), float('inf')):
            row = output(); row['vertices'][0][0] = bad
            self.reject(contract.validate_output, row)
    def test_exact_promoted_f32_zero(self):
        row = output(); row['vertices'][0][0] = -0.0
        self.reject(contract.validate_output, row)
    def test_boolean_index_rejected(self):
        row = output(); row['faces'][0][0] = False
        self.reject(contract.validate_output, row)

if __name__ == '__main__': unittest.main(verbosity=2)
