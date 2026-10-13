"""CPU-only regression checks for the neural decoder's preflight contracts."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import amd_shape_decoder_preflight as audit


class PreflightTests(unittest.TestCase):
    def arrays(self):
        return np.ones((3, 32), dtype=np.float32), np.array(
            [[0, 0, 0, 0], [0, 1, 2, 3], [0, 31, 31, 31]], dtype=np.int32)

    def test_valid_saved_layout(self):
        x, c = self.arrays()
        self.assertEqual(audit.validate_arrays(x, c, count=3)['tokens'], 3)

    def test_bad_array_contracts(self):
        x, c = self.arrays()
        cases = [(x.astype(np.float16), c), (x[:, :31], c),
                 (x, c.astype(np.int64)), (x, c[:, :3]), (x, c[::-1][:2])]
        for a, b in cases:
            with self.subTest(shape=(a.shape, b.shape)), self.assertRaises(ValueError):
                audit.validate_arrays(a, b, count=3)

    def test_nonfinite_duplicate_batch_and_bounds(self):
        x, c = self.arrays()
        for kind in ('nonfinite', 'duplicate', 'batch', 'negative', 'upper'):
            a, b = x.copy(), c.copy()
            if kind == 'nonfinite': a[0, 0] = np.nan
            if kind == 'duplicate': b[1] = b[0]
            if kind == 'batch': b[0, 0] = 1
            if kind == 'negative': b[0, 1] = -1
            if kind == 'upper': b[0, 1] = 32
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit.validate_arrays(a, b, count=3)

    def test_reader_rejects_wrong_hash_and_trailing_bytes(self):
        with tempfile.TemporaryDirectory(dir=audit.ROOT) as directory:
            Path(directory).resolve().relative_to(audit.ROOT.resolve())
            p = Path(directory) / 'saved.npy'
            np.save(p, self.arrays()[0], allow_pickle=False)
            expected = audit.digest(p)
            self.assertEqual(audit.read_npy(p, expected).shape, (3, 32))
            with self.assertRaises(ValueError): audit.read_npy(p, '0' * 64)
            with p.open('ab') as stream: stream.write(b'extra')
            with self.assertRaises(ValueError): audit.read_npy(p, audit.digest(p))

    def test_exact_state_and_scoped_precision(self):
        header = {'from_latent.weight': {'shape': [2, 32], 'dtype': 'F16'},
                  'blocks.0.0.conv.weight': {'shape': [2, 3, 3, 3, 2], 'dtype': 'F16'}}
        state = {k: dict(v, device='meta') for k, v in header.items()}
        state['from_latent.weight']['dtype'] = 'F32'
        result = audit.validate_state(header, state, {})
        self.assertEqual(result['fp32_keys'], ['from_latent.weight'])
        for change in ('missing', 'extra', 'shape', 'dtype', 'foreign_widen', 'buffer'):
            h, s, b = copy.deepcopy(header), copy.deepcopy(state), {}
            if change == 'missing': s.pop('from_latent.weight')
            if change == 'extra': s['foreign'] = s['from_latent.weight']
            if change == 'shape': s['from_latent.weight']['shape'] = [1, 32]
            if change == 'dtype': s['from_latent.weight']['dtype'] = 'F16'
            if change == 'foreign_widen': s['blocks.0.0.conv.weight']['dtype'] = 'F32'
            if change == 'buffer': b['unexpected'] = {'shape': [1], 'dtype': 'F32'}
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.validate_state(h, s, b)

    def test_subdivision_and_memory_are_not_predicted_output(self):
        result = audit.growth_budget(3743, 948598492)
        self.assertEqual(result['maximum_support'], [3743, 29944, 239552, 1916416, 15331328])
        self.assertGreater(result['worst_visible_final_bytes'], 12 * 1024**3)
        self.assertFalse(result['unconditional_12gib_completion_established'])
        self.assertIn('excludes', result['limitations'])

    def test_cache_metadata_is_not_current_source_attestation(self):
        row = {'hash': 'a' * 64, 'target': {'backend': 'hip', 'arch': 'gfx1201', 'warp_size': 32},
               'triton_version': '3.8.0', 'global_scratch_size': 0, 'profile_scratch_size': 0}
        self.assertEqual(audit.validate_cache_metadata(row)['current_source_cache_key_verified'], False)
        for key in ('arch', 'backend', 'warp_size'):
            invalid = copy.deepcopy(row); invalid['target'][key] = 'foreign'
            with self.assertRaises(ValueError): audit.validate_cache_metadata(invalid)
        invalid = copy.deepcopy(row); invalid['global_scratch_size'] = 1
        with self.assertRaises(ValueError): audit.validate_cache_metadata(invalid)

    def test_metadata_result_cannot_accept_nominal_pass_with_violation(self):
        row = dict(returncode=0, worker_exit_verified=True,
                   job_cleanup=dict(owned_tree_exit_verified=True, forced_job_cleanup=False),
                   worker=dict(status='PASS', gpu_initialized=False, weights_loaded=False, model_forward_calls=0))
        self.assertTrue(audit.metadata_passed(row))
        for key, value in [('gpu_initialized', True), ('weights_loaded', True), ('model_forward_calls', 1),
                           ('model_forward_calls', False), ('error', 'raised after nominal success')]:
            invalid = copy.deepcopy(row); invalid['worker'][key] = value
            with self.subTest(key=key): self.assertFalse(audit.metadata_passed(invalid))
        for key in ('gpu_initialized', 'weights_loaded', 'model_forward_calls'):
            invalid = copy.deepcopy(row); invalid['worker'].pop(key)
            self.assertFalse(audit.metadata_passed(invalid))
        invalid = copy.deepcopy(row); invalid['error'] = 'supervisor failure after worker exited'
        self.assertFalse(audit.metadata_passed(invalid))

    def test_cache_container_complete_and_directory_identity(self):
        with tempfile.TemporaryDirectory(dir=audit.ROOT) as directory:
            p = Path(directory).resolve(); p.relative_to(audit.ROOT.resolve())
            children = {}
            for suffix in ('.source', '.ttir', '.ttgir', '.llir', '.amdgcn', '.hsaco', '.json'):
                child = p / ('kernel' + suffix); child.write_bytes(b'CPU fixture, never executed')
                children[child.name] = str(child)
            meta = p / 'kernel.json'
            self.assertEqual(len(audit.cache_group_paths(meta, children)), 7)
            missing = dict(children); missing.pop('kernel.hsaco')
            with self.assertRaises(ValueError): audit.cache_group_paths(meta, missing)
            foreign = p / 'foreign'; foreign.mkdir(); (foreign / 'kernel.hsaco').write_bytes(b'foreign')
            invalid = dict(children); invalid['kernel.hsaco'] = str(foreign / 'kernel.hsaco')
            with self.assertRaises(ValueError): audit.cache_group_paths(meta, invalid)


if __name__ == '__main__': unittest.main()
