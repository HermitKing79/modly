"""CPU-only regression for exact helper identities and cache-only admission."""
import hashlib
import unittest
import amd_shape_cache_identity as contract


class CacheIdentityTests(unittest.TestCase):
    def test_helper_key_follows_installed_ordered_configuration(self):
        config = dict(language=['c'], library_dirs=None, include_dirs=['D:\\runtime\\include'], libraries=None, ccflags=None)
        h = hashlib.sha256(b'original source'); h.update(b'AMD64,Windows,64bit,WindowsPE')
        for key, values in config.items():
            if values is None: continue
            h.update(key.encode())
            for value in values: h.update(value.encode()); h.update(b':')
        self.assertEqual(contract.helper_key(b'original source', 'AMD64,Windows,64bit,WindowsPE', config), h.hexdigest())
        changed = dict(config, include_dirs=['D:/runtime/include'])
        self.assertNotEqual(contract.helper_key(b'original source', 'AMD64,Windows,64bit,WindowsPE', changed), h.hexdigest())

    def test_kernel_first_dispatch_signature_uses_runtime_dimensions(self):
        signature, constants, attrs = contract.first_signature(splitk=2)
        self.assertEqual(signature['N'], 'i32'); self.assertEqual(signature['Ci'], 'i32')
        self.assertNotIn((6,), constants)
        self.assertEqual(constants[(10,)], 27)
        self.assertEqual(constants[(14,)], 2)
        self.assertEqual(attrs[(8,)], [['tt.divisibility', 16]])
        self.assertEqual(signature['neighbor'], '*u32')
        self.assertEqual(signature['valid_kernel'], '*i32')
        self.assertEqual(signature['valid_kernel_seg'], '*i32')
        self.assertEqual(set(attrs), {(i,) for i in range(len(signature))})
        self.assertEqual(attrs[(6,)], [])
        self.assertEqual(attrs[(10,)], [])

    def test_cache_only_admission_rejects_misses_before_callback(self):
        calls = []
        gate = contract.CacheOnly({'a'*64: 'verified product'})
        self.assertEqual(gate.require('a'*64), 'verified product')
        for key in ('b'*64, '', None):
            with self.assertRaises(ValueError):
                product = gate.require(key); calls.append(product)
        self.assertEqual(calls, [])
        for action in ('JIT compile', 'helper build', 'autotune', 'foreign native entry'):
            with self.assertRaises(RuntimeError): gate.forbidden(action)

    def test_first_dispatch_rejects_unverified_variants(self):
        for splitk in (1,4,True,2.0):
            with self.assertRaises(ValueError):contract.first_signature(splitk=splitk)


if __name__ == '__main__': unittest.main()
