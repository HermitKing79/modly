"""CPU-only opt-in and final acceptance contracts for the six added tile sites."""
import inspect
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_decoder import contract_document
import run_contract
import admission


class TilingContractTests(unittest.TestCase):
    def test_manifest_default_is_byte_equivalent_and_opt_in_is_authenticated(self):
        doc=contract_document()
        old=run_contract.parse(doc)
        self.assertFalse(getattr(old,'large_convolution_tiling',None),
                         'explicit default-off tiling contract is missing')
        self.assertIn('large_convolution_tiling',inspect.signature(run_contract.RunContract).parameters)
        self.assertEqual(old.document(),doc)
        enabled=run_contract.parse(dict(doc,large_convolution_tiling=True))
        self.assertTrue(enabled.large_convolution_tiling)
        self.assertNotEqual(enabled.sha256,old.sha256)
        self.assertEqual(enabled.document(),dict(doc,large_convolution_tiling=True))
        for bad in (1,'true',None,[],{}):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                run_contract.parse(dict(doc,large_convolution_tiling=bad))

    def test_historical_hooks_do_not_enable_new_tiling(self):
        self.assertIn('tile_large_convolutions',inspect.signature(admission.AdmissionHooks).parameters)
        self.assertFalse(admission.AdmissionHooks(None,None).tile_large_convolutions)
        with self.assertRaises(admission.AdmissionStop):
            admission.AdmissionHooks(None,None,tile_large_convolutions=1)


if __name__=='__main__':unittest.main(verbosity=2)
