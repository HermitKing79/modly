import sys
sys.dont_write_bytecode=True
from pathlib import Path
import unittest,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
try:from decision import evaluate
except ImportError:evaluate=None


class DecisionTests(unittest.TestCase):
    def run_case(self,predicted,raw,groups,zero=None):
        self.assertTrue(callable(evaluate),'Per-region fixed-factor decision implementation required')
        n=len(raw);qf=np.arange(n,dtype=np.int32);kind=np.arange(n,dtype=np.int8)%6
        return evaluate(predicted,raw,qf,kind,groups,np.zeros(n,bool) if zero is None else zero)

    def base(self,n):
        p=np.zeros((n,3),np.float64);r=np.zeros((n,6),np.float32);r[:,4:]=1
        return p,r

    def test_region_failure_is_not_hidden_in_successful_global_average(self):
        p,r=self.base(6006);p[-6:]=.03
        groups={'good':dict(query_start=0,query_stop=6000),'bad':dict(query_start=6000,query_stop=6006)}
        result=self.run_case(p,r,groups)
        self.assertEqual(result['material_suitability'],'BLOCKED')
        self.assertTrue(result['regions']['good']['pass']);self.assertFalse(result['regions']['bad']['pass'])
        self.assertLess(np.sqrt(np.mean(p*p)),.02)

    def test_each_fixed_scalar_and_opaque_alpha_is_gated_separately(self):
        p,r=self.base(6);r[:,3]=.031
        result=self.run_case(p,r,{'all':dict(query_start=0,query_stop=6)})
        self.assertFalse(result['regions']['all']['all']['scalars']['metallic']['pass'])
        self.assertFalse(result['regions']['all']['pass'])
        p,r=self.base(6);r[:,5]=.95
        self.assertFalse(self.run_case(p,r,{'all':dict(query_start=0,query_stop=6)})['regions']['all']['all']['scalars']['alpha']['pass'])

    def test_corner_centroid_and_interior_results_remain_separate(self):
        p,r=self.base(6);p[:3]=.021
        result=self.run_case(p,r,{'all':dict(query_start=0,query_stop=6)})
        self.assertTrue(result['regions']['all']['all']['RGB']['pass'])
        self.assertFalse(result['regions']['all']['corners']['RGB']['pass'])
        self.assertEqual(result['material_suitability'],'BLOCKED')

    def test_unsupported_reference_blocks_without_dropping_measurements(self):
        p,r=self.base(6);zero=np.zeros(6,bool);zero[1]=True
        result=self.run_case(p,r,{'all':dict(query_start=0,query_stop=6)},zero)
        self.assertEqual(result['zero_support'],1);self.assertEqual(result['material_suitability'],'BLOCKED')
        self.assertEqual(result['regions']['all']['all']['samples'],6)

    def test_exact_case_passes_and_bad_schema_is_rejected(self):
        p,r=self.base(6)
        result=self.run_case(p,r,{'all':dict(query_start=0,query_stop=6)})
        self.assertEqual(result['material_suitability'],'PASS')
        p[0,0]=float('nan')
        with self.assertRaises(ValueError):self.run_case(p,r,{'all':dict(query_start=0,query_stop=6)})


if __name__=='__main__':unittest.main()
