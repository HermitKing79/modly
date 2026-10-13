"""Fixed CPU oracle, corruption and owned-scratch persistence checks."""
import sys
from pathlib import Path
import unittest
import uuid
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import amd_cached_kernel_oracle as o
import amd_native_convolution_contract as c

class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f,cls.co,cls.w,cls.b=c.fixture();cls.meta=o.build(cls.co)
    def test_complete_membership_and_pointer_bounds(self):
        c.validate_fixture(self.f,self.co,self.w,self.b)
        proof=o.validate(self.co,self.meta)
        self.assertEqual((proof['signals'],proof['kernel_entries']),(35353,599))
        self.assertEqual(self.meta['valid_kernel_seg'].tolist(),[0,17,35,53,71,89,107,125,143,161,179,197,215,233,251,269,290,311,329,347,365,383,401,419,437,455,473,491,509,527,545,563,581,599])
    def test_corrupt_membership_sort_segment_or_kernel_rejected(self):
        for name,index,value in [('neighbor',(0,13),2049),('neighbor',(0,0),0),('sorted_idx',0,2049),('valid_kernel',0,27),('valid_kernel_seg',1,0)]:
            m={k:v.copy() for k,v in self.meta.items()};m[name][index]=value
            with self.assertRaises(ValueError):o.validate(self.co,m)
    def test_wrong_metadata_dtype_and_coordinate_identity(self):
        m=self.meta.copy();m['neighbor']=m['neighbor'].astype(np.int64)
        with self.assertRaises(ValueError):o.validate(self.co,m)
        coords=self.co.copy();coords[1]=coords[0]
        with self.assertRaises(ValueError):o.build(coords)
    def test_exact_reduction_and_nonfinite_unwritten_or_rounding_errors(self):
        raw=np.zeros((2,2049,1024),np.float32);raw[0]=self.f
        out,stats=o.compare(raw,self.f);self.assertTrue(np.array_equal(out,self.f));self.assertEqual(stats['mismatched_elements'],0)
        for bad in (np.nan,np.inf,1/8192):
            altered=raw.copy();altered[1,0,0]=bad
            with self.assertRaises(ValueError):o.compare(altered,self.f)
    def test_owned_output_readback_exclusive_and_invalid_names(self):
        base=Path(__file__).resolve().parents[2]/'docs/compatibility/amd-cached-kernel-convolution-evidence/persistence-tests'/str(uuid.uuid4());base.mkdir(parents=True)
        rows=o.save_arrays(base,dict(raw_splits=self.f[:2].copy()));self.assertTrue(Path(rows['raw_splits']['path']).exists())
        with self.assertRaises(FileExistsError):o.save_arrays(base,dict(raw_splits=self.f[:2].copy()))
        for name in ('../foreign','foreign/file','..','_',''):
            with self.assertRaises(ValueError):o.save_arrays(base,{name:self.f[:2]})
    def test_actual_array_assembly_and_counts_classification(self):
        import amd_cached_kernel_convolution as proof
        values,coords,_=proof.arrays()
        self.assertEqual(set(values),{'input','weight','bias','neighbor','sorted_idx','valid_kernel','valid_kernel_seg','output'})
        self.assertTrue(np.shares_memory(values['weight'],values['weight'].base))
        self.assertTrue(np.isnan(values['output']).all())
        self.assertTrue(np.array_equal(coords,self.co))
        counts={name:int(name in ('convolution','jit','binary','launch')) for name in c.Permit().counts}
        self.assertTrue(proof.counts_valid(counts))
        for name in counts:
            altered=counts.copy();altered[name]+=1;self.assertFalse(proof.counts_valid(altered))
        self.assertFalse(proof.counts_valid({**counts,'foreign':1}))
        self.assertFalse(proof.counts_valid({k:v for k,v in counts.items() if k!='neighbor'}))
        self.assertEqual(proof.classify({},True,True,True),'UNKNOWN')

def installed_policy_tests(torch,private,meta,source,leaf):
    from amd_cached_kernel_policy import CachedKernelPolicy
    from amd_forward_diagnostics import Trace
    policy=CachedKernelPolicy(torch,private,meta,source,Trace(lambda line:None))
    try:
        policy.select_leaf(leaf)
        for foreign in (leaf,object()):
            try:policy.select_leaf(foreign)
            except RuntimeError:pass
            else:raise AssertionError('Foreign/repeated leaf accepted')
        # Genuine CPU tensors cannot enter the GPU-only owned-buffer contract.
        try:policy.bind(dict(input=torch.zeros((2049,1024),dtype=torch.float16)))
        except RuntimeError:pass
        else:raise AssertionError('Partial/foreign buffers accepted')
        if any(policy.permit.counts.values()) or policy.helper_calls or policy.lookups:raise AssertionError('Native execution in CPU policy test')
        return dict(passed=True,real_policy_constructed=True,foreign_leaf_and_buffers_rejected=True,native_counts=policy.permit.counts.copy())
    finally:policy.close()

if __name__=='__main__':unittest.main()
