"""Focused numerical and reuse tests; no GPU/native execution."""
import sys,unittest,uuid
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import amd_neighbor_mixing_oracle as m
import amd_cached_kernel_oracle as old
import amd_native_convolution_contract as c

class MixingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f,cls.co,cls.w,cls.b=m.fixture()
        cls.meta=old.build(cls.co)
        cls.ref,cls.coverage=m.reference(cls.f,cls.co,cls.w,cls.b)
        cls.partials=m.split_reference(cls.f,cls.co,cls.w,cls.meta,cls.ref)
    def test_only_weights_changed_and_three_exact_taps(self):
        f,co,w,b=c.fixture()
        self.assertTrue(np.array_equal(self.f,f));self.assertTrue(np.array_equal(self.co,co));self.assertTrue(np.array_equal(self.b,b))
        self.assertEqual(np.count_nonzero(self.w),3072)
        self.assertEqual({float(v) for v in np.unique(self.w)},{0.,1.,.5,-.25})
        self.assertEqual(self.w.dtype,np.float16)
        m.validate_fixture(self.f,self.co,self.w,self.b)
        corrupted=self.w.copy();corrupted[0,1,1,2,0]=0
        with self.assertRaises(ValueError):m.validate_fixture(self.f,self.co,corrupted,self.b)
    def test_tuple_reference_boundaries_sparse_miss_and_coverage(self):
        self.assertEqual(self.ref.dtype,np.float32)
        for row,numerator in ((0,-6110),(31,-1988),(1024,1010),(2048,4040)):
            self.assertEqual(float(self.ref[row,0]),numerator/4096)
        self.assertEqual(self.coverage['valid_rows'],[2049,1984,1025])
        self.assertEqual(self.coverage['out_of_bounds_missing'],[0,64,1024])
        self.assertEqual(self.coverage['in_bounds_sparse_missing'],[0,1,0])
        self.assertEqual(self.coverage['valid_noncentral_terms'],3081216)
        self.assertEqual(self.coverage['nonidentity_elements'],2064894)
        self.assertEqual(self.coverage['reference_numerator_min'],-6119)
        self.assertEqual(self.coverage['reference_numerator_max'],6085)
    def test_correct_split_reduction_and_exact_raw_planes(self):
        out,stats=m.compare(self.partials,self.ref,self.partials)
        self.assertEqual(out.dtype,np.float32)
        self.assertTrue(np.array_equal(out,self.ref))
        self.assertEqual(stats['mismatched_elements'],0)
        self.assertTrue(stats['fp32_reference_exact']);self.assertTrue(stats['per_split_reference_exact'])
        self.assertTrue(np.count_nonzero(self.partials[0])>100000)
        self.assertTrue(np.count_nonzero(self.partials[1])>100000)
    def test_nonfinite_errors_and_compensating_split_corruption_rejected(self):
        for bad in (np.nan,np.inf,1/8192):
            raw=self.partials.copy();raw[0,0,0]+=bad
            with self.assertRaises(ValueError):m.compare(raw,self.ref,self.partials)
        raw=self.partials.copy();raw[0,0,0]+=.25;raw[1,0,0]-=.25
        self.assertTrue(np.array_equal(raw.sum(axis=0),self.ref))
        with self.assertRaises(ValueError):m.compare(raw,self.ref,self.partials)
        with self.assertRaises(ValueError):m.compare(self.partials.astype(np.float16),self.ref,self.partials)
        self.assertFalse(np.array_equal(self.ref.astype(np.float16).astype(np.float32),self.ref))
    def test_weight_order_sign_and_missing_neighbor_faults_detectable(self):
        for index,value in (((0,1,1,0,0),.5),((0,0,1,1,0),.25)):
            weights=self.w.copy();weights[index]=value
            with self.assertRaises(ValueError):m.reference(self.f,self.co,weights,self.b)
        wrong=self.partials.copy();wrong[0,2048,0]+=.5*float(self.f[0,0])
        with self.assertRaises(ValueError):m.compare(wrong,self.ref,self.partials)
    def test_scratch_readback_exclusive_and_invalid_names(self):
        path=Path(__file__).resolve().parents[2]/'docs/compatibility/amd-cached-neighbour-mixing-evidence/persistence-tests'/str(uuid.uuid4())/'scratch';path.mkdir(parents=True)
        rows=old.save_arrays(path,{'cpu_reference':self.ref,'raw_splits':self.partials})
        self.assertEqual(len(rows),2)
        with self.assertRaises(FileExistsError):old.save_arrays(path,{'cpu_reference':self.ref})
        with self.assertRaises(ValueError):old.save_arrays(path,{'../foreign':self.ref})
    def test_real_reused_arrays_and_classifier_contract(self):
        import amd_cached_kernel_convolution as b
        previous=b.NUMERICAL
        try:
            b.NUMERICAL=m
            values,coords,proof=b.arrays();references,coverage=b.cpu_references(values,coords)
            self.assertEqual(set(references),{'cpu_reference','cpu_split_reference'})
            self.assertTrue(np.array_equal(references['cpu_reference'],self.ref))
            self.assertTrue(np.isnan(values['output']).all())
            self.assertTrue(np.shares_memory(values['weight'],values['weight'].base))
            r={'status':'PASS','native_counts':{k:int(k in ('convolution','jit','binary','launch')) for k in c.Permit().counts},'helper_calls':1,'lookup_keys':[b.p.HIT_KEY],'forbidden_calls':[],
                'elementwise':{'mismatched_elements':0,'fp32_reference_exact':True,'per_split_reference_exact':True},'coordinates_unchanged':True,'release':{'allocated_bytes':0,'reserved_bytes':0,'owned_tensors_collected':True}}
            self.assertEqual(b.classify(r,True,True,True),'PASS')
            r['elementwise']['per_split_reference_exact']=False;self.assertEqual(b.classify(r,True,True,True),'UNKNOWN')
        finally:b.NUMERICAL=previous

if __name__=='__main__':unittest.main()
