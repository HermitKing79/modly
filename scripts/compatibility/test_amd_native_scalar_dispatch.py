"""Actual installed CPU ATen operators through the production native dispatcher."""
import contextlib
import io
import unittest
import numpy as np

def run_scalar_suite(torch,private,meta,source,*,red=False):
    import amd_native_convolution_policy as production
    from amd_native_post1_scalar import LENGTH,N,EXPECTED
    from amd_forward_diagnostics import Trace
    @contextlib.contextmanager
    def constructed():
        policy=production.NativePolicy(torch,private,meta,source,Trace(lambda line:None))
        try:
            policy.permit.phase='convolution'
            with torch.inference_mode():yield policy,production.dispatch_guard(torch,policy)
        finally:policy.close()
    def invoke(mode,op,*args,**kw):return mode.__torch_dispatch__(op,(torch.Tensor,),args,kw)
    def mask_prefix(mode,*,values=None):
        mask=invoke(mode,torch.ops.aten.empty.memory_format,[LENGTH],dtype=torch.int32,device=torch.device('cpu'))
        mask.numpy()[:]=valid_flags if values is None else values
        return mask,invoke(mode,torch.ops.aten.cumsum.default,mask,0,dtype=torch.int32)
    def last(mode,prefix):return invoke(mode,torch.ops.aten.select.int,prefix,0,-1)
    def read(mode,scalar,op=None):return invoke(mode,torch.ops.aten.item.default if op is None else op,scalar)
    def scope(policy):
        policy.permit.take('post1');return policy.post1_scalar.scope('cpu')

    # Independent, bounded CPU neighbor reference for the exact unchanged support.
    i=np.arange(N,dtype=np.int32)
    coords=np.stack([i//1024,(i//32)%32,i%32],1)
    offsets=np.array([(x,y,z) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1)],dtype=np.int32)
    neighbors=coords[:,None,:]+offsets[None,:,:]
    inside=((neighbors>=0)&(neighbors<np.array([3,32,32]))).all(axis=2)
    keys=neighbors[:,:,0]*1024+neighbors[:,:,1]*32+neighbors[:,:,2]
    support=coords[:,0]*1024+coords[:,1]*32+coords[:,2]
    valid_flags=(inside & np.isin(keys,support)).T.reshape(-1).astype(np.int32)
    if int(valid_flags.sum())!=EXPECTED:raise AssertionError('Independent fixture neighbor count differs')
    observed={}
    class ScalarTests(unittest.TestCase):
        def test_foreign_scalar_rejected_without_native_scope(self):
            with constructed() as (policy,mode):
                scalar=torch.tensor(35353,dtype=torch.int32)
                with self.assertRaises(RuntimeError):
                    mode.__torch_dispatch__(torch.ops.aten.item.default,(torch.Tensor,),(scalar,))
        def test_exact_installed_item_through_production_handler(self):
            with constructed() as (policy,mode):
                with scope(policy):
                    _,prefix=mask_prefix(mode);scalar=last(mode,prefix)
                    self.assertEqual(read(mode,scalar),EXPECTED)
                self.assertEqual(policy.post1_scalar.reads,1)
                self.assertIsNone(policy.post1_scalar.mask);self.assertIsNone(policy.post1_scalar.prefix);self.assertIsNone(policy.post1_scalar.last)
                observed['exact_item_result']=policy.post1_scalar.value
        def test_normal_torch_dispatch_mode_and_local_scalar_dense(self):
            with constructed() as (policy,mode):
                with scope(policy):
                    with mode:mask=torch.ops.aten.empty.memory_format([LENGTH],dtype=torch.int32,device=torch.device('cpu'))
                    mask.numpy()[:]=valid_flags
                    with mode:
                        prefix=torch.ops.aten.cumsum.default(mask,0,dtype=torch.int32)
                        scalar=torch.ops.aten.select.int(prefix,0,-1)
                        self.assertEqual(torch.ops.aten.item.default(scalar),EXPECTED)
                self.assertEqual(policy.post1_scalar.reads,1)
            with constructed() as (policy,mode):
                with scope(policy):
                    _,prefix=mask_prefix(mode);scalar=last(mode,prefix)
                    self.assertEqual(read(mode,scalar,torch.ops.aten._local_scalar_dense.default),EXPECTED)
        def test_foreign_same_metadata_scalar_rejected(self):
            foreign=torch.tensor(EXPECTED,dtype=torch.int32)
            with constructed() as (policy,mode):
                with scope(policy):
                    _,prefix=mask_prefix(mode);scalar=last(mode,prefix)
                    with self.assertRaises(RuntimeError):read(mode,foreign)
                    self.assertEqual(policy.post1_scalar.reads,0);self.assertEqual(read(mode,scalar),EXPECTED)
        def test_duplicate_reads_rejected_before_second_operator(self):
            with constructed() as (policy,mode):
                with scope(policy):
                    _,prefix=mask_prefix(mode);scalar=last(mode,prefix);read(mode,scalar)
                    with self.assertRaises(RuntimeError):read(mode,scalar)
                    with self.assertRaises(RuntimeError):read(mode,scalar,torch.ops.aten._local_scalar_dense.default)
                    self.assertEqual(policy.post1_scalar.reads,1)
        def test_wrong_dtype_shape_and_arguments_rejected(self):
            variants=[torch.tensor(EXPECTED,dtype=torch.int64),torch.tensor(float(EXPECTED)),torch.tensor([EXPECTED],dtype=torch.int32)]
            with constructed() as (policy,mode):
                with scope(policy):
                    _,prefix=mask_prefix(mode);scalar=last(mode,prefix)
                    for value in variants:
                        with self.assertRaises(RuntimeError):read(mode,value)
                    with self.assertRaises(RuntimeError):invoke(mode,torch.ops.aten.item.default,scalar,0)
                    with self.assertRaises(RuntimeError):invoke(mode,torch.ops.aten.item.default,scalar,unknown=True)
                    self.assertEqual(policy.post1_scalar.reads,0);read(mode,scalar)
        def test_changed_eligible_tensor_metadata_rejected(self):
            with constructed() as (policy,mode):
                with self.assertRaises(RuntimeError):
                    with scope(policy):
                        _,prefix=mask_prefix(mode);scalar=last(mode,prefix)
                        scalar.data=torch.tensor(float(EXPECTED))
                        read(mode,scalar)
                self.assertEqual(policy.post1_scalar.reads,0)
        def test_same_dtype_foreign_scalar_storage_rejected(self):
            with constructed() as (policy,mode):
                with self.assertRaisesRegex(RuntimeError,'storage provenance'):
                    with scope(policy):
                        _,prefix=mask_prefix(mode);scalar=last(mode,prefix)
                        scalar.data=torch.tensor(EXPECTED,dtype=torch.int32)
                        read(mode,scalar)
                self.assertEqual(policy.post1_scalar.reads,0)
        def test_mask_and_prefix_foreign_storage_rejected(self):
            with constructed() as (policy,mode):
                with scope(policy):
                    mask=invoke(mode,torch.ops.aten.empty.memory_format,[LENGTH],dtype=torch.int32,device=torch.device('cpu'))
                    mask.numpy()[:]=valid_flags;original=mask.detach()
                    mask.data=torch.zeros_like(mask)
                    with self.assertRaisesRegex(RuntimeError,'storage provenance'):invoke(mode,torch.ops.aten.cumsum.default,mask,0,dtype=torch.int32)
                    mask.data=original
                    prefix=invoke(mode,torch.ops.aten.cumsum.default,mask,0,dtype=torch.int32);original=prefix.detach()
                    prefix.data=torch.zeros_like(prefix)
                    with self.assertRaisesRegex(RuntimeError,'storage provenance'):last(mode,prefix)
                    prefix.data=original;read(mode,last(mode,prefix))
        def test_swapped_prefix_and_scalar_together_rejected(self):
            with constructed() as (policy,mode):
                with self.assertRaisesRegex(RuntimeError,'storage provenance'):
                    with scope(policy):
                        _,prefix=mask_prefix(mode);scalar=last(mode,prefix)
                        foreign=torch.full_like(prefix,EXPECTED)
                        prefix.data=foreign;scalar.data=foreign[-1]
                        read(mode,scalar)
                self.assertEqual(policy.post1_scalar.reads,0)
        def test_same_storage_prefix_metadata_change_rejected(self):
            with constructed() as (policy,mode):
                with scope(policy):
                    _,prefix=mask_prefix(mode);original=prefix.detach()
                    prefix.data=original.view(1,LENGTH)
                    with self.assertRaisesRegex(RuntimeError,'tensor metadata'):last(mode,prefix)
                    prefix.data=original;scalar=last(mode,prefix)
                    prefix.data=original.view(1,LENGTH)
                    with self.assertRaisesRegex(RuntimeError,'tensor metadata'):read(mode,scalar)
                    prefix.data=original;read(mode,scalar)
        def test_foreign_cumsum_repeated_allocation_and_select_rejected(self):
            foreign=torch.zeros([LENGTH],dtype=torch.int32)
            with constructed() as (policy,mode):
                with scope(policy):
                    mask,prefix=mask_prefix(mode)
                    with self.assertRaises(RuntimeError):invoke(mode,torch.ops.aten.cumsum.default,foreign,0,dtype=torch.int32)
                    with self.assertRaises(RuntimeError):invoke(mode,torch.ops.aten.cumsum.default,mask,0,dtype=torch.int32)
                    with self.assertRaises(RuntimeError):invoke(mode,torch.ops.aten.empty.memory_format,[LENGTH],dtype=torch.int32,device=torch.device('cpu'))
                    for dim,index in ((1,-1),(0,0),(True,-1)):
                        with self.assertRaises(RuntimeError):invoke(mode,torch.ops.aten.select.int,prefix,dim,index)
                    scalar=last(mode,prefix)
                    with self.assertRaises(RuntimeError):last(mode,prefix)
                    read(mode,scalar)
        def test_incorrect_cumsum_dtype_and_arguments_rejected(self):
            with constructed() as (policy,mode):
                with scope(policy):
                    mask=invoke(mode,torch.ops.aten.empty.memory_format,[LENGTH],dtype=torch.int32,device=torch.device('cpu'))
                    mask.numpy()[:]=valid_flags
                    for dim,dtype in ((1,torch.int32),(0,torch.int64),(True,torch.int32)):
                        with self.assertRaises(RuntimeError):invoke(mode,torch.ops.aten.cumsum.default,mask,dim,dtype=dtype)
                    prefix=invoke(mode,torch.ops.aten.cumsum.default,mask,0,dtype=torch.int32)
                    read(mode,last(mode,prefix))
        def test_out_of_bounds_and_wrong_fixture_counts_rejected(self):
            for value in (N-1,LENGTH+1,EXPECTED+1):
                values=np.zeros(LENGTH,dtype=np.int32);values[-1]=value
                with constructed() as (policy,mode):
                    with self.assertRaisesRegex(RuntimeError,'bounds or deterministic'):
                        with scope(policy):
                            _,prefix=mask_prefix(mode,values=values);read(mode,last(mode,prefix))
                    self.assertEqual(policy.post1_scalar.reads,1);self.assertIsNone(policy.post1_scalar.value)
                    self.assertIsNone(policy.post1_scalar.mask)
        def test_other_operations_remain_rejected(self):
            with constructed() as (policy,mode):
                with scope(policy):
                    _,prefix=mask_prefix(mode)
                    with self.assertRaises(RuntimeError):invoke(mode,torch.ops.aten.add.Tensor,prefix,prefix)
                    read(mode,last(mode,prefix))
        def test_lifecycle_nested_repeated_scope_and_post2_rejected(self):
            with constructed() as (policy,mode):
                with scope(policy):
                    with self.assertRaises(RuntimeError):
                        with policy.post1_scalar.scope('cpu'):pass
                    _,prefix=mask_prefix(mode);scalar=last(mode,prefix);read(mode,scalar)
                with self.assertRaises(RuntimeError):
                    with policy.post1_scalar.scope('cpu'):pass
                with self.assertRaises(RuntimeError):read(mode,scalar)
                # Second native operation is intentionally NOT permitted here.
                policy.permit.take('post2')
                other=torch.tensor(33,dtype=torch.int32)
                with self.assertRaises(RuntimeError):read(mode,other)
            with constructed() as (policy,mode):
                policy.permit.phase='output'
                with self.assertRaises(RuntimeError):
                    with scope(policy):pass
        def test_missing_read_fails_closed_and_clears_references(self):
            with constructed() as (policy,mode):
                with self.assertRaisesRegex(RuntimeError,'Exactly one'):
                    with scope(policy):mask_prefix(mode)
                self.assertFalse(policy.post1_scalar.active);self.assertIsNone(policy.post1_scalar.mask)
        def test_actual_operator_schemas(self):
            self.assertEqual(str(torch.ops.aten.item.default._schema),'aten::item(Tensor self) -> Scalar')
            self.assertEqual(str(torch.ops.aten.cumsum.default._schema),'aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor')
            self.assertIn('select.int(Tensor(a) self, int dim, SymInt index)',str(torch.ops.aten.select.int._schema))
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(ScalarTests)
    if red:suite=unittest.TestSuite([ScalarTests('test_foreign_scalar_rejected_without_native_scope')])
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);print(stream.getvalue(),flush=True)
    return dict(passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),gpu_initialized=torch.cuda.is_initialized(),
        operator_schemas={name:str(getattr(torch.ops.aten,name).default._schema) for name in ('item','_local_scalar_dense','cumsum')},
        expected_fixture_count=int(valid_flags.sum()),source_bounds=[N,LENGTH],post2_permitted=False,observed=observed)
