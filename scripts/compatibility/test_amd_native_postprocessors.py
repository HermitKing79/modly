"""Source-visible postprocessor host operations, genuine CPU ATen, no native code."""
import contextlib
import io
import unittest
import numpy as np

def reference():
    i=np.arange(2049,dtype=np.int32)
    coords=np.stack([i//1024,(i//32)%32,i%32],1)
    offsets=np.array([(x,y,z) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1)],dtype=np.int32)
    points=coords[:,None,:]+offsets[None,:,:]
    inside=((points>=0)&(points<np.array([3,32,32]))).all(axis=2)
    keys=points[:,:,0]*1024+points[:,:,1]*32+points[:,:,2]
    flags=inside & np.isin(keys,np.arange(2049))
    gray=(flags.astype(np.int32)*(1<<np.arange(27))).sum(axis=1).astype(np.int32)
    binary=gray.copy()
    for shift in range(1,27):binary^=gray>>shift
    ordered=gray[np.argsort(binary)]
    lengths=np.array([0]+[int(np.bitwise_or.reduce(ordered[a:a+64])).bit_count() for a in range(0,2049,64)],dtype=np.int32)
    assert int(flags.sum())==35353 and int(lengths.sum())==599
    return flags.T.reshape(-1).astype(np.int32),gray,binary,lengths

def run_postprocessor_suite(torch,private,meta,source,*,red=False):
    import amd_native_convolution_policy as production
    from amd_forward_diagnostics import Trace
    flags,gray,binary,lengths=reference();observed={}
    @contextlib.contextmanager
    def constructed():
        policy=production.NativePolicy(torch,private,meta,source,Trace(lambda line:None))
        try:
            policy.permit.phase='convolution'
            with torch.inference_mode():yield policy,production.dispatch_guard(torch,policy)
        finally:
            policy.close()
            assert not policy.restores
    def empty(mode,shape,dtype):
        with mode:return torch.ops.aten.empty.memory_format(shape,dtype=dtype,device=torch.device('cpu'))
    def chain(mode,mask):
        with mode:
            prefix=torch.ops.aten.cumsum.default(mask,0,dtype=torch.int32)
            last=torch.ops.aten.select.int(prefix,0,-1)
        return prefix,last
    def item(mode,last):
        with mode:return torch.ops.aten.item.default(last)
    def post1(policy,mode):
        policy.permit.take('post1')
        with policy.post1_scalar.scope('cpu'):
            g=empty(mode,[2049],torch.int32);b=empty(mode,[2049],torch.int32)
            mask=empty(mode,[55323],torch.int32);trans=empty(mode,[55323],torch.uint32)
            g.numpy()[:]=gray;b.numpy()[:]=binary;mask.numpy()[:]=flags
            with mode:sorted_idx=torch.ops.aten.argsort.default(b)
            prefix,last=chain(mode,mask)
            assert item(mode,last)==35353
            vi=empty(mode,[35353],torch.uint32);vo=empty(mode,[35353],torch.uint32);seg=empty(mode,[28],torch.uint32)
            outputs=(g,sorted_idx,vi,vo,seg)
            policy.post1_scalar.finish(outputs)
        policy.remember('post1',outputs)
        return outputs
    def post2(policy,mode):
        policy.permit.take('post2')
        with policy.post2_scalar.scope('cpu'):
            reduced=empty(mode,[33],torch.int32);seg=empty(mode,[34],torch.int32)
            seg.numpy()[:]=lengths
            prefix,last=chain(mode,seg);assert item(mode,last)==599
            valid=empty(mode,[599],torch.int32)
            outputs=(valid,prefix);policy.post2_scalar.finish(outputs)
        return outputs
    class PostprocessorTests(unittest.TestCase):
        def test_exact_post2_read_through_production_mode(self):
            with constructed() as (p,m):
                p.permit.take('post2');contract=getattr(p,'post2_scalar',None)
                context=contract.scope('cpu') if contract else contextlib.nullcontext()
                actual=None
                try:
                    with context:
                        empty(m,[33],torch.int32);seg=empty(m,[34],torch.int32);seg.numpy()[:]=lengths
                        prefix,last=chain(m,seg);actual=item(m,last)
                        if contract:
                            valid=empty(m,[599],torch.int32);contract.finish((valid,prefix))
                except RuntimeError:
                    if not red:raise
                self.assertEqual(actual,599)
        def test_complete_postprocessors_and_convolution_tail(self):
            with constructed() as (p,m):
                outputs=post1(p,m)
                p.known_input(outputs[0],'post1',0);p.known_input(outputs[1],'post1',1)
                kernel,segments=post2(p,m)
                self.assertEqual(tuple(kernel.shape),(599,));self.assertEqual(tuple(segments.shape),(34,))
                # The genuine source tail uses only these existing overloads.
                # The same production guard applies in the convolution phase
                # on CPU too; no test-only permission or replacement operator.
                weight=torch.zeros((1024,3,3,3,1024),dtype=torch.float16)
                with m:
                    view=torch.ops.aten.view.default(weight,[1024,27,1024])
                    split=torch.ops.aten.empty.memory_format([2,2049,1024],dtype=torch.float32,device=torch.device('cpu'))
                split.numpy()[:]=0
                with m:
                    summed=torch.ops.aten.sum.dim_IntList(split,[0],False)
                    out=torch.ops.aten._to_copy.default(summed,dtype=torch.float16)
                self.assertEqual(tuple(out.shape),(2049,1024));self.assertEqual(out.dtype,torch.float16)
                self.assertEqual(view.untyped_storage()._cdata,weight.untyped_storage()._cdata)
                observed.update(post1=p.post1_scalar.value,post2=p.post2_scalar.value,tail_shape=list(out.shape))
                p.release()
                self.assertFalse(p.post_outputs)
        def test_exact_python_convolution_tail_interfaces(self):
            with constructed() as (p,m):
                weight=torch.zeros((1024,3,3,3,1024),dtype=torch.float16)
                with m:
                    view=weight.reshape(1024,27,1024)
                    split=torch.empty((2,2049,1024),device=weight.device,dtype=torch.float32)
                split.numpy()[:]=0
                with m:out=split.sum(dim=0).to(torch.float16)
                self.assertEqual(tuple(out.shape),(2049,1024));self.assertEqual(out.dtype,torch.float16)
                self.assertEqual(view.untyped_storage()._cdata,weight.untyped_storage()._cdata)
                observed['python_tail_interfaces']=True
        def test_wrong_factory_shape_dtype_count_and_read_order(self):
            for tag in ('post1','post2'):
                with constructed() as (p,m):
                    p.permit.take(tag);s=getattr(p,tag+'_scalar')
                    with self.assertRaises(RuntimeError):
                        with s.scope('cpu'):
                            for shape,dtype in (([35],torch.int32),([34],torch.float32),([599],torch.int32)):
                                with self.assertRaises(RuntimeError):empty(m,shape,dtype)
                            if tag=='post2':
                                empty(m,[33],torch.int32)
                                with self.assertRaises(RuntimeError):empty(m,[33],torch.int32)
        def test_malformed_factory_arguments_rejected_before_allocation(self):
            variants=[dict(device='cuda:1'),dict(layout=torch.sparse_coo),dict(pin_memory=True),dict(pin_memory=0),
                      dict(memory_format=torch.preserve_format),dict(unexpected=True)]
            for tag in ('post1','post2'):
                for bad in variants:
                    with constructed() as (p,m):
                        p.permit.take(tag);s=getattr(p,tag+'_scalar')
                        with self.assertRaises(RuntimeError):
                            with s.scope('cpu'):
                                shape=[55323] if tag=='post1' else [34]
                                kw=dict(dtype=torch.int32,device=torch.device('cpu'));kw.update(bad)
                                with self.assertRaises(RuntimeError):
                                    m.__torch_dispatch__(torch.ops.aten.empty.memory_format,(torch.Tensor,),(shape,),kw)
                                self.assertEqual(s.factories,{});self.assertEqual(s.created,[])
        def test_post2_prefix_and_cumsum_contract(self):
            with constructed() as (p,m):
                p.permit.take('post2')
                with p.post2_scalar.scope('cpu'):
                    empty(m,[33],torch.int32);seg=empty(m,[34],torch.int32);seg.numpy()[:]=lengths
                    for dim,dtype in ((1,torch.int32),(0,torch.int64),(True,torch.int32)):
                        with self.assertRaises(RuntimeError):
                            m.__torch_dispatch__(torch.ops.aten.cumsum.default,(torch.Tensor,),(seg,dim),dict(dtype=dtype))
                    prefix,last=chain(m,seg)
                    original=prefix.detach();prefix.data=torch.zeros_like(prefix)
                    with self.assertRaises(RuntimeError):item(m,last)
                    prefix.data=original;self.assertEqual(item(m,last),599)
                    with self.assertRaises(RuntimeError):
                        m.__torch_dispatch__(torch.ops.aten._local_scalar_dense.default,(torch.Tensor,),(last,),{})
                    valid=empty(m,[599],torch.int32);p.post2_scalar.finish((valid,prefix))
        def test_post2_foreign_dtype_shape_storage_and_duplicate_read(self):
            with constructed() as (p,m):
                p.permit.take('post2')
                with p.post2_scalar.scope('cpu'):
                    empty(m,[33],torch.int32);seg=empty(m,[34],torch.int32);seg.numpy()[:]=lengths
                    prefix,last=chain(m,seg)
                    for foreign in (torch.tensor(599,dtype=torch.int32),torch.tensor(599,dtype=torch.int64),torch.tensor([599],dtype=torch.int32)):
                        with self.assertRaises(RuntimeError):item(m,foreign)
                    original=last.detach();last.data=torch.tensor(599,dtype=torch.int32)
                    with self.assertRaises(RuntimeError):item(m,last)
                    last.data=original
                    self.assertEqual(item(m,last),599)
                    with self.assertRaises(RuntimeError):item(m,last)
                    valid=empty(m,[599],torch.int32);p.post2_scalar.finish((valid,prefix))
        def test_wrong_post2_scalar_values(self):
            for value in (32,892,600):
                with constructed() as (p,m):
                    p.permit.take('post2')
                    with self.assertRaises(RuntimeError):
                        with p.post2_scalar.scope('cpu'):
                            empty(m,[33],torch.int32);seg=empty(m,[34],torch.int32);seg.numpy()[:]=0;seg.numpy()[-1]=value
                            prefix,last=chain(m,seg);item(m,last)
                    self.assertIsNone(p.post2_scalar.value)
        def test_wrong_returned_outputs_and_known_input(self):
            with constructed() as (p,m):
                out=post1(p,m)
                with self.assertRaises(RuntimeError):p.known_input(torch.zeros_like(out[0]),'post1',0)
                original=out[0].detach();out[0].data=torch.zeros_like(out[0])
                with self.assertRaises(RuntimeError):p.known_input(out[0],'post1',0)
                out[0].data=original;p.known_input(out[0],'post1',0)
            with constructed() as (p,m):
                p.permit.take('post2')
                with p.post2_scalar.scope('cpu'):
                    empty(m,[33],torch.int32);seg=empty(m,[34],torch.int32);seg.numpy()[:]=lengths
                    prefix,last=chain(m,seg);item(m,last);valid=empty(m,[599],torch.int32)
                    for bad in ((valid,), (torch.empty_like(valid),prefix),(valid,seg)):
                        with self.assertRaises(RuntimeError):p.post2_scalar.finish(bad)
                    p.post2_scalar.finish((valid,prefix))
        def test_unexpected_operations_and_lifecycle(self):
            with constructed() as (p,m):
                p.permit.take('post2')
                with self.assertRaises(RuntimeError):
                    with p.post2_scalar.scope('cpu'):
                        a=empty(m,[34],torch.int32)
                        with m:
                            with self.assertRaises(RuntimeError):torch.ops.aten.add.Tensor(a,a)
                        with self.assertRaises(RuntimeError):
                            with p.post1_scalar.scope('cpu'):pass
                self.assertFalse(p.post2_scalar.active);self.assertIsNone(p.post2_scalar.prefix)
                with self.assertRaises(RuntimeError):
                    with p.post2_scalar.scope('cpu'):pass
        def test_missing_factories_and_invalid_sort_inputs(self):
            with constructed() as (p,m):
                p.permit.take('post1')
                with p.post1_scalar.scope('cpu'):
                    a=empty(m,[2049],torch.int32);b=empty(m,[2049],torch.int32)
                    with m:
                        with self.assertRaises(RuntimeError):torch.ops.aten.argsort.default(a)
                    mask=empty(m,[55323],torch.int32);mask.numpy()[:]=flags
                    prefix,last=chain(m,mask);item(m,last)
                    with self.assertRaises(RuntimeError):p.post1_scalar.finish((a,b,mask,mask,mask))
    stream=io.StringIO()
    suite=unittest.TestSuite([PostprocessorTests('test_exact_post2_read_through_production_mode')]) if red else unittest.defaultTestLoader.loadTestsFromTestCase(PostprocessorTests)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);print(stream.getvalue(),flush=True)
    return dict(passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),observed=observed,
        gpu_initialized=torch.cuda.is_initialized(),native_executed=False,source_counts=dict(post1=35353,post2=599),block_popcounts=lengths[1:].tolist())
