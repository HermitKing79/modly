"""Focused CPU-only arithmetic-policy tests. No genuine checkpoint/native imports."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

def denied(*args, **kwargs):
    raise AssertionError('GPU entry is prohibited in CPU tests')
torch.cuda._lazy_init = denied
torch._C._cuda_init = denied

class RuntimeTests(unittest.TestCase):
    def test_selective_reference_conversion_preserves_fp32_sensitive_layers(self):
        import runtime
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks=torch.nn.Sequential(torch.nn.Linear(2,2).half(),torch.nn.LayerNorm(2))
                self.from_latent=torch.nn.Linear(2,2)
                self.output_layer=torch.nn.Linear(2,7)
        m=Model().eval().requires_grad_(False)
        m.blocks[0].weight.fill_(1.0009765625)
        originals={k:p.detach().clone() for k,p in m.named_parameters() if p.dtype==torch.float32}
        conv=type('Config',(),{'FLEX_GEMM_HASHMAP_RATIO':2.0})()
        triton=type('Config',(),{})()
        changes=runtime.configure(m,torch,(torch.nn.Linear,),conv,triton)
        self.assertEqual(conv.FLEX_GEMM_ALGO,'explicit_gemm')
        self.assertIs(triton.allow_tf32,False)
        self.assertEqual(m.dtype,torch.bfloat16)
        self.assertEqual(m.blocks[0].weight.dtype,torch.bfloat16)
        self.assertEqual(float(m.blocks[0].weight[0,0]),1.0)  # BF16 rounding is explicit.
        self.assertEqual(set(changes),{'blocks.0.weight','blocks.0.bias'})
        for k,p in m.named_parameters():
            if k in originals:self.assertTrue(torch.equal(p,originals[k]))
        self.assertFalse(torch.cuda.is_initialized())

    def test_foreign_hashmap_ratio_rejected_before_conversion(self):
        import runtime
        m=type('Model',(),{})()
        with self.assertRaises(RuntimeError):
            runtime.configure(m,torch,(torch.nn.Linear,),type('Config',(),{'FLEX_GEMM_HASHMAP_RATIO':1.0})(),None)

    def test_full_input_is_only_accepted_contract(self):
        import runtime
        x=torch.zeros(3743,32,dtype=torch.float32);c=torch.zeros(3743,4,dtype=torch.int32)
        runtime.validate_tensor_contract(x,c)
        for bad in (x[:2],x.half()):
            with self.assertRaises(RuntimeError):runtime.validate_tensor_contract(bad,c)
        with self.assertRaises(RuntimeError):runtime.validate_tensor_contract(x,c.long())

    def test_actual_output_writer_roundtrip_and_mask_linkage(self):
        import tempfile
        import numpy as np
        import neural
        original=neural.ARTIFACTS
        e=Path(__file__).resolve().parent
        coords=torch.from_numpy(np.load(e.parent/'amd-shape-slat-512-sampling-evidence/shape-latent-coordinates.npy',allow_pickle=False))
        values={'features':torch.ones(3743,7),'coordinates':coords*torch.tensor([1,16,16,16],dtype=torch.int32)}
        for i in range(4):
            values[f'subdivision_{i}_coordinates']=coords*torch.tensor([1,2**i,2**i,2**i],dtype=torch.int32)
            logits=torch.full((3743,8),-1,dtype=torch.bfloat16);logits[:,0]=1
            values[f'subdivision_{i}_logits']=logits
        with tempfile.TemporaryDirectory(prefix='owned-writer-test-',dir=e) as owned:
            self.assertTrue(Path(owned).resolve().is_relative_to(e))
            neural.ARTIFACTS=Path(owned)/'output'
            try:
                r=neural.preserve_output(values,torch,{'CPU_fixture':True,'no_model':True})
                self.assertTrue(r['verified']);self.assertEqual(len(r['tensor_payload_sha256']),10)
                values['subdivision_0_logits'][0,0]=-1
                with self.assertRaisesRegex(RuntimeError,'linkage'):neural.validate_output(values,torch)
                values['subdivision_0_logits'][0,0]=1
                values['features'][0,0]=float('nan')
                with self.assertRaisesRegex(RuntimeError,'Nonfinite'):neural.validate_output(values,torch)
            finally:neural.ARTIFACTS=original

    def test_production_native_boundary_and_restoration_with_metadata_only_inputs(self):
        import runtime
        from types import SimpleNamespace as NS
        compat=Path(__file__).resolve().parents[3]/'scripts/compatibility'
        sys.path.insert(0,str(compat))
        from amd_wsl_native_import_policy import NATIVE_BINDINGS
        # Simulation ends at the native result boundary. No CUDA tensor/native load.
        dev=torch.device('cuda',0)
        def tensor(shape,dtype,device=dev):
            return NS(shape=shape,dtype=dtype,device=device,is_contiguous=lambda:True)
        coords=tensor((3743,4),torch.int32)
        x=NS(coords=coords,feats=tensor((3743,1024),torch.bfloat16),spatial_shape=(32,32,32))
        hook=NS(active_conv=('blocks.0.0.conv',None,x))
        count=[]
        def original(*args):count.append(1);return tensor((3743,27),torch.uint32)
        cuda=NS(**{n:original for n in NATIVE_BINDINGS})
        proxy=NS(uint32=torch.uint32,uint64=torch.uint64,int32=torch.int32,cuda=NS(current_stream=lambda _:NS(cuda_stream=0)))
        cfg=NS(FLEX_GEMM_ALGO='explicit_gemm',FLEX_GEMM_HASHMAP_RATIO=2.0)
        ops=NS(ALGORITHM='explicit_gemm',HASHMAP_RATIO=2.0)
        before=sys.getprofile();native=runtime.NativeBoundary(cuda,NATIVE_BINDINGS,hook,proxy,cfg,ops,lambda *a,**k:None).install()
        args=(tensor((7486,),torch.uint32),tensor((7486,),torch.uint32),coords,32,32,32,3,3,3,1,1,1)
        try:
            with self.assertRaises(RuntimeError):native.call(runtime.ALLOWED_NATIVE,args,{})
            native.phase='neural'
            for wrong in (args[:-1],(*args[:3],32,32,32,1,3,3,1,1,1),
                          (tensor((1,),torch.uint32),*args[1:]),
                          (*args[:2],tensor((3743,4),torch.int32),*args[3:])):
                with self.assertRaises(RuntimeError):native.call(runtime.ALLOWED_NATIVE,wrong,{})
            with self.assertRaises(RuntimeError):native.call(NATIVE_BINDINGS[0],args,{})
            self.assertEqual(count,[])
            native.call(runtime.ALLOWED_NATIVE,args,{})
            self.assertEqual(native.calls,1);self.assertEqual(count,[1])
            with self.assertRaises(RuntimeError):native.call(runtime.ALLOWED_NATIVE,args,{})
        finally:native.remove()
        self.assertIs(sys.getprofile(),before)
        for n in NATIVE_BINDINGS:self.assertIs(getattr(cuda,n),original)
        self.assertFalse(torch.cuda.is_initialized())

    def test_failure_detaches_real_tensor_frames_without_losing_original_error(self):
        import gc,weakref
        import neural
        from admission import AdmissionStop
        refs=[]
        def failing_stage():
            tensor=torch.ones(3)
            refs.append(weakref.ref(tensor))
            raise AdmissionStop('known resource boundary')
        try:failing_stage()
        except AdmissionStop as exc:
            self.assertIsNotNone(refs[0]())
            detail,rethrow=neural.detach_failure(exc)
            self.assertEqual(detail['error'],'known resource boundary')
            self.assertIn('failing_stage',detail['traceback'])
            self.assertIs(rethrow[0],AdmissionStop)
            self.assertIsNone(exc.__traceback__)
            gc.collect()
            self.assertIsNone(refs[0]())

if __name__=='__main__':unittest.main(verbosity=2)
