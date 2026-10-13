"""Actual production CopyPolicy tests; CUDA initialization and checkpoint opens prohibited."""
import hashlib,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_shape_slat_forward as f
def audit(e,a):
    if e.startswith('socket.') and e!='socket.gethostname':raise RuntimeError('CPU test forbids network')
    if e=='open' and isinstance(a[0],(str,bytes)) and str(a[0]).lower().endswith(('.safetensors','.pt','.pth','.ckpt','.gguf')):raise RuntimeError('CPU test forbids checkpoints')
sys.addaudithook(audit)
import torch
def forbidden(*a,**k):raise RuntimeError('CPU test forbids GPU initialization/compilation')
assert not torch.cuda.is_initialized()
torch.cuda._lazy_init=forbidden;torch.compile=forbidden;torch.load=forbidden

class ControlTests(unittest.TestCase):
    def policy(self):return f.CopyPolicy(torch,7)
    def test_actual_inputs_outputs_and_duplicates(self):
        cases=[('x',torch.zeros(7,32)),('coords',torch.zeros(7,4,dtype=torch.int32)),('batch_map',torch.zeros(7,dtype=torch.int64)),('t',torch.zeros(1)),('cond',torch.zeros(1,1029,1024))]
        for k,x in cases:
            p=self.policy();p.phase='inputs';p.active=(k,x,'cuda:0')
            p.check_copy('aten._to_copy.default',(x,),{'device':torch.device('cuda:0')})
            self.assertEqual(p.counts[k],1)
            with self.assertRaises(RuntimeError):p.check_copy('aten._to_copy.default',(x,),{'device':torch.device('cuda:0')})
    def test_wrong_identity_device_dtype_operation_phase(self):
        x=torch.zeros(7,32)
        for args,kwargs,name,phase in [((x.clone(),),{'device':torch.device('cuda:0')},'aten._to_copy.default','inputs'),
            ((x,),{'device':torch.device('cuda:1')},'aten._to_copy.default','inputs'),
            ((x,),{'device':torch.device('cuda:0'),'dtype':torch.float16},'aten._to_copy.default','inputs'),
            ((x,),{'device':torch.device('cuda:0')},'aten.copy_.default','inputs'),
            ((x,),{'device':torch.device('cuda:0')},'aten._to_copy.default','cpu')]:
            p=self.policy();p.phase=phase;p.active=('x',x,'cuda:0')
            with self.assertRaises(RuntimeError):p.check_copy(name,args,kwargs)
    def test_genuine_frequency_overload_and_negative_arguments(self):
        import numpy as np
        x=torch.exp(-np.log(10000)*torch.arange(128,dtype=torch.float32)/128)
        p=self.policy();p.phase='forward';p.scopes=['','t_embedder'];p.frequency=x
        kw=dict(dtype=torch.float32,layout=torch.strided,device=torch.device('cuda:0'),pin_memory=False,non_blocking=False,copy=False)
        p.check_copy('aten.to.dtype_layout',(x,),kw);self.assertEqual(p.counts['frequency'],1)
        for changes in ({'copy':True},{'non_blocking':True},{'device':torch.device('cuda:1')},{'dtype':torch.bfloat16},{'layout':torch.sparse_coo},{'foreign':1}):
            p=self.policy();p.phase='forward';p.scopes=['t_embedder'];p.frequency=x
            with self.assertRaises(RuntimeError):p.check_copy('aten.to.dtype_layout',(x,),{**kw,**changes})
        p=self.policy();p.phase='forward';p.scopes=['t_embedder'];p.frequency=x
        with self.assertRaises(RuntimeError):p.check_copy('aten.to.dtype_layout',(x.clone(),),kw)
    def test_rope_scope_values_and_one_transfer(self):
        # Avoid native/source imports: the real production value checker is supplied
        # the same genuine constructor result already proven by the owned CPU probe.
        x=1/(10000**(torch.arange(21,dtype=torch.float32)/21))
        from unittest.mock import patch
        with patch.object(f.local,'rope_frequencies',return_value=x.clone()):
            p=self.policy();p.phase='forward';p.scopes=['blocks.0.self_attn.rope'];p.rope=x
            p.check_copy('aten._to_copy.default',(x,),{'device':torch.device('cuda:0')})
            with self.assertRaises(RuntimeError):p.check_copy('aten._to_copy.default',(x,),{'device':torch.device('cuda:0')})
            p=self.policy();p.phase='forward';p.scopes=['blocks.1.self_attn.rope'];p.rope=x
            with self.assertRaises(RuntimeError):p.check_copy('aten._to_copy.default',(x,),{'device':torch.device('cuda:0')})
    def test_actual_diagnostics_journal(self):
        import tempfile
        from amd_forward_diagnostics import Trace,Journal
        lines=[];trace=Trace(lines.append)
        trace.emit('state','gpu',gpu=dict(name='AMD Radeon RX 9070 XT',architecture='gfx1201',total_bytes=17095983104))
        trace.emit('memory','release',memory=dict(allocated_bytes=0,reserved_bytes=0))
        trace.call('module:<root>',lambda:None,input=dict(shape=[1,32],dtype='torch.float32',device='cuda:0'))
        with tempfile.TemporaryDirectory() as d:
            j=Journal(Path(d)/'trace.jsonl')
            for line in lines:j.consume(line)
            self.assertTrue(j.close())
    def test_classifier_rejects_incomplete_release(self):
        self.assertEqual(f.contract.classify({},True,True,True),'PARTIAL')
        self.assertEqual(f.contract.classify({'status':'PASS','token_count':7,'release':{'allocated_bytes':79691776}},True,True,True),'PARTIAL')
    def test_exact_visibility_restored_after_offline_guard(self):
        import os
        from unittest.mock import patch
        names=('HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES')
        with patch.dict(os.environ,{k:'' for k in names}):
            self.assertEqual(f.restore_execution_environment({k:'0' for k in names}),{k:'0' for k in names})
            for bad in ('','1','0,1'):
                with self.assertRaises(RuntimeError):f.restore_execution_environment({k:bad for k in names})

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ControlTests))
    assert not torch.cuda.is_initialized()
    print(json.dumps({'gpu_initialized':False,'weights_loaded':False,'schema':str(torch.ops.aten.to.dtype_layout._schema)}))
    raise SystemExit(not result.wasSuccessful())
