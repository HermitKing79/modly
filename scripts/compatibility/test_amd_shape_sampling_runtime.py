"""Genuine sparse sampler with CPU fixtures and actual production hooks/dispatcher."""
import ast,copy,gc,inspect,json,sys,tempfile,unittest,weakref
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
import amd_shape_slat_forward as forward
import amd_shape_sampling as entry
from amd_shape_sampling_control import ShapeSamplingControl,PHASE_KEY
import amd_forward_diagnostics as diagnostics
from test_amd_sparse_flow_transfer_guard import CPUSink
def forbidden(*a,**k):raise RuntimeError('CPU proof forbids GPU/checkpoint/compilation')
assert not torch.cuda.is_initialized()
torch.cuda._lazy_init=forbidden;torch.load=forbidden;torch.compile=forbidden;torch.set_num_threads(1)
def audit(e,a):
    if e=='open' and isinstance(a[0],(str,bytes)) and str(a[0]).lower().endswith(('.pt','.pth','.ckpt','.bin')):return forbidden()
sys.addaudithook(audit)
import os
os.environ['ATTN_BACKEND']='sdpa'
forward.local.import_class()
from trellis2.modules.sparse import SparseTensor
class Identity(torch.nn.Module):
    def forward(self,x):return x
class Rope(torch.nn.Module):
    def __init__(self):super().__init__();self.freqs=forward.local.rope_frequencies()
    def forward(self,x):
        if x.get_spatial_cache(PHASE_KEY) is None:
            torch.ops.aten.to.dtype_layout(self.freqs,device=torch.device('cuda:0'))
            angles=torch.arange(3743*64,dtype=torch.float32).reshape(3743,64)*.001
            x.register_spatial_cache(PHASE_KEY,torch.polar(torch.ones_like(angles),angles))
        return x
class SelfAttn(torch.nn.Module):
    def __init__(self):super().__init__();self.rope=Rope()
    def forward(self,x):return self.rope(x)
class Block(torch.nn.Module):
    def __init__(self):super().__init__();self.self_attn=SelfAttn();self.cross_attn=Identity()
    def forward(self,x):return self.cross_attn(self.self_attn(x))
class Frequency(torch.nn.Module):
    def forward(self,t):
        freq=torch.exp(-np.log(10000)*torch.arange(128,dtype=torch.float32)/128)
        torch.ops.aten.to.dtype_layout(freq,device=torch.device('cuda:0'));return t
class Fixture(torch.nn.Module):
    def __init__(self):super().__init__();self.t_embedder=Frequency();self.blocks=torch.nn.ModuleList(Block() for _ in range(30))
    def forward(self,x,t,cond):
        self.t_embedder(t)
        for b in self.blocks:x=b(x)
        return x.replace(x.feats*.1+cond.mean()*.02+t.reshape(1,1)*.00001)
def hooks(model,control,trace):
    tree=ast.parse(Path(forward.__file__).read_text());nodes=[]
    for name in ('module_call','timestep_tensor'):
        node=copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name));nodes.append(node)
    nodes.append(next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='BoundedDispatch'))
    mem=lambda:dict(allocated_bytes=0,reserved_bytes=0,peak_allocated_bytes=0,peak_reserved_bytes=0)
    p=control.copy_policy(3743);p.phase='forward';p.rope=model.blocks[0].self_attn.rope.freqs
    ns=dict(torch=torch,sampling=control,policy=p,original_call=torch.nn.Module.__call__,original_tensor=torch.tensor,
        modules={id(v):n for n,v in model.named_modules()},record={'calls':{}},trace=trace,describe=forward.describe,forbidden=forbidden,
        contract=forward.contract,strict=forward.strict,local=type('L',(),{'loading':type('P',(),{'gpu_memory':staticmethod(lambda t:mem())})})(),
        op_counts={},TorchDispatchMode=TorchDispatchMode,tree_leaves=tree_leaves)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),forward.__file__,'exec'),ns)
    ns['BoundedDispatch'].__torch_dispatch__=inspect.unwrap(ns['BoundedDispatch'].__torch_dispatch__)
    return ns,mem
class RuntimeTests(unittest.TestCase):
    def test_actual_native_sampler_hooks_counts_cache_layout_cleanup_and_journal(self):
        cls=entry.import_sampler();model=Fixture();control=ShapeSamplingControl(torch,cls,device='cpu')
        coords=torch.from_numpy(np.load(entry.COORDS,allow_pickle=False));noise=torch.randn(3743,32,generator=torch.Generator().manual_seed(entry.policy.SEED))
        x=SparseTensor(noise,coords);x.register_spatial_cache('layout',[slice(0,3743)]);x.register_spatial_cache('seqlen',torch.tensor([3743],dtype=torch.int64))
        inputs=dict(cond=torch.ones(1,1029,1024)*.25,neg_cond=torch.zeros(1,1029,1024));lines=[];trace=diagnostics.Trace(lines.append,limit=192)
        ns,mem=hooks(model,control,trace)
        with torch.inference_mode(),CPUSink(),ns['BoundedDispatch'](),patch.object(torch.nn.Module,'__call__',ns['module_call']),patch.object(torch,'tensor',ns['timestep_tensor']):
            result=control.run(model,x,inputs,trace,lambda:None,mem)
        self.assertEqual(control.budget.calls,21);self.assertEqual(control.t_copies,21);self.assertEqual(control.frequency_copies,21)
        self.assertEqual(set(ns['record']['calls'].values()),{21});self.assertEqual(len(ns['record']['calls']),91)
        self.assertEqual([r['forwards'] for r in control.steps],[2]*9+[1]*3)
        self.assertEqual([r['role'] for r in control.models],['positive','negative']*9+['positive']*3)
        self.assertIs(result.samples.coords,coords);self.assertTrue(torch.isfinite(result.samples.feats).all());self.assertEqual(len(result.pred_x_t),12)
        self.assertTrue(all(r['cache_reused'] for r in control.models[1:]));self.assertLessEqual(len(lines),192)
        with tempfile.TemporaryDirectory() as d:
            journal=diagnostics.Journal(Path(d)/'trace.jsonl')
            for line in lines:journal.consume(line)
            self.assertTrue(journal.close())
        refs=control.issued.copy();result.samples.clear_spatial_cache();control.drop_references();result=x=inputs=coords=noise=None;gc.collect()
        self.assertTrue(all(r() is None for r in refs))
        with self.assertRaises(RuntimeError):control.budget.begin_step(1,.9)
    def test_native_segment_reduce_available_and_population_std(self):
        self.assertTrue(torch._C._dispatch_has_kernel_for_dispatch_key('aten::segment_reduce','CUDA'))
        coords=torch.from_numpy(np.load(entry.COORDS,allow_pickle=False));v=torch.linspace(-1,1,3743*32).reshape(3743,32)
        x=SparseTensor(v,coords);actual=x.std(dim=[1],keepdim=True)
        torch.testing.assert_close(actual,torch.sqrt((v*v).mean()-v.mean()**2).reshape(1,1))
    def test_actual_copy_policy_negative_cond_and_seqlen(self):
        p=ShapeSamplingControl(torch,None).copy_policy(3743)
        for key,v in [('neg_cond',torch.zeros(1,1029,1024)),('seqlen',torch.tensor([3743],dtype=torch.int64))]:
            p.phase='inputs';p.active=(key,v,'cuda:0');p.check_copy('aten._to_copy.default',(v,),{'device':torch.device('cuda:0')})
            with self.assertRaises(RuntimeError):p.check_copy('aten._to_copy.default',(v,),{'device':torch.device('cuda:0')})
    def test_step_factory_wrong_values_kwargs_duplicate_and_invocation(self):
        c=ShapeSamplingControl(torch,None,device='cpu');c.budget.begin_step(*entry.policy.times()[:2])
        for args,kw in [(([999.],),dict(device=torch.device('cpu'),dtype=torch.float32)),(([float('nan')],),dict(device=torch.device('cpu'),dtype=torch.float32)),(([1000.],),dict(device=torch.device('cuda:1'),dtype=torch.float32))]:
            with self.assertRaises(RuntimeError):c.next_timestep(args,kw)
        c.next_timestep(([1000.],),dict(device=torch.device('cpu'),dtype=torch.float32))
        with self.assertRaises(RuntimeError):c.next_timestep(([1000.],),dict(device=torch.device('cpu'),dtype=torch.float32))
        with self.assertRaises(RuntimeError):c.begin_model((),{},c.copy_policy(3743))
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeTests));assert not torch.cuda.is_initialized()
    print(json.dumps(dict(gpu_initialized=False,real_weights_loaded=False,real_model_forwards=0,schema=str(torch.ops.aten.to.dtype_layout._schema))))
    raise SystemExit(not r.wasSuccessful())
