"""Real pinned Euler sampler, actual worker hooks/dispatch, CPU fixtures only."""
import ast
import copy
import gc
import inspect
import json
import tempfile
from pathlib import Path
import unittest
import weakref
from unittest.mock import patch
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
import amd_sparse_flow_forward as forward
import amd_sparse_sampling_contract as policy
from amd_sparse_sampling_control import SamplingControl
import amd_forward_diagnostics as diagnostics
from test_amd_sparse_flow_transfer_guard import CPUSink

ROWS=[]
class Identity(torch.nn.Module):
    def forward(self,x):return x
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__();self.self_attn=Identity();self.cross_attn=Identity()
    def forward(self,x):return self.cross_attn(self.self_attn(x))
class Frequency(torch.nn.Module):
    def forward(self,t):
        f=torch.exp(-np.log(10000)*torch.arange(128,dtype=torch.float32)/128)
        torch.ops.aten.to.dtype_layout(f,device=torch.device('cuda:0'))
        return t
class FixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.t_embedder=Frequency();self.blocks=torch.nn.ModuleList(Block() for _ in range(30))
    def forward(self,x,t,cond):
        self.t_embedder(t)
        for b in self.blocks:x=b(x)
        return x*.1+cond.mean()*.02+t.reshape(1,1,1,1,1)*.00001

def hooks(model,control,trace):
    def forbidden(*a,**k):raise RuntimeError('Actual worker hook rejected')
    tree=ast.parse(Path(forward.__file__).read_text())
    required=('gpu','module_call','timestep_tensor')
    nodes=[]
    for name in required:
        node=next((n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name),None)
        if node is None:raise AssertionError('Missing actual production sampling hook: '+name)
        node=copy.deepcopy(node)
        node.body=[ast.copy_location(ast.Global(names=n.names),n) if isinstance(n,ast.Nonlocal) else n for n in node.body]
        nodes.append(node)
    nodes.append(next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='BoundedDispatch'))
    mem=lambda:{'allocated_bytes':0,'reserved_bytes':0,'peak_allocated_bytes':0,'peak_reserved_bytes':0}
    ns=dict(torch=torch,sampling=control,original_tensor=torch.tensor,original_call=torch.nn.Module.__call__,
        phase='forward',module_names={id(v):n for n,v in model.named_modules()},in_timestep=False,
        record={'calls':{}},trace=trace,forbidden=forbidden,contract=policy,
        strict_contract=forward.strict_contract,describe=forward.describe,
        local=type('L',(),{'loading':type('P',(),{'gpu_memory':staticmethod(lambda t:mem())})})(),
        frequency=torch.exp(-np.log(10000)*torch.arange(128,dtype=torch.float32)/128),frequency_sources=[],
        transfers=dict(x=1,t=0,cond=1,neg_cond=1,output=0,frequency=0),op_counts={},placement_sources=set(),copy_source=None,
        TorchDispatchMode=TorchDispatchMode,tree_leaves=tree_leaves)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),forward.__file__,'exec'),ns)
    ns['BoundedDispatch'].__torch_dispatch__=inspect.unwrap(ns['BoundedDispatch'].__torch_dispatch__)
    return ns,mem

def genuine_sampler():
    import amd_sparse_sampling as entry
    return entry.import_sampler()

class SamplingRuntimeTests(unittest.TestCase):
    def test_actual_sampler_12steps_22calls_frequency_guards_and_logging(self):
        cls=genuine_sampler();model=FixtureModel();control=SamplingControl(torch,cls,device='cpu')
        lines=[];trace=diagnostics.Trace(lines.append,limit=192);ns,mem=hooks(model,control,trace)
        inputs={'x':torch.randn(1,8,16,16,16,generator=torch.Generator().manual_seed(policy.SEED)),
            'cond':torch.ones(1,1029,1024)*.25,'neg_cond':torch.zeros(1,1029,1024)}
        with CPUSink(),ns['BoundedDispatch'](),patch.object(torch.nn.Module,'__call__',ns['module_call']), \
             patch.object(torch,'tensor',ns['timestep_tensor']):
            result=control.run(model,inputs,trace,lambda:None,mem)
        self.assertEqual(control.budget.calls,22);self.assertEqual(control.t_copies,22);self.assertEqual(control.frequency_copies,22)
        self.assertEqual(ns['record']['calls']['<root>'],22)
        self.assertTrue(all(v==22 for v in ns['record']['calls'].values()));self.assertEqual(len(ns['record']['calls']),91)
        self.assertEqual([r['forwards'] for r in control.steps],[2]*10+[1]*2)
        self.assertEqual([r['role'] for r in control.models],['positive','negative']*10+['positive']*2)
        self.assertEqual(list(result.samples.shape),[1,8,16,16,16]);self.assertTrue(torch.isfinite(result.samples).all())
        self.assertEqual(len(result.pred_x_t),12);self.assertEqual(len(result.pred_x_0),12)
        self.assertLessEqual(len(lines),192)
        for line in lines:diagnostics.validate_record(json.loads(line.removeprefix(diagnostics.MARKER)))
        with tempfile.TemporaryDirectory() as tmp:
            journal=diagnostics.Journal(Path(tmp)/'compact.jsonl')
            for line in lines:journal.consume(line)
            self.assertTrue(journal.close());self.assertEqual(journal.rows,68)
        ROWS.extend(json.loads(line.removeprefix(diagnostics.MARKER)) for line in lines)
        with self.assertRaises(RuntimeError):control.run(model,inputs,trace,lambda:None,mem)
        with self.assertRaises(RuntimeError):control.budget.next_ticket()
        refs=[weakref.ref(model),weakref.ref(result.samples),*[weakref.ref(v) for v in inputs.values()],*control.issued]
        control.drop_references();model=result=inputs=None;gc.collect()
        self.assertTrue(all(ref() is None for ref in refs),'Sampler closure/history retains owned model/tensors')
    def test_foreign_root_inputs_rejected_before_model_execution(self):
        cls=genuine_sampler()
        for corrupt in ('latent','timestep','conditioning','value'):
            c=SamplingControl(torch,cls,device='cpu');model=FixtureModel();trace=diagnostics.Trace(lambda s:None)
            ns,mem=hooks(model,c,trace);x=torch.ones(1,8,16,16,16);positive=torch.ones(1,1029,1024);negative=torch.zeros_like(positive)
            c.budget.begin_step(0,1,55/56);c.step_x=x;c.positive=positive;c.negative=negative
            c.next_timestep(([1000.],),{'device':torch.device('cpu'),'dtype':torch.float32})
            t=torch.tensor([1000.]);c.staged_timestep(t,t);ns['transfers']['t']=1
            args=[x,t,positive]
            if corrupt=='latent':args[0]=x.clone()
            elif corrupt=='timestep':args[1]=t.clone()
            elif corrupt=='conditioning':args[2]=positive.clone()
            else:t[0]=999
            with self.subTest(corrupt=corrupt),self.assertRaises(RuntimeError):ns['module_call'](model,*args)
            self.assertEqual(c.budget.started,0);self.assertEqual(ns['record']['calls'],{})
    def test_real_cfg_arithmetic_matches_independent_affine_reference(self):
        cls=genuine_sampler();sampler=cls(sigma_min=1e-5)
        x=torch.tensor([-2.,-1.,1.,2.]).reshape(1,1,1,1,4);positive=torch.ones(1);negative=torch.zeros(1);calls=[]
        def model(x,t,c):
            calls.append(c);return x*(2 if c is positive else 3)+(1 if c is positive else 0)
        result=sampler._inference_model(model,x,.5,positive,neg_cond=negative,guidance_strength=7.5,
            guidance_rescale=.7,guidance_interval=(.6,1.0))
        self.assertEqual(len(calls),1);torch.testing.assert_close(result,2*x+1)
        calls.clear();t=.625;result=sampler._inference_model(model,x,t,positive,neg_cond=negative,
            guidance_strength=7.5,guidance_rescale=.7,guidance_interval=(.6,1.0))
        self.assertEqual(calls,[positive,negative])
        a=1-1e-5;b=1e-5+a*t
        cfg=-4.5*x+7.5;ratio=abs(a-2*b)/abs(a+4.5*b)
        expected=(a*x-(.7*ratio+.3)*(a*x-b*cfg))/b
        torch.testing.assert_close(result,expected,rtol=2e-5,atol=2e-5)
    def test_factory_role_identity_and_extra_timestep_rejected(self):
        cls=genuine_sampler();c=SamplingControl(torch,cls,device='cpu');c.budget.begin_step(0,1,55/56)
        good={'device':torch.device('cpu'),'dtype':torch.float32}
        for args,kw in [(([500.],),good),(([1000.+1e-8],),good),(([1000.],),{**good,'dtype':torch.bfloat16}),
                        (([1000.],),{**good,'device':torch.device('cuda:1')})]:
            with self.assertRaises(RuntimeError):c.next_timestep(args,kw)
        c.next_timestep(([1000.],),good)
        with self.assertRaises(RuntimeError):c.next_timestep(([1000.],),good)

if __name__=='__main__':unittest.main()
