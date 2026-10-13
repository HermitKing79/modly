"""Actual production dispatcher with installed Torch CPU tensors and a GPU sink.

The sink prevents backend execution; it does not replace the guard. These tests
catch a missing overload, missing identity/value/scope checks, and loose options.
Run only through the GPU-initialization-blocked cpu-validation entry point.
"""
import ast
import inspect
from pathlib import Path
import unittest
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
import amd_sparse_flow_forward as forward

COPIES=[]
OP=torch.ops.aten.to.dtype_layout

class CPUSink(TorchDispatchMode):
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        values=tree_leaves((args,kwargs))
        gpu=any(isinstance(v,torch.device) and v.type=='cuda' or
                isinstance(v,torch.Tensor) and v.device.type=='cuda' for v in values)
        if gpu:
            if str(func) not in ('aten.to.dtype_layout','aten._to_copy.default'):
                raise AssertionError('CPU sink refuses unexpected GPU operation')
            assert args[0].device.type=='cpu'
            COPIES.append({'operation':str(func),'shape':list(args[0].shape),
                           'kwargs':{k:str(v) for k,v in kwargs.items()}})
            return args[0].clone() # Only CPU work; no target device execution.
        return func(*args,**kwargs)

# Torch decorates mode methods with a lazy Dynamo import. CPU guard replay
# executes the unchanged method body, without importing compiler machinery.
CPUSink.__torch_dispatch__=inspect.unwrap(CPUSink.__torch_dispatch__)

def replay():
    def forbidden(*args,**kwargs):raise RuntimeError('rejected by actual production guard')
    tree=ast.parse(Path(forward.__file__).read_text())
    nodes=[next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='gpu'),
           next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='BoundedDispatch')]
    frequency=torch.exp(-np.log(10000)*torch.arange(128,dtype=torch.float32)/128)
    ns=dict(sampling=None,torch=torch,tree_leaves=tree_leaves,TorchDispatchMode=TorchDispatchMode,
        forbidden=forbidden,phase='forward',in_timestep=True,frequency=frequency,frequency_sources=[],
        transfers={k:0 for k in ('x','t','cond','output','frequency')},op_counts={},
        describe=forward.describe,strict_contract=forward.strict_contract,
        contract=forward.contract,placement_sources=set(),copy_source=('x',None,'cuda:0'))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),forward.__file__,'exec'),ns)
    ns['BoundedDispatch'].__torch_dispatch__=inspect.unwrap(ns['BoundedDispatch'].__torch_dispatch__)
    mode=ns['BoundedDispatch']()
    with CPUSink(),mode:
        source=torch.exp(-np.log(10000)*torch.arange(128,dtype=torch.float32)/128)
    return mode,ns,source

def invoke(mode,source,kwargs=None,args=None,operation=OP):
    with CPUSink():
        return mode.__torch_dispatch__(operation,(),(source,) if args is None else args,
            {'device':torch.device('cuda:0')} if kwargs is None else kwargs)

class TransferGuardTests(unittest.TestCase):
    def test_observed_aten_overload_counts_once_in_inference_mode(self):
        mode,ns,source=replay();start=len(COPIES)
        with CPUSink(),mode:
            result=OP(source,device=torch.device('cuda:0'))
        self.assertTrue(torch.equal(result,source));self.assertEqual(result.device.type,'cpu')
        self.assertEqual(ns['transfers']['frequency'],1)
        self.assertEqual(COPIES[start]['operation'],'aten.to.dtype_layout')
        with self.assertRaises(RuntimeError):invoke(mode,source)
        self.assertEqual(len(COPIES),start+1)
    def test_explicit_valid_defaults_do_not_change_tensor_contract(self):
        mode,ns,source=replay()
        result=invoke(mode,source,{'device':torch.device('cuda:0'),'dtype':torch.float32,
            'layout':torch.strided,'pin_memory':False,'non_blocking':False,'copy':False,
            'memory_format':torch.preserve_format})
        self.assertTrue(torch.equal(result,source));self.assertEqual(ns['transfers']['frequency'],1)
    def test_foreign_equal_clone_is_rejected_before_backend(self):
        mode,ns,source=replay();foreign=source.clone();start=len(COPIES)
        with self.assertRaises(RuntimeError):invoke(mode,foreign)
        self.assertEqual(ns['transfers']['frequency'],0);self.assertEqual(len(COPIES),start)
    def test_unrelated_cpu_to_gpu_overload_is_rejected(self):
        mode,ns,source=replay();start=len(COPIES)
        with self.assertRaises(RuntimeError):
            invoke(mode,source,{'device':torch.device('cuda:0'),'dtype':torch.float32},
                operation=torch.ops.aten.to.device)
        self.assertEqual(ns['transfers']['frequency'],0);self.assertEqual(len(COPIES),start)
    def test_frequency_requires_one_verified_cpu_construction(self):
        mode,ns,source=replay();ns['frequency_sources'].clear();start=len(COPIES)
        with self.assertRaises(RuntimeError):invoke(mode,source)
        self.assertEqual(ns['transfers']['frequency'],0);self.assertEqual(len(COPIES),start)
        mode,ns,source=replay()
        with CPUSink(),mode,self.assertRaises(RuntimeError):
            torch.exp(-np.log(10000)*torch.arange(128,dtype=torch.float32)/128)
    def test_invalid_frequency_construction_fails_before_transfer(self):
        mode,ns,source=replay();ns['frequency_sources'].clear()
        with CPUSink(),mode,self.assertRaises(RuntimeError):torch.exp(torch.ones(128))
        self.assertEqual(ns['frequency_sources'],[]);self.assertEqual(ns['transfers']['frequency'],0)
    def test_wrong_values_shape_or_dtype_are_rejected(self):
        for corruption in ('value','shape','dtype'):
            with self.subTest(corruption=corruption):
                mode,ns,source=replay()
                if corruption=='value':source[0]=2
                elif corruption=='shape':source.resize_(127)
                else:source=source.to(torch.bfloat16)
                start=len(COPIES)
                with self.assertRaises(RuntimeError):invoke(mode,source)
                self.assertEqual(ns['transfers']['frequency'],0);self.assertEqual(len(COPIES),start)
    def test_wrong_devices_and_all_unexpected_options_are_rejected(self):
        changes=[{'device':torch.device(d)} for d in ('cuda:1','cuda','cpu','meta')]+[
            {'device':'cuda:0'},{'dtype':torch.bfloat16},{'layout':torch.sparse_coo},
            {'pin_memory':True},{'pin_memory':0},{'non_blocking':True},{'non_blocking':0},
            {'copy':True},{'copy':0},{'memory_format':torch.contiguous_format},{'alien':False}]
        for change in changes:
            with self.subTest(change=str(change)):
                mode,ns,source=replay();start=len(COPIES)
                with self.assertRaises(RuntimeError):invoke(mode,source,{'device':torch.device('cuda:0'),**change})
                self.assertEqual(ns['transfers']['frequency'],0);self.assertEqual(len(COPIES),start)
    def test_positional_extras_missing_self_and_missing_device_are_rejected(self):
        for kind in ('extra','missing','foreign','no-device'):
            mode,ns,source=replay();start=len(COPIES)
            args={'extra':(source,torch.float32),'missing':(),'foreign':(object(),),'no-device':(source,)}[kind]
            kw={} if kind=='no-device' else {'device':torch.device('cuda:0')}
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):invoke(mode,source,kw,args)
            self.assertEqual(ns['transfers']['frequency'],0);self.assertEqual(len(COPIES),start)
    def test_wrong_phase_or_timestep_scope_is_rejected(self):
        for phase,scope in [('cpu',True),('placement',True),('inputs',True),('output',True),('release',True),('other',True),('forward',False)]:
            mode,ns,source=replay();ns.update(phase=phase,in_timestep=scope);start=len(COPIES)
            with self.subTest(phase=phase),self.assertRaises(RuntimeError):invoke(mode,source)
            self.assertEqual(ns['transfers']['frequency'],0);self.assertEqual(len(COPIES),start)
    def test_legacy_copy_and_shared_duplicate_budget_are_preserved(self):
        for first,second in [(torch.ops.aten._to_copy.default,OP),(OP,torch.ops.aten._to_copy.default)]:
            mode,ns,source=replay();start=len(COPIES)
            invoke(mode,source,operation=first);self.assertEqual(ns['transfers']['frequency'],1)
            with self.assertRaises(RuntimeError):invoke(mode,source,operation=second)
            self.assertEqual(len(COPIES),start+1)

if __name__=='__main__':unittest.main()
