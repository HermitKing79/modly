"""CPU-only allocator fixture and genuine safetensors round-trip tests."""
import gc
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
try:import amd_dinov3_allocation as allocation
except ImportError:allocation=None
try:import amd_feature_checkpoint as checkpoint
except ImportError:checkpoint=None

class AllocationTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(allocation,'Production allocation diagnostics are missing')
    def snapshot(self):
        return {'segments':[{'device':0,'address':1024,'total_size':4096,'allocated_size':2048,'active_size':2048,
            'blocks':[{'address':1024,'size':2048,'requested_size':2000,'state':'active_allocated',
                       'frames':[{'filename':'CublasHandlePool.cpp','name':'getCurrentCUDABlasHandle','line':90}]},
                      {'address':3072,'size':2048,'requested_size':0,'state':'inactive','frames':[]}]}],
            'device_traces':[[{'action':'alloc','addr':1024,'size':2048,'frames':[]}]]}
    def test_states_and_origin_are_not_ownership_claims(self):
        row=allocation.summarize(self.snapshot(),2048,4096,[])
        self.assertEqual(row['active_allocated_bytes'],2048)
        self.assertEqual(row['inactive_bytes'],2048)
        self.assertEqual(row['active_blocks'][0]['observed_python_owners'],[])
        self.assertIn('getCurrentCUDABlasHandle',row['active_blocks'][0]['frames'][0]['name'])
    def test_unknown_state_wrong_totals_and_unbounded_input_fail(self):
        for change in ('state','size','frames','devices'):
            s=self.snapshot()
            if change=='state':s['segments'][0]['blocks'][0]['state']='unknown'
            elif change=='size':s['segments'][0]['allocated_size']=100
            elif change=='frames':s['segments'][0]['blocks'][0]['frames']*=257
            else:s['segments'][0]['device']=1
            with self.assertRaises(ValueError):allocation.summarize(s,2048,4096,[])
    def test_weak_registry_never_retains_tensor(self):
        class Tensor:
            device=SimpleNamespace(type='cuda',index=0);dtype='torch.float32';shape=[4]
            def untyped_storage(self):return SimpleNamespace(data_ptr=lambda:1024,nbytes=lambda:16)
        owners=allocation.WeakOwners(limit=2);obj=Tensor()
        owners.track(obj,'owned');self.assertEqual(len(owners.living()),1)
        del obj;gc.collect();self.assertEqual(owners.living(),[])
        owners.track(Tensor(),'second')
        with self.assertRaises(ValueError):owners.track(Tensor(),'unbounded')
    def test_actual_history_snapshot_calls_and_exclusive_capture(self):
        calls=[];s=self.snapshot()
        memory=SimpleNamespace(_record_memory_history=lambda **k:calls.append(('history',k)),
                               _snapshot=lambda **k:calls.append(('snapshot',k)) or s)
        torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True,memory=memory),
            _C=SimpleNamespace(_cuda_getCublasWorkspaceSize=lambda:2048,_cuda_getCublasLtWorkspaceSize=lambda:0))
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            diag=allocation.AllocationDiagnostics(Path(tmp));diag.begin(torch)
            result=diag.capture(torch,2048,4096)
            self.assertEqual(result['summary']['active_allocated_bytes'],2048)
            self.assertEqual(calls[0][1]['max_entries'],512)
            self.assertEqual(calls[0][1]['context'],'state')
            self.assertEqual(calls[0][1]['stacks'],'all')
            self.assertEqual(calls[1],('snapshot',{'augment_with_fx_traces':False}))
            self.assertIsNone(calls[2][1]['enabled'])
            self.assertTrue((Path(tmp)/'allocator-release.json').is_file())
            with self.assertRaises(RuntimeError):diag.capture(torch,2048,4096)
    def test_history_is_forbidden_before_authorization_or_twice(self):
        calls=[];torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False,
            memory=SimpleNamespace(_record_memory_history=lambda **k:calls.append(k))))
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            diag=allocation.AllocationDiagnostics(Path(tmp))
            with self.assertRaises(RuntimeError):diag.begin(torch)
            self.assertEqual(calls,[])
            torch.cuda.is_initialized=lambda:True;diag.begin(torch)
            with self.assertRaises(RuntimeError):diag.begin(torch)
            self.assertEqual(len(calls),1)
    def test_installed_memory_wrapper_replay_uses_bounded_native_arguments(self):
        import ast
        from typing import Literal
        installed=Path('D:/AI/Trellis2AMD/venv/Lib/site-packages/torch/cuda/memory.py')
        tree=ast.parse(installed.read_text());calls=[]
        native=SimpleNamespace(_cuda_record_memory_history=lambda *a:calls.append(('history',a)),
                               _cuda_memorySnapshot=lambda *a:calls.append(('snapshot',a)) or {'segments':[],'device_traces':[]})
        ns=dict(sys=sys,Literal=Literal,_C=native)
        wanted={'_record_memory_history','_record_memory_history_impl','_snapshot'}
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
        self.assertEqual(len(nodes),3)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(installed),'exec'),ns)
        ns['_record_memory_history'](**allocation.HISTORY)
        ns['_snapshot'](augment_with_fx_traces=False)
        ns['_record_memory_history'](enabled=None,context=None,stacks='all',max_entries=512)
        self.assertEqual(calls[0],('history',('all','state','all',512,True,False,False,[])))
        self.assertEqual(calls[1],('snapshot',(None,)))
        self.assertEqual(calls[2],('history',(None,None,'all',512,False,False,False,[])))

class IntegrationTests(unittest.TestCase):
    def test_actual_feature_access_audit_is_exact_and_phase_scoped(self):
        from test_amd_dinov3_control import function
        import amd_dinov3_conditioning as proof
        ns=dict(os=proof.os,Path=Path,phase='cpu',allocation_observer=None,
                checkpoint=Path('D:/models/dino/model.safetensors').resolve(),
                feature_path=Path('C:/owned/conditioning.safetensors').resolve())
        audit=function('audit',ns)
        with self.assertRaises(RuntimeError):audit('open',(str(ns['feature_path']),'wb',0))
        ns.update(phase='feature-save',allocation_observer=object())
        audit('open',(str(ns['feature_path']),'wb',0));audit('open',(str(ns['feature_path']),'rb',0))
        with self.assertRaises(RuntimeError):audit('open',('C:/owned/other.safetensors','wb',0))
        ns['phase']='release'
        with self.assertRaises(RuntimeError):audit('open',(str(ns['feature_path']),'rb',0))
    def test_actual_new_logging_expressions_pass_real_journal(self):
        import ast
        import amd_dinov3_conditioning as proof
        import amd_forward_diagnostics as diagnostics
        from test_amd_dinov3_control import function
        tree=ast.parse(Path(proof.__file__).read_text());messages=[]
        ns=dict(trace=diagnostics.Trace(messages.append),record={'feature_checkpoint':{'verified':True,'shape':[1,1029,1024],
            'provenance':{'component':'DINOv3ViTModel','input_png_sha256':'1'*64}},
            'allocation_diagnosis':{'snapshot_verified':True,'workspace_settings':{'blas_workspace_bytes':79691776},
                'summary':{'active_allocated_bytes':79691776,'active_blocks':[{'frames':[{'name':'native allocation'}]}]}}})
        rows=[n for n in ast.walk(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
            and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name)
            and n.value.func.value.id=='trace' and n.value.func.attr=='emit'
            and len(n.value.args)>1 and isinstance(n.value.args[1],ast.Constant)
            and n.value.args[1].value in ('feature-checkpoint','allocator-release')]
        self.assertEqual(len(rows),2)
        for row in sorted(rows,key=lambda n:n.lineno):exec(compile(ast.Module(body=[row],type_ignores=[]),proof.__file__,'exec'),ns)
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            journal=diagnostics.Journal(Path(tmp)/'events.jsonl')
            for line in messages:journal.consume(line)
            self.assertTrue(journal.close());self.assertIsNone(journal.error)
            saved=[json.loads(v) for v in (Path(tmp)/'events.jsonl').read_text().splitlines()]
            self.assertEqual(saved[0]['checkpoint']['shape'],[1,1029,1024])
            self.assertEqual(saved[1]['allocation_totals']['active_allocated_bytes'],79691776)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(checkpoint,'Production CPU feature writer is missing')
        import numpy as np
        self.np=np;self.array=np.zeros((1,1029,1024),dtype=np.float32)
        self.provenance=dict(component='DINOv3ViTModel',checkpoint_sha256='1'*64,config_sha256='2'*64,
            input_pixel_sha256='3'*64,input_png_sha256='4'*64,extractor_source_sha256='5'*64,
            pipeline_source_sha256='6'*64,harness_commit='7'*40,runtime='PyTorch 2.13 ROCm 10')
    def test_genuine_cpu_safetensors_round_trip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            path=Path(tmp)/'conditioning.safetensors'
            result=checkpoint.write(path,Path(tmp),self.array,self.provenance)
            self.assertEqual(result['shape'],[1,1029,1024]);self.assertEqual(result['dtype'],'F32')
            self.assertTrue(result['verified']);self.assertEqual(result['tensor_bytes'],4214784)
            checkpoint.verify(path,Path(tmp),self.provenance,result['output_sha256'])
            with self.assertRaises(FileExistsError):checkpoint.write(path,Path(tmp),self.array,self.provenance)
    def test_dtype_shape_nonfinite_provenance_and_path_rejected(self):
        arrays=[self.array.astype(self.np.float16),self.array[:,1:,:],self.array.transpose(0,2,1)]
        nan=self.array.copy();nan[0,0,0]=self.np.nan;arrays.append(nan)
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            path=Path(tmp)/'conditioning.safetensors'
            for array in arrays:
                with self.assertRaises(ValueError):checkpoint.write(path,Path(tmp),array,self.provenance)
            with self.assertRaises(ValueError):checkpoint.write(path,Path(tmp),self.array,{**self.provenance,'checkpoint_sha256':'bad'})
            with self.assertRaises(ValueError):checkpoint.write(Path(tmp)/'other.safetensors',Path(tmp),self.array,self.provenance)
    def test_corruption_and_wrong_provenance_fail_verification(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            path=Path(tmp)/'conditioning.safetensors';r=checkpoint.write(path,Path(tmp),self.array,self.provenance)
            with self.assertRaises(ValueError):checkpoint.verify(path,Path(tmp),{**self.provenance,'input_png_sha256':'8'*64},r['output_sha256'])
            data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
            with self.assertRaises(ValueError):checkpoint.verify(path,Path(tmp),self.provenance,r['output_sha256'])

if __name__=='__main__':unittest.main()
