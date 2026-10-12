"""CPU replay of the actual guarded worker teardown, with no AI imports."""
import ast
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref
try:import amd_blaslt_teardown as teardown
except ImportError:teardown=None

sys.path.insert(0,str(Path(__file__).parent))
import amd_forward_diagnostics as diagnostics
import amd_dinov3_allocation as allocation
import amd_decoder_contract as strict_contract
import amd_dinov3_contract as inference_contract

SOURCE=Path(__file__).with_name('amd_dinov3_conditioning.py')

class TeardownTests(unittest.TestCase):
    def replay(self,enabled=True,fail=None,residue=0):
        events=[];journal=[];record={};lines=SOURCE.read_text()
        tree=ast.parse(lines)
        body=next(node.body for node in ast.walk(tree) if isinstance(node,ast.With)
                  and ast.unparse(node.items[0].context_expr)=='BoundedDispatch()')
        start=next(i for i,node in enumerate(body) if isinstance(node,ast.Expr)
                   and ast.unparse(node.value)=="stage('release')")
        class Tensor: pass
        class Component:
            def __init__(self,model):self.model=model
            def unload(self):events.append('unload');self.model=None
        class Cache:
            size=1
            def cache_info(self):return SimpleNamespace(currsize=self.size)
            def cache_clear(self):events.append('rope-clear');self.size=0
        def sync(device):
            self.assertEqual(device,0);events.append('sync')
            if fail=='sync':raise RuntimeError('synthetic synchronization failure')
        def clear():
            events.append('clear')
            if fail=='clear':raise RuntimeError('synthetic workspace failure')
        def empty():events.append('empty-cache')
        class Observer:
            def capture(self,torch,allocated,reserved):
                events.append('snapshot')
                return {'history_disabled':True,'summary':{},'snapshot_verified':True}
            def disable_history(self,torch):
                events.append('disable-history')
                return {'history_disabled':True,'snapshot_captured':False,'observed_python_owner_count':0}
        def memory(torch):
            events.append('measure')
            return {'allocated_bytes':residue,'reserved_bytes':residue,'available_bytes':999999}
        def stage(name):record['stage']=name
        trace=diagnostics.Trace(lambda line:journal.append(json.loads(line.removeprefix(diagnostics.MARKER))))
        model=Tensor();output=Tensor()
        namespace=dict(record=record,phase='output',refs=None,workspace_teardown=enabled,
            model=model,output=output,component=Component(model),feature=SimpleNamespace(model=model),
            pipeline=SimpleNamespace(image_cond_model=model),cache=Cache(),allocation_observer=Observer(),
            cpu_output=object(),cond=object(),processed=object(),image=object(),input_source=object(),original_features=object(),
            stage=stage,weakref=weakref,gc=SimpleNamespace(collect=lambda:events.append('gc') or gc.collect()),
            torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=sync,empty_cache=empty),
                                  _C=SimpleNamespace(_cuda_clearCublasWorkspaces=clear)),
            loading=SimpleNamespace(gpu_memory=memory,process_memory=lambda:{}),trace=trace,strict_contract=strict_contract)
        model=output=None
        error=None
        try:exec(compile(ast.Module(body=body[start:],type_ignores=[]),str(SOURCE),'exec'),namespace)
        except Exception as exc:error=exc
        return events,record,journal,error

    def test_synchronized_one_clear_after_refs_before_empty_cache(self):
        events,record,journal,error=self.replay()
        self.assertIsNone(error)
        self.assertEqual(events,['rope-clear','unload','gc','sync','clear','empty-cache','sync','measure','disable-history'])
        self.assertEqual(record['workspace_clear_calls'],1)
        self.assertEqual(record['status'],'PASS')
        self.assertTrue(record['release']['model_collected'])
        self.assertTrue(record['release']['output_collected'])
        self.assertFalse(record['allocation_diagnosis']['snapshot_captured'])
        self.assertEqual([row['event'] for row in journal if row['name']=='blas:clear-workspaces'],['enter','return'])

    def test_clear_failure_stops_no_allocator_recovery_or_retry(self):
        events,record,journal,error=self.replay(fail='clear')
        self.assertIsInstance(error,RuntimeError)
        self.assertEqual(events,['rope-clear','unload','gc','sync','clear'])
        self.assertEqual(record['workspace_clear_calls'],1)
        self.assertNotIn('release',record)
        self.assertEqual([row['event'] for row in journal if row['name']=='blas:clear-workspaces'],['enter','error'])

    def test_failed_sync_never_clears_workspaces(self):
        events,record,journal,error=self.replay(fail='sync')
        self.assertIsInstance(error,RuntimeError)
        self.assertNotIn('clear',events);self.assertNotIn('empty-cache',events)

    def test_residual_allocation_still_fails_and_gets_one_snapshot(self):
        events,record,journal,error=self.replay(residue=79691776)
        self.assertIsInstance(error,ValueError)
        self.assertEqual(events.count('clear'),1);self.assertEqual(events.count('snapshot'),1)
        self.assertEqual(record['release']['allocated_bytes'],79691776)

    def test_original_mode_does_not_invoke_private_cleanup(self):
        events,record,journal,error=self.replay(enabled=False)
        self.assertIsNone(error);self.assertNotIn('clear',events)
        self.assertIn('snapshot',events);self.assertNotIn('workspace_clear_calls',record)

    def test_no_snapshot_history_disable_is_scoped_and_once(self):
        calls=[]
        torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True,
            memory=SimpleNamespace(_record_memory_history=lambda **kw:calls.append(kw))))
        diag=allocation.AllocationDiagnostics(Path('.'))
        self.assertTrue(hasattr(diag,'disable_history'),'Missing bounded no-snapshot history shutdown')
        with self.assertRaises(RuntimeError):diag.disable_history(torch)
        diag.begin(torch);result=diag.disable_history(torch)
        self.assertTrue(result['history_disabled']);self.assertFalse(result['snapshot_captured'])
        self.assertEqual(calls[-1],dict(enabled=None,context=None,stacks='all',max_entries=512))
        with self.assertRaises(RuntimeError):diag.disable_history(torch)

    def test_inference_contract_and_zero_release_unchanged(self):
        permit=inference_contract.SingleConditioning();self.assertEqual(permit.run(lambda:7),7)
        with self.assertRaises(RuntimeError):permit.run(lambda:8)
        out=dict(shape=[1,1029,1024],dtype='torch.float32',device='cuda:0',all_finite=True,count=1053696,conditioning_width=1024)
        inference_contract.validate_output(out)
        for key,value in [('shape',[1,1024,1024]),('dtype','torch.float16'),('device','cpu'),('all_finite',False)]:
            with self.assertRaises(ValueError):inference_contract.validate_output({**out,key:value})
        with self.assertRaises(ValueError):strict_contract.validate_release(dict(allocated_bytes=1,reserved_bytes=0,model_collected=True))

    def test_unverified_python_replacement_or_initialized_gpu_rejected(self):
        self.assertIsNotNone(teardown,'Missing production binding validator')
        for initialized in (False,True):
            fake=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:initialized),
                _C=SimpleNamespace(_cuda_clearCublasWorkspaces=lambda:None))
            with self.assertRaises(RuntimeError):teardown.verify_runtime_binding(fake)

    def test_classifier_requires_exact_clear_and_unchanged_release(self):
        self.assertIsNotNone(teardown,'Missing production teardown classifier')
        good=dict(status='PASS',stage='complete',calls={'extractor':1,'features':1,'embeddings':1,'rope_embeddings':1,
            **{'layer.'+str(i):1 for i in range(24)}},load_calls=1,conditioning_calls=1,
            transfer_calls={'input':1,'output':1},normalization_copies={'mean':1,'std':1},cpu_load={'status':'PASS'},
            output=dict(shape=[1,1029,1024],dtype='torch.float32',device='cuda:0',all_finite=True,count=1053696,
                conditioning_width=1024,requires_grad=False,grad_enabled=False,is_inference_tensor=True),
            release=dict(allocated_bytes=0,reserved_bytes=0,model_collected=True,output_collected=True,rope_cache_entries_after=0),
            workspace_clear_calls=1,workspace_binding={'verified':True},
            allocation_diagnosis={'snapshot_captured':False,'history_disabled':True})
        self.assertEqual(teardown.classify(good,True,True,True),'PASS')
        for key,value in [('workspace_clear_calls',0),('workspace_clear_calls',2),('workspace_clear_calls',True),
                          ('workspace_binding',{}),('allocation_diagnosis',{})]:
            self.assertEqual(teardown.classify({**good,key:value},True,True,True),'PARTIAL')
        self.assertEqual(teardown.classify({**good,'release':{**good['release'],'allocated_bytes':1}},True,True,True),'PARTIAL')

    def test_workspace_guard_rejects_before_attempt_can_be_consumed(self):
        self.assertTrue(hasattr(teardown,'verify_workspace'),'Missing CLI workspace precondition')
        with patch.dict('os.environ',{'WORKSPACE_DIR':str(teardown.previous.local.ROOT)}):
            teardown.verify_workspace()
        for value in ('',str(teardown.previous.local.ROOT.parent)):
            with patch.dict('os.environ',{'WORKSPACE_DIR':value}):
                with self.assertRaises(RuntimeError):teardown.verify_workspace()

if __name__=='__main__':unittest.main()
