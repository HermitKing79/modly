"""Focused CPU-only saved-input, occupancy and transfer regressions."""
import json
import ast
import gc
import io
from pathlib import Path
import tempfile
import unittest
import weakref
from types import SimpleNamespace
import numpy as np
try:
    import amd_sparse_occupancy_data as data
    import amd_sparse_occupancy_contract as contract
except ModuleNotFoundError:data=contract=None

class OccupancyTests(unittest.TestCase):
    def test_true_committed_reader_and_tamper_rejection(self):
        self.assertIsNotNone(data,'saved-latent reader missing')
        array,row=data.read_input()
        self.assertEqual(list(array.shape),[1,8,16,16,16]);self.assertEqual(array.dtype,np.float32)
        self.assertTrue(np.isfinite(array).all());self.assertEqual(row['file_sha256'],data.INPUT_SHA)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sparse-structure-latent.safetensors';blob=bytearray(data.INPUT.read_bytes());blob[-1]^=1;path.write_bytes(blob)
            with self.assertRaises(ValueError):data.read_input(path)
    def test_native_threshold_coordinates_components_and_readback(self):
        self.assertIsNotNone(data,'occupancy writer missing')
        logits=np.zeros((1,1,64,64,64),np.float32);logits[0,0,1,2,3]=2;logits[0,0,2,3,3]=.1;logits[0,0,1,2,4]=-1
        with tempfile.TemporaryDirectory() as tmp:
            row=data.write_output(Path(tmp),logits,{'fixture':'CPU'})
            self.assertEqual(row['occupied_count'],2);self.assertEqual(row['components']['count'],2)
            self.assertEqual(row['bounding_box'],{'minimum':[1,2,3],'maximum':[2,3,3]})
            self.assertEqual(np.load(Path(tmp)/'occupancy-coordinates.npy',allow_pickle=False).tolist(),[[0,1,2,3],[0,2,3,3]])
            mask=np.load(Path(tmp)/'occupancy-mask.npy',allow_pickle=False);self.assertEqual(int(mask.sum()),2)
            self.assertTrue(row['readback_verified'])
            with self.assertRaises(FileExistsError):data.write_output(Path(tmp),logits,{'fixture':'CPU'})
    def test_output_shape_dtype_nonfinite_and_empty_full_degeneracy(self):
        self.assertIsNotNone(data)
        for value in (np.zeros((64,64,64),np.float32),np.zeros((1,1,64,64,64),np.float16),np.full((1,1,64,64,64),np.nan,np.float32)):
            with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):data.write_output(Path(tmp),value,{})
        for value,empty,degenerate in [(0,True,True),(1,False,True)]:
            with tempfile.TemporaryDirectory() as tmp:
                row=data.write_output(Path(tmp),np.full((1,1,64,64,64),value,np.float32),{})
                self.assertEqual(row['empty'],empty);self.assertEqual(row['degenerate'],degenerate)
    def test_exact_real_shapes_transfer_identity_and_arguments(self):
        self.assertIsNotNone(contract)
        c=contract.TensorTransfers();source=object()
        row=dict(shape=[1,8,16,16,16],stride=[32768,4096,256,16,1],dtype='torch.float32',device='cpu',layout='torch.strided',contiguous=True,requires_grad=False)
        c.begin('input',source,row,'input-transfer')
        for args,kw in [((object(),),{'device':'cuda:0'}),((source,),{'device':'cuda:1'}),((source,),{'device':'cuda:0','dtype':'torch.float16'})]:
            with self.assertRaises(RuntimeError):c.authorize('aten._to_copy.default',args,kw,'input-transfer',row)
        c.authorize('aten._to_copy.default',(source,),{'device':'cuda:0'},'input-transfer',row)
        with self.assertRaises(RuntimeError):c.begin('input',source,row,'input-transfer')
        out={**row,'shape':[1,1,64,64,64],'stride':[262144,262144,4096,64,1],'device':'cuda:0'}
        c.begin('output',source,out,'output-transfer');c.authorize('aten._to_copy.default',(source,),{'device':'cpu'},'output-transfer',out)
        self.assertEqual(c.counts,{'input':1,'output':1})
    def test_preview_exposed_faces_and_limits(self):
        self.assertIsNotNone(data)
        mask=np.zeros((64,64,64),bool);mask[1:3,2:4,3:5]=True
        vertices,faces=data.surface(mask,face_limit=100)
        self.assertEqual(len(faces),24);self.assertEqual(len(vertices),26)
        self.assertEqual(min(v[0] for v in vertices),1);self.assertEqual(max(v[2] for v in vertices),5)
        with self.assertRaises(ValueError):data.surface(mask,face_limit=2)
        with tempfile.TemporaryDirectory() as tmp:
            row=data.write_preview(Path(tmp),mask);self.assertEqual(row['faces'],24)
            self.assertTrue(Path(row['png_path']).is_file());self.assertLess(row['obj_file_bytes'],12*1024**2)
            with self.assertRaises(FileExistsError):data.write_preview(Path(tmp),mask)
    def test_actual_preflight_identity_workspace_and_bounds_cpu_only(self):
        import os
        import amd_sparse_occupancy as entry
        from unittest.mock import patch
        with patch.dict(os.environ,{'WORKSPACE_DIR':str(data.ROOT)}):entry.teardown.verify_workspace()
        row=entry.identity();self.assertEqual(row['tensor_count'],74)
        p=entry.profile();self.assertEqual((p.cpu_seconds,p.gpu_seconds,p.overall_seconds),(120,180,360))
        self.assertEqual(contract.MEMORY_LIMIT,12*1024**3);self.assertEqual(p.validate_config(row['config'])['output_shape'],[1,1,64,64,64])
        self.assertFalse('torch' in __import__('sys').modules)
    def cleanup_replay(self,keep=False,fail=False):
        import amd_sparse_decoder_forward as base
        import amd_forward_diagnostics as diagnostics
        tree=ast.parse(Path(base.__file__).read_text());worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        block=next(n for n in ast.walk(worker) if isinstance(n,ast.With) and any(isinstance(c.context_expr,ast.Call) and isinstance(c.context_expr.func,ast.Name) and c.context_expr.func.id=='BoundedDispatch' for c in n.items))
        start=next(i for i,n in enumerate(block.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='stage' and n.value.args[0].value=='release')
        events=[]
        class Owned:pass
        class Component:
            def __init__(self,m):self.model=m
            def unload(self):events.append('unload');self.model=None
        def clear():
            events.append('clear')
            if fail:raise RuntimeError('clear failed')
        fake=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda n:events.append('sync'),empty_cache=lambda:events.append('empty')),_C=SimpleNamespace(_cuda_clearCublasWorkspaces=clear))
        model=Owned();component=Component(model)
        ns=dict(gc=gc,weakref=weakref,record={},stage=lambda n:events.append(n),phase='output',occupancy=object(),component=component,
            model=model,latent=Owned(),output=Owned(),cpu_output=Owned(),cpu_input=Owned(),saved_array=None,torch=fake,
            trace=diagnostics.Trace(lambda s:None),previous=SimpleNamespace(gpu_memory=lambda t:dict(allocated_bytes=0,reserved_bytes=0),process_memory=lambda:{}),loading_contract=base.loading_contract)
        held=model if keep else None;model=component=None
        exec(compile(ast.Module(body=block.body[start:],type_ignores=[]),base.__file__,'exec'),ns)
        self.assertIsNone(held)
        return ns,events
    def test_actual_cleanup_order_zero_and_binding_failure(self):
        ns,events=self.cleanup_replay();self.assertEqual(events,['release','unload','sync','clear','empty','sync','complete'])
        self.assertTrue(ns['record']['release']['model_collected']);self.assertTrue(ns['record']['release']['input_collected']);self.assertEqual(ns['record']['workspace_clear_calls'],1)
        with self.assertRaisesRegex(RuntimeError,'clear failed'):self.cleanup_replay(fail=True)
        with self.assertRaises(ValueError):self.cleanup_replay(keep=True)
    def good_record(self):
        output=dict(shape=[1,1,64,64,64],dtype='torch.float32',device='cuda:0',requires_grad=False,inference_mode=True,
            all_finite=True,count=262144,minimum=-1.,maximum=1.,mean=0.,stddev=.5)
        return dict(status='PASS',stage='complete',load_calls=1,forward_calls=1,root_decoder_calls=1,fixture=contract.fixture(contract.CONFIG),
            transfer_calls={'input':1,'output':1},forward=dict(status='PASS',output=output,grad_enabled=False,output_is_inference_tensor=True),
            cpu_load={'status':'PASS'},workspace_clear_calls=1,workspace_binding={'verified':True},
            release=dict(allocated_bytes=0,reserved_bytes=0,model_collected=True,input_collected=True,output_collected=True),
            occupancy_artifacts=dict(readback_verified=True,empty=False,degenerate=False))
    def test_acceptance_rejects_bad_output_empty_counts_cleanup_and_monitor(self):
        good=self.good_record();self.assertEqual(contract.classify(good,True,True,True),'PASS')
        for key,value in [('forward_calls',2),('workspace_clear_calls',0),('workspace_binding',{}),('occupancy_artifacts',{'readback_verified':True,'empty':True,'degenerate':True})]:
            self.assertEqual(contract.classify({**good,key:value},True,True,True),'PARTIAL')
        for key,value in [('shape',[1,1,32,32,32]),('dtype','torch.float16'),('all_finite',False)]:
            with self.assertRaises(ValueError):contract.validate_output({**good['forward']['output'],key:value},contract.fixture(contract.CONFIG))
        self.assertEqual(contract.classify(good,True,False,True),'PARTIAL')
    def test_actual_supervisor_occupancy_profile_and_single_attempt_protocol(self):
        import amd_sparse_occupancy as entry
        from unittest.mock import patch
        base=entry.base;expected=entry.identity();written=[]
        gate=dict(status='PASS',strict=True,assign=True,gpu_initialized=False,missing_keys=[],unexpected_keys=[],parameter_bytes=expected['tensor_data_bytes'],
            tensors={k:dict(shape=v['shape'],dtype=v['dtype'],device='cpu',is_meta=False) for k,v in expected['tensors'].items()})
        blob=(base.EVENT+json.dumps(dict(event='cpu-ready',gate_one=gate,checkpoint_sha256=expected['checkpoint_sha256']))+'\n'+base.RESULT+json.dumps(self.good_record())+'\n').encode()
        class Input(io.BytesIO):
            def write(self,value):written.append(value);return super().write(value)
        child=SimpleNamespace(pid=123,stdin=Input(),stdout=io.BytesIO(blob),stderr=io.BytesIO(),returncode=0,wait=lambda timeout:None)
        class Monitor:
            error=None;samples=2
            def __init__(self,*a,**k):pass
            def start(self,timeout):pass
            def require_healthy(self):pass
            def close(self):return True
        class Job:
            def attach(self,child):pass
            def finish(self):return dict(owned_tree_exit_verified=True,forced_job_cleanup=False)
        with tempfile.TemporaryDirectory(dir=data.ROOT.parent) as tmp:
            p=entry.profile();p.evidence=Path(tmp);p.identity=lambda:expected;p.source_identity=lambda:{};p.prepare_private_cache=lambda scratch:{}
            with patch.object(base,'verify_committed_files',return_value={}),patch.object(base.subprocess,'run',return_value=SimpleNamespace(stdout=b'a'*40)), \
                 patch.object(base.subprocess,'Popen',return_value=child),patch.object(base,'windows_observation',return_value={}),patch.object(base,'healthy_observation',return_value=True), \
                 patch.object(base,'OwnedJob',Job),patch.object(base.diagnostics,'ActivityMonitor',Monitor),patch.object(base,'WindowsMonitor',Monitor), \
                 patch.object(base.previous,'reap_owned_child',side_effect=lambda c,p:p.update(worker_exit_verified=True)):
                self.assertEqual(base.supervise(p.evidence/'run','a'*40,profile=p),0)
            marker=json.loads((p.evidence/'attempt-started.json').read_text());self.assertEqual(marker['maximum_forwards'],1)
            result=json.loads((p.evidence/'run/results.json').read_text());self.assertEqual(result['process']['stage_timeouts_seconds'],dict(cpu=120,forward_release=180,overall=360))
        self.assertEqual(written,[b'OWNED_JOB_ATTACHED\n',b'EXECUTE_ONE_DECODER_FORWARD\n'])

if __name__=='__main__':unittest.main()
