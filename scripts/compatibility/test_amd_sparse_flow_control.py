"""CPU replay of actual production dispatch and teardown; no Torch import."""
import ast
import gc
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref
sys.path.insert(0,str(Path(__file__).parent))
import amd_sparse_flow_forward as forward
import amd_sparse_flow_contract as policy
import amd_forward_diagnostics as diagnostics

SOURCE=Path(forward.__file__)
TREE=ast.parse(SOURCE.read_text())
class Device:
    def __init__(self,value):
        value=str(value);self.type=value.split(':')[0];self.index=int(value.split(':')[1]) if ':' in value else None
    def __str__(self):return self.type+(':'+str(self.index) if self.index is not None else '')
    def __eq__(self,other):return isinstance(other,Device) and str(self)==str(other)
class Tensor:
    def __init__(self,shape,dtype='torch.float32',device='cpu'):
        self.shape=shape;self.dtype=dtype;self.device=Device(device)
    def is_floating_point(self):return self.dtype in ('torch.float32','torch.bfloat16','torch.float16','torch.float64')
    def is_complex(self):return self.dtype.startswith('torch.complex')
def leaves(value):
    if isinstance(value,dict):return [v for i in value.values() for v in leaves(i)]
    if isinstance(value,(list,tuple)):return [v for i in value for v in leaves(i)]
    return [value]
class Operation:
    def __init__(self,name):self.name=name;self.calls=0
    def __str__(self):return self.name
    def __call__(self,*args,**kwargs):self.calls+=1;return args[0] if args else None

def dispatch_replay(phase,source=None,key='x',target='cuda:0',in_timestep=False):
    def forbidden(*a,**k):raise RuntimeError('rejected by actual dispatch')
    torch=SimpleNamespace(Tensor=Tensor,device=Device,float32='torch.float32',bfloat16='torch.bfloat16',
        complex64='torch.complex64',float16='torch.float16',float64='torch.float64',complex128='torch.complex128',
        preserve_format='preserve',strided='torch.strided',equal=lambda a,b:a is b)
    frequency=Tensor([128])
    trace=diagnostics.Trace(lambda line:diagnostics.validate_record(json.loads(line.removeprefix(diagnostics.MARKER))))
    namespace=dict(torch=torch,tree_leaves=leaves,phase=phase,forbidden=forbidden,
        TorchDispatchMode=object,strict_contract=forward.strict_contract,contract=policy,
        placement_sources={id(source)} if source is not None else set(),copy_source=(key,source,target),
        transfers={k:0 for k in ('x','t','cond','output','frequency')},in_timestep=in_timestep,
        frequency=frequency,frequency_sources=[],op_counts={},describe=forward.describe,trace=trace)
    gpu_node=next(n for n in ast.walk(TREE) if isinstance(n,ast.FunctionDef) and n.name=='gpu')
    node=next(n for n in ast.walk(TREE) if isinstance(n,ast.ClassDef) and n.name=='BoundedDispatch')
    exec(compile(ast.Module(body=[gpu_node,node],type_ignores=[]),str(SOURCE),'exec'),namespace)
    mode=namespace['BoundedDispatch']()
    if phase=='forward' and in_timestep:
        mode.__torch_dispatch__(Operation('aten.exp.default'),(),(frequency,),{})
    return mode,namespace

class ControlTests(unittest.TestCase):
    def good_record(self):
        return dict(status='PASS',stage='complete',load_calls=1,forward_calls=1,workspace_clear_calls=1,
            workspace_binding={'verified':True},transfers={k:1 for k in ('x','t','cond','output','frequency')},
            calls={'<root>':1,**{f'blocks.{i}':1 for i in range(30)},**{f'blocks.{i}.{a}':1 for i in range(30) for a in ('self_attn','cross_attn')}},
            output=dict(shape=[1,8,16,16,16],dtype='torch.float32',device='cuda:0',all_finite=True,count=32768,
                requires_grad=False,is_inference_tensor=True,grad_enabled=False),
            release=dict(allocated_bytes=0,reserved_bytes=0,model_collected=True,owned_tensors_collected=True))
    def test_actual_module_calls_journal_and_phase_scope(self):
        node=next(n for n in ast.walk(TREE) if isinstance(n,ast.FunctionDef) and n.name=='module_call')
        node=__import__('copy').deepcopy(node)
        node.body=[ast.copy_location(ast.Global(names=n.names),n) if isinstance(n,ast.Nonlocal) else n for n in node.body]
        root=object();block=object();timestep=object();record={'calls':{}};lines=[]
        def forbidden(*a,**k):raise RuntimeError('module rejected')
        ns=dict(phase='forward',module_names={id(root):'',id(block):'blocks.0',id(timestep):'t_embedder'},
            in_timestep=False,record=record,forbidden=forbidden,describe=forward.describe,
            original_call=lambda *a,**k:Tensor([1],device='cuda:0'),trace=diagnostics.Trace(lines.append),
            local=SimpleNamespace(loading=SimpleNamespace(gpu_memory=lambda t:{'allocated_bytes':0,'reserved_bytes':0})),
            torch=object(),contract=policy)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
        call=ns['module_call'];call(root);call(block);call(timestep)
        self.assertEqual(record['calls'],{'<root>':1,'blocks.0':1});self.assertFalse(ns['in_timestep'])
        for line in lines:diagnostics.validate_record(json.loads(line.removeprefix(diagnostics.MARKER)))
        with self.assertRaises(RuntimeError):call(root)
        with self.assertRaises(RuntimeError):call(object())
        ns['phase']='cpu'
        with self.assertRaises(RuntimeError):call(block)
    def test_acceptance_fails_for_bad_counts_release_or_observer(self):
        good=self.good_record();self.assertEqual(policy.classify(good,True,True,True),'PASS')
        for change in [{'forward_calls':2},{'load_calls':2},{'calls':{}},{'transfers':{}},
                       {'release':{**good['release'],'reserved_bytes':1}},{'workspace_clear_calls':0}]:
            self.assertEqual(policy.classify({**good,**change},True,True,True),'PARTIAL')
        for flags in [(False,True,True),(True,False,True),(True,True,False)]:
            self.assertEqual(policy.classify(good,*flags),'PARTIAL')
    def test_actual_supervisor_sparse_flow_protocol_and_gate_fail_closed(self):
        written=[]
        class Input(io.BytesIO):
            def write(self,value):written.append(value);return super().write(value)
        expected={'config':policy.CONFIG,'checkpoint_sha256':'verified','tensor_data_bytes':2584359952,
                  'tensors':{k:{**v,'dtype':'BF16'} for k,v in policy.schema().items()}}
        gate=dict(status='PASS',gpu_initialized=False,strict=True,assign=True,missing_keys=[],unexpected_keys=[],
            parameter_bytes=expected['tensor_data_bytes'],payload_entries=640,reversible_widening_count=220,feature_verified=True,
            tensors={k:{**v,'device':'cpu','is_meta':False} for k,v in policy.schema().items()},
            buffers={'rope_phases':{'shape':[4096,64],'dtype':'C64','device':'cpu','is_meta':False}})
        for bad in (False,True):
            written.clear();candidate={**gate,'reversible_widening_count':219} if bad else gate
            data=(forward.supervisor.EVENT+json.dumps({'event':'cpu-ready','gate_one':candidate,'checkpoint_sha256':'verified'})+'\n'+
                  forward.supervisor.RESULT+json.dumps(self.good_record())+'\n').encode()
            child=SimpleNamespace(pid=123,stdin=Input(),stdout=io.BytesIO(data),stderr=io.BytesIO(),returncode=0,wait=lambda timeout:None)
            class Monitor:
                error=None;samples=2
                def __init__(self,*a,**k):pass
                def start(self,timeout):pass
                def require_healthy(self):pass
                def close(self):return True
            class Job:
                def attach(self,child):pass
                def finish(self):return dict(owned_tree_exit_verified=True,forced_job_cleanup=False)
            def reap(c,p):p.update(worker_exit_verified=True)
            with tempfile.TemporaryDirectory(dir=forward.local.ROOT.parent) as tmp:
                profile=forward.profile();profile.evidence=Path(tmp);profile.identity=lambda:expected;profile.source_identity=lambda:{}
                with patch.object(forward.supervisor,'verify_committed_files',return_value={}), \
                     patch.object(forward.supervisor.subprocess,'run',return_value=SimpleNamespace(stdout=b'published')), \
                     patch.object(forward.supervisor.subprocess,'Popen',return_value=child), \
                     patch.object(forward.supervisor,'windows_observation',return_value={}), \
                     patch.object(forward.supervisor,'healthy_observation',return_value=True), \
                     patch.object(forward.supervisor,'OwnedJob',Job), \
                     patch.object(forward.supervisor.diagnostics,'ActivityMonitor',Monitor), \
                     patch.object(forward.supervisor,'WindowsMonitor',Monitor), \
                     patch.object(forward.supervisor.previous,'reap_owned_child',side_effect=reap):
                    result=forward.supervisor.supervise(profile.evidence/'run','published',profile=profile)
                self.assertEqual(result,1 if bad else 0)
                self.assertEqual(written,[b'OWNED_JOB_ATTACHED\n'] if bad else [b'OWNED_JOB_ATTACHED\n',forward.AUTHORIZE])
                saved=json.loads((profile.evidence/'run/results.json').read_text())
                if not bad:self.assertEqual(saved['process']['stage_timeouts_seconds'],{'cpu':180,'forward_release':240,'overall':480})
    def test_actual_exact_transfers_and_duplicate_rejection(self):
        for key,shape in [('x',[1,8,16,16,16]),('t',[1]),('cond',[1,1029,1024]),('output',[1,8,16,16,16])]:
            phase='output' if key=='output' else 'inputs';target='cpu' if key=='output' else 'cuda:0'
            source=Tensor(shape,device='cuda:0' if key=='output' else 'cpu')
            mode,ns=dispatch_replay(phase,source,key,target)
            op=Operation('aten._to_copy.default');mode.__torch_dispatch__(op,(),(source,),{'device':Device(target)})
            self.assertEqual(op.calls,1);self.assertEqual(ns['transfers'][key],1)
            with self.assertRaises(RuntimeError):mode.__torch_dispatch__(op,(),(source,),{'device':Device(target)})
    def test_transfer_negatives_actual_guard(self):
        for key,source,kwargs in [('x',Tensor([1]),{'device':Device('cuda:0')}),
            ('x',Tensor([1,8,16,16,16]),{'device':Device('cuda:1')}),
            ('x',Tensor([1,8,16,16,16]),{'device':Device('cuda:0'),'dtype':'torch.bfloat16'}),
            ('x',Tensor([1,8,16,16,16],dtype='torch.bfloat16'),{'device':Device('cuda:0')})]:
            mode,_=dispatch_replay('inputs',source,key)
            with self.assertRaises((RuntimeError,ValueError)):mode.__torch_dispatch__(Operation('aten._to_copy.default'),(),(source,),kwargs)
    def test_phase_device_and_dtype_fail_closed(self):
        tensor=Tensor([1],device='cuda:0')
        for phase,op,kwargs in [('cpu','aten.add.Tensor',{}),('placement','aten.mm.default',{}),
            ('release','aten.mm.default',{}),('other','aten.detach.default',{}),
            ('forward','aten._to_copy.default',{'dtype':'torch.float16'}),
            ('forward','aten._to_copy.default',{'dtype':'torch.int8'}),
            ('forward','aten._to_copy.default',{'device':Device('cpu')}),
            ('forward','aten._to_copy.default',{'layout':'sparse'}),('forward','not_aten.mm',{})]:
            mode,_=dispatch_replay(phase)
            with self.assertRaises(RuntimeError):mode.__torch_dispatch__(Operation(op),(),(tensor,),kwargs)
        mode,_=dispatch_replay('forward')
        with self.assertRaises(RuntimeError):mode.__torch_dispatch__(Operation('aten.add.Tensor'),(),(Tensor([1],device='cuda:1'),),{})
    def test_cpu_frequency_only_in_timestep_once(self):
        mode,ns=dispatch_replay('forward',in_timestep=True);freq=ns['frequency']
        op=Operation('aten._to_copy.default');kw={'device':Device('cuda:0')}
        mode.__torch_dispatch__(op,(),(freq,),kw);self.assertEqual(ns['transfers']['frequency'],1)
        with self.assertRaises(RuntimeError):mode.__torch_dispatch__(op,(),(freq,),kw)
        mode,ns=dispatch_replay('forward',in_timestep=True)
        mode.__torch_dispatch__(op,(),(ns['frequency'],),{**kw,'dtype':'torch.float32','layout':'torch.strided'})
        for enabled,source in [(False,freq),(True,Tensor([128])),(True,Tensor([127]))]:
            mode,ns=dispatch_replay('forward',in_timestep=enabled)
            if not enabled:source=ns['frequency']
            with self.assertRaises(RuntimeError):mode.__torch_dispatch__(op,(),(source,),kw)
    def test_mixed_types_permitted_without_changes(self):
        for dtype in ('torch.float32','torch.bfloat16','torch.complex64'):
            mode,_=dispatch_replay('forward');op=Operation('aten.mul.Tensor')
            mode.__torch_dispatch__(op,(),(Tensor([1],dtype,device='cuda:0'),),{})
            self.assertEqual(op.calls,1)
    def test_unrelated_cpu_to_gpu_copy_is_rejected(self):
        mode,ns=dispatch_replay('forward',in_timestep=True)
        op=Operation('aten.copy_.default')
        with self.assertRaises(RuntimeError):
            mode.__torch_dispatch__(op,(),(Tensor([128],device='cuda:0'),ns['frequency']),{})
        self.assertEqual(op.calls,0);self.assertEqual(ns['transfers']['frequency'],0)
    def test_placement_allows_only_unchanged_observed_defaults(self):
        for dtype in ('torch.float32','torch.bfloat16','torch.complex64'):
            source=Tensor([2],dtype);mode,_=dispatch_replay('placement',source)
            options={'dtype':dtype,'layout':'torch.strided','device':Device('cuda:0')}
            op=Operation('aten._to_copy.default');mode.__torch_dispatch__(op,(),(source,),options)
            self.assertEqual(op.calls,1)
            for changes in ({'dtype':'torch.float16'},{'layout':'sparse'},{'non_blocking':True},{'device':Device('cuda:1')},{'memory_format':'channels_last'}):
                with self.assertRaises(RuntimeError):mode.__torch_dispatch__(op,(),(source,),{**options,**changes})
    def test_internal_float_casts_and_complex_preservation(self):
        for source_dtype,target_dtype,ok in [('torch.float32','torch.bfloat16',True),('torch.bfloat16','torch.float32',True),
            ('torch.complex64','torch.complex64',True),('torch.float32','torch.complex64',False),('torch.complex64','torch.float32',False)]:
            mode,_=dispatch_replay('forward');op=Operation('aten._to_copy.default');args=(Tensor([2],source_dtype,device='cuda:0'),)
            kwargs={'dtype':target_dtype,'layout':'torch.strided'}
            if ok:mode.__torch_dispatch__(op,(),args,kwargs);self.assertEqual(op.calls,1)
            else:
                with self.assertRaises(RuntimeError):mode.__torch_dispatch__(op,(),args,kwargs)
    def test_inert_pipeline_import_only(self):
        self.assertFalse(policy.prohibited_import('trellis2.pipelines'))
        for name in ('trellis2.pipelines.trellis2_image_to_3d','flex_gemm','cumesh._C','o_voxel'):
            self.assertTrue(policy.prohibited_import(name))
    def test_exact_deadline_profile_and_production_gates(self):
        p=forward.profile()
        self.assertEqual((p.cpu_seconds,p.gpu_seconds,p.overall_seconds),(180,240,480))
        self.assertEqual(policy.MEMORY_LIMIT,12*1024**3)
        self.assertIs(p.authorize_placement,policy.authorize_placement)
        self.assertIn('scripts/compatibility/amd_sparse_flow_forward.py',p.committed_files)
        self.assertEqual(p.prepare_private_cache(Path('.'))['copied_entries'],0)
    def test_release_actual_order_and_zero_memory(self):
        events=[];lines=[]
        class Component:
            def __init__(self,m):self.model=m
            def unload(self):events.append('unload');self.model=None
        model=Tensor([1]);output=Tensor([1]);refs=[weakref.ref(model),weakref.ref(output)]
        component=Component(model);model=output=None
        fake=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda d:events.append('sync'),empty_cache=lambda:events.append('empty')),
                             _C=SimpleNamespace(_cuda_clearCublasWorkspaces=lambda:events.append('clear')))
        memory={'allocated_bytes':0,'reserved_bytes':0,'available_bytes':999}
        trace=diagnostics.Trace(lines.append)
        with patch.object(forward.local.loading,'gpu_memory',return_value=memory),patch.object(forward.local.loading,'process_memory',return_value={}):
            row=forward.release(fake,component,refs,trace)
        self.assertEqual(events,['unload','sync','clear','empty','sync'])
        self.assertTrue(row['model_collected']);self.assertTrue(row['owned_tensors_collected'])
        for line in lines:diagnostics.validate_record(json.loads(line.removeprefix(diagnostics.MARKER)))
    def test_release_clear_failure_and_retained_tensor(self):
        class Component:
            def unload(self):pass
        def bad():raise RuntimeError('clear failed')
        fake=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda d:None,empty_cache=lambda:None),
                             _C=SimpleNamespace(_cuda_clearCublasWorkspaces=bad))
        kept=Tensor([1]);refs=[weakref.ref(kept)]
        with self.assertRaisesRegex(RuntimeError,'clear failed'):forward.release(fake,Component(),refs,diagnostics.Trace(lambda _:None))
        fake._C._cuda_clearCublasWorkspaces=lambda:None
        with patch.object(forward.local.loading,'gpu_memory',return_value={'allocated_bytes':0,'reserved_bytes':0}),patch.object(forward.local.loading,'process_memory',return_value={}):
            with self.assertRaises(ValueError):forward.release(fake,Component(),refs,diagnostics.Trace(lambda _:None))

if __name__=='__main__':unittest.main()
