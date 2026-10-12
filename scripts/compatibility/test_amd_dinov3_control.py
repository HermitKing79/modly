"""Replay the actual DINO transfer/module functions and supervisor with CPU fixtures."""
import ast
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import amd_dinov3_conditioning as proof
import amd_dinov3_contract as contract
import amd_forward_diagnostics as diagnostics
from test_amd_dinov3_contract import CONFIG


def function(name,namespace):
    tree=ast.parse(Path(proof.__file__).read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    node.body=[ast.copy_location(ast.Global(names=n.names),n) if isinstance(n,ast.Nonlocal) else n for n in node.body]
    exec(compile(ast.Module(body=[node],type_ignores=[]),proof.__file__,'exec'),namespace)
    return namespace[name]


class ControlTests(unittest.TestCase):
    def test_actual_dispatch_rejects_unarmed_copies_and_outside_math(self):
        class Device(str):
            @property
            def type(self):return self.split(':')[0]
            @property
            def index(self):return int(self.split(':')[1]) if ':' in self else None
        class Tensor:
            def __init__(self,device,shape,dtype='torch.float32'):
                self.device=Device(device);self.shape=shape;self.dtype=dtype
            def is_floating_point(self):return True
        class Op:
            def __init__(self,name):self.name=name
            def __str__(self):return self.name
            def __call__(self,*args,**kwargs):return 'executed'
        def leaves(v):
            if isinstance(v,dict):return sum((leaves(x) for x in v.values()),[])
            if isinstance(v,(list,tuple)):return sum((leaves(x) for x in v),[])
            return [v]
        def forbidden(*a,**k):raise RuntimeError('forbidden')
        describe=lambda v:dict(shape=v.shape,dtype=v.dtype,device=str(v.device))
        cpu=Tensor('cpu',[1,3,512,512]);gpu=Tensor('cuda:0',[1,3,512,512])
        ns=dict(phase='cpu',input_source=cpu,output=gpu,normalizing=False,constant_copy=None,
                placement_sources={id(cpu)},transfers={'input':0,'output':0},contract=contract,
                strict_contract=proof.strict_contract,describe=describe,forbidden=forbidden,
                tree_leaves=leaves,TorchDispatchMode=object,trace=diagnostics.Trace(lambda v:None),
                operation_counts={},
                torch=SimpleNamespace(Tensor=Tensor,device=Device,float32='torch.float32',float16='torch.float16',
                    bfloat16='torch.bfloat16',float64='torch.float64'))
        function('gpu',ns)
        tree=ast.parse(Path(proof.__file__).read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='BoundedDispatch')
        exec(compile(ast.Module(body=[node],type_ignores=[]),proof.__file__,'exec'),ns)
        dispatch=ns['BoundedDispatch']().__torch_dispatch__
        copyop=Op('aten._to_copy.default');target={'device':Device('cuda:0')}
        with self.assertRaises(RuntimeError):dispatch(copyop,(),(cpu,),target)
        ns['phase']='placement'
        self.assertEqual(dispatch(copyop,(),(cpu,),target),'executed')
        with self.assertRaises(RuntimeError):dispatch(Op('aten.mm.default'),(),(gpu,gpu),{})
        with self.assertRaises(RuntimeError):dispatch(copyop,(),(Tensor('cpu',[1]),),target)
        ns['phase']='conditioning'
        with self.assertRaises(RuntimeError):dispatch(Op('aten.clone.default'),(),(Tensor('cuda:1',[1]),),{})
        self.assertEqual(dispatch(copyop,(),(cpu,),target),'executed');self.assertEqual(ns['transfers']['input'],1)
        with self.assertRaises(RuntimeError):dispatch(copyop,(),(cpu,),target)
        with self.assertRaises(RuntimeError):dispatch(copyop,(),(cpu,),{**target,'dtype':'torch.float16'})
        ns['normalizing']=True;constant=Tensor('cpu',[3]);ns['constant_copy']=constant
        self.assertEqual(dispatch(copyop,(),(constant,),target),'executed')
        ns['constant_copy']=None
        with self.assertRaises(RuntimeError):dispatch(copyop,(),(constant,),target)
        with self.assertRaises(RuntimeError):dispatch(Op('triton.launch'),(),(gpu,),{})
        ns['phase']='output'
        # Validation rejects the wrong output shape before any actual operation.
        with self.assertRaises(ValueError):dispatch(copyop,(),(gpu,),{'device':Device('cpu')})
        ns['output']=Tensor('cuda:0',[1,1029,1024])
        self.assertEqual(dispatch(copyop,(),(ns['output'],),{'device':Device('cpu')}),'executed')
        with self.assertRaises(RuntimeError):dispatch(copyop,(),(ns['output'],),{'device':Device('cpu')})
        ns['phase']='release'
        with self.assertRaises(RuntimeError):dispatch(Op('aten.mm.default'),(),(gpu,gpu),{})

    def test_actual_cuda_wrapper_and_consumed_transfer(self):
        copied=[]
        def forbidden(*a,**k):raise RuntimeError('forbidden')
        cpu=SimpleNamespace(device=SimpleNamespace(type='cpu'),shape=[1,3,512,512],dtype='torch.float32')
        gpu=SimpleNamespace(device=SimpleNamespace(type='cuda'),shape=[1,3,512,512],dtype='torch.float32')
        def desc(v):return dict(shape=v.shape,dtype=v.dtype,device='cpu' if v.device.type=='cpu' else 'cuda:0')
        ns=dict(phase='conditioning',input_source=None,contract=contract,transfers={'input':0,'output':0},
                describe=desc,forbidden=forbidden,torch=SimpleNamespace(device=lambda v:v,
                ops=SimpleNamespace(aten=SimpleNamespace(_to_copy=SimpleNamespace(default=lambda t,**kw:copied.append((t,kw)) or gpu)))))
        cuda=function('exact_cuda',ns)
        self.assertIs(cuda(cpu),gpu);self.assertIs(ns['input_source'],cpu)
        self.assertIs(cuda(gpu),gpu);self.assertEqual(len(copied),1)
        ns['transfers']['input']=1
        with self.assertRaises(RuntimeError):cuda(cpu)
        for phase,args,kwargs in [('cpu',(),{}),('conditioning',(0,),{}),('conditioning',(),{'device':1})]:
            ns['phase']=phase
            with self.assertRaises(RuntimeError):cuda(gpu,*args,**kwargs)
        ns['phase']='conditioning';gpu.dtype='torch.float16'
        with self.assertRaises(ValueError):contract.validate_transfer('input',dict(shape=cpu.shape,dtype=gpu.dtype,device='cpu'),'cuda:0',{},'conditioning')
        with self.assertRaises(RuntimeError):cuda(gpu)

    def test_actual_module_entry_and_return_use_valid_journal(self):
        obj=object();calls={};messages=[]
        trace=diagnostics.Trace(messages.append)
        def count(name):
            calls[name]=calls.get(name,0)+1
            if calls[name]>1:raise RuntimeError('duplicate')
        def forbidden(*a,**k):raise RuntimeError('forbidden')
        ns=dict(phase='conditioning',allowed_modules={id(obj):'layer.0'},count=count,trace=trace,
                original_call=lambda *a,**k:42,describe=lambda v:{},model=object(),forbidden=forbidden,
                torch=SimpleNamespace(Tensor=type(None)))
        invoke=function('module_call',ns)
        self.assertEqual(invoke(obj),42)
        with tempfile.TemporaryDirectory(dir=proof.local.ROOT.parent) as tmp:
            journal=diagnostics.Journal(Path(tmp)/'trace.jsonl')
            for line in messages:journal.consume(line)
            self.assertTrue(journal.close());self.assertIsNone(journal.error)
            rows=[json.loads(line) for line in (Path(tmp)/'trace.jsonl').read_text().splitlines()]
            self.assertEqual([(r['event'],r['name']) for r in rows],[('enter','module:layer.0'),('return','module:layer.0')])
        with self.assertRaises(RuntimeError):invoke(obj)
        ns['phase']='cpu'
        with self.assertRaises(RuntimeError):invoke(obj)
        ns['phase']='conditioning'
        with self.assertRaises(RuntimeError):invoke(object())

    def good_record(self):
        return dict(status='PASS',stage='complete',load_calls=1,conditioning_calls=1,
            transfer_calls={'input':1,'output':1},cpu_load={'status':'PASS'},
            normalization_copies={'mean':1,'std':1},
            calls={'extractor':1,'features':1,'embeddings':1,'rope_embeddings':1,**{'layer.'+str(i):1 for i in range(24)}},
            output={'shape':[1,1029,1024],'dtype':'torch.float32','device':'cuda:0','count':1053696,
                    'conditioning_width':1024,'all_finite':True,'requires_grad':False,'grad_enabled':False,'is_inference_tensor':True},
            release={'allocated_bytes':0,'reserved_bytes':0,'model_collected':True,'output_collected':True,'rope_cache_entries_after':0})

    def test_incomplete_attempt_cannot_pass(self):
        record=self.good_record();self.assertEqual(contract.classify(record,True,True,True),'PASS')
        for change in [{'load_calls':2},{'conditioning_calls':0},{'transfer_calls':{'input':2,'output':1}},
                       {'release':{**record['release'],'rope_cache_entries_after':1}},
                       {'output':{**record['output'],'all_finite':False}}]:
            self.assertEqual(contract.classify({**record,**change},True,True,True),'PARTIAL')
        for flags in [(False,True,True),(True,False,True),(True,True,False)]:
            self.assertEqual(contract.classify(record,*flags),'PARTIAL')

    def test_actual_normalization_constant_copy_scope(self):
        copied=[];created=[]
        def forbidden(*a,**k):raise RuntimeError('forbidden')
        def make(data,**kw):created.append((data,kw));return data
        ns=dict(phase='conditioning',normalizing=True,constant_copy=None,
                normalization_copies={'mean':0,'std':0},forbidden=forbidden,
                gpu=lambda kw:str(kw.get('device','')).startswith('cuda'),original_as_tensor=make,
                torch=SimpleNamespace(float32='float32',device=lambda v:v,ops=SimpleNamespace(aten=SimpleNamespace(
                    _to_copy=SimpleNamespace(default=lambda v,**kw:copied.append((v,kw)) or v)))))
        invoke=function('exact_as_tensor',ns)
        good={'dtype':'float32','device':'cuda:0'}
        with self.assertRaises(RuntimeError):invoke([.229,.224,.225],**good)
        for invalid in ({**good,'device':'cuda:1'},{**good,'dtype':'float16'},{**good,'extra':True}):
            with self.assertRaises(RuntimeError):invoke([.485,.456,.406],**invalid)
        ns['normalizing']=False
        with self.assertRaises(RuntimeError):invoke([.485,.456,.406],**good)
        ns['normalizing']=True
        invoke([.485,.456,.406],**good);invoke([.229,.224,.225],**good)
        self.assertEqual(ns['normalization_copies'],{'mean':1,'std':1});self.assertIsNone(ns['constant_copy'])
        self.assertEqual(len(copied),2);self.assertTrue(all(kw=={'device':'cpu','dtype':'float32'} for _,kw in created))
        with self.assertRaises(RuntimeError):invoke([.485,.456,.406],**good)
        with self.assertRaises(RuntimeError):invoke([0.,0.,0.],**good)

    def test_profile_and_actual_supervisor_protocol(self):
        written=[]
        class Input(io.BytesIO):
            def write(self,value):written.append(value);return super().write(value)
        expected={'config':CONFIG,'checkpoint_sha256':'verified','tensor_data_bytes':4,
                  'tensors':{'x':{'shape':[1],'dtype':'F32'}}}
        gate={'status':'PASS','gpu_initialized':False,'strict':True,'assign':True,'missing_keys':[],'unexpected_keys':[],
              'parameter_bytes':4,'tensors':{'x':{'shape':[1],'dtype':'F32','device':'cpu','is_meta':False}}}
        data=(proof.supervisor.EVENT+json.dumps({'event':'cpu-ready','gate_one':gate,'checkpoint_sha256':'verified'})+'\n'+
              proof.supervisor.RESULT+json.dumps(self.good_record())+'\n').encode()
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
        with tempfile.TemporaryDirectory(dir=proof.local.ROOT.parent) as tmp:
            profile=proof.profile();profile.evidence=Path(tmp);profile.identity=lambda:expected;profile.source_identity=lambda:{}
            with patch.object(proof.supervisor,'verify_committed_files',return_value={}) as verify, \
                 patch.object(proof.supervisor.subprocess,'run',return_value=SimpleNamespace(stdout=b'published')), \
                 patch.object(proof.supervisor.subprocess,'Popen',return_value=child) as launch, \
                 patch.object(proof.supervisor,'windows_observation',return_value={}), \
                 patch.object(proof.supervisor,'healthy_observation',return_value=True), \
                 patch.object(proof.supervisor,'OwnedJob',Job), \
                 patch.object(proof.supervisor.diagnostics,'ActivityMonitor',Monitor), \
                 patch.object(proof.supervisor,'WindowsMonitor',Monitor), \
                 patch.object(proof.supervisor.previous,'reap_owned_child',side_effect=reap):
                self.assertEqual(proof.supervisor.supervise(profile.evidence/'run','published',profile=profile),0)
                self.assertEqual(written,[b'OWNED_JOB_ATTACHED\n',proof.AUTHORIZE])
                self.assertEqual(launch.call_args.args[0][3],str(Path(proof.__file__).resolve()))
                verify.assert_called_once_with('published',profile.committed_files)
            result=json.loads((profile.evidence/'run/results.json').read_text())
            self.assertEqual(result['classification'],'PASS')
            self.assertEqual(result['process']['stage_timeouts_seconds'],{'cpu':120,'forward_release':180,'overall':360})
            with self.assertRaises(FileExistsError):
                with patch.object(proof.supervisor.subprocess,'run',return_value=SimpleNamespace(stdout=b'published')), \
                     patch.object(proof.supervisor,'verify_committed_files',return_value={}):
                    proof.supervisor.supervise(profile.evidence/'run','published',profile=profile)


if __name__=='__main__':unittest.main()
