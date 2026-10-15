"""Real CPU Torch operations; no native import, model or GPU execution."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
import hashlib

E=Path(__file__).resolve().parent
sys.path.insert(0,str(E))
sys.path.insert(0,str(E.parents[2]/'scripts/compatibility'))
import torch
GPU_ENTRIES=[]
def deny_gpu(*a,**k):
    GPU_ENTRIES.append('GPU entry attempted')
    raise RuntimeError('CPU test forbids GPU entry')
assert not torch.cuda.is_initialized()
torch.cuda._lazy_init=deny_gpu
torch.cuda.init=deny_gpu
torch._C._cuda_init=deny_gpu
torch.set_num_threads(2)
import admission
candidate=None
if (E/'chunked_final.py').exists():
    spec=importlib.util.spec_from_file_location('chunked_final',E/'chunked_final.py')
    candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)

source=E.parent/'amd-wsl-native-decoder-preparation-evidence/pinned-linux-submanifold_conv3d.py.txt'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='67641d13e06ca135a4f90eb71a01f7d8d32448bb5aa7c31b858c69368811fb6b'
cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='SubMConv3dFunction')
fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_sparse_submanifold_conv_forward')
fn.decorator_list=[]
env={'torch':torch,'Algorithm':types.SimpleNamespace(EXPLICIT_GEMM='explicit_gemm'),
     'spconv':types.SimpleNamespace(ALGORITHM='explicit_gemm'),'SubMConv3dNeighborCache':dict,
     'Optional':__import__('typing').Optional}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),env)
original=env['_sparse_submanifold_conv_forward']
OBSERVED=[]

def fixture(dtype):
    # Deliberately non-spatial row order and gaps. Adjacent cells cross row tiles.
    coords=[(0,1,0,0),(0,0,0,0),(0,2,1,0),(0,1,1,0),(0,3,0,1),
            (0,2,0,0),(0,0,1,0),(0,3,1,0),(0,1,0,1),(0,2,0,1),(0,0,0,1)]
    slots=[(a,b,c) for a in (-1,0,1) for b in (-1,0,1) for c in (-1,0,1)]
    membership={c:i for i,c in enumerate(coords)}
    indices=[[membership.get((batch,d+a,h+b,w+c),0xffffffff) for a,b,c in slots] for batch,d,h,w in coords]
    nm=torch.tensor(indices,dtype=torch.uint32)
    features=((torch.arange(len(coords)*4).reshape(len(coords),4)%17)-8).to(dtype)/32
    weight=torch.zeros((3,3,3,3,4),dtype=dtype)
    weight[:,1,1,1,:]=torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,1]],dtype=dtype)
    weight[:,2,1,1,:]=torch.tensor([[0,.5,0,0],[0,0,.25,0],[.125,0,0,0]],dtype=dtype)
    weight[:,1,0,1,:]=torch.tensor([[0,0,0,.25],[.125,0,0,0],[0,.5,0,0]],dtype=dtype)
    bias=torch.tensor([.03125,-.0625,.125],dtype=dtype)
    return coords,slots,features,nm,weight,bias

class Equivalence(unittest.TestCase):
    def test_global_reference_bias_sentinels_cross_chunks_short_tail(self):
        self.assertIsNotNone(candidate,'row-chunked implementation not yet written')
        for dtype in (torch.float32,torch.float16):
            coords,slots,f,nm,w,b=fixture(dtype)
            for bias in (None,b):
                expected=original(f,{'neighbor_map':nm},w,bias)
                # Independent coordinate-tuple reference, never kernel arrays/GPU output.
                ref=torch.zeros_like(expected,dtype=torch.float64)
                lookup={c:i for i,c in enumerate(coords)}
                for row,(batch,d,h,z) in enumerate(coords):
                    for ki,(a,c,e) in enumerate(slots):
                        index=lookup.get((batch,d+a,h+c,z+e))
                        if index is not None:ref[row]+=w.view(3,27,4)[:,ki].double() @ f[index].double()
                    if bias is not None:ref[row]+=bias.double()
                self.assertTrue(torch.equal(expected,ref.to(dtype)))
                for tile in (1,3,4,32):
                    events=[]
                    actual=candidate.row_chunked_explicit(torch,f,nm,w,bias,tile,
                        lambda phase,start,end:events.append((phase,start,end)))
                    self.assertTrue(torch.equal(actual,expected))
                    self.assertEqual(actual.dtype,dtype)
                    self.assertEqual(events[0],('entry',0,min(tile,len(coords))))
                    self.assertEqual(events[-1],('return',(len(coords)-1)//tile*tile,len(coords)))
                    self.assertEqual(sum(e[2]-e[1] for e in events if e[0]=='return'),len(coords))
        self.assertTrue(any(nm[row,col].item()!=0xffffffff and nm[row,col].item()//3!=row//3 for row in range(11) for col in range(27)))

    def test_dense_weights_declared_tolerance(self):
        self.assertIsNotNone(candidate,'row-chunked implementation not yet written')
        g=torch.Generator().manual_seed(20261015)
        for dtype,rtol,atol in [(torch.float32,1e-5,1e-6),(torch.float16,1/1024,1/8192)]:
            _,_,f,nm,w,b=fixture(dtype)
            w=(torch.randn(w.shape,generator=g)/32).to(dtype)
            expected=original(f,{'neighbor_map':nm},w,b)
            actual=candidate.row_chunked_explicit(torch,f,nm,w,b,3,lambda *args:None)
            torch.testing.assert_close(actual,expected,rtol=rtol,atol=atol)
            OBSERVED.append({'dtype':str(dtype),'rtol':rtol,'atol':atol,'max_abs':float((actual.float()-expected.float()).abs().max())})

    def test_invalid_layout_and_tile_rejected(self):
        self.assertIsNotNone(candidate,'row-chunked implementation not yet written')
        _,_,f,nm,w,b=fixture(torch.float16)
        for args in [(f,nm.long(),w,b,3),(f,nm,w,b,0),(f,nm,w,b,32769),(f,nm,w.float(),b,3),(f,nm,w,b[:2],3)]:
            with self.assertRaises(admission.AdmissionStop):candidate.row_chunked_explicit(torch,*args,lambda *x:None)

class Memory(unittest.TestCase):
    def test_exact_final_contract_and_original_cap(self):
        self.assertTrue(hasattr(admission,'final_c2s_plan'),'fixed-support admission not yet written')
        p=admission.final_c2s_plan(378891,512,128,1842968,64)
        self.assertEqual(p['visible_bytes'],2766585408)
        for args in [(378891,512,128,1842969,64),(378890,512,128,1842968,64),(378891,256,128,1842968,64),(2**63,512,128,2**63,64)]:
            with self.assertRaises(admission.AdmissionStop):admission.final_c2s_plan(*args)
        with self.assertRaises(admission.AdmissionStop):admission.c2s_plan(378891,512,128,1842968,64)
        with self.assertRaises(admission.AdmissionStop):admission.conv_plan(1842968,64,64,512**3)
        self.assertEqual(admission.MAX_ROWS,1000000)

    def test_chunk_and_artifact_formula(self):
        self.assertTrue(hasattr(admission,'chunked_conv_plan'),'chunked admission not yet written')
        p=admission.chunked_conv_plan(1842968,32768,90243072)
        self.assertEqual(p['visible_bytes'],3069154208)
        self.assertEqual(p['chunks'],57);self.assertEqual(p['last_chunk_rows'],7960)
        self.assertEqual(p['im2col_bytes'],113246208)
        self.assertEqual(admission.output_payload_bytes(),117949952)
        for n,t,v in [(1842969,32768,90243072),(1842968,32769,90243072),(1842968,0,90243072),(2**63,32768,2**63)]:
            with self.assertRaises(admission.AdmissionStop):admission.chunked_conv_plan(n,t,v)

    def test_live_memory_rejection_precedes_allocator_callback(self):
        self.assertTrue(hasattr(admission,'chunked_conv_plan'),'chunked admission not yet written')
        events=[]
        hooks=admission.AdmissionHooks(None,torch,emit=events.append,memory=lambda:dict(allocated_bytes=12*admission.GIB,reserved_bytes=12*admission.GIB,free_bytes=13*admission.GIB,total_bytes=16*admission.GIB))
        with self.assertRaises(admission.AdmissionStop):hooks._admit('final-convolution',admission.chunked_conv_plan(1842968,32768,90243072))
        self.assertFalse(events[-1]['admitted'])
        self.assertEqual(events[-1]['memory']['allocated_bytes'],12*admission.GIB)

    def test_live_admission_at_historical_baseline_and_free_reserve_refusal(self):
        events=[]
        memory=dict(allocated_bytes=1594233856,reserved_bytes=6129975296,free_bytes=8342953984,total_bytes=16974905344)
        h=admission.AdmissionHooks(None,torch,emit=events.append,memory=lambda:memory)
        h._admit('fourth-first',admission.final_c2s_plan(378891,512,128,1842968,64))
        h._admit('fourth-second',admission.final_c2s_plan(378891,512,128,1842968,64))
        h._admit('final-conv',admission.chunked_conv_plan(1842968,32768,90243072))
        self.assertTrue(all(e['admitted'] for e in events))
        memory['free_bytes']=2*admission.GIB
        with self.assertRaises(admission.AdmissionStop):h._admit('last',admission.chunked_conv_plan(1842968,1,90243072))
        self.assertFalse(events[-1]['admitted'])

    def test_final_plan_rejects_non_integer_metadata(self):
        with self.assertRaises(admission.AdmissionStop):admission.final_c2s_plan(378891,512,128,1842968.0,64)
        with self.assertRaises(admission.AdmissionStop):admission.chunked_conv_plan(1842968.0,32768,90243072)

class PatchLifecycle(unittest.TestCase):
    def make(self):
        self.assertIsNotNone(candidate)
        # Native owner boundary simulated only; exact original Python source and
        # genuine nn.Module hook machinery. No FlexGEMM/native import.
        owner=type('SubMConv3dFunction',(object,),{'_sparse_submanifold_conv_forward':staticmethod(original)})
        module=torch.nn.Identity()
        hooks=types.SimpleNamespace(active_conv=None,convolution_completed=39,up_states={3:{'completed':2}})
        patch=candidate.FinalConvolutionPatch(owner,module,hooks,torch,lambda *a,**k:None)
        return owner,module,hooks,patch

    def test_original_unmodified_until_final_scope_and_restored_after_failure(self):
        owner,module,hooks,patch=self.make();descriptor=owner.__dict__[patch.name]
        patch.install()
        self.assertIs(owner.__dict__[patch.name],descriptor)
        with self.assertRaises(admission.AdmissionStop):module(torch.zeros(1,4))
        x=types.SimpleNamespace(feats=torch.zeros(11,4))
        hooks.active_conv=(candidate.SITE,module,x)
        patch._enter(module,(x,))
        self.assertIsNot(owner.__dict__[patch.name],descriptor)
        with self.assertRaises(admission.AdmissionStop):patch.forward(x.feats.clone(),{},torch.zeros(3,3,3,3,4))
        patch.remove()
        self.assertIs(owner.__dict__[patch.name],descriptor)
        self.assertFalse(module._forward_pre_hooks);self.assertFalse(module._forward_hooks)
        self.assertFalse(patch.active)

    def test_foreign_descriptor_and_repeated_lifecycle_rejected(self):
        owner,module,hooks,patch=self.make();descriptor=owner.__dict__[patch.name]
        hooks.active_conv=(candidate.SITE,module,object())
        owner._sparse_submanifold_conv_forward=staticmethod(lambda *a:None)
        with self.assertRaises(admission.AdmissionStop):patch._enter(module,(hooks.active_conv[2],))
        owner._sparse_submanifold_conv_forward=descriptor
        patch.calls=1
        with self.assertRaises(admission.AdmissionStop):patch._enter(module,(hooks.active_conv[2],))

    def test_admission_failure_stops_tile_loop(self):
        _,_,f,nm,w,b=fixture(torch.float16);entries=[]
        def observe(phase,start,end):
            if phase=='entry':
                entries.append(start)
                if start==3:raise admission.AdmissionStop('live reserve')
        with self.assertRaises(admission.AdmissionStop):candidate.row_chunked_explicit(torch,f,nm,w,b,3,observe)
        self.assertEqual(entries,[0,3])

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    evidence={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'torch':torch.__version__,'GPU_entries':GPU_ENTRIES,'GPU_initialized':torch.cuda.is_initialized(),'CPU_equivalence':OBSERVED,'native_execution':False,'weights_loaded':False}
    print(json.dumps(evidence))
    raise SystemExit(0 if result.wasSuccessful() and not GPU_ENTRIES and not torch.cuda.is_initialized() else 1)
