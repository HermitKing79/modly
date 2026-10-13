"""CPU-only contracts for the one installed convolution, never import FlexGEMM."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import amd_native_convolution_contract as c

class ContractTests(unittest.TestCase):
    def test_fixture_and_full_identity(self):
        f,co,w,b=c.fixture()
        c.validate_fixture(f,co,w,b)
        self.assertEqual(len(np.unique(co,axis=0)),2049)
        self.assertEqual(len(np.unique(f[:,0])),2049)
        self.assertTrue(np.array_equal(f.astype('float32').astype('float16'),f))
        self.assertEqual(np.count_nonzero(w),1024)
        self.assertEqual(c.compare(f.copy(),f)['mismatched_elements'],0)
        bad=f.copy();bad[-1,-1]+=np.float16(1/1024)
        with self.assertRaises(ValueError):c.compare(bad,f)
    def test_foreign_fixture(self):
        f,co,w,b=c.fixture();co[1]=co[0]
        with self.assertRaises(ValueError):c.validate_fixture(f,co,w,b)
    def test_tuning_miss_and_config(self):
        self.assertEqual(c.outer_key(2049,1024,1024,27),'(2^11, 1024, 1024, 27)')
        c.require_tuning({c.OUTER_KEY:{'SPLITK':2}},{c.INNER_KEY:c.INNER_CONFIG})
        with self.assertRaises(ValueError):c.require_tuning({}, {c.INNER_KEY:c.INNER_CONFIG})
        with self.assertRaises(ValueError):c.require_tuning({c.OUTER_KEY:{'SPLITK':1}},{c.INNER_KEY:c.INNER_CONFIG})
    def test_once_and_phase(self):
        p=c.Permit()
        with self.assertRaises(RuntimeError):p.take('jit')
        p.phase='convolution';p.take('jit')
        with self.assertRaises(RuntimeError):p.take('jit')
        with self.assertRaises(RuntimeError):p.take('unlisted')
    def test_memory_admission(self):
        c.admit(2*1024**3,13*1024**3)
        with self.assertRaises(ValueError):c.admit(1024**3-1,13*1024**3)
        with self.assertRaises(ValueError):c.admit(2*1024**3,13*1024**3-1)
        c.allocation([2,2049,1024],4)
        with self.assertRaises(ValueError):c.allocation([2049,27,1024],4)
        with self.assertRaises(ValueError):c.allocation([-1],4)
    def test_cached_helper_rejects_before_loader(self):
        calls=[]
        load=c.helper_loader('source',['inc'],Path('helper.pyd'),'hash',lambda p:'hash',lambda p,n:calls.append((p,n)))
        with self.assertRaises(ValueError):load(src='foreign',name='hip_utils',include_dirs=['inc'])
        self.assertEqual(calls,[])
        load(src='source',name='hip_utils',include_dirs=['inc'])
        self.assertEqual(len(calls),1)
        with self.assertRaises(RuntimeError):load(src='source',name='hip_utils',include_dirs=['inc'])
    def test_deadlines_preserve_existing(self):
        import amd_sparse_decoder_forward as s
        self.assertEqual(s.profile_deadlines('native-convolution'),(45,120,180))
        self.assertEqual(s.profile_deadlines('shape-sampling'),(180,240,480))
        self.assertEqual(s.profile_deadlines(None),(120,180,360))
    def test_weight_view_is_not_allocation(self):
        self.assertEqual(c.result_allocation('aten.view.default',(1024,27,1024),2,True),0)
        with self.assertRaises(ValueError):c.result_allocation('aten.view.default',(1024,27,1024),2,False)
        with self.assertRaises(ValueError):c.result_allocation('aten.empty.memory_format',(1024,27,1024),2,False)
    def test_actual_trace_callbacks_and_candidate_call_sites(self):
        import ast
        from amd_forward_diagnostics import Trace,validate_record
        lines=[];t=Trace(lines.append);calls=[]
        self.assertEqual(t.call('native:neighbor',lambda:(calls.append('neighbor'),17)[1]),17)
        t.emit('state','gpu',gpu=dict(name='AMD Radeon RX 9070 XT',architecture='gfx1201',total_bytes=17095983104,index=0))
        self.assertEqual(calls,['neighbor'])
        import json
        for line in lines:validate_record(json.loads(line.split('=',1)[1]))
        for name in ('amd_native_convolution_policy.py','amd_native_convolution_smoke.py'):
            tree=ast.parse((Path(__file__).parent/name).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='call':
                    self.assertEqual(len(node.args),2,(name,node.lineno));self.assertFalse(node.keywords,(name,node.lineno))
    def test_cpu_gate_and_pass_require_every_safety_check(self):
        import amd_native_convolution_smoke as s
        gate=dict(status='PASS',gpu_initialized=False,weights_loaded=False,fixture_verified=True)
        s.authorize(gate,dict(native_contract_sha256='hash'),'hash')
        with self.assertRaises(RuntimeError):s.authorize(gate,dict(native_contract_sha256='hash'),'foreign')
        self.assertEqual(s.classify(dict(status='PASS'),True,True,True),'UNKNOWN')

if __name__=='__main__':unittest.main()
