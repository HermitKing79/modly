"""CPU fixtures exercise the real pre-hook and production durable journal bridge.

Only _sparse's GPU-device boundary and memory query are simulated. No Torch,
weights, native modules or GPU operations are imported or executed.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest

E=Path(__file__).resolve().parent;ROOT=E.parents[2]
sys.path[:0]=[str(E),str(ROOT/'scripts/compatibility')]
from admission import AdmissionHooks,AdmissionStop
import amd_wsl_rocdxg_smoke as base

class CpuMask:
    def __init__(self,rows,coords):
        self.rows=rows;self.shape=(len(rows),8);self.dtype=bool;self.device='cpu';self.sum_calls=0
    def sum(self):
        self.sum_calls+=1
        return NS(item=lambda:sum(sum(row) for row in self.rows))

class CpuC2S:
    def forward(self,*args):raise AssertionError('Original expansion must never execute in this CPU fixture')

class CpuUpBlock:
    out_channels=64;channels=128
    def _forward(self,*args):raise AssertionError('Model forward prohibited in CPU fixture')

def setup(rows,output):
    n=len(rows);coords=NS(dtype='int32',device='cpu',shape=(n,4))
    mask=CpuMask(rows,coords)
    x=NS(coords=coords,feats=NS(shape=(n,512),device='cpu'),spatial_shape=(256,256,256),get_spatial_cache=lambda key:None)
    sub=NS(feats=mask,coords=coords)
    # Execute the exact production emit/event_hook definitions, including fsync.
    tree=ast.parse((E/'neural.py').read_text());run=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='run_neural')
    functions=[v for v in run.body if isinstance(v,ast.FunctionDef) and v.name in ('emit','event_hook')]
    ns=dict(time=time,gpu_started=time.monotonic(),record={},base=base,output=output)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(E/'neural.py'),'exec'),ns)
    hooks=AdmissionHooks(None,NS(bool=bool,int32='int32',equal=lambda a,b:a is b),emit=ns['event_hook'],
        memory=lambda:dict(allocated_bytes=1,reserved_bytes=2,free_bytes=14*1024**3,total_bytes=16*1024**3))
    block=CpuUpBlock();module=CpuC2S();site=('blocks.3.4.updown',module,block,3)
    hooks._sparse=lambda sparse,width:(n,width) # CPU fixture: only the GPU/device seam.
    hooks.convolution_completed=39;hooks.sequence=[None]*39+[('final',None,block,'conv2',3)]
    hooks.up_states={3:dict(completed=0)}
    plans=[];hooks._admit=lambda name,plan:plans.append(plan)
    return hooks,site,(x,sub),mask,plans

class Tests(unittest.TestCase):
    def rejected(self,rows,count):
        with tempfile.TemporaryDirectory(dir=E) as directory:
            output=Path(directory);hooks,site,args,mask,plans=setup(rows,output)
            with self.assertRaises(AdmissionStop):hooks._up_pre(site,site[1],args)
            record=json.loads((output/'phases.jsonl').read_text().strip())
            self.assertEqual(record['phase'],'c2s-count-before-admission')
            self.assertEqual((record['site'],record['call'],record['parents'],record['children']),('blocks.3.4.updown',1,len(rows),count))
            self.assertEqual(mask.sum_calls,1);self.assertEqual(plans,[])
            self.assertTrue(record['row_limits_would_reject'])
            self.assertEqual(record['mask_shape'],[len(rows),8]);self.assertEqual(record['mask_dtype'],str(mask.dtype))
            self.assertEqual(record['source']['sha256'],hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            self.assertEqual(record['feature_channels'],512);self.assertEqual(record['skip_channels'],128);self.assertEqual(record['out_channels'],64)

    def test_zero_persists_before_original_rejection(self):self.rejected([(False,)*8]*2,0)
    def test_over_cap_persists_before_original_rejection(self):self.rejected([(True,)*8]*125000+[(True,)+(False,)*7],1000001)

    def test_within_cap_keeps_original_plan_and_mask(self):
        with tempfile.TemporaryDirectory(dir=E) as directory:
            output=Path(directory);hooks,site,args,mask,plans=setup([(True,)*8]*2,output)
            hooks._up_pre(site,site[1],args)
            record=json.loads((output/'phases.jsonl').read_text().strip())
            self.assertFalse(record['row_limits_would_reject']);self.assertEqual(record['children_per_parent_mean'],8)
            self.assertEqual(plans[0]['children'],16);self.assertIs(hooks.up_states[3]['mask'],mask)
            self.assertEqual(mask.sum_calls,1)

if __name__=='__main__':unittest.main()
