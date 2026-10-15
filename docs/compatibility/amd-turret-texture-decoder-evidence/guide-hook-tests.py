"""CPU-only tests of production guided C2S admission hooks with real guides.

The tensor facade exposes only CPU NumPy values. _sparse retains every shape,
dtype and contiguity check while allowing CPU devices. Convolution execution is
outside this test: each stage starts at its legitimate next-conv2 boundary.
The production mask/coordinate/order/cache, admission and paired-call checks run
unchanged. No Torch, model, native library, GPU or WSL is imported or executed.
"""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[2]/'scripts/compatibility')]
import admission
import amd_texture_decoder_inputs as preserved


class Array:
    def __init__(self,value,device='cpu'):
        self.value=value;self.device=device;self.shape=value.shape;self.dtype=value.dtype
    def is_contiguous(self): return self.value.flags.c_contiguous
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value
    def sum(self): return self.value.sum()


class Features:
    def __init__(self,rows,width,dtype=np.dtype('<f2'),device='cpu'):
        self.shape=(rows,width);self.dtype=dtype;self.device=device
    def is_contiguous(self): return True


class Sparse:
    def __init__(self,coords,width,*,features=None,cache=None,grid=32):
        self.coords=coords if isinstance(coords,Array) else Array(coords)
        self.feats=features or Features(len(self.coords.value),width)
        self.shape=(1,width);self.spatial_shape=(grid,grid,grid);self.cache=cache
    def get_spatial_cache(self,name): return self.cache if name=='channel2spatial_2' else None


class CPU:
    int32=np.dtype('<i4');float16=np.dtype('<f2');bool=np.dtype('bool')
    @staticmethod
    def equal(left,right): return np.array_equal(left.value,right.value)


class Handle:
    def remove(self): pass


class Module:
    def register_forward_pre_hook(self,_): return Handle()
    def register_forward_hook(self,_): return Handle()
    def forward(self,*args): raise AssertionError('No original model/native allocation is permitted in this CPU test')


class SparseConv3d(Module): pass
class SparseChannel2Spatial(Module): factor=2


def original_block_forward(*args):
    raise AssertionError('No model block is executed in this hook contract test')


def decoder():
    blocks=[];modules=[]
    for stage,count in enumerate(admission.COUNTS):
        rows=[]
        for index in range(count):
            conv=SparseConv3d();rows.append(SimpleNamespace(conv=conv,mlp=[SimpleNamespace(out_features=4*admission.WIDTHS[stage])]))
            modules.append((f'blocks.{stage}.{index}.conv',conv))
        if stage<4:
            up=SimpleNamespace(channels=admission.WIDTHS[stage],out_channels=admission.WIDTHS[stage+1],
                pred_subdiv=False,conv1=SparseConv3d(),conv2=SparseConv3d(),updown=SparseChannel2Spatial(),
                _forward=original_block_forward)
            rows.append(up)
            for suffix in ('conv1','conv2','updown'):
                modules.append((f'blocks.{stage}.{count}.{suffix}',getattr(up,suffix)))
        blocks.append(rows)
    return SimpleNamespace(num_blocks=admission.COUNTS,model_channels=admission.WIDTHS,
        blocks=blocks,pred_subdiv=False,out_channels=6,named_modules=lambda:iter(modules))


class CPUHooks(admission.AdmissionHooks):
    def _sparse(self,x,width=None,final=False):
        n,c=x.feats.shape
        admission.require((0<n<=admission.MAX_ROWS or (final and n==admission.FINAL_ROWS and c in (16,64)))
                          and (width is None or c==width),'feature rows/width mismatch')
        admission.require(tuple(x.coords.shape)==(n,4) and x.coords.dtype==self.torch.int32,
                          'coordinate rows/shape/dtype mismatch')
        admission.require(x.feats.dtype==self.torch.float16 and x.feats.is_contiguous()
                          and x.coords.is_contiguous(),'features/coords must be contiguous F16/int32')
        admission.require(x.feats.device==x.coords.device=='cpu','CPU-only test tensor required')
        admission.require(tuple(x.shape)==(1,c) and len(x.spatial_shape)==3 and all(v>0 for v in x.spatial_shape),
                          'unexpected sparse batch/spatial shape')
        return n,c


class GuideHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.inputs=preserved.read_inputs()

    def new(self):
        self.events=[]
        memory=lambda:dict(free_bytes=14*admission.GIB,total_bytes=16*admission.GIB,
                            allocated_bytes=admission.GIB,reserved_bytes=admission.GIB)
        hooks=CPUHooks(decoder(),CPU,emit=self.events.append,memory=memory,guidance=self.inputs).install()
        self.addCleanup(hooks.remove)
        return hooks

    def boundary(self,hooks,stage):
        hooks.convolution_completed=next(i for i,site in enumerate(hooks.sequence) if site[3:]==('conv2',stage))
        block=hooks.decoder.blocks[stage][-1]
        site=(f'blocks.{stage}.{admission.COUNTS[stage]}.updown',block.updown,block,stage)
        guide=self.inputs['guides'][stage]
        sub=Sparse(guide['coords'],8,features=Array(guide['logits_f32']>0),grid=32*2**stage)
        return site,sub

    def expand(self,hooks,site,sub,call):
        stage=site[3];block=site[2]
        width=block.out_channels*8 if call==0 else block.channels
        x=Sparse(sub.coords,width,grid=32*2**stage)
        hooks._up_pre(site,site[1],(x,sub))
        coords=self.inputs['guides'][stage+1]['coords'] if stage<3 else self.inputs['geometry_coords']
        out=Sparse(coords,width//8,grid=64*2**stage)
        hooks._up_post(site,site[1],(x,sub),out)
        return x,out

    def assert_rejected_without_admission(self,hooks,site,x,sub):
        before=sum(event['event']=='admission' for event in self.events)
        with self.assertRaises((ValueError,admission.AdmissionStop)): hooks._up_pre(site,site[1],(x,sub))
        self.assertEqual(sum(event['event']=='admission' for event in self.events),before)

    def test_all_eight_real_guided_expansions_with_production_hook_state(self):
        hooks=self.new()
        for stage in range(4):
            site,sub=self.boundary(hooks,stage)
            for call in range(2): self.expand(hooks,site,sub,call)
        self.assertEqual([hooks.up_states[i]['completed'] for i in range(4)],[2,2,2,2])
        self.assertEqual(len(hooks.subdivision_records),8)
        self.assertEqual([e['children'] for e in hooks.subdivision_records],
                         [n for n in preserved.CHILD_ROWS for _ in range(2)])
        self.assertTrue(all(e['admitted'] for e in self.events if e['event']=='admission'))
        self.assertEqual(hooks.guide_order.finalize(self.inputs['geometry_coords'])['calls'],[2,2,2,2])
        hooks.convolution_completed=40;hooks.assert_complete()

    def test_permuted_parent_rows_and_changed_mask_are_rejected_before_admission(self):
        for mutation in ('parent-order','mask','mask-dtype','cache'):
            with self.subTest(mutation=mutation):
                hooks=self.new();site,sub=self.boundary(hooks,0)
                if mutation=='parent-order': sub.coords=Array(sub.coords.value[::-1].copy())
                if mutation=='mask':
                    changed=sub.feats.value.copy();changed[0,0]=~changed[0,0];sub.feats=Array(changed)
                if mutation=='mask-dtype': sub.feats=Array(sub.feats.value.astype(np.uint8))
                x=Sparse(sub.coords,site[2].out_channels*8,cache=object() if mutation=='cache' else None)
                self.assert_rejected_without_admission(hooks,site,x,sub)

    def test_repeated_unpaired_and_third_calls_and_new_second_mask_object_rejected(self):
        hooks=self.new();site,sub=self.boundary(hooks,0)
        x=Sparse(sub.coords,site[2].out_channels*8)
        hooks._up_pre(site,site[1],(x,sub))
        self.assert_rejected_without_admission(hooks,site,x,sub)
        hooks=self.new();site,sub=self.boundary(hooks,0)
        self.expand(hooks,site,sub,0)
        cloned=Sparse(sub.coords,8,features=Array(sub.feats.value.copy()))
        self.assert_rejected_without_admission(hooks,site,Sparse(sub.coords,site[2].channels),cloned)
        hooks=self.new();site,sub=self.boundary(hooks,0)
        for call in range(2): self.expand(hooks,site,sub,call)
        self.assert_rejected_without_admission(hooks,site,Sparse(sub.coords,site[2].channels),sub)

    def test_permuted_child_rows_rejected_before_marking_operation_complete(self):
        hooks=self.new();site,sub=self.boundary(hooks,0)
        x=Sparse(sub.coords,site[2].out_channels*8)
        hooks._up_pre(site,site[1],(x,sub))
        out=Sparse(self.inputs['guides'][1]['coords'][::-1].copy(),site[2].out_channels,grid=64)
        with self.assertRaises(ValueError): hooks._up_post(site,site[1],(x,sub),out)
        self.assertEqual(hooks.up_states[0]['completed'],0)
        self.assertEqual(hooks.subdivision_records,[])

    def test_memory_rejection_prevents_original_expansion(self):
        hooks=self.new();site,sub=self.boundary(hooks,0)
        hooks.memory=lambda:dict(free_bytes=0,total_bytes=16*admission.GIB,allocated_bytes=0,reserved_bytes=0)
        with self.assertRaises(admission.AdmissionStop): hooks._up_pre(site,site[1],(Sparse(sub.coords,4096),sub))
        self.assertEqual(hooks.up_states[0]['completed'],0)
        self.assertFalse(next(e for e in self.events if e['event']=='admission')['admitted'])

    def test_exact_f16_and_pred_subdiv_false_are_required(self):
        for location in ('decoder','block'):
            invalid=decoder()
            target=invalid if location=='decoder' else invalid.blocks[0][-1]
            target.pred_subdiv=True
            hooks=CPUHooks(invalid,CPU,guidance=self.inputs)
            reason='decoder architecture' if location=='decoder' else 'C2S width/mask/factor'
            with self.assertRaisesRegex(admission.AdmissionStop,reason): hooks.install()
        # Exercise the original _sparse implementation itself: no CPU override.
        hooks=self.new();x=Sparse(self.inputs['coords'],1024)
        device=SimpleNamespace(type='cuda',index=0)
        x.feats.device=x.coords.device=device
        self.assertEqual(admission.AdmissionHooks._sparse(hooks,x,1024),(3743,1024))
        for dtype in (np.dtype('<f4'),'bfloat16'):
            x.feats.dtype=dtype
            with self.assertRaisesRegex(admission.AdmissionStop,'contiguous F16/int32'):
                admission.AdmissionHooks._sparse(hooks,x,1024)

    def test_original_convolution_weight_dtype_check_rejects_bf16_f32_and_wrong_bias(self):
        for wrong in (None,'bfloat16',np.dtype('<f4'),'bias-f32'):
            hooks=self.new();site=hooks.sequence[0];module=site[1]
            module.weight=SimpleNamespace(shape=(1024,3,3,3,1024),dtype=np.dtype('<f2'),device='cpu')
            module.bias=None;module.kernel_size=(3,3,3);module.dilation=(1,1,1);module.stride=(1,1,1)
            if wrong=='bias-f32': module.bias=SimpleNamespace(dtype=np.dtype('<f4'))
            elif wrong is not None: module.weight.dtype=wrong
            if wrong is None:
                hooks._conv_pre(site,module,(Sparse(self.inputs['coords'],1024),))
                self.assertEqual(hooks.active_conv[0],site[0])
                self.assertTrue(next(e for e in self.events if e['event']=='admission')['admitted'])
            else:
                with self.assertRaisesRegex(admission.AdmissionStop,'weights must be F16'):
                    hooks._conv_pre(site,module,(Sparse(self.inputs['coords'],1024),))
                self.assertFalse(any(e['event']=='admission' for e in self.events))


if __name__=='__main__': unittest.main(verbosity=2)
