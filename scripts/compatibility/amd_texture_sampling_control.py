"""Twelve genuine texture calls; reuse proven shape invocation/transfer guards."""
import time,weakref
import amd_texture_sampling_contract as policy
from amd_shape_sampling_control import ShapeSamplingControl
from amd_shape_slat_forward import CopyPolicy

class TextureCopyPolicy(CopyPolicy):
    def __init__(self,torch,n):
        super().__init__(torch,n);self.contract=policy;self.counts.update(shape_cond=0,seqlen=0)

class TextureSamplingControl(ShapeSamplingControl):
    def __init__(self,torch,sampler_class,*,device='cuda:0',token_count=policy.TOKEN_COUNT,maximum_tokens=policy.TOKEN_COUNT):
        if type(maximum_tokens) is not int or not 0<maximum_tokens<=4430:raise ValueError('Texture input bound exceeds qualified4430 maximum')
        super().__init__(torch,sampler_class,device=device,token_count=token_count,maximum_tokens=maximum_tokens)
        self.policy=policy;self.budget=policy.Budget();self.shape_cond=None;self.concat_calls=0;self.concat_checked=False
    def copy_policy(self,n):return TextureCopyPolicy(self.torch,n)
    def begin_model(self,args,kwargs,copy):
        if set(kwargs)!={'concat_cond'} or kwargs['concat_cond'] is not self.shape_cond:raise RuntimeError('Foreign/missing shape concatenation condition')
        self.sparse_check(self.shape_cond);self.concat_checked=False
        super().begin_model(args,{},copy)
    def validate_concat(self,x):
        if self.budget.active is None or self.concat_checked:raise RuntimeError('Foreign/duplicate texture concatenation')
        from trellis2.modules.sparse import SparseTensor
        if not isinstance(x,SparseTensor) or list(x.shape)!=[1,64] or x.coords is not self.coords or x.layout!=[slice(0,self.token_count)]:raise RuntimeError('Texture64 sparse layout differs')
        self.tensor_check(x.feats,[self.token_count,64])
        if not self.torch.equal(x.feats[:,:32],self.step_x.feats) or not self.torch.equal(x.feats[:,32:],self.shape_cond.feats):raise RuntimeError('Texture concat must be exact noise/state BEFORE shape condition')
        self.concat_checked=True;self.concat_calls+=1
    def end_model(self,result,copy):
        if not self.concat_checked:raise RuntimeError('Genuine64-channel projection not observed')
        super().end_model(result,copy)
    def check_memory(self,row):
        if row['allocated_bytes']>policy.MEMORY_LIMIT or row['reserved_bytes']>policy.MEMORY_LIMIT or row['available_bytes']<policy.FREE_RESERVE+policy.OPAQUE_HEADROOM:raise RuntimeError('Texture memory/reserve/workspace admission failed')
    def block_memory(self,row):self.check_memory(row);super().block_memory(row)
    def run(self,model,x,inputs,trace,synchronize,memory):
        if self.sequences:raise RuntimeError('Texture sampling sequence already consumed')
        self.sequences=1;self.current_x=x;self.coords=x.coords;self.positive=inputs['cond'];self.negative=None;self.shape_cond=inputs['shape_cond']
        self.sparse_check(x);self.sparse_check(self.shape_cond);self.tensor_check(self.positive,[1,1029,1024])
        seqlen=x.get_spatial_cache('seqlen')
        if (not isinstance(seqlen,self.torch.Tensor) or list(seqlen.shape)!=[1] or seqlen.dtype!=self.torch.int64 or str(seqlen.device)!=self.device or seqlen.item()!=self.token_count):raise RuntimeError('Texture sparse sequence cache differs')
        for value in (x.feats,self.shape_cond.feats,self.positive):
            if not self.torch.isfinite(value).all().item():raise RuntimeError('Nonfinite texture input')
        self.check_memory(memory());sampler=self.sampler_class(sigma_min=policy.SIGMA_MIN);original=sampler.sample_once
        expected={'neg_cond':None,'concat_cond':self.shape_cond,'guidance_strength':1.0,'guidance_interval':(.6,.9),'guidance_rescale':0.0}
        def step(model_arg,x,t,t_prev,cond=None,**kwargs):
            if model_arg is not model or x is not self.current_x or cond is not self.positive or set(kwargs)!=set(expected) or any(kwargs[k] is not expected[k] if k in ('concat_cond','neg_cond') else kwargs[k]!=expected[k] for k in expected):raise RuntimeError('Foreign canonical texture sampler arguments')
            self.budget.begin_step(t,t_prev);self.step_x=x;index=self.budget.step;before=self.budget.calls;began=time.monotonic()
            trace.emit('enter','texture-step-'+str(index),step={'index':index,'t':t,'next_t':t_prev})
            self.check_memory(memory());out=original(model_arg,x,t,t_prev,cond,**kwargs)
            self.sparse_check(out.pred_x_prev);self.sparse_check(out.pred_x_0)
            finite=bool(self.torch.isfinite(out.pred_x_prev.feats).all().item() and self.torch.isfinite(out.pred_x_0.feats).all().item())
            if not finite:raise RuntimeError('Nonfinite genuine texture Euler output')
            synchronize();mem=memory();self.check_memory(mem);self.budget.end_step();self.current_x=out.pred_x_prev;self.step_x=None
            self.issued.extend([weakref.ref(out.pred_x_prev),weakref.ref(out.pred_x_0)])
            row=dict(step=index,t=t,next_t=t_prev,forwards=self.budget.calls-before,total_forwards=self.budget.calls,all_finite=finite,coords_unchanged=True,synchronized=True,elapsed_seconds=time.monotonic()-began,memory=mem)
            self.steps.append(row);trace.emit('state','texture-step-complete',step=row);return out
        sampler.sample_once=step
        result=sampler.sample(model,x,cond=self.positive,neg_cond=None,concat_cond=self.shape_cond,**policy.PARAMS,verbose=False,tqdm_desc='Bounded texture-SLat512')
        self.budget.require_complete()
        if result.samples is not self.current_x or len(result.pred_x_t)!=12 or len(result.pred_x_0)!=12 or self.concat_calls!=12:raise RuntimeError('Texture history/concat count differs')
        return result
    def report(self):return {**super().report(),'concat_calls':self.concat_calls}
    def drop_references(self):self.shape_cond=None;super().drop_references()
