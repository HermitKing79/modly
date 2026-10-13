"""Source-faithful sparse sampler controls; one21-call sequence, no model implementation."""
import math,time,weakref
import amd_shape_sampling_contract as policy
from amd_shape_slat_forward import CopyPolicy
MODULES={'<root>':1,**{f'blocks.{i}':1 for i in range(30)},**{f'blocks.{i}.{a}':1 for i in range(30) for a in ('self_attn','cross_attn')}}
PHASE_KEY='rope_phase_3d_freq1.0-10000.0_hd128'
class SamplingCopyPolicy(CopyPolicy):
    def __init__(self,torch,n):
        super().__init__(torch,n);self.contract=policy;self.counts.update(neg_cond=0,seqlen=0)
class ShapeSamplingControl:
    def __init__(self,torch,sampler_class,*,device='cuda:0'):
        self.torch=torch;self.sampler_class=sampler_class;self.device=device;self.policy=policy
        self.budget=policy.Budget();self.steps=[];self.models=[];self.issued=[];self.sequences=0
        self.current_x=self.step_x=self.positive=self.negative=self.coords=self.cache_phase=None
        self.pending_t=self.factory_ticket=self.active_counts=None;self.t_copies=0;self.frequency_copies=0;self.max_memory={}
    def copy_policy(self,n):return SamplingCopyPolicy(self.torch,n)
    def tensor_check(self,t,shape):
        if (not isinstance(t,self.torch.Tensor) or list(t.shape)!=shape or t.dtype!=self.torch.float32
            or str(t.device)!=self.device or t.requires_grad):raise RuntimeError('Foreign shape sampling tensor')
    def sparse_check(self,x):
        from trellis2.modules.sparse import SparseTensor
        if not isinstance(x,SparseTensor) or list(x.shape)!=[1,32] or x.coords is not self.coords or x.layout!=[slice(0,policy.TOKEN_COUNT)]:raise RuntimeError('Sparse layout/coordinate identity changed')
        self.tensor_check(x.feats,[policy.TOKEN_COUNT,32])
    def next_timestep(self,args,kwargs):
        ticket=self.budget.next_ticket()
        if (self.pending_t is not None or self.factory_ticket is not None or len(args)!=1 or type(args[0]) is not list
            or len(args[0])!=1 or type(args[0][0]) not in (int,float) or not math.isfinite(args[0][0])
            or abs(args[0][0]-1000*ticket['t'])>1e-10 or kwargs!={'device':self.torch.device(self.device),'dtype':self.torch.float32}):raise RuntimeError('Foreign/duplicate timestep factory')
        self.factory_ticket=ticket;return ticket
    def staged_timestep(self,cpu,placed):
        import numpy as np
        if self.factory_ticket is None or cpu.device.type!='cpu' or list(cpu.shape)!=[1] or cpu.dtype!=self.torch.float32 or cpu.item()!=float(np.float32(1000*self.factory_ticket['t'])):raise RuntimeError('Timestep staging differs')
        self.tensor_check(placed,[1]);self.pending_t=placed;self.t_copies+=1;self.issued.append(weakref.ref(placed))
    def begin_model(self,args,kwargs,copy):
        import numpy as np
        ticket=self.budget.next_ticket()
        if (kwargs or len(args)!=3 or self.factory_ticket is not ticket or self.pending_t is None or args[0] is not self.step_x
            or args[1] is not self.pending_t or args[2] is not (self.positive if ticket['role']=='positive' else self.negative)
            or copy.counts['t']!=1 or args[1].item()!=float(np.float32(1000*ticket['t']))):raise RuntimeError('Foreign latent/time/conditioning or invocation order')
        self.sparse_check(args[0]);self.tensor_check(args[1],[1]);self.tensor_check(args[2],[1,1029,1024])
        cached=args[0].get_spatial_cache(PHASE_KEY)
        if ticket['index']==1:
            if cached is not None or copy.counts['rope_frequency']:raise RuntimeError('Unexpected warm RoPE cache')
        elif cached is not self.cache_phase or copy.counts['rope_frequency']!=1:raise RuntimeError('RoPE cache reuse/transfer count differs')
        self.budget.begin_forward(ticket);self.active_counts={};copy.frequency=None;copy.counts['frequency']=0
    def module_enter(self,key):
        if self.active_counts is None or key not in MODULES or self.active_counts.get(key,0):raise RuntimeError('Foreign/duplicate module invocation')
        self.active_counts[key]=1
    def end_model(self,result,copy):
        self.sparse_check(result)
        if self.active_counts!=MODULES or copy.counts['frequency']!=1 or copy.counts['rope_frequency']!=1 or not self.torch.isfinite(result.feats).all().item():raise RuntimeError('Incomplete/nonfinite model computation')
        phase=result.get_spatial_cache(PHASE_KEY)
        if (not isinstance(phase,self.torch.Tensor) or list(phase.shape)!=[policy.TOKEN_COUNT,64] or phase.dtype!=self.torch.complex64 or str(phase.device)!=self.device):raise RuntimeError('RoPE cache structure differs')
        if self.cache_phase is None:self.cache_phase=phase;self.issued.append(weakref.ref(phase))
        elif phase is not self.cache_phase:raise RuntimeError('RoPE cache regenerated')
        ticket=self.budget.active;self.budget.end_forward(ticket);self.frequency_copies+=1;self.issued.append(weakref.ref(result))
        self.models.append({**ticket,'modules':self.active_counts.copy(),'frequency':1,'cache_reused':ticket['index']>1})
        self.active_counts=self.pending_t=self.factory_ticket=None
    def block_memory(self,row):
        for key in ('allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes'):self.max_memory[key]=max(self.max_memory.get(key,0),row[key])
    def run(self,model,x,inputs,trace,synchronize,memory):
        if self.sequences:raise RuntimeError('Sampling sequence already consumed')
        self.sequences=1;self.current_x=x;self.coords=x.coords;self.positive=inputs['cond'];self.negative=inputs['neg_cond']
        self.sparse_check(x);self.tensor_check(self.positive,[1,1029,1024]);self.tensor_check(self.negative,[1,1029,1024])
        seqlen=x.get_spatial_cache('seqlen')
        if (not isinstance(seqlen,self.torch.Tensor) or list(seqlen.shape)!=[1] or seqlen.dtype!=self.torch.int64 or str(seqlen.device)!=self.device or seqlen.item()!=policy.TOKEN_COUNT):raise RuntimeError('Exact precached sparse sequence length required')
        if not self.torch.isfinite(x.feats).all().item() or not self.torch.isfinite(self.positive).all().item() or not self.torch.equal(self.negative,self.torch.zeros_like(self.negative)):raise RuntimeError('Invalid initial sampling tensors')
        sampler=self.sampler_class(sigma_min=policy.SIGMA_MIN);original=sampler.sample_once
        def step(model_arg,x,t,t_prev,cond=None,**kwargs):
            if (model_arg is not model or x is not self.current_x or cond is not self.positive or kwargs!={'neg_cond':self.negative,'guidance_strength':7.5,'guidance_interval':(.6,1.0),'guidance_rescale':.5}):raise RuntimeError('Foreign source sampler arguments')
            self.budget.begin_step(t,t_prev);self.step_x=x;index=self.budget.step;before=self.budget.calls;began=time.monotonic()
            trace.emit('enter','shape-step-'+str(index),step={'index':index,'t':t,'next_t':t_prev})
            out=original(model_arg,x,t,t_prev,cond,**kwargs)
            self.sparse_check(out.pred_x_prev);self.sparse_check(out.pred_x_0)
            finite=bool(self.torch.isfinite(out.pred_x_prev.feats).all().item() and self.torch.isfinite(out.pred_x_0.feats).all().item())
            if not finite:raise RuntimeError('Nonfinite CFG/Euler output; no math repair allowed')
            synchronize();mem=memory()
            if mem['allocated_bytes']>policy.MEMORY_LIMIT or mem['reserved_bytes']>policy.MEMORY_LIMIT:raise RuntimeError('Sampling memory limit exceeded')
            self.budget.end_step();self.current_x=out.pred_x_prev;self.step_x=None;self.issued.extend([weakref.ref(out.pred_x_prev),weakref.ref(out.pred_x_0)])
            row=dict(step=index,t=t,next_t=t_prev,forwards=self.budget.calls-before,total_forwards=self.budget.calls,all_finite=finite,coords_unchanged=True,synchronized=True,elapsed_seconds=time.monotonic()-began,memory=mem)
            self.steps.append(row);trace.emit('state','shape-step-complete',step=row);return out
        sampler.sample_once=step
        result=sampler.sample(model,x,cond=self.positive,neg_cond=self.negative,**policy.PARAMS,verbose=False,tqdm_desc='Bounded shape-SLat512')
        self.budget.require_complete()
        if result.samples is not self.current_x or len(result.pred_x_t)!=12 or len(result.pred_x_0)!=12:raise RuntimeError('Source final samples/history differs')
        return result
    def report(self):
        return dict(sequences=self.sequences,started=self.budget.started,completed=self.budget.calls,positive_calls=sum(r['role']=='positive' for r in self.models),negative_calls=sum(r['role']=='negative' for r in self.models),steps=self.steps,models=self.models,max_block_memory=self.max_memory)
    def transfers(self,current):return {**current,'t':self.t_copies,'frequency':self.frequency_copies}
    def drop_references(self):
        self.current_x=self.step_x=self.positive=self.negative=self.coords=self.cache_phase=self.pending_t=self.factory_ticket=self.active_counts=None
