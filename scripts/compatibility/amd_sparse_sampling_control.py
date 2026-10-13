"""Source-faithful sampler instrumentation; per-invocation identities and compact diagnostics."""
import inspect
import math
import time
import weakref
import amd_sparse_sampling_contract as policy

MODULES={'<root>':1,**{f'blocks.{i}':1 for i in range(30)},
         **{f'blocks.{i}.{a}':1 for i in range(30) for a in ('self_attn','cross_attn')}}

class SamplingControl:
    def __init__(self,torch,sampler_class,*,device='cuda:0'):
        self.torch=torch;self.sampler_class=sampler_class;self.device=device;self.policy=policy
        self.budget=policy.Budget();self.steps=[];self.models=[];self.issued=[]
        self.active_counts=None;self.pending_t=None;self.factory_ticket=None
        self.current_x=self.step_x=self.positive=self.negative=None
        self.sequences=0;self.t_copies=0;self.frequency_copies=0;self.max_memory={}
    def tensor_check(self,value,shape):
        t=self.torch
        if (not isinstance(value,t.Tensor) or list(value.shape)!=shape or value.dtype!=t.float32
            or str(value.device)!=self.device or value.requires_grad):raise RuntimeError('Sampling tensor identity/shape/dtype/device mismatch')
    def next_timestep(self,args,kwargs):
        ticket=self.budget.next_ticket()
        if (self.pending_t is not None or self.factory_ticket is not None or len(args)!=1
            or type(args[0]) is not list or len(args[0])!=1
            or type(args[0][0]) not in (int,float) or not math.isfinite(args[0][0])
            or abs(args[0][0]-1000*ticket['t'])>1e-10
            or kwargs!={'device':self.torch.device(self.device),'dtype':self.torch.float32}):
            raise RuntimeError('Unapproved/duplicate sampler timestep factory')
        self.factory_ticket=ticket
        return ticket
    def staged_timestep(self,cpu,placed):
        import numpy as np
        ticket=self.factory_ticket
        if (ticket is None or cpu.device.type!='cpu' or list(cpu.shape)!=[1] or cpu.dtype!=self.torch.float32
            or cpu.item()!=float(np.float32(1000*ticket['t']))):raise RuntimeError('CPU timestep staging values differ')
        self.tensor_check(placed,[1]);self.pending_t=placed;self.t_copies+=1;self.issued.append(weakref.ref(placed))
    def begin_model(self,args,kwargs,transfers,frequency_sources):
        import numpy as np
        ticket=self.budget.next_ticket()
        if (kwargs or len(args)!=3 or self.factory_ticket is not ticket or self.pending_t is None
            or args[0] is not self.step_x or args[1] is not self.pending_t
            or args[2] is not (self.positive if ticket['role']=='positive' else self.negative)
            or transfers['t']!=1 or args[1].item()!=float(np.float32(1000*ticket['t']))):
            raise RuntimeError('Foreign sampling latent/timestep/conditioning or wrong invocation order')
        self.tensor_check(args[0],[1,8,16,16,16]);self.tensor_check(args[1],[1]);self.tensor_check(args[2],[1,1029,1024])
        self.budget.begin_forward(ticket);self.active_counts={};frequency_sources.clear();transfers['frequency']=0
    def module_enter(self,key):
        if self.active_counts is None or key not in MODULES or self.active_counts.get(key,0):
            raise RuntimeError('Unapproved or duplicate model/block/attention invocation')
        self.active_counts[key]=1
    def end_model(self,result,transfers):
        self.tensor_check(result,[1,8,16,16,16])
        if self.active_counts!=MODULES or not self.torch.isfinite(result).all().item():
            raise RuntimeError('Incomplete/nonfinite model result')
        ticket=self.budget.active;self.budget.end_forward(ticket,transfers['frequency'])
        self.frequency_copies+=transfers['frequency'];self.issued.append(weakref.ref(result))
        self.models.append(dict(ticket,modules=self.active_counts.copy(),frequency=transfers['frequency']))
        self.active_counts=None;self.pending_t=None;self.factory_ticket=None
    def block_memory(self,row):
        for key in ('allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes'):
            self.max_memory[key]=max(self.max_memory.get(key,0),row[key])
    def run(self,model,inputs,trace,synchronize,memory):
        if self.sequences:raise RuntimeError('Sampling sequence already consumed')
        self.sequences=1;self.current_x=inputs['x'];self.positive=inputs['cond'];self.negative=inputs['neg_cond']
        sampler=self.sampler_class(sigma_min=policy.SIGMA_MIN);original=sampler.sample_once
        def step(model_arg,x,t,t_prev,cond=None,**kwargs):
            if (model_arg is not model or x is not self.current_x or cond is not self.positive
                or kwargs!={'neg_cond':self.negative,'guidance_strength':7.5,
                    'guidance_interval':(.6,1.0),'guidance_rescale':.7}):
                raise RuntimeError('Foreign sampler step model/input/parameters')
            self.budget.begin_step(self.budget.steps,t,t_prev);self.step_x=x
            index=self.budget.step;before=self.budget.calls;began=time.monotonic()
            trace.emit('enter','euler-step-'+str(index),step={'index':index,'t':t,'next_t':t_prev})
            out=original(model_arg,x,t,t_prev,cond,**kwargs)
            self.tensor_check(out.pred_x_prev,[1,8,16,16,16]);self.tensor_check(out.pred_x_0,[1,8,16,16,16])
            finite=bool(self.torch.isfinite(out.pred_x_prev).all().item() and self.torch.isfinite(out.pred_x_0).all().item())
            if not finite:raise RuntimeError('Nonfinite Euler step')
            synchronize();mem=memory()
            if mem['allocated_bytes']>policy.MEMORY_LIMIT or mem['reserved_bytes']>policy.MEMORY_LIMIT:
                raise RuntimeError('Sampling step memory cap exceeded')
            self.budget.end_step();self.current_x=out.pred_x_prev;self.step_x=None
            self.issued.extend((weakref.ref(out.pred_x_prev),weakref.ref(out.pred_x_0)))
            row=dict(step=index,t=t,next_t=t_prev,forwards=self.budget.calls-before,total_forwards=self.budget.calls,
                all_finite=finite,synchronized=True,elapsed_seconds=time.monotonic()-began,memory=mem)
            self.steps.append(row);trace.emit('state','euler-step-complete',step=row)
            return out
        sampler.sample_once=step
        result=sampler.sample(model,self.current_x,cond=self.positive,neg_cond=self.negative,
            **policy.PARAMS,verbose=False,tqdm_desc='Bounded sparse structure')
        self.budget.require_complete()
        if result.samples is not self.current_x or len(result.pred_x_t)!=12 or len(result.pred_x_0)!=12:
            raise RuntimeError('Sampler result/history does not match exactly12steps')
        return result
    def report(self):
        return dict(sequences=self.sequences,started=self.budget.started,completed=self.budget.calls,
            positive_calls=sum(r['role']=='positive' for r in self.models),negative_calls=sum(r['role']=='negative' for r in self.models),
            steps=self.steps,models=self.models,max_block_memory=self.max_memory)
    def transfers(self,current):
        return {**current,'t':self.t_copies,'frequency':self.frequency_copies}
    def drop_references(self):
        self.current_x=self.step_x=self.positive=self.negative=self.pending_t=self.factory_ticket=None
        self.active_counts=None
