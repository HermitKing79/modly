"""Fixed12-step Euler budget. No framework import, model loading or device query."""
import math
from fractions import Fraction
from amd_sparse_flow_contract import (CPU_SECONDS,GPU_SECONDS,OVERALL_SECONDS,MEMORY_LIMIT,
    CONFIG,SingleForward,validate_manifest,validate_header,authorize_placement,validate_output)
import amd_sparse_flow_contract as previous

SEED=20261013
SIGMA_MIN=1e-5
REQUIRED_RAM=11411181632
PARAMS=dict(steps=12,rescale_t=5.0,guidance_strength=7.5,guidance_rescale=.7,
            guidance_interval=(.6,1.0),dino_lock=0.0)
SAMPLER_MODULES={'trellis2.pipelines.samplers','trellis2.pipelines.samplers.base',
    'trellis2.pipelines.samplers.flow_euler','trellis2.pipelines.samplers.classifier_free_guidance_mixin',
    'trellis2.pipelines.samplers.guidance_interval_mixin'}

def times():
    return [float(5*Fraction(12-i,12)/(1+4*Fraction(12-i,12))) for i in range(13)]

class Budget:
    def __init__(self):
        self.calls=0;self.started=0;self.steps=0;self.step=None;self.active=None
        self.tickets=[]
        for i,t in enumerate(times()[:-1]):
            for role in (('positive','negative') if i<10 else ('positive',)):
                self.tickets.append(dict(index=len(self.tickets)+1,step=i+1,role=role,t=t,next_t=times()[i+1]))
    def begin_step(self,index,t,next_t):
        if (type(index) is not int or index!=self.steps or self.step is not None or not 0<=index<12
            or type(t) not in (int,float) or type(next_t) not in (int,float)
            or not math.isfinite(t) or not math.isfinite(next_t)
            or abs(t-times()[index])>1e-14 or abs(next_t-times()[index+1])>1e-14):
            raise RuntimeError('Unapproved Euler step/time/order')
        self.step=index+1
    def next_ticket(self):
        if (self.step is None or self.active is not None or self.calls>=22
            or self.tickets[self.calls]['step']!=self.step):raise RuntimeError('No legitimate next model invocation')
        return self.tickets[self.calls]
    def begin_forward(self,ticket):
        if ticket is not self.next_ticket():raise RuntimeError('Foreign or altered sampling ticket')
        self.active=ticket;self.started+=1
    def end_forward(self,ticket,frequency):
        if ticket is not self.active or frequency!=1:raise RuntimeError('Incomplete model/frequency contract')
        self.calls+=1;self.active=None
    def end_step(self):
        if self.step is None:raise RuntimeError('No admitted step to complete')
        target=2*self.step if self.step<=10 else 20+self.step-10
        if self.step is None or self.active is not None or self.calls!=target:
            raise RuntimeError('Incomplete Euler step invocation count')
        self.steps+=1;self.step=None
    def require_complete(self):
        if self.steps!=12 or self.calls!=22 or self.started!=22 or self.active is not None or self.step is not None:
            raise RuntimeError('Sampling sequence did not complete exactly12steps/22forwards')

def prohibited_import(name):
    return previous.prohibited_import(name) and name not in SAMPLER_MODULES

def validate_transfer(key,description,target,phase):
    if key=='neg_cond':key='cond'
    previous.validate_transfer(key,description,target,phase)

def classify(record,process_ok,healthy,preserved):
    try:
        validate_output(record.get('output',{}))
        from amd_decoder_contract import validate_release
        validate_release(record.get('release',{}))
        expected={'<root>':22,**{f'blocks.{i}':22 for i in range(30)},
                  **{f'blocks.{i}.{a}':22 for i in range(30) for a in ('self_attn','cross_attn')}}
        if record.get('calls')!=expected:raise ValueError('Sampling module counts differ')
        models=record.get('sampling',{}).get('models',[])
        tickets=Budget().tickets
        if len(models)!=22:raise ValueError('Missing per-forward records')
        per_model={'<root>':1,**{f'blocks.{i}':1 for i in range(30)},
                   **{f'blocks.{i}.{a}':1 for i in range(30) for a in ('self_attn','cross_attn')}}
        for row,ticket in zip(models,tickets):
            if row!={**ticket,'modules':per_model,'frequency':1}:raise ValueError('Per-forward contract record differs')
        rows=record.get('sampling',{}).get('steps',[])
        if len(rows)!=12:raise ValueError('Missing durable steps')
        for i,row in enumerate(rows):
            if (row.get('step')!=i+1 or row.get('forwards')!=(2 if i<10 else 1)
                or row.get('synchronized') is not True or row.get('all_finite') is not True):
                raise ValueError('Invalid durable sampling step')
        artifact=record['latent_checkpoint']
        if (artifact.get('verified') is not True or artifact['output_sha256']!=record['output']['sha256']
            or type(artifact.get('file_bytes')) is not int or not 131072<artifact['file_bytes']<=262144
            or artifact.get('tensor_bytes')!=131072 or artifact.get('shape')!=[1,8,16,16,16]
            or artifact.get('dtype')!='F32'):
            raise ValueError('Latent artifact not verified')
    except (KeyError,ValueError,TypeError):return 'PARTIAL'
    sampling=record.get('sampling',{})
    return 'PASS' if (record.get('status')=='PASS' and record.get('stage')=='complete'
        and record.get('load_calls')==1 and record.get('forward_calls')==22
        and sampling.get('sequences')==1 and sampling.get('started')==22
        and sampling.get('positive_calls')==12 and sampling.get('negative_calls')==10
        and record.get('workspace_clear_calls')==1
        and record.get('transfers')==dict(x=1,t=22,cond=1,neg_cond=1,output=1,frequency=22)
        and record['release'].get('owned_tensors_collected') is True
        and record.get('workspace_binding',{}).get('verified') is True
        and process_ok is True and healthy is True and preserved is True) else 'PARTIAL'
