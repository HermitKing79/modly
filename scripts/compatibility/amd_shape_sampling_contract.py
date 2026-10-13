"""Source-derived shape schedule; no GPU or model imports/authorization."""
import math,re
import numpy as np
import amd_shape_slat_contract as previous
CPU_SECONDS,GPU_SECONDS,OVERALL_SECONDS=180,240,480
MEMORY_LIMIT=12*1024**3
REQUIRED_RAM=11411771648
TOKEN_COUNT=3743
SAMPLER_MODULES={'trellis2.pipelines.samplers','trellis2.pipelines.samplers.base','trellis2.pipelines.samplers.flow_euler',
    'trellis2.pipelines.samplers.classifier_free_guidance_mixin','trellis2.pipelines.samplers.guidance_interval_mixin'}
PARAMS=dict(steps=12,rescale_t=3.0,guidance_strength=7.5,guidance_rescale=.5,guidance_interval=(.6,1.0),dino_lock=0.0)
SIGMA_MIN=.00001
SEED=20261013
def times():
    seq=np.linspace(1,0,PARAMS['steps']+1)
    return (PARAMS['rescale_t']*seq/(1+(PARAMS['rescale_t']-1)*seq)).tolist()
def tickets():
    result=[];seq=times();low,high=PARAMS['guidance_interval']
    for i,t in enumerate(seq[:-1]):
        for role in (('positive','negative') if low<=t<=high else ('positive',)):
            result.append(dict(index=len(result)+1,step=i+1,role=role,t=t,next_t=seq[i+1]))
    return result
def require_count(total,positive,negative):
    rows=tickets();actual=(len(rows),sum(r['role']=='positive' for r in rows),sum(r['role']=='negative' for r in rows))
    if (total,positive,negative)!=actual:raise ValueError(f'Requested total/positive/negative {(total,positive,negative)} conflicts with canonical source schedule {actual}')
class Budget:
    def __init__(self):self.rows=tickets();self.calls=0;self.started=0;self.steps=0;self.step=None;self.active=None
    def begin_step(self,t,next_t):
        if type(t) not in (int,float) or type(next_t) not in (int,float) or not math.isfinite(t) or not math.isfinite(next_t):raise RuntimeError('Nonfinite/invalid Euler time')
        if self.step is not None or self.active is not None or self.steps>=12 or abs(t-times()[self.steps])>1e-14 or abs(next_t-times()[self.steps+1])>1e-14:raise RuntimeError('Foreign/extra Euler step')
        self.step=self.steps+1
    def next_ticket(self):
        if self.step is None or self.active is not None or self.calls>=21 or self.rows[self.calls]['step']!=self.step:raise RuntimeError('No legitimate next model ticket')
        return self.rows[self.calls]
    def begin_forward(self,ticket):
        if ticket is not self.next_ticket():raise RuntimeError('Foreign invocation ticket')
        self.active=ticket;self.started+=1
    def end_forward(self,ticket):
        if ticket is not self.active:raise RuntimeError('Incomplete invocation')
        self.calls+=1;self.active=None
    def end_step(self):
        target=sum(r['step']<=self.step for r in self.rows) if self.step is not None else -1
        if self.active is not None or self.calls!=target:raise RuntimeError('Incomplete step count')
        self.steps+=1;self.step=None
    def require_complete(self):
        if self.steps!=12 or self.calls!=21 or self.started!=21 or self.step is not None or self.active is not None:raise RuntimeError('Canonical12/21 sequence incomplete')
def prohibited_import(name):return previous.prohibited_import(name) and name not in SAMPLER_MODULES
def validate_transfer(key,row,target,phase,n):
    if key=='neg_cond':key='cond'
    if key=='seqlen':
        if row!=dict(shape=[1],dtype='torch.int64',device='cpu') or target!='cuda:0' or phase!='inputs':raise ValueError('Sparse sequence-length transfer differs')
    else:previous.validate_transfer(key,row,target,phase,n)
def classify(record,process_ok,healthy,preserved):
    try:
        def seconds(value,limit):
            if type(value) not in (float,int) or not math.isfinite(value) or not 0<=value<=limit:raise ValueError('Invalid/unbounded timing')
        def digest(value):
            if type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None:raise ValueError('Invalid artifact digest')
        seconds(record['forward_seconds'],GPU_SECONDS);seconds(record['cpu_load']['elapsed_seconds'],CPU_SECONDS)
        previous.validate_output(record['output'],TOKEN_COUNT)
        from amd_decoder_contract import validate_release
        validate_release(record['release'])
        per_model={'<root>':1,**{f'blocks.{i}':1 for i in range(30)},**{f'blocks.{i}.{a}':1 for i in range(30) for a in ('self_attn','cross_attn')}}
        if record['calls']!={k:21 for k in per_model}:raise ValueError('Model/module counts differ')
        s=record['sampling']
        if (s['sequences'],s['started'],s['completed'],s['positive_calls'],s['negative_calls'])!=(1,21,21,12,9):raise ValueError('Sequence counts differ')
        if len(s['models'])!=21 or len(s['steps'])!=12:raise ValueError('Per-model/step records missing')
        for r,ticket in zip(s['models'],tickets()):
            if r!={**ticket,'modules':per_model,'frequency':1,'cache_reused':ticket['index']>1}:raise ValueError('Invocation identity/cache contract differs')
        total=0
        for i,row in enumerate(s['steps']):
            forwards=2 if i<9 else 1;total+=forwards
            if (row['step']!=i+1 or row['forwards']!=forwards or row['total_forwards']!=total or row['t']!=times()[i] or row['next_t']!=times()[i+1]
                or any(row[k] is not True for k in ('all_finite','synchronized','coords_unchanged'))):raise ValueError('Step evidence incomplete')
            seconds(row['elapsed_seconds'],GPU_SECONDS)
            for key in ('allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes'):
                if type(row['memory'][key]) is not int or not 0<=row['memory'][key]<=MEMORY_LIMIT:raise ValueError('Step memory exceeds budget')
        transfers={k:1 for k in ('x','coords','batch_map','seqlen','cond','neg_cond','output','output_coords','rope_frequency')};transfers.update(t=21,frequency=21)
        if record['transfers']!=transfers:raise ValueError('Transfer counts differ')
        artifacts=record['artifacts']
        if set(artifacts)!={'shape-latent-raw.npy','shape-latent-denormalized.npy','shape-latent-coordinates.npy','shape-initial-noise.npy','shape-latent-provenance.json'}:raise ValueError('Unexpected/missing artifacts')
        for key,dtype,shape,size in [('shape-latent-raw.npy','float32',[TOKEN_COUNT,32],479232),('shape-latent-denormalized.npy','float32',[TOKEN_COUNT,32],479232),('shape-latent-coordinates.npy','int32',[TOKEN_COUNT,4],60016),('shape-initial-noise.npy','float32',[TOKEN_COUNT,32],479232)]:
            if artifacts[key]['verified'] is not True or artifacts[key]['dtype']!=dtype or artifacts[key]['shape']!=shape or artifacts[key]['file_bytes']!=size:raise ValueError('Saved latent artifact incomplete')
            digest(artifacts[key]['sha256']);digest(artifacts[key]['payload_sha256'])
        for name,expected in [('shape-latent-raw.npy',record['output']['sha256']),('shape-latent-coordinates.npy',record['input']['coords']['sha256']),('shape-initial-noise.npy',record['input']['x']['sha256'])]:
            if artifacts[name]['payload_sha256']!=expected:raise ValueError('Artifact identity changed')
        provenance=artifacts['shape-latent-provenance.json']
        if provenance['verified'] is not True or type(provenance['file_bytes']) is not int or not 0<provenance['file_bytes']<=65536:raise ValueError('Required bounded provenance missing')
        digest(provenance['sha256'])
    except (KeyError,ValueError,TypeError):return 'PARTIAL'
    return 'PASS' if (record['status']=='PASS' and record['stage']=='complete' and record['load_calls']==1 and record['forward_calls']==21
        and record['workspace_clear_calls']==1 and record['release']['owned_tensors_collected'] is True and record['workspace_binding']['verified'] is True
        and process_ok and healthy and preserved) else 'PARTIAL'
