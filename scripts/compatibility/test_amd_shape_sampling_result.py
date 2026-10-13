"""CPU-only classifier regressions: missing/contradictory evidence fails closed."""
import copy,unittest
import amd_shape_sampling_contract as c
def candidate():
    modules={'<root>':1,**{f'blocks.{i}':1 for i in range(30)},**{f'blocks.{i}.{a}':1 for i in range(30) for a in ('self_attn','cross_attn')}}
    steps=[];total=0
    for i,t in enumerate(c.times()[:-1]):
        count=2 if i<9 else 1;total+=count
        steps.append(dict(step=i+1,t=t,next_t=c.times()[i+1],forwards=count,total_forwards=total,all_finite=True,synchronized=True,coords_unchanged=True,elapsed_seconds=1.,memory=dict(allocated_bytes=1,reserved_bytes=1,peak_allocated_bytes=1,peak_reserved_bytes=1)))
    sha='a'*64;artifacts={}
    for name,dtype,shape,size in [('shape-latent-raw.npy','float32',[3743,32],479232),('shape-latent-denormalized.npy','float32',[3743,32],479232),('shape-latent-coordinates.npy','int32',[3743,4],60016),('shape-initial-noise.npy','float32',[3743,32],479232)]:
        artifacts[name]=dict(verified=True,dtype=dtype,shape=shape,file_bytes=size,sha256=sha,payload_sha256=sha)
    artifacts['shape-latent-provenance.json']=dict(verified=True,file_bytes=100,sha256=sha)
    return dict(status='PASS',stage='complete',load_calls=1,forward_calls=21,workspace_clear_calls=1,workspace_binding={'verified':True},forward_seconds=20.,
        cpu_load={'elapsed_seconds':10.},output=dict(shape=[3743,32],dtype='torch.float32',device='cuda:0',all_finite=True,count=3743*32,requires_grad=False,is_inference_tensor=True,grad_enabled=False,coords_unchanged=True,sparse_shape=[1,32],sha256=sha),
        input={k:{'sha256':sha} for k in ('x','coords')},release=dict(allocated_bytes=0,reserved_bytes=0,model_collected=True,owned_tensors_collected=True),calls={k:21 for k in modules},
        sampling=dict(sequences=1,started=21,completed=21,positive_calls=12,negative_calls=9,steps=steps,models=[{**t,'modules':modules,'frequency':1,'cache_reused':t['index']>1} for t in c.tickets()]),
        transfers={**{k:1 for k in ('x','coords','batch_map','seqlen','cond','neg_cond','output','output_coords','rope_frequency')},'t':21,'frequency':21},artifacts=artifacts)
class ResultTests(unittest.TestCase):
    def test_complete_synthetic_evidence_and_reject_changed_execution(self):
        self.assertEqual(c.classify(candidate(),True,True,True),'PASS')
        for key,value in [('t',.5),('next_t',float('nan')),('total_forwards',22),('elapsed_seconds',-1.),('all_finite',1),('elapsed_seconds',float('inf'))]:
            r=candidate();r['sampling']['steps'][0][key]=value;self.assertEqual(c.classify(r,True,True,True),'PARTIAL')
        r=candidate();r['sampling']['steps'][0]['memory']['reserved_bytes']=c.MEMORY_LIMIT+1;self.assertEqual(c.classify(r,True,True,True),'PARTIAL')
    def test_required_noise_provenance_and_integrity(self):
        for name in ('shape-initial-noise.npy','shape-latent-provenance.json'):
            r=candidate();r['artifacts'].pop(name);self.assertEqual(c.classify(r,True,True,True),'PARTIAL')
        for key,value in [('sha256','invalid'),('payload_sha256','b'*64),('file_bytes',5000000)]:
            r=candidate();r['artifacts']['shape-initial-noise.npy'][key]=value;self.assertEqual(c.classify(r,True,True,True),'PARTIAL')
        for flags in ((False,True,True),(True,False,True),(True,True,False)):
            self.assertEqual(c.classify(candidate(),*flags),'PARTIAL')
if __name__=='__main__':unittest.main()
