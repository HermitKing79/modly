"""Host-only real supervisor replay with synthetic process/observer boundaries."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import amd_sparse_sampling as entry
import amd_sparse_sampling_contract as policy
import amd_sparse_flow_forward as forward
import test_amd_sparse_flow_control as controls

def good_record():
    row=controls.ControlTests().good_record();row.update(forward_calls=22)
    row['calls']={k:22 for k in row['calls']};row['transfers']=dict(x=1,t=22,cond=1,neg_cond=1,output=1,frequency=22)
    per={k:1 for k in row['calls']};times=[1,55/56,25/26,15/16,10/11,7/8,5/6,25/32,5/7,5/8,1/2,5/16,0]
    models=[]
    for i in range(12):
        for role in (['positive','negative'] if i<10 else ['positive']):
            models.append(dict(index=len(models)+1,step=i+1,role=role,t=times[i],next_t=times[i+1],modules=per,frequency=1))
    row['sampling']=dict(sequences=1,started=22,completed=22,positive_calls=12,negative_calls=10,models=models,
        steps=[dict(step=i+1,forwards=2 if i<10 else 1,synchronized=True,all_finite=True) for i in range(12)])
    row['output']['sha256']='a'*64
    row['latent_checkpoint']=dict(verified=True,output_sha256='a'*64,file_bytes=140000,tensor_bytes=131072,shape=[1,8,16,16,16],dtype='F32')
    return row

class SamplingSupervisionTests(unittest.TestCase):
    def test_acceptance_rejects_counts_frequency_cleanup_missing_steps_and_artifact(self):
        good=good_record();self.assertEqual(policy.classify(good,True,True,True),'PASS')
        for key,value in [('forward_calls',23),('calls',{}),('latent_checkpoint',{}),('workspace_clear_calls',0)]:
            self.assertEqual(policy.classify({**good,key:value},True,True,True),'PARTIAL')
        bad=json.loads(json.dumps(good));bad['sampling']['models'][1]['frequency']=0
        self.assertEqual(policy.classify(bad,True,True,True),'PARTIAL')
        bad=json.loads(json.dumps(good));bad['sampling']['models'][1]['role']='positive'
        self.assertEqual(policy.classify(bad,True,True,True),'PARTIAL')
        for key,value in [('file_bytes',262145),('dtype','BF16')]:
            bad=json.loads(json.dumps(good));bad['latent_checkpoint'][key]=value
            self.assertEqual(policy.classify(bad,True,True,True),'PARTIAL')
        for flags in [(False,True,True),(True,False,True),(True,True,False)]:
            self.assertEqual(policy.classify(good,*flags),'PARTIAL')
    def test_ram_and_sampler_dependency_fail_before_admission(self):
        with patch('psutil.virtual_memory',return_value=SimpleNamespace(available=policy.REQUIRED_RAM-1)):
            with self.assertRaises(RuntimeError):entry.admission()
        with patch.object(entry,'sources',return_value={}):
            with self.assertRaises(RuntimeError):entry.verify_sampler()
    def test_actual_supervisor_publishes_exact_sampling_marker_and_limits(self):
        written=[]
        class Input(io.BytesIO):
            def write(self,value):written.append(value);return super().write(value)
        expected={'config':policy.CONFIG,'checkpoint_sha256':'verified','tensor_data_bytes':2584359952,
            'tensors':{k:{**v,'dtype':'BF16'} for k,v in policy.previous.schema().items()}}
        gate=dict(status='PASS',gpu_initialized=False,strict=True,assign=True,missing_keys=[],unexpected_keys=[],
            parameter_bytes=2584359952,payload_entries=640,reversible_widening_count=220,feature_verified=True,
            tensors={k:{**v,'device':'cpu','is_meta':False} for k,v in policy.previous.schema().items()},
            buffers={'rope_phases':{'shape':[4096,64],'dtype':'C64','device':'cpu','is_meta':False}})
        data=(forward.supervisor.EVENT+json.dumps(dict(event='cpu-ready',gate_one=gate,checkpoint_sha256='verified'))+'\n'+
              forward.supervisor.RESULT+json.dumps(good_record())+'\n').encode()
        child=SimpleNamespace(pid=123,stdin=Input(),stdout=io.BytesIO(data),stderr=io.BytesIO(),returncode=0,wait=lambda timeout:None)
        class Monitor:
            error=None;samples=2
            def __init__(self,*a,**k):pass
            def start(self,timeout):pass
            def require_healthy(self):pass
            def close(self):return True
        class Job:
            def attach(self,child):pass
            def finish(self):return dict(owned_tree_exit_verified=True,forced_job_cleanup=False)
        with tempfile.TemporaryDirectory(dir=forward.local.ROOT.parent) as tmp:
            p=entry.profile();p.evidence=Path(tmp);p.identity=lambda:expected;p.source_identity=lambda:{}
            with patch.object(forward.supervisor,'verify_committed_files',return_value={}), \
                 patch.object(forward.supervisor.subprocess,'run',return_value=SimpleNamespace(stdout=b'published')), \
                 patch.object(forward.supervisor.subprocess,'Popen',return_value=child), \
                 patch.object(forward.supervisor,'windows_observation',return_value={}), \
                 patch.object(forward.supervisor,'healthy_observation',return_value=True), \
                 patch.object(forward.supervisor,'OwnedJob',Job), \
                 patch.object(forward.supervisor.diagnostics,'ActivityMonitor',Monitor), \
                 patch.object(forward.supervisor,'WindowsMonitor',Monitor), \
                 patch.object(forward.supervisor.previous,'reap_owned_child',side_effect=lambda c,p:p.update(worker_exit_verified=True)):
                self.assertEqual(forward.supervisor.supervise(p.evidence/'run','published',profile=p),0)
            marker=json.loads((p.evidence/'attempt-started.json').read_text())
            self.assertEqual(marker['maximum_forwards'],22);self.assertEqual(marker['maximum_steps'],12);self.assertEqual(marker['maximum_sequences'],1)
            result=json.loads((p.evidence/'run/results.json').read_text())
            self.assertEqual(result['process']['stage_timeouts_seconds'],dict(cpu=180,forward_release=240,overall=480))
        self.assertEqual(written,[b'OWNED_JOB_ATTACHED\n',entry.AUTHORIZE])

if __name__=='__main__':unittest.main()
