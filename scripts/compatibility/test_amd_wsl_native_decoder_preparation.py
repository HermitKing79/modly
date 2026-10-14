"""Focused host-only admission/authorization/result contracts; no WSL/Torch."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parent))
import amd_wsl_native_decoder_preparation as harness

class PreparationContracts(unittest.TestCase):
    def good(self):
        z={'allocated':0,'reserved':0}
        parameters=json.loads((harness.ROOT/'docs/compatibility/amd-shape-decoder-neural-evidence/cpu-probe-4/results.json').read_text())['worker']['parameters']
        widen=sorted(k for k,v in parameters.items() if v['dtype']=='F32')
        return dict(decoder_cpu='PASS',CPU_model_collected=True,decoder_class='trellis2.models.sc_vaes.sparse_unet_vae.SparseUnetVaeDecoder',
                    decoder_diagnostics=dict(weights_loaded=True,strict=True,entries=292,dtype_counts={'F16':216,'F32':76},roundtrip_verified=True,widened_keys=widen,payload_sha256={k:'0'*64 for k in parameters},model_forward_calls=0),policy_final={'rejected_calls':0,'native_calls':0},memory_before=z,memory_release=z,memory_peak=z,no_neural_forward=True,
                    native_calls=0,gate_authorized=True,native_import='PASS',bindings_count=18,
                    guards_rejected_calls=0,driver_helper_inactive=True)

    def test_zero_allocation_and_no_computation_contract(self):
        harness.validate_qualified(self.good())
        for key,value in [('native_calls',1),('no_neural_forward',False),('bindings_count',17),
                          ('guards_rejected_calls',1),('driver_helper_inactive',False),
                          ('gate_authorized',False),('native_import','ENTERED')]:
            r=self.good();r[key]=value
            with self.assertRaises(RuntimeError): harness.validate_qualified(r)
        for key,value in [('decoder_cpu','NOT_ATTEMPTED'),('CPU_model_collected',False),('decoder_diagnostics',{}),('cleanup_error','synthetic cleanup failed'),('error','synthetic failure'),('policy_final',{'rejected_calls':1,'native_calls':0})]:
            r=self.good();r[key]=value
            with self.assertRaises(RuntimeError): harness.validate_qualified(r)
        for key in ('memory_before','memory_release','memory_peak'):
            r=self.good();r[key]={'allocated':1,'reserved':0}
            with self.assertRaises(RuntimeError): harness.validate_qualified(r)

    def test_production_command_preserves_named_distro_binary_gate_and_timeout(self):
        cmd=harness.worker_command('/mnt/c/work/script.py','/mnt/c/work/evidence')
        self.assertEqual(cmd[cmd.index('-d')+1], 'Modly-AMD-Ubuntu-24.04')
        self.assertIn('240s',cmd);self.assertIn('--kill-after=5s',cmd)
        self.assertIn('/opt/modly-wsl/venv/bin/python',cmd);self.assertIn('-I',cmd)
        self.assertNotIn('docker-desktop',cmd)
        with self.assertRaises(ValueError): harness.worker_command('/wrong/script','/mnt/c/evidence')

    def test_marker_is_exclusive_and_identifies_actual_candidate(self):
        with tempfile.TemporaryDirectory(dir=harness.ROOT) as tmp,patch.object(harness,'EV',Path(tmp)):
            out=Path(tmp)/'run';harness.reserve_attempt(out)
            value=json.loads((Path(tmp)/'native-attempt.marker').read_text())
            self.assertEqual(value['script_sha256'],harness.digest(harness.SCRIPT))
            self.assertEqual(value['neural_forwards_authorized'],0)
            with self.assertRaises(FileExistsError): harness.reserve_attempt(out)

    def test_complete_actual_supervisor_lifecycle_with_host_only_fixtures(self):
        import io
        from concurrent.futures import Future
        from types import SimpleNamespace
        health={'returncode':0,'timed_out':False,'observation':{
            'observer':{'system_log_probe':{'RecordId':1}},
            'controllers':[{'Name':'AMD Radeon RX 9070 XT','Status':'OK'}], 'events':[]}}
        class Monitor:
            def __init__(self,*args):self.observations=[{'capture':health}]
            def poll(self,now):pass
            def healthy_since(self,after):return True
            def latest(self):return health
        class Job:
            def attach(self,child):self.attached=True
            def active(self):return 0
            def finish(self):return {'owned_tree_exit_verified':True,'forced_job_cleanup':False}
        class Pipe(io.BytesIO):
            def close(self):self.sent=self.getvalue();self.closed_for_test=True
            @property
            def closed(self):return getattr(self,'closed_for_test',False)
        pipe=Pipe()
        class Child:
            pid=123;stdin=pipe;returncode=0
            def poll(self):return 0 if pipe.closed else None
        def instant(fn):
            result=Future();result.set_result(fn());return result
        with tempfile.TemporaryDirectory(dir=harness.ROOT) as tmp:
            out=Path(tmp)/'run'
            def popen(*args,**kwargs):
                (out/'guest').mkdir()
                record=self.good();record['classification']='QUALIFIED'
                (out/'guest/results.json').write_text(json.dumps(record))
                return Child()
            actual_health=harness.base.load_sibling('amd_bounded_gpu_proof').healthy_observation
            def sibling(name):
                return SimpleNamespace(OwnedJob=Job) if name=='amd_forward_owned_job' else SimpleNamespace(windows_observation=lambda since:health,healthy_observation=actual_health)
            with patch.object(harness,'EV',Path(tmp)),patch.object(harness,'host_admission',return_value={'available_ram_bytes':16*harness.GiB,'D_free_bytes':70*harness.GiB}),patch.object(harness.base,'HealthMonitor',Monitor),patch.object(harness.base,'load_sibling',side_effect=sibling),patch.object(harness.subprocess,'Popen',side_effect=popen),patch.object(harness.base,'in_thread',side_effect=instant),patch.object(harness.base,'terminate_owned_distro',return_value={'returncode':0}):
                # base guest_path enforces the genuine checkout's path.
                self.assertEqual(harness.supervise(out),0,(out/'results.json').read_text())
            result=json.loads((out/'results.json').read_text())
            self.assertEqual(pipe.sent,b'GO\n')
            self.assertTrue(result['observer_healthy'])
            self.assertFalse(result['windows_job_cleanup']['forced_job_cleanup'])

    def test_deadlines_are_bounded(self):
        harness.check_deadlines(100,0,10)
        with self.assertRaises(TimeoutError): harness.check_deadlines(270,0,None)
        with self.assertRaises(TimeoutError): harness.check_deadlines(230,0,10)

    def test_authorization_remains_actual_binary_production_gate(self):
        import io
        harness.base.await_go(io.BytesIO(b'GO\n'))
        for token in (b'GO\r\n',b'GO\nX',b'',b'GO'):
            with self.assertRaises(RuntimeError): harness.base.await_go(io.BytesIO(token))

if __name__=='__main__':unittest.main()
