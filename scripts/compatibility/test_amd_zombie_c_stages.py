"""Synthetic adapter tests; original neural entrypoints are never executed."""
import contextlib,importlib.util,io,subprocess,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
SCRIPT=Path(__file__).with_name('amd_zombie_c_stages.py');subject=None
if SCRIPT.exists():
    spec=importlib.util.spec_from_file_location('c_stages',SCRIPT);subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)

class Exists(unittest.TestCase):
    def test_bridge_exists(self):self.assertIsNotNone(subject,'Scoped new-image bridge missing')

@unittest.skipIf(subject is None,'not implemented')
class BridgePolicy(unittest.TestCase):
    def test_default_bridge_cannot_run(self):
        with patch.object(subject,'worker',side_effect=AssertionError('worker')):
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(subject.main([]),0)

    def test_actual_windows_isolated_worker_script_default_is_safe(self):
        result=subprocess.run([sys.executable,'-B','-I','-S',str(SCRIPT)],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('DRY_RUN',result.stdout)

    def test_actual_isolated_worker_rejects_missing_parent_permit(self):
        import os
        env={k:v for k,v in os.environ.items() if not k.startswith('MODLY_ZOMBIE_C_')}
        result=subprocess.run([sys.executable,'-B','-I','-S',str(SCRIPT),'--worker','--output','foreign'],
                              env=env,capture_output=True,text=True,timeout=15)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('exact parent pipeline permit',result.stderr)

    def test_private_authority_does_not_remove_code_checks(self):
        calls=[];gates=[]
        b=subject.Bridge('a'*40);b._authority=dict(source_path='new/source.png',input_contract_path='new/input-contract.json',source_sha256='1',input_contract_sha256='2')
        b._verify_original=lambda commit,names,**kw:calls.append((commit,names,kw)) or {n:'sha' for n in names}
        b._check_private=lambda:None;b._gate=lambda label:gates.append(label)
        b._pending_gate='dino-marker'
        result=b.verify_files('a'*40,['old-code.py','new/source.png','new/input-contract.json'],timeout_seconds=5)
        self.assertIn('old-code.py',calls[0][1]);self.assertNotIn('new/source.png',calls[0][1]);self.assertIn(subject.h.REFERENCE,calls[0][1])
        self.assertEqual(result['new/source.png'],'1');self.assertEqual(gates,['dino-marker']);self.assertIsNone(b._pending_gate)

    def test_unexpected_commit_and_old_stage_destination_rejected(self):
        b=subject.Bridge('a'*40)
        with self.assertRaises(ValueError):b.verify_files('b'*40,[])
        with self.assertRaises(ValueError):b.supervise(subject.h.ROOT/'old/run','a'*40)

    def test_source_verification_failure_prevents_marker_gate(self):
        b=subject.Bridge('a'*40);b._check_private=lambda:(_ for _ in ()).throw(ValueError('modified'))
        b._pending_gate='stage';calls=[];b._gate=calls.append
        with self.assertRaises(ValueError):b.verify_files('a'*40,['code.py'])
        self.assertEqual(calls,[])

    def test_stage_reservation_requires_verifier_gate(self):
        b=subject.Bridge('a'*40);b._phase='occupancy'
        b._supervise_original=lambda *a,**k:0
        with self.assertRaises(RuntimeError):b.supervise(subject.h.CASE/'occupancy/dino/run','a'*40)

    def test_healthy_observer_keeps_original_check_and12_floor(self):
        b=subject.Bridge('a'*40);b.read=lambda:dict(available_bytes=12*1024**3)
        b._healthy_original=lambda capture:capture['ok']
        self.assertTrue(b.healthy(dict(ok=True)));self.assertFalse(b.healthy(dict(ok=False)))
        b.read=lambda:dict(available_bytes=12*1024**3-1)
        with self.assertRaises(RuntimeError):b.healthy(dict(ok=True))

    def test_decode_reservation_uses_original_host_guard_and22_gate(self):
        b=subject.Bridge('a'*40);calls=[];b._gate=lambda label:calls.append(label)
        runtime=NS(evidence=subject.h.CASE/'decode',host_admission=lambda:calls.append('host') or {})
        b.decode_reservation(runtime,lambda output:calls.append(output),runtime.evidence/'run')
        self.assertEqual(calls,['host','geometry-decode-marker',runtime.evidence/'run'])
        with self.assertRaises(ValueError):b.decode_reservation(runtime,lambda _:None,subject.h.ROOT/'foreign')

    def test_worker_requires_exact_pipeline_permit(self):
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaises(ValueError):subject.worker(Path('foreign'))

if __name__=='__main__':unittest.main()
