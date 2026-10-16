"""Actual CPU child writer / independent Windows host receipt qualification."""
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path('C:/Users/wattz/Documents/ModlyArtifacts/amd-brick-pbr-cpu-3942181-tests')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'experimental/extensions/trellis2-amd'))


def cpu_only(event, args):
    if event == 'import' and str(args[0]).split('.')[0] in {
            'torch', 'triton', 'trellis2', 'flex_gemm', 'o_voxel', 'cumesh'}:
        raise RuntimeError('Receipt qualification forbids AI/native imports')
    if event.startswith('socket.'):
        raise RuntimeError('Receipt qualification forbids network access')


sys.addaudithook(cpu_only)


class BrickPbrReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from inference_texture import profile
        cls.document = json.loads((ROOT.parent/'artifacts/amd-brick-neural-completion-c9447dd/neural-output/run-contract.json').read_text())
        cls.profile = profile.authenticate(profile.NAME, cls.document)

    def context(self):
        from inference_texture import receipts
        # mkdtemp's Windows0700 ACL uses OWNER RIGHTS rather than the actual
        # user SID; the unchanged original verifier correctly refuses it.
        base = FIXTURES / ('receipt-cpu-'+uuid.uuid4().hex)
        base.mkdir()
        base = base.resolve()
        self.assertEqual(base.parent, FIXTURES.resolve(strict=True))
        def cleanup():
            if base != base.resolve(strict=True) or base.parent != FIXTURES.resolve(strict=True):
                raise ValueError('Fixture cleanup target escaped its exact approved parent')
            shutil.rmtree(base)
        self.addCleanup(cleanup)
        field, control = base/'field', base/'probe'
        field.mkdir(); control.mkdir()
        attestation = receipts.mint_host_attestation(field, control/'ownership.json', profile=self.profile, mode='fixture')
        marker = control/'cpu-only-attempt.marker'
        marker.write_text(json.dumps(dict(schema='modly.amd.brick-pbr-receipt-marker.v1',
            mode='fixture', profile_sha256=self.profile.sha256, artifact_root=field.as_posix())), encoding='utf-8')
        config = dict(root=str(field), receipt_dir=str(control), marker_path=str(marker),
            marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
            attestation_path=attestation['path'], attestation_sha256=attestation['sha256'])
        bridge = receipts.ReceiptBridge(**config, profile=self.profile)
        return config, bridge

    def child(self, config):
        capsule = Path(config['receipt_dir'])/'cpu-writer-config.json'
        capsule.write_text(json.dumps(dict(config=config, geometry_document=self.document)), encoding='utf-8')
        code = r'''
import sys,json,numpy as np
from pathlib import Path
def guard(event,args):
    if event=='import' and str(args[0]).split('.')[0] in {'torch','triton','trellis2','flex_gemm','o_voxel','cumesh'}:
        raise RuntimeError('CPU writer forbids AI/native imports')
    if event.startswith('socket.'): raise RuntimeError('Network forbidden')
sys.addaudithook(guard)
sys.path.insert(0,sys.argv[1])
from inference_texture import receipts,profile
capsule=json.loads(Path(sys.argv[2]).read_text()); config=capsule['config']
p=profile.authenticate(profile.NAME,capsule['geometry_document'])
bridge=receipts.ReceiptBridge(**config,profile=p)
raw=np.array([[-3,-2,-1,0,1,2],[3,4,5,6,7,8]],dtype='<f4')
coords=np.array([[0,511,2,3],[0,0,1,0]],dtype='<i4')
r=bridge.write_output(raw,raw*np.float32(.5)+np.float32(.5),coords,
    dict(mode='fixture',synthetic=True,purpose='SYNTHETIC CPU child writer with actual independent host acceptance'),
    config['receipt_dir'],profile=p)
assert r['verified'] and r['independently_verified_before_release'] and 'torch' not in sys.modules
print(json.dumps(dict(verified=True,independently_verified_before_release=True,
                     artifact_sha256=r['metadata']['artifact_sha256'],mode='SYNTHETIC_CPU_FIXTURE')))
'''
        child = subprocess.Popen([sys.executable, '-B', '-I', '-c', code,
            str(ROOT/'experimental/extensions/trellis2-amd'), str(capsule)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        def cleanup():
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
        self.addCleanup(cleanup)
        return child

    def test_actual_child_writer_waits_for_independent_host_acceptance(self):
        from inference_texture import artifact
        config, bridge = self.context()
        child = self.child(config)
        host = bridge.host_receipt()
        began = time.monotonic()
        while not host.done and child.poll() is None:
            host.poll()
            if time.monotonic()-began > 30:
                self.fail('CPU writer/host receipt did not complete within fixed deadline')
            time.sleep(.025)
        stdout, stderr = child.communicate(timeout=5)
        self.assertEqual(child.returncode, 0, stderr)
        self.assertTrue(host.done)
        verdict = json.loads(stdout)
        self.assertTrue(verdict['independently_verified_before_release'])
        restored = artifact.verify(Path(config['root'])/artifact.ARTIFACT_NAME, profile=self.profile)
        self.assertEqual(restored['metadata']['artifact_sha256'], verdict['artifact_sha256'])
        self.assertEqual(restored['metadata']['payload_bytes'], 128)
        self.assertFalse((Path(config['receipt_dir'])/'neural-attempt.marker').exists())
        self.assertEqual(host.result['profile_sha256'], self.profile.sha256)
        print(json.dumps(dict(classification='PASS_ACTUAL_CPU_CHILD_HOST_ACK_SYNTHETIC_ONLY',
            payload_bytes=128, actual_windows_owner_attestation=True, independent_host_reader=True,
            before_release_ack=True, clean_child_exit=True, post_exit_readback=True,
            unused_GPU_marker_absent=True, torch_imported=False, WSL_executed=False, GPU_executed=False)))

    def test_actual_waiting_writer_refuses_wrong_ack(self):
        from inference_texture import receipts
        config, bridge = self.context()
        child = self.child(config)
        ready = Path(config['receipt_dir'])/'artifact-ready.json'
        began = time.monotonic()
        while not ready.exists() and child.poll() is None:
            if time.monotonic()-began > 20:
                self.fail('CPU fixture writer did not publish ready receipt')
            time.sleep(.025)
        self.assertTrue(ready.exists())
        saved = receipts._decode(ready)
        wrong = dict(bridge._identity(saved['metadata']), status='PASS', independent_of_worker=True,
                     ownership_verified=True, marker_sha256='0'*64)
        receipts._publish(Path(config['receipt_dir'])/'host-artifact-verified.json', wrong)
        stdout, stderr = child.communicate(timeout=5)
        self.assertNotEqual(child.returncode, 0)
        self.assertIn('acknowledgement differs', stderr)
        self.assertEqual(stdout, '')

    def test_foreign_root_profile_marker_and_cancellation_refused_before_write(self):
        import dataclasses
        import numpy as np
        from inference_texture import receipts
        config, bridge = self.context()
        foreign = dict(config, root=str(Path(config['root']).parent/'probe'))
        with self.assertRaises(ValueError):
            receipts.ReceiptBridge(**foreign, profile=self.profile)
        with self.assertRaises(ValueError):
            receipts.ReceiptBridge(**config, profile=dataclasses.replace(self.profile, final_rows=1))
        with self.assertRaises(ValueError):
            receipts.ReceiptBridge(**dict(config, marker_sha256='0'*64), profile=self.profile)
        raw = np.zeros((1, 6), dtype='<f4')
        coords = np.array([[0, 1, 2, 3]], dtype='<i4')
        fixture = dict(mode='fixture', synthetic=True, purpose='SYNTHETIC cancellation refusal')
        bridge.cancelled = lambda: True
        with self.assertRaises(InterruptedError):
            bridge.write_output(raw, raw+np.float32(.5), coords, fixture, config['receipt_dir'], profile=self.profile)
        self.assertEqual(list(Path(config['root']).iterdir()), [])
        self.assertFalse((Path(config['receipt_dir'])/'artifact-ready.json').exists())

    def test_destination_resources_preserve_original_exact_gate_before_write(self):
        import numpy as np
        from inference_texture import receipts
        config, bridge = self.context()
        boundary = 2 * 1024**3
        raw = np.zeros((1, 6), dtype='<f4')
        coords = np.array([[0, 1, 2, 3]], dtype='<i4')
        fixture = dict(mode='fixture', synthetic=True, purpose='SYNTHETIC destination admission refusal')
        for free, writable in ((boundary-1, True), (boundary, False)):
            with self.subTest(free=free, writable=writable), \
                    patch('shutil.disk_usage', return_value=SimpleNamespace(free=free)), \
                    patch('os.access', return_value=writable), \
                    patch.object(receipts.artifact, 'write') as writer:
                with self.assertRaisesRegex(RuntimeError, 'free-space/permission admission refused'):
                    bridge.write_output(raw, raw+np.float32(.5), coords, fixture,
                                        config['receipt_dir'], profile=self.profile)
                writer.assert_not_called()
                self.assertEqual(list(Path(config['root']).iterdir()), [])
                self.assertFalse((Path(config['receipt_dir'])/'artifact-ready.json').exists())
        with patch('shutil.disk_usage', return_value=SimpleNamespace(free=boundary)), \
                patch('os.access', return_value=True):
            self.assertEqual(receipts._check_directory_resources(Path(config['root'])), boundary)


if __name__ == '__main__':
    unittest.main(verbosity=2)
