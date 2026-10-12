"""Pure CPU contract tests; no AI imports, checkpoints or GPU initialization."""
import importlib.util
import math
from pathlib import Path
import unittest
import sys
import io
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parent))

spec = importlib.util.spec_from_file_location('forward_contract', Path(__file__).with_name('amd_decoder_forward_contract.py'))
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
proof_spec=importlib.util.spec_from_file_location('forward_proof',Path(__file__).with_name('amd_sparse_decoder_forward.py'))
proof=importlib.util.module_from_spec(proof_spec)
proof_spec.loader.exec_module(proof)

CONFIG = {'name':'SparseStructureDecoder','args':{'out_channels':1,'latent_channels':8,'num_res_blocks':2,
          'num_res_blocks_middle':2,'channels':[512,128,32],'use_fp16':True}}


def fake_owned_job():
    return SimpleNamespace(attach=lambda child:None,finish=lambda:{'owned_tree_exit_verified':True,'forced_job_cleanup':False},close=lambda:None)


class ForwardContractTests(unittest.TestCase):
    def test_minimal_architecture_valid_fixture_and_upsampling(self):
        fixture = contract.fixture(CONFIG)
        self.assertEqual(fixture['input_shape'], [1,8,1,1,1])
        self.assertEqual(fixture['output_shape'], [1,1,4,4,4])
        self.assertEqual(fixture['input_dtype'], 'torch.float32')
        self.assertEqual(len(fixture['values']), 8)
        self.assertEqual(fixture, contract.fixture(CONFIG))

    def test_configuration_changes_require_new_inspection(self):
        for change in ({'latent_channels':4},{'channels':[512,128]},{'use_fp16':False},{'norm_type':'group'}):
            with self.assertRaises(ValueError):
                contract.fixture({**CONFIG,'args':{**CONFIG['args'],**change}})

    def test_forward_permit_is_single_use_even_after_exception(self):
        permit = contract.SingleForward()
        calls=[]
        self.assertEqual(permit.invoke(lambda: calls.append(1) or 42),42)
        with self.assertRaises(RuntimeError): permit.invoke(lambda: calls.append(2))
        self.assertEqual(calls,[1])
        failed=contract.SingleForward()
        def error(): raise ValueError('native operation failed')
        with self.assertRaises(ValueError): failed.invoke(error)
        with self.assertRaises(RuntimeError): failed.invoke(lambda: 0)
        self.assertEqual(failed.calls,1)

    def output(self):
        return {'shape':[1,1,4,4,4],'dtype':'torch.float32','device':'cuda:0',
                'requires_grad':False,'inference_mode':True,'values':[float(i)/64 for i in range(64)]}

    def test_output_shape_dtype_device_and_finiteness_are_required(self):
        self.assertEqual(contract.validate_output(self.output(), contract.fixture(CONFIG))['count'],64)
        for change in ({'shape':[1,1,2,2,2]},{'dtype':'torch.float16'},{'device':'cpu'},
                       {'requires_grad':True},{'inference_mode':False},{'values':[math.nan]*64}, {'values':[1.]*63}):
            with self.assertRaises(ValueError):
                contract.validate_output({**self.output(),**change}, contract.fixture(CONFIG))

    def test_cpu_statistics_have_no_gpu_dependency(self):
        stats=contract.validate_output(self.output(), contract.fixture(CONFIG))
        self.assertEqual(stats['minimum'],0.)
        self.assertEqual(stats['maximum'],63/64)
        self.assertEqual(stats['mean'],63/128)
        self.assertEqual(len(stats['representative_values']),8)

    def good_record(self):
        return {'status':'PASS','stage':'complete','forward_calls':1,'root_decoder_calls':1,'load_calls':1,
                'transfer_calls':{'input':1,'output':1},
                'fixture':contract.fixture(CONFIG),'cpu_load':{'status':'PASS'},
                'forward':{'status':'PASS','output':self.output(),'grad_enabled':False,'output_is_inference_tensor':True},
                'release':{'allocated_bytes':0,'reserved_bytes':0,'model_collected':True,'input_collected':True,'output_collected':True}}

    def test_pass_requires_single_complete_forward_clean_release_and_all_gates(self):
        record=self.good_record()
        self.assertEqual(contract.classify(record, True, True, True), 'PASS')
        for flags in ((False,True,True),(True,False,True),(True,True,False)):
            self.assertEqual(contract.classify(record,*flags),'PARTIAL')
        for change in ({'forward_calls':0},{'forward_calls':2},{'status':'FAIL'},
                       {'root_decoder_calls':2},{'load_calls':2},
                       {'forward':{'status':'FAIL'}},{'release':{**record['release'],'output_collected':False}}):
            self.assertEqual(contract.classify({**record,**change},True,True,True),'PARTIAL')

    def test_success_label_does_not_override_invalid_output(self):
        record=self.good_record()
        record['forward']['output']['values'][0]=math.inf
        self.assertEqual(contract.classify(record,True,True,True),'PARTIAL')

    def test_transfer_guard_has_no_compute_allowance(self):
        self.assertTrue(contract.placement_operation('aten._to_copy.default'))
        for op in ('aten.convolution.default','aten.mm.default','aten.add.Tensor'):
            self.assertFalse(contract.placement_operation(op))

    def test_verified_lazy_pipeline_package_is_allowed_but_pipeline_classes_and_mesh_modules_are_not(self):
        self.assertFalse(contract.prohibited_import('trellis2.pipelines'))
        self.assertFalse(contract.prohibited_import('trellis2.models.sparse_structure_vae'))
        for name in ('trellis2.pipelines.trellis2_image_to_3d','trellis2.pipelines.trellis2_texturing','cumesh','o_voxel._C','nvdiffrast.torch'):
            self.assertTrue(contract.prohibited_import(name))

    def test_unhealthy_observer_prevents_launch_and_existing_attempt_cannot_repeat(self):
        with tempfile.TemporaryDirectory(dir=proof.ROOT.parent) as tmp:
            evidence=Path(tmp)
            with patch.object(proof,'EVIDENCE',evidence), patch.object(proof,'source_identity',return_value={}), \
                 patch.object(proof,'verify_committed_files',return_value={}), \
                 patch.object(proof.previous,'identity',return_value={'config':CONFIG}), \
                 patch.object(proof.subprocess,'run',return_value=SimpleNamespace(stdout=b'published')), \
                 patch.object(proof,'windows_observation',return_value={}), \
                 patch.object(proof,'healthy_observation',return_value=False), patch.object(proof.subprocess,'Popen') as launch:
                self.assertEqual(proof.supervise(evidence/'run','published'),1)
                launch.assert_not_called()
                self.assertTrue((evidence/'attempt-started.json').is_file())
                before=(evidence/'run/results.json').read_bytes()
                with self.assertRaises(FileExistsError):proof.supervise(evidence/'run','published')
                self.assertEqual(before,(evidence/'run/results.json').read_bytes())

    def test_parent_protocol_error_unconditionally_reaps_owned_worker(self):
        child=SimpleNamespace(pid=123,stdin=io.BytesIO(),stdout=io.BytesIO(b'FORWARD_RESULT=invalid-json\n'),stderr=io.BytesIO())
        def reap(owned,process):
            self.assertIs(owned,child)
            process.update(returncode=-1,worker_exit_verified=True,forced_cleanup=True)
        with tempfile.TemporaryDirectory(dir=proof.ROOT.parent) as tmp:
            evidence=Path(tmp)
            with patch.object(proof,'EVIDENCE',evidence), patch.object(proof,'source_identity',return_value={}), \
                 patch.object(proof,'verify_committed_files',return_value={}), \
                 patch.object(proof.previous,'identity',return_value={'config':CONFIG}), \
                 patch.object(proof.subprocess,'run',return_value=SimpleNamespace(stdout=b'published')), \
                 patch.object(proof,'windows_observation',return_value={}), \
                 patch.object(proof,'healthy_observation',return_value=True), \
                 patch.object(proof,'prepare_private_cache',return_value={}), \
                 patch.object(proof.subprocess,'Popen',return_value=child), \
                 patch.object(proof,'OwnedJob',side_effect=fake_owned_job), \
                 patch.object(proof.previous,'reap_owned_child',side_effect=reap) as cleanup:
                self.assertEqual(proof.supervise(evidence/'run','published'),1)
                cleanup.assert_called_once()
                self.assertTrue(child.stdin.closed)

    def test_executed_harness_must_match_tracked_published_bytes(self):
        with tempfile.TemporaryDirectory(dir=proof.ROOT.parent) as tmp:
            root=Path(tmp);path=root/'harness.py';path.write_bytes(b'original\r\n')
            with patch.object(proof,'ROOT',root),patch.object(proof,'COMMITTED_FILES',['harness.py']), \
                 patch.object(proof.subprocess,'run',return_value=SimpleNamespace(stdout=b'original\n')):
                self.assertIn('harness.py',proof.verify_committed_files('a'*40))
                path.write_bytes(b'changed\n')
                with self.assertRaisesRegex(RuntimeError,'differs'):proof.verify_committed_files('a'*40)
            with patch.object(proof,'ROOT',root),patch.object(proof,'COMMITTED_FILES',['untracked.py']), \
                 patch.object(proof.subprocess,'run',side_effect=proof.subprocess.CalledProcessError(128,'git show')):
                with self.assertRaisesRegex(RuntimeError,'absent'):proof.verify_committed_files('a'*40)

    def test_cpu_gate_deadline_expiry_never_authorizes_gpu_and_reaps_worker(self):
        for expiry in ('observer','monitor'):
            now=[0.];observations=[];written=[];started=[]
            class Input(io.BytesIO):
                def write(self,value):written.append(value);return super().write(value)
            event=proof.EVENT+'{"event":"cpu-ready","gate_one":{},"checkpoint_sha256":"verified"}\n'
            child=SimpleNamespace(pid=123,stdin=Input(),stdout=io.BytesIO(event.encode()),stderr=io.BytesIO())
            def observe(since):
                observations.append(1)
                if len(observations)==2:now[0]=46. if expiry=='observer' else 44.5
                return {}
            class Monitor:
                error=None
                def __init__(self,*args):pass
                def start(self,timeout):started.append(timeout);now[0]=46.
                def close(self):return True
            def reap(owned,process):process.update(returncode=-1,worker_exit_verified=True,forced_cleanup=True)
            with tempfile.TemporaryDirectory(dir=proof.ROOT.parent) as tmp:
                evidence=Path(tmp)
                with patch.object(proof,'EVIDENCE',evidence),patch.object(proof,'source_identity',return_value={}), \
                     patch.object(proof,'verify_committed_files',return_value={}), \
                     patch.object(proof.previous,'identity',return_value={'config':CONFIG,'checkpoint_sha256':'verified'}), \
                     patch.object(proof.subprocess,'run',return_value=SimpleNamespace(stdout=b'published')), \
                     patch.object(proof,'windows_observation',side_effect=observe), \
                     patch.object(proof,'healthy_observation',return_value=True), \
                     patch.object(proof,'prepare_private_cache',return_value={}), \
                     patch.object(proof.loading_contract,'authorize_placement'), \
                     patch.object(proof.subprocess,'Popen',return_value=child), \
                     patch.object(proof,'OwnedJob',side_effect=fake_owned_job), \
                     patch.object(proof.diagnostics,'ActivityMonitor',Monitor), \
                     patch.object(proof.time,'monotonic',side_effect=lambda:now[0]), \
                     patch.object(proof.previous,'reap_owned_child',side_effect=reap) as cleanup:
                    self.assertEqual(proof.supervise(evidence/'run','published'),1)
                    cleanup.assert_called_once()
                self.assertEqual(written,[b'OWNED_JOB_ATTACHED\n'])
                self.assertEqual(started,[] if expiry=='observer' else [.5])
                result=__import__('json').loads((evidence/'run/results.json').read_text())
                self.assertTrue(result['process']['timed_out'])


if __name__ == '__main__': unittest.main()
