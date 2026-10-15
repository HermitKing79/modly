"""CPU-only host routing/readback tests; no WSL, model or GPU entry is allowed."""
import hashlib
import importlib.abc
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout

class NoAI(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'torch','triton','trellis2','flex_gemm','transformers'}:
            raise AssertionError('Unexpected model/GPU import: '+name)
sys.meta_path.insert(0,NoAI())
try:import amd_generation_decode as subject
except ModuleNotFoundError as error:
    if error.name!='amd_generation_decode':raise
    subject=None

@unittest.skipIf(subject is None,'Host adapter not implemented')
class DecodeHostTests(unittest.TestCase):
    def test_git_read_timeout_preserves_default_and_exact_content_checks(self):
        import amd_sparse_decoder_forward as verifier
        path=self.root/'published.py';path.write_bytes(b'expected\r\n')
        with patch.object(verifier,'ROOT',self.root),patch.object(verifier.subprocess,'run',return_value=SimpleNamespace(stdout=b'expected\n')) as call:
            verifier.verify_committed_files('1'*40,['published.py'])
            self.assertEqual(call.call_args.kwargs['timeout'],5)
            verifier.verify_committed_files('1'*40,['published.py'],timeout_seconds=subject.GIT_READ_SECONDS)
            self.assertEqual(call.call_args.kwargs['timeout'],15)
            call.return_value.stdout=b'changed\n'
            with self.assertRaisesRegex(RuntimeError,'differs from published'):
                verifier.verify_committed_files('1'*40,['published.py'],timeout_seconds=15)
    def test_git_read_timeout_rejects_invalid_budgets_before_command(self):
        import amd_sparse_decoder_forward as verifier
        with patch.object(verifier.subprocess,'run') as call:
            for value in (0,-1,31,True,float('inf'),float('nan'),'15'):
                with self.assertRaises(ValueError):verifier.verify_committed_files('1'*40,[],timeout_seconds=value)
            call.assert_not_called()
    def setUp(self):
        import numpy as np
        self.np=np
        self.temp=tempfile.TemporaryDirectory(prefix='_decode-test-',dir=subject.APPROVED_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.patch=patch.object(subject,'APPROVED_ROOT',self.root);self.patch.start();self.addCleanup(self.patch.stop)
        self.real_lineage_checker=subject.verify_shape_lineage
        self.lineage_patch=patch.object(subject,'verify_shape_lineage',return_value=[])
        self.lineage_patch.start();self.addCleanup(self.lineage_patch.stop)
        self.shape=self.root/'shape';scratch=self.shape/'run/scratch';scratch.mkdir(parents=True)
        arrays={'shape-latent-denormalized.npy':np.zeros((2,32),dtype='<f4'),
                'shape-latent-coordinates.npy':np.array([[0,0,0,0],[0,1,0,0]],dtype='<i4')}
        artifacts={}
        for name,a in arrays.items():
            path=scratch/name;np.save(path,a,allow_pickle=False)
            artifacts[name]={'path':str(path),'file_bytes':path.stat().st_size,'sha256':subject.digest(path),
                'payload_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'shape':list(a.shape),'dtype':str(a.dtype),'verified':True}
        process={'returncode':0,'timed_out':False,'job_cleanup':{'owned_tree_exit_verified':True,'forced_job_cleanup':False}}
        for key in ('worker_exit_verified','gpu_authorized','output_readers_completed','cpu_observer_stopped',
                    'windows_periodic_stopped','diagnostic_journal_stopped','windows_journal_stopped',
                    'job_attached_before_preparation'):process[key]=True
        self.result={'classification':'PASS','observer_healthy':True,'checkpoint_unchanged':True,'process':process,
            'worker':{'status':'PASS','stage':'complete','token_count':2,'forward_calls':21,
                'sampling':{'sequences':1,'started':21,'completed':21,'positive_calls':12,'negative_calls':9},
                'release':{'allocated_bytes':0,'reserved_bytes':0,'model_collected':True,'owned_tensors_collected':True},
                'artifacts':artifacts}}
        (self.shape/'run/results.json').write_text(json.dumps(self.result))
        self.runtime=subject.Runtime(self.shape,self.root/'decode','1'*40,'run-cpu-fixture')

    def test_source_shape_rejects_changed_array_and_unreleased_worker(self):
        self.assertEqual(self.runtime.bundle['count'],2)
        path=Path(self.result['worker']['artifacts']['shape-latent-denormalized.npy']['path'])
        original=path.read_bytes();path.write_bytes(original+b'x')
        with self.assertRaises(ValueError):subject.shape_inputs(self.shape)
        path.write_bytes(original)
        self.result['worker']['release']['allocated_bytes']=1
        (self.shape/'run/results.json').write_text(json.dumps(self.result))
        with self.assertRaises(ValueError):subject.shape_inputs(self.shape)

    def test_genuine_source_checker_rejects_a_claimed_pass_rejected_by_classifier(self):
        import amd_generation_shape as generation
        with patch.object(generation,'classify',return_value='PARTIAL'):
            with self.assertRaises(ValueError):
                self.real_lineage_checker(self.shape,self.result,self.result['worker'],None,None)

    def test_preflight_rejects_wrong_head_before_marker_or_resource_work(self):
        with patch.dict('os.environ',{'WORKSPACE_DIR':str(subject.ROOT)}),\
                patch('subprocess.check_output',return_value='0'*40+'\n'),\
                patch.object(self.runtime,'host_admission',side_effect=AssertionError('too late')):
            with self.assertRaises(ValueError):self.runtime.preflight()
        self.assertFalse(self.runtime.evidence.exists())

    def test_large_tiling_is_explicit_and_bound_to_run_contract(self):
        self.assertFalse(self.runtime.contract.large_convolution_tiling)
        self.assertNotIn('large_convolution_tiling',self.runtime.contract.document())
        enabled=subject.Runtime(self.shape,self.root/'decode','1'*40,'run-cpu-fixture',
                                tile_large_convolutions=True)
        self.assertTrue(enabled.contract.large_convolution_tiling)
        self.assertNotEqual(enabled.contract.sha256,self.runtime.contract.sha256)
        self.assertIs(enabled.contract.document()['large_convolution_tiling'],True)
        with self.assertRaises(ValueError):
            subject.Runtime(self.shape,self.root/'decode','1'*40,'run-cpu-fixture',tile_large_convolutions=1)
        self.assertFalse(self.runtime.evidence.exists())

    def test_live_admission_enforces_twelve_gib_and_records_failure_values(self):
        def memory(pointer):pointer._obj.available=11*1024**3;return 1
        capture=io.StringIO()
        with patch('ctypes.windll.kernel32.GlobalMemoryStatusEx',side_effect=memory),\
                patch('shutil.disk_usage',return_value=SimpleNamespace(free=100*1024**3)),redirect_stdout(capture):
            with self.assertRaises(RuntimeError):self.runtime.host_admission()
        row=json.loads(capture.getvalue().split('HOST_RESOURCE_ADMISSION=',1)[1])
        self.assertEqual(row['required_ram_bytes'],12*1024**3)
        self.assertEqual(row['available_ram_bytes'],11*1024**3)
        self.assertFalse(row['admitted'])
        self.assertTrue(row['timestamp_utc'])
        self.assertFalse(self.runtime.evidence.exists())

    def test_bootstrap_waits_for_gated_child_then_copies_inside_outer_owner(self):
        calls=[];mirror=self.root/'mirror';mirror.mkdir()
        contract=SimpleNamespace(sha256='a'*64,output_directory=self.root/'fresh-guest-parent/geometry',host_evidence_directory=mirror)
        fake_contract=SimpleNamespace(load=lambda path:contract)
        fake_paths=SimpleNamespace(attest_parent=lambda c:calls.append('parent'),
            attest_output=lambda c:calls.append('output') or Path(c.output_directory))
        def run(command,**kwargs):
            self.assertNotIn('stdin',kwargs)  # GO/EOF is inherited, never replaced.
            calls.append('child-exited');return SimpleNamespace(returncode=0)
        with subject.module_binding('run_contract',fake_contract),subject.module_binding('artifact_paths',fake_paths),\
                patch('subprocess.run',side_effect=run),patch('os.getuid',return_value=1000,create=True),\
                patch.object(subject,'copy_files',side_effect=lambda *args:calls.append('copy') or {}):
            self.assertEqual(subject.guest_bootstrap(str(subject.GUEST_ENTRY),'contract.json','a'*64,subject.COPY_LIMITS),0)
        self.assertEqual(calls,['parent','child-exited','output','copy'])

    def test_command_keeps_owned_distro_deadlines_and_private_parent(self):
        command=self.runtime.worker_command(subject.guest_path(subject.GUEST_ENTRY),subject.guest_path(self.runtime.evidence/'run/guest'))
        self.assertEqual(command[:4],['wsl.exe','-d','Modly-AMD-Ubuntu-24.04','-u'])
        self.assertIn('450s',command)
        self.assertIn('--kill-after=5s',command)
        self.assertIn('/opt/modly-wsl/venv/bin/python',command)
        self.assertIn(self.runtime.contract.sha256,command)
        self.assertIn('exist_ok=False',command[command.index('-c')+1])
        self.assertEqual(str(self.runtime.contract.output_directory),'/home/modly/run-cpu-fixture/geometry')
        with self.assertRaises(ValueError):subject.Runtime(self.shape,self.root/'decode2','1'*40,'../escape')

    def test_adapter_globals_do_not_change_shared_helper_script(self):
        import amd_wsl_rocdxg_smoke as shared
        before=shared.SCRIPT
        supervisor=self.runtime.supervisor()
        self.assertEqual(supervisor.SCRIPT,subject.GUEST_ENTRY)
        self.assertIsNot(supervisor.base,shared)
        self.assertIs(supervisor.base.terminate_owned_distro,shared.terminate_owned_distro)
        self.assertEqual(shared.SCRIPT,before)
        self.assertEqual((supervisor.NATIVE_DEADLINE,supervisor.CPU_DEADLINE,supervisor.GPU_DEADLINE,supervisor.HOST_DEADLINE),(60.,120.,240.,480.))

    def test_marker_and_mirror_are_exclusive_and_contract_is_exact(self):
        self.runtime.reserve_attempt(self.runtime.evidence/'run')
        path=self.runtime.evidence/'run-contract.json'
        self.assertEqual(subject.read_json(path),self.runtime.contract.document())
        self.assertEqual(list((self.runtime.evidence/'run/guest').iterdir()),[])
        with self.assertRaises(FileExistsError):self.runtime.reserve_attempt(self.runtime.evidence/'run')

    def fixture_guest(self):
        np=self.np;rt=self.runtime
        rt.reserve_attempt(rt.evidence/'run');target=rt.evidence/'run/guest'
        source=self.root/'guest-files';source.mkdir()
        parent=rt.bundle['coords'].copy();tensors={};stages=[]
        for i in range(4):
            logits=np.full((2,8),0xbf80,dtype='<u2');logits[:,0]=0x3f80
            tensors[f'subdivision_{i}_coordinates']=('I32',parent.copy())
            tensors[f'subdivision_{i}_logits']=('BF16',logits)
            stages.append({'parents':2,'selected_children':2});parent=parent.copy();parent[:,1:]*=2
        tensors.update(features=('F32',np.zeros((2,7),dtype='<f4')),coordinates=('I32',parent))
        summary={'rows':2,'subdivisions':stages,'finite':True}
        inputs={**rt.bundle['summary'],'run_contract_sha256':rt.contract.sha256,'source_job_sha256':rt.contract.source_job_sha256,
            'input_paths':{k:str(v) for k,v in rt.contract.input_paths.items()},'input_sha256':dict(rt.contract.input_sha256)}
        provenance={'run_contract_sha256':rt.contract.sha256,'source_job_sha256':rt.contract.source_job_sha256,
            'checkpoint_sha256':subject.CHECKPOINT_SHA,'inputs':inputs,'input_file_sha256':dict(rt.contract.input_sha256),
            'resolution':512,'forward_calls':1,'convolutions':40,'mesh_extraction':False}
        header={'__metadata__':{'provenance':json.dumps(provenance,sort_keys=True),'summary':json.dumps(summary,sort_keys=True)}}
        payload=b'';hashes={}
        for name,(dtype,array) in tensors.items():
            blob=array.tobytes();header[name]={'dtype':dtype,'shape':list(array.shape),'data_offsets':[len(payload),len(payload)+len(blob)]}
            hashes[name]=hashlib.sha256(blob).hexdigest();payload+=blob
        encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
        path=source/'shape-decoded.safetensors';path.write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
        output={**summary,'sha256':subject.digest(path),'bytes':path.stat().st_size,'tensor_payload_sha256':hashes,
            'verified':True,'readback_verified_before_release':True,'path':str(rt.contract.output_directory/'shape-decoded.safetensors')}
        worker=dict(classification='QUALIFIED',decoder_cpu='PASS',native_import='PASS',forward_calls=1,convolutions_completed=40,
            final_convolution_chunks=1,final_convolution_rows=2,native_map_calls=5,final_convolution_patch_restored=True,
            subdivision_operations=[{}]*8,output=output,CPU_model_collected=True,CPU_parameters_collected=True,CPU_component_collected=True,
            memory_release={'allocated':0,'reserved':0},workspace_clear_calls=1,runtime_dtype_counts={'BF16':216,'F32':76},
            constructed_on_meta=True,buffers_count=0,decoder_diagnostics={'strict':True,'roundtrip_verified':True,'entries':292,
                'dtype_counts':{'F16':216,'F32':76},'missing_keys':[],'unexpected_keys':[]},policy_final={'rejected_calls':0},
            run_contract_sha256=rt.contract.sha256,source_job_sha256=rt.contract.source_job_sha256,saved_inputs=inputs)
        for name,row in [('results.json',worker),('run-contract.json',rt.contract.document()),
            ('provenance.json',{'provenance':provenance,'output':output,'run_contract':rt.contract.document()})]:
            (source/name).write_text(json.dumps(row))
        journal=b'{"phase":"worker-finished"}\n';(source/'phases.jsonl').write_bytes(journal);(target/'phases.jsonl').write_bytes(journal)
        return source,target

    def test_allowlisted_copy_and_ten_tensor_readback_reject_corruption(self):
        source,target=self.fixture_guest()
        (source/'unexpected.bin').write_bytes(b'not copied')
        self.runtime.copy_guest_output(source)
        self.assertFalse((target/'unexpected.bin').exists())
        receipt=self.runtime.verify_outputs()
        self.assertTrue(receipt['ten_tensors_verified'])
        self.assertEqual(receipt['rows'],2)
        path=target/'shape-decoded.safetensors';blob=bytearray(path.read_bytes());blob[-1]^=1;path.write_bytes(blob)
        with self.assertRaises(ValueError):self.runtime.verify_outputs()

    def test_final_cleanup_error_cannot_pass_host_readback(self):
        source,target=self.fixture_guest();self.runtime.copy_guest_output(source)
        row=subject.read_json(target/'results.json');row['cleanup_error']='fixture failure'
        (target/'results.json').write_text(json.dumps(row))
        with self.assertRaises(RuntimeError):self.runtime.verify_outputs()

    def test_missing_artifact_still_preserves_durable_failure_results(self):
        source,target=self.fixture_guest();(source/'shape-decoded.safetensors').unlink()
        with self.assertRaises((ValueError,FileNotFoundError)):self.runtime.copy_guest_output(source)
        self.assertTrue((target/'results.json').is_file())
        self.assertEqual(subject.read_json(target/'run-contract.json'),self.runtime.contract.document())

    def test_overall_pass_requires_independent_readback_after_qualified_supervisor(self):
        source,target=self.fixture_guest();self.runtime.copy_guest_output(source)
        path=target/'shape-decoded.safetensors';blob=bytearray(path.read_bytes());blob[-1]^=1;path.write_bytes(blob)
        (self.runtime.evidence/'run/results.json').write_text(json.dumps({'classification':'QUALIFIED','elapsed_seconds':1.}))
        with patch.object(self.runtime,'supervisor',return_value=SimpleNamespace(supervise=lambda output:0)):
            self.assertEqual(self.runtime.supervise(),1)
        self.assertEqual(subject.read_json(self.runtime.evidence/'final-result.json')['classification'],'FAIL')

class PresenceTests(unittest.TestCase):
    def test_adapter_exists(self):self.assertIsNotNone(subject,'Reusable Windows decode owner missing')

class ActualFreshShapeTests(unittest.TestCase):
    def test_actual_4430_source_passes_genuine_cpu_lineage_reader(self):
        row=subject.shape_inputs(subject.ROOT/'docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/occupancy/shape')
        self.assertEqual(row['count'],4430)
        self.assertEqual(row['source_job_sha256'],'b92aade7aca2e832fbf425ec9d4e54d40aa7ffa4b4afdea54c38f71f61b8e62f')
        self.assertTrue(row['required_files'])
        self.assertFalse(any(name in sys.modules for name in ('torch','triton','trellis2')))

if __name__=='__main__':unittest.main()
