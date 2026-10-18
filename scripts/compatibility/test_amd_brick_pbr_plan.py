"""Actual production reservation path; only publication side effects are redirected."""
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'docs/compatibility/amd-brick-pbr-plan-repair-evidence'
P=ROOT.parent/'artifacts/amd-brick-pbr-plan-repair-375ed55'


def cpu_only(event,args):
    if event=='import' and str(args[0]).split('.')[0] in {'torch','triton','trellis2','flex_gemm','o_voxel','cumesh'}:
        raise RuntimeError('Plan qualification forbids model/native/GPU imports')
    if event in ('subprocess.Popen','os.system','socket.connect','socket.bind','socket.getaddrinfo'):
        raise RuntimeError('Plan qualification forbids processes, WSL and network')


sys.dont_write_bytecode=True
sys.pycache_prefix=str(P/'nonexistent-test-bytecode')
sys.addaudithook(cpu_only)
spec=importlib.util.spec_from_file_location('brick_pbr_actual_plan_test',E/'host.py')
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)


class Publications:
    """Keep real serialization/writes, but use ordinary CPU fixture filenames."""
    def __enter__(self):
        fixtures=h.DESTINATION/'cpu-fixtures';fixtures.mkdir(exist_ok=True)
        self.directory=Path(tempfile.mkdtemp(prefix='plan-',dir=fixtures))
        self.files={};self.order=[];self.mkdir_calls=[];self.stack=ExitStack()
        publish=h.receipts._publish;digest=h.digest;save=h.save;mkdir=Path.mkdir
        def capture_publish(path,value):
            path=Path(path)
            if path not in (h.PROBE/'neural-attempt.marker',h.ACTIVE_PLAN_PATH):
                raise AssertionError('Unexpected production publication')
            target=self.directory/('cpu-only-'+path.name)
            publish(target,value);self.files[path]=target;self.order.append(path.name)
        def capture_save(path,value):
            path=Path(path)
            if path!=h.EVIDENCE/'neural-attempt.marker':raise AssertionError('Unexpected host publication')
            target=self.directory/'cpu-only-evidence-marker.json'
            save(target,value);self.files[path]=target;self.order.append('evidence-marker')
        def capture_mkdir(path,*args,**kwargs):
            if path==h.EVIDENCE/'run':self.mkdir_calls.append(path);return None
            return mkdir(path,*args,**kwargs)
        self.stack.enter_context(patch.object(h.receipts,'_publish',side_effect=capture_publish))
        self.stack.enter_context(patch.object(h,'digest',side_effect=lambda path:digest(self.files.get(Path(path),Path(path)))))
        self.stack.enter_context(patch.object(h,'save',side_effect=capture_save))
        self.stack.enter_context(patch.object(Path,'mkdir',capture_mkdir))
        self.stack.enter_context(patch.object(h,'HARNESS_COMMIT','375ed55000cce4e724e04d61ace218462160b666'))
        self.stack.enter_context(patch.object(h,'ACTIVE_PLAN_SHA',None))
        return self
    def __exit__(self,*args):return self.stack.__exit__(*args)


class CompleteProductionPlanTests(unittest.TestCase):
    def test_exact_original_contract_conversion(self):
        self.assertEqual(h.to_guest(h.GEOMETRY/'run-contract.json'),
            '/mnt/c/Users/wattz/Documents/ChatGPT/modly/artifacts/amd-brick-neural-completion-c9447dd/neural-output/run-contract.json')

    def test_complete_actual_reserve_constructs_every_field_and_real_publisher_readback(self):
        with Publications() as capture:
            h.reserve_attempt(h.EVIDENCE/'run')
            plan_path=capture.files[h.ACTIVE_PLAN_PATH]
            plan=json.loads(plan_path.read_text())
            self.assertEqual(set(plan),{'schema','harness_commit','manifest_path','manifest_sha256','scratch',
                'field','probe','marker_path','marker_sha256','attestation_path','attestation_sha256',
                'geometry_contract_path','geometry_contract_file_sha256'})
            self.assertEqual(plan['schema'],'modly.amd.brick-pbr-execution-plan.v1')
            self.assertEqual(plan['harness_commit'],'375ed55000cce4e724e04d61ace218462160b666')
            self.assertEqual(plan['geometry_contract_file_sha256'],'4676626d1061e6cd7090b560f952a974a584f8ea9e6218378b7902294b93bd4f')
            self.assertEqual(plan['field'],'/mnt/c/Users/wattz/Documents/ModlyArtifacts/amd-brick-pbr-plan-repair-375ed55/field')
            self.assertEqual(plan['probe'],'/mnt/c/Users/wattz/Documents/ModlyArtifacts/amd-brick-pbr-plan-repair-375ed55/probe')
            self.assertEqual(plan['scratch'],'/home/modly/brick-pbr-plan-repair-375ed55')
            self.assertEqual(plan['marker_sha256'],hashlib.sha256(capture.files[h.PROBE/'neural-attempt.marker'].read_bytes()).hexdigest())
            self.assertEqual(h.ACTIVE_PLAN_SHA,hashlib.sha256(plan_path.read_bytes()).hexdigest())
            self.assertEqual(capture.order,['neural-attempt.marker','execution-plan.json','evidence-marker'])
            self.assertEqual(capture.mkdir_calls,[h.EVIDENCE/'run'])
            # Run the genuine guest verifier against the same real files using
            # only a virtual /mnt/c-to-Windows read bridge; no guest is started.
            worker=h.plan_worker(h.digest(h.WORKER));required=worker.required_candidate_paths()
            plain=worker.read_plain;verify=worker.verify_manifest
            guest_root=Path('/mnt/c/'+ROOT.as_posix()[3:]);guest_ev=guest_root/h.EVIDENCE.relative_to(ROOT)
            mapped={Path(plan['probe'])/'execution-plan.json':plan_path,
                Path(plan['marker_path']):capture.files[h.PROBE/'neural-attempt.marker'],
                Path(plan['attestation_path']):h.PROBE/'ownership.json',
                Path(plan['geometry_contract_path']):h.GEOMETRY/'run-contract.json',
                Path(plan['manifest_path']):h.EVIDENCE/'candidate-identities.json'}
            with patch.multiple(worker,ROOT=guest_root,EV=guest_ev,FIELD=Path(plan['field']),PROBE=Path(plan['probe']),SCRATCH=Path(plan['scratch'])),\
                 patch.object(worker,'read_plain',side_effect=lambda path,maximum:plain(mapped.get(Path(path),Path(path)),maximum)),\
                 patch.object(worker,'verify_manifest',side_effect=lambda path,sha:verify(mapped[Path(path)],sha,root=ROOT,required=required)):
                accepted=worker.authenticate_plan(Path(plan['probe'])/'execution-plan.json',h.ACTIVE_PLAN_SHA,guest_ev/'run/guest')
            self.assertEqual({key:accepted[key] for key in plan},plan)
            self.assertGreaterEqual(len(accepted['candidate_identities']),len(required))
            # Actual command builder only returns argv; audit forbids execution.
            command=h.worker_command(h.to_guest(h.WORKER),h.to_guest(h.EVIDENCE/'run/guest'))
            self.assertIn('450s',command);self.assertEqual(command[-4:],['--plan',plan['probe']+'/execution-plan.json','--plan-sha256',h.ACTIVE_PLAN_SHA])
        self.assertFalse((h.PROBE/'neural-attempt.marker').exists())
        self.assertFalse(h.ACTIVE_PLAN_PATH.exists());self.assertFalse((h.EVIDENCE/'run').exists())

    def test_foreign_last_plan_path_cannot_publish_marker(self):
        with Publications() as capture,patch.object(h,'GEOMETRY',Path('C:/foreign/neural-output')):
            with self.assertRaises((ValueError,FileNotFoundError)):h.reserve_attempt(h.EVIDENCE/'run')
            self.assertEqual(capture.order,[]);self.assertEqual(capture.mkdir_calls,[])

    def test_malformed_commit_cannot_publish_marker(self):
        with Publications() as capture,patch.object(h,'HARNESS_COMMIT','not-a-commit'):
            with self.assertRaises(ValueError):h.reserve_attempt(h.EVIDENCE/'run')
            self.assertEqual(capture.order,[]);self.assertEqual(capture.mkdir_calls,[])

    def test_real_publisher_bytes_match_prediction_including_windows_newlines(self):
        with Publications() as capture:
            value={'schema':'cpu-serializer-proof','unicode':'\u03a9','nested':{'count':1957891}}
            path=capture.directory/'ordinary-cpu-serializer.json'
            h.receipts._frozen_receipt.durable_json(path,value)
            self.assertEqual(h.serialized_publication(value),path.read_bytes())
            self.assertEqual(json.loads(path.read_bytes()),value)
            self.assertIn(b'\r\n',path.read_bytes())

    def test_last_record_serialization_failure_cannot_publish_marker(self):
        original=h.serialized_publication
        def fail_last(value):
            if 'plan_sha256' in value:raise ValueError('Injected final evidence serialization refusal')
            return original(value)
        with Publications() as capture,patch.object(h,'serialized_publication',side_effect=fail_last):
            with self.assertRaisesRegex(ValueError,'final evidence'):h.reserve_attempt(h.EVIDENCE/'run')
            self.assertEqual(capture.order,[]);self.assertEqual(capture.mkdir_calls,[])

    def test_missing_and_altered_required_source_manifest_cannot_publish_marker(self):
        for changed in ('missing','altered'):
            with self.subTest(changed=changed),Publications() as capture:
                values=json.loads((h.EVIDENCE/'candidate-identities.json').read_text())
                key='scripts/compatibility/amd_forward_owned_job.py'
                if changed=='missing':del values[key]
                else:values[key]='0'*64
                (capture.directory/'candidate-identities.json').write_text(json.dumps(values))
                with patch.object(h,'EVIDENCE',capture.directory):
                    with self.assertRaises(ValueError):h.reserve_attempt(capture.directory/'run')
                self.assertEqual(capture.order,[]);self.assertEqual(capture.mkdir_calls,[])

    def test_changed_texture_input_and_unsupported_scratch_cannot_publish_marker(self):
        with Publications() as capture:
            foreign=capture.directory/'texture-latent.safetensors'
            blob=h.inputs.TEXTURE_PATH.read_bytes();foreign.write_bytes(blob[:-1]+bytes([blob[-1]^1]))
            (capture.directory/'texture-latent-metadata.json').write_bytes(h.inputs.TEXTURE_PATH.with_name('texture-latent-metadata.json').read_bytes())
            with patch.object(h.inputs,'TEXTURE_PATH',foreign):
                with self.assertRaises(ValueError):h.reserve_attempt(h.EVIDENCE/'run')
            self.assertEqual(capture.order,[])
        with Publications() as capture,patch.object(h,'SCRATCH','/home/modly/foreign'):
            with self.assertRaises(ValueError):h.reserve_attempt(h.EVIDENCE/'run')
            self.assertEqual(capture.order,[])

    def test_preexisting_control_or_output_never_consumes_another_marker(self):
        for kind in ('control','output','pending'):
            with self.subTest(kind=kind),Publications() as capture:
                if kind=='control':
                    with patch.object(h,'ACTIVE_PLAN_PATH',capture.directory/'foreign-plan.json'):
                        with self.assertRaises(ValueError):h.reserve_attempt(h.EVIDENCE/'run')
                else:
                    if kind=='pending':(capture.directory/'probe').mkdir()
                    occupied=capture.directory/'run' if kind=='output' else capture.directory/'probe/neural-attempt.marker.publishing'
                    occupied.write_bytes(b'preserved CPU occupied sentinel')
                    if kind=='output':
                        with patch.object(h,'EVIDENCE',capture.directory):
                            with self.assertRaises(FileExistsError):h.reserve_attempt(capture.directory/'run')
                    else:
                        with patch.object(h,'DESTINATION',capture.directory),patch.object(h,'FIELD',capture.directory/'field'),\
                             patch.object(h,'PROBE',capture.directory/'probe'),patch.object(h,'ACTIVE_PLAN_PATH',capture.directory/'probe/execution-plan.json'):
                            with self.assertRaises(FileExistsError):h.reserve_attempt(h.EVIDENCE/'run')
                    self.assertEqual(occupied.read_bytes(),b'preserved CPU occupied sentinel')
                self.assertEqual(capture.order,[])


if __name__=='__main__':unittest.main(verbosity=2)
