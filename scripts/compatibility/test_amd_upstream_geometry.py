"""Synthetic CPU host-chain handoff to the real current geometry bridge.

Only predecessor neural verifiers are substituted, as explicitly labelled in
the baseline fixture. Actual host tickets, receipts, byte/NPY readers and
current geometry binding are exercised. No worker or model is run.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'api'),str(Path(__file__).parent)]
import test_amd_current_geometry as fixture
from services.amd_completed_result import FilePin


class UpstreamGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard=fixture.NoAI();sys.meta_path.insert(0,cls.guard)

    @classmethod
    def tearDownClass(cls):sys.meta_path.remove(cls.guard)

    def setUp(self):
        # Reuse fixture logic while keeping all new disposable evidence out of
        # the now-historical geometry qualification directory.
        original_temp=tempfile.TemporaryDirectory
        def current_temp(*args,**kwargs):
            kwargs['dir']=ROOT/'docs/compatibility/amd-upstream-image-shape-evidence'
            return original_temp(*args,**kwargs)
        with patch.object(fixture.tempfile,'TemporaryDirectory',side_effect=current_temp):
            fixture.CurrentGeometryBridgeTests.setUp(self)
        from services import amd_upstream_job as upstream
        self.upstream=upstream;self.event=threading.Event()
        self.parent=tempfile.TemporaryDirectory(prefix='synthetic-chain-',
            dir=ROOT.parent/'artifacts/amd-upstream-image-shape-5501dfd')
        self.addCleanup(self.parent.cleanup)
        source=self.occ/'source-image.png';source.write_bytes(self.image)
        image_contract={'source_sha256':self.context.input_sha256,
                        'source_path':source.relative_to(ROOT).as_posix()}
        contract=self.occ/'input-contract.json';contract.write_text(json.dumps(image_contract))
        self.input_hash=fixture.host.digest(contract)
        raw=self.bundle['source_job']
        # Preserve bytes and restore only this test's own files after issuing
        # the corresponding host ticket. Historical artifacts are never used.
        raw.write_text(json.dumps({'worker':{'generation_inputs':{'input_contract_sha256':self.input_hash}}}))
        outputs=[raw]+[row['path'] for row in self.bundle['inputs'].values()]
        self.payloads={path:path.read_bytes() for path in outputs}
        for path in outputs:path.unlink()
        self.conditioning=self.shape/'synthetic-conditioning.json'
        self.occupancy=self.shape/'synthetic-occupancy.json'
        stages=(upstream.StageSpec('conditioning',(upstream.OutputSpec(self.conditioning,1024),)),
                upstream.StageSpec('occupancy',(upstream.OutputSpec(self.occupancy,1024),)),
                upstream.StageSpec('shape',tuple(upstream.OutputSpec(path,2*1024**2) for path in outputs)))
        code=ROOT/'api/services/amd_upstream_job.py'
        self.job=upstream.UpstreamJob(Path(self.parent.name),context=self.context,image_bytes=self.image,
            params=self.params,shape_evidence=self.shape,source_pins={'host':FilePin(code,fixture.host.digest(code))},
            runtime_input_pins={'image':FilePin(source,self.context.input_sha256),
                                'input_contract':FilePin(contract,self.input_hash)},
            stages=stages,cancel_event=self.event)
        self.occupancy_patch=patch('amd_generation_occupancy.verify_sequence',return_value={
            'input_contract':image_contract,'input_contract_sha256':self.input_hash})
        self.occupancy_patch.start();self.addCleanup(self.occupancy_patch.stop)
        self.bundle['required_files']=[source.relative_to(ROOT).as_posix(),contract.relative_to(ROOT).as_posix()]

    def complete(self):
        for stage in ('conditioning','occupancy','shape'):
            ticket=self.job.issue(stage)
            if stage=='shape':
                for path,blob in self.payloads.items():path.write_bytes(blob)
                self.bundle['source_job_sha256']=fixture.host.digest(self.bundle['source_job'])
            else:
                path=self.conditioning if stage=='conditioning' else self.occupancy
                path.write_text(json.dumps({'synthetic_contract_only':True,'stage':stage}))
            observed=self.upstream.HostStageObservation(ticket,stage,self.context.to_dict(),ticket.nonce,0,True,True,False)
            self.job.accept(ticket,observed)
        return self.job.complete()

    def prepare(self,authority=None,**overrides):
        args=dict(context=self.context,image_bytes=self.image,params=self.params,shape_evidence=self.shape,
                  expected_source_job_sha256=self.bundle['source_job_sha256'],harness_commit='1'*40,
                  run_id='run-current-upstream-synthetic',upstream_completion=authority)
        args.update(overrides)
        return fixture.host.prepare_current_input(**args)

    def test_real_host_chain_binds_unchanged_raw_receipt_and_private_inputs(self):
        authority=self.complete();raw=self.bundle['source_job'];before=raw.read_bytes()
        with patch('amd_sparse_decoder_forward.verify_committed_files',return_value={}) as verifier:
            contract=self.prepare(authority)
        self.assertEqual(raw.read_bytes(),before)
        self.assertNotIn('product_context',json.loads(before))
        self.assertEqual(contract.document()['current_input']['product_context'],self.context.to_dict())
        names=verifier.call_args.args[1]
        self.assertTrue(all(path not in names for path in self.bundle['required_files']))
        self.assertIn('api/services/amd_upstream_job.py',names)
        self.assertIn('experimental/extensions/trellis2-amd/inference_decoder/admission.py',names)
        self.assertEqual(contract.maximum_final_rows,1842968)
        self.assertTrue(contract.large_convolution_tiling)

    def test_worker_dictionary_or_unsealed_handoff_cannot_replace_host_authority(self):
        authority=self.complete()
        with self.assertRaisesRegex(ValueError,'context'):self.prepare()
        with self.assertRaises(ValueError):self.prepare({'product_context':self.context.to_dict(),'status':'PASS'})
        forged=object.__new__(self.upstream.AuthenticatedShapeInput)
        with self.assertRaises(ValueError):self.prepare(forged)
        cloned=copy.copy(authority)
        with self.assertRaises(ValueError):self.prepare(cloned)

    def test_cancelled_chain_cannot_admit_geometry(self):
        authority=self.complete();self.event.set()
        with self.assertRaises((ValueError,InterruptedError)):self.prepare(authority)

    def test_coordinate_change_after_host_completion_rejected(self):
        authority=self.complete();path=self.bundle['inputs']['coords.npy']['path']
        blob=path.read_bytes();path.write_bytes(blob[:-1]+bytes([blob[-1]^1]))
        with self.assertRaises(ValueError):self.prepare(authority)

    def test_private_image_or_contract_change_after_completion_rejected(self):
        authority=self.complete();path=self.occ/'source-image.png';path.write_bytes(self.image+b'changed')
        with self.assertRaises(ValueError):self.prepare(authority)

    def test_foreign_context_or_wrong_completed_job_hash_rejected(self):
        from dataclasses import replace
        authority=self.complete()
        with self.assertRaises(ValueError):self.prepare(authority,context=replace(self.context,nonce='foreign'))
        with self.assertRaises(ValueError):self.prepare(authority,expected_source_job_sha256='0'*64)

    def test_host_sidecar_never_hides_a_conflicting_worker_context(self):
        raw=self.bundle['source_job']
        document=json.loads(self.payloads[raw]);document['product_context']=dict(self.context.to_dict(),job_id='foreign-job')
        self.payloads[raw]=json.dumps(document).encode()
        authority=self.complete()
        with self.assertRaisesRegex(ValueError,'request/job context'):self.prepare(authority)


if __name__=='__main__':unittest.main()
