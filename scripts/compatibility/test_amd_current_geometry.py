"""CPU-only product bridge tests. Labelled synthetic predecessor receipts.

The inherited shape/occupancy verifier is replaced only in bridge wiring tests;
real bounded files, request validation, freshness and payload readers are used.
This does not claim a new genuine shape sampling job has completed.
"""
import hashlib
import importlib.abc
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'api'), str(Path(__file__).parent)]


class NoAI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','triton','flex_gemm','trellis2','transformers'}:
            raise AssertionError('CPU contract route imported AI: '+fullname)


_import_guard=NoAI();sys.meta_path.insert(0,_import_guard)
try:
    import amd_generation_decode as host
    from services.amd_completed_result import CompletionContext
finally:sys.meta_path.remove(_import_guard)


class CurrentGeometryBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard=NoAI();sys.meta_path.insert(0,cls.guard)

    @classmethod
    def tearDownClass(cls):sys.meta_path.remove(cls.guard)

    def setUp(self):
        import numpy as np
        self.np = np
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-current-',
            dir=ROOT/'docs/compatibility/amd-current-image-geometry-evidence')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        # Standard-library PNG fixture; no image package or environment change.
        import struct, zlib
        def chunk(kind,payload):
            return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload))
        pixels=b''.join(b'\0'+b''.join(bytes((120,90,50,255)) if 8<=x<24 and 8<=y<24
                       else b'\0\0\0\0' for x in range(32)) for y in range(32))
        self.image=(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>2I5B',32,32,8,6,0,0,0))
                    +chunk(b'IDAT',zlib.compress(pixels))+chunk(b'IEND',b''))
        self.params={'resolution':512}
        self.context=CompletionContext.new('synthetic-current-job',self.image,self.params,model_id='trellis2-amd')
        self.shape=self.root/'shape'; (self.shape/'run/scratch').mkdir(parents=True)
        self.occ=self.root/'occupancy'; self.occ.mkdir()
        self.features=np.arange(3*32,dtype=np.float32).reshape(3,32)/np.float32(128)
        self.coords=np.array([[0,2,1,0],[0,1,3,2],[0,0,0,1]],np.int32)
        inputs={}
        for name,array in [('shape-latent-denormalized.npy',self.features),('coords.npy',self.coords)]:
            leaf='shape-latent-coordinates.npy' if name=='coords.npy' else name
            path=self.shape/'run/scratch'/leaf
            with path.open('xb') as f: np.save(f,array,allow_pickle=False)
            inputs[name]={'path':path,'sha256':host.digest(path)}
        image_contract={'source_sha256':hashlib.sha256(self.image).hexdigest()}
        (self.occ/'input-contract.json').write_text(json.dumps(image_contract),encoding='utf-8')
        self.input_hash=host.digest(self.occ/'input-contract.json')
        (self.shape/'pre-execution-admission.json').write_text(json.dumps({'context':{'occupancy_evidence':str(self.occ)}}))
        source=self.shape/'run/results.json'
        source.write_text(json.dumps({'product_context':self.context.to_dict(),
            'worker':{'generation_inputs':{'input_contract_sha256':self.input_hash}}}))
        self.bundle={'source_job':source,'source_job_sha256':host.digest(source),'count':3,
                     'inputs':inputs,'coords':self.coords,'required_files':[]}
        self.patches=[patch.object(host,'APPROVED_ROOT',self.root),
                      patch.object(host,'shape_inputs',return_value=self.bundle),
                      patch('amd_generation_occupancy.verify_sequence',return_value={'input_contract':image_contract,
                            'input_contract_sha256':self.input_hash}),
                      patch('subprocess.check_output',return_value='1'*40+'\n'),
                      patch('amd_sparse_decoder_forward.verify_committed_files',return_value={})]
        for p in self.patches: p.start(); self.addCleanup(p.stop)

    def prepare(self,**overrides):
        args=dict(context=self.context,image_bytes=self.image,params=self.params,
                  shape_evidence=self.shape,expected_source_job_sha256=self.bundle['source_job_sha256'],
                  harness_commit='1'*40,run_id='run-current-synthetic')
        args.update(overrides)
        return host.prepare_current_input(**args)

    def test_actual_files_bind_distinct_current_request_and_payloads(self):
        contract=self.prepare()
        self.assertEqual(contract.initial_count,3)
        self.assertIsNone(contract.final_support_profile)
        self.assertTrue(contract.large_convolution_tiling)
        binding=contract.document()['current_input']
        self.assertEqual(binding['product_context'],self.context.to_dict())
        self.assertEqual(binding['initial_coords_payload_sha256'],hashlib.sha256(self.coords.tobytes()).hexdigest())
        self.assertEqual(binding['initial_features_payload_sha256'],hashlib.sha256(self.features.tobytes()).hexdigest())
        self.assertNotIn('brick-1957891-v1',json.dumps(contract.document()))
        self.assertFalse((self.root/'decode').exists())

    def test_wrong_actual_image_or_params_rejected(self):
        for overrides in ({'image_bytes':self.image+b'x'}, {'params':{'resolution':256}}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError): self.prepare(**overrides)

    def test_same_image_foreign_job_nonce_and_completed_job_pin_rejected(self):
        from dataclasses import replace
        for ctx in (replace(self.context,job_id='other-job'),replace(self.context,nonce='other-nonce'),
                    replace(self.context,request_id='other-request')):
            with self.subTest(context=ctx),self.assertRaisesRegex(ValueError,'request/job'):self.prepare(context=ctx)
        with self.assertRaisesRegex(ValueError,'Foreign source'):self.prepare(expected_source_job_sha256='0'*64)

    def test_occupancy_image_or_contract_lineage_rejected(self):
        with patch('amd_generation_occupancy.verify_sequence',return_value={
                'input_contract':{'source_sha256':'0'*64},'input_contract_sha256':self.input_hash}):
            with self.assertRaisesRegex(ValueError,'image'): self.prepare()
        with patch('amd_generation_occupancy.verify_sequence',return_value={
                'input_contract':{'source_sha256':self.context.input_sha256},'input_contract_sha256':'0'*64}):
            with self.assertRaises(ValueError): self.prepare()

    def test_stale_source_job_and_stale_input_rejected(self):
        import os
        for path in (self.bundle['source_job'], self.bundle['inputs']['coords.npy']['path']):
            info=path.stat(); os.utime(path,ns=(info.st_atime_ns,self.context.created_ns-1_000_000))
            with self.assertRaisesRegex(ValueError,'[Ss]tale'): self.prepare()
            os.utime(path,ns=(info.st_atime_ns,time.time_ns()))

    def test_changed_source_or_foreign_coordinate_receipt_rejected(self):
        source=self.bundle['source_job']; original=source.read_bytes(); source.write_bytes(original+b' ')
        with self.assertRaises(ValueError): self.prepare()
        source.write_bytes(original)
        path=self.bundle['inputs']['coords.npy']['path']; raw=path.read_bytes()
        path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        with self.assertRaises(ValueError): self.prepare()

    def test_replay_mode_and_foreign_model_rejected(self):
        from dataclasses import replace
        for ctx in (replace(self.context,mode='replay_fixture'), replace(self.context,model_id='other')):
            with self.assertRaises(ValueError): self.prepare(context=ctx)

    def test_genuine_predecessor_verifier_is_mandatory(self):
        with patch.object(host,'shape_inputs',side_effect=ValueError('real lineage failed')) as verifier:
            with self.assertRaisesRegex(ValueError,'real lineage failed'): self.prepare()
            verifier.assert_called_once_with(self.shape)

    def test_inherited_routing_and_environment_restored_on_success_and_failure(self):
        import os
        import amd_generation_occupancy as occupancy
        original=occupancy.original
        keys=('EVIDENCE','CONTRACT','SOURCE','SOURCE_SHA','selected_image','prepare_profile','preflight')
        before={key:getattr(original,key) for key in keys}
        contract_sha=occupancy._contract_sha;preview=original.occupancy.data.write_preview
        env={key:os.environ.get(key) for key in (occupancy.ENV_IMAGE,occupancy.ENV_EVIDENCE,
            occupancy.ENV_SOURCE_SHA,occupancy.ENV_CONTRACT_SHA,'MODLY_OBJECT_HARNESS_COMMIT')}
        def change(fail=False):
            for key in keys:setattr(original,key,object())
            occupancy._contract_sha='temporary';original.occupancy.data.write_preview=lambda:None
            os.environ.update({key:'temporary' for key in env})
            if fail:raise ValueError('inherited failure')
            return self.bundle
        for fail in (False,True):
            with patch.object(host,'shape_inputs',side_effect=lambda path:change(fail)):
                if fail:
                    with self.assertRaisesRegex(ValueError,'inherited failure'):self.prepare()
                else:self.prepare()
            self.assertEqual({key:getattr(original,key) for key in keys},before)
            self.assertEqual(occupancy._contract_sha,contract_sha)
            self.assertIs(original.occupancy.data.write_preview,preview)
            self.assertEqual({key:os.environ.get(key) for key in env},env)

    def test_foreign_source_path_or_unpublished_sources_rejected(self):
        self.bundle['inputs']['coords.npy']['path']=self.root/'foreign.npy'
        with self.assertRaises((ValueError,FileNotFoundError)): self.prepare()
        self.bundle['inputs']['coords.npy']['path']=self.shape/'run/scratch/shape-latent-coordinates.npy'
        with patch('subprocess.check_output',return_value='2'*40+'\n'):
            with self.assertRaisesRegex(ValueError,'HEAD'): self.prepare()
        with patch('amd_sparse_decoder_forward.verify_committed_files',side_effect=RuntimeError('uncommitted')):
            with self.assertRaisesRegex(RuntimeError,'uncommitted'): self.prepare()


if __name__=='__main__': unittest.main()
