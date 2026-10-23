"""Labelled tiny CPU tensors exercise actual current-output writer/readers.

No saved neural output is claimed; no model, native binding or GPU runs.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path[:0]=[str(HERE),str(ROOT/'api'),str(ROOT/'scripts/compatibility')]
import run_contract as rc
import artifact_io
from test_current_input import base_document, context_document


class CurrentArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        cls.np=np;cls.torch=torch
        assert not torch.cuda.is_initialized()
        torch.set_num_threads(1)
        cls.guards=[patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU entry forbidden')),
                    patch.object(torch,'load',side_effect=AssertionError('Model load forbidden')),
                    patch.object(torch,'compile',side_effect=AssertionError('Compiler forbidden'))]
        for guard in cls.guards:guard.start()

    @classmethod
    def tearDownClass(cls):
        for guard in cls.guards:guard.stop()
        assert not cls.torch.cuda.is_initialized()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='synthetic-output-',
            dir=ROOT/'docs/compatibility/amd-current-image-geometry-evidence')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        np=self.np;torch=self.torch
        self.initial=np.array([[0,1,2,3],[0,4,5,6]],np.int32)
        self.features=np.arange(64,dtype=np.float32).reshape(2,32)/np.float32(128)
        self.current=rc._current_api()
        self.contract=self.current.create(base_document(),product_context=context_document(),
            image_input_contract_sha256='f'*64,
            initial_features_payload_sha256=hashlib.sha256(self.features.tobytes()).hexdigest(),
            initial_coords_payload_sha256=hashlib.sha256(self.initial.tobytes()).hexdigest())
        self.tensors={};parents=self.initial.copy()
        for i in range(4):
            logits=np.full((len(parents),8),-1,np.float32);logits[:,0]=1
            if i in (0,2):logits[0,7]=1
            self.tensors[f'subdivision_{i}_coordinates']=torch.from_numpy(parents.copy())
            self.tensors[f'subdivision_{i}_logits']=torch.from_numpy(logits).to(torch.bfloat16)
            parents=rc.reconstruct_children(parents,logits,contract=self.contract)
        self.tensors['coordinates']=torch.from_numpy(parents)
        self.tensors['features']=torch.arange(len(parents)*7,dtype=torch.float32).reshape(-1,7)/128
        import neural
        self.summary=neural.validate_output(self.tensors,torch,2,self.initial,contract=self.contract)
        runtime=dict(self.current.IDENTITY_POLICY['runtime']);runtime.pop('python_prefix')
        runtime.update(python='3.12.3',gpu_initialized_before=False)
        record={'source_sha256':self.current._source_identity(),
            'checkpoint':{'freshly_streamed_sha256':self.current.IDENTITY_POLICY['checkpoint_sha256']},
            'saved_inputs':{},'runtime':runtime}
        self.provenance=neural.output_provenance(record,self.contract,len(parents),1,[])
        self.hashes={k:hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest() for k,t in self.tensors.items()}

    def save(self,tensors=None,metadata=None):
        from safetensors.torch import save_file
        path=self.root/'fixture.safetensors'
        save_file(self.tensors if tensors is None else tensors,str(path),
                  metadata=artifact_io.output_metadata(self.provenance,self.summary,self.contract) if metadata is None else metadata)
        return path

    def test_real_writer_and_both_independent_readers(self):
        with patch('artifact_paths.attest_output',return_value=self.root):
            result=artifact_io.write_output(self.tensors,self.torch,self.provenance,self.summary,self.contract)
        self.assertTrue(result['verified']);self.assertTrue(result['readback_verified_before_release'])
        self.assertEqual(result['rows'],4)
        import amd_generation_mesh as reader
        arrays,counts=reader.read_neural(self.root/artifact_io.FILENAME,result['sha256'],contract=self.contract)
        try:
            self.assertEqual(counts,[3,3,4,4])
            for name,array in arrays.items():
                reference=self.tensors[name]
                expected=reference.view(self.torch.uint16).numpy() if reference.dtype==self.torch.bfloat16 else reference.numpy()
                self.np.testing.assert_array_equal(array,expected)
        finally:
            for array in arrays.values():array._mmap.close()
        self.assertEqual(json.loads((self.root/'provenance.json').read_text())['run_contract'],self.contract.document())

    def test_provenance_and_product_metadata_are_bound(self):
        path=self.save()
        self.assertEqual(self.provenance['product_context'],context_document())
        artifact_io.verify_file(path,self.provenance,self.summary,self.hashes,self.contract)
        wrong=copy.deepcopy(self.provenance);wrong['product_context']['nonce']='foreign'
        with self.assertRaises(ValueError): artifact_io.verify_file(path,wrong,self.summary,self.hashes,self.contract)
        metadata=artifact_io.output_metadata(self.provenance,self.summary,self.contract)
        metadata['product_context']=json.dumps(dict(context_document(),job_id='other'))
        path.unlink();path=self.save(metadata=metadata)
        with self.assertRaisesRegex(ValueError,'metadata'):artifact_io.verify_file(path,self.provenance,self.summary,self.hashes,self.contract)

    def test_output_is_readable_by_existing_product_stage_contract(self):
        from services.amd_completed_result import CompletionContext
        from services.amd_stage_contract import Artifact,validate_artifact
        path=self.save()
        tensors={name:{'dtype':dtype,'shape':shape} for name,(dtype,shape)
                 in artifact_io.expected_schema(self.summary,self.contract).items()}
        descriptor=Artifact(path.name,rc.digest(path),path.stat().st_size,'safetensors',
            {'tensors':tensors,'max_rows':self.contract.maximum_final_rows,'coordinate_tensor':'coordinates'})
        result=validate_artifact(self.root,descriptor,CompletionContext.from_dict(context_document()),
                                 expected_coordinates_sha256=self.hashes['coordinates'])
        self.assertEqual(result['coordinates_sha256'],self.hashes['coordinates'])
        self.assertEqual(set(result['tensors']),set(self.tensors))

    def test_intermediate_oversize_rejected_before_c2s_allocation(self):
        from types import SimpleNamespace
        import admission
        events=[]
        facade=SimpleNamespace(bool='bool',int32='int32',equal=lambda a,b:a is b)
        hooks=admission.AdmissionHooks(None,facade,contract=self.contract,tile_large_convolutions=True,
            emit=events.append,memory=lambda:dict(allocated_bytes=0,reserved_bytes=0,free_bytes=13*admission.GIB))
        hooks.sequence=[('next',None,None,'conv2',2)];hooks.up_states={2:{'completed':0}}
        coords=SimpleNamespace(dtype='int32',device='cpu')
        mask=SimpleNamespace(shape=(200000,8),dtype='bool',device='cpu',
                             sum=lambda:SimpleNamespace(item=lambda:1000001))
        x=SimpleNamespace(feats=SimpleNamespace(shape=(200000,1024),device='cpu'),coords=coords,
                          spatial_shape=(128,128,128),get_spatial_cache=lambda key:None)
        block=SimpleNamespace(out_channels=128,channels=256,_forward=lambda:None)
        module=SimpleNamespace(forward=lambda:None)
        with patch.object(hooks,'_sparse',return_value=(200000,1024)):
            with self.assertRaises(admission.AdmissionStop):
                hooks._up_pre(('blocks.2.8.updown',module,block,2),module,
                              (x,SimpleNamespace(feats=mask,coords=coords)))
        self.assertEqual([event['event'] for event in events],['c2s-count-before-admission'])
        self.assertEqual(events[0]['children'],1000001)

    def test_independent_reader_rejects_consistently_hashed_nonfinite_output(self):
        self.tensors['features'][0,0]=float('nan')
        path=self.save()
        hashes=dict(self.hashes,features=hashlib.sha256(self.tensors['features'].numpy().tobytes()).hexdigest())
        with self.assertRaisesRegex(ValueError,'Finite'):artifact_io.verify_file(path,self.provenance,self.summary,hashes,self.contract)

    def test_reader_rejects_changed_initial_support_even_with_valid_foreign_chain(self):
        from safetensors.torch import save_file
        import amd_generation_mesh as reader
        # Translation at each resolution preserves every child linkage but is
        # still foreign to the current authenticated starting coordinates.
        tensors={k:v.clone() for k,v in self.tensors.items()}
        for i in range(4):tensors[f'subdivision_{i}_coordinates'][:,1]+=2**i
        tensors['coordinates'][:,1]+=16
        path=self.save(tensors=tensors)
        with self.assertRaisesRegex(ValueError,'initial coordinate'):
            reader.read_neural(path,rc.digest(path),contract=self.contract)

    def test_corrupt_chain_and_missing_tensor_are_rejected(self):
        import amd_generation_mesh as reader
        tensors={k:v.clone() for k,v in self.tensors.items()};tensors['coordinates'][0,3]+=1
        path=self.save(tensors=tensors)
        with self.assertRaisesRegex(ValueError,'child coordinates'):reader.read_neural(path,rc.digest(path),contract=self.contract)
        path.unlink();tensors.pop('subdivision_0_logits');path=self.save(tensors=tensors)
        with self.assertRaisesRegex(ValueError,'ten tensors'):reader.read_neural(path,rc.digest(path),contract=self.contract)

    def test_both_execution_entries_deny_current_contract_before_side_effects(self):
        import decode, neural
        with patch('artifact_paths.claim_directory',side_effect=AssertionError('Filesystem claim forbidden')):
            with self.assertRaisesRegex(ValueError,'CPU'):decode.worker(self.contract)
        with self.assertRaisesRegex(ValueError,'CPU'):neural.run_neural(None,None,None,None,None,None,self.contract)
        self.assertEqual(list(self.root.iterdir()),[])

    def test_actual_cli_can_validate_but_cannot_execute_without_ai_import(self):
        path=self.root/'contract.json';path.write_text(json.dumps(self.contract.document()))
        command=[sys.executable,'-B','-I',str(HERE/'decode.py'),'--run-contract',str(path)]
        validated=subprocess.run(command+['--validate-contract'],capture_output=True,text=True,timeout=15)
        self.assertEqual(validated.returncode,0,validated.stderr)
        self.assertFalse(json.loads(validated.stdout)['AI_modules_imported'])
        denied=subprocess.run(command,capture_output=True,text=True,timeout=15)
        self.assertNotEqual(denied.returncode,0);self.assertIn('CPU-qualified only',denied.stderr)


if __name__=='__main__':unittest.main()
