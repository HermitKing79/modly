"""Bounded CPU tests: no model, native GPU package, WSL or GPU initialization."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load(name):
    path = HERE / (name + '.py')
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def contract_document():
    return {'schema': 'modly.amd.neural-run.v1', 'initial_count': 2,
            'input_paths': {'shape-latent-denormalized.npy': '/home/modly/run-test/input/shape.npy',
                            'coords.npy': '/home/modly/run-test/input/coords.npy'},
            'input_sha256': {'shape-latent-denormalized.npy': 'a'*64, 'coords.npy': 'b'*64},
            'source_job_sha256': 'c'*64,
            'output_directory': '/home/modly/run-test/geometry',
            'scratch_directory': '/home/modly/run-test/shape-scratch',
            'repo_root': '/mnt/c/Users/wattz/Documents/ChatGPT/modly/proof',
            'maximum_final_rows': 1842968, 'maximum_intermediate_rows': 1000000,
            'native_stage': '/home/modly/native-decoder-preparation-6dd94a85'}


class RunContractTests(unittest.TestCase):
    def setUp(self):
        self.module = load('run_contract')
        self.assertIsNotNone(self.module, 'per-run input contract is missing')

    def test_accepts_fresh_inputs_without_fixture_hashes(self):
        value = self.module.parse(contract_document())
        self.assertEqual(value.initial_count, 2)
        self.assertEqual(str(value.output_directory), '/home/modly/run-test/geometry')
        self.assertEqual(value.input_sha256['coords.npy'], 'b'*64)

    def test_refuses_envelope_expansion_and_unsafe_directories(self):
        for field, value in [('initial_count', 4431), ('initial_count', True),
                             ('maximum_final_rows', 1842969), ('maximum_intermediate_rows', 1000001),
                             ('native_stage', '/home/modly/unqualified'),
                             ('output_directory', '/mnt/c/run-test/output'),
                             ('scratch_directory', '/home/modly/run-test/geometry'),
                             ('output_directory', '/home/modly/run-test/../other/output')]:
            document = contract_document(); document[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.module.parse(document)

    def test_measured_4430_inputs_reopen_exactly_and_4431_is_rejected(self):
        from dataclasses import replace
        import numpy as np
        import torch
        document=contract_document();document['initial_count']=4430
        contract=self.module.parse(document)
        sys.path.insert(0,str(HERE.parents[3]/'scripts/compatibility'))
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root=Path(directory);paths={'shape-latent-denormalized.npy':root/'shape.npy','coords.npy':root/'coords.npy'}
            features=np.arange(4430*32,dtype=np.float32).reshape(4430,32)/np.float32(4096)
            linear=np.arange(4430,dtype=np.int32)
            coords=np.column_stack((np.zeros(4430,np.int32),linear//1024,linear//32%32,linear%32)).astype(np.int32)
            np.save(paths['shape-latent-denormalized.npy'],features,allow_pickle=False)
            np.save(paths['coords.npy'],coords,allow_pickle=False)
            hashes={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in paths.items()}
            fixture=replace(contract,input_paths=paths,input_sha256=hashes)
            actual,c,summary=self.module.load_inputs(fixture)
            np.testing.assert_array_equal(actual,features);np.testing.assert_array_equal(c,coords)
            self.assertEqual(summary['tokens'],4430)
            runtime=load('runtime');runtime.validate_tensor_contract(torch.from_numpy(actual),torch.from_numpy(c),4430)
            with self.assertRaises(ValueError):self.module.parse(dict(document,initial_count=4431))
            invalid=np.zeros((4431,32),np.float32);np.save(root/'over.npy',invalid,allow_pickle=False)
            with self.assertRaises(ValueError):
                self.module.read_npy(root/'over.npy',hashlib.sha256((root/'over.npy').read_bytes()).hexdigest(),(4431,32),'float32')
            with self.assertRaises(RuntimeError):
                runtime.validate_tensor_contract(torch.from_numpy(invalid),torch.zeros(4431,4,dtype=torch.int32),4431)
        self.assertFalse(torch.cuda.is_initialized())

    def test_optional_host_journal_is_confined_to_fresh_stage_guest_directory(self):
        document=contract_document()
        host=document['repo_root']+'/docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/shape-decode/run/guest'
        document['host_evidence_directory']=host
        value=self.module.parse(document)
        self.assertEqual(str(value.host_evidence_directory),host)
        self.assertEqual(value.document()['host_evidence_directory'],host)
        self.assertIsNone(self.module.parse(contract_document()).host_evidence_directory)
        for path in ('/mnt/c/unrelated/run/guest',host.replace('/new-image/','/old-image/'),
                     host+'/other',host.replace('/run/guest','/run/../guest')):
            with self.subTest(path=path),self.assertRaises(ValueError):
                self.module.parse(dict(document,host_evidence_directory=path))

    def test_npy_reader_verifies_hash_shape_and_trailing_bytes(self):
        import numpy as np
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory)/'shape.npy'
            np.save(path, np.array([[1]*32, [2]*32], np.float32), allow_pickle=False)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            value = self.module.read_npy(path, digest, (2,32), 'float32')
            self.assertEqual(value[1,0], 2)
            with self.assertRaises(ValueError): self.module.read_npy(path, '0'*64, (2,32), 'float32')
            path.write_bytes(path.read_bytes()+b'X')
            with self.assertRaises(ValueError):
                self.module.read_npy(path, hashlib.sha256(path.read_bytes()).hexdigest(), (2,32), 'float32')


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.module = load('admission')
        self.assertIsNotNone(self.module, 'bounded dynamic admission is missing')

    def test_historical_memory_formulas_and_small_dynamic_tail(self):
        a = self.module
        self.assertEqual(a.final_c2s_plan(378891,512,128,1842968,64)['visible_bytes'], 2766585408)
        old = a.chunked_conv_plan(1842968,32768,90243072)
        self.assertEqual(old['visible_bytes'], 3069154208)
        self.assertEqual(old['chunks'], 57)
        small = a.chunked_conv_plan(32769,32768,512**3)
        self.assertEqual((small['chunks'], small['last_chunk_rows']), (2,1))
        self.assertEqual(a.final_c2s_plan(2,512,128,3,64)['children'], 3)

    def test_refuses_expanded_support_and_insufficient_live_memory(self):
        a = self.module
        for values in [(2,512,128,17,64), (1000001,512,128,2,64),
                       (378891,512,128,1842969,64), (2,256,128,3,64)]:
            with self.assertRaises(a.AdmissionStop): a.final_c2s_plan(*values)
        for n in (0, True, 1842969):
            with self.assertRaises(a.AdmissionStop): a.chunked_conv_plan(n,32768,512**3)
        events = []
        hooks = a.AdmissionHooks(None, None, emit=events.append,
            memory=lambda: dict(allocated_bytes=12*a.GIB,reserved_bytes=12*a.GIB,
                                free_bytes=13*a.GIB,total_bytes=16*a.GIB))
        with self.assertRaises(a.AdmissionStop): hooks._admit('test', a.chunked_conv_plan(3,3,512**3))
        self.assertFalse(events[-1]['admitted'])


class ChainTests(unittest.TestCase):
    def test_original_c2s_order_and_all_four_links(self):
        module = load('run_contract')
        self.assertIsNotNone(module, 'four-stage support validation is missing')
        import numpy as np
        parents = np.array([[0,1,0,0],[0,0,0,0]], np.int32)
        logits = np.full((2,8), -1, np.float32)
        logits[0,1] = 1; logits[1,4] = 1
        child = module.reconstruct_children(parents, logits)
        np.testing.assert_array_equal(child, [[0,3,0,0],[0,0,0,1]])
        stages = []
        for i in range(4):
            values = np.full((len(parents),8), -1, np.float32); values[:,0] = 1
            stages.append((parents, values)); parents = module.reconstruct_children(parents, values)
        module.validate_chain(stages, parents, 2)
        with self.assertRaises(ValueError): module.validate_chain(stages, parents[::-1].copy(), 2)

    def test_output_preserves_nonbinary_bf16_logits_and_initial_coordinate_identity(self):
        module=load('neural'); self.assertIsNotNone(module)
        import torch
        coords=torch.tensor([[0,1,0,0],[0,0,0,0]],dtype=torch.int32)
        original=coords.clone();tensors={}
        for i in range(4):
            logits=torch.full((2,8),-183.,dtype=torch.bfloat16);logits[:,0]=338.
            tensors[f'subdivision_{i}_coordinates']=coords.clone()
            tensors[f'subdivision_{i}_logits']=logits
            coords=coords.clone();coords[:,1:]*=2
        tensors.update(coordinates=coords,features=torch.zeros(2,7))
        result=module.validate_output(tensors,torch,2,original.numpy())
        self.assertEqual([s['selected_children'] for s in result['subdivisions']],[2]*4)
        self.assertEqual(tensors['subdivision_0_logits'][0,0].item(),338.)
        with self.assertRaises(ValueError):module.validate_output(tensors,torch,2,original.flip(0).numpy())


class ChunkArithmeticTests(unittest.TestCase):
    def test_source_equivalent_convolution_preserves_cross_tile_global_indices(self):
        module = load('chunked_final')
        self.assertIsNotNone(module, 'source-faithful dynamic final convolution is missing')
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        feats = torch.tensor([[1.,2.],[3.,4.],[5.,6.]], dtype=torch.float32)
        neighbor = torch.full((3,27), 0xffffffff, dtype=torch.uint32)
        neighbor[:,13] = torch.tensor([2,0,1], dtype=torch.uint32)
        weight = torch.zeros((1,3,3,3,2), dtype=torch.float32)
        weight[0,1,1,1] = torch.tensor([2.,-1.])
        for tile in (1,2,3):
            actual = module.row_chunked_explicit(torch, feats, neighbor, weight, None, tile, lambda *args:None)
            torch.testing.assert_close(actual, torch.tensor([[4.],[0.],[2.]]), rtol=0, atol=0)
        self.assertFalse(torch.cuda.is_initialized())


class ArtifactTests(unittest.TestCase):
    def test_independent_reader_rejects_payload_corruption_and_foreign_schema(self):
        module = load('artifact_io')
        self.assertIsNotNone(module, 'dynamic output readback is missing')
        import torch
        from safetensors.torch import save_file
        tensors = {'features':torch.zeros(2,7), 'coordinates':torch.zeros(2,4,dtype=torch.int32)}
        for i in range(4):
            tensors[f'subdivision_{i}_coordinates'] = torch.zeros(2,4,dtype=torch.int32)
            tensors[f'subdivision_{i}_logits'] = torch.ones(2,8,dtype=torch.bfloat16)
        provenance = {'run_contract_sha256':'a'*64,'source_job_sha256':'b'*64}
        summary = {'rows':2, 'subdivisions':[{'parents':2} for i in range(4)]}
        hashes = {k:hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest() for k,t in tensors.items()}
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory)/'shape.safetensors'
            save_file(tensors,str(path),metadata={'provenance':json.dumps(provenance,sort_keys=True),
                                                'summary':json.dumps(summary,sort_keys=True)})
            result = module.verify_file(path,provenance,summary,hashes)
            self.assertTrue(result['verified'])
            with self.assertRaises(ValueError):
                module.verify_file(path,provenance,dict(summary,rows=3),hashes)
            blob=bytearray(path.read_bytes()); blob[-1]^=1; path.write_bytes(blob)
            with self.assertRaises(ValueError): module.verify_file(path,provenance,summary,hashes)


class EntryTests(unittest.TestCase):
    def test_entry_import_and_contract_validation_do_not_import_ai(self):
        self.assertTrue((HERE/'decode.py').exists(), 'factored guest entry is missing')
        import subprocess
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path=Path(directory)/'contract.json'; path.write_text(json.dumps(contract_document()))
            result=subprocess.run([sys.executable,'-B','-I',str(HERE/'decode.py'),
                                   '--run-contract',str(path),'--validate-contract'],
                                  capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertFalse(json.loads(result.stdout)['AI_modules_imported'])

    def test_acceptance_requires_zero_release_and_current_run_identity(self):
        module=load('decode');self.assertIsNotNone(module)
        contract=load('run_contract').parse(contract_document())
        row=dict(decoder_cpu='PASS',native_import='PASS',forward_calls=1,convolutions_completed=40,
                 final_convolution_chunks=1,final_convolution_rows=2,native_map_calls=5,
                 final_convolution_patch_restored=True,subdivision_operations=[{}]*8,
                 output=dict(rows=2,verified=True,readback_verified_before_release=True),
                 CPU_model_collected=True,CPU_parameters_collected=True,CPU_component_collected=True,
                 memory_release=dict(allocated=0,reserved=0),workspace_clear_calls=1,
                 runtime_dtype_counts=dict(BF16=216,F32=76),constructed_on_meta=True,buffers_count=0,
                 decoder_diagnostics=dict(strict=True,roundtrip_verified=True,entries=292,
                                          dtype_counts=dict(F16=216,F32=76),missing_keys=[],unexpected_keys=[]),
                 policy_final=dict(rejected_calls=0),run_contract_sha256=contract.sha256,
                 source_job_sha256=contract.source_job_sha256)
        module.validate_qualified(row,contract)
        self.assertTrue(hasattr(module,'final_acceptance'),'post-cleanup acceptance is missing')
        self.assertEqual(module.final_acceptance(row,contract),0)
        self.assertEqual(row['classification'],'QUALIFIED')
        for key,value in [('cleanup_error','late synchronization failed'),
                          ('memory_release',dict(allocated=1,reserved=0)),('CPU_parameters_collected',False)]:
            bad=dict(row,**{key:value})
            self.assertEqual(module.final_acceptance(bad,contract),1)
            self.assertNotEqual(bad['classification'],'QUALIFIED')
        for key,value in [('memory_release',dict(allocated=1,reserved=0)),('source_job_sha256','0'*64),
                          ('final_convolution_chunks',57),('convolutions_completed',39)]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                module.validate_qualified(dict(row,**{key:value}),contract)


class JournalTests(unittest.TestCase):
    def test_identical_durable_stage_events_and_only_one_host_file_allowed(self):
        module=load('stage_journal');self.assertIsNotNone(module,'host stage bridge is missing')
        base=load('decode').base
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root=Path(directory);output=root/'linux';host=root/'host'
            output.mkdir();host.mkdir()
            journal=module.StageJournal(base,output,host)
            journal(output,'native-import-entry',gpu={'free_bytes':13*1024**3})
            journal(output,'gpu-stage-entry',memory={'allocated':0,'reserved':0})
            self.assertEqual((output/'phases.jsonl').read_bytes(),(host/'phases.jsonl').read_bytes())
            self.assertEqual([json.loads(s)['phase'] for s in (host/'phases.jsonl').read_text().splitlines()],
                             ['native-import-entry','gpu-stage-entry'])
            self.assertEqual([p.name for p in host.iterdir()],['phases.jsonl'])
            journal.audit('open',(str(host/'phases.jsonl'),'a',0))
            with self.assertRaises(RuntimeError):journal.audit('open',(str(host/'phases.jsonl'),'w',0))
            with self.assertRaises(RuntimeError):journal.audit('open',(str(host/'model.safetensors'),'w',0))
            with self.assertRaises(RuntimeError):journal.audit('os.mkdir',(str(host/'cache'),0o700,-1))

    def test_bridge_refuses_missing_or_preexisting_shared_leaf_contents(self):
        module=load('stage_journal');self.assertIsNotNone(module)
        base=load('decode').base
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root=Path(directory);output=root/'linux';output.mkdir()
            with self.assertRaises((ValueError,FileNotFoundError)):
                module.StageJournal(base,output,root/'missing')
            host=root/'host';host.mkdir();(host/'phases.jsonl').write_text('stale')
            with self.assertRaises((ValueError,FileExistsError)):
                module.StageJournal(base,output,host)


if __name__ == '__main__':
    unittest.main(verbosity=2)
