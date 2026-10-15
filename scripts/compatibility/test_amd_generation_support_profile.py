"""CPU-only support-profile consumer tests; every tensor fixture is synthetic."""
import copy
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/'experimental/extensions/trellis2-amd/inference_decoder'
RAW_RECORD=ROOT/'docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/occupancy/row-tiling-resume-43a637d/decode/run/guest/results.json'
METRICS=os.environ.get('MODLY_FINAL_SUPPORT_CPU_METRICS')
sys.path.insert(0,str(CODE))
import run_contract as contracts
import artifact_io
import amd_generation_mesh as mesh


def contract_document():
    return dict(schema='modly.amd.neural-run.v1',initial_count=4430,
        input_paths={'shape-latent-denormalized.npy':'/home/modly/profile-test-input/shape.npy',
                     'coords.npy':'/home/modly/profile-test-input/coords.npy'},
        input_sha256={'shape-latent-denormalized.npy':'e6a6eec09029db43b7b2356ee71a943ad2f225c45d48a03bfaa519a031b8c808',
                      'coords.npy':'17a8ddc94edaa9da5b02834157eef9f8d4f7f99bfa8e2f4d2b5806f1b1cfe93c'},
        source_job_sha256='b92aade7aca2e832fbf425ec9d4e54d40aa7ffa4b4afdea54c38f71f61b8e62f',
        output_directory='/home/modly/profile-test/geometry',scratch_directory='/home/modly/profile-test/scratch',
        repo_root='/mnt/c/test/proof',maximum_final_rows=1957891,maximum_intermediate_rows=1000000,
        native_stage='/home/modly/native-decoder-preparation-6dd94a85',
        large_convolution_tiling=True,final_support_profile='brick-1957891-v1')


def summary():
    return dict(rows=1957891,subdivisions=[{'parents':n} for n in (4430,21391,97313,438684)])


def provenance(contract):
    return dict(run_contract_sha256=contract.sha256,source_job_sha256=contract.source_job_sha256,
                input_file_sha256=dict(contract.input_sha256),final_support_profile=contract.final_support_profile,
                checkpoint_sha256='e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581',
                synthetic_cpu_fixture=True)


class SupportConsumerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='_support-consumer-',dir=CODE)
        self.root=Path(self.temp.name);self.contract=contracts.parse(contract_document())

    def tearDown(self):self.temp.cleanup()

    def test_artifact_schema_accepts_only_authenticated_exact_profile_support(self):
        schema=artifact_io.expected_schema(summary(),contract=self.contract)
        self.assertEqual(schema['features'],('F32',[1957891,7]))
        self.assertEqual(schema['subdivision_3_logits'],('BF16',[438684,8]))
        with self.assertRaises(ValueError):artifact_io.expected_schema(summary())
        bare=replace(self.contract,final_support_profile=None)
        with self.assertRaises(ValueError):artifact_io.expected_schema(summary(),contract=bare)
        for field,value in [('rows',1957892),('subdivisions',[{'parents':n} for n in (4430,21391,97313,438683)])]:
            with self.assertRaises(ValueError):artifact_io.expected_schema({**summary(),field:value},contract=self.contract)

    def test_mesh_profile_contract_cannot_admit_a_different_small_chain(self):
        from test_amd_generation_mesh import fixture
        path=self.root/'synthetic-small.safetensors';sha=fixture(path,varied_logits=True)
        arrays,counts=mesh.read_neural(path,sha)
        try:self.assertEqual(counts,[1,1,1,4])
        finally:
            for array in arrays.values():array._mmap.close()
        with self.assertRaises(ValueError):mesh.read_neural(path,sha,contract=self.contract)

    def test_mesh_loads_exact_contract_code_instead_of_bare_module_binding(self):
        from test_amd_generation_mesh import fixture
        path=self.root/'small.safetensors';sha=fixture(path)
        fake=SimpleNamespace(final_limit=lambda contract=None:999999999,parse=lambda document:None)
        with patch.dict(sys.modules,{'run_contract':fake}),patch.object(mesh,'_CONTRACT_API',None):
            with self.assertRaises(ValueError):mesh.read_neural(path,sha,contract=replace(self.contract,final_support_profile=None))


def synthetic_tensors(torch):
    count=4430;linear=np.arange(count,dtype=np.int32)
    coords=np.column_stack((np.zeros(count,np.int32),linear//1024,linear//32%32,linear%32)).astype(np.int32)
    initial=coords.copy();tensors={}
    for stage,children in enumerate((21391,97313,438684,1957891)):
        tensors[f'subdivision_{stage}_coordinates']=torch.from_numpy(coords)
        # Raw BF16 -2/+2 logits, selected in original parent-major order.
        bits=np.full((len(coords),8),0xc000,dtype=np.uint16);bits.reshape(-1)[:children]=0x4000
        tensors[f'subdivision_{stage}_logits']=torch.from_numpy(bits).view(torch.bfloat16)
        parent=np.arange(children,dtype=np.int64)//8;child=np.arange(children,dtype=np.int64)%8
        following=coords[parent].copy();following[:,1:]*=2
        for axis in range(3):following[:,axis+1]+=((child//2**axis)%2).astype(np.int32)
        coords=following
    tensors['coordinates']=torch.from_numpy(coords)
    tensors['features']=torch.zeros((len(coords),7),dtype=torch.float32)
    return tensors,initial


class FullProfileRoundtripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.started=time.perf_counter();cls.metrics={};cls.checks=[]
        import torch
        cls.torch=torch
        if torch.cuda.is_initialized():raise AssertionError('CPU fixture requires uninitialized GPU')
        cls.previous_threads=torch.get_num_threads();torch.set_num_threads(1)
        def forbidden(*args,**kwargs):raise AssertionError('CPU fixture forbids GPU/model loading/compilation')
        cls.lazy=patch.object(torch.cuda,'_lazy_init',forbidden);cls.lazy.start()
        cls.load=patch.object(torch,'load',forbidden);cls.load.start()
        cls.compile=patch.object(torch,'compile',forbidden);cls.compile.start()
        cls.temp=tempfile.TemporaryDirectory(prefix='_profile-writer-',dir=CODE);cls.root=Path(cls.temp.name)
        cls.addClassCleanup(cls.temp.cleanup)
        cls.addClassCleanup(cls.lazy.stop);cls.addClassCleanup(cls.load.stop);cls.addClassCleanup(cls.compile.stop)
        cls.addClassCleanup(torch.set_num_threads,cls.previous_threads)
        cls.contract=contracts.parse(contract_document())
        if not RAW_RECORD.is_file():raise unittest.SkipTest('Latest genuine local record is unavailable for production metadata qualification')
        raw_record=json.loads(RAW_RECORD.read_text())
        cls.metrics['production_record_sha256']=hashlib.sha256(RAW_RECORD.read_bytes()).hexdigest()
        cls.metrics['production_record_path']=RAW_RECORD.relative_to(ROOT).as_posix()
        began=time.perf_counter()
        tensors,initial=synthetic_tensors(torch)
        cls.metrics['synthetic_tensor_build_seconds']=time.perf_counter()-began
        neural=importlib.import_module('neural')
        cls.provenance=neural.output_provenance(raw_record,cls.contract,1957891,60,raw_record['large_tiled_convolutions'])
        cls.provenance['synthetic_cpu_fixture']=True
        began=time.perf_counter()
        cls.summary=neural.validate_output(tensors,torch,4430,initial,contract=cls.contract)
        cls.metrics['actual_finite_and_chain_validation_seconds']=time.perf_counter()-began
        # Only the Linux-owned path boundary is replaced. Serializer, metadata,
        # finite/chain validation and independent readers are actual consumers.
        with patch('artifact_paths.attest_output',return_value=cls.root):
            began=time.perf_counter()
            cls.written=artifact_io.write_output(tensors,torch,cls.provenance,cls.summary,cls.contract)
            cls.metrics['actual_writer_and_readback_seconds']=time.perf_counter()-began
        cls.path=Path(cls.written['path']);cls.hashes=cls.written['tensor_payload_sha256']
        del tensors

    @classmethod
    def tearDownClass(cls):
        import psutil
        memory=psutil.Process().memory_info()
        with cls.path.open('rb') as stream:header_bytes=int.from_bytes(stream.read(8),'little')
        cls.metrics.update(schema='modly.amd.final-support-consumer-cpu-tests.v1',
            classification='CPU_CONSUMER_PASS' if len(cls.checks)==4 else 'CPU_CHECK_INCOMPLETE',
            fixture='synthetic CPU coordinates/logits/features; no genuine decoder mask recovery',
            production_metadata='actual neural.output_provenance builder with latest genuine raw source/runtime/input record',
            full_fixture_checks=list(cls.checks),payload_bytes=cls.summary['payload_bytes'],header_bytes=header_bytes,
            file_bytes=cls.path.stat().st_size,header_bound_bytes=65536,payload_bound_bytes=256*1024**2,
            total_full_fixture_seconds=time.perf_counter()-cls.started,
            process_working_set_bytes=memory.rss,peak_process_working_set_bytes=getattr(memory,'peak_wset',None),
            gpu_initialized=cls.torch.cuda.is_initialized(),model_weights_loaded=False,native_packages_imported=False,
            mesh_extraction_executed=False,cpu_threads=1,
            command='D:/AI/Trellis2AMD/venv/Scripts/python.exe -B -m unittest test_amd_generation_support_profile',
            cwd='scripts/compatibility',
            observed_red_and_fix={'failure':'Actual write_output fsync on rb handle raised OSError [Errno 9] Bad file descriptor on Windows',
                                  'fix':'Use r+b for the owned freshly written artifact fsync; actual durability/readback remains enabled'},
            source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (CODE/'artifact_io.py',CODE/'run_contract.py',CODE/'neural.py',Path(mesh.__file__),Path(__file__))})
        cls.temp.cleanup();cls.metrics['temporary_fixture_removed']=not cls.root.exists()
        if METRICS:
            with Path(METRICS).open('x',encoding='utf-8') as stream:json.dump(cls.metrics,stream,indent=2,allow_nan=False);stream.write('\n')
        else:
            keys=('classification','payload_bytes','header_bytes','file_bytes','total_full_fixture_seconds',
                  'peak_process_working_set_bytes','gpu_initialized','temporary_fixture_removed')
            print(json.dumps({**{key:cls.metrics[key] for key in keys},
                              'test_source_sha256':cls.metrics['source_sha256'][Path(__file__).relative_to(ROOT).as_posix()]},sort_keys=True))

    def tearDown(self):self.assertFalse(self.torch.cuda.is_initialized())

    def test_full_synthetic_profile_actual_writer_finite_validator_and_both_readers(self):
        self.assertEqual(self.summary['payload_bytes'],104125380)
        self.assertTrue(self.summary['finite']);self.assertTrue(self.written['readback_verified_before_release'])
        with self.path.open('rb') as stream:length=int.from_bytes(stream.read(8),'little')
        self.assertLessEqual(length,65536)
        verified=artifact_io.verify_file(self.path,self.provenance,self.summary,self.hashes,contract=self.contract)
        self.assertTrue(verified['verified']);self.assertEqual(verified['payload_bytes'],104125380)
        arrays,counts=mesh.read_neural(self.path,mesh.digest(self.path),contract=self.contract)
        try:
            self.assertEqual(counts,[21391,97313,438684,1957891])
            self.assertEqual(arrays['features'].shape,(1957891,7))
            self.assertEqual(int(arrays['subdivision_3_logits'][0,0]),0x4000)
        finally:
            for array in arrays.values():array._mmap.close()
        with self.assertRaises(ValueError):artifact_io.verify_file(self.path,self.provenance,self.summary,self.hashes)
        with self.assertRaises(ValueError):mesh.read_neural(self.path,mesh.digest(self.path))
        self.checks.append('actual writer, finite/chain validator and both full-size readers; <=64KiB production header; default rejects expanded rows')

    def test_altered_profile_provenance_and_bare_high_limit_rejected_by_consumers(self):
        for key in ('run_contract_sha256','source_job_sha256','checkpoint_sha256','input_file_sha256','final_support_profile'):
            altered=copy.deepcopy(self.provenance)
            altered[key]=dict(altered[key],**{'coords.npy':'0'*64}) if key=='input_file_sha256' else '0'*64
            with self.subTest(key=key),self.assertRaises(ValueError):
                artifact_io.verify_file(self.path,altered,self.summary,self.hashes,contract=self.contract)
        with self.assertRaises(ValueError):mesh.read_neural(self.path,mesh.digest(self.path),contract=replace(self.contract,final_support_profile=None))
        with self.path.open('r+b') as stream:
            length=int.from_bytes(stream.read(8),'little');original=stream.read(length)
            altered=original.replace(self.provenance['checkpoint_sha256'].encode(),b'0'*64)
            self.assertNotEqual(original,altered)
            stream.seek(8);stream.write(altered);stream.flush()
        try:
            with patch.object(mesh.np,'memmap',side_effect=AssertionError('Unbound profile must fail before mapping')):
                with self.assertRaises(ValueError):mesh.read_neural(self.path,mesh.digest(self.path),contract=self.contract)
        finally:
            with self.path.open('r+b') as stream:stream.seek(8);stream.write(original)
        self.checks.append('altered provenance and unauthenticated raised limit rejected before mapping')

    def test_profile_reader_rejects_changed_chain_and_closes_all_mappings(self):
        with self.path.open('r+b') as stream:
            length=int.from_bytes(stream.read(8),'little');header=json.loads(stream.read(length))
            offset=8+length+header['coordinates']['data_offsets'][0]
            stream.seek(offset);original=stream.read(32);stream.seek(offset);stream.write(original[16:]+original[:16])
        opened=[];actual_memmap=np.memmap
        def track(*args,**kwargs):
            value=actual_memmap(*args,**kwargs);opened.append(value);return value
        try:
            with patch.object(mesh.np,'memmap',side_effect=track):
                with self.assertRaisesRegex(ValueError,'child.*order'):mesh.read_neural(self.path,mesh.digest(self.path),contract=self.contract)
            self.assertEqual(len(opened),10);self.assertTrue(all(array._mmap.closed for array in opened))
        finally:
            with self.path.open('r+b') as stream:stream.seek(offset);stream.write(original)
        self.checks.append('changed C2S coordinate order rejected and all ten memmaps closed')

    def test_nonfinite_features_rejected_by_actual_neural_validator_and_mesh_reader(self):
        from safetensors.torch import load_file
        tensors=load_file(str(self.path),device='cpu');tensors['features'][0,0]=float('nan')
        neural=importlib.import_module('neural')
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):
            neural.validate_output(tensors,self.torch,4430,tensors['subdivision_0_coordinates'].numpy(),contract=self.contract)
        del tensors
        with self.path.open('r+b') as stream:
            length=int.from_bytes(stream.read(8),'little');header=json.loads(stream.read(length))
            offset=8+length+header['features']['data_offsets'][0]
            stream.seek(offset);original=stream.read(4);stream.seek(offset);stream.write(np.array([np.nan],'<f4').tobytes())
        try:
            with self.assertRaisesRegex(ValueError,'Finite'):mesh.read_neural(self.path,mesh.digest(self.path),contract=self.contract)
        finally:
            with self.path.open('r+b') as stream:stream.seek(offset);stream.write(original)
        self.checks.append('NaN rejected by actual neural validator and CPU mesh reader')


if __name__=='__main__':unittest.main(verbosity=2)
