"""Stdlib-only synthetic worker control tests; no WSL/Torch/model/native import."""
import hashlib
import importlib.util
import io
import json
import marshal
from contextlib import contextmanager
from pathlib import Path
import shutil
import sys
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref
import uuid

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'docs/compatibility/amd-brick-pbr-postrestart-evidence/worker.py'
sys.dont_write_bytecode = True


def forbidden(event, args):
    if event == 'import' and str(args[0]).split('.')[0] in {
            'torch', 'triton', 'trellis2', 'flex_gemm', 'o_voxel', 'cumesh'}:
        raise RuntimeError('Synthetic worker tests forbid AI/native/model imports')
    if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system'):
        raise RuntimeError('Synthetic worker tests forbid network and child execution')


sys.addaudithook(forbidden)


@contextmanager
def fixture_directory():
    parent = (ROOT/'.tmp/brick-pbr-worker-fixtures').resolve()
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent/uuid.uuid4().hex
    directory.mkdir()
    try:
        yield directory
    finally:
        if directory.parent != parent or directory.resolve(strict=True) != directory:
            raise ValueError('CPU fixture cleanup escaped the exact workspace parent')
        shutil.rmtree(directory)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SOURCE.is_file(), 'experiment worker adapter is missing')
        spec = importlib.util.spec_from_file_location('_brick_worker_cpu_test', SOURCE)
        self.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.worker)

    def test_manifest_authenticates_payload_and_refuses_missing_or_escaped_source(self):
        # Break caught: accepting changed source bytes, omitted closure or traversal.
        with fixture_directory() as directory:
            root = Path(directory).resolve()
            (root/'source.py').write_bytes(b'qualified source\n')
            manifest = root/'manifest.json'
            payload = {'source.py': hashlib.sha256(b'qualified source\n').hexdigest()}
            manifest.write_text(json.dumps(payload), encoding='utf-8')
            sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            self.assertEqual(self.worker.verify_manifest(manifest, sha, root=root, required={'source.py'}), payload)
            (root/'source.py').write_bytes(b'changed source\n')
            with self.assertRaisesRegex(ValueError, 'identity'):
                self.worker.verify_manifest(manifest, sha, root=root, required={'source.py'})
            with self.assertRaisesRegex(ValueError, 'closure'):
                self.worker.verify_manifest(manifest, sha, root=root, required={'other.py'})
            manifest.write_text(json.dumps({'../source.py': 'a'*64}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'canonical'):
                self.worker.verify_manifest(manifest, hashlib.sha256(manifest.read_bytes()).hexdigest(), root=root, required=set())

    def test_stage_ack_requires_exact_identity_and_fresh_host_admission(self):
        # Break caught: carrying a stale/foreign stage GO or accepting a floor failure.
        ready = dict(schema='modly.amd.brick-pbr-stage-ready.v1', stage='cpu',
                     marker_sha256='a'*64, attestation_sha256='b'*64, profile_sha256='c'*64,
                     run_contract_sha256='d'*64, plan_sha256='e'*64)
        ack = dict(ready, schema='modly.amd.brick-pbr-stage-go.v1', status='GO',
                   healthy_observation=True,
                   host_admission=dict(admitted=True, available_ram_bytes=20*1024**3, D_free_bytes=60*1024**3),
                   reserve_assessment=dict(admitted=True,source='closest-historical-whole-host-reference-not-bound',
                       remaining_reference_bytes=1143431168,remaining_after_reference_bytes=20331405312))
        self.worker.validate_stage_go(ack, ready)
        for change in ({'stage': 'native'}, {'marker_sha256': 'f'*64}, {'healthy_observation': False},
                       {'reserve_assessment': {'admitted': False, 'source': 'insufficient headroom'}},
                       {'reserve_assessment': dict(ack['reserve_assessment'],remaining_reference_bytes=1)},
                       {'reserve_assessment': dict(ack['reserve_assessment'],remaining_after_reference_bytes=12*1024**3-1)},
                       {'host_admission': dict(admitted=True, available_ram_bytes=12*1024**3-1, D_free_bytes=60*1024**3)}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.worker.validate_stage_go(dict(ack, **change), ready)

    def test_resource_release_collects_owners_and_clears_workspace_once(self):
        # Break caught: retaining owned model/parameter/component or duplicate workspace clear.
        calls = []
        cuda = SimpleNamespace(is_initialized=lambda: True, synchronize=lambda *_: calls.append('sync'),
            empty_cache=lambda: calls.append('empty'), memory_allocated=lambda *_: 0,
            memory_reserved=lambda *_: 0, mem_get_info=lambda *_: (14*1024**3,16*1024**3),
            max_memory_allocated=lambda *_: 123, max_memory_reserved=lambda *_: 456)
        torch = SimpleNamespace(cuda=cuda, _C=SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: calls.append('clear')))
        class Owner:
            pass
        class Component:
            def __init__(self, model):
                self.model = model
            def unload(self):
                self.model = None
        state = self.worker.WorkerResources()
        state.torch = torch
        state.gpu_started = 1.0
        state.model = Owner()
        state.model.parameter = Owner()
        state.component = Component(state.model)
        state.model_ref = weakref.ref(state.model)
        state.parameter_refs = [weakref.ref(state.model.parameter)]
        state.component_ref = weakref.ref(state.component)
        record = {}
        state.release(record)
        state.release(record)
        self.assertEqual(calls.count('clear'), 1)
        self.assertEqual(record['workspace_clear_calls'], 1)
        self.assertEqual(record['memory_release'], {'allocated': 0, 'reserved': 0})
        self.assertTrue(all(record[key] for key in ('CPU_model_collected','CPU_parameters_collected','CPU_component_collected')))

    def test_missing_binary_go_records_failure_without_admitted_work(self):
        # Break caught: executing bootstrap despite EOF or malformed binary token.
        with fixture_directory() as directory:
            output = Path(directory)/'guest'
            scratch = directory/'unused-scratch'
            original_control = self.worker.control
            def isolated_control():
                self.assertEqual(sys.pycache_prefix,str(scratch/'forbidden-bytecode'))
                self.assertTrue(sys.dont_write_bytecode)
                self.assertFalse(scratch.exists())
                return original_control()
            with patch.object(self.worker,'validate_output',return_value=output), \
                 patch.object(self.worker,'SCRATCH',scratch), \
                 patch.object(self.worker,'control',side_effect=isolated_control), \
                 patch.object(sys,'pycache_prefix',None), \
                 patch.object(self.worker, 'authenticate_plan', return_value={'candidate_identities': {}}), \
                 patch.object(self.worker, 'execute_admitted', side_effect=AssertionError('must stay closed')), \
                 patch.object(self.worker.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(b'GO\r\n'))):
                code = self.worker.worker(output, Path(directory)/'plan.json', 'a'*64)
            result = json.loads((output/'results.json').read_text())
            self.assertEqual(code, 1)
            self.assertFalse(result['gate_authorized'])
            self.assertEqual(result['native_import'], 'NOT_ATTEMPTED')
            self.assertEqual(result['decoder_cpu'], 'NOT_ATTEMPTED')
            self.assertEqual(result['forward_calls'], 0)
            self.assertIn('Host authorization', result['error'])
            self.assertFalse(scratch.exists())

    def test_worker_uses_one_approved_execution_and_natural_result(self):
        # Break caught: missing GO state, repeat admitted execution, or wrong result schema.
        with fixture_directory() as directory:
            output = Path(directory)/'guest'
            def execute(plan, record, resources, output):
                self.assertTrue(record['gate_authorized'])
                record['classification'] = 'QUALIFIED'
                record['forward_calls'] = 1
            with patch.object(self.worker,'validate_output',return_value=output), \
                 patch.object(sys,'pycache_prefix',None), \
                 patch.object(self.worker, 'authenticate_plan', return_value={'candidate_identities': {}}), \
                 patch.object(self.worker, 'execute_admitted', side_effect=execute) as admitted, \
                 patch.object(self.worker.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(b'GO\n'))):
                code = self.worker.worker(output, Path(directory)/'plan.json', 'a'*64)
            result = json.loads((output/'results.json').read_text())
            self.assertEqual(code, 0)
            self.assertEqual(admitted.call_count, 1)
            self.assertEqual(result['classification'], 'QUALIFIED')
            self.assertEqual(result['forward_calls'], 1)
            self.assertEqual(result['execution_plan_sha256'], 'a'*64)

    def test_failure_after_gpu_entry_releases_owned_references_and_keeps_failure(self):
        # Break caught: traceback/cleanup retaining model owners or promoting failure.
        calls = []
        cuda = SimpleNamespace(is_initialized=lambda: True, synchronize=lambda *_: None,
            empty_cache=lambda: None, memory_allocated=lambda *_: 0, memory_reserved=lambda *_: 0,
            mem_get_info=lambda *_: (14*1024**3,16*1024**3),
            max_memory_allocated=lambda *_: 0,max_memory_reserved=lambda *_: 0)
        torch = SimpleNamespace(cuda=cuda,_C=SimpleNamespace(_cuda_clearCublasWorkspaces=lambda:calls.append('clear')))
        class Model:
            pass
        class Component:
            def __init__(self, model):
                self.model = model
            def unload(self):
                self.model = None
        def execute(plan, record, resources, output):
            resources.torch = torch
            resources.gpu_started = 1.0
            resources.model = Model()
            resources.component = Component(resources.model)
            resources.model_ref = weakref.ref(resources.model)
            resources.component_ref = weakref.ref(resources.component)
            record['forward_calls'] = 1
            raise RuntimeError('synthetic learned-forward boundary refusal')
        with fixture_directory() as directory:
            with patch.object(self.worker,'validate_output',return_value=directory/'guest'), \
                 patch.object(sys,'pycache_prefix',None), \
                 patch.object(self.worker,'authenticate_plan',return_value={'candidate_identities':{}}), \
                 patch.object(self.worker,'execute_admitted',side_effect=execute), \
                 patch.object(self.worker.sys,'stdin',SimpleNamespace(buffer=io.BytesIO(b'GO\n'))):
                code = self.worker.worker(directory/'guest',directory/'plan.json','a'*64)
            result = json.loads((directory/'guest/results.json').read_text())
            self.assertEqual(code,1)
            self.assertEqual(result['classification'],'UNKNOWN')
            self.assertEqual(result['forward_calls'],1)
            self.assertEqual(result['workspace_clear_calls'],1)
            self.assertEqual(calls,['clear'])
            self.assertTrue(result['CPU_model_collected'])
            self.assertTrue(result['CPU_component_collected'])
            self.assertEqual(result['memory_release'],{'allocated':0,'reserved':0})
            self.assertIn('synthetic learned-forward',result['error'])

    def test_stage_barrier_publishes_identity_then_refuses_expired_host_authority(self):
        # Break caught: no durable ready record, or proceeding without timely host GO.
        with fixture_directory() as directory:
            receipts = SimpleNamespace(_publish=lambda path,value:path.write_text(json.dumps(value),encoding='utf-8'))
            package = SimpleNamespace(receipts=receipts)
            plan = dict(probe=str(directory),marker_sha256='a'*64,attestation_sha256='b'*64)
            profile = SimpleNamespace(sha256='c'*64,run_contract_sha256='d'*64)
            with patch.dict(sys.modules,{'inference_texture':package}), \
                 patch.object(self.worker.time,'monotonic',side_effect=[100.0,100.0,131.0]), \
                 patch.object(self.worker.time,'sleep'):
                with self.assertRaises(TimeoutError):
                    self.worker.stage_ready(plan,profile,'native','e'*64,160.0)
            ready = json.loads((directory/'native-stage-ready.json').read_text())
            self.assertEqual(ready,dict(schema='modly.amd.brick-pbr-stage-ready.v1',stage='native',
                marker_sha256='a'*64,attestation_sha256='b'*64,profile_sha256='c'*64,
                run_contract_sha256='d'*64,plan_sha256='e'*64))
            self.assertFalse((directory/'native-stage-go.json').exists())

    def test_guest_memory_snapshot_counts_only_owned_descendants(self):
        # Break caught: attributing another process's RSS to the owned worker tree.
        with fixture_directory() as directory:
            for pid, parent, rss, peak in ((101,1,100,120),(102,101,30,40),(103,102,5,8),(104,1,999,999)):
                (directory/str(pid)).mkdir()
                (directory/str(pid)/'status').write_text(
                    f'Name:\tfixture\nPid:\t{pid}\nPPid:\t{parent}\nVmRSS:\t{rss} kB\nVmHWM:\t{peak} kB\n',encoding='utf-8')
            result = self.worker.process_memory(pid=101,proc_root=directory)
            self.assertEqual(result['self']['VmRSS_bytes'],102400)
            self.assertEqual(result['descendant_pids'],[102,103])
            self.assertEqual(result['tree_RSS_sum_bytes'],138240)
            self.assertEqual(result['tree_HWM_sum_bytes'],172032)
            self.assertEqual(result['scope'],'self and current descendants; RSS/HWM sums may count shared pages')

    def test_control_executes_verified_source_despite_a_timestamp_valid_bytecode_cache(self):
        # Break caught: authenticated source bytes followed by execution of unchecked .pyc.
        with fixture_directory() as directory:
            source = directory/'scripts/compatibility/amd_wsl_rocdxg_smoke.py'
            source.parent.mkdir(parents=True)
            blob = b"MARKER = 'verified source'\n"
            source.write_bytes(blob)
            cache = source.parent/'__pycache__'/('amd_wsl_rocdxg_smoke.'+sys.implementation.cache_tag+'.pyc')
            cache.parent.mkdir()
            header = importlib.util.MAGIC_NUMBER+struct.pack('<III',0,int(source.stat().st_mtime),len(blob))
            cache.write_bytes(header+marshal.dumps(compile("MARKER = 'unchecked cache'\n",str(source),'exec')))
            with patch.object(self.worker,'ROOT',directory), \
                 patch.object(self.worker,'CONTROL_SHA256',hashlib.sha256(blob).hexdigest()), \
                 patch.object(sys,'pycache_prefix',None), patch.dict(sys.modules):
                sys.modules.pop('amd_wsl_rocdxg_smoke',None)
                actual = self.worker.control()
            self.assertEqual(actual.MARKER,'verified source')

    def test_bytecode_prefix_is_exclusive_absent_and_never_created(self):
        # Break caught: looking up stale project .pyc or creating a writable code cache.
        with fixture_directory() as directory:
            scratch = directory/'scratch'
            prefix = scratch/'forbidden-bytecode'
            with patch.object(self.worker,'SCRATCH',scratch), \
                 patch.object(sys,'pycache_prefix',None), patch.object(sys,'dont_write_bytecode',False):
                self.worker.isolate_bytecode()
                self.assertEqual(sys.pycache_prefix,str(prefix))
                self.assertTrue(sys.dont_write_bytecode)
                self.assertFalse(scratch.exists())
                self.assertFalse(prefix.exists())
            scratch.mkdir()
            prefix.mkdir()
            with patch.object(self.worker,'SCRATCH',scratch), \
                 patch.object(sys,'pycache_prefix',None), patch.object(sys,'dont_write_bytecode',False):
                with self.assertRaises(FileExistsError):
                    self.worker.isolate_bytecode()
                self.assertIsNone(sys.pycache_prefix)
                self.assertFalse(sys.dont_write_bytecode)

    def test_foreign_output_is_refused_before_directory_creation(self):
        # Break caught: creating an arbitrary --worker directory before checking its route.
        exact = self.worker.EV/'run/guest'
        self.assertEqual(self.worker.validate_output(exact),exact)
        with fixture_directory() as directory:
            foreign = directory/'foreign-output'
            with self.assertRaises(ValueError):
                self.worker.worker(foreign,directory/'missing-plan.json','a'*64)
            self.assertFalse(foreign.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
