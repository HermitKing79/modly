"""CPU integration replay: completed outputs are never fresh image inference."""
import asyncio
import hashlib
import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import routers.generation as generation
import services.generator_registry as registry
from schemas.generation import JobStatus
from services.amd_completed_result import (
    CONTRACT, CompletionAuthority, CompletionContext, CompletedWorkerOutput,
    FilePin, authenticate_completion,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tiny_glb():
    binary = struct.pack('<18f3I', 0, 0, 0, 1, 0, 0, 0, 1, 0,
                         0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0, 1, 2)
    document = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
                'nodes': [{'mesh': 0}], 'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'COLOR_0': 1}, 'indices': 2}]}],
                'buffers': [{'byteLength': len(binary)}],
                'bufferViews': [{'buffer': 0, 'byteOffset': 0, 'byteLength': 36},
                                {'buffer': 0, 'byteOffset': 36, 'byteLength': 36},
                                {'buffer': 0, 'byteOffset': 72, 'byteLength': 12}],
                'accessors': [{'bufferView': 0, 'componentType': 5126, 'count': 3, 'type': 'VEC3'},
                              {'bufferView': 1, 'componentType': 5126, 'count': 3, 'type': 'VEC3'},
                              {'bufferView': 2, 'componentType': 5125, 'count': 3, 'type': 'SCALAR'}]}
    data = json.dumps(document, separators=(',', ':')).encode()
    data += b' ' * (-len(data) % 4)
    return struct.pack('<III', 0x46546c67, 2, 28 + len(data) + len(binary)) + struct.pack('<I4s', len(data), b'JSON') + data + struct.pack('<I4s', len(binary), b'BIN\0') + binary


class ReplayGenerator:
    completed_results_enabled = True

    def __init__(self, root):
        self.root = root
        self.outputs_dir = None
        self.cancel_on_return = False
        self.source = root / 'accepted.glb'
        self.source.write_bytes(_tiny_glb())
        self.worker = root / 'qualified-worker.py'
        self.worker.write_text('# pinned CPU replay fixture; no inference\n')
        self.proofs = {}
        for name in ('export', 'export_host', 'verification', 'verification_host'):
            path = root / (name + '.json')
            path.write_text(json.dumps({'classification': 'PASS', 'fixture': True}))
            self.proofs[name] = FilePin(path, _sha(path))

    def generate(self, *args):
        raise AssertionError('Replay must not be dispatched as image inference')

    def generate_completed(self, image, params, job_id, progress_cb=None, cancel_event=None):
        context = CompletionContext.new(job_id, image, params, mode='replay_fixture', model_id='test-amd/generate')
        record = {'contract': CONTRACT, 'product_context': context.to_dict(), 'status': 'PASS',
                  'source_run_id': 'accepted-test-fixture', 'worker_source_sha256': _sha(self.worker),
                  'output': {'path': self.source.name, 'bytes': self.source.stat().st_size, 'sha256': _sha(self.source)},
                  'proof_sha256': {name: pin.sha256 for name, pin in self.proofs.items()}}
        receipt = self.root / (context.run_id + '.json')
        receipt.write_text(json.dumps(record))
        candidate = CompletedWorkerOutput(str(receipt), _sha(receipt))
        authority = CompletionAuthority(self.root, receipt.name, _sha(receipt), context.input_sha256,
                                        'accepted-test-fixture', FilePin(self.worker, _sha(self.worker)), self.proofs, 'replay_fixture')
        result = authenticate_completion(context, candidate, authority, cancel_event)
        if self.cancel_on_return:
            cancel_event.set()
        return result


class ReplayRegistry:
    def __init__(self, generator):
        self.generator = generator

    def assert_weight_variant_installed(self, *args):
        pass

    def get_generator(self, *args):
        return self.generator

    def activate_ready_generator(self, *args):
        return self.generator

    def model_status(self, *args):
        return {'loaded': True, 'downloaded': True, 'name': 'Explicit completed-artifact replay'}


class CompletedGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_root = self.root / 'saved-output'
        self.source_root.mkdir()
        self.workspace = self.root / 'workspace'
        self.workspace.mkdir()
        self.generator = ReplayGenerator(self.source_root)
        self.patches = [patch.object(generation, 'generator_registry', ReplayRegistry(self.generator)),
                        patch.object(registry, 'WORKSPACE_DIR', self.workspace)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        for store in (generation._jobs, generation._cancelled, generation._cancel_events,
                      generation._completed_at, generation._job_generators):
            store.clear()
        self.temp.cleanup()

    def run_job(self):
        import uuid
        job_id = str(uuid.uuid4())
        generation._jobs[job_id] = JobStatus(job_id=job_id, status='pending', progress=0)
        generation._cancel_events[job_id] = threading.Event()
        asyncio.run(generation._run_generation(job_id, b'explicit replay fixture image', {}, 'Default', 'mesh', 'test-amd/generate'))
        return generation._jobs[job_id]

    def test_completed_replay_uses_existing_job_and_workspace_result_without_recolouring(self):
        before = self.generator.source.read_bytes()
        job = self.run_job()
        self.assertEqual(job.status, 'done', job.error)
        self.assertEqual(job.progress, 100)
        self.assertEqual(job.result_origin, 'amd-replay')
        self.assertIn('replay', job.step.lower())
        self.assertTrue(job.output_url.startswith('/workspace/Workflows/'))
        output = self.workspace / job.output_url.removeprefix('/workspace/')
        self.assertEqual(output.read_bytes(), before)
        self.assertEqual(self.generator.source.read_bytes(), before)
        self.assertEqual(job.result_metadata['mode'], 'replay_fixture')
        self.assertEqual(_sha(output), job.result_metadata['output_sha256'])
        self.assertEqual(asyncio.run(generation.job_status(job.job_id)).output_url, job.output_url)
        self.assertTrue(output.with_suffix('.json').exists())
        from routers.workflow_runs import get_run
        workflow = asyncio.run(get_run(job.job_id))
        self.assertEqual(workflow.status, 'done')
        self.assertEqual(workflow.output_url, job.output_url)
        self.assertEqual(workflow.scene_candidate['workspace_path'], output.relative_to(self.workspace).as_posix())
        self.assertIn('replay', workflow.step.lower())

    def test_cancellation_before_promotion_never_reports_success(self):
        self.generator.cancel_on_return = True
        job = self.run_job()
        self.assertEqual(job.status, 'cancelled')
        self.assertIsNone(job.output_url)
        self.assertFalse(list(self.workspace.rglob('*.glb')))

    def test_late_cancellation_removes_only_the_new_promotion_and_never_reports_success(self):
        import services.amd_completed_result as results
        promote = results.promote_completion
        def cancel_after_copy(*args, **kwargs):
            published = promote(*args, **kwargs)
            kwargs['cancel_event'].set()
            return published
        with patch.object(results, 'promote_completion', cancel_after_copy):
            job = self.run_job()
        self.assertEqual(job.status, 'cancelled')
        self.assertIsNone(job.output_url)
        self.assertFalse(list(self.workspace.rglob('*.glb')))
        self.assertTrue(self.generator.source.exists())

    def test_unverified_plain_path_cannot_complete_an_opted_in_job(self):
        self.generator.generate_completed = lambda *args: self.generator.source
        job = self.run_job()
        self.assertEqual(job.status, 'error')
        self.assertIsNone(job.output_url)

    def test_async_task_cancellation_uses_the_terminal_commit_lock(self):
        async def scenario():
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            held = []
            class CommitLock:
                def __enter__(self): held.append(True)
                def __exit__(self, *args): held.pop()
            class CancelEvent:
                def set(inner): self.assertTrue(held, 'Cancellation must serialize with publication')
            job_id = 'task-cancellation'
            generation._jobs[job_id] = JobStatus(job_id=job_id, status='running')
            generation._cancel_events[job_id] = CancelEvent()
            with patch.object(loop, 'run_in_executor', return_value=future), patch.object(generation, '_result_commit_lock', CommitLock()):
                task = asyncio.create_task(generation._run_generation(job_id, b'image', {}, model_id='test'))
                await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError): await task
            self.assertEqual(generation._jobs[job_id].status, 'cancelled')
        asyncio.run(scenario())

    def test_async_task_cancellation_after_committed_result_does_not_cancel_it(self):
        async def scenario():
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            job_id = 'already-committed'
            generation._jobs[job_id] = JobStatus(job_id=job_id, status='done', progress=100)
            event = threading.Event(); generation._cancel_events[job_id] = event
            with patch.object(loop, 'run_in_executor', return_value=future):
                task = asyncio.create_task(generation._run_generation(job_id, b'image', {}, model_id='test'))
                await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError): await task
            self.assertEqual(generation._jobs[job_id].status, 'done')
            self.assertFalse(event.is_set())
            self.assertNotIn(job_id, generation._cancelled)
        asyncio.run(scenario())


if __name__ == '__main__':
    unittest.main()
