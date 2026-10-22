"""Contract tests use synthetic payloads, never neural execution or a generated GLB."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest
import zlib

from services.amd_completed_result import CompletionContext, FilePin
from services.amd_stage_orchestrator import (
    JobOwnedStageGenerator, OutputSpec, StageDefinition, StagePlan, StageBlocked,
    StageCancelled, StageOrchestrator,
)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CPUReceiptRunner:
    """Only substitutes the OS boundary; real contracts/files/journal are exercised."""
    def __init__(self):
        self.calls = []
        self.fail = None
        self.cancel_after = None
        self.forced = False
        self.foreign = False

    def run(self, command, cancel_event, tick=None):
        self.calls.append(command.stage_id)
        request = json.loads((command.cwd / 'stage-request.json').read_text())
        context = dict(request['product_context'])
        if self.foreign: context['nonce'] = 'foreign'
        output = command.cwd / 'output.json'
        output.write_text(json.dumps({'product_context': context, 'synthetic_only': True, 'value': len(self.calls)}))
        receipt = dict(schema='modly.amd.stage-receipt.v1', product_context=context,
                       mode='cpu_lifecycle_test', stage_id=command.stage_id, status='PASS',
                       predecessor_sha256=request['predecessor_sha256'],
                       inputs=request['inputs'], source_sha256=request['source_sha256'],
                       outputs=[dict(relative_path='output.json', sha256=digest(output),
                                     bytes=output.stat().st_size, kind='json', schema={})])
        if self.fail == 'malformed': receipt['predecessor_sha256'] = '0' * 64
        (command.cwd / 'stage-result.json').write_text(json.dumps(receipt))
        if tick: tick({'test': True})
        if self.cancel_after == command.stage_id: cancel_event.set()
        return dict(terminal='failed' if self.fail == 'crash' else 'succeeded', returncode=1 if self.fail == 'crash' else 0,
                    windows_settled=True, linux_settled=True if command.kind == 'wsl' else None,
                    natural=not self.forced, forced=self.forced)


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'pinned.py'; self.source.write_text('# synthetic only\n')
        def chunk(kind, data): return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
        pixels = b''.join(b'\0' + b''.join(bytes((50, 80, 100, 255 if (x, y) == (3, 3) else 0)) for x in range(8)) for y in range(8))
        self.image = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 8, 8, 8, 6, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b'')
        self.params = {'image_mode': 'alpha-only', 'resolution': 512}
        self.context = CompletionContext.new('test-job', self.image, self.params, model_id='test-only/stage-chain')
        self.runner = CPUReceiptRunner()
        self.cancel = threading.Event()

    def tearDown(self): self.temp.cleanup()

    def plan(self, mode='cpu_lifecycle_test'):
        return StagePlan(self.context, mode, self.params, (
            StageDefinition('control_windows', 'windows', ('python', str(self.source)),
                            {'worker': FilePin(self.source, digest(self.source))}, (OutputSpec('output.json', 'json', {}),)),
            StageDefinition('control_wsl', 'wsl', ('/usr/bin/python3', str(self.source)),
                            {'worker': FilePin(self.source, digest(self.source))}, (OutputSpec('output.json', 'json', {}),)),
        ))

    def run_plan(self, plan=None, progress=None):
        return StageOrchestrator(self.root, self.runner).run(plan or self.plan(), self.image, self.params, self.cancel, progress)

    def test_real_receipts_are_sequential_current_context_and_predecessor_bound(self):
        progress = []
        result = self.run_plan(progress=lambda value, label: progress.append((value, label)))
        self.assertEqual(self.runner.calls, ['control_windows', 'control_wsl'])
        self.assertEqual(result['terminal'], 'succeeded')
        self.assertFalse(result['generated_asset'])
        self.assertTrue(all(value < 100 for value, _ in progress))
        run = self.root / self.context.run_id
        rows = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
        self.assertEqual([row['stage_id'] for row in rows if row['event'] == 'stage_succeeded'], self.runner.calls)
        first = run / 'stages/00-control_windows/stage-settled.json'
        second = json.loads((run / 'stages/01-control_wsl/stage-request.json').read_text())
        self.assertEqual(second['predecessor_sha256'], digest(first))
        self.assertEqual(second['inputs'][0]['sha256'], digest(first.parent / 'output.json'))

    def test_cancel_before_dispatch_never_launches_worker(self):
        self.cancel.set()
        with self.assertRaises(StageCancelled): self.run_plan()
        self.assertEqual(self.runner.calls, [])

    def test_cancel_between_stages_rejects_late_success_and_stops_chain(self):
        self.runner.cancel_after = 'control_windows'
        with self.assertRaises(StageCancelled): self.run_plan()
        self.assertEqual(self.runner.calls, ['control_windows'])
        self.assertFalse(list(self.root.rglob('*.glb')))

    def test_forced_or_unverified_settlement_is_not_clean_cancellation(self):
        self.runner.forced = True
        self.runner.cancel_after = 'control_windows'
        with self.assertRaises(StageBlocked): self.run_plan()
        self.assertEqual(self.runner.calls, ['control_windows'])

    def test_malformed_or_foreign_receipt_blocks_successor(self):
        for setting in ('malformed', 'foreign', 'crash'):
            with self.subTest(setting=setting):
                self.context = CompletionContext.new('test-job', self.image, self.params, model_id='test-only/stage-chain')
                self.runner = CPUReceiptRunner()
                if setting == 'foreign': self.runner.foreign = True
                else: self.runner.fail = setting
                with self.assertRaises(StageBlocked): self.run_plan()
                self.assertEqual(self.runner.calls, ['control_windows'])

    def test_changed_image_params_source_or_existing_run_are_rejected(self):
        with self.assertRaises((ValueError, StageBlocked)):
            StageOrchestrator(self.root, self.runner).run(self.plan(), b'foreign', self.params, self.cancel)
        changed = dict(self.params, resolution=1024)
        with self.assertRaises((ValueError, StageBlocked)):
            StageOrchestrator(self.root, self.runner).run(self.plan(), self.image, changed, self.cancel)
        plan = self.plan(); self.source.write_text('# changed\n')
        with self.assertRaises((ValueError, StageBlocked)): self.run_plan(plan)
        self.assertFalse(self.runner.calls)

    def test_generated_mode_cannot_launch_without_neural_qualification(self):
        with self.assertRaisesRegex(StageBlocked, 'qualification|inference|CPU'):
            self.run_plan(self.plan(mode='generated'))
        self.assertFalse(self.runner.calls)

    def test_synthetic_adapter_never_returns_generated_completion(self):
        def factory(context, image, params):
            plan = self.plan()
            return StagePlan(context, plan.mode, plan.preprocess, plan.stages)
        adapter = JobOwnedStageGenerator('test-only/stage-chain', self.root, factory, self.runner)
        with self.assertRaisesRegex(StageBlocked, 'synthetic|Synthetic|inference'):
            adapter.generate_completed(self.image, self.params, 'test-job', cancel_event=self.cancel)
        self.assertFalse(list(self.root.rglob('*.glb')))
        self.assertFalse(adapter.request_cancel('another-job'))

    def test_final_stage_cannot_modify_its_predecessor_artifact(self):
        original = self.runner.run
        def corrupt(command, event, tick=None):
            result = original(command, event, tick)
            if command.stage_id == 'control_wsl':
                request = json.loads((command.cwd / 'stage-request.json').read_text())
                (Path(request['input_root']) / request['inputs'][0]['relative_path']).write_bytes(b'changed')
            return result
        self.runner.run = corrupt
        with self.assertRaisesRegex(StageBlocked, 'Predecessor'): self.run_plan()

    def test_cancel_after_stage_started_notification_prevents_dispatch(self):
        with self.assertRaises(StageCancelled):
            self.run_plan(progress=lambda value, label: self.cancel.set())
        self.assertEqual(self.runner.calls, [])

    def test_missing_linux_ack_blocks_after_client_exit(self):
        original = self.runner.run
        def missing(command, event, tick=None):
            result = original(command, event, tick)
            if command.kind == 'wsl': result['linux_settled'] = False
            return result
        self.runner.run = missing
        with self.assertRaisesRegex(StageBlocked, 'ownership|cleanup'): self.run_plan()

    def test_existing_run_is_not_reused_or_overwritten(self):
        self.run_plan()
        count = len(self.runner.calls)
        with self.assertRaises(FileExistsError): self.run_plan()
        self.assertEqual(len(self.runner.calls), count)

    def test_final_worker_cannot_remove_the_host_source_pins(self):
        plan = self.plan()
        original = self.runner.run
        def mutate(command, event, tick=None):
            result = original(command, event, tick)
            if command.kind == 'wsl':
                plan.stages[-1].source_pins.clear()
                self.source.write_text('# changed after dispatch\n')
            return result
        self.runner.run = mutate
        with self.assertRaisesRegex(StageBlocked, 'plan mutated'): self.run_plan(plan)

    def test_cancel_at_final_completion_notification_never_returns_success(self):
        def cancel_final(value, label):
            if label == 'CPU lifecycle test completed: control_wsl': self.cancel.set()
        with self.assertRaises(StageCancelled): self.run_plan(progress=cancel_final)


if __name__ == '__main__': unittest.main()
