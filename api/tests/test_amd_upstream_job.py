"""Synthetic host contract tests; no worker, inference, OS job, or model runs."""
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from services.amd_completed_result import CompletionContext, FilePin

try:
    from services import amd_upstream_job as subject
except ImportError:
    subject = None


def pin(path):
    return FilePin(path, hashlib.sha256(path.read_bytes()).hexdigest())


def seal_synthetic_job(private_root, *, context, image_bytes, params, shape_evidence,
                       source_pins, runtime_input_pins, stage_outputs, cancel_event=None):
    """Authenticate test-owned bytes only; never claims genuine AMD completion."""
    job = subject.UpstreamJob(private_root, context=context, image_bytes=image_bytes, params=params,
        shape_evidence=shape_evidence, source_pins=source_pins, runtime_input_pins=runtime_input_pins,
        stages=tuple(subject.StageSpec(stage, tuple(subject.OutputSpec(path, 2*1024**2) for path in paths))
                     for stage, paths in stage_outputs.items()), cancel_event=cancel_event)
    for stage in stage_outputs:
        ticket = job.issue(stage)
        for path in stage_outputs[stage]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('synthetic host-contract output for ' + stage).encode())
        job.accept(ticket, subject.HostStageObservation(ticket, ticket.stage_id, context.to_dict(),
                    ticket.nonce, 0, True, True, False))
    return job.complete()


class NoAI(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'torch', 'triton', 'trellis2', 'flex_gemm', 'transformers'}:
            raise AssertionError('Upstream host contract attempted AI import: ' + name)


class UpstreamJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        cls.guard = NoAI()
        sys.meta_path.insert(0, cls.guard)

    @classmethod
    def tearDownClass(cls):
        import sys
        sys.meta_path.remove(cls.guard)

    def setUp(self):
        self.assertIsNotNone(subject, 'Host-owned upstream receipt integration is missing')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.private = self.root / 'private'
        self.private.mkdir()
        self.evidence = self.root / 'evidence' / 'request'
        self.shape = self.evidence / 'shape'
        self.image = b'contract-test synthetic image bytes; not an AMD image'
        self.params = {'image_mode': 'alpha-only', 'resolution': 512}
        self.context = CompletionContext.new('synthetic-upstream', self.image, self.params, model_id='trellis2-amd')
        self.cancel = threading.Event()
        image_path = self.evidence / 'occupancy' / 'source-image.bin'
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(self.image)
        contract_path = image_path.parent / 'input-contract.json'
        contract_path.write_text(json.dumps({'source_sha256': self.context.input_sha256, 'synthetic_only': True}))
        self.runtime = {'image': pin(image_path), 'input_contract': pin(contract_path)}
        source = self.root / 'host-selected-policy.py'
        source.write_text('# synthetic policy, never executable by this module\n')
        self.sources = {'policy': pin(source)}
        self.stage_outputs = {
            'conditioning': (self.evidence / 'occupancy' / 'dino' / 'run' / 'conditioning.bin',),
            'occupancy': (self.evidence / 'occupancy' / 'decoder' / 'run' / 'occupancy.bin',),
            'shape': (self.shape / 'run' / 'results.json',
                      self.shape / 'run' / 'scratch' / 'shape-latent-denormalized.npy',
                      self.shape / 'run' / 'scratch' / 'shape-latent-coordinates.npy')}
        for stage, paths in self.stage_outputs.items():
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, stage):
        for path in self.stage_outputs[stage]:
            path.write_bytes(('synthetic ' + stage + ' contract bytes; no genuine worker claim').encode())

    def job(self, **changes):
        options = dict(context=self.context, image_bytes=self.image, params=self.params, shape_evidence=self.shape,
            source_pins=self.sources, runtime_input_pins=self.runtime,
            stages=tuple(subject.StageSpec(stage, tuple(subject.OutputSpec(path, 2*1024**2) for path in paths))
                         for stage, paths in self.stage_outputs.items()), cancel_event=self.cancel)
        options.update(changes)
        return subject.UpstreamJob(self.private, **options)

    def observe(self, ticket, **changes):
        values = dict(ticket=ticket, stage_id=ticket.stage_id, product_context=self.context.to_dict(), nonce=ticket.nonce,
                      returncode=0, natural=True, settled=True, forced=False)
        values.update(changes)
        return subject.HostStageObservation(**values)

    def complete(self):
        job = self.job()
        for stage in self.stage_outputs:
            ticket = job.issue(stage)
            self.emit(stage)
            job.accept(ticket, self.observe(ticket))
        return job, job.complete()

    def test_sealed_handoff_reopens_actual_bytes_and_preserves_raw_receipt(self):
        job, handoff = self.complete()
        raw = ('synthetic shape contract bytes; no genuine worker claim').encode()
        self.assertIs(type(handoff), subject.AuthenticatedShapeInput)
        self.assertIs(handoff.validate(self.context, self.image, self.params, self.shape,
                                     pin(self.shape / 'run/results.json').sha256, self.cancel), handoff)
        self.assertEqual(handoff.product_context, self.context.to_dict())
        self.assertEqual(handoff.runtime_input_pins, self.runtime)
        self.assertEqual(handoff.source_pins, self.sources)
        self.assertEqual((self.shape / 'run/results.json').read_bytes(), raw)
        receipt = json.loads(handoff.receipt_path.read_text())
        self.assertEqual(receipt['qualification'], 'CPU_CONTRACT_ONLY')
        self.assertFalse(receipt['genuine_execution_qualified'])
        self.assertEqual(len(receipt['stages']), 3)
        self.assertEqual(handoff.sha256, pin(handoff.receipt_path).sha256)

    def test_host_observation_requires_exact_ticket_stage_context_nonce_and_settlement(self):
        changes = ({'stage_id': 'shape'}, {'nonce': 'foreign'},
                   {'product_context': dict(self.context.to_dict(), job_id='other')},
                   {'returncode': 1}, {'returncode': False}, {'natural': False}, {'settled': False}, {'forced': True})
        for index, change in enumerate(changes):
            with self.subTest(change=change):
                private = self.root / ('private-' + str(index))
                private.mkdir()
                self.private = private
                job = self.job()
                ticket = job.issue('conditioning')
                with self.assertRaises(ValueError):
                    job.accept(ticket, self.observe(ticket, **change))

    def test_worker_status_dictionary_cannot_be_completion_authority(self):
        job = self.job()
        ticket = job.issue('conditioning')
        with self.assertRaises(ValueError):
            job.accept(ticket, {'status': 'PASS', 'ticket': ticket, 'natural': True, 'settled': True})
        with self.assertRaises(ValueError):
            job.complete()

    def test_ticket_is_one_use_ordered_and_held_by_exact_host_instance(self):
        job = self.job()
        with self.assertRaises(ValueError):
            job.issue('shape')
        with self.assertRaises(ValueError):
            job.issue('conditioning')
        self.private = self.root / 'duplicate-issue-job'
        self.private.mkdir()
        job = self.job()
        ticket = job.issue('conditioning')
        with self.assertRaises(ValueError):
            job.issue('conditioning')
        with self.assertRaises(ValueError):
            job.accept(ticket, self.observe(ticket))
        self.private = self.root / 'copied-ticket-job'
        self.private.mkdir()
        job = self.job()
        ticket = job.issue('conditioning')
        from dataclasses import replace
        with self.assertRaises(ValueError):
            job.accept(replace(ticket), self.observe(ticket))
        with self.assertRaises(ValueError):
            job.accept(ticket, self.observe(ticket))
        self.private = self.root / 'second-ticket-job'
        self.private.mkdir()
        job = self.job()
        ticket = job.issue('conditioning')
        self.emit('conditioning')
        job.accept(ticket, self.observe(ticket))
        with self.assertRaises(ValueError):
            job.accept(ticket, self.observe(ticket))

    def test_request_nonce_and_job_run_reservation_cannot_be_reused(self):
        self.job()
        with self.assertRaises((ValueError, FileExistsError)):
            self.job()
        changed = CompletionContext.from_dict(dict(self.context.to_dict(), job_id='other-job', run_id='other-run'))
        with self.assertRaises((ValueError, FileExistsError)):
            self.job(context=changed)

    def test_wrong_image_parameters_context_mode_and_private_source_root_are_rejected(self):
        for change in ({'image_bytes': b'other'}, {'params': {'resolution': 1024}},
                       {'context': CompletionContext.from_dict(dict(self.context.to_dict(), mode='replay_fixture'))},
                       {'context': CompletionContext.from_dict(dict(self.context.to_dict(), model_id='other'))}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.job(**change)
        source_tree = Path(__file__).resolve().parents[2]
        with self.assertRaises(ValueError):
            subject.UpstreamJob(source_tree, context=self.context, image_bytes=self.image, params=self.params,
                shape_evidence=self.shape, source_pins=self.sources, runtime_input_pins=self.runtime, stages=())

    def test_runtime_image_and_contract_pins_must_match_request_actual_bytes(self):
        changed = dict(self.runtime, image=FilePin(self.runtime['image'].path, '0'*64))
        with self.assertRaises(ValueError):
            self.job(runtime_input_pins=changed)
        self.runtime['input_contract'].path.write_text(json.dumps({'source_sha256': '0'*64}))
        changed = dict(self.runtime, input_contract=pin(self.runtime['input_contract'].path))
        with self.assertRaises(ValueError):
            self.job(runtime_input_pins=changed)

    def test_changed_sources_runtime_predecessor_outputs_and_receipts_are_rejected(self):
        for index, target in enumerate(('source', 'image', 'prior-output', 'prior-receipt')):
            with self.subTest(target=target):
                self.private = self.root / ('change-' + str(index))
                self.private.mkdir()
                job = self.job()
                ticket = job.issue('conditioning')
                self.emit('conditioning')
                receipt = job.accept(ticket, self.observe(ticket))
                path = {'source': self.sources['policy'].path, 'image': self.runtime['image'].path,
                        'prior-output': self.stage_outputs['conditioning'][0], 'prior-receipt': receipt.path}[target]
                previous = path.read_bytes()
                path.write_bytes(previous + b'changed')
                with self.assertRaises(ValueError):
                    job.issue('occupancy')
                path.write_bytes(previous)
                for paths in self.stage_outputs.values():
                    for output in paths:
                        if output.exists():
                            output.unlink()

    def test_cancel_before_and_after_stages_permanently_blocks_completion(self):
        self.cancel.set()
        with self.assertRaises(InterruptedError):
            self.job()
        self.cancel.clear()
        job = self.job()
        ticket = job.issue('conditioning')
        self.cancel.set()
        with self.assertRaises(InterruptedError):
            job.accept(ticket, self.observe(ticket))
        self.cancel.clear()
        with self.assertRaises((InterruptedError, ValueError)):
            job.complete()

    def test_handoff_rejects_foreign_request_hash_tamper_and_late_cancel(self):
        job, handoff = self.complete()
        other = CompletionContext.from_dict(dict(self.context.to_dict(), nonce='another-nonce'))
        for context, expected in ((other, handoff.source_job_sha256), (self.context, '0'*64)):
            with self.subTest(context=context, expected=expected), self.assertRaises(ValueError):
                handoff.validate(context, self.image, self.params, self.shape, expected, self.cancel)
        handoff.receipt_path.write_bytes(b'changed sidecar')
        with self.assertRaises(ValueError):
            handoff.validate(self.context, self.image, self.params, self.shape, handoff.source_job_sha256, self.cancel)

    def test_cancel_after_handoff_is_sticky_and_context_views_are_detached(self):
        job, handoff = self.complete()
        context = handoff.product_context
        context['job_id'] = 'mutated'
        self.assertEqual(handoff.product_context['job_id'], self.context.job_id)
        with self.assertRaises(TypeError):
            handoff.runtime_input_pins['image'] = self.runtime['input_contract']
        self.cancel.set()
        with self.assertRaises(InterruptedError):
            handoff.validate(self.context, self.image, self.params, self.shape, handoff.source_job_sha256, self.cancel)
        self.cancel.clear()
        with self.assertRaises((InterruptedError, ValueError)):
            handoff.validate(self.context, self.image, self.params, self.shape, handoff.source_job_sha256, self.cancel)

    def test_preexisting_outputs_and_outputs_before_ticket_cannot_be_relabelled(self):
        self.emit('conditioning')
        with self.assertRaises(ValueError):
            self.job()
        for path in self.stage_outputs['conditioning']:
            path.unlink()
        job = self.job()
        self.emit('conditioning')
        with self.assertRaises(ValueError):
            job.issue('conditioning')

    def test_stage_output_must_be_fresh_for_its_exact_ticket(self):
        job = self.job()
        ticket = job.issue('conditioning')
        self.emit('conditioning')
        path = self.stage_outputs['conditioning'][0]
        os.utime(path, ns=(self.context.created_ns, self.context.created_ns))
        with self.assertRaises(ValueError):
            job.accept(ticket, self.observe(ticket))

    def test_copied_seal_cannot_create_another_authenticated_handoff(self):
        from dataclasses import replace
        _, handoff = self.complete()
        copied = replace(handoff)
        with self.assertRaises(ValueError):
            copied.validate(self.context, self.image, self.params, self.shape, handoff.source_job_sha256, self.cancel)

    def test_uninitialised_handoff_is_rejected_as_invalid_authority(self):
        empty = object.__new__(subject.AuthenticatedShapeInput)
        try:
            empty.validate(self.context, self.image, self.params, self.shape, '0'*64, self.cancel)
        except ValueError:
            return
        except Exception as exc:
            self.fail('Uninitialised authority must fail closed with ValueError, got ' + type(exc).__name__)
        self.fail('Uninitialised authority was accepted')

    def test_failed_observation_or_incomplete_completion_permanently_closes_chain(self):
        job = self.job()
        ticket = job.issue('conditioning')
        with self.assertRaises(ValueError):
            job.accept(ticket, self.observe(ticket, settled=False))
        self.emit('conditioning')
        with self.assertRaises(ValueError):
            job.accept(ticket, self.observe(ticket))
        for path in self.stage_outputs['conditioning']:
            path.unlink()
        self.private = self.root / 'incomplete-chain'
        self.private.mkdir()
        job = self.job()
        with self.assertRaises(ValueError):
            job.complete()
        with self.assertRaises(ValueError):
            job.issue('conditioning')

    def test_shape_profile_cannot_omit_actual_latent_or_coordinate_artifact(self):
        stages = tuple(subject.StageSpec(stage, tuple(subject.OutputSpec(path, 2*1024**2) for path in paths))
                       for stage, paths in self.stage_outputs.items())
        for missing in range(3):
            self.private = self.root / ('omitted-artifact-' + str(missing))
            self.private.mkdir()
            final = subject.StageSpec('shape', tuple(o for i, o in enumerate(stages[-1].outputs) if i != missing))
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.job(stages=(*stages[:-1], final))

    def test_next_ticket_reauthenticates_predecessor_and_sources_before_dispatch(self):
        for index, target in enumerate(('prior-output', 'source')):
            with self.subTest(target=target):
                self.private = self.root / ('predispatch-' + str(index))
                self.private.mkdir()
                job = self.job()
                ticket = job.issue('conditioning')
                self.emit('conditioning')
                job.accept(ticket, self.observe(ticket))
                path = self.stage_outputs['conditioning'][0] if target == 'prior-output' else self.sources['policy'].path
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b'changed before dispatch')
                    with self.assertRaises(ValueError):
                        job.issue('occupancy')
                    path.write_bytes(original)
                    with self.assertRaises(ValueError):
                        job.issue('occupancy')
                finally:
                    path.write_bytes(original)
                    for output in self.stage_outputs['conditioning']:
                        output.unlink()


if __name__ == '__main__':
    unittest.main()
