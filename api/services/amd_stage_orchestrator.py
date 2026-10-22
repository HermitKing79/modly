"""Job-owned stage sequencing. No public capability or neural execution is enabled.

Plans and source pins are host configuration, never request parameters. The
existing completed-result authenticator/publisher remains the only asset gate.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import os
import threading
import time

from services.amd_completed_result import (
    CompletionContext, FilePin, _canonical, _child, _name, _need, _opened, _plain,
)
from services.amd_stage_contract import Artifact, STAGE_ORDER, validate_artifact, _json
from services.generators.base import GenerationCancelled


class StageBlocked(RuntimeError):
    """Admission or complete ownership could not be established."""


class StageCancelled(GenerationCancelled):
    """Cancellation before dispatch or after verified natural settlement."""


@dataclass(frozen=True)
class OutputSpec:
    relative_path: str
    kind: str
    schema: dict


@dataclass(frozen=True)
class StageDefinition:
    stage_id: str
    kind: str
    argv: tuple[str, ...]
    source_pins: dict[str, FilePin]
    outputs: tuple[OutputSpec, ...]
    timeout_s: float = 180
    estimated_bytes: int = 64 * 1024**2


@dataclass(frozen=True)
class StagePlan:
    context: CompletionContext
    mode: str
    preprocess: dict
    stages: tuple[StageDefinition, ...]


def _save(path, value):
    with Path(path).open('xb') as stream:
        stream.write(_canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())


def _digest(path, limit=4 * 1024**2):
    with _opened(path, limit) as (stream, _):
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _read(path):
    with _opened(path, 4 * 1024**2) as (stream, _):
        value = _json(stream.read(4 * 1024**2 + 1))
    _need(type(value) is dict, 'Stage receipt object required')
    return value


class StageOrchestrator:
    def __init__(self, private_root, runner):
        self.private_root = _plain(private_root)
        _need(self.private_root.is_dir(), 'Existing private parent required')
        source_tree = Path(__file__).resolve().parents[2]
        _need(not self.private_root.is_relative_to(source_tree)
              and not any(part.lower() == '.git' for part in self.private_root.parts),
              'Private run artifacts must remain outside the product source tree')
        self.runner = runner

    @staticmethod
    def _plan_identity(plan):
        return _canonical(dict(context=plan.context.to_dict(), mode=plan.mode, preprocess=plan.preprocess,
            stages=[dict(stage_id=s.stage_id, kind=s.kind, argv=list(s.argv), timeout_s=s.timeout_s,
                         estimated_bytes=s.estimated_bytes,
                         sources={k: dict(path=str(p.path), sha256=p.sha256) for k, p in s.source_pins.items()},
                         outputs=[dict(relative_path=o.relative_path, kind=o.kind, schema=o.schema) for o in s.outputs])
                    for s in plan.stages]))

    def _admit_plan(self, plan, image, params):
        _need(type(plan) is StagePlan, 'Host-owned stage plan required')
        context = CompletionContext.from_dict(plan.context.to_dict())
        context.validate_request(image, params)
        _need(len(image) <= 2 * 1024**2, 'Original 2 MiB source bound exceeded')
        _need(plan.preprocess == {'image_mode': 'alpha-only', 'resolution': 512}, 'Qualified preprocessing request required')
        if plan.mode == 'generated':
            # There is deliberately no parameter/manifest switch around this gate.
            raise StageBlocked('Fresh inference qualification is incomplete: current neural support/guide profiles are brick-only; CPU milestone permits no neural execution')
        _need(plan.mode == 'cpu_lifecycle_test' and 1 <= len(plan.stages) <= 16, 'Explicit bounded CPU lifecycle plan required')
        ids = [s.stage_id for s in plan.stages]
        _need(len(set(ids)) == len(ids) and not any(name in STAGE_ORDER for name in ids), 'Synthetic stages must not claim neural pipeline stages')
        for stage in plan.stages:
            _name(stage.stage_id)
            _need(stage.kind in ('windows', 'wsl') and type(stage.argv) is tuple and stage.argv
                  and all(type(arg) is str and arg and '\0' not in arg for arg in stage.argv), 'Exact stage command required')
            _need(type(stage.timeout_s) in (int, float) and 0 < stage.timeout_s <= 180
                  and type(stage.estimated_bytes) is int and 0 < stage.estimated_bytes <= 6 * 1024**3, 'CPU stage resource envelope exceeded')
            _need(type(stage.source_pins) is dict and stage.source_pins and 'worker' in stage.source_pins,
                  'Pinned stage implementation required')
            worker_path = str(_plain(stage.source_pins['worker'].path))
            paths = {worker_path, worker_path.replace('\\', '/')}
            if stage.kind == 'wsl' and worker_path[:3].lower().replace('\\', '/') == 'c:/':
                paths.add('/mnt/c/' + worker_path[3:].replace('\\', '/'))
            _need(any(path in stage.argv for path in paths),
                  'Command must reference its pinned worker')
            _need(stage.outputs and len(stage.outputs) <= 16, 'Exact bounded stage outputs required')
            for output in stage.outputs:
                _need(type(output) is OutputSpec and output.kind == 'json' and output.schema == {},
                      'Synthetic chain permits contextual JSON payloads only')
                _child(self.private_root, output.relative_path)

    def run(self, plan, image, params, cancel_event, progress=None):
        if cancel_event.is_set(): raise StageCancelled()
        self._admit_plan(plan, image, params)
        frozen = self._plan_identity(plan)
        # Authenticate code before reserving a run; repeat before each dispatch.
        for stage in plan.stages:
            for pin in stage.source_pins.values():
                _need(type(pin) is FilePin and _digest(pin.path) == pin.sha256, 'Stage source identity changed')
        root = _child(self.private_root, _name(plan.context.run_id)); root.mkdir()
        stages = root / 'stages'; stages.mkdir()
        _save(root / 'run-plan.json', _json(frozen))
        with (root / 'input.bin').open('xb') as stream:
            stream.write(image); stream.flush(); os.fsync(stream.fileno())
        initial = dict(schema='modly.amd.stage-input.v1', product_context=plan.context.to_dict(),
                       mode=plan.mode, input_sha256=plan.context.input_sha256,
                       preprocess=plan.preprocess, preprocessing_complete=False)
        _save(root / 'input.json', initial)
        previous = root / 'input.json'
        inputs = [dict(relative_path='input.bin', bytes=len(image), sha256=plan.context.input_sha256,
                       kind='submitted-image', schema={})]
        results = []
        def journal(event, stage_id=None, **details):
            row = dict(event=event, stage_id=stage_id, timestamp_ns=time.time_ns(),
                       monotonic_seconds=time.monotonic(), product_context=plan.context.to_dict(), **details)
            with (root / 'events.jsonl').open('ab') as stream:
                stream.write(_canonical(row) + b'\n'); stream.flush(); os.fsync(stream.fileno())
        journal('run_started', synthetic_only=True)
        try:
            for index, stage in enumerate(plan.stages):
                if cancel_event.is_set(): raise StageCancelled()
                _need(self._plan_identity(plan) == frozen, 'Host plan mutated during run')
                for pin in stage.source_pins.values():
                    _need(_digest(pin.path) == pin.sha256, 'Stage source identity changed')
                predecessor = _digest(previous)
                # Inputs remain the preceding stage's authenticated files, in order.
                for descriptor in inputs:
                    candidate = _child(previous.parent, descriptor['relative_path'])
                    _need(_digest(candidate, 256 * 1024**2) == descriptor['sha256'], 'Predecessor artifact changed')
                leaf = _child(stages, f'{index:02d}-' + stage.stage_id); leaf.mkdir()
                source_hashes = {key: pin.sha256 for key, pin in stage.source_pins.items()}
                request = dict(schema='modly.amd.stage-request.v1', mode=plan.mode,
                               product_context=plan.context.to_dict(), stage_id=stage.stage_id,
                               predecessor_path=str(previous), predecessor_sha256=predecessor,
                               inputs=inputs, input_root=str(previous.parent), source_sha256=source_hashes,
                               outputs=[dict(relative_path=o.relative_path, kind=o.kind, schema=o.schema) for o in stage.outputs])
                _save(leaf / 'stage-request.json', request)
                journal('stage_started', stage.stage_id)
                if progress: progress(index * 99 // len(plan.stages), 'CPU lifecycle test: ' + stage.stage_id)
                if cancel_event.is_set(): raise StageCancelled()
                from services.amd_stage_process import StageCommand
                command = StageCommand(stage.stage_id, stage.kind, stage.argv, leaf, plan.context.to_dict(), stage.timeout_s)
                settlement = self.runner.run(command, cancel_event)
                _save(leaf / 'process-settlement.json', settlement)
                _need(self._plan_identity(plan) == frozen, 'Host plan mutated during execution')
                settled = settlement.get('windows_settled') is True and (stage.kind != 'wsl' or settlement.get('linux_settled') is True)
                clean = settled and settlement.get('natural') is True and settlement.get('forced') is False
                if not clean:
                    raise StageBlocked('Stage ownership/cleanup is forced or unverified: ' + stage.stage_id)
                if cancel_event.is_set() or settlement.get('terminal') == 'cancelled':
                    raise StageCancelled()
                if settlement.get('terminal') != 'succeeded' or settlement.get('returncode') != 0:
                    raise StageBlocked('Stage failed without a complete output: ' + stage.stage_id)
                receipt = _read(leaf / 'stage-result.json')
                _need(set(receipt) == {'schema', 'product_context', 'mode', 'stage_id', 'status', 'predecessor_sha256', 'inputs', 'source_sha256', 'outputs'}
                      and receipt['schema'] == 'modly.amd.stage-receipt.v1' and receipt['product_context'] == plan.context.to_dict()
                      and receipt['mode'] == plan.mode and receipt['stage_id'] == stage.stage_id and receipt['status'] == 'PASS'
                      and receipt['predecessor_sha256'] == predecessor and receipt['inputs'] == inputs
                      and receipt['source_sha256'] == source_hashes, 'Foreign or incomplete stage receipt')
                _need(type(receipt['outputs']) is list and len(receipt['outputs']) == len(stage.outputs), 'Complete stage output closure required')
                actual = []
                for spec, descriptor in zip(stage.outputs, receipt['outputs']):
                    artifact = Artifact.from_dict(descriptor)
                    _need((artifact.relative_path, artifact.kind, artifact.schema) == (spec.relative_path, spec.kind, spec.schema), 'Stage output schema differs from host plan')
                    validate_artifact(leaf, artifact, plan.context)
                    actual.append(artifact.to_dict())
                _need(_digest(previous) == predecessor, 'Predecessor receipt changed during stage')
                for descriptor in inputs:
                    candidate = _child(previous.parent, descriptor['relative_path'])
                    _need(_digest(candidate, 256 * 1024**2) == descriptor['sha256'], 'Predecessor artifact changed during stage')
                for pin in stage.source_pins.values():
                    _need(_digest(pin.path) == pin.sha256, 'Stage source changed during execution')
                if cancel_event.is_set(): raise StageCancelled()
                row = dict(schema='modly.amd.stage-settled.v1', product_context=plan.context.to_dict(),
                           mode=plan.mode, stage_id=stage.stage_id, predecessor_sha256=predecessor,
                           worker_receipt_sha256=_digest(leaf / 'stage-result.json'), outputs=actual,
                           source_sha256=source_hashes, settlement=settlement, completed_ns=time.time_ns())
                previous = leaf / 'stage-settled.json'; _save(previous, row)
                inputs = actual; results.append(row)
                journal('stage_succeeded', stage.stage_id, receipt_sha256=_digest(previous))
                if progress: progress((index + 1) * 99 // len(plan.stages), 'CPU lifecycle test completed: ' + stage.stage_id)
            _need(self._plan_identity(plan) == frozen, 'Host plan mutated before terminal acceptance')
            if cancel_event.is_set(): raise StageCancelled()
            final = dict(terminal='succeeded', mode=plan.mode, generated_asset=False,
                         product_context=plan.context.to_dict(), stages=len(results), final_receipt_sha256=_digest(previous))
            _save(root / 'result.json', final); journal('run_succeeded', synthetic_only=True)
            return final
        except BaseException as exc:
            terminal = 'cancelled' if isinstance(exc, StageCancelled) else 'blocked'
            journal('run_' + terminal, reason=type(exc).__name__ + ': ' + str(exc))
            _save(root / 'result.json', dict(terminal=terminal, generated_asset=False, mode=plan.mode,
                                           completed_stages=len(results), reason=type(exc).__name__ + ': ' + str(exc)))
            if isinstance(exc, (StageCancelled, StageBlocked)): raise
            raise StageBlocked(str(exc)) from exc


class JobOwnedStageGenerator:
    """Unregistered host adapter. A CPU test chain can never return a model."""
    completed_results_enabled = True
    job_owned_stages = True

    def __init__(self, model_id, private_root, plan_factory, runner):
        self.MODEL_ID, self.private_root = model_id, private_root
        self.plan_factory, self.runner = plan_factory, runner
        self.outputs_dir = None
        self._active = None
        self._guard = threading.Lock()

    def request_cancel(self, job_id):
        with self._guard:
            if self._active is None or self._active[0] != job_id: return False
            self._active[1].set()
            return True

    def generate_completed(self, image, params, job_id, progress_cb=None, cancel_event=None):
        event = cancel_event if cancel_event is not None else threading.Event()
        with self._guard:
            if self._active is not None: raise StageBlocked('This stage owner already holds a job')
            self._active = job_id, event
        try:
            context = CompletionContext.new(job_id, image, params, model_id=self.MODEL_ID)
            plan = self.plan_factory(context, image, params)
            _need(plan.context == context, 'Plan context differs from current product job')
            StageOrchestrator(self.private_root, self.runner).run(plan, image, params, event, progress_cb)
            raise StageBlocked('Synthetic lifecycle test completed; no inference or generated asset exists')
        finally:
            with self._guard: self._active = None
