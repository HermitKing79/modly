"""Host-owned upstream byte receipts, CPU contract qualification only.

There is no executor or worker-status deserializer. Trusted host integration
must supply actual stage settlement observations. Independent AMD lineage and
numeric readers remain mandatory consumers of the sealed handoff.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import os
import time
from types import MappingProxyType
import uuid

from services.amd_completed_result import (
    CompletionContext, FilePin, _canonical, _child, _identity, _json, _name, _need,
    _opened, _plain, _sha,
)

STAGES = ('conditioning', 'occupancy', 'shape')
_SEAL = object()
_MAX_FILE = 256 * 1024**2


@dataclass(frozen=True)
class OutputSpec:
    path: Path
    maximum_bytes: int


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    outputs: tuple[OutputSpec, ...]


@dataclass(frozen=True)
class _Ticket:
    stage_id: str
    nonce: str
    predecessor_sha256: str
    created_ns: int


@dataclass(frozen=True)
class HostStageObservation:
    """Trusted host input, never reconstructed from worker JSON or parameters."""
    ticket: object
    stage_id: str
    product_context: object
    nonce: str
    returncode: int
    natural: bool
    settled: bool
    forced: bool

    def __post_init__(self):
        context = CompletionContext.from_dict(self.product_context)
        object.__setattr__(self, 'product_context', MappingProxyType(context.to_dict()))


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    sha256: str
    bytes: int
    identity: tuple
    maximum_bytes: int
    fresh_ns: int

    def document(self):
        return dict(path=str(self.path), sha256=self.sha256, bytes=self.bytes, identity=list(self.identity))

    def recheck(self):
        actual = _snapshot(self.path, self.maximum_bytes, self.fresh_ns, self.sha256)
        _need(actual == self, 'Host-authenticated file identity changed')


def _snapshot(path, limit, fresh_ns=0, expected=None):
    path = _plain(path)
    with _opened(path, limit, fresh_ns) as (stream, info):
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if expected is not None:
        _need(digest == _sha(expected), 'Host-selected file SHA256 differs')
    return _Snapshot(path, digest, info.st_size, _identity(info), limit, fresh_ns)


def _save(path, document, fresh_ns):
    blob = _canonical(document) + b'\n'
    _need(len(blob) <= 4*1024**2, 'Host receipt exceeds bounded JSON size')
    with path.open('xb') as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    return _snapshot(path, 4*1024**2, fresh_ns, hashlib.sha256(blob).hexdigest())


class UpstreamJob:
    """One private host reservation; owns exact tickets, pins and receipt chain."""
    def __init__(self, private_root, *, context, image_bytes, params, shape_evidence,
                 source_pins, runtime_input_pins, stages, cancel_event=None):
        self._blocked = False
        self._cancel_event = cancel_event
        self._pending = self._completion = None
        self._bound_ticket = None
        self._accepted, self._outputs = [], []
        self._check()
        self._context = CompletionContext.from_dict(context.to_dict())
        self._context.validate_request(image_bytes, params)
        _need(self._context.mode == 'generated' and self._context.model_id == 'trellis2-amd',
              'Original generated trellis2-amd product request required')
        private_root = _plain(private_root)
        source_tree = Path(__file__).resolve().parents[2]
        _need(private_root.is_dir() and not private_root.is_relative_to(source_tree)
              and not any(part.lower() == '.git' for part in private_root.parts),
              'Existing private root outside the product source tree required')
        self._shape_evidence = _plain(shape_evidence)
        _need(self._shape_evidence.name == 'shape', 'Exact host-owned shape evidence leaf required')
        evidence_root = self._shape_evidence.parent
        _need(type(source_pins) is dict and 0 < len(source_pins) <= 128,
              'Bounded host-selected published source pins required')
        self._sources = tuple((_name(name), self._pin(pin, 4*1024**2)) for name, pin in source_pins.items())
        _need(type(runtime_input_pins) is dict and set(runtime_input_pins) == {'image', 'input_contract'},
              'Exactly the submitted image and input-contract runtime pins required')
        self._runtime = tuple((name, self._pin(runtime_input_pins[name], limit, self._context.created_ns))
                              for name, limit in (('image', 2*1024**2), ('input_contract', 65536)))
        runtime = dict(self._runtime)
        _need(all(row.path.is_relative_to(evidence_root) for row in runtime.values())
              and runtime['image'].path != runtime['input_contract'].path
              and runtime['input_contract'].path.name == 'input-contract.json',
              'Distinct runtime inputs must belong to this exact evidence root')
        _need(runtime['image'].sha256 == self._context.input_sha256, 'Runtime image differs from the submitted bytes')
        with _opened(runtime['input_contract'].path, 65536, self._context.created_ns) as (stream, _):
            image_contract = _json(stream.read(65537))
        _need(type(image_contract) is dict and image_contract.get('source_sha256') == self._context.input_sha256,
              'Actual image input contract differs from the submitted image')
        _need(type(stages) is tuple and all(type(stage) is StageSpec for stage in stages)
              and tuple(stage.stage_id for stage in stages) == STAGES, 'Exact ordered upstream stage profile required')
        self._stages = []
        paths = set()
        for stage in stages:
            _need(type(stage.outputs) is tuple and 0 < len(stage.outputs) <= 16, 'Bounded exact stage output expectations required')
            outputs = []
            for output in stage.outputs:
                _need(type(output) is OutputSpec and type(output.maximum_bytes) is int
                      and 0 < output.maximum_bytes <= _MAX_FILE, 'Bounded host output expectation required')
                path = _plain(output.path)
                _need(path.is_relative_to(evidence_root) and path not in paths
                      and path not in {row.path for row in runtime.values()}, 'Distinct confined stage outputs required')
                _need(not path.exists() and not path.is_symlink(), 'Preexisting output cannot become a current upstream artifact')
                paths.add(path)
                outputs.append(OutputSpec(path, output.maximum_bytes))
            self._stages.append(StageSpec(stage.stage_id, tuple(outputs)))
        self._stages = tuple(self._stages)
        required = {self._shape_evidence / 'run' / name for name in (
            'results.json', 'scratch/shape-latent-denormalized.npy', 'scratch/shape-latent-coordinates.npy')}
        _need(required <= {row.path for row in self._stages[-1].outputs},
              'Exact raw shape-job receipt, denormalized latent and coordinates must be shape outputs')
        self._check()
        # Both are permanent exclusive reservations, including failed runs.
        _save(_child(private_root, 'nonce-' + self._context.nonce + '.json'),
              {'product_context': self._context.to_dict(), 'qualification': 'CPU_CONTRACT_ONLY'}, self._context.created_ns)
        self._directory = _child(private_root, 'amd-upstream-' + self._context.job_id + '-' + self._context.run_id)
        self._directory.mkdir(exist_ok=False)
        self._request = _save(self._directory / 'request.json', {
            'schema': 'modly.amd.upstream-request.v1', 'product_context': self._context.to_dict(),
            'params': _json(_canonical(params)), 'qualification': 'CPU_CONTRACT_ONLY',
            'source_pins': {name: row.document() for name, row in self._sources},
            'runtime_input_pins': {name: row.document() for name, row in self._runtime},
            'stages': [dict(stage_id=s.stage_id, outputs=[dict(path=str(o.path), maximum_bytes=o.maximum_bytes)
                                                       for o in s.outputs]) for s in self._stages]}, self._context.created_ns)
        self._reauthenticate()

    @staticmethod
    def _pin(pin, limit, fresh_ns=0):
        _need(type(pin) is FilePin, 'Host-selected FilePin required; worker descriptors are not authority')
        return _snapshot(pin.path, limit, fresh_ns, pin.sha256)

    def _check(self, cancel_event=None):
        if any(event is not None and event.is_set() for event in (self._cancel_event, cancel_event)):
            self._blocked = True
            raise InterruptedError('Upstream product request cancelled')
        _need(not self._blocked, 'Upstream host completion is permanently blocked')

    def _reauthenticate(self):
        for row in [self._request, *[r for _, r in self._sources], *[r for _, r in self._runtime],
                    *self._accepted, *self._outputs]:
            row.recheck()

    def issue(self, stage_id):
        try:
            self._check()
            _need(self._completion is None and self._pending is None and len(self._accepted) < len(self._stages)
                  and stage_id == self._stages[len(self._accepted)].stage_id, 'No next host-owned upstream stage ticket')
            self._reauthenticate()
            previous = self._accepted[-1] if self._accepted else self._request
            _need(all(not spec.path.exists() and not spec.path.is_symlink()
                      for spec in self._stages[len(self._accepted)].outputs), 'Output exists before its exact host stage ticket')
            self._check()
            self._pending = _Ticket(stage_id, uuid.uuid4().hex, previous.sha256, time.time_ns())
            return self._pending
        except BaseException:
            self._blocked = True
            raise

    def accept(self, ticket, observation):
        try:
            self._check()
            _need(self._bound_ticket is None, 'Windows-bound tickets require original runner authority')
            _need(ticket is self._pending and ticket is not None and type(observation) is HostStageObservation
                  and observation.ticket is ticket and observation.stage_id == ticket.stage_id
                  and observation.nonce == ticket.nonce and dict(observation.product_context) == self._context.to_dict(),
                  'Exact host ticket, stage, nonce and original product context observation required')
            _need(type(observation.returncode) is int and observation.returncode == 0
                  and observation.natural is True and observation.settled is True and observation.forced is False,
                  'Naturally settled successful host stage observation required')
            return self._accept_outputs(ticket, 'CPU_CONTRACT_ONLY')
        except BaseException:
            self._blocked = True
            raise

    def bind_windows(self, ticket, runner, command):
        """Bind one host ticket to the original Windows runner; launches nothing."""
        return self._bind_supervised(ticket, runner, command, 'windows')

    def bind_wsl(self, ticket, runner, command):
        """Bind one host ticket to the original WSL broker route; launches nothing."""
        return self._bind_supervised(ticket, runner, command, 'wsl')

    def _bind_supervised(self, ticket, runner, command, backendkind):
        try:
            self._check()
            _need(ticket is self._pending and ticket is not None and self._bound_ticket is None,
                  'One exact unbound pending upstream ticket required')
            from services.amd_windows_authority import bind
            token = bind(self, ticket, runner, command, backendkind=backendkind)
            self._bound_ticket = ticket
            return token
        except BaseException:
            self._blocked = True
            raise

    def accept_windows(self, ticket, runner, authority):
        """Consume runner-issued lifecycle authority, never a worker/result dict."""
        return self._accept_supervised(ticket, runner, authority, 'windows')

    def accept_wsl(self, ticket, runner, authority):
        """Consume original WSL broker and Windows client lifecycle authority."""
        return self._accept_supervised(ticket, runner, authority, 'wsl')

    def _accept_supervised(self, ticket, runner, authority, backendkind):
        try:
            self._check()
            _need(ticket is self._pending and ticket is self._bound_ticket and ticket is not None,
                  'Exact pending Windows-bound upstream ticket required')
            from services.amd_windows_authority import consume, document, reopen_sources
            summary, runner_pin = consume(authority, self, ticket, runner, backendkind=backendkind)
            summary = document(summary)
            runner_receipt = _snapshot(runner_pin.path, 4*1024**2, ticket.created_ns, runner_pin.sha256)
            fixed = reopen_sources(summary['binding'])

            def validate():
                self._check()
                _need(time.monotonic() < summary['binding']['deadline_monotonic_s'],
                      'Windows authority expired during acceptance')
                for row in (*fixed, runner_receipt):
                    row.recheck()

            return self._accept_outputs(ticket, backendkind.upper() + '_CPU_SUPERVISED',
                extra={'genuine_execution_qualified': False, backendkind + '_supervision': summary,
                       'runner_receipt': runner_receipt.document()},
                witnesses=(*fixed, runner_receipt), validate=validate)
        except BaseException:
            self._blocked = True
            raise

    def _accept_outputs(self, ticket, qualification, *, extra=None, witnesses=(), validate=None):
        self._reauthenticate()
        if validate is not None:
            validate()
        stage = self._stages[len(self._accepted)]
        outputs = tuple(_snapshot(spec.path, spec.maximum_bytes, ticket.created_ns) for spec in stage.outputs)
        receipt = _save(self._directory / (stage.stage_id + '.json'), {
            'schema': 'modly.amd.upstream-stage.v1', 'qualification': qualification,
            'product_context': self._context.to_dict(), 'stage_id': ticket.stage_id, 'nonce': ticket.nonce,
            'predecessor_sha256': ticket.predecessor_sha256, 'natural': True, 'settled': True, 'forced': False,
            'outputs': [row.document() for row in outputs], **(extra or {})}, self._context.created_ns)
        self._outputs.extend((*outputs, *witnesses))
        self._accepted.append(receipt)
        self._reauthenticate()
        if validate is not None:
            validate()
        self._check()
        self._pending = self._bound_ticket = None
        return FilePin(receipt.path, receipt.sha256)

    def complete(self):
        try:
            self._check()
            _need(self._completion is None and self._pending is None and len(self._accepted) == len(STAGES),
                  'One complete ordered upstream host receipt chain required')
            self._reauthenticate()
            source = next(row for row in self._outputs if row.path == self._shape_evidence / 'run/results.json')
            receipt = _save(self._directory / 'shape-completion.json', {
                'schema': 'modly.amd.upstream-shape-completion.v1', 'qualification': 'CPU_CONTRACT_ONLY',
                'genuine_execution_qualified': False, 'product_context': self._context.to_dict(),
                'shape_evidence': str(self._shape_evidence), 'source_job': source.document(),
                'request_sha256': self._request.sha256,
                'stages': [row.document() for row in self._accepted],
                'source_pins': {name: row.document() for name, row in self._sources},
                'runtime_input_pins': {name: row.document() for name, row in self._runtime}}, self._context.created_ns)
            self._reauthenticate()
            self._check()
            self._completion = AuthenticatedShapeInput(self, receipt, source, _SEAL)
            return self._completion
        except BaseException:
            self._blocked = True
            raise


@dataclass(frozen=True)
class AuthenticatedShapeInput:
    _job: object
    _receipt: _Snapshot
    _source: _Snapshot
    _seal: object

    @property
    def product_context(self):
        return self._job._context.to_dict()

    @property
    def shape_evidence(self):
        return self._job._shape_evidence

    @property
    def source_job_sha256(self):
        return self._source.sha256

    @property
    def sha256(self):
        return self._receipt.sha256

    @property
    def receipt_path(self):
        return self._receipt.path

    @property
    def runtime_input_pins(self):
        return MappingProxyType({name: FilePin(row.path, row.sha256) for name, row in self._job._runtime})

    @property
    def source_pins(self):
        return MappingProxyType({name: FilePin(row.path, row.sha256) for name, row in self._job._sources})

    def validate(self, context, image_bytes, params, shape_evidence, expected_source_job_sha256, cancel_event=None):
        job = getattr(self, '_job', None)
        _need(type(self) is AuthenticatedShapeInput and getattr(self, '_seal', None) is _SEAL
              and type(job) is UpstreamJob and getattr(job, '_completion', None) is self,
              'Exact sealed host-authenticated shape input required')
        self._job._check(cancel_event)
        context = CompletionContext.from_dict(context.to_dict())
        context.validate_request(image_bytes, params)
        _need(context == self._job._context and _plain(shape_evidence) == self.shape_evidence
              and _sha(expected_source_job_sha256) == self.source_job_sha256,
              'Host shape handoff differs from the exact request or completed source job')
        try:
            self._job._reauthenticate()
            self._receipt.recheck()
            self._source.recheck()
            self._job._check(cancel_event)
        except BaseException:
            self._job._blocked = True
            raise
        return self
