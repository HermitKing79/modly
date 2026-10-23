"""Current-image byte binding for CPU qualification, never execution authority.

The host authenticates the submitted request and upstream shape lineage before
constructing this contract. Pins identify the inherited policy; they do not
claim that installed model, runtime, or source bytes were freshly verified.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import MappingProxyType

SCHEMA = 'modly.amd.current-image-neural-run.v1'
ROOT = Path(__file__).resolve().parents[4]
SOURCE_MANIFEST = 'docs/compatibility/amd-wsl-neural-decoder-preparation-evidence/decoder-contract-readback.json'
_POLICY = {
    'source_manifest_sha256': '77cec60d7769c8f448c5880e620bf7007568c24671450037f74819f02d02a0c3',
    'checkpoint_sha256': 'e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581',
    'configuration_sha256': '5f1b856dffce79466fb18f5f0eefc08f08c673b24b3c05b20359c0b6d318a209',
    'native_binary_sha256': '91f66fa2c0f3567cee71246257bea8eb0fe5af9b6c70617d3c5894aa9f103f01',
    'native_import_policy_sha256': '8b9207a14757ccda197e11c054d694f9b4ff47006a1cccc46d71f1d494ab96cd',
    'explicit_gemm_source_sha256': '67641d13e06ca135a4f90eb71a01f7d8d32448bb5aa7c31b858c69368811fb6b',
    'precision_policy_sha256': 'ea4199df31941795b7a5a5bb58d0f180f610c6ad17e8358b16393696a30c3c2e',
    'runtime_policy_sha256': 'c10dc0afc08be765bfcfb1ddba6acd1aba22ebdb214da0c82bbdb9e12ae58cc2',
    'algorithm': 'explicit_gemm',
    'BF16_policy': '216 body tensors after original strict F16/F32 verification',
    'runtime': {'torch': '2.9.1+rocm7.2.1.gitff65f5bc',
                'distribution_version': '2.9.1+rocm7.2.1.lw.gitff65f5bc',
                'git_version': 'ff65f5bc672795c5e5033900ea0a0c4f8566c8cf',
                'hip': '7.2.53211-e1a6bc5663', 'cuda': None,
                'prefix': '/opt/modly-wsl/venv', 'python_prefix': '3.12.'}}
_BINDING_FIELDS = {'product_context', 'image_input_contract_sha256',
                   'initial_features_payload_sha256', 'initial_coords_payload_sha256', 'identity_policy'}


def _freeze(value):
    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze(item) for item in value)
    return value


def _document(value):
    if isinstance(value, (dict, MappingProxyType)):
        return {key: _document(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_document(item) for item in value]
    return value


IDENTITY_POLICY = _freeze(_POLICY)
del _POLICY


def _local_module(name, path):
    path = path.resolve()
    for module in tuple(sys.modules.values()):
        if getattr(module, '__file__', None) == str(path):
            return module
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return module


def _base_api():
    return _local_module('_modly_current_input_run_contract', Path(__file__).with_name('run_contract.py'))


def _checked_base(document, api):
    if (type(document) is not dict or document.get('schema') != api.SCHEMA
            or 'final_support_profile' in document or document.get('large_convolution_tiling') is not True):
        raise ValueError('Current-image input requires the generic legacy envelope and explicit large tiling')
    return api.parse(document)


def _checked_binding(binding, api):
    if type(binding) is not dict or set(binding) != _BINDING_FIELDS:
        raise ValueError('Exact current-image input binding required')
    completion = _local_module('_modly_current_input_completion_context', ROOT / 'api/services/amd_completed_result.py')
    context = completion.CompletionContext.from_dict(binding['product_context'])
    if context.mode != 'generated' or context.model_id != 'trellis2-amd':
        raise ValueError('Current-image geometry requires a generated trellis2-amd request context')
    for name in _BINDING_FIELDS - {'product_context', 'identity_policy'}:
        api._sha(binding[name])
    if binding['identity_policy'] != _document(IDENTITY_POLICY):
        raise ValueError('Current-image input differs from the pinned identity policy')
    return dict(binding, product_context=context.to_dict())


@dataclass(frozen=True)
class CurrentImageRunContract:
    base: object
    current_input: object

    def __post_init__(self):
        api = _base_api()
        base = _checked_base(self.base.document(), api)
        binding = _checked_binding(_document(self.current_input), api)
        object.__setattr__(self, 'base', base)
        object.__setattr__(self, 'current_input', _freeze(binding))

    def __getattr__(self, name):
        if name in self.base.__dataclass_fields__:
            return getattr(self.base, name)
        raise AttributeError(name)

    def document(self):
        return {'schema': SCHEMA, 'base': self.base.document(), 'current_input': _document(self.current_input)}

    @property
    def sha256(self):
        blob = json.dumps(self.document(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        return hashlib.sha256(blob).hexdigest()


def parse(document, *, base_api=None):
    if type(document) is not dict or set(document) != {'schema', 'base', 'current_input'} or document.get('schema') != SCHEMA:
        raise ValueError('Exact current-image neural run schema required')
    api = _base_api() if base_api is None else base_api
    base = _checked_base(document['base'], api)
    binding = _checked_binding(document['current_input'], api)
    return CurrentImageRunContract(base, binding)


def create(base, *, product_context, image_input_contract_sha256,
           initial_features_payload_sha256, initial_coords_payload_sha256):
    """Pure constructor; the host must first authenticate request and lineage."""
    base = base if type(base) is dict else base.document()
    return parse({'schema': SCHEMA, 'base': base, 'current_input': {
        'product_context': product_context, 'image_input_contract_sha256': image_input_contract_sha256,
        'initial_features_payload_sha256': initial_features_payload_sha256,
        'initial_coords_payload_sha256': initial_coords_payload_sha256,
        'identity_policy': _document(IDENTITY_POLICY)}})


def is_current(contract):
    return contract is not None and contract.document().get('schema') == SCHEMA


def binding_document(contract):
    return _document(parse(contract.document()).current_input)


def metadata(contract):
    binding = binding_document(contract)
    return {'product_context': dict(binding['product_context']), 'current_input': binding}


def _source_identity():
    path = ROOT / SOURCE_MANIFEST
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 4*1024**2:
        raise ValueError('Bounded pinned decoder source manifest required')
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != IDENTITY_POLICY['source_manifest_sha256']:
        raise ValueError('Pinned decoder source manifest changed')
    checks = json.loads(blob)['source_checks']
    return {'/mnt/' + name[0].lower() + '/' + name[3:].replace('\\', '/'): row['expected_sha256']
            for name, row in checks.items()}


def artifact_binding(provenance, contract):
    contract = parse(contract.document())
    binding = binding_document(contract)
    expected = dict(run_contract_sha256=contract.sha256, source_job_sha256=contract.source_job_sha256,
                    input_file_sha256=dict(contract.input_sha256), source=_source_identity(),
                    checkpoint_sha256=IDENTITY_POLICY['checkpoint_sha256'],
                    native_binary_sha256=IDENTITY_POLICY['native_binary_sha256'],
                    algorithm=IDENTITY_POLICY['algorithm'], BF16_policy=IDENTITY_POLICY['BF16_policy'],
                    product_context=binding['product_context'], current_input=binding)
    if (type(provenance) is not dict or 'final_support_profile' in provenance
            or any(provenance.get(key) != value for key, value in expected.items())):
        raise ValueError('Current-image artifact request, inputs or pinned identity differs')
    runtime = provenance.get('runtime')
    expected_runtime = dict(IDENTITY_POLICY['runtime'])
    python_prefix = expected_runtime.pop('python_prefix')
    if (type(runtime) is not dict or any(runtime.get(key) != value for key, value in expected_runtime.items())
            or type(runtime.get('python')) is not str or not runtime['python'].startswith(python_prefix)
            or runtime.get('gpu_initialized_before') is not False):
        raise ValueError('Current-image artifact runtime identity differs')
