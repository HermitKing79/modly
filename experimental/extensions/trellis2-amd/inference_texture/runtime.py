"""F16 schema, native provenance and final consumer acceptance; no GPU entrypoint."""
import sys
import threading
import json

from ._source import load_pinned
from .profile import require_authenticated
from .admission import HISTORICAL, NATIVE_SITES, LARGE_TILED_SITES, FINAL_SITE, require
from .tiling import validate_records, validate_final_record

_runtime = load_pinned('pbr_model_runtime', 'api/services/amd_runtime.py',
    '16499c8c8b5ef3f133ee872bade91a4332f4195d757da662022511c5049bf756')
_metadata = load_pinned('pbr_model_metadata', 'api/services/amd_model_metadata.py',
    'ca7b9f01ea09cb1820208e22f9bb5605bf8cf67159ea0dead9be7036eaf2ab96',
    imports={'services.amd_runtime': _runtime})
_strict = load_pinned('pbr_strict_loader', 'experimental/extensions/trellis2-amd/strict_loader.py',
    '0a4b2aab6fb34a034be1725d4514d71f17c59376151b549ab631ebc8b5c8fc79',
    imports={'services.amd_runtime': _runtime, 'services.amd_model_metadata': _metadata})
loader = load_pinned('pbr_texture_loader', 'experimental/extensions/trellis2-amd/texture_decoder_loader.py',
    '409bf691bb3d7f3638d40cb46f02250111249af31f19a4377345da84ec2fb988',
    imports={'services.amd_runtime': _runtime, 'services.amd_model_metadata': _metadata,
             'strict_loader': _strict})
_frozen = load_pinned('pbr_native_boundary', HISTORICAL+'runtime.py',
    '935d431c107ec6ea856bdccff9d1c88298403c7e9afe07232b22e74525bf0e10',
    imports={'texture_decoder_loader': loader})
ALLOWED_NATIVE = _frozen.ALLOWED_NATIVE


def configure(model, torch, mix_types, conv_config, triton_config):
    """Original configuration plus the same caller-owned low-VRAM/TF32 settings."""
    changes = _frozen.configure(model, torch, mix_types, conv_config, triton_config)
    model.low_vram = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return changes


def validate_configured_model(model, torch):
    require(loader.manifest(model) == loader.schema(device='cpu')
            and model.dtype == torch.float16 and not model.pred_subdiv and model.out_channels == 6
            and model.low_vram is True and not model.training
            and not any(p.requires_grad for p in model.parameters())
            and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
            'original F16/F32 schema, low-VRAM, evaluation or TF32 settings differ')


def validate_tensor_contract(x, coordinates, *, profile):
    p = require_authenticated(profile)
    require(tuple(x.shape) == (p.initial_rows, 32) and str(x.dtype) == 'torch.float32'
            and tuple(coordinates.shape) == (p.initial_rows, 4) and str(coordinates.dtype) == 'torch.int32'
            and x.is_contiguous() and coordinates.is_contiguous(),
            'require entire authenticated F32 brick texture latent and original I32 support')


def expected_native_records(profile):
    p = require_authenticated(profile)
    counts = (p.initial_rows, *p.child_rows)
    return [dict(site=site, rows=rows, shape=[rows, 27], dtype='torch.uint32',
                 native_original_binding=True) for site, rows in zip(NATIVE_SITES, counts)]


class NativeBoundary(_frozen.NativeBoundary):
    """All original restrictions plus exact five maps and one shared large map."""
    def __init__(self, cuda, names, hooks, torch, conv_config, spconv, emit, *, profile):
        self.profile_token = require_authenticated(profile)
        require(hooks.profile == self.profile_token, 'native/admission profile differs')
        hooks.spconv = spconv
        super().__init__(cuda, names, hooks, torch, conv_config, spconv, emit)
        self.records, self.changed_bindings = [], []
        self.profile_installed = False
        self.installed = False

    def install(self):
        require(not self.installed and not self.changed_bindings, 'native boundary installed twice')
        self.old_profile, self.old_thread_profile = sys.getprofile(), threading.getprofile()
        try:
            for name in self.names:
                def wrapper(*args, _name=name, **kwargs):
                    return self.call(_name, args, kwargs)
                setattr(self.cuda, name, wrapper)
                self.changed_bindings.append(name)
            sys.setprofile(self.profile)
            self.profile_installed = True
            threading.setprofile(self.profile)
            self.installed = True
        except BaseException:
            self.remove()
            raise
        return self

    def call(self, name, args, kwargs):
        require(self.installed and self.calls < len(NATIVE_SITES) and self.hooks.active_conv is not None,
                'native boundary closed or map count exceeded')
        site, _, x = self.hooks.active_conv
        require(site == NATIVE_SITES[self.calls], 'native global-map site/order differs')
        if site == LARGE_TILED_SITES[0]:
            require(self.hooks.large_neighbor_map is None and self.hooks.large_neighbor_coords is None,
                    'duplicate admitted large global native map')
        result = super().call(name, args, kwargs)
        if site == LARGE_TILED_SITES[0]:
            self.hooks.large_neighbor_map, self.hooks.large_neighbor_coords = result, x.coords
        self.records.append(dict(site=site, rows=x.coords.shape[0], shape=list(result.shape),
                                 dtype=str(result.dtype), native_original_binding=True))
        return result

    def assert_complete(self):
        require(self.calls == 5 and self.records == expected_native_records(self.profile_token),
                'five exact original global native maps required')

    def remove(self):
        self.phase = 'closed'
        if self.profile_installed:
            sys.setprofile(self.old_profile)
            threading.setprofile(self.old_thread_profile)
            self.profile_installed = False
        for name in reversed(self.changed_bindings):
            setattr(self.cuda, name, self.originals[name])
        self.changed_bindings.clear()
        self.hooks = None
        self.installed = False


def validate_result(record, *, profile):
    """Future host acceptance of a genuine worker receipt, never manufacture one."""
    p = require_authenticated(profile)
    d = record.get('decoder_diagnostics', {})
    required_true = ('large_convolution_patch_restored', 'final_convolution_patch_restored',
                     'native_bindings_restored', 'convolution_hooks_removed',
                     'GPU_input_output_tensors_collected', 'CPU_model_collected',
                     'CPU_parameters_collected', 'CPU_component_collected', 'constructed_on_meta')
    require(record.get('pbr_profile') == p.document() and record.get('profile_sha256') == p.sha256
            and record.get('run_contract_sha256') == p.run_contract_sha256,
            'worker profile authentication differs')
    require(record.get('decoder_cpu') == 'PASS' and record.get('native_import') == 'PASS'
            and record.get('forward_calls') == 1 and record.get('convolutions_completed') == 40
            and record.get('final_convolution_chunks') == 60
            and record.get('final_convolution_rows') == p.final_rows
            and record.get('large_convolution_chunks') == 84 and record.get('native_map_calls') == 5
            and record.get('native_map_records') == expected_native_records(p)
            and all(record.get(key) is True for key in required_true)
            and record.get('memory_release') == {'allocated': 0, 'reserved': 0}
            and record.get('workspace_clear_calls') == 1
            and record.get('runtime_dtype_counts') == {'F16': 208, 'F32': 76}
            and record.get('runtime_dtype_changes') == {} and record.get('buffers_count') == 0
            and d.get('strict') is True and d.get('roundtrip_verified') is True
            and d.get('entries') == 284 and d.get('dtype_counts') == {'F16': 208, 'F32': 76}
            and not d.get('missing_keys') and not d.get('unexpected_keys')
            and record.get('policy_final', {}).get('rejected_calls') == 0
            and not any(key in record for key in ('error', 'cleanup_error', 'neural_failure')),
            'incomplete strict F16 load/decode/restoration/zero-release acceptance')
    validate_records(record.get('large_convolution_records'), profile=p)
    validate_final_record(record.get('final_convolution_record'), profile=p)
    subdivisions = record.get('subdivision_operations')
    require(type(subdivisions) is list and len(subdivisions) == 8,
            'eight genuine guided C2S records required')
    for index, row in enumerate(subdivisions):
        stage, call = divmod(index, 2)
        require(row.get('site') == f'blocks.{stage}.{COUNTS_FOR_RESULT[stage]}.updown'
                and row.get('call') == call+1 and row.get('parents') == p.parent_rows[stage]
                and row.get('children') == p.child_rows[stage], 'C2S receipt lineage/order differs')
    output = record.get('output', {})
    require(output.get('verified') is True and output.get('independently_verified_before_release') is True,
            'durable field writer and independent consumer must acknowledge before release')
    provenance = output.get('metadata', {}).get('provenance', {})
    require(provenance.get('mode') == 'production'
            and provenance.get('pbr_profile_sha256') == p.sha256
            and provenance.get('pbr_profile_document') == json.dumps(
                p.document(), sort_keys=True, separators=(',', ':'), allow_nan=False),
            'genuine production field with exact authenticated profile binding required')
    require(record.get('output_validation') == dict(rows=p.final_rows,
                raw_shape=[p.final_rows, 6], features_shape=[p.final_rows, 6],
                coordinates_shape=[p.final_rows, 4], all_finite=True,
                geometry_coordinate_order_exact=True,
                original_transform='output * F32(0.5) + F32(0.5), CPU elementwise; no clamp', clamped=False),
            'final field support/transform/finite-value acceptance differs')
    require(record.get('guide_order_validation') == dict(verified=True, calls=[2, 2, 2, 2],
                                                       final_geometry_order_exact=True),
            'genuine guide-order completion missing')
    return True


COUNTS_FOR_RESULT = (4, 16, 8, 4)
