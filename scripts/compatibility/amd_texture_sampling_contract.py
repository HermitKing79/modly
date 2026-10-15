"""Exact texture-SLat512 architecture and twelve positive source evaluations."""
import copy
import json
import math
import re

import numpy as np

import amd_shape_slat_contract as shape
import amd_shape_sampling_contract as sampling

CPU_SECONDS, GPU_SECONDS, OVERALL_SECONDS = 180, 240, 480
MEMORY_LIMIT = REQUIRED_RAM = 12 * 1024**3
FREE_RESERVE = 2 * 1024**3
OPAQUE_HEADROOM = 1024**3
TOKEN_COUNT = 3743
SEED = 20261015
SIGMA_MIN = .00001
PARAMS = dict(steps=12, rescale_t=3., guidance_strength=1., guidance_rescale=0.,
              guidance_interval=(.6, .9), dino_lock=0.)
CONFIG = copy.deepcopy(shape.CONFIG)
CONFIG['args']['in_channels'] = 64
SAMPLER_MODULES = sampling.SAMPLER_MODULES
validate_coords = shape.validate_coords
validate_output = shape.validate_output
prohibited_import = sampling.prohibited_import

SHAPE_NORMALIZATION = {
    'mean': [0.781296, 0.018091, -0.495192, -0.558457, 1.060530, 0.093252, 1.518149, -0.933218,
             -0.732996, 2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944,
             0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
             -0.059621, 2.220780, 1.621212, 0.877230, 0.567247, -3.175944, -3.186688, 1.578665],
    'std': [5.972266, 4.706852, 5.445010, 5.209927, 5.320220, 4.547237, 5.020802, 5.444004,
            5.226681, 5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578,
            4.805265, 5.124013, 5.530808, 5.619001, 5.103930, 5.417670, 5.269677, 5.547194,
            5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691]}
TEXTURE_NORMALIZATION = {
    'mean': [3.501659, 2.212398, 2.226094, 0.251093, -0.026248, -0.687364, 0.439898, -0.928075,
             0.029398, -0.339596, -0.869527, 1.038479, -0.972385, 0.126042, -1.129303, 0.455149,
             -1.209521, 2.069067, 0.544735, 2.569128, -0.323407, 2.293000, -1.925608, -1.217717,
             1.213905, 0.971588, -0.023631, 0.106750, 2.021786, 0.250524, -0.662387, -0.768862],
    'std': [2.665652, 2.743913, 2.765121, 2.595319, 3.037293, 2.291316, 2.144656, 2.911822,
            2.969419, 2.501689, 2.154811, 3.163343, 2.621215, 2.381943, 3.186697, 3.021588,
            2.295916, 3.234985, 3.233086, 2.260140, 2.874801, 2.810596, 3.292720, 2.674999,
            2.680878, 2.372054, 2.451546, 2.353556, 2.995195, 2.379849, 2.786195, 2.775190]}


def schema():
    result = shape.schema()
    result['input_layer.weight']['shape'] = [1536, 64]
    return result


def validate_header(config, tensors):
    if config != CONFIG or set(tensors) != set(schema()):
        raise ValueError('Exact texture-SLat architecture/640 keys mismatch')
    for key, row in schema().items():
        if tensors[key]['shape'] != row['shape'] or tensors[key]['dtype'] != 'BF16':
            raise ValueError('Texture checkpoint shape/BF16 mismatch: ' + key)


def validate_manifest(parameters, buffers, device):
    if buffers or set(parameters) != set(schema()):
        raise ValueError('Texture-SLat parameter/buffer keys mismatch')
    for key, row in schema().items():
        if parameters[key] != {**row, 'device': device, 'is_meta': device == 'meta'}:
            raise ValueError('Texture runtime shape/dtype/device mismatch: ' + key)


def authorize_placement(gate, expected, checkpoint_sha256, worker_sha256):
    if (gate.get('status') != 'PASS' or gate.get('gpu_initialized') is not False
            or gate.get('strict') is not True or gate.get('assign') is not True
            or gate.get('missing_keys') != [] or gate.get('unexpected_keys') != []
            or gate.get('parameter_bytes') != expected['tensor_data_bytes']
            or worker_sha256 != checkpoint_sha256 or gate.get('payload_entries') != 640
            or gate.get('reversible_widening_count') != 220 or gate.get('feature_verified') is not True
            or gate.get('input_token_count') != expected['token_count']):
        raise ValueError('Texture-SLat CPU gate incomplete')
    validate_header(expected['config'], expected['tensors'])
    validate_manifest(gate.get('tensors', {}), gate.get('buffers', {}), 'cpu')


def times():
    seq = np.linspace(1, 0, PARAMS['steps'] + 1)
    return (PARAMS['rescale_t'] * seq / (1 + (PARAMS['rescale_t'] - 1) * seq)).tolist()


def tickets():
    seq = times()
    return [dict(index=i + 1, step=i + 1, role='positive', t=t, next_t=seq[i + 1])
            for i, t in enumerate(seq[:-1])]


def require_count(total, positive, negative):
    if (any(type(n) is not int for n in (total, positive, negative))
            or (total, positive, negative) != (12, 12, 0)):
        raise ValueError('Texture sampling requires exactly12 positive and zero negative forwards')


class Budget:
    def __init__(self):
        self.rows = tickets()
        self.calls = self.started = self.steps = 0
        self.step = self.active = None

    def begin_step(self, t, next_t):
        if (type(t) not in (int, float) or type(next_t) not in (int, float)
                or not math.isfinite(t) or not math.isfinite(next_t) or self.step is not None
                or self.active is not None or self.steps >= 12
                or abs(t - times()[self.steps]) > 1e-14
                or abs(next_t - times()[self.steps + 1]) > 1e-14):
            raise RuntimeError('Foreign/extra texture Euler step')
        self.step = self.steps + 1

    def next_ticket(self):
        if (self.step is None or self.active is not None or self.calls >= 12
                or self.rows[self.calls]['step'] != self.step):
            raise RuntimeError('No legitimate texture model ticket')
        return self.rows[self.calls]

    def begin_forward(self, ticket):
        if ticket is not self.next_ticket():
            raise RuntimeError('Foreign texture invocation ticket')
        self.active = ticket
        self.started += 1

    def end_forward(self, ticket):
        if ticket is not self.active or ticket is None:
            raise RuntimeError('Incomplete texture invocation')
        self.calls += 1
        self.active = None

    def end_step(self):
        if self.step is None or self.active is not None or self.calls != self.step:
            raise RuntimeError('Incomplete texture step count')
        self.steps += 1
        self.step = None

    def require_complete(self):
        if (self.steps, self.calls, self.started) != (12, 12, 12) or self.step is not None or self.active is not None:
            raise RuntimeError('Canonical12/12 texture sequence incomplete')


def validate_transfer(key, row, target, phase, n):
    if key == 'shape_cond':
        key = 'x'
    if key == 'neg_cond':
        raise ValueError('Texture guidance1.0 permits no negative conditioning transfer')
    sampling.validate_transfer(key, row, target, phase, n)


def validate_latent(x,*,maximum_tokens=TOKEN_COUNT):
    from amd_shape_sampling_artifact import validate
    validate(x,maximum_tokens=maximum_tokens)


def normalization_arrays(stats, expected):
    if stats != expected:
        raise ValueError('Texture route requires the exact pinned normalization identity')
    return np.array(stats['mean'], np.float32), np.array(stats['std'], np.float32)


def normalize_shape(x, stats=None,*,maximum_tokens=TOKEN_COUNT):
    validate_latent(x,maximum_tokens=maximum_tokens)
    mean, std = normalization_arrays(SHAPE_NORMALIZATION if stats is None else stats, SHAPE_NORMALIZATION)
    result = np.ascontiguousarray((x - mean[None]) / std[None])
    validate_latent(result,maximum_tokens=maximum_tokens)
    return result


def denormalize_texture(x, stats=None,*,maximum_tokens=TOKEN_COUNT):
    validate_latent(x,maximum_tokens=maximum_tokens)
    mean, std = normalization_arrays(TEXTURE_NORMALIZATION if stats is None else stats, TEXTURE_NORMALIZATION)
    result = np.ascontiguousarray(x * std[None] + mean[None])
    validate_latent(result,maximum_tokens=maximum_tokens)
    return result


def classify(record, process_ok, healthy, preserved,*,token_count=TOKEN_COUNT,maximum_tokens=TOKEN_COUNT,feature_sha256=None):
    """Require the exact sequence and independently verified external artifact row."""
    try:
        import amd_texture_sampling_local as local
        import amd_texture_sampling_artifact as artifact
        from amd_decoder_contract import validate_release
        if type(maximum_tokens) is not int or not 0<maximum_tokens<=4430 or type(token_count) is not int or not 0<token_count<=maximum_tokens:raise ValueError("Texture token bound differs")
        layouts=artifact.layouts(token_count,maximum_tokens=maximum_tokens)
        payload_bytes=token_count*400
        expected_feature_sha256=local.FEATURE_SHA if feature_sha256 is None else feature_sha256

        def seconds(value, limit):
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= limit:
                raise ValueError('Invalid/unbounded texture timing')

        def digest(value):
            if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
                raise ValueError('Invalid texture digest')

        def count(value, expected):
            if type(value) is not int or value != expected:
                raise ValueError('Texture count differs')

        environment = {key: '0' for key in ('PYTORCH_TUNABLEOP_ENABLED', 'PYTORCH_TUNABLEOP_TUNING',
                                            'PYTORCH_TUNABLEOP_RECORD_UNTUNED')}
        flag_keys = {'is_enabled', 'tuning_is_enabled', 'record_untuned_is_enabled'}
        for phase, initialized in [('tunable_import', False), ('tunable_before_forward', True)]:
            tunable = record[phase]
            if (set(tunable) != {'environment', 'flags', 'gpu_initialized'}
                    or tunable['environment'] != environment or set(tunable['flags']) != flag_keys
                    or any(type(value) is not bool or value for value in tunable['flags'].values())
                    or tunable['gpu_initialized'] is not initialized):
                raise ValueError('Texture actual TunableOp policy/phase evidence differs')
        digest(expected_feature_sha256)
        seconds(record['forward_seconds'], GPU_SECONDS)
        seconds(record['cpu_load']['elapsed_seconds'], CPU_SECONDS)
        validate_output(record['output'], token_count)
        validate_release(record['release'])
        modules = {'<root>': 1, **{f'blocks.{i}': 1 for i in range(30)},
                   **{f'blocks.{i}.{a}': 1 for i in range(30) for a in ('self_attn', 'cross_attn')}}
        if record['calls'] != {key: 12 for key in modules}:
            raise ValueError('Texture model/module counts differ')
        sampled = record['sampling']
        for key, expected in [('sequences', 1), ('started', 12), ('completed', 12),
                               ('positive_calls', 12), ('negative_calls', 0), ('concat_calls', 12)]:
            count(sampled[key], expected)
        if len(sampled['models']) != 12 or len(sampled['steps']) != 12:
            raise ValueError('Texture invocation/step records missing')
        for row, ticket in zip(sampled['models'], tickets()):
            if row != {**ticket, 'modules': modules, 'frequency': 1, 'cache_reused': ticket['index'] > 1}:
                raise ValueError('Texture invocation identity/cache evidence differs')
        for index, row in enumerate(sampled['steps']):
            if (row['step'] != index + 1 or row['forwards'] != 1 or row['total_forwards'] != index + 1
                    or row['t'] != times()[index] or row['next_t'] != times()[index + 1]
                    or any(row[key] is not True for key in ('all_finite', 'synchronized', 'coords_unchanged'))):
                raise ValueError('Texture step evidence differs')
            seconds(row['elapsed_seconds'], GPU_SECONDS)
            for key in ('allocated_bytes', 'reserved_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes'):
                value = row['memory'][key]
                if type(value) is not int or not 0 <= value <= MEMORY_LIMIT:
                    raise ValueError('Texture step memory exceeds cap')
            available = row['memory']['available_bytes']
            if type(available) is not int or available < FREE_RESERVE + OPAQUE_HEADROOM:
                raise ValueError('Texture free reserve/opaque headroom was lost')
        transfers = {key: 1 for key in ('x', 'coords', 'batch_map', 'seqlen', 'cond', 'shape_cond',
                                       'output', 'output_coords', 'rope_frequency')}
        transfers.update(t=12, frequency=12)
        if record['transfers'] != transfers:
            raise ValueError('Texture transfer counts differ')
        saved = record['artifacts']
        if saved['verified'] is not True:
            raise ValueError('Texture artifact independent readback missing')
        metadata = saved['metadata']
        if (metadata['format'] != artifact.FORMAT or metadata['artifact_file'] != artifact.ARTIFACT_NAME
                or type(metadata['artifact_bytes']) is not int
                or not payload_bytes + 8 < metadata['artifact_bytes'] <= 8 + artifact.MAX_HEADER_BYTES + payload_bytes):
            raise ValueError('Texture artifact format/size differs')
        digest(metadata['artifact_sha256'])
        if set(metadata['tensors']) != set(layouts):
            raise ValueError('Texture artifact tensor set differs')
        for key, (dtype, layout) in layouts.items():
            row = metadata['tensors'][key]
            if row != dict(dtype=dtype, shape=list(layout), tensor_bytes=math.prod(layout) * 4, sha256=row['sha256']):
                raise ValueError('Texture artifact tensor layout differs')
            digest(row['sha256'])
        for key, expected in [('raw', record['output']['sha256']), ('coords', record['input']['coords']['sha256']),
                               ('noise', record['input']['x']['sha256'])]:
            if metadata['tensors'][key]['sha256'] != expected:
                raise ValueError('Texture artifact payload identity differs')
        provenance = metadata['provenance']
        expected_params = json.loads(json.dumps(PARAMS))
        if (provenance['parameters'] != expected_params or provenance['sigma_min'] != SIGMA_MIN
                or type(provenance['seed']) is not int or provenance['seed'] != SEED
                or provenance['shape_normalization'] != SHAPE_NORMALIZATION
                or provenance['texture_normalization'] != TEXTURE_NORMALIZATION
                or provenance['input'] != record['input'] or provenance['sampling'] != sampled
                or type(provenance['harness_commit']) is not str
                or re.fullmatch('[0-9a-f]{40}', provenance['harness_commit']) is None):
            raise ValueError('Texture artifact canonical provenance differs')
        checkpoint = provenance['checkpoint']
        if (checkpoint['checkpoint_sha256'] != local.CHECKPOINT_SHA
                or checkpoint['config_sha256'] != local.CONFIG_SHA
                or checkpoint['source_sha256'] != '3959f1f36f17ecd59ba06b91c948ac85303a3aab7d572f6248a846bc50f68d0c'
                or checkpoint['conditioning']['verified'] is not True
                or checkpoint['conditioning']['file_sha256'] != expected_feature_sha256
                or checkpoint['conditioning']['output_sha256'] != record['input']['cond']['sha256']):
            raise ValueError('Texture checkpoint/source/conditioning provenance differs')
    except (KeyError, ValueError, TypeError, IndexError):
        return 'PARTIAL'
    return 'PASS' if (record['status'] == 'PASS' and record['stage'] == 'complete'
                      and record['load_calls'] == 1 and record['forward_calls'] == 12
                      and record['workspace_clear_calls'] == 1
                      and record['release']['owned_tensors_collected'] is True
                      and record['workspace_binding']['verified'] is True
                      and process_ok is True and healthy is True and preserved is True) else 'PARTIAL'
