"""Pure decisions for the one approved sparse-decoder forward."""
import math
import statistics
from amd_decoder_contract import gpu_operation_allowed, validate_release

CONFIG = {'name':'SparseStructureDecoder','args':{'out_channels':1,'latent_channels':8,
          'num_res_blocks':2,'num_res_blocks_middle':2,'channels':[512,128,32],'use_fp16':True}}


def fixture(config):
    if config != CONFIG:
        raise ValueError('Decoder configuration differs from the inspected minimal-forward architecture')
    return {'input_shape':[1,8,1,1,1], 'output_shape':[1,1,4,4,4], 'input_dtype':'torch.float32',
            'output_dtype':'torch.float32', 'values':[-.875,-.625,-.375,-.125,.125,.375,.625,.875]}


class SingleForward:
    def __init__(self): self.calls = 0
    def invoke(self, call):
        if self.calls:
            raise RuntimeError('Single forward permit already consumed; no retries or warm-ups')
        self.calls = 1
        return call()


class TensorTransfers:
    """Exact single-use input/output copies; never an operator-wide allowance."""
    def __init__(self, input_shape=None, output_shape=None):
        self.input_shape=[1,8,1,1,1] if input_shape is None else list(input_shape)
        self.output_shape=[1,1,4,4,4] if output_shape is None else list(output_shape)
        if (self.input_shape,self.output_shape) not in [([1,8,1,1,1],[1,1,4,4,4]),([1,8,16,16,16],[1,1,64,64,64])]:
            raise ValueError('Only the reviewed tiny or saved-latent decoder shapes are permitted')
        self.counts = {'input': 0, 'output': 0}
        self.active = None

    def begin(self, direction, source, description, phase):
        if direction not in self.counts or self.active is not None or self.counts[direction]:
            raise RuntimeError('Invalid, overlapping or consumed tensor transfer permit')
        if direction == 'output' and self.counts['input'] != 1:
            raise RuntimeError('Output copy requires the completed input transfer')
        stride = description.get('stride')
        if not isinstance(stride,list) or len(stride)!=5 or any(type(value) is not int or value<0 for value in stride):
            raise RuntimeError('Invalid transfer stride metadata')
        expected = {'shape': self.input_shape if direction == 'input' else self.output_shape,
                    'stride': [math.prod(self.input_shape[i+1:]) for i in range(5)] if direction == 'input' else list(stride),
                    'dtype': 'torch.float32', 'device': 'cpu' if direction == 'input' else 'cuda:0',
                    'layout': 'torch.strided', 'contiguous': True, 'requires_grad': False}
        if (phase != direction+'-transfer' or description != expected
                or description.get('contiguous') is not True or description.get('requires_grad') is not False):
            raise RuntimeError('Unexpected tensor metadata or transfer lifecycle phase')
        self.active = (direction, source, expected)

    def authorize(self, operation, args, kwargs, phase, description):
        if self.active is None:
            raise RuntimeError('Unarmed or repeated tensor transfer')
        direction, source, expected = self.active
        target = 'cuda:0' if direction == 'input' else 'cpu'
        if (operation != 'aten._to_copy.default' or type(args) is not tuple or len(args) != 1
                or args[0] is not source or phase != direction+'-transfer'
                or description != expected or kwargs != {'device': target}):
            raise RuntimeError('Unapproved tensor copy operation, source, arguments or phase')
        # Consume before redispatch: even a native failure cannot enable a retry.
        self.counts[direction] += 1
        self.active = None


def placement_operation(name):
    return gpu_operation_allowed(name)


def prohibited_import(name):
    # trellis2/__init__.py imports the pinned, inert lazy pipelines registry.
    # Real pipeline modules/classes and all mesh runtimes stay out of scope.
    return any(name.startswith(prefix) for prefix in ('cumesh','o_voxel','nvdiffrast','trellis2.pipelines.'))


def validate_output(output, expected):
    values = output.get('values', [])
    if (output.get('shape') != expected['output_shape'] or output.get('dtype') != expected['output_dtype']
            or output.get('device') != 'cuda:0' or output.get('requires_grad') is not False
            or output.get('inference_mode') is not True or not isinstance(values,list)
            or len(values) != math.prod(expected['output_shape'])
            or any(type(value) not in (int,float) or not math.isfinite(value) for value in values)):
        raise ValueError('Decoder output shape/dtype/device/inference/finiteness validation failed')
    return {'count':len(values),'finite':True,'minimum':min(values),'maximum':max(values),
            'mean':statistics.fmean(values),'population_stddev':statistics.pstdev(values),
            'representative_values':values[:8]}


def classify(record, process_ok, healthy, preserved):
    try:
        validate_output(record.get('forward',{}).get('output',{}),fixture(CONFIG))
        release=record.get('release',{})
        validate_release(release)
        if release.get('input_collected') is not True or release.get('output_collected') is not True:
            raise ValueError('Live input/output tensors remain')
    except (ValueError,KeyError,TypeError,AttributeError):
        return 'PARTIAL'
    return 'PASS' if (record.get('status')=='PASS' and record.get('stage')=='complete'
                      and all(type(record.get(key)) is int and record[key]==1 for key in ('forward_calls','root_decoder_calls','load_calls'))
                      and record.get('transfer_calls')=={'input':1,'output':1}
                      and all(type(value) is int for value in record['transfer_calls'].values())
                      and record.get('fixture')==fixture(CONFIG)
                      and record.get('cpu_load',{}).get('status')=='PASS'
                      and record.get('forward',{}).get('status')=='PASS'
                      and record['forward'].get('grad_enabled') is False
                      and record['forward'].get('output_is_inference_tensor') is True
                      and process_ok is True and healthy is True and preserved is True) else 'PARTIAL'
