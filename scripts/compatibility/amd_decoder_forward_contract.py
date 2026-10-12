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
                      and record.get('fixture')==fixture(CONFIG)
                      and record.get('cpu_load',{}).get('status')=='PASS'
                      and record.get('forward',{}).get('status')=='PASS'
                      and record['forward'].get('grad_enabled') is False
                      and record['forward'].get('output_is_inference_tensor') is True
                      and process_ok is True and healthy is True and preserved is True) else 'PARTIAL'
