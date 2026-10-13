"""Only the committed 16-cubed latent and native 64-cubed decoder output."""
import amd_decoder_forward_contract as base
from amd_decoder_forward_contract import CONFIG,SingleForward,placement_operation,prohibited_import
from amd_decoder_contract import validate_release
INPUT_SHAPE=[1,8,16,16,16];OUTPUT_SHAPE=[1,1,64,64,64]
MEMORY_LIMIT=12*1024**3
CPU_SECONDS=120;GPU_SECONDS=180;OVERALL_SECONDS=360

class TensorTransfers(base.TensorTransfers):
    def __init__(self):super().__init__(INPUT_SHAPE,OUTPUT_SHAPE)

def fixture(config):
    if config!=CONFIG:raise ValueError('Pinned sparse decoder configuration changed')
    return dict(input_shape=INPUT_SHAPE,output_shape=OUTPUT_SHAPE,input_dtype='torch.float32',output_dtype='torch.float32',input_mode='committed sparse latent')

def validate_output(row,expected):
    if (row.get('shape')!=expected['output_shape'] or row.get('dtype')!='torch.float32'
        or row.get('device')!='cuda:0' or row.get('requires_grad') is not False
        or row.get('inference_mode') is not True or row.get('all_finite') is not True or row.get('count')!=262144):
        raise ValueError('Native occupancy logits failed shape/dtype/device/finiteness contract')
    return {key:row[key] for key in ('count','all_finite','minimum','maximum','mean','stddev')}

def classify(record,process_ok,healthy,preserved):
    try:
        validate_output(record['forward']['output'],fixture(CONFIG));validate_release(record['release'])
        if not record['release']['input_collected'] or not record['release']['output_collected']:return 'PARTIAL'
        artifact=record['occupancy_artifacts']
        if artifact['readback_verified'] is not True or artifact['empty'] or artifact['degenerate']:return 'PARTIAL'
    except (KeyError,ValueError,TypeError):return 'PARTIAL'
    return 'PASS' if (record['status']=='PASS' and record['stage']=='complete' and record['load_calls']==record['forward_calls']==record['root_decoder_calls']==1
        and record['transfer_calls']=={'input':1,'output':1} and record.get('workspace_clear_calls')==1
        and record.get('workspace_binding',{}).get('verified') is True
        and record.get('cpu_load',{}).get('status')=='PASS' and record.get('fixture')==fixture(CONFIG)
        and record['forward'].get('grad_enabled') is False and record['forward'].get('output_is_inference_tensor') is True
        and process_ok and healthy and preserved) else 'PARTIAL'
