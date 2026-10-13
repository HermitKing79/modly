"""One source-faithful shape-SLat512 call; no runtime/framework imports."""
import copy
import amd_sparse_flow_contract as old
CPU_SECONDS,GPU_SECONDS,OVERALL_SECONDS=180,240,480
MEMORY_LIMIT=12*1024**3
SEED=20261013
TIMESTEP=1000.0
CONFIG=copy.deepcopy(old.CONFIG)
CONFIG['name']='SLatFlowModel'
CONFIG['args'].update(resolution=32,in_channels=32,out_channels=32)
SingleForward=old.SingleForward
prohibited_import=old.prohibited_import
def schema():
    result=old.schema()
    result['input_layer.weight']['shape']=[1536,32]
    result['out_layer.weight']['shape']=[32,1536]
    result['out_layer.bias']['shape']=[32]
    return result
def validate_header(config,tensors):
    if config!=CONFIG or set(tensors)!=set(schema()):raise ValueError('Exact shape-SLat architecture/640 keys mismatch')
    for k,v in schema().items():
        if tensors[k]['shape']!=v['shape'] or tensors[k]['dtype']!='BF16':raise ValueError('Checkpoint shape/BF16 mismatch: '+k)
def validate_manifest(parameters,buffers,device):
    if buffers or set(parameters)!=set(schema()):raise ValueError('Shape-SLat parameter/buffer keys mismatch')
    for k,v in schema().items():
        if parameters[k]!={**v,'device':device,'is_meta':device=='meta'}:raise ValueError('Runtime shape/dtype/device mismatch: '+k)
def validate_coords(coords,count):
    import numpy as np
    if (type(coords) is not np.ndarray or coords.dtype!=np.int32 or coords.shape!=(count,4) or not 0<count<=32768
        or np.any(coords[:,0]!=0) or np.any(coords[:,1:]<0) or np.any(coords[:,1:]>31)):
        raise ValueError('Coordinates require bounded unique int32 [batch,D,H,W]')
    linear=(coords[:,1].astype(np.int64)*32+coords[:,2])*32+coords[:,3]
    if np.any(np.diff(linear)<=0):raise ValueError('Coordinates must be unique and lexicographically ordered')
def reduce_occupancy(logits,mask):
    import numpy as np
    if (type(logits) is not np.ndarray or logits.shape!=(1,1,64,64,64) or logits.dtype!=np.float32
        or not np.isfinite(logits).all() or type(mask) is not np.ndarray or mask.shape!=(64,64,64)
        or mask.dtype!=np.bool_ or not np.array_equal(logits[0,0]>0,mask)):
        raise ValueError('Saved native logits/mask mismatch')
    # Exact max_pool3d((decoded>0).float(),2,2,0)>0.5; no averaging/logit pooling.
    pooled=mask.reshape(32,2,32,2,32,2).max(axis=(1,3,5))
    xyz=np.argwhere(pooled).astype(np.int32)
    coords=np.concatenate((np.zeros((len(xyz),1),np.int32),xyz),axis=1)
    if len(coords):validate_coords(coords,len(coords))
    return pooled,coords
def validate_transfer(key,row,target,phase,n):
    shapes={'x':([n,32],'torch.float32'),'coords':([n,4],'torch.int32'),'batch_map':([n],'torch.int64'),
            't':([1],'torch.float32'),'cond':([1,1029,1024],'torch.float32'),'output':([n,32],'torch.float32'),'output_coords':([n,4],'torch.int32')}
    if key not in shapes:raise ValueError('Foreign transfer identity')
    shape,dtype=shapes[key];out=key in ('output','output_coords')
    if row!=dict(shape=shape,dtype=dtype,device='cuda:0' if out else 'cpu') or target!=('cpu' if out else 'cuda:0') or phase!=('output' if out else 'inputs'):
        raise ValueError('Exact shape-SLat transfer mismatch')
def authorize_placement(gate,expected,checkpoint_sha256,worker_sha256):
    if (gate.get('status')!='PASS' or gate.get('gpu_initialized') is not False or gate.get('strict') is not True
        or gate.get('assign') is not True or gate.get('missing_keys')!=[] or gate.get('unexpected_keys')!=[]
        or gate.get('parameter_bytes')!=expected['tensor_data_bytes'] or worker_sha256!=checkpoint_sha256
        or gate.get('payload_entries')!=640 or gate.get('reversible_widening_count')!=220
        or gate.get('feature_verified') is not True or gate.get('input_token_count')!=expected['token_count']):raise ValueError('Shape-SLat CPU gate incomplete')
    validate_header(expected['config'],expected['tensors']);validate_manifest(gate['tensors'],gate['buffers'],'cpu')
def validate_output(row,n):
    expected=dict(shape=[n,32],dtype='torch.float32',device='cuda:0',all_finite=True,count=n*32,
        requires_grad=False,is_inference_tensor=True,grad_enabled=False,coords_unchanged=True,sparse_shape=[1,32])
    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Shape-SLat numerical/structure output mismatch')
def classify(record,process_ok,healthy,preserved):
    try:
        validate_output(record.get('output',{}),record['token_count'])
        from amd_decoder_contract import validate_release
        validate_release(record.get('release',{}))
        calls={'<root>':1,**{f'blocks.{i}':1 for i in range(30)},**{f'blocks.{i}.{a}':1 for i in range(30) for a in ('self_attn','cross_attn')}}
        if record.get('calls')!=calls:raise ValueError('Genuine module counts mismatch')
    except (ValueError,KeyError):return 'PARTIAL'
    transfers={k:1 for k in ('x','coords','batch_map','t','cond','output','output_coords','frequency','rope_frequency')}
    return 'PASS' if (record.get('status')=='PASS' and record.get('stage')=='complete' and record.get('load_calls')==1
        and record.get('forward_calls')==1 and record.get('workspace_clear_calls')==1 and record.get('transfers')==transfers
        and record.get('release',{}).get('owned_tensors_collected') is True and record.get('workspace_binding',{}).get('verified') is True
        and process_ok and healthy and preserved) else 'PARTIAL'
