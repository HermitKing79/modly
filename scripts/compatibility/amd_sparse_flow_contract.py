"""Explicit source-derived contracts for the single sparse-flow computation proof."""
CONFIG={'name':'SparseStructureFlowModel','args':dict(resolution=16,in_channels=8,out_channels=8,
    model_channels=1536,cond_channels=1024,num_blocks=30,num_heads=12,mlp_ratio=5.3334,
    pe_mode='rope',share_mod=True,initialization='scaled',qk_rms_norm=True,qk_rms_norm_cross=True,dtype='bfloat16')}
CPU_SECONDS=180
GPU_SECONDS=240
OVERALL_SECONDS=480
MEMORY_LIMIT=12*1024**3

def schema():
    result={}
    def pair(prefix,shape,dtype):
        result[prefix+'.weight']={'shape':shape,'dtype':dtype}
        result[prefix+'.bias']={'shape':[shape[0]],'dtype':dtype}
    for name,shape in [('t_embedder.mlp.0',[1536,256]),('t_embedder.mlp.2',[1536,1536]),
                       ('adaLN_modulation.1',[9216,1536]),('input_layer',[1536,8]),('out_layer',[8,1536])]:
        pair(name,shape,'F32')
    for i in range(30):
        p=f'blocks.{i}.'
        result[p+'modulation']={'shape':[9216],'dtype':'F32'}
        for name in ('weight','bias'):result[p+'norm2.'+name]={'shape':[1536],'dtype':'F32'}
        for attn in ('self_attn','cross_attn'):
            for qk in ('q','k'):result[p+attn+'.'+qk+'_rms_norm.gamma']={'shape':[12,128],'dtype':'F32'}
        for name,shape in [('self_attn.to_qkv',[4608,1536]),('self_attn.to_out',[1536,1536]),
                           ('cross_attn.to_q',[1536,1536]),('cross_attn.to_kv',[3072,1024]),
                           ('cross_attn.to_out',[1536,1536]),('mlp.mlp.0',[8192,1536]),('mlp.mlp.2',[1536,8192])]:
            pair(p+name,shape,'BF16')
    return result

def validate_header(config,tensors):
    if config!=CONFIG:raise ValueError('Sparse-flow configuration differs from exact audited architecture')
    expected=schema()
    if set(tensors)!=set(expected):raise ValueError('Sparse-flow checkpoint keys differ from exact640 parameter schema')
    for key,row in expected.items():
        if tensors[key]['shape']!=row['shape'] or tensors[key]['dtype']!='BF16':
            raise ValueError('Sparse-flow checkpoint shape/storage dtype mismatch: '+key)

def validate_manifest(parameters,buffers,device):
    expected=schema()
    if set(parameters)!=set(expected):raise ValueError('Runtime parameter keys differ')
    for key,row in expected.items():
        if parameters[key]!={**row,'device':device,'is_meta':device=='meta'}:
            raise ValueError('Runtime parameter contract mismatch: '+key)
    if buffers!={'rope_phases':{'shape':[4096,64],'dtype':'C64','device':'cpu' if device=='meta' else device,'is_meta':False}}:
        raise ValueError('Generated RoPE buffer contract mismatch')

class SingleForward:
    def __init__(self):self.calls=0
    def run(self,operation):
        if self.calls:raise RuntimeError('Sparse-flow forward permit already consumed')
        self.calls=1
        return operation()

def validate_output(row):
    expected=dict(shape=[1,8,16,16,16],dtype='torch.float32',device='cuda:0',all_finite=True,
                  count=32768,requires_grad=False,is_inference_tensor=True,grad_enabled=False)
    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Sparse-flow output contract failed')

def authorize_placement(gate,expected,checkpoint_sha256,worker_sha256):
    if (gate.get('status')!='PASS' or gate.get('gpu_initialized') is not False
        or gate.get('strict') is not True or gate.get('assign') is not True
        or gate.get('missing_keys')!=[] or gate.get('unexpected_keys')!=[]
        or gate.get('parameter_bytes')!=expected['tensor_data_bytes'] or worker_sha256!=checkpoint_sha256
        or gate.get('payload_entries')!=640 or gate.get('reversible_widening_count')!=220
        or gate.get('feature_verified') is not True):raise ValueError('Sparse-flow CPU gate is incomplete')
    validate_header(expected['config'],expected['tensors'])
    validate_manifest(gate.get('tensors',{}),gate.get('buffers',{}),'cpu')

def prohibited_import(name):
    # trellis2 imports this inert lazy package initializer; no pipeline classes.
    return name.split('.')[0] in ('cumesh','o_voxel','flex_gemm','nvdiffrast') or name.startswith('trellis2.pipelines.')

def validate_transfer(key,description,target,phase):
    shapes={'x':[1,8,16,16,16],'t':[1],'cond':[1,1029,1024],'output':[1,8,16,16,16]}
    if key not in shapes:raise ValueError('Unapproved sparse-flow transfer identity')
    source='cuda:0' if key=='output' else 'cpu'
    if description!={'shape':shapes[key],'dtype':'torch.float32','device':source} or target!=('cpu' if key=='output' else 'cuda:0') or phase!=('output' if key=='output' else 'inputs'):
        raise ValueError('Unapproved sparse-flow transfer contract')

def classify(record,process_ok,healthy,preserved):
    try:
        validate_output(record.get('output',{}))
        from amd_decoder_contract import validate_release
        validate_release(record.get('release',{}))
        expected={'<root>':1,**{f'blocks.{i}':1 for i in range(30)},
                  **{f'blocks.{i}.{a}':1 for i in range(30) for a in ('self_attn','cross_attn')}}
        if record.get('calls')!=expected:raise ValueError('Model/block/attention call counts differ')
    except ValueError:return 'PARTIAL'
    return 'PASS' if (record.get('status')=='PASS' and record.get('stage')=='complete'
        and record.get('load_calls')==1 and record.get('forward_calls')==1 and record.get('workspace_clear_calls')==1
        and record.get('transfers')=={'x':1,'t':1,'cond':1,'output':1,'frequency':1}
        and record['release'].get('owned_tensors_collected') is True
        and record.get('workspace_binding',{}).get('verified') is True
        and process_ok is True and healthy is True and preserved is True) else 'PARTIAL'
