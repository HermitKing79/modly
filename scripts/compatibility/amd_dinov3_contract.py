"""Fail-closed, CPU-testable policy for one DINOv3 conditioning experiment."""


def expected_shape(config):
    required=dict(model_type='dinov3_vit', architectures=['DINOv3ViTModel'], hidden_size=1024,
                  patch_size=16, num_channels=3, num_register_tokens=4, num_hidden_layers=24,
                  num_attention_heads=16, torch_dtype='float32')
    if any(config.get(key)!=value for key,value in required.items()):
        raise ValueError('DINOv3 configuration differs from the audited float32 ViT-L/16 architecture')
    return [1,1+config['num_register_tokens']+(512//config['patch_size'])**2,config['hidden_size']]


def validate_buffers(buffers):
    expected={'rope_embeddings.inv_freq':{'shape':[16],'dtype':'F32','device':'cpu','is_meta':False}}
    if buffers!=expected:
        raise ValueError('Unverified DINOv3 nonpersistent-buffer initialization: '+repr(buffers))


def validate_transfer(kind, source, target, options, phase):
    expected=({'shape':[1,3,512,512], 'dtype':'torch.float32','device':'cpu'},'cuda:0','conditioning') if kind=='input' else (
        {'shape':[1,1029,1024], 'dtype':'torch.float32','device':'cuda:0'},'cpu','output')
    if kind not in ('input','output') or (source,target,phase)!=expected or options:
        raise ValueError('Unapproved DINOv3 tensor transfer: '+repr((kind,source,target,options,phase)))


class SingleConditioning:
    def __init__(self):self.calls=0
    def run(self,operation):
        if self.calls:raise RuntimeError('DINOv3 conditioning attempt already consumed')
        self.calls=1
        return operation()


def validate_counts(counts):
    expected={'extractor':1,'features':1,'embeddings':1,'rope_embeddings':1,
              **{'layer.'+str(i):1 for i in range(24)}}
    if counts!=expected:raise ValueError('DINOv3 execution count differs: '+repr(counts))


def validate_output(result):
    expected={'shape':[1,1029,1024],'dtype':'torch.float32','device':'cuda:0',
              'all_finite':True,'count':1053696,'conditioning_width':1024}
    if any(result.get(key)!=value for key,value in expected.items()):
        raise ValueError('Invalid DINOv3 conditioning output: '+repr(result))


def classify(record, process_ok, healthy, preserved):
    try:
        validate_counts(record.get('calls',{}));validate_output(record.get('output',{}))
        from amd_decoder_contract import validate_release
        validate_release(record.get('release',{}))
        if record['release'].get('rope_cache_entries_after')!=0:raise ValueError('DINO cache retained')
    except (ValueError,KeyError):return 'PARTIAL'
    return 'PASS' if (record.get('status')=='PASS' and record.get('stage')=='complete'
        and type(record.get('load_calls')) is int and record['load_calls']==1
        and type(record.get('conditioning_calls')) is int and record['conditioning_calls']==1
        and record.get('transfer_calls')=={'input':1,'output':1}
        and record.get('normalization_copies')=={'mean':1,'std':1}
        and all(type(v) is int for v in record['transfer_calls'].values())
        and record.get('cpu_load',{}).get('status')=='PASS'
        and record['output'].get('requires_grad') is False and record['output'].get('grad_enabled') is False
        and record['output'].get('is_inference_tensor') is True and record['release'].get('output_collected') is True
        and process_ok is True and healthy is True and preserved is True) else 'PARTIAL'
