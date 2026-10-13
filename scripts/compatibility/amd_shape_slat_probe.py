"""Bounded CPU-only real metadata construction; reuse the existing owned probe supervisor."""
import argparse, hashlib, json, os, sys, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_shape_slat_local as local
import amd_blaslt_teardown as teardown
from services.amd_offline_guard import block_external_actions
def worker(scratch):
    record=dict(status='FAIL',weights_loaded=False,real_model_forward_calls=0,gpu_initialized=False,sdk_discovery=[])
    try:
        if sys.stdin.readline().strip()!='OWNED_JOB_ATTACHED':raise RuntimeError('Owned Job required')
        scratch=scratch.resolve();scratch.relative_to(local.EVIDENCE.resolve())
        local.verify_sources();teardown.verify_workspace();teardown.verify_binding_evidence()
        expected=local.identity(hash_weights=False)
        env=local.loading.gpu_proof.scratch_environment(scratch)
        for key in local.loading.gpu_proof.CACHE_KEYS:
            if key!='FLEX_GEMM_AUTOTUNE_CACHE_PATH':Path(env[key]).mkdir(parents=True,exist_ok=True)
        env.update(ATTN_BACKEND='sdpa',WORKSPACE_DIR=str(local.ROOT));os.environ.update(env)
        block_external_actions(local.VENV/'Scripts/rocm-sdk.exe',record['sdk_discovery'],scratch)
        phase='metadata'
        def forbidden(*a,**k):raise RuntimeError('CPU proof prohibits GPU/model forward/remote factory/compilation')
        def audit(event,args):
            if event.startswith('socket.'):return forbidden()
            if event=='import' and local.contract.prohibited_import(str(args[0])):return forbidden()
            if event=='open' and isinstance(args[0],(str,bytes)):
                p=Path(os.fsdecode(args[0]))
                if p.suffix.lower() in ('.safetensors','.pt','.pth','.bin','.gguf','.ckpt'):
                    allowed=p.resolve() in (local.FEATURE.resolve(),scratch/'tiny.safetensors')
                    allowed=allowed or phase=='hash-after-contract' and p.resolve()==local.PREFIX.with_suffix('.safetensors').resolve()
                    if not allowed:return forbidden()
        sys.addaudithook(audit)
        import torch
        if torch.cuda.is_initialized():return forbidden()
        torch.cuda._lazy_init=forbidden;torch.load=forbidden;torch.compile=forbidden;torch.hub.load=forbidden;torch.set_num_threads(1)
        torch.nn.Module.__call__=forbidden
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_leaves
        class CPUOnly(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                if any(isinstance(v,torch.Tensor) and v.device.type not in ('cpu','meta') or isinstance(v,torch.device) and v.type not in ('cpu','meta') for v in tree_leaves((args,kwargs or {}))):return forbidden()
                return func(*args,**(kwargs or {}))
        with CPUOnly():
            with torch.device('meta'):model=local.factory(expected['config'])
            record['parameters']=local.manifest(dict(model.named_parameters()));record['buffers']=local.manifest(dict(model.named_buffers()))
            local.contract.validate_manifest(record['parameters'],record['buffers'],'meta')
            frequencies=[b.self_attn.rope.freqs for b in model.blocks]
            reference=1/(10000**(torch.arange(21,dtype=torch.float32)/21))
            if any(v.device.type!='cpu' or v.dtype!=torch.float32 or not torch.equal(v,reference) for v in frequencies):raise ValueError('Genuine ordinary RoPE frequencies mismatch')
            record.update(strict_header_match=True,frequency_sha256=hashlib.sha256(reference.numpy().tobytes()).hexdigest(),frequency_objects=30,
                class_path=__import__('inspect').getfile(type(model)),sparse_attention_backend='sdpa',sparse_convolution_backend='none',dtype_counts={'BF16':420,'F32':220})
            model=frequencies=None
            feature,mask,coords=local.verify_inputs()
            from trellis2.modules.sparse import SparseTensor
            sparse=SparseTensor(torch.zeros(len(coords),32,dtype=torch.float32),torch.from_numpy(coords))
            if sparse.shape!=torch.Size([1,32]) or sparse.layout!=[slice(0,len(coords))] or not torch.equal(sparse.batch_boardcast_map,torch.zeros(len(coords),dtype=torch.int64)):raise ValueError('Real SparseTensor layout/broadcast contract mismatch')
            pooled=torch.nn.functional.max_pool3d(torch.from_numpy(__import__('numpy').load(local.TURRET/'occupancy-mask.npy',allow_pickle=False)).float()[None,None],2,2,0)>.5
            if not torch.equal(torch.argwhere(pooled)[:,[0,2,3,4]].int(),torch.from_numpy(coords)):raise ValueError('NumPy pooling differs from actual PyTorch source contract')
            record.update(token_count=len(coords),feature=feature,coords_sha256=hashlib.sha256(coords.tobytes()).hexdigest(),cpu_sparse_layout_verified=True,pooling_torch_equal=True)
            from strict_loader import StrictLocalComponent
            import safetensors.torch
            path=scratch/'tiny.safetensors';config_path=scratch/'tiny.json';config_path.write_text('{}')
            def tiny_factory(config):return torch.nn.Linear(2,2,bias=False,dtype=torch.bfloat16)
            safetensors.torch.save_file({'weight':torch.tensor([[1,-2],[.5,4]],dtype=torch.bfloat16)},str(path))
            component=StrictLocalComponent(str(config_path),str(path),tiny_factory);tiny=component.load_cpu()
            local.old.widen(tiny,component.diagnostics['payload_sha256'],{'weight':{'shape':[2,2],'dtype':'F32'}})
            if tiny.weight.dtype!=torch.float32:raise ValueError('Strict synthetic widening failed')
            component.unload();tiny=component=None
            failures=[]
            for label,state in [('missing',{}),('unexpected',{'weight':torch.ones(2,2,dtype=torch.bfloat16),'extra':torch.ones(1)}),('shape',{'weight':torch.ones(1,2,dtype=torch.bfloat16)}),('dtype',{'weight':torch.ones(2,2,dtype=torch.float32)})]:
                safetensors.torch.save_file(state,str(path))
                component=None
                try:
                    component=StrictLocalComponent(str(config_path),str(path),tiny_factory)
                    component.load_cpu()
                except ValueError as e:failures.append({'case':label,'error':str(e)})
                else:raise ValueError('Invalid synthetic checkpoint accepted')
                if component is not None:component.unload()
                component=None
            record['synthetic']={'strict_positive':True,'negative':failures,'real_weights_loaded':False}
            phase='hash-after-contract';record['checkpoint_sha256']=local.digest(local.PREFIX.with_suffix('.safetensors'));phase='metadata'
            record['runtime']=dict(python=sys.version,torch=str(torch.__version__),hip=torch.version.hip,cuda=torch.version.cuda,torch_path=torch.__file__)
            if torch.cuda.is_initialized():return forbidden()
            record.update(status='PASS',gpu_initialized=False)
    except BaseException as e:record.update(error=repr(e),traceback=traceback.format_exc())
    print('FLOW_METADATA='+json.dumps(record),flush=True)
    return 0 if record['status']=='PASS' else 1
def supervise(output):
    import amd_sparse_flow_probe as proven
    proven.local=local;proven.__file__=__file__
    return proven.supervise(output)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    teardown.verify_workspace()
    raise SystemExit(worker(args.output) if args.worker else supervise(args.output))
