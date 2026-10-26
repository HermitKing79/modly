"""Synthetic operators only. Stdlib at import; no work without owned GO/EOF."""
import __future__,argparse,ast,gc,hashlib,json,math,os,sys,time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional,Tuple,Union
sys.path.insert(0,str(Path(__file__).parent))
import amd_pixal3d_operator_common_v3 as c

def extract(path,names,namespace):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes}!=names:raise ValueError('Exact pinned operator definitions required')
    target=dict(namespace)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),target)
    return {name:target[name] for name in names}

def attention_reference(q,k,v,scale):
    result=[]
    for query in q:
        scores=[sum(x*y for x,y in zip(query,key))*scale for key in k];maximum=max(scores)
        exp=[math.exp(s-maximum) for s in scores];total=sum(exp)
        result.append([sum(exp[i]*v[i][j] for i in range(len(v)))/total for j in range(len(v[0]))])
    return result

def hadamard(size):
    if size<4 or size&(size-1) or round(math.log(size,4))!=math.log(size,4):raise ValueError('Power-of-four regular Hadamard required')
    base=[[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]];h=base
    while len(h)<size:h=[[base[i//len(h)][j//len(h)]*h[i%len(h)][j%len(h)] for j in range(len(h)*4)] for i in range(len(h)*4)]
    return [[v/math.sqrt(size) for v in row] for row in h]

def assert_values(actual,expected,atol,rtol):
    if len(actual)!=len(expected) or not actual:raise RuntimeError('Reference shape/length mismatch')
    errors=[]
    for a,b in zip(actual,expected):
        if not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>atol+rtol*abs(b):raise RuntimeError('Nonfinite or CPU-reference numerical failure')
        errors.append(abs(a-b))
    return dict(max_abs_error=max(errors),atol=atol,rtol=rtol,values=len(actual))

def require_witness(device,elapsed_ms,calls,selected_HIP):
    if device!='cuda' or elapsed_ms<=0 or not selected_HIP or not any(v>0 and name.startswith('int8_gemm') for name,v in calls.items()):
        raise RuntimeError('Missing native HIP/GPU execution witness; fallback rejected')

def quantize(rows):
    q=[];scales=[]
    for row in rows:
        scale=max(map(abs,row))/127 or 1.;scales.append(scale)
        q.append([max(-127,min(127,round(value/scale))) for value in row])
    return q,scales

def rotate(rows,h):return [[sum(value*column for value,column in zip(row,col)) for col in zip(*h)] for row in rows]

def policy(root,case):
    def audit(event,args):
        if event in ('subprocess.Popen','os.system','os.posix_spawn','os.fork','socket.connect','socket.bind'):raise RuntimeError('No compile, subprocess or network in operator worker')
        if event=='import':
            name=str(args[0])
            if name=='triton' or name.startswith('triton.'):raise ImportError('Global Triton disabled for this proof')
            if name=='comfy_kitchen.backends.cuda._C':raise ImportError('CUDA native extension disabled; HIP only')
            if name.split('.')[0] in ('pip','transformers','diffusers','trellis2','comfy','huggingface_hub','natten','spconv','nvdiffrast'):raise ImportError('Installer/model/foreign kernel module forbidden')
        if event=='ctypes.dlopen' and args[0] and any(n in str(args[0]).lower() for n in ('libcuda.so','libcublas','/nvidia/')):raise OSError('CUDA library disabled')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0]));writing=any(v in str(args[1] or '') for v in 'wax+') or bool(args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing:
                path=c.plain(path)
                if not (path.is_relative_to(root) or path.is_relative_to(case)):raise RuntimeError('Operator write outside isolated roots')
            elif path.suffix.lower() in ('.safetensors','.pt','.pth','.ckpt'):raise RuntimeError('No weights/checkpoints in model-free proof')
        if event=='os.mkdir':
            path=c.plain(args[0])
            if not(path.is_relative_to(root) or path.is_relative_to(case)):raise RuntimeError('Foreign directory creation')
        if event in ('os.remove','os.rename','os.rmdir','os.link','os.symlink','os.truncate'):raise RuntimeError('Destructive mutation forbidden in operator proof')
    return audit

def run(case,token):
    from amd_pixal3d_operator_guest_v3 import validate_case,baseline
    case=validate_case(case,token);cfg=c.lock();root=c.plain(cfg['guest_root']);leaf=case/'operators'
    if sys.stdin.buffer.read(4)!=b'GO\n' or os.environ.get('MODLY_PIXAL_PROOF_TOKEN')!=token:raise RuntimeError('Owned GO/exec nonce required before imports')
    if Path(sys.prefix)!=root/'.venv' or sys.version.split()[0]!=cfg['runtime']['python']:raise RuntimeError('Only the isolated qualified Python prefix')
    sys.addaudithook(policy(c.plain(cfg['temporary_guest_root']),case))
    result=dict(classification='FAIL',operators={name:dict(status='NOT_RUN') for name in c.OPERATORS},no_weights_or_models=True)
    result['worker_identity']=dict(pid=os.getpid(),prefix=sys.prefix,python=sys.executable)
    before=baseline();torch=None;current='gfx1201'
    def phase(name,kind):c.journal(leaf/'operator-phases.jsonl',name,kind=kind,monotonic_start=time.monotonic())
    def tensor_check(actual,expected,atol,rtol):
        if actual.device.type!='cuda':raise RuntimeError('GPU output unexpectedly fell back to CPU')
        return assert_values(actual.detach().float().cpu().flatten().tolist(),expected,atol,rtol)
    try:
        phase('imports','CPU')
        import importlib.metadata
        import torch
        from torch.nn import functional as F
        rt=cfg['runtime']
        if (torch.__version__!=rt['torch'] or importlib.metadata.version('torch')!=rt['distribution'] or torch.version.git_version!=rt['git']
            or torch.version.hip!=rt['hip'] or torch.version.cuda is not None
            or c.digest(Path(torch.__file__).with_name('version.py'))!=rt['torch_version_file_sha256']):raise RuntimeError('Qualified ROCm/Torch identity differs')
        if not Path(torch.__file__).resolve().is_relative_to(root/'.venv'):raise RuntimeError('Shared/baseline Torch import forbidden')
        if torch.cuda.is_initialized():raise RuntimeError('Unexpected GPU initialization during Torch import')
        def forbidden(*args,**kwargs):raise RuntimeError('Model load, compiler or fallback forbidden')
        torch.load=forbidden;torch.jit.load=forbidden;torch.compile=forbidden;torch.hub.load=forbidden
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        phase('device-admission','GPU')
        if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one unmasked visible GPU required')
        properties=torch.cuda.get_device_properties(0)
        if properties.gcnArchName.split(':')[0]!='gfx1201' or '9070 XT' not in properties.name:raise RuntimeError('Exact RX9070XT gfx1201 required')
        free,total=torch.cuda.mem_get_info()
        if free<cfg['guards']['initial_free_VRAM_bytes']:raise RuntimeError('Original initial free-VRAM gate failed')
        torch.cuda.set_per_process_memory_fraction(cfg['guards']['torch_allocator_cap_bytes']/total,0);torch.cuda.reset_peak_memory_stats()
        result['operators']['gfx1201']=dict(status='PASS',name=properties.name,arch=properties.gcnArchName,visible_devices=1,free_bytes=free,total_bytes=total)
        c.save(leaf/'gfx1201.json',result['operators']['gfx1201'])
        current='HIP_extension';phase('Kitchen-HIP-load','GPU')
        import comfy_kitchen as ck
        from comfy_kitchen.backends import hip
        for name in ('cuda','triton','eager'):ck.disable_backend(name)
        ck.set_backend_priority(['hip'])
        if not hip._EXT_AVAILABLE or not ck.registry.is_available('hip') or hip._C is None:raise RuntimeError('Genuine HIP extension unavailable: '+str(hip._EXT_ERROR))
        ext=Path(hip._C.__file__).resolve()
        if not ext.is_relative_to(root/'.venv') or c.digest(Path(hip.__file__))!=cfg['kitchen_HIP_source_sha256']:raise RuntimeError('Unexpected HIP source/extension origin')
        payload=json.loads((case/'payload.json').read_text())['hip_payload']
        if c.digest(ext)!=payload['extension_sha256']:raise RuntimeError('Loaded HIP ELF differs from approved wheel')
        calls={};native=hip._C
        class NativeWitness:
            def __getattr__(self,name):
                value=getattr(native,name)
                if not callable(value):return value
                def call(*args,**kwargs):calls[name]=calls.get(name,0)+1;return value(*args,**kwargs)
                return call
        hip._C=NativeWitness()
        hip._eager.int8_linear=forbidden;hip._eager.quantize_and_rotate_rowwise=forbidden;hip._eager.quantize_int8_convrot_weight=forbidden
        result['operators'][current]=dict(status='PASS',extension_sha256=payload['extension_sha256'],backend='hip',CUDA_and_global_Triton_disabled=True)
        c.save(leaf/'HIP_extension.json',result['operators'][current]);current=None
        ns=dict(torch=torch,F=F,Optional=Optional,Tuple=Tuple,Union=Union,_CHUNK_BYTES=2**29,
                comfy=SimpleNamespace(model_management=SimpleNamespace(get_free_memory=lambda device:torch.cuda.mem_get_info(device)[0])))
        for row in cfg['sources']:c.verify(c.inside(root/'ComfyUI',row['path']),row)
        flex=extract(root/'ComfyUI/comfy/ldm/trellis2/flexgemm.py',{'TorchHashMap','compute_kernel_offsets','build_submanifold_neighbor_map','get_recommended_chunk_mem','sparse_submanifold_conv3d'},ns)
        # All extracted definitions share their original globals in their compiled namespace.
        naf=extract(root/'ComfyUI/comfy/image_encoders/naf.py',{'na2d_pure'},ns)['na2d_pure']
        conversion=extract(root/'ComfyUI/comfy/ldm/trellis2/vae.py',{'flexible_dual_grid_to_mesh'},dict(ns,TorchHashMap=flex['TorchHashMap']))['flexible_dual_grid_to_mesh']

        def measure(operation):
            torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();value=operation();b.record();torch.cuda.synchronize()
            return value,a.elapsed_time(b)

        def timed(name,operation):
            nonlocal current
            current=name;phase(name,'GPU')
            if torch.cuda.mem_get_info()[0]<cfg['guards']['live_shape_free_VRAM_bytes']:raise RuntimeError('Original live free-VRAM floor failed')
            prior=dict(calls);began=time.monotonic();metrics,device,elapsed=operation()
            if device!='cuda' or elapsed<=0:raise RuntimeError('No actual HIP-resident timed result')
            if torch.cuda.mem_get_info()[0]<cfg['guards']['live_shape_free_VRAM_bytes']:raise RuntimeError('Original live free-VRAM floor failed after operation')
            metrics.update(status='PASS',output_device=device,HIP_event_ms=elapsed,fixture_and_reference_wall_seconds=time.monotonic()-began,native_calls={k:v-prior.get(k,0) for k,v in calls.items() if v-prior.get(k,0)})
            if name.startswith('HIP_'):
                require_witness(device,elapsed,metrics['native_calls'],metrics.pop('selected_HIP'))
                if name=='HIP_ConvRot' and metrics['native_calls'].get('quantize_int8_convrot',0)==0:raise RuntimeError('ConvRot native producer not executed')
            result['operators'][name]=metrics
            c.save(leaf/(name+'.json'),metrics)

        def linear(convrot):
            x=[[(i*13+j*7)%15/8-7/8 for j in range(256)] for i in range(64)]
            if convrot:
                w=[[(i*3+j*11)%17/16-8/16 for j in range(256)] for i in range(64)]
                h=hadamard(256);xr=rotate(x,h);wr=rotate(w,h);wq,ws=quantize(wr)
            else:xr=x;wq=[[(i*3+j*11)%15-7 for j in range(256)] for i in range(64)];ws=[1/32]*64
            xq,xs=quantize(xr)
            reference=[sum(a*b for a,b in zip(xq[i],wq[j]))*xs[i]*ws[j] for i in range(64) for j in range(64)]
            kwargs=dict(x=torch.tensor(x,device='cuda',dtype=torch.bfloat16),weight=torch.tensor(wq,device='cuda',dtype=torch.int8),
                        weight_scale=torch.tensor(ws,device='cuda',dtype=torch.float32),out_dtype=torch.bfloat16,convrot=convrot,convrot_groupsize=256)
            implementation=ck.registry.get_implementation('int8_linear',backend='hip',kwargs=kwargs)
            selected=implementation.__module__=='comfy_kitchen.backends.hip'
            with ck.use_backend('hip'):actual,elapsed=measure(lambda:ck.int8_linear(**kwargs))
            metrics=tensor_check(actual,reference,.02,.02);metrics.update(selected_HIP=selected,reference='Independent Python row quantization/regular-Hadamard/integer dot product',dtype='bfloat16')
            return metrics,actual.device.type,elapsed

        def sdpa():
            q=[[((i*3+j)%7-3)/8 for j in range(4)] for i in range(4)];k=[[((i+j*5)%9-4)/8 for j in range(4)] for i in range(4)];v=[[((i*7+j)%11-5)/8 for j in range(4)] for i in range(4)]
            reference=[n for row in attention_reference(q,k,v,.5) for n in row]
            operands=[torch.tensor(z,device='cuda',dtype=torch.float16)[None,None] for z in (q,k,v)]
            from torch.nn.attention import SDPBackend,sdpa_kernel
            with sdpa_kernel(SDPBackend.MATH):actual,elapsed=measure(lambda:F.scaled_dot_product_attention(*operands))
            return dict(tensor_check(actual,reference,.002,.002),dtype='float16',backend='PyTorch GPU math SDPA; flash/memory-efficient SDPA not qualified'),actual.device.type,elapsed

        def naf_probe():
            q=torch.tensor([((i*3)%13-6)/16 for i in range(64)],dtype=torch.float32).reshape(1,2,2,4,4)
            k=torch.tensor([((i*5)%11-5)/16 for i in range(16)],dtype=torch.float32).reshape(1,4,2,2)
            v=torch.tensor([((i*7)%17-8)/16 for i in range(16)],dtype=torch.float32).reshape(1,4,2,2)
            expected=[]
            for head in range(2):
                outputs={}
                for y in range(4):
                    for x in range(4):
                        keys=[];values=[]
                        for dy in (-1,0,1):
                            for dx in (-1,0,1):
                                yy=y//2+dy;xx=x//2+dx;valid=0<=yy<2 and 0<=xx<2
                                keys.append([float(k[0,head*2+d,yy,xx]) if valid else 0. for d in range(2)])
                                values.append([float(v[0,head*2+d,yy,xx]) if valid else 0. for d in range(2)])
                        outputs[y,x]=attention_reference([[float(q[0,head,d,y,x]) for d in range(2)]],keys,values,2**-.5)[0]
                expected.extend(outputs[y,x][d] for d in range(2) for y in range(4) for x in range(4))
            q,k,v=[value.to(device='cuda',dtype=torch.float16) for value in (q,k,v)];actual,elapsed=measure(lambda:naf(q,k,v,(3,3),2**-.5))
            return dict(tensor_check(actual,expected,.002,.002),dtype='float16',reference='Independent zero-padded LR-neighborhood Python attention loops'),actual.device.type,elapsed

        def sparse_probe():
            coords=[[0,x,y,z] for x in range(2) for y in range(2) for z in range(2)];feats=[[((i*3+j)%9-4)/8 for j in range(4)] for i in range(8)]
            weight=torch.tensor([((i*7)%13-6)/16 for i in range(3*27*4)],dtype=torch.float32).reshape(3,3,3,3,4);bias=[1/8,2/8,3/8];lookup={tuple(p):i for i,p in enumerate(coords)};expected=[]
            for coord in coords:
                for out in range(3):
                    value=bias[out]
                    for x in range(3):
                        for y in range(3):
                            for z in range(3):
                                index=lookup.get((0,coord[1]+x-1,coord[2]+y-1,coord[3]+z-1))
                                if index is not None:value+=sum(feats[index][i]*float(weight[out,x,y,z,i]) for i in range(4))
                    expected.append(value)
            arguments=(torch.tensor(feats,device='cuda',dtype=torch.bfloat16),torch.tensor(coords,device='cuda',dtype=torch.int32),(2,2,2),weight.to(device='cuda',dtype=torch.bfloat16),torch.tensor(bias,device='cuda',dtype=torch.bfloat16),None,(1,1,1))
            (actual,_),elapsed=measure(lambda:flex['sparse_submanifold_conv3d'](*arguments))
            return dict(tensor_check(actual,expected,.02,.02),dtype='bfloat16',source='Unmodified pinned native PyTorch sparse gather/GEMM function'),actual.device.type,elapsed

        def convert_probe():
            coords=[[0,0,0],[0,0,1],[0,1,1],[0,1,0]]
            arguments=(torch.tensor(coords,device='cuda',dtype=torch.int32),torch.full((4,3),.5,device='cuda'),torch.tensor([[True,False,False]]+[[False]*3]*3,device='cuda'),torch.ones(4,device='cuda'),[[-1,-1,-1],[1,1,1]])
            (vertices,faces),elapsed=measure(lambda:conversion(*arguments,grid_size=2))
            expected=[float(n)-.5 for row in coords for n in row]
            metrics=tensor_check(vertices,expected,0,0)
            if faces.device.type!='cuda' or faces.cpu().tolist()!=[[0,1,3],[3,1,2]]:raise RuntimeError('Independent tiny dual-grid face reference differs')
            return dict(metrics,vertices=4,triangles=2,reference='Independent literal quad split and unit-cell coordinates'),vertices.device.type,elapsed

        for name,function in [('HIP_INT8',lambda:linear(False)),('HIP_ConvRot',lambda:linear(True)),('PyTorch_SDPA',sdpa),('NAF',naf_probe),('native_sparse',sparse_probe),('native_conversion',convert_probe)]:timed(name,function)
        result['classification']='PASS'
    except BaseException as exc:
        result['error']=type(exc).__name__+': '+str(exc)
        if current is not None:result['operators'][current]=dict(status='FAIL',error=result['error'])
    finally:
        import resource
        result['Linux_worker_peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        try:
            phase('release','GPU')
            if torch is not None and torch.cuda.is_initialized():
                gc.collect();result['memory_peak']=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
                torch.cuda.empty_cache();torch.cuda.synchronize()
                result['memory_release']=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved());result['post_release_free_VRAM_bytes']=torch.cuda.mem_get_info()[0]
                if result['memory_release']!=dict(allocated=0,reserved=0):raise RuntimeError('Owned Torch allocation release incomplete')
                if result['post_release_free_VRAM_bytes']<cfg['guards']['live_shape_free_VRAM_bytes']:raise RuntimeError('Original live free-VRAM floor failed at release')
            if baseline()!=before:raise RuntimeError('Original runtime critical files changed')
        except BaseException as exc:result.update(classification='FAIL',cleanup_error=str(exc))
        c.save(leaf/'child-result.json',result)
    return 0 if result['classification']=='PASS' else 2

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--case');parser.add_argument('--token');args=parser.parse_args(argv)
    if not args.run:print('DRY_RUN: no Torch, GPU or native import');return 0
    if not args.case or not args.token:parser.error('Owned case and token required')
    return run(args.case,args.token)

if __name__=='__main__':raise SystemExit(main())
