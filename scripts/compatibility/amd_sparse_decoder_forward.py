"""Exactly one offline sparse-decoder forward, with parent gates and durable logs."""
import argparse
import datetime
import gc
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback
import weakref

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'api'),str(Path(__file__).parent),str(ROOT/'experimental/extensions/trellis2-amd')]
import amd_sparse_decoder_loading as previous
import amd_decoder_contract as loading_contract
import amd_decoder_forward_contract as contract
from services.amd_offline_guard import block_external_actions
from services.amd_observer import windows_observation, healthy_observation

SOURCE,VENV,PREFIX=previous.SOURCE,previous.VENV,previous.PREFIX
EVIDENCE=ROOT/'docs/compatibility/amd-sparse-decoder-forward-evidence'
EVENT='FORWARD_EVENT='
RESULT='FORWARD_RESULT='
OLD=ROOT/'docs/compatibility/amd-sparse-decoder-loading-evidence/run/results.json'
COMMITTED_FILES=[
    'scripts/compatibility/'+name+'.py' for name in ('amd_sparse_decoder_forward','amd_decoder_forward_contract',
    'amd_sparse_decoder_loading','amd_decoder_contract','amd_bounded_gpu_proof','amd_native_import_preflight','amd_model_metadata')
] + ['api/services/'+name+'.py' for name in ('amd_runtime','amd_model_metadata','amd_offline_guard','amd_observer')] + [
    'experimental/extensions/trellis2-amd/strict_loader.py',
    OLD.relative_to(ROOT).as_posix(),previous.ARCHIVED.relative_to(ROOT).as_posix()]


def verify_committed_files(commit):
    if len(commit)!=40 or any(char not in '0123456789abcdef' for char in commit):
        raise ValueError('Supply the exact published harness commit SHA')
    hashes={}
    for name in COMMITTED_FILES:
        try:
            committed=subprocess.run(['git','show',commit+':'+name],cwd=ROOT,capture_output=True,check=True,timeout=5).stdout
        except subprocess.CalledProcessError as exc:
            raise RuntimeError('Required harness dependency is absent from the published commit: '+name) from exc
        working=(ROOT/name).read_bytes()
        # Git's Windows checkout conversion is the only permitted difference.
        if working.replace(b'\r\n',b'\n')!=committed.replace(b'\r\n',b'\n'):
            raise RuntimeError('Executed harness dependency differs from published commit: '+name)
        hashes[name]=hashlib.sha256(working).hexdigest()
    return hashes


def source_identity():
    rows=json.loads(OLD.read_text())['worker']['imported_source']
    for name,row in rows.items():
        path=Path(row['path']).resolve();path.relative_to(SOURCE.resolve())
        if previous.digest(path)!=row['sha256']:
            raise RuntimeError('Previously verified decoder support source changed: '+name)
    return rows


def worker(scratch):
    record={'status':'FAIL','stage':'startup','forward_calls':0,'sdk_discovery':[]}
    component=model=latent=output=cpu_output=cpu_input=torch=None
    permit=contract.SingleForward()
    phase='cpu'; allowed_modules=set();root_id=None;root_calls=0
    def stage(name):
        record['stage']=name
        print('FORWARD_STAGE='+name,flush=True)
    try:
        scratch=scratch.resolve();scratch.relative_to(EVIDENCE)
        env=previous.gpu_proof.scratch_environment(scratch)
        env['WORKSPACE_DIR']=str(scratch)
        for key in previous.gpu_proof.CACHE_KEYS:
            if key!='FLEX_GEMM_AUTOTUNE_CACHE_PATH':Path(env[key]).mkdir(parents=True,exist_ok=True)
        os.environ.update(env)
        record['sdk_verification']=block_external_actions(VENV/'Scripts/rocm-sdk.exe',record['sdk_discovery'],scratch)
        os.environ.update(env)
        checkpoint=PREFIX.with_suffix('.safetensors').resolve()
        def audit(event,args):
            if event.startswith('socket.'):
                raise RuntimeError('Forward proof prohibits network: '+event)
            if event=='import' and contract.prohibited_import(str(args[0])):
                raise RuntimeError('Unapproved pipeline/mesh import: '+str(args[0]))
            if event=='open' and isinstance(args[0],(str,bytes)):
                path=Path(os.fsdecode(args[0]))
                if path.suffix.lower() in ('.safetensors','.pt','.pth','.bin','.ckpt','.gguf') and path.resolve()!=checkpoint:
                    raise RuntimeError('Only the audited sparse decoder may be read: '+str(path))
        sys.addaudithook(audit)
        record['verified_support_source']=source_identity()
        verified=previous.identity();record['checkpoint_identity']=verified
        fixture=contract.fixture(verified['config']);record['fixture']=fixture
        import torch
        Path(sys.executable).resolve().relative_to(VENV)
        Path(torch.__file__).resolve().relative_to(VENV/'Lib/site-packages')
        if (sys.version.split()[0]!='3.12.10' or torch.__version__!='2.13.0+rocm10.0.0'
                or not torch.version.hip or torch.version.cuda is not None or torch.cuda.is_initialized()):
            raise RuntimeError('Unexpected AMD runtime or pre-existing GPU context')
        record['runtime']={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
                           'torch_path':torch.__file__,'hip':torch.version.hip,'cuda':torch.version.cuda}
        torch.set_num_threads(1)
        original_init=torch.cuda._lazy_init
        def forbidden(*args,**kwargs):raise RuntimeError('Unapproved model loading, initialization or compilation')
        torch.cuda._lazy_init=forbidden
        torch.load=forbidden;torch.compile=forbidden
        original_call=torch.nn.Module.__call__
        def module_call(instance,*args,**kwargs):
            nonlocal root_calls
            if phase!='forward' or id(instance) not in allowed_modules:
                raise RuntimeError('Only the verified decoder module tree may execute during the single forward')
            if id(instance)==root_id:
                root_calls+=1
                if root_calls!=1:raise RuntimeError('Duplicate decoder invocation')
            return original_call(instance,*args,**kwargs)
        torch.nn.Module.__call__=module_call
        from torch.utils._python_dispatch import TorchDispatchMode
        def has_gpu(value):
            if isinstance(value,torch.Tensor):return value.device.type=='cuda'
            if isinstance(value,torch.device):return value.type=='cuda'
            if isinstance(value,(list,tuple)):return any(has_gpu(item) for item in value)
            if isinstance(value,dict):return any(has_gpu(item) for item in value.values())
            return isinstance(value,str) and value.startswith('cuda')
        class BoundedDispatch(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                kwargs=kwargs or {}
                if (has_gpu(args) or has_gpu(kwargs)) and (phase=='cpu' or
                        (phase!='forward' and not contract.placement_operation(str(func)))):
                    raise RuntimeError('Unapproved GPU operation outside forward: '+str(func))
                return func(*args,**kwargs)
        with BoundedDispatch():
            stage('decoder-import')
            sys.path.insert(0,str(SOURCE))
            cls=importlib.import_module('trellis2.models.sparse_structure_vae').SparseStructureDecoder
            if Path(inspect.getfile(cls)).resolve()!=(SOURCE/'trellis2/models/sparse_structure_vae.py').resolve():
                raise RuntimeError('Decoder class provenance mismatch')
            from trellis2 import models
            models.from_pretrained=forbidden
            import triton
            triton.compile=forbidden;triton.runtime.JITFunction.run=forbidden;triton.runtime.Autotuner.run=forbidden
            import safetensors.torch
            original_load=safetensors.torch.load_file
            def load_only(path,device='cpu',**kwargs):
                if Path(path).resolve()!=checkpoint or device!='cpu' or kwargs or record.get('load_calls',0):
                    raise RuntimeError('Only one CPU load of the exact decoder is approved')
                record['load_calls']=1
                return original_load(str(checkpoint),device='cpu')
            safetensors.torch.load_file=load_only
            from strict_loader import StrictLocalComponent
            stage('strict-CPU-load');began=time.monotonic();cpu_before=previous.process_memory()
            component=StrictLocalComponent(str(PREFIX.with_suffix('.json')),str(checkpoint),lambda config:cls(**config['args']))
            model=component.load_cpu()
            manifest=previous.tensor_manifest(model.state_dict())
            loading_contract.validate_tensors(verified['tensors'],manifest,'cpu')
            if torch.cuda.is_initialized():raise RuntimeError('CPU loading initialized GPU')
            record['cpu_load']={'status':'PASS','elapsed_seconds':time.monotonic()-began,'strict':True,'assign':True,
                'missing_keys':[],'unexpected_keys':[],'gpu_initialized':False,'parameter_bytes':verified['tensor_data_bytes'],
                'tensors':manifest,'memory_before':cpu_before,'memory_after':previous.process_memory(),
                'loader_diagnostics':component.diagnostics.copy()}
            print(EVENT+json.dumps({'event':'cpu-ready','gate_one':record['cpu_load'],
                                   'checkpoint_sha256':verified['checkpoint_sha256']}),flush=True)
            stage('waiting-for-observer')
            if sys.stdin.readline().strip()!='EXECUTE_ONE_DECODER_FORWARD':
                raise RuntimeError('GPU forward not authorized by parent')
            phase='placement';torch.cuda._lazy_init=original_init
            stage('GPU-metadata')
            properties=torch.cuda.get_device_properties(0)
            if 'RX 9070 XT' not in properties.name or properties.gcnArchName.split(':')[0]!='gfx1201':
                raise RuntimeError('Unexpected AMD GPU')
            record['gpu']={'name':properties.name,'architecture':properties.gcnArchName,'total_bytes':properties.total_memory}
            torch.cuda.reset_peak_memory_stats(0)
            record['before_placement']=previous.gpu_memory(torch)
            if record['before_placement']['allocated_bytes'] or record['before_placement']['reserved_bytes'] or record['before_placement']['available_bytes']<verified['tensor_data_bytes']+512*1024**2:
                raise RuntimeError('Baseline allocation or insufficient VRAM')
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.use_deterministic_algorithms(True)
            record['determinism']={'algorithms':True,'cudnn_benchmark':False,'cudnn_deterministic':True}
            stage('decoder-placement');began=time.monotonic()
            model.to(device='cuda:0');torch.cuda.synchronize(0)
            loading_contract.validate_tensors(verified['tensors'],previous.tensor_manifest(model.state_dict()),'cuda:0')
            record['placement_seconds']=time.monotonic()-began
            with torch.inference_mode():
                cpu_input=torch.tensor(fixture['values'],dtype=torch.float32,device='cpu').reshape(fixture['input_shape'])
                latent=cpu_input.to('cuda:0');torch.cuda.synchronize(0)
                record['before_forward']=previous.gpu_memory(torch)
                allowed_modules={id(item) for item in model.modules()};root_id=id(model)
                stage('single-forward');phase='forward';began=time.monotonic()
                output=permit.invoke(lambda:model(latent))
                phase='placement';torch.cuda.synchronize(0)
                elapsed=time.monotonic()-began
                record['after_forward']=previous.gpu_memory(torch)
                stage('output-copy-validation')
                cpu_output=output.cpu();torch.cuda.synchronize(0)
                description={'shape':list(output.shape),'dtype':str(output.dtype),'device':str(output.device),
                    'requires_grad':output.requires_grad,'inference_mode':torch.is_inference_mode_enabled(),
                    'values':cpu_output.reshape(-1).tolist()}
                statistics=contract.validate_output(description,fixture)
                record['forward']={'status':'PASS','elapsed_seconds':elapsed,'input_shape':list(latent.shape),
                    'input_dtype':str(latent.dtype),'input_device':str(latent.device),'output':description,
                    'statistics':statistics,'output_sha256':hashlib.sha256(cpu_output.numpy().tobytes()).hexdigest(),
                    'grad_enabled':torch.is_grad_enabled(),'output_is_inference_tensor':output.is_inference()}
            record['forward_calls']=permit.calls
            if root_calls!=1 or permit.calls!=1:raise RuntimeError('Forward count invariant failed')
            stage('release')
            model_ref,input_ref,output_ref=weakref.ref(model),weakref.ref(latent),weakref.ref(output)
            component.unload();component=None;model=latent=output=cpu_output=cpu_input=None
            gc.collect();torch.cuda.synchronize(0);torch.cuda.empty_cache();torch.cuda.synchronize(0)
            record['release']={**previous.gpu_memory(torch),'model_collected':model_ref() is None,
                              'input_collected':input_ref() is None,'output_collected':output_ref() is None,
                              'cpu_memory':previous.process_memory()}
            loading_contract.validate_release(record['release'])
            if record['release']['input_collected'] is not True or record['release']['output_collected'] is not True:
                raise RuntimeError('Forward cleanup contract failed')
            stage('complete');record['status']='PASS'
    except BaseException as exc:
        record.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        record['forward_calls']=permit.calls;record['root_decoder_calls']=root_calls
        # After any native error, do not issue another GPU operation. Process exit
        # and parent reaping own cleanup; no retry or attempted GPU repair.
        if component is not None:component.unload()
        component=model=latent=output=cpu_output=cpu_input=None
        gc.collect()
        print(RESULT+json.dumps(record),flush=True)
    return 0 if record['status']=='PASS' else 1


def supervise(output,harness_commit):
    output=output.resolve()
    if output!=EVIDENCE/'run':raise ValueError('Use the single canonical evidence run directory')
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,check=True,timeout=5).stdout.decode().strip()
    if head!=harness_commit:raise RuntimeError('Run only the exact committed/published harness')
    committed_hashes=verify_committed_files(harness_commit)
    output.mkdir(parents=True,exist_ok=False)
    with (EVIDENCE/'attempt-started.json').open('x',encoding='utf-8') as stream:
        json.dump({'harness_commit':head,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'maximum_forwards':1},stream)
        stream.flush();os.fsync(stream.fileno())
    def save(name,value):
        with (output/name).open('x',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    since=datetime.datetime.now(datetime.timezone.utc).isoformat()
    record={'status':'NOT ATTEMPTED','stage':'before-launch'};process={'returncode':None,'timed_out':False}
    healthy=False;preserved=False;verified=None;child=None
    stdout=[];stderr=[];messages=queue.Queue();reader=err_reader=None
    try:
        save('committed-harness.json',{'commit':head,'verified_file_sha256':committed_hashes})
        source_identity();verified=previous.identity();contract.fixture(verified['config']);save('checkpoint-before.json',verified)
        before=windows_observation(since);save('windows-before.json',before)
        if not healthy_observation(before):raise RuntimeError('Unhealthy Windows observer before launch')
        healthy=True
        scratch=output/'scratch';scratch.mkdir()
        command=[str(VENV/'Scripts/python.exe'),'-B','-I',str(Path(__file__).resolve()),'--worker','--output',str(scratch)]
        env={**os.environ,**previous.gpu_proof.scratch_environment(scratch)}
        child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
        process.update(pid=child.pid,command=command,harness_commit=head)
        def read_stdout():
            # Append+fsync each line so the last completed/started stage survives power loss.
            with (output/'worker.stdout.txt').open('xb') as stream:
                for line in iter(child.stdout.readline,b''):
                    stream.write(line);stream.flush();os.fsync(stream.fileno())
                    stdout.append(line);messages.put(line.decode('utf-8',errors='replace').strip())
            messages.put(None)
        def read_stderr():
            with (output/'worker.stderr.txt').open('xb') as stream:
                for line in iter(child.stderr.readline,b''):
                    stream.write(line);stream.flush();os.fsync(stream.fileno());stderr.append(line)
        reader=threading.Thread(target=read_stdout,daemon=True);err_reader=threading.Thread(target=read_stderr,daemon=True)
        reader.start();err_reader.start();began=time.monotonic();deadline=began+45;authorized=False;final_seen=False
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Decoder forward stage deadline exceeded')
            line=messages.get(timeout=remaining)
            if line is None:break
            if line.startswith('FORWARD_STAGE='):process['last_stage']=line.split('=',1)[1]
            if line.startswith(EVENT):
                event=json.loads(line[len(EVENT):])
                if authorized or event.get('event')!='cpu-ready':raise RuntimeError('Duplicate/invalid CPU gate')
                loading_contract.authorize_placement(event.get('gate_one',{}),verified,verified['checkpoint_sha256'],event.get('checkpoint_sha256'))
                save('cpu-gate.json',event)
                observer=windows_observation(since);save('windows-cpu-gate.json',observer)
                if not healthy_observation(observer):
                    healthy=False
                    raise RuntimeError('Unhealthy observer at CPU gate')
                child.stdin.write(b'EXECUTE_ONE_DECODER_FORWARD\n');child.stdin.flush();child.stdin.close()
                authorized=True;deadline=min(began+90,time.monotonic()+30)
            if line.startswith(RESULT):
                if final_seen:raise RuntimeError('Duplicate final result')
                final_seen=True;record=json.loads(line[len(RESULT):])
        child.wait(timeout=max(.1,min(5,began+90-time.monotonic())))
        process.update(returncode=child.returncode,elapsed_seconds=time.monotonic()-began,gpu_authorized=authorized,
                       stage_timeouts_seconds={'cpu':45,'forward_release':30,'overall':90})
    except BaseException as exc:
        process.update(error=repr(exc),timed_out=isinstance(exc,(TimeoutError,queue.Empty,subprocess.TimeoutExpired)))
    finally:
        if child is not None:
            previous.reap_owned_child(child,process)
            if not child.stdin.closed:
                try:child.stdin.close()
                except OSError:pass
        for thread in (reader,err_reader):
            if thread is not None:thread.join(timeout=5)
        process['output_readers_completed']=all(thread is None or not thread.is_alive() for thread in (reader,err_reader))
        try:
            after=windows_observation(since);save('windows-after-exit.json',after)
            healthy=healthy and healthy_observation(after)
            after_identity=previous.identity();save('checkpoint-after.json',after_identity);preserved=verified==after_identity
        except BaseException as exc:
            process['postcheck_error']=repr(exc);healthy=False
    process_ok=(process['returncode']==0 and not process['timed_out'] and not process.get('forced_cleanup',False)
        and process.get('worker_exit_verified') is True and process.get('gpu_authorized') is True
        and process['output_readers_completed'] and 'error' not in process and not process.get('cleanup_unverified',False))
    result={'classification':contract.classify(record,process_ok,healthy,preserved),'observer_healthy':healthy,
            'checkpoint_unchanged':preserved,'worker':record,'process':process}
    save('results.json',result)
    print(json.dumps({key:value for key,value in result.items() if key not in ('worker','process')}))
    return 0 if result['classification']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--harness-commit')
    args=parser.parse_args()
    raise SystemExit(worker(args.output) if args.worker else supervise(args.output,args.harness_commit))
