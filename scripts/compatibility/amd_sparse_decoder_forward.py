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
import amd_forward_diagnostics as diagnostics
import amd_forward_cache_reuse as cache_reuse
from amd_forward_windows_monitor import WindowsMonitor
from amd_forward_owned_job import OwnedJob
from services.amd_offline_guard import block_external_actions
from services.amd_observer import windows_observation, healthy_observation

SOURCE,VENV,PREFIX=previous.SOURCE,previous.VENV,previous.PREFIX
EVIDENCE=ROOT/'docs/compatibility/amd-instrumented-forward-resume-evidence'
EVENT='FORWARD_EVENT='
RESULT='FORWARD_RESULT='
OLD=ROOT/'docs/compatibility/amd-sparse-decoder-loading-evidence/run/results.json'
COMMITTED_FILES=[
    'scripts/compatibility/'+name+'.py' for name in ('amd_sparse_decoder_forward','amd_decoder_forward_contract',
    'amd_sparse_decoder_loading','amd_decoder_contract','amd_bounded_gpu_proof','amd_native_import_preflight','amd_model_metadata','amd_forward_diagnostics',
    'amd_forward_cache_reuse','amd_forward_windows_monitor','amd_forward_owned_job')
] + ['api/services/'+name+'.py' for name in ('amd_runtime','amd_model_metadata','amd_offline_guard','amd_observer')] + [
    'experimental/extensions/trellis2-amd/strict_loader.py',
    OLD.relative_to(ROOT).as_posix(),previous.ARCHIVED.relative_to(ROOT).as_posix(),
    'docs/compatibility/amd-instrumented-forward-resume-evidence/cache-plan.json']

CPU_SECONDS=45;GPU_SECONDS=120;OVERALL_SECONDS=180


def prepare_private_cache(scratch):
    plan=json.loads((EVIDENCE/'cache-plan.json').read_text())
    baseline=ROOT/'docs/compatibility/amd-forward-timeout-investigation-evidence/inventory-before.json'
    if cache_reuse.attest_runtime(cache_reuse.SITE,json.loads(baseline.read_text()))!=plan['runtime_libraries']:
        raise RuntimeError('Runtime library attestation changed after preparation')
    cache_reuse.verify_copies(EVIDENCE/'prepared-cache',plan['selected'])
    # Fresh scratch already exists; copy into a temporary sibling, then move only
    # its two owned cache subdirectories into the empty owned scratch tree.
    staging=scratch.parent/'cache-staging'
    cache_reuse.copy_entries(EVIDENCE/'prepared-cache',staging,plan['selected'])
    for name in ('amd_comgr_cache_dir','miopen_custom_cache_dir'):
        source=(staging/name).resolve();target=(scratch/name).resolve()
        source.relative_to(EVIDENCE.resolve());target.relative_to(EVIDENCE.resolve())
        source.rename(target)
    staging.rmdir()
    cache_reuse.verify_copies(scratch,plan['selected'])
    return {'selected_files':len(plan['selected']),'plan_sha256':cache_reuse.digest(EVIDENCE/'cache-plan.json'),
            'runtime_libraries':plan['runtime_libraries'],'private_copy_hashes_verified':True}


def verify_committed_files(commit,files=None):
    if len(commit)!=40 or any(char not in '0123456789abcdef' for char in commit):
        raise ValueError('Supply the exact published harness commit SHA')
    hashes={}
    for name in COMMITTED_FILES if files is None else files:
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


def worker(scratch, occupancy=None):
    contract=globals()['contract'] if occupancy is None else occupancy.contract
    record={'status':'FAIL','stage':'startup','forward_calls':0,'sdk_discovery':[]}
    component=model=latent=output=cpu_output=cpu_input=torch=saved_array=None
    permit=contract.SingleForward()
    transfers=contract.TensorTransfers()
    phase='cpu'; allowed_modules=set();module_names={};root_id=None;root_calls=0
    trace=diagnostics.Trace(lambda line:print(line,flush=True))
    def stage(name):
        record['stage']=name
        print('FORWARD_STAGE='+name,flush=True)
        trace.emit('stage',name,phase=phase,forward_calls=permit.calls,root_decoder_calls=root_calls)
    try:
        stage('waiting-for-owned-job')
        if sys.stdin.readline().strip()!='OWNED_JOB_ATTACHED':
            raise RuntimeError('Worker requires parent-owned job containment before preparation')
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
                allowed=(checkpoint,) if occupancy is None else (checkpoint,occupancy.data.INPUT.resolve())
                if path.suffix.lower() in ('.safetensors','.pt','.pth','.bin','.ckpt','.gguf') and path.resolve() not in allowed:
                    raise RuntimeError('Only the audited sparse decoder may be read: '+str(path))
        sys.addaudithook(audit)
        record['verified_support_source']=source_identity()
        verified=previous.identity();record['checkpoint_identity']=verified
        fixture=contract.fixture(verified['config']);record['fixture']=fixture
        if occupancy is not None:
            saved_array,record['saved_latent']=occupancy.data.read_input()
            record['workspace_binding']=occupancy.teardown.verify_binding_evidence()
        import torch
        Path(sys.executable).resolve().relative_to(VENV)
        Path(torch.__file__).resolve().relative_to(VENV/'Lib/site-packages')
        if (sys.version.split()[0]!='3.12.10' or torch.__version__!='2.13.0+rocm10.0.0'
                or not torch.version.hip or torch.version.cuda is not None or torch.cuda.is_initialized()):
            raise RuntimeError('Unexpected AMD runtime or pre-existing GPU context')
        record['runtime']={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
                           'torch_path':torch.__file__,'hip':torch.version.hip,'cuda':torch.version.cuda}
        trace.emit('state','runtime',**record['runtime'])
        if occupancy is not None:
            record['workspace_binding']=occupancy.teardown.verify_runtime_binding(torch)
            record['ram_before_load']=occupancy.admission()
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
            name=module_names[id(instance)]
            if diagnostics.key_module(name,type(instance).__name__):
                detail=tensor_description(args[0]) if args and isinstance(args[0],torch.Tensor) else {}
                return trace.call('module:'+(name or '<root>'),lambda:original_call(instance,*args,**kwargs),
                                  input=detail,forward_calls=permit.calls,root_decoder_calls=root_calls)
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
                if phase in ('input-transfer','output-transfer'):
                    description=tensor_description(args[0]) if args and isinstance(args[0],torch.Tensor) else {}
                    normalized={key:str(value) if isinstance(value,torch.device) else value for key,value in kwargs.items()}
                    transfers.authorize(str(func),args,normalized,phase,description)
                elif (has_gpu(args) or has_gpu(kwargs)) and (phase=='cpu' or
                        (phase!='forward' and (phase not in ('placement','release') or not contract.placement_operation(str(func))))):
                    raise RuntimeError('Unapproved GPU operation outside forward: '+str(func))
                if phase=='forward' and str(func) in ('aten.convolution.default','aten._convolution.default','aten.miopen_convolution.default'):
                    return trace.call('aten:'+str(func),lambda:func(*args,**kwargs),
                                      input=tensor_description(args[0]) if args else {})
                return func(*args,**kwargs)
        def tensor_description(tensor):
            return {'shape':list(tensor.shape),'dtype':str(tensor.dtype),'device':str(tensor.device),
                    'stride':list(tensor.stride()),
                    'layout':str(tensor.layout),'contiguous':tensor.is_contiguous(),'requires_grad':tensor.requires_grad}
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
            trace.emit('state','gpu',gpu=record['gpu'])
            torch.cuda.reset_peak_memory_stats(0)
            record['before_placement']=previous.gpu_memory(torch)
            trace.emit('memory','before-placement',**record['before_placement'])
            if record['before_placement']['allocated_bytes'] or record['before_placement']['reserved_bytes'] or record['before_placement']['available_bytes']<verified['tensor_data_bytes']+512*1024**2:
                raise RuntimeError('Baseline allocation or insufficient VRAM')
            if occupancy is not None:
                if record['before_placement']['available_bytes']<13*1024**3:raise RuntimeError('Less than13GiB available VRAM')
                torch.cuda.set_per_process_memory_fraction(contract.MEMORY_LIMIT/properties.total_memory,0)
                record['budget']={'allocator_limit_bytes':contract.MEMORY_LIMIT,'minimum_available_bytes':13*1024**3,
                                  'cpu_seconds':120,'gpu_seconds':180,'overall_seconds':360}
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.use_deterministic_algorithms(True)
            record['determinism']={'algorithms':True,'cudnn_benchmark':False,'cudnn_deterministic':True}
            stage('decoder-placement');began=time.monotonic()
            model.to(device='cuda:0');torch.cuda.synchronize(0)
            loading_contract.validate_tensors(verified['tensors'],previous.tensor_manifest(model.state_dict()),'cuda:0')
            record['placement_seconds']=time.monotonic()-began
            with torch.inference_mode():
                cpu_input=(torch.tensor(fixture['values'],dtype=torch.float32,device='cpu').reshape(fixture['input_shape'])
                           if occupancy is None else torch.from_numpy(saved_array))
                stage('input-transfer');phase='input-transfer'
                transfers.begin('input',cpu_input,tensor_description(cpu_input),phase)
                latent=torch.ops.aten._to_copy.default(cpu_input,device=torch.device('cuda:0'))
                phase='placement';torch.cuda.synchronize(0)
                if tensor_description(latent)!={**tensor_description(cpu_input),'device':'cuda:0'}:
                    raise RuntimeError('Input copy changed shape, dtype or layout')
                record['before_forward']=previous.gpu_memory(torch)
                trace.emit('memory','before-forward',**record['before_forward'])
                module_names={id(item):name for name,item in model.named_modules()}
                allowed_modules=set(module_names);root_id=id(model)
                stage('single-forward');phase='forward';began=time.monotonic()
                output=permit.invoke(lambda:model(latent))
                phase='placement';trace.call('synchronize:after-forward',lambda:torch.cuda.synchronize(0))
                elapsed=time.monotonic()-began
                record['after_forward']=previous.gpu_memory(torch)
                trace.emit('memory','after-forward',**record['after_forward'])
                stage('output-copy-validation')
                phase='output-transfer'
                transfers.begin('output',output,tensor_description(output),phase)
                cpu_output=torch.ops.aten._to_copy.default(output,device=torch.device('cpu'))
                phase='placement';torch.cuda.synchronize(0)
                expected_copy={**tensor_description(output),'device':'cpu'}
                if tensor_description(cpu_output)!=expected_copy:
                    raise RuntimeError('Output copy changed shape, dtype or layout')
                description={'shape':list(output.shape),'dtype':str(output.dtype),'device':str(output.device),
                    'requires_grad':output.requires_grad,'inference_mode':torch.is_inference_mode_enabled(),
                    **({'values':cpu_output.reshape(-1).tolist()} if occupancy is None else occupancy.output_summary(cpu_output.numpy()))}
                statistics=contract.validate_output(description,fixture)
                record['forward']={'status':'PASS','elapsed_seconds':elapsed,'input_shape':list(latent.shape),
                    'input_dtype':str(latent.dtype),'input_device':str(latent.device),'output':description,
                    'statistics':statistics,'output_sha256':hashlib.sha256(cpu_output.numpy().tobytes()).hexdigest(),
                    'grad_enabled':torch.is_grad_enabled(),'output_is_inference_tensor':output.is_inference()}
                if occupancy is not None:
                    record['occupancy_artifacts']=occupancy.data.write_output(scratch,cpu_output.numpy(),occupancy.provenance(record))
            record['forward_calls']=permit.calls
            if root_calls!=1 or permit.calls!=1:raise RuntimeError('Forward count invariant failed')
            stage('release')
            phase='release'
            model_ref,input_ref,output_ref=weakref.ref(model),weakref.ref(latent),weakref.ref(output)
            component.unload();component=None;model=latent=output=cpu_output=cpu_input=None
            saved_array=None;gc.collect();torch.cuda.synchronize(0)
            if occupancy is not None:
                record['workspace_clear_calls']=1;trace.call('blas:clear-workspaces',torch._C._cuda_clearCublasWorkspaces)
            torch.cuda.empty_cache();torch.cuda.synchronize(0)
            record['release']={**previous.gpu_memory(torch),'model_collected':model_ref() is None,
                              'input_collected':input_ref() is None,'output_collected':output_ref() is None,
                              'cpu_memory':previous.process_memory()}
            trace.emit('memory','release',**record['release'])
            loading_contract.validate_release(record['release'])
            if record['release']['input_collected'] is not True or record['release']['output_collected'] is not True:
                raise RuntimeError('Forward cleanup contract failed')
            stage('complete');record['status']='PASS'
    except BaseException as exc:
        record.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        record['forward_calls']=permit.calls;record['root_decoder_calls']=root_calls
        record['transfer_calls']=transfers.counts.copy()
        transfers.active=None
        # After any native error, do not issue another GPU operation. Process exit
        # and parent reaping own cleanup; no retry or attempted GPU repair.
        if component is not None:component.unload()
        component=model=latent=output=cpu_output=cpu_input=saved_array=None
        gc.collect()
        print(RESULT+json.dumps(record),flush=True)
    return 0 if record['status']=='PASS' else 1


def supervise(output,harness_commit,*,profile=None):
    # A second single-model proof reuses the same ownership, journals, observer,
    # deadlines and cleanup path. The original decoder default is unchanged.
    evidence=EVIDENCE if profile is None else profile.evidence
    cpu_seconds=CPU_SECONDS if profile is None else profile.cpu_seconds
    gpu_seconds=GPU_SECONDS if profile is None else profile.gpu_seconds
    overall_seconds=OVERALL_SECONDS if profile is None else profile.overall_seconds
    identity=previous.identity if profile is None else profile.identity
    inspect_source=source_identity if profile is None else profile.source_identity
    validate_config=contract.fixture if profile is None else profile.validate_config
    classify=contract.classify if profile is None else profile.classify
    cache=prepare_private_cache if profile is None else profile.prepare_private_cache
    worker_script=Path(__file__).resolve() if profile is None else profile.worker_script
    authorize=b'EXECUTE_ONE_DECODER_FORWARD\n' if profile is None else profile.authorize
    sparse_profile=getattr(profile,'kind',None) in ('sparse-flow','sparse-sampling','shape-slat')
    deadline_contract=(180,240,480) if sparse_profile else (120,180,360)
    if profile is not None and (cpu_seconds,gpu_seconds,overall_seconds)!=deadline_contract:
        raise ValueError('Profile deadlines differ from the explicit reviewed contract')
    authorize_gate=profile.authorize_placement if sparse_profile else loading_contract.authorize_placement
    output=output.resolve()
    if output!=evidence/'run':raise ValueError('Use the single canonical evidence run directory')
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,check=True,timeout=5).stdout.decode().strip()
    if head!=harness_commit:raise RuntimeError('Run only the exact committed/published harness')
    committed_hashes=verify_committed_files(harness_commit) if profile is None else verify_committed_files(harness_commit,profile.committed_files)
    output.mkdir(parents=True,exist_ok=False)
    with (evidence/'attempt-started.json').open('x',encoding='utf-8') as stream:
        attempt={'harness_commit':head,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'maximum_forwards':1 if profile is None else getattr(profile,'maximum_forwards',1)}
        if profile is not None and getattr(profile,'kind',None)=='sparse-sampling':
            attempt.update(maximum_steps=12,maximum_sequences=1)
        json.dump(attempt,stream)
        stream.flush();os.fsync(stream.fileno())
    def save(name,value):
        with (output/name).open('x',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    since=datetime.datetime.now(datetime.timezone.utc).isoformat()
    record={'status':'NOT ATTEMPTED','stage':'before-launch'};process={'returncode':None,'timed_out':False}
    healthy=False;preserved=False;verified=None;child=None
    stdout=[];stderr=[];messages=queue.Queue();reader=err_reader=None;journal=monitor=None
    windows_monitor=windows_journal=owned_job=None
    try:
        save('committed-harness.json',{'commit':head,'verified_file_sha256':committed_hashes})
        inspect_source();verified=identity();validate_config(verified['config']);save('checkpoint-before.json',verified)
        before=windows_observation(since);save('windows-before.json',before)
        if not healthy_observation(before):raise RuntimeError('Unhealthy Windows observer before launch')
        healthy=True
        scratch=output/'scratch';scratch.mkdir()
        save('private-cache.json',cache(scratch))
        journal=diagnostics.Journal(output/'trace.jsonl')
        windows_journal=diagnostics.Journal(output/'windows-periodic.jsonl')
        command=[str(VENV/'Scripts/python.exe'),'-B','-I',str(worker_script),'--worker','--output',str(scratch)]
        env={**os.environ,**previous.gpu_proof.scratch_environment(scratch)}
        owned_job=OwnedJob()
        began=time.monotonic();deadline=began+cpu_seconds
        child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
        process.update(pid=child.pid,command=command,harness_commit=head)
        owned_job.attach(child)
        process['job_attached_before_preparation']=True
        child.stdin.write(b'OWNED_JOB_ATTACHED\n');child.stdin.flush()
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
        reader.start();err_reader.start();authorized=False;final_seen=False
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Decoder forward stage deadline exceeded')
            if authorized:monitor.require_healthy();windows_monitor.require_healthy()
            try:line=messages.get(timeout=min(remaining,.5) if authorized else remaining)
            except queue.Empty:
                if authorized:continue
                raise TimeoutError('CPU gate deadline exceeded')
            if line is None:break
            if line.startswith(diagnostics.MARKER):
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Deadline expired before durable diagnostic write')
                row=journal.consume(line,timeout=min(1.,remaining))
                if row['origin']=='worker':
                    process['last_worker_trace']=row
                    if row['event']=='memory':process.setdefault('gpu_memory_samples',{})[row['name']]=row
                else:process['last_cpu_activity']=row
            if line.startswith('FORWARD_STAGE='):process['last_stage']=line.split('=',1)[1]
            if line.startswith(EVENT):
                event=json.loads(line[len(EVENT):])
                if authorized or event.get('event')!='cpu-ready':raise RuntimeError('Duplicate/invalid CPU gate')
                authorize_gate(event.get('gate_one',{}),verified,verified['checkpoint_sha256'],event.get('checkpoint_sha256'))
                save('cpu-gate.json',event)
                observer=windows_observation(since);save('windows-cpu-gate.json',observer)
                if not healthy_observation(observer):
                    healthy=False
                    raise RuntimeError('Unhealthy observer at CPU gate')
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('CPU gate deadline expired before monitor startup')
                monitor=diagnostics.ActivityMonitor(child.pid,messages.put)
                monitor.start(timeout=min(2.,remaining))
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('CPU gate deadline expired before periodic Windows startup')
                windows_monitor=WindowsMonitor(since,windows_journal.consume)
                windows_monitor.start(timeout=min(5.,remaining))
                monitor.require_healthy();windows_monitor.require_healthy()
                if time.monotonic()>=deadline:raise TimeoutError('CPU gate deadline expired before GPU authorization')
                child.stdin.write(authorize);child.stdin.flush();child.stdin.close()
                authorized=True;deadline=min(began+overall_seconds,time.monotonic()+gpu_seconds)
                process['gpu_authorized']=True
            if line.startswith(RESULT):
                if final_seen:raise RuntimeError('Duplicate final result')
                final_seen=True;record=json.loads(line[len(RESULT):])
        if time.monotonic()>=deadline:raise TimeoutError('Worker-exit stage deadline exceeded')
        child.wait(timeout=min(5,deadline-time.monotonic(),began+overall_seconds-time.monotonic()))
        process.update(returncode=child.returncode,elapsed_seconds=time.monotonic()-began,gpu_authorized=authorized,
                       stage_timeouts_seconds={'cpu':cpu_seconds,'forward_release':gpu_seconds,'overall':overall_seconds})
    except BaseException as exc:
        process.update(error=repr(exc),timed_out=isinstance(exc,(TimeoutError,queue.Empty,subprocess.TimeoutExpired)))
    finally:
        if owned_job is not None:
            try:process['job_cleanup']=owned_job.finish()
            except BaseException as exc:
                process['job_cleanup_error']=repr(exc);process['cleanup_unverified']=True
                try:owned_job.close()
                except BaseException as close_error:process['job_close_error']=repr(close_error)
        if child is not None:
            previous.reap_owned_child(child,process)
            if not child.stdin.closed:
                try:child.stdin.close()
                except OSError:pass
        if monitor is not None:
            process['cpu_observer_stopped']=monitor.close()
            process['cpu_observer_error']=monitor.error
        if windows_monitor is not None:
            process['windows_periodic_stopped']=windows_monitor.close()
            process['windows_periodic_error']=windows_monitor.error
            process['windows_periodic_samples']=windows_monitor.samples
            healthy=healthy and not windows_monitor.error and process['windows_periodic_stopped']
        for thread in (reader,err_reader):
            if thread is not None:thread.join(timeout=5)
        process['output_readers_completed']=all(thread is None or not thread.is_alive() for thread in (reader,err_reader))
        if journal is not None:
            process['diagnostic_journal_stopped']=journal.close()
            process['diagnostic_journal_error']=journal.error
        if windows_journal is not None:
            process['windows_journal_stopped']=windows_journal.close()
            process['windows_journal_error']=windows_journal.error
        if record['status']=='NOT ATTEMPTED' and process.get('last_stage'):
            record={'status':'INCOMPLETE','stage':process['last_stage'],'last_worker_trace':process.get('last_worker_trace')}
        try:
            after=windows_observation(since);save('windows-after-exit.json',after)
            healthy=healthy and healthy_observation(after)
            after_identity=identity();save('checkpoint-after.json',after_identity);preserved=verified==after_identity
        except BaseException as exc:
            process['postcheck_error']=repr(exc);healthy=False
    process_ok=(process['returncode']==0 and not process['timed_out'] and not process.get('forced_cleanup',False)
        and process.get('worker_exit_verified') is True and process.get('gpu_authorized') is True
        and process['output_readers_completed'] and 'error' not in process and not process.get('cleanup_unverified',False))
    process_ok=process_ok and process.get('cpu_observer_stopped') is True and not process.get('cpu_observer_error')
    process_ok=process_ok and process.get('diagnostic_journal_stopped') is True and not process.get('diagnostic_journal_error')
    process_ok=process_ok and process.get('windows_periodic_stopped') is True and not process.get('windows_periodic_error')
    process_ok=process_ok and process.get('windows_journal_stopped') is True and not process.get('windows_journal_error')
    process_ok=process_ok and process.get('job_attached_before_preparation') is True \
        and process.get('job_cleanup',{}).get('owned_tree_exit_verified') is True \
        and not process.get('job_cleanup',{}).get('forced_job_cleanup') and not process.get('job_cleanup_error')
    result={'classification':classify(record,process_ok,healthy,preserved),'observer_healthy':healthy,
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
