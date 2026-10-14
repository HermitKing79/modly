"""One authorised FlexGEMM import, then conditional genuine CPU decoder loading.
No decoder forward, GPU tensor operation, compiler, helper build or mesh path.
"""
import argparse
import gc
import ctypes
import shutil
import hashlib
import importlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import weakref

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[3]
EV = SCRIPT.parent
OLD_EV = ROOT / 'docs/compatibility/amd-wsl-native-decoder-preparation-evidence'
SCRATCH = Path('/home/modly/shape-decoder-cpu-c4552110')
CPU_RAM = 6972719480
STAGE = Path('/home/modly/native-decoder-preparation-6dd94a85')
SOURCE = Path('/mnt/d/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD')
PREFIX = Path('/mnt/d/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16')
HOST_DEADLINE = 270.0
NATIVE_DEADLINE = 60.0
CPU_DEADLINE = 120.0
GiB = 1024**3
sys.path.insert(0, str(ROOT/'scripts/compatibility'))
import amd_wsl_rocdxg_smoke as base


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def checkpoint_record(checkpoint):
    """The same explicit, read-only checkpoint contract on host and guest."""
    document = json.loads((EV/'windows-preservation-before.json').read_text())
    row = document['decoder_checkpoint_metadata_and_header_only']
    verified = json.loads((EV/'checkpoint-and-resource-preflight.json').read_text())
    expected = {'bytes': 948490494, 'mtime_ns': 1791274722716299500,
                'verified_prior_stream_sha256': 'e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581'}
    if document['status'] != 'PASS' or not isinstance(row, dict):
        raise ValueError('Verified checkpoint preservation record required')
    for key, value in expected.items():
        if type(row.get(key)) is not type(value) or row[key] != value:
            raise ValueError('Checkpoint record field differs: '+key)
    windows_path = 'D:/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors'
    if type(row.get('path')) is not str or row['path'].replace('\\', '/') != windows_path:
        raise ValueError('Foreign checkpoint record path')
    expected_path = windows_path if sys.platform == 'win32' else '/mnt/d/'+windows_path[3:]
    if checkpoint.resolve(strict=True).as_posix() != expected_path:
        raise ValueError('Foreign checkpoint file identity')
    for key, recorded in (('checkpoint_bytes', expected['bytes']), ('checkpoint_mtime_ns', expected['mtime_ns']),
                          ('checkpoint_sha256', expected['verified_prior_stream_sha256'])):
        if type(verified.get(key)) is not type(recorded) or verified[key] != recorded:
            raise ValueError('Checkpoint hash provenance differs: '+key)
    before = checkpoint.stat()
    if not checkpoint.is_file() or before.st_size != row['bytes'] or before.st_mtime_ns != row['mtime_ns']:
        raise ValueError('Verified checkpoint metadata changed')
    actual = digest(checkpoint)
    after = checkpoint.stat()
    if actual != expected['verified_prior_stream_sha256'] or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Checkpoint payload identity changed during read-only verification')
    return dict(row, freshly_streamed_sha256=actual)


def worker_command(script_linux, output_linux):
    command = base.worker_command(script_linux, output_linux)
    command[command.index('90s')] = '240s'
    return command


def check_deadlines(now, began, go):
    if now - began >= HOST_DEADLINE: raise TimeoutError('270-second host deadline')
    if go is not None and now - go >= 220: raise TimeoutError('220-second authorised worker deadline')


class HostMemory(ctypes.Structure):
    _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]


def host_admission():
    value=HostMemory();value.length=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)): raise ctypes.WinError()
    free=shutil.disk_usage('D:/').free
    if value.available<CPU_RAM or free<50*GiB: raise RuntimeError('Host RAM/D-disk floor failed')
    return {'available_ram_bytes':value.available,'D_free_bytes':free}


def reserve_attempt(output):
    # Exact candidate identity; retain marker even on any later failure.
    base.save_json(EV/'CPU-loading-attempt.marker',{'started_utc':base.utc_now(),
        'distro':base.DISTRO,'script_sha256':digest(SCRIPT),'output':str(output),
        'native_import_attempts_authorized':1,'neural_forwards_authorized':0})
    output.mkdir(parents=True,exist_ok=False)


def require_zero(record):
    for key in ('memory_before', 'memory_release', 'memory_peak'):
        if record.get(key) != {'allocated': 0, 'reserved': 0}:
            raise RuntimeError(key + ': zero PyTorch GPU bytes required')
    if record.get('no_neural_forward') is not True or record.get('native_calls') != 0:
        raise RuntimeError('Unexpected native/neural computation')


def validate_qualified(record):
    diagnostics=record.get('decoder_diagnostics',{})
    parameters=json.loads((ROOT/'docs/compatibility/amd-shape-decoder-neural-evidence/cpu-probe-4/results.json').read_text())['worker']['parameters']
    widen=sorted(k for k,v in parameters.items() if v['dtype']=='F32')
    final_policy=record.get('policy_final',record.get('policy',{}))
    if ('error' in record or 'cleanup_error' in record
        or record.get('decoder_cpu')!='PASS'
        or record.get('constructed_on_meta') is not True or record.get('evaluation_and_gradients_disabled') is not True
        or record.get('buffers_count')!=0 or len(record.get('loaded_parameters',{}))!=292
        or record.get('CPU_parameters_collected') is not True or record.get('CPU_component_collected') is not True or record.get('CPU_model_collected') is not True
        or record.get('decoder_class')!='trellis2.models.sc_vaes.sparse_unet_vae.SparseUnetVaeDecoder'
        or any(diagnostics.get(k)!=v for k,v in {'weights_loaded':True,'strict':True,'entries':292,
            'dtype_counts':{'F16':216,'F32':76},'roundtrip_verified':True,'widened_keys':widen,'model_forward_calls':0}.items())
        or set(diagnostics.get('payload_sha256',{}))!=set(parameters)
        or final_policy.get('rejected_calls')!=0 or final_policy.get('native_calls')!=0):
        raise RuntimeError('Incomplete strict CPU decoder/release/final policy qualification')
    require_zero(record)
    if (record.get('gate_authorized') is not True or record.get('native_import') != 'PASS'
            or record.get('bindings_count') != 18 or record.get('guards_rejected_calls') != 0
            or record.get('driver_helper_inactive') is not True):
        raise RuntimeError('Incomplete native import qualification')


def cpu_memory():
    values={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')): values[line.split(':')[0]+'_bytes']=int(line.split()[1])*1024
    values['ru_maxrss_bytes']=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss*1024
    return values

def worker(output):
    output.mkdir(exist_ok=False)
    record = {'classification': 'UNKNOWN', 'native_import': 'NOT_ATTEMPTED',
              'decoder_cpu': 'NOT_ATTEMPTED', 'no_neural_forward': True,
              'started_utc': base.utc_now(), 'gate_authorized': False}
    torch = policy = component = model = None
    model_ref = None
    began = None
    try:
        base.journal(output, 'gate-wait', gpu_allowed=False)
        base.await_go(sys.stdin.buffer)
        record['gate_authorized'] = True
        began = time.monotonic()
        base.journal(output, 'gate-authorized', gpu_allowed=True)
        # Own every cache/home/temp write. Never stage in the baseline venv.
        SCRATCH.mkdir(exist_ok=False)
        for name in ('home', 'tmp', 'triton-cache', 'hip-cache', 'comgr-cache', 'torch-extensions'):
            (SCRATCH / name).mkdir()
        os.environ.update(HOME=str(SCRATCH/'home'), TMPDIR=str(SCRATCH/'tmp'),
            TRITON_CACHE_DIR=str(SCRATCH/'triton-cache'), HIP_CACHE_DIR=str(SCRATCH/'hip-cache'),
            AMD_COMGR_CACHE_DIR=str(SCRATCH/'comgr-cache'), TORCH_EXTENSIONS_DIR=str(SCRATCH/'torch-extensions'),
            FLEX_GEMM_USE_AUTOTUNE_CACHE='0', FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE='0',
            SPARSE_CONV_BACKEND='flex_gemm', HSA_ENABLE_DXG_DETECTION='1',
            HIP_VISIBLE_DEVICES='0', ROCR_VISIBLE_DEVICES='0', CUDA_VISIBLE_DEVICES='0')
        sys.path[:0] = [str(STAGE/'site-packages'), str(ROOT/'api'),
                       str(ROOT/'experimental/extensions/trellis2-amd')]
        if digest(ROOT/'scripts/compatibility/amd_wsl_native_import_policy.py') != '8b9207a14757ccda197e11c054d694f9b4ff47006a1cccc46d71f1d494ab96cd': raise RuntimeError('Qualified policy changed')
        staged = json.loads((OLD_EV/'staging.json').read_text())
        if staged['target'] != str(STAGE/'site-packages'): raise RuntimeError('Foreign isolated target')
        accepted = json.loads((EV.parent/'amd-wsl-flexgemm-build-evidence/static-acceptance.json').read_text())
        wheel = EV.parent/'amd-wsl-flexgemm-build-evidence/flex_gemm-1.0.0-cp312-cp312-linux_x86_64.whl'
        if digest(wheel) != accepted['wheel_sha256']: raise RuntimeError('Native wheel changed')
        import zipfile
        with zipfile.ZipFile(wheel) as archive:
            for info in archive.infolist():
                if info.is_dir() or '.dist-info/' in info.filename: continue
                if (STAGE/'site-packages'/info.filename).read_bytes() != archive.read(info):
                    raise RuntimeError('Native package payload changed: '+info.filename)
        import numpy as np
        import amd_shape_decoder_preflight as contract
        record['saved_inputs'] = contract.validate_arrays(
            contract.read_npy(contract.SAMPLE/'shape-latent-denormalized.npy', contract.INPUTS['shape-latent-denormalized.npy']),
            contract.read_npy(contract.SAMPLE/'shape-latent-coordinates.npy', contract.INPUTS['shape-latent-coordinates.npy']))
        if digest(contract.SAMPLE/'shape-latent-provenance.json') != contract.INPUTS['shape-latent-provenance.json']:
            raise RuntimeError('Turret provenance changed')
        source_contract = json.loads((EV.parent/'amd-wsl-neural-decoder-preparation-evidence/decoder-contract-readback.json').read_text())
        source_checks = {}
        for windows_path, row in source_contract['source_checks'].items():
            guest_path = '/mnt/'+windows_path[0].lower()+'/'+windows_path[3:].replace('\\','/')
            actual = digest(guest_path)
            if actual != row['expected_sha256']: raise RuntimeError('Pinned decoder source changed: '+guest_path)
            source_checks[guest_path] = actual
        record['source_sha256'] = source_checks
        if digest(PREFIX.with_suffix('.json')) != '5f1b856dffce79466fb18f5f0eefc08f08c673b24b3c05b20359c0b6d318a209':
            raise RuntimeError('Decoder config changed')
        before = checkpoint_record(PREFIX.with_suffix('.safetensors'))
        record['checkpoint'] = dict(before, guest_path=str(PREFIX.with_suffix('.safetensors')))
        available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
        free = os.statvfs('/').f_bavail*os.statvfs('/').f_frsize
        dfree = os.statvfs('/mnt/d').f_bavail*os.statvfs('/mnt/d').f_frsize
        record['admission'] = {'guest_ram_available':available,'guest_disk_free':free,'D_disk_free':dfree,
                               'CPU_ram_requirement':CPU_RAM,'guest_disk_requirement':5*GiB,'D_disk_requirement':50*GiB}
        if available<CPU_RAM or free<5*GiB or dfree<50*GiB: raise RuntimeError('CPU/disk admission refused')
        import torch
        record['runtime'] = {'torch':torch.__version__, 'distribution_version':md.version('torch'),
            'git_version':torch.version.git_version,'hip':torch.version.hip,'cuda':torch.version.cuda,
            'python':platform.python_version(),'prefix':sys.prefix,'gpu_initialized_before':torch.cuda.is_initialized()}
        base.validate_runtime_identity(record['runtime'])
        if md.version('triton') != '3.5.1+rocm7.2.1.gita272dfa8': raise RuntimeError('Triton identity mismatch')
        import triton
        from amd_wsl_native_import_policy import ImportPolicy
        policy = ImportPolicy(SCRATCH, writable_roots=(output,))
        policy.install_tensor_guards(torch)
        policy.install_runtime_guards(triton)
        policy.install_audit_hook()
        policy.install_profile()
        def memory():
            return {'allocated':torch.cuda.memory_allocated(0),'reserved':torch.cuda.memory_reserved(0)}
        with policy.tensor_mode(torch):
            count = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            vfree, total = torch.cuda.mem_get_info(0)
            record['gpu'] = {'name':props.name,'architecture':props.gcnArchName,'count':count,
                             'total_bytes':total,'free_bytes':vfree,'property_total_bytes':props.total_memory}
            if count!=1 or 'RX 9070 XT' not in props.name or props.gcnArchName.split(':')[0]!='gfx1201' or vfree<13*GiB:
                raise RuntimeError('GPU identity/free-VRAM admission refused')
            torch.cuda.set_per_process_memory_fraction(12*GiB/total,0)
            torch.cuda.reset_peak_memory_stats(0)
            record['memory_before'] = memory()
            if record['memory_before']!={'allocated':0,'reserved':0}: raise RuntimeError('Unexpected initial GPU allocation')
            record['gpu_initialized_after_admission'] = torch.cuda.is_initialized()
            base.journal(output,'native-import-entry',gpu=record['gpu'])
            record['native_import'] = 'ENTERED'
            native_began = time.monotonic()
            import flex_gemm
            from flex_gemm.kernels import cuda
            record['package_path'] = flex_gemm.__file__
            record['native_path'] = cuda.__file__
            if Path(cuda.__file__).resolve() != (STAGE/'site-packages'/accepted['native_member']).resolve() or digest(cuda.__file__)!=accepted['native_sha256']:
                raise RuntimeError('Unexpected native module provenance')
            names = accepted['source_binding_names']
            if len(names)!=18 or not all(callable(getattr(cuda,name,None)) for name in names):
                raise RuntimeError('Missing actual native bindings')
            record['bindings']=names; record['bindings_count']=len(names)
            torch.cuda.synchronize(0)
            record['native_import_synchronized_seconds']=time.monotonic()-native_began
            record['gpu_initialized_after_import']=torch.cuda.is_initialized()
            record['native_memory_after']=memory()
            if time.monotonic()-began>=NATIVE_DEADLINE or memory()!={'allocated':0,'reserved':0}:
                raise RuntimeError('Native import deadline/allocation gate failed')
            record['driver_helper_inactive'] = (triton.runtime.driver._active is None and
                triton.runtime.driver._default is None and not any(n in sys.modules for n in ('hip_utils','__triton_launcher')))
            if not record['driver_helper_inactive']: raise RuntimeError('Unexpected Triton helper/driver activation')
            record['native_import']='PASS'
            base.journal(output,'native-qualified-CPU-preparation-entry',memory=memory())
            # Same owned worker: no second native risk import. GPU ops remain blocked.
            policy.set_phase('cpu_meta')
            cpu_began=time.monotonic()
            record['cpu_memory_before_loading']=cpu_memory()
            fresh_available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
            record['guest_RAM_before_loading']=fresh_available
            if fresh_available<CPU_RAM: raise RuntimeError('Fresh CPU load RAM admission refused')
            sys.path.insert(0,str(SOURCE))
            from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
            from shape_decoder_loader import ShapeDecoderComponent
            component=ShapeDecoderComponent(PREFIX.with_suffix('.json'), PREFIX.with_suffix('.safetensors'))
            original_factory=component._component.factory
            def timed_factory(config):
                construction_began=time.monotonic()
                constructed=original_factory(config)
                record['construction_seconds']=time.monotonic()-construction_began
                record['constructed_on_meta']=all(p.is_meta for p in constructed.parameters())
                if not record['constructed_on_meta']: raise RuntimeError('Parent not constructed on meta')
                return constructed
            component._component.factory=timed_factory
            loading_began=time.monotonic()
            model=component.load_cpu()
            record['strict_load_and_widen_seconds']=time.monotonic()-loading_began
            component._component.factory=original_factory
            original_factory=timed_factory=None
            record['loaded_parameters']={k:{'shape':list(p.shape),'dtype':str(p.dtype),'device':str(p.device),'requires_grad':p.requires_grad} for k,p in model.named_parameters()}
            parameter_refs=[weakref.ref(p) for p in model.parameters()]
            component_ref=weakref.ref(component)
            if model.training or any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()): raise RuntimeError('Model not eval/grad-disabled')
            record['evaluation_and_gradients_disabled']=True
            record['buffers_count']=len(list(model.named_buffers()))
            model_ref=weakref.ref(model)
            record['decoder_cpu']='PASS'
            record['decoder_class']=type(model).__module__+'.'+type(model).__name__
            record['decoder_diagnostics']=json.loads(json.dumps(component.diagnostics))
            record['decoder_source_path']=sys.modules[type(model).__module__].__file__
            record['cpu_memory_loaded']=cpu_memory()
            record['decoder_cpu_seconds']=time.monotonic()-cpu_began
            record['cpu_max_rss_bytes']=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss*1024
            if time.monotonic()-cpu_began>=CPU_DEADLINE: raise TimeoutError('CPU preparation deadline')
            model=None
            component.unload();component=None
            gc.collect()
            record['CPU_model_collected']=model_ref() is None
            record['CPU_parameters_collected']=all(ref() is None for ref in parameter_refs)
            record['CPU_component_collected']=component_ref() is None
            if not record['CPU_parameters_collected'] or not record['CPU_component_collected']: raise RuntimeError('Parameter/component references survived')
            if not record['CPU_model_collected']: raise RuntimeError('CPU model references survived cleanup')
            record['cpu_memory_after_release']=cpu_memory()
            base.journal(output,'CPU-preparation-complete',decoder_cpu=record['decoder_cpu'])
            torch.cuda.synchronize(0)
            torch.cuda.empty_cache()
            record['memory_release']=memory()
            record['memory_peak']={'allocated':torch.cuda.max_memory_allocated(0),'reserved':torch.cuda.max_memory_reserved(0)}
            snap=policy.snapshot();record['policy']=snap
            record['native_calls']=snap['native_calls'];record['guards_rejected_calls']=snap['rejected_calls']
            require_zero(record)
            validate_qualified(record)
            record['classification']='QUALIFIED'
    except BaseException as exc:
        record.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        model=None
        original_factory=timed_factory=None
        if component is not None: component.unload()
        component=None;gc.collect()
        if policy is not None:
            record['policy_final']=policy.snapshot()
            policy.restore_profile();policy.restore_runtime_guards();policy.restore_tensor_guards()
        if torch is not None and torch.cuda.is_initialized():
            try:
                torch.cuda.synchronize(0);torch.cuda.empty_cache()
                record['memory_release']={'allocated':torch.cuda.memory_allocated(0),'reserved':torch.cuda.memory_reserved(0)}
            except BaseException as exc: record['cleanup_error']=repr(exc)
        record['elapsed_seconds']=time.monotonic()-began if began else None
        record['finished_utc']=base.utc_now()
        base.journal(output,'worker-finished',classification=record['classification'])
        base.save_json(output/'results.json',record)
    return 0 if record['classification']=='QUALIFIED' else 1


def supervise(output):
    if sys.platform != "win32":
        raise RuntimeError("Supervisor requires Windows; Linux runs only the gated worker")
    output = output.resolve()
    output.relative_to(ROOT)
    preflight()
    initial_admission=host_admission()
    reserve_attempt(output)
    began = time.monotonic()
    since = base.utc_now()
    helper = base.load_sibling("amd_bounded_gpu_proof")
    monitor = base.HealthMonitor(lambda: helper.windows_observation(since), helper.healthy_observation)
    record = {"classification": "UNKNOWN", "distro": base.DISTRO, "started_utc": since,
              "script_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
              "host_admission_before": initial_admission, "containment_scope": "Windows Job: wsl.exe client; Linux: guest timeout and named-distro termination"}
    child = job = None
    go = None
    observer_failed = False
    def tick(gpu_allowed=True):
        nonlocal observer_failed
        now = time.monotonic()
        if record.get('last_resource_poll',0)+1<=now:
            record['last_resource_poll']=now
            admission=host_admission()
            record['minimum_host_ram_bytes']=min(record.get('minimum_host_ram_bytes',admission['available_ram_bytes']),admission['available_ram_bytes'])
            record['minimum_D_free_bytes']=min(record.get('minimum_D_free_bytes',admission['D_free_bytes']),admission['D_free_bytes'])
        check_deadlines(now, began, go if gpu_allowed else None)
        if go is not None and gpu_allowed:
            phases = output/'guest/phases.jsonl'
            text = phases.read_text() if phases.exists() else ''
            entered_cpu = 'native-qualified-CPU-preparation-entry' in text
            if not entered_cpu and now-go >= NATIVE_DEADLINE:
                raise TimeoutError('60-second native stage deadline')
            if entered_cpu:
                rows=[json.loads(l) for l in text.splitlines() if l.strip()]
                entry=next(r for r in rows if r['phase']=='native-qualified-CPU-preparation-entry')
                # Linux monotonic is a different clock: use host first-observed stage time.
                if record.get('CPU_stage_host_observed') is None: record['CPU_stage_host_observed']=now
                if now-record['CPU_stage_host_observed']>=CPU_DEADLINE:
                    raise TimeoutError('120-second CPU preparation deadline')

        try:
            monitor.poll(now)
        except BaseException:
            observer_failed = True
            raise
    def wait_health(after):
        while not monitor.healthy_since(after):
            tick()
            time.sleep(0.025)
        tick()
        return monitor.latest()
    base.journal(output, "supervisor-start", gpu_allowed=False)
    try:
        before = wait_health(began)
        record["observation_before"] = before
        command = worker_command(base.guest_path(SCRIPT), base.guest_path(output / "guest"))
        record["worker_command"] = command
        job = base.load_sibling("amd_forward_owned_job").OwnedJob()
        with (output / "worker.stdout.txt").open("xb") as stdout, (output / "worker.stderr.txt").open("xb") as stderr:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
            base.journal(output, "wsl-client-started", pid=child.pid, gpu_allowed=False)
            job.attach(child)
            attached = time.monotonic()
            base.journal(output, "windows-job-attached", gpu_allowed=False)
            after = wait_health(attached)
            record["observation_after_attachment"] = after
            tick()
            if child.poll() is not None:
                raise RuntimeError("Worker exited before GO")
            base.authorize_go(child.stdin, True, before, after)
            go = time.monotonic()
            child.stdin.close()
            base.journal(output, "go-sent", gpu_allowed=True)
            while child.poll() is None:
                tick()
                time.sleep(0.025)
            record["worker_returncode"] = child.returncode
            tick()
            base.journal(output, "wsl-client-exited", returncode=child.returncode, gpu_allowed=False)
    except BaseException as exc:
        record.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        # Continue monitoring during cleanup. A failed observer cannot restore PASS.
        def await_cleanup(future, seconds):
            deadline = time.monotonic() + seconds
            while not future.done():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Owned cleanup completion unverified")
                if not observer_failed_local[0]:
                    try:
                        monitor.poll(time.monotonic())
                    except BaseException as exc:
                        record["cleanup_observer_error"] = str(exc)
                        observer_failed_local[0] = True
                time.sleep(0.025)
            return future.result()
        observer_failed_local = [observer_failed]
        if child is not None and child.stdin is not None and not child.stdin.closed:
            child.stdin.close()  # EOF keeps the Linux launch gate closed.
        if job is not None:
            # A normal WSL client can exit milliseconds before its owned
            # wslhost/conhost helpers. Give them a bounded monitored grace;
            # unchanged job.finish() remains the last-resort containment.
            if child is not None and child.poll() is not None and not observer_failed_local[0]:
                grace_began = time.monotonic()
                try:
                    while job.active():
                        if time.monotonic() - grace_began >= 5.0:
                            break
                        check_deadlines(time.monotonic(), began, None)
                        try:
                            monitor.poll(time.monotonic())
                        except BaseException:
                            observer_failed_local[0] = True
                            raise
                        time.sleep(0.025)
                    record["windows_job_grace"] = {"empty": job.active() == 0,
                        "seconds": time.monotonic() - grace_began, "limit_seconds": 5.0}
                except BaseException as exc:
                    record["windows_job_grace_error"] = str(exc)
            try:
                record["windows_job_cleanup"] = await_cleanup(base.in_thread(job.finish), 6.0)
            except BaseException as exc:
                record["windows_job_cleanup_error"] = str(exc)
        if child is not None:
            try:
                record["distro_termination"] = await_cleanup(base.in_thread(base.terminate_owned_distro), 11.0)
                if child.poll() is None:
                    child.wait(timeout=1)
                record.setdefault("worker_returncode", child.returncode)
                termination_finished = time.monotonic()
                base.journal(output, "owned-distro-terminated", capture=record["distro_termination"])
                if not observer_failed_local[0]:
                    while not monitor.healthy_since(termination_finished):
                        check_deadlines(time.monotonic(), began, None)
                        monitor.poll(time.monotonic())
                        time.sleep(0.025)
                    record["observation_after_exit"] = monitor.latest()
            except BaseException as exc:
                record["distro_cleanup_or_post_observer_error"] = str(exc)
        record["observations"] = monitor.observations
        record["observer_healthy"] = not observer_failed_local[0] and "observation_after_exit" in record
        try:
            result = json.loads((output / "guest/results.json").read_text(encoding="utf-8"))
            record["worker"] = result
            if result.get('classification') == 'QUALIFIED':
                validate_qualified(result)
                if ('error' not in record and record.get('worker_returncode') == 0 and record['observer_healthy']
                    and time.monotonic()-began < HOST_DEADLINE
                    and record.get('distro_termination',{}).get('returncode') == 0
                    and record.get('windows_job_cleanup',{}).get('owned_tree_exit_verified') is True
                    and record['windows_job_cleanup'].get('forced_job_cleanup') is False
                    and 'windows_job_grace_error' not in record
                    and 'distro_cleanup_or_post_observer_error' not in record):
                    record['classification']='QUALIFIED'
        except BaseException as exc:
            record["worker_evidence_error"] = type(exc).__name__ + ": " + str(exc)
        record.update(elapsed_seconds=time.monotonic() - began, finished_utc=base.utc_now())
        base.journal(output, "supervisor-finished", classification=record["classification"])
        base.save_json(output / "results.json", record)
    print("WSL_NATIVE_PREPARATION_RESULT=" + record["classification"], flush=True)
    return 0 if record["classification"] == "QUALIFIED" else 1



def preflight():
    if (EV/'CPU-loading-attempt.marker').exists() or (EV/'run').exists(): raise RuntimeError('Exclusive attempt already reserved')
    checkpoint = checkpoint_record(PREFIX_HOST.with_suffix('.safetensors'))
    print('CHECKPOINT_SCHEMA_PASS='+json.dumps(checkpoint), flush=True)
    linux_baseline = ROOT/'docs/compatibility/amd-wsl-shape-decoder-cpu-loading-evidence/linux-preservation-after.json'
    if json.loads(linux_baseline.read_text())['status']!='PASS': raise RuntimeError('Previous Linux preservation gate refused')
    preserved=json.loads((EV/'preservation-before.json').read_text())
    for name,expected in preserved['unchanged_control_code'].items():
        if digest(ROOT/'scripts/compatibility'/name)!=expected: raise RuntimeError('Control identity changed: '+name)
    for name,expected in preserved['loader_code'].items():
        if digest(ROOT/'experimental/extensions/trellis2-amd'/name)!=expected: raise RuntimeError('Existing loader changed')
    helpers={}
    for name in ('amd_bounded_gpu_proof','amd_forward_owned_job'):
        helper=base.load_sibling(name)
        path=(ROOT/'scripts/compatibility'/(name+'.py')).resolve()
        if Path(helper.__file__).resolve()!=path: raise RuntimeError('Wrong helper path')
        helpers[name]={'path':str(path),'sha256':digest(path)}
    if digest(PREFIX_HOST.with_suffix('.json'))!='5f1b856dffce79466fb18f5f0eefc08f08c673b24b3c05b20359c0b6d318a209': raise RuntimeError('Config changed')
    if json.loads(linux_baseline.read_text())['RAM_available_bytes']<CPU_RAM: raise RuntimeError('Recorded guest baseline RAM insufficient')
    command=worker_command(base.guest_path(SCRIPT),base.guest_path(EV/'run/guest'))
    if command[command.index('--worker')-1]!=base.guest_path(SCRIPT): raise RuntimeError('Wrong guest')
    admission = host_admission()
    print('READ_ONLY_PATH_PREFLIGHT='+json.dumps({'helpers':helpers,'selected_guest':str(SCRIPT),
        'worker_command':command,'marker_absent':True,'host_admission':admission,
        'linux_identity_source':str(linux_baseline),'linux_identity_freshly_observed':False}), flush=True)
    # Validate the actual unchanged worker argument types against the existing
    # loader's first path gate. This does not import the decoder, Torch or native code.
    sys.path.insert(0, str(ROOT/'api'))
    from services.amd_runtime import absolute_file
    absolute_file(PREFIX.with_suffix('.json'), 'component configuration')
    absolute_file(PREFIX.with_suffix('.safetensors'), 'component weights')
    return {'helpers':helpers,'selected_guest':str(SCRIPT),'worker_command':command,'marker_absent':True,'host_admission':host_admission()}

PREFIX_HOST=Path('D:/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.preflight: print(json.dumps(preflight()));return 0
    if args.worker is not None: return worker(args.worker)
    if args.output is None: parser.error('--output required')
    return supervise(args.output)

if __name__=='__main__': raise SystemExit(main())
