"""One qualified shape decode from authenticated run inputs; no build or retry.

The Windows owner must supply GO\n plus EOF after attaching containment and
observing health. The owner retains its 480-second deadline and guest timeout;
this entry preserves the 60/120/240 native/CPU/GPU deadlines.
"""
import argparse
import gc
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import weakref

SCRIPT=Path(__file__).resolve()
ROOT=SCRIPT.parents[4]
CODE=SCRIPT.parent
DOCS=ROOT/'docs/compatibility'
EV=DOCS/'amd-wsl-turret-artifact-writer-ready-evidence'
OLD_EV=DOCS/'amd-wsl-native-decoder-preparation-evidence'
PKG_EV=DOCS/'amd-wsl-packaging-decoder-loading-evidence'
STAGE=Path('/home/modly/native-decoder-preparation-6dd94a85')
SOURCE=Path('/mnt/d/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD')
PREFIX=Path('/mnt/d/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16')
GPU_DEADLINE=240.0
NATIVE_DEADLINE=60.0
CPU_DEADLINE=120.0
CPU_RAM=6972719480
GiB=1024**3
sys.dont_write_bytecode=True
sys.path[:0]=[str(CODE),str(ROOT/'scripts/compatibility')]
import amd_wsl_rocdxg_smoke as base
from run_contract import load,digest


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



def validate_qualified(record,contract):
    if contract.large_convolution_tiling:
        from chunked_large import validate_records
        validate_records(record.get('large_tiled_convolutions'))
        if record.get('large_convolution_patch_restored') is not True:
            raise RuntimeError('Large convolution patch was not restored')
    rows=record.get('output',{}).get('rows',0)
    if type(rows) is not int or not 0<rows<=contract.maximum_final_rows:
        raise RuntimeError('Final run support outside qualified bounds')
    d=record.get('decoder_diagnostics',{})
    if (record.get('decoder_cpu')!='PASS' or record.get('native_import')!='PASS'
        or record.get('forward_calls')!=1 or record.get('convolutions_completed')!=40
        or record.get('final_convolution_chunks')!=(rows+32767)//32768 or record.get('final_convolution_rows')!=rows
        or record.get('native_map_calls')!=5 or record.get('final_convolution_patch_restored') is not True
        or len(record.get('subdivision_operations',[]))!=8 or record.get('output',{}).get('verified') is not True
        or record.get('output',{}).get('readback_verified_before_release') is not True
        or not all(record.get(k) is True for k in ('CPU_model_collected','CPU_parameters_collected','CPU_component_collected'))
        or record.get('memory_release')!={'allocated':0,'reserved':0}
        or record.get('run_contract_sha256')!=contract.sha256
        or record.get('source_job_sha256')!=contract.source_job_sha256
        or record.get('workspace_clear_calls')!=1 or record.get('runtime_dtype_counts')!={'BF16':216,'F32':76}
        or record.get('constructed_on_meta') is not True or record.get('buffers_count')!=0
        or d.get('strict') is not True or d.get('roundtrip_verified') is not True or d.get('entries')!=292
        or d.get('dtype_counts')!={'F16':216,'F32':76} or d.get('missing_keys') or d.get('unexpected_keys')
        or 'error' in record or 'cleanup_error' in record or record.get('policy_final',{}).get('rejected_calls')!=0):
        raise RuntimeError('Incomplete strict load/neural decode/zero-release acceptance')


def final_acceptance(record,contract):
    """Classify only the final record, after every cleanup observation completed."""
    try:
        validate_qualified(record,contract)
    except Exception as exc:
        record['classification']='PARTIAL' if record.get('forward_calls')==1 else 'FAIL'
        record['acceptance_error']=str(exc)
        return 1
    record['classification']='QUALIFIED'
    return 0


def cpu_memory():
    values={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')): values[line.split(':')[0]+'_bytes']=int(line.split()[1])*1024
    values['ru_maxrss_bytes']=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss*1024
    return values


def verify_tunable(torch,record,phase):
    flags={'enabled':torch.cuda.tunable.is_enabled(),
           'tuning':torch.cuda.tunable.tuning_is_enabled(),
           'record_untuned':torch.cuda.tunable.record_untuned_is_enabled()}
    environment={key:os.environ.get(key) for key in (
        'PYTORCH_TUNABLEOP_ENABLED','PYTORCH_TUNABLEOP_TUNING','PYTORCH_TUNABLEOP_RECORD_UNTUNED')}
    record.setdefault('tunable_admission',{})[phase]={'flags':flags,'environment':environment}
    if (any(type(value) is not bool for value in flags.values())
        or flags!={'enabled':False,'tuning':False,'record_untuned':False}
        or any(value!='0' for value in environment.values())):
        raise RuntimeError('TunableOp configuration admission failed')


def worker(contract):
    from artifact_paths import claim_directory
    from stage_journal import StageJournal
    output=claim_directory(contract,'output')
    journal=StageJournal(base,output,contract.host_evidence_directory)
    journal.install_guard()
    SCRATCH=Path(str(contract.scratch_directory))
    ARTIFACTS=output
    base.save_json(output/'run-contract.json',contract.document())
    record = {'classification': 'UNKNOWN', 'native_import': 'NOT_ATTEMPTED',
              'decoder_cpu': 'NOT_ATTEMPTED', 'forward_calls': 0,
              'started_utc': base.utc_now(), 'gate_authorized': False}
    torch = policy = component = model = None
    model_ref = None
    began = None
    gpu_started = None
    workspace_cleared = False
    try:
        journal(output, 'gate-wait', gpu_allowed=False)
        base.await_go(sys.stdin.buffer)
        record['gate_authorized'] = True
        began = time.monotonic()
        journal(output, 'gate-authorized', gpu_allowed=True)
        os.environ.update(
            PYTORCH_TUNABLEOP_ENABLED='0',
            PYTORCH_TUNABLEOP_TUNING='0',
            PYTORCH_TUNABLEOP_RECORD_UNTUNED='0')
        record['artifact_directory']={'path':str(output),'guest_local':True}
        record['run_contract_sha256']=contract.sha256
        record['source_job_sha256']=contract.source_job_sha256
        # Own every cache/home/temp write. Never stage in the baseline venv.
        claim_directory(contract,'scratch')
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
        from services.amd_runtime import absolute_file
        config_argument = str(PREFIX.with_suffix('.json'))
        weights_argument = str(PREFIX.with_suffix('.safetensors'))
        config_path = absolute_file(config_argument, 'component configuration')
        weights_path = absolute_file(weights_argument, 'component weights')
        if config_path.as_posix() != config_argument or weights_path.as_posix() != weights_argument:
            raise RuntimeError('Unexpected Linux decoder path resolution')
        if digest(config_path) != '5f1b856dffce79466fb18f5f0eefc08f08c673b24b3c05b20359c0b6d318a209':
            raise RuntimeError('Linux configuration identity mismatch')
        record['linux_path_validation'] = {'config':str(config_path),'weights':str(weights_path),
            'argument_types':[type(config_argument).__name__,type(weights_argument).__name__],
            'before_torch_and_flexgemm':not any(n in sys.modules for n in ('torch','flex_gemm'))}
        if digest(ROOT/'scripts/compatibility/amd_wsl_native_import_policy.py') != '8b9207a14757ccda197e11c054d694f9b4ff47006a1cccc46d71f1d494ab96cd': raise RuntimeError('Qualified policy changed')
        packaging_stage = json.loads((PKG_EV/'packaging-staging.json').read_text())
        if packaging_stage['status']!='PASS' or packaging_stage['version']!='25.0':raise RuntimeError('Packaging staging not qualified')
        for name,row in packaging_stage['installed_files'].items():
            path = STAGE/'site-packages'/name
            if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:raise RuntimeError('Packaging payload changed')
        staged = json.loads((OLD_EV/'staging.json').read_text())
        if staged['target'] != str(STAGE/'site-packages'): raise RuntimeError('Foreign isolated target')
        accepted = json.loads((DOCS/'amd-wsl-flexgemm-build-evidence/static-acceptance.json').read_text())
        wheel = DOCS/'amd-wsl-flexgemm-build-evidence/flex_gemm-1.0.0-cp312-cp312-linux_x86_64.whl'
        if accepted['wheel_sha256']!='cadaa86c428a93c3aee10c4dd8dfb66604a424c8145670051ca0ce1a0f296285' or digest(wheel) != accepted['wheel_sha256']: raise RuntimeError('Native wheel changed')
        import zipfile
        with zipfile.ZipFile(wheel) as archive:
            for info in archive.infolist():
                if info.is_dir() or '.dist-info/' in info.filename: continue
                if (STAGE/'site-packages'/info.filename).read_bytes() != archive.read(info):
                    raise RuntimeError('Native package payload changed: '+info.filename)
        from run_contract import load_inputs
        _,_,record['saved_inputs']=load_inputs(contract)
        source_contract = json.loads((DOCS/'amd-wsl-neural-decoder-preparation-evidence/decoder-contract-readback.json').read_text())
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
        verify_tunable(torch,record,'after_torch_import')
        journal(output,'tunable-after-torch-import',tunable=record['tunable_admission']['after_torch_import'])
        record['runtime'] = {'torch':torch.__version__, 'distribution_version':md.version('torch'),
            'git_version':torch.version.git_version,'hip':torch.version.hip,'cuda':torch.version.cuda,
            'python':platform.python_version(),'prefix':sys.prefix,'gpu_initialized_before':torch.cuda.is_initialized()}
        base.validate_runtime_identity(record['runtime'])
        if md.version('triton') != '3.5.1+rocm7.2.1.gita272dfa8': raise RuntimeError('Triton identity mismatch')
        import triton
        from amd_wsl_native_import_policy import ImportPolicy
        policy = ImportPolicy(SCRATCH, writable_roots=(output, ARTIFACTS, *journal.writable_roots))
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
            journal(output,'native-import-entry',gpu=record['gpu'])
            record['native_import'] = 'ENTERED'
            native_began = time.monotonic()
            import flex_gemm
            from flex_gemm.kernels import cuda
            record['package_path'] = flex_gemm.__file__
            record['native_path'] = cuda.__file__
            if Path(cuda.__file__).resolve() != (STAGE/'site-packages'/accepted['native_member']).resolve() or accepted['native_sha256']!='91f66fa2c0f3567cee71246257bea8eb0fe5af9b6c70617d3c5894aa9f103f01' or digest(cuda.__file__)!=accepted['native_sha256']:
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
            journal(output,'native-qualified-CPU-preparation-entry',memory=memory())
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
            component=ShapeDecoderComponent(
                str(PREFIX.with_suffix('.json')),
                str(PREFIX.with_suffix('.safetensors')),
            )
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
            from runtime import configure
            from trellis2.modules.utils import MIX_PRECISION_MODULES
            from trellis2.modules.sparse.conv import config as conv_config
            from flex_gemm.kernels.triton.spconv import config as triton_config
            changes=configure(model,torch,MIX_PRECISION_MODULES,conv_config,triton_config)
            if len(changes)!=216 or set(changes)!={k for k,v in record['loaded_parameters'].items() if v['dtype']=='torch.float16'}:
                raise RuntimeError('Selective reference BF16 parameter set differs')
            record['runtime_dtype_changes']=changes
            record['runtime_dtype_counts']={'BF16':216,'F32':76}
            model.low_vram=True
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            if not callable(getattr(torch._C,'_cuda_clearCublasWorkspaces',None)):
                raise RuntimeError('Required workspace cleanup binding absent')
            verify_tunable(torch,record,'pre_gpu')
            journal(output,'tunable-pre-gpu',tunable=record['tunable_admission']['pre_gpu'])
            policy.restore_profile();policy.restore_tensor_guards()
        # Import/CPU dispatch scope ended. Compilation/build/audit guards remain.
        gpu_started=time.monotonic()
        journal(output,'gpu-stage-entry',memory=memory())
        import neural
        from neural import run_neural
        run_neural(model,torch,cuda,record,output,gpu_started,contract,journal=journal)
        model=None
        component.unload();component=None;gc.collect()
        record['CPU_model_collected']=model_ref() is None
        record['CPU_parameters_collected']=all(ref() is None for ref in parameter_refs)
        record['CPU_component_collected']=component_ref() is None
        if not all(record[k] for k in ('CPU_model_collected','CPU_parameters_collected','CPU_component_collected')):
            raise RuntimeError('Model/component/parameter references survived')
        torch.cuda.synchronize(0)
        torch._C._cuda_clearCublasWorkspaces();workspace_cleared=True
        torch.cuda.empty_cache()
        record['workspace_clear_calls']=1
        record['memory_release']=memory()
        free,total=torch.cuda.mem_get_info(0)
        record['device_memory_after_release']={'free_bytes':free,'total_bytes':total}
        record['memory_peak']={'allocated':torch.cuda.max_memory_allocated(0),'reserved':torch.cuda.max_memory_reserved(0)}
        record['gpu_stage_seconds']=time.monotonic()-gpu_started
        if record['gpu_stage_seconds']>=GPU_DEADLINE:raise TimeoutError('240-second GPU stage deadline')
        record['policy_final']=policy.snapshot()
    except BaseException as exc:
        record.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        if type(exc).__name__=='AdmissionStop' and record.get('forward_calls')==1:record['classification']='PARTIAL'
    finally:
        model=None
        original_factory=timed_factory=None
        if component is not None: component.unload()
        component=None;gc.collect()
        if model_ref is not None:
            record['CPU_model_collected']=model_ref() is None
            record['CPU_parameters_collected']=all(ref() is None for ref in parameter_refs)
            record['CPU_component_collected']=component_ref() is None
        if policy is not None:
            record['policy_final']=policy.snapshot()
            policy.restore_profile();policy.restore_runtime_guards();policy.restore_tensor_guards()
        if torch is not None and torch.cuda.is_initialized():
            try:
                torch.cuda.synchronize(0)
                if gpu_started is not None and not workspace_cleared:
                    torch._C._cuda_clearCublasWorkspaces();workspace_cleared=True
                    record['workspace_clear_calls']=1
                torch.cuda.empty_cache()
                record['memory_release']={'allocated':torch.cuda.memory_allocated(0),'reserved':torch.cuda.memory_reserved(0)}
                record['memory_peak']={'allocated':torch.cuda.max_memory_allocated(0),'reserved':torch.cuda.max_memory_reserved(0)}
                free,total=torch.cuda.mem_get_info(0)
                record['device_memory_after_release']={'free_bytes':free,'total_bytes':total}
            except BaseException as exc: record['cleanup_error']=repr(exc)
        record['elapsed_seconds']=time.monotonic()-began if began else None
        record['finished_utc']=base.utc_now()
        exit_code=final_acceptance(record,contract)
        journal(output,'worker-finished',classification=record['classification'])
        base.save_json(output/'results.json',record)
    return exit_code



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-contract',type=Path,required=True)
    parser.add_argument('--validate-contract',action='store_true')
    args=parser.parse_args();contract=load(args.run_contract)
    if args.validate_contract:
        print(json.dumps({'status':'PASS','run_contract_sha256':contract.sha256,
                          'AI_modules_imported':any(n in sys.modules for n in ('torch','triton','flex_gemm','trellis2'))}))
        return 0
    if sys.platform!='linux' or Path(str(contract.repo_root)).resolve(strict=True)!=ROOT:
        raise RuntimeError('Execution requires exact qualified guest checkout')
    return worker(contract)


if __name__=='__main__':raise SystemExit(main())
