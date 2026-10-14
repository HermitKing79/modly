"""One genuine contained FlexGEMM import. No binding, model or kernel calls."""
import argparse
import gc
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parent
SITE = Path('/home/modly/native-decoder-preparation-6dd94a85/site-packages')
SCRATCH = Path('/home/modly/flexgemm-native-import-9d439e0')
BUILD = ROOT/'docs/compatibility/amd-wsl-flexgemm-build-evidence'
WHEEL = BUILD/'flex_gemm-1.0.0-cp312-cp312-linux_x86_64.whl'
WHEEL_SHA = 'cadaa86c428a93c3aee10c4dd8dfb66604a424c8145670051ca0ce1a0f296285'
POLICY_SHA = '8292bd7dae51a724aa3a15d5e346b4fe1be8e372adb18f87b58df5a1fdf88058'
GiB = 1024**3

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

smoke = load('existing_smoke', ROOT/'scripts/compatibility/amd_wsl_rocdxg_smoke.py')

def static_preflight(save=True):
    selected = Path(__file__).resolve(strict=True)
    if selected != (E/'native_import.py').resolve(strict=True):
        raise RuntimeError('Unexpected guest script')
    expected = json.loads((ROOT/'docs/compatibility/amd-wsl-launcher-path-correction-evidence/preservation-before.json').read_text())['unchanged_control_code']
    helpers = {}
    rows = {}
    directory = (ROOT/'scripts/compatibility').resolve(strict=True)
    for name in ('amd_bounded_gpu_proof', 'amd_forward_owned_job'):
        path = (directory/(name+'.py')).resolve(strict=True)
        if path.parent != directory or digest(path) != expected[name+'.py']:
            raise RuntimeError('Unverified helper: '+name)
        helpers[name] = load(name, path)
        rows[name] = {'path':str(path), 'sha256':digest(path)}
    if digest(directory/'amd_wsl_native_import_policy.py') != POLICY_SHA:
        raise RuntimeError('Reviewed policy changed')
    if digest(WHEEL) != WHEEL_SHA:
        raise RuntimeError('Original wheel changed')
    accepted = json.loads((BUILD/'static-acceptance.json').read_text())
    with zipfile.ZipFile(WHEEL) as archive:
        if hashlib.sha256(archive.read(accepted['native_member'])).hexdigest() != accepted['native_sha256']:
            raise RuntimeError('Wheel binary mismatch')
    if not (E/'linux-preservation-before.json').is_file():
        raise RuntimeError('Missing passive Linux preflight')
    linux = json.loads((E/'linux-preservation-before.json').read_text())
    if linux['status'] != 'PASS':
        raise RuntimeError('Linux preservation/admission failed')
    windows = json.loads((E/'windows-preservation-before.json').read_text())
    if windows['status'] != 'PASS':
        raise RuntimeError('Windows preservation/admission failed')
    command = smoke.worker_command(smoke.guest_path(selected), smoke.guest_path(E/'run/guest'))
    if command[command.index('--worker')-1] != smoke.guest_path(selected):
        raise RuntimeError('Wrong guest command')
    if (E/'native-attempt-started.json').exists() or (E/'run').exists():
        raise RuntimeError('Exclusive attempt already reserved')
    def fixed_helper(name):
        if name not in helpers:
            raise RuntimeError('Unexpected helper request: '+name)
        return helpers[name]
    record = {'status':'PASS', 'helpers':rows, 'guest':str(selected),
              'script_sha256':digest(selected), 'worker_command':command,
              'marker_absent':True, 'policy_sha256':POLICY_SHA,
              'wheel_sha256':WHEEL_SHA, 'native_sha256':accepted['native_sha256']}
    if save:
        smoke.save_json(E/'helper-preflight.json', record)
    return fixed_helper, record

def guest(output):
    output.mkdir(exist_ok=False)
    record = {'classification':'UNKNOWN', 'native_import':'NOT_ATTEMPTED',
              'started_utc':smoke.utc_now(), 'gate_authorized':False,
              'native_binding_calls':0, 'models_loaded':False, 'neural_forwards':0}
    torch = policy = flex_gemm = cuda = None
    began = None
    def memory():
        return {'allocated':torch.cuda.memory_allocated(0), 'reserved':torch.cuda.memory_reserved(0)}
    try:
        smoke.journal(output, 'gate-wait', gpu_allowed=False)
        smoke.await_go(sys.stdin.buffer)
        began = time.monotonic()
        record['gate_authorized'] = True
        smoke.journal(output, 'gate-authorized', gpu_allowed=True)
        SCRATCH.mkdir(exist_ok=False)
        for name in ('home','tmp','triton-cache','hip-cache','comgr-cache','torch-extensions'):
            (SCRATCH/name).mkdir()
        os.environ.update(HOME=str(SCRATCH/'home'), TMPDIR=str(SCRATCH/'tmp'),
            TRITON_CACHE_DIR=str(SCRATCH/'triton-cache'), HIP_CACHE_DIR=str(SCRATCH/'hip-cache'),
            AMD_COMGR_CACHE_DIR=str(SCRATCH/'comgr-cache'), TORCH_EXTENSIONS_DIR=str(SCRATCH/'torch-extensions'),
            FLEX_GEMM_USE_AUTOTUNE_CACHE='0', FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE='0')
        if digest(WHEEL) != WHEEL_SHA or digest(ROOT/'scripts/compatibility/amd_wsl_native_import_policy.py') != POLICY_SHA:
            raise RuntimeError('Frozen input identity changed')
        accepted = json.loads((BUILD/'static-acceptance.json').read_text())
        payloads = 0
        with zipfile.ZipFile(WHEEL) as archive:
            for info in archive.infolist():
                if info.is_dir() or '.dist-info/' in info.filename:
                    continue
                if (SITE/info.filename).read_bytes() != archive.read(info):
                    raise RuntimeError('Staged native payload changed: '+info.filename)
                payloads += 1
        record['wheel_sha256'] = WHEEL_SHA
        record['staged_payloads_checked'] = payloads
        sys.path[:0] = [str(SITE), str(ROOT/'scripts/compatibility')]
        import torch
        import triton
        record['runtime'] = {'torch':torch.__version__, 'distribution_version':md.version('torch'),
            'git_version':torch.version.git_version, 'hip':torch.version.hip, 'cuda':torch.version.cuda,
            'python':platform.python_version(), 'prefix':sys.prefix, 'gpu_initialized_before':torch.cuda.is_initialized(),
            'triton':md.version('triton'), 'torch_path':torch.__file__, 'triton_path':triton.__file__}
        smoke.validate_runtime_identity(record['runtime'])
        if record['runtime']['python'] != '3.12.3' or record['runtime']['triton'] != '3.5.1+rocm7.2.1.gita272dfa8':
            raise RuntimeError('Pinned Python/Triton mismatch')
        from amd_wsl_native_import_policy import ImportPolicy
        policy = ImportPolicy(SCRATCH, writable_roots=(output,))
        policy.install_tensor_guards(torch)
        policy.install_runtime_guards(triton)
        policy.install_audit_hook()
        policy.install_profile()
        with policy.tensor_mode(torch):
            count = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            record['gpu'] = {'name':props.name, 'architecture':props.gcnArchName, 'count':count,
                'free_bytes':free, 'total_bytes':total, 'property_total_bytes':props.total_memory}
            if count != 1 or 'RX 9070 XT' not in props.name or props.gcnArchName.split(':')[0] != 'gfx1201' or free < 13*GiB:
                raise RuntimeError('GPU identity/free-VRAM admission refused')
            torch.cuda.set_per_process_memory_fraction(12*GiB/total, 0)
            torch.cuda.reset_peak_memory_stats(0)
            record['allocator_cap_bytes'] = 12*GiB
            record['memory_before'] = memory()
            record['gpu_initialized_after_admission'] = torch.cuda.is_initialized()
            if record['memory_before'] != {'allocated':0, 'reserved':0}:
                raise RuntimeError('Unexpected initial Torch allocation')
            if time.monotonic()-began >= 60:
                raise TimeoutError('Native-stage deadline before import')
            record['native_import'] = 'ENTERED'
            smoke.journal(output, 'native-import-entry', gpu=record['gpu'], memory=record['memory_before'])
            entered = time.monotonic()
            import flex_gemm
            from flex_gemm.kernels import cuda
            record['import_host_return_seconds'] = time.monotonic()-entered
            smoke.journal(output, 'native-import-host-return', memory=memory())
            record['package_path'] = flex_gemm.__file__
            record['native_path'] = cuda.__file__
            record['native_sha256'] = digest(cuda.__file__)
            if Path(cuda.__file__).resolve() != (SITE/accepted['native_member']).resolve() or record['native_sha256'] != accepted['native_sha256']:
                raise RuntimeError('Loaded native provenance mismatch')
            if Path(flex_gemm.__file__).resolve() != (SITE/'flex_gemm/__init__.py').resolve():
                raise RuntimeError('Foreign FlexGEMM package')
            from amd_wsl_native_import_policy import NATIVE_BINDINGS
            names = accepted['source_binding_names']
            if tuple(names) != NATIVE_BINDINGS or len(names) != 18 or not all(callable(getattr(cuda, name, None)) for name in names):
                raise RuntimeError('Missing/unexpected native bindings')
            record['bindings'] = names
            record['bindings_count'] = len(names)
            torch.cuda.synchronize(0)
            record['import_synchronized_seconds'] = time.monotonic()-entered
            record['gpu_initialized_after_import'] = torch.cuda.is_initialized()
            record['memory_after_import'] = memory()
            record['vram_after_import'] = dict(zip(('free_bytes','total_bytes'), torch.cuda.mem_get_info(0)))
            record['memory_peak'] = {'allocated':torch.cuda.max_memory_allocated(0), 'reserved':torch.cuda.max_memory_reserved(0)}
            record['driver_helper_inactive'] = (triton.runtime.driver._active is None and triton.runtime.driver._default is None
                and not any(name in sys.modules for name in ('hip_utils','__triton_launcher')))
            if not record['driver_helper_inactive'] or any(record['memory_peak'].values()):
                raise RuntimeError('Unexpected helper activation or Torch allocation')
            record['native_import'] = 'PASS'
            smoke.journal(output, 'native-import-synchronized', memory=record['memory_after_import'])
    except BaseException as exc:
        record.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        if record['native_import'] == 'ENTERED' and isinstance(exc, (ImportError, OSError)) and any(s in str(exc).lower() for s in ('undefined symbol','cannot open shared object','invalid elf','wrong elf','version `')):
            record['classification'] = 'FAIL'
    finally:
        flex_gemm = cuda = None
        gc.collect()
        if torch is not None and torch.cuda.is_initialized():
            try:
                smoke.journal(output, 'release-entry')
                torch.cuda.synchronize(0)
                torch.cuda.empty_cache()
                record['memory_release'] = memory()
                record['vram_after_release'] = dict(zip(('free_bytes','total_bytes'), torch.cuda.mem_get_info(0)))
                smoke.journal(output, 'release-return', memory=record['memory_release'])
            except BaseException as exc:
                record['cleanup_error'] = type(exc).__name__+': '+str(exc)
        if policy is not None:
            record['policy_final'] = policy.snapshot()
            policy.restore_profile()
            policy.restore_runtime_guards()
            policy.restore_tensor_guards()
        record['elapsed_seconds'] = time.monotonic()-began if began else None
        if (record['native_import']=='PASS' and not any('error' in key for key in record)
            and record.get('memory_release')=={'allocated':0,'reserved':0}
            and record.get('policy_final',{}).get('rejected_calls')==0 and record['elapsed_seconds']<60):
            record['classification'] = 'PASS'
        record['finished_utc'] = smoke.utc_now()
        smoke.journal(output, 'worker-finished', classification=record['classification'])
        smoke.save_json(output/'results.json', record)
    return 0 if record['classification']=='PASS' else 1

def host(output):
    from unittest.mock import patch
    if output.resolve() != (E/'run').resolve():
        raise RuntimeError('Exclusive output path required')
    helper, _ = static_preflight()
    with patch.object(smoke,'SCRIPT',Path(__file__).resolve()), patch.object(smoke,'ATTEMPT_MARKER',E/'native-attempt-started.json'), patch.object(smoke,'load_sibling',side_effect=helper):
        raw_exit = smoke.supervise(output)
    raw = json.loads((output/'results.json').read_text())
    worker = raw.get('worker',{})
    passed = (worker.get('classification')=='PASS' and worker.get('gate_authorized') is True
        and raw_exit==1 and raw.get('classification')=='UNKNOWN'
        and raw.get('worker_evidence_error')=='NumericalError: CPU result differs from the exact 256-element F32 reference'
        and worker.get('bindings_count')==18 and worker.get('native_binding_calls')==0
        and worker.get('models_loaded') is False and worker.get('neural_forwards')==0
        and worker.get('memory_release')=={'allocated':0,'reserved':0}
        and worker.get('policy_final',{}).get('rejected_calls')==0
        and raw.get('worker_returncode')==0 and raw.get('observer_healthy') is True
        and raw.get('windows_job_cleanup',{}).get('owned_tree_exit_verified') is True
        and raw['windows_job_cleanup'].get('forced_job_cleanup') is False
        and raw.get('windows_job_grace',{}).get('empty') is True
        and raw.get('distro_termination',{}).get('returncode')==0
        and raw.get('elapsed_seconds',121)<120
        and not any('error' in key for key in raw if key!='worker_evidence_error'))
    # Unchanged GPU supervisor expects vector-add evidence; preserve its raw result.
    classification = 'PASS' if passed else ('FAIL' if worker.get('classification')=='FAIL' else 'UNKNOWN')
    verdict = {'classification':classification,'attempts':1,'native_binding_calls':worker.get('native_binding_calls'),
        'raw_supervisor_exit':raw_exit,'raw_supervisor_classification':raw['classification'],
        'raw_GPU_contract_unused':True,'worker_classification':worker.get('classification')}
    smoke.save_json(output/'native-verdict.json',verdict)
    print(json.dumps(verdict))
    return 0 if passed else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=Path);p.add_argument('--output',type=Path);p.add_argument('--preflight',action='store_true');args=p.parse_args()
    if args.worker is not None:
        raise SystemExit(guest(args.worker))
    if args.preflight:
        print(json.dumps(static_preflight(save=False)[1]));raise SystemExit(0)
    if args.output is None:
        p.error('--output required')
    raise SystemExit(host(args.output))
