"""Experiment-only, binary-GO-gated composition of the qualified brick F16 core.

Importing this adapter is stdlib-only. There is no retry, installation, source
rewrite, geometry runner, alternate forward or replacement receipt capability.
"""
import argparse
import gc
import hashlib
import importlib
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
import time
import traceback
from types import ModuleType
import weakref
import zipfile

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[3]
EV = SCRIPT.parent
FROZEN = 'docs/compatibility/amd-turret-texture-decode-resume-evidence/'
SCRATCH = Path('/home/modly/brick-pbr-gpu-4f91b52')
ARTIFACT_ROOT = Path('/mnt/c/Users/wattz/Documents/ModlyArtifacts/amd-brick-pbr-gpu-4f91b52')
FIELD, PROBE = ARTIFACT_ROOT/'field', ARTIFACT_ROOT/'probe'
STAGE = Path('/home/modly/native-decoder-preparation-6dd94a85')
SOURCE = Path('/mnt/d/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD')
PREFIX = Path('/mnt/d/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16')
GiB = 1024**3
CPU_RAM = 6972719480
NATIVE_DEADLINE, CPU_DEADLINE, GPU_DEADLINE, HOST_DEADLINE = 60.0, 120.0, 240.0, 480.0
POLICY_SHA256 = '8b9207a14757ccda197e11c054d694f9b4ff47006a1cccc46d71f1d494ab96cd'
CONTROL_SHA256 = 'c10dc0afc08be765bfcfb1ddba6acd1aba22ebdb214da0c82bbdb9e12ae58cc2'
ORIGINAL_SHA256 = '8e6d52b84a4be0aceb57f9b942e6bc83f8998decb931c287f40d2a31263b543d'
NATIVE_SHA256 = '91f66fa2c0f3567cee71246257bea8eb0fe5af9b6c70617d3c5894aa9f103f01'
WHEEL_SHA256 = 'cadaa86c428a93c3aee10c4dd8dfb66604a424c8145670051ca0ce1a0f296285'


def read_plain(path, maximum):
    path = Path(path)
    if path != path.resolve(strict=True):
        raise ValueError('Canonical plain file required')
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or getattr(before, 'st_file_attributes', 0) & 1024 or before.st_size > maximum):
        raise ValueError('Bounded plain single-link file required')
    identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_nlink)
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        blob = stream.read(maximum+1)
        after = os.fstat(stream.fileno())
    if (len(blob) != before.st_size or len(blob) > maximum or identity(before) != identity(opened)
            or identity(before) != identity(after) or identity(before) != identity(path.lstat())):
        raise ValueError('Plain file identity changed during read')
    return blob


def digest(path):
    path = Path(path)
    before = path.stat()
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Streamed file identity changed')
    return value.hexdigest()


def required_candidate_paths():
    required = {p.relative_to(ROOT).as_posix() for p in
                (ROOT/'experimental/extensions/trellis2-amd/inference_texture').glob('*.py')}
    required.update(FROZEN+name for name in ('decode.py','runtime.py','neural.py','admission.py',
                                           'chunked_final.py','artifact_io.py','output_owner.py','windows_owner.ps1'))
    required.update(('api/services/amd_runtime.py','api/services/amd_model_metadata.py',
        'experimental/extensions/trellis2-amd/strict_loader.py',
        'experimental/extensions/trellis2-amd/texture_decoder_loader.py',
        'experimental/extensions/trellis2-amd/inference_decoder/run_contract.py',
        'experimental/extensions/trellis2-amd/inference_decoder/artifact_io.py',
        'scripts/compatibility/amd_texture_decoder_inputs.py',
        'scripts/compatibility/amd_texture_sampling_artifact.py',
        'scripts/compatibility/amd_texture_field_artifact.py',
        'scripts/compatibility/amd_wsl_rocdxg_smoke.py',
        'scripts/compatibility/amd_wsl_native_import_policy.py',
        'scripts/compatibility/amd_forward_owned_job.py',
        'scripts/compatibility/amd_bounded_gpu_proof.py',
        'scripts/compatibility/test_amd_brick_pbr_worker.py',
        'docs/compatibility/amd-turret-texture-route-evidence/local-assets.json',
        'docs/compatibility/amd-wsl-native-decoder-preparation-evidence/staging.json',
        'docs/compatibility/amd-wsl-packaging-decoder-loading-evidence/packaging-staging.json',
        'docs/compatibility/amd-wsl-flexgemm-build-evidence/static-acceptance.json',
        'docs/compatibility/amd-wsl-flexgemm-build-evidence/flex_gemm-1.0.0-cp312-cp312-linux_x86_64.whl'))
    required.update((EV/name).relative_to(ROOT).as_posix() for name in ('worker.py','host.py'))
    return required


def verify_manifest(path, expected_sha256, *, root=ROOT, required=None):
    blob = read_plain(path, 2*1024**2)
    if hashlib.sha256(blob).hexdigest() != expected_sha256:
        raise ValueError('Candidate manifest identity differs')
    values = json.loads(blob)
    if type(values) is not dict or not values:
        raise ValueError('Complete candidate source closure required')
    required = required_candidate_paths() if required is None else required
    if not required.issubset(values):
        raise ValueError('Candidate source closure is incomplete')
    root = Path(root).resolve(strict=True)
    for name, expected in values.items():
        if (type(name) is not str or not name or '\\' in name or ':' in name
                or Path(name).is_absolute() or '..' in Path(name).parts or Path(name).as_posix() != name
                or type(expected) is not str or not re.fullmatch('[0-9a-f]{64}', expected)):
            raise ValueError('Candidate requires canonical relative paths and SHA256 values')
        source = root/name
        if source != source.resolve(strict=True) or not source.is_relative_to(root):
            raise ValueError('Candidate source escaped canonical root')
        info = source.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or getattr(info, 'st_file_attributes', 0) & 1024 or digest(source) != expected):
            raise ValueError('Candidate source identity differs: '+name)
    return values


def control():
    name, path = 'amd_wsl_rocdxg_smoke', ROOT/'scripts/compatibility/amd_wsl_rocdxg_smoke.py'
    blob = read_plain(path,128*1024)
    if hashlib.sha256(blob).hexdigest() != CONTROL_SHA256:
        raise ValueError('Frozen binary-GO supervisor helper identity differs')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Execute exactly the bytes just authenticated. SourceFileLoader may use an
    # existing timestamp-valid pyc even with -B/dont_write_bytecode enabled.
    exec(compile(blob,str(path),'exec',dont_inherit=True),vars(module))
    return module


def isolate_bytecode():
    prefix = SCRATCH/'forbidden-bytecode'
    if os.path.lexists(prefix):
        raise FileExistsError('Exclusive nonexistent project bytecode prefix required')
    sys.pycache_prefix = str(prefix)
    sys.dont_write_bytecode = True


def validate_output(output):
    output = Path(output)
    expected = EV/'run/guest'
    if not output.is_absolute() or output != expected or output.resolve(strict=False) != expected:
        raise ValueError('Exact canonical experiment worker output required')
    return output


def authenticate_plan(path, expected_sha256, output):
    blob = read_plain(path, 128*1024)
    if hashlib.sha256(blob).hexdigest() != expected_sha256:
        raise ValueError('Execution plan identity differs')
    plan = json.loads(blob)
    keys = {'schema','harness_commit','manifest_path','manifest_sha256','scratch','field','probe',
            'marker_path','marker_sha256','attestation_path','attestation_sha256',
            'geometry_contract_path','geometry_contract_file_sha256'}
    if type(plan) is not dict or set(plan) != keys or plan['schema'] != 'modly.amd.brick-pbr-execution-plan.v1':
        raise ValueError('Exact experiment execution plan schema required')
    for key in ('manifest_sha256','marker_sha256','attestation_sha256','geometry_contract_file_sha256'):
        if type(plan[key]) is not str or not re.fullmatch('[0-9a-f]{64}', plan[key]):
            raise ValueError('Execution plan SHA256 differs: '+key)
    if not re.fullmatch('[0-9a-f]{40}', plan['harness_commit']):
        raise ValueError('Exact published harness commit required')
    exact = dict(scratch=SCRATCH, field=FIELD, probe=PROBE, marker_path=PROBE/'neural-attempt.marker',
                 attestation_path=PROBE/'ownership.json', manifest_path=EV/'candidate-identities.json',
                 geometry_contract_path=ROOT.parent/'artifacts/amd-brick-neural-completion-c9447dd/neural-output/run-contract.json')
    if any(type(plan[key]) is not str or Path(plan[key]) != value for key, value in exact.items()):
        raise ValueError('Foreign experiment execution path')
    if Path(path) != PROBE/'execution-plan.json' or Path(output) != EV/'run/guest':
        raise ValueError('Exact fresh experiment plan/output required')
    if os.path.lexists(SCRATCH):
        raise FileExistsError('Exclusive experiment scratch already exists')
    for key, hash_key in (('marker_path','marker_sha256'), ('attestation_path','attestation_sha256'),
                          ('geometry_contract_path','geometry_contract_file_sha256')):
        if hashlib.sha256(read_plain(plan[key], 128*1024)).hexdigest() != plan[hash_key]:
            raise ValueError('Execution input identity differs: '+key)
    plan['candidate_identities'] = verify_manifest(plan['manifest_path'], plan['manifest_sha256'])
    plan['execution_plan_sha256'] = expected_sha256
    return plan


def validate_stage_go(ack, ready):
    identity = {key: value for key, value in ready.items() if key != 'schema'}
    admission = ack.get('host_admission', {})
    reserve = ack.get('reserve_assessment', {})
    remaining_reference = {'native':3067854848,'cpu':1143431168}.get(ready.get('stage'))
    if (ack.get('schema') != 'modly.amd.brick-pbr-stage-go.v1' or ack.get('status') != 'GO'
            or any(ack.get(key) != value for key, value in identity.items())
            or ack.get('healthy_observation') is not True or admission.get('admitted') is not True
            or type(admission.get('available_ram_bytes')) is not int or admission['available_ram_bytes'] < 12*GiB
            or type(admission.get('D_free_bytes')) is not int or admission['D_free_bytes'] < 50*GiB
            or reserve.get('admitted') is not True
            or reserve.get('source') != 'closest-historical-whole-host-reference-not-bound'
            or remaining_reference is None or type(reserve.get('remaining_reference_bytes')) is not int
            or reserve['remaining_reference_bytes'] != remaining_reference
            or type(reserve.get('remaining_after_reference_bytes')) is not int
            or reserve['remaining_after_reference_bytes'] != admission['available_ram_bytes']-remaining_reference
            or reserve['remaining_after_reference_bytes'] < 12*GiB):
        raise RuntimeError('Fresh healthy host stage GO/resource/reserve authority refused')


def stage_ready(plan, profile, stage, plan_sha256, deadline):
    from inference_texture import receipts
    if stage not in ('native', 'cpu'):
        raise ValueError('Exact native/CPU stage required')
    ready = dict(schema='modly.amd.brick-pbr-stage-ready.v1', stage=stage,
        marker_sha256=plan['marker_sha256'], attestation_sha256=plan['attestation_sha256'],
        profile_sha256=profile.sha256, run_contract_sha256=profile.run_contract_sha256,
        plan_sha256=plan_sha256)
    began = time.monotonic()
    receipts._publish(Path(plan['probe'])/(stage+'-stage-ready.json'), ready)
    ack = Path(plan['probe'])/(stage+'-stage-go.json')
    while not ack.exists():
        if time.monotonic() >= deadline or time.monotonic()-began >= 30.0:
            raise TimeoutError('Host stage GO deadline expired: '+stage)
        time.sleep(.025)
    value = receipts._decode(ack)
    validate_stage_go(value, ready)
    if time.monotonic() >= deadline or time.monotonic()-began >= 30.0:
        raise TimeoutError('Host stage GO deadline expired: '+stage)
    return value


def process_memory(*, pid=None, proc_root=Path('/proc')):
    """Bounded read-only proc snapshot; never treats RSS sums as whole-host use."""
    pid = os.getpid() if pid is None else pid
    table, raced = {}, 0
    with os.scandir(proc_root) as entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            if len(table)+raced >= 8192:
                raise RuntimeError('Bounded Linux process snapshot exceeded 8192 entries')
            try:
                with (Path(proc_root)/entry.name/'status').open('rb') as stream:
                    blob = stream.read(65537)
                if len(blob) > 65536:
                    raise ValueError('Unbounded proc status')
                rows = dict(line.split(':',1) for line in blob.decode('ascii',errors='replace').splitlines() if ':' in line)
                row = dict(pid=int(rows['Pid']),parent_pid=int(rows['PPid']))
                for key in ('VmRSS','VmHWM','VmSize'):
                    if key in rows:
                        parts = rows[key].split()
                        if len(parts) != 2 or parts[1] != 'kB':
                            raise ValueError('Unexpected proc memory units')
                        row[key+'_bytes'] = int(parts[0])*1024
                table[row['pid']] = row
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                raced += 1
    if pid not in table:
        raise RuntimeError('Worker self memory snapshot absent')
    owned = {pid}
    for _ in range(len(table)):
        children = {key for key,row in table.items() if row['parent_pid'] in owned}
        if children.issubset(owned):
            break
        owned.update(children)
    rows = [table[key] for key in sorted(owned)]
    return dict(scope='self and current descendants; RSS/HWM sums may count shared pages',
        self=table[pid],descendant_pids=sorted(owned-{pid}),processes=rows,
        tree_RSS_sum_bytes=sum(row.get('VmRSS_bytes',0) for row in rows),
        tree_HWM_sum_bytes=sum(row.get('VmHWM_bytes',0) for row in rows),
        process_entries_observed=len(table),raced_or_unreadable_entries=raced)


def sample_guest_memory(record, phase, cpu_memory):
    rows = record.setdefault('guest_process_memory', [])
    if len(rows) >= 64:
        raise RuntimeError('Bounded guest memory event count exceeded')
    rows.append(dict(phase=phase,monotonic_seconds=time.monotonic(),
                     original_self_rss_hwm=cpu_memory(),process_tree=process_memory()))


class WorkerResources:
    """Worker-owned references and idempotent original zero-VRAM settlement."""
    def __init__(self):
        self.torch = self.policy = self.component = self.model = self.prepared = None
        self.model_ref = self.component_ref = None
        self.parameter_refs, self.prepared_refs = [], []
        self.gpu_started = None
        self.workspace_cleared = False
        self.cpu_memory = None

    def release(self, record):
        if self.cpu_memory is not None:
            sample_guest_memory(record,'cleanup-entry',self.cpu_memory)
        self.model = self.prepared = None
        if self.component is not None:
            self.component.unload()
        self.component = None
        gc.collect()
        if self.model_ref is not None:
            record['CPU_model_collected'] = self.model_ref() is None
            record['CPU_parameters_collected'] = all(ref() is None for ref in self.parameter_refs)
            record['CPU_component_collected'] = self.component_ref() is None
        if self.prepared_refs:
            record['CPU_prepared_inputs_collected'] = all(ref() is None for ref in self.prepared_refs)
        if self.torch is not None and self.torch.cuda.is_initialized():
            torch = self.torch
            torch.cuda.synchronize(0)
            if self.gpu_started is not None and not self.workspace_cleared:
                torch._C._cuda_clearCublasWorkspaces()
                self.workspace_cleared = True
                record['workspace_clear_calls'] = 1
            torch.cuda.empty_cache()
            record['memory_release'] = dict(allocated=torch.cuda.memory_allocated(0), reserved=torch.cuda.memory_reserved(0))
            free, total = torch.cuda.mem_get_info(0)
            record['device_memory_after_release'] = dict(free_bytes=free,total_bytes=total)
            record['memory_peak'] = dict(allocated=torch.cuda.max_memory_allocated(0),reserved=torch.cuda.max_memory_reserved(0))
        if self.cpu_memory is not None:
            sample_guest_memory(record,'cleanup-settled',self.cpu_memory)

    def restore_policy(self, record):
        if self.policy is not None:
            record['policy_final'] = self.policy.snapshot()
            self.policy.restore_profile()
            self.policy.restore_runtime_guards()
            self.policy.restore_tensor_guards()


def bootstrap(plan, record):
    if any(name in sys.modules for name in ('torch','triton','flex_gemm','trellis2')):
        raise RuntimeError('Cold experiment worker requires no prior AI/native/model import')
    SCRATCH.mkdir(exist_ok=False)
    for name in ('home','tmp','triton-cache','hip-cache','comgr-cache','torch-extensions'):
        (SCRATCH/name).mkdir()
    os.environ.update(PYTORCH_TUNABLEOP_ENABLED='0',PYTORCH_TUNABLEOP_TUNING='0',PYTORCH_TUNABLEOP_RECORD_UNTUNED='0',
        HOME=str(SCRATCH/'home'), TMPDIR=str(SCRATCH/'tmp'), TRITON_CACHE_DIR=str(SCRATCH/'triton-cache'),
        HIP_CACHE_DIR=str(SCRATCH/'hip-cache'), AMD_COMGR_CACHE_DIR=str(SCRATCH/'comgr-cache'),
        TORCH_EXTENSIONS_DIR=str(SCRATCH/'torch-extensions'), FLEX_GEMM_USE_AUTOTUNE_CACHE='0',
        FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE='0', SPARSE_CONV_BACKEND='flex_gemm', HSA_ENABLE_DXG_DETECTION='1',
        HIP_VISIBLE_DEVICES='0',ROCR_VISIBLE_DEVICES='0',CUDA_VISIBLE_DEVICES='0')
    sys.path[:0] = [str(STAGE/'site-packages'),str(ROOT/'api'),
                   str(ROOT/'experimental/extensions/trellis2-amd'),str(ROOT/'scripts/compatibility')]
    from inference_texture import profile, inputs, runtime, neural, receipts
    from inference_texture._source import load_pinned
    private_sys = ModuleType('sys')
    vars(private_sys).update(vars(sys))
    private_sys.path = list(sys.path)
    original = load_pinned('postrestart_original_worker', FROZEN+'decode.py', ORIGINAL_SHA256, imports={'sys': private_sys})
    authenticated = profile.authenticate(profile.NAME, profile.contracts.load(plan['geometry_contract_path']).document())
    bridge = receipts.ReceiptBridge(FIELD, PROBE, PROBE/'neural-attempt.marker', plan['marker_sha256'],
        profile=authenticated, attestation_path=PROBE/'ownership.json',attestation_sha256=plan['attestation_sha256'])
    if bridge.mode != 'production':
        raise ValueError('Production live WSL ownership authority required')
    record['artifact_directory'] = dict(root=str(ARTIFACT_ROOT),field=str(FIELD),probe=str(PROBE),
        production_WSL_factory_qualified=True, actual_worker_identity=[os.getuid(),os.geteuid(),os.getgid(),os.getegid()])
    record['bootstrap_scratch'] = str(SCRATCH)
    return authenticated, inputs, runtime, neural, bridge, original


def qualify_staged_native():
    if digest(ROOT/'scripts/compatibility/amd_wsl_native_import_policy.py') != POLICY_SHA256:
        raise ValueError('Qualified native policy changed')
    packaging = json.loads((ROOT/'docs/compatibility/amd-wsl-packaging-decoder-loading-evidence/packaging-staging.json').read_text())
    if packaging['status'] != 'PASS' or packaging['version'] != '25.0':
        raise ValueError('Packaging staging is not qualified')
    for name, row in packaging['installed_files'].items():
        path = STAGE/'site-packages'/name
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Packaging staged payload changed')
    staged = json.loads((ROOT/'docs/compatibility/amd-wsl-native-decoder-preparation-evidence/staging.json').read_text())
    if staged['target'] != str(STAGE/'site-packages'):
        raise ValueError('Foreign native isolated stage')
    directory = ROOT/'docs/compatibility/amd-wsl-flexgemm-build-evidence'
    accepted = json.loads((directory/'static-acceptance.json').read_text())
    wheel = directory/'flex_gemm-1.0.0-cp312-cp312-linux_x86_64.whl'
    if accepted['native_sha256'] != NATIVE_SHA256 or accepted['wheel_sha256'] != WHEEL_SHA256 or digest(wheel) != WHEEL_SHA256:
        raise ValueError('Original native/wheel source identity changed')
    with zipfile.ZipFile(wheel) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or '.dist-info/' in entry.filename:
                continue
            if (STAGE/'site-packages'/entry.filename).read_bytes() != archive.read(entry):
                raise ValueError('Native package payload changed: '+entry.filename)
    return accepted


def guest_available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:')))*1024


def execute_admitted(plan, record, resources, output):
    base = control()
    began = time.monotonic()
    profile, inputs, runtime, neural, bridge, original = bootstrap(plan, record)
    resources.cpu_memory = original.cpu_memory
    sample_guest_memory(record,'runtime-entry-before-torch',original.cpu_memory)
    accepted = qualify_staged_native()
    config_argument, weights_argument = str(PREFIX.with_suffix('.json')), str(PREFIX.with_suffix('.safetensors'))
    config_path = runtime._runtime.absolute_file(config_argument, 'component configuration')
    weights_path = runtime._runtime.absolute_file(weights_argument, 'component weights')
    if config_path.as_posix() != config_argument or weights_path.as_posix() != weights_argument or digest(config_path) != runtime.loader.CONFIG_SHA256:
        raise ValueError('Original Linux configuration/path identity differs')
    record['linux_path_validation'] = dict(config=str(config_path),weights=str(weights_path),
        argument_types=[type(config_argument).__name__,type(weights_argument).__name__],
        before_torch_and_flexgemm=not any(name in sys.modules for name in ('torch','flex_gemm')))
    resources.prepared = inputs.prepare(profile)
    inputs.require_prepared(resources.prepared, profile)
    record['saved_inputs'] = resources.prepared['metadata']
    resources.prepared_refs = [weakref.ref(resources.prepared[key]) for key in ('denormalized','coords','geometry_coords')]
    resources.prepared_refs.extend(weakref.ref(value) for guide in resources.prepared['guides'] for value in guide.values())
    metadata_component = runtime.loader.TextureDecoderComponent(str(config_path),str(weights_path))
    metadata_component._verify_source()
    record['source_sha256'] = dict(runtime.loader.SOURCE_SHA256)
    metadata_component.unload()
    metadata_component = None
    record['checkpoint'] = dict(original.checkpoint_record(weights_path),guest_path=str(weights_path))
    available, fs, dfs = guest_available(), os.statvfs('/'), os.statvfs('/mnt/d')
    record['admission'] = dict(guest_ram_available=available,guest_disk_free=fs.f_bavail*fs.f_frsize,
        D_disk_free=dfs.f_bavail*dfs.f_frsize,CPU_ram_requirement=CPU_RAM,guest_disk_requirement=5*GiB,D_disk_requirement=50*GiB)
    if available < CPU_RAM or fs.f_bavail*fs.f_frsize < 5*GiB or dfs.f_bavail*dfs.f_frsize < 50*GiB:
        raise RuntimeError('Original CPU/disk admission refused')
    torch = resources.torch = importlib.import_module('torch')
    original.verify_tunable(torch,record,'after_torch_import')
    base.journal(output,'tunable-after-torch-import',tunable=record['tunable_admission']['after_torch_import'])
    record['runtime'] = dict(torch=str(torch.__version__),distribution_version=md.version('torch'),
        git_version=torch.version.git_version,hip=torch.version.hip,cuda=torch.version.cuda,
        python=platform.python_version(),prefix=sys.prefix,gpu_initialized_before=torch.cuda.is_initialized())
    base.validate_runtime_identity(record['runtime'])
    sample_guest_memory(record,'runtime-identity-qualified',original.cpu_memory)
    if md.version('triton') != '3.5.1+rocm7.2.1.gita272dfa8':
        raise RuntimeError('Original Triton identity differs')
    triton = importlib.import_module('triton')
    from amd_wsl_native_import_policy import ImportPolicy
    policy = resources.policy = ImportPolicy(SCRATCH,writable_roots=(output,FIELD,PROBE))
    policy.install_tensor_guards(torch)
    policy.install_runtime_guards(triton)
    policy.install_audit_hook()
    policy.install_profile()
    memory = lambda: dict(allocated=torch.cuda.memory_allocated(0),reserved=torch.cuda.memory_reserved(0))
    with policy.tensor_mode(torch):
        count, props = torch.cuda.device_count(), torch.cuda.get_device_properties(0)
        vfree, total = torch.cuda.mem_get_info(0)
        record['gpu'] = dict(name=props.name,architecture=props.gcnArchName,count=count,total_bytes=total,free_bytes=vfree,property_total_bytes=props.total_memory)
        if count != 1 or 'RX 9070 XT' not in props.name or props.gcnArchName.split(':')[0] != 'gfx1201' or vfree < 13*GiB:
            raise RuntimeError('Original GPU identity/free-VRAM admission refused')
        torch.cuda.set_per_process_memory_fraction(12*GiB/total,0)
        torch.cuda.reset_peak_memory_stats(0)
        record['memory_before'] = memory()
        if record['memory_before'] != {'allocated':0,'reserved':0}:
            raise RuntimeError('Unexpected initial GPU allocation')
        record['gpu_initialized_after_admission'] = torch.cuda.is_initialized()
        base.journal(output,'native-import-entry',gpu=record['gpu'])
        record['native_import'] = 'ENTERED'
        native_began = time.monotonic()
        flex_gemm = importlib.import_module('flex_gemm')
        cuda = importlib.import_module('flex_gemm.kernels.cuda')
        record['package_path'], record['native_path'] = flex_gemm.__file__, cuda.__file__
        if Path(cuda.__file__).resolve() != (STAGE/'site-packages'/accepted['native_member']).resolve() or digest(cuda.__file__) != NATIVE_SHA256:
            raise RuntimeError('Original native module provenance differs')
        names = accepted['source_binding_names']
        if len(names) != 18 or not all(callable(getattr(cuda,name,None)) for name in names):
            raise RuntimeError('Missing original 18 native bindings')
        record['bindings'], record['bindings_count'] = names, len(names)
        torch.cuda.synchronize(0)
        record['native_import_synchronized_seconds'] = time.monotonic()-native_began
        record['gpu_initialized_after_import'], record['native_memory_after'] = torch.cuda.is_initialized(), memory()
        if time.monotonic()-began >= NATIVE_DEADLINE or memory() != {'allocated':0,'reserved':0}:
            raise RuntimeError('Original native import deadline/allocation gate failed')
        record['driver_helper_inactive'] = (triton.runtime.driver._active is None and triton.runtime.driver._default is None
                                          and not any(name in sys.modules for name in ('hip_utils','__triton_launcher')))
        if not record['driver_helper_inactive']:
            raise RuntimeError('Unexpected Triton driver/helper activation')
        record['native_import'] = 'PASS'
        record['native_stage_host_go'] = stage_ready(plan,profile,'native',record['execution_plan_sha256'],began+NATIVE_DEADLINE)
        base.journal(output,'native-qualified-CPU-preparation-entry',memory=memory())
        policy.set_phase('cpu_meta')
        cpu_began = time.monotonic()
        record['cpu_memory_before_loading'], record['guest_RAM_before_loading'] = original.cpu_memory(), guest_available()
        if record['guest_RAM_before_loading'] < CPU_RAM:
            raise RuntimeError('Fresh CPU load RAM admission refused')
        resources.component = runtime.loader.TextureDecoderComponent(str(config_path),str(weights_path))
        original_factory = resources.component._component.factory
        def timed_factory(config):
            construction_began = time.monotonic()
            constructed = original_factory(config)
            record['construction_seconds'] = time.monotonic()-construction_began
            record['constructed_on_meta'] = all(parameter.is_meta for parameter in constructed.parameters())
            if not record['constructed_on_meta']:
                raise RuntimeError('Genuine decoder not constructed on meta')
            return constructed
        resources.component._component.factory = timed_factory
        loading_began = time.monotonic()
        try:
            resources.model = resources.component.load_cpu()
        finally:
            resources.component._component.factory = original_factory
            original_factory = timed_factory = None
        record['strict_load_and_widen_seconds'] = time.monotonic()-loading_began
        sample_guest_memory(record,'strict-load-return',original.cpu_memory)
        resources.model_ref, resources.component_ref = weakref.ref(resources.model), weakref.ref(resources.component)
        resources.parameter_refs = [weakref.ref(parameter) for parameter in resources.model.parameters()]
        model = resources.model
        record['loaded_parameters'] = {name:dict(shape=list(parameter.shape),dtype=str(parameter.dtype),device=str(parameter.device),requires_grad=parameter.requires_grad)
                                       for name,parameter in model.named_parameters()}
        if model.training or any(module.training for module in model.modules()) or any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError('Genuine decoder evaluation/gradient state differs')
        record['evaluation_and_gradients_disabled'], record['buffers_count'] = True, len(list(model.named_buffers()))
        record['decoder_cpu'], record['decoder_class'] = 'PASS', type(model).__module__+'.'+type(model).__name__
        record['decoder_diagnostics'] = json.loads(json.dumps(resources.component.diagnostics))
        record['decoder_source_path'] = sys.modules[type(model).__module__].__file__
        record['cpu_memory_loaded'], record['decoder_cpu_seconds'] = original.cpu_memory(), time.monotonic()-cpu_began
        from trellis2.modules.utils import MIX_PRECISION_MODULES
        from trellis2.modules.sparse.conv import config as conv_config
        from flex_gemm.kernels.triton.spconv import config as triton_config
        changes = runtime.configure(model,torch,MIX_PRECISION_MODULES,conv_config,triton_config)
        if changes != {}:
            raise RuntimeError('Original F16/F32 precision changed')
        runtime.validate_configured_model(model,torch)
        record['runtime_dtype_changes'], record['runtime_dtype_counts'] = changes, {'F16':208,'F32':76}
        if not callable(getattr(torch._C,'_cuda_clearCublasWorkspaces',None)):
            raise RuntimeError('Original workspace cleanup binding absent')
        original.verify_tunable(torch,record,'pre_gpu')
        base.journal(output,'tunable-pre-gpu',tunable=record['tunable_admission']['pre_gpu'])
        model = None
        record['cpu_stage_host_go'] = stage_ready(plan,profile,'cpu',record['execution_plan_sha256'],cpu_began+CPU_DEADLINE)
        if time.monotonic()-cpu_began >= CPU_DEADLINE:
            raise TimeoutError('120-second CPU preparation deadline')
        policy.restore_profile()
        policy.restore_tensor_guards()
    resources.gpu_started = time.monotonic()
    base.journal(output,'gpu-stage-entry',memory=memory())
    def emit_neural(event, **details):
        phases = {'texture-forward-entry':'forward-entry',
                  'texture-forward-host-return':'forward-host-return-before-synchronization',
                  'texture-forward-synchronized':'pre-persistence-before-cpu-copy',
                  'texture-field-readback-verified':'post-persistence-independent-readback'}
        if event in phases:
            sample_guest_memory(record,phases[event],original.cpu_memory)
        base.journal(output,event,**details)
    neural.run_neural(resources.model,torch,cuda,record,PROBE,resources.gpu_started,
        profile=profile,prepared=resources.prepared,write_output=bridge.write_output,
        emit=emit_neural)
    resources.release(record)
    if not all(record.get(key) is True for key in ('CPU_model_collected','CPU_parameters_collected','CPU_component_collected','CPU_prepared_inputs_collected')):
        raise RuntimeError('Owned model/component/parameter/prepared input references survived')
    record['gpu_stage_seconds'] = time.monotonic()-resources.gpu_started
    if record['gpu_stage_seconds'] >= GPU_DEADLINE or time.monotonic()-began >= HOST_DEADLINE:
        raise TimeoutError('Original bounded worker stage deadline expired')
    record['policy_final'] = policy.snapshot()
    runtime.validate_result(record,profile=profile)
    record['classification'] = 'QUALIFIED'


def worker(output, plan_path, plan_sha256):
    output = validate_output(output)
    isolate_bytecode()
    output.mkdir(exist_ok=False)
    base = control()
    record = dict(classification='UNKNOWN',native_import='NOT_ATTEMPTED',decoder_cpu='NOT_ATTEMPTED',
        forward_calls=0,started_utc=base.utc_now(),gate_authorized=False,execution_plan_sha256=plan_sha256)
    resources, began = WorkerResources(), None
    try:
        plan = authenticate_plan(plan_path,plan_sha256,output)
        record['candidate_identities'] = plan['candidate_identities']
        record['adapter_source_sha256'] = dict(plan['candidate_identities'])
        record['source_manifest_sha256'] = plan.get('manifest_sha256')
        record['harness_commit'] = plan.get('harness_commit')
        base.journal(output,'gate-wait',gpu_allowed=False)
        base.await_go(sys.stdin.buffer)
        record['gate_authorized'] = True
        began = time.monotonic()
        base.journal(output,'gate-authorized',gpu_allowed=True)
        execute_admitted(plan,record,resources,output)
    except BaseException as exc:
        record.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        exc.__traceback__ = None
        if type(exc).__name__ == 'AdmissionStop' and record.get('forward_calls') == 1:
            record['classification'] = 'PARTIAL'
    finally:
        try:
            resources.release(record)
        except BaseException as exc:
            record['cleanup_error'] = type(exc).__name__+': '+str(exc)
            record['classification'] = 'UNKNOWN'
            exc.__traceback__ = None
        try:
            resources.restore_policy(record)
        except BaseException as exc:
            record['policy_cleanup_error'] = type(exc).__name__+': '+str(exc)
            record['classification'] = 'UNKNOWN'
            exc.__traceback__ = None
        record.update(elapsed_seconds=time.monotonic()-began if began is not None else None,finished_utc=base.utc_now())
        base.journal(output,'worker-finished',classification=record['classification'])
        base.save_json(output/'results.json',record)
    return 0 if record['classification'] == 'QUALIFIED' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Only the exact host-supervised WSL worker is executable')
    return worker(args.worker,args.plan,args.plan_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
