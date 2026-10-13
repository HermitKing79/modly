"""CPU-only shape-decoder audit. This entry point deliberately cannot run a GPU attempt."""
import argparse
import ast
import base64
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'api'), str(Path(__file__).parent)]
SOURCE = Path('D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD')
VENV = Path('D:/AI/Trellis2AMD/venv')
SITE = VENV / 'Lib/site-packages'
PREFIX = Path('D:/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16')
EV = ROOT / 'docs/compatibility/amd-shape-decoder-neural-evidence'
SAMPLE = ROOT / 'docs/compatibility/amd-shape-slat-512-sampling-evidence'
INPUTS = {
    'shape-latent-denormalized.npy': '90eaf72030f303b209a4d02ae9adc7d4ecd39662344f35f0adc8120bc1e871b9',
    'shape-latent-coordinates.npy': '949a9e6a2548abcad66354384502c06834d4d56ea3f2fa86b619c25926379ed4',
    'shape-latent-provenance.json': '4834412e700434f5645cd919dd0d75e9209b6b5d67cc7cb765e98b76890f9a81',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def read_npy(path, expected_hash):
    import numpy as np
    if path.stat().st_size > 2 * 1024**2 or digest(path) != expected_hash:
        raise ValueError('Saved input hash/size mismatch: ' + str(path))
    with path.open('rb') as stream:
        a = np.load(stream, allow_pickle=False, max_header_size=4096)
        if stream.read(1): raise ValueError('Trailing NPY bytes')
    return a


def validate_arrays(x, c, *, count=3743):
    import numpy as np
    if x.shape != (count, 32) or x.dtype != np.float32 or c.shape != (count, 4) or c.dtype != np.int32:
        raise ValueError('Expected denormalized F32 [N,32] and I32 [N,4]')
    if not x.flags.c_contiguous or not c.flags.c_contiguous or not np.isfinite(x).all():
        raise ValueError('Nonfinite/noncontiguous saved latent')
    if (c[:, 0] != 0).any() or (c[:, 1:] < 0).any() or (c[:, 1:] >= 32).any() or len(np.unique(c, axis=0)) != count:
        raise ValueError('Invalid batch/duplicate/out-of-bounds sparse coordinates')
    return dict(tokens=count, features_shape=list(x.shape), coordinates_shape=list(c.shape),
                coordinate_order='batch,D,H,W; preserved row order', bounds=[c[:, 1:].min(0).tolist(), c[:, 1:].max(0).tolist()],
                payload_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                coordinates_payload_sha256=hashlib.sha256(c.tobytes()).hexdigest(),
                minimum=float(x.min()), maximum=float(x.max()), mean=float(x.mean(dtype=np.float64)),
                standard_deviation=float(x.std(dtype=np.float64)), finite=True)


def fp32_key(key):
    return key in ('from_latent.weight', 'from_latent.bias', 'output_layer.weight', 'output_layer.bias') or bool(
        re.fullmatch(r'blocks\.\d+\.\d+\.norm1?\.(weight|bias)', key))


def validate_state(header, parameters, buffers):
    if buffers or set(header) != set(parameters): raise ValueError('Complete parameter/buffer key mismatch')
    counts = {'F16': 0, 'F32': 0}; elements = {'F16': 0, 'F32': 0}; widened = []
    for key, row in header.items():
        p = parameters[key]; dtype = 'F32' if fp32_key(key) else 'F16'
        if row['dtype'] != 'F16' or row['shape'] != p['shape'] or p['dtype'] != dtype or p['device'] != 'meta':
            raise ValueError('Shape/precision/metadata-device mismatch: ' + key)
        counts[dtype] += 1; elements[dtype] += math.prod(row['shape'])
        if dtype == 'F32': widened.append(key)
    return dict(entries=len(header), dtype_counts=counts, dtype_elements=elements, fp32_keys=sorted(widened),
                parameter_bytes=elements['F16'] * 2 + elements['F32'] * 4, persistent_buffers=0,
                weights_loaded=False, value_integrity_at_real_load='not exercised; require unchanged StrictLocalComponent plus scoped reversible F16-to-F32 verification')


def growth_budget(initial, parameter_bytes):
    support = [initial * 8**stage for stage in range(5)]
    # Conservative sum of visible neighbor-map/postprocess allocations, including
    # hashmap keys/values. Not a guarantee for sort/cumsum/allocator workspace.
    neighbor = 680 * support[-1] + 112
    # Final C2S conv2 input:64 F16 channels; child skip input:128/8=16.
    # Its output, widened/chunked activations and skip projection are extra.
    features = support[-1] * (64 + 16) * 2
    return dict(maximum_support=support, resolution=[32, 64, 128, 256, 512],
                native_neighbor_visible_bytes=neighbor, primary_final_features_bytes=features,
                model_parameter_bytes=parameter_bytes, worst_visible_final_bytes=neighbor + features + parameter_bytes,
                allocator_cap_bytes=12 * 1024**3, free_vram_admission_bytes=13 * 1024**3,
                unconditional_12gib_completion_established=False,
                limitations='Conservative visible sum excludes sorting/cumsum workspace, retained earlier-stage caches, outputs/subdivision tensors and other temporaries; learned support is unknown. A pre-call support/memory gate is still required.')


def validate_cache_metadata(row):
    if row.get('target') != {'backend': 'hip', 'arch': 'gfx1201', 'warp_size': 32} or row.get('triton_version') != '3.8.0':
        raise ValueError('Foreign Triton cache target/version')
    if any(row.get(k) != 0 for k in ('global_scratch_size', 'profile_scratch_size')):
        raise ValueError('Unbounded cached-kernel scratch')
    if not re.fullmatch('[0-9a-f]{64}', row.get('hash', '')): raise ValueError('Invalid cache digest')
    return dict(target=row['target'], current_source_cache_key_verified=False,
                loaded_or_executed=False, global_scratch_size=0, profile_scratch_size=0)


def cache_group_paths(metadata_path, children):
    expected = {metadata_path.stem + suffix for suffix in ('.source', '.ttir', '.ttgir', '.llir', '.amdgcn', '.hsaco', '.json')}
    if not isinstance(children, dict) or set(children) != expected: raise ValueError('Incomplete/foreign cache container')
    result = {}
    for name, original in children.items():
        if not isinstance(original, str): raise ValueError('Malformed cache child path')
        p = Path(original)
        # NTFS file identity handles the existing \\?\ long-path prefix without
        # accepting another directory or rewriting any historical manifest.
        if p.name != name or not p.is_file() or not os.path.samefile(p.parent, metadata_path.parent):
            raise ValueError('Foreign/missing cache group child')
        result[name] = p
    return result


def checkpoint_header():
    from services.amd_model_metadata import json_metadata, safetensors_metadata
    actual = safetensors_metadata(PREFIX.with_suffix('.safetensors'))
    original = json.loads((ROOT / 'docs/compatibility/amd-offline-model-audit-evidence/final-assets/shape_dec_next_dc_f16c32_fp16.metadata.json').read_text(encoding='utf-8'))
    config = json_metadata(PREFIX.with_suffix('.json'))
    for key in ('file_bytes', 'mtime_ns', 'header_bytes', 'header_sha256', 'tensor_count', 'parameter_count', 'tensors', 'tensor_data_bytes'):
        if actual[key] != original[key]: raise ValueError('Audited checkpoint changed: ' + key)
    if config != original['config'] or digest(SOURCE / 'trellis2/models/sc_vaes/fdg_vae.py') != original['source_module_sha256']:
        raise ValueError('Audited architecture/configuration changed')
    return actual, config


def metadata_worker(scratch):
    """Genuine parent constructor, with its exact backend initializer on meta only.

    The source initializer is extracted unchanged to avoid FlexGEMM's import-time
    GPU discovery. No decoder/native forward is supplied or executed. This
    deliberately proves a parameter manifest, NOT backend import/execution.
    """
    record = dict(status='FAIL', weights_loaded=False, gpu_initialized=False, model_forward_calls=0)
    torch = None
    try:
        if sys.stdin.readline().strip() != 'OWNED_JOB_ATTACHED': raise RuntimeError('Owned Job required')
        from services.amd_offline_guard import block_external_actions
        from amd_bounded_gpu_proof import scratch_environment, CACHE_KEYS
        scratch = scratch.resolve(); scratch.relative_to(EV.resolve())
        if scratch.name != 'scratch' or not scratch.is_dir(): raise ValueError('Owned CPU scratch required')
        env = scratch_environment(scratch)
        for key in CACHE_KEYS:
            if key != 'FLEX_GEMM_AUTOTUNE_CACHE_PATH': Path(env[key]).mkdir(parents=True, exist_ok=True)
        os.environ.update(env)
        discovery = []; block_external_actions(VENV / 'Scripts/rocm-sdk.exe', discovery, scratch)
        def deny(*a, **k): raise RuntimeError('CPU manifest forbids GPU, model loading, forward and compilation')
        def audit(event, args):
            if event.startswith('socket.'): return deny()
            if event == 'import' and (str(args[0]).split('.')[0] in ('flex_gemm', 'cumesh', 'o_voxel', 'nvdiffrast') or str(args[0]) == 'triton.backends.amd.driver'): return deny()
            if event == 'open' and isinstance(args[0], (str, bytes)) and Path(os.fsdecode(args[0])).suffix.lower() in ('.safetensors', '.pt', '.pth', '.ckpt', '.gguf'): return deny()
        # Read ONLY checkpoint/configuration metadata before enforcing payload-open denial.
        header, config = checkpoint_header()
        sys.addaudithook(audit)
        import torch
        if torch.cuda.is_initialized(): return deny()
        if sys.version.split()[0] != '3.12.10' or str(torch.__version__) != '2.13.0+rocm10.0.0' or torch.version.cuda is not None or not torch.version.hip:
            raise ValueError('Pinned AMD runtime identity mismatch')
        torch.cuda._lazy_init = deny; torch.load = deny; torch.compile = deny; torch.nn.Module.__call__ = deny
        # PyTorch's meta-device dispatcher imports Dynamo, which imports Triton
        # for package detection. Import alone is CPU-only; executable paths stay denied.
        import triton
        import triton.runtime.build
        triton.compile = deny; triton.runtime.jit.JITFunction.run = deny
        triton.runtime.Autotuner.run = deny; triton.runtime.build._build = deny
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_leaves
        class MetaOnly(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                for v in tree_leaves((args, kwargs or {})):
                    if isinstance(v, (torch.Tensor, torch.device)) and (v.device.type if isinstance(v, torch.Tensor) else v.type) != 'meta': return deny()
                return func(*args, **(kwargs or {}))
        sys.path.insert(0, str(SOURCE))
        conv = importlib.import_module('trellis2.modules.sparse.conv.conv')
        parent = importlib.import_module('trellis2.models.sc_vaes.sparse_unet_vae').SparseUnetVaeDecoder
        initializer = SOURCE / 'trellis2/modules/sparse/conv/conv_flex_gemm.py'
        tree = ast.parse(initializer.read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sparse_conv3d_init')
        namespace = dict(torch=torch, nn=torch.nn, math=math)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(initializer), 'exec'), namespace)
        conv.config.set_conv_backend('flex_gemm')
        conv._backends['flex_gemm'] = SimpleNamespace(sparse_conv3d_init=namespace['sparse_conv3d_init'], sparse_conv3d_forward=deny)
        args = dict(config['args']); args.pop('resolution')
        with MetaOnly(), torch.device('meta'):
            model = parent(out_channels=7, **args)
            model.eval(); model.low_vram = True
            def manifest(values):
                return {k: dict(shape=list(v.shape), dtype={torch.float32:'F32', torch.float16:'F16'}[v.dtype], device=v.device.type) for k, v in values}
            parameters = manifest(model.named_parameters()); buffers = manifest(model.named_buffers())
            precision = validate_state(header['tensors'], parameters, buffers)
            modules = {name: type(m).__name__ for name, m in model.named_modules()}
            record.update(precision=precision, parameters=parameters, buffers=buffers, modules=modules,
                          parent_source=str(SOURCE / 'trellis2/models/sc_vaes/sparse_unet_vae.py'),
                          backend_initializer_source_sha256=digest(initializer), initializer_policy='unchanged AST initializer only; native forward denied',
                          low_vram=True, native_backend_imported=False, checkpoint_payload_read=False)
            model = None
        record.update(gpu_initialized=torch.cuda.is_initialized(),
                      runtime=dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__), hip=torch.version.hip, cuda=torch.version.cuda, torch_path=torch.__file__))
        if record['gpu_initialized']: return deny()
        record.update(status='PASS', harness_sha256=digest(Path(__file__)))
    except BaseException as exc: record.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
    print('SHAPE_METADATA=' + json.dumps(record), flush=True)
    return 0 if record['status'] == 'PASS' else 1


def metadata_passed(record):
    w = record.get('worker', {})
    return (not record.get('error') and record.get('returncode') == 0 and w.get('status') == 'PASS' and w.get('gpu_initialized') is False
            and w.get('weights_loaded') is False and type(w.get('model_forward_calls')) is int and w['model_forward_calls'] == 0
            and not w.get('error') and record.get('worker_exit_verified') is True
            and record.get('job_cleanup', {}).get('owned_tree_exit_verified') is True
            and record['job_cleanup'].get('forced_job_cleanup') is False)


def supervise_metadata(name):
    from amd_forward_owned_job import OwnedJob
    from amd_bounded_gpu_proof import scratch_environment
    if not re.fullmatch(r'cpu-probe(?:-\d+)?', name): raise ValueError('Exclusive CPU-probe name required')
    output = EV / name; output.mkdir(); scratch = output / 'scratch'; scratch.mkdir()
    job = OwnedJob(); child = None; record = {}; started = time.monotonic()
    try:
        with (output / 'stdout.txt').open('xb') as out, (output / 'stderr.txt').open('xb') as err:
            env = {**os.environ, **scratch_environment(scratch), 'WORKSPACE_DIR': str(ROOT)}
            child = subprocess.Popen([str(VENV / 'Scripts/python.exe'), '-B', '-I', str(Path(__file__).resolve()), '--metadata-worker', '--scratch', str(scratch)], stdin=subprocess.PIPE, stdout=out, stderr=err, env=env)
            job.attach(child); child.stdin.write(b'OWNED_JOB_ATTACHED\n'); child.stdin.flush(); child.stdin.close()
            child.wait(timeout=120); record.update(returncode=child.returncode, elapsed_seconds=time.monotonic()-started)
    except BaseException as exc: record.update(error=repr(exc))
    finally:
        record['job_cleanup'] = job.finish()
        if child is not None:
            from amd_sparse_decoder_loading import reap_owned_child
            reap_owned_child(child, record)
    rows = [json.loads(s.split('=', 1)[1]) for s in (output / 'stdout.txt').read_text(encoding='utf-8').splitlines() if s.startswith('SHAPE_METADATA=')]
    if len(rows) == 1: record['worker'] = rows[0]
    passed = metadata_passed(record)
    record['passed'] = bool(passed)
    (output / 'results.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k != 'worker'}))
    return 0 if passed else 1


def passive_audit(probe_name):
    """Hash/read existing data and code only. Never import an AI runtime."""
    import numpy as np
    import psutil
    import importlib.metadata
    import amd_blaslt_teardown as teardown
    teardown.verify_workspace()
    if not re.fullmatch(r'cpu-probe(?:-\d+)?', probe_name): raise ValueError('CPU probe name required')
    probe = json.loads((EV / probe_name / 'results.json').read_text(encoding='utf-8'))
    if not metadata_passed(probe): raise ValueError('CPU architecture probe did not pass')
    current_sources = {str(p): digest(p) for p in (SOURCE / 'trellis2').rglob('*.py')}
    prior_sources = json.loads((ROOT / 'docs/compatibility/amd-shape-slat-512-forward-evidence/source-contract.json').read_text(encoding='utf-8'))['sources']
    if current_sources != prior_sources: raise ValueError('Pinned AMD source changed')
    for name, expected in INPUTS.items():
        if digest(SAMPLE / name) != expected: raise ValueError('Published artifact changed: ' + name)
    x = read_npy(SAMPLE / 'shape-latent-denormalized.npy', INPUTS['shape-latent-denormalized.npy'])
    c = read_npy(SAMPLE / 'shape-latent-coordinates.npy', INPUTS['shape-latent-coordinates.npy'])
    arrays = validate_arrays(x, c)
    provenance = json.loads((SAMPLE / 'shape-latent-provenance.json').read_text(encoding='utf-8'))
    if (provenance['harness_commit'] != '133dcb41f866488ff017595547f8d8c77bc446f3'
            or provenance['sequence']['completed'] != 21 or len(provenance['sequence']['steps']) != 12
            or provenance['input']['coords']['sha256'] != arrays['coordinates_payload_sha256']):
        raise ValueError('Published input provenance mismatch')
    raw_path = SAMPLE / 'shape-latent-raw.npy'
    raw = read_npy(raw_path, '03fd8208614fcf20c37e40e59c9eb48972c3c7a3ee18a31bab1b0537e86ab2a1')
    canonical = raw * np.array(provenance['normalization']['std'], dtype=np.float32) + np.array(provenance['normalization']['mean'], dtype=np.float32)
    if not np.array_equal(x, canonical): raise ValueError('Saved canonical denormalization differs')
    header, config = checkpoint_header()
    root = Path('C:/Users/wattz/.triton/cache')
    metadata = []
    for path in sorted(root.rglob('*.json')):
        if path.name.startswith('__grp'): continue
        if path.stat().st_size > 1024**2: raise ValueError('Unbounded cache metadata')
        row = json.loads(path.read_text(encoding='utf-8'))
        validation = validate_cache_metadata(row)
        folder = base64.b32encode(bytes.fromhex(row['hash'])).decode().rstrip('=')
        if folder != path.parent.name: raise ValueError('Cache folder/hash mismatch')
        group = path.parent / ('__grp__' + path.name)
        if not group.is_file(): raise ValueError('Missing cache group manifest')
        children = json.loads(group.read_text(encoding='utf-8'))['child_paths']
        files = {}
        for name, p in cache_group_paths(path, children).items():
            if p.stat().st_size > 32 * 1024**2: raise ValueError('Unbounded cache product')
            if p.suffix == '.hsaco':
                with p.open('rb') as stream: prefix = stream.read(20)
                if prefix[:6] != b'\x7fELF\x02\x01' or int.from_bytes(prefix[18:20], 'little') != 224:
                    raise ValueError('Unexpected HSACO container/machine')
            files[name] = dict(bytes=p.stat().st_size, sha256=digest(p))
        metadata.append(dict(path=str(path), metadata=row, group_sha256=digest(group), files=files, **validation))
    tuning_path = Path('C:/Users/wattz/.flex_gemm/autotune_cache.json')
    tuning = json.loads(tuning_path.read_text(encoding='utf-8'))['AMD Radeon RX 9070 XT']
    suffix = 'flex_gemm.kernels.triton.spconv.sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk.'
    outer = tuning[suffix + 'sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk']
    known = {key: outer.get(key) for key in ('(2^11, 1024, 1024, 27)', '(2^11, 1024, 4096, 27)')}
    native_paths = [SITE / 'flex_gemm/kernels/cuda.cp312-win_amd64.pyd', SITE / 'triton/backends/amd/driver.c',
                    SITE / 'triton/backends/amd/compiler.py', SITE / 'triton/runtime/cache.py', SITE / 'triton/runtime/build.py', SITE / 'triton/runtime/jit.py']
    native_paths += list((SITE / 'flex_gemm').rglob('*.hip')) + list((SITE / 'flex_gemm').rglob('*.py'))
    helper = list(root.rglob('hip_utils.cp312-win_amd64.pyd'))
    native_paths += helper
    selected_packages = {}
    for dist in importlib.metadata.distributions(path=[str(SITE)]):
        name = dist.metadata['Name']
        if any(s in name.lower() for s in ('torch', 'triton', 'flex', 'rocm', 'voxel', 'cumesh')): selected_packages[name] = dist.version
    parameter_bytes = probe['worker']['precision']['parameter_bytes']
    result = dict(classification='UNKNOWN', gpu_attempted=False, weights_loaded=False, neural_forward_calls=0,
                  gpu_initialized=False, decoded_artifacts_created=False,
                  inputs={name: dict(path=str(SAMPLE/name), sha256=h, bytes=(SAMPLE/name).stat().st_size) for name,h in INPUTS.items()},
                  arrays=arrays, canonical_denormalization_equal=True, inherited_provenance=provenance,
                  checkpoint=dict(header=header, config=config, checkpoint_sha256=digest(PREFIX.with_suffix('.safetensors')), config_sha256=digest(PREFIX.with_suffix('.json'))),
                  cpu_probe=dict(path=str(EV/probe_name/'results.json'), sha256=digest(EV/probe_name/'results.json'), precision=probe['worker']['precision']),
                  source_sha256=current_sources, native_source_and_binary_sha256={str(p):digest(p) for p in native_paths},
                  installed_package_versions=selected_packages, cache_metadata=metadata,
                  tuning=dict(path=str(tuning_path), sha256=digest(tuning_path), known_first_stage_keys=known,
                              current_source_compatibility_verified=False, later_dynamic_keys_complete=False),
                  helper_cache=dict(files=[str(p) for p in helper], current_source_platform_library_key_verified=False, loaded=False),
                  growth=growth_budget(3743,parameter_bytes), available_host_ram_bytes=psutil.virtual_memory().available,
                  proposed_not_authorized_execution_limits=dict(host_ram_admission_bytes=10*1024**3, output_file_total_bytes=256*1024**2,
                      maximum_tokens_per_stage=1000000, cpu_seconds=180, gpu_seconds=240, overall_seconds=480,
                      caveat='Candidate limits only. Token/memory pre-call enforcement and cache-only native path have not been reviewed/implemented; GPU gate remains closed.'),
                  blocker='Existing guards prohibit Triton launch; cached-kernel source/environment and HIP helper identities are unverified. Cold custom autotuning and native helper build must not be implicitly admitted. Pre-call learned-growth/native-memory gates are absent.',
                  boundary='Pinned original parent forward returns raw F32 [dynamic_N,7] plus four F16 sparse subdivision tensors before child mesh conversion; preserve eval, low_vram, resolution512 and clear_spatial_cache.',
                  forbidden_imports_loaded=[name for name in ('torch','triton','flex_gemm','cumesh','o_voxel') if name in sys.modules])
    if result['forbidden_imports_loaded']: raise ValueError('Passive audit imported AI runtime')
    with (EV / 'preflight.json').open('x', encoding='utf-8') as stream: json.dump(result, stream, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k in ('classification','gpu_attempted','weights_loaded','arrays','growth','available_host_ram_bytes','blocker')}))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--metadata-worker', action='store_true'); parser.add_argument('--metadata-probe', action='store_true')
    parser.add_argument('--scratch', type=Path); parser.add_argument('--name', default='cpu-probe')
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if args.metadata_worker:
        if args.scratch is None: parser.error('Owned scratch required')
        raise SystemExit(metadata_worker(args.scratch))
    if args.metadata_probe: raise SystemExit(supervise_metadata(args.name))
    if args.audit: raise SystemExit(passive_audit(args.name))
    parser.error('CPU metadata probe only; GPU execution is not implemented')
