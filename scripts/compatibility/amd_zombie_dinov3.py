"""One exact-image standalone DINO proof; no production or downstream dispatch."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amd_blaslt_teardown as teardown
import amd_dinov3_conditioning as dino
import amd_feature_checkpoint as features

ROOT = dino.local.ROOT
ARCHIVE = ROOT.parent / 'artifacts/amd-preview-dinov3-7d11b36/cpu-input'
SOURCE = ARCHIVE / 'source-image.png'
SOURCE_SHA = 'db1fbb6bd27717f4b6a0298d64ff1ddff17e80027d782bf5dbfbc959d70fa06e'
PIXEL_SHA = 'e958193cad52cb3b15efd69410c3727853df61fd35310468d1b4540284ea2fb5'
EVIDENCE = ROOT / 'artifacts/amd-zombie-dinov3-b8afdd5/dino'
PRIOR = {'input-contract.json': 'de2ed14e6d7ed9d37eb675c4990c3f91e9f311c0804f40f18761edc1451217f4',
         'cpu-result.json': '0af4bcf4f3592919272ad32277b6d0125ea08b6b30853a8340893a5869025cee'}
STATIC = ['scripts/compatibility/' + name for name in (
    'amd_zombie_dinov3.py', 'test_amd_zombie_dinov3.py', 'amd_feature_checkpoint.py',
    'amd_object_occupancy.py', 'amd_sparse_sampling.py', 'amd_sparse_occupancy.py',
    'amd_sparse_flow_forward.py', 'amd_sparse_flow_local.py', 'amd_sparse_flow_contract.py',
    'amd_sparse_sampling_contract.py', 'amd_sparse_sampling_control.py',
    'amd_sparse_occupancy_contract.py', 'amd_sparse_occupancy_data.py', 'amd_sparse_latent_checkpoint.py')]


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def plain_path(value, *, existing=False):
    raw = str(value).replace('\\', '/')
    if raw.startswith('//') or '..' in raw.split('/'):
        raise ValueError('Local canonical paths without traversal required')
    path = Path(value).absolute()
    for ancestor in (path, *path.parents):
        if ancestor.exists() or ancestor.is_symlink():
            info = ancestor.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError('Reparse points and symlinks prohibited')
            if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                raise ValueError('Hardlinked input/output prohibited')
    resolved = path.resolve(strict=existing)
    if path != resolved:
        raise ValueError('Path alias prohibited')
    return resolved


def bounded(path, limit):
    with plain_path(path, existing=True).open('rb') as stream:
        blob = stream.read(limit + 1)
    if len(blob) > limit:
        raise ValueError('Input exceeds explicit bound')
    return blob


def validate_image(blob, prior_cpu_result, prior_input_contract):
    if len(blob) != 642414 or sha(blob) != SOURCE_SHA:
        raise ValueError('Exact validated zombie PNG required')
    from alpha_input import decode_alpha_image
    image, alpha = decode_alpha_image(blob)
    row = prior_cpu_result.get('contract', {})
    if (image.mode != 'RGBA' or image.size != (1216, 1294) or sha(image.tobytes()) != PIXEL_SHA
            or prior_cpu_result.get('status') != 'PASS_CPU_IMAGE_REVALIDATION'
            or prior_cpu_result.get('image_sha256') != SOURCE_SHA
            or prior_input_contract.get('source_sha256') != SOURCE_SHA
            or prior_input_contract.get('qualification') != 'CPU_ADMISSION_ONLY'
            or prior_input_contract.get('original_cpu_input_contract') != row
            or prior_cpu_result.get('canonical_parameters') != dict(image_mode='alpha-only', resolution=512,
                stage='conditioning-only', include_neg_cond=False)
            or any(row.get(k) != v for k, v in dict(input_sha256=SOURCE_SHA, source_bytes=642414,
                source_size=[1216, 1294], source_mode='RGBA', pixel_sha256=PIXEL_SHA,
                foreground_bbox=alpha['foreground_bbox'], resized_foreground_bbox=alpha['resized_foreground_bbox'],
                preprocessing=dict(image_mode='alpha-only', resolution=512),
                preprocessing_complete=False, dino_executed=False).items())):
        raise ValueError('Archived CPU eligibility or decoded pixel lineage differs')
    # Reuse the original CPU-only extraction of the pinned preprocessing method.
    # Never call the object/generation entrypoints, profile or three-stage loop.
    from amd_object_occupancy import preprocess_cpu
    crop = preprocess_cpu(image)
    return image, dict(source_sha256=SOURCE_SHA, source_bytes=len(blob), pixel_sha256=PIXEL_SHA,
        source_size=list(image.size), source_mode=image.mode, input_png_sha256=sha(blob),
        crop_size=list(crop.size), crop_mode=crop.mode, crop_pixel_sha256=sha(crop.tobytes()),
        alpha=alpha, weights_loaded=False, gpu_initialized=False)


def selected_image():
    records = {}
    for name, expected in PRIOR.items():
        blob = bounded(ARCHIVE / name, 65536)
        if sha(blob) != expected:
            raise ValueError('Archived CPU receipt changed: ' + name)
        records[name] = json.loads(blob, object_pairs_hook=features.unique)
    return validate_image(bounded(SOURCE, 2 * 1024**2), records['cpu-result.json'], records['input-contract.json'])


def image_factory(path):
    path = plain_path(path)
    if path != plain_path(EVIDENCE / 'run/scratch/input.png'):
        raise ValueError('Only exact fresh owned input.png allowed')
    image, row = selected_image()
    blob = bounded(SOURCE, 2 * 1024**2)
    if sha(blob) != row['input_png_sha256']:
        raise ValueError('Image changed before exclusive copy')
    with path.open('xb') as stream:
        stream.write(blob); stream.flush(); os.fsync(stream.fileno())
    if sha(bounded(path, 2 * 1024**2)) != SOURCE_SHA:
        raise ValueError('Scratch input byte readback failed')
    return image


def profile():
    result = teardown.profile()
    result.evidence = EVIDENCE
    result.worker_script = Path(__file__).resolve()
    result.committed_files = list(dict.fromkeys([*result.committed_files, *STATIC]))
    return result


def expected_provenance(commit):
    pins = dino.local.verify_sources()
    return dict(component='DINOv3ViTModel',
        checkpoint_sha256=json.loads(dino.local.PINS.read_text())['checkpoint_sha256'],
        config_sha256=dino.local.digest(dino.local.MODEL / 'config.json'),
        input_pixel_sha256=PIXEL_SHA, input_png_sha256=SOURCE_SHA, harness_commit=commit,
        extractor_source_sha256=pins[str(dino.local.SOURCE / 'trellis2/modules/image_feature_extractor.py')],
        pipeline_source_sha256=pins[str(dino.local.SOURCE / 'trellis2/pipelines/trellis2_image_to_3d.py')],
        runtime='Python 3.12.10; PyTorch 2.13.0+rocm10.0.0; HIP 7.15.26333')


def validate_saved(result, commit):
    worker = result['worker']; process = result['process']
    command = [str(dino.local.VENV / 'Scripts/python.exe'), '-B', '-I',
        str(Path(__file__).resolve()), '--worker', '--output', str(EVIDENCE / 'run/scratch')]
    if (teardown.classify(worker, True, True, True) != 'PASS' or result.get('classification') != 'PASS'
            or result.get('observer_healthy') is not True or result.get('checkpoint_unchanged') is not True
            or process.get('harness_commit') != commit or process.get('returncode') != 0
            or process.get('command') != command
            or process.get('stage_timeouts_seconds') != dict(cpu=120, forward_release=180, overall=360)
            or process.get('timed_out') is not False or process.get('gpu_authorized') is not True
            or process.get('worker_exit_verified') is not True or process.get('forced_cleanup', False)
            or process.get('cleanup_unverified', False) or process.get('error') or process.get('job_cleanup_error')
            or process.get('job_cleanup', {}).get('owned_tree_exit_verified') is not True
            or process.get('job_cleanup', {}).get('active_before_cleanup') != 0
            or process.get('job_cleanup', {}).get('active_after_cleanup') != 0
            or process.get('job_cleanup', {}).get('forced_job_cleanup') is not False
            or any(process.get(k) is not True for k in ('job_attached_before_preparation', 'output_readers_completed',
                'cpu_observer_stopped', 'diagnostic_journal_stopped', 'windows_periodic_stopped', 'windows_journal_stopped'))
            or any(process.get(k) for k in ('cpu_observer_error', 'diagnostic_journal_error', 'windows_periodic_error', 'windows_journal_error'))):
        raise ValueError('Original host, neural, monitoring or natural settlement acceptance failed')
    _, image = selected_image()
    if any(worker['input'].get(k) != v for k, v in dict(mode='RGBA', size=[1216, 1294],
            pixel_sha256=PIXEL_SHA, png_sha256=SOURCE_SHA, crop_mode=image['crop_mode'],
            crop_size=image['crop_size'], resolution=512, alpha_only=True, rmbg_model=None).items()):
        raise ValueError('Runtime selected-image lineage differs')
    artifact = worker['feature_checkpoint']
    path = plain_path(EVIDENCE / 'run/scratch/conditioning.safetensors', existing=True)
    restored = features.verify(path, path.parent, expected_provenance(commit), worker['output']['sha256'])
    if restored != artifact:
        raise ValueError('Independent artifact/container/tensor readback differs')
    return restored


def save(path, value):
    with plain_path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def preflight():
    teardown.verify_workspace(); teardown.verify_binding_evidence()
    image, row = selected_image(); del image
    verified = dino.verify_preflight()
    import psutil
    versions = {name: importlib.metadata.version(name) for name in (
        'torch', 'torchvision', 'transformers', 'numpy', 'pillow', 'safetensors', 'psutil')}
    expected = dict(torch='2.13.0+rocm10.0.0', torchvision='0.28.0+rocm10.0.0',
        transformers='4.56.0', numpy='2.5.3', pillow='12.3.0', safetensors='0.8.0', psutil='7.2.2')
    available = psutil.virtual_memory().available
    needed = verified['tensor_data_bytes'] * 4 + 1024**3
    # The absent exclusive leaf is on the checkout's same C: volume. Do not
    # require its parent to exist (or reserve the attempt) just to query space.
    destination = plain_path(ROOT, existing=True)
    free = psutil.disk_usage(str(destination)).free
    executable = plain_path(sys.executable, existing=True)
    if (executable != dino.local.VENV / 'Scripts/python.exe' or sys.version.split()[0] != '3.12.10'
            or dino.local.digest(executable) != '0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14'
            or versions != expected or available < needed or free < 2 * 1024**3 or 'torch' in sys.modules):
        raise RuntimeError('CPU runtime, destination or original RAM admission failed')
    return dict(input=row, checkpoint_sha256=verified['checkpoint_sha256'], config_sha256=verified['config_sha256'],
        header_sha256=verified['header_sha256'], header_hash_domain='8-byte length prefix plus padded JSON header',
        tensor_count=verified['tensor_count'], parameter_bytes=verified['tensor_data_bytes'], dependencies=versions,
        available_ram_bytes=available, required_ram_bytes=needed, destination_free_bytes=free,
        extra_destination_floor_bytes=2 * 1024**3, original_initial_device_free_bytes=verified['tensor_data_bytes'] + 2 * 1024**3,
        deadlines_seconds=dict(cpu=120, gpu=180, overall=360), torch_imported=False, weights_loaded=False,
        scope='standalone conditioning only; no production authority or downstream execution')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--harness-commit')
    args = parser.parse_args()
    teardown.verify_workspace()
    if args.worker:
        if args.preflight or args.harness_commit or plain_path(args.output, existing=True) != EVIDENCE / 'run/scratch':
            raise ValueError('Exact single-stage worker destination required')
        dino.local.fixture = image_factory
        return dino.worker(args.output, allocation_evidence=EVIDENCE, workspace_teardown=True, save_features=True)
    row = preflight()
    if args.preflight:
        plain_path(EVIDENCE).mkdir(parents=True, exist_ok=False)
        save(EVIDENCE / 'cpu-admission.json', row)
        print(json.dumps(row, indent=2)); return 0
    if args.output != EVIDENCE / 'run' or not args.harness_commit:
        raise ValueError('Use exact standalone run destination and published commit')
    admitted = json.loads(bounded(EVIDENCE / 'cpu-admission.json', 65536))
    if any(admitted[k] != row[k] for k in ('input', 'checkpoint_sha256', 'config_sha256', 'header_sha256', 'dependencies')):
        raise ValueError('Prepared CPU identity changed before attempt')
    result = dino.supervisor.supervise(args.output, args.harness_commit, profile=profile())
    if result:
        return result
    restored = validate_saved(json.loads(bounded(EVIDENCE / 'run/results.json', 4 * 1024**2)), args.harness_commit)
    save(EVIDENCE / 'host-readback.json', restored)
    print(json.dumps(dict(standalone='PASS', artifact_sha256=restored['file_sha256'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
