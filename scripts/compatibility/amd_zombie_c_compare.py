"""Prepare untouched Zombie C/original comparisons after a future PASS decode.

Default execution is a stdlib-only dry run. The explicit run reuses the original
owned CPU supervisor, never a generation, neural-model or WSL entrypoint.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
C_ROOT = ROOT / 'docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/zombie-c-60862722-v1'
DECODE = C_ROOT / 'decode'
OCCUPANCY = C_ROOT / 'occupancy'
SHAPE = C_ROOT / 'shape'
SOURCE = DECODE / 'run/guest/shape-decoded.safetensors'
BASE = ROOT.parent / 'artifacts/amd-zombie-c-comparison-60862722-v1'
ORIGINAL = ROOT.parent / 'artifacts/amd-zombie-cpu-mesh-20261025-v1'
REFERENCE = ROOT / 'docs/compatibility/amd-zombie-c-reference-contract.json'
IMAGE_SHA = '60862722c989772ad973870890601a6a95ed9a5cd5a9150a378e989db35ad437'
ORIGINAL_SHA = '5301f7dc3e22abf6bfd7c1253a4d6253935ad3054658e2eddbb6ac024e6fc12c'
ORIGINAL_RENDER_SHA = '5d756d6939bf1b16c8f9eb544006cb58dd45eeedf4628b57ce3e5c49b5087c88'
ORIGINAL_METADATA_SHA = 'a1f5cc97f00ed5a64b344d8230ed6b8b2b9c96af3059f6bc3445a70f93f3a91c'
CHECKPOINT_SHA = 'e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581'
BLENDER = Path('C:/Program Files/Blender Foundation/Blender 5.2/blender.exe')
PYTHON = Path('D:/AI/Trellis2AMD/venv/Scripts/python.exe')
PYTHON_SHA = '0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14'
MAX_ROWS = 1_842_968
COPY_LIMITS = {'run-contract.json': 65536, 'phases.jsonl': 8 * 1024**2,
               'results.json': 8 * 1024**2, 'provenance.json': 1024**2,
               'shape-decoded.safetensors': 256 * 1024**2}
TENSOR_NAMES = {'features', 'coordinates'} | {
    f'subdivision_{i}_{suffix}' for i in range(4) for suffix in ('coordinates', 'logits')}
FILES = ['scripts/compatibility/' + name for name in (
    'amd_zombie_c_compare.py', 'amd_zombie_c_compare_render.py', 'test_amd_zombie_c_compare.py',
    'amd_zombie_cpu_mesh.py', 'amd_zombie_ram_diagnostic.py', 'amd_forward_owned_job.py',
    'amd_generation_mesh.py', 'amd_flexible_dual_grid_cpu.py')]
FILES += ['experimental/extensions/trellis2-amd/inference_decoder/run_contract.py',
          'docs/compatibility/amd-zombie-c-reference-contract.json',
          'docs/compatibility/amd-zombie-c-comparison-plan.json']


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def plain(path):
    path = Path(path).absolute()
    need(path == path.resolve() and not str(path).startswith('\\\\'), 'Canonical local path required')
    for parent in (path, *path.parents):
        if parent.exists() or parent.is_symlink():
            need(not parent.is_symlink() and not getattr(parent, 'is_junction', lambda: False)()
                 and not getattr(parent.lstat(), 'st_file_attributes', 0) & 0x400, 'Redirected path denied')
    if path.is_file():
        need(path.stat().st_nlink == 1, 'Single-link file required')
    return path


def unique(pairs):
    value = {}
    for key, item in pairs:
        need(key not in value, 'Duplicate JSON key')
        value[key] = item
    return value


def read(path, limit=8 * 1024**2):
    path = plain(path)
    need(path.is_file() and 0 < path.stat().st_size <= limit, 'Bounded regular JSON required')
    def nonfinite(value):
        raise ValueError('Nonfinite JSON denied: ' + value)
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique, parse_constant=nonfinite)


def save(path, value):
    with plain(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def digest(path):
    with plain(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sha(value):
    need(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'Exact SHA256 required')
    return value


def snapshot(paths, *, allow_empty=False):
    rows = {}
    for path in paths:
        path = plain(path); before = path.stat()
        need(path.is_file() and (0 if allow_empty else 1) <= before.st_size <= 256 * 1024**2, 'Bounded immutable file required')
        rows[str(path)] = dict(bytes=before.st_size, mtime_ns=before.st_mtime_ns, sha256=digest(path))
        need(path.stat().st_mtime_ns == before.st_mtime_ns, 'Input changed during snapshot')
    return rows


def verify_snapshot(rows):
    for name, expected in rows.items():
        path = plain(name); info = path.stat()
        need((info.st_size, info.st_mtime_ns) == (expected['bytes'], expected['mtime_ns'])
             and digest(path) == expected['sha256'], 'Immutable input changed: ' + name)


def reserve(path):
    plain(path).mkdir(exist_ok=False)


def validate_receipts(final, host, guest, sidecar, commit):
    """Dynamic support is accepted only inside the original qualified envelope."""
    try:
        independent, output, provenance = final['independent_verification'], guest['output'], sidecar['provenance']
        rows = independent['rows']; source_sha = sha(independent['sha256'])
        contract_sha = sha(final['run_contract_sha256']); job_sha = sha(independent['source_job_sha256'])
        need(type(commit) is str and re.fullmatch('[0-9a-f]{40}', commit), 'Exact execution commit required')
        need(final['classification'] == 'PASS' and final['supervisor'] == 'QUALIFIED'
             and final['copy_error'] is None and final['harness_commit'] == commit
             and independent['ten_tensors_verified'] is True and independent['independent_of_worker'] is True
             and independent['model_or_GPU_imported'] is False, 'Complete independent decode PASS required')
        need(type(rows) is int and 0 < rows <= MAX_ROWS and output['rows'] == rows
             and output['sha256'] == source_sha and independent['run_contract_sha256'] == contract_sha,
             'Dynamic final support/hash exceeds the qualified bound')
        need(host['classification'] == 'QUALIFIED' and type(host['worker_returncode']) is int
             and host['worker_returncode'] == 0 and host['observer_healthy'] is True
             and type(host['minimum_host_ram_bytes']) is int and host['minimum_host_ram_bytes'] >= 12 * 1024**3
             and 0 <= host['elapsed_seconds'] < 480
             and host['windows_job_cleanup']['owned_tree_exit_verified'] is True
             and host['windows_job_cleanup']['forced_job_cleanup'] is False
             and host['distro_termination']['returncode'] == 0, 'Natural owned decode settlement required')
        need(guest['classification'] == 'QUALIFIED' and guest['native_import'] == guest['decoder_cpu'] == 'PASS'
             and guest['forward_calls'] == 1 and guest['convolutions_completed'] == 40
             and guest['memory_release'] == dict(allocated=0, reserved=0)
             and guest['run_contract_sha256'] == provenance['run_contract_sha256'] == contract_sha
             and guest['source_job_sha256'] == provenance['source_job_sha256'] == job_sha
             and sidecar['output'] == output and provenance['inputs'] == guest['saved_inputs']
             and provenance['checkpoint_sha256'] == CHECKPOINT_SHA, 'Complete guest/sidecar provenance required')
        hashes = independent['tensor_payload_sha256']
        need(type(hashes) is dict and set(hashes) == TENSOR_NAMES
             and hashes == output['tensor_payload_sha256'], 'Exactly ten tensor hashes required')
        for value in hashes.values():
            sha(value)
        initial = guest['saved_inputs']['tokens']
        need(type(initial) is int and 0 < initial <= 4430, 'Qualified initial support required')
        chain = output['subdivisions']; need(type(chain) is list and len(chain) == 4, 'Four learned subdivisions required')
        parent, children = initial, []
        for i, stage in enumerate(chain):
            child = stage['selected_children']
            need(type(stage['parents']) is int and stage['parents'] == parent
                 and stage['stage'] == i and stage['logits_dtype'] == 'torch.bfloat16'
                 and type(child) is int and 0 < child <= (MAX_ROWS if i == 3 else 1_000_000),
                 'Bounded dynamic learned subdivision linkage required')
            parent = child; children.append(child)
        need(children[-1] == rows, 'Final learned support differs')
        return dict(source_sha256=source_sha, rows=rows, learned_subdivisions=children,
                    initial_rows=initial, run_contract_sha256=contract_sha, source_job_sha256=job_sha,
                    tensor_payload_sha256=hashes)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('Incomplete C decode receipts') from exc


def flat_cpu(metadata):
    need(metadata['classification'] == 'PASS_CPU_PREVIEWS' and metadata['device'] == 'CPU'
         and metadata['engine'] == 'CYCLES' and metadata['threads'] == 4 and metadata['samples'] == 8
         and metadata['resolution'] == [768, 768] and metadata['up_axis'] == '+Z'
         and metadata['geometry_unchanged'] is True and metadata['imported_vertex_and_face_payloads_exact'] is True
         and metadata['smooth_shading'] is False and metadata['modifiers'] == [], 'Flat unchanged CPU previews required')


def validate_original_records(metadata, mesh):
    try:
        flat_cpu(metadata)
        need(metadata['source_PLY_sha256'] == mesh['artifacts']['mesh.ply']['sha256'] == ORIGINAL_SHA
             and mesh['PLY_readback_exact'] is True and mesh['topology_repair'] is False
             and mesh['material_decoded'] is False and mesh['zero_area_triangles'] == 0
             and metadata['vertices'] == mesh['vertices'] and metadata['triangles'] == mesh['triangles'],
             'Immutable original mesh binding differs')
        need(len(metadata['views']) == 8, 'Exactly eight original views required')
        for i, row in enumerate(metadata['views']):
            need(row['name'] == f'view-{i * 45:03d}.png' and row['azimuth_degrees'] == i * 45,
                 'Ordered distinct original camera views required')
            need(len(row['camera_location']) == 3 and all(type(v) in (int, float) and math.isfinite(v) for v in row['camera_location'])
                 and type(row['ortho_scale']) in (int, float) and math.isfinite(row['ortho_scale'])
                 and row['ortho_scale'] > 0, 'Finite original camera required')
            sha(row['sha256'])
        low, high = mesh['bounds']['min'], mesh['bounds']['max']
        need(len(low) == len(high) == 3 and all(math.isfinite(v) for v in low + high)
             and all(a < b for a, b in zip(low, high)), 'Finite original bounds required')
        return dict(target=[(a + b) / 2 for a, b in zip(low, high)],
                    extent=max(b - a for a, b in zip(low, high)))
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('Incomplete original camera/mesh records') from exc


def validate_render_metadata(original, candidate, source_sha):
    try:
        flat_cpu(candidate)
        need(candidate['source_PLY_sha256'] == source_sha and candidate['original_PLY_sha256'] == ORIGINAL_SHA
             and candidate['matched_original_cameras'] is True and len(candidate['views']) == len(original['views']) == 8,
             'New preview source or matched camera binding differs')
        for old, new in zip(original['views'], candidate['views']):
            need(all(old[key] == new[key] for key in ('name', 'azimuth_degrees', 'camera_location', 'ortho_scale')),
                 'Camera registration differs from the immutable original')
            sha(new['sha256'])
    except (KeyError, TypeError) as exc:
        raise ValueError('Incomplete matched preview receipt') from exc


def original_inputs():
    need(digest(ORIGINAL / 'mesh/mesh.ply') == ORIGINAL_SHA
         and digest(ORIGINAL / 'mesh/mesh.json') == ORIGINAL_METADATA_SHA
         and digest(ORIGINAL / 'previews/render-metadata.json') == ORIGINAL_RENDER_SHA, 'Pinned original files changed')
    metadata = read(ORIGINAL / 'previews/render-metadata.json'); mesh = read(ORIGINAL / 'mesh/mesh.json')
    camera = validate_original_records(metadata, mesh)
    files = [ORIGINAL / 'mesh/mesh.json', ORIGINAL / 'previews/render-metadata.json']
    need(set(mesh['artifacts']) == {'mesh.ply', 'vertices.npy', 'faces.npy'}, 'Original artifact closure differs')
    for name, row in mesh['artifacts'].items():
        path = plain(ORIGINAL / 'mesh' / name)
        need(digest(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Original mesh artifact changed')
        files.append(path)
    for row in metadata['views']:
        path = plain(ORIGINAL / 'previews' / row['name'])
        need(path.stat().st_size <= 4 * 1024**2 and digest(path) == row['sha256'], 'Original preview changed')
        files.append(path)
    return metadata, mesh, camera, snapshot(files)


def validate_lineage(contract, shape, admission, input_contract, input_sha, feature_metadata, commit):
    """Check saved current-image linkage without constructing a neural Runtime."""
    try:
        need(contract['schema'] == 'modly.amd.neural-run.v1' and 'final_support_profile' not in contract
             and contract['large_convolution_tiling'] is True and contract['maximum_final_rows'] == MAX_ROWS
             and contract['maximum_intermediate_rows'] == 1_000_000, 'Generic qualified C contract required')
        need(admission['context'] == dict(occupancy_evidence=str(OCCUPANCY), evidence=str(SHAPE), harness_commit=commit)
             and shape['classification'] == 'PASS' and shape['process']['harness_commit'] == commit,
             'Exact C shape/occupancy context required')
        lineage = shape['worker']['generation_inputs']
        need(lineage['schema'] == 'modly.amd.generation-shape-inputs.v1'
             and lineage['token_count'] == contract['initial_count']
             and lineage['input_contract_sha256'] == input_sha and lineage['occupancy_harness_commit'] == commit,
             'Fresh C shape conditioning lineage differs')
        need(input_contract['source_path'] == (OCCUPANCY / 'source-image.png').relative_to(ROOT).as_posix()
             and input_contract['source_sha256'] == IMAGE_SHA
             and feature_metadata['input_pixel_sha256'] == input_contract['pixel_sha256']
             and feature_metadata['input_png_sha256'] == input_contract['input_png_sha256']
             and feature_metadata['harness_commit'] == commit
             and feature_metadata['output_sha256'] == lineage['feature_payload_sha256'], 'C source image or feature provenance differs')
    except (KeyError, TypeError) as exc:
        raise ValueError('Incomplete C image lineage') from exc


def feature_header(path):
    path = plain(path); total = path.stat().st_size
    need(4214784 + 8 < total <= 4214784 + 16384, 'Bounded original conditioning artifact required')
    with path.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        need(0 < length <= 16384 and total == 8 + length + 4214784, 'Conditioning byte closure differs')
        header = json.loads(stream.read(length), object_pairs_hook=unique)
        need(set(header) == {'__metadata__', 'cond'} and header['cond'] == dict(dtype='F32', shape=[1, 1029, 1024], data_offsets=[0, 4214784]),
             'Original conditioning tensor layout required')
        need(hashlib.file_digest(stream, 'sha256').hexdigest() == header['__metadata__']['output_sha256'], 'Conditioning payload differs')
    return header['__metadata__']


def validate_reference(reference, input_contract, input_sha):
    try:
        need(reference['schema'] == 'modly.amd.zombie-c-reference.v1'
             and reference['source_sha256'] == input_contract['source_sha256'] == IMAGE_SHA
             and reference['source_path'] == input_contract['source_path']
             == (OCCUPANCY / 'source-image.png').relative_to(ROOT).as_posix()
             and reference['input_contract_path'] == (OCCUPANCY / 'input-contract.json').relative_to(ROOT).as_posix()
             and reference['input_contract_sha256'] == input_sha
             and reference['source_pixel_sha256'] == input_contract['pixel_sha256']
             and reference['encoded_worker_input_png_sha256'] == input_contract['input_png_sha256'],
             'Private C source/contract differs from published reference authority')
    except (KeyError, TypeError) as exc:
        raise ValueError('Complete published C reference authority required') from exc


def receipt_inputs(commit):
    final = read(DECODE / 'final-result.json'); host = read(DECODE / 'run/results.json')
    guest = read(SOURCE.parent / 'results.json'); sidecar = read(SOURCE.parent / 'provenance.json')
    binding = validate_receipts(final, host, guest, sidecar, commit)
    copied = final['allowlisted_copy']; need(set(copied) == set(COPY_LIMITS), 'Complete allowlisted decode copy required')
    files = [DECODE / 'final-result.json', DECODE / 'run/results.json', DECODE / 'run-contract.json']
    for name, limit in COPY_LIMITS.items():
        path = plain(SOURCE.parent / name); row = copied[name]
        need(type(row['bytes']) is int and 0 < row['bytes'] <= limit
             and path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], 'Copied C evidence differs: ' + name)
        files.append(path)
    need(copied[SOURCE.name]['sha256'] == binding['source_sha256'], 'Dynamic neural source hash differs')
    contract = read(DECODE / 'run-contract.json', 65536)
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    need(contract_sha == binding['run_contract_sha256'] and contract == read(SOURCE.parent / 'run-contract.json', 65536)
         and sidecar['run_contract'] == contract, 'Original host/guest C contract differs')
    shape_path = SHAPE / 'run/results.json'; shape = read(shape_path)
    need(digest(shape_path) == contract['source_job_sha256'] == binding['source_job_sha256'], 'C source shape job changed')
    need(digest(OCCUPANCY / 'source-image.png') == IMAGE_SHA, 'Raw C image changed')
    input_path = OCCUPANCY / 'input-contract.json'; input_sha = digest(input_path); input_contract = read(input_path, 65536)
    validate_reference(read(REFERENCE, 65536), input_contract, input_sha)
    feature = OCCUPANCY / 'dino/run/scratch/conditioning.safetensors'
    admission_path = SHAPE / 'pre-execution-admission.json'
    validate_lineage(contract, shape, read(admission_path), input_contract, input_sha, feature_header(feature), commit)
    need(digest(feature) == shape['worker']['generation_inputs']['feature_file_sha256'], 'C conditioning file changed')
    need(sidecar['provenance']['input_file_sha256'] == contract['input_sha256'], 'Decode input file binding differs')
    for name, leaf in (('coords.npy', 'shape-latent-coordinates.npy'), ('shape-latent-denormalized.npy', 'shape-latent-denormalized.npy')):
        path = plain(SHAPE / 'run/scratch' / leaf)
        guest_path = '/mnt/' + str(path)[0].lower() + '/' + str(path)[3:].replace('\\', '/')
        need(contract['input_paths'][name] == guest_path and digest(path) == contract['input_sha256'][name]
             and path.stat().st_size <= 2 * 1024**2, 'C decode input path/hash differs')
        files.append(path)
    binding.update(source_image_sha256=IMAGE_SHA, input_contract_sha256=input_sha,
                   feature_file_sha256=digest(feature), checkpoint_sha256=CHECKPOINT_SHA)
    files += [shape_path, admission_path, input_path, OCCUPANCY / 'source-image.png', feature, REFERENCE]
    return contract, binding, snapshot(files)


def authenticate(commit):
    import numpy as np
    import amd_generation_mesh as mesh
    document, binding, unchanged = receipt_inputs(commit)
    api = mesh._contract_api(); contract = api.load(DECODE / 'run-contract.json')
    need(contract.document() == document and contract.sha256 == binding['run_contract_sha256'], 'Qualified C contract differs')
    arrays, children = mesh.read_neural(SOURCE, binding['source_sha256'], contract=contract)
    try:
        hashes = {name: hashlib.sha256(memoryview(array)).hexdigest() for name, array in arrays.items()}
        need(set(arrays) == TENSOR_NAMES and hashes == binding['tensor_payload_sha256']
             and children == binding['learned_subdivisions'] and len(arrays['features']) == binding['rows'], 'Independent C ten-tensor readback differs')
        original = api.read_npy(SHAPE / 'run/scratch/shape-latent-coordinates.npy', contract.input_sha256['coords.npy'],
                                (contract.initial_count, 4), 'int32')
        need(np.array_equal(arrays['subdivision_0_coordinates'], original), 'Initial C coordinates/order differ')
        api.read_npy(SHAPE / 'run/scratch/shape-latent-denormalized.npy', contract.input_sha256['shape-latent-denormalized.npy'],
                     (contract.initial_count, 32), 'float32')
        with SOURCE.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(length), object_pairs_hook=unique)
        sidecar = read(SOURCE.parent / 'provenance.json')
        need(json.loads(header['__metadata__']['provenance'], object_pairs_hook=unique) == sidecar['provenance']
             and json.loads(header['__metadata__']['summary'], object_pairs_hook=unique)['subdivisions'] == sidecar['output']['subdivisions'],
             'Complete raw C header/sidecar metadata differs')
    finally:
        for array in arrays.values():
            array._mmap.close()
    verify_snapshot(unchanged)
    return contract, dict(binding, classification='PASS_TEN_TENSOR_PROVENANCE', model_or_GPU_imported=False)


def audit_policy(base):
    base = plain(base)
    denied = {'torch', 'triton', 'trellis2', 'flex_gemm', 'o_voxel', 'cupy', 'cumesh', 'nvdiffrast',
              'transformers', 'diffusers', 'timm', 'onnxruntime', 'comfy', 'comfyui', 'amd_generation_decode',
              'amd_generation_shape', 'amd_generation_occupancy'}
    def audit(event, args):
        if event == 'import' and str(args[0]).split('.')[0].lower() in denied:
            raise RuntimeError('Model/GPU/generation import denied')
        if event in ('subprocess.Popen', 'os.system', 'socket.connect', 'socket.bind', 'os.fork', 'os.posix_spawn'):
            raise RuntimeError('Child/network denied')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            writing = any(c in str(args[1]) for c in 'wax+') or args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            if writing:
                plain(os.fsdecode(args[0])).relative_to(base)
        if event == 'os.mkdir':
            plain(args[0]).relative_to(base)
        if event in ('os.remove', 'os.rename', 'os.rmdir', 'os.link', 'os.symlink', 'os.truncate'):
            raise RuntimeError('Destructive mutation denied')
    return audit


def validate_mesh(metadata, binding, mesh_base=None):
    need(metadata['input_sha256'] == binding['source_sha256'] and metadata['vertices'] == binding['rows']
         and metadata['learned_subdivisions'] == binding['learned_subdivisions'] and metadata['triangles'] > 0
         and metadata['PLY_readback_exact'] is True and metadata['zero_area_triangles'] == 0
         and metadata['topology_repair'] is False and metadata['material_decoded'] is False
         and metadata['native_extraction'] is False, 'Untouched raw C mesh binding differs')
    need(set(metadata['artifacts']) == {'mesh.ply', 'vertices.npy', 'faces.npy'}, 'Exact raw mesh artifact closure required')
    for name, row in metadata['artifacts'].items():
        path = plain((BASE if mesh_base is None else mesh_base) / 'mesh' / name)
        need(path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], 'Raw C mesh artifact changed')


def route_command(stage, command):
    entry = Path(__file__).resolve()
    if stage == 'render':
        expected = [str(BLENDER), '--background', '--factory-startup', '--disable-autoexec', '--threads', '4',
                    '--python-exit-code', '1', '--python', str(entry.with_name('amd_zombie_mesh_render.py'))]
        need(command == expected, 'Inherited CPU render command changed')
        return command[:-1] + [str(entry.with_name('amd_zombie_c_compare_render.py'))]
    need(stage in ('extract', 'review') and command == [sys._base_executable, '-B', '-I', '-S', str(entry), '--worker', stage],
         'Inherited isolated CPU worker command changed')
    return command


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def supervise(stage):
    # A fresh module namespace retains the exact original owner, caps, GO/EOF,
    # live floors, deadline, peak measurements and cleanup. Only routing changes.
    owner = load('_zombie_c_scoped_cpu_owner', Path(__file__).with_name('amd_zombie_cpu_mesh.py'))
    owner.BASE = BASE; owner.__file__ = str(Path(__file__).resolve())
    launch = subprocess.Popen
    def routed(command, **kwargs):
        return launch(route_command(stage, command), **kwargs)
    owner.subprocess = SimpleNamespace(Popen=routed, PIPE=subprocess.PIPE)
    return owner.supervise(stage)


def quality_review(original_mesh, candidate_mesh):
    criteria = [
        ('finger_separation', 'Inspect both hands in all eight full-size pairs; record distinct visible digits and persistent gaps rather than mesh vertex count.'),
        ('toe_separation', 'Inspect both feet in front, rear and oblique full-size pairs; record visible toe gaps and fused or missing forms.'),
        ('unwanted_holes', 'Inspect the whole surface and silhouette for unintended openings and tears; distinguish intended mouth, eyes, clothing edges and reference wounds.'),
        ('wound_preservation', 'Compare intended shoulder, chest, abdominal, thigh and shin wounds against each input reference: location, ragged rim, visible recess and lost or invented walls. Do not reward closure of intended wounds; mark unsupported depth unassessable.'),
        ('surface_defects', 'Inspect head, torso, limbs, palms and soles for spikes, floating fragments, folds, rough patches and loss of form.')]
    return dict(visual_status='UNASSESSED_REQUIRES_HUMAN_VISUAL_REVIEW', overall_quality_winner=None,
                semantic_digit_classifier_used=False, self_intersections_measured=False,
                geometry_counts_are_secondary=True,
                secondary_counts={label: {key: mesh.get(key) for key in ('vertices', 'triangles', 'zero_area_triangles', 'conversion')}
                                  for label, mesh in (('original', original_mesh), ('candidate_C', candidate_mesh))},
                criteria=[dict(criterion=name, instruction=text, assessment=None,
                               allowed_assessments=['C_better', 'original_better', 'similar', 'unassessable'],
                               evidence_required='Record view angles, visible regions and any occlusion/resolution limitation.') for name, text in criteria],
                limitation='Eight matched 768px views may leave digits hidden or too small. Mark unassessable and do not infer quality from counts.')


def review(mesh_base=None):
    from PIL import Image, ImageDraw
    mesh_base = BASE if mesh_base is None else mesh_base
    inputs = read(BASE / 'comparison-inputs.json'); verify_snapshot(inputs['immutable_original']); verify_snapshot(inputs['decoded_inputs'])
    original = read(ORIGINAL / 'previews/render-metadata.json'); original_mesh = read(ORIGINAL / 'mesh/mesh.json')
    candidate = read(BASE / 'previews/render-metadata.json'); candidate_mesh = read(mesh_base / 'mesh/mesh.json')
    validate_mesh(candidate_mesh, inputs['decode_binding'], mesh_base)
    validate_render_metadata(original, candidate, candidate_mesh['artifacts']['mesh.ply']['sha256'])
    need(read(mesh_base / 'extract-run/host-result.json')['classification'] == read(BASE / 'render-run/host-result.json')['classification'] == 'PASS',
         'Natural owned extraction/render PASS required before review')
    output = BASE / 'review'; reserve(output); reserve(output / 'original'); reserve(output / 'candidate-C')
    sheet = Image.new('RGB', (768, 8 * 412), '#30353d'); draw = ImageDraw.Draw(sheet)
    pairs = []; sections = []
    for i, (old, new) in enumerate(zip(original['views'], candidate['views'])):
        for label, folder, row in (('original', ORIGINAL / 'previews', old), ('candidate-C', BASE / 'previews', new)):
            source = plain(folder / row['name']); target = output / label / row['name']
            need(digest(source) == row['sha256'] and source.stat().st_size <= 4 * 1024**2, 'Paired source PNG changed')
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst, 65536)
            need(digest(target) == row['sha256'], 'Exact private PNG copy differs')
            with Image.open(target) as picture:
                need(picture.size == (768, 768) and picture.format == 'PNG', 'Native matched 768px PNG required')
                sheet.paste(picture.convert('RGB').resize((384, 384), Image.Resampling.LANCZOS), (0 if label == 'original' else 384, i * 412))
            draw.text((8 if label == 'original' else 392, i * 412 + 388), f"{label} | azimuth {i * 45} degrees", fill='white')
        pairs.append(dict(azimuth_degrees=i * 45, original='original/' + old['name'], candidate='candidate-C/' + new['name'],
                          original_sha256=old['sha256'], candidate_sha256=new['sha256']))
        sections.append(f'<h2>Azimuth {i * 45} degrees</h2><div class="pair"><figure><figcaption>Original</figcaption><a href="original/{old["name"]}"><img src="original/{old["name"]}" width="768" height="768"></a></figure>'
                        f'<figure><figcaption>Candidate C</figcaption><a href="candidate-C/{new["name"]}"><img src="candidate-C/{new["name"]}" width="768" height="768"></a></figure></div>')
    sheet_path = output / 'paired-eight-view-contact-sheet.png'
    with sheet_path.open('xb') as stream:
        sheet.save(stream, format='PNG')
    quality = quality_review(original_mesh, candidate_mesh)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Original / Zombie C mesh comparison</title><style>body{background:#20242b;color:#fff;font:16px system-ui;margin:24px}.pair{display:flex;gap:16px;overflow:auto}figure{margin:0}img{max-width:none}li{margin:10px 0}a{color:#8ecbff}</style><h1>Original / candidate C</h1><p><strong>UNASSESSED: human visual review required.</strong> Cameras, scale, lighting and flat shading match the original. Click an image for its unchanged 768px PNG; use browser zoom to inspect visible digits and surfaces.</p><ul>'
    page += ''.join('<li>' + row['instruction'] + '</li>' for row in quality['criteria'])
    page += '</ul><p>' + quality['limitation'] + '</p>' + ''.join(sections) + '</html>'
    with (output / 'paired-review.html').open('x', encoding='utf-8') as stream:
        stream.write(page)
    quality.update(classification='PASS_PAIRED_EVIDENCE_VISUAL_UNASSESSED', pairs=pairs,
                   original_PLY_sha256=ORIGINAL_SHA, candidate_PLY_sha256=candidate['source_PLY_sha256'],
                   original_files_unchanged=True, matched_camera_locations_and_ortho_scales=True,
                   native_PNG_copies_exact=True, thumbnail_resampling_only=True,
                   contact_sheet_sha256=digest(sheet_path), paired_review_sha256=digest(output / 'paired-review.html'))
    verify_snapshot(inputs['immutable_original']); verify_snapshot(inputs['decoded_inputs'])
    save(output / 'comparison-review.json', quality)


def worker(stage):
    if sys.stdin.buffer.read(4) != b'GO\n':
        raise RuntimeError('Exact owned GO plus EOF required before CPU runtime imports')
    sys.addaudithook(audit_policy(BASE))
    sys.path[:0] = [str(ROOT / 'scripts/compatibility'), 'D:/AI/Trellis2AMD/venv/Lib/site-packages']
    inputs = read(BASE / 'comparison-inputs.json'); verify_snapshot(inputs['immutable_original']); verify_snapshot(inputs['decoded_inputs'])
    if stage == 'extract':
        contract, binding = authenticate(inputs['harness_commit'])
        need(all(binding[key] == value for key, value in inputs['decode_binding'].items()), 'C input binding changed after reservation')
        save(BASE / 'authentication-extract.json', binding)
        import amd_generation_mesh as mesh
        result = mesh.extract(SOURCE, binding['source_sha256'], BASE / 'mesh', contract=contract)
        validate_mesh(result, binding)
        save(BASE / 'raw-mesh-preservation.json', dict(source_sha256=binding['source_sha256'],
             PLY_sha256=result['artifacts']['mesh.ply']['sha256'], geometry_repaired=False,
             geometry_smoothed=False, remeshed=False, material_decoded=False, texture_decoded=False))
    elif stage == 'review':
        review()
    else:
        raise ValueError('Unknown CPU comparison worker')
    verify_snapshot(inputs['immutable_original']); verify_snapshot(inputs['decoded_inputs'])


def verify_publication(commit):
    need(type(commit) is str and re.fullmatch('[0-9a-f]{40}', commit), 'Exact published execution commit required')
    git = shutil.which('git.exe') or shutil.which('git'); need(git, 'Git executable required')
    def read_git(args):
        return subprocess.check_output([git, *args], cwd=ROOT, timeout=15)
    need(read_git(['rev-parse', 'HEAD']).decode().strip() == commit, 'Published checkout commit differs')
    for name in FILES:
        committed = read_git(['show', commit + ':' + name]).replace(b'\r\n', b'\n')
        need(committed == plain(ROOT / name).read_bytes().replace(b'\r\n', b'\n'), 'Published CPU comparison source differs: ' + name)


def run(commit):
    import ctypes
    need(os.name == 'nt' and not ctypes.windll.shell32.IsUserAnAdmin(), 'Normal non-elevated Windows user required')
    need(Path(sys.executable).resolve() == PYTHON.resolve() and digest(PYTHON) == PYTHON_SHA, 'Qualified Windows interpreter required')
    verify_publication(commit)
    need(not plain(BASE).exists(), 'Single private comparison destination already consumed; no retry')
    original, original_mesh, camera, unchanged = original_inputs()
    _, binding, decoded_inputs = receipt_inputs(commit)
    owner = load('_zombie_c_cpu_admission', Path(__file__).with_name('amd_zombie_cpu_mesh.py'))
    owner.BASE = BASE.parent
    need(owner.resource()['admitted'], 'Original CPU RAM/disk admission failed')
    need(plain(BLENDER).is_file(), 'Original qualified Blender executable missing')
    reserve(BASE)
    save(BASE / 'comparison-inputs.json', dict(harness_commit=commit, decode_binding=binding,
         immutable_original=unchanged, decoded_inputs=decoded_inputs, camera=camera,
         published_CPU_sources=snapshot(ROOT / name for name in FILES),
         planned_stages=['extract', 'render', 'review'], automatic_retries=0))
    final = dict(classification='FAIL', visual_status='UNASSESSED_REQUIRES_HUMAN_VISUAL_REVIEW',
                 harness_commit=commit, source_image_sha256=IMAGE_SHA, stages=[])
    try:
        for stage in ('extract', 'render', 'review'):
            inputs = read(BASE / 'comparison-inputs.json')
            verify_snapshot(unchanged); verify_snapshot(decoded_inputs); verify_snapshot(inputs['published_CPU_sources'])
            code = supervise(stage); final['stages'].append(dict(stage=stage, returncode=code))
            if code:
                raise RuntimeError('Owned CPU comparison stage failed; no retry: ' + stage)
        candidate_mesh = read(BASE / 'mesh/mesh.json'); validate_mesh(candidate_mesh, binding)
        candidate = read(BASE / 'previews/render-metadata.json')
        validate_render_metadata(original, candidate, candidate_mesh['artifacts']['mesh.ply']['sha256'])
        reviewed = read(BASE / 'review/comparison-review.json')
        need(reviewed['classification'] == 'PASS_PAIRED_EVIDENCE_VISUAL_UNASSESSED'
             and reviewed['visual_status'] == 'UNASSESSED_REQUIRES_HUMAN_VISUAL_REVIEW'
             and reviewed['overall_quality_winner'] is None and reviewed['semantic_digit_classifier_used'] is False
             and len(reviewed['criteria']) == 5 and all(row['assessment'] is None for row in reviewed['criteria'])
             and len(reviewed['pairs']) == 8 and reviewed['original_PLY_sha256'] == ORIGINAL_SHA
             and reviewed['candidate_PLY_sha256'] == candidate['source_PLY_sha256'], 'Complete unassessed paired review required')
        for old, new, pair in zip(original['views'], candidate['views'], reviewed['pairs']):
            need(pair['azimuth_degrees'] == old['azimuth_degrees']
                 and pair['original'] == 'original/' + old['name'] and pair['candidate'] == 'candidate-C/' + new['name']
                 and pair['original_sha256'] == digest(BASE / 'review' / pair['original']) == old['sha256']
                 and pair['candidate_sha256'] == digest(BASE / 'review' / pair['candidate']) == new['sha256'], 'Paired native image copy changed')
        need(digest(BASE / 'review/paired-eight-view-contact-sheet.png') == reviewed['contact_sheet_sha256']
             and digest(BASE / 'review/paired-review.html') == reviewed['paired_review_sha256'], 'Paired review artifacts missing or changed')
        verify_snapshot(unchanged); verify_snapshot(decoded_inputs)
        final.update(classification='PASS_PAIRED_EVIDENCE_VISUAL_UNASSESSED', original_files_unchanged=True,
                     review=str(BASE / 'review/paired-review.html'), contact_sheet=str(BASE / 'review/paired-eight-view-contact-sheet.png'))
    except Exception as exc:
        final['error'] = type(exc).__name__ + ': ' + str(exc)
    save(BASE / 'comparison-final.json', final)
    print(json.dumps(final), flush=True)
    return 0 if final['classification'] == 'PASS_PAIRED_EVIDENCE_VISUAL_UNASSESSED' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true'); parser.add_argument('--harness-commit')
    parser.add_argument('--worker', choices=('extract', 'review'))
    args = parser.parse_args(argv)
    if args.worker:
        if args.run or args.harness_commit:
            parser.error('Owned workers consume only the private reserved input receipt')
        worker(args.worker); return 0
    if not args.run:
        if args.harness_commit:
            parser.error('--harness-commit requires --run')
        print('DRY_RUN: C comparison prepared; no model, WSL, extraction, rendering or output reservation')
        return 0
    if not args.harness_commit:
        parser.error('--run requires --harness-commit')
    return run(args.harness_commit)


if __name__ == '__main__':
    raise SystemExit(main())
