"""One exclusive CPU render/review recovery of the already extracted Zombie C.

Default is a stdlib-only dry run. No generation or extraction stage is exposed.
"""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_zombie_c_recovery_policy', HERE / 'amd_zombie_c_compare.py')
p = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(p)
PRIOR = p.BASE
BASE = p.ROOT.parent / 'artifacts/amd-zombie-c-render-recovery-60862722-v1'
SOURCE_COMMIT = '4ab4616c4019705d5f53802b053156ef65cae150'
CONTRACT = p.ROOT / 'docs/compatibility/amd-zombie-c-render-recovery-contract.json'
STAGES = ('authenticate', 'render', 'review')
FILES = p.FILES + ['scripts/compatibility/' + name for name in (
    'amd_zombie_c_render_recovery.py', 'Invoke-ZombieCRenderRecovery.ps1', 'test_amd_zombie_c_render_recovery.py')]
FILES += [CONTRACT.relative_to(p.ROOT).as_posix()]


def natural_pass(row, stage):
    p.need(row['classification'] == 'PASS' and row['stage'] == stage and row['worker_returncode'] == 0
           and row['worker_exit_verified'] is True and row['owned_job_attached'] is True
           and row['process_cap_bytes'] == 6 * 1024**3 and row['deadline_seconds'] == 180
           and row['initial_resource']['admitted'] is True
           and row['initial_resource']['required_RAM_bytes'] == 12 * 1024**3
           and row['initial_resource']['required_disk_bytes'] == 2 * 1024**3
           and all(s['admitted'] is True for s in row['samples'])
           and row['job_cleanup']['owned_tree_exit_verified'] is True
           and row['job_cleanup']['forced_job_cleanup'] is False
           and row['cleanup_errors'] == [] and not row.get('error') and not row.get('measurement_error'),
           'Natural owned CPU PASS and unchanged limits required: ' + stage)


def prior_inputs():
    """Read-only receipt/hash closure. The failed comparison stays a failure."""
    authority = p.read(CONTRACT, 65536)
    p.need(authority['source_commit'] == SOURCE_COMMIT and authority['schema'] == 'modly.amd.zombie-c-render-recovery.v1',
           'Original recovery authority differs')
    files = []
    for name, row in authority['previous_files'].items():
        path = p.plain(PRIOR / name)
        p.need(path.is_relative_to(PRIOR) and path.stat().st_size == row['bytes'] and p.digest(path) == row['sha256'],
               'Pinned previous evidence differs: ' + name)
        files.append(path)
    p.need(set(name for name in authority['previous_files'] if name.startswith('mesh/')) ==
           {'mesh/mesh.json', 'mesh/mesh.ply', 'mesh/vertices.npy', 'mesh/faces.npy'}, 'Complete saved mesh required')
    old = p.read(PRIOR / 'comparison-inputs.json')
    p.need(old['harness_commit'] == SOURCE_COMMIT and old['planned_stages'] == ['extract', 'render', 'review']
           and old['automatic_retries'] == 0, 'Original comparison run contract differs')
    p.verify_snapshot(old['immutable_original']); p.verify_snapshot(old['decoded_inputs'])
    _, binding, decoded = p.receipt_inputs(SOURCE_COMMIT)
    original, original_mesh, camera, unchanged = p.original_inputs()
    p.need(binding == old['decode_binding'] and decoded == old['decoded_inputs']
           and unchanged == old['immutable_original'] and camera == old['camera'], 'Saved C/source/original provenance differs')
    authenticated = p.read(PRIOR / 'authentication-extract.json')
    p.need(authenticated == dict(binding, classification='PASS_TEN_TENSOR_PROVENANCE', model_or_GPU_imported=False),
           'Original extraction authentication differs')
    metadata = p.read(PRIOR / 'mesh/mesh.json'); p.validate_mesh(metadata, binding, PRIOR)
    p.need(metadata['vertices'] == 325074 and metadata['triangles'] == 665232, 'Saved C mesh counts differ')
    natural_pass(p.read(PRIOR / 'extract-run/host-result.json'), 'extract')
    preserved = p.read(PRIOR / 'raw-mesh-preservation.json')
    p.need(preserved == dict(source_sha256=binding['source_sha256'], PLY_sha256=metadata['artifacts']['mesh.ply']['sha256'],
           geometry_repaired=False, geometry_smoothed=False, remeshed=False, material_decoded=False, texture_decoded=False),
           'Raw geometry preservation receipt differs')
    failed = p.read(PRIOR / 'comparison-final.json'); render = p.read(PRIOR / 'render-run/host-result.json')
    p.need(failed['classification'] == 'FAIL' and failed['harness_commit'] == SOURCE_COMMIT
           and failed['stages'] == [dict(stage='extract', returncode=0), dict(stage='render', returncode=1)]
           and render['classification'] == 'FAIL' and render['worker_returncode'] == 1
           and render['job_cleanup']['owned_tree_exit_verified'] is True
           and render['job_cleanup']['forced_job_cleanup'] is False and render['cleanup_errors'] == [],
           'Expected settled render-only failure required')
    # Freeze all prior files, including stdout/stderr, not just the mesh subset.
    previous = p.snapshot((path for path in PRIOR.rglob('*') if path.is_file()), allow_empty=True)
    return dict(decode_binding=binding, immutable_original=unchanged, decoded_inputs=decoded,
                camera=camera, preserved_previous=previous)


def verify_publication(commit):
    old = p.FILES
    try:
        p.FILES = FILES
        p.verify_publication(commit)
    finally:
        p.FILES = old


def environment():
    p.need(os.name == 'nt' and not ctypes.windll.shell32.IsUserAnAdmin(), 'Normal non-elevated Windows user required')
    p.need(Path(sys.executable).resolve() == p.PYTHON.resolve() and p.digest(p.PYTHON) == p.PYTHON_SHA,
           'Qualified Windows interpreter required')
    p.need(p.digest(p.BLENDER) == p.read(CONTRACT, 65536)['blender_executable_sha256'], 'Qualified Blender identity differs')


def verify_worker_inputs():
    inputs = p.read(BASE / 'comparison-inputs.json')
    p.need(inputs['source_commit'] == SOURCE_COMMIT and inputs['planned_stages'] == list(STAGES)
           and inputs['automatic_retries'] == 0, 'Exclusive render-only recovery receipt required')
    for key in ('immutable_original', 'decoded_inputs', 'preserved_previous', 'published_CPU_sources'):
        p.verify_snapshot(inputs[key])
    return inputs


def authenticate_mesh():
    """Verify tensors and read back saved PLY/NPY; never invoke a converter."""
    import numpy as np
    import amd_flexible_dual_grid_cpu as reader
    inputs = verify_worker_inputs()
    _, binding = p.authenticate(SOURCE_COMMIT)
    p.need(binding == dict(inputs['decode_binding'], classification='PASS_TEN_TENSOR_PROVENANCE', model_or_GPU_imported=False),
           'Independent ten-tensor authentication differs')
    metadata = p.read(PRIOR / 'mesh/mesh.json'); p.validate_mesh(metadata, inputs['decode_binding'], PRIOR)
    v = np.load(PRIOR / 'mesh/vertices.npy', mmap_mode='r', allow_pickle=False)
    f = np.load(PRIOR / 'mesh/faces.npy', mmap_mode='r', allow_pickle=False)
    pv, pf = reader.read_ply(PRIOR / 'mesh/mesh.ply')
    try:
        p.need(v.dtype == np.dtype('<f4') and f.dtype == np.dtype('<i4')
               and v.shape == (325074, 3) and f.shape == (665232, 3)
               and np.isfinite(v).all() and f.min() >= 0 and f.max() < len(v), 'Finite bounded saved arrays required')
        p.need(v.shape == pv.shape and f.shape == pf.shape and v.tobytes() == pv.tobytes() and f.tobytes() == pf.tobytes(),
               'Bit-exact saved PLY/NPY geometry/order required')
        p.need(dict(min=v.min(axis=0).tolist(), max=v.max(axis=0).tolist()) == metadata['bounds'], 'Saved C bounds differ')
        zero = 0
        for start in range(0, len(f), 32768):
            tri = v[f[start:start + 32768]].astype('f8')
            zero += int(np.count_nonzero(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) == 0))
        p.need(zero == metadata['zero_area_triangles'] == 0, 'Saved mesh zero-area count differs')
    finally:
        for array in (v, f, pv, pf):
            array._mmap.close()
    verify_worker_inputs()
    p.save(BASE / 'authentication-recovery.json', dict(binding, classification='PASS_EXISTING_GEOMETRY_READBACK',
           PLY_sha256=metadata['artifacts']['mesh.ply']['sha256'], vertices=325074, triangles=665232,
           PLY_NPY_payloads_bit_exact=True, zero_area_triangles=zero, mesh_extracted=False, geometry_modified=False))


def route_command(stage, command):
    p.need(stage in STAGES, 'Recovery has no extraction or generation stage')
    if stage == 'render':
        expected = [str(p.BLENDER), '--background', '--factory-startup', '--disable-autoexec', '--threads', '4',
                    '--python-exit-code', '1', '--python', str(HERE / 'amd_zombie_mesh_render.py')]
        p.need(command == expected, 'Original CPU render command changed')
        return command[:-1] + [str(HERE / 'amd_zombie_c_compare_render.py'), '--', '--render-recovery']
    p.need(command == [sys._base_executable, '-B', '-I', '-S', str(Path(__file__).resolve()), '--worker', stage],
           'Original isolated CPU worker command changed')
    return command


def supervise(stage):
    owner = p.load('_zombie_c_recovery_owner', HERE / 'amd_zombie_cpu_mesh.py')
    owner.BASE = BASE; owner.__file__ = str(Path(__file__).resolve())
    launch = subprocess.Popen
    def routed(command, **kwargs):
        return launch(route_command(stage, command), **kwargs)
    owner.subprocess = SimpleNamespace(Popen=routed, PIPE=subprocess.PIPE)
    return owner.supervise(stage)


def render_context(policy):
    inputs = verify_worker_inputs()
    natural_pass(p.read(BASE / 'authenticate-run/host-result.json'), 'authenticate')
    p.need(p.read(BASE / 'authentication-recovery.json')['classification'] == 'PASS_EXISTING_GEOMETRY_READBACK',
           'Independent saved geometry authentication required')
    policy.BASE = BASE
    return PRIOR


def worker(stage):
    if sys.stdin.buffer.read(4) != b'GO\n':
        raise RuntimeError('Exact owned GO plus EOF required before CPU imports')
    p.need(stage in ('authenticate', 'review'), 'Unknown render recovery worker')
    sys.addaudithook(p.audit_policy(BASE))
    sys.path[:0] = [str(HERE), 'D:/AI/Trellis2AMD/venv/Lib/site-packages']
    verify_worker_inputs()
    if stage == 'authenticate':
        authenticate_mesh()
    else:
        natural_pass(p.read(BASE / 'authenticate-run/host-result.json'), 'authenticate')
        natural_pass(p.read(BASE / 'render-run/host-result.json'), 'render')
        p.BASE = BASE
        p.review(mesh_base=PRIOR)
    verify_worker_inputs()


def acceptance():
    inputs = verify_worker_inputs()
    for stage in STAGES:
        natural_pass(p.read(BASE / (stage + '-run/host-result.json')), stage)
    auth = p.read(BASE / 'authentication-recovery.json')
    p.need(auth['classification'] == 'PASS_EXISTING_GEOMETRY_READBACK' and auth['mesh_extracted'] is False
           and auth['geometry_modified'] is False and auth['PLY_NPY_payloads_bit_exact'] is True,
           'Independent immutable geometry PASS required')
    original = p.read(p.ORIGINAL / 'previews/render-metadata.json')
    candidate = p.read(BASE / 'previews/render-metadata.json')
    p.validate_render_metadata(original, candidate, auth['PLY_sha256'])
    reviewed = p.read(BASE / 'review/comparison-review.json')
    p.need(reviewed['classification'] == 'PASS_PAIRED_EVIDENCE_VISUAL_UNASSESSED'
           and reviewed['visual_status'] == 'UNASSESSED_REQUIRES_HUMAN_VISUAL_REVIEW'
           and reviewed['overall_quality_winner'] is None and len(reviewed['criteria']) == 5
           and all(row['assessment'] is None for row in reviewed['criteria'])
           and reviewed['candidate_PLY_sha256'] == auth['PLY_sha256'], 'Complete unassessed paired review required')
    for row, old, new in zip(reviewed['pairs'], original['views'], candidate['views'], strict=True):
        p.need(row['azimuth_degrees'] == old['azimuth_degrees'] == new['azimuth_degrees'], 'Paired view differs')
        for label, record in (('original', old), ('candidate-C', new)):
            p.need(p.digest(BASE / 'review' / label / record['name']) == record['sha256'], 'Private PNG copy differs')
    p.need(p.digest(BASE / 'review/paired-review.html') == reviewed['paired_review_sha256']
           and p.digest(BASE / 'review/paired-eight-view-contact-sheet.png') == reviewed['contact_sheet_sha256'],
           'Review page/contact sheet differs')
    return reviewed


def run(commit):
    environment(); verify_publication(commit)
    p.need(not p.plain(BASE).exists(), 'Recovery destination already consumed; stop for review, no retry')
    prior = prior_inputs()
    owner = p.load('_zombie_c_recovery_admission', HERE / 'amd_zombie_cpu_mesh.py'); owner.BASE = BASE.parent
    p.need(owner.resource()['admitted'], 'Original CPU RAM/disk admission failed')
    p.reserve(BASE)
    p.save(BASE / 'comparison-inputs.json', dict(prior, harness_commit=commit, source_commit=SOURCE_COMMIT,
           published_CPU_sources=p.snapshot(p.ROOT / name for name in FILES), planned_stages=list(STAGES), automatic_retries=0))
    final = dict(classification='FAIL', harness_commit=commit, source_commit=SOURCE_COMMIT, stages=[],
                 neural_execution=False, mesh_extracted=False, geometry_modified=False)
    try:
        for stage in STAGES:
            verify_worker_inputs()
            code = supervise(stage); final['stages'].append(dict(stage=stage, returncode=code))
            p.need(code == 0, 'CPU recovery stage failed; stop without retry: ' + stage)
        acceptance()
        final.update(classification='PASS_PAIRED_EVIDENCE_VISUAL_UNASSESSED',
                     visual_status='UNASSESSED_REQUIRES_HUMAN_VISUAL_REVIEW', original_evidence_unchanged=True,
                     review_page=str(BASE / 'review/paired-review.html'))
    except Exception as exc:
        final['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        try:
            verify_worker_inputs()
        except Exception as exc:
            final.update(classification='FAIL', preservation_error=type(exc).__name__ + ': ' + str(exc))
    p.save(BASE / 'recovery-final.json', final)
    print(json.dumps(final), flush=True)
    return 0 if final['classification'] == 'PASS_PAIRED_EVIDENCE_VISUAL_UNASSESSED' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true'); parser.add_argument('--harness-commit')
    parser.add_argument('--worker', choices=('authenticate', 'review'))
    args = parser.parse_args(argv)
    if args.worker:
        if args.run or args.harness_commit:
            parser.error('Owned workers consume only reserved recovery inputs')
        worker(args.worker); return 0
    if not args.run:
        if args.harness_commit: parser.error('--harness-commit requires --run')
        print('DRY_RUN: render-only recovery; no process, runtime, output, generation or extraction')
        return 0
    if not args.harness_commit or not re.fullmatch('[0-9a-f]{40}', args.harness_commit):
        parser.error('--run requires exact --harness-commit')
    return run(args.harness_commit)


if __name__ == '__main__':
    raise SystemExit(main())
