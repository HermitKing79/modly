"""Synthetic CPU postprocessing contracts; never decode or render a real mesh."""
import builtins
import copy
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import ctypes
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import runpy
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name('amd_zombie_d_compare.py')
subject = None
if PATH.exists():
    spec = importlib.util.spec_from_file_location('zombie_d_compare', PATH)
    subject = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(subject)

COMMIT = '1' * 40
SOURCE_SHA = '2' * 64
CONTRACT_SHA = '3' * 64
JOB_SHA = '4' * 64
TENSORS = {'features': '5' * 64, 'coordinates': '6' * 64,
           **{f'subdivision_{i}_{suffix}': str(7 + i) * 64 if i < 3 else 'b' * 64
              for i in range(4) for suffix in ('coordinates', 'logits')}}


@contextmanager
def scratch_directory():
    parent = PATH.parents[3] / 'artifacts/amd-zombie-d-comparison-policy-tests'
    parent.mkdir(exist_ok=True)
    # Keep only new synthetic scratch. Never delete or reuse historical evidence.
    yield tempfile.mkdtemp(prefix='policy-', dir=parent)


def receipts():
    subdivisions = [dict(stage=i, parents=p, selected_children=c, logits_dtype='torch.bfloat16')
                    for i, (p, c) in enumerate(((2, 7), (7, 19), (19, 73), (73, 251)))]
    output = dict(rows=251, sha256=SOURCE_SHA, bytes=1234,
                  subdivisions=subdivisions, tensor_payload_sha256=TENSORS)
    inputs = dict(tokens=2, run_contract_sha256=CONTRACT_SHA, source_job_sha256=JOB_SHA)
    guest = dict(classification='QUALIFIED', native_import='PASS', decoder_cpu='PASS',
                 forward_calls=1, convolutions_completed=40, memory_release=dict(allocated=0, reserved=0),
                 run_contract_sha256=CONTRACT_SHA, source_job_sha256=JOB_SHA,
                 output=output, saved_inputs=inputs)
    final = dict(classification='PASS', supervisor='QUALIFIED', copy_error=None,
                 harness_commit=COMMIT, run_contract_sha256=CONTRACT_SHA,
                 independent_verification=dict(ten_tensors_verified=True, independent_of_worker=True,
                     model_or_GPU_imported=False, rows=251, sha256=SOURCE_SHA,
                     run_contract_sha256=CONTRACT_SHA, source_job_sha256=JOB_SHA,
                     tensor_payload_sha256=TENSORS))
    host = dict(classification='QUALIFIED', worker_returncode=0, observer_healthy=True,
                minimum_host_ram_bytes=12884901888, elapsed_seconds=479,
                windows_job_cleanup=dict(owned_tree_exit_verified=True, forced_job_cleanup=False),
                distro_termination=dict(returncode=0))
    sidecar = dict(output=copy.deepcopy(output), provenance=dict(inputs=copy.deepcopy(inputs),
                     checkpoint_sha256='e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581',
                     run_contract_sha256=CONTRACT_SHA, source_job_sha256=JOB_SHA))
    return final, host, guest, sidecar


def original_records():
    metadata = dict(classification='PASS_CPU_PREVIEWS', source_PLY_sha256='5301f7dc3e22abf6bfd7c1253a4d6253935ad3054658e2eddbb6ac024e6fc12c',
                    imported_vertex_and_face_payloads_exact=True, geometry_unchanged=True,
                    modifiers=[], smooth_shading=False, engine='CYCLES', device='CPU', samples=8,
                    threads=4, resolution=[768, 768], up_axis='+Z', blender_version='5.2.1 LTS',
                    vertices=326232, triangles=673604,
                    views=[dict(name=f'view-{i * 45:03d}.png', azimuth_degrees=i * 45,
                                camera_location=[float(i), -4., .3], ortho_scale=1.2, sha256='a' * 64)
                           for i in range(8)])
    mesh = dict(vertices=326232, triangles=673604, PLY_readback_exact=True,
                topology_repair=False, material_decoded=False, zero_area_triangles=0,
                bounds=dict(min=[-.5, -.1, -.5], max=[.5, .1, .5]),
                artifacts={'mesh.ply': dict(sha256=metadata['source_PLY_sha256'], bytes=12671864)})
    return metadata, mesh


def lineage_records():
    contract = dict(schema='modly.amd.neural-run.v1', large_convolution_tiling=True,
                    maximum_final_rows=1842968, maximum_intermediate_rows=1000000, initial_count=2)
    lineage = dict(schema='modly.amd.generation-shape-inputs.v1', token_count=2,
                   input_contract_sha256='c' * 64, occupancy_harness_commit=COMMIT,
                   feature_payload_sha256='d' * 64)
    shape = dict(classification='PASS', process=dict(harness_commit=COMMIT), worker=dict(generation_inputs=lineage))
    admission = dict(context=dict(occupancy_evidence=str(subject.OCCUPANCY), evidence=str(subject.SHAPE), harness_commit=COMMIT))
    source = dict(source_path=(subject.OCCUPANCY / 'source-image.png').relative_to(subject.ROOT).as_posix(),
                  source_sha256='9ac0364f409077cd3fdeb853a1baed3b851c56f4fdc539c9fde73e0e3ab79db9',
                  pixel_sha256='e' * 64, input_png_sha256='f' * 64)
    feature = dict(input_pixel_sha256='e' * 64, input_png_sha256='f' * 64,
                   output_sha256='d' * 64, harness_commit=COMMIT)
    return contract, shape, admission, source, 'c' * 64, feature, COMMIT


class Exists(unittest.TestCase):
    def test_bounded_C_comparison_utility_exists(self):
        self.assertIsNotNone(subject, 'C comparison utility is missing')


@unittest.skipIf(subject is None, 'Not implemented')
class ComparisonPolicy(unittest.TestCase):
    def test_dry_run_loads_no_runtime_and_creates_no_output(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'PIL', 'bpy', 'torch', 'amd_generation_mesh', 'amd_generation_decode'}:
                raise AssertionError('Dry run imported a runtime')
            return original(name, *args, **kwargs)
        with scratch_directory() as folder, patch.object(subject, 'BASE', Path(folder) / 'absent'), \
             patch.object(builtins, '__import__', side_effect=guarded), \
             patch.object(subject.subprocess, 'Popen', side_effect=AssertionError('Dry run spawned')):
            self.assertEqual(subject.main([]), 0)
            self.assertFalse(subject.BASE.exists())

    def test_missing_commit_rejects_before_runtime(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            subject.main(['--run'])
        self.assertEqual(caught.exception.code, 2)

    def test_dynamic_success_is_accepted_without_historical_row_counts(self):
        binding = subject.validate_receipts(*receipts(), COMMIT)
        self.assertEqual(binding['rows'], 251)
        self.assertEqual(binding['learned_subdivisions'], [7, 19, 73, 251])
        self.assertEqual(binding['source_sha256'], SOURCE_SHA)

    def test_current_image_lineage_rejects_old_conditioning_context_and_larger_profile(self):
        subject.validate_lineage(*lineage_records())
        for mutation in ('old-pixels', 'old-png', 'old-context', 'old-commit', 'contract-hash', 'larger-profile'):
            with self.subTest(mutation=mutation):
                contract, shape, admission, source, input_sha, feature, commit = copy.deepcopy(lineage_records())
                if mutation == 'old-pixels': feature['input_pixel_sha256'] = '0' * 64
                if mutation == 'old-png': feature['input_png_sha256'] = '0' * 64
                if mutation == 'old-context': admission['context']['occupancy_evidence'] += '-original'
                if mutation == 'old-commit': shape['worker']['generation_inputs']['occupancy_harness_commit'] = '0' * 40
                if mutation == 'contract-hash': shape['worker']['generation_inputs']['input_contract_sha256'] = '0' * 64
                if mutation == 'larger-profile': contract['final_support_profile'] = 'brick-1957891-v1'
                with self.assertRaises(ValueError):
                    subject.validate_lineage(contract, shape, admission, source, input_sha, feature, commit)

    def test_private_contract_must_match_published_source_authority(self):
        source = lineage_records()[3]
        reference = dict(schema='modly.amd.zombie-d-reference.v1', source_sha256=source['source_sha256'],
                         source_path=source['source_path'], source_pixel_sha256=source['pixel_sha256'],
                         input_contract_path=(subject.OCCUPANCY / 'input-contract.json').relative_to(subject.ROOT).as_posix(),
                         input_contract_sha256='c' * 64, encoded_worker_input_png_sha256=source['input_png_sha256'])
        self.assertIsNone(subject.validate_reference(reference, source, 'c' * 64))
        for mutation in ('private-hash', 'image', 'pixels', 'source-path'):
            changed = copy.deepcopy(reference)
            if mutation == 'private-hash': changed['input_contract_sha256'] = '0' * 64
            if mutation == 'image': changed['source_sha256'] = '0' * 64
            if mutation == 'pixels': changed['source_pixel_sha256'] = '0' * 64
            if mutation == 'source-path': changed['source_path'] = 'old/source-image.png'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                subject.validate_reference(changed, source, 'c' * 64)

    def test_incomplete_independent_or_unclean_decode_never_passes(self):
        mutations = [(0, ('classification',), 'FAIL'),
                     (0, ('independent_verification', 'ten_tensors_verified'), False),
                     (0, ('independent_verification', 'independent_of_worker'), False),
                     (0, ('independent_verification', 'model_or_GPU_imported'), True),
                     (0, ('harness_commit',), 'f' * 40),
                     (1, ('minimum_host_ram_bytes',), 12884901887),
                     (1, ('observer_healthy',), False),
                     (1, ('windows_job_cleanup', 'forced_job_cleanup'), True),
                     (1, ('elapsed_seconds',), 480),
                     (2, ('forward_calls',), 2),
                     (2, ('convolutions_completed',), 39),
                     (2, ('memory_release',), dict(allocated=1, reserved=0)),
                     (3, ('provenance', 'checkpoint_sha256'), '0' * 64)]
        for index, keys, value in mutations:
            with self.subTest(keys=keys):
                rows = list(copy.deepcopy(receipts())); target = rows[index]
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with self.assertRaises(ValueError):
                    subject.validate_receipts(*rows, COMMIT)

    def test_learned_chain_requires_bounded_integer_counts_and_full_tensor_closure(self):
        for mutation in ('over-final', 'zero-final', 'bool-final', 'bad-link', 'over-parent', 'tensor-missing'):
            with self.subTest(mutation=mutation):
                final, host, guest, sidecar = copy.deepcopy(receipts())
                if mutation in ('over-final', 'zero-final', 'bool-final'):
                    n = {'over-final': 1842969, 'zero-final': 0, 'bool-final': True}[mutation]
                    final['independent_verification']['rows'] = guest['output']['rows'] = sidecar['output']['rows'] = n
                    guest['output']['subdivisions'][3]['selected_children'] = n
                    sidecar['output']['subdivisions'][3]['selected_children'] = n
                elif mutation == 'bad-link':
                    guest['output']['subdivisions'][2]['parents'] = 18
                    sidecar['output']['subdivisions'][2]['parents'] = 18
                elif mutation == 'over-parent':
                    guest['saved_inputs']['tokens'] = sidecar['provenance']['inputs']['tokens'] = 4431
                    guest['output']['subdivisions'][0]['parents'] = sidecar['output']['subdivisions'][0]['parents'] = 4431
                else:
                    for hashes in (final['independent_verification']['tensor_payload_sha256'], guest['output']['tensor_payload_sha256'], sidecar['output']['tensor_payload_sha256']):
                        hashes.pop('features', None)
                with self.assertRaises(ValueError):
                    subject.validate_receipts(final, host, guest, sidecar, COMMIT)

    def test_original_cameras_are_distinct_finite_flat_CPU_views(self):
        metadata, mesh = original_records()
        camera = subject.validate_original_records(metadata, mesh)
        self.assertEqual(camera['target'], [0., 0., 0.])
        self.assertEqual(camera['extent'], 1.)
        for mutation in ('missing', 'repeated', 'path', 'nonfinite', 'smooth', 'gpu', 'threads'):
            with self.subTest(mutation=mutation):
                altered = copy.deepcopy(metadata)
                if mutation == 'missing': altered['views'].pop()
                if mutation == 'repeated': altered['views'][1] = altered['views'][0]
                if mutation == 'path': altered['views'][0]['name'] = '../escape.png'
                if mutation == 'nonfinite': altered['views'][0]['ortho_scale'] = float('nan')
                if mutation == 'smooth': altered['smooth_shading'] = True
                if mutation == 'gpu': altered['device'] = 'GPU'
                if mutation == 'threads': altered['threads'] = 8
                with self.assertRaises(ValueError): subject.validate_original_records(altered, mesh)

    def test_camera_drift_or_mesh_hash_drift_invalidates_new_render(self):
        original, mesh = original_records()
        new = copy.deepcopy(original)
        new.update(source_PLY_sha256=SOURCE_SHA, matched_original_cameras=True,
                   original_PLY_sha256=original['source_PLY_sha256'])
        subject.validate_render_metadata(original, new, SOURCE_SHA)
        for mutation in ('camera', 'scale', 'angle', 'source', 'smooth', 'modifiers'):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(new)
                if mutation == 'camera': changed['views'][2]['camera_location'][0] += .001
                if mutation == 'scale': changed['views'][2]['ortho_scale'] += .001
                if mutation == 'angle': changed['views'][2]['azimuth_degrees'] += 1
                if mutation == 'source': changed['source_PLY_sha256'] = '0' * 64
                if mutation == 'smooth': changed['smooth_shading'] = True
                if mutation == 'modifiers': changed['modifiers'] = ['Subdivision']
                with self.assertRaises(ValueError): subject.validate_render_metadata(original, changed, SOURCE_SHA)

    def test_immutable_original_hash_and_size_are_rechecked(self):
        with scratch_directory() as folder:
            path = Path(folder) / 'original.ply'; path.write_bytes(b'immutable')
            snapshot = subject.snapshot([path])
            subject.verify_snapshot(snapshot)
            path.write_bytes(b'altered!!')
            with self.assertRaises(ValueError): subject.verify_snapshot(snapshot)

    def test_audit_denies_models_children_network_writes_outside_and_destructive_mutation(self):
        with scratch_directory() as folder:
            root = Path(folder); audit = subject.audit_policy(root)
            audit('open', (str(root / 'owned.json'), 'x', 0))
            audit('open', (str(root.parent / 'read-only.json'), 'r', 0))
            for event, args in [('import', ('torch.cuda',)), ('import', ('trellis2',)),
                                ('subprocess.Popen', ('wsl.exe',)), ('socket.connect', (None, ('127.0.0.1', 9))),
                                ('open', (str(root.parent / 'escape.json'), 'w', 0)),
                                ('os.remove', (str(root / 'owned.json'),)), ('os.mkdir', (str(root.parent / 'escape'), 511, -1))]:
                with self.subTest(event=event), self.assertRaises((RuntimeError, ValueError)):
                    audit(event, args)

    def test_renderer_and_worker_reject_bad_GO_before_runtime_imports(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'bpy', 'mathutils', 'amd_generation_mesh'}:
                raise AssertionError('Runtime import preceded GO')
            return original(name, *args, **kwargs)
        for token in (b'', b'NO\n', b'GO\nx'):
            with self.subTest(token=token), patch.object(sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(token))), \
                 patch.object(builtins, '__import__', side_effect=guarded):
                with self.assertRaisesRegex(RuntimeError, 'GO'):
                    subject.worker('extract')
                sys.stdin.buffer = io.BytesIO(token)
                with self.assertRaisesRegex(RuntimeError, 'GO'):
                    runpy.run_path(str(PATH.with_name('amd_zombie_d_compare_render.py')), run_name='__main__')

    def test_owned_routing_changes_only_render_script_or_exact_worker_target(self):
        original_render = PATH.with_name('amd_zombie_mesh_render.py')
        renderer = [str(subject.BLENDER), '--background', '--factory-startup', '--disable-autoexec',
                    '--threads', '4', '--python-exit-code', '1', '--python', str(original_render)]
        routed = subject.route_command('render', renderer)
        self.assertEqual(routed[:-1], renderer[:-1])
        self.assertEqual(routed[-1], str(PATH.with_name('amd_zombie_d_compare_render.py')))
        worker = [sys._base_executable, '-B', '-I', '-S', str(PATH.resolve()), '--worker', 'extract']
        self.assertEqual(subject.route_command('extract', worker), worker)
        for changed in (renderer + ['--debug'], ['wsl.exe'], renderer[:5] + ['8'] + renderer[6:]):
            with self.assertRaises(ValueError): subject.route_command('render', changed)

    def test_scoped_inherited_owner_keeps_GO_cap_deadline_and_unattached_child_cleanup(self):
        import amd_forward_owned_job
        for attachment_fails, finish_fails in ((False, False), (True, False), (True, True)):
            with self.subTest(attachment_fails=attachment_fails, finish_fails=finish_fails), scratch_directory() as folder:
                base = Path(folder); actual_load = subject.load
                owner = actual_load('fixture_cpu_owner', PATH.with_name('amd_zombie_cpu_mesh.py'))
                original_base, original_file, original_process_api = owner.BASE, owner.__file__, owner.subprocess
                owner.resource = lambda: dict(admitted=True, available_RAM_bytes=12884901888, free_disk_bytes=2147483648)
                class Child:
                    def __init__(self):
                        self.attached = False; self.returncode = None; self.reaped = False
                        child = self
                        class Gate(io.BytesIO):
                            def write(self, value):
                                if not child.attached: raise AssertionError('GO preceded owned attachment')
                                if value != b'GO\n': raise AssertionError('GO changed')
                                return super().write(value)
                            def close(self):
                                if child.attached: child.returncode = 0
                                super().close()
                        self.stdin = Gate()
                    def poll(self): return self.returncode
                    def terminate(self): self.returncode = -1
                    def kill(self): self.returncode = -1
                    def wait(self, timeout):
                        if self.returncode is None: raise subject.subprocess.TimeoutExpired('owned fixture', timeout)
                        self.reaped = True; return self.returncode
                child = Child()
                class Job:
                    handle = 1
                    api = SimpleNamespace(SetInformationJobObject=lambda *args: True, QueryInformationJobObject=lambda *args: True)
                    def attach(self, process):
                        if attachment_fails: raise OSError('synthetic attachment failure')
                        process.attached = True
                    def active(self): return 0
                    def finish(self):
                        if finish_fails: raise OSError('synthetic job cleanup failure')
                        return dict(owned_tree_exit_verified=True, forced_job_cleanup=False)
                commands = []
                def spawn(command, **kwargs):
                    commands.append(command)
                    self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '1')
                    return child
                with patch.object(subject, 'BASE', base), patch.object(subject, 'load', return_value=owner), \
                     patch.object(subject.subprocess, 'Popen', side_effect=spawn), \
                     patch.object(amd_forward_owned_job, 'OwnedJob', return_value=Job()), redirect_stdout(io.StringIO()):
                    code = subject.supervise('render')
                result = subject.read(base / 'render-run/host-result.json')
                self.assertEqual(result['process_cap_bytes'], 6442450944)
                self.assertEqual(result['deadline_seconds'], 180)
                self.assertTrue(child.reaped)
                self.assertEqual(commands[0][-1], str(PATH.with_name('amd_zombie_d_compare_render.py')))
                self.assertEqual(code, 1 if attachment_fails else 0)
                self.assertEqual(result['classification'], 'FAIL' if attachment_fails else 'PASS')
                if attachment_fails:
                    self.assertTrue(result['unattached_owned_child_terminated'])
                    self.assertIn('attachment failure', result['error'])
                # The actual historical module remains an independent namespace.
                historical = actual_load('independent_historical_owner', PATH.with_name('amd_zombie_cpu_mesh.py'))
                self.assertEqual(historical.BASE, original_base)
                self.assertEqual(historical.__file__, original_file)
                self.assertIs(historical.subprocess, original_process_api)

    def test_quality_report_never_infers_digits_or_surface_quality_from_counts(self):
        report = subject.quality_review(dict(vertices=99999, triangles=199999), dict(vertices=3, triangles=1))
        self.assertEqual(report['visual_status'], 'UNASSESSED_REQUIRES_HUMAN_VISUAL_REVIEW')
        self.assertIsNone(report['overall_quality_winner'])
        self.assertFalse(report['semantic_digit_classifier_used'])
        self.assertEqual({row['criterion'] for row in report['criteria']},
                         {'finger_separation', 'toe_separation', 'unwanted_holes', 'wound_preservation', 'surface_defects'})
        self.assertTrue(all(row['assessment'] is None for row in report['criteria']))

    def test_exclusive_run_directory_is_never_reused(self):
        with scratch_directory() as folder:
            path = Path(folder) / 'fresh'
            subject.reserve(path)
            with self.assertRaises(FileExistsError): subject.reserve(path)
            self.assertEqual(list(path.iterdir()), [])

    def test_clean_cpu_exit_without_review_receipt_cannot_claim_comparison_ready(self):
        with scratch_directory() as folder:
            base = Path(folder) / 'absent-comparison'
            interpreter = Path(sys.executable)
            admission = SimpleNamespace(BASE=None, resource=lambda: dict(admitted=True))
            stages = []
            with patch.multiple(subject, BASE=base, PYTHON=interpreter, PYTHON_SHA=subject.digest(interpreter),
                                BLENDER=interpreter, FILES=[]), \
                 patch.object(ctypes.windll.shell32, 'IsUserAnAdmin', return_value=False), \
                 patch.object(subject, 'verify_publication'), \
                 patch.object(subject, 'original_inputs', return_value=({}, {}, {}, {})), \
                 patch.object(subject, 'receipt_inputs', return_value=({}, {}, {})), \
                 patch.object(subject, 'secondary_inputs', return_value=({}, {})), \
                 patch.object(subject, 'load', return_value=admission), \
                 patch.object(subject, 'supervise', side_effect=lambda stage: stages.append(stage) or 0), \
                 redirect_stdout(io.StringIO()):
                result = subject.run(COMMIT)
            self.assertEqual(stages, ['extract', 'render', 'review'])
            self.assertEqual(result, 1)
            self.assertEqual(subject.read(base / 'comparison-final.json')['classification'], 'FAIL')

    def test_missing_independent_decode_stops_before_private_output_reservation(self):
        with scratch_directory() as folder:
            base = Path(folder) / 'absent-comparison'; interpreter = Path(sys.executable)
            with patch.multiple(subject, BASE=base, PYTHON=interpreter, PYTHON_SHA=subject.digest(interpreter)), \
                 patch.object(ctypes.windll.shell32, 'IsUserAnAdmin', return_value=False), \
                 patch.object(subject, 'verify_publication'), \
                 patch.object(subject, 'original_inputs', return_value=({}, {}, {}, {})), \
                 patch.object(subject, 'receipt_inputs', side_effect=ValueError('Independent decode PASS missing')), \
                 patch.object(subject, 'supervise', side_effect=AssertionError('CPU launch before decode PASS')):
                with self.assertRaisesRegex(ValueError, 'Independent decode PASS'):
                    subject.run(COMMIT)
            self.assertFalse(base.exists())

    def test_duplicate_or_nonfinite_JSON_is_rejected(self):
        with scratch_directory() as folder:
            path = Path(folder) / 'input.json'
            for contents in ('{"x":1,"x":2}', '{"x":NaN}'):
                path.write_text(contents)
                with self.assertRaises(ValueError): subject.read(path)

    def test_saved_C_receipt_closure_is_verified_without_importing_any_neural_runtime(self):
        with scratch_directory() as folder:
            root = Path(folder) / 'proof'
            c_root = root / 'docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/zombie-d-9ac0364f-v1'
            occupancy, shape, decode = c_root / 'occupancy', c_root / 'shape', c_root / 'decode'
            source = decode / 'run/guest/shape-decoded.safetensors'
            reference = root / 'docs/compatibility/amd-zombie-d-reference-contract.json'
            for directory in (occupancy / 'dino/run/scratch', shape / 'run/scratch', source.parent):
                directory.mkdir(parents=True)
            def write_json(path, value):
                path.write_text(json.dumps(value), encoding='utf-8')
            def file_sha(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()
            image = occupancy / 'source-image.png'; image.write_bytes(b'synthetic raw image fixture')
            image_sha = file_sha(image)
            source.write_bytes(b'synthetic future neural artifact fixture')
            source_sha = file_sha(source)
            input_contract = dict(source_path=(image.relative_to(root)).as_posix(), source_sha256=image_sha,
                                  pixel_sha256='e' * 64, input_png_sha256='f' * 64)
            input_path = occupancy / 'input-contract.json'; write_json(input_path, input_contract)
            input_sha = file_sha(input_path)
            payload = bytes(4214784); payload_sha = hashlib.sha256(payload).hexdigest()
            feature_metadata = dict(input_pixel_sha256='e' * 64, input_png_sha256='f' * 64,
                                    output_sha256=payload_sha, harness_commit=COMMIT)
            feature_path = occupancy / 'dino/run/scratch/conditioning.safetensors'
            header = json.dumps({'__metadata__': feature_metadata,
                                 'cond': dict(dtype='F32', shape=[1, 1029, 1024], data_offsets=[0, 4214784])}).encode()
            feature_path.write_bytes(struct.pack('<Q', len(header)) + header + payload)
            shape_result = dict(classification='PASS', process=dict(harness_commit=COMMIT), worker=dict(generation_inputs=dict(
                schema='modly.amd.generation-shape-inputs.v1', token_count=2, input_contract_sha256=input_sha,
                occupancy_harness_commit=COMMIT, feature_file_sha256=file_sha(feature_path), feature_payload_sha256=payload_sha)))
            shape_path = shape / 'run/results.json'; write_json(shape_path, shape_result)
            write_json(shape / 'pre-execution-admission.json', dict(context=dict(occupancy_evidence=str(occupancy), evidence=str(shape), harness_commit=COMMIT)))
            paths, hashes = {}, {}
            for name, leaf in (('coords.npy', 'shape-latent-coordinates.npy'), ('shape-latent-denormalized.npy', 'shape-latent-denormalized.npy')):
                path = shape / 'run/scratch' / leaf; path.write_bytes(b'synthetic ' + name.encode())
                paths[name] = '/mnt/' + str(path)[0].lower() + '/' + str(path)[3:].replace('\\', '/')
                hashes[name] = file_sha(path)
            contract = dict(schema='modly.amd.neural-run.v1', large_convolution_tiling=True, initial_count=2,
                            maximum_final_rows=1842968, maximum_intermediate_rows=1000000,
                            input_paths=paths, input_sha256=hashes, source_job_sha256=file_sha(shape_path))
            contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            write_json(decode / 'run-contract.json', contract); write_json(source.parent / 'run-contract.json', contract)
            (source.parent / 'phases.jsonl').write_bytes(b'{"synthetic":true}\n')
            final, host, guest, sidecar = copy.deepcopy(receipts())
            final['run_contract_sha256'] = contract_sha
            final['independent_verification'].update(sha256=source_sha, run_contract_sha256=contract_sha, source_job_sha256=file_sha(shape_path))
            guest.update(run_contract_sha256=contract_sha, source_job_sha256=file_sha(shape_path))
            guest['output'].update(sha256=source_sha, bytes=source.stat().st_size)
            guest['saved_inputs'].update(run_contract_sha256=contract_sha, source_job_sha256=file_sha(shape_path), input_sha256=hashes)
            sidecar.update(output=guest['output'], run_contract=contract)
            sidecar['provenance'].update(inputs=guest['saved_inputs'], run_contract_sha256=contract_sha,
                                        source_job_sha256=file_sha(shape_path), input_file_sha256=hashes)
            write_json(source.parent / 'results.json', guest); write_json(source.parent / 'provenance.json', sidecar)
            final['allowlisted_copy'] = {name: dict(bytes=(source.parent / name).stat().st_size, sha256=file_sha(source.parent / name))
                                        for name in ('run-contract.json', 'phases.jsonl', 'results.json', 'provenance.json', 'shape-decoded.safetensors')}
            write_json(decode / 'final-result.json', final); write_json(decode / 'run/results.json', host)
            write_json(reference, dict(schema='modly.amd.zombie-d-reference.v1', source_sha256=image_sha,
                source_path=image.relative_to(root).as_posix(), input_contract_path=input_path.relative_to(root).as_posix(),
                input_contract_sha256=input_sha, source_pixel_sha256='e' * 64, encoded_worker_input_png_sha256='f' * 64))
            original_import = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name.split('.')[0] in {'numpy', 'torch', 'amd_generation_mesh', 'amd_generation_decode', 'amd_generation_shape'}:
                    raise AssertionError('Metadata verification imported neural runtime')
                return original_import(name, *args, **kwargs)
            with patch.multiple(subject, ROOT=root, D_ROOT=c_root, DECODE=decode, OCCUPANCY=occupancy, SHAPE=shape,
                                SOURCE=source, REFERENCE=reference, IMAGE_SHA=image_sha), \
                 patch.object(builtins, '__import__', side_effect=guarded):
                actual_contract, binding, snapshots = subject.receipt_inputs(COMMIT)
                self.assertEqual(actual_contract, contract)
                self.assertEqual(binding['source_sha256'], source_sha)
                self.assertEqual(binding['rows'], 251)
                subject.verify_snapshot(snapshots)
                (source.parent / 'phases.jsonl').write_bytes(b'{"synthetic":false}\n')
                with self.assertRaises(ValueError): subject.receipt_inputs(COMMIT)


if __name__ == '__main__':
    unittest.main()
