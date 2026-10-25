"""Render recovery policy tests; never run the character renderer or extraction."""
import sys
_initial_bytecode_flag = sys.dont_write_bytecode
# Test harness imports must not create caches before the renderer's own audit.
sys.dont_write_bytecode = True
import ast
import builtins
import copy
from contextlib import redirect_stdout
import ctypes
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PATH = HERE / 'amd_zombie_c_render_recovery.py'
subject = None
if PATH.exists():
    spec = importlib.util.spec_from_file_location('recovery_policy', PATH)
    subject = importlib.util.module_from_spec(spec); spec.loader.exec_module(subject)


def scratch():
    base = HERE.parents[2] / 'artifacts/amd-zombie-c-render-recovery-tests'
    base.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='policy-', dir=base))


def natural(stage):
    return dict(classification='PASS', stage=stage, process_cap_bytes=6*1024**3, deadline_seconds=180,
                initial_resource=dict(admitted=True, required_RAM_bytes=12*1024**3, required_disk_bytes=2*1024**3),
                samples=[dict(admitted=True)], worker_returncode=0, worker_exit_verified=True, owned_job_attached=True,
                job_cleanup=dict(owned_tree_exit_verified=True, forced_job_cleanup=False), cleanup_errors=[])


class EmbeddedInterpreter(unittest.TestCase):
    def test_bytecode_disabled_in_entrypoint_before_third_party_imports(self):
        tree = ast.parse((HERE / 'amd_zombie_c_compare_render.py').read_text())
        disabled = False
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                    and t.value.id == 'sys' and t.attr == 'dont_write_bytecode' for t in node.targets):
                disabled = isinstance(node.value, ast.Constant) and node.value.value is True
            if isinstance(node, ast.Import) and any(a.name in ('numpy', 'bpy') for a in node.names):
                self.assertTrue(disabled, 'Embedded Blender ignores PYTHONDONTWRITEBYTECODE; set sys flag before imports')

    def test_original_scene_settings_and_lighting_are_unchanged(self):
        source=(HERE/'amd_zombie_c_compare_render.py').read_text()
        sections=[('bpy.ops.wm.read_factory_settings(use_empty=True)',
                   "props = set(bpy.ops.wm.ply_import.get_rna_type().properties.keys())",
                   '69083bd721695ae042919701e00107b9857a0ac993ae1ae25b9e5be66d65a17f'),
                  ('center = (Vector(original_mesh','views = []; began =',
                   '138e4412c78ecaf1a804219bfcfccca47b880c64b6bc6fe0e505a2a73a1ad14f')]
        for begin,end,expected in sections:
            semantic=ast.dump(ast.parse(source[source.index(begin):source.index(end)]),include_attributes=False)
            self.assertEqual(hashlib.sha256(semantic.encode()).hexdigest(),expected)

    @unittest.skipUnless(os.environ.get('MODLY_BLENDER_IMPORT_TEST') == '1', 'Explicit harmless Blender import integration')
    def test_actual_embedded_numpy_import_has_no_cache_attempt_and_denies_external_writes(self):
        folder = scratch()
        owner = subject.p.load('_blender_import_test_owner', HERE / 'amd_zombie_cpu_mesh.py'); owner.BASE = folder
        launch = subprocess.Popen
        def routed(command, **kwargs):
            expected = [str(owner.BLENDER), '--background', '--factory-startup', '--disable-autoexec', '--threads', '4',
                        '--python-exit-code', '1', '--python', str(HERE / 'amd_zombie_mesh_render.py')]
            self.assertEqual(command, expected)
            return launch(command[:-1] + [str(Path(__file__).resolve()), '--', '--blender-import-worker'], **kwargs)
        owner.subprocess = SimpleNamespace(Popen=routed, PIPE=subprocess.PIPE)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(owner.supervise('render'), 0)
        row = subject.p.read(folder / 'probe-result.json')
        self.assertEqual(row['classification'], 'PASS')
        self.assertEqual(row['initial']['env'], '1')
        self.assertEqual(row['initial']['ignore_environment'], 1)
        self.assertFalse(row['initial']['dont_write_bytecode'])
        self.assertTrue(row['final_dont_write_bytecode'])
        self.assertTrue(row['numpy_imported'])
        self.assertEqual(row['bootstrap_writes'], [])
        self.assertEqual(row['deliberate_external_writes_denied'], ['open', 'mkdir'])
        subject.natural_pass(subject.p.read(folder / 'render-run/host-result.json'), 'render')


class Exists(unittest.TestCase):
    def test_render_only_recovery_exists(self):
        self.assertIsNotNone(subject, 'Render-only recovery missing')


@unittest.skipIf(subject is None, 'Not implemented')
class RecoveryPolicy(unittest.TestCase):
    def test_default_does_not_spawn_import_runtime_or_reserve(self):
        original = builtins.__import__
        def guard(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'PIL', 'bpy', 'torch', 'amd_generation_mesh'}:
                raise AssertionError('Default imported runtime')
            return original(name, *args, **kwargs)
        with patch.object(builtins, '__import__', side_effect=guard), \
             patch.object(subject.subprocess, 'Popen', side_effect=AssertionError('Spawned')), \
             patch.object(subject.p, 'reserve', side_effect=AssertionError('Reserved')), redirect_stdout(io.StringIO()):
            self.assertEqual(subject.main([]), 0)

    def test_bad_arguments_and_extract_worker_reject(self):
        for args in (['--run'], ['--run', '--harness-commit', 'bad'], ['--worker', 'extract'],
                     ['--worker', 'review', '--run', '--harness-commit', 'a'*40]):
            with self.subTest(args=args), patch.object(subject, 'run', side_effect=AssertionError('Reached run')), \
                 self.assertRaises(SystemExit):
                subject.main(args)

    def test_routes_exact_original_owner_commands_only(self):
        for stage in ('authenticate', 'review'):
            cmd = [sys._base_executable, '-B', '-I', '-S', str(PATH), '--worker', stage]
            self.assertEqual(subject.route_command(stage, cmd), cmd)
            with self.assertRaises(ValueError): subject.route_command(stage, cmd + ['--run'])
        cmd = [str(subject.p.BLENDER), '--background', '--factory-startup', '--disable-autoexec', '--threads', '4',
               '--python-exit-code', '1', '--python', str(HERE / 'amd_zombie_mesh_render.py')]
        expected = cmd[:-1] + [str(HERE / 'amd_zombie_c_compare_render.py'), '--', '--render-recovery']
        self.assertEqual(subject.route_command('render', cmd), expected)
        for stage, command in (('extract', cmd), ('render', ['wsl.exe']), ('review', ['cmd.exe']), ('render', cmd + ['--python-expr', 'evil'])):
            with self.assertRaises(ValueError): subject.route_command(stage, command)

    def test_natural_pass_receipt_requires_limits_and_cleanup(self):
        good = natural('extract')
        subject.natural_pass(good, 'extract')
        for key, value in [('classification','FAIL'), ('worker_returncode',1), ('process_cap_bytes',7*1024**3),
                           ('deadline_seconds',181), ('cleanup_errors',['error']), ('measurement_error','missing')]:
            row = dict(good, **{key:value})
            with self.subTest(key=key), self.assertRaises(ValueError): subject.natural_pass(row, 'extract')

    def test_recovery_exclusive_directory_cannot_be_reused(self):
        folder = scratch()
        with self.assertRaises(FileExistsError): subject.p.reserve(folder)
        self.assertEqual(list(folder.iterdir()), [])

    def test_no_go_token_rejects_before_imports(self):
        with patch.object(sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(b'NO\n'))), \
             patch.object(subject, 'prior_inputs', side_effect=AssertionError('Read inputs')):
            with self.assertRaises(RuntimeError): subject.worker('authenticate')

    def test_failed_stage_stops_without_retry_or_extraction(self):
        for fail in ('authenticate', 'render', 'review'):
            base = scratch() / 'new'
            prior = dict(decode_binding={}, immutable_original={}, decoded_inputs={}, camera={}, preserved_previous={})
            owner = SimpleNamespace(BASE=None, resource=lambda:dict(admitted=True))
            stages=[]
            def supervise(stage):
                stages.append(stage)
                return int(stage == fail)
            with patch.object(subject, 'BASE', base), patch.object(subject, 'environment'), \
                 patch.object(subject, 'verify_publication'), patch.object(subject, 'prior_inputs', return_value=prior), \
                 patch.object(subject.p, 'snapshot', return_value={}), patch.object(subject.p, 'load', return_value=owner), \
                 patch.object(subject, 'supervise', side_effect=supervise), redirect_stdout(io.StringIO()):
                self.assertEqual(subject.run('a'*40), 1)
                self.assertEqual(stages, list(subject.STAGES)[:subject.STAGES.index(fail)+1])
                self.assertNotIn('extract', stages)
                self.assertEqual(subject.p.read(base/'recovery-final.json')['classification'], 'FAIL')

    def test_clean_stage_exits_without_output_receipts_cannot_claim_pass(self):
        base = scratch() / 'new'
        prior = dict(decode_binding={}, immutable_original={}, decoded_inputs={}, camera={}, preserved_previous={})
        owner = SimpleNamespace(BASE=None, resource=lambda:dict(admitted=True))
        with patch.object(subject, 'BASE', base), patch.object(subject, 'environment'), \
             patch.object(subject, 'verify_publication'), patch.object(subject, 'prior_inputs', return_value=prior), \
             patch.object(subject.p, 'snapshot', return_value={}), patch.object(subject.p, 'load', return_value=owner), \
             patch.object(subject, 'supervise', return_value=0), redirect_stdout(io.StringIO()):
            self.assertEqual(subject.run('a'*40), 1)

    def test_audit_is_original_strict_policy(self):
        audit = subject.p.audit_policy(scratch())
        for event,args in [('open',(str(subject.p.BLENDER.parent/'denied.pyc'),'wb',os.O_WRONLY|os.O_CREAT)),
                           ('os.mkdir',(str(subject.PRIOR/'new-cache'),0o777,-1)),
                           ('subprocess.Popen',('wsl.exe',['wsl.exe'],None,None)),
                           ('import',('torch',None,None,None,None)), ('socket.connect',(None,('localhost',80))),
                           ('os.rename',('a','b',-1,-1))]:
            with self.subTest(event=event), self.assertRaises((ValueError,RuntimeError)):
                audit(event,args)

    def test_unchanged_input_snapshot_is_enforced(self):
        folder=scratch(); path=folder/'input';path.write_bytes(b'original')
        rows=subject.p.snapshot([path]);subject.p.verify_snapshot(rows)
        path.write_bytes(b'changed')
        with self.assertRaises(ValueError):subject.p.verify_snapshot(rows)

    def test_empty_failed_log_can_be_frozen_but_not_rewritten(self):
        path=scratch()/'stderr.txt';path.write_bytes(b'')
        rows=subject.p.snapshot([path], allow_empty=True)
        subject.p.verify_snapshot(rows)
        with self.assertRaises(ValueError):subject.p.snapshot([path])
        path.write_bytes(b'changed')
        with self.assertRaises(ValueError):subject.p.verify_snapshot(rows)

    def test_pinned_previous_file_mutation_is_rejected_before_decode_auth(self):
        folder=scratch();prior=folder/'prior';prior.mkdir();(prior/'mesh.ply').write_bytes(b'changed')
        contract=folder/'contract.json'
        subject.p.save(contract, dict(schema='modly.amd.zombie-c-render-recovery.v1',source_commit=subject.SOURCE_COMMIT,
                       previous_files={'mesh.ply':dict(bytes=7,sha256='a'*64)}))
        with patch.object(subject,'PRIOR',prior),patch.object(subject,'CONTRACT',contract), \
             patch.object(subject.p,'receipt_inputs',side_effect=AssertionError('Reached later auth')):
            with self.assertRaisesRegex(ValueError,'Pinned previous evidence'):subject.prior_inputs()

    def test_current_publication_files_include_recovery_and_unchanged_cpu_owner(self):
        self.assertIn('scripts/compatibility/amd_zombie_cpu_mesh.py',subject.FILES)
        self.assertIn('scripts/compatibility/amd_zombie_c_compare_render.py',subject.FILES)
        self.assertIn('scripts/compatibility/Invoke-ZombieCRenderRecovery.ps1',subject.FILES)
        with patch.object(subject.p,'verify_publication') as check:
            original=subject.p.FILES
            subject.verify_publication('a'*40)
            check.assert_called_once_with('a'*40)
            self.assertIs(subject.p.FILES,original)

    def test_existing_mesh_can_be_read_from_explicit_base_without_copying(self):
        folder=scratch();(folder/'mesh').mkdir()
        artifacts={}
        for name in ('mesh.ply','vertices.npy','faces.npy'):
            path=folder/'mesh'/name;path.write_bytes(name.encode())
            artifacts[name]=dict(bytes=path.stat().st_size,sha256=subject.p.digest(path))
        metadata=dict(input_sha256='a'*64,vertices=2,triangles=1,learned_subdivisions=[2],
                      PLY_readback_exact=True,zero_area_triangles=0,topology_repair=False,material_decoded=False,
                      native_extraction=False,artifacts=artifacts)
        binding=dict(source_sha256='a'*64,rows=2,learned_subdivisions=[2])
        subject.p.validate_mesh(metadata,binding,folder)
        self.assertEqual(set(p.name for p in (folder/'mesh').iterdir()),set(artifacts))

    def test_isolated_worker_startup_rejects_missing_go_without_runtime(self):
        result=subprocess.run([sys._base_executable,'-B','-I','-S',str(PATH),'--worker','authenticate'],
                              input=b'',stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15)
        self.assertNotEqual(result.returncode,0)
        self.assertIn(b'Exact owned GO plus EOF',result.stderr)
        self.assertNotIn(b'ModuleNotFoundError',result.stderr)

    def test_complete_matched_outputs_accept_and_camera_drift_rejects(self):
        fixture=subject.p.load('_comparison_synthetic_records',HERE/'test_amd_zombie_c_compare.py')
        original,_=fixture.original_records()
        base=scratch();(base/'previews').mkdir();(base/'review').mkdir()
        for label in ('original','candidate-C'):(base/'review'/label).mkdir()
        candidate=copy.deepcopy(original)
        candidate.update(source_PLY_sha256='b'*64,original_PLY_sha256=subject.p.ORIGINAL_SHA,matched_original_cameras=True)
        for row in candidate['views']:
            path=base/'previews'/row['name'];path.write_bytes(b'candidate '+row['name'].encode());row['sha256']=subject.p.digest(path)
        pairs=[]
        for old,new in zip(original['views'],candidate['views']):
            path=base/'review/original'/old['name'];path.write_bytes(b'original '+old['name'].encode());old['sha256']=subject.p.digest(path)
            (base/'review/candidate-C'/new['name']).write_bytes((base/'previews'/new['name']).read_bytes())
            pairs.append(dict(azimuth_degrees=old['azimuth_degrees']))
        origin=base/'saved-original';(origin/'previews').mkdir(parents=True)
        subject.p.save(origin/'previews/render-metadata.json',original)
        subject.p.save(base/'previews/render-metadata.json',candidate)
        for name in ('paired-review.html','paired-eight-view-contact-sheet.png'):(base/'review'/name).write_bytes(name.encode())
        reviewed=dict(classification='PASS_PAIRED_EVIDENCE_VISUAL_UNASSESSED',
                      visual_status='UNASSESSED_REQUIRES_HUMAN_VISUAL_REVIEW',overall_quality_winner=None,
                      criteria=[dict(assessment=None) for _ in range(5)],candidate_PLY_sha256='b'*64,pairs=pairs,
                      paired_review_sha256=subject.p.digest(base/'review/paired-review.html'),
                      contact_sheet_sha256=subject.p.digest(base/'review/paired-eight-view-contact-sheet.png'))
        subject.p.save(base/'review/comparison-review.json',reviewed)
        subject.p.save(base/'authentication-recovery.json',dict(classification='PASS_EXISTING_GEOMETRY_READBACK',
                       mesh_extracted=False,geometry_modified=False,PLY_NPY_payloads_bit_exact=True,PLY_sha256='b'*64))
        for stage in subject.STAGES:
            (base/(stage+'-run')).mkdir();subject.p.save(base/(stage+'-run/host-result.json'),natural(stage))
        with patch.object(subject,'BASE',base),patch.object(subject.p,'ORIGINAL',origin), \
             patch.object(subject,'verify_worker_inputs',return_value={}):
            self.assertEqual(subject.acceptance(),reviewed)
            candidate['views'][0]['ortho_scale']+=.1
            # Only synthetic test fixture is edited; no actual evidence is used.
            (base/'previews/render-metadata.json').write_text(json.dumps(candidate))
            with self.assertRaises(ValueError):subject.acceptance()


def blender_import_worker():
    """Execute the actual renderer through NumPy import, stopping before geometry."""
    base = Path(os.environ['TEMP'])
    entry = HERE / 'amd_zombie_c_compare_render.py'
    prefix = []
    for node in ast.parse(entry.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BASE' for t in node.targets):
            break
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'sys.addaudithook':
            prefix.extend(ast.parse('policy.BASE = __probe_base\nsys.addaudithook(__probe_audit)').body)
        prefix.append(node)
    writes = []
    def observe(event, args):
        if event == 'os.mkdir': writes.append(dict(event=event, path=str(args[0])))
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            if any(c in str(args[1]) for c in 'wax+') or args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                writes.append(dict(event=event, path=os.fsdecode(args[0])))
    initial = dict(dont_write_bytecode=_initial_bytecode_flag, ignore_environment=sys.flags.ignore_environment,
                   env=os.environ.get('PYTHONDONTWRITEBYTECODE'), numpy_already_loaded='numpy' in sys.modules)
    p = subject.p
    namespace = dict(__file__=str(entry), __name__='__probe__', __probe_base=base, __probe_audit=observe)
    sys.argv = sys.argv[:sys.argv.index('--')]
    sys.dont_write_bytecode = _initial_bytecode_flag
    exec(compile(ast.fix_missing_locations(ast.Module(body=prefix, type_ignores=[])), str(entry), 'exec'), namespace)
    assert not writes and 'np' in namespace and not initial['numpy_already_loaded']
    assert namespace['np'].arange(3).sum() == 3
    result = dict(classification='PASS', initial=initial, final_dont_write_bytecode=sys.dont_write_bytecode,
                  bootstrap_writes=writes.copy(), numpy_imported=True)
    outside = p.BLENDER.parent / '5.2/python/Lib/site-packages/numpy/__modly_denied_probe__'
    rejected = []
    for name, call in [('open', lambda:outside.open('xb')), ('mkdir', lambda:outside.mkdir())]:
        try: call()
        except (ValueError, RuntimeError): rejected.append(name)
        else: raise AssertionError('External write permitted')
    assert rejected == ['open', 'mkdir']
    result['deliberate_external_writes_denied'] = rejected
    p.save(base / 'probe-result.json', result)


if __name__ == '__main__':
    if '--' in sys.argv and sys.argv[sys.argv.index('--')+1:] == ['--blender-import-worker']:
        blender_import_worker()
    else:
        unittest.main()
