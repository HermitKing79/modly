"""Offline recovery tests; no WSL, GPU, model loading or neural marker."""
import contextlib,io,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).parent))
try:
    import amd_closed_solid_recovery as h
    import amd_closed_solid_recovery_cpu as c
except ModuleNotFoundError:h=c=None

class Implemented(unittest.TestCase):
    def test_prepared(self):self.assertIsNotNone(h);self.assertIsNotNone(c)

@unittest.skipIf(h is None,'Not implemented')
class Recovery(unittest.TestCase):
    def test_original_six_tiling_records_and_resource_limits_unchanged(self):
        code=h.ROOT/'experimental/extensions/trellis2-amd/inference_decoder'
        with patch.object(sys,'path',[str(code),*sys.path]):
            policy=h.original.b.load('_recovery_original_admission',code/'admission.py')
            tiled=h.original.b.load('_recovery_original_large_tiling',code/'chunked_large.py')
        self.assertEqual((policy.ALLOCATOR_LIMIT,policy.INITIAL_FREE_REQUIRED,policy.FREE_RESERVE,policy.OPAQUE_HEADROOM),(12*1024**3,13*1024**3,2*1024**3,1024**3))
        self.assertEqual((policy.MAX_ROWS,policy.FINAL_ROWS,policy.TILE_ROWS),(1000000,1842968,32768))
        rows=[dict(site=site,rows=100,in_channels=128,out_channels=512 if i==5 else 128,tile_rows=32768,chunks=1,rows_completed=100,global_map_reused=i>0,completed=True) for i,site in enumerate(policy.LARGE_TILED_SITES)]
        tiled.validate_records(rows)
        with self.assertRaises(RuntimeError):tiled.validate_records(rows[:-1])
        for change in (dict(completed=False),dict(tile_rows=65536),dict(rows_completed=99)):
            bad=[dict(r) for r in rows];bad[0].update(change)
            with self.assertRaises(RuntimeError):tiled.validate_records(bad)
    def test_backup_postcheck_runs_when_decode_raises(self):
        with tempfile.TemporaryDirectory(dir=h.ROOT.parent/'artifacts') as d:
            identity=dict(file_id=7);metadata=dict(virtual_capacity_bytes=32*1024**3)
            cold=NS(native_file=lambda *a:contextlib.nullcontext(object()),file_identity=Mock(return_value=identity),metadata=Mock(return_value=metadata))
            with patch.object(h,'PRIVATE',Path(d)):
                with self.assertRaisesRegex(RuntimeError,'decode failed'):
                    with h.held_backup(cold,Path('unused'),dict(identity=identity,metadata=metadata)):raise RuntimeError('decode failed')
            self.assertEqual(cold.file_identity.call_count,2)
            self.assertTrue(json.loads((Path(d)/'backup-after.json').read_text())['unchanged'])
    def test_private_scratch_created_exclusively(self):
        with tempfile.TemporaryDirectory(dir=h.ROOT.parent/'artifacts') as d,patch.object(h,'PRIVATE',Path(d)/'private'):
            h.create_private();self.assertTrue((h.PRIVATE/'tmp').is_dir())
            with self.assertRaises(FileExistsError):h.create_private()
    def test_private_runtime_keeps_tiling_and_git_verifies_unknown_files(self):
        from types import ModuleType
        sup=ModuleType('amd_sparse_decoder_forward');sup.COMMITTED_FILES=[];sup.verify_committed_files=Mock(return_value={'code.py':'ok'})
        captured={}
        class Base:
            def __init__(self,*args,**kw):
                captured.update(args=args,kwargs=kw);self.commit=args[2];self.evidence=args[1]
                self.bundle=dict(count=3888,source_job_sha256='498948bb64a88e1f869a2c01ed0f38c51346ae347bc1775aa263f1c8a69f7ea5')
            def preflight(self):
                from amd_sparse_decoder_forward import verify_committed_files
                return verify_committed_files(self.commit,['code.py','private/source.png','private/unknown.png'])
        @contextlib.contextmanager
        def binding(name,value):
            old=sys.modules.get(name);sys.modules[name]=value
            try:yield
            finally:
                if old is None:sys.modules.pop(name,None)
                else:sys.modules[name]=old
        module=NS(Runtime=Base,STATIC_FILES=[],module_binding=binding)
        with patch.object(h.original.b,'load',return_value=module),patch.object(h,'authority'),patch.object(h,'read_authority',return_value=dict(private_repo={'private/source.png':'a'*64})),patch.object(h,'digest',return_value='a'*64),patch.dict(sys.modules,{'amd_sparse_decoder_forward':sup}):
            runtime=h.make_runtime('b'*40);result=runtime.preflight()
            self.assertEqual(captured['args'],(h.SHAPE,h.DECODE,'b'*40,h.RUN_ID))
            self.assertEqual(captured['kwargs'],dict(tile_large_convolutions=True,final_support_profile=None))
            self.assertEqual(result['private/source.png'],'a'*64)
            checked=sup.verify_committed_files.call_args.args[1]
            self.assertIn('private/unknown.png',checked);self.assertNotIn('private/source.png',checked)
            self.assertIs(sys.modules['amd_sparse_decoder_forward'],sup)
            with patch.object(h,'digest',return_value='wrong'):
                with self.assertRaises(ValueError):runtime.preflight()
    def test_default_is_inert_and_live_approval_explicit(self):
        with patch.object(h,'run',side_effect=AssertionError('live')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(h.main([]),0)
            with self.assertRaises(SystemExit):h.main(['--run'])
    def test_only_one_decode_then_cpu_no_retries(self):
        for value in (0,1):
            runtime=NS(supervise=Mock(return_value=value));cpu=Mock(return_value=0)
            if value:
                with self.assertRaises(RuntimeError):h.execute_once(runtime,cpu)
                cpu.assert_not_called()
            else:h.execute_once(runtime,cpu);cpu.assert_called_once()
            runtime.supervise.assert_called_once()
    def test_no_cpu_retry_after_cpu_failure(self):
        runtime=NS(supervise=Mock(return_value=0));cpu=Mock(return_value=1)
        with self.assertRaises(RuntimeError):h.execute_once(runtime,cpu)
        runtime.supervise.assert_called_once();cpu.assert_called_once()
    def test_new_paths_distinct_from_all_sources(self):
        self.assertEqual(h.RUN_ID,'run-closed-solid-box-decode-recovery-v1')
        self.assertEqual(h.SHAPE,h.original.CASE/'shape')
        self.assertNotEqual(h.DECODE,h.original.CASE/'decode')
        self.assertNotEqual(h.RAW,h.original.b.PREP)
        self.assertEqual(h.FORWARDS,dict(dino=0,occupancy_sampling=0,occupancy_decode=0,shape=0,geometry_decode=1))
    def test_consumed_output_refused(self):
        with tempfile.TemporaryDirectory(dir=h.ROOT.parent/'artifacts') as d:
            p=Path(d);paths=(p/'private',p/'case',p/'raw')
            with patch.multiple(h,PRIVATE=paths[0],RECOVERY=paths[1],RAW=paths[2]):
                for path in paths:
                    path.mkdir()
                    with self.assertRaises(FileExistsError):h.require_fresh()
                    path.rmdir()
    def test_original_failure_must_be_zero_forward_before_native(self):
        row=dict(forward_calls=0,native_import='NOT_ATTEMPTED',error='RuntimeError: CPU/disk admission refused',admission=dict(guest_disk_free=2649030656,guest_disk_requirement=5*1024**3))
        h.zero_forward_failure(row)
        for change in (dict(forward_calls=1),dict(native_import='PASS'),dict(error='other')):
            bad=dict(row,**change)
            with self.assertRaises(ValueError):h.zero_forward_failure(bad)
    def test_private_artifacts_only_exact_allowlist(self):
        authority=dict(private_repo={'private/source.png':'a'*64})
        self.assertEqual(h.partition_files(['code.py','private/source.png'],authority),(['code.py'],['private/source.png']))
        self.assertEqual(h.partition_files(['private/unknown.png'],authority),(['private/unknown.png'],[]))
    def test_final22_guard_stops_before_marker(self):
        with tempfile.TemporaryDirectory(dir=h.ROOT.parent/'artifacts') as d:
            runtime=NS(evidence=Path(d)/'decode',commit='a'*40,contract=NS(sha256='b'*64),host_admission=lambda:dict(available_ram_bytes=22*1024**3-1,D_free_bytes=100*1024**3))
            reserve=Mock()
            with self.assertRaises(RuntimeError):h.guard.reserve_guarded(runtime,reserve,runtime.evidence/'run',Path(d)/'gate.json',disk_free=lambda _:10*1024**3)
            reserve.assert_not_called()
    def test_historical_artifact_is_never_a_git_exemption_by_prefix(self):
        code,private=h.partition_files(['private/anything.npy'],dict(private_repo={}))
        self.assertEqual(code,['private/anything.npy']);self.assertEqual(private,[])

@unittest.skipIf(c is None,'Not implemented')
class CPU(unittest.TestCase):
    def test_real_cpu_entries_reject_absent_go_without_model_or_blender(self):
        environment=dict(os.environ,MODLY_CLOSED_SOLID_RECOVERY_COMMIT='a'*40)
        for flags in (['--worker','extract'],['--render','--generated']):
            result=subprocess.run([sys._base_executable,'-B','-I','-S',str(Path(c.__file__).resolve()),*flags],input=b'',capture_output=True,timeout=15,env=environment)
            self.assertNotEqual(result.returncode,0);self.assertIn(b'Owned GO plus EOF required',result.stderr)
    def test_two_commits_only_lineage_uses_upstream(self):
        old=Mock();wrapped=c.lineage_for('a'*40,old)
        wrapped(1,2,3,4,5,6,'a'*40)
        old.assert_called_once_with(1,2,3,4,5,6,h.UPSTREAM)
        with self.assertRaises(ValueError):wrapped(1,2,3,4,5,6,'b'*40)
    def test_render_routes_only_generated_mode_to_recovery_worker(self):
        expected=c.original_render_command()
        routed=c.render_command(expected)
        self.assertIn(str(Path(c.__file__).resolve()),routed)
        self.assertEqual(routed[-3:],['--','--render','--generated'])
        for args in (expected[:-1]+['--reference'],['wsl.exe'],['foreign.exe']):
            with self.assertRaises(ValueError):c.render_command(args)
    def test_closure_counters_cannot_claim_intersection_free(self):
        row=dict(edges=dict(boundary_incidence_1=0,nonmanifold_incidence_gt_2=0),components=dict(face_components=1),two_face_winding_conflicts=0,duplicate_index_triangles_ignoring_winding=0,degeneracy=dict(zero_area_faces_float64=0,repeated_index_faces=0,nonfinite_vertex_rows=0,faces_with_nonfinite_coordinates=0))
        result=c.closure(row)
        self.assertTrue(result['closed_oriented_edge_manifold_single_component'])
        self.assertFalse(result['self_intersections_verified']);self.assertFalse(result['texture_ready'])
        row['degeneracy']['nonfinite_vertex_rows']=1
        self.assertFalse(c.closure(row)['closed_oriented_edge_manifold_single_component'])
        row['degeneracy']['nonfinite_vertex_rows']=0
        row['edges']['boundary_incidence_1']=1
        self.assertFalse(c.closure(row)['closed_oriented_edge_manifold_single_component'])
    def test_cpu_binding_preserves_original_case_and_neural_tensor_source(self):
        with c.configured('a'*40) as cpu:
            reader=cpu.decode_reader()
            self.assertEqual(reader.SOURCE,h.DECODE/'run/guest/shape-decoded.safetensors')
            self.assertEqual(reader.SHAPE,h.SHAPE);self.assertEqual(cpu.b.RAW,h.RAW)
            self.assertNotEqual(reader.SOURCE,h.original.b.PREP/'source/mesh.ply')
        self.assertEqual(h.original.b.RAW,h.ROOT.parent/'artifacts/amd-closed-solid-raw-v1')

if __name__=='__main__':unittest.main(verbosity=2)
