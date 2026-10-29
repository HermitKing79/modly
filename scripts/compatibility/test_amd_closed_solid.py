"""Synthetic CPU-only controls; never dispatch model stages."""
import ast,contextlib,io,itertools,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).parent))
try:import amd_closed_solid_benchmark as b
except ImportError:b=None

class GroundTruth(unittest.TestCase):
    def test_fixed_closed_convex_geometry(self):
        self.assertIsNotNone(b,'Closed-solid benchmark missing')
        v,f=b.geometry();r=b.small_topology(v,f)
        self.assertEqual((len(v),len(f)),(24,44));self.assertEqual(r['components'],1)
        self.assertEqual((r['boundary'],r['nonmanifold'],r['winding'],r['euler']),(0,0,0,2))
        self.assertGreater(r['signed_volume'],0);self.assertTrue(r['convex'])
        expected=1*.65*.8-2*b.CHAMFER**2*sum(b.DIMENSIONS)+(16/3)*b.CHAMFER**3
        self.assertAlmostEqual(r['signed_volume'],expected,places=7)
        self.assertEqual((v,f),b.geometry());self.assertEqual(b.DIMENSIONS,(1.0,.65,.8))
    def test_geometry_does_not_smuggle_extra_or_duplicate_faces(self):
        self.assertIsNotNone(b)
        v,f=b.geometry();self.assertEqual(len(set(map(tuple,v))),24)
        self.assertEqual(len({tuple(sorted(t)) for t in f}),44)
        self.assertTrue(all(len(set(t))==3 for t in f))
    def test_one_fixed_view_and_ground_truth_are_not_generated_output(self):
        self.assertIsNotNone(b)
        self.assertEqual(b.REFERENCE_RESOLUTION,1024);self.assertEqual(b.SEED,20261013)
        self.assertNotEqual(b.PREP,b.RAW);self.assertFalse(b.RAW.is_relative_to(b.PREP))
        self.assertEqual(b.CAMERA,[2.4,-3.2,2.5]);self.assertEqual(b.ORTHO,1.7)
    def test_cpu_policy_denies_model_network_and_original_writes(self):
        self.assertIsNotNone(b)
        with tempfile.TemporaryDirectory(dir=b.ROOT.parent/'artifacts') as t:
            guard=b.cpu_guard(Path(t));guard('open',(str(Path(t)/'new.json'),'x',0))
            for event,args in [('import',('torch',)),('subprocess.Popen',('wsl.exe',)),('socket.connect',('localhost',)),('os.remove',(str(Path(t)/'x'),))]:
                with self.assertRaises(RuntimeError):guard(event,args)
            with self.assertRaises(ValueError):guard('open',(str(b.ROOT/'original.txt'),'w',0))
    def test_blender_disables_bytecode_before_third_party_imports(self):
        p=Path(__file__).with_name('amd_closed_solid_render.py');self.assertTrue(p.exists(),'Owned renderer missing')
        tree=ast.parse(p.read_text());text=p.read_text()
        self.assertLess(text.index('sys.dont_write_bytecode = True'),text.index('import bpy'))
        self.assertIn("b'GO\\n'",text);self.assertIn("scene.cycles.device='CPU'",text)
    def test_cpu_reference_driver_has_no_implicit_work(self):
        p=Path(__file__).with_name('amd_closed_solid_cpu.py');self.assertTrue(p.exists(),'CPU driver missing')
        import amd_closed_solid_cpu as c
        with patch.object(c,'prepare',side_effect=AssertionError('reference')),patch.object(c,'future',side_effect=AssertionError('future')):
            with contextlib.redirect_stdout(io.StringIO()) as out:self.assertEqual(c.main([]),0)
        self.assertIn('DRY_RUN',out.getvalue())
    def test_executed_reference_core_matches_final_published_core(self):
        def function(path,name):return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        for file,names in [('amd_closed_solid_benchmark.py',['geometry','small_topology']),('amd_closed_solid_cpu.py',['owned','qualify'])]:
            for name in names:self.assertEqual(function(Path(__file__).with_name(file),name),function(b.PREP/'executed-source'/file,name))
        self.assertEqual(b.digest(Path(__file__).with_name('amd_closed_solid_render.py')),b.digest(b.PREP/'executed-source/amd_closed_solid_render.py'))

class Handoff(unittest.TestCase):
    def subject(self):
        p=Path(__file__).with_name('amd_closed_solid_handoff.py');self.assertTrue(p.exists(),'Conditional handoff missing')
        import amd_closed_solid_handoff as h
        return h
    def test_no_work_default_and_separate_approval(self):
        h=self.subject()
        with patch.object(h,'run',side_effect=AssertionError('execution')),patch.object(h,'windows_memory',side_effect=AssertionError('memory')):
            with contextlib.redirect_stdout(io.StringIO()) as out:self.assertEqual(h.main([]),0)
        r=json.loads(out.getvalue());self.assertTrue(r['execution_hold']);self.assertFalse(r['neural_execution'])
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):h.main(['--run','--harness-commit','a'*40])
    def test_memory_gates_reuse_exact_qualified_functions(self):
        h=self.subject();import amd_zombie_d_handoff as old
        self.assertIs(h.fresh_admission,old.fresh_admission);self.assertIs(h.guard_reservation,old.guard_reservation)
        self.assertEqual((h.FLOOR,h.PREFERRED,h.LIVE_FLOOR),(22*1024**3,24*1024**3,12*1024**3))
        sleeps=[];rows=[];h.fresh_admission(lambda:dict(available_bytes=h.FLOOR),sleeps.append,rows.append)
        self.assertEqual(sleeps,[4,4]);self.assertEqual(len(rows),3)
        with self.assertRaises(RuntimeError):h.guard_reservation(lambda:dict(available_bytes=h.FLOOR-1),lambda r:None,lambda:self.fail('reserved'))
    def test_no_retry_at_any_stage(self):
        h=self.subject()
        for i,name in enumerate(h.STAGES):
            calls=[]
            with self.assertRaises(RuntimeError):h.execute_once(lambda stage:calls.append(stage) or int(stage==name))
            self.assertEqual(calls,list(h.STAGES[:i+1]))
    def test_case_is_not_an_old_source_or_ground_truth(self):
        h=self.subject()
        self.assertEqual(h.CASE,b.CASE);self.assertEqual(h.COMPARISON,b.RAW)
        self.assertNotIn('zombie',str(h.CASE).lower());self.assertNotEqual(h.CASE,b.PREP)
        self.assertEqual(h.FORWARDS,dict(dino=1,occupancy_sampling=22,occupancy_decode=1,shape=21,geometry_decode=1))
    def test_private_stage_namespace_uses_explicit_memory_and_box_dispatch(self):
        self.subject()
        p=Path(__file__).with_name('amd_closed_solid_stages.py');self.assertTrue(p.exists(),'Scoped stage adapter missing')
        import amd_closed_solid_stages as s
        import amd_zombie_d_handoff as old
        self.assertIs(s.base.Bridge,s.Bridge);self.assertIs(s.base.h,s.h)
        self.assertEqual(Path(s.base.__file__),p)
        self.assertNotEqual(s.base.h.CASE,old.CASE)
        read=lambda:dict(available_bytes=0);bridge=s.Bridge('a'*40,read=read);self.assertIs(bridge.read,read)
        self.assertFalse({'torch','trellis2','amd_generation_occupancy','amd_generation_decode','amd_generation_shape'}.intersection(sys.modules))
    def test_fresh_adapter_refuses_existing_runtime(self):
        self.subject();import amd_closed_solid_stages as s
        with patch.dict(sys.modules,{'amd_generation_occupancy':object()}):
            with self.assertRaises(RuntimeError):s.Bridge('a'*40).install()
    def test_decode_preserves_qualified_tiling_limits_and_uses_new_run_id(self):
        h=self.subject();import amd_closed_solid_stages as s
        bridge=s.Bridge('a'*40);bridge.install=lambda:None;bridge._check_private=lambda:None
        runtime=NS(reserve_attempt=lambda out:None,supervise=lambda:0);constructor=Mock(return_value=runtime);bridge.decode=NS(Runtime=constructor)
        self.assertEqual(bridge.dispatch('decode'),0)
        constructor.assert_called_once_with(h.CASE/'shape',h.CASE/'decode','a'*40,'run-closed-solid-box-v1',tile_large_convolutions=True,final_support_profile=None)
    def test_real_reference_authority_and_unused_paths(self):
        h=self.subject();r=h.authority();self.assertEqual(r['reference_views'],1);self.assertTrue(r['neural_execution_hold'])
        h.require_fresh_outputs()
        for p in (h.PRIVATE,h.COMPARISON,h.CASE/'shape',h.CASE/'decode'):self.assertFalse(p.exists())
        self.assertEqual({p.name for p in (h.CASE/'occupancy').iterdir()},{'source-image.png','input-contract.json'})
        with patch.object(h.legacy,'verify_qualification',side_effect=ValueError('tampered')):
            with self.assertRaises(ValueError):h.authority()
    def test_publication_rejects_wrong_head_before_any_execution(self):
        h=self.subject()
        with patch.object(h.legacy,'git',return_value='b'*40):
            with self.assertRaises(ValueError):h.verify_publication('a'*40)
    def test_freshness_rejects_every_occupied_output(self):
        h=self.subject()
        with tempfile.TemporaryDirectory(dir=b.ROOT.parent/'artifacts') as t:
            root=Path(t);case=root/'case';(case/'occupancy').mkdir(parents=True)
            for name in ('source-image.png','input-contract.json'):(case/'occupancy'/name).write_bytes(b'x')
            with patch.multiple(h,CASE=case,PRIVATE=root/'private',COMPARISON=root/'raw'):
                for p in (h.PRIVATE,h.COMPARISON,case/'shape',case/'decode'):
                    p.mkdir()
                    with self.assertRaises(FileExistsError):h.require_fresh_outputs()
                    p.rmdir()
                (case/'occupancy'/'attempt-started.json').write_text('{}')
                with self.assertRaises(FileExistsError):h.require_fresh_outputs()

class Measurements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):sys.path.insert(0,'D:/AI/Trellis2AMD/venv/Lib/site-packages')
    def test_changed_preview_stops_before_pixel_measurement(self):
        import amd_closed_solid_cpu as c
        with tempfile.TemporaryDirectory(dir=b.ROOT.parent/'artifacts') as t:
            root=Path(t);gt=root/'gt';raw=root/'raw';gt.mkdir();(raw/'view').mkdir(parents=True);(raw/'mesh').mkdir()
            ref=json.loads((b.PREP/'render.json').read_text());meta=dict(ref,ground_truth=False,image_sha256='0'*64)
            (gt/'render.json').write_text(json.dumps(ref));(raw/'view/render.json').write_text(json.dumps(meta))
            (raw/'mesh/mesh.json').write_text(json.dumps(dict(artifacts={'mesh.ply':{'sha256':ref['source_PLY_sha256']}})))
            (raw/'benchmark-inputs.json').write_text(json.dumps(dict(binding={})));(raw/'view/raw-reference-camera.png').write_bytes(b'changed')
            fake=NS(validate_mesh=lambda *a:None,snapshot=lambda *a:{})
            with patch.multiple(b,PREP=gt,RAW=raw),patch.object(c,'decode_reader',return_value=fake):
                with self.assertRaisesRegex(ValueError,'Raw preview PNG differs'):c.compare_view()
            self.assertFalse((raw/'comparison').exists())
    def test_worker_checks_source_before_and_after_measurements(self):
        import amd_closed_solid_cpu as c
        with tempfile.TemporaryDirectory(dir=b.ROOT.parent/'artifacts') as t:
            root=Path(t);(root/'benchmark-inputs.json').write_text(json.dumps(dict(decoded_inputs={},published_sources={},binding={})))
            calls=[];snap={'mesh':'identity'}
            def verify(rows):
                if rows==snap:calls.append('after');raise ValueError('changed raw array')
            fake=NS(verify_snapshot=verify,validate_mesh=lambda *a:calls.append('before'),read=lambda *a:{},snapshot=lambda *a:snap)
            with patch.object(b,'RAW',root),patch.object(sys,'stdin',NS(buffer=io.BytesIO(b'GO\n'))),patch.object(sys,'addaudithook'),\
                 patch.object(c,'decode_reader',return_value=fake),patch.object(c,'topology_and_shape',side_effect=lambda:calls.append('measure')):
                with self.assertRaisesRegex(ValueError,'changed raw array'):c.worker('analyze')
            self.assertEqual(calls,['before','measure','after'])
    def test_small_synthetic_cpu_measurement_preserves_arrays_and_does_not_promote(self):
        sys.path.insert(0,'D:/AI/Trellis2AMD/venv/Lib/site-packages')
        import numpy as np
        import amd_closed_solid_cpu as c
        with tempfile.TemporaryDirectory(dir=b.ROOT.parent/'artifacts') as t:
            root=Path(t);(root/'mesh').mkdir();v,f=b.geometry()
            np.save(root/'mesh/vertices.npy',np.asarray(v,'f4'));np.save(root/'mesh/faces.npy',np.asarray(f,'i4'))
            before={n:b.digest(root/'mesh'/n) for n in ('vertices.npy','faces.npy')}
            with patch.object(b,'RAW',root):c.topology_and_shape()
            r=json.loads((root/'raw-quality.json').read_text());self.assertEqual(r['edges']['boundary_incidence_1'],0);self.assertEqual(r['edges']['nonmanifold_incidence_gt_2'],0)
            self.assertEqual(r['two_face_winding_conflicts'],0);self.assertTrue(r['degeneracy']['duplicate_faces_assessed'])
            self.assertEqual(r['euler_referenced_vertices'],2)
            self.assertEqual(r['shape']['GT_to_raw']['maximum'],0);self.assertFalse(r['texture_ready']);self.assertFalse(r['geometry_repair'])
            self.assertEqual(before,{n:b.digest(root/'mesh'/n) for n in before})
    def test_procedural_mesh_cannot_be_used_as_neural_decode(self):
        sys.path.insert(0,'D:/AI/Trellis2AMD/venv/Lib/site-packages')
        import amd_generation_mesh as m
        with self.assertRaises(ValueError):m.read_neural(b.PREP/'source/mesh.ply',b.digest(b.PREP/'source/mesh.ply'))
    def test_ten_tensor_reader_is_bound_only_to_box_and_rejects_gt_receipts(self):
        import amd_closed_solid_cpu as c
        r=c.decode_reader()
        self.assertEqual(r.SOURCE,b.CASE/'decode/run/guest/shape-decoded.safetensors');self.assertNotEqual(r.SOURCE,b.PREP/'source/mesh.ply')
        with self.assertRaises(ValueError):r.validate_receipts({'ground_truth':True},{},{},{},'a'*40)
        self.assertFalse({'torch','trellis2','transformers'}.intersection(sys.modules))

# Execute the unchanged qualified adapter safety cases against the Box binding,
# including actual harmless isolated Python worker rejection. No model dispatch.
import amd_closed_solid_stages as _scoped
_reused=b.load('_closed_solid_reused_stage_tests',Path(__file__).with_name('test_amd_zombie_d_stages.py'))
_reused.subject=_scoped;_reused.SCRIPT=Path(__file__).with_name('amd_closed_solid_stages.py')
class ReusedStageGuards(_reused.BridgePolicy):pass

if __name__=='__main__':unittest.main()
