"""CPU fixtures for read-only layered source inspection."""
import builtins,importlib.util,io,runpy,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).parent),'D:/AI/Trellis2AMD/venv/Lib/site-packages']
PATH=Path(__file__).with_name('amd_zombie_wound_source.py');subject=None
if PATH.exists():
    spec=importlib.util.spec_from_file_location('wound',PATH);subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)
class Exists(unittest.TestCase):
    def test_implementation_exists(self):self.assertIsNotNone(subject)
@unittest.skipIf(subject is None,'Not implemented')
class Audit(unittest.TestCase):
    def test_two_sided_hits_ignore_winding_and_retain_all_layers(self):
        import numpy as np
        p=np.array([[[0,0,0],[1,0,0],[0,0,1]],[[0,1,0],[0,1,1],[1,1,0]]],np.float64)
        a=subject.intersections(np,np.array([.2,-1,.2]),np.array([0,1,0]),p)
        b=subject.intersections(np,np.array([.2,-1,.2]),np.array([0,1,0]),p[:,[0,2,1]])
        self.assertEqual(a[0].tolist(),[0,1]);self.assertEqual(a[1].tolist(),[1.,2.]);self.assertEqual(a[1].tolist(),b[1].tolist())

    def test_equal_depth_shared_edge_hits_group_without_inventing_sheets(self):
        r=subject.depth_groups([1.,1.0000001,1.02],.000001)
        self.assertEqual(r,[[0,1],[2]])

    def test_unsupported_and_nonmanifold_role_remains_uncertain(self):
        self.assertEqual(subject.role(visible_views=0,branch=False,skin=True,cavity=False,rim=False),0)
        self.assertEqual(subject.role(visible_views=3,branch=True,skin=False,cavity=True,rim=False),0)
        self.assertEqual(subject.role(visible_views=2,branch=False,skin=True,cavity=False,rim=False),1)
        self.assertEqual(subject.role(visible_views=1,branch=False,skin=False,cavity=True,rim=False),3)

    def test_conflicting_semantic_candidates_remain_uncertain(self):
        self.assertEqual(subject.role(visible_views=3,branch=False,skin=True,cavity=True,rim=False),0)
        self.assertEqual(subject.role(visible_views=1,branch=False,skin=False,cavity=False,rim=True),2)

    def test_candidate_path_requires_rim_and_avoids_branch_barrier(self):
        graph=[[1],[0,2],[1,3],[2]];points=[[0,0,0],[1,0,0],[2,0,0],[3,0,0]]
        self.assertEqual(subject.shortest_rim_path(graph,points,[0],{3},{1},[True]*4)['path'],[0,1,2,3])
        self.assertIsNone(subject.shortest_rim_path(graph,points,[0],{3},{1},[True,False,True,True]))
        self.assertIsNone(subject.shortest_rim_path(graph,points,[0],{3},set(),[True]*4))

    def test_visual_wound_constraint_rejects_below_wound_torso(self):
        self.assertTrue(subject.inside_polygon((240,160),subject.WOUND_OUTLINE))
        self.assertFalse(subject.inside_polygon((300,345),subject.WOUND_OUTLINE))
        self.assertFalse(subject.inside_polygon((380,225),subject.WOUND_OUTLINE))

    def test_baseline_is_original_orphan_not_winding_variant(self):
        self.assertEqual(subject.BASELINE_SHA,'377a15dc4b5a27936f7e62b311dd03e8337685344b0f8204b8f831252176b1a5')
        self.assertEqual(subject.OLD.name,'amd-zombie-orphan-cleanup-20261025-v1')

    def test_owner_blocks_low_resource_before_spawn(self):
        module=subject.owner('authenticate')
        with patch.object(module,'resource',return_value={'admitted':False}),patch.object(module.subprocess,'Popen',side_effect=AssertionError('Spawn')):
            with self.assertRaisesRegex(RuntimeError,'admission'):module.supervise('authenticate')

    def test_blender_gate_precedes_heavy_imports(self):
        original=builtins.__import__
        def guard(name,*a,**kw):
            if name.split('.')[0] in {'bpy','numpy','mathutils'}:raise AssertionError('Heavy import')
            return original(name,*a,**kw)
        with patch.object(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(b'NO\n'))),patch.object(builtins,'__import__',side_effect=guard):
            with self.assertRaisesRegex(RuntimeError,'GO'):runpy.run_path(str(PATH.with_name('amd_zombie_wound_source_blender.py')),run_name='__main__')

    def test_audit_has_no_geometry_write_or_repair_calls(self):
        import ast
        forbidden={'write_ply','voxel_remesh','quadriflow_remesh','modifier_apply','reverse_rows','chart','compact','save_as_mainfile'}
        for p in (PATH,PATH.with_name('amd_zombie_wound_source_blender.py')):
            for node in ast.walk(ast.parse(p.read_text())):
                if not isinstance(node,ast.Call):continue
                name=node.func.attr if isinstance(node.func,ast.Attribute) else (node.func.id if isinstance(node.func,ast.Name) else '')
                self.assertNotIn(name,forbidden)
                if name=='new' and isinstance(node.func.value,ast.Attribute):self.assertNotEqual(node.func.value.attr,'modifiers')
if __name__=='__main__':unittest.main()
