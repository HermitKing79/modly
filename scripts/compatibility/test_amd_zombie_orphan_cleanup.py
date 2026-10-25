"""Small CPU fixtures for exact compaction and symbolic winding analysis."""
import importlib.util
import builtins
import io
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).parent),'D:/AI/Trellis2AMD/venv/Lib/site-packages']
PATH=Path(__file__).with_name('amd_zombie_orphan_cleanup.py')
subject=None
if PATH.exists():
    spec=importlib.util.spec_from_file_location('cleanup',PATH);subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)


class Exists(unittest.TestCase):
    def test_copy_only_cleanup_exists(self):self.assertIsNotNone(subject,'Copy-only cleanup implementation required')


@unittest.skipIf(subject is None,'Not implemented')
class Geometry(unittest.TestCase):
    def fixture(self):
        import numpy as np
        v=np.array([[-0.,0,0],[.2,.2,0],[1,0,0],[0,1,0],[.3,.3,0],[1,1,0]],np.float32)
        f=np.array([[0,2,3],[3,2,5]],np.int32)
        return np,v,f

    def test_exact_ids_remapped_without_surface_or_order_change(self):
        np,v,f=self.fixture();before=v.tobytes(),f.tobytes()
        vv,ff,new_to_old,old_to_new=subject.compact(np,v,f,np.array([1,4],np.int32))
        self.assertEqual(new_to_old.tolist(),[0,2,3,5]);self.assertEqual(old_to_new.tolist(),[0,-1,1,2,-1,3])
        self.assertEqual(ff.tolist(),[[0,1,2],[2,1,3]])
        subject.surface_equal(np,v,f,vv,ff)
        self.assertEqual(before,(v.tobytes(),f.tobytes()))

    def test_wrong_or_referenced_removal_ids_are_rejected(self):
        np,v,f=self.fixture()
        for ids in ([1],[1,3],[1,4,4]):
            with self.assertRaises(ValueError):subject.compact(np,v,f,np.array(ids,np.int32))

    def test_triangle_reordering_winding_and_signed_zero_changes_are_rejected(self):
        np,v,f=self.fixture();vv,ff,_,_=subject.compact(np,v,f,np.array([1,4],np.int32))
        for bad in (ff[::-1].copy(),ff[:,::-1].copy()):
            with self.assertRaises(ValueError):subject.surface_equal(np,v,f,vv,bad)
        altered=vv.copy();altered[0,0]=np.float32(0.)
        with self.assertRaises(ValueError):subject.surface_equal(np,v,f,altered,ff)

    def test_symbolic_complete_quad_flip_resolves_four_conflicts_without_edit(self):
        import numpy as np
        f=np.array([[0,1,2],[0,2,3],[0,1,4],[1,2,5],[2,3,6],[3,0,7]],np.int32)
        before=f.tobytes();result=subject.winding_proposal(np,f,[0,1],8)
        self.assertTrue(result['all_touched_edges_two_face'])
        self.assertEqual(result['before_conflicts'],4);self.assertEqual(result['predicted_after_conflicts'],0)
        self.assertEqual(f.tobytes(),before)

    def test_three_face_edge_prevents_simple_manifold_winding_proposal(self):
        import numpy as np
        f=np.array([[0,1,2],[0,2,3],[0,1,4],[0,1,5]],np.int32)
        self.assertFalse(subject.winding_proposal(np,f,[0,1],6)['all_touched_edges_two_face'])

    def test_vertex_link_distinguishes_closed_fan_from_branched_fan(self):
        import numpy as np
        closed=np.array([[0,1,2],[0,2,3],[0,3,1]],np.int32)
        self.assertTrue(subject.vertex_fan(np,closed,0)['closed_single_fan'])
        branched=np.concatenate((closed,np.array([[0,1,4]],np.int32)))
        self.assertFalse(subject.vertex_fan(np,branched,0)['closed_single_fan'])

    def test_prediction_is_serialisable_without_constructing_repaired_mesh(self):
        import numpy as np,json
        f=np.array([[0,1,2],[0,2,3],[0,1,4],[0,1,5]],np.int32)
        row=subject.winding_proposal(np,f,[0,1],6)
        decoded=json.loads(json.dumps(row))
        self.assertIs(decoded['faces_reversed_or_repaired'],False)

    def test_owner_rejects_low_resources_before_spawn(self):
        route=subject.owner('cleanup')
        with patch.object(route,'resource',return_value={'admitted':False}),patch.object(route.subprocess,'Popen',side_effect=AssertionError('Unexpected spawn')):
            with self.assertRaisesRegex(RuntimeError,'admission'):route.supervise('cleanup')

    def test_render_requires_owned_gate_before_Blender_imports(self):
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in {'bpy','mathutils','numpy'}:raise AssertionError('Import before GO')
            return original(name,*args,**kwargs)
        with patch.object(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(b'NO\n'))),patch.object(builtins,'__import__',side_effect=guarded):
            with self.assertRaisesRegex(RuntimeError,'GO'):
                runpy.run_path(str(PATH.with_name('amd_zombie_orphan_render.py')),run_name='__main__')


if __name__=='__main__':unittest.main()
