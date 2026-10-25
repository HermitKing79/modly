"""Independent small topology fixtures; no models, Blender or real analysis run."""
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
PATH=Path(__file__).with_name('amd_zombie_topology.py')
subject=None
if PATH.exists():
    spec=importlib.util.spec_from_file_location('topology',PATH);subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)


class Exists(unittest.TestCase):
    def test_readonly_topology_diagnostic_exists(self):self.assertIsNotNone(subject,'Topology diagnostic missing')


@unittest.skipIf(subject is None,'Not implemented')
class Geometry(unittest.TestCase):
    def test_three_faces_share_one_edge(self):
        import numpy as np
        edges,counts,inverse,winding=subject.incidence(np,np.array([[0,1,2],[1,0,3],[0,1,4]],np.int32))
        self.assertEqual(int((counts==1).sum()),6)
        self.assertEqual(int((counts>2).sum()),1)
        self.assertEqual(edges[counts>2].tolist(),[[0,1]])
        self.assertEqual(int(counts.max()),3)

    def test_closed_tetrahedron_has_no_boundary_or_excess_incidence(self):
        import numpy as np
        f=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],np.int32)
        edges,counts,inverse,winding=subject.incidence(np,f)
        self.assertEqual(len(edges),6);self.assertTrue(np.all(counts==2));self.assertTrue(np.all(winding==0))

    def test_split_diagonals_do_not_change_quad_perimeter_defects(self):
        import numpy as np
        q=np.array([[0,1,2,3],[1,0,4,5],[0,1,6,7]],np.int32)
        one=q[:,[0,1,2,0,2,3]].reshape(-1,3)
        two=q[:,[0,1,3,3,1,2]].reshape(-1,3)
        for f in (one,two):
            recovered=subject.recover_quads(np,f)
            self.assertTrue(np.array_equal(recovered,q))
            perimeter,counts=subject.quad_incidence(np,recovered)
            e,tc,_,_=subject.incidence(np,f)
            self.assertEqual(int((counts==1).sum()),9)
            self.assertEqual(int((counts>2).sum()),1)
            self.assertEqual(e[tc>2].tolist(),[[0,1]])
            self.assertEqual(perimeter[counts>2].tolist(),[[0,1]])

    def test_component_ids_ignore_unreferenced_vertices(self):
        import numpy as np
        f=np.array([[0,1,2],[2,1,3],[4,5,6]],np.int32)
        vl,fl=subject.component_labels(np,f,8)
        self.assertEqual(vl.tolist(),[0,0,0,0,1,1,1,-1])
        self.assertEqual(fl.tolist(),[0,0,1])

    def test_geometric_bins_are_exclusive_and_keep_hands_separate(self):
        import numpy as np
        p=np.array([[0,0,.48],[.46,0,.25],[0,0,.07],[0,0,-.49]],np.float32)
        self.assertEqual(subject.region_labels(np,p).tolist(),['crown-scalp','hands','abdomen','feet'])

    def test_direct_learned_quad_lookup_preserves_row_order_and_missing_corners(self):
        import numpy as np
        c=np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0]],np.int32)
        flags=np.zeros((4,3),bool);flags[0,2]=True;flags[3,2]=True
        quads,missing=subject.source_quads(np,c,flags)
        self.assertEqual(quads.tolist(),[[0,1,2,3]])
        self.assertEqual(missing,1)

    def test_private_owner_keeps_low_resource_rejection_before_spawn(self):
        original=subject.inherited.BASE
        route=subject.owner('analyze')
        self.assertEqual(subject.inherited.BASE,original)
        self.assertIs(route.supervise.__globals__,route.resource.__globals__)
        with patch.object(route,'resource',return_value={'admitted':False}),patch.object(route.subprocess,'Popen',side_effect=AssertionError('Spawn before admission')):
            with self.assertRaisesRegex(RuntimeError,'admission'):route.supervise('analyze')

    def test_render_gate_rejects_before_Blender_import(self):
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in {'bpy','mathutils','numpy'}:raise AssertionError('Import before GO')
            return original(name,*args,**kwargs)
        with patch.object(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(b'NO\n'))),patch.object(builtins,'__import__',side_effect=guarded):
            with self.assertRaisesRegex(RuntimeError,'GO'):
                runpy.run_path(str(PATH.with_name('amd_zombie_topology_render.py')),run_name='__main__')


if __name__=='__main__':unittest.main()
