"""Small CPU fixtures for the bounded retopology feasibility experiment."""
import builtins,importlib.util,io,runpy,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).parent),'D:/AI/Trellis2AMD/venv/Lib/site-packages']
PATH=Path(__file__).with_name('amd_zombie_retopology.py');subject=None
if PATH.exists():
    spec=importlib.util.spec_from_file_location('retopo',PATH);subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)
class Exists(unittest.TestCase):
    def test_implementation_exists(self):self.assertIsNotNone(subject)
@unittest.skipIf(subject is None,'Not implemented')
class Geometry(unittest.TestCase):
    def test_crop_distinguishes_global_openings_from_artificial_cuts(self):
        import numpy as np
        whole=np.array([[0,1,2],[0,2,3]],np.int32)
        r=subject.crop_boundary(np,whole,whole[:1],4)
        self.assertEqual(r['genuine_source_boundary_edges'],2);self.assertEqual(r['artificial_crop_boundary_edges'],1)

    def plane(self):
        import numpy as np
        x,z=np.meshgrid(np.arange(5)*.25,np.arange(5)*.25)
        return np,np.stack((x,np.zeros_like(x),z),axis=2).astype(np.float32)

    def test_chart_normals_follow_known_exterior_not_source_winding(self):
        np,points=self.plane();before=points.tobytes();valid=np.ones((5,5),bool)
        for sign in (-1,1):
            v,f,receipt=subject.chart(np,points,valid,sign,1.)
            self.assertEqual(len(f),8);self.assertEqual(receipt['accepted_cells'],4)
            p=v[f];normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);self.assertTrue(np.all(normal[:,1]*sign>0))
        self.assertEqual(points.tobytes(),before)

    def test_missing_dense_witness_prevents_bridging_opening(self):
        np,points=self.plane();valid=np.ones((5,5),bool);valid[1,1]=False
        v,f,r=subject.chart(np,points,valid,1,1.)
        self.assertEqual(r['missing_witness_cells'],1);self.assertEqual(len(f),6)

    def test_depth_jump_prevents_cross_sheet_bridge(self):
        np,points=self.plane();points[:,2:,1]=10.
        v,f,r=subject.chart(np,points,np.ones((5,5),bool),1,1.)
        self.assertGreater(r['depth_discontinuity_cells'],0);self.assertEqual(len(f),4)

    def test_diagonal_touching_chart_cells_get_separate_vertex_fans(self):
        import numpy as np
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],np.float32)
        f=np.array([[0,1,2],[0,3,4]],np.int32);vv,ff,n=subject.split_fans(np,v,f)
        self.assertEqual(n,1);self.assertNotEqual(ff[0,0],ff[1,0])
        self.assertEqual(v[f].tobytes(),vv[ff].tobytes())

    def test_invalid_chart_dimensions_rejected(self):
        np,points=self.plane()
        with self.assertRaises(ValueError):subject.chart(np,points[:4],np.ones((4,5),bool),1,1.)

    def test_depth_quantile_uses_recorded_float64_interpolation(self):
        import numpy as np
        a=np.array([2.,2.000123456,2.003729],np.float32);b=np.full_like(a,2.)
        expected=float(np.quantile(abs(a-b).astype(np.float64),.95))
        self.assertEqual(subject.depth_p95(np,a,b,np.ones(3,bool)),expected)

    def test_orphan_baseline_is_pinned_not_winding_experiment(self):
        self.assertEqual(subject.BASELINE_SHA,'377a15dc4b5a27936f7e62b311dd03e8337685344b0f8204b8f831252176b1a5')
        self.assertEqual(subject.OLD.name,'amd-zombie-orphan-cleanup-20261025-v1')

    def test_owner_refuses_low_resource_before_spawn(self):
        module=subject.owner('prepare')
        with patch.object(module,'resource',return_value={'admitted':False}),patch.object(module.subprocess,'Popen',side_effect=AssertionError('Spawn')):
            with self.assertRaisesRegex(RuntimeError,'admission'):module.supervise('prepare')

    def test_blender_gate_precedes_heavy_imports(self):
        original=builtins.__import__
        def guard(name,*a,**kw):
            if name.split('.')[0] in {'bpy','numpy','mathutils'}:raise AssertionError('Heavy import')
            return original(name,*a,**kw)
        with patch.object(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(b'NO\n'))),patch.object(builtins,'__import__',side_effect=guard):
            with self.assertRaisesRegex(RuntimeError,'GO'):runpy.run_path(str(PATH.with_name('amd_zombie_retopology_blender.py')),run_name='__main__')
if __name__=='__main__':unittest.main()
