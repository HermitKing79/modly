import sys,unittest
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np

class Contract(unittest.TestCase):
    def module(self):
        import reduction_contract as m
        return m
    def test_positions_and_colour_transfer_use_original_source_ids(self):
        m=self.module();v=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],np.float32)
        f=np.array([[0,1,2],[1,3,2]],np.int32)
        ids,compact=m.compact(f,4)
        self.assertTrue(np.array_equal(ids,np.arange(4,dtype=np.int32)))
        self.assertTrue(np.array_equal(ids[compact],f))
        self.assertEqual(m.scan(v,f)['zero_area'],0)
    def test_nonfinite_invalid_indices_and_degenerate_triangles_reject(self):
        m=self.module();v=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
        for f in [np.array([[0,0,2]],np.int32),np.array([[0,1,3]],np.int32)]:
            with self.assertRaises(ValueError):m.scan(v,f)
        v[0,0]=np.nan
        with self.assertRaises(ValueError):m.scan(v,np.array([[0,1,2]],np.int32))
    def test_no_disconnected_component_bridging_or_silent_component_loss(self):
        m=self.module();labels=np.array([0,0,0,1,1,1],np.int32)
        good=np.array([[0,1,2],[3,4,5]],np.int32)
        self.assertTrue(m.component_lineage(good,labels,[0,1])['all_components_retained'])
        with self.assertRaises(ValueError):m.component_lineage(np.array([[0,1,3]],np.int32),labels,[0,1])
        self.assertFalse(m.component_lineage(good[:1],labels,[0,1])['all_components_retained'])
    def test_approximate_reported_meshopt_error_does_not_replace_quality_gates(self):
        m=self.module()
        self.assertEqual(m.TARGET_TRIANGLES,2086792)
        self.assertEqual(m.FLAGS,['LockBorder'])
        self.assertEqual(m.ERROR,.005)
        self.assertEqual(m.GATES['silhouette_IoU'],.98)
        self.assertEqual(m.GATES['RGB'],[.02,.05,.15])
        self.assertEqual(m.GATES['geometry_diagonal'],[.005,.01,.03])
    def test_oriented_keys_preserve_cyclic_order_and_distinguish_reversal(self):
        m=self.module();f=np.array([[0,1,2],[1,2,0],[0,2,1]],np.int32);k=m.oriented_keys(f,3)
        self.assertEqual(k[0],k[1]);self.assertNotEqual(k[0],k[2])
    def test_vertex_distance_is_conservative_and_removed_vertex_cannot_rescue(self):
        m=self.module();p=np.array([[.5,.5,0]],np.float32);v=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
        distance,_=m.upper_bounds(p,v)
        self.assertGreater(distance[0],0) # true surface distance is zero
        self.assertAlmostEqual(distance[0],np.sqrt(.5))
    def test_reverse_population_is_independent_area_stratified_and_deterministic(self):
        m=self.module();v=np.array([[0,0,0],[1,0,0],[0,1,0],[4,0,0]],np.float32)
        faces=np.array([[0,1,2],[0,3,2]],np.int32);areas=np.array([.5,2.],np.float64)
        a,b,p=m.reverse_samples(v,faces,areas,n=100)
        self.assertEqual(int((a==0).sum()),20)
        self.assertTrue((b>0).all() and (b<1).all());np.testing.assert_allclose(b.sum(1),1)
        for x,y in zip((a,b,p),m.reverse_samples(v,faces,areas,n=100)):np.testing.assert_array_equal(x,y)
        with self.assertRaises(ValueError):m.reverse_samples(v,faces,np.array([0.,1.]),100)

if __name__=='__main__':unittest.main()
