"""Exact, synthetic geometry controls; no original assets or native imports."""
import itertools,sys,unittest
from fractions import Fraction as Q
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_brick_four_edge_forensics as f

class ExactRelations(unittest.TestCase):
    def test_receipt_keys_use_portable_separators_on_windows(self):
        self.assertEqual(f.evidence_key(Path('prepare'),'vertices.npy'),'prepare/vertices.npy')
    def relation(self,a,b):
        self.assertTrue(hasattr(f,'triangle_relation'),'Exact local intersection predicate missing')
        before=(tuple(a),tuple(b));r=f.triangle_relation(a,b);self.assertEqual((tuple(a),tuple(b)),before)
        self.assertEqual(f.triangle_relation(b,a)['kind'],r['kind']);return r
    def test_non_coplanar_shared_edge_has_no_interior_overlap(self):
        self.assertEqual(self.relation([(0,0,0),(4,0,0),(0,4,0)],[(0,0,0),(4,0,0),(0,4,1)])['kind'],'shared_edge_only')
    def test_coplanar_same_side_overlap_is_not_an_ordinary_adjacency(self):
        self.assertEqual(self.relation([(0,0,0),(4,0,0),(0,4,0)],[(0,0,0),(4,0,0),(1,2,0)])['kind'],'coplanar_positive_area_overlap')
    def test_coplanar_opposite_sides_only_touch_the_shared_edge(self):
        self.assertEqual(self.relation([(0,0,0),(4,0,0),(0,4,0)],[(0,0,0),(4,0,0),(0,-4,0)])['kind'],'shared_edge_only')
    def test_reversed_geometry_duplicate_is_not_a_separate_sheet(self):
        t=[(0,0,0),(4,0,0),(0,4,0)]
        self.assertEqual(self.relation(t,list(reversed(t)))['kind'],'coincident_triangle')
    def test_disjoint_and_nonshared_point_contact(self):
        a=[(0,0,0),(4,0,0),(0,4,0)]
        self.assertEqual(self.relation(a,[(5,0,0),(6,0,0),(5,1,0)])['kind'],'disjoint')
        self.assertEqual(self.relation(a,[(0,0,1),(1,0,1),(0,1,1)])['kind'],'disjoint')
        self.assertEqual(self.relation(a,[(1,1,0),(1,1,2),(2,1,2)])['kind'],'unshared_point_contact')
    def test_crossing_interior_segments_are_detected(self):
        self.assertEqual(self.relation([(-2,-2,0),(2,-2,0),(0,2,0)],[(0,-1,-1),(0,-1,1),(0,1,1)])['kind'],'interior_intersection_segment')
    def test_a_shared_vertex_does_not_hide_an_extended_intersection(self):
        self.assertEqual(self.relation([(0,0,0),(4,0,0),(0,4,0)],[(0,0,0),(3,1,-1),(1,3,1)])['kind'],'interior_intersection_segment')
    def test_edge_through_face_is_distinguished_from_shared_vertex_only(self):
        a=[(0,0,0),(4,0,0),(0,4,0)]
        self.assertEqual(self.relation(a,[(0,0,0),(1,1,0),(1,1,1)])['kind'],'edge_face_contact_segment')
        self.assertEqual(self.relation(a,[(0,0,0),(0,0,1),(1,1,1)])['kind'],'shared_vertex_only')
    def test_dyadic_scale_and_large_translation_keep_classification(self):
        a=[(0,0,0),(4,0,0),(0,4,0)];b=[(0,0,0),(4,0,0),(0,4,1)]
        scale=2**35;shift=2**90
        transformed=lambda t:[tuple(x*scale+shift for x in p) for p in t]
        self.assertEqual(self.relation(transformed(a),transformed(b))['kind'],'shared_edge_only')
    def test_zero_area_input_stops_instead_of_false_clean_result(self):
        self.assertTrue(hasattr(f,'triangle_relation'))
        with self.assertRaises(ValueError):f.triangle_relation([(0,0,0),(1,0,0),(2,0,0)],[(0,0,0),(1,1,0),(1,0,0)])
    def test_corner_order_and_winding_do_not_select_a_sheet(self):
        a=[(0,0,0),(4,0,0),(0,4,0)]
        for b,kind in [([(0,0,0),(4,0,0),(1,2,0)],'coplanar_positive_area_overlap'),
                       ([(0,0,0),(3,1,-1),(1,3,1)],'interior_intersection_segment')]:
            for pa in itertools.permutations(a):
                for pb in itertools.permutations(b):self.assertEqual(self.relation(pa,pb)['kind'],kind)
    def test_coplanar_containment_and_exact_subpixel_separation(self):
        a=[(0,0,0),(4,0,0),(0,4,0)]
        self.assertEqual(self.relation(a,[(1,1,0),(2,1,0),(1,2,0)])['kind'],'coplanar_positive_area_overlap')
        b=[(0,0,Q(1,2**100)),(4,0,Q(1,2**100)),(0,4,Q(1,2**100))]
        self.assertEqual(self.relation(a,b)['kind'],'disjoint')

if __name__=='__main__':unittest.main()
