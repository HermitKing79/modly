"""Synthetic exact-arithmetic applicability/refusal tests, without native code."""
import itertools
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from fractions import Fraction as Q
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import amd_patch_cell_audit as a


def square():
    planes=[(0,0,1,0),(1,0,0,0),(1,0,0,1),(0,1,0,0),(0,1,0,1)]
    points=[(0,0,0),(1,0,0),(1,1,0),(0,1,0)]
    return [tuple(map(Q,p)) for p in points],[tuple(map(Q,p)) for p in planes]


class ExactCellTests(unittest.TestCase):
    def test_complete_cell(self):
        points,planes=square();r=a.full_cell(points,0,planes)
        self.assertEqual(r['projected_area_twice'],2)
        self.assertEqual(set(r['edge_planes']),{1,2,3,4})

    def test_missing_interior_cut_refused(self):
        points,planes=square();planes.append((Q(1),Q(0),Q(0),Q(1,2)))
        with self.assertRaisesRegex(ValueError,'Missing interior'):a.full_cell(points,0,planes)

    def test_duplicate_degenerate_wrong_order(self):
        points,planes=square()
        for p in (points+[points[0]],points[:2],[points[i] for i in (0,2,1,3)]):
            with self.assertRaises(ValueError):a.full_cell(p,0,planes)

    def test_duplicate_or_coincident_plane_refused(self):
        points,planes=square()
        for plane in (planes[0],planes[1]):
            with self.assertRaises(ValueError):a.full_cell(points,0,planes+[plane])

    def test_missing_edge_plane_refused(self):
        points,planes=square()
        with self.assertRaisesRegex(ValueError,'unique plane'):a.full_cell(points,0,planes[:-1])

    def test_narrow_oblique_patch_has_no_size_or_axis_assumption(self):
        # 1e-6-wide tilted parallelogram; all coordinates are exact.
        e=Q(1,1000000)
        points=[(Q(0),Q(0),Q(0)),(e,Q(0),e),(e,Q(1),e),(Q(0),Q(1),Q(0))]
        planes=[(-Q(1),Q(0),Q(1),Q(0)),(1,0,0,0),(1,0,0,e),(0,1,0,0),(0,1,0,1)]
        self.assertGreater(a.full_cell(points,0,planes)['projected_area_twice'],0)

    def test_exact_triple_and_degenerate_intersection(self):
        p=a.intersection((Q(1),0,0,Q(1,3)),(0,Q(1),0,Q(2,7)),(0,0,Q(1),Q(1,11)))
        self.assertEqual(p,(Q(1,3),Q(2,7),Q(1,11)))
        with self.assertRaises(ValueError):a.intersection((1,0,0,0),(1,0,0,1),(0,0,1,0))

    def test_exact_triangle_distance(self):
        v=[(Q(0),Q(0),Q(0)),(Q(1),Q(0),Q(0)),(Q(0),Q(1),Q(0))]
        self.assertEqual(a.point_triangle_squared((Q(1,4),Q(1,4),Q(2)),*v),4)
        self.assertEqual(a.point_triangle_squared((Q(1),Q(1),Q(0)),*v),Q(1,2))
        self.assertEqual(a.point_triangle_squared((Q(-1),Q(0),Q(0)),*v),1)


class SubdivisionClosureTests(unittest.TestCase):
    def setUp(self):
        self.edges=[(3*y+x,3*y+x+1) for y in range(3) for x in range(2)]
        self.edges += [(3*y+x,3*(y+1)+x) for y in range(2) for x in range(3)]

    def test_all_binary_choices_closed_only_if_constant(self):
        # Independent enumeration of two-face edge incidence: zero or two.
        valid=[bits for bits in itertools.product((0,1),repeat=9)
               if all(bits[i]+bits[j] in (0,2) for i,j in self.edges)]
        self.assertEqual(valid,[(0,)*9,(1,)*9])

    def test_hole_concavity_disconnected_domains_and_extension(self):
        # Keep complement cells explicitly as zero: dropping them cannot hide
        # the new open boundary. Each connected full cell must then be zero.
        for forbidden in ([4],[4,5,7,8],[1,3,4,5,7],[8]):
            self.assertEqual(a.zero_by_internal_edges(9,self.edges,forbidden),list(range(9)))

    def test_all_supported_cell_not_rejected(self):
        self.assertEqual(a.zero_by_internal_edges(9,self.edges,[]),[])

    def test_two_conforming_cells_preserve_coarse_edge_constraint(self):
        # Two patches each split into a 2x2 grid. Every shared edge segment
        # independently requires zero/two incidence, hence coarse equality.
        internal=[(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7)]
        shared=[(1,4),(3,6)]
        valid=[bits for bits in itertools.product((0,1),repeat=8)
               if all(bits[i]+bits[j] in (0,2) for i,j in internal+shared)]
        self.assertEqual(valid,[(0,)*8,(1,)*8])

    def test_invalid_index_refused(self):
        for count,edges,forbidden in [(0,[],[]),(3,[(0,3)],[]),(3,[(1,1)],[]),(3,[],[-1]),(3,[],[3])]:
            with self.assertRaises(ValueError):a.zero_by_internal_edges(count,edges,forbidden)


class CoverageTests(unittest.TestCase):
    def test_complete_population_and_outward_bounds(self):
        data=struct.pack('<ddd',0,0,0)*32768+struct.pack('<ddd',3,0,0)*32768
        p=SimpleNamespace(stat=lambda:SimpleNamespace(st_size=len(data)),read_bytes=lambda:data)
        result=a.broad_coverage(p,[([Q(-1)]*3,[Q(1)]*3)])
        self.assertEqual(result['samples'],65536)
        self.assertEqual(result['violations_above_1_native_plus_1e_8'],32768)
        self.assertLessEqual(result['maximum_native'],2)

    def test_wrong_sample_count_refused(self):
        p=SimpleNamespace(stat=lambda:SimpleNamespace(st_size=0),read_bytes=lambda:b'')
        with self.assertRaises(ValueError):a.broad_coverage(p,[([Q(0)]*3,[Q(1)]*3)])


if __name__=='__main__':unittest.main()
