import sys
sys.dont_write_bytecode=True
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).parent))
import amd_cgal_proof as p
sys.path.insert(0,str(p.CPU_SITE))
import numpy as np
import amd_cgal_ai_input as ai
from amd_cgal_official_check import mesh_counts


class NumericalAndTopology(unittest.TestCase):
    def test_plane_inverse_transform_preserves_signed_distance(self):
        n=np.array([.3,.4,np.sqrt(.75)]);center=np.array([12.,-3.,.5]);scale=51.7
        offset=.731;point=np.array([.2,.17,-.4]);native=(point-center)*scale
        original=ai.original_plane(np,n,offset,center,scale)
        self.assertAlmostEqual((native@n-offset)/scale,point@n-original,places=13)

    def test_strict_distance_and_unsigned_normals(self):
        points=np.array([[0.,0,0],[.1,0,.0020001]])
        normals=np.array([[0.,0,1],[0.,0,-1]])
        report,_,_=ai.evaluate(np,points,normals,[dict(id=0,normal=[0.,0,1],offset=0.)],1.)
        self.assertEqual(report['covered'],1)
        self.assertFalse(report['gate_pass'])
        self.assertTrue(report['infinite_plane_distance_is_not_surface_distance'])

    def test_vertex_links_and_open_surface_are_distinguished(self):
        vertices=[(0.,0,0),(1.,0,0),(0.,2,0),(0.,0,3)]
        faces=[[0,2,1],[0,1,3],[1,2,3],[2,0,3]]
        closed=mesh_counts(vertices,faces)
        self.assertEqual(closed['boundary_edges'],0);self.assertEqual(closed['nonmanifold_vertex_links'],0)
        self.assertEqual(mesh_counts(vertices,faces[:-1])['boundary_edges'],3)
        duplicate=mesh_counts(vertices,faces+[faces[0]])
        self.assertEqual(duplicate['nonmanifold_edges'],3)

    def test_frozen_identity_and_budget(self):
        self.assertEqual(ai.frozen()['candidate_budget']['maximum_triangles'],1000)


if __name__=='__main__':unittest.main(verbosity=2)
