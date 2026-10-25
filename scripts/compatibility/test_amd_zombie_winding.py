"""CPU fixtures: exact authorized corner permutations and orientation constraints."""
import builtins
import importlib.util
import io
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).parent),'D:/AI/Trellis2AMD/venv/Lib/site-packages']
PATH=Path(__file__).with_name('amd_zombie_winding.py')
subject=None
if PATH.exists():
    spec=importlib.util.spec_from_file_location('winding',PATH);subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)

class Exists(unittest.TestCase):
    def test_implementation_exists(self):self.assertIsNotNone(subject)

@unittest.skipIf(subject is None,'Not implemented')
class Winding(unittest.TestCase):
    def fixture(self):
        import numpy as np
        return np,np.array([[-0.,0,0],[1,0,0],[1,1,0],[0,1,0]],np.float32),np.array([[0,1,2],[0,2,3],[0,1,3]],np.int32)

    def test_only_selected_rows_reversed_and_original_immutable(self):
        np,v,f=self.fixture();old=f.tobytes();g=subject.reverse_rows(np,f,[0,1])
        self.assertEqual(g.tolist(),[[0,2,1],[0,3,2],[0,1,3]])
        subject.validate_edit(np,v,f,v,g,[0,1]);self.assertEqual(f.tobytes(),old)

    def test_outside_face_or_vertex_change_rejected(self):
        np,v,f=self.fixture();g=subject.reverse_rows(np,f,[0,1]);g[2]=g[2,[0,2,1]]
        with self.assertRaises(ValueError):subject.validate_edit(np,v,f,v,g,[0,1])
        g=subject.reverse_rows(np,f,[0,1]);altered=v.copy();altered[0,0]=0.
        with self.assertRaises(ValueError):subject.validate_edit(np,v,f,altered,g,[0,1])

    def test_duplicate_out_of_range_or_empty_rows_rejected(self):
        np,v,f=self.fixture()
        for rows in ([],[0,0],[-1],[3]):
            with self.assertRaises(ValueError):subject.reverse_rows(np,f,rows)

    def test_odd_parity_cycle_produces_falsifiable_witness(self):
        r=subject.solve_constraints([10,20,30],[(10,20,1,1),(20,30,1,2),(30,10,1,3)],[])
        c=r['components'][0];self.assertFalse(c['internally_consistent'])
        self.assertEqual(c['contradictory_cycles'][0]['xor'],1)
        self.assertEqual(set(c['contradictory_cycles'][0]['edge_ids']),{1,2,3})
        self.assertEqual(c['safe_fixed_exterior_flip_face_ids'],[])

    def test_even_cycle_and_fixed_exterior_solved_coherently(self):
        r=subject.solve_constraints([0,1,2,3],[(0,1,1,0),(1,2,1,1),(2,3,1,2),(3,0,1,3)],[(0,0,4),(2,0,5)])
        c=r['components'][0];self.assertTrue(c['internally_consistent']);self.assertTrue(c['fixed_exterior_consistent'])
        self.assertEqual(c['safe_fixed_exterior_flip_face_ids'],[1,3])
        self.assertEqual(c['internal_conflicts_before'],4)

    def test_fixed_exterior_conflict_distinct_from_nonorientable_cycle(self):
        r=subject.solve_constraints([0,1],[(0,1,1,0)],[(0,0,1),(1,0,2)])
        c=r['components'][0];self.assertTrue(c['internally_consistent']);self.assertFalse(c['fixed_exterior_consistent'])
        self.assertTrue(c['fixed_boundary_contradictions']);self.assertEqual(c['safe_fixed_exterior_flip_face_ids'],[])

    def test_disconnected_islands_and_minimal_free_gauge(self):
        r=subject.solve_constraints([0,1,2,3],[(0,1,1,0),(1,2,0,1)],[])
        self.assertEqual(len(r['components']),2)
        self.assertEqual(r['components'][0]['minimum_free_flip_face_ids'],[0])

    def test_invalid_constraints_rejected(self):
        for edges,fixed in (([(0,2,1,0)],[]),([(0,1,2,0)],[]),([],[(2,0,1)])):
            with self.assertRaises(ValueError):subject.solve_constraints([0,1],edges,fixed)

    def test_partial_no_new_solution_matches_exhaustive_oracle(self):
        import itertools
        ids=[0,1,2,3];edges=[(0,1,1,0),(1,2,1,1),(2,3,1,2)];fixed=[(0,0,3),(3,0,4)]
        r=subject.no_new_optimum(ids,edges,fixed)
        best=999
        for x in itertools.product((0,1),repeat=4):
            if any(p==0 and x[a] for a,p,e in fixed):continue
            if any(p==0 and x[a]^x[b] for a,b,p,e in edges):continue
            residual=sum(x[a]^x[b]^p for a,b,p,e in edges)+sum(x[a]^p for a,p,e in fixed)
            best=min(best,residual)
        self.assertEqual(r['minimum_residual_conflicts'],best);self.assertEqual(best,1)
        self.assertEqual(r['conflicts_removed'],2);self.assertEqual(r['new_conflicts'],0)

    def test_no_new_optimum_preserves_good_edges_and_rejects_odd_cycles(self):
        r=subject.no_new_optimum([0,1,2],[(0,1,0,0),(1,2,1,1)],[(0,0,2)])
        self.assertEqual(r['flip_face_ids'],[2]);self.assertEqual(r['minimum_residual_conflicts'],0)
        with self.assertRaisesRegex(ValueError,'consistent'):
            subject.no_new_optimum([0,1,2],[(0,1,1,0),(1,2,1,1),(2,0,1,2)],[])

    def test_cut_certificate_matches_small_exhaustive_orientable_graphs(self):
        import itertools,random
        rng=random.Random(57)
        for trial in range(12):
            ids=list(range(6));labels=[rng.randrange(2) for _ in ids]
            edges=[(a,b,labels[a]^labels[b],i) for i,(a,b) in enumerate(itertools.combinations(ids,2)) if rng.random()<.5]
            fixed=[(a,rng.randrange(2),100+a) for a in ids if rng.random()<.6]
            best=999
            for x in itertools.product((0,1),repeat=len(ids)):
                states=[(p,x[a]^x[b]^p) for a,b,p,e in edges]+[(p,x[a]^p) for a,p,e in fixed]
                if any(p==0 and q==1 for p,q in states):continue
                best=min(best,sum(q for p,q in states))
            result=subject.no_new_optimum(ids,edges,fixed)
            self.assertEqual(result['minimum_residual_conflicts'],best)
            self.assertEqual(result['max_flow'],result['cut_capacity'])

    def test_owner_refuses_low_resources_without_spawn(self):
        route=subject.owner('edit')
        with patch.object(route,'resource',return_value={'admitted':False}),patch.object(route.subprocess,'Popen',side_effect=AssertionError('Unexpected spawn')):
            with self.assertRaisesRegex(RuntimeError,'admission'):route.supervise('edit')

    def test_renderer_gate_precedes_blender_imports(self):
        original=builtins.__import__
        def guard(name,*a,**kw):
            if name.split('.')[0] in {'bpy','numpy','mathutils'}:raise AssertionError('Early import')
            return original(name,*a,**kw)
        with patch.object(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(b'NO\n'))),patch.object(builtins,'__import__',side_effect=guard):
            with self.assertRaisesRegex(RuntimeError,'GO'):runpy.run_path(str(PATH.with_name('amd_zombie_winding_render.py')),run_name='__main__')

if __name__=='__main__':unittest.main()
