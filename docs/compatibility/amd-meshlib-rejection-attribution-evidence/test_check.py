"""Independent link and record checks on synthetic values only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest
import copy
import json
try:
    from check import link_evidence,_same
except ModuleNotFoundError:
    link_evidence=_same=None

class IndependentChecks(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(link_evidence,'Independent checker is not implemented')
    def test_two_open_components(self):
        row=link_evidence([[0,1,2],[0,3,4]],0,[0,1])
        self.assertEqual((row['component_count'],row['open_chain_components']),(2,2))
    def test_ring(self):
        row=link_evidence([[0,1,2],[0,2,3],[0,3,1]],0,[0,1,2])
        self.assertEqual((row['component_count'],row['closed_chain_components'],row['maximum_link_degree']),(1,1,2))
    def test_branch_is_separate_from_raw_connectivity(self):
        row=link_evidence([[0,1,2],[0,1,3],[0,1,4]],0,[0,1,2])
        self.assertEqual((row['component_count'],row['branching_components'],row['maximum_link_degree']),(1,1,3))
    def test_parallel_edges_keep_multiplicity_and_direction(self):
        row=link_evidence([[0,1,2],[0,1,2]],0,[0,1])
        self.assertEqual(row['repeated_undirected_link_edges'],1)
        self.assertEqual(row['co_directed_link_edge_excess'],1)
        self.assertEqual(row['closed_chain_components'],1)
    def test_repeated_central_corner_is_explicitly_ambiguous(self):
        row=link_evidence([[0,0,2]],0,[0])
        self.assertEqual(row['ambiguous_repeated_central_vertex_faces'],[0])
        self.assertEqual(row['component_count'],0)
    def test_prefix_subset_is_not_full_star(self):
        f=[[0,1,2],[0,3,4]]
        self.assertEqual(link_evidence(f,0,[0])['component_count'],1)
        self.assertEqual(link_evidence(f,0,[0,1])['component_count'],2)
    def test_exact_record_rejects_unknown_fields_ID_types_and_zero_sign(self):
        for actual,expected in [({'id':True},{'id':1}),({'x':1,'extra':0},{'x':1}),([-0.0],[0.0])]:
            with self.assertRaises(ValueError):_same(actual,expected)
    def test_float64_values_match_without_rounding_or_tolerance(self):
        self.assertTrue(_same({'area':0.5,'ids':[1,2]},{'area':0.5,'ids':[1,2]}))
        with self.assertRaises(ValueError):_same({'area':0.5000000000000001},{'area':0.5})

class SyntheticHandoff(unittest.TestCase):
    def test_complete_producer_schema_and_mutated_records(self):
        import numpy as np
        from analyze import analyze_arrays
        from test_analyze import fixture
        from check import check_arrays
        cases=[fixture(),fixture(failed=(3,1)),
               fixture([[0,1,2],[0,1,3]],[[0,0,0],[1,0,0],[0,1,0],[2,0,0]]),
               fixture([[0,1,2],[0,3,4]],[[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]]),
               fixture([[0,1,2],[1,0,3],[4,5,6]],
                       [[-0.0,0,0],[1,.25,.125],[.5,1,.25],[0,-1,0],[3,0,0],[4,0,0],[3,1,0]],
                       failed=(2,),split=((0,1),))]
        for original,saved,failed in cases:
            public,private=analyze_arrays(np,original,saved,failed)
            public,private=json.loads(json.dumps([public,private],allow_nan=False))
            with self.subTest(failed=failed.tolist(),split=public['split_endpoint_pair_count']):
                self.assertEqual(check_arrays(np,original,saved,failed,public,private)['classification'],'PASS')
                corrupt=copy.deepcopy(private)
                corrupt['witnesses'][0]['source_face_id']+=1
                with self.assertRaises(ValueError):check_arrays(np,original,saved,failed,public,corrupt)
                corrupt=copy.deepcopy(private)
                corrupt['witnesses'][0]['vertex_stars'][0]['links']['original']['component_count']+=1
                with self.assertRaises(ValueError):check_arrays(np,original,saved,failed,public,corrupt)
                corrupt=copy.deepcopy(private);corrupt['context_faces'].pop()
                with self.assertRaises(ValueError):check_arrays(np,original,saved,failed,public,corrupt)
        original,saved,failed=cases[-1]
        public,private=analyze_arrays(np,original,saved,failed)
        corrupt=copy.deepcopy(public);corrupt['split_pairs'][0]['native_edge_count']+=1
        with self.assertRaises(ValueError):check_arrays(np,original,saved,failed,corrupt,private)

    def test_saved_native_reciprocity_is_rechecked(self):
        import numpy as np
        from analyze import analyze_arrays
        from test_analyze import fixture
        from check import check_arrays
        original,saved,failed=fixture()
        public,private=analyze_arrays(np,original,saved,failed)
        saved['native_right_faces']=saved['native_right_faces'].copy()
        at=np.argwhere(saved['native_right_faces']>=0)[0]
        saved['native_right_faces'][tuple(at)]=-1
        with self.assertRaises(ValueError):check_arrays(np,original,saved,failed,public,private)

if __name__=='__main__':unittest.main(verbosity=2)
