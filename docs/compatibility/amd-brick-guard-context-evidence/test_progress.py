"""Tiny complete-ring and meaningful-screen contracts, no model arrays."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest,numpy as np
from progress_contract import build_contexts,one_ring,screen,select_first

def index(f,n):
    rows=sorted((min(int(t[j]),int(t[(j+1)%3]))*n+max(int(t[j]),int(t[(j+1)%3])),i*3+j) for i,t in enumerate(f) for j in range(3))
    k=np.array(sorted({x[0] for x in rows}),np.uint64);o=np.array([x[1] for x in rows],np.int32)
    off=np.r_[0,np.cumsum([sum(x[0]==q for x in rows) for q in k])].astype(np.int32)
    return k,o,off

class ProgressTests(unittest.TestCase):
    def test_first_crossings_and_largest_complete_ring(self):
        f=np.array([[i,i+1,i+2] for i in range(8)],np.int32);k,o,off=index(f,10)
        r=build_contexts(f,k,o,off,10,np.array([0],np.int32),np.array([0],np.int32),lambda:None,targets=(2,4,6),max_faces=7)
        self.assertEqual([(x['depth'],len(x['source_face_ids']),x['labels']) for x in r['snapshots']],[(4,2,['2']),(6,4,['4']),(8,6,['6']),(9,7,['max'])])
        self.assertEqual(r['terminal']['reason'],'exact_cap');self.assertEqual(len(r['ring_records']),7)
    def test_nonmanifold_group_is_complete_and_crossed_checkpoints_dedup(self):
        f=np.array([[0,1,2],[1,0,3],[0,1,4],[1,0,5]],np.int32);k,o,off=index(f,6)
        r=build_contexts(f,k,o,off,6,np.array([0],np.int32),np.array([0],np.int32),lambda:None,targets=(2,3,4),max_faces=4)
        self.assertEqual(len(r['snapshots']),1);self.assertEqual(r['snapshots'][0]['labels'],['2','3','4','max'])
        np.testing.assert_array_equal(r['snapshots'][0]['source_face_ids'],[0,1,2,3])
        r=build_contexts(f,k,o,off,6,np.array([0],np.int32),np.array([0],np.int32),lambda:None,targets=(2,3,4),max_faces=3)
        self.assertEqual(r['snapshots'],[]);self.assertEqual(r['terminal']['prospective_face_count'],4);self.assertEqual(r['baseline_labels'],['baseline','max'])
    def test_fixed_point_and_no_baseline_requalification(self):
        f=np.array([[0,1,2]],np.int32);k,o,off=index(f,3)
        r=build_contexts(f,k,o,off,3,np.array([0],np.int32),np.array([0],np.int32),lambda:None)
        self.assertEqual(r['snapshots'],[]);self.assertEqual(r['terminal']['reason'],'fixed_point')
    def test_bad_cache_work_cap_and_cancellation_reject(self):
        f=np.array([[0,1,2]],np.int32);k,o,off=index(f,3)
        with self.assertRaises(ValueError):one_ring(f,k+1,o,off,np.array([0],np.int32),3,lambda:None)
        with self.assertRaises(ValueError):build_contexts(f,k,o,off,3,np.array([0],np.int32),np.array([0],np.int32),lambda:None,max_faces=6001)
        def stop():raise TimeoutError('cancelled')
        with self.assertRaises(TimeoutError):one_ring(f,k,o,off,np.array([0],np.int32),3,stop)
    def test_screen_exact_equalities_each_failure_and_earliest(self):
        good=dict(initial_non_touching_opportunities=10,eligible_face_population=100,original_patch_faces=400,qualified=True)
        self.assertTrue(screen(good)['meaningful_pass'])
        for key,val in [('initial_non_touching_opportunities',9),('eligible_face_population',101),('original_patch_faces',401),('qualified',False)]:
            bad=dict(good);bad[key]=val;self.assertFalse(screen(bad)['meaningful_pass'])
        rows=[dict(depth=4,screening=screen(dict(good,qualified=False))),dict(depth=7,screening=screen(good)),dict(depth=9,screening=screen(dict(good,initial_non_touching_opportunities=20)))]
        self.assertEqual(select_first(rows),7)
        with self.assertRaises(ValueError):select_first(rows+rows[:1])

if __name__=='__main__':unittest.main(verbosity=2)
