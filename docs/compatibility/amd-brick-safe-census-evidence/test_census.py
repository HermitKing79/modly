"""Small synthetic census only; no original arrays or mesh mutation."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest,importlib.util
import numpy as np
import census_core as core

def grid(size=12):
    v=np.array([(x,y,0) for y in range(size) for x in range(size)],np.float32)
    f=[]
    for y in range(size-1):
        for x in range(size-1):
            a=y*size+x;f.extend(((a,a+1,a+size+1),(a,a+size+1,a+size)))
    return v,np.array(f,np.int32)

def census(v,f):
    keys,inc=np.unique(core.keys_for(f,len(v)),return_counts=True)
    order,off,used,deg,consistent,conflict=core.occurrences(f,keys,inc,len(v),lambda:None)
    sheets,mf,ms,stats=core.fan_and_sheets(f,order,off,consistent,used,lambda:None)
    masks=core.protection(v,f,keys,inc,order,off,used,mf,ms,conflict,lambda:None)
    eligible,link,_=core.full_links(f,keys,order,off,consistent,masks['protected_faces'],masks['locked_vertices'],len(v),lambda:None)
    return dict(keys=keys,inc=inc,order=order,offsets=off,used=used,degree=deg,consistent=consistent,conflict=conflict,sheets=sheets,multi_fan=mf,multi_sheet=ms,stats=stats,masks=masks,eligible=eligible,link=link)

class CensusTests(unittest.TestCase):
    def test_free_grid_and_pinned_local_contract_equivalence(self):
        v,f=grid();r=census(v,f);self.assertGreater(len(r['eligible']),0)
        old=Path(__file__).resolve().parents[1]/'amd-brick-junction-protected-evidence/patch_contract.py'
        import hashlib
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(),'336e49bd7836f4a66c02ade831d6b37b66f290cb42e3aa30c72e7a009599d60c')
        s=importlib.util.spec_from_file_location('pinned_local_contract',old);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        summary,a=m.classify(v,f,np.arange(len(v),dtype=np.int32),f,[],r['keys'],r['inc'],len(v))
        np.testing.assert_array_equal(r['masks']['protected_faces'],a['protected_faces'])
        np.testing.assert_array_equal(r['masks']['locked_vertices'],a['vertex_locks'])
        np.testing.assert_array_equal(r['keys'][r['eligible']],a['edge_keys'][a['eligible_edge_rows']])
        self.assertEqual(summary['strict_eligible_edges'],r['link']['certified_edges'])
        self.assertFalse(r['masks']['locked_vertices'][6*12+6]);self.assertTrue(r['masks']['locked_vertices'][0])
    def test_tetrahedron_full_link_edge_rejection(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
        f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int32);r=census(v,f)
        self.assertEqual(r['link']['pre_link_candidate_edges'],6)
        self.assertEqual(r['link']['rejected_common_link_edge'],6);self.assertEqual(len(r['eligible']),0)
    def test_complete_four_face_junction_is_frozen(self):
        v=np.array([[0,0,0],[0,0,1],[1,0,.5],[0,1,.5],[-1,0,.5],[0,-1,.5]],np.float32)
        f=np.array([[0,1,2],[1,0,3],[0,1,4],[1,0,5]],np.int32);r=census(v,f)
        self.assertEqual(int((r['inc']>2).sum()),1);self.assertEqual(r['masks']['nonmanifold_vertices'].sum(),2)
        self.assertTrue(r['masks']['protected_faces'].all());self.assertEqual(len(r['eligible']),0)
    def test_winding_conflict_preserves_both_source_faces(self):
        v,f=grid();f[80]=f[80,[0,2,1]];r=census(v,f)
        self.assertEqual(len(r['conflict']),3);self.assertTrue(r['masks']['protected_faces'][80])
        self.assertTrue(r['masks']['protected_faces'][r['order'][r['offsets'][r['conflict']]]//3].all())
        self.assertTrue(r['masks']['protected_faces'][r['order'][r['offsets'][r['conflict']]+1]//3].all())
    def test_vertex_fan_pinch_despite_one_global_sheet(self):
        v=np.array([[0,0,0],[1,0,0],[1,1,0],[-1,0,0],[-1,-1,0],[2,1,1],[-2,1,1]],np.float32)
        f=np.array([[0,1,2],[0,3,4],[2,1,5],[5,1,6],[5,6,3],[3,6,4]],np.int32);r=census(v,f)
        self.assertEqual(r['stats']['oriented_sheet_components'],1);self.assertTrue(r['multi_fan'][0]);self.assertFalse(r['multi_sheet'][0])
        self.assertTrue(r['masks']['protected_vertices'][0])
    def test_unused_vertices_not_phantom_multiple_fans(self):
        v,f=grid();v=np.concatenate((v,[[999,999,999]])).astype(np.float32);r=census(v,f)
        self.assertFalse(r['used'][-1]);self.assertFalse(r['multi_fan'][-1]);self.assertFalse(r['multi_sheet'][-1])
        self.assertEqual(r['degree'][-1],0)
    def test_signed_zero_alias_interfaces_are_locked(self):
        v,f=grid();v=np.concatenate((v,[[-0.,0,0],[1,-1,0],[0,-1,0]])).astype(np.float32)
        f=np.concatenate((f,[[len(v)-3,len(v)-2,len(v)-1]])).astype(np.int32);r=census(v,f)
        self.assertTrue(r['masks']['alias_vertices'][0]);self.assertTrue(r['masks']['alias_vertices'][-3])
        self.assertTrue(r['masks']['locked_vertices'][0]);self.assertTrue(r['masks']['locked_vertices'][-3])
    def test_cached_incidence_and_foreign_keys_fail_closed(self):
        v,f=grid(4);keys,inc=np.unique(core.keys_for(f,len(v)),return_counts=True)
        with self.assertRaises(ValueError):core.occurrences(f,keys,inc+1,len(v),lambda:None)
        bad=keys.copy();bad[0]+=1
        with self.assertRaises(ValueError):core.occurrences(f,bad,inc,len(v),lambda:None)
        badf=f.copy();badf[0,0]=len(v)
        with self.assertRaises(ValueError):core.occurrences(badf,keys,inc,len(v),lambda:None)
    def test_work_and_cancellation_fail_closed(self):
        v,f=grid();keys,inc=np.unique(core.keys_for(f,len(v)),return_counts=True)
        def stop():raise TimeoutError('synthetic cancellation')
        with self.assertRaises(TimeoutError):core.occurrences(f,keys,inc,len(v),stop)
        old=core.MAX_LINK_COMPARISONS;core.MAX_LINK_COMPARISONS=0
        try:
            with self.assertRaisesRegex(ValueError,'work exceeds'):census(v,f)
        finally:core.MAX_LINK_COMPARISONS=old
    def test_staged_distribution_denominators_and_source_components(self):
        v,f=grid();r=census(v,f);labels=np.zeros(len(v),np.int32)
        d,regions=core.distributions(v,f,labels,r['masks'],r['keys'],r['eligible'],lambda:None)
        self.assertEqual(d['original_face_bearing_vertex_components'],1)
        self.assertEqual(sum(x['original_faces'] for x in d['by_octant'].values()),len(f))
        self.assertEqual(sum(x['certified_edges'] for x in d['by_octant'].values()),len(r['eligible']))
        self.assertEqual(sum(x['locked_vertices'] for x in d['by_octant'].values()),int(r['masks']['locked_vertices'].sum()))
        self.assertEqual(len(regions),len(f))

if __name__=='__main__':unittest.main(verbosity=2)
