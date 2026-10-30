"""Independent tiny lattice fixtures. No original assets, WSL or GPU needed."""
import io
import itertools
import json
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import amd_closed_solid_support_census as c

KEYS=[((1,1,0),1),((1,1,1),1),((1,0,1),2),((1,1,1),2)]

def fixture(removed=(),values=(1,2,3,-.25)):
    coords=np.array([p for p in itertools.product(range(3),repeat=3) if p not in removed],np.int32)
    lookup={tuple(p):i for i,p in enumerate(coords.tolist())};features=np.zeros((len(coords),7),np.float32)
    for (anchor,axis),value in zip(KEYS,values):
        if anchor in lookup:features[lookup[anchor],3+axis]=value
    # Independent exhaustive oracle: scan anchors/axes, collect fully present flagged squares.
    offsets=(((0,0,0),(0,0,1),(0,1,1),(0,1,0)),((0,0,0),(1,0,0),(1,0,1),(0,0,1)),((0,0,0),(0,1,0),(1,1,0),(1,0,0)))
    q=[];index=np.full((len(coords),3),-1,np.int32)
    for anchor,row in lookup.items():
        for axis,off in enumerate(offsets):
            corners=[tuple(anchor[k]+p[k] for k in range(3)) for p in off]
            if features[row,3+axis]>0 and all(p in lookup for p in corners):
                index[row,axis]=len(q);q.append([lookup[p] for p in corners])
    return coords,features,lookup,index,np.array(q,np.int32).reshape(-1,4)

def records(**kw):
    coords,features,lookup,index,quads=fixture(**kw)
    edge=[lookup[(1,1,1)],lookup[(2,1,1)]]
    rows=c.candidates(edge,coords,features,lookup,index,quads)
    return {(tuple(r['anchor_grid']),r['axis']):r for r in rows}

class CensusTests(unittest.TestCase):
    def test_four_distinct_candidates_and_three_contributing_quads(self):
        r=records();self.assertEqual(set(r),set(KEYS));self.assertEqual(sum(x['contributes_to_edge'] for x in r.values()),3)
        self.assertEqual([x['state'] for x in r.values()].count('off_complete'),1)
        for x in r.values():self.assertTrue(x['all_corners_present']);self.assertEqual(len(x['corner_vertex_indices']),4)

    def test_absent_anchor_has_null_not_zero_logit(self):
        r=records(removed=((1,1,0),))[KEYS[0]]
        self.assertEqual(r['state'],'absent_anchor');self.assertFalse(r['anchor_exists'])
        self.assertIsNone(r['logit']);self.assertIsNone(r['logit_bits']);self.assertIsNone(r['flag_on'])

    def test_on_flag_with_missing_corner_is_not_emitted(self):
        r=records(removed=((2,1,0),))[KEYS[0]]
        self.assertEqual(r['state'],'flagged_incomplete');self.assertTrue(r['flag_on'])
        self.assertFalse(r['all_corners_present']);self.assertFalse(r['contributes_to_edge'])

    def test_off_incomplete_separate_from_off_complete(self):
        r=records(removed=((2,1,0),),values=(-.125,2,3,-.25))
        self.assertEqual(r[KEYS[0]]['state'],'off_incomplete');self.assertEqual(r[KEYS[3]]['state'],'off_complete')

    def test_original_F32_and_signed_zero_threshold(self):
        for value,bits in [(-0.,0x80000000),(0.,0),(np.nextafter(np.float32(0),np.float32(1)),1)]:
            r=records(values=(1,2,3,value))[KEYS[3]]
            self.assertEqual(r['logit_bits'],bits);self.assertEqual(r['flag_on'],bool(value>0))
            self.assertEqual(struct.unpack('<I',struct.pack('<f',r['logit']))[0],bits)

    def test_corner_mapping_and_actual_source_membership(self):
        r=records()[KEYS[0]]
        self.assertEqual(r['corner_grid'],[[1,1,0],[2,1,0],[2,1,1],[1,1,1]])
        self.assertTrue(r['contributes_to_edge']);self.assertIsNotNone(r['quad_row'])

    def test_nonunit_and_diagonal_edges_rejected(self):
        coords,f,lookup,index,q=fixture()
        for endpoint in ((2,1,1),(1,2,1),(0,1,1)):
            a=lookup[(0,1,1)];b=lookup[endpoint]
            with self.assertRaises(ValueError):c.candidates([a,b],coords,f,lookup,index,q)

    def test_false_source_quad_identity_rejected(self):
        coords,f,lookup,index,q=fixture();index[lookup[(1,1,0)],1]=-1
        with self.assertRaisesRegex(ValueError,'membership'):c.candidates([lookup[(1,1,1)],lookup[(2,1,1)]],coords,f,lookup,index,q)

    def test_addition_screen_rejects_transfer_and_new_junction(self):
        self.assertTrue(c.incidence_effect([1,1,1,1],False)['all_sides_become_two'])
        self.assertFalse(c.incidence_effect([1,1,1,0],False)['all_sides_become_two'])
        self.assertEqual(c.incidence_effect([1,1,1,0],False)['new_boundary_edges'],1)
        self.assertEqual(c.incidence_effect([1,1,1,2],False)['new_nonmanifold_edges'],1)

    def test_removal_screen_rejects_new_hole_or_erasure(self):
        self.assertTrue(c.incidence_effect([3,3,3,3],True)['all_sides_become_two'])
        a=c.incidence_effect([3,2,2,2],True);self.assertFalse(a['all_sides_become_two']);self.assertEqual(a['new_boundary_edges'],3)
        self.assertEqual(c.incidence_effect([3,1,2,2],True)['erased_edges'],1)

    def test_global_counts_lookup_handles_absent_edge_without_wrap(self):
        edges=np.array([[0,1],[1,2]],np.int32);counts=np.array([1,3],np.int32)
        self.assertEqual(c.perimeter_counts([0,1,2,3],edges,counts,4),[1,3,0,0])

    def test_logit_statistics_preserve_sign_and_count_unavailable(self):
        s=c.logit_stats([-1.,-0.,0.,.001,2.,None])
        self.assertEqual((s['negative'],s['zero'],s['positive'],s['unavailable']),(1,2,2,1))
        self.assertEqual(s['negative_zero'],1)
        self.assertEqual(json.loads(json.dumps(s,allow_nan=False))['negative_zero'],1)

    def test_duplicate_candidate_support_must_match(self):
        a=records()[KEYS[0]];b=dict(a);b['logit']=99
        with self.assertRaisesRegex(ValueError,'Duplicate'):c.deduplicate([a,b])
        self.assertEqual(len(c.deduplicate([a,a])),1)

    def test_inert_default_and_GO_required(self):
        with patch.object(c,'analyze',side_effect=AssertionError('No live work')):
            self.assertEqual(c.main([]),0)
            with patch.object(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(b'NO\n'))):
                with self.assertRaisesRegex(RuntimeError,'GO'):c.main(['--worker','analyze'])

    def test_owned_route_resource_limits_and_restricted_writes(self):
        import amd_zombie_cpu_mesh as owner
        import amd_closed_solid_benchmark as b
        with patch.object(owner,'supervise',return_value=0) as launch,patch.object(owner,'BASE',owner.BASE),patch.object(owner,'__file__',owner.__file__):
            self.assertEqual(c.main(['--run']),0);launch.assert_called_once_with('analyze')
            self.assertEqual(owner.BASE,c.BASE);self.assertEqual(owner.CAP,6*1024**3);self.assertEqual(owner.DEADLINE,180)
        self.assertFalse(owner.admission(12*1024**3-1,2*1024**3)['admitted'])
        guard=b.cpu_guard(c.BASE)
        for event,args in [('import',('torch',)),('subprocess.Popen',('wsl.exe',)),('os.remove',(str(c.PREVIOUS/'summary.json'),))]:
            with self.assertRaises(RuntimeError):guard(event,args)
        with self.assertRaises(ValueError):guard('open',(str(c.PREVIOUS/'evidence-manifest.json'),'w',os.O_WRONLY))

    def test_saved_completion_route_cannot_rerun_census(self):
        import amd_zombie_cpu_mesh as owner
        with patch.object(owner,'supervise',return_value=0) as launch,patch.object(owner,'plain') as path,patch.object(owner,'BASE',owner.BASE),patch.object(owner,'__file__',owner.__file__),patch.object(c,'analyze',side_effect=AssertionError('No census rerun')):
            self.assertEqual(c.main(['--finalize-saved']),0)
            launch.assert_called_once_with('finalize');path.return_value.mkdir.assert_called_once_with(exist_ok=False)
            self.assertEqual(owner.BASE,c.COMPLETION)

    def test_saved_completion_rejects_tampering_before_original_reads(self):
        import amd_zombie_d_compare as reader
        with patch.object(reader,'digest',return_value='0'*64),patch.object(reader,'verify_snapshot') as verify:
            with self.assertRaisesRegex(ValueError,'Saved census identity'):c.finalize_saved()
            verify.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
