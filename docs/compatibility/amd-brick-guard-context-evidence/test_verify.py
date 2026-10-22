"""Independent tiny readback negatives; no original model arrays or worker run."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest
import numpy as np
from verify_worker import link_ok,guard_closure,screen,replay_progression,verify_snapshot

def fixture():
    v=np.array([[-0.,0,0],[1,0,0],[0,1,0],[1,1,0]],np.float32)
    f=np.array([[0,1,2],[2,1,3]],np.int32);entries=[]
    for row,t in enumerate(f):
        for j in range(3):
            a,b=map(int,(t[j],t[(j+1)%3]));entries.append((min(a,b)*4+max(a,b),3*row+j))
    entries.sort();keys=np.array(sorted({x[0] for x in entries}),np.uint64)
    inc=np.array([sum(x[0]==k for x in entries) for k in keys],np.uint8)
    order=np.array([x[1] for x in entries],np.int32);off=np.r_[0,np.cumsum(inc)].astype(np.int32)
    ctx=dict(order=order,offsets=off,consistent=inc==2,protected_faces=np.zeros(2,bool),protected_vertices=np.zeros(4,bool),locked_vertices=np.zeros(4,bool),global_sheet_labels=np.zeros(2,np.int32),eligible_edge_ids=np.empty(0,np.int32),face_degree=np.array([1,2,2,1],np.int32))
    def classify(*args):
        return {},dict(protected_faces=np.ones(2,bool),protected_vertices=np.ones(4,bool),vertex_locks=np.ones(4,bool),edge_keys=keys,eligible_edge_rows=np.empty(0,np.int32))
    choice=dict(selected_component_faces=2,selected_component_edges=1,eligible_face_components=1,seed_edge_id=0)
    arrays=dict(vertices=v.copy(),faces=f.copy(),source_faces=f.copy(),source_face_ids=np.array([0,1],np.int32),source_vertex_ids=np.arange(4,dtype=np.int32),original_native_RGBA=np.full((4,4),255,np.uint8),source_component_labels=np.zeros(4,np.int32),global_sheet_labels=np.zeros(2,np.int32),outer_face_ids=np.array([1],np.int32),eligible_edge_ids=np.empty(0,np.int32),non_touching_edge_ids=np.empty(0,np.int32),eligible_faces=np.zeros(2,bool),complete_star_vertex_ids=np.arange(4,dtype=np.int32),complete_star_offsets=np.array([0,1,3,5,6],np.int32),complete_star_source_face_ids=np.array([0,0,1,0,1,1],np.int32))
    arrays.update({name:np.zeros(n,bool) for name,n in [('inherited_protected_faces',2),('inherited_protected_vertices',4),('inherited_vertex_locks',4)]})
    arrays.update({name:np.ones(n,bool) for name,n in [('local_protected_faces',2),('local_protected_vertices',4),('local_vertex_locks',4),('protected_faces',2),('protected_vertices',4),('vertex_locks',4)]})
    s=dict(original_patch_faces=2,original_patch_vertices=4,source_component=0,raw_original_edge_connected_components=1,eligible_subset_of_independent_local_classifier=True,selected_component_faces=2,selected_component_edges=1,eligible_face_components=1,seed_edge_id=0,globally_certified_edges_touching_patch=0,both_edge_faces_represented=0,complete_global_endpoint_star_edges=0,post_crop_inherited_guard_edges=0,lost_incomplete_edge_face_context=0,lost_incomplete_endpoint_star_context=0,lost_to_actual_crop_or_inherited_guards=0,full_link_rejected=0,strict_eligible_edges=0,eligible_face_population=0,immutable_face_floor=2,optimistic_removable_face_ceiling=0,guard_locked_vertices=4,initial_non_touching_opportunities=0,initial_topological_removal_estimate_faces=0,estimate_fraction_of_eligible_faces=None,initial_estimate_fraction_of_entire_patch=0.,optimistic_removable_ceiling_fraction_of_entire_patch=0.,twenty_percent_is_informational_not_admission=True,qualified=False)
    return v,f,arrays['original_native_RGBA'].copy(),keys,inc,np.zeros(4,np.int32),ctx,arrays,s,{},choice,classify

class IndependentTests(unittest.TestCase):
    def test_full_link_rejects_shared_link_edge_and_off_crop_neighbour(self):
        f=np.array([[0,1,2],[1,0,3]],np.int32)
        self.assertTrue(link_ok(f,0,1,[0,1],[0,1]))
        tetra=np.vstack((f,[[0,2,3],[1,3,2]])).astype(np.int32)
        self.assertFalse(link_ok(tetra,0,1,[0,1,2],[0,1,3]))
        extra=np.vstack((f,[[0,4,5],[1,6,4]])).astype(np.int32)
        self.assertFalse(link_ok(extra,0,1,[0,1,2],[0,1,3]))
    def test_guard_closure_protects_all_neighbours_and_position_aliases(self):
        f=np.array([[0,1,2],[2,3,4]],np.int32);v=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[2,1,0],[-0.,0,0]],np.float32)
        p,pv,lock=guard_closure(v,f,np.array([1,0],bool),np.zeros(2,bool),np.zeros(6,bool),np.zeros(6,bool))
        np.testing.assert_array_equal(p,[1,0]);np.testing.assert_array_equal(pv,[1,1,1,0,0,0]);self.assertTrue(lock.all())
    def test_exact_integer_screen_boundaries(self):
        s=dict(initial_non_touching_opportunities=10,eligible_face_population=100,original_patch_faces=400,qualified=True)
        self.assertTrue(screen(s)['meaningful_pass'])
        for key,value in [('initial_non_touching_opportunities',9),('eligible_face_population',101),('original_patch_faces',401),('qualified',False)]:
            self.assertFalse(screen(dict(s,**{key:value}))['meaningful_pass'])
        with self.assertRaises(ValueError):screen(dict(s,initial_non_touching_opportunities=True))
    def test_complete_snapshot_and_every_saved_array_negative(self):
        args=fixture();verify_snapshot(*args,check=lambda:None)
        for name in ('vertices','source_faces','original_native_RGBA','local_protected_faces','vertex_locks','complete_star_offsets'):
            bad=list(fixture());a=bad[7][name]
            if name=='vertices':a[0,0]=0. # signed zero is part of the contract
            elif a.dtype==np.bool_:a.flat[0]=not a.flat[0]
            else:a.flat[0]=int(a.flat[0])+(-1 if a.dtype==np.uint8 else 1)
            with self.subTest(name=name),self.assertRaises(ValueError):verify_snapshot(*bad,check=lambda:None)
        bad=list(fixture());bad[8]['guard_locked_vertices']=3
        with self.assertRaises(ValueError):verify_snapshot(*bad,check=lambda:None)
    def test_budget_cancellation_stays_incomplete(self):
        def stop():raise TimeoutError('synthetic budget')
        with self.assertRaises(TimeoutError):verify_snapshot(*fixture(),check=stop)
    def test_real_pinned_classifier_and_producer_schema_handoff(self):
        # Synthetic only: catches the real adapter/independent-checker handoff,
        # including every saved array and every summary field, before a run.
        import common as c
        import qualification as q
        pc=c.pinnedpc();size=20;n=size*size
        v=np.array([(x,y,0) for y in range(size) for x in range(size)],np.float32);rows=[]
        for y in range(size-1):
            for x in range(size-1):a=y*size+x;rows.extend(((a,a+1,a+size+1),(a,a+size+1,a+size)))
        f=np.array(rows,np.int32);entries=[]
        for row,t in enumerate(f):
            for j in range(3):a,b=map(int,(t[j],t[(j+1)%3]));entries.append((min(a,b)*n+max(a,b),3*row+j))
        entries.sort();keys=np.array(sorted({x[0] for x in entries}),np.uint64);inc=np.array([sum(x[0]==k for x in entries) for k in keys],np.uint8)
        order=np.array([x[1] for x in entries],np.int32);off=np.r_[0,np.cumsum(inc)].astype(np.int32)
        prior=c.pin('independent_test_local_classifier',c.ROOT/'docs/compatibility/amd-brick-junction-protected-evidence/patch_contract.py','336e49bd7836f4a66c02ade831d6b37b66f290cb42e3aa30c72e7a009599d60c')
        _,base=prior.classify(v,f,np.arange(n,dtype=np.int32),f,[],keys,inc,n)
        ctx=dict(order=order,offsets=off,consistent=inc==2,protected_faces=base['protected_faces'],protected_vertices=base['protected_vertices'],locked_vertices=base['vertex_locks'],global_sheet_labels=base['sheet_labels'],eligible_edge_ids=base['eligible_edge_rows'],face_degree=np.bincount(f.ravel(),minlength=n).astype(np.int32),classify=prior.classify,check=lambda:None)
        raw_choice=pc.choose_component(f,keys,order,off,ctx['eligible_edge_ids']);selection=pc.extract(f,keys,order,off,raw_choice['seed_edge_id'],n)
        choice=dict(selected_component_faces=len(raw_choice['component_face_ids']),selected_component_edges=len(raw_choice['component_edge_ids']),eligible_face_components=raw_choice['eligible_component_count'],seed_edge_id=raw_choice['seed_edge_id'])
        rgba=(np.arange(n*4).reshape(n,4)%256).astype(np.uint8);labels=np.zeros(n,np.int32)
        for locks in (ctx['locked_vertices'],np.ones(n,bool)):
            context=dict(ctx,locked_vertices=locks);s,a,local=q.qualify(v,f,rgba,keys,inc,labels,context,choice,selection);a['outer_face_ids']=selection['ring_added_face_ids'][-1]
            verify_snapshot(v,f,rgba,keys,inc,labels,context,a,s,local,choice,prior.classify,lambda:None)
    def test_progression_complete_rings_and_deduplicated_max_label(self):
        f=np.array([[0,1,2],[1,3,2],[2,3,4],[3,5,4],[4,5,6],[5,7,6]],np.int32);entries=[];n=8
        for row,t in enumerate(f):
            for j in range(3):a,b=map(int,(t[j],t[(j+1)%3]));entries.append((min(a,b)*n+max(a,b),3*row+j))
        entries.sort();keys=np.array(sorted({x[0] for x in entries}),np.uint64);counts=[sum(x[0]==k for x in entries) for k in keys]
        order=np.array([x[1] for x in entries],np.int32);off=np.r_[0,np.cumsum(counts)].astype(np.int32)
        result=replay_progression(f,keys,order,off,n,np.array([0],np.int32),np.array([0],np.int32),lambda:None,targets=(2,3,5),cap=5)
        self.assertEqual([x['depth'] for x in result['snapshots']],[4,5,7]);self.assertEqual(result['snapshots'][-1]['labels'],['5','max'])
        self.assertEqual(result['terminal'],dict(reason='exact_cap',largest_depth=7,largest_face_count=5))

if __name__=='__main__':unittest.main(verbosity=2)
