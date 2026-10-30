import itertools
import unittest
import subprocess
from pathlib import Path
from amd_closure_graph import prune

class Closure(unittest.TestCase):
    def test_closed_tetra_is_not_rejected(self):
        # Two possible faces plus two vetoed alternatives on each tetra edge.
        fans=[[a,b,4,5] for a,b in itertools.combinations(range(4),2)]
        self.assertEqual(prune(6,fans,[4,5])['remaining'],list(range(4)))
    def test_missing_tetra_face_propagates(self):
        fans=[[a,b,4,5] for a,b in itertools.combinations(range(4),2)]
        r=prune(6,fans,[0,4,5]);self.assertEqual(r['remaining'],[])
        self.assertTrue(r['steps'])
    def test_nonfour_is_zero_even_if_two(self):
        self.assertEqual(prune(4,[[0,1],[1,2,3]],[])['remaining'],[])
    def test_unknown_unresolved_is_potentially_allowed(self):
        self.assertEqual(prune(4,[[0,1,2,3]],[])['remaining'],list(range(4)))
    def test_malformed_rejected(self):
        for fans in ([[0,0,1,2]],[[0,1,2,4]],[[0,1,2]],[]):
            with self.assertRaises(ValueError):prune(4,fans,[])
    def test_exhaustive_equivalence_necessary_only(self):
        fans=[[0,1,2,3],[1,2,4,5],[0,3,4,5],[0,5]]
        for mask in range(64):
            dead=[i for i in range(6) if mask>>i&1]
            r=prune(6,fans,dead)
            for x in itertools.product((0,1),repeat=6):
                valid=not any(x[i] for i in dead) and all(sum(x[i] for i in f) in ((0,2) if len(f)==4 else (0,)) for f in fans)
                if valid:self.assertFalse(any(x[i] for i in set(range(6))-set(r['remaining'])))
    def test_private_hooks_are_exact(self):
        import amd_closure_proof as r
        from amd_closure_overlay import capture_header,diagnostic_header,common_source
        original=(r.PHASE15/'project/overlay/CGAL/Polygonal_surface_reconstruction.h').read_text()
        self.assertEqual(diagnostic_header(original),(r.BASE/'project/overlay/CGAL/Polygonal_surface_reconstruction.h').read_text())
        source=Path(__file__).with_name('amd_finite_patch_reconstruct.cpp').read_text()
        self.assertEqual(common_source(source),(r.BASE/'project/phase15-common.inc').read_text())
        with self.assertRaises(ValueError):capture_header(original+original)
        with self.assertRaises(ValueError):common_source('unqualified')
    def test_exact_cpu_spawn_allowlist(self):
        import amd_closure_proof as r
        for stage in ('configure','build','build-mapping','build-diagnostic','configure-certify','build-certify','build-refine','certify','refine','capture','capture-mapping','diagnostic'):
            c=r.command(stage)
            for args in (c,tuple(c),subprocess.list2cmdline(c)):
                self.assertTrue(r.spawn_allowed(stage,(c[0],args,str(r.BASE),{})))
            self.assertFalse(r.spawn_allowed(stage,(c[0],c+['--extra'],str(r.BASE),{})))
            self.assertFalse(r.spawn_allowed(stage,('wsl.exe',c,str(r.BASE),{})))
            self.assertFalse(r.spawn_allowed(stage,(c[0],c,str(r.PRIOR),{})))
        self.assertFalse(r.spawn_allowed('analyse',('wsl.exe',[],str(r.BASE),{})))
        self.assertNotIn('production',r.STAGES)
        with self.assertRaises(ValueError):r.command('production')
    def test_cyclic_correspondence_and_ambiguity(self):
        from amd_closure_mapping import map_faces
        a=dict(variable=0,key='a',group=0,state='CONFIRMED_UNSUPPORTED',vertices=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        b=dict(a,key='b',vertices=[[0.,1.,0.],[1.,0.,0.],[1e-14,0.,0.]])
        r=map_faces([a],[b]);self.assertFalse(r['eligible_certificates_transferred']);self.assertLess(r['maximum_displacement_native'],1e-10)
        for changed in (dict(b,group=1),dict(b,vertices=[[0.,1.,0.],[1.,0.,0.],[1e-5,0.,0.]])):
            with self.assertRaises(ValueError):map_faces([a],[changed])
        with self.assertRaises(ValueError):map_faces([a,dict(a,variable=1,key='c')],[b,dict(b,variable=1,key='d')])
    def test_certificate_core_preserves_all_geometry_guards(self):
        import amd_closure_proof as r
        from amd_closure_overlay import certificate_core,refine_core
        source=Path(__file__).with_name('amd_finite_patch_reconstruct.cpp').read_text()
        original=source[source.index('enum State {'):source.index('template<class S>void modly_observe_planes(S& set){')]
        expected=original.replace('static std::vector<PNI> input_points;\n','').replace('QUERY_LIMIT=2000000;','QUERY_LIMIT=32000000;')
        self.assertEqual(certificate_core(source),expected)
        self.assertEqual((r.BASE/'retained-depth20-core.inc').read_text(),expected)
        self.assertEqual((r.BASE/'certify-project/certificate-core.inc').read_text(),refine_core(source))
        self.assertIn('DEPTH_LIMIT=28',refine_core(source))
        self.assertIn('EPS=.2,MARGIN=1e-8,PLANARITY_LIMIT=1e-10',refine_core(source))
        self.assertIn('EPS=.2,MARGIN=1e-8,PLANARITY_LIMIT=1e-10',expected)
        self.assertIn('DEPTH_LIMIT=20',expected)
    def test_box_lower_bound_is_not_triangle_distance(self):
        from amd_closure_fidelity import decimal_box_witness
        tri=dict(variable=0,vertices=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        # This point is outside the triangle but in its enclosing box.
        r=decimal_box_witness([1.,1.,0.],[tri]);self.assertEqual(float(r['distance_lower_bound_native']),0)
        self.assertFalse(r['strict_frozen_max_violation'])
    def test_box_outward_guard_and_all_possible_faces(self):
        from amd_closure_fidelity import decimal_box_witness
        a=dict(variable=0,vertices=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        b=dict(variable=1,vertices=[[4.,0.,0.],[5.,0.,0.],[4.,1.,0.]])
        self.assertTrue(decimal_box_witness([4.,0.,0.],[a])['strict_frozen_max_violation'])
        self.assertFalse(decimal_box_witness([4.,0.,0.],[a,b])['strict_frozen_max_violation'])
        self.assertFalse(decimal_box_witness([2.,0.,0.],[a])['strict_frozen_max_violation'])
        with self.assertRaises(ValueError):decimal_box_witness([float('nan'),0.,0.],[a])
    def test_certificate_merge_scope_and_identity(self):
        import json
        import amd_closure_proof as r
        a=[json.loads(x) for x in (r.BASE/'certification/certificates.jsonl').read_text().splitlines()]
        b=[json.loads(x) for x in (r.BASE/'refinement/certificates.jsonl').read_text().splitlines()]
        c=r.merge_certificates(a,b);self.assertEqual(sum(x['state']=='CERTIFIED_ELIGIBLE' for x in c),137)
        self.assertEqual(sum(x['state']=='UNRESOLVED' for x in c),0)
        with self.assertRaises(ValueError):r.merge_certificates(a,b[:1])
        bad=[dict(x) for x in b];bad[0]['key']='spoof'
        with self.assertRaises(ValueError):r.merge_certificates(a,bad)

if __name__=='__main__':unittest.main()
