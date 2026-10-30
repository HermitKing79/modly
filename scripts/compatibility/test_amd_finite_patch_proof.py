"""Offline isolation and routing tests; native geometry/SCIP fixtures are separate."""
import subprocess
import sys
import unittest
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import amd_finite_patch_proof as p

class FinitePatchTests(unittest.TestCase):
    def test_consumed_query_limited_case_cannot_authorize_future_solve(self):
        import json
        summary=json.loads((p.BASE/'owned-session/eligibility-summary.json').read_text())
        records=[json.loads(line) for line in (p.BASE/'owned-session/eligibility.jsonl').read_text().splitlines()]
        self.assertEqual(p.eligibility_admission(summary,records),'GLOBAL_QUERY_WORK_INCOMPLETE')
    def test_local_depth_uncertainty_keeps_veto_policy(self):
        records=[dict(variable=i,key=str(i),state='CERTIFIED_ELIGIBLE',reason='CERTIFIED') for i in range(4)]
        records.append(dict(variable=4,key='4',state='UNRESOLVED',reason='DEPTH_LIMIT_OR_NUMERICAL_BAND'))
        self.assertEqual(p.eligibility_admission(dict(candidate_faces=5,eligibility_records=5,eligible_faces=4),records),'ADMITTED')
    def test_exact_query_cap_is_not_itself_incomplete_work(self):
        records=[dict(variable=i,key=str(i),state='CERTIFIED_ELIGIBLE',reason='CERTIFIED') for i in range(4)]
        self.assertEqual(p.eligibility_admission(dict(candidate_faces=4,eligibility_records=4,eligible_faces=4,queries=2000000),records),'ADMITTED')
    def test_exact_private_patch_roundtrip_and_compiled_header(self):
        from amd_finite_patch_overlay import patched_header,HOOKS
        original=(p.PRIOR/'owned-project/overlay/CGAL/Polygonal_surface_reconstruction.h').read_text()
        actual=(p.BASE/'project/overlay/CGAL/Polygonal_surface_reconstruction.h').read_text()
        self.assertEqual(patched_header(original),actual)
        restored=actual
        for before,after in reversed(HOOKS):restored=restored.replace(after,before)
        self.assertEqual(restored,original)
    def test_changed_or_ambiguous_dependency_anchor_refused(self):
        from amd_finite_patch_overlay import patched_header,HOOKS
        with self.assertRaises(ValueError):patched_header('unqualified header')
        original=(p.PRIOR/'owned-project/overlay/CGAL/Polygonal_surface_reconstruction.h').read_text()
        with self.assertRaises(ValueError):patched_header(original+'\n'+HOOKS[0][0])
    def test_windows_manifest_separator_normalization(self):
        self.assertEqual(p.manifest_identity({'input\\mesh.bin':{'sha256':'pinned'}},'input/mesh.bin'),{'sha256':'pinned'})
        self.assertEqual(p.manifest_identity({'input/mesh.bin':{'sha256':'pinned'}},'input\\mesh.bin'),{'sha256':'pinned'})
    def test_manifest_alias_collision_fails_closed(self):
        with self.assertRaises(ValueError):p.manifest_identity({'input/mesh.bin':{},'input\\mesh.bin':{}},'input/mesh.bin')
        with self.assertRaises(ValueError):p.manifest_identity({},'input/mesh.bin')
    def test_inert_default(self):self.assertEqual(p.main([]),0)
    def test_no_neural_or_wsl_route(self):
        self.assertFalse(any(x in p.STAGES for x in ('generate','decode','inference','wsl','remesh')))
        with self.assertRaises(ValueError):p.command('wsl')
    def test_exact_windows_native_event(self):
        c=p.command('reconstruct')
        self.assertTrue(p.spawn_allowed('reconstruct',(c[0],subprocess.list2cmdline(c),str(p.BASE),{})))
        self.assertTrue(p.spawn_allowed('reconstruct',(c[0],c,str(p.BASE),{})))
    def test_foreign_executable_denied(self):
        c=p.command('reconstruct')
        self.assertFalse(p.spawn_allowed('reconstruct',('wsl.exe',c,str(p.BASE),{})))
    def test_wrong_cwd_denied(self):
        c=p.command('reconstruct')
        self.assertFalse(p.spawn_allowed('reconstruct',(c[0],c,str(p.PRIOR),{})))
    def test_extra_arguments_and_injection_denied(self):
        c=p.command('reconstruct')
        self.assertFalse(p.spawn_allowed('reconstruct',(c[0],c+['; wsl.exe'],str(p.BASE),{})))
        self.assertFalse(p.spawn_allowed('reconstruct',(c[0],subprocess.list2cmdline(c)+' & whoami',str(p.BASE),{})))
    def test_native_modes_are_exactly_scoped(self):
        self.assertEqual(p.command('fixtures')[1],'fixtures')
        self.assertEqual(p.command('reconstruct')[-1],'finite-patch-v1')
        with self.assertRaises(ValueError):p.command('authenticate')
    def test_cpu_resource_limits_unchanged(self):
        self.assertEqual(p.owner.CAP,6*1024**3);self.assertEqual(p.owner.DEADLINE,180)
        self.assertFalse(p.owner.admission(12*1024**3-1,2*1024**3))
    def test_source_scope_blocks_procedural_reference(self):
        self.assertFalse(p.source_allowed(p.owner.ROOT/'scripts/compatibility/amd_closed_solid_benchmark.py'))
        self.assertFalse(p.source_allowed(p.BASE.parent/'amd-closed-solid-benchmark-v1/source/mesh.ply'))
        self.assertTrue(p.source_allowed(p.PRIOR/'input/mesh.bin'))
        self.assertTrue(p.source_allowed(p.BASE/'owned-session/candidate/raw-reconstruction.off'))

if __name__=='__main__':unittest.main()
