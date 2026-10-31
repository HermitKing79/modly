import sys,subprocess,unittest
import json
from unittest.mock import patch
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import amd_alpha_wrap_proof as p

class AlphaGuardTests(unittest.TestCase):
    def test_inert_default(self):self.assertEqual(p.main([]),0)
    def test_exact_windows_spawn(self):
        c=p.command('official')
        for argv in (c,tuple(c),subprocess.list2cmdline(c)):
            self.assertTrue(p.spawn_allowed('official',(c[0],argv,str(p.BASE/'official'),{})))
    def test_foreign_execution_and_injection(self):
        c=p.command('official')
        for exe,argv,cwd in [('wsl.exe',c,str(p.BASE/'official')),(c[0],c+['extra'],str(p.BASE/'official')),(c[0],subprocess.list2cmdline(c)+' & whoami',str(p.BASE/'official')),(c[0],c,str(p.BASE))]:
            self.assertFalse(p.spawn_allowed('official',(exe,argv,cwd,{})))
    def test_no_unauthorised_route(self):
        for s in ('wsl','neural','install','download','retry'):
            self.assertNotIn(s,p.STAGES)
            with self.assertRaises(ValueError):p.command(s)
    def test_unchanged_limits_and_parameters(self):
        self.assertEqual(p.owner.CAP,6*1024**3);self.assertEqual(p.owner.DEADLINE,180)
        self.assertFalse(p.owner.admission(12*1024**3-1,2*1024**3))
        self.assertEqual(p.ALPHA_DENOMINATOR,20);self.assertEqual(p.OFFSET_DENOMINATOR,2000)
    def test_unknown_dependency_refused(self):
        with self.assertRaises(ValueError):p.dependency_spec('torch')
    def test_missing_stage_approval_refused(self):
        row=dict(classification='APPROVED',allowed_stages=['official'],files={})
        with patch.object(Path,'read_text',return_value=json.dumps(row)):
            with self.assertRaises(ValueError):p.reviewed('wrap')
    def test_reviewed_file_tampering_refused(self):
        row=dict(classification='APPROVED',allowed_stages=['official'],files={'reviewed.py':'expected'})
        with patch.object(Path,'read_text',return_value=json.dumps(row)),patch.object(p.owner,'sha',return_value='changed'):
            with self.assertRaises(ValueError):p.reviewed('official')
    def test_exact_stage_pin_accepted(self):
        row=dict(classification='APPROVED',allowed_stages=['official'],files={'reviewed.py':'expected'})
        with patch.object(Path,'read_text',return_value=json.dumps(row)),patch.object(p.owner,'sha',return_value='expected'):
            p.reviewed('official')
    def test_failed_prerequisite_refused(self):
        with patch.object(Path,'read_text',return_value='{"classification":"FAIL"}'):
            with self.assertRaises(ValueError):p.successful('official')
    def test_artifact_path_alias_refused(self):
        with patch.object(Path,'read_text',return_value='{"path":"unrelated"}'):
            with self.assertRaises(ValueError):p.artifact('wrap',p.BASE/'raw-wrap.off')

if __name__=='__main__':unittest.main()
