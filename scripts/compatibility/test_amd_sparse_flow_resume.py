"""CPU-only admission and run-directory isolation; no AI imports."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import amd_sparse_flow_resume as resume

class ResumeTests(unittest.TestCase):
    def test_unchanged_exact_margin_and_available_sample(self):
        margin=11411181632
        self.assertEqual(resume.REQUIRED_RAM,margin)
        for available in (margin,19599089664):
            row=resume.cpu_admission(SimpleNamespace(total=34270023680,available=available))
            self.assertEqual(row['host_available_bytes'],available)
            self.assertEqual(row['required_margin_bytes'],margin)
            self.assertTrue(row['admitted']);self.assertFalse(row['torch_imported'])
        self.assertFalse(resume.cpu_admission(SimpleNamespace(total=34270023680,available=margin-1))['admitted'])
    def test_invalid_memory_sample_fails_closed(self):
        for available in (True,-1,'19599089664'):
            with self.assertRaises(ValueError):resume.cpu_admission(SimpleNamespace(total=34270023680,available=available))
    def test_only_new_evidence_and_entrypoint_change(self):
        old=resume.previous.local.EVIDENCE
        try:
            with patch.object(resume,'verify_reviewed_code',return_value=True):
                profile=resume.profile()
            self.assertEqual(profile.evidence,resume.EVIDENCE)
            self.assertEqual(profile.worker_script,resume.Path(resume.__file__).resolve())
            self.assertEqual((profile.cpu_seconds,profile.gpu_seconds,profile.overall_seconds),(180,240,480))
            self.assertIs(profile.authorize_placement,resume.previous.contract.authorize_placement)
            self.assertIs(profile.classify,resume.previous.contract.classify)
            self.assertEqual(profile.authorize,resume.previous.AUTHORIZE)
            self.assertEqual(resume.previous.PROBE,resume.ORIGINAL/'cpu-metadata-final-candidate/results.json')
            self.assertEqual(resume.previous.local.PINS,resume.ORIGINAL/'source-contract.json')
            self.assertIn('scripts/compatibility/amd_sparse_flow_resume.py',profile.committed_files)
        finally:resume.previous.local.EVIDENCE=old
    def test_review_failure_blocks_configuration(self):
        old=resume.previous.local.EVIDENCE
        with patch.object(resume,'verify_reviewed_code',side_effect=RuntimeError('changed')):
            with self.assertRaises(RuntimeError):resume.configure()
        self.assertEqual(resume.previous.local.EVIDENCE,old)

if __name__=='__main__':unittest.main()
