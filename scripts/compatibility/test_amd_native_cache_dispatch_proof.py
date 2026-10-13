"""CPU-only acceptance regressions for the real entry-point proof record."""
import copy
from pathlib import Path
import unittest
import amd_native_cache_dispatch_proof as proof


def valid_record():
    return dict(returncode=0,worker_exit_verified=True,
        job_cleanup=dict(owned_tree_exit_verified=True,forced_job_cleanup=False),
        worker=dict(status='PASS',source_identity_verified=True,gpu_initialized=False,weights_loaded=False,
            installed_compile_calls=2,installed_constructor_calls=1,installed_group_calls=1,
            forbidden_calls={k:0 for k in proof.COUNTERS},
            miss=dict(hash='1'*64,missing_from_historical_and_private_cache=True,rejected=True),
            hit=dict(hash='d5de3d60ef4c1dfce927f5a59a91fdfa35c7de3400fb71acb214abe5dc4fbea5',
                kernel_sha256='e639008883feef1e37079d2a809a45cdfef85240aaec7ae9f018d8175cf7fe01',
                module=None,function=None,run_handle=None),
            runtime=dict(python='3.12.10',torch='2.13.0+rocm10.0.0',cuda=None,hip='7.15.26333')))


class ProofAcceptance(unittest.TestCase):
    def test_owned_windows_alias_preserves_real_directory_and_foreign_alias_rejects(self):
        approved=proof.EV.resolve()
        alias='\\\\?\\'+str(approved)
        self.assertEqual(proof.normalize_cache_mkdir(alias,approved),str(approved))
        with self.assertRaises(ValueError):proof.normalize_cache_mkdir('\\\\?\\'+str(proof.ROOT.resolve()),approved)
        for foreign in ('\\\\?\\UNC\\server\\share','\\\\?\\GLOBALROOT\\Device\\HarddiskVolume1'):
            with self.assertRaises(ValueError):proof.normalize_cache_mkdir(foreign,approved)
        self.assertEqual(proof.normalize_cache_mkdir(str(approved),approved),str(approved))

    def test_genuine_entry_hit_and_miss_with_clean_release_are_accepted(self):
        self.assertTrue(proof.result_passed(valid_record()))

    def test_native_callback_or_gpu_initialization_invalidates_proof(self):
        for name in proof.COUNTERS:
            r=valid_record();r['worker']['forbidden_calls'][name]=1
            self.assertFalse(proof.result_passed(r))
        for field in ('gpu_initialized','weights_loaded'):
            r=valid_record();r['worker'][field]=True
            self.assertFalse(proof.result_passed(r))

    def test_missing_counter_or_boolean_counter_cannot_pass(self):
        r=valid_record();del r['worker']['forbidden_calls']['helper_load']
        self.assertFalse(proof.result_passed(r))
        r=valid_record();r['worker']['forbidden_calls']['gpu_init']=False
        self.assertFalse(proof.result_passed(r))

    def test_unverified_hit_handles_or_false_miss_cannot_pass(self):
        for field,value in (('hash','2'*64),('kernel_sha256','3'*64),('module',123),('run_handle','loaded')):
            r=valid_record();r['worker']['hit'][field]=value
            self.assertFalse(proof.result_passed(r))
        for field in ('missing_from_historical_and_private_cache','rejected'):
            r=valid_record();r['worker']['miss'][field]=False
            self.assertFalse(proof.result_passed(r))

    def test_abnormal_exit_missing_real_calls_and_worker_error_cannot_pass(self):
        r=valid_record();r['returncode']=1;self.assertFalse(proof.result_passed(r))
        r=valid_record();r['job_cleanup']['forced_job_cleanup']=True;self.assertFalse(proof.result_passed(r))
        r=valid_record();r['worker']['installed_compile_calls']=0;self.assertFalse(proof.result_passed(r))
        r=valid_record();r['worker']['error']='diagnostic failed';self.assertFalse(proof.result_passed(r))
        self.assertFalse(proof.result_passed({}))


if __name__=='__main__':unittest.main()
