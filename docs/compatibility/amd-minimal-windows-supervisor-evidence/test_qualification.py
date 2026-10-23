"""Mocked admission checks only; never creates a Job or native worker."""
import copy
import importlib.util
from pathlib import Path
import unittest

P = Path(__file__).with_name('qualification.py')
if P.exists():
    spec = importlib.util.spec_from_file_location('minimal_qualification', P)
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
else:
    q = None


def proof_inputs():
    context = {'nonce': 'a' * 32, 'job_id': 'synthetic-target'}
    ready = dict(context=context, stage_id='target', role='target', pid=42,
                 release_nonce='b' * 32, job_nonce='c' * 32, monotonic_s=9)
    identity = dict(pid=42, creation_filetime=123, job_owner='private-job', job_handle=17,
        job_membership_verified=True, image='C:/p/python.exe',
        observed_alive={'alive': True}, identity_source='native-image-query',
        image_authentication=dict(api='QueryFullProcessImageNameW', succeeded=True,
            pid=42, creation_filetime=123, job_owner='private-job', job_handle=17,
            image='C:/p/python.exe', observed_utc='mocked'),
        exit_observation=dict(api='GetExitCodeProcess', succeeded=True, exit_code=259))
    snapshot = dict(monotonic_s=10, active=2, enumerated_pids=[41, 42],
                    job_owner='private-job', process_identities=[identity])
    expected = dict(context=context, stage_id='target', role='target',
                    release_nonce='b' * 32, job_nonce='c' * 32)
    return ready, snapshot, expected


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(q, 'qualification implementation not yet present')

    def test_exact_current_native_identity(self):
        ready, snapshot, expected = proof_inputs()
        self.assertEqual(q.live_proof(ready, snapshot, expected, 10.1, 9)['identity']['creation_filetime'], 123)

    def test_cached_but_not_enumerated_rejected(self):
        args = list(proof_inputs()); args[1]['enumerated_pids'] = [41]
        self.assertIsNone(q.live_proof(*args, 10.1, 9))

    def test_stale_and_future_observations_rejected(self):
        for stamp in (8, 11):
            with self.subTest(stamp=stamp):
                ready, snapshot, expected = proof_inputs(); snapshot['monotonic_s'] = stamp
                self.assertIsNone(q.live_proof(ready, snapshot, expected, 10.1, 9))

    def test_foreign_readiness_rejected(self):
        for key in ('context', 'stage_id', 'role', 'release_nonce', 'job_nonce'):
            with self.subTest(key=key):
                ready, snapshot, expected = proof_inputs(); ready[key] = 'foreign'
                with self.assertRaises(ValueError): q.live_proof(ready, snapshot, expected, 10.1, 9)

    def test_identity_provenance_failure_rejected(self):
        for field, value in (('job_owner', 'foreign'), ('creation_filetime', 124), ('succeeded', False)):
            with self.subTest(field=field):
                ready, snapshot, expected = proof_inputs()
                snapshot['process_identities'][0]['image_authentication'][field] = value
                self.assertIsNone(q.live_proof(ready, snapshot, expected, 10.1, 9))

    def test_empty_job_is_not_enough(self):
        row = dict(terminal='succeeded', natural=True, forced=False, windows_settled=True,
                   windows={'active': 0}, returncode=0, bootstrap_returncode=0,
                   elapsed_s=2, diagnostic_errors=[])
        self.assertTrue(q.settled(row, 'succeeded'))
        for key, value in (('forced', True), ('natural', False), ('elapsed_s', 25),
                           ('returncode', 1), ('bootstrap_returncode', 1),
                           ('diagnostic_errors', [{'error': 'access denied'}])):
            with self.subTest(key=key):
                bad = dict(row); bad[key] = value
                self.assertFalse(q.settled(bad, 'succeeded'))

    def test_post_settlement_requires_a_second_native_snapshot(self):
        ready, snapshot, expected = proof_inputs()
        self.assertIsNone(q.proof_after_fence(ready, snapshot, expected, 10.1, 10))
        snapshot['monotonic_s'] = 10.05
        self.assertIsNotNone(q.proof_after_fence(ready, snapshot, expected, 10.1, 10))


if __name__ == '__main__': unittest.main()
