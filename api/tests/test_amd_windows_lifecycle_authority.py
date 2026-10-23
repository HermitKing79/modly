"""Synthetic lifecycle facts only: these tests do not confer native authority."""
import unittest
from services import amd_stage_process as stage


def snapshot(active=1):
    rows = []
    for pid in (10, 20):
        row = dict(pid=pid, creation_filetime=100+pid, job_owner='private', job_handle=42,
                   image='C:/python.exe', job_membership_verified=True,
                   identity_source='native-image-query', observed_alive={'alive': True},
                   exit_observation=dict(api='GetExitCodeProcess', succeeded=True, exit_code=259))
        row['image_authentication'] = dict(api='QueryFullProcessImageNameW', succeeded=True,
            observed_utc='synthetic', **{key: row[key] for key in
                ('pid', 'creation_filetime', 'job_owner', 'job_handle', 'image')})
        rows.append(row)
    return dict(active=active, total_processes=2, job_owner='private', process_identities=rows,
                enumerated_pids=[10, 20] if active else [], peak_process_bytes=1000, peak_job_bytes=2000)


class LifecycleFactsTests(unittest.TestCase):
    def prepared(self):
        p = stage._WindowsCompletion(10, 'private', 42)
        p.observe(snapshot())
        p.control('GO')
        p.event(dict(event='control-received', action='GO'))
        p.event(dict(event='stage-started', pid=20))
        return p

    def finish(self, p):
        p.event(dict(event='child-exited', child_pid=20, returncode=0, cancellation_sent=False, forced=False))
        p.event(dict(event='terminal', terminal='succeeded', returncode=0, natural=True, forced=False,
                     cooperative_cancelled=False))
        p.observe(snapshot(0))
        return p.finish(0, True)

    def test_complete_native_facts_and_no_mutable_snapshot_alias(self):
        p = self.prepared()
        public = snapshot()
        p.observe(public)
        public['process_identities'][0]['creation_filetime'] = 0
        result = self.finish(p)
        self.assertEqual(result['owned_remaining'], 0)
        self.assertEqual(result['child_identity']['creation_filetime'], 120)

    def test_missing_child_membership_cannot_be_upgraded(self):
        p = self.prepared()
        p.identities.pop((20, 120))
        with self.assertRaises(ValueError):
            p.event(dict(event='child-exited', child_pid=20, returncode=0, cancellation_sent=False, forced=False))

    def test_unknown_descendant_count_blocks(self):
        p = self.prepared()
        self.finish(p)
        row = snapshot(0); row['total_processes'] = 3
        p.observe(row)
        with self.assertRaises(ValueError):
            p.finish(0, True)

    def test_membership_image_and_creation_are_mandatory(self):
        for key, bad in [('job_membership_verified', False), ('creation_filetime', 0), ('job_owner', 'foreign')]:
            with self.subTest(key=key):
                p = stage._WindowsCompletion(10, 'private', 42)
                row = snapshot(); row['process_identities'][0][key] = bad
                with self.assertRaises(ValueError): p.observe(row)

    def test_protocol_failures_and_duplicate_start_block(self):
        for event in [dict(event='ack-error'), dict(event='control-error'), dict(event='unrecognized'),
                      dict(event='stage-started', pid=20), dict(event='control-received', action='CANCEL')]:
            with self.subTest(event=event):
                p = self.prepared()
                with self.assertRaises(ValueError): p.event(event)

    def test_cancel_and_missing_go_block(self):
        p = self.prepared()
        with self.assertRaises(ValueError): p.control('CANCEL')
        p = stage._WindowsCompletion(10, 'private', 42)
        with self.assertRaises(ValueError): p.event(dict(event='stage-started', pid=20))

    def test_forced_cancelled_failed_child_never_success(self):
        for changes in [dict(returncode=1), dict(forced=True), dict(cancellation_sent=True)]:
            p = self.prepared()
            with self.assertRaises(ValueError):
                p.event(dict(dict(event='child-exited', child_pid=20, returncode=0,
                                  cancellation_sent=False, forced=False), **changes))

    def test_open_job_failed_transport_and_incomplete_terminal_block(self):
        for closed, code in [(False, 0), (True, 1)]:
            p = self.prepared()
            self.finish(p)
            with self.assertRaises(ValueError): p.finish(code, closed)


if __name__ == '__main__': unittest.main()
