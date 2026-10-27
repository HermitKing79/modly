"""Offline proc/pidfd fixtures; no real /proc reads or process signals."""
import io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import amd_pixal3d_owned_proc_v3 as p
TMP=None
OWNER=dict(pid=10,pgid=10,start_ticks=1,token='a'*32)

class ProcTests(unittest.TestCase):
    def test_permission_denied_live_member_must_fail_but_proven_terminal_may_be_omitted(self):
        for state,ok in [('S',False),('Z',True),('X',True)]:
            fs=FakeFS(states=['S','S',state],env_error=PermissionError('environ'));fs.dead_value=ok
            if ok:self.assertEqual(p.process_rows(10,fs=fs),[])
            else:
                with self.assertRaises(p.OwnershipError):p.process_rows(10,fs=fs)

    def test_permission_error_with_unknown_terminal_state_still_fails(self):
        fs=FakeFS(states=['S','S',PermissionError('stat denied')],env_error=PermissionError('environ'))
        with self.assertRaises((p.OwnershipError,PermissionError)):p.process_rows(10,fs=fs)
        self.assertIn(120,fs.closed)

    def test_disappearance_uses_the_same_pinned_directory_and_accepts_no_member(self):
        fs=FakeFS(states=['S','S',ProcessLookupError('gone')],env_error=PermissionError('environ'));fs.dead_value=True;events=[]
        self.assertEqual(p.process_rows(10,fs=fs,report=events.append),[])
        self.assertEqual(fs.stat_fds,[120,120,120]);self.assertEqual(events[0]['event'],'verified_process_exit')

    def test_PID_reuse_or_group_change_between_reads_is_rejected(self):
        for changed in (dict(pid=20,pgid=10,start_ticks=31,state='S'),dict(pid=20,pgid=99,start_ticks=30,state='S')):
            fs=FakeFS(states=['S',changed])
            with self.assertRaises(p.OwnershipError):p.process_rows(10,fs=fs)

    def test_foreign_missing_and_duplicate_nonces_are_rejected_before_any_signal(self):
        for env in (b'',b'MODLY_PIXAL_PROOF_TOKEN='+b'b'*32,b'MODLY_PIXAL_PROOF_TOKEN='+b'a'*32+b'\0MODLY_PIXAL_PROOF_TOKEN='+b'a'*32):
            fs=FakeFS(env=env)
            with self.assertRaises(p.OwnershipError):p.signal_verified(OWNER,15,fs=fs)
            self.assertEqual(fs.signals,[])

    def test_reused_group_leader_is_rejected(self):
        fs=FakeFS();fs.pids=[10]
        with self.assertRaises(p.OwnershipError):p.signal_verified(OWNER,15,fs=fs)
        self.assertEqual(fs.signals,[])

    def test_unavailable_pidfds_never_fall_back_to_numeric_signals(self):
        fs=FakeFS();fs.support=False
        with self.assertRaises(p.OwnershipError):p.signal_verified(OWNER,15,fs=fs)
        self.assertEqual(fs.signals,[]);self.assertEqual(fs.stat_fds,[])

    def test_pidfd_binding_rejects_reused_or_ambiguous_target(self):
        for states,dead in ((['S','S',dict(pid=20,pgid=10,start_ticks=31,state='S')],False),(['S']*3,True),(['S','S','Z'],False)):
            fs=FakeFS(states=states);fs.dead_value=dead
            with self.assertRaises(p.OwnershipError):p.signal_verified(OWNER,15,fs=fs)
            self.assertEqual(fs.signals,[]);self.assertIn(220,fs.closed)

    def test_signal_targets_only_a_verified_pidfd_and_closes_both_handles(self):
        fs=FakeFS();sent=p.signal_verified(OWNER,15,fs=fs)
        self.assertEqual(fs.signals,[(220,15)]);self.assertEqual(sent,[dict(pid=20,start_ticks=30,signal=15)])
        self.assertCountEqual(fs.closed,[120,220])

    def test_whole_snapshot_foreign_member_prevents_signalling_earlier_valid_member(self):
        fs=FakeFS();fs.pids=[20,21];fs.foreign_fd=121
        with self.assertRaises(p.OwnershipError):p.signal_verified(OWNER,15,fs=fs)
        self.assertEqual(fs.signals,[]);self.assertIn(220,fs.closed)

    def test_terminal_target_after_binding_is_not_signalled(self):
        fs=FakeFS(states=['S','S','S','Z']);fs.dead_values=iter([False,True,True])
        self.assertEqual(p.signal_verified(OWNER,15,fs=fs),[]);self.assertEqual(fs.signals,[])

    def test_disappearance_after_binding_requires_dead_pidfd_before_omission(self):
        fs=FakeFS(states=['S','S','S',ProcessLookupError('reaped')]);fs.dead_values=iter([False,True])
        self.assertEqual(p.signal_verified(OWNER,15,fs=fs),[]);self.assertEqual(fs.signals,[])
        fs=FakeFS(states=['S','S','S',ProcessLookupError('ambiguous')])
        with self.assertRaises(p.OwnershipError):p.signal_verified(OWNER,15,fs=fs)
        self.assertEqual(fs.signals,[])

    def test_zombie_leader_with_surviving_threads_is_never_omitted(self):
        for states,env_error in ((['Z','Z'],None),(['S','S','Z'],PermissionError('environ'))):
            fs=FakeFS(states=states,env_error=env_error)
            with self.assertRaises(p.OwnershipError):p.process_rows(10,fs=fs)
            self.assertEqual(fs.signals,[])

    def test_initial_unidentified_disappearance_is_unverifiable_not_ignored(self):
        fs=FakeFS(states=[ProcessLookupError('no binding')])
        with self.assertRaises(p.OwnershipError):p.process_rows(10,fs=fs)

    def test_pidfd_preflight_checks_kernel_availability_and_closes_probe(self):
        fs=p.ProcFS()
        with patch.object(p.os,'pidfd_open',return_value=88,create=True) as opened,patch.object(p.signal,'pidfd_send_signal',create=True),patch.object(fs,'dead',return_value=False),patch.object(p.os,'close') as close:
            fs.require_pidfds();self.assertEqual(opened.call_args.args,(p.os.getpid(),0));close.assert_called_once_with(88)
        with patch.object(p.os,'pidfd_open',side_effect=PermissionError('kernel refusal'),create=True),patch.object(p.signal,'pidfd_send_signal',create=True):
            with self.assertRaises(p.OwnershipError):fs.require_pidfds()

    def test_live_permission_failure_during_signal_recheck_is_not_ignored(self):
        fs=FakeFS();original=fs.read;counts=[]
        def read(fd,name):
            if name=='environ':
                counts.append(1)
                if len(counts)>1:raise PermissionError('changed permissions')
            return original(fd,name)
        fs.read=read
        with self.assertRaises(p.OwnershipError):p.signal_verified(OWNER,15,fs=fs)
        self.assertEqual(fs.signals,[])

    def test_missing_live_RSS_cannot_be_reported_as_zero(self):
        fs=FakeFS();fs.status=b'State:\tS\n'
        with self.assertRaises(p.OwnershipError):p.process_rows(10,fs=fs)

    def test_stat_parser_handles_parentheses_and_requires_exact_PID(self):
        text='20 (name (with) parens)) S 1 10 '+'0 '*16+'30'
        self.assertEqual(p.stat_identity(text,20),dict(pid=20,state='S',pgid=10,start_ticks=30))
        with self.assertRaises(p.OwnershipError):p.stat_identity(text,21)

    def test_kernel_reads_are_relative_to_pinned_proc_fd(self):
        with patch.object(p.os,'O_NOFOLLOW',0x20000,create=True),patch.object(p.os,'open',return_value=88) as opened,patch.object(p.os,'fdopen',return_value=io.BytesIO(b'value')):
            self.assertEqual(p.ProcFS().read(77,'environ'),b'value')
            self.assertEqual(opened.call_args.args[0],'environ');self.assertEqual(opened.call_args.kwargs,dict(dir_fd=77))

    def test_worker_diagnostics_never_overwrite_controller_failure_or_promote_PASS(self):
        import amd_pixal3d_operator_guest_v3 as guest
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            leaf=Path(d);data=dict(classification='PASS',error='worker detail',operators={'gfx1201':{'status':'PASS'}})
            (leaf/'child-result.json').write_text(json.dumps(data))
            record=dict(classification='FAIL',error='controller ownership failed')
            guest.harvest_worker(record,leaf)
            self.assertEqual(record['classification'],'FAIL');self.assertEqual(record['error'],'controller ownership failed')
            self.assertEqual(record['worker_error'],'worker detail');self.assertEqual(record['operators']['gfx1201']['status'],'PASS')

    def test_teardown_error_cannot_bypass_worker_diagnostics(self):
        import amd_pixal3d_operator_guest_v3 as guest
        class Child:
            stdin=None
            def poll(self):return None
            def terminate(self):raise PermissionError('teardown denied')
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            leaf=Path(d);(leaf/'child-result.json').write_text(json.dumps(dict(classification='FAIL',error='worker audit denied')))
            record=dict(classification='FAIL',error='controller ownership error')
            guest.reap_child(record,Child());guest.harvest_worker(record,leaf)
            self.assertEqual(record['error'],'controller ownership error');self.assertTrue(record['child_exit_unverified'])
            self.assertEqual(record['worker_error'],'worker audit denied');self.assertEqual(record['classification'],'FAIL')

    def test_process_pidfd_uses_zero_flags_for_whole_process_exit_semantics(self):
        with patch.object(p.os,'pidfd_open',return_value=88,create=True) as opened:
            self.assertEqual(p.ProcFS().pidfd(123),88);opened.assert_called_once_with(123,0)

class FakeFS:
    def __init__(self,states=None,env_error=None,env=None):
        self.states=iter(states or ['S']*100);self.env_error=env_error;self.closed=[];self.signals=[];self.stat_fds=[];self.pids=[20];self.support=True;self.dead_value=False;self.dead_values=None;self.foreign_fd=None
        self.env=b'MODLY_PIXAL_PROOF_TOKEN='+b'a'*32+b'\0' if env is None else env;self.status=b'VmRSS:\t16 kB\n'
    def entries(self):return self.pids
    def open_proc(self,pid):return pid+100
    def stat(self,fd,pid):
        self.stat_fds.append(fd);value=next(self.states)
        if isinstance(value,BaseException):raise value
        return dict(pid=pid,pgid=10,start_ticks=30,state=value) if isinstance(value,str) else value
    def read(self,fd,name):
        if name=='environ':
            if self.env_error:raise self.env_error
            return b'MODLY_PIXAL_PROOF_TOKEN=foreign' if fd==self.foreign_fd else self.env
        return self.status
    def close(self,fd):self.closed.append(fd)
    def require_pidfds(self):
        if not self.support:raise p.OwnershipError('Unavailable')
    def pidfd(self,pid):return pid+200
    def dead(self,fd):return next(self.dead_values) if self.dead_values is not None else self.dead_value
    def signal(self,fd,sig):self.signals.append((fd,sig))

if __name__=='__main__':unittest.main()
