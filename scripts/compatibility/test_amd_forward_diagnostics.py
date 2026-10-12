"""Focused CPU diagnostic tests. No torch, models, GPU or cached-code execution."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parent))

try:
    import amd_forward_diagnostics as diagnostics
except ModuleNotFoundError:
    diagnostics=None


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(diagnostics,'Missing durable host diagnostic interface')

    def test_entry_is_emitted_before_call_and_return_does_not_retain_output(self):
        rows=[];clock=iter([10.,11.,12.,15.,16.])
        trace=diagnostics.Trace(rows.append,clock=lambda:next(clock))
        result=object()
        def call():
            self.assertEqual(json.loads(rows[0][len(diagnostics.MARKER):])['event'],'enter')
            return result
        self.assertIs(trace.call('module:input_layer',call),result)
        parsed=[json.loads(row[len(diagnostics.MARKER):]) for row in rows]
        self.assertEqual([row['event'] for row in parsed],['enter','return'])
        self.assertEqual(parsed[1]['completion'],'host-return-only')
        self.assertGreater(parsed[1]['duration_seconds'],0)

    def test_failure_emits_error_and_never_retries(self):
        rows=[];trace=diagnostics.Trace(rows.append);calls=[]
        def fail(): calls.append(1);raise RuntimeError('native call failed')
        with self.assertRaisesRegex(RuntimeError,'native call failed'):trace.call('aten:convolution',fail)
        self.assertEqual(calls,[1])
        self.assertEqual(json.loads(rows[-1][len(diagnostics.MARKER):])['event'],'error')

    def test_journal_is_exclusive_and_each_record_is_durable_before_return(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]) as tmp:
            path=Path(tmp)/'trace.jsonl'
            with diagnostics.Journal(path) as journal:
                trace=diagnostics.Trace(journal.consume)
                trace.emit('memory','before-forward',allocated_bytes=123)
                saved=[json.loads(row) for row in path.read_text().splitlines()]
                self.assertEqual(saved[0]['allocated_bytes'],123)
                with self.assertRaises(FileExistsError):diagnostics.Journal(path)

    def test_invalid_or_nonfinite_and_unbounded_records_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]) as tmp:
            with diagnostics.Journal(Path(tmp)/'trace') as journal:
                for line in ('invalid',diagnostics.MARKER+'{}',diagnostics.MARKER+'{"event":"memory","name":"x","seq":1,"elapsed_seconds":NaN}'):
                    with self.assertRaises(ValueError): journal.consume(line)
            trace=diagnostics.Trace(lambda line:None,limit=1)
            trace.emit('memory','first')
            with self.assertRaises(RuntimeError):trace.emit('memory','second')

    def test_stalled_journal_write_cannot_block_supervisor_deadline(self):
        release=threading.Event();entered=threading.Event()
        def stalled_fsync(fd):entered.set();release.wait()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]) as tmp:
            with patch.object(diagnostics.os,'fsync',side_effect=stalled_fsync):
                journal=diagnostics.Journal(Path(tmp)/'trace')
                try:
                    began=time.monotonic()
                    with self.assertRaises(TimeoutError):
                        diagnostics.Trace(lambda line:journal.consume(line,timeout=.02)).emit('stage','forward')
                    self.assertTrue(entered.is_set())
                    self.assertLess(time.monotonic()-began,.5)
                    self.assertFalse(journal.close(timeout=.01))
                finally:release.set();journal.close(timeout=1.)

    def test_writer_error_is_fail_closed_and_acknowledged(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]) as tmp:
            with patch.object(diagnostics.os,'fsync',side_effect=OSError('disk failure')):
                journal=diagnostics.Journal(Path(tmp)/'trace')
                try:
                    with self.assertRaisesRegex(RuntimeError,'disk failure'):
                        diagnostics.Trace(journal.consume).emit('stage','forward')
                finally:self.assertTrue(journal.close())

    def test_live_owned_cpu_process_is_sampled_and_exits_without_gpu_imports(self):
        child=subprocess.Popen([sys.executable,'-B','-I','-c','import time; time.sleep(3)'])
        try:
            observer=diagnostics.ProcessActivity(child.pid)
            rows=observer.sample()['processes']
            self.assertEqual([row['pid'] for row in rows],[child.pid])
            self.assertGreaterEqual(rows[0]['cpu_seconds'],0.)
            self.assertGreaterEqual(rows[0]['cpu_delta_seconds'],0.)
            self.assertNotIn('torch',sys.modules)
        finally:
            child.terminate();child.wait(timeout=2)

    def test_activity_tracks_owned_descendants_and_rejects_pid_reuse(self):
        class Process:
            def __init__(self,pid,birth=10.):self.pid=pid;self.birth=birth;self.cpu=1.
            def create_time(self):return self.birth
            def cpu_times(self):return type('Times',(),{'user':self.cpu,'system':.5})()
            def children(self,recursive=True):return [descendant]
            def name(self):return 'compiler.exe'
        root=Process(10);descendant=Process(11)
        module=type('Module',(),{'__version__':'7.2.2','__file__':__file__,'NoSuchProcess':ProcessLookupError,'Process':lambda pid:root})
        observer=diagnostics.ProcessActivity(10,module=module)
        first=observer.sample();descendant.cpu=2.
        second=observer.sample()
        self.assertEqual([row['pid'] for row in first['processes']],[10,11])
        self.assertEqual(second['processes'][1]['cpu_delta_seconds'],1.)
        root.birth=99.
        with self.assertRaises(RuntimeError):observer.sample()

    def test_monitor_failure_is_reported_without_claiming_activity(self):
        class Process:
            pid=10
            def create_time(self):return 1.
            def children(self,recursive=True):raise PermissionError('denied')
        module=type('Module',(),{'__version__':'7.2.2','__file__':__file__,'NoSuchProcess':ProcessLookupError,'Process':lambda pid:Process()})
        observer=diagnostics.ProcessActivity(10,module=module)
        with self.assertRaises(PermissionError):observer.sample()

    def test_root_exit_between_identity_and_tree_query_is_normal(self):
        class Process:
            pid=10
            def create_time(self):return 1.
            def children(self,recursive=True):raise ProcessLookupError('exited')
        module=type('Module',(),{'__version__':'7.2.2','__file__':__file__,'NoSuchProcess':ProcessLookupError,'Process':lambda pid:Process()})
        observer=diagnostics.ProcessActivity(10,module=module)
        result=observer.sample()
        self.assertTrue(result['root_exited'])
        self.assertEqual(result['processes'],[])

    def test_cache_audit_output_cannot_touch_historical_artifacts(self):
        import amd_forward_cache_audit as cache
        with patch.object(Path,'read_text',side_effect=AssertionError('must reject before reading')):
            for path in (cache.OLD/'run/scratch/new.json',cache.ROOT/'unapproved.json'):
                with self.assertRaises(ValueError):cache.audit(path)

    def test_activity_monitor_startup_failure_and_stalled_queries_are_bounded(self):
        def fail(pid):raise PermissionError('denied')
        monitor=diagnostics.ActivityMonitor(10,lambda line:None,activity_factory=fail)
        with self.assertRaises(RuntimeError):monitor.start(timeout=.05)
        self.assertTrue(monitor.close())
        stalled=__import__('threading').Event()
        monitor=diagnostics.ActivityMonitor(10,lambda line:None,activity_factory=lambda pid:stalled.wait())
        try:
            with self.assertRaises(RuntimeError):monitor.start(timeout=.01)
            self.assertFalse(monitor.close(timeout=.01))
        finally:stalled.set();monitor.thread.join(timeout=1)

    def test_key_modules_and_clock_freshness_do_not_expand_computation(self):
        for name,kind in (('','SparseStructureDecoder'),('input_layer','Conv3d'),('blocks.2','UpsampleBlock3d')):
            self.assertTrue(diagnostics.key_module(name,kind))
        self.assertFalse(diagnostics.key_module('other','UnapprovedModel'))
        monitor=diagnostics.ActivityMonitor(10,lambda line:None)
        monitor.last_sample=time.monotonic()-10
        with self.assertRaises(RuntimeError):monitor.require_healthy()


if __name__=='__main__':unittest.main()
