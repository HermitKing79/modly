"""Offline audit evidence and unchanged-denial tests; no native/model imports."""
import json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import amd_pixal3d_operator_ops_v3 as ops
from amd_pixal3d_operator_audit_v3 import AuditLog
TMP=None

class AuditTests(unittest.TestCase):
    def test_denied_event_records_path_phase_and_actual_caller_without_allowing_mutation(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);path=root/'keep';path.write_bytes(b'unchanged');log=AuditLog(root/'audit.jsonl');log.phase='Kitchen-HIP-load'
            try:
                with self.assertRaises(RuntimeError):ops.policy(root,root,log)('os.remove',(str(path),-1))
            finally:log.close()
            row=json.loads((root/'audit.jsonl').read_text());self.assertEqual(row['event'],'os.remove');self.assertEqual(row['arguments'],[str(path),-1])
            self.assertEqual(row['phase'],'Kitchen-HIP-load');self.assertFalse(row['operation_allowed']);self.assertEqual(path.read_bytes(),b'unchanged')
            self.assertTrue(any(f['function'].startswith('test_denied_event') for f in row['caller_frames']))

    def test_all_six_destructive_event_types_still_fail_inside_and_outside_scratch(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);audit=ops.policy(root/'scratch',root/'case')
            for directory in (root/'scratch',root/'unrelated'):
                for event in ('os.remove','os.rename','os.rmdir','os.link','os.symlink','os.truncate'):
                    with self.subTest(event=event,directory=directory),self.assertRaises(RuntimeError):audit(event,(str(directory/'a'),str(directory/'b'),-1,-1))

    def test_original_scratch_creation_is_allowed_and_protected_writes_remain_denied(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);scratch=root/'scratch';case=root/'case';audit=ops.policy(scratch,case)
            for directory in (scratch,case):audit('open',(str(directory/'new'),'wb',os.O_WRONLY));audit('os.mkdir',(str(directory/'newdir'),0o700,-1))
            for directory in ('V2','TRELLIS','cache','unrelated'):
                with self.assertRaises(RuntimeError):audit('open',(str(root/directory/'file'),'wb',os.O_WRONLY))
                with self.assertRaises(RuntimeError):audit('os.mkdir',(str(root/directory/'new'),0o700,-1))

    def test_symlink_or_parent_escape_remains_rejected(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);audit=ops.policy(root,root/'case')
            with patch.object(ops.c,'plain',side_effect=ValueError('Redirected path denied')):
                with self.assertRaises(ValueError):audit('open',(str(root/'alias/file'),'wb',os.O_WRONLY))
                with self.assertRaises(ValueError):audit('os.mkdir',(str(root/'alias/new'),0o700,-1))

    def test_existing_evidence_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            path=Path(d)/'audit.jsonl';path.write_bytes(b'history')
            with self.assertRaises(FileExistsError):AuditLog(path)
            self.assertEqual(path.read_bytes(),b'history')

    def test_fd_based_denial_records_descriptor_identity_without_mutating_file(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);path=root/'keep';path.write_bytes(b'unchanged');log=AuditLog(root/'audit.jsonl')
            try:
                with path.open('rb') as f:
                    with self.assertRaises(RuntimeError):ops.policy(root,root,log)('os.truncate',(f.fileno(),0))
            finally:log.close()
            row=json.loads((root/'audit.jsonl').read_text());self.assertTrue(row['descriptors'][0]['identity_stable'])
            self.assertEqual(row['descriptors'][0]['inode'],path.stat().st_ino);self.assertEqual(path.read_bytes(),b'unchanged')

    def test_close_is_serialized_and_closed_logger_never_writes_a_reused_descriptor(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);log=AuditLog(root/'audit.jsonl');real_close=os.close
            def close(fd):self.assertTrue(log.lock.locked());real_close(fd)
            try:
                with patch('amd_pixal3d_operator_audit_v3.os.close',side_effect=close):log.close()
            finally:
                if log.fd is not None:real_close(log.fd);log.fd=None
            with patch('amd_pixal3d_operator_audit_v3.os.write',side_effect=AssertionError('closed descriptor reused')):
                with self.assertRaises(RuntimeError):log('os.remove',(str(root/'keep'),-1))

    def test_evidence_failure_and_record_bound_never_grant_the_operation(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);log=AuditLog(root/'audit.jsonl');audit=ops.policy(root,root,log)
            try:
                with patch('amd_pixal3d_operator_audit_v3.os.write',side_effect=OSError('disk failure')):
                    with self.assertRaises(OSError):audit('os.remove',(str(root/'keep'),-1))
                for _ in range(32):
                    with self.assertRaises(RuntimeError):audit('os.remove',(str(root/'keep'),-1))
                with self.assertRaises(RuntimeError):audit('os.remove',(str(root/'keep'),-1))
                self.assertEqual(log.records,32)
            finally:log.close()

    def test_instrumented_import_and_subprocess_prohibitions_remain(self):
        audit=ops.policy(Path('scratch'),Path('case'))
        for name in ('triton','comfy_kitchen.backends.cuda._C','pip','transformers'):
            with self.assertRaises(ImportError):audit('import',(name,None,None,None,None))
        with self.assertRaises(RuntimeError):audit('subprocess.Popen',('foreign',[],None,None))

    def test_live_operator_entry_stops_before_admission_or_child_launch(self):
        import amd_pixal3d_operator_host_v3 as host
        with patch.object(host.c,'admit_operator',side_effect=AssertionError('live admission')),patch.object(host.subprocess,'Popen',side_effect=AssertionError('child launch')):
            with self.assertRaisesRegex(RuntimeError,'NO_GO'):host.run('a'*40,approved=True)

def real_tempfile_fixture(path):
    """Explicit harmless CPU control; never an operator/provisioning route."""
    root=ops.c.plain(path);root.relative_to(ops.c.plain(ops.c.ROOT.parent/'artifacts'))
    root.mkdir(exist_ok=False);scratch=root/'scratch';scratch.mkdir()
    os.environ['TMPDIR']=str(scratch);tempfile.tempdir=None
    log=AuditLog(root/'audit.jsonl');log.phase='stdlib-tempfile-control'
    sys.addaudithook(ops.policy(root,root,log))
    try:
        try:tempfile.gettempdir()
        except RuntimeError as exc:
            if 'Destructive mutation forbidden' not in str(exc):raise
        else:raise AssertionError('Expected original unlink prohibition')
    finally:log.close()
    row=json.loads((root/'audit.jsonl').read_text())
    assert row['event']=='os.remove' and row['operation_allowed'] is False
    assert any(f['function']=='_get_default_tempdir' for f in row['caller_frames'])
    assert Path(row['arguments'][0]).is_relative_to(scratch)
    assert Path(row['arguments'][0]).exists() # the actual rejected unlink did not run
    assert not ({'torch','triton','comfy_kitchen'}&sys.modules.keys())
    result=dict(classification='CPU_CONTROL_DENIAL_CAPTURED',event=row['event'],operation_allowed=False,
        caller='_get_default_tempdir',historical_trigger_proven=False,GPU_operations=0)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--real-audit-fixture':real_tempfile_fixture(sys.argv[2])
    else:unittest.main()
