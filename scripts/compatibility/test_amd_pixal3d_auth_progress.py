"""Offline equivalence against 8bb9624 and bounded authentication diagnostics."""
import hashlib,io,json,os,sys,tempfile,unittest,zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import amd_pixal3d_wheel_payload as p
from test_amd_pixal3d_wheel_payload import fixture
TMP=None

# Frozen verify_manifest from reviewed commit 8bb96247. The immutable reader
# and mapper remain the real production helpers for both sides of equivalence.
ORIGINAL='''def verify_manifest(manifest,env,site,progress=None,trusted_paths=None):
    files=0;total=0;hint_errors=[];installed_record=None
    for name,expected in manifest['files'].items():
        path=installed_path(name,env,site)
        if trusted_paths is not None and path.is_relative_to(site):trusted_paths.add(path.relative_to(site).as_posix())
        with readonly(path) as stream:
            actual=stream_identity(stream,progress)
            if name==manifest['record']:
                installed_record=dict(path=name,**actual,pip_rewritten_metadata=True)
            elif actual!=expected:raise RuntimeError('Installed payload identity differs: '+name)
            else:files+=1;total+=actual['bytes']
            # Same read-only cache hint as V3; neither a write nor a recovery guarantee.
            if hasattr(os,'posix_fadvise'):
                try:os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                except OSError as exc:hint_errors.append(dict(file=name,error=str(exc)))
        if progress:progress()
    return dict(files_verified=files,bytes_verified=total,archive=manifest['archive'],archive_identity=manifest['archive_identity'],
        RECORD_discrepancies=manifest['RECORD_discrepancies'],installed_RECORD=installed_record,
        read_only_cache_hint_errors=hint_errors,physical_recovery_guaranteed=False)
'''
space={};exec(compile(ORIGINAL,'reviewed-8bb96247-verify_manifest','exec'),p.__dict__,space);original=space['verify_manifest']

def tracker(test,guard=None):
    test.assertTrue(hasattr(p,'AuthenticationProgress'),'Bounded authentication progress is missing')
    return p.AuthenticationProgress(guard)

class Equivalence(unittest.TestCase):
    def test_valid_bytes_hashes_callbacks_and_reduced_validation_work(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            archive,pin,env,site=fixture(Path(d));m=p.wheel_manifest(archive,pin,env,site);outputs=[];calls=[];callbacks=[]
            for fn in (original,p.verify_manifest):
                ticks=[]
                with patch.object(p,'plain',wraps=p.plain) as checked:
                    outputs.append(fn(m,env,site,progress=lambda:ticks.append(1)))
                    calls.append(checked.call_count);callbacks.append(len(ticks))
            self.assertEqual(outputs[0],outputs[1]);self.assertEqual(callbacks[0],callbacks[1])
            self.assertEqual(calls,[12,8]);self.assertEqual(outputs[1]['files_verified'],2)

    def test_same_errors_for_corruption_missing_and_unsafe_members(self):
        for mode in ('hash','missing','directory','unsafe'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(dir=TMP) as d:
                archive,pin,env,site=fixture(Path(d));m=p.wheel_manifest(archive,pin,env,site)
                if mode=='hash':(site/'tiny.py').write_bytes(b'qualified payloae')
                if mode=='missing':(site/'tiny.py').unlink()
                if mode=='directory':(site/'tiny.py').unlink();(site/'tiny.py').mkdir()
                if mode=='unsafe':m['files']={'../escape':dict(bytes=0,sha256='0'*64)}
                errors=[]
                for fn in (original,p.verify_manifest):
                    try:fn(m,env,site)
                    except (RuntimeError,ValueError,OSError) as exc:errors.append(type(exc))
                self.assertEqual(len(errors),2);self.assertIs(errors[0],errors[1])

    def test_same_errors_for_parent_redirection_and_real_hardlinks(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            archive,pin,env,site=fixture(Path(d));m=p.wheel_manifest(archive,pin,env,site);real=Path.is_symlink
            for redirect in (site,site/'tiny.py'):
                for fn in (original,p.verify_manifest):
                    with patch.object(Path,'is_symlink',lambda path:True if path==redirect else real(path)):
                        with self.assertRaises(ValueError):fn(m,env,site)
            os.link(site/'tiny.py',Path(d)/'alias')
            for fn in (original,p.verify_manifest):
                with self.assertRaises(ValueError):fn(m,env,site)

    def test_both_versions_reject_file_identity_change_before_reader_exit(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            archive,pin,env,site=fixture(Path(d));m=p.wheel_manifest(archive,pin,env,site);fstat=p.os.fstat
            for fn in (original,p.verify_manifest):
                calls=[]
                def changed(fd):
                    s=fstat(fd);calls.append(1);values={k:getattr(s,k) for k in ('st_dev','st_ino','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')}
                    if len(calls)>1:values['st_mtime_ns']+=1
                    return SimpleNamespace(**values)
                with patch.object(p.os,'fstat',side_effect=changed):
                    with self.assertRaisesRegex(RuntimeError,'File changed during read'):fn(m,env,site)

class Progress(unittest.TestCase):
    def test_chunk_bytes_precede_the_unchanged_callback_and_exception(self):
        data=b'x'*(p.BLOCK+17);seen=[];stop=TimeoutError('unchanged120-second guard');state=None
        def guard():
            seen.append(state.snapshot()['file_bytes_hashed'])
            if len(seen)==2:raise stop
        state=tracker(self,guard);state.enter('installed-files','installed','tiny.py','installed-sha256',package='tiny')
        with self.assertRaises(TimeoutError) as caught:p.stream_identity(io.BytesIO(data),state)
        self.assertIs(caught.exception,stop);self.assertEqual(seen,[1048576,1048593])
        row=state.snapshot();self.assertEqual(row['file'],'tiny.py');self.assertEqual(row['verified']['installed'],dict(files=0,bytes=0))

    def test_stage_durations_and_failure_freeze_exclude_cleanup(self):
        now=[10.0]
        with patch('amd_pixal3d_wheel_payload.time.monotonic',side_effect=lambda:now[0]):
            state=tracker(self);state.enter('archive-manifests','archive','tiny.whl','archive-sha256',package='tiny')
            now[0]=12;state.enter('installed-files','installed','tiny.py','installed-sha256')
            now[0]=15;state.finish('failed');row=state.snapshot();now[0]=100
            self.assertEqual(state.snapshot(),row)
        self.assertEqual(row['elapsed_seconds'],5);self.assertEqual(row['stage_seconds'],{'archive-manifests':2,'installed-files':3})
        self.assertEqual(row['status'],'failed');self.assertEqual(row['checkpoint'],'installed-sha256')

    def test_payload_and_installed_counters_distinguish_rewritten_RECORD(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            archive,pin,env,site=fixture(Path(d));ticks=[];state=tracker(self,lambda:ticks.append(1));state.package='tiny'
            m=p.wheel_manifest(archive,pin,env,site,state)
            (site/'tiny-1.dist-info/RECORD').write_bytes(b'pip rewritten metadata')
            result=p.verify_manifest(m,env,site,state);row=state.snapshot()
            self.assertEqual(row['verified']['archive'],dict(files=1,bytes=pin['bytes']))
            self.assertEqual(row['verified']['payload']['files'],3)
            self.assertEqual(row['verified']['installed'],dict(files=2,bytes=39))
            self.assertEqual(row['installed_RECORD_hashed'],dict(files=1,bytes=22))
            self.assertEqual(result['installed_RECORD']['sha256'],hashlib.sha256(b'pip rewritten metadata').hexdigest())

    def test_missing_and_hash_failure_retain_current_file_without_verifying_it(self):
        for mode in ('missing','hash'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(dir=TMP) as d:
                archive,pin,env,site=fixture(Path(d));m=p.wheel_manifest(archive,pin,env,site);state=tracker(self);state.package='tiny'
                if mode=='missing':(site/'tiny.py').unlink()
                else:(site/'tiny.py').write_bytes(b'qualified payloae')
                with self.assertRaises((FileNotFoundError,RuntimeError)):p.verify_manifest(m,env,site,state)
                row=state.snapshot();self.assertEqual(row['package'],'tiny');self.assertEqual(row['file'],'tiny.py')
                self.assertEqual(row['checkpoint'],'open-installed-readonly' if mode=='missing' else 'compare-installed-sha256')
                self.assertEqual(row['verified']['installed'],dict(files=0,bytes=0))

    def test_exit_identity_failure_does_not_increment_verified_counters(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            archive,pin,env,site=fixture(Path(d));m=p.wheel_manifest(archive,pin,env,site);state=tracker(self);calls=[];fstat=p.os.fstat
            def changed(fd):
                s=fstat(fd);calls.append(1);values={k:getattr(s,k) for k in ('st_dev','st_ino','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')}
                if len(calls)>1:values['st_mtime_ns']+=1
                return SimpleNamespace(**values)
            with patch.object(p.os,'fstat',side_effect=changed):
                with self.assertRaises(RuntimeError):p.verify_manifest(m,env,site,state)
            row=state.snapshot();self.assertEqual(row['checkpoint'],'installed-readonly-exit');self.assertEqual(row['verified']['installed'],dict(files=0,bytes=0))

    def test_zero_byte_payload_and_empty_manifest_keep_path_protections(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);archive,pin,env,site=fixture(root,files={'tiny.py':b''});state=tracker(self)
            m=p.wheel_manifest(archive,pin,env,site,state);p.verify_manifest(m,env,site,state)
            self.assertEqual(state.snapshot()['verified']['installed'],dict(files=1,bytes=0))
            m['files']={}
            with self.assertRaises(ValueError):p.verify_manifest(m,env,root,state)

    def test_progress_labels_are_bounded_and_no_history_grows_per_file(self):
        state=tracker(self)
        for index in range(2000):state.enter('installed-files','installed','x'*5000+str(index),'open-installed-readonly',package='tiny')
        row=state.snapshot();self.assertLessEqual(len(row['file']),4096);self.assertTrue(row['file_truncated']);self.assertEqual(row['file_sha256'],hashlib.sha256(('x'*5000+'1999').encode()).hexdigest())
        self.assertLess(len(json.dumps(row)),8192)

    def test_archive_structure_and_RECORD_failures_name_the_actual_checkpoint(self):
        for mode,point in [('structure','open-authenticated-ZIP'),('RECORD','validate-top-level-RECORD')]:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(dir=TMP) as d:
                root=Path(d);archive=root/'tiny-1-py3-none-any.whl'
                if mode=='structure':archive.write_bytes(b'Pinned but not a ZIP')
                else:
                    with zipfile.ZipFile(archive,'w') as z:z.writestr('tiny.py',b'code')
                pin=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest());state=tracker(self)
                with self.assertRaises((RuntimeError,zipfile.BadZipFile)):p.wheel_manifest(archive,pin,root,root,state)
                self.assertEqual(state.snapshot()['checkpoint'],point)

    def test_overlay_post_manifest_failure_is_not_reported_as_archive_complete(self):
        import amd_pixal3d_kitchen_overlay as overlay
        state=tracker(self);state.enter('overlay','archive',overlay.PIN['filename'],'overlay-authentication',package='packaging')
        with patch.object(overlay.email,'message_from_bytes',return_value={'Name':'wrong'}):
            with self.assertRaises(RuntimeError):overlay.authenticate(state)
        row=state.snapshot();self.assertEqual(row['stage'],'overlay');self.assertEqual(row['checkpoint'],'validate-overlay-runtime-metadata')

if __name__=='__main__':unittest.main()
