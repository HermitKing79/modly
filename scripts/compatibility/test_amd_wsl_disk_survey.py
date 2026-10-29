"""Offline fixtures only. Never start WSL or query live host RAM."""
import contextlib,importlib,io,json,os,stat,struct,sys,tempfile,unittest,uuid
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
TEMP=Path(__file__).resolve().parents[3]/'artifacts'
try:
    import amd_wsl_disk_survey as host
    import amd_wsl_disk_survey_guest as guest
except ModuleNotFoundError:
    host=guest=None

class Implementation(unittest.TestCase):
    def test_survey_handoff_exists(self):self.assertIsNotNone(host,'Survey owner is not implemented')

@unittest.skipIf(host is None,'Implementation not yet present')
class Policy(unittest.TestCase):
    def test_default_does_no_work(self):
        with patch.object(host,'run',side_effect=AssertionError('live')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(host.main([]),0)
    def test_approval_is_required_before_any_live_action(self):
        with patch.object(host,'run',side_effect=AssertionError('live')):
            with self.assertRaises(SystemExit):host.main(['--run','--commit','a'*40])
    def test_memory_policy_unchanged(self):
        self.assertEqual((host.GIB,host.START,host.LIVE,host.DISK,host.CAP),(1024**3,22*1024**3,12*1024**3,50*1024**3,512*1024**2))
    def test_low_first_reading_stops_without_sleep_or_launch(self):
        with self.assertRaises(RuntimeError):host.admit(lambda:dict(available_bytes=host.START-1),lambda _:self.fail('retry'),lambda _:None)
    def test_admission_has_three_reads_and_two_four_second_waits(self):
        events=[]
        host.admit(lambda:dict(available_bytes=host.START),lambda n:events.append(n),lambda _:None)
        self.assertEqual(events,[4,4])
    def test_final_gate_cannot_use_the_live_floor(self):
        with self.assertRaises(RuntimeError):host.require_memory(dict(available_bytes=host.START-1),host.START)
        host.require_memory(dict(available_bytes=host.LIVE),host.LIVE)
    def test_exact_command_and_no_generation_or_shell(self):
        token='a'*32;cmd=host.command(token)
        self.assertEqual(cmd[1:5],['-d','Modly-AMD-Ubuntu-24.04','-u','modly'])
        self.assertIn('-S',cmd);self.assertIn('MODLY_PIXAL_PROOF_TOKEN='+token,cmd)
        self.assertIn('--guest',cmd);self.assertIn('--cleanup',host.command(token,True))
        self.assertFalse(any(x in cmd for x in ('bash','sh','--terminate','--shutdown','--install')))
        with self.assertRaises(ValueError):host.command('a;whoami')
    def test_consumed_evidence_or_redirected_path_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            with patch.object(host,'CASE',Path(d)):
                with self.assertRaises(FileExistsError):host.require_fresh()
    def test_running_distro_stops_but_other_distro_is_not_terminated(self):
        with self.assertRaises(RuntimeError):host.require_stopped('Modly-AMD-Ubuntu-24.04\n'.encode('utf-16-le'))
        host.require_stopped('Unrelated\n'.encode('utf-16-le'))
    def test_bad_or_ambiguous_running_list_fails_closed(self):
        with self.assertRaises(ValueError):host.require_stopped(b'\x00')
    def test_guest_audit_rejects_mutations_and_neural_imports(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            audit=guest.audit(Path(d))
            audit('open',(str(Path(d)/'result.json'),'x',os.O_CREAT|os.O_EXCL|os.O_WRONLY))
            for event,args in [('open',('/home/modly/a','w',os.O_WRONLY)),('os.remove',('/tmp/a',-1)),('os.mkdir',('/tmp/a',0,-1)),('subprocess.Popen',('evil',[],None,None)),('import',('torch',None,None,None,None))]:
                with self.assertRaises((RuntimeError,ImportError)):audit(event,args)
    def test_receipts_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            with self.assertRaises(RuntimeError):guest.audit(Path(d))('open',(str(Path(d)/'result.json'),'w',os.O_WRONLY|os.O_TRUNC))
    def test_mount_parser_handles_spaces_and_bind_mounts(self):
        text='24 1 8:16 / / rw,relatime - ext4 /dev/sdb rw\n25 24 8:16 /a /mount\\040point rw - ext4 /dev/sdb rw\n'
        rows=guest.mounts(text);self.assertEqual(rows[1]['target'],'/mount point');self.assertEqual(rows[0]['source'],'/dev/sdb')
    def test_filesystem_numbers_keep_reserved_blocks_and_inodes_distinct(self):
        from types import SimpleNamespace
        row=guest.fs_numbers(SimpleNamespace(f_blocks=100,f_frsize=4096,f_bfree=20,f_bavail=10,f_files=80,f_ffree=30,f_favail=25))
        self.assertEqual((row['capacity_bytes'],row['used_bytes'],row['free_bytes'],row['available_bytes']),(409600,327680,81920,40960))
        self.assertEqual(row['inodes_available'],25)
    def test_permission_failure_is_incomplete_not_zero_usage(self):
        with patch.object(guest,'walk_usage',return_value=dict(complete=False,errors=[dict(error='PermissionError')],directories=[],entries=0)):
            result=guest.collect_usage([],0,lambda:None)
        self.assertFalse(result['complete']);self.assertEqual(result['errors'][0]['error'],'PermissionError')
    def test_production_usage_scope_is_only_the_approved_directories(self):
        with patch.object(guest,'walk_usage',return_value={}) as walk:
            guest.collect_usage([],1,lambda:None)
        self.assertEqual(walk.call_args.kwargs.get('roots'),('/home/modly','/opt','/usr','/var/cache','/var/log','/tmp'))
    def test_short_cleanup_self_deadline_precedes_host_cleanup_timeout(self):
        self.assertTrue(hasattr(guest,'wall_limit'),'cleanup wall deadline policy missing')
        self.assertLess(guest.wall_limit(True),20)
    def test_listing_and_cleanup_have_a_bounded_client_owner(self):
        self.assertTrue(hasattr(host,'bounded_client'),'file-backed bounded client required')

@unittest.skipIf(host is None,'Implementation not yet present')
class Metadata(unittest.TestCase):
    class FS:
        def __init__(self):
            from types import SimpleNamespace as N
            def s(ino,mode=stat.S_IFREG,dev=1):return N(st_dev=dev,st_ino=ino,st_mode=mode,st_blocks=1,st_size=100)
            self.nodes={'/':s(1,stat.S_IFDIR),'/file':s(2),'/link':s(3,stat.S_IFLNK),'/alias':s(2),
                '/other':s(4,stat.S_IFDIR,2),'/bind':s(5,stat.S_IFDIR),'/secret':s(6,stat.S_IFDIR),'/good':s(7,stat.S_IFDIR)}
            self.closed=[];self.opened=[]
        def root(self):return '/'
        def info(self,fd):return self.nodes[fd]
        @contextlib.contextmanager
        def entries(self,fd):
            from types import SimpleNamespace as N
            names=['file','alias','link','other','bind','secret','good'] if fd=='/' else []
            yield iter([N(name=n) for n in names])
        def child_info(self,fd,name):return self.nodes[fd.rstrip('/')+'/'+name]
        def child(self,fd,name):
            self.opened.append(name)
            if name=='secret':raise PermissionError('synthetic permission gap')
            return fd.rstrip('/')+'/'+name
        def close(self,fd):self.closed.append(fd)
    def test_same_filesystem_no_follow_mounts_hardlink_accounting_and_permission_gap(self):
        fs=self.FS();r=guest.walk_usage([dict(target='/'),dict(target='/bind')],1,lambda:None,fs=fs)
        self.assertFalse(r['complete']);self.assertIn('PermissionError',r['errors'][0]['error'])
        self.assertEqual(r['unique_nondirectory_inodes'],2)
        self.assertEqual(fs.opened,['secret','good']);self.assertEqual(fs.closed,['/good','/'])
        self.assertEqual(next(x for x in r['directories'] if x['path']=='/')['allocated_bytes'],4*512)
    def test_entry_limit_is_bounded_and_closes_root(self):
        fs=self.FS()
        r=guest.walk_usage([],1,lambda:None,fs=fs,entry_limit=1)
        self.assertTrue(r['fatal_stop']);self.assertIn('entry limit',r['errors'][0]['error'])
        self.assertEqual(fs.closed,['/'])
    def test_retained_directory_rows_are_bounded(self):
        from types import SimpleNamespace as N
        fs=self.FS()
        @contextlib.contextmanager
        def entries(fd):yield iter([N(name='good')]*200 if fd=='/' else [])
        fs.entries=entries
        r=guest.walk_usage([],1,lambda:None,fs=fs)
        self.assertLessEqual(len(r['directories']),64)
        self.assertGreater(r['directory_rows_omitted'],0)
    def test_directory_swap_is_rejected(self):
        fs=self.FS();original=fs.child
        def changed(fd,name):
            out=original(fd,name)
            from types import SimpleNamespace as N
            fs.nodes[out]=N(st_dev=1,st_ino=999,st_mode=stat.S_IFDIR,st_blocks=0,st_size=0)
            return out
        fs.child=changed
        r=guest.walk_usage([dict(target='/bind')],1,lambda:None,fs=fs)
        self.assertTrue(r['fatal_stop']);self.assertIn('identity/device changed',r['errors'][-1]['error'])
        self.assertIn('/good',fs.closed);self.assertIn('/',fs.closed)
    def test_vhd_virtual_file_length_and_physical_allocation_are_distinct(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            p=Path(d)/'synthetic.vhdx';blob=bytearray(2*1024**2);blob[:8]=b'vhdxfile';blob[196608:196612]=b'regi'
            struct.pack_into('<I',blob,196616,1)
            struct.pack_into('<16sQII',blob,196624,uuid.UUID('8b7ca206-4790-4b9a-b8fe-575f050f886e').bytes_le,1048576,1048576,0)
            blob[1048576:1048584]=b'metadata';struct.pack_into('<H',blob,1048586,1)
            struct.pack_into('<16sIIII',blob,1048608,uuid.UUID('2fa54224-cd1b-4876-b211-5dbed83bf4b8').bytes_le,65536,8,0,0)
            struct.pack_into('<Q',blob,1048576+65536,32*1024**3);p.write_bytes(blob)
            with patch.object(host,'allocated_size',return_value=12345):r=host.vhdx_metadata(p)
            self.assertEqual(r['virtual_capacity_bytes'],32*1024**3);self.assertEqual(r['container_file_length_bytes'],len(blob))
            self.assertEqual(r['Windows_allocated_storage_bytes_GetCompressedFileSizeW'],12345)
            p.write_bytes(b'not-a-vhdx')
            with self.assertRaisesRegex(ValueError,'identity'):host.vhdx_metadata(p)
    def test_original_completed_shape_hashes_authenticate_without_models(self):
        before=set(sys.modules);r=host.verify_inputs()
        self.assertEqual(len(r),8);self.assertEqual(r['shape/run/scratch/shape-latent-denormalized.npy']['bytes'],497792)
        self.assertFalse({'torch','triton','trellis2','flex_gemm'}.intersection(set(sys.modules)-before))
    def test_cpu_only_metadata_partial_facts_survive_inventory_failure(self):
        from types import SimpleNamespace as N
        facts={};v=N(f_blocks=100,f_frsize=4096,f_bfree=20,f_bavail=10,f_files=80,f_ffree=30,f_favail=25)
        with patch.object(guest,'bounded_text',return_value='24 1 8:16 / / rw - ext4 /dev/sdb rw\n'),patch.object(guest.os,'stat',return_value=N(st_dev=1)),patch.object(guest.os,'major',create=True,return_value=8),patch.object(guest.os,'minor',create=True,return_value=16),patch.object(guest.os,'statvfs',create=True,return_value=v),patch.object(guest.os,'uname',create=True,return_value=N(release='synthetic')),patch.object(guest,'collect_usage',side_effect=TimeoutError('fixture deadline')):
            with self.assertRaises(TimeoutError):guest.collect(lambda:None,facts)
        self.assertEqual(facts['filesystems']['/']['capacity_bytes'],409600)
    def test_hard_aggregate_capture_cap(self):
        from types import SimpleNamespace as N
        with tempfile.TemporaryDirectory(dir=TEMP) as d,patch.object(host,'CASE',Path(d)):
            cap=host.Capture('fixture',10);cap.start(N(stdout=io.BytesIO(b'x'*30),stderr=io.BytesIO(b'y'*30)))
            with self.assertRaisesRegex(RuntimeError,'output limit'):cap.finish()
            self.assertEqual(sum(p.stat().st_size for p in Path(d).iterdir()),10)

@unittest.skipIf(host is None,'Implementation not yet present')
class OwnedRouting(unittest.TestCase):
    def test_unattached_client_uses_only_its_Popen_handle_then_reaps(self):
        from unittest.mock import Mock
        from types import SimpleNamespace as N
        child=N(wait=Mock(side_effect=[host.subprocess.TimeoutExpired('owned',5),0]),poll=Mock(return_value=None),kill=Mock())
        self.assertTrue(hasattr(host,'reap_client'),'Handle-scoped fallback missing')
        self.assertTrue(host.reap_client(child,False));child.kill.assert_called_once();self.assertEqual(child.wait.call_count,2)
    def test_attached_client_cannot_use_unattached_fallback(self):
        from unittest.mock import Mock
        from types import SimpleNamespace as N
        child=N(wait=Mock(side_effect=host.subprocess.TimeoutExpired('owned',5)),poll=Mock(return_value=None),kill=Mock())
        self.assertTrue(hasattr(host,'reap_client'),'Handle-scoped fallback missing')
        with self.assertRaises(RuntimeError):host.reap_client(child,True)
        child.kill.assert_not_called()
    def auxiliary(self,mode):
        from types import SimpleNamespace as N
        from unittest.mock import Mock
        job=N(handle=123,api=N(SetInformationJobObject=Mock(return_value=True)),attach=Mock(),finish=Mock(return_value={'owned_tree_exit_verified':True,'forced_job_cleanup':mode=='forced'}))
        child=N(stdout=io.BytesIO(b'x'*9000 if mode=='overflow' else b''),stderr=io.BytesIO(),poll=Mock(return_value=None if mode=='timeout' else 0),returncode=0,wait=Mock(return_value=0))
        clock=N(monotonic=Mock(side_effect=[0,100]),sleep=lambda _:None)
        with tempfile.TemporaryDirectory(dir=TEMP) as d,patch.object(host,'CASE',Path(d)),patch.object(host,'OwnedJob',return_value=job),patch.object(host.subprocess,'Popen',return_value=child) as spawn,patch.object(host,'time',clock):
            with self.assertRaises((RuntimeError,TimeoutError)):host.bounded_client(host.command('a'*32,True),'cleanup',20)
            self.assertTrue(job.finish.called);self.assertTrue(child.wait.called);self.assertEqual(spawn.call_count,1)
            self.assertLessEqual(sum(p.stat().st_size for p in Path(d).glob('*.txt')),8192)
    def test_auxiliary_timeout_is_owned_and_not_retried(self):self.auxiliary('timeout')
    def test_auxiliary_output_overflow_is_hard_capped_and_owned(self):self.auxiliary('overflow')
    def test_auxiliary_forced_tree_cleanup_cannot_pass(self):self.auxiliary('forced')
    def exercise(self,mode):
        from types import SimpleNamespace as N
        from unittest.mock import Mock
        class Input(io.BytesIO):
            written=b''
            def write(self,b):self.written+=b;return super().write(b)
        token='a'*32;owner=dict(pid=10,pgid=10,start_ticks=100,token=token)
        job=N(handle=123,api=N(SetInformationJobObject=Mock(return_value=True),QueryInformationJobObject=Mock(return_value=True)),
            attach=Mock(side_effect=RuntimeError('attach fixture') if mode=='attach' else None),finish=Mock(return_value={'owned_tree_exit_verified':True,'forced_job_cleanup':False}),close=Mock())
        child=N(pid=99,stdin=Input(),stdout=io.BytesIO(),stderr=io.BytesIO(),poll=Mock(side_effect=[None,None,0]),returncode=0,wait=Mock(return_value=0))
        receipts={'ready.json':dict(owner=owner,pidfd_prerequisite='PASS',process_memory_limit_bytes=host.CAP),'ownership.json':owner,
            'result.json':dict(classification='SURVEY_COMPLETE',process_identity=owner,peak_RSS_bytes=100,seconds=1,neural_execution=False,guest_file_writes=False,inventory_started=True)}
        floors=[]
        def live(floor=host.LIVE):
            floors.append(floor)
            if (mode=='startup' and len(floors)==2) or (mode=='GO' and len(floors)==3):raise RuntimeError('22GiB fixture denial')
            return dict(available_bytes=floor)
        with tempfile.TemporaryDirectory(dir=TEMP) as d,patch.object(host,'CASE',Path(d)),patch.object(host,'allocated_size',return_value=1),patch.object(host,'OwnedJob',return_value=job),patch.object(host.subprocess,'Popen',return_value=child) as spawn,patch.object(host,'read_receipt',side_effect=lambda name,limit=0:receipts[name]),patch.object(host.time,'sleep',lambda _:None):
            (Path(d)/'ownership.json').write_text(json.dumps(owner))
            runner=host.Runner(token);runner.live=live;runner.cleanup_guest=Mock(return_value=dict(natural_exit=True))
            result=runner.invoke();self.assertEqual(spawn.call_count,1);self.assertTrue(child.stdin.closed)
            self.assertTrue(job.finish.called);self.assertTrue(runner.cleanup_guest.called)
        return result,floors,child.stdin.written
    def test_startup_decline_stops_before_GO_and_cleans_owned_work(self):
        result,floors,writes=self.exercise('startup')
        self.assertEqual(floors,[host.START,host.START]);self.assertEqual(writes,b'');self.assertEqual(result['classification'],'FAIL')
    def test_final_GO_decline_stops_without_retry(self):
        result,floors,writes=self.exercise('GO')
        self.assertEqual(floors,[host.START]*3);self.assertEqual(writes,b'');self.assertFalse(result['GO_sent'])
    def test_job_attach_failure_cannot_authorise_guest(self):
        result,_,writes=self.exercise('attach');self.assertEqual(writes,b'');self.assertEqual(result['classification'],'FAIL')
    def test_exact_GO_EOF_then_live_floor_and_natural_cleanup(self):
        result,floors,writes=self.exercise('pass')
        self.assertEqual(floors,[host.START]*3+[host.LIVE]);self.assertEqual(writes,b'GO\n');self.assertEqual(result['classification'],'SURVEY_COMPLETE')

if __name__=='__main__':unittest.main()
