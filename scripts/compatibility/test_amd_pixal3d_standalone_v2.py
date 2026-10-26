"""CPU/mocked v2 tests. Real WSL, admission, network and Torch are forbidden."""
import io,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import amd_pixal3d_proof_common_v2 as c
import amd_pixal3d_standalone_v2 as host
import amd_pixal3d_memory_v2 as memory
import amd_pixal3d_guest_v2 as guest
G=1024**3
TMP=Path(__file__).resolve().parents[3]/'artifacts/amd-pixal3d-standalone-preparation-20261026-v2'

class Cache(unittest.TestCase):
    def fixture(self,root):
        import hashlib
        (root/'downloads').mkdir();(root/'sources').mkdir()
        (root/'downloads/tiny.whl').write_bytes(b'archive');(root/'sources/tiny.py').write_bytes(b'source')
        def row(name,data):return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),**name)
        return dict(cache_root=str(root),packages=[row(dict(filename='tiny.whl'),b'archive')],sources=[row(dict(path='tiny.py'),b'source')])
    def test_cache_is_read_only_and_no_copy_or_download_is_needed(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            cfg=self.fixture(Path(d));before={str(p):p.stat().st_mtime_ns for p in Path(d).rglob('*')}
            result=c.verify_cache(cfg)
            self.assertEqual(result['files_verified'],2)
            self.assertEqual(before,{str(p):p.stat().st_mtime_ns for p in Path(d).rglob('*')})
    def test_missing_or_invalid_cache_fails_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            cfg=self.fixture(Path(d));p=Path(d)/'downloads/tiny.whl';p.write_bytes(b'bad')
            with self.assertRaises(RuntimeError):c.verify_cache(cfg)
            self.assertEqual(p.read_bytes(),b'bad');p.unlink()
            with self.assertRaises(FileNotFoundError):c.verify_cache(cfg)
    def test_cache_policy_rejects_writes_rename_delete_and_metadata_changes(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);p=root/'x';audit=c.cache_policy(root)
            audit('open',(str(p),'rb',0))
            for event,args in [('open',(str(p),'wb',os.O_WRONLY)),('os.remove',(str(p),-1)),('os.rename',(str(p),str(root/'y'),-1,-1)),('os.utime',(str(p),None,None,-1)),('os.mkdir',(str(root/'sub'),0,-1))]:
                with self.assertRaises(RuntimeError):audit(event,args)
    def test_v2_keeps_all_original_runtime_pins_and_guards(self):
        import amd_pixal3d_proof_common as old
        v1=old.lock();v2=c.lock()
        for name in ('guards','packages','sources','runtime','provision_seconds','operator_seconds','cleanup_seconds','guest_growth_limit_bytes','D_disk_floor_bytes','guest_disk_floor_bytes'):
            self.assertEqual(v1[name],v2[name],name)
        self.assertEqual(v2['cache_root'],'D:/AI/Modly-Pixal3D-Proof-v1')
        self.assertEqual(v2['guest_root'],'/home/modly/pixal3d-native-proof-v2')
        self.assertFalse(v2['downloads_allowed']);self.assertFalse(v2['guards']['preferred_is_mandatory'])

class Startup(unittest.TestCase):
    def owner(self,root):
        r=host.Runner.__new__(host.Runner);r.case=root;r.startup_active=False;r.memory_log=root/'memory.jsonl'
        r.cfg=c.lock();r.token='a'*32;r.vhd_before=0;r.vhd=root/'vhd';r.vhd.write_bytes(b'x')
        return r
    def test_first_startup_crossing_stops_before_GO(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            r=self.owner(Path(d));r.startup_active=True
            with patch.object(r,'memory',return_value=dict(available_bytes=22*G-1)),patch.object(host.shutil,'disk_usage',return_value=type('D',(),dict(free=100*G))()),patch.object(host,'allocated_size',return_value=0):
                with self.assertRaisesRegex(RuntimeError,'22 GiB'):r.live()
            self.assertLess(json.loads(r.memory_log.read_text())['available_bytes'],22*G)
    def test_startup_exact22_passes_24_is_not_required(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            r=self.owner(Path(d));r.startup_active=True
            with patch.object(r,'memory',return_value=dict(available_bytes=22*G)),patch.object(host.shutil,'disk_usage',return_value=type('D',(),dict(free=100*G))()),patch.object(host,'allocated_size',return_value=0):
                self.assertEqual(r.live()['available_bytes'],22*G)
    def test_VM_observation_is_explicitly_unavailable_not_zero_on_access_error(self):
        row=memory.process_observation(dict(pid=42,name='VmmemWSL'),lambda _:(_ for _ in ()).throw(PermissionError('denied')))
        self.assertEqual(row['status'],'UNAVAILABLE');self.assertNotIn('working_set_bytes',row)
        self.assertEqual(row['scope'],'shared_WSL_VM_candidate_not_distro_attribution')
    def test_client_is_never_labelled_VM_memory(self):
        row=memory.process_observation(dict(pid=3,name='wsl.exe'),lambda _:dict(working_set_bytes=123))
        self.assertEqual(row['scope'],'Windows_client_only_not_VM')
    def test_linux_meminfo_parsing_keeps_distinct_available_and_free(self):
        row=memory.parse_meminfo('MemTotal: 100 kB\nMemFree: 20 kB\nMemAvailable: 60 kB\nCached: 30 kB\nHugePages_Total: 0\n')
        self.assertEqual(row['MemAvailable'],61440);self.assertEqual(row['MemFree'],20480)
        with self.assertRaises(ValueError):memory.parse_meminfo('MemFree: 1 kB\n')

class Invocation(unittest.TestCase):
    def exercise(self,values,timeout=False,ready=True,after_GO_running=False,attach_failure=False):
        import amd_forward_owned_job as job_module
        import subprocess
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);r=Startup().owner(root);values=iter(values);calls=[]
            class Pipe:
                closed=False
                data=b''
                def write(self,data):self.data+=data
                def close(self):self.closed=True
            class Child:
                stdin=Pipe();returncode=0
                waits=0
                polled_after_GO=False
                killed=False
                def poll(self):
                    if self.stdin.data:
                        if after_GO_running and not self.polled_after_GO:self.polled_after_GO=True;return None
                        return 0
                    return None
                def wait(self,timeout=None):
                    calls.append('client_wait')
                    self.waits+=1
                    if attach_failure and not self.killed:raise subprocess.TimeoutExpired('unattached client',timeout)
                    if timeout is not None and self.waits==1 and timed_out:raise subprocess.TimeoutExpired('synthetic WSL client',timeout)
                    return 0
                def kill(self):calls.append('kill_owned_unattached_client');self.killed=True
            child=Child();timed_out=timeout
            class API:
                def SetInformationJobObject(self,*args):return True
                def QueryInformationJobObject(self,handle,kind,pointer,*args):pointer._obj.PeakProcessMemory=65536;return True
            class Job:
                api=API();handle=42
                def attach(self,child):
                    if attach_failure:raise OSError('synthetic Job attach failure')
                    calls.append('attached')
                def close(self):pass
                def finish(self):return dict(forced_job_cleanup=False,owned_tree_exit_verified=True)
            def spawn(*args,**kwargs):
                leaf=root/'provision';owner=dict(pid=42,pgid=42,start_ticks=1,token='a'*32,stage='provision')
                c.save(leaf/'ownership.json',owner)
                if ready:c.save(leaf/'ready.json',dict(owner=owner))
                c.save(leaf/'result.json',dict(classification='PASS'))
                return child
            def cleanup(stage):calls.append('owned_cleanup');return dict(natural_exit=True,group_empty=True)
            with patch.object(job_module,'OwnedJob',Job),patch.object(host.subprocess,'Popen',spawn),patch.object(r,'memory',side_effect=lambda:dict(available_bytes=next(values))),patch.object(host.shutil,'disk_usage',return_value=type('D',(),dict(free=100*G))()),patch.object(host,'allocated_size',return_value=0),patch.object(host.telemetry,'windows_processes',return_value=dict(status='UNAVAILABLE')),patch.object(r,'cleanup_guest',side_effect=cleanup),patch.object(host.time,'sleep'):
                try:r.invoke('provision')
                except RuntimeError:pass
            receipt=json.loads((root/'provision/host-result.json').read_text())
            return child.stdin.data,receipt,calls
    def test_actual_invocation_blocks_GO_on_first_startup_crossing(self):
        sent,row,calls=self.exercise([23*G,22*G-1,23*G])
        self.assertEqual(sent,b'');self.assertFalse(row['GO_sent']);self.assertEqual(row['startup_classification'],'FAIL')
        self.assertIn('owned_cleanup',calls)
    def test_actual_invocation_preserves_final22_gate(self):
        sent,row,_=self.exercise([23*G,23*G,22*G-1,23*G])
        self.assertEqual(sent,b'');self.assertFalse(row['GO_sent'])
    def test_actual_invocation_sends_one_GO_when_exact22_passes(self):
        sent,row,_=self.exercise([22*G]*5)
        self.assertEqual(sent,b'GO\n');self.assertEqual(row['classification'],'PASS');self.assertEqual(row['startup_classification'],'PASS')
    def test_owned_cleanup_runs_even_if_client_natural_exit_wait_times_out(self):
        sent,row,calls=self.exercise([23*G,22*G-1,23*G],timeout=True)
        self.assertEqual(sent,b'');self.assertIn('owned_cleanup',calls)
        self.assertEqual(row['classification'],'FAIL')
    def test_live_floor_failure_after_GO_cleans_Linux_before_waiting_for_client(self):
        sent,row,calls=self.exercise([23*G]*4+[12*G-1,23*G],after_GO_running=True)
        self.assertEqual(sent,b'GO\n');self.assertEqual(row['classification'],'FAIL')
        self.assertLess(calls.index('owned_cleanup'),calls.index('client_wait'))
    def test_failed_Job_attachment_still_cleans_exact_owned_Windows_client(self):
        sent,row,calls=self.exercise([23*G,23*G],attach_failure=True)
        self.assertEqual(sent,b'');self.assertIn('kill_owned_unattached_client',calls);self.assertEqual(row['classification'],'FAIL')

class Ownership(unittest.TestCase):
    def test_v2_cleanup_rejects_foreign_or_reused_processes(self):
        owner=dict(pid=1,pgid=1,start_ticks=123,token='a'*32)
        with self.assertRaises(RuntimeError):guest.owned_members(owner,[dict(owner,token='b'*32)])
        with self.assertRaises(RuntimeError):guest.owned_members(owner,[dict(owner,start_ticks=124)])
    def test_existing_guest_root_fails_before_GO_read_or_child_start(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            case=Path(d);(case/'provision').mkdir();root=case/'guest-already-used';root.mkdir()
            cfg=dict(c.lock(),guest_root=str(root));owner=dict(pid=42,pgid=42,start_ticks=1,token='a'*32)
            class NoRead:
                def read(self,*args):raise AssertionError('Existing root must fail before reading GO')
            with patch.object(guest,'validate_case',return_value=case),patch.object(c,'lock',return_value=cfg),patch.dict(os.environ,MODLY_PIXAL_PROOF_TOKEN='a'*32),patch.object(guest.os,'getpid',return_value=42),patch.object(guest.os,'getpgrp',return_value=42,create=True),patch.object(guest,'process_rows',return_value=[owner]),patch.object(guest,'linux_memory',return_value=dict(meminfo_bytes={})),patch.object(guest.sys,'stdin',type('Input',(),dict(buffer=NoRead()))()):
                self.assertEqual(guest.controller('provision',str(case),'a'*32),2)
            result=json.loads((case/'provision/result.json').read_text())
            self.assertFalse(result['child_started']);self.assertIn('already exists',result['error']);self.assertFalse((case/'provision/ready.json').exists())

class Leases(unittest.TestCase):
    def test_real_Windows_read_lease_blocks_writes_to_owned_synthetic_cache(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            cfg=Cache().fixture(Path(d));p=Path(d)/'downloads/tiny.whl'
            with c.CacheReadLease(cfg):
                self.assertEqual(p.read_bytes(),b'archive')
                with self.assertRaises(PermissionError):p.write_bytes(b'mutation')
            self.assertEqual(p.read_bytes(),b'archive')

class Routing(unittest.TestCase):
    def test_default_entrypoints_do_no_work(self):
        with patch.object(host,'run',side_effect=AssertionError('live')),patch.object(guest,'controller',side_effect=AssertionError('WSL')):
            self.assertEqual(host.main([]),0);self.assertEqual(guest.main([]),0)
    def test_wsl_routing_is_exact_new_worker_no_shell(self):
        cmd=host.wsl_command('provision','/mnt/c/Users/test/case','a'*32)
        self.assertIn('--exec',cmd);self.assertTrue(any(str(s).endswith('amd_pixal3d_guest_v2.py') for s in cmd))
        self.assertEqual(cmd[1:5],['-d','Modly-AMD-Ubuntu-24.04','-u','modly']);self.assertNotIn('sh',cmd)
    def test_reporting_preserves_separate_admission_startup_provision_and_HIP(self):
        result=host.initial_summary('a'*40,Path('case'))
        self.assertEqual(set(result['stages']),{'windows_admission','cache_verification','wsl_startup','provisioning','hip_operators'})
        self.assertTrue(all(v['status']=='NOT_RUN' for v in result['stages'].values()))
    def test_stop_sequence_has_no_automatic_retry(self):
        called=[]
        def fail(stage):called.append(stage);raise RuntimeError('stop')
        with self.assertRaises(RuntimeError):host.sequence(fail)
        self.assertEqual(called,['provision'])
    def test_later_WSL_startup_failure_cannot_be_reported_as_PASS(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);summary=host.initial_summary('a'*40,root)
            for stage,status,sent in [('provision','PASS',True),('operators','FAIL',False)]:
                (root/stage).mkdir();c.save(root/stage/'host-result.json',dict(startup_classification=status,classification=status,GO_sent=sent,error='startup RAM' if not sent else None))
            host.harvest(summary,root)
            self.assertEqual(summary['stages']['wsl_startup']['status'],'FAIL')
            self.assertEqual(summary['stages']['provisioning']['status'],'PASS')
            self.assertEqual(summary['stages']['hip_operators']['status'],'NOT_RUN')

class Reports(unittest.TestCase):
    def exercise(self,failure):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d)/'proof';root.mkdir();(root.parent/'artifacts').mkdir();case_seen=[];called=[]
            cfg=dict(c.lock(),windows_root=str(root.parent/'unused-v2'),cache_root=str(root.parent/'immutable-cache'))
            cache=dict(files_verified=0,files=[]);counts={'reads':0}
            class Lease:
                def __init__(self,cfg):pass
                def __enter__(self):return self
                def __exit__(self,*args):pass
            class Fake:
                def __init__(self,case,token,cfg):self.case=case;case_seen.append(case)
                def memory(self):counts['reads']+=1;return dict(available_bytes=22*G-1 if failure=='admission' else 22*G)
                def live(self):raise AssertionError('Live memory forbidden')
                def observe(self,*args):return dict(available_bytes=23*G)
                def prepare_payload(self):called.append('cache');c.save(self.case/'cache-verification.json',cache)
                def invoke(self,stage):
                    called.append(stage);leaf=self.case/stage;leaf.mkdir()
                    failed=failure in (stage,'startup') and (stage=='provision' or failure=='operators')
                    receipt=dict(classification='FAIL' if failed else 'PASS',startup_classification='FAIL' if failure=='startup' else 'PASS',GO_sent=failure!='startup')
                    c.save(leaf/'host-result.json',receipt)
                    if failed:raise RuntimeError('synthetic '+failure)
                    if stage=='operators':c.save(leaf/'result.json',dict(classification='PASS',operators={k:dict(status='PASS') for k in c.OPERATORS}))
            with patch.object(c,'ROOT',root),patch.object(c,'lock',return_value=cfg),patch.object(c,'digest',return_value=cfg['windows_python_sha256']),patch.object(c,'verify_cache',return_value=cache),patch.object(c,'CacheReadLease',Lease),patch.object(host,'Runner',Fake),patch.object(host,'verify_publication'),patch.object(host,'desktop_absent'),patch.object(host.sys,'addaudithook'),patch.object(host.time,'sleep'),patch.object(host.ctypes.windll.shell32,'IsUserAnAdmin',return_value=False):
                result=host.run('a'*40)
            summary=json.loads((case_seen[0]/'summary.json').read_text());report=(case_seen[0]/'REPORT.txt').read_text()
            return result,summary,report,called,counts
    def test_failed_admission_never_enters_cache_or_WSL(self):
        result,s,report,calls,counts=self.exercise('admission')
        self.assertEqual(result,2);self.assertEqual(counts['reads'],1);self.assertEqual(calls,[])
        self.assertEqual(s['stages']['windows_admission']['status'],'FAIL');self.assertIn('wsl_startup: NOT_RUN',report)
    def test_startup_failure_leaves_provision_and_HIP_NOT_RUN(self):
        result,s,report,calls,_=self.exercise('startup')
        self.assertEqual(result,2);self.assertEqual(calls,['cache','provision'])
        self.assertEqual(s['stages']['windows_admission']['status'],'PASS');self.assertEqual(s['stages']['wsl_startup']['status'],'FAIL')
        self.assertIn('provisioning: NOT_RUN',report);self.assertIn('hip_operators: NOT_RUN',report)
    def test_provision_failure_stops_before_operator_stage(self):
        result,s,report,calls,_=self.exercise('provision')
        self.assertEqual(result,2);self.assertEqual(calls,['cache','provision'])
        self.assertEqual(s['stages']['wsl_startup']['status'],'PASS');self.assertIn('provisioning: FAIL',report)
    def test_success_is_model_free_and_reports_all_distinct_stages(self):
        result,s,report,calls,_=self.exercise(None)
        self.assertEqual(result,0);self.assertTrue(all(r['status']=='PASS' for r in s['stages'].values()))
        self.assertFalse(s['generation']);self.assertFalse(s['model_weights']);self.assertFalse(s['automatic_retry'])

if __name__=='__main__':unittest.main()
