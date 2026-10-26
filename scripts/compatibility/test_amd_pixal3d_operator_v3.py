"""Offline v3 profile/authentication tests; no real admission, WSL or Torch."""
import base64,csv,hashlib,io,json,sys,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import amd_pixal3d_operator_common_v3 as c
import amd_pixal3d_operator_host_v3 as host
import amd_pixal3d_operator_guest_v3 as guest
G=1024**3
TMP=Path(__file__).resolve().parents[3]/'artifacts/amd-pixal3d-operator-v3-preparation-20261026-v1'

class Policy(unittest.TestCase):
    def test_derived_budget_is_explicit_and_neural_guards_remain22(self):
        cfg=c.lock();p=cfg['operator_policy']
        self.assertEqual(p['ready_bytes'],18*G);self.assertEqual(p['observed_startup_decline_bytes'],2360119296)
        self.assertEqual(p['prelaunch_bytes'],21687472128);self.assertEqual(cfg['guards']['admission_RAM_bytes'],22*G)
        self.assertEqual(cfg['guards']['live_host_RAM_floor_bytes'],12*G);self.assertFalse(cfg['execution_authorized_in_chat'])
    def test_three_fresh_readings_and_final_are_required_before_any_side_effect(self):
        values=iter([21687472128]*4);events=[]
        c.admit_operator(lambda:dict(available_bytes=next(values)),lambda v:events.append(('sleep',v)),lambda r:events.append('record'),lambda:events.append('execute'))
        self.assertEqual(events,["record",('sleep',4),'record',('sleep',4),'record','record','execute'])
    def test_low_first_later_or_final_stops_without_retry_or_execute(self):
        for values in ([21687472127],[21687472128,21687472127],[21687472128]*3+[21687472127]):
            values=iter(values);calls=[]
            with self.assertRaises(RuntimeError):c.admit_operator(lambda:dict(available_bytes=next(values)),lambda _:None,lambda _:None,lambda:calls.append('execute'))
            self.assertEqual(calls,[])
    def test_profile_refuses_invalid_measurement_types(self):
        for value in (None,True,float(18*G)):
            with self.assertRaises(RuntimeError):c.require_profile(dict(available_bytes=value),18*G)

class Authentication(unittest.TestCase):
    def fixture(self,base):
        env=base/'venv';site=env/'lib/python3.12/site-packages';site.mkdir(parents=True);(env/'share/man').mkdir(parents=True)
        source=b'qualified tiny module\n';(site/'tiny.py').write_bytes(source)
        data=b'manpage';(env/'share/man/tiny.1').write_bytes(data)
        sha=lambda b:base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b'=').decode()
        record='tiny.py,sha256='+sha(source)+','+str(len(source))+'\ntiny-1.data/data/share/man/tiny.1,sha256='+sha(data)+','+str(len(data))+'\ntiny-1.dist-info/RECORD,,\n'
        archive=base/'tiny.whl'
        with zipfile.ZipFile(archive,'w') as z:z.writestr('tiny-1.dist-info/RECORD',record)
        return env,site,archive
    def test_trusted_wheel_manifest_authenticates_installed_bytes_without_writes(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            env,site,z=self.fixture(Path(d));before=c.environment_snapshot(env)
            self.assertEqual(c.verify_installed_wheel(z,env,site)['files_verified'],2)
            self.assertEqual(c.environment_snapshot(env),before)
    def test_changed_missing_or_escaping_installed_files_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            env,site,z=self.fixture(Path(d));(site/'tiny.py').write_bytes(b'changed')
            with self.assertRaises(RuntimeError):c.verify_installed_wheel(z,env,site)
            (site/'tiny.py').unlink()
            with self.assertRaises(FileNotFoundError):c.verify_installed_wheel(z,env,site)
            for name in ('../../../../outside','/absolute','a\\b'):
                with self.assertRaises(ValueError):c.installed_path(name,env,site)
    def test_provisioning_receipt_must_include_all_exact_packages_and_natural_cleanup(self):
        c.validate_v2_evidence(c.lock())
        with patch.object(c,'evidence_document',return_value={'classification':'FAIL'}):
            with self.assertRaises(RuntimeError):c.validate_v2_evidence(c.lock())
    def test_audit_denies_environment_writes_and_allows_only_fresh_temp_or_evidence(self):
        import amd_pixal3d_operator_ops_v3 as ops
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);temporary=root/'temporary';case=root/'case';environment=root/'v2-env';audit=ops.policy(temporary,case)
            audit('open',(str(environment/'module.py'),'rb',0))
            for event,args in [('open',(str(environment/'cache.pyc'),'wb',1)),('os.mkdir',(str(environment/'new'),0,-1))]:
                with self.assertRaises(RuntimeError):audit(event,args)
            audit('open',(str(temporary/'new-file'),'wb',1))
    def test_extra_python_or_native_startup_modules_fail_current_inventory(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            site=Path(d);(site/'tiny.py').write_bytes(b'trusted')
            self.assertEqual(c.verify_import_inventory(site,{'tiny.py'},set())['unknown_files'],0)
            for name in ('extra.py','sitecustomize.cpython-312-x86_64-linux-gnu.so','usercustomize.so'):
                p=site/name;p.write_bytes(b'unapproved')
                with self.assertRaises(RuntimeError):c.verify_import_inventory(site,{'tiny.py'},set())
                p.unlink()
    def test_pip_bootstrap_is_explicit_and_cannot_import_in_operator_worker(self):
        import amd_pixal3d_operator_ops_v3 as ops
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            site=Path(d);(site/'pip').mkdir();(site/'pip/__init__.py').write_bytes(b'bootstrap')
            self.assertEqual(c.verify_import_inventory(site,set(),set())['bootstrap_files_excluded_from_operator_imports'],1)
            with self.assertRaises(ImportError):ops.policy(site/'temp',site/'case')('import',('pip',None,None,None,None))
    def test_inner_root_startup_directory_alias_is_rejected_without_following_it(self):
        class Alias:
            def is_symlink(self):return True
            def is_file(self):return False
            def relative_to(self,_):return Path('sitecustomize')
        class Site:
            def rglob(self,_):return [Alias()]
        with self.assertRaises(RuntimeError):c.verify_import_inventory(Site(),set(),set())
    def test_venv_configuration_rejects_duplicate_site_flag_and_foreign_home(self):
        valid='home = /usr/bin\ninclude-system-site-packages = false\nversion = 3.12.3\nexecutable = /usr/bin/python3.12\ncommand = qualified creation metadata\n'
        self.assertFalse(c.validate_venv_config(valid,'3.12.3')['system_site_packages'])
        for text in (valid+'include-system-site-packages = true\n',valid.replace('/usr/bin\n','/tmp/foreign\n'),valid.replace('false','true')):
            with self.assertRaises(RuntimeError):c.validate_venv_config(text,'3.12.3')

class Routing(unittest.TestCase):
    def test_no_work_defaults_and_missing_explicit_approval_never_run(self):
        with patch.object(host,'run',side_effect=AssertionError('live')):
            self.assertEqual(host.main([]),0)
            with self.assertRaises(SystemExit):host.main(['--run','--commit','a'*40])
        self.assertEqual(guest.main([]),0)
    def test_only_operator_route_is_exposed_no_provisioning(self):
        command=host.wsl_command('operators','/mnt/c/Users/test/case','a'*32)
        self.assertTrue(any(str(s).endswith('amd_pixal3d_operator_guest_v3.py') for s in command))
        for stage in ('provision','install','download'):
            with self.assertRaises(ValueError):host.wsl_command(stage,'/mnt/c/Users/test/case','a'*32)
    def test_owned_cleanup_never_accepts_foreign_processes(self):
        owner=dict(pid=1,pgid=1,token='a'*32,start_ticks=1)
        with self.assertRaises(RuntimeError):guest.owned_members(owner,[dict(owner,token='b'*32)])
    def test_changed_executed_v2_guest_cannot_bypass_publication_verification(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);dependency='scripts/compatibility/amd_pixal3d_guest_v2.py';published={n:'published\n' for n in set(host.FILES)|{dependency}}
            for name,text in published.items():p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
            (root/dependency).write_text('changed executed dependency\n')
            def git(argv,**kwargs):return 'a'*40 if 'rev-parse' in argv else published[argv[-1].split(':',1)[1]]
            with patch.object(c,'ROOT',root),patch.object(host.subprocess,'check_output',side_effect=git),patch.object(c,'validate_v2_evidence',return_value={}),patch.object(c,'lock',return_value={}):
                with self.assertRaises(ValueError):host.verify_publication('a'*40)

class Invocation(unittest.TestCase):
    def exercise(self,values,wait_timeout=False,attach_failure=False):
        import amd_forward_owned_job as jobs
        import subprocess
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);r=host.Runner.__new__(host.Runner);r.case=root;r.token='a'*32;r.cfg=c.lock();r.startup_active=False;r.vhd_before=0;r.vhd=root/'vhd';r.vhd.write_bytes(b'x');r.memory_log=root/'memory.jsonl';values=iter(values);calls=[]
            class Pipe:
                closed=False;data=b''
                def write(self,data):self.data+=data
                def close(self):self.closed=True
            class Child:
                pid=42;stdin=Pipe();returncode=0;waits=0;killed=False
                def poll(self):return 0 if self.stdin.data or self.killed else None
                def wait(self,timeout=None):
                    calls.append('wait');self.waits+=1
                    if attach_failure and not self.killed or wait_timeout and self.waits==1:raise subprocess.TimeoutExpired('owned client',timeout)
                    return 0
                def kill(self):self.killed=True;calls.append('kill_owned_client')
            child=Child()
            class API:
                def SetInformationJobObject(self,*args):return True
                def QueryInformationJobObject(self,h,k,p,*args):p._obj.PeakProcessMemory=65536;return True
            class Job:
                handle=42;api=API()
                def attach(self,child):
                    if attach_failure:raise OSError('attach failure')
                def close(self):pass
                def finish(self):return dict(forced_job_cleanup=False,owned_tree_exit_verified=True)
            def spawn(*args,**kwargs):
                calls.append('spawn');leaf=root/'operators';owner=dict(pid=42,pgid=42,start_ticks=1,token='a'*32,stage='operators')
                c.save(leaf/'ownership.json',owner);c.save(leaf/'ready.json',dict(owner=owner))
                c.save(leaf/'result.json',dict(classification='PASS',operators={k:dict(status='PASS') for k in c.OPERATORS},memory_release=dict(allocated=0,reserved=0)))
                return child
            def cleanup(stage):calls.append('owned_cleanup');return dict(natural_exit=True,group_empty=True)
            with patch.object(jobs,'OwnedJob',Job),patch.object(host.subprocess,'Popen',spawn),patch.object(r,'memory',side_effect=lambda:dict(available_bytes=next(values))),patch.object(host.shutil,'disk_usage',return_value=type('D',(),dict(free=100*G))()),patch.object(host,'allocated_size',return_value=0),patch.object(host.telemetry,'windows_processes',return_value=dict(status='UNAVAILABLE')),patch.object(r,'cleanup_guest',side_effect=cleanup),patch.object(host.time,'sleep'):
                try:r.invoke('operators')
                except RuntimeError:pass
            return child.stdin.data,json.loads((root/'operators/host-result.json').read_text()),calls
    def test_new_prelaunch_budget_is_checked_before_Popen(self):
        sent,row,calls=self.exercise([20*G,21*G])
        self.assertEqual(sent,b'');self.assertNotIn('spawn',calls);self.assertFalse(row['client_started'])
    def test_ready18_is_accepted_but_exactly_one_GO_is_sent(self):
        sent,row,calls=self.exercise([21*G,18*G,18*G,18*G,18*G])
        self.assertEqual(sent,b'GO\n');self.assertEqual(row['classification'],'PASS');self.assertTrue(row['client_started'])
    def test_startup_below18_withholds_GO_and_cleans_owned_group(self):
        sent,row,calls=self.exercise([21*G,18*G-1,21*G])
        self.assertEqual(sent,b'');self.assertIn('owned_cleanup',calls);self.assertEqual(row['classification'],'FAIL')
    def test_final_ready_gate_cannot_be_skipped(self):
        sent,row,_=self.exercise([21*G,19*G,19*G,18*G-1,21*G])
        self.assertEqual(sent,b'');self.assertFalse(row['GO_sent'])
    def test_client_wait_timeout_does_not_skip_owned_cleanup(self):
        sent,row,calls=self.exercise([21*G,18*G-1,21*G],wait_timeout=True)
        self.assertEqual(sent,b'');self.assertIn('owned_cleanup',calls);self.assertEqual(row['classification'],'FAIL')
    def test_failed_Job_attach_still_cleans_exact_owned_client(self):
        sent,row,calls=self.exercise([21*G,21*G],attach_failure=True)
        self.assertEqual(sent,b'');self.assertIn('kill_owned_client',calls);self.assertEqual(row['classification'],'FAIL')

if __name__=='__main__':unittest.main()
