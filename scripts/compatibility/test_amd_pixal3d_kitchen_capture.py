"""Offline forensic routing/evidence tests. No Torch, WSL or GPU import."""
import ast,atexit,getpass,hashlib,json,os,re,shutil,sys,tempfile,unittest,zipfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
from amd_pixal3d_operator_audit_v3 import AuditLog
import amd_pixal3d_kitchen_capture as cap
TMP=None

class Evidence(unittest.TestCase):
    def test_missing_cwd_does_not_hide_first_denied_event(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            path=Path(d)/'audit.jsonl';log=AuditLog(path)
            try:
                with patch('amd_pixal3d_operator_audit_v3.os.getcwd',side_effect=FileNotFoundError('deleted cwd')):log('os.remove',(str(Path(d)/'target'),-1))
            finally:log.close()
            row=json.loads(path.read_text());self.assertEqual(row['event'],'os.remove');self.assertIn('cwd_unavailable',row)

    def test_target_identity_is_observed_but_not_used_as_permission(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            path=Path(d)/'keep';path.write_bytes(b'keep');log=AuditLog(Path(d)/'audit.jsonl')
            try:log('os.remove',(str(path),-1))
            finally:log.close()
            target=json.loads((Path(d)/'audit.jsonl').read_text())['targets'][0]
            self.assertEqual(target['inode'],path.stat().st_ino);self.assertTrue(target['observation_only']);self.assertTrue(target['resolution_is_unpinned'])
            self.assertEqual(target['identity_role'],'candidate_directory_entry_lstat');self.assertEqual(path.read_bytes(),b'keep')

    def test_relative_dir_fd_uses_observed_descriptor_base_and_unknown_base_stays_unknown(self):
        from amd_pixal3d_operator_audit_v3 import path_observations
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);(root/'keep').write_bytes(b'keep')
            args=('keep',77);rows=path_observations('os.remove',args,None,[dict(argument_index=1,target=str(root),identity_stable=True)])
            self.assertEqual(rows[0]['candidate'],str(root/'keep'));self.assertEqual(rows[0]['inode'],(root/'keep').stat().st_ino)
            self.assertIn('unavailable',path_observations('os.remove',args,None,[])[0])

    def test_first_denial_cannot_be_swallowed_by_an_Exception_handler(self):
        class Exit(BaseException):pass
        events=[]
        def leave(code):events.append(('exit',code));raise Exit()
        sink=cap.FirstDenial(lambda event,args:events.append(('persisted',event,args)),leave)
        with self.assertRaises(Exit):
            try:sink('os.remove',('owned/probe',-1))
            except Exception:self.fail('Import caught the first-denial stop')
        self.assertEqual(events,[('persisted','os.remove',('owned/probe',-1)),('exit',23)])

    def test_logging_failure_stops_with_distinct_code(self):
        class Exit(BaseException):pass
        codes=[]
        def leave(code):codes.append(code);raise Exit()
        def broken(*args):raise OSError('evidence disk unavailable')
        with self.assertRaises(Exit):cap.FirstDenial(broken,leave)('os.remove',('probe',-1))
        self.assertEqual(codes,[24])

    def test_empty_or_partial_phase_write_cannot_break_supervision(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            path=Path(d)/'phase';path.write_bytes(b'');self.assertIsNone(cap.last_phase(path))
            path.write_bytes(b'{"partial":');self.assertIsNone(cap.last_phase(path))
            path.write_bytes(b'{"kind":"CPU","monotonic_start":1}\n{"partial":')
            self.assertEqual(cap.last_phase(path),dict(kind='CPU',monotonic_start=1))

    def test_ambiguous_denial_evidence_is_not_accepted(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);path=root/'first-denial.jsonl';row=dict(event='os.remove',operation_allowed=False)
            path.write_text(json.dumps(row)+'\n');self.assertEqual(cap.denial_record(root),row)
            path.write_text((json.dumps(row)+'\n')*2)
            with self.assertRaises(RuntimeError):cap.denial_record(root)

    def test_unraisable_hook_reports_then_stops_even_without_filesystem_denial(self):
        class Exit(BaseException):pass
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            path=Path(d)/'unraisable.json';codes=[]
            def leave(code):codes.append(code);raise Exit()
            args=type('Args',(),dict(exc_value=RuntimeError('observer teardown failed')))()
            with self.assertRaises(Exit):cap.unraisable_handler(path,leave)(args)
            self.assertEqual(codes,[24]);self.assertIn('observer teardown failed',json.loads(path.read_text())['error'])

    def test_observer_is_armed_for_unraisable_errors_before_runtime_import(self):
        class NativeBoundary(BaseException):pass
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);(root/'capture').mkdir();cfg=cap.config();cfg['temporary_guest_root']=str(root/'scratch');old_hook=sys.unraisablehook;logs=[]
            def logger(path):log=AuditLog(path);logs.append(log);return log
            try:
                with patch.object(cap,'validate_case',return_value=root),patch.object(cap,'config',return_value=cfg),patch.object(cap,'worker_context'),patch.object(cap.sys,'addaudithook'),patch.object(cap,'AuditLog',side_effect=logger),patch.object(cap,'runtime',side_effect=NativeBoundary):
                    with self.assertRaises(NativeBoundary):cap.worker(root,'a'*32,True)
                self.assertIsNot(sys.unraisablehook,old_hook)
            finally:
                sys.unraisablehook=old_hook
                for log in logs:log.close()

class Routing(unittest.TestCase):
    def test_no_work_default_missing_approval_and_missing_commit_never_launch(self):
        with patch.object(cap,'run_host',side_effect=AssertionError('live')):
            self.assertEqual(cap.main([]),0)
            for args in (['--capture-kitchen-import'],['--capture-kitchen-import','--approve-kitchen-import-capture']):
                with self.assertRaises(SystemExit):cap.main(args)
        with patch.object(cap,'verify_publication',side_effect=AssertionError('preparation')):
            with self.assertRaises(RuntimeError):cap.run_host('a'*40,False)

    def test_only_import_worker_or_query_cleanup_child_is_routable(self):
        cfg=cap.config()
        for mode in ('--capture-worker','--release-observer'):
            cmd=cap.child_command(mode,Path('case'),'a'*32,cfg)
            self.assertEqual(cmd[0],cfg['guest_root']+'/.venv/bin/python');self.assertIn(mode,cmd)
        for mode in ('--run','--operators','--provision','--inference'):
            with self.assertRaises(ValueError):cap.child_command(mode,Path('case'),'a'*32,cfg)

    def test_original_operator_hold_and_resource_limits_are_retained(self):
        cfg=cap.config();base=cap.c.lock()
        self.assertTrue(cfg['operator_execution_hold']);self.assertEqual(cfg['guards'],base['guards']);self.assertEqual(cfg['operator_policy'],base['operator_policy'])
        self.assertEqual(cfg['packages'],base['packages']);self.assertEqual(cfg['guest_root'],base['guest_root'])
        self.assertFalse(cfg['capture_spec']['numerical_checks']);self.assertTrue(cfg['capture_spec']['approval_pending'])
        import amd_pixal3d_operator_host_v3 as old
        with self.assertRaisesRegex(RuntimeError,'NO_GO'):old.run('a'*40,True)

    def test_forensic_PASS_is_not_operator_qualification_and_bad_resources_fail(self):
        import copy
        result=dict(classification='PASS',purpose=cap.PURPOSE,capture_status='FIRST_DENIAL_CAPTURED',numerical_checks=False,models_loaded=False,operator_execution_hold=True,
            V2_preserved=True,natural_child_exit=True,first_denial=dict(event='os.remove',operation_allowed=False,caller_frames=[dict(function='probe')]),
            release_observer=dict(numerical_checks=False,allocated=0,reserved=0,gpu=dict(free_bytes=3*cap.c.GIB)))
        cleanup=dict(natural_exit=True,group_empty=True);cap.accept_capture(result,cleanup)
        for key,value in [('numerical_checks',True),('models_loaded',True),('operator_execution_hold',False),('V2_preserved',False),('natural_child_exit',False),('capture_status','HIP_PASS')]:
            bad=copy.deepcopy(result);bad[key]=value
            with self.assertRaises(RuntimeError):cap.accept_capture(bad,cleanup)
        for field,value in [('allocated',1),('reserved',1)]:
            bad=copy.deepcopy(result);bad['release_observer'][field]=value
            with self.assertRaises(RuntimeError):cap.accept_capture(bad,cleanup)
        bad=copy.deepcopy(result);bad['release_observer']['gpu']['free_bytes']-=1
        with self.assertRaises(RuntimeError):cap.accept_capture(bad,cleanup)

class Controller(unittest.TestCase):
    def exercise(self,capture_exit=23,observer_exit=0,pidfd_failure=False,auth_failure=False,auth_samples=False):
        import io
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);leaf=root/'capture';leaf.mkdir();cfg=cap.config();cfg['guest_root']=str(root/'v2');cfg['temporary_guest_root']=str(root/'scratch')
            cfg['runtime']=dict(cfg['runtime'],python=sys.version.split()[0]);token='a'*32;events=[];children=[];scans=[];clock=[10.0]
            owner=dict(pid=303,pgid=303,start_ticks=1,token=token,RSS_bytes=1024,state='S')
            def rows(*args,**kwargs):
                scans.append(1)
                return [owner]+[dict(pid=child.pid,pgid=303,start_ticks=child.pid,token=token,RSS_bytes=1024,state='S') for child in children if not child.stdin.closed]
            class Pipe:
                closed=False
                def write(self,data):
                    if data!=b'GO\n':raise AssertionError('wrong GO')
                def close(self):self.closed=True
            class Child:
                def __init__(self,pid,code):self.pid=pid;self.returncode=code;self.stdin=Pipe()
                def poll(self):return self.returncode if self.stdin.closed else None
                def wait(self,timeout=None):return self.returncode
            def spawn(argv,**kw):
                mode=argv[4];events.append(mode)
                self.assertEqual(argv[0],cfg['guest_root']+'/.venv/bin/python');self.assertIn(mode,('--capture-worker','--release-observer'))
                if mode=='--capture-worker':
                    (leaf/'first-denial.jsonl').write_text(json.dumps(dict(event='os.remove',arguments=['probe',-1],phase='Kitchen-import',operation_allowed=False,caller_frames=[dict(function='probe')]))+'\n')
                    child=Child(400,capture_exit)
                else:
                    (leaf/'release-observer.json').write_text(json.dumps(dict(purpose=cap.PURPOSE,numerical_checks=False,allocated=0,reserved=0,gpu=dict(free_bytes=3*cap.c.GIB))))
                    child=Child(500,observer_exit)
                children.append(child);return child
            def kernel():
                events.append('pidfd-preflight')
                if pidfd_failure:raise RuntimeError('kernel unavailable')
            def auth(*args,**kwargs):
                events.append('authenticated')
                if auth_samples:
                    state=args[1];state.enter('installed-files','installed','torch/lib/example.so','installed-sha256',package='torch')
                    for when,count in [(10,17),(10.1,18),(10.25,19),(130.25,20)]:
                        clock[0]=when;state.file_bytes_hashed=count;state()
                if auth_failure:
                    args[1].enter('installed-files','installed','torch/lib/example.so','installed-sha256',package='torch')
                    args[1].file_bytes_hashed=17
                    raise TimeoutError('Existing120-second authentication deadline')
                return dict(status='PASS')
            proxy_os=SimpleNamespace(name='posix',getpid=lambda:303,getpgrp=lambda:303,environ={'MODLY_PIXAL_PROOF_TOKEN':token})
            proxy_sys=SimpleNamespace(prefix='/usr',version=sys.version,stdin=SimpleNamespace(buffer=io.BytesIO(b'GO\n')),addaudithook=lambda _:None)
            timing=SimpleNamespace(monotonic=lambda:clock[0]) if auth_samples else cap.time
            with patch.object(shutil,'disk_usage',return_value=SimpleNamespace(free=cfg['guest_disk_floor_bytes'])),patch.object(cap,'time',timing),patch.object(cap,'validate_case',return_value=root),patch.object(cap,'config',return_value=cfg),patch.object(cap,'os',proxy_os),patch.object(cap,'sys',proxy_sys),patch.object(cap.ProcFS,'require_pidfds',side_effect=kernel),patch.object(cap,'process_rows',side_effect=rows),patch.object(cap.c,'environment_snapshot',return_value={'immutable':'same'}),patch.object(cap.c,'authenticate_environment',side_effect=auth),patch.object(cap.c.overlay,'authenticate',return_value=dict(status='PASS')),patch.object(cap.subprocess,'Popen',side_effect=spawn):
                if pidfd_failure:
                    with self.assertRaises(RuntimeError):cap.controller(root,token)
                    return events,None
                cap.controller(root,token)
            resource_file=leaf/'resources.jsonl';self.resource_rows=[json.loads(line) for line in resource_file.read_text().splitlines()] if resource_file.exists() else [];self.scans=len(scans)
            return events,json.loads((leaf/'result.json').read_text())

    def test_one_capture_then_one_query_only_observer_and_no_operator_route(self):
        events,result=self.exercise()
        self.assertEqual(events,['pidfd-preflight','authenticated','--capture-worker','--release-observer'])
        self.assertEqual(result['classification'],'PASS');self.assertEqual(result['capture_status'],'FIRST_DENIAL_CAPTURED')
        cap.accept_capture(result,dict(natural_exit=True,group_empty=True))

    def test_authentication_timeout_keeps_checkpoint_and_never_starts_worker(self):
        events,result=self.exercise(auth_failure=True)
        self.assertEqual(events,['pidfd-preflight','authenticated']);self.assertEqual(result['classification'],'FAIL')
        self.assertIn('authentication_progress',result)
        row=result['authentication_progress'];self.assertEqual(row['status'],'failed');self.assertEqual(row['file'],'torch/lib/example.so')
        self.assertEqual(row['package'],'torch');self.assertEqual(row['file_bytes_hashed'],17);self.assertEqual(row['checkpoint'],'installed-sha256')

    def test_auth_samples_use_the_existing_throttle_scans_and_120_second_deadline(self):
        events,result=self.exercise(auth_samples=True)
        self.assertEqual(events,['pidfd-preflight','authenticated']);self.assertEqual(result['classification'],'FAIL');self.assertIn('authentication/overall deadline',result['error'])
        self.assertEqual(result['seconds'],120.25);self.assertEqual(self.scans,5)
        self.assertEqual([row['authentication_progress']['file_bytes_hashed'] for row in self.resource_rows],[17,19,20])
        self.assertTrue(all(row['authentication_progress']['file']=='torch/lib/example.so' for row in self.resource_rows))
        self.assertEqual(result['authentication_progress']['status'],'failed')

    def test_failed_stop_or_failed_release_cannot_qualify_or_retry(self):
        events,result=self.exercise(capture_exit=0)
        self.assertEqual(events.count('--capture-worker'),1);self.assertNotIn('--release-observer',events);self.assertEqual(result['classification'],'FAIL')
        self.assertIn('observed_first_denial',result)
        events,result=self.exercise(observer_exit=1)
        self.assertEqual(events.count('--release-observer'),1);self.assertEqual(result['classification'],'FAIL')

    def test_kernel_preflight_failure_prevents_authentication_and_worker_GO(self):
        events,result=self.exercise(pidfd_failure=True)
        self.assertEqual(events,['pidfd-preflight']);self.assertIsNone(result)

def cpu_fixture(root,mode):
    """Explicit offline controls only: authenticated definitions plus stdlib."""
    if mode not in ('cold','cached','unraisable'):raise ValueError('Only declared CPU controls')
    root=cap.c.plain(root);root.relative_to(cap.c.plain(cap.c.ROOT.parent/'artifacts'));root.mkdir(exist_ok=False)
    if mode=='unraisable':
        sys.unraisablehook=cap.unraisable_handler(root/'unraisable.json')
        def fail():raise RuntimeError('fixture teardown exception')
        atexit.register(fail);return
    scratch=root/'scratch';scratch.mkdir();os.environ['TMPDIR']=str(scratch)
    tempfile.tempdir=str(scratch) if mode=='cached' else None
    cfg=cap.config();pin=next(p for p in cfg['packages'] if p['name']=='torch');archive=Path(cfg['cache_root'])/'downloads'/pin['filename'];cap.c.verify(archive,pin)
    with zipfile.ZipFile(archive) as z:
        environment=z.read('torch/_environment.py');cache=z.read('torch/_inductor/runtime/cache_dir_utils.py')
    env_tree=ast.parse(environment);cache_tree=ast.parse(cache)
    nodes=[n for n in env_tree.body if isinstance(n,ast.FunctionDef) and n.name=='is_fbcode']+[n for n in cache_tree.body if isinstance(n,ast.FunctionDef) and n.name=='default_cache_dir']
    namespace=dict(os=os,re=re,getpass=getpass,tempfile=tempfile)
    for node in nodes:
        filename='authenticated-wheel:'+pin['filename']+'/'+('torch/_environment.py' if node.name=='is_fbcode' else 'torch/_inductor/runtime/cache_dir_utils.py')
        exec(compile(ast.Module(body=[node],type_ignores=[]),filename,'exec'),namespace)
    (root/'source.json').write_text(json.dumps(dict(mode=mode,Torch_wheel_sha256=pin['sha256'],cache_source_sha256=hashlib.sha256(cache).hexdigest(),historical_trigger_proven=False)))
    log=AuditLog(root/'first-denial.jsonl');log.phase='offline-exact-default-cache-dir'
    sys.addaudithook(cap.policy(root,root,cap.FirstDenial(log)))
    try:
        try:value=namespace['default_cache_dir']()
        except BaseException:
            (root/'incorrectly-swallowed.json').write_text('{}');raise
        (root/'cached-result.json').write_text(json.dumps(dict(path=value,historical_trigger_proven=False)))
    finally:
        # A real first denial exits before reaching even this marker.
        (root/'returned.json').write_text('{}')
    assert not ({'torch','triton','comfy_kitchen'}&sys.modules.keys())

if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--cpu-fixture':cpu_fixture(sys.argv[3],sys.argv[2])
    else:unittest.main()
