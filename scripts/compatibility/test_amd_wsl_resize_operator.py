"""No live admission, real disks, WSL or resize; synthetic control fixtures."""
import contextlib,io,json,sys,unittest,tempfile,ast,os
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
try:import amd_wsl_resize_operator as o
except ModuleNotFoundError:o=None

class Implementation(unittest.TestCase):
    def test_operator_exists(self):self.assertIsNotNone(o,'Attended handoff missing')

@unittest.skipIf(o is None,'Implementation missing')
class Operator(unittest.TestCase):
    def test_guest_digest_survives_checkout_line_endings_only(self):
        with tempfile.TemporaryDirectory(dir=o.ROOT.parent/'artifacts') as d:
            path=Path(d)/'guest.py'
            with patch.object(o,'GUEST',path):
                path.write_bytes(b'first\nsecond\n');expected=o.guest_digest()
                path.write_bytes(b'first\r\nsecond\r\n');self.assertEqual(o.guest_digest(),expected)
                path.write_bytes(b'changed\r\nsecond\r\n');self.assertNotEqual(o.guest_digest(),expected)

    def test_guest_effective_readonly_flags_cannot_pass(self):
        tree=ast.parse(o.GUEST.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='root_writable']
        self.assertEqual(len(nodes),1,'Effective root writability helper missing')
        from types import SimpleNamespace
        ns={'os':SimpleNamespace(ST_RDONLY=1)}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(o.GUEST),'exec'),ns)
        good=ns['root_writable']
        self.assertTrue(good('ext4','rw,relatime','rw',0))
        for args in [('ext4','rw','ro',0),('ext4','rw','rw',1),('ext4','ro','rw',0),('overlay','rw','rw',0)]:self.assertFalse(good(*args))

    def test_capacity_setup_failure_still_closes_input_and_settles(self):
        from types import SimpleNamespace
        class Child:
            stdout=io.BytesIO();stdin=io.BytesIO();polls=0
            def poll(self):self.polls+=1;return 1 if self.stdin.closed else None
        with tempfile.TemporaryDirectory(dir=o.ROOT.parent/'artifacts') as d:
            child=Child();ob=o.Observer()
            with patch.object(o,'CASE',Path(d)),patch.object(ob,'check'),patch.object(o.subprocess,'Popen',return_value=child),patch.object(o.threading.Thread,'start',side_effect=RuntimeError('thread startup failed')):
                with self.assertRaisesRegex(RuntimeError,'thread startup failed'):o.capacity_query(ob)
            self.assertTrue(child.stdin.closed);self.assertGreater(child.polls,0)

    def test_capacity_eof_error_still_settles_owned_child(self):
        class Broken(io.BytesIO):
            def close(self):super().close();raise BrokenPipeError('closed pipe')
        class Child:
            stdout=io.BytesIO();stdin=Broken();polls=0
            def poll(self):self.polls+=1;return 1
        with tempfile.TemporaryDirectory(dir=o.ROOT.parent/'artifacts') as d:
            child=Child();ob=o.Observer()
            with patch.object(o,'CASE',Path(d)),patch.object(ob,'check'),patch.object(o.subprocess,'Popen',return_value=child),patch.object(o.threading.Thread,'start',side_effect=RuntimeError('thread startup failed')):
                with self.assertRaisesRegex(RuntimeError,'thread startup failed'):o.capacity_query(ob)
            self.assertTrue(child.stdin.closed);self.assertGreater(child.polls,0)
            self.assertTrue(any('Capacity EOF' in f for f in ob.faults.rows))

    def test_run_late_backup_close_or_report_error_never_leaves_pass(self):
        from types import SimpleNamespace
        for failure in ('none','backup-close','report-write','final-fsync'):
            with tempfile.TemporaryDirectory(dir=o.ROOT.parent/'artifacts') as d:
                parent=Path(d);case=parent/'case';auth=parent/'auth.json';auth.write_text(json.dumps(dict(receipts={},native_binaries={},guest_sha256=o.guest_digest())))
                class Stream(io.BytesIO):
                    def __init__(self,kind):super().__init__(b'fixture');self.kind=kind
                @contextlib.contextmanager
                def opened(path,purpose):
                    kind='backup' if path==o.BACKUP else 'source';f=Stream(kind)
                    try:yield f
                    finally:
                        f.close()
                        if kind=='backup' and failure=='backup-close':raise OSError('backup close failure')
                identity=lambda f:dict(size=o.r.BACKUP_BYTES,file_id=1 if f.kind=='source' else 2,volume_serial='serial')
                ob=o.Observer();state={'resized':False}
                def meta(f):return dict(virtual_capacity_bytes=64*1024**3 if f.kind=='source' and state['resized'] else 32*1024**3)
                def native(_):ob.native_started=ob.native_settled=True;state['resized']=True;return 64*1024**3
                class Client:
                    def child(self,*a,**kw):return json.dumps(dict(running=['Modly-AMD-Ubuntu-24.04'],vm_processes=[]))
                original=Path.open
                save=o.b.save
                def save_receipt(path,row):
                    if failure=='final-fsync' and path.name=='final-result.pending.json':
                        with patch.object(o.os,'fsync',side_effect=OSError('final fsync failure')):return save(path,row)
                    return save(path,row)
                def file_open(path,*a,**kw):
                    if failure=='report-write' and path.name=='REPORT.txt':raise OSError('report write failure')
                    return original(path,*a,**kw)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(o.b,'save',side_effect=save_receipt))
                    for target,name,value in [(o,'CASE',case),(o,'AUTH',auth),(o,'publication',lambda c:None),(o,'no_inherited_job',lambda:None),(o,'inventory',lambda *a:{}),(o,'hash_disk',lambda *a:None),(o,'invoke_native',native),(o,'capacity_query',lambda *a:dict(root_device_bytes=64*1024**3,ext4_rw=True,available_bytes=5*1024**3)),(o,'docker_quiet',lambda:None),(o,'Observer',lambda:ob),(o.time,'sleep',lambda _:None),(o.b,'authenticate_worker_python',lambda:None),(o.b,'authority',lambda:{}),(o.b,'digest',lambda _:o.AUTH_SHA),(o.b,'inputs',lambda _:{}),(o.b,'native_file',opened),(o.b,'file_identity',identity),(o.b,'metadata',meta),(o.b,'Controller',Client),(o.b,'authenticate_inventory',lambda *a:None),(o.b,'Pins',lambda:SimpleNamespace(add=lambda *a:None,close=lambda:None)),(o.r,'verify_backup',lambda *a:None),(o.shutil,'disk_usage',lambda _:SimpleNamespace(free=200*1024**3)),(ob,'check',lambda *a:dict(available_bytes=24*1024**3)),(ob,'preflight',lambda:None),(Path,'open',file_open)]:
                        stack.enter_context(patch.object(target,name,value))
                    code=o.run('a'*40)
                if failure=='final-fsync':
                    self.assertEqual(code,1);self.assertFalse((case/'final-result.json').exists());self.assertTrue((case/'final-result.pending.json').exists());continue
                result=json.loads((case/'final-result.json').read_text())
                self.assertEqual(code,0 if failure=='none' else 1)
                self.assertEqual(result['classification'],'PASS' if failure=='none' else 'FAIL')
                if failure!='none':self.assertTrue(result['faults'])

    def test_breaching_sample_is_recorded_before_refusal(self):
        records=[];ob=o.Observer()
        row=dict(available_bytes=22*1024**3-1,private_commit_bytes=100,working_set_bytes=100)
        from types import SimpleNamespace
        with patch.object(o.b,'windows_memory',return_value=row),patch.object(o.shutil,'disk_usage',return_value=SimpleNamespace(free=200*1024**3)),patch.object(o,'journal',side_effect=lambda event,**data:records.append((event,data))):
            with self.assertRaises(ValueError):ob.check()
        self.assertEqual(ob.minimum,row['available_bytes'])
        self.assertTrue(any(event=='SAMPLE' and data['available_bytes']==row['available_bytes'] for event,data in records))

    def test_actual_invocation_control_has_no_cancellation_even_on_fault(self):
        class Child:
            pid=123;polls=0
            def poll(self):self.polls+=1;return 0 if self.polls>=4 else None
            def kill(self):raise AssertionError('kill forbidden')
            def terminate(self):raise AssertionError('termination forbidden')
        for breach in (False,True):
            with tempfile.TemporaryDirectory(dir=o.ROOT.parent/'artifacts') as d:
                ob=o.Observer();calls=[];child=Child()
                def check(*a):
                    calls.append(1)
                    if breach and len(calls)>1:raise ValueError('RAM below22')
                def spawn(*args,**kw):
                    self.assertEqual(list(args[0]),o.native_command())
                    self.assertEqual(kw['creationflags'],0x08000200)
                    kw['stdout'].write(b'The filesystem on /dev/sdb is now 16777216 (4k) blocks long.\n')
                    return child
                with patch.object(o,'CASE',Path(d)),patch.object(ob,'check',side_effect=check),patch.object(o.subprocess,'Popen',side_effect=spawn),patch.object(o.time,'sleep'):
                    if breach:
                        with self.assertRaisesRegex(ValueError,'Latched native fault'):o.invoke_native(ob)
                    else:self.assertEqual(o.invoke_native(ob),68719476736)
                self.assertTrue(ob.native_started);self.assertTrue(ob.native_settled)
                self.assertTrue((Path(d)/'native-attempt.json').exists())

    def test_default_is_inert(self):
        with patch.object(o,'run',side_effect=AssertionError('live')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(o.main([]),0)

    def test_both_approvals_and_exact_commit_required(self):
        with patch.object(o,'run',side_effect=AssertionError('live')):
            for flags in (['--run'],['--run','--approve-resize','--commit','a'*40]):
                with self.assertRaises(SystemExit):o.main(flags)

    def test_faults_never_clear(self):
        latch=o.Faults(lambda *a,**k:None)
        latch.add('below22');latch.add('below22');latch.add('deadline')
        self.assertEqual(latch.rows,['below22','deadline'])

    def test_settlement_continues_after_resource_error_and_interrupt(self):
        class Child:
            polls=0
            def poll(self):
                self.polls+=1
                return 0 if self.polls>=5 else None
            def kill(self):raise AssertionError('must not kill')
            def terminate(self):raise AssertionError('must not terminate')
        calls=[];latch=o.Faults(lambda *a,**k:None)
        def observe():
            calls.append(1)
            if len(calls)==1:raise RuntimeError('below22')
            if len(calls)==2:raise KeyboardInterrupt()
        self.assertEqual(o.settle(Child(),observe,latch,lambda _:None),0)
        self.assertGreaterEqual(len(calls),4)
        self.assertTrue(any('below22' in x for x in latch.rows))
        self.assertTrue(any('KeyboardInterrupt' in x for x in latch.rows))

    def test_log_failure_does_not_abort_fault_latch(self):
        latch=o.Faults(lambda *a,**k:(_ for _ in ()).throw(OSError('full')))
        latch.add('disk');self.assertIn('disk',latch.rows)

    def test_native_block_geometry_exact_and_unambiguous(self):
        self.assertEqual(o.filesystem_bytes('The filesystem on /dev/sdb is now 16777216 (4k) blocks long.'),68719476736)
        for text in ('Operation completed successfully.','The filesystem on /dev/sdb is now 8388608 (4k) blocks long.'):
            with self.assertRaises(ValueError):o.filesystem_bytes(text)

    def test_no_mutating_fallback_or_native_job(self):
        self.assertEqual(o.native_command()[-3:],['Modly-AMD-Ubuntu-24.04','--resize','68719476736B'])
        import inspect
        source=inspect.getsource(o.invoke_native)
        for word in ('OwnedJob','Terminate','terminate(','kill(','--shutdown','--terminate'):
            self.assertNotIn(word,source)

    def test_cold_hash_mismatch_fails(self):
        with self.assertRaises(ValueError):o.require_hash('0'*64)
        o.require_hash('36d8679eaad94fdfa7b0cc12a48c728ee4ec73f83de90d339005295852365c5c')

if __name__=='__main__':unittest.main(verbosity=2)
