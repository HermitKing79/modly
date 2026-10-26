"""Mock/CPU-only tests: never call Windows admission, WSL, pip or GPU."""
import ast,hashlib,importlib,json,math,sys,tempfile,unittest,zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import amd_pixal3d_proof_common as c
import amd_pixal3d_standalone as host
import amd_pixal3d_hip_ops as ops
import amd_pixal3d_guest as guest
G=1024**3
TEST_TMP=Path(__file__).resolve().parents[3]/'artifacts/amd-pixal3d-standalone-preparation-20261026-v1'

class Admission(unittest.TestCase):
    def run_gate(self,values,fail_stage=None):
        values=iter(values);events=[]
        def read():events.append('read');return dict(available_bytes=next(values))
        def execute(name):
            events.append(name)
            if fail_stage==name:raise RuntimeError('injected failure')
        c.admit_once(read,lambda t:events.append(('sleep',t)),lambda row:events.append(('record',row['available_bytes'])),execute)
        return events
    def test_exact22_three_spaced_readings_and_final_before_side_effect(self):
        e=self.run_gate([22*G]*4)
        self.assertEqual(e.count('read'),4);self.assertEqual([x for x in e if isinstance(x,tuple) and x[0]=='sleep'],[('sleep',4),('sleep',4)])
        self.assertEqual(e[-1],'admitted');self.assertFalse(c.lock()['guards']['preferred_is_mandatory'])
    def test_first_later_and_final_failure_never_execute(self):
        for values in ([22*G-1],[22*G,22*G-1],[22*G]*3+[22*G-1]):
            called=[];values=iter(values)
            with self.assertRaises(RuntimeError):c.admit_once(lambda:dict(available_bytes=next(values)),lambda _:None,lambda _:None,called.append)
            self.assertEqual(called,[])
    def test_reader_error_never_sleeps_or_executes(self):
        with self.assertRaises(OSError):c.admit_once(lambda:(_ for _ in ()).throw(OSError('fail')),lambda _:self.fail(),lambda _:None,lambda _:self.fail())
    def test_bad_read_types_and_final_threshold(self):
        for value in (float(22*G),True,None,22*G-1):
            with self.assertRaises(RuntimeError):c.require_ram(dict(available_bytes=value),22*G)
    def test_live_floor_and_disk_do_not_replace_22_gate(self):
        c.require_resources(12*G,50*G,2*G)
        for args in ((12*G-1,50*G,2*G),(12*G,50*G-1,2*G),(12*G,50*G,2*G-1)):
            with self.assertRaises(RuntimeError):c.require_resources(*args)

class Isolation(unittest.TestCase):
    def test_owned_cache_hint_does_not_change_new_environment_contents(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as d:
            root=Path(d);path=root/'new-wheel-file';path.write_bytes(b'owned bytes');hints=[]
            row=guest.release_owned_cache(root,lambda fd:hints.append(fd),lambda fd:None)
            self.assertEqual(row['files'],1);self.assertEqual(len(hints),1);self.assertEqual(path.read_bytes(),b'owned bytes')
    def test_partial_child_JSON_cannot_suppress_the_failure_report(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as d:
            root=Path(d)/'proof';root.mkdir();(root.parent/'artifacts').mkdir()
            class Fake:
                def memory(self):return dict(available_bytes=22*G)
                def live(self):return dict(available_bytes=22*G)
                def prepare_payload(self):self.case=next((root.parent/'artifacts').iterdir())
                def invoke(self,stage):
                    if stage=='operators':
                        leaf=self.case/'operators';leaf.mkdir();(leaf/'result.json').write_text('{')
                        raise RuntimeError('synthetic guard stop')
            fake=Fake()
            with patch.object(host,'verify_publication'),patch.object(host,'desktop_absent'),patch.object(host,'Runner',return_value=fake),patch.object(c,'ROOT',root),patch.object(host.time,'sleep'):
                self.assertEqual(host.run('a'*40),2)
            self.assertIn('synthetic guard stop',(fake.case/'REPORT.txt').read_text())
            self.assertEqual(json.loads((fake.case/'summary.json').read_text())['classification'],'FAIL')
    def test_child_failure_details_survive_controller_summary(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as d:
            leaf=Path(d);data=dict(classification='FAIL',operators={'HIP_INT8':dict(status='FAIL',error='synthetic mismatch')},memory_peak=dict(allocated=123,reserved=456))
            (leaf/'child-result.json').write_text(json.dumps(data))
            record=dict(classification='FAIL')
            guest.capture_child(record,leaf)
            self.assertEqual(record['operators'],data['operators']);self.assertEqual(record['memory_peak'],data['memory_peak'])
    def test_payload_routes_to_owned_child_not_inline_network(self):
        owner=host.Runner.__new__(host.Runner)
        with patch.object(owner,'payload_owned',return_value='owned') as method:
            self.assertEqual(owner.prepare_payload(),'owned');method.assert_called_once_with()
    def test_failed_live_admission_is_logged_without_a_fifth_or_retry_read(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as d:
            root=Path(d)/'proof';root.mkdir();(root.parent/'artifacts').mkdir()
            class Fake:
                calls=0
                def memory(self):self.calls+=1;return dict(available_bytes=22*G-1)
                def live(self):raise AssertionError('No post-exit reading when execution never began')
                def prepare_payload(self):raise AssertionError('No payload on low RAM')
                def invoke(self,_):raise AssertionError('No WSL')
            fake=Fake()
            with patch.object(host,'verify_publication'),patch.object(host,'desktop_absent'),patch.object(host,'Runner',return_value=fake),patch.object(c,'ROOT',root):
                self.assertEqual(host.run('a'*40),2)
            self.assertEqual(fake.calls,1)
            report=next((root.parent/'artifacts').glob('*/summary.json'))
            self.assertFalse(json.loads(report.read_text())['execution_started'])
    def test_no_work_defaults_do_not_call_live_hooks(self):
        with patch.object(host,'run',side_effect=AssertionError('live')),patch.object(guest,'controller',side_effect=AssertionError('guest live')):
            self.assertEqual(host.main([]),0);self.assertEqual(guest.main([]),0)
    def test_path_traversal_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as d:
            root=Path(d)
            for name in ('../escape','/absolute','x/../../bad','C:/outside','a\\b'):
                with self.assertRaises(ValueError):c.inside(root,name)
    def test_stage_failure_stops_without_next_stage_or_retry(self):
        for failed in ('provision','operators'):
            called=[]
            def invoke(name):
                called.append(name)
                if name==failed:raise RuntimeError('stop')
            with self.assertRaises(RuntimeError):host.sequence(invoke)
            self.assertEqual(called,['provision'] if failed=='provision' else ['provision','operators'])
    def test_exact_wsl_routing_no_shell_or_foreign_distro(self):
        cmd=host.wsl_command('provision','/mnt/c/Users/test/case','a'*32)
        self.assertEqual(cmd[1:5],['-d',c.lock()['distro'],'-u','modly'])
        self.assertNotIn('bash',cmd);self.assertNotIn('sh',cmd);self.assertIn('--controller',cmd)
        for path in ('/mnt/c/a;touch bad','../escape','/tmp/not-approved'):
            with self.assertRaises(ValueError):host.wsl_command('provision',path,'a'*32)
    def test_cleanup_foreign_members_or_reused_leader_cannot_signal(self):
        owner=dict(token='a'*32,pgid=42,pid=42,start_ticks=123)
        good=[dict(pid=42,pgid=42,start_ticks=123,token='a'*32),dict(pid=43,pgid=42,start_ticks=124,token='a'*32)]
        self.assertTrue(guest.owned_members(owner,good))
        for members in ([dict(good[0],token='b'*32)],[dict(good[0],start_ticks=999)],[dict(good[1],token=None)]):
            with self.assertRaises(RuntimeError):guest.owned_members(owner,members)
    def test_pass_requires_all_operators_and_natural_cleanup(self):
        good={name:dict(status='PASS') for name in c.OPERATORS}
        host.accept_operators(dict(classification='PASS',operators=good,memory_release=dict(allocated=0,reserved=0)),dict(natural_exit=True,group_empty=True))
        for result,cleanup in ((dict(classification='PASS',operators={}),dict(natural_exit=True,group_empty=True)),
             (dict(classification='PASS',operators=good,memory_release=dict(allocated=1,reserved=0)),dict(natural_exit=True,group_empty=True)),
             (dict(classification='PASS',operators=good,memory_release=dict(allocated=0,reserved=0)),dict(natural_exit=False,group_empty=True))):
            with self.assertRaises(RuntimeError):host.accept_operators(result,cleanup)

class Payload(unittest.TestCase):
    def test_native_disk_allocation_reader_on_owned_synthetic_file(self):
        import os
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as d:
            path=Path(d)/'allocation.bin';path.write_bytes(os.urandom(8192))
            self.assertGreaterEqual(host.allocated_size(path),8192)
    def test_realistic_synthetic_HIP_ELF_is_read_without_loading(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as d:
            path=Path(d)/'hip.whl';header=bytearray(64);header[:6]=b'\x7fELF\x02\x01';header[18:20]=(62).to_bytes(2,'little')
            with zipfile.ZipFile(path,'w') as z:
                z.writestr('comfy_kitchen/backends/hip/_C.abi3.so',header)
                z.writestr('comfy_kitchen/backends/hip/architectures.json','{"gfx1201":true}')
            self.assertEqual(c.hip_payload(path)['extension'],'comfy_kitchen/backends/hip/_C.abi3.so')
    def test_lock_reuses_pins_and_never_lists_model_files(self):
        lock=c.lock();self.assertFalse(lock['model_downloads_allowed']);self.assertEqual(len(lock['packages']),13)
        for row in lock['packages']:
            self.assertTrue(row['filename'].endswith('.whl'));self.assertEqual(len(row['sha256']),64)
        self.assertEqual(lock['guards']['admission_RAM_bytes'],22*G)
        self.assertEqual(lock['guards']['live_host_RAM_floor_bytes'],12*G)
    def test_missing_HIP_and_non_elf_payload_stop(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as d:
            path=Path(d)/'k.whl'
            with zipfile.ZipFile(path,'w') as z:z.writestr('comfy_kitchen/backends/cuda/_C.abi3.so',b'bad')
            with self.assertRaises(RuntimeError):c.hip_payload(path)
    def test_hash_or_size_failure_preserves_rejected_file(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as d:
            path=Path(d)/'x.whl';path.write_bytes(b'bad')
            with self.assertRaises(RuntimeError):c.verify(path,dict(bytes=3,sha256='0'*64))
            self.assertEqual(path.read_bytes(),b'bad')
    def test_pinned_functions_extract_without_model_or_torch_import(self):
        source=Path(__file__).resolve().parents[3]/'artifacts/amd-pixal3d-native-assessment-20261025-v1/core--comfy__image_encoders__naf.py'
        ns=ops.extract(source,{'na2d_pure'},{})
        self.assertEqual(set(ns),{'na2d_pure'});self.assertEqual(ns['na2d_pure'].__name__,'na2d_pure')
        with self.assertRaises(ValueError):ops.extract(source,{'PretrainedFake'},{} )

class References(unittest.TestCase):
    def test_independent_attention_known_constant_values(self):
        self.assertEqual(ops.attention_reference([[0,0]],[[0,0],[0,0]],[[2,4],[6,8]],1),[[4,6]])
    def test_hadamard_reference_is_orthonormal(self):
        h=ops.hadamard(16)
        for i in range(16):
            for j in range(16):self.assertAlmostEqual(sum(h[i][k]*h[j][k] for k in range(16)),float(i==j))
    def test_nonfinite_wrong_device_or_backend_rejected(self):
        for value in (math.nan,math.inf,-math.inf):
            with self.assertRaises(RuntimeError):ops.assert_values([value],[0],.01,.01)
        with self.assertRaises(RuntimeError):ops.assert_values([1],[0],.01,.01)
        with self.assertRaises(RuntimeError):ops.require_witness('cpu',1,{'int8_gemm':1},True)
        with self.assertRaises(RuntimeError):ops.require_witness('cuda',1,{},True)
        with self.assertRaises(RuntimeError):ops.require_witness('cuda',1,{'int8_gemm':1},False)

if __name__=='__main__':unittest.main()
