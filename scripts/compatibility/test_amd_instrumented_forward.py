"""CPU-only cache/monitor tests: synthetic bytes and mock passive observers."""
import hashlib
import base64
import io
import json
from pathlib import Path
import struct
import sqlite3
import bz2
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).parent))
try:
    import amd_forward_cache_reuse as cache
    import amd_forward_windows_monitor as monitoring
except ModuleNotFoundError:cache=monitoring=None
try:import amd_forward_owned_job as ownership
except ModuleNotFoundError:ownership=None


def elf(flags=0x4e):
    data=bytearray(64+64+8)
    data[:16]=b'\x7fELF\x02\x01\x01\x40'+b'\0'*8
    struct.pack_into('<HHIQQQIHHHHHH',data,16,3,224,1,0,0,64,flags,64,56,0,64,1,0)
    struct.pack_into('<IIQQQQIIQQ',data,64,0,1,6,0,128,8,0,0,8,0)
    return bytes(data)


def container(payload):return struct.pack('<Q',len(payload))+payload+struct.pack('<Q',0)


class CacheTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(cache)

    def test_complete_direct_elf_is_eligible_but_never_executed(self):
        row=cache.inspect_comgr(container(elf()))
        self.assertTrue(row['eligible']);self.assertEqual(row['architecture'],'gfx1201')
        self.assertNotIn('torch',sys.modules)

    def test_uncertain_framing_bitcode_empty_and_incomplete_entries_are_excluded(self):
        for payload in (struct.pack('<II',0,len(elf()))+elf(),b'BC\xc0\xde'+b'\0'*20,b''):
            self.assertFalse(cache.inspect_comgr(container(payload))['eligible'])
        self.assertFalse(cache.inspect_comgr(container(elf())[:-1])['eligible'])

    def test_wrong_architecture_and_incomplete_elf_are_excluded(self):
        for data in (elf(0x41),elf()[:-3]):
            self.assertFalse(cache.inspect_comgr(container(data))['eligible'])

    def test_private_copies_are_exclusive_identical_and_do_not_modify_source(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]) as tmp:
            source=Path(tmp)/'source';source.mkdir();name='amd_comgr_cache_dir/llvmcache-'+'a'*64
            file=source/name;file.parent.mkdir();data=container(elf());file.write_bytes(data);before=file.stat()
            rows=[{'path':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'eligible':True}]
            target=Path(tmp)/'private'
            cache.copy_entries(source,target,rows);cache.verify_copies(target,rows)
            self.assertEqual(file.read_bytes(),data);self.assertEqual(file.stat().st_mtime_ns,before.st_mtime_ns)
            with self.assertRaises(FileExistsError):cache.copy_entries(source,target,rows)
            (target/name).write_bytes(data[:-1])
            with self.assertRaises(RuntimeError):cache.verify_copies(target,rows)

    def test_unsafe_paths_unselected_entries_and_extra_files_fail_closed(self):
        for name in ('../historical','tmp/a','/absolute','C:/file','amd_comgr_cache_dir/llvmcache-invalid'):
            with self.assertRaises(ValueError):cache.cache_path(name)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]) as tmp:
            source=Path(tmp)/'source';source.mkdir()
            with self.assertRaises(ValueError):cache.copy_entries(source,Path(tmp)/'private',[{'eligible':False}])
            other=Path(tmp)/'other';other.mkdir();(other/'extra').write_text('extra')
            with self.assertRaises(RuntimeError):cache.verify_copies(other,[])

    def test_changed_native_attestation_is_rejected_before_copy(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]) as tmp:
            site=Path(tmp)
            with self.assertRaises(RuntimeError):cache.attest_runtime(site,{'amd':{'all_distributions':[]}})

    def test_native_record_and_library_content_must_both_match_historical_packages(self):
        packages={'rocm-sdk-core':('_rocm_sdk_core/bin/amd_comgr.dll','_rocm_sdk_core/bin/amdhip64_7.dll','_rocm_sdk_core/bin/hiprtc0715.dll'),
                  'rocm-sdk-libraries':('_rocm_sdk_libraries/bin/MIOpen.dll',)}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]) as tmp:
            site=Path(tmp);old={'amd':{'all_distributions':[]}}
            for package,names in packages.items():
                rows=[]
                for name in names:
                    file=site/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'owned synthetic bytes')
                    signature=base64.urlsafe_b64encode(hashlib.sha256(file.read_bytes()).digest()).decode().rstrip('=')
                    rows.append(name+',sha256='+signature+','+str(file.stat().st_size))
                record=site/(package.replace('-','_')+'-10.0.0.dist-info/RECORD');record.parent.mkdir()
                record.write_text('\n'.join(rows))
                old['amd']['all_distributions'].append({'name':package,'version':'10.0.0','record_sha256':hashlib.sha256(record.read_text().encode()).hexdigest()})
            self.assertEqual(len(cache.attest_runtime(site,old)),4)
            (site/'_rocm_sdk_core/bin/amd_comgr.dll').write_bytes(b'modified')
            with self.assertRaisesRegex(RuntimeError,'differs'):cache.attest_runtime(site,old)
            (site/'_rocm_sdk_core/bin/amd_comgr.dll').write_bytes(b'owned synthetic bytes')
            record.write_text('modified RECORD')
            with self.assertRaisesRegex(RuntimeError,'RECORD changed'):cache.attest_runtime(site,old)

    def test_database_compression_target_and_schema_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]) as tmp:
            path=Path(tmp)/'synthetic.ukdb';connection=sqlite3.connect(path)
            connection.execute('CREATE TABLE kern_db(id INTEGER,kernel_name TEXT,kernel_args TEXT,kernel_hash TEXT,uncompressed_size INTEGER,kernel_blob BLOB)')
            expected=[];data=elf()
            for index in (1,2):
                args='-mcpu=gfx1201';name=f'kernel-{index}';key=str(index)
                connection.execute('INSERT INTO kern_db VALUES(?,?,?,?,?,?)',(index,name,args,key,len(data),bz2.compress(data)))
                expected.append({'id':index,'name':name,'arguments':args,'recorded_hash':key,'uncompressed_bytes':len(data),'uncompressed_sha256':hashlib.sha256(data).hexdigest()})
            connection.commit();cache.validate_database(path,expected)
            connection.execute('UPDATE kern_db SET kernel_blob=? WHERE id=1',(b'broken',));connection.commit()
            with self.assertRaises((OSError,RuntimeError,ValueError)):cache.validate_database(path,expected)
            connection.execute('UPDATE kern_db SET kernel_blob=?,kernel_args=? WHERE id=1',(bz2.compress(data),'-mcpu=gfx1100'));connection.commit()
            with self.assertRaises(RuntimeError):cache.validate_database(path,expected)
            connection.execute('DROP TABLE kern_db');connection.commit()
            with self.assertRaises(sqlite3.DatabaseError):cache.validate_database(path,expected)
            connection.close()


class WindowsMonitorTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(monitoring)

    def test_healthy_capture_is_durable_before_ready_and_contains_elapsed_time(self):
        lines=[];calls=[]
        def observe(since,timeout):calls.append(timeout);return {'ok':True}
        monitor=monitoring.WindowsMonitor('time',lines.append,observe=observe,healthy=lambda row:row.get('ok',False),interval=.05)
        try:
            monitor.start(timeout=.5);monitor.require_healthy()
            row=json.loads(lines[0][len(monitoring.MARKER):])
            self.assertEqual(row['name'],'windows-observer');self.assertTrue(row['healthy'])
            self.assertEqual(calls,[4.]);self.assertGreaterEqual(row['query_seconds'],0.)
        finally:self.assertTrue(monitor.close())

    def test_unhealthy_capture_or_durability_failure_blocks_authorization(self):
        for ok in (False,True):
            def send(line):
                if ok:raise OSError('disk failure')
            monitor=monitoring.WindowsMonitor('time',send,observe=lambda *a,**kw:{'ok':ok},healthy=lambda row:row['ok'])
            try:
                with self.assertRaises(RuntimeError):monitor.start(timeout=.5)
            finally:self.assertTrue(monitor.close())

    def test_stalled_query_and_stale_observation_fail_closed_without_blocking_parent(self):
        release=threading.Event()
        monitor=monitoring.WindowsMonitor('time',lambda line:None,observe=lambda *a,**kw:release.wait(),healthy=lambda row:True)
        try:
            began=time.monotonic()
            with self.assertRaises(RuntimeError):monitor.start(timeout=.02)
            self.assertLess(time.monotonic()-began,.5)
            self.assertFalse(monitor.close(timeout=.02))
        finally:release.set();monitor.thread.join(timeout=1)
        monitor=monitoring.WindowsMonitor('time',lambda line:None)
        monitor.last_healthy=time.monotonic()-10
        with self.assertRaises(RuntimeError):monitor.require_healthy()

    def test_fixed_rate_checks_do_not_exceed_ten_seconds_or_allow_longer_queries(self):
        with self.assertRaises(ValueError):monitoring.WindowsMonitor('time',lambda line:None,interval=11)
        ticks=[]
        def observe(*args,**kwargs):ticks.append(time.monotonic());return {}
        monitor=monitoring.WindowsMonitor('time',lambda line:None,observe=observe,healthy=lambda row:True,interval=.04)
        try:
            monitor.start(timeout=.5);time.sleep(.13)
            self.assertGreaterEqual(len(ticks),3)
            self.assertLess(max(b-a for a,b in zip(ticks,ticks[1:])),.15)
        finally:self.assertTrue(monitor.close())

    def test_periodic_service_timeout_is_passed_to_owned_powershell_capture(self):
        import services.amd_observer as observer
        with patch.object(observer,'_capture',return_value={'returncode':1}) as capture:
            observer.windows_observation('2026-10-12T00:00:00+00:00',timeout=4.)
            self.assertEqual(capture.call_args.args[1],4.)
        with self.assertRaises(ValueError):observer.windows_observation('ignored',timeout=11)

    def test_unhealthy_periodic_supervisor_gate_never_writes_gpu_permit(self):
        import test_amd_decoder_forward as common
        proof=common.proof;written=[]
        class Input(io.BytesIO):
            def write(self,value):written.append(value);return super().write(value)
        event=proof.EVENT+'{"event":"cpu-ready","gate_one":{},"checkpoint_sha256":"verified"}\n'
        child=SimpleNamespace(pid=123,stdin=Input(),stdout=io.BytesIO(event.encode()),stderr=io.BytesIO())
        class CPU:
            error=None
            def __init__(self,*args):pass
            def start(self,timeout):pass
            def require_healthy(self):pass
            def close(self):return True
        class Windows(CPU):
            error='unhealthy';samples=1
            def start(self,timeout):raise RuntimeError('unhealthy periodic observer')
        def reap(owned,process):process.update(returncode=-1,worker_exit_verified=True,forced_cleanup=True)
        with tempfile.TemporaryDirectory(dir=proof.ROOT.parent) as tmp:
            evidence=Path(tmp)
            with patch.object(proof,'EVIDENCE',evidence),patch.object(proof,'source_identity',return_value={}), \
                 patch.object(proof,'verify_committed_files',return_value={}), \
                 patch.object(proof,'prepare_private_cache',return_value={}), \
                 patch.object(proof.previous,'identity',return_value={'config':common.CONFIG,'checkpoint_sha256':'verified'}), \
                 patch.object(proof.subprocess,'run',return_value=SimpleNamespace(stdout=b'published')), \
                 patch.object(proof,'windows_observation',return_value={}),patch.object(proof,'healthy_observation',return_value=True), \
                 patch.object(proof.loading_contract,'authorize_placement'),patch.object(proof.subprocess,'Popen',return_value=child), \
                 patch.object(proof,'OwnedJob',side_effect=common.fake_owned_job), \
                 patch.object(proof.diagnostics,'ActivityMonitor',CPU),patch.object(proof,'WindowsMonitor',Windows), \
                 patch.object(proof.previous,'reap_owned_child',side_effect=reap) as cleanup:
                self.assertEqual(proof.supervise(evidence/'run','published'),1);cleanup.assert_called_once()
            self.assertNotIn(b'EXECUTE_ONE_DECODER_FORWARD\n',written)
            self.assertFalse(json.loads((evidence/'run/results.json').read_text())['observer_healthy'])


class OwnershipTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(ownership)

    def test_real_cpu_job_contains_child_after_parent_exit_and_reaps_it(self):
        job=ownership.OwnedJob();child=descendant=None
        try:
            script='import subprocess,sys; sys.stdin.readline(); p=subprocess.Popen([sys.executable,"-B","-I","-c","import time; time.sleep(10)"]); print(p.pid,flush=True)'
            child=subprocess.Popen([sys.executable,'-B','-I','-c',script],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
            job.attach(child);child.stdin.write('owned\n');child.stdin.flush();child.stdin.close()
            import psutil
            descendant=psutil.Process(int(child.stdout.readline().strip()));child.wait(timeout=2)
            self.assertGreaterEqual(job.active(),1)
            result=job.finish(timeout=2)
            self.assertTrue(result['owned_tree_exit_verified']);self.assertTrue(result['forced_job_cleanup'])
            descendant.wait(timeout=1)
            self.assertNotIn('torch',sys.modules)
        finally:
            job.close()
            if child is not None:
                if child.poll() is None:child.kill()
                child.wait(timeout=2);child.stdout.close()
            if descendant is not None:
                try:descendant.kill()
                except __import__('psutil').NoSuchProcess:pass

    def test_ownership_assignment_failure_is_fail_closed(self):
        job=ownership.OwnedJob()
        try:
            with self.assertRaises(OSError):job.attach(SimpleNamespace(_handle=0))
            self.assertEqual(job.active(),0)
        finally:job.close()

    def test_job_cleanup_timeout_never_claims_owned_tree_exit(self):
        job=ownership.OwnedJob()
        try:
            with patch.object(job,'active',return_value=1):
                with self.assertRaises(TimeoutError):job.finish(timeout=.01)
        finally:job.close()


if __name__=='__main__':unittest.main()
