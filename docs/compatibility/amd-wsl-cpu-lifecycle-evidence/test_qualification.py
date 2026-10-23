"""Bounded synthetic receipt checks. No Job, WSL or native worker execution."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

E=Path(__file__).resolve().parent
sys.path.insert(0,str(E.parents[2]/'api'))
s=importlib.util.spec_from_file_location('wsl_pair',E/'coordinator.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.context={'nonce':'a'*32,'job_id':'synthetic-only'}
        self.command=SimpleNamespace(context=self.context,stage_id='wsl-target')
        leader=dict(pid=11,ppid=10,uid=1000,start_ticks=101,pgrp=11,session=11)
        self.ready=dict(uid=1000,broker=dict(pid=10,uid=1000,start_ticks=100),leader=leader,
            boot_id='12345678-1234-1234-1234-123456789abc',memory=dict(rlimit_as_bytes=6*1024**3,
            aggregate_rss_limit_bytes=6*1024**3,atomic_cgroup_limit=False,aggregate_rss_peak_bytes=1024))
        linux=dict(self.ready,event='terminal',terminal='cancelled',returncode=0,linux_settled=True,
            reaped_all=True,owned_remaining=[],natural=True,forced=False,errors=[],
            cooperative_cancelled=True,reaped=[dict(pid=11,status=0)],process_identities=[leader])
        self.row=dict(terminal='cancelled',natural=True,forced=False,windows_settled=True,linux_settled=True,
            returncode=0,bootstrap_returncode=0,linux_client_returncode=0,elapsed_s=1,diagnostic_errors=[],
            context=self.context,stage_id='wsl-target',nonce=self.context['nonce'],windows=dict(active=0,total_processes=1,
            process_identities=[dict(job_membership_verified=True)]),linux=linux,
            control_messages=[dict(action='CANCEL',flushed=True)])

    def check(self):
        c.settled(self.row,'cancelled',self.command,self.ready)

    def test_exact_cooperative_linux_and_windows_receipt(self):
        self.check()

    def test_windows_exit_alone_never_proves_linux_settlement(self):
        self.row.pop('linux')
        with self.assertRaises(ValueError):self.check()

    def test_missing_cancel_ack_blocks(self):
        self.row['linux']['cooperative_cancelled']=False
        with self.assertRaises(ValueError):self.check()

    def test_changed_boot_or_identity_blocks(self):
        for key,value in (('boot_id','foreign'),('leader',dict(self.ready['leader'],start_ticks=999))):
            with self.subTest(key=key):
                original=self.row['linux'][key];self.row['linux'][key]=value
                with self.assertRaises(ValueError):self.check()
                self.row['linux'][key]=original

    def test_unverified_reaping_or_owned_processes_block(self):
        for key,value in (('reaped_all',False),('owned_remaining',[self.ready['leader']]),('reaped',[])):
            with self.subTest(key=key):
                original=self.row['linux'][key];self.row['linux'][key]=value
                with self.assertRaises(ValueError):self.check()
                self.row['linux'][key]=original

    def test_forced_uncertain_and_incomplete_windows_accounting_block(self):
        for mutate in (lambda r:r.update(forced=True),lambda r:r.update(windows_settled=False),
                       lambda r:r['windows'].update(total_processes=2)):
            with self.subTest(mutate=mutate):
                original=copy.deepcopy(self.row);mutate(self.row)
                with self.assertRaises(ValueError):self.check()
                self.row=original

    def test_foreign_nonce_and_wrong_stage_binding_block(self):
        for key,value in (('nonce','b'*32),('stage_id','wsl-sentinel')):
            original=self.row[key];self.row[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()
            self.row[key]=original

    def test_cancel_cannot_masquerade_as_sentinel_success(self):
        self.row.update(terminal='succeeded');self.row['linux'].update(terminal='succeeded',cooperative_cancelled=False)
        with self.assertRaises(ValueError):c.settled(self.row,'succeeded',self.command,self.ready)

    def test_broken_session_or_uid_blocks_readiness(self):
        for key,value in (('uid',0),('session',99),('ppid',99)):
            ready=copy.deepcopy(self.ready);ready['leader'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):c.linux_identity(ready)

    def test_repeated_nonce_readiness_and_foreign_protocol_rejected(self):
        row=dict(context=self.context,stage_id='wsl-target',nonce=self.context['nonce'],event='ready')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'protocol'
            for rows in ([row,row],[dict(row,nonce='b'*32)],[dict(row,event='worker-PASS')]):
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaises(ValueError):c.records(path,self.context,'wsl-target')

    def test_preflight_never_starts_wsl_or_reserves(self):
        s=importlib.util.spec_from_file_location('wsl_q',E/'qualification.py')
        q=importlib.util.module_from_spec(s);s.loader.exec_module(q)
        with patch.object(q,'read',side_effect=ValueError('missing admission')), \
             patch.object(q.subprocess,'Popen') as launch,patch.object(Path,'mkdir') as reserve:
            with self.assertRaises(ValueError):q.preflight()
            launch.assert_not_called();reserve.assert_not_called()

    def test_native_wsl_basename_without_original_image_is_insufficient(self):
        image=str(Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/wsl.exe')
        identity=dict(pid=22,creation_filetime=123,image=image,job_owner='test-only',job_handle=7)
        row=dict(identity,job_membership_verified=True,observed_alive={'alive':True},
                 image_authentication=dict(identity,api='QueryFullProcessImageNameW',succeeded=True),
                 exit_observation=dict(api='GetExitCodeProcess',succeeded=True,exit_code=259))
        snapshot=dict(monotonic_s=10,job_owner='test-only',enumerated_pids=[22],process_identities=[row])
        self.assertEqual(len(c.native_live(snapshot,10.1)),1)
        row['image']=str(Path(image).parent.parent/'foreign/wsl.exe')
        row['image_authentication']['image']=row['image']
        with self.assertRaises(ValueError):c.native_live(snapshot,10.1)

    def test_changed_registration_or_wsl_executable_blocks_before_execution(self):
        s=importlib.util.spec_from_file_location('wsl_q_pins',E/'qualification.py')
        q=importlib.util.module_from_spec(s);s.loader.exec_module(q)
        admission=dict(registration_before_sha256='baseline',wsl_executable_sha256='exe')
        registration={'DistributionName':'Modly-AMD-Ubuntu-24.04','Version':2,'DefaultUid':0}
        def sha(path):return 'baseline' if Path(path).name=='registration-before-private.json' else 'exe'
        with patch.object(q,'read',return_value={'registration':registration}), \
             patch.object(q,'registration_identity',return_value=registration) as query,patch.object(q,'sha',side_effect=sha):
            q.wsl_unchanged(admission)
            query.return_value=dict(registration,DefaultUid=999)
            with self.assertRaises(ValueError):q.wsl_unchanged(admission)
            query.return_value=registration
            with patch.object(q,'sha',return_value='altered'):
                with self.assertRaises(ValueError):q.wsl_unchanged(admission)


if __name__=='__main__':unittest.main(verbosity=2)
