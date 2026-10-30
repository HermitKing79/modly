"""Pure offline receipt/acceptance fixtures; never access a VHDX or run WSL."""
import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).parent))
try: import amd_wsl_resize_readiness as r
except ModuleNotFoundError: r=None

class Implementation(unittest.TestCase):
    def test_readiness_checks_exist(self): self.assertIsNotNone(r,'Offline resize checks missing')

@unittest.skipIf(r is None,'Implementation missing')
class Receipts(unittest.TestCase):
    def fixture(self):
        identity={'size':29581377536,'file_id':7,'mtime_ns':11,'final_path':r.SOURCE_PATH}
        dest=dict(identity,file_id=8,final_path=r.BACKUP_PATH)
        meta={'virtual_capacity_bytes':34359738368,'region_table_sha256':'region','metadata_table_sha256':'meta'}
        worker=dict(classification='PASS',bytes=29581377536,source_identity=identity,source_after=identity,
                    destination_identity=dest,source_metadata=meta,destination_metadata=meta,handles_closed=True,
                    copy_sha256=r.BACKUP_SHA,source_sha256=r.BACKUP_SHA,destination_sha256=r.BACKUP_SHA)
        final=dict(classification='PASS',commit=r.BACKUP_COMMIT,source=r.SOURCE_PATH,destination=r.BACKUP_PATH,
                   controller_handles_closed=True,verification=worker)
        jobs=[dict(returncode=0,job_attached=True,peak_job_commit_bytes=67108864,cleanup=dict(owned_tree_exit_verified=True,forced_job_cleanup=False,
                    active_after_cleanup=0)) for _ in range(3)]
        current={'source':{'identity':identity,'metadata':meta},'backup':{'identity':dest,'metadata':meta}}
        independent={'size':29581377536,'sha256':r.BACKUP_SHA}
        return final,worker,independent,jobs,current

    def test_matching_completed_receipts(self):
        self.assertEqual(r.verify_backup(*self.fixture())['sha256'],r.BACKUP_SHA)

    def test_fixed_hash_not_derived_from_matching_receipts(self):
        f,w,d,j,c=self.fixture()
        for k in ('copy_sha256','source_sha256','destination_sha256'):w[k]='0'*64
        d['sha256']='0'*64
        with self.assertRaises(ValueError):r.verify_backup(f,w,d,j,c)

    def test_failed_top_level_or_worker_cannot_accept_provisional_manifest(self):
        for which in (0,1):
            args=list(copy.deepcopy(self.fixture()));args[which]['classification']='FAIL'
            with self.assertRaises(ValueError):r.verify_backup(*args)

    def test_wrong_commit_size_current_identity_or_capacity_rejected(self):
        for field in ('commit','bytes','identity','capacity'):
            f,w,d,j,c=copy.deepcopy(self.fixture())
            if field=='commit':f['commit']='0'*40
            elif field=='bytes':w['bytes']-=1
            elif field=='identity':c['backup']['identity']=dict(c['backup']['identity'],mtime_ns=12)
            else:c['source']['metadata']=dict(c['source']['metadata'],virtual_capacity_bytes=68719476736)
            with self.assertRaises(ValueError):r.verify_backup(f,w,d,j,c)

    def test_unsafe_cleanup_rejected(self):
        for key,value in [('returncode',1),('job_attached',False),('forced_job_cleanup',True),('owned_tree_exit_verified',False)]:
            f,w,d,j,c=self.fixture()
            if key in j[0]:j[0][key]=value
            else:j[0]['cleanup'][key]=value
            with self.assertRaises(ValueError):r.verify_backup(f,w,d,j,c)

@unittest.skipIf(r is None,'Implementation missing')
class Policy(unittest.TestCase):
    def good(self):
        return dict(native_exit=0,natural_completion=True,virtual_bytes=68719476736,
                    filesystem_bytes=68719476736,root_device_bytes=68719476736,ext4_rw=True,
                    available_bytes=5368709120,D_free_bytes=53687091200,backup_unchanged=True,
                    registration_unchanged=True,inputs_unchanged=True,guard_faults=[])

    def test_exact_native_byte_target(self):
        self.assertEqual(r.NATIVE_COMMAND[-1],'68719476736B')
        self.assertEqual(r.TARGET_BYTES,64*1024**3)

    def test_complete_result_meets_exact_disk_floors(self):
        self.assertTrue(r.accept_completion(self.good()))

    def test_virtual_growth_alone_is_not_filesystem_success(self):
        row=self.good();row['filesystem_bytes']=34359738368
        with self.assertRaises(ValueError):r.accept_completion(row)

    def test_unknown_or_incomplete_results_fail_closed(self):
        for field,value in [('native_exit',1),('natural_completion',False),('available_bytes',5368709119),
                            ('D_free_bytes',53687091199),('backup_unchanged',False),('ext4_rw',False),
                            ('filesystem_bytes',None),('registration_unchanged',False),('inputs_unchanged',False)]:
            row=self.good();row[field]=value
            with self.assertRaises(ValueError):r.accept_completion(row)

    def test_fault_latches_failure_even_if_native_later_succeeds(self):
        for fault in ('below22GiB','below12GiB','deadline','disk_floor'):
            row=self.good();row['guard_faults']=[fault]
            with self.assertRaises(ValueError):r.accept_completion(row)

    def test_startup_window_does_not_drop_to_maintenance_live_floor(self):
        r.require_native_ram(22*1024**3)
        with self.assertRaises(ValueError):r.require_native_ram(22*1024**3-1)
        with self.assertRaises(ValueError):r.require_native_ram(12*1024**3)

if __name__=='__main__':unittest.main(verbosity=2)
