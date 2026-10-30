"""Offline WinAPI mocks. No real process is opened, signaled or terminated."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
sys.path.insert(0,str(Path(__file__).parent))
import amd_cgal_proof as p


class ExactHelperOwnership(unittest.TestCase):
    def exercise(self, failure=None):
        folder=Path(tempfile.mkdtemp(prefix='helper-ownership-fixture-',dir=p.BASE))
        expected=folder/'expected.exe';expected.write_bytes(b'non-executable unit fixture')
        other=folder/'other.exe';other.write_bytes(b'non-executable foreign fixture')
        st=expected.stat();pin=dict(path=str(expected),bytes=st.st_size,file_id=st.st_ino,mtime_ns=st.st_mtime_ns,sha256=p.COMPILER_HELPER_SHA)
        if failure=='pin-hash':pin['sha256']='0'*64
        kernel=MagicMock();kernel.OpenProcess.return_value=0x1234
        def query(_job,_kind,result,_size,_returned):
            result._obj.assigned=2 if failure=='multiple' else 1
            result._obj.count=result._obj.assigned
            result._obj.ids[0]=123
            return 1
        def member(_handle,_job,value):value._obj.value=int(failure!='nonmember');return 1
        def image(_handle,_flags,buffer,_length):buffer.value=str(other if failure=='path' else expected);return 1
        start=int((time.time()+11644473600-10)*10000000)
        born=start+10000000;calls=[]
        def process_times(_handle,creation,_end,_kernel,_user):
            calls.append(True)
            creation._obj.value=start-1 if failure=='start' else born+(1 if failure=='identity-change' and len(calls)>1 else 0)
            return 1
        kernel.IsProcessInJob.side_effect=member
        kernel.QueryFullProcessImageNameW.side_effect=image
        kernel.GetProcessTimes.side_effect=process_times
        kernel.TerminateProcess.return_value=1
        kernel.WaitForSingleObject.return_value=0
        job=SimpleNamespace(handle=77,api=SimpleNamespace(QueryInformationJobObject=query),active=lambda:0)
        deadline=time.monotonic()+30 if failure!='deadline' else time.monotonic()-1
        read_text=Path.read_text
        def read(path,*a,**kw):
            return json.dumps(pin) if path==p.BASE/'compiler-helper-pin.json' else read_text(path,*a,**kw)
        with patch.object(p.ctypes,'WinDLL',return_value=kernel), patch.object(p,'COMPILER_HELPER',expected), \
             patch.object(p,'compiler_outputs_complete',return_value=True),patch.object(Path,'read_text',read), \
             patch.object(p,'sha',return_value='0'*64 if failure=='hash' else p.COMPILER_HELPER_SHA):
            if failure:
                with self.assertRaises((RuntimeError,OSError)):
                    p.finish_exact_compiler_helper(job,'build-runtime',start,deadline,lambda:dict(admitted=failure!='resource'))
                kernel.TerminateProcess.assert_not_called()
            else:
                result=p.finish_exact_compiler_helper(job,'build-runtime',start,deadline,lambda:dict(admitted=True))
                self.assertFalse(result['natural_helper_exit'])
                self.assertTrue(result['owned_job_empty'])
                kernel.TerminateProcess.assert_called_once_with(0x1234,0)
                kernel.WaitForSingleObject.assert_called_once_with(0x1234,1000)
                kernel.CloseHandle.assert_called_once_with(0x1234)

    def test_positive_exact_retained_handle_cleanup(self):self.exercise()
    def test_wrong_path_never_signalled(self):self.exercise('path')
    def test_wrong_hash_never_signalled(self):self.exercise('hash')
    def test_wrong_pin_hash_never_signalled(self):self.exercise('pin-hash')
    def test_old_process_start_never_signalled(self):self.exercise('start')
    def test_creation_identity_change_never_signalled(self):self.exercise('identity-change')
    def test_nonmember_never_signalled(self):self.exercise('nonmember')
    def test_multiple_processes_never_signalled(self):self.exercise('multiple')
    def test_expired_deadline_never_signalled(self):self.exercise('deadline')
    def test_resource_failure_never_signalled(self):self.exercise('resource')


if __name__=='__main__':unittest.main(verbosity=2)
