"""Private Windows job containment for the one owned worker and its descendants."""
import ctypes
from ctypes import wintypes as wt
import time


class BasicLimits(ctypes.Structure):
    _fields_=[('ProcessTime',ctypes.c_int64),('JobTime',ctypes.c_int64),('Flags',wt.DWORD),
              ('MinWorkingSet',ctypes.c_size_t),('MaxWorkingSet',ctypes.c_size_t),('ActiveLimit',wt.DWORD),
              ('Affinity',ctypes.c_size_t),('Priority',wt.DWORD),('Scheduling',wt.DWORD)]


class IO(ctypes.Structure):
    _fields_=[(name,ctypes.c_uint64) for name in ('ReadOps','WriteOps','OtherOps','ReadBytes','WriteBytes','OtherBytes')]


class ExtendedLimits(ctypes.Structure):
    _fields_=[('Basic',BasicLimits),('Counters',IO),('ProcessMemory',ctypes.c_size_t),('JobMemory',ctypes.c_size_t),
              ('PeakProcessMemory',ctypes.c_size_t),('PeakJobMemory',ctypes.c_size_t)]


class Accounting(ctypes.Structure):
    _fields_=[(name,ctypes.c_int64) for name in ('UserTime','KernelTime','PeriodUserTime','PeriodKernelTime')]+[
        (name,wt.DWORD) for name in ('PageFaults','TotalProcesses','ActiveProcesses','TerminatedProcesses')]


class OwnedJob:
    def __init__(self):
        if ctypes.sizeof(ExtendedLimits)!=144 or ctypes.sizeof(Accounting)!=48:
            raise RuntimeError('Unexpected Windows 64-bit job ABI')
        self.api=ctypes.WinDLL('kernel32',use_last_error=True);self.handle=None
        signatures={
            'CreateJobObjectW':([ctypes.c_void_p,wt.LPCWSTR],wt.HANDLE),
            'SetInformationJobObject':([wt.HANDLE,ctypes.c_int,ctypes.c_void_p,wt.DWORD],wt.BOOL),
            'AssignProcessToJobObject':([wt.HANDLE,wt.HANDLE],wt.BOOL),
            'QueryInformationJobObject':([wt.HANDLE,ctypes.c_int,ctypes.c_void_p,wt.DWORD,ctypes.c_void_p],wt.BOOL),
            'TerminateJobObject':([wt.HANDLE,wt.UINT],wt.BOOL),'CloseHandle':([wt.HANDLE],wt.BOOL)}
        for name,(args,result) in signatures.items():
            func=getattr(self.api,name);func.argtypes=args;func.restype=result
        # NULL security attributes: non-inheritable. Unnamed: no external owner.
        self.handle=self.api.CreateJobObjectW(None,None)
        if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
        limits=ExtendedLimits();limits.Basic.Flags=0x2000  # KILL_ON_JOB_CLOSE; no breakaway flags.
        if not self.api.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            error=ctypes.WinError(ctypes.get_last_error());self.close();raise error

    def attach(self,child):
        # Existing Python3.12 Windows Popen owns this live process handle.
        if not self.api.AssignProcessToJobObject(self.handle,int(child._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def active(self):
        row=Accounting()
        if not self.api.QueryInformationJobObject(self.handle,1,ctypes.byref(row),ctypes.sizeof(row),None):
            raise ctypes.WinError(ctypes.get_last_error())
        return row.ActiveProcesses

    def finish(self,timeout=5.):
        began=time.monotonic()
        try:
            before=self.active()
            if before and not self.api.TerminateJobObject(self.handle,1):raise ctypes.WinError(ctypes.get_last_error())
            while self.active():
                if time.monotonic()-began>=timeout:raise TimeoutError('Owned job tree exit unverified')
                time.sleep(min(.025,max(0.,timeout-(time.monotonic()-began))))
            return {'owned_tree_exit_verified':True,'active_before_cleanup':before,'active_after_cleanup':0,
                    'forced_job_cleanup':bool(before),'cleanup_seconds':time.monotonic()-began}
        finally:self.close()

    def close(self):
        if self.handle is not None:
            handle=self.handle
            if not self.api.CloseHandle(handle):raise ctypes.WinError(ctypes.get_last_error())
            self.handle=None
