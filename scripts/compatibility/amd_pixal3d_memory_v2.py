"""Read-only process observations. Windows clients are never treated as VM totals."""
import ctypes,os
from pathlib import Path

def process_observation(item,query):
    scope='shared_WSL_VM_candidate_not_distro_attribution' if item['name'].lower() in ('vmmem','vmmemwsl') else 'Windows_client_only_not_VM' if item['name'].lower()=='wsl.exe' else 'Windows_WSL_service_not_VM_total'
    row=dict(item,scope=scope)
    try:row.update(query(item['pid']),status='OBSERVED')
    except OSError as exc:row.update(status='UNAVAILABLE',error=type(exc).__name__+': '+str(exc))
    return row

def windows_processes():
    from ctypes import wintypes as w
    class Entry(ctypes.Structure):
        _fields_=[('size',w.DWORD),('usage',w.DWORD),('pid',w.DWORD),('heap',ctypes.c_size_t),('module',w.DWORD),('threads',w.DWORD),('parent',w.DWORD),('priority',w.LONG),('flags',w.DWORD),('exe',w.WCHAR*260)]
    class Counters(ctypes.Structure):
        _fields_=[('cb',w.DWORD),('faults',w.DWORD)]+[(n,ctypes.c_size_t) for n in ('peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile','private')]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes=[w.DWORD,w.DWORD];kernel.CreateToolhelp32Snapshot.restype=w.HANDLE
    kernel.Process32FirstW.argtypes=[w.HANDLE,ctypes.POINTER(Entry)];kernel.Process32NextW.argtypes=kernel.Process32FirstW.argtypes
    kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];kernel.OpenProcess.restype=w.HANDLE;kernel.CloseHandle.argtypes=[w.HANDLE]
    kernel.GetProcessTimes.argtypes=[w.HANDLE,*([ctypes.POINTER(w.FILETIME)]*4)]
    psapi.GetProcessMemoryInfo.argtypes=[w.HANDLE,ctypes.c_void_p,w.DWORD]
    def query(pid):
        handle=kernel.OpenProcess(0x1000|0x10,False,pid)
        if not handle:raise ctypes.WinError()
        try:
            counters=Counters();counters.cb=ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(counters),counters.cb):raise ctypes.WinError()
            times=[w.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle,*(ctypes.byref(t) for t in times)):raise ctypes.WinError()
            return dict(creation_FILETIME=times[0].dwHighDateTime*2**32+times[0].dwLowDateTime,working_set_bytes=counters.ws,
                        private_commit_bytes=counters.private,OS_peak_working_set_bytes=counters.peak_ws)
        finally:kernel.CloseHandle(handle)
    snapshot=kernel.CreateToolhelp32Snapshot(2,0)
    if snapshot==ctypes.c_void_p(-1).value:raise ctypes.WinError()
    rows=[]
    try:
        item=Entry();item.size=ctypes.sizeof(item);ok=kernel.Process32FirstW(snapshot,ctypes.byref(item))
        if not ok:raise ctypes.WinError()
        while ok:
            if item.exe.lower() in ('vmmem','vmmemwsl','wsl.exe','wslhost.exe','wslservice.exe'):
                rows.append(process_observation(dict(pid=item.pid,parent_pid=item.parent,name=item.exe),query))
            ok=kernel.Process32NextW(snapshot,ctypes.byref(item))
        if ctypes.get_last_error()!=18:raise ctypes.WinError()
    finally:kernel.CloseHandle(snapshot)
    return dict(status='OBSERVED',processes=rows,VM_process_present=any(r['name'].lower() in ('vmmem','vmmemwsl') for r in rows),
                attribution='Read-only candidate counters; may cover other distros/shared pages; not exclusive proof attribution')

def parse_meminfo(text):
    values={}
    for line in text.splitlines():
        name,rest=line.split(':',1);parts=rest.split()
        if len(parts)==2 and parts[1]=='kB':values[name]=int(parts[0])*1024
    if not all(k in values for k in ('MemTotal','MemFree','MemAvailable')):raise ValueError('Linux memory statistics incomplete')
    return values

def linux_memory():
    if os.name!='posix':raise RuntimeError('Linux-only observation')
    status={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:','VmSize:','VmHWM:')):status[line.split(':')[0]]=int(line.split()[1])*1024
    return dict(meminfo_bytes=parse_meminfo(Path('/proc/meminfo').read_text()),controller_pid=os.getpid(),controller_memory_bytes=status,
                attribution='Linux VM-wide statistics plus controller only; includes shared/unrelated guest activity')
