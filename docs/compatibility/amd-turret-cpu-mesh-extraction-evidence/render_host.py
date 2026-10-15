"""Bound only the installed Blender CPU preview process; never modify the PLY."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

E=Path(__file__).resolve().parent;ROOT=E.parents[2]
sys.path.insert(0,str(ROOT/'scripts/compatibility'))
from amd_forward_owned_job import OwnedJob,ExtendedLimits
from extract import digest,save,OUTPUT,GIB

class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]

if __name__=='__main__':
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    if m.available<10*GIB:raise RuntimeError('10GiB RAM admission for optional CPU preview refused')
    if (E/'preview-attempt.marker').exists():raise RuntimeError('No automatic render retry')
    scratch=OUTPUT/'blender-scratch';scratch.mkdir(exist_ok=False)
    env=dict(os.environ,BLENDER_USER_CONFIG=str(scratch),BLENDER_USER_SCRIPTS=str(scratch),
             BLENDER_USER_EXTENSIONS=str(scratch),TEMP=str(scratch),TMP=str(scratch))
    job=OwnedJob();limit=ExtendedLimits();limit.Basic.Flags=0x2000|0x100;limit.ProcessMemory=6*GIB
    if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limit),ctypes.sizeof(limit)):raise ctypes.WinError()
    command=['C:/Program Files/Blender Foundation/Blender 5.2/blender.exe','--background','--factory-startup',
             '--disable-autoexec','--threads','4','--python-exit-code','1','--python',str(E/'render_preview.py')]
    save(E/'preview-attempt.marker',{'CPU_only':True,'command':command,'available_RAM_bytes':m.available,
         'memory_cap_bytes':6*GIB,'deadline_seconds':180,'script_sha256':digest(E/'render_preview.py')})
    record={'command':command,'CPU_only':True};child=None
    try:
        with (E/'preview.stdout.txt').open('xb') as out,(E/'preview.stderr.txt').open('xb') as err:
            child=subprocess.Popen(command,stdout=out,stderr=err,env=env)
            job.attach(child);child.wait(timeout=180);record['returncode']=child.returncode
            peak=ExtendedLimits()
            if not job.api.QueryInformationJobObject(job.handle,9,ctypes.byref(peak),ctypes.sizeof(peak),None):raise ctypes.WinError()
            record['peak_committed_bytes']=peak.PeakProcessMemory
    except BaseException as exc:record['error']=str(exc)
    finally:
        record['job_cleanup']=job.finish()
        if child is not None:child.wait(timeout=5)
        save(E/'preview-host.json',record)
    raise SystemExit(0 if record.get('returncode')==0 and not record['job_cleanup']['forced_job_cleanup'] else 1)
