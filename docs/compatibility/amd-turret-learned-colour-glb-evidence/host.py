"""Reuse owned Windows Job containment; bounded CPU exporter and fresh importer."""
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import contract as c
sys.path.insert(0,str(c.ROOT/'scripts/compatibility'))
from amd_forward_owned_job import OwnedJob, ExtendedLimits
import numpy as np

class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in
        ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]

def main():
    phase=sys.argv[1]
    assert phase in ('prepare','fixture-export','fixture-import','fixture-import-confirmation','full-export','full-import')
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    assert m.available>=12*1024**3,'12 GiB host RAM admission failed'
    assert shutil.disk_usage(c.OUT.parent).free>=3*1024**3,'3 GiB disk admission failed'
    sources=c.preserve()
    if phase=='prepare':
        assert not c.OUT.exists()
        c.OUT.mkdir(exist_ok=False)
        xyz,faces,rgb=c.original()
        fixture_xyz=np.array([[.125,.25,.375],[1.125,.25,.375],[.125,1.25,.375],[.125,.25,1.375]],dtype='<f4')
        fixture_faces=np.array([[0,1,2],[0,3,1],[0,2,3]],dtype='<i4')
        np.savez(c.OUT/'fixture-source.npz',xyz=fixture_xyz,faces=fixture_faces,rgb=rgb[[100,1000,10000,100000]])
        c.save(c.E/'preservation-before.json',{'sources':sources,'host_available_RAM_bytes':m.available,
            'disk_free_bytes':shutil.disk_usage(c.OUT.parent).free,'fixture_uses_genuine_field_RGB_rows':[100,1000,10000,100000]})
        return 0
    if phase in ('fixture-import','fixture-import-confirmation'):assert json.loads((c.E/'fixture-export-host.json').read_text())['acceptance']
    if phase=='full-export':assert json.loads((c.E/'fixture-import-confirmation-host.json').read_text())['acceptance']
    if phase=='full-import':assert json.loads((c.E/'full-export-host.json').read_text())['acceptance']
    deadline=300 if phase=='full-export' else 180 if phase=='full-import' else 60
    cap=(8 if phase.startswith('full-') else 2)*1024**3
    scratch=c.OUT/(phase+'-scratch');scratch.mkdir(exist_ok=False)
    blender=Path('C:/Program Files/Blender Foundation/Blender 5.2/blender.exe');assert blender.is_file()
    command=[str(blender),'--background','--factory-startup','--disable-autoexec','--threads','4','--python-exit-code','1']
    if phase=='full-export':command.append(str(c.SCENE))
    command+=['--python',str(c.E/'blender_glb.py')]
    env=dict(os.environ,MODLY_GLB_PHASE=phase,PYTHONDONTWRITEBYTECODE='1',
        BLENDER_USER_CONFIG=str(scratch),BLENDER_USER_SCRIPTS=str(scratch),BLENDER_USER_EXTENSIONS=str(scratch),
        TEMP=str(scratch),TMP=str(scratch),CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',ONEAPI_DEVICE_SELECTOR='*:cpu')
    job=OwnedJob();limits=ExtendedLimits();limits.Basic.Flags=0x2000|0x100;limits.ProcessMemory=cap
    if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):job.close();raise ctypes.WinError()
    record={'phase':phase,'CPU_only':True,'command':command,'deadline_seconds':deadline,'memory_cap_bytes':cap,
        'host_available_RAM_bytes':m.available,'disk_free_bytes':shutil.disk_usage(c.OUT.parent).free,
        'scripts':{name:c.digest(c.E/name) for name in ('blender_glb.py','contract.py','host.py')}}
    c.save(c.E/(phase+'-attempt.json'),record)
    child=None;attached=False;began=time.monotonic()
    try:
        with (c.E/(phase+'.stdout.txt')).open('xb') as out,(c.E/(phase+'.stderr.txt')).open('xb') as err:
            child=subprocess.Popen(command,stdout=out,stderr=err,env=env);job.attach(child);attached=True
            child.wait(timeout=deadline);record['returncode']=child.returncode
            peak=ExtendedLimits()
            if not job.api.QueryInformationJobObject(job.handle,9,ctypes.byref(peak),ctypes.sizeof(peak),None):raise ctypes.WinError()
            record['peak_committed_bytes']=peak.PeakProcessMemory
    except BaseException as exc:
        record['error']=repr(exc)
        if child is not None and not attached and child.poll() is None:child.kill();child.wait(timeout=5)
    finally:
        record['job_cleanup']=job.finish()
        if child is not None:child.wait(timeout=5)
        record['elapsed_seconds']=time.monotonic()-began
        record['sources_after']=c.preserve()
        record['acceptance']=(record.get('returncode')==0 and not record['job_cleanup']['forced_job_cleanup']
            and record['job_cleanup']['owned_tree_exit_verified'] and (c.E/(phase+'.json')).is_file())
        c.save(c.E/(phase+'-host.json'),record)
    print(json.dumps(record),flush=True)
    return 0 if record['acceptance'] else 1

if __name__=='__main__':raise SystemExit(main())
