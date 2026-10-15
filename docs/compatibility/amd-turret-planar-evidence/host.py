"""One bounded CPU Blender worker; no master mesh enters a modifier."""
import ctypes,json,os,shutil,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as c
sys.path.insert(0,str(c.ROOT/'scripts/compatibility'))
from amd_forward_owned_job import OwnedJob,ExtendedLimits
class Memory(ctypes.Structure):
 _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
def main():
 m=Memory();m.length=ctypes.sizeof(m)
 if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
 assert m.available>=6*1024**3 and shutil.disk_usage(c.OUT.parent).free>=2*1024**3
 sources=c.preserve()
 assert not c.OUT.exists();c.OUT.mkdir(exist_ok=False)
 c.save(c.E/'preservation-before.json',sources)
 scratch=c.OUT/'scratch';scratch.mkdir(exist_ok=False)
 command=['C:/Program Files/Blender Foundation/Blender 5.2/blender.exe','--background','--factory-startup','--disable-autoexec','--threads','4','--python-exit-code','1','--python',str(c.E/'worker.py')]
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',BLENDER_USER_CONFIG=str(scratch),BLENDER_USER_SCRIPTS=str(scratch),BLENDER_USER_EXTENSIONS=str(scratch),TEMP=str(scratch),TMP=str(scratch),CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',ONEAPI_DEVICE_SELECTOR='*:cpu')
 job=OwnedJob();limits=ExtendedLimits();limits.Basic.Flags=0x2000|0x100;limits.ProcessMemory=2*1024**3
 if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):job.close();raise ctypes.WinError()
 paths={n:c.E/n for n in ('host.py','common.py','worker.py')}
 paths.update({'prior/'+n:c.PRIOR/n for n in ('common.py','core.py','metrics.py')})
 record={'command':command,'CPU_only':True,'deadline_seconds':120,'memory_cap_bytes':2*1024**3,'host_available_RAM_bytes':m.available,'scripts':{n:c.digest(p) for n,p in paths.items()}}
 c.save(c.E/'attempt.json',record)
 child=None;attached=False;began=time.monotonic()
 try:
  with (c.E/'stdout.txt').open('xb') as out,(c.E/'stderr.txt').open('xb') as err:
   child=subprocess.Popen(command,stdout=out,stderr=err,env=env);job.attach(child);attached=True;child.wait(timeout=120)
   record['returncode']=child.returncode;peak=ExtendedLimits()
   if not job.api.QueryInformationJobObject(job.handle,9,ctypes.byref(peak),ctypes.sizeof(peak),None):raise ctypes.WinError()
   record['peak_committed_bytes']=peak.PeakProcessMemory
 except BaseException as exc:
  record['error']=repr(exc)
  if child is not None and not attached and child.poll() is None:child.kill();child.wait(timeout=5)
 finally:
  record['job_cleanup']=job.finish()
  if child is not None:child.wait(timeout=5)
  record['elapsed_seconds']=time.monotonic()-began;record['sources_after']=c.preserve()
  record['completed']=(record.get('returncode')==0 and not record['job_cleanup']['forced_job_cleanup'] and (c.E/'results.json').is_file())
  c.save(c.E/'host.json',record);c.save(c.OUT/'host.json',record)
 print(json.dumps(record),flush=True)
 return 0 if record['completed'] else 1
if __name__=='__main__':raise SystemExit(main())
