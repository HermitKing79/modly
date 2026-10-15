"""Existing owned Blender containment, one render and optional CPU readback."""
import ctypes,json,os,subprocess,sys,time,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pbr_contract as c
sys.path.insert(0,str(c.ROOT/'scripts/compatibility'))
from amd_forward_owned_job import OwnedJob,ExtendedLimits
class Memory(ctypes.Structure):
 _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
def main():
 verify=sys.argv[1:]==['--verify'];assert not sys.argv[1:] or verify
 phase='readback' if verify else 'render';deadline=60 if verify else 180
 m=Memory();m.length=ctypes.sizeof(m)
 if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
 if m.available<10*1024**3:raise RuntimeError('10 GiB CPU preview host RAM admission failed')
 if c.digest(c.INPUT)!=c.INPUT_SHA or c.digest(c.FIELD)!=c.FIELD_SHA:raise RuntimeError('Original input hashes differ')
 proof=json.loads((c.OLD/'independent-coloured-ply-review.json').read_text());assert proof['classification']=='PASS'
 if shutil.disk_usage(c.OUT.parent).free<2*1024**3:raise RuntimeError('2 GiB preview disk admission failed')
 if not verify:
  if c.OUT.exists():raise FileExistsError('Fresh independent output required')
  c.OUT.mkdir(exist_ok=False);(c.OUT/'blender-scratch').mkdir(exist_ok=False)
 else:
  assert json.loads((c.E/'render-host.json').read_text())['returncode']==0
  assert c.BLEND.is_file() and not (c.E/'scene-readback.json').exists()
 marker=c.E/(phase+'-attempt.marker');assert not marker.exists()
 blender=Path('C:/Program Files/Blender Foundation/Blender 5.2/blender.exe');assert blender.is_file()
 scene=c.BLEND if verify else c.INPUT
 scene_sha=c.digest(scene);scratch=c.OUT/'blender-scratch'
 env=dict(os.environ,MODLY_PBR_READBACK='1' if verify else '0',PYTHONDONTWRITEBYTECODE='1',BLENDER_USER_CONFIG=str(scratch),BLENDER_USER_SCRIPTS=str(scratch),BLENDER_USER_EXTENSIONS=str(scratch),TEMP=str(scratch),TMP=str(scratch),CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',ONEAPI_DEVICE_SELECTOR='*:cpu')
 command=[str(blender),'--background','--factory-startup','--disable-autoexec','--threads','4','--python-exit-code','1',str(scene),'--python',str(c.E/'pbr_preview.py')]
 job=OwnedJob();limit=ExtendedLimits();limit.Basic.Flags=0x2000|0x100;limit.ProcessMemory=6*1024**3
 if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limit),ctypes.sizeof(limit)):job.close();raise ctypes.WinError()
 c.save(marker,{'CPU_only':True,'render_calls_authorized':0 if verify else 1,'command':command,'deadline_seconds':deadline,'memory_cap_bytes':6*1024**3,'source_scene_sha256':scene_sha,'script_sha256':c.digest(c.E/'pbr_preview.py'),'contract_sha256':c.digest(c.E/'pbr_contract.py')})
 record={'CPU_only':True,'command':command,'deadline_seconds':deadline,'memory_cap_bytes':6*1024**3,'host_available_RAM_bytes':m.available};child=None;attached=False;began=time.monotonic()
 try:
  with (c.E/(phase+'.stdout.txt')).open('xb') as out,(c.E/(phase+'.stderr.txt')).open('xb') as err:
   child=subprocess.Popen(command,stdout=out,stderr=err,env=env);job.attach(child);attached=True;child.wait(timeout=deadline);record['returncode']=child.returncode
   peak=ExtendedLimits()
   if not job.api.QueryInformationJobObject(job.handle,9,ctypes.byref(peak),ctypes.sizeof(peak),None):raise ctypes.WinError()
   record['peak_committed_bytes']=peak.PeakProcessMemory
 except BaseException as exc:
  record['error']=str(exc)
  if child is not None and not attached and child.poll() is None:child.kill();child.wait(timeout=5)
 finally:
  record['job_cleanup']=job.finish()
  if child is not None:child.wait(timeout=5)
  record['elapsed_seconds']=time.monotonic()-began;record['loaded_scene_unchanged']=c.digest(scene)==scene_sha
  record['original_scene_sha256_after']=c.digest(c.INPUT);record['material_sha256_after']=c.digest(c.FIELD)
  c.save(c.E/(phase+'-host.json'),record)
 return 0 if record.get('returncode')==0 and record['loaded_scene_unchanged'] and not record['job_cleanup']['forced_job_cleanup'] and record['original_scene_sha256_after']==c.INPUT_SHA and record['material_sha256_after']==c.FIELD_SHA else 1
if __name__=='__main__':raise SystemExit(main())
