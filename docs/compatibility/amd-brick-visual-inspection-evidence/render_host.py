"""One owned CPU Blender import: verify the first frame before continuing."""
import ctypes,datetime,hashlib,json,os,shutil,stat,subprocess,sys,time,traceback
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
PRIVATE=ROOT.parent/'artifacts/amd-brick-visual-inspection-b536fb6';OUT=PRIVATE/'renders'
SOURCE=ROOT.parent/'artifacts/amd-brick-cpu-mesh-8473579/mesh/mesh.ply'
SOURCE_SHA='ec9736e2510c26303196c4e3fc2045a1a1d36401b2576ee68f102ee29b551ffd'
BLENDER=Path('C:/Program Files/Blender Foundation/Blender 5.2/blender.exe')
CAP=6*1024**3;RAM_FLOOR=12*1024**3;DISK_FLOOR=2*1024**3;DEADLINE=180
sys.path.insert(0,str(ROOT/'scripts/compatibility'))
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,r):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def plain(p):
 p=Path(p).absolute()
 for q in (p,*p.parents):
  s=q.lstat()
  if q.is_symlink() or getattr(s,'st_file_attributes',0)&0x400:raise ValueError('Plain non-reparse path required')
 if p.is_file() and p.stat().st_nlink!=1:raise ValueError('Single-link file required')
 return p
def admission(ram,disk):
 return dict(available_ram_bytes=ram,disk_free_bytes=disk,required_ram_bytes=RAM_FLOOR,required_disk_bytes=DISK_FLOOR,admitted=ram>=RAM_FLOOR and disk>=DISK_FLOOR)
def resources():
 class M(ctypes.Structure):
  _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
 m=M();m.length=ctypes.sizeof(m)
 if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
 return admission(m.available,shutil.disk_usage(PRIVATE).free)
def first_image(out,r):
 from PIL import Image,ImageStat
 if r.get('name')!='pos-x' or r.get('PNG')!='view-pos-x.png':raise ValueError('Exact positive-X first frame required')
 if r.get('device')!='CPU' or r.get('denoising_use_gpu') is not False or r.get('source_PLY_sha256')!=SOURCE_SHA or (r.get('vertex_count'),r.get('face_count'))!=(1957891,4173584) or r.get('geometry_unchanged_after_first_view') is not True:raise ValueError('First frame CPU/source/geometry contract')
 p=plain(Path(out)/'view-pos-x.png')
 if digest(p)!=r.get('sha256') or p.stat().st_size!=r.get('bytes'):raise ValueError('First PNG hash/size mismatch')
 with Image.open(p) as image:
  if image.format!='PNG' or image.size!=(1024,1024) or image.mode!='RGB':raise ValueError('First PNG image schema')
  image.load();spread=max(ImageStat.Stat(image).stddev)
  if spread<5:raise ValueError('First frame blank or nearly uniform')
 return dict(verified=True,path=str(p),sha256=r['sha256'],bytes=r['bytes'],dimensions=[1024,1024],maximum_channel_stddev=spread)
def process_memory(p):
 class C(ctypes.Structure):
  _fields_=[('cb',ctypes.c_uint32),('faults',ctypes.c_uint32)]+[(n,ctypes.c_size_t) for n in ('peak_working_set','working_set','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile','private')]
 c=C();c.cb=ctypes.sizeof(c);api=ctypes.WinDLL('psapi',use_last_error=True)
 api.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(C),ctypes.c_uint32];api.GetProcessMemoryInfo.restype=ctypes.c_int
 if not api.GetProcessMemoryInfo(int(p._handle),ctypes.byref(c),ctypes.sizeof(c)):raise ctypes.WinError(ctypes.get_last_error())
 return dict(peak_working_set_bytes=c.peak_working_set,peak_pagefile_bytes=c.peak_pagefile)
def main():
 from amd_forward_owned_job import OwnedJob,ExtendedLimits
 initial=resources()
 if not initial['admitted']:raise RuntimeError('Pre-marker resource admission: '+json.dumps(initial))
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!='b536fb6d140c297ec845cb5741b19370884105a5':raise ValueError('Exact approved Git revision required')
 if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='codex/amd-repeatable-generation-v1':raise ValueError('Development branch required')
 plain(PRIVATE);plain(SOURCE);plain(BLENDER);plain(E/'render_views.py')
 if digest(SOURCE)!=SOURCE_SHA or SOURCE.stat().st_size!=77751514:raise ValueError('Original mesh identity')
 old=json.loads((SOURCE.parent/'mesh.json').read_text())
 if (old['vertices'],old['triangles'])!=(1957891,4173584):raise ValueError('Original geometry counts')
 watched={SOURCE,directory_file(SOURCE,'vertices.npy'),directory_file(SOURCE,'faces.npy'),SOURCE.parent/'mesh.json',BLENDER}
 before={str(p):dict(sha256=digest(plain(p)),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in watched}
 if OUT.exists():raise FileExistsError('Fresh exclusive render directory required; no retry')
 OUT.mkdir(exist_ok=False);scratch=OUT/'blender-scratch';scratch.mkdir(exist_ok=False)
 command=[str(BLENDER),'--background','--factory-startup','--disable-autoexec','--threads','4','--python-exit-code','1','--python',str(E/'render_views.py'),'--','--output',str(OUT)]
 save(OUT/'attempt-marker.json',dict(CPU_only=True,command=command,source_sha256=SOURCE_SHA,renderer_sha256=digest(E/'render_views.py'),host_sha256=digest(__file__),memory_cap_bytes=CAP,deadline_seconds=DEADLINE,initial_resources=initial))
 job=OwnedJob();limits=ExtendedLimits();limits.Basic.Flags=0x2100;limits.ProcessMemory=CAP
 if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):job.close();raise ctypes.WinError()
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',BLENDER_USER_CONFIG=str(scratch),BLENDER_USER_SCRIPTS=str(scratch),BLENDER_USER_EXTENSIONS=str(scratch),TEMP=str(scratch),TMP=str(scratch),CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',ONEAPI_DEVICE_SELECTOR='*:cpu',OMP_NUM_THREADS='4')
 result=dict(CPU_only=True,command=command,initial_resources=initial,resource_samples=[],source_before=before,remaining_authorized=False)
 child=None;attached=False;started=time.monotonic();last_sample=-1
 try:
  with (OUT/'stdout.txt').open('xb') as out,(OUT/'stderr.txt').open('xb') as err:
   child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=out,stderr=err,env=env)
   job.attach(child);attached=True;result['attached_before_GO_START']=True
   child.stdin.write(b'GO_START\n');child.stdin.flush()
   while child.poll() is None:
    elapsed=time.monotonic()-started
    if elapsed>=DEADLINE:raise TimeoutError('180-second Blender deadline')
    if elapsed-last_sample>=1:
     r=resources();r['elapsed_seconds']=elapsed;result['resource_samples'].append(r);last_sample=elapsed
     if not r['admitted']:raise RuntimeError('Live CPU resource admission: '+json.dumps(r))
    checkpoint=OUT/'first-view.json'
    if not result['remaining_authorized'] and checkpoint.exists():
     # Writer flushes and closes the exclusive checkpoint before stdout marker.
     if b'BRICK_VISUAL_READY_GO_REMAINING' in (OUT/'stdout.txt').read_bytes()[-8192:]:
      if checkpoint.stat().st_size>32768:raise ValueError('Bounded first-frame checkpoint required')
      first=json.loads(checkpoint.read_text(encoding='utf-8'))
      result['first_frame_verified']=first_image(OUT,first)
      if DEADLINE-elapsed<60:raise TimeoutError('Insufficient remaining bounded render time')
      save(OUT/'first-frame-host-verification.json',result['first_frame_verified'])
      child.stdin.write(b'GO_REMAINING\n');child.stdin.flush();child.stdin.close()
      result['remaining_authorized']=True;result['remaining_authorized_elapsed_seconds']=elapsed
    time.sleep(.1)
   result['returncode']=child.returncode;result['process_memory']=process_memory(child)
   grace=time.monotonic()
   while job.active() and time.monotonic()-grace<5:time.sleep(.025)
 except BaseException as exc:
  result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
  if child is not None and not attached and child.poll() is None:child.kill();child.wait(timeout=5);result['unattached_owned_child_terminated']=True
 finally:
  if child is not None and child.stdin is not None and not child.stdin.closed:child.stdin.close()
  try:
   peak=ExtendedLimits()
   if not job.api.QueryInformationJobObject(job.handle,9,ctypes.byref(peak),ctypes.sizeof(peak),None):raise ctypes.WinError()
   result['peak_process_committed_bytes']=peak.PeakProcessMemory
  except BaseException as exc:result['measurement_error']=str(exc)
  result['job_cleanup']=job.finish()
  if child is not None:child.wait(timeout=5)
  after={str(p):dict(sha256=digest(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in watched}
  result.update(source_after=after,source_unchanged=after==before,elapsed_seconds=time.monotonic()-started)
  result['classification']='PASS' if result.get('returncode')==0 and result['remaining_authorized'] and result['source_unchanged'] and not any(k in result for k in ('error','measurement_error')) and result['job_cleanup']['owned_tree_exit_verified'] and not result['job_cleanup']['forced_job_cleanup'] else 'PARTIAL'
  save(OUT/'host-result.json',result)
 print(json.dumps(result),flush=True);return 0 if result['classification']=='PASS' else 1
def directory_file(source,name):return source.parent/name
if __name__=='__main__':raise SystemExit(main())
