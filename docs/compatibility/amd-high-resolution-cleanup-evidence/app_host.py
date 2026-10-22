"""Owned installed-app acceptance session, not a neural/CPU-worker result."""
import sys
sys.dont_write_bytecode = True
import ctypes, hashlib, importlib.util, json, os, socket, subprocess, time, traceback
from pathlib import Path
E=Path(__file__).resolve().parent; ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-high-resolution-cleanup-bc9515f'
EXE=Path('C:/Users/wattz/AppData/Local/Programs/Modly/Modly.exe')
CAP=6*1024**3; SESSION_SECONDS=360; LOAD_SECONDS=180
HELPER=ROOT/'docs/compatibility/amd-brick-pbr-exposure-evidence/render_host.py'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def need(ok,msg):
    if not ok:raise ValueError(msg)
def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]

def main(preflight=False):
    need(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='bc9515fbbdc8e28cdc58b47bc1b32f4ed3369ad4','Approved source base')
    need(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='codex/amd-high-resolution-cleanup-qualification','Isolated branch')
    need(sha(HELPER)=='2d98a925a0b97aee29d21d259e40eeb290ab44386ab035e15a3463c5df0c1418','Pinned resource reader')
    spec=importlib.util.spec_from_file_location('app_resources',HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);helper.PRIVATE=P
    need(helper.resources()['admitted'],'12GiB RAM/2GiB output floor')
    with socket.socket() as s:need(s.connect_ex(('127.0.0.1',8765))!=0,'Port8765 must be unused; never kill an unrelated backend')
    manifest=json.loads((P/'app-inputs.json').read_text())
    for name,item in manifest['files'].items():
        path=Path(item['path'])
        for node in (path,*path.parents):need(not node.is_symlink() and not getattr(node.lstat(),'st_file_attributes',0)&0x400,'Plain input path')
        need(path.stat().st_nlink==1 and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'Exact app/source input: '+name)
    review=json.loads((E/'session-review.json').read_text())
    controls=[E/'app_host.py',E/'startup_probe.js',E/'renderer_observer.js',E/'inspect_app.mjs',E/'cleanup_observer.js',E/'shutdown_control.js']
    need(review['classification']=='GO' and review['sources']=={x.name:sha(x) for x in controls},'Reviewed exact session controls')
    watched=[Path(item['path']) for item in manifest['files'].values()]+[P/'app-inputs.json',*controls,E/'session-review.json',HELPER,ROOT/'scripts/compatibility/amd_forward_owned_job.py']
    before={str(x):dict(bytes=x.stat().st_size,mtime_ns=x.stat().st_mtime_ns,sha256=sha(x)) for x in watched}
    need(not any((P/n).exists() for n in ('app-session-attempt.json','app-host.json','app-running.json')),'Exclusive app session')
    if preflight:print(json.dumps(dict(classification='PASS',marker_reserved=False,resources=helper.resources())));return 0
    inspector,renderer=port(),port();need(inspector!=renderer,'Distinct local debugging ports')
    command=[str(EXE),'--inspect-brk=127.0.0.1:'+str(inspector),'--remote-debugging-address=127.0.0.1','--remote-debugging-port='+str(renderer),'--user-data-dir='+str(P/'profile')]
    sys.path.insert(0,str(ROOT/'scripts/compatibility'));from amd_forward_owned_job import OwnedJob,ExtendedLimits
    job=OwnedJob();limits=ExtendedLimits();limits.Basic.Flags=0x2300;limits.ProcessMemory=limits.JobMemory=CAP
    need(job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)),'Both6GiB app process/Job caps')
    record=dict(classification='BLOCKED',application_graphics_session=True,CPU_worker_result=False,command=command,memory_cap_bytes=CAP,session_deadline_seconds=SESSION_SECONDS,per_import_deadline_seconds=LOAD_SECONDS,resource_samples=[])
    save(P/'app-session-attempt.json',dict(command=command,source_sha256=review['sources'],input_manifest_sha256=sha(P/'app-inputs.json'),review_sha256=sha(E/'session-review.json')))
    child=None;began=time.monotonic()
    try:
        with (P/'app.stdout.txt').open('xb') as stdout,(P/'app.stderr.txt').open('xb') as stderr:
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',SELECTED_MODEL_ID='')
            child=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,cwd=P);job.attach(child)
            record['attached_before_debugger_resume']=True
            save(P/'app-running.json',dict(pid=child.pid,inspector_port=inspector,renderer_port=renderer,attached_before_debugger_resume=True,started_epoch=time.time(),shutdown_no_later_epoch=time.time()+220))
            last=-1
            while child.poll() is None:
                elapsed=time.monotonic()-began;need(elapsed<SESSION_SECONDS,'Bounded6minute cleanup session')
                need(elapsed<240 or (P/'shutdown-request.json').exists(),'Normal shutdown must begin before240seconds');sample=helper.resources();sample['elapsed_seconds']=elapsed;record['resource_samples'].append(sample);need(sample['admitted'],'Live RAM/disk floor: '+json.dumps(sample))
                stage=P/'stage.json'
                if stage.exists():
                    phase=json.loads(stage.read_text())
                    if phase.get('loading'):need(0<=time.time()-phase['started_epoch']<LOAD_SECONDS,'180second import deadline')
                time.sleep(1)
            record['returncode']=child.returncode
            grace=time.monotonic()
            while job.active() and time.monotonic()-grace<5:time.sleep(.025)
            need(job.active()==0,'Owned application tree settled without forced Job cleanup')
            need(child.returncode==0,'Installed app exited successfully')
            record['classification']='PASS'
    except BaseException as error:record.update(error=str(error),traceback=traceback.format_exc())
    finally:
        peak=ExtendedLimits()
        if job.api.QueryInformationJobObject(job.handle,9,ctypes.byref(peak),ctypes.sizeof(peak),None):record.update(peak_process_committed_bytes=peak.PeakProcessMemory,peak_job_committed_bytes=peak.PeakJobMemory)
        else:record['measurement_error']=str(ctypes.WinError())
        record['job_cleanup']=job.finish()
        if child is not None:child.wait(timeout=5)
        record['elapsed_seconds']=time.monotonic()-began
        record['input_sources_unchanged']=before=={str(x):dict(bytes=x.stat().st_size,mtime_ns=x.stat().st_mtime_ns,sha256=sha(x)) for x in watched}
        if record['job_cleanup']['forced_job_cleanup'] or 'measurement_error' in record or not record['input_sources_unchanged'] or record.get('peak_process_committed_bytes',CAP+1)>CAP or record.get('peak_job_committed_bytes',CAP+1)>CAP:record['classification']='BLOCKED'
        save(P/'app-host.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='resource_samples'}));return 0 if record['classification']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main('--preflight' in sys.argv))
