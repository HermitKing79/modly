"""Two explicit stages of this CPU experiment; unchanged owned Job boundaries."""
import sys
sys.dont_write_bytecode=True
import argparse,ctypes,hashlib,importlib.util,json,os,subprocess,time,traceback
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
sys.path.insert(0,str(E))
from inputs import P,authenticate,digest,plain,save,require
BASE='bad7ba4e9e8c3b14539d02a0c3930080891e0c56'
BRANCH='codex/amd-connected-foliage-uv'
HELPER=ROOT/'docs/compatibility/amd-brick-pbr-exposure-evidence/render_host.py'
require(digest(plain(HELPER))=='2d98a925a0b97aee29d21d259e40eeb290ab44386ab035e15a3463c5df0c1418','Original CPU resource policy')
spec=importlib.util.spec_from_file_location('connected_original_cpu_limits',HELPER)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);helper.PRIVATE=P
sys.path.insert(0,str(ROOT/'scripts/compatibility'))
from amd_forward_owned_job import OwnedJob,ExtendedLimits
BLENDER=Path('C:/Program Files/Blender Foundation/Blender 5.2/blender.exe')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('capacity','unwrap'));ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE,'Exact approved base')
    require(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==BRANCH,'Isolated branch')
    initial=helper.resources();require(initial['admitted'],'Pre-marker resources: '+json.dumps(initial))
    plain(P);_,_,watched,_=authenticate()
    reused=[HELPER,ROOT/'scripts/compatibility/amd_forward_owned_job.py']
    for folder,names in [('amd-brick-pbr-mip-evidence',('assets.py',)),('amd-brick-defect-diagnosis-evidence',('diagnose.py',)),('amd-brick-seam-uv-bake-evidence',('common.py','uv_layout.py')),('amd-brick-mip-safe-atlas-evidence',('footprint.py',)),('amd-pbr-foliage-evidence',('foliage_layout.py',))]:
        reused += [ROOT/'docs/compatibility'/folder/n for n in names]
    for path in reused:
        require(path.read_bytes()==subprocess.check_output(['git','show',BASE+':'+path.relative_to(ROOT).as_posix()],cwd=ROOT),'Historical source pin: '+str(path))
    sources=list(E.glob('*.py'));watched+=reused+sources+[E/'qualification-plan.md']
    for path in sources:compile(path.read_bytes(),str(path),'exec')
    review_path=plain(E/(args.stage+'-candidate-review.json'));review=json.loads(review_path.read_text())
    require(review['classification']=='GO','Independent exact candidate GO required')
    for name,sha in review['candidate_source_SHA256'].items():require(digest(plain(E/name))==sha,'Reviewed source changed: '+name)
    require(digest(E/'qualification-plan.md')==review['qualification_plan_SHA256'],'Reviewed limits/method changed')
    watched.append(review_path)
    if args.stage=='capacity':
        command=['D:/AI/Trellis2AMD/venv/Scripts/python.exe','-I','-B',str(E/'capacity.py')]
    else:
        capacity=P/'capacity/run/result.json'
        require(json.loads(plain(capacity).read_text())['classification']=='PASS','Actual capacity gate must pass first')
        require(digest(capacity)==review['capacity_result_SHA256'],'Reviewed actual capacity result')
        require(digest(plain(BLENDER))==review['blender_executable_SHA256'],'Installed Blender identity')
        watched += [capacity,BLENDER]
        command=[str(BLENDER),'--background','--factory-startup','--disable-autoexec','--threads','4','--python-exit-code','1','--python',str(E/'unwrap.py')]
    watched=list(dict.fromkeys(watched));phase=P/args.stage
    if args.preflight:
        print(json.dumps(dict(classification='PASS',stage=args.stage,no_marker_reserved=True,resources=initial,watched_files=len(watched),command=command)));return 0
    require(not phase.exists(),'Exclusive one-time stage; no retry')
    before={str(path):dict(bytes=plain(path).stat().st_size,sha256=digest(path),mtime_ns=path.stat().st_mtime_ns) for path in watched}
    final=helper.resources();require(final['admitted'],'Final pre-marker resource floor: '+json.dumps(final))
    phase.mkdir();out=phase/'run';out.mkdir();scratch=out/'scratch';scratch.mkdir()
    save(phase/'attempt.json',dict(CPU_only=True,stage=args.stage,command=command,base=BASE,source_before=before,initial_resources=final,process_memory_bytes=6*1024**3,deadline_seconds=180))
    job=OwnedJob();limit=ExtendedLimits();limit.Basic.Flags=0x2100;limit.ProcessMemory=6*1024**3
    if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limit),ctypes.sizeof(limit)):job.close();raise ctypes.WinError()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TEMP=str(scratch),TMP=str(scratch),BLENDER_USER_CONFIG=str(scratch),BLENDER_USER_SCRIPTS=str(scratch),BLENDER_USER_EXTENSIONS=str(scratch),CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',ONEAPI_DEVICE_SELECTOR='*:cpu',OPENCV_OPENCL_RUNTIME='disabled',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    record=dict(CPU_only=True,command=command,initial_resources=final,resource_samples=[])
    child=None;attached=False;began=time.monotonic();last=-1
    try:
        with (out/'stdout.txt').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
            child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,env=env);job.attach(child);attached=True
            record['attached_before_GO']=True;child.stdin.write(b'GO\n');child.stdin.flush();child.stdin.close()
            while child.poll() is None:
                elapsed=time.monotonic()-began
                if elapsed>=180:raise TimeoutError('Unchanged 180-second CPU deadline')
                if elapsed-last>=1:
                    sample=helper.resources();sample['elapsed_seconds']=elapsed;record['resource_samples'].append(sample);last=elapsed
                    require(sample['admitted'],'Live resource floor: '+json.dumps(sample))
                time.sleep(.1)
            record['returncode']=child.returncode
            grace=time.monotonic()
            while job.active() and time.monotonic()-grace<5:time.sleep(.025)
    except BaseException as exc:
        record.update(error=str(exc),traceback=traceback.format_exc())
        if child is not None and not attached and child.poll() is None:child.kill();child.wait(timeout=5)
    finally:
        if child is not None and child.stdin is not None and not child.stdin.closed:child.stdin.close()
        peak=ExtendedLimits()
        if job.api.QueryInformationJobObject(job.handle,9,ctypes.byref(peak),ctypes.sizeof(peak),None):record['peak_process_committed_bytes']=peak.PeakProcessMemory
        else:record['measurement_error']=str(ctypes.WinError())
        record['job_cleanup']=job.finish()
        if child is not None:child.wait(timeout=5)
        after={str(path):dict(bytes=path.stat().st_size,sha256=digest(path),mtime_ns=path.stat().st_mtime_ns) for path in watched}
        record.update(watched_sources_unchanged=before==after,elapsed_seconds=time.monotonic()-began)
        natural=record.get('returncode')==0 and before==after and not any(k in record for k in ('error','measurement_error')) and record['job_cleanup']['owned_tree_exit_verified'] and not record['job_cleanup']['forced_job_cleanup']
        record['execution_classification']='PASS_NATURAL_CPU_COMPLETION' if natural else 'BLOCKED'
        if natural:record['experiment_classification']=json.loads((out/'result.json').read_text())['classification']
        save(phase/'host-result.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='resource_samples'}),flush=True)
    return 0 if record['execution_classification']=='PASS_NATURAL_CPU_COMPLETION' else 1


if __name__=='__main__':raise SystemExit(main())
