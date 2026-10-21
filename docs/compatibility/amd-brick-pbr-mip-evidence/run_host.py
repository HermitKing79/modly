"""One CPU-only filtering worker, reusing the established owned Job limits."""
import argparse,ctypes,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from assets import ROOT,P,PRIOR,AUTH,AUTH_SHA,authenticate,digest,plain,save,decode_glb,align_uv,GLB
BASE='0f2fd9250e8a2825df240a1f948afed0b07d31fa'
BRANCH='codex/amd-brick-pbr-mipmap-qualification'
HELPER=ROOT/'docs/compatibility/amd-brick-pbr-exposure-evidence/render_host.py'
if digest(plain(HELPER))!='2d98a925a0b97aee29d21d259e40eeb290ab44386ab035e15a3463c5df0c1418':raise ValueError('CPU containment source pin')
spec=importlib.util.spec_from_file_location('original_CPU_containment',HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);helper.PRIVATE=P
sys.path.insert(0,str(ROOT/'scripts/compatibility'))
from amd_forward_owned_job import OwnedJob,ExtendedLimits

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=BASE:raise ValueError('Exact approved starting commit')
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!=BRANCH:raise ValueError('Isolated development branch')
    initial=helper.resources()
    if not initial['admitted']:raise RuntimeError('Pre-marker live resource floor: '+json.dumps(initial))
    plain(P);_,_,watched,_=authenticate()
    watched.extend(list(E.glob('*.py'))+[HELPER,E/'qualification-plan.md'])
    for path in E.glob('*.py'):compile(path.read_bytes(),str(path),'exec')
    # Actual accepted GLB/data metadata checks before reserving the only run.
    import numpy as np
    glb=decode_glb(GLB.read_bytes());b=PRIOR/'bake'
    v,f,uv=[np.load(b/(n+'.npy'),allow_pickle=False) for n in ('vertices','faces','uv')]
    _,error=align_uv(v,f,uv,glb)
    review=plain(E/'independent-candidate-review.json');r=json.loads(review.read_text())
    if r['classification']!='GO':raise ValueError('Independent pre-execution GO required')
    for name,sha in r['candidate_source_SHA256'].items():
        if digest(plain(E/name))!=sha:raise ValueError('Reviewed source changed: '+name)
    if r['qualification_plan_SHA256']!=digest(E/'qualification-plan.md'):raise ValueError('Reviewed criteria changed')
    watched.append(review)
    command=['D:/AI/Trellis2AMD/venv/Scripts/python.exe','-I','-B',str(E/'experiment.py')]
    if args.preflight:
        print(json.dumps(dict(classification='PASS',no_marker_reserved=True,command=command,resources=initial,GLB_UV_error=error,watched_files=len(watched))));return 0
    before={str(path):dict(bytes=plain(path).stat().st_size,sha256=digest(path),mtime_ns=path.stat().st_mtime_ns) for path in watched}
    out=P/'run'
    if out.exists() or (P/'attempt.json').exists():raise FileExistsError('Exclusive one-time filtering experiment; no retry')
    final=helper.resources()
    if not final['admitted']:raise RuntimeError('Final pre-marker resource floor: '+json.dumps(final))
    out.mkdir(exist_ok=False);scratch=out/'scratch';scratch.mkdir(exist_ok=False)
    save(P/'attempt.json',dict(CPU_only=True,command=command,base=BASE,source_before=before,initial_resources=final,process_memory_bytes=6*1024**3,deadline_seconds=180))
    record=dict(CPU_only=True,command=command,resource_samples=[],initial_resources=final)
    job=OwnedJob();limit=ExtendedLimits();limit.Basic.Flags=0x2100;limit.ProcessMemory=6*1024**3
    if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limit),ctypes.sizeof(limit)):job.close();raise ctypes.WinError()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TEMP=str(scratch),TMP=str(scratch),CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',ONEAPI_DEVICE_SELECTOR='*:cpu',OPENCV_OPENCL_RUNTIME='disabled',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    child=None;attached=False;began=time.monotonic();last=-1
    try:
        with (out/'stdout.txt').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
            child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,env=env);job.attach(child);attached=True
            record['attached_before_GO']=True;child.stdin.write(b'GO\n');child.stdin.flush();child.stdin.close()
            while child.poll() is None:
                elapsed=time.monotonic()-began
                if elapsed>=180:raise TimeoutError('Unchanged 180-second CPU deadline')
                if elapsed-last>=1:
                    resources=helper.resources();resources['elapsed_seconds']=elapsed;record['resource_samples'].append(resources);last=elapsed
                    if not resources['admitted']:raise RuntimeError('Live resource floor: '+json.dumps(resources))
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
        save(P/'host-result.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='resource_samples'}),flush=True)
    return 0 if record['execution_classification']=='PASS_NATURAL_CPU_COMPLETION' else 1

if __name__=='__main__':raise SystemExit(main())
