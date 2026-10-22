"""Experiment-specific adapter to the unchanged owned Windows CPU Job."""
import sys
sys.dont_write_bytecode=True
import ctypes,importlib.util,json,os,subprocess,time,traceback
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from common import ROOT,P,plain,digest,save,require,authenticate
BASE='e262ad7c37f108e1cea456775b9b8c4b11cc9ed8'
BRANCH='codex/amd-brick-safe-simplification-census'
CAP=6*1024**3;DEADLINE=180
HELPER=ROOT/'docs/compatibility/amd-brick-pbr-exposure-evidence/render_host.py'
PYTHON='D:/AI/Trellis2AMD/venv/Scripts/python.exe'

def main(preflight=False,verify=False):
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE,'Approved base')
    require(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==BRANCH,'Isolated diagnostic branch')
    require(digest(plain(HELPER))=='2d98a925a0b97aee29d21d259e40eeb290ab44386ab035e15a3463c5df0c1418','Unchanged CPU monitor')
    spec=importlib.util.spec_from_file_location('attribution_cpu_monitor',HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);helper.PRIVATE=P
    require((helper.CAP,helper.DEADLINE,helper.RAM_FLOOR,helper.DISK_FLOOR)==(CAP,DEADLINE,12*1024**3,2*1024**3),'Unchanged safeguards')
    initial=helper.resources();require(initial['admitted'],'Initial resource floor: '+json.dumps(initial))
    sources=sorted(E.glob('*.py'));review=plain(E/'independent-candidate-review.json');r=json.loads(review.read_text())
    require(r['classification']=='GO' and r['source_SHA256']=={p.name:digest(plain(p)) for p in sources},'Exact independently reviewed diagnostic sources')
    require(r['plan_SHA256']==digest(E/'census-plan.md') and r['focused_tests_SHA256']==digest(E/'focused-tests.txt'),'Frozen diagnostic plan and focused tests')
    watched=authenticate()+sources+[review,E/'census-plan.md',E/'focused-tests.txt',HELPER,ROOT/'scripts/compatibility/amd_forward_owned_job.py']
    if verify:
        predecessor=plain(P/'diagnose-host.json');result=plain(P/'diagnose/result.json')
        require(predecessor.stat().st_size<2*1024**2 and result.stat().st_size<1024**2,'Bounded exact census predecessor')
        prior_host=json.loads(predecessor.read_text());prior=json.loads(result.read_text())
        require(prior_host['classification']=='PASS' and prior_host['worker_receipt_sha256']==digest(result) and prior.get('census_completed') is True,'Completed owned census required before readback')
        watched.extend((predecessor,result))
        for name,row in prior['files'].items():
            require(Path(name).name==name,'Private census manifest basename')
            path=plain(P/'diagnose'/name)
            require(path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],'Exact durable census predecessor: '+name)
            watched.append(path)
    watched=list(dict.fromkeys(watched));before={str(p):dict(sha256=digest(plain(p)),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in watched}
    phase='verify' if verify else 'diagnose'
    command=[PYTHON,'-I','-B',str(E/('verify_worker.py' if verify else 'worker.py'))]
    plain(command[-1])
    out=P/phase;marker=P/(phase+'-attempt.json');receipt=P/(phase+'-host.json')
    require(not any(p.exists() for p in (out,marker,receipt)),'Fresh exclusive diagnostics; no overwrite')
    if preflight:print(json.dumps(dict(classification='PASS',marker_reserved=False,resources=initial,watched_files=len(watched),command=command)));return 0
    final=helper.resources();require(final['admitted'],'Final resource floor: '+json.dumps(final))
    out.mkdir();scratch=out/'scratch';scratch.mkdir()
    sys.path.insert(0,str(ROOT/'scripts/compatibility'));from amd_forward_owned_job import OwnedJob,ExtendedLimits
    job=OwnedJob();limits=ExtendedLimits();limits.Basic.Flags=0x2300;limits.ProcessMemory=limits.JobMemory=CAP
    if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):job.close();raise ctypes.WinError()
    save(marker,dict(CPU_only=True,source_before=before,command=command,base=BASE,memory_cap_bytes=CAP,deadline_seconds=DEADLINE,resources=final))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TEMP=str(scratch),TMP=str(scratch),TMPDIR=str(scratch),CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    child=None;began=time.monotonic();last=-1
    record=dict(CPU_only=True,command=command,resource_samples=[],initial_resources=final,memory_cap_bytes=CAP,deadline_seconds=DEADLINE)
    try:
        with (out/'stdout.txt').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
            child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,env=env);job.attach(child);record['attached_before_GO']=True
            child.stdin.write(b'GO\n');child.stdin.flush();child.stdin.close();record['exact_binary_GO_sent']=True
            while child.poll() is None:
                elapsed=time.monotonic()-began
                if elapsed>=DEADLINE:raise TimeoutError('180-second CPU deadline')
                if elapsed-last>=1:
                    sample=helper.resources();sample['elapsed_seconds']=elapsed;record['resource_samples'].append(sample);last=elapsed
                    require(sample['admitted'],'Live CPU floor: '+json.dumps(sample))
                time.sleep(.1)
            record['returncode']=child.returncode
            grace=time.monotonic()
            while job.active() and time.monotonic()-grace<5 and time.monotonic()-began<DEADLINE:time.sleep(.025)
            require(job.active()==0 and time.monotonic()-began<=DEADLINE,'Natural owned settlement inside deadline')
            require(child.returncode==0,'Diagnostic worker failed')
            result=plain(out/'result.json');allowed={'PASS','BLOCKED'} if verify else {'BLOCKED','CENSUS_COMPLETE'}
            require(result.stat().st_size<1024**2 and json.loads(result.read_text())['classification'] in allowed,'Read-only phase concluded without simplification')
            record['worker_receipt_sha256']=digest(result)
    except BaseException as error:
        record.update(error=str(error),traceback=traceback.format_exc())
        if child is not None and not record.get('attached_before_GO') and child.poll() is None:child.kill();child.wait(timeout=5)
    finally:
        if child is not None and child.stdin is not None and not child.stdin.closed:child.stdin.close()
        peak=ExtendedLimits()
        if job.api.QueryInformationJobObject(job.handle,9,ctypes.byref(peak),ctypes.sizeof(peak),None):record.update(peak_process_committed_bytes=peak.PeakProcessMemory,peak_job_committed_bytes=peak.PeakJobMemory)
        else:record['measurement_error']=str(ctypes.WinError())
        record['job_cleanup']=job.finish()
        if child is not None:child.wait(timeout=5)
        record['sources_unchanged']=before=={str(p):dict(sha256=digest(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in watched}
        record['elapsed_seconds']=time.monotonic()-began
        cleanup=record['job_cleanup'];record['classification']='PASS' if record.get('returncode')==0 and record.get('worker_receipt_sha256') and record['sources_unchanged'] and not any(x in record for x in ('error','measurement_error')) and cleanup['owned_tree_exit_verified'] and not cleanup['forced_job_cleanup'] and record.get('peak_process_committed_bytes',CAP+1)<=CAP and record.get('peak_job_committed_bytes',CAP+1)<=CAP else 'BLOCKED'
        save(receipt,record)
    print(json.dumps({k:v for k,v in record.items() if k!='resource_samples'}));return 0 if record['classification']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main('--preflight' in sys.argv[1:],'--verify' in sys.argv[1:]))
