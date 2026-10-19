"""Fixed three-stage CPU proof. Each stage is exclusive, 6 GiB / 180 seconds."""
import argparse,ctypes,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-brick-uv-bake-18c0c92/qualified'
HELPER=ROOT/'docs/compatibility/amd-brick-pbr-exposure-evidence/render_host.py'
if hashlib.sha256(HELPER.read_bytes()).hexdigest()!='2d98a925a0b97aee29d21d259e40eeb290ab44386ab035e15a3463c5df0c1418':raise ValueError('CPU containment source pin')
spec=importlib.util.spec_from_file_location('original_CPU_host',HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);helper.PRIVATE=P
sys.path.insert(0,str(ROOT/'scripts/compatibility'))
from amd_forward_owned_job import OwnedJob,ExtendedLimits


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['bake','export','compare','finish']);args=ap.parse_args()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!='18c0c92fe348e8a7fc04e9b5c9443423484ee5b6':raise ValueError('Approved exact base required')
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='codex/amd-brick-uv-bake-proof':raise ValueError('Isolated development branch required')
    initial=helper.resources()
    if not initial['admitted']:raise RuntimeError('Pre-marker CPU RAM/disk: '+json.dumps(initial))
    helper.plain(P)
    watched=list(E.glob('*.py'))+[E/'source-semantics.json',HELPER]
    sem=json.loads((E/'source-semantics.json').read_text())
    for row in sem['sources'].values():
        path=helper.plain(row['path'])
        if path.stat().st_size!=row['bytes'] or helper.digest(path)!=row['sha256']:raise ValueError('Changed source semantics pin: '+str(path))
        watched.append(path)
    for path,sha in [(helper.SOURCE,helper.SOURCE_SHA),(helper.MATERIAL,helper.MATERIAL_SHA),(helper.MAPPING,helper.MAPPING_SHA256)]:
        if helper.digest(helper.plain(path))!=sha:raise ValueError('Changed original asset')
        watched.append(path)
    watched.extend(helper.SOURCE.parent/name for name in ('vertices.npy','faces.npy','mesh.json'))
    for path in E.glob('*.py'):compile(path.read_bytes(),str(path),'exec')
    if args.stage!='bake':
        b=json.loads((P/'bake/bake.json').read_text())
        if b['classification']!='PASS' or not b['inputs_unchanged']:raise ValueError('Qualified CPU bake required')
        for name,row in b['files'].items():
            p=helper.plain(P/'bake'/name)
            if helper.digest(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise ValueError('Changed baked private artifact')
        watched.extend(x for x in (P/'bake').iterdir() if x.is_file())
    if args.stage in ('compare','finish'):
        if not (P/'export/patch.glb').is_file():raise ValueError('Actual portable patch required before fresh import')
        watched.extend(x for x in (P/'export').iterdir() if x.is_file())
    if args.stage=='finish':
        if json.loads((P/'compare/compare-checkpoint.json').read_text())['classification']!='PASS':raise ValueError('Qualified fresh import and renders required')
        watched.extend(x for x in (P/'compare').iterdir() if x.is_file())
    if args.stage in ('bake','finish'):
        command=[str(Path('D:/AI/Trellis2AMD/venv/Scripts/python.exe')),'-I','-B',str(E/('bake_patch.py' if args.stage=='bake' else 'finish.py'))]
    else:
        blender=helper.plain(helper.BLENDER)
        if helper.digest(blender)!='284f4041f98e113f3dc10654a7193ffaaa9bfdfec8b87fa116620a48b5f6d4cb':raise ValueError('Installed Blender identity changed')
        watched.append(blender)
        command=[str(blender),'--background','--factory-startup','--disable-autoexec','--threads','4','--python-exit-code','1','--python',str(E/'render_patch.py'),'--','--stage',args.stage,'--output',str(P/args.stage)]
    before={str(p):dict(sha256=helper.digest(helper.plain(p)),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in watched}
    out=P/args.stage
    if out.exists():raise FileExistsError('No stage retry or overwritten evidence')
    final=helper.resources()
    if not final['admitted']:raise RuntimeError('Final pre-marker RAM/disk floor')
    out.mkdir(exist_ok=False);scratch=out/'scratch';scratch.mkdir(exist_ok=False)
    helper.save(P/(args.stage+'-attempt.json'),dict(stage=args.stage,CPU_only=True,command=command,source_before=before,
        memory_cap_bytes=6*1024**3,deadline_seconds=180,initial_resources=final))
    record=dict(CPU_only=True,stage=args.stage,command=command,resource_samples=[],initial_resources=final)
    job=OwnedJob();limit=ExtendedLimits();limit.Basic.Flags=0x2100;limit.ProcessMemory=6*1024**3
    if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limit),ctypes.sizeof(limit)):job.close();raise ctypes.WinError()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',BLENDER_USER_CONFIG=str(scratch),BLENDER_USER_SCRIPTS=str(scratch),BLENDER_USER_EXTENSIONS=str(scratch),TEMP=str(scratch),TMP=str(scratch),
             CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',ONEAPI_DEVICE_SELECTOR='*:cpu',OPENCV_OPENCL_RUNTIME='disabled',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    child=None;attached=False;began=time.monotonic();last=-1
    try:
        with (out/'stdout.txt').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
            child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,env=env)
            job.attach(child);attached=True;record['attached_before_GO']=True
            child.stdin.write(b'GO\n');child.stdin.flush();child.stdin.close()
            while child.poll() is None:
                elapsed=time.monotonic()-began
                if elapsed>=180:raise TimeoutError('Unchanged 180-second CPU deadline')
                if elapsed-last>=1:
                    r=helper.resources();r['elapsed_seconds']=elapsed;record['resource_samples'].append(r);last=elapsed
                    if not r['admitted']:raise RuntimeError('Live CPU RAM/disk floor: '+json.dumps(r))
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
        after={str(p):dict(sha256=helper.digest(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in watched}
        record['watched_sources_unchanged']=before==after;record['elapsed_seconds']=time.monotonic()-began
        record['classification']='PASS' if record.get('returncode')==0 and before==after and not any(k in record for k in ('error','measurement_error')) and record['job_cleanup']['owned_tree_exit_verified'] and not record['job_cleanup']['forced_job_cleanup'] else 'BLOCKED'
        helper.save(P/(args.stage+'-host-result.json'),record)
    print(json.dumps({k:v for k,v in record.items() if k!='resource_samples'}),flush=True)
    return 0 if record['classification']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
