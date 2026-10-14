"""One real WSL CPU cleanup/FileLock acceptance. No AI/native/GPU imports."""
import argparse
import gc
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

ROOT=Path(__file__).resolve().parents[3]
E=Path(__file__).resolve().parent
SCRATCH=Path('/home/modly/owned-cleanup-9a53e6b')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
smoke=load('existing_smoke',ROOT/'scripts/compatibility/amd_wsl_rocdxg_smoke.py')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def guest(output):
    output.mkdir(exist_ok=False)
    result={'classification':'CPU_UNKNOWN','checks':[],'forbidden_import_attempts':[],
        'native_imports':False,'GPU_actions':False,'started_utc':smoke.utc_now()}
    policy=None;fds=[]
    try:
        smoke.await_go(sys.stdin.buffer)
        result['authorization_accepted']=True
        def deny(event,args):
            if event=='import' and str(args[0]).split('.')[0] in {'torch','triton','flex_gemm','trellis2','numpy','safetensors','cumesh','o_voxel'}:
                result['forbidden_import_attempts'].append(str(args[0]));raise RuntimeError('CPU acceptance forbids AI/native import')
            if event=='ctypes.dlopen':raise RuntimeError('CPU acceptance forbids ctypes loading')
        sys.addaudithook(deny)
        versions={name:md.version(name) for name in ('torch','triton','filelock')}
        if versions!={'torch':'2.9.1+rocm7.2.1.lw.gitff65f5bc','triton':'3.5.1+rocm7.2.1.gita272dfa8','filelock':'4.1.1'}:
            raise RuntimeError('Installed dependency identity changed')
        result['installed_versions']=versions
        result['python']=sys.version.split()[0]
        if result['python']!='3.12.3' or sys.prefix!='/opt/modly-wsl/venv':raise RuntimeError('Foreign CPU runtime')
        # All fixture setup is explicitly owned. Foreign means outside policy's
        # narrower allowed subtree, never an installed application directory.
        SCRATCH.mkdir(exist_ok=False)
        owned=SCRATCH/'allowed';foreign=SCRATCH/'foreign';owned.mkdir();foreign.mkdir()
        for name in ('tmp','home'): (owned/name).mkdir()
        sentinel=foreign/'sentinel';sentinel.write_bytes(b'foreign unchanged\n')
        marker=owned/'marker';marker.write_bytes(b'owned retained\n')
        link=owned/'foreign-link';link.symlink_to(foreign,target_is_directory=True)
        os.environ.update(HOME=str(owned/'home'),TMPDIR=str(owned/'tmp'))
        tempfile.tempdir=str(owned/'tmp')
        sys.path.insert(0,str(ROOT/'scripts/compatibility'))
        from amd_wsl_native_import_policy import ImportPolicy,ImportPolicyError
        policy=ImportPolicy(owned,writable_roots=(output,))
        events=[]
        def observe(event,args):
            if event in ('os.remove','os.rmdir'):
                if len(events)>=256:raise RuntimeError('Bounded cleanup journal full')
                row={'event':event,'args':repr(args)[:600]}
                if len(args)==2 and type(args[1]) is int and args[1]>=0:
                    try:
                        s=os.fstat(args[1]);row['directory']={'fd':args[1],'path':os.readlink('/proc/self/fd/'+str(args[1])),
                            'device':s.st_dev,'inode':s.st_ino,'mode':s.st_mode,'nlink':s.st_nlink}
                    except OSError as exc:row['descriptor_error']=str(exc)
                events.append(row)
        sys.addaudithook(observe)
        policy.install_audit_hook();policy.install_profile()
        def check(name,fn,reject=False):
            before=policy.snapshot()['rejected_calls']
            smoke.journal(output,'check-entry',name=name)
            try:
                fn()
            except ImportPolicyError as exc:
                if not reject:raise
                if policy.snapshot()['rejected_calls']!=before+1:raise RuntimeError('Unexpected rejection count')
                result['checks'].append({'name':name,'outcome':'rejected_before_operation','error':str(exc)})
            else:
                if reject:
                    result['classification']='CPU_NO_GO'
                    raise RuntimeError('Unsafe request was not rejected: '+name)
                if policy.snapshot()['rejected_calls']!=before:raise RuntimeError('Positive route blocked')
                result['checks'].append({'name':name,'outcome':'accepted'})
            smoke.journal(output,'check-return',name=name)
        def temporary_cleanup():
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory);(p/'nested').mkdir();(p/'nested/file').write_bytes(b'owned')
                (p/'link').symlink_to(p/'nested/file')
            if p.exists():raise RuntimeError('TemporaryDirectory survived cleanup')
        check('actual-TemporaryDirectory-nested-cleanup',temporary_cleanup)
        tree=owned/'rmtree';tree.mkdir();(tree/'sub').mkdir();(tree/'sub/file').write_bytes(b'owned')
        def remove_tree():
            shutil.rmtree(tree)
            if tree.exists():raise RuntimeError('shutil.rmtree survived')
        check('actual-shutil-rmtree',remove_tree)
        owned_fd=os.open(owned,os.O_RDONLY|os.O_DIRECTORY);fds.append(owned_fd)
        foreign_fd=os.open(foreign,os.O_RDONLY|os.O_DIRECTORY);fds.append(foreign_fd)
        escaped_fd=os.open(link,os.O_RDONLY|os.O_DIRECTORY);fds.append(escaped_fd)
        file_fd=os.open(marker,os.O_RDONLY);fds.append(file_fd)
        (owned/'file').write_bytes(b'owned');(owned/'empty').mkdir()
        check('owned-fd-unlink',lambda:os.unlink('file',dir_fd=owned_fd))
        check('owned-fd-rmdir',lambda:os.rmdir('empty',dir_fd=owned_fd))
        check('foreign-directory',lambda:os.unlink('sentinel',dir_fd=foreign_fd),True)
        check('symlink-escaped-directory',lambda:os.unlink('sentinel',dir_fd=escaped_fd),True)
        check('nondirectory-fd',lambda:os.unlink('marker',dir_fd=file_fd),True)
        for name in ('','.', '..','../foreign/sentinel','tmp/escape','a\\b',str(sentinel)):
            check('unsafe-basename:'+repr(name),lambda name=name:os.unlink(name,dir_fd=owned_fd),True)
        deleted=owned/'deleted';deleted.mkdir();deleted_fd=os.open(deleted,os.O_RDONLY|os.O_DIRECTORY);fds.append(deleted_fd);os.rmdir(deleted)
        check('deleted-directory',lambda:os.unlink('entry',dir_fd=deleted_fd),True)
        closed_fd=os.open(owned,os.O_RDONLY|os.O_DIRECTORY);os.close(closed_fd)
        check('closed-directory-fd',lambda:os.unlink('marker',dir_fd=closed_fd),True)
        check('negative-directory-fd',lambda:os.unlink('marker',dir_fd=-2),True)
        def filelock_import():
            import filelock
            if md.version('filelock')!='4.1.1' or not Path(filelock.__file__).resolve().is_relative_to(Path('/opt/modly-wsl/venv')):
                raise RuntimeError('Foreign FileLock import')
            result['filelock_path']=filelock.__file__;result['filelock_sha256']=digest(filelock.__file__)
            strict=sys.modules['filelock._strict'];result['filelock_strict_path']=strict.__file__;result['filelock_strict_sha256']=digest(strict.__file__)
        check('genuine-FileLock-4.1.1-import',filelock_import)
        if list((owned/'tmp').iterdir()):raise RuntimeError('FileLock/TemporaryDirectory files survived')
        check('owned-symlink-entry-unlink',lambda:os.unlink('foreign-link',dir_fd=owned_fd))
        if sentinel.read_bytes()!=b'foreign unchanged\n' or marker.read_bytes()!=b'owned retained\n':raise RuntimeError('Sentinel modified')
        if set(p.name for p in owned.iterdir())!={'home','tmp','marker'}:raise RuntimeError('Unexpected owned artifacts')
        result['temporary_files_cleaned']=True;result['foreign_sentinel_unchanged']=True
        result['cleanup_events']=events;result['policy']=policy.snapshot()
        forbidden=sorted(name for name in sys.modules if name.split('.')[0] in {'torch','triton','flex_gemm','trellis2','numpy','safetensors','cumesh','o_voxel'})
        result['forbidden_modules']=forbidden
        if forbidden or result['forbidden_import_attempts']:raise RuntimeError('Forbidden library attempted')
        result['classification']='CPU_PASS'
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        for fd in fds:os.close(fd)
        if policy is not None:policy.restore_profile()
        gc.collect();result['finished_utc']=smoke.utc_now()
        smoke.save_json(output/'results.json',result)
    return 0 if result['classification']=='CPU_PASS' else 1

def helper_preflight(save=True):
    helper_dir=(ROOT/'scripts/compatibility').resolve(strict=True)
    expected=json.loads((E/'preservation-before.json').read_text())['unchanged_control_code']
    helpers={};rows={}
    for name in ('amd_bounded_gpu_proof','amd_forward_owned_job'):
        path=(helper_dir/(name+'.py')).resolve(strict=True)
        if path.parent!=helper_dir or digest(path)!=expected[name+'.py']:raise RuntimeError('Helper identity changed')
        helpers[name]=load(name,path);rows[name]={'path':str(path),'sha256':digest(path)}
    for phase in ('windows','linux'):
        if json.loads((E/(phase+'-preservation-before.json')).read_text())['status']!='PASS':raise RuntimeError('Preservation refused')
    selected=Path(__file__).resolve(strict=True)
    if selected!=(E/'acceptance.py').resolve(strict=True):raise RuntimeError('Unexpected guest')
    if (E/'CPU-attempt-started.json').exists() or (E/'run').exists():raise RuntimeError('Attempt already reserved')
    command=smoke.worker_command(smoke.guest_path(selected),smoke.guest_path(E/'run/guest'))
    if command[command.index('--worker')-1]!=smoke.guest_path(selected):raise RuntimeError('Wrong guest command')
    def fixed_helper(name):
        if name not in helpers:raise RuntimeError('Unexpected helper')
        return helpers[name]
    record={'status':'PASS','helpers':rows,'guest':str(selected),'guest_sha256':digest(selected),
        'policy_sha256':digest(helper_dir/'amd_wsl_native_import_policy.py'),'worker_command':command,'marker_absent':True,'GPU_authorized':False}
    if save:smoke.save_json(E/'helper-preflight.json',record)
    return fixed_helper,record

def host(output):
    from unittest.mock import patch
    if output.resolve()!=(E/'run').resolve():raise RuntimeError('Exclusive output required')
    helper,_=helper_preflight()
    with patch.object(smoke,'SCRIPT',Path(__file__).resolve()),patch.object(smoke,'ATTEMPT_MARKER',E/'CPU-attempt-started.json'),patch.object(smoke,'load_sibling',side_effect=helper):
        raw_exit=smoke.supervise(output)
    raw=json.loads((output/'results.json').read_text());worker=raw.get('worker',{})
    passed=(raw_exit==1 and raw.get('classification')=='UNKNOWN' and raw.get('worker_returncode')==0
        and worker.get('classification')=='CPU_PASS' and worker.get('forbidden_import_attempts')==[] and worker.get('forbidden_modules')==[]
        and worker.get('temporary_files_cleaned') is True and worker.get('foreign_sentinel_unchanged') is True
        and raw.get('observer_healthy') is True and raw.get('windows_job_cleanup',{}).get('owned_tree_exit_verified') is True
        and raw['windows_job_cleanup'].get('forced_job_cleanup') is False and raw.get('windows_job_grace',{}).get('empty') is True
        and raw.get('distro_termination',{}).get('returncode')==0 and raw.get('elapsed_seconds',121)<120
        and not any('error' in key for key in raw))
    verdict={'classification':'PASS' if passed else ('NO-GO' if worker.get('classification')=='CPU_NO_GO' else 'UNKNOWN'),'acceptance_attempts':1,'GPU_authorized':False,
        'checks':len(worker.get('checks',[])),'production_supervisor_exit':raw_exit,'raw_supervisor_classification':raw['classification'],
        'raw_GPU_contract_unused':True,'worker_classification':worker.get('classification')}
    smoke.save_json(output/'CPU-verdict.json',verdict);print(json.dumps(verdict));return 0 if passed else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=Path);p.add_argument('--output',type=Path);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    if a.worker is not None:raise SystemExit(guest(a.worker))
    if a.preflight:print(json.dumps(helper_preflight(save=False)[1]));raise SystemExit(0)
    if a.output is None:p.error('--output required')
    raise SystemExit(host(a.output))
