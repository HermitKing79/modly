"""One metadata-only guest survey; no binary contents, children or model imports."""
import argparse,heapq,json,os,re,signal,stat,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import amd_pixal3d_proof_common as c
from amd_pixal3d_owned_proc_v3 import ProcFS,process_rows,owned_members,signal_verified

CASE=c.ROOT.parent/'artifacts/amd-closed-solid-disk-survey-standalone-v2'
CAP=512*1024**2
OUTPUT_LIMIT=768*1024
ENTRY_LIMIT=1_000_000
RECEIPTS={'ownership.json','ready.json','result.json','cleanup.json'}
SURVEY_ROOTS=('/home/modly','/opt','/usr','/var/cache','/var/log','/tmp')

def wall_limit(cleanup):return 15 if cleanup else 115

def audit(case):
    def hook(event,args):
        if event=='import' and args[0].split('.')[0] in {'torch','triton','trellis2','flex_gemm','comfy_kitchen','transformers','bpy'}:
            raise ImportError('No neural/native imports in disk survey')
        if event in {'subprocess.Popen','os.system','os.fork','os.posix_spawn','os.exec','socket.connect','socket.bind','os.kill','os.killpg'}:
            raise RuntimeError('No children/network/numeric signals in survey')
        if event=='open':
            path,mode,flags=args
            if any(x in str(mode or '') for x in 'wax+') or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
                if (not isinstance(path,(str,bytes,os.PathLike)) or c.plain(os.fsdecode(path)).parent!=case
                    or Path(os.fsdecode(path)).name not in RECEIPTS or not flags&os.O_EXCL):
                    raise RuntimeError('Only exclusive Windows survey receipts are writable')
        if event in {'os.mkdir','os.remove','os.rmdir','os.rename','os.link','os.symlink','os.truncate','os.utime','os.chmod','os.chown'}:
            raise RuntimeError('Filesystem mutation forbidden in disk survey')
    return hook

def fs_numbers(v):
    return dict(capacity_bytes=v.f_blocks*v.f_frsize,used_bytes=(v.f_blocks-v.f_bfree)*v.f_frsize,
        free_bytes=v.f_bfree*v.f_frsize,available_bytes=v.f_bavail*v.f_frsize,
        reserved_or_unavailable_bytes=(v.f_bfree-v.f_bavail)*v.f_frsize,
        fragment_bytes=v.f_frsize,blocks=v.f_blocks,free_blocks=v.f_bfree,available_blocks=v.f_bavail,
        inodes=v.f_files,inodes_free=v.f_ffree,inodes_available=v.f_favail)

def mounts(text):
    rows=[]
    def unescape(value):return re.sub(r'\\(040|011|012|134)',lambda m:chr(int(m[1],8)),value)
    for line in text.splitlines():
        a,b=line.split(' - ',1);left=a.split();right=b.split()
        if len(left)<6 or len(right)<3:raise ValueError('Incomplete mountinfo')
        rows.append(dict(id=int(left[0]),parent=int(left[1]),device=left[2],root=unescape(left[3]),
            target=unescape(left[4]),options=left[5],optional=left[6:],filesystem=right[0],source=unescape(right[1]),super_options=right[2]))
    if not rows or len(rows)>4096:raise ValueError('Bounded complete mount inventory required')
    return rows

def bounded_text(path,limit=1024**2):
    with Path(path).open('r',encoding='utf-8') as f:text=f.read(limit+1)
    if len(text.encode())>limit:raise ValueError('Metadata text limit: '+str(path))
    return text

class DirectoryFS:
    """Linux dir_fd operations: never traverse a substituted path or symlink."""
    def root(self):return os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    def open_path(self,path):
        fd=self.root()
        try:
            for name in Path(path).parts[1:]:
                child=self.child(fd,name);self.close(fd);fd=child
            return fd
        except BaseException:self.close(fd);raise
    def info(self,fd):return os.fstat(fd)
    def entries(self,fd):return os.scandir(fd)
    def child_info(self,fd,name):return os.stat(name,dir_fd=fd,follow_symlinks=False)
    def child(self,fd,name):return os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
    def close(self,fd):os.close(fd)

def walk_usage(mount_rows,rootdev,check,*,fs=None,entry_limit=ENTRY_LIMIT,roots=('/',)):
    fs=fs or DirectoryFS();excluded={r['target'] for r in mount_rows if r['target']!='/'}
    seen=set();rows=[];errors=[];skipped=[];entries=0;omitted=0;row_count=0;last_path='/';fatal=False;retained=[];retained_omitted=0
    def problem(path,exc):
        nonlocal omitted
        if len(errors)<32:errors.append(dict(path=path[:2048],error=(type(exc).__name__+': '+str(exc))[:512]))
        else:omitted+=1
    def visit(fd,path,depth,expected=None):
        nonlocal entries,row_count,last_path,retained_omitted
        last_path=path
        check();s=fs.info(fd)
        if s.st_dev!=rootdev or expected is not None and (s.st_dev,s.st_ino,s.st_mode)!=(expected.st_dev,expected.st_ino,expected.st_mode):
            raise RuntimeError('Directory identity/device changed: '+path)
        if depth>64:raise RuntimeError('Depth limit exceeded')
        allocated=s.st_blocks*512;apparent=s.st_size;complete=True
        try:
            with fs.entries(fd) as iterator:
                for entry in iterator:
                    check();entries+=1
                    if entries>entry_limit:raise RuntimeError('Metadata entry limit exceeded; stop without retry')
                    name=entry.name
                    if name in ('.','..') or '/' in name:raise RuntimeError('Invalid directory entry')
                    target=path.rstrip('/')+'/'+name
                    last_path=target
                    try:
                        child=fs.child_info(fd,name)
                        if target in excluded or child.st_dev!=rootdev:
                            if len(skipped)<128:skipped.append(target)
                            continue
                        if stat.S_ISDIR(child.st_mode):
                            handle=fs.child(fd,name)
                            try:a,b,ok=visit(handle,target,depth+1,child)
                            finally:fs.close(handle)
                            allocated+=a;apparent+=b;complete=complete and ok
                        else:
                            # Count a hardlinked inode once; symlinks are metadata only.
                            key=(child.st_dev,child.st_ino)
                            if key not in seen:
                                seen.add(key);allocated+=child.st_blocks*512;apparent+=child.st_size
                    except OSError as exc:problem(target,exc);complete=False
        except OSError as exc:problem(path,exc);complete=False
        if depth<=3 or path.startswith('/home/modly/') and (depth<=4 or Path(path).name in ('scratch','geometry')):
            row_count+=1;row=dict(path=path,allocated_bytes=allocated,apparent_inode_bytes=apparent,complete=complete,
                uid=getattr(s,'st_uid',None),gid=getattr(s,'st_gid',None),mode=s.st_mode,device=s.st_dev,inode=s.st_ino)
            item=(allocated,row_count,row)
            if len(rows)<64:heapq.heappush(rows,item)
            elif item[0]>rows[0][0]:heapq.heapreplace(rows,item)
            p=Path(path)
            if p.name in ('geometry','scratch') and p.parent.name.startswith('run-') and p.parent.parent==Path('/home/modly'):
                if len(retained)<64:retained.append(row)
                else:retained_omitted+=1
        return allocated,apparent,complete
    complete=True
    for path in roots:
        root=None
        try:
            if any(path==m or path.startswith(m.rstrip('/')+'/') for m in excluded):skipped.append(path);continue
            root=fs.root() if path=='/' else fs.open_path(path)
            _,_,ok=visit(root,path,0);complete=complete and ok
        except OSError as exc:problem(path,exc);complete=False
        except (RuntimeError,TimeoutError,MemoryError) as exc:
            problem(last_path,exc);complete=False;fatal=True;break
        finally:
            if root is not None:fs.close(root)
    return dict(complete=complete and not errors and not omitted,fatal_stop=fatal,last_path=last_path,entries=entries,unique_nondirectory_inodes=len(seen),
        errors=errors,errors_omitted=omitted,excluded_mounts=sorted(excluded),skipped_mount_or_device_examples=skipped,
        roots=list(roots),directory_rows_omitted=max(0,row_count-len(rows)),
        retained_run_leaves=retained,retained_run_leaves_omitted=retained_omitted,
        directories=[r[2] for r in sorted(rows,reverse=True)],
        accounting='Metadata st_blocks*512; hardlinks counted once at first encounter; symlinks never followed. Nested rows overlap. Not equivalent to df used blocks or an atomic snapshot.')

def collect_usage(mount_rows,rootdev,check):return walk_usage(mount_rows,rootdev,check,roots=SURVEY_ROOTS)

def collect(check,result=None):
    result={} if result is None else result
    rows=mounts(bounded_text('/proc/self/mountinfo'));root=os.stat('/');filesystems={};fs_errors={}
    result.update(mounts=rows,filesystems=filesystems,filesystem_errors=fs_errors)
    for path in ('/','/home/modly','/mnt/d'):
        try:
            check();s=os.stat(path)
            filesystems[path]=dict(device=s.st_dev,major=os.major(s.st_dev),minor=os.minor(s.st_dev),**fs_numbers(os.statvfs(path)))
        except OSError as exc:fs_errors[path]=type(exc).__name__+': '+str(exc)
    if '/' not in filesystems:raise RuntimeError('Root filesystem measurement unavailable')
    device=filesystems['/'];sysfs=Path('/sys/dev/block')/(str(device['major'])+':'+str(device['minor']))
    block={}
    try:
        resolved=sysfs.resolve(strict=True);resolved.relative_to('/sys')
        block=dict(sysfs_path=str(resolved),sectors_512=int(bounded_text(sysfs/'size',1024).strip()))
        block['block_device_bytes']=block['sectors_512']*512
    except (OSError,ValueError) as exc:block=dict(error=type(exc).__name__+': '+str(exc))
    result.update(mounts=rows,filesystems=filesystems,root_block_device=block,
        root_mounts=[r for r in rows if r['target']=='/'],kernel=os.uname().release)
    result['usage']=collect_usage(rows,root.st_dev,check)
    return result

def save(case,name,value):
    if name not in RECEIPTS:raise ValueError('Unknown survey receipt')
    blob=json.dumps(value,allow_nan=False,indent=2).encode()
    if len(blob)>(OUTPUT_LIMIT if name=='result.json' else 8192):raise ValueError('Bounded survey receipt exceeded')
    with c.plain(case/name).open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())

def validate_case(case,token):
    if not re.fullmatch('[a-f0-9]{32}',token or '') or c.plain(case)!=c.plain(CASE):raise ValueError('Exact new survey case/token required')
    if os.environ.get('MODLY_DISK_SURVEY_TOKEN')!=token or os.environ.get('MODLY_PIXAL_PROOF_TOKEN')!=token:
        raise RuntimeError('Exec-time ownership nonce missing')
    return c.plain(case)

def guest(case,token,cleanup=False):
    if os.name!='posix':raise RuntimeError('Named Linux guest required')
    import resource
    case=validate_case(case,token);ProcFS().require_pidfds()
    resource.setrlimit(resource.RLIMIT_AS,(CAP,CAP));resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    resource.setrlimit(resource.RLIMIT_FSIZE,(OUTPUT_LIMIT,OUTPUT_LIMIT))
    def expired(*_):raise TimeoutError('Bounded survey wall deadline')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,wall_limit(cleanup))
    sys.addaudithook(audit(case))
    if cleanup:
        owner=json.loads(bounded_text(case/'ownership.json',8192))
        if (owner.get('token')!=token or owner.get('pid')!=owner.get('pgid') or owner.get('pid')==os.getpid() or owner.get('pgid')==os.getpgrp()
            or type(owner.get('start_ticks')) is not int):raise RuntimeError('Invalid survey ownership')
        members=process_rows(owner['pgid']);owned_members(owner,members);forced=bool(members);sent=[]
        for sig in (signal.SIGTERM,signal.SIGKILL):
            if not members:break
            sent.extend(signal_verified(owner,sig));until=time.monotonic()+5
            while time.monotonic()<until:
                members=process_rows(owner['pgid']);owned_members(owner,members)
                if not members:break
                time.sleep(.05)
        if members:raise RuntimeError('Owned guest exit unverified; no distro termination')
        save(case,'cleanup.json',dict(group_empty=True,natural_exit=not forced,forced_owned_cleanup=forced,
            pidfd_signals=sent,no_distro_termination=True));return 0
    if os.getpgrp()!=os.getpid():os.setsid()
    if os.getpgrp()!=os.getpid():raise RuntimeError('Owned process group required')
    own=next(r for r in process_rows(os.getpgrp()) if r['pid']==os.getpid())
    owner={k:own[k] for k in ('pid','pgid','start_ticks')};owner['token']=token
    owned_members(owner,[own]);save(case,'ownership.json',owner)
    save(case,'ready.json',dict(owner=owner,pidfd_prerequisite='PASS',process_memory_limit_bytes=CAP))
    began=time.monotonic();result=dict(classification='FAIL',inventory_started=False,neural_execution=False,guest_file_writes=False)
    def check():
        if time.monotonic()-began>=110:raise TimeoutError('Survey metadata deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>CAP:raise RuntimeError('Guest process memory ceiling')
    try:
        if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Exact GO plus EOF required; no survey')
        result['inventory_started']=True;result['survey']={};collect(check,result['survey'])
        result['classification']='FAIL' if result['survey']['usage']['fatal_stop'] else ('SURVEY_COMPLETE'
            if result['survey']['usage']['complete'] and not result['survey']['filesystem_errors'] and 'error' not in result['survey']['root_block_device'] else 'SURVEY_INCOMPLETE')
    except BaseException as exc:result['error']=type(exc).__name__+': '+str(exc)
    finally:
        result.update(seconds=time.monotonic()-began,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            process_identity=owner,finished_utc=c.now())
        save(case,'result.json',result)
    return 0 if result['classification'] in ('SURVEY_COMPLETE','SURVEY_INCOMPLETE') else 2

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument('--guest',action='store_true');g.add_argument('--cleanup',action='store_true')
    p.add_argument('--case');p.add_argument('--token');a=p.parse_args(argv)
    if not a.guest and not a.cleanup:print('DRY_RUN: no guest activity');return 0
    return guest(Path(a.case),a.token,a.cleanup)
if __name__=='__main__':raise SystemExit(main())
