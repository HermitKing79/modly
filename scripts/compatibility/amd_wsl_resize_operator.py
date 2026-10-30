"""Attended one-shot native resize. Default inert; separate execution approval."""
import argparse,contextlib,ctypes,hashlib,json,os,queue,re,shutil,signal,subprocess,sys,threading,time,uuid
from pathlib import Path
from datetime import datetime,timezone
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import amd_wsl_cold_backup as b
import amd_wsl_resize_readiness as r

ROOT=b.ROOT
CASE=ROOT.parent/'artifacts/amd-wsl-resize-64g-v1'
ACCEPTED=ROOT.parent/'artifacts/amd-wsl-cold-backup-v2'
BACKUP=Path(r.BACKUP_PATH)
AUTH=ROOT.parent/'artifacts/amd-wsl-resize-operator-preparation-20261030-v1/authority.json'
AUTH_SHA='56fcc0ae93bdaa2dac6e5d71ae28250cc8ebf8681adc2dac5fd1a414dc6ee6c5'
GUEST=Path(__file__).with_name('amd_wsl_resize_capacity.py')
GIB=1024**3
WINDOW=1800
FILES=b.PUBLISHED+['scripts/compatibility/'+x for x in ('amd_wsl_resize_operator.py','amd_wsl_resize_capacity.py','amd_wsl_resize_readiness.py','Invoke-ModlyWslResize.ps1')]

def utc():return datetime.now(timezone.utc).isoformat()

def journal(event,**data):
    with (CASE/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(utc=utc(),event=event,**data),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

class Faults:
    def __init__(self,record):self.rows=[];self.record=record
    def add(self,error):
        value=str(error)[:512]
        if value in self.rows:return
        if len(self.rows)>=64:return
        self.rows.append(value)
        try:self.record('FAULT',error=value)
        except BaseException:pass
        try:print('LATCHED FAIL: '+value,flush=True)
        except BaseException:pass

def settle(child,observe,faults,sleep=time.sleep):
    """No cancellation path: even observation, logging and interrupts may fail."""
    while True:
        try:code=child.poll()
        except BaseException as exc:faults.add(type(exc).__name__+': '+str(exc));code=None
        try:observe()
        except BaseException as exc:faults.add(type(exc).__name__+': '+str(exc))
        if code is not None:return code
        try:sleep(.25)
        except BaseException as exc:faults.add(type(exc).__name__+': '+str(exc))

def native_command():return list(r.NATIVE_COMMAND)

def require_hash(value):r.require(value==r.BACKUP_SHA,'Fresh complete disk SHA-256 differs')

def guest_digest():return hashlib.sha256(GUEST.read_bytes().replace(b'\r\n',b'\n')).hexdigest()

def filesystem_bytes(text):
    found=set()
    for count,size,unit in re.findall(r'filesystem\s+on\s+\S+\s+is\s+now\s+(\d+)\s+\((\d+)([kKmM]?)\)\s+blocks\s+long',text,re.I):
        found.add(int(count)*int(size)*{'':1,'k':1024,'m':1024**2}[unit.lower()])
    r.require(found=={r.TARGET_BYTES},'Exact native filesystem block geometry unavailable/different')
    return r.TARGET_BYTES

def no_inherited_job():
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p
    k.IsProcessInJob.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]
    flag=ctypes.c_int()
    if not k.IsProcessInJob(k.GetCurrentProcess(),None,ctypes.byref(flag)):raise ctypes.WinError()
    r.require(not flag.value,'Controller is in an external Job; use a normal standalone console, no fallback')

def docker_quiet():
    class Entry(ctypes.Structure):
        _fields_=[('size',ctypes.c_uint32),('usage',ctypes.c_uint32),('pid',ctypes.c_uint32),('heap',ctypes.c_size_t),('module',ctypes.c_uint32),('threads',ctypes.c_uint32),('parent',ctypes.c_uint32),('priority',ctypes.c_long),('flags',ctypes.c_uint32),('name',ctypes.c_wchar*260)]
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateToolhelp32Snapshot.argtypes=[ctypes.c_uint32,ctypes.c_uint32];k.CreateToolhelp32Snapshot.restype=ctypes.c_void_p
    for name in ('Process32FirstW','Process32NextW'):getattr(k,name).argtypes=[ctypes.c_void_p,ctypes.POINTER(Entry)]
    k.CloseHandle.argtypes=[ctypes.c_void_p]
    h=k.CreateToolhelp32Snapshot(2,0)
    if h==ctypes.c_void_p(-1).value:raise ctypes.WinError()
    try:
        row=Entry();row.size=ctypes.sizeof(row);ok=k.Process32FirstW(h,ctypes.byref(row));names=[]
        if not ok:raise ctypes.WinError()
        while ok:
            if row.name.lower() in {'docker desktop.exe','com.docker.backend.exe','com.docker.proxy.exe'}:names.append(row.name)
            ok=k.Process32NextW(h,ctypes.byref(row))
        if ctypes.get_last_error()!=18:raise ctypes.WinError()
        r.require(not names,'Conflicting Docker activity: '+str(names))
    finally:k.CloseHandle(h)

class Observer:
    def __init__(self):
        self.began=time.monotonic();self.last=0;self.minimum=None;self.faults=Faults(journal)
        self.native_started=False;self.native_settled=False
    def check(self,floor=22*GIB):
        row=b.windows_memory()
        available=row.get('available_bytes')
        if type(available) is int:self.minimum=min(self.minimum if self.minimum is not None else available,available)
        disks={};limits={'D:/':50*GIB,'E:/':2*GIB,str(CASE):2*GIB}
        try:
            for path in limits:disks[path]=shutil.disk_usage(path).free
        except BaseException:
            journal('SAMPLE',seconds=time.monotonic()-self.began,disk_free=disks,**row);raise
        bad=(type(available) is not int or available<floor or row.get('private_commit_bytes',b.CAP+1)>b.CAP
             or any(disks[path]<limit for path,limit in limits.items()))
        if bad or time.monotonic()-self.last>=1:
            journal('SAMPLE',seconds=time.monotonic()-self.began,required_RAM_bytes=floor,disk_free=disks,**row);self.last=time.monotonic()
        r.require(type(available) is int and available>=12*GIB,'Emergency RAM floor')
        r.require(available>=floor,'RAM below '+str(floor//GIB)+' GiB')
        r.require(row['private_commit_bytes']<=b.CAP,'Observer process commit ceiling')
        for path,limit in limits.items():r.require(disks[path]>=limit,'Disk floor: '+path)
        return row
    def preflight(self):
        r.require(time.monotonic()-self.began<WINDOW,'Preflight deadline')
        return self.check(12*GIB)

def publication(commit):
    b.published(commit)
    for file in FILES:
        original=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+file],timeout=15)
        r.require(original.replace(b'\r\n',b'\n')==(ROOT/file).read_bytes().replace(b'\r\n',b'\n'),'Published source differs: '+file)

def inventory(ob,prefix,authority):
    previous=b.CASE;b.CASE=CASE
    try:
        c=b.Controller();c.health=ob.preflight
        data=json.loads(c.child(b.helper_command('inventory'),prefix,30))
    finally:b.CASE=previous
    b.authenticate_inventory(data,authority);docker_quiet();b.save(CASE/(prefix+'.json'),data)
    return data

def hash_disk(f,ob,label):
    before=b.file_identity(f);h=hashlib.sha256();total=0;last=0;f.seek(0)
    journal('HASH_START',disk=label,identity=before)
    while data:=f.read(b.BUFFER):
        ob.preflight();h.update(data);total+=len(data)
        if time.monotonic()-last>=5:journal('HASH_PROGRESS',disk=label,bytes=total);last=time.monotonic()
    r.require(total==r.BACKUP_BYTES and b.file_identity(f)==before,'Disk identity/length changed during hash')
    require_hash(h.hexdigest());journal('HASH_PASS',disk=label,bytes=total,sha256=h.hexdigest());f.seek(0)

def invoke_native(ob):
    """Exactly one Microsoft operation; never attach a cancellation Job."""
    ob.check();journal('FINAL_NATIVE_ADMISSION');b.save(CASE/'native-attempt.json',dict(utc=utc(),command=native_command(),policy='natural-settlement',target_bytes=r.TARGET_BYTES))
    child=None;began=time.monotonic();handlers={}
    def interrupted(signum,frame):ob.faults.add('Operator interrupt '+str(signum)+'; waiting for natural settlement')
    with (CASE/'native-stdout.txt').open('xb',buffering=0) as out,(CASE/'native-stderr.txt').open('xb',buffering=0) as err:
        for signum in (signal.SIGINT,signal.SIGBREAK):handlers[signum]=signal.signal(signum,interrupted)
        def observe():
            if time.monotonic()-began>=WINDOW:ob.faults.add('Native deadline exceeded; settlement continues')
            if out.tell()+err.tell()>16*1024**2:ob.faults.add('Native output budget exceeded; streams stay open')
            ob.check()
        try:
            # New group, no console and no Job: no inherited diagnostic kill policy.
            child=subprocess.Popen(native_command(),executable=native_command()[0],stdin=subprocess.DEVNULL,
                                   stdout=out,stderr=err,shell=False,cwd=CASE,creationflags=0x08000200)
            ob.native_started=True
            try:b.save(CASE/'native-client.json',dict(pid=child.pid,utc=utc(),new_process_group=True,no_cancellation_job=True))
            except BaseException as exc:ob.faults.add('Native identity receipt: '+str(exc))
            code=settle(child,observe,ob.faults)
        finally:
            if child is not None:
                settle(child,observe,ob.faults);ob.native_settled=True
            for signum,handler in handlers.items():signal.signal(signum,handler)
        out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
    journal('NATIVE_SETTLED',returncode=code,pid=child.pid,seconds=time.monotonic()-began)
    r.require(code==0,'Native command failed')
    r.require(not ob.faults.rows,'Latched native fault: no guest/mutating follow-up')
    texts=[]
    for name in ('native-stdout.txt','native-stderr.txt'):
        with (CASE/name).open('rb') as f:raw=f.read(16*1024**2+1)
        r.require(len(raw)<=16*1024**2,'Output incomplete for geometry validation')
        texts.append(raw.replace(b'\0',b'').decode('utf-8',errors='replace'))
    return filesystem_bytes('\n'.join(texts))

def capacity_command(token):
    path='/mnt/c/'+str(GUEST.resolve())[3:].replace('\\','/')
    return [native_command()[0],'-d','Modly-AMD-Ubuntu-24.04','-u','modly','--exec','/usr/bin/python3','-B','-I','-S',path,token]

def capacity_query(ob):
    token=uuid.uuid4().hex;ob.check();journal('CAPACITY_START');began=time.monotonic();messages=queue.Queue();child=None
    with (CASE/'capacity-stderr.txt').open('xb') as err:
        thread=None;reader_started=False;ready=None;result=None;sent=False
        def close_input():
            if child is not None:
                try:child.stdin.close()
                except BaseException as exc:ob.faults.add('Capacity EOF: '+type(exc).__name__+': '+str(exc))
        def read():
            try:
                while line:=child.stdout.readline(65537):
                    if len(line)>65536:raise ValueError('Capacity line too large')
                    messages.put(json.loads(line))
            except BaseException as exc:messages.put(dict(error=type(exc).__name__+': '+str(exc)))
        try:
            child=subprocess.Popen(capacity_command(token),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,
                                   shell=False,cwd=CASE,creationflags=0x08000200)
            thread=threading.Thread(target=read,daemon=True);thread.start();reader_started=True
            while child.poll() is None or not messages.empty() or thread.is_alive():
                try:
                    ob.check()
                    if time.monotonic()-began>30:ob.faults.add('Capacity deadline exceeded')
                except BaseException as exc:ob.faults.add(type(exc).__name__+': '+str(exc))
                try:row=messages.get(timeout=.2)
                except queue.Empty:row=None
                if row is not None:
                    if row.get('phase')=='READY':
                        r.require(ready is None and row.get('token')==token,'Capacity ready identity differs');ready=row
                        if not ob.faults.rows:child.stdin.write(b'GO\n');child.stdin.flush();sent=True
                        close_input()
                    else:result=row
                if ob.faults.rows and not child.stdin.closed:close_input()
        finally:
            close_input()
            if child is not None:
                settle(child,lambda:ob.check(),ob.faults)
                if reader_started:thread.join(2)
                child.stdout.close()
        r.require(not thread.is_alive() and child.returncode==0 and sent and not ob.faults.rows,'Capacity natural exit/guards failed')
        r.require(result is not None and ready is not None and result.get('token')==token and result.get('pid')==ready['pid']
                  and result.get('start_ticks')==ready['start_ticks'],'Capacity result ownership differs')
        r.require(type(result.get('peak_RSS_bytes')) is int and result['peak_RSS_bytes']<=b.CAP,'Capacity process memory differs')
        b.save(CASE/'linux-capacity.json',result);return result

def run(commit):
    publication(commit);b.authenticate_worker_python();no_inherited_job()
    r.require(b.digest(AUTH)==AUTH_SHA,'Operator authority changed')
    authority=json.loads(AUTH.read_text());a=b.authority()
    for p in (CASE,):
        b.plain(p,False);r.require(not p.exists(),'Consumed resize evidence path; no retry')
    pins=b.Pins();created=False;container_captured=False;report=dict(classification='FAIL',commit=commit,started=utc());ob=Observer()
    try:
        pins.add(CASE.parent);CASE.mkdir();created=True;pins.add(CASE)
        ob.preflight();journal('BEGIN',critical_policy_explicitly_accepted=True)
        for name,wanted in authority['receipts'].items():r.require(b.digest(ACCEPTED/name)==wanted,'Accepted receipt changed: '+name)
        for path,expected in authority['native_binaries'].items():r.require(b.digest(Path(path))==expected,'Native WSL binary changed: '+path)
        inventory(ob,'initial-inventory',a);b.save(CASE/'inputs-before.json',b.inputs(a))
        r.require(shutil.disk_usage('D:/').free-r.TARGET_BYTES>=50*GIB,'Insufficient D: capacity-growth reserve')
        pins.add(BACKUP.parent);pins.add(b.SOURCE.parent)
        with b.native_file(BACKUP,'verify') as backup:
            backup_id=b.file_identity(backup);backup_meta=b.metadata(backup)
            with b.native_file(b.SOURCE,'source') as source:
                current=dict(source=dict(identity=b.file_identity(source),metadata=b.metadata(source)),backup=dict(identity=backup_id,metadata=backup_meta))
                load=lambda name:json.loads((ACCEPTED/name).read_text())
                r.verify_backup(load('final-result.json'),load('worker-result.json'),load('destination-hash-stdout.txt'),
                    [load(x) for x in ('inventory-process.json','backup-worker-process.json','final-inventory-process.json')],current)
                b.save(CASE/'cold-identities.json',current);hash_disk(source,ob,'source');hash_disk(backup,ob,'backup')
                inventory(ob,'final-cold-inventory',a)
                for i in range(3):
                    row=ob.check();journal('ADMISSION',index=i,**row)
                    if i<2:time.sleep(4)
                r.require(b.file_identity(source)==current['source']['identity'],'Source changed during admission')
                journal('PREFERRED_HEADROOM',preferred_24GiB=row['available_bytes']>=24*GIB,mandatory_22GiB=True)
                for path,expected in authority['native_binaries'].items():r.require(b.digest(Path(path))==expected,'Native WSL binary changed before launch')
            ob.check()
            fs_bytes=invoke_native(ob)
            with b.native_file(b.SOURCE,'verify') as source:
                after=b.file_identity(source);meta=b.metadata(source)
                r.require(after['file_id']==current['source']['identity']['file_id'] and after['volume_serial']==current['source']['identity']['volume_serial'],'Backing file identity changed')
                r.require(meta['virtual_capacity_bytes']==r.TARGET_BYTES,'Container capacity differs')
                b.save(CASE/'container-after.json',dict(identity=after,metadata=meta))
                container_captured=True
            r.require(guest_digest()==authority['guest_sha256'],'Capacity source changed')
            result=capacity_query(ob)
            # Named distro is now running intentionally; verify immutable registration/hardware only.
            saved=b.CASE;b.CASE=CASE
            try:
                c=b.Controller();c.health=lambda:ob.check()
                final=json.loads(c.child(b.helper_command('inventory'),'after-inventory',30))
            finally:b.CASE=saved
            b.save(CASE/'after-inventory.json',final)
            r.require(set(final['running'])<={'Modly-AMD-Ubuntu-24.04'},'Unrelated distro started during operation')
            docker_quiet()
            identity_view=dict(final,running=[],vm_processes=[])
            b.authenticate_inventory(identity_view,a)
            r.require(b.file_identity(backup)==backup_id and b.metadata(backup)==backup_meta,'Accepted backup changed')
            b.save(CASE/'inputs-after.json',b.inputs(a));ob.check()
            accepted=dict(native_exit=0,natural_completion=True,virtual_bytes=meta['virtual_capacity_bytes'],filesystem_bytes=fs_bytes,
                root_device_bytes=result['root_device_bytes'],ext4_rw=result['ext4_rw'],available_bytes=result['available_bytes'],
                D_free_bytes=shutil.disk_usage('D:/').free,backup_unchanged=True,registration_unchanged=True,inputs_unchanged=True,guard_faults=ob.faults.rows)
            r.accept_completion(accepted);report.update(classification='PASS',verification=accepted)
    except BaseException as exc:
        ob.faults.add(type(exc).__name__+': '+str(exc));report.update(classification='FAIL',error=type(exc).__name__+': '+str(exc))
    finally:
        if created and ob.native_settled and not container_captured:
            try:
                with b.native_file(b.SOURCE,'verify') as source:
                    b.save(CASE/'settled-container-state.json',dict(identity=b.file_identity(source),metadata=b.metadata(source)))
            except BaseException as exc:
                ob.faults.add('Settled metadata unavailable: '+str(exc))
        try:pins.close();report['handles_closed']=True
        except BaseException as exc:ob.faults.add('Handle cleanup: '+str(exc));report['classification']='FAIL'
        if ob.faults.rows:report['classification']='FAIL'
        report.update(faults=ob.faults.rows,minimum_observed_available_bytes=ob.minimum,seconds=time.monotonic()-ob.began,
                      native_started=ob.native_started,native_settled=ob.native_settled,
                      no_retry=True,no_automatic_rollback=True,no_decoder=True)
        if created:
            try:
                with (CASE/'REPORT.txt').open('x',encoding='utf-8') as f:
                    f.write('Final authority: final-result.json (absent means incomplete).\n'+json.dumps(report,indent=2)+'\n');f.flush();os.fsync(f.fileno())
            except BaseException as exc:
                ob.faults.add('Report write: '+str(exc));report['classification']='FAIL'
            try:
                # A failed flush/close must never publish a readable PASS authority.
                b.save(CASE/'final-result.pending.json',report)
                os.rename(CASE/'final-result.pending.json',CASE/'final-result.json')
            except BaseException as exc:
                ob.faults.add('Final receipt publication: '+str(exc));report['classification']='FAIL'
        print('RESIZE='+report['classification']+'\nEVIDENCE='+str(CASE),flush=True)
    return 0 if report['classification']=='PASS' else 1

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--approve-resize',action='store_true');p.add_argument('--accept-native-critical-section',action='store_true');p.add_argument('--commit');args=p.parse_args(argv)
    if not args.run:print('DRY_RUN: no admission, inventory, disk access, WSL or resize. Execution approval required.');return 0
    if not args.approve_resize or not args.accept_native_critical_section or not re.fullmatch('[a-f0-9]{40}',args.commit or ''):p.error('Separate resize approval, accepted native critical section and exact commit required')
    return run(args.commit)

if __name__=='__main__':raise SystemExit(main())
