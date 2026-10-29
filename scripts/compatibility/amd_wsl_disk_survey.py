"""Prepared read-only survey. Default does no work; separate approval is required."""
import argparse,ctypes,hashlib,json,os,re,shutil,struct,subprocess,sys,threading,time,uuid
from pathlib import Path
from datetime import datetime,timezone
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import amd_zombie_d_handoff as guard
from amd_pixal3d_standalone_v2 import allocated_size,desktop_absent,linux_path
from amd_forward_owned_job import OwnedJob,ExtendedLimits
import amd_wsl_disk_survey_guest as guest

ROOT=Path(__file__).resolve().parents[2]
CASE=ROOT.parent/'artifacts/amd-closed-solid-disk-survey-standalone-v2'
VHD=Path('D:/AI/Modly-WSL/Ubuntu24.04/ext4.vhdx')
DISTRO='Modly-AMD-Ubuntu-24.04'
GIB=1024**3;START=22*GIB;LIVE=12*GIB;DISK=50*GIB;CAP=512*1024**2
FILES=['scripts/compatibility/'+name for name in ('amd_wsl_disk_survey.py','amd_wsl_disk_survey_guest.py',
    'Invoke-ModlyDiskSurvey.ps1','test_amd_wsl_disk_survey.py','amd_zombie_d_handoff.py','amd_zombie_ram_diagnostic.py',
    'amd_forward_owned_job.py','amd_pixal3d_standalone_v2.py','amd_pixal3d_proof_common_v2.py',
    'amd_pixal3d_proof_common.py','amd_pixal3d_memory_v2.py','amd_pixal3d_owned_proc_v3.py')]
UPSTREAM='853a9a28da970beeb53685f967de059bb90c7364'
INPUT=ROOT/'docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/closed-solid-box-v1'
INPUTS={
    'occupancy/source-image.png':'fddf6f7de10a548271c3d80b2b813e3be1191f1636ae5f51e08d548c07117c8a',
    'occupancy/input-contract.json':'c5eee9373aeb8be783184bf3cf47b29fc48b50780bf4602c8640d1eeabb16edc',
    'shape/run/results.json':'498948bb64a88e1f869a2c01ed0f38c51346ae347bc1775aa263f1c8a69f7ea5',
    'shape/run/scratch/shape-latent-raw.npy':'d0a5c8b5f490484227ed6a822400b56dfeb9432b7637542903c7ffaba97c8505',
    'shape/run/scratch/shape-latent-denormalized.npy':'caabc77639b7ccb355836fe2dee6f36bdbcdbd0bb7d32b207854ac74f1455f3e',
    'shape/run/scratch/shape-latent-coordinates.npy':'dc958a216097954dbc1844669a24b645503c2e87f69fd2c08fb1aa372d3dbdec',
    'shape/run/scratch/shape-initial-noise.npy':'d77e2190c9d891f3c7629d69dc5e3e4854b27ddc32963cae1aed935e50d74266',
    'shape/run/scratch/shape-latent-provenance.json':'92978cd372b9ad6b5814c767fbbfe4a7d20b5b1d3d20c85addf99c5f7f283790'}
admit=guard.fresh_admission

def now():return datetime.now(timezone.utc).isoformat()
def require_memory(row,floor):
    if type(row.get('available_bytes')) is not int or row['available_bytes']<floor:raise RuntimeError('Windows available RAM below '+str(floor))
def require_fresh():
    if guard.plain(CASE).exists():raise FileExistsError('One-shot survey destination consumed; no retry')
def record(name,value):guard.save(CASE/name,value)
def journal(name,value):
    path=guard.plain(CASE/name)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(utc=now(),**value),allow_nan=False)+'\n');f.flush()

def verify_inputs():
    result={}
    for name,expected in INPUTS.items():
        p=guard.plain(INPUT/name);s=p.stat()
        if not 0<s.st_size<=8*1024**2 or guard.digest(p)!=expected:raise ValueError('Preserved box input changed: '+name)
        after=p.stat()
        if (s.st_size,s.st_mtime_ns,s.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise ValueError('Input identity changed during verification')
        result[name]=dict(bytes=s.st_size,sha256=expected,file_id=s.st_ino,mtime_ns=s.st_mtime_ns)
    receipt=json.loads((INPUT/'shape/run/results.json').read_bytes())
    if (receipt['classification']!='PASS' or receipt['worker']['token_count']!=3888
        or receipt['worker']['forward_calls']!=21 or receipt['process']['harness_commit']!=UPSTREAM):raise ValueError('Original authenticated 3888-row shape job required')
    return result

def verify_publication(commit):
    if not re.fullmatch('[a-f0-9]{40}',commit or ''):raise ValueError('Exact published commit required')
    def git(args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True,encoding='utf-8',timeout=15)
    if git(['rev-parse','HEAD']).strip()!=commit:raise ValueError('Exact published checkout required')
    for name in FILES:
        if git(['show',commit+':'+name]).replace('\r\n','\n')!=(ROOT/name).read_text(encoding='utf-8').replace('\r\n','\n'):
            raise ValueError('Published survey source differs: '+name)

def command(token,cleanup=False):
    if not re.fullmatch('[a-f0-9]{32}',token or ''):raise ValueError('Canonical survey nonce required')
    return [str(Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/wsl.exe'),'-d',DISTRO,'-u','modly','--exec','/usr/bin/env',
        'MODLY_DISK_SURVEY_TOKEN='+token,'MODLY_PIXAL_PROOF_TOKEN='+token,'PYTHONDONTWRITEBYTECODE=1',
        '/usr/bin/python3','-B','-I','-S',linux_path(Path(__file__).with_name('amd_wsl_disk_survey_guest.py')),
        '--cleanup' if cleanup else '--guest','--case',linux_path(CASE),'--token',token]

def require_stopped(data):
    if len(data)>8192:raise ValueError('Bounded WSL running inventory required')
    text=data.decode('utf-16-le' if b'\x00' in data else 'utf-8').lstrip('\ufeff')
    names=[s.strip() for s in text.splitlines() if s.strip()]
    if any(not re.fullmatch('[A-Za-z0-9_.-]+',s) for s in names):raise ValueError('Unrecognized WSL running inventory')
    if DISTRO in names:raise RuntimeError('Named distro already running; preserve unrelated work and stop')
    return names

def vhdx_metadata(path=VHD):
    """Bounded container headers, never mount or read Linux filesystem contents."""
    p=guard.plain(path);before=p.stat()
    with p.open('rb') as f:
        if f.read(8)!=b'vhdxfile':raise ValueError('VHDX identity missing')
        f.seek(196608);table=f.read(65536)
        if table[:4]!=b'regi':raise ValueError('VHDX region table missing')
        count=struct.unpack_from('<I',table,8)[0]
        if not 0<count<2048:raise ValueError('Bounded VHDX regions required')
        found=[]
        for i in range(count):
            guid,offset,length,_=struct.unpack_from('<16sQII',table,16+i*32)
            if str(uuid.UUID(bytes_le=guid))=='8b7ca206-4790-4b9a-b8fe-575f050f886e':found.append((offset,length))
        if len(found)!=1:raise ValueError('Unique VHDX metadata region required')
        offset,length=found[0]
        if offset+length>before.st_size:raise ValueError('VHDX metadata outside container')
        f.seek(offset);meta=f.read(65536)
        if meta[:8]!=b'metadata':raise ValueError('VHDX metadata signature missing')
        count=struct.unpack_from('<H',meta,10)[0]
        if not 0<count<2048:raise ValueError('Bounded VHDX metadata table required')
        sizes=[]
        for i in range(count):
            guid,start,size,_,_=struct.unpack_from('<16sIIII',meta,32+i*32)
            if str(uuid.UUID(bytes_le=guid))=='2fa54224-cd1b-4876-b211-5dbed83bf4b8':
                if size!=8 or start+8>length:raise ValueError('Invalid VHDX virtual size item')
                f.seek(offset+start);sizes.append(struct.unpack('<Q',f.read(8))[0])
        if len(sizes)!=1 or sizes[0]<=0:raise ValueError('Unique VHDX virtual capacity required')
    after=p.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise ValueError('VHDX changed during header inspection')
    return dict(path=str(p),virtual_capacity_bytes=sizes[0],container_file_length_bytes=after.st_size,
        Windows_allocated_storage_bytes_GetCompressedFileSizeW=allocated_size(p),
        region_table_sha256=hashlib.sha256(table).hexdigest(),metadata_table_sha256=hashlib.sha256(meta).hexdigest(),
        mtime_ns=after.st_mtime_ns,file_id=after.st_ino,
        scope='Container capacity, file length and Windows-reported allocated storage are distinct; not guest free space or physical NAND sectors.')

def read_receipt(name,limit=guest.OUTPUT_LIMIT):
    p=guard.plain(CASE/name)
    with p.open('rb') as f:data=f.read(limit+1)
    if len(data)>limit:raise ValueError('Oversized survey receipt')
    return json.loads(data)

class Capture:
    """Hard aggregate file-write cap; overflow is signalled to the owner."""
    def __init__(self,prefix,limit):
        if not re.fullmatch('[a-z-]+',prefix):raise ValueError('Fixed capture prefix required')
        self.prefix=prefix;self.limit=limit;self.bytes=0;self.error=None;self.threads=[];self.lock=threading.Lock()
    def start(self,child):
        def reader(pipe,name):
            try:
                with (CASE/(self.prefix+'-'+name+'.txt')).open('xb') as out:
                    while block:=pipe.read(4096):
                        with self.lock:
                            kept=block[:max(0,self.limit-self.bytes)];out.write(kept);self.bytes+=len(kept)
                            if len(kept)!=len(block):self.error='Captured output limit exceeded'
            except BaseException as exc:self.error=type(exc).__name__+': '+str(exc)
            finally:pipe.close()
        for name,pipe in [('stdout',child.stdout),('stderr',child.stderr)]:
            t=threading.Thread(target=reader,args=(pipe,name),daemon=True);self.threads.append(t);t.start()
    def check(self):
        if self.error:raise RuntimeError(self.error)
    def finish(self):
        for t in self.threads:t.join(1)
        if any(t.is_alive() for t in self.threads):raise RuntimeError('Capture readers did not exit')
        self.check()

def reap_client(child,attached):
    try:child.wait(timeout=5);return False
    except subprocess.TimeoutExpired:
        if attached or child.poll() is not None:raise RuntimeError('Owned client exit unverified')
        # Windows Popen.kill uses its retained process HANDLE, not a PID lookup.
        child.kill();child.wait(timeout=5);return True

def bounded_client(cmd,prefix,timeout,cap=8192,observe=None):
    """Read-only listing or pidfd cleanup only, under its own Windows Job."""
    allowed=(cmd==[command('0'*32)[0],'--list','--running','--quiet']
        or len(cmd)>2 and '--cleanup' in cmd and cmd==command(cmd[-1],True))
    if not allowed:raise ValueError('Only exact listing/owned cleanup commands')
    job=OwnedJob();child=None;attached=False;began=time.monotonic();capture=Capture(prefix,cap);health_error=None
    limits=ExtendedLimits();limits.Basic.Flags=0x2100;limits.ProcessMemory=CAP
    try:
        if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):raise ctypes.WinError()
        child=subprocess.Popen(cmd,executable=cmd[0],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=CASE,creationflags=0x08000000)
        capture.start(child);job.attach(child);attached=True
        while True:
            if observe:
                try:observe()
                except Exception as exc:
                    if prefix!='cleanup':raise
                    health_error=type(exc).__name__+': '+str(exc)
            capture.check()
            if child.poll() is not None:break
            if time.monotonic()-began>=timeout:raise TimeoutError('Bounded client deadline')
            time.sleep(.05)
        capture.finish()
        if child.returncode:raise RuntimeError('Read-only client failed: '+prefix)
        if health_error:raise RuntimeError('Cleanup health failure after bounded owned cleanup: '+health_error)
        return (CASE/(prefix+'-stdout.txt')).read_bytes()
    finally:
        try:
            cleanup=job.finish();record(prefix+'-job.json',cleanup)
            if cleanup.get('owned_tree_exit_verified') is not True or cleanup.get('forced_job_cleanup') is not False:
                raise RuntimeError('Auxiliary cleanup was forced or unverified')
        finally:
            if child:
                forced=reap_client(child,attached)
                record(prefix+'-client-exit.json',dict(unattached_handle_cleanup=forced,returncode=child.returncode))
            capture.finish()

class Runner:
    def __init__(self,token):self.token=token;self.began=None;self.started=False;self.initial_allocated=allocated_size(VHD)
    def live(self,floor=LIVE,deadline=True):
        row=dict(guard.windows_memory(),D_free_bytes=shutil.disk_usage('D:/').free,output_free_bytes=shutil.disk_usage(CASE).free,
            VHD_allocated_storage_bytes=allocated_size(VHD))
        journal('host-memory.jsonl',row)
        require_memory(row,floor)
        if row['D_free_bytes']<DISK or row['output_free_bytes']<2*GIB:raise RuntimeError('Preserved D:/output disk floor failed')
        if row['VHD_allocated_storage_bytes']-self.initial_allocated>20*GIB:raise RuntimeError('Original VHD growth ceiling failed')
        if deadline and self.began is not None and time.monotonic()-self.began>=120:raise TimeoutError('120-second survey host deadline')
        if hasattr(self,'capture'):self.capture.check()
        return row
    def cleanup_guest(self):
        cmd=command(self.token,True)
        bounded_client(cmd,'cleanup',20,observe=lambda:self.live(deadline=False))
        return read_receipt('cleanup.json')
    def invoke(self):
        receipt=dict(classification='FAIL',WSL_started=False,GO_sent=False);job=OwnedJob();child=None
        limits=ExtendedLimits();limits.Basic.Flags=0x2100;limits.ProcessMemory=CAP
        if not job.api.SetInformationJobObject(job.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            job.close();raise ctypes.WinError()
        try:
            cmd=command(self.token)
            record('final-before-WSL.json',self.live(START));self.began=time.monotonic()
            self.capture=Capture('survey',128*1024)
            child=subprocess.Popen(cmd,executable=cmd[0],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=CASE,creationflags=0x08000000)
            self.started=True
            self.capture.start(child)
            receipt.update(WSL_started=True,client_pid=child.pid);job.attach(child);receipt['job_attached']=True
            while True:
                self.live(START)
                if child.poll() is not None:raise RuntimeError('Survey exited before GO')
                try:ready=read_receipt('ready.json',8192);break
                except (FileNotFoundError,json.JSONDecodeError):time.sleep(.2)
            owner=read_receipt('ownership.json',8192)
            if (ready['owner']!=owner or owner['token']!=self.token or type(owner['pid']) is not int or owner['pid']<=0
                or owner['pid']!=owner['pgid'] or type(owner['start_ticks']) is not int
                or ready['pidfd_prerequisite']!='PASS' or ready['process_memory_limit_bytes']!=CAP):raise RuntimeError('Guest identity/prerequisites differ')
            receipt['owner']=owner;record('final-before-GO.json',self.live(START))
            child.stdin.write(b'GO\n');child.stdin.close();receipt['GO_sent']=True
            while child.poll() is None:self.live();time.sleep(.2)
            receipt['exit_code']=child.returncode
            result=read_receipt('result.json')
            if (result.get('process_identity')!=owner or result.get('peak_RSS_bytes',CAP+1)>CAP
                or result.get('seconds',121)>=120 or result.get('neural_execution') is not False
                or result.get('guest_file_writes') is not False or not result.get('inventory_started')):raise RuntimeError('Survey result identity/limits differ')
            receipt['classification']=result['classification']
            if child.returncode or result['classification'] not in ('SURVEY_COMPLETE','SURVEY_INCOMPLETE'):raise RuntimeError('Survey failed; no retry')
        except BaseException as exc:receipt.update(classification='FAIL',error=type(exc).__name__+': '+str(exc))
        finally:
            if child and child.stdin and not child.stdin.closed:child.stdin.close()
            if child and not receipt['GO_sent']:
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:receipt['EOF_wait_timeout']=True
            try:
                if (CASE/'ownership.json').exists():
                    receipt['guest_cleanup']=self.cleanup_guest()
                    if not receipt['guest_cleanup']['natural_exit']:raise RuntimeError('Forced owned guest cleanup required')
                elif child:raise RuntimeError('Guest ownership unavailable; no group/distro signal permitted')
            except BaseException as exc:receipt.update(classification='FAIL',cleanup_error=str(exc))
            peak=ExtendedLimits()
            if job.api.QueryInformationJobObject(job.handle,9,ctypes.byref(peak),ctypes.sizeof(peak),None):receipt['Windows_client_peak_commit_bytes']=peak.PeakProcessMemory
            else:receipt.update(classification='FAIL',measurement_error='Windows Job peak unavailable')
            try:
                receipt['Windows_job_cleanup']=job.finish()
                if receipt['Windows_job_cleanup']['forced_job_cleanup']:receipt['classification']='FAIL'
            except BaseException as exc:receipt.update(classification='FAIL',job_cleanup_error=str(exc))
            if child:
                try:receipt['unattached_handle_cleanup']=reap_client(child,receipt.get('job_attached',False))
                except (OSError,RuntimeError,subprocess.TimeoutExpired) as exc:receipt.update(classification='FAIL',client_reap_error=str(exc))
            if hasattr(self,'capture'):
                try:self.capture.finish()
                except BaseException as exc:receipt.update(classification='FAIL',capture_error=str(exc))
            if receipt.get('EOF_wait_timeout'):receipt['classification']='FAIL'
            receipt['seconds']=0 if self.began is None else time.monotonic()-self.began
            record('host-result.json',receipt)
        return receipt

def run(commit,approved=False):
    if not approved:raise RuntimeError('Separate survey approval required')
    if os.name!='nt' or ctypes.windll.shell32.IsUserAnAdmin():raise RuntimeError('Normal non-admin Windows required')
    if Path(sys.executable).resolve()!=guard.PYTHON.resolve() or guard.digest(guard.PYTHON)!=guard.PYTHON_SHA:raise ValueError('Qualified Windows Python required')
    require_fresh();CASE.mkdir(exist_ok=False)
    summary=dict(classification='FAIL',execution_commit=commit,upstream_commit=UPSTREAM,WSL_started=False,neural_execution=False,automatic_retry=False)
    before=None;runner=None
    try:
        verify_publication(commit);desktop_absent();before=verify_inputs();record('inputs-before.json',before)
        record('vhd-before.json',vhdx_metadata())
        rows=admit(guard.windows_memory,time.sleep,lambda row:journal('admission.jsonl',row))
        guard.validate_admission(rows,datetime.now(timezone.utc));runner=Runner(uuid.uuid4().hex)
        record('before-running-inventory.json',runner.live(START))
        # Listing does not start a distro. Never shut down an existing session.
        wsl=command(runner.token)[0]
        listed=bounded_client([wsl,'--list','--running','--quiet'],'running-inventory',10,observe=lambda:runner.live(START))
        record('running-distros.json',dict(names=require_stopped(listed)))
        receipt=runner.invoke();summary.update(classification=receipt['classification'],WSL_started=receipt['WSL_started'])
    except BaseException as exc:summary.update(classification='FAIL',error=type(exc).__name__+': '+str(exc))
    finally:
        if runner:summary['WSL_started']=runner.started
        try:
            if before is not None:
                after=verify_inputs();record('inputs-after.json',after)
                if after!=before:raise RuntimeError('Original neural/reference files changed')
            record('vhd-after.json',vhdx_metadata())
            if runner:
                runner.began=None;record('post-exit-resource.json',runner.live())
        except BaseException as exc:summary.update(classification='FAIL',preservation_or_closeout_error=str(exc))
        record('summary.json',summary)
        with (CASE/'REPORT.txt').open('x',encoding='utf-8') as f:
            f.write(summary['classification']+'\nNo storage changes, neural work or automatic retry.\nSee result.json for measured data and explicit gaps.\n')
            if 'error' in summary:f.write(summary['error']+'\n')
        print(CASE/'REPORT.txt');print(summary['classification'])
    return 0 if summary['classification']=='SURVEY_COMPLETE' else 2

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--approve-survey',action='store_true');p.add_argument('--commit');a=p.parse_args(argv)
    if not a.run:
        print(json.dumps(dict(mode='DRY_RUN',execution_hold=True,WSL_launched=False,RAM_read=False,case=str(CASE),startup_bytes=START,live_bytes=LIVE,
            D_floor_bytes=DISK,preferred_only_bytes=24*GIB,guest_decode_floor_unchanged_bytes=5*GIB,seconds=120,process_limit_bytes=CAP,neural_execution=False)));return 0
    if not a.approve_survey or not re.fullmatch('[a-f0-9]{40}',a.commit or ''):p.error('Separate --approve-survey and exact published --commit required')
    return run(a.commit,True)
if __name__=='__main__':raise SystemExit(main())
