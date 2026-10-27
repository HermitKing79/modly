"""V3 ownership: pinned proc directories, terminal rechecks and pidfd signals."""
import os,select,signal
from pathlib import Path

class OwnershipError(RuntimeError):pass
MISSING=(FileNotFoundError,ProcessLookupError)
TERMINAL={'Z','X','x'}

def stat_identity(text,pid):
    try:
        head,tail=text.rsplit(')',1);fields=tail.split()
        if int(head.split(' ',1)[0])!=pid or len(fields)<20:raise ValueError()
        return dict(pid=pid,state=fields[0],pgid=int(fields[2]),start_ticks=int(fields[19]))
    except (ValueError,IndexError) as exc:raise OwnershipError('Invalid pinned proc stat') from exc

def same(a,b):return all(a[k]==b[k] for k in ('pid','pgid','start_ticks'))

class ProcFS:
    def entries(self):return [int(p.name) for p in Path('/proc').iterdir() if p.name.isdecimal()]
    def open_proc(self,pid):
        if os.name!='posix':raise OwnershipError('Linux procfs required')
        return os.open('/proc/'+str(pid),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    def read(self,fd,name):
        bound=1024**2 if name=='environ' else 65536
        child=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
        with os.fdopen(child,'rb') as f:data=f.read(bound+1)
        if len(data)>bound:raise OwnershipError('Bounded proc field exceeded: '+name)
        return data
    def stat(self,fd,pid):return stat_identity(self.read(fd,'stat').decode('utf-8'),pid)
    def close(self,fd):os.close(fd)
    def require_pidfds(self):
        if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):raise OwnershipError('Stable pidfd support required; no numeric-signal fallback')
        try:fd=os.pidfd_open(os.getpid(),0)
        except OSError as exc:raise OwnershipError('Kernel pidfd support unavailable') from exc
        try:
            if self.dead(fd):raise OwnershipError('Self pidfd unexpectedly terminal')
            # Signal zero tests the actual kernel API and permission without
            # delivering a signal. A Python symbol alone proves neither.
            try:signal.pidfd_send_signal(fd,0,None,0)
            except OSError as exc:raise OwnershipError('Kernel pidfd zero-signal prerequisite unavailable') from exc
        finally:os.close(fd)
    def pidfd(self,pid):return os.pidfd_open(pid,0)
    def dead(self,fd):
        p=select.poll();p.register(fd,select.POLLIN);rows=p.poll(0)
        if any(mask&(select.POLLERR|select.POLLNVAL) for _,mask in rows):raise OwnershipError('Ambiguous pidfd poll')
        return bool(rows)
    def signal(self,fd,sig):signal.pidfd_send_signal(fd,sig,None,0)

def note(report,event,**values):
    if report:report(dict(event=event,**values))

def prove_exit(fs,pidfd,before,report,error=None):
    # A zombie group leader can still have live native threads. Only a
    # correctly bound process pidfd proves that the whole process has exited.
    if pidfd is None or not fs.dead(pidfd):raise OwnershipError('Terminal/unreadable leader has no proven process exit')
    note(report,'verified_process_exit',identity=before,error=str(error) if error else None)
    return None

def checked_after(fs,fd,before,report,error=None,pidfd=None):
    try:after=fs.stat(fd,before['pid'])
    except MISSING:
        return prove_exit(fs,pidfd,before,report,error)
    if not same(before,after):raise OwnershipError('Pinned process identity changed; refuse ownership')
    if after['state'] in TERMINAL:
        return prove_exit(fs,pidfd,after,report,error)
    return after

def read_member(fs,fd,pid,pgid,report=None,*,pidfd=None,before=None):
    before=fs.stat(fd,pid) if before is None else before
    if before['pgid']!=pgid:return None
    if before['state'] in TERMINAL:return prove_exit(fs,pidfd,before,report)
    try:env=fs.read(fd,'environ');status=fs.read(fd,'status').decode('utf-8')
    except (PermissionError,*MISSING) as exc:
        after=checked_after(fs,fd,before,report,exc,pidfd)
        if after is None:return None
        note(report,'live_process_unverifiable',identity=after,error=type(exc).__name__+': '+str(exc))
        raise OwnershipError('Live process ownership unreadable: '+str(pid)) from exc
    after=checked_after(fs,fd,before,report,pidfd=pidfd)
    if after is None:return None
    prefix=b'MODLY_PIXAL_PROOF_TOKEN=';tokens=[v[len(prefix):] for v in env.split(b'\0') if v.startswith(prefix)]
    token=tokens[0].decode('ascii',errors='replace') if len(tokens)==1 else None
    rss=[line.split() for line in status.splitlines() if line.startswith('VmRSS:')]
    if len(rss)!=1 or len(rss[0])!=3 or not rss[0][1].isdecimal() or rss[0][2]!='kB':raise OwnershipError('Live RSS measurement unavailable')
    return dict(after,token=token,RSS_bytes=int(rss[0][1])*1024)

def owned_members(owner,members):
    for row in members:
        if row['pgid']!=owner['pgid'] or row.get('token')!=owner['token']:raise OwnershipError('Foreign process-group member; refuse signal')
        if row['pid']==owner['pid'] and row['start_ticks']!=owner['start_ticks']:raise OwnershipError('Reused leader PID; refuse signal')
    return True

def scan(pgid,fs,report=None,owner=None):
    rows=[];targets=[]
    try:
        fs.require_pidfds()
        for pid in fs.entries():
            fd=None;pidfd=None;retained=False
            try:
                fd=fs.open_proc(pid);before=fs.stat(fd,pid)
                if before['pgid']!=pgid:continue
                try:pidfd=fs.pidfd(pid)
                except MISSING as exc:raise OwnershipError('Cannot bind process pidfd; no terminal-state guess') from exc
                bound=checked_after(fs,fd,before,report,pidfd=pidfd)
                if bound is None:continue
                row=read_member(fs,fd,pid,pgid,report,pidfd=pidfd,before=bound)
                if row is None:continue
                if fs.dead(pidfd):raise OwnershipError('Live proc stat/pidfd identities disagree')
                if owner is not None:
                    owned_members(owner,[row])
                    targets.append((fd,pidfd,row));retained=True
                rows.append(row)
            except MISSING as exc:
                # Membership and a process pidfd were not bound. Do not guess
                # that an unreadable group leader has no surviving threads.
                note(report,'process_disappeared_before_identity',pid=pid)
                raise OwnershipError('Proc identity disappeared before a stable binding') from exc
            except PermissionError as exc:
                note(report,'unverifiable_proc_read',pid=pid,error=type(exc).__name__+': '+str(exc))
                raise OwnershipError('Proc identity or terminal state could not be verified') from exc
            finally:
                if not retained:
                    if pidfd is not None:fs.close(pidfd)
                    if fd is not None:fs.close(fd)
        if owner is not None:owned_members(owner,rows)
        return rows,targets
    except BaseException:
        for fd,pidfd,_ in targets:fs.close(pidfd);fs.close(fd)
        raise

def process_rows(pgid,*,fs=None,report=None):return scan(pgid,fs or ProcFS(),report)[0]

def signal_verified(owner,sig,*,fs=None,report=None):
    fs=fs or ProcFS();rows,targets=scan(owner['pgid'],fs,report,owner);sent=[]
    try:
        # Validate the whole set before signalling; bind each target again
        # immediately before its pidfd send. Never signal a numeric PID/PGID.
        for fd,pidfd,row in targets:
            try:current=read_member(fs,fd,row['pid'],owner['pgid'],report,pidfd=pidfd)
            except MISSING:
                if not fs.dead(pidfd):raise OwnershipError('Live pidfd lost pinned proc identity')
                note(report,'verified_pidfd_target_exited',identity=row);continue
            if current is None:
                if not fs.dead(pidfd):raise OwnershipError('Live pidfd lost pinned proc identity')
                continue
            if not same(current,row):raise OwnershipError('Signal target identity changed')
            owned_members(owner,[current])
            try:fs.signal(pidfd,sig);sent.append(dict(pid=row['pid'],start_ticks=row['start_ticks'],signal=int(sig)))
            except ProcessLookupError:
                if not fs.dead(pidfd):raise OwnershipError('Signal failed for unverifiable live target')
        return sent
    finally:
        for fd,pidfd,_ in targets:fs.close(pidfd);fs.close(fd)
