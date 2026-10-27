"""Bounded denial evidence; grants no filesystem permissions."""
import json,os,sys,threading,time
from datetime import datetime,timezone
from pathlib import Path
from amd_pixal3d_proof_common import plain

def frames(frame=None):
    frame=frame or sys._getframe(1);rows=[]
    while frame is not None and len(rows)<24:
        rows.append(dict(file=frame.f_code.co_filename,function=frame.f_code.co_name,line=frame.f_lineno));frame=frame.f_back
    return rows

def argument(value):
    if value is None or isinstance(value,(int,bool)):return value
    if isinstance(value,bytes):value=os.fsdecode(value)
    if isinstance(value,(str,Path)):
        text=str(value)
        return text if len(text)<=8192 else dict(prefix=text[:8192],truncated=True)
    return dict(unserialized_type=type(value).__name__)

def descriptor_observations(event,args):
    indices={'os.remove':(1,),'os.rmdir':(1,),'os.rename':(2,3),'os.link':(2,3),'os.symlink':(2,),'os.truncate':(0,)}.get(event,())
    rows=[]
    for index in indices:
        if index>=len(args) or type(args[index]) is not int or args[index]<0:continue
        fd=args[index];row=dict(argument_index=index,fd=fd,observation_only=True)
        try:
            before=os.fstat(fd);row['target']=os.readlink('/proc/self/fd/'+str(fd)) if os.name=='posix' else None
            after=os.fstat(fd);row.update(device=before.st_dev,inode=before.st_ino,mode=before.st_mode,
                identity_stable=(before.st_dev,before.st_ino)==(after.st_dev,after.st_ino))
        except OSError as exc:row['unavailable']=str(exc)
        rows.append(row)
    return rows

class AuditLog:
    def __init__(self,path):
        self.path=plain(path);self.records=0;self.phase='before-imports';self.last=None;self.lock=threading.Lock()
        self.fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_BINARY',0),0o600)
    def __call__(self,event,args):
        with self.lock:self._record(event,args)
    def _record(self,event,args):
        if self.fd is None:raise RuntimeError('Audit logger closed; mutation still denied')
        if self.records>=32:raise RuntimeError('Audit evidence record bound reached; mutation still denied')
        row=dict(utc=datetime.now(timezone.utc).isoformat(),monotonic=time.monotonic(),pid=os.getpid(),thread=threading.get_ident(),phase=self.phase,
            event=event,arguments=[argument(v) for v in args],cwd=os.getcwd(),descriptors=descriptor_observations(event,args),
            caller_frames=frames(sys._getframe(1)),operation_allowed=False)
        data=(json.dumps(row,ensure_ascii=True,allow_nan=False)+'\n').encode('utf-8')
        if len(data)>65536:raise RuntimeError('Audit evidence byte bound reached; mutation still denied')
        view=memoryview(data)
        while view:
            count=os.write(self.fd,view)
            if count<=0:raise RuntimeError('Audit evidence write failed; mutation still denied')
            view=view[count:]
        os.fsync(self.fd);self.records+=1;self.last=row
    def close(self):
        with self.lock:
            fd=self.fd;self.fd=None
            if fd is not None:os.close(fd)

def exception_evidence(exc):
    rows=[];tb=exc.__traceback__
    while tb is not None and len(rows)<24:
        rows.append(dict(file=tb.tb_frame.f_code.co_filename,function=tb.tb_frame.f_code.co_name,line=tb.tb_lineno));tb=tb.tb_next
    return dict(utc=datetime.now(timezone.utc).isoformat(),monotonic=time.monotonic(),error=type(exc).__name__+': '+str(exc),frames=rows)
