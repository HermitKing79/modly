"""Host-only diagnostics. Never imports torch, executes artifacts or changes budgets."""
import json
import datetime
import math
import os
from pathlib import Path
import queue
import threading
import time

MARKER='FORWARD_TRACE='
RESERVED_FIELDS=frozenset(('event','name','origin','seq','elapsed_seconds',
                          'monotonic_seconds','timestamp_utc','self','call'))
CALL_FIELDS=frozenset(('duration_seconds','completion','error_type','error'))


def validate_json(value,depth=0):
    """Reject lossy JSON coercions, nonfinite numbers and unbounded nesting."""
    if depth>16:raise ValueError('Diagnostic metadata nesting exceeds bound')
    # PyTorch's version identity is TorchVersion(str); JSON preserves it as text.
    if value is None or isinstance(value,str) or type(value) in (bool,int):return
    if type(value) is float and math.isfinite(value):return
    if type(value) is list:
        for item in value:validate_json(item,depth+1)
        return
    if type(value) is dict:
        for key,item in value.items():
            if type(key) is not str:raise ValueError('Diagnostic metadata keys must be strings')
            validate_json(item,depth+1)
        return
    raise ValueError('Diagnostic metadata must contain finite JSON values')


def validate_details(details,reserved=RESERVED_FIELDS):
    conflict=reserved.intersection(details)
    if conflict:raise ValueError('Reserved diagnostic metadata fields: '+', '.join(sorted(conflict)))
    validate_json(details)


def validate_record(row):
    validate_json(row)
    if (type(row) is not dict or row.get('event') not in ('enter','return','error','stage','memory','state','process')
            or type(row.get('name')) is not str or not 0<len(row['name'])<=256
            or type(row.get('seq')) is not int or not 0<row['seq']<=512
            or row.get('origin') not in ('worker','supervisor')
            or any(type(row.get(key)) not in (float,int) or row[key]<0
                   for key in ('elapsed_seconds','monotonic_seconds'))):
        raise ValueError('Invalid or unbounded diagnostic record')
    try:
        stamp=datetime.datetime.fromisoformat(row['timestamp_utc'])
        if stamp.utcoffset()!=datetime.timedelta(0):raise ValueError('UTC required')
    except (KeyError,ValueError,TypeError) as exc:
        raise ValueError('Invalid diagnostic UTC timestamp') from exc


def unique_fields(pairs):
    row={}
    for key,value in pairs:
        if key in row:raise ValueError('Duplicate diagnostic JSON field: '+key)
        row[key]=value
    return row


class Trace:
    def __init__(self,send,clock=time.monotonic,limit=512,origin='worker'):
        self.send=send;self.clock=clock;self.start=clock();self.seq=0;self.limit=limit;self.origin=origin

    def emit(self,event,name,/,**details):
        validate_details(details)
        if self.seq>=self.limit:raise RuntimeError('Diagnostic event bound exceeded')
        now=self.clock()
        row={'event':event,'name':name,'origin':self.origin,'seq':self.seq+1,
             'elapsed_seconds':now-self.start,'monotonic_seconds':now,
             'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**details}
        validate_record(row)
        line=MARKER+json.dumps(row,allow_nan=False)
        if len(line)>16384:raise ValueError('Diagnostic line exceeds bound')
        self.seq+=1
        self.send(line)

    def call(self,name,call,/,**details):
        validate_details(details,RESERVED_FIELDS|CALL_FIELDS)
        began=self.clock();self.emit('enter',name,**details)
        try:result=call()
        except BaseException as exc:
            self.emit('error',name,duration_seconds=self.clock()-began,error_type=type(exc).__name__,error=str(exc)[:2048])
            raise
        self.emit('return',name,duration_seconds=self.clock()-began,completion='host-return-only')
        return result


class Journal:
    def __init__(self,path):
        self.stream=Path(path).open('x',encoding='utf-8',newline='\n');self.last=None;self.rows=0
        self.pending=queue.Queue(maxsize=2);self.error=None;self.closing=False
        self.thread=threading.Thread(target=self._write,daemon=True);self.thread.start()

    def _write(self):
        try:
            while True:
                item=self.pending.get()
                if item is None:break
                text,ack=item
                try:
                    self.stream.write(text);self.stream.flush();os.fsync(self.stream.fileno())
                except BaseException as exc:self.error=repr(exc)
                finally:ack.set()
        finally:
            try:self.stream.close()
            except BaseException as exc:self.error=repr(exc)

    def consume(self,line,timeout=1.):
        if self.closing or self.error:raise RuntimeError('Diagnostic journal unavailable: '+str(self.error))
        if not 0<timeout<=1.:raise ValueError('Journal acknowledgment bound must be within one second')
        if not line.startswith(MARKER) or len(line)>16384:raise ValueError('Invalid diagnostic line')
        def invalid(value):raise ValueError('Nonfinite diagnostic value')
        row=json.loads(line[len(MARKER):],parse_constant=invalid,object_pairs_hook=unique_fields)
        validate_record(row)
        if self.rows>=1024:raise ValueError('Unbounded diagnostic record count')
        ack=threading.Event()
        self.pending.put_nowait((json.dumps(row,allow_nan=False)+'\n',ack))
        if not ack.wait(timeout):
            self.error='Durable diagnostic acknowledgment timed out'
            raise TimeoutError(self.error)
        if self.error:raise RuntimeError('Diagnostic journal failed: '+self.error)
        self.rows+=1;self.last=row
        return row

    def close(self,timeout=2.):
        if not self.closing:self.closing=True;self.pending.put_nowait(None)
        self.thread.join(timeout)
        return not self.thread.is_alive()
    def __enter__(self):return self
    def __exit__(self,*args):
        if not self.close() and args[0] is None:raise TimeoutError('Diagnostic journal shutdown timed out')


def key_module(name,type_name):
    return name in ('','middle_block','out_layer') or type_name in ('Conv3d','ResBlock3d','UpsampleBlock3d')


class ProcessActivity:
    """Nonblocking cumulative CPU counters for the owned root and descendants."""
    def __init__(self,pid,module=None):
        if module is None:
            import psutil as module
        if module.__version__!='7.2.2':raise RuntimeError('Unexpected supervisor psutil version; no installation allowed')
        self.module=module;self.root=module.Process(pid);self.birth=self.root.create_time();self.previous={}

    def sample(self):
        try:
            observed=self.module.Process(self.root.pid).create_time()
            if observed!=self.birth:raise RuntimeError('Owned process PID was reused')
            owned=[self.root]+self.root.children(recursive=True)
        except self.module.NoSuchProcess:
            return {'processes':[],'vanished_pids':[self.root.pid],'root_exited':True,
                    'psutil_version':self.module.__version__,'psutil_path':self.module.__file__}
        if len(owned)>16:raise RuntimeError('Owned process tree exceeds diagnostic bound')
        rows=[];vanished=[]
        for process in owned:
            try:
                identity=(process.pid,process.create_time());times=process.cpu_times();cpu=times.user+times.system
                rows.append({'pid':process.pid,'created_seconds':identity[1],'name':process.name(),'executable':process.exe(),
                             'cpu_seconds':cpu,'cpu_delta_seconds':cpu-self.previous.get(identity,cpu)})
                self.previous[identity]=cpu
            except self.module.NoSuchProcess:vanished.append(process.pid)
        return {'processes':sorted(rows,key=lambda row:row['pid']),'vanished_pids':vanished,
                'psutil_version':self.module.__version__,'psutil_path':self.module.__file__}


class ActivityMonitor:
    """A stalled OS query cannot stall the supervisor's deadline thread."""
    def __init__(self,pid,send,interval=2.,activity_factory=ProcessActivity):
        self.pid=pid;self.send=send;self.interval=interval;self.factory=activity_factory
        self.stop_event=threading.Event();self.ready=threading.Event();self.last_sample=None;self.error=None
        self.thread=threading.Thread(target=self._run,daemon=True)

    def _run(self):
        try:
            activity=self.factory(self.pid);trace=Trace(self.send,origin='supervisor')
            while not self.stop_event.is_set():
                trace.emit('process','owned-tree',**activity.sample())
                self.last_sample=time.monotonic();self.ready.set()
                self.stop_event.wait(self.interval)
        except BaseException as exc:
            self.error=repr(exc);self.ready.set()

    def start(self,timeout=2.):
        self.thread.start()
        if not self.ready.wait(timeout):raise RuntimeError('CPU activity observer startup timed out')
        self.require_healthy()

    def require_healthy(self,max_age=5.):
        if self.error or self.last_sample is None or time.monotonic()-self.last_sample>max_age:
            raise RuntimeError('CPU activity observer unreliable: '+str(self.error))

    def close(self,timeout=2.):
        self.stop_event.set();self.thread.join(timeout)
        return not self.thread.is_alive()
