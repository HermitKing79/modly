"""One synthetic nonrecursive WSL conditioning output, never neural evidence."""
import importlib.util
import json
import os
from pathlib import Path
import pwd
import queue
import resource
import sys
import threading
import time


def publish(path, value):
    temporary=path.with_name(path.name+'.writing')
    with temporary.open('xb') as stream:
        stream.write((json.dumps(value,sort_keys=True)+'\n').encode())
        stream.flush();os.fsync(stream.fileno())
    os.link(temporary,path);temporary.unlink()


def main(leaf):
    leaf=Path(leaf)
    context=json.loads(os.environ['MODLY_COMPLETION_CONTEXT_JSON'])
    value=dict(context=context,stage_id=os.environ['MODLY_STAGE_ID'],
               nonce=os.environ['MODLY_STAGE_CONTROL_NONCE'],synthetic=True)
    if value['stage_id']!='conditioning' or context['nonce']==value['nonce']:
        raise ValueError('Independent original context and control ticket nonce required')
    if (os.getuid()!=1000 or os.geteuid()!=1000 or pwd.getpwuid(1000).pw_name!='modly'
            or os.path.realpath('/proc/self/exe')!='/usr/bin/python3.12'
            or resource.getrlimit(resource.RLIMIT_AS)!=(6*1024**3,6*1024**3)):
        raise ValueError('Original UID/interpreter/limit required')
    source=Path(__file__).resolve().parents[3]/'api/services/amd_wsl_session.py'
    spec=importlib.util.spec_from_file_location('original_broker_identity',source)
    broker=importlib.util.module_from_spec(spec);spec.loader.exec_module(broker)
    publish(leaf/'ready.json',dict(value,pid=os.getpid(),uid=os.getuid(),euid=os.geteuid(),user='modly',
        executable=os.path.realpath('/proc/self/exe'),rlimit_as=list(resource.getrlimit(resource.RLIMIT_AS)),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        identity=broker._parse_stat(Path('/proc/self/stat').read_text(),os.getuid())))
    controls=queue.Queue(maxsize=1)
    def control():
        try:
            line=sys.stdin.buffer.raw.readline(65537)
            if not line or len(line)>65536 or not line.endswith(b'\n'):raise ValueError('Invalid cancellation channel')
            if json.loads(line)!=dict(action='CANCEL',stage_id=value['stage_id'],nonce=value['nonce']):
                raise ValueError('Foreign cancellation')
            controls.put('CANCEL')
        except BaseException as error:controls.put(error)
    threading.Thread(target=control,daemon=True).start()
    end=time.monotonic()+20
    while True:
        try:item=controls.get_nowait()
        except queue.Empty:item=None
        if isinstance(item,BaseException):raise item
        if item=='CANCEL':
            publish(Path(os.environ['MODLY_STAGE_CANCEL_ACK']),dict(value,terminal='cancelled'))
            return 0
        release=leaf/'release.json'
        if release.exists() and not release.with_name(release.name+'.writing').exists():
            if json.loads(release.read_text())!=value:raise ValueError('Foreign host release')
            break
        if time.monotonic()>=end:raise TimeoutError('Host native-identity release missing')
        time.sleep(.01)
    publish(leaf/'synthetic-output.json',dict(value,value='synthetic conditioning transport only; no model evidence'))
    return 0


if __name__=='__main__':raise SystemExit(main(sys.argv[1]))
