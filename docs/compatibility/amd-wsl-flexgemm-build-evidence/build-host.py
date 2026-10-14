"""Own one WSL client/guest compiler sequence and enforce host RAM/disk floors."""
import ctypes
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
E=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('existing_owned_job',ROOT/'scripts/compatibility/amd_forward_owned_job.py')
job_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(job_module)
class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
def host_ram():
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise ctypes.WinError()
    return m.available
guest='/mnt/c/'+str(E/'build-monitor.py').replace('\\','/')[3:]
command=['wsl.exe','-d','Modly-AMD-Ubuntu-24.04','-u','modly','--exec',
         '/usr/bin/timeout','--signal=TERM','--kill-after=5s','2460s',
         '/opt/modly-wsl/venv/bin/python','-B','-I','-S',guest]
record={'command':command,'minimum_host_available_RAM_bytes':host_ram(),
        'minimum_D_free_bytes':shutil.disk_usage('D:\\').free,
        'GPU_actions':False,'build_retry':False}
job=None;proc=None;start=time.monotonic()
try:
    if record['minimum_host_available_RAM_bytes']<4*1024**3 or record['minimum_D_free_bytes']<50*1024**3:
        raise RuntimeError('Host admission failed')
    job=job_module.OwnedJob()
    with (E/'build-host.stdout.txt').open('xb') as out,(E/'build-host.stderr.txt').open('xb') as err:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=out,stderr=err)
        job.attach(proc)
        proc.stdin.write(b'GO\n');proc.stdin.flush();proc.stdin.close()
        while proc.poll() is None:
            ram=host_ram();disk=shutil.disk_usage('D:\\').free
            record['minimum_host_available_RAM_bytes']=min(record['minimum_host_available_RAM_bytes'],ram)
            record['minimum_D_free_bytes']=min(record['minimum_D_free_bytes'],disk)
            if ram<4*1024**3 or disk<50*1024**3:raise RuntimeError('Host resource floor violated')
            if time.monotonic()-start>2475:raise TimeoutError('WSL build supervision deadline')
            time.sleep(1)
        record['worker_exit']=proc.returncode
    if proc.returncode:raise RuntimeError('Single native build worker failed: '+str(proc.returncode))
except BaseException:
    record['error']=traceback.format_exc()
finally:
    if proc is not None and proc.poll() is None:
        stop=subprocess.run(['wsl.exe','--terminate','Modly-AMD-Ubuntu-24.04'],capture_output=True,timeout=15)
        record['emergency_named_distro_termination_exit']=stop.returncode
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:record['client_exit_unverified']=True
    if job is not None:
        began=time.monotonic()
        while job.active() and time.monotonic()-began<5:time.sleep(.025)
        record['natural_job_grace_seconds']=time.monotonic()-began
        record['job']=job.finish()
    stop=subprocess.run(['wsl.exe','--terminate','Modly-AMD-Ubuntu-24.04'],capture_output=True,timeout=15)
    record['final_named_distro_termination_exit']=stop.returncode
    record['total_seconds']=time.monotonic()-start
    with (E/'build-host-result.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
    print(json.dumps(record,indent=2),flush=True)
if 'error' in record:raise SystemExit(1)
