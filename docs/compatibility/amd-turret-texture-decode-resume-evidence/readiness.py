"""Exactly three Windows-only RAM readings after the approved 60s settling delay."""
import ctypes,datetime,json,os,time
from pathlib import Path
class Memory(ctypes.Structure):
 _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
E=Path(__file__).resolve().parent
start=datetime.datetime(2026,10,15,8,35,34,tzinfo=datetime.timezone.utc).timestamp()
required=37*1024**3//2
while time.time()<start+60:time.sleep(min(1,start+60-time.time()))
rows=[]
with (E/'ram-readiness-samples.jsonl').open('x',encoding='utf-8') as log:
 for index in range(3):
  m=Memory();m.length=ctypes.sizeof(m)
  if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
  row={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'available_ram_bytes':m.available,'required_prelaunch_ram_bytes':required,'pass':m.available>=required,'seconds_since_request':time.time()-start}
  rows.append(row);log.write(json.dumps(row)+'\n');log.flush();os.fsync(log.fileno());print(json.dumps(row),flush=True)
  if index<2:time.sleep(5)
passed=all(row['pass'] for row in rows)
v={'classification':'PASS' if passed else 'BLOCKED','required_bytes':required,'settling_seconds_observed':rows[0]['seconds_since_request'],'consecutive_passing_samples':3 if passed else 0,'samples':rows,'GPU_initialized':False,'WSL_launched':False,'inference_marker_reserved':False}
with (E/'ram-readiness.json').open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
print('RAM_READINESS='+v['classification'],flush=True)
raise SystemExit(0 if passed else 2)
