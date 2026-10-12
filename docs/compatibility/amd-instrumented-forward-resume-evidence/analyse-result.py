"""Post-run verification and inert cache analysis. No AI imports or GPU queries."""
import bz2
import datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import struct
import sys

p=Path(__file__).parent;root=p.parents[2]
sys.path.insert(0,str(root/'scripts/compatibility'))
from amd_forward_cache_audit import elf_header

def save(name,data):
    with (p/name).open('x',encoding='utf-8') as stream:json.dump(data,stream,indent=2)

r=json.loads((p/'run/results.json').read_text());w=r['worker'];process=r['process']
assert r['classification']=='PASS' and r['observer_healthy'] and r['checkpoint_unchanged']
assert w['status']=='PASS' and w['stage']=='complete'
assert w['load_calls']==w['forward_calls']==w['root_decoder_calls']==1
assert w['transfer_calls']=={'input':1,'output':1}
assert w['cpu_load']['status']=='PASS' and w['cpu_load']['strict'] and not w['cpu_load']['gpu_initialized']
assert w['cpu_load']['missing_keys']==w['cpu_load']['unexpected_keys']==[]
assert w['forward']['input_shape']==[1,8,1,1,1] and w['forward']['input_dtype']=='torch.float32'
output=w['forward']['output'];values=output['values']
assert output['shape']==[1,1,4,4,4] and output['dtype']=='torch.float32' and output['device']=='cuda:0'
assert len(values)==64 and all(math.isfinite(value) for value in values)
assert output['requires_grad'] is False and output['inference_mode'] is True
assert w['forward']['grad_enabled'] is False and w['forward']['output_is_inference_tensor'] is True
assert hashlib.sha256(struct.pack('<64f',*values)).hexdigest()==w['forward']['output_sha256']
assert statistics.fmean(values)==w['forward']['statistics']['mean']
assert w['release']['allocated_bytes']==w['release']['reserved_bytes']==0
assert all(w['release'][name] is True for name in ('model_collected','input_collected','output_collected'))
assert process['returncode']==0 and not process['timed_out'] and process['worker_exit_verified']
assert process['job_cleanup']['owned_tree_exit_verified'] and process['job_cleanup']['active_after_cleanup']==0
assert not process['job_cleanup']['forced_job_cleanup']
assert all(process[key] for key in ('cpu_observer_stopped','windows_periodic_stopped','output_readers_completed',
                                   'diagnostic_journal_stopped','windows_journal_stopped'))
assert not any(process.get(key) for key in ('error','cpu_observer_error','windows_periodic_error',
                                          'diagnostic_journal_error','windows_journal_error'))
assert (p/'run/checkpoint-before.json').read_bytes()==(p/'run/checkpoint-after.json').read_bytes()

trace=[json.loads(line) for line in (p/'run/trace.jsonl').read_text().splitlines()]
worker=[row for row in trace if row['origin']=='worker'];cpu=[row for row in trace if row['event']=='process']
assert [row['seq'] for row in worker]==list(range(1,len(worker)+1))
assert next(row for row in worker if row['name']=='gpu')['gpu']==w['gpu']
stack=[];operations=[]
for row in worker:
    if row['event']=='enter':stack.append(row)
    elif row['event']=='return':
        entered=stack.pop();assert entered['name']==row['name']
        operations.append({'name':row['name'],'entry':entered,'return':row})
    assert row['event']!='error'
assert not stack
sync=next(row for row in worker if row['name']=='synchronize:after-forward' and row['event']=='return')
assert sum(row['event']=='enter' and row['name']=='module:<root>' for row in worker)==1
windows=[json.loads(line) for line in (p/'run/windows-periodic.jsonl').read_text().splitlines()]
assert len(windows)==process['windows_periodic_samples'] and all(row['healthy'] for row in windows)
gaps=[b['monotonic_seconds']-a['monotonic_seconds'] for a,b in zip(windows,windows[1:])]
assert max(gaps)<10

plan=json.loads((p/'cache-plan.json').read_text());seeds={row['path']:row for row in plan['selected']}
scratch=p/'run/scratch';files=[]
for path in sorted(scratch.rglob('*')):
    if not path.is_file():continue
    before=path.stat();name=path.relative_to(scratch).as_posix()
    raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    row={'path':name,'bytes':len(raw),'sha256':sha,'mtime_ns':before.st_mtime_ns,
         'copied_seed':name in seeds,'same_seed_bytes':sha==seeds[name]['sha256'] if name in seeds else None}
    if path.name.startswith('llvmcache-'):
        length=struct.unpack_from('<Q',raw)[0]
        assert length<=len(raw)-16
        log_length=struct.unpack_from('<Q',raw,8+length)[0];assert length+log_length+16==len(raw)
        payload=raw[8:8+length]
        offset=0 if payload.startswith(b'\x7fELF') else 8 if payload[8:12]==b'\x7fELF' else None
        row.update(container_complete=True,compiler_log_bytes=log_length,
                   elf_offset=offset,elf=elf_header(payload[offset:]) if offset is not None else None)
    assert path.stat().st_mtime_ns==before.st_mtime_ns
    files.append(row)
save('scratch-side-effects.json',{'file_count':len(files),'total_bytes':sum(row['bytes'] for row in files),'files':files,
                                 'read_only':True,'executed_artifacts':0})

db=scratch/'miopen_custom_cache_dir/3.6.0.8d1ae90e/gfx1201_32.ukdb'
connection=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
try:
    connection.execute('PRAGMA query_only=ON')
    check=connection.execute('PRAGMA quick_check').fetchall();assert check==[('ok',)]
    kernels=[]
    for ident,name,args,key,size,blob in connection.execute('SELECT id,kernel_name,kernel_args,kernel_hash,uncompressed_size,kernel_blob FROM kern_db ORDER BY id'):
        assert 0<size<=2*1024**2
        decoder=bz2.BZ2Decompressor();decoded=decoder.decompress(blob,max_length=size+1)
        assert decoder.eof and not decoder.unused_data and len(decoded)==size
        kernels.append({'id':ident,'name':name,'arguments':args,'recorded_hash':key,'bytes':size,
                        'decoded_sha256':hashlib.sha256(decoded).hexdigest(),'elf':elf_header(decoded)})
finally:connection.close()

new_comgr=[row for row in files if row['path'].startswith('amd_comgr_cache_dir/llvmcache-') and not row['copied_seed']]
delays=sorted(operations,key=lambda row:row['return']['duration_seconds'],reverse=True)
for op in delays:
    a=datetime.datetime.fromisoformat(op['entry']['timestamp_utc']).timestamp()*1e9
    b=datetime.datetime.fromisoformat(op['return']['timestamp_utc']).timestamp()*1e9
    op['new_COMGR_mtimes_in_interval']=sum(a<=row['mtime_ns']<=b for row in new_comgr)
cpu_processes=[proc for row in cpu for proc in row['processes']]
interpretation={'classification':'PASS','harness_commit':process['harness_commit'],
    'exact_load_root_forward_transfer_counts_verified':True,'independent_output_sha256':w['forward']['output_sha256'],
    'forward_seconds_including_synchronization':w['forward']['elapsed_seconds'],
    'last_confirmed_worker_event':worker[-1],'synchronization_return':sync,
    'host_operation_returns_are_not_per_operation_GPU_completion':True,
    'worker_trace_events':len(worker),'all_trace_events':len(trace),'operator_aten_markers':sum(row['name'].startswith('aten:') for row in worker),
    'CPU_samples':len(cpu),'observed_process_names':sorted(set(proc['name'] for proc in cpu_processes)),
    'CPU_seconds_by_pid':{str(pid):{'first':next(proc['cpu_seconds'] for proc in cpu_processes if proc['pid']==pid),
                                  'last':next(proc['cpu_seconds'] for proc in reversed(cpu_processes) if proc['pid']==pid)}
                           for pid in sorted(set(proc['pid'] for proc in cpu_processes))},
    'separate_compiler_child_observed':any(proc['name']!='python.exe' for proc in cpu_processes),
    'periodic_windows_samples':len(windows),'max_observer_interval_seconds':max(gaps),
    'scratch_files':len(files),'new_COMGR_entries':len(new_comgr),'miopen_records':kernels,
    'longest_host_operations':delays[:4],
    'native_computation_failure':False,'guard_error':False,'timeout':False,
    'warm_up_or_retry':False,'AI_imported_by_analysis':False}
assert not any(name.split('.')[0] in ('torch','triton','trellis2','safetensors') for name in sys.modules)
save('attempt-interpretation.json',interpretation)
print(json.dumps({key:interpretation[key] for key in ('classification','forward_seconds_including_synchronization',
    'worker_trace_events','operator_aten_markers','CPU_samples','observed_process_names','periodic_windows_samples',
    'max_observer_interval_seconds','scratch_files','new_COMGR_entries')}))
print(json.dumps({'MIOpen_records':len(kernels),'longest':[{ 'name':op['name'],'host_seconds':op['return']['duration_seconds'],
    'new_COMGR_mtimes_in_interval':op['new_COMGR_mtimes_in_interval']} for op in delays[:4]]}))
