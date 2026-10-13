"""Bounded stdlib NPY/header/payload verification and byte-identical publication."""
import ast
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

root=Path(__file__).resolve().parents[3];evidence=Path(__file__).parent;scratch=evidence/'run/scratch'
def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in ('torch','triton','trellis2','cumesh','flex_gemm','o_voxel'):
        raise RuntimeError('Independent readback prohibits AI/native GPU imports')
sys.addaudithook(audit)
sys.path.insert(0,str(root/'scripts/compatibility'))
import amd_forward_diagnostics as diagnostics
results=json.loads((evidence/'run/results.json').read_text());worker=results['worker'];process=results['process']
assert results['classification']=='PASS' and results['observer_healthy'] and results['checkpoint_unchanged']
assert worker['load_calls']==worker['forward_calls']==worker['root_decoder_calls']==1
assert worker['transfer_calls']=={'input':1,'output':1}
assert worker['release']['allocated_bytes']==worker['release']['reserved_bytes']==0
assert all(worker['release'][n] for n in ('model_collected','input_collected','output_collected'))
assert worker['workspace_clear_calls']==1 and process['returncode']==0 and process['worker_exit_verified']
assert process['job_cleanup']['owned_tree_exit_verified'] and not process['job_cleanup']['forced_job_cleanup']
provenance=json.loads((scratch/'occupancy-provenance.json').read_text())
assert provenance==worker['occupancy_artifacts']
input_path=root/'docs/compatibility/amd-sparse-structure-sampling-evidence/sparse-structure-latent.safetensors'
assert hashlib.sha256(input_path.read_bytes()).hexdigest()==provenance['provenance']['input']['file_sha256']=='8a4da87ae1bbc2d5e5186f300adf11597bf986cb58dfb24a564a2eec287b44f2'
assert provenance['provenance']['decoder_checkpoint_sha256']=='1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a'
assert provenance['provenance']['harness_commit']==process['harness_commit']=='fe02093614605a0621c738b0e7827b287660fd14'
assert provenance['threshold']=='decoded > 0' and provenance['coordinate_order']==['batch','D','H','W']
def read(name,limit,expected_shape,descriptor,item_bytes,allow_fortran=False):
    path=scratch/name
    with path.open('rb') as f:blob=f.read(limit+1)
    assert len(blob)<=limit and len(blob)==provenance['files'][name]['file_bytes']
    assert hashlib.sha256(blob).hexdigest()==provenance['files'][name]['file_sha256']
    assert blob[:8]==b'\x93NUMPY\x01\x00'
    header_size=struct.unpack('<H',blob[8:10])[0];assert 0<header_size<=4096
    header=ast.literal_eval(blob[10:10+header_size].decode('ascii'))
    assert set(header)=={'descr','fortran_order','shape'}
    assert header['descr']==descriptor and type(header['fortran_order']) is bool and header['shape']==expected_shape
    assert allow_fortran or header['fortran_order'] is False
    payload=blob[10+header_size:];assert len(payload)==math.prod(expected_shape)*item_bytes
    if header['fortran_order'] is False:
        assert hashlib.sha256(payload).hexdigest()==provenance['files'][name]['payload_sha256']
    return blob,payload,header
logits_blob,logits,logits_header=read('occupancy-logits.npy',2*1024**2,(1,1,64,64,64),'<f4',4)
mask_blob,mask,mask_header=read('occupancy-mask.npy',512*1024,(64,64,64),'|b1',1)
values=[v[0] for v in struct.iter_unpack('<f',logits)]
assert len(values)==262144 and all(math.isfinite(v) for v in values)
assert mask==bytes(int(v>0) for v in values)
indices=[i for i,v in enumerate(mask) if v]
expected_coordinates=[(0,i//4096,(i//64)%64,i%64) for i in indices]
coords_blob,coords,coords_header=read('occupancy-coordinates.npy',5*1024**2,(len(indices),4),'<i4',4,allow_fortran=True)
if coords_header['fortran_order']:
    flat=[v[0] for v in struct.iter_unpack('<i',coords)]
    actual_coordinates=[tuple(flat[column*len(indices)+row] for column in range(4)) for row in range(len(indices))]
else:
    actual_coordinates=list(struct.iter_unpack('<4i',coords))
assert actual_coordinates==expected_coordinates
logical_coords=b''.join(struct.pack('<4i',*coordinate) for coordinate in actual_coordinates)
assert hashlib.sha256(logical_coords).hexdigest()==provenance['files']['occupancy-coordinates.npy']['payload_sha256']
assert len(indices)==provenance['occupied_count'] and len(indices)/262144==provenance['occupancy_fraction']
bbox=dict(minimum=[min(c[i] for c in expected_coordinates) for i in (1,2,3)],maximum=[max(c[i] for c in expected_coordinates) for i in (1,2,3)])
assert bbox==provenance['bounding_box']
remaining=set(indices);component_sizes=[]
while remaining:
    queue=deque([remaining.pop()]);count=0
    while queue:
        value=queue.popleft();count+=1;d,h,w=value//4096,(value//64)%64,value%64
        neighbors=[]
        for coordinate,stride in ((d,4096),(h,64),(w,1)):
            if coordinate>0:neighbors.append(value-stride)
            if coordinate<63:neighbors.append(value+stride)
        for adjacent in neighbors:
            if adjacent in remaining:remaining.remove(adjacent);queue.append(adjacent)
    component_sizes.append(count)
assert sorted(component_sizes,reverse=True)[:20]==provenance['components']['largest_sizes']
assert len(component_sizes)==provenance['components']['count'] and sum(component_sizes)==len(indices)
copies={}
for name,blob in [('occupancy-logits.npy',logits_blob),('occupancy-mask.npy',mask_blob),('occupancy-coordinates.npy',coords_blob),('occupancy-provenance.json',(scratch/'occupancy-provenance.json').read_bytes())]:
    target=evidence/name
    with target.open('xb') as f:f.write(blob)
    assert target.read_bytes()==blob
    copies[name]=dict(path=str(target),file_bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),byte_identical_to_owned_scratch=True)
trace=[json.loads(line,object_pairs_hook=diagnostics.unique_fields) for line in (evidence/'run/trace.jsonl').read_text().splitlines()]
windows=[json.loads(line,object_pairs_hook=diagnostics.unique_fields) for line in (evidence/'run/windows-periodic.jsonl').read_text().splitlines()]
for row in trace+windows:diagnostics.validate_record(row)
assert all(row['healthy'] for row in windows)
gap=max(b['monotonic_seconds']-a['monotonic_seconds'] for a,b in zip(windows,windows[1:]))
assert gap<=10
started=next(row for row in trace if row['origin']=='worker' and row['event']=='stage' and row['name']=='single-forward')
complete=next(row for row in trace if row['origin']=='worker' and row['event']=='stage' and row['name']=='complete')
synchronized=next(row for row in trace if row['origin']=='worker' and row['event']=='return' and row['name']=='synchronize:after-forward')
assert complete['monotonic_seconds']>synchronized['monotonic_seconds']>=started['monotonic_seconds']
origins=[row for row in trace if row['event']=='process']
convolutions=[dict(name=row['name'],duration_seconds=row['duration_seconds'],completion=row['completion']) for row in trace if row['origin']=='worker' and row['event']=='return' and row['name'].startswith('aten:')]
mean=math.fsum(values)/len(values)
row=dict(classification='PASS',independent_validation='stdlib bounded NPY header, complete raw F32/bool/I32 payloads, threshold, ordered coordinates,6-neighbor connectivity, hashes and provenance',
    artifacts=copies,native_shape=[1,1,64,64,64],dtype='float32',all_finite=True,threshold='decoded > 0',occupied_count=len(indices),occupancy_fraction=len(indices)/262144,
    npy_storage_order=dict(logits='C',mask='C',coordinates='Fortran' if coords_header['fortran_order'] else 'C'),
    payload_hash_policy='logical C-order bytes, distinct from complete NPY file hash; column-major coordinates decoded explicitly',
    bounding_box=bbox,coordinate_order=['batch','D','H','W'],bounding_box_extents=[b-a+1 for a,b in zip(bbox['minimum'],bbox['maximum'])],
    components=dict(connectivity=6,count=len(component_sizes),largest_sizes=sorted(component_sizes,reverse=True)),
    structure_quality='thin connected occupancy, two cells wide on W and touches both D volume boundaries; no detailed-shape quality claim',
    boundary_occupied_count=sum(any(c in (0,63) for c in coordinate[1:]) for coordinate in expected_coordinates),
    statistics=dict(minimum=min(values),maximum=max(values),mean_float64=mean,stddev_float64=math.sqrt(math.fsum((v-mean)**2 for v in values)/len(values))),
    cpu_loading_seconds=worker['cpu_load']['elapsed_seconds'],synchronized_forward_seconds=worker['forward']['elapsed_seconds'],
    gpu_stage_through_cleanup_seconds=complete['monotonic_seconds']-next(r for r in trace if r['origin']=='worker' and r['name']=='GPU-metadata')['monotonic_seconds'],
    overall_worker_seconds=process['elapsed_seconds'],peak_allocated_bytes=worker['release']['peak_allocated_bytes'],peak_reserved_bytes=worker['release']['peak_reserved_bytes'],
    release=worker['release'],process_exit_status=process['returncode'],owned_tree_exit_verified=process['job_cleanup']['owned_tree_exit_verified'],
    windows_periodic_samples=len(windows),maximum_windows_gap_seconds=gap,trace_records=len(trace),process_activity_samples=len(origins),
    observed_process_names=sorted({p['name'] for r in origins for p in r['processes']}),convolution_host_return_times=convolutions,
    host_return_is_gpu_completion=False,after_forward_synchronization_observed=True,torch_imported='torch' in sys.modules,additional_gpu_execution=False)
assert not row['torch_imported']
with (evidence/'independent-result-analysis.json').open('x',encoding='utf-8') as f:json.dump(row,f,indent=2)
print(json.dumps({k:row[k] for k in ('classification','occupied_count','bounding_box','bounding_box_extents','components','synchronized_forward_seconds','peak_allocated_bytes','peak_reserved_bytes','maximum_windows_gap_seconds','torch_imported')}))
