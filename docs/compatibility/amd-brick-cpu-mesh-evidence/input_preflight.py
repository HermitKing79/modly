"""Authenticate the saved ten tensors using existing production CPU readers."""
import hashlib,json,sys,time
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
sys.path[:0]=[str(ROOT/'scripts/compatibility'),str(ROOT/'experimental/extensions/trellis2-amd/inference_decoder')]
def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in {'torch','triton','flex_gemm','trellis2','o_voxel','cupy'}:
        raise RuntimeError('CPU preflight forbids AI/GPU import')
sys.addaudithook(audit)
import numpy as np
import amd_generation_mesh as mesh
import artifact_io
from mesh_host import SOURCE,SOURCE_SHA,OUTPUT,resource,save
began=time.monotonic();admission=resource()
if not admission['admitted']:raise RuntimeError('CPU preflight resources: '+json.dumps(admission))
if OUTPUT.exists():raise FileExistsError('Exclusive mesh output already exists')
sidecar=json.loads(SOURCE.with_name('provenance.json').read_text())
policy=mesh._contract_api();contract=policy.load(SOURCE.with_name('run-contract.json'))
if contract.final_support_profile!='brick-1957891-v1' or contract.maximum_final_rows!=1957891:raise ValueError('Exact qualified profile required')
before=SOURCE.stat()
if before.st_size!=104148612 or mesh.digest(SOURCE)!=SOURCE_SHA:raise ValueError('Neural size/hash')
with SOURCE.open('rb') as f:
    size=int.from_bytes(f.read(8),'little');header=json.loads(f.read(size),object_pairs_hook=mesh._object)
summary=json.loads(header['__metadata__']['summary'])
verified=artifact_io.verify_file(SOURCE,sidecar['provenance'],summary,sidecar['output']['tensor_payload_sha256'],contract)
arrays,children=mesh.read_neural(SOURCE,SOURCE_SHA,contract=contract)
try:
    hashes={k:hashlib.sha256(memoryview(v)).hexdigest() for k,v in arrays.items()}
    if hashes!=sidecar['output']['tensor_payload_sha256']:raise ValueError('All ten tensor hashes differ')
    original=ROOT/'docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/occupancy/shape/run/scratch/shape-latent-coordinates.npy'
    if mesh.digest(original)!=contract.input_sha256['coords.npy']:raise ValueError('Original input identity')
    if not np.array_equal(np.load(original,allow_pickle=False),arrays['subdivision_0_coordinates']):raise ValueError('Original input coordinate order')
finally:
    for a in arrays.values():a._mmap.close()
try:mesh.read_neural(SOURCE,SOURCE_SHA)
except ValueError:default_rejected=True
else:raise AssertionError('Historical default unexpectedly accepts expanded support')
after=SOURCE.stat()
if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise ValueError('Source changed during read')
record=dict(classification='PASS',CPU_only=True,source=str(SOURCE),source_sha256=SOURCE_SHA,
 source_bytes=before.st_size,run_contract_sha256=contract.sha256,profile=contract.final_support_profile,
 rows=1957891,children=children,ten_tensor_payload_sha256=hashes,production_readback=verified,
 original_coordinate_identity=True,historical_default_rejected=default_rejected,
 resources=admission,output_absent=True,elapsed_seconds=time.monotonic()-began,models_loaded=False,GPU_execution=False)
save(E/'input-preflight.json',record);print(json.dumps(record),flush=True)
