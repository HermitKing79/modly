"""Read-only CPU admission/identity evidence before the sole owned GPU worker."""
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[3];evidence=Path(__file__).parent
sys.path.insert(0,str(root/'scripts/compatibility'))
def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in ('torch','triton','trellis2','cumesh','flex_gemm','o_voxel'):
        raise RuntimeError('CPU preflight prohibits AI/native GPU imports')
sys.addaudithook(audit)
import amd_sparse_occupancy as entry
entry.teardown.verify_workspace()
assert not (evidence/'attempt-started.json').exists() and not (evidence/'run').exists()
identity=entry.identity();array,latent=entry.data.read_input();plan=entry.cache_plan()
assert array.shape==(1,8,16,16,16) and str(array.dtype)=='float32'
row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workspace_dir=os.environ['WORKSPACE_DIR'],
    actual_verify_workspace_passed=True,ram=entry.admission(),input=latent,decoder_identity=identity,
    dependencies={name:importlib.metadata.version(name) for name in ('numpy','safetensors','psutil')},
    pipeline_source_sha256=entry.PIPELINE_SHA,eligible_cache_entries=len(plan['selected']),runtime_libraries=plan['runtime_libraries'],
    source_contract=dict(input_shape=[1,8,16,16,16],output_shape=[1,1,64,64,64],input_dtype='float32',output_dtype='float32',
        intermediate_dtype='float16',upsampling='two pixel_shuffle_3d(...,2) blocks',threshold='decoded > 0',
        coordinates=['batch','D','H','W'],native_resolution_preserved=True,
        optional_later_32_cubed_rule='max_pool3d(mask.float(),2,2,0)>0.5; not executed'),
    deadlines=dict(cpu_seconds=120,gpu_seconds=180,overall_seconds=360),allocator_cap_bytes=12*1024**3,
    live_rocm_vram_admission='at least13GiB available, zero allocated/reserved; checked by owned worker before placement',
    torch_imported='torch' in sys.modules,gpu_initialized=False,model_weights_loaded=False,attempt_marker_exists=False)
assert row['torch_imported'] is False
with (evidence/'preparation.json').open('x',encoding='utf-8') as f:json.dump(row,f,indent=2)
print(json.dumps(dict(workspace_verified=True,ram=row['ram'],input_sha256=entry.data.INPUT_SHA,
    checkpoint_sha256=identity['checkpoint_sha256'],checkpoint_entries=len(identity['tensors']),
    eligible_cache_entries=len(plan['selected']),torch_imported=False)))
