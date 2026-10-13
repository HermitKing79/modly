"""Read-only identity, available RAM and Windows observer; no framework import."""
import datetime
import json
from pathlib import Path
import sys
import time

root=Path(__file__).resolve().parents[3];evidence=Path(__file__).parent
sys.path.insert(0,str(root/'scripts/compatibility'))
def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in ('torch','trellis2','triton','cumesh','o_voxel','flex_gemm'):
        raise RuntimeError('Passive preflight prohibits AI framework imports')
sys.addaudithook(audit)
import amd_sparse_sampling as entry
began=time.monotonic();expected=entry.preflight();feature=entry.forward.local.verify_feature()
since=datetime.datetime.now(datetime.timezone.utc).isoformat()
observation=entry.forward.supervisor.windows_observation(since)
healthy=entry.forward.supervisor.healthy_observation(observation)
record=dict(checkpoint_sha256=expected['checkpoint_sha256'],config_sha256=expected['config_sha256'],
    checkpoint_entries=expected['tensor_count'],parameter_count=expected['parameter_count'],
    checkpoint_payload_bytes=expected['tensor_data_bytes'],feature=feature,sampler=entry.verify_sampler(),
    ram=entry.admission(),windows_observation=observation,observer_healthy=healthy,
    elapsed_seconds=time.monotonic()-began,torch_imported='torch' in sys.modules,
    real_model_weights_loaded=False,gpu_initialized=False)
with (evidence/'passive-preflight.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
print(json.dumps({k:v for k,v in record.items() if k not in ('sampler','feature','windows_observation')}))
if not healthy:raise RuntimeError('Passive Windows observer unhealthy')
