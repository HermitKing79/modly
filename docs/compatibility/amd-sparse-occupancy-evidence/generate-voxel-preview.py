"""Optional CPU exposed-voxel diagnostic after independently verified worker exit."""
import hashlib
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[3];evidence=Path(__file__).parent
def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in ('torch','triton','trellis2','cumesh','flex_gemm','o_voxel','nvdiffrast'):
        raise RuntimeError('Voxel preview prohibits AI/native GPU imports')
sys.addaudithook(audit)
sys.path.insert(0,str(root/'scripts/compatibility'))
import amd_sparse_occupancy_data as data
import numpy as np
validation=json.loads((evidence/'independent-result-analysis.json').read_text())
assert validation['classification']=='PASS' and validation['owned_tree_exit_verified'] and validation['process_exit_status']==0
mask_path=evidence/'occupancy-mask.npy';assert hashlib.sha256(mask_path.read_bytes()).hexdigest()==validation['artifacts']['occupancy-mask.npy']['sha256']
mask=np.load(mask_path,allow_pickle=False,max_header_size=4096)
assert mask.shape==(64,64,64) and mask.dtype==bool and mask.sum()==validation['occupied_count']
assert validation['components']['count']==1 and all(v>1 for v in validation['bounding_box_extents'])
row=data.write_preview(evidence,mask)
row.update(mask_sha256=validation['artifacts']['occupancy-mask.npy']['sha256'],no_downsampling=True,
    structure_quality=validation['structure_quality'],source='exact exposed faces of committed native occupancy mask',torch_imported='torch' in sys.modules)
assert not row['torch_imported']
with (evidence/'voxel-preview.json').open('x',encoding='utf-8') as f:json.dump(row,f,indent=2)
print(json.dumps(row))
