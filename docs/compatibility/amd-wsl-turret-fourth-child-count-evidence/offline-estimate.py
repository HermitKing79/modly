"""Post-run CPU arithmetic only; never admits or invokes a GPU operation."""
import argparse
from hashlib import sha256
import json
from math import prod
from pathlib import Path
from admission import ALLOCATOR_LIMIT,FREE_RESERVE,OPAQUE_HEADROOM,MAX_ROWS

def buffers(n,ci=64,co=64):
    if type(n) is not int or n<0:raise ValueError('Nonnegative actual integer rows required')
    # Pinned explicit-GEMM source and existing conv_plan, uncached 3x3x3 BF16.
    terms=dict(im2col_bytes=n*27*ci*2,gathered_bytes=n*27*ci*2,
        neighbor_map_bytes=n*27*4,long_index_bytes=2*n*27*8,mask_bytes=n*27,
        hash_bytes=2*n*(4+4),output_bytes=n*co*2,block_extra_bytes=n*(16*ci+4*co))
    terms['visible_bytes']=sum(terms.values())
    return terms

def estimate(row):
    n=row['children'];parents=row['parents'];ci=co=row['out_channels']
    if row['site']!='blocks.3.4.updown' or row['call']!=1 or parents!=378891 or ci!=64:
        raise ValueError('Require the actual first fourth-stage record')
    if type(n) is not int or not 0<=n<=8*parents:raise ValueError('Invalid genuine boolean-mask count')
    volume_bound=prod(2*v for v in row['parent_spatial_shape'])
    if not 0<volume_bound<2**32:raise ValueError('Four-byte hashmap key assumption not proven')
    terms=buffers(n,ci,co);padded=(terms['visible_bytes']*5+3)//4
    memory=row['memory'];retained=max(memory['allocated_bytes'],memory['reserved_bytes'])
    source=Path(__file__).parent.parent/'amd-wsl-native-decoder-preparation-evidence/pinned-linux-submanifold_conv3d.py.txt'
    return {'diagnostic_only':True,'GPU_execution_authorized':False,'actual_children':n,
        'row_gate_rejects':not 0<n<=MAX_ROWS,'zero_children':n==0,'out_channels':co,
        'target':'blocks.3.4.conv2, original uncached explicit_gemm, 3x3x3, BF16',
        'source_sha256':sha256(source.read_bytes()).hexdigest(),'child_spatial_volume_upper_bound':volume_bound,
        'buffer_upper_bounds':terms,'gathered_self_neighbor_lower_bound_bytes':n*ci*2,
        'padded_visible_bytes':padded,'opaque_headroom_bytes':OPAQUE_HEADROOM,'free_reserve_bytes':FREE_RESERVE,
        'pre_expansion_memory':memory,'allocator_ceiling_bytes':ALLOCATOR_LIMIT,
        'allocator_required_using_pre_expansion_retained_bytes':retained+padded+OPAQUE_HEADROOM,
        'allocator_formula_passes_using_pre_expansion_baseline':retained+padded+OPAQUE_HEADROOM<=ALLOCATOR_LIMIT,
        'device_free_formula_passes_using_pre_expansion_baseline':memory['free_bytes']>=padded+FREE_RESERVE,
        'limitation':'No child coordinates or neighbour map exists. Gather/index sizes are upper bounds; real valid-neighbour counts and post-C2S retained memory are unmeasured. No admission is granted.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('journal',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    rows=[json.loads(line) for line in a.journal.read_text().splitlines()]
    matches=[r for r in rows if r.get('phase')=='c2s-count-before-admission' and r['site']=='blocks.3.4.updown' and r['call']==1]
    if len(matches)!=1:raise RuntimeError('Exactly one genuine first fourth-stage count required')
    result=estimate(matches[0])
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
