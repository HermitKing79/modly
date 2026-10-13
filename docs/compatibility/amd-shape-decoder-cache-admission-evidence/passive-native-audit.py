"""One read-only audit of saved manifests/current files; no AI/native imports."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
EV=Path(__file__).parent
sys.path.insert(0,str(ROOT/'scripts/compatibility'))
import amd_shape_decoder_preflight as prior


def deny_ai(event,args):
    if event=='import' and str(args[0]).split('.')[0] in ('torch','triton','flex_gemm','cumesh','o_voxel','trellis2'):
        raise RuntimeError('Passive audit forbids AI/native imports')


sys.addaudithook(deny_ai)
old=json.loads((ROOT/'docs/compatibility/amd-shape-decoder-neural-evidence/preflight.json').read_text(encoding='utf-8'))
probe=json.loads((EV/'cpu-probe-5/results.json').read_text(encoding='utf-8'))
assert probe['cpu_probe_passed'] and not probe['worker']['gpu_initialized']
assert probe['worker']['harness_sha256']==prior.digest(ROOT/'scripts/compatibility/amd_shape_cache_identity.py')
native=old['native_source_and_binary_sha256'];sources=old['source_sha256']
for name,expected in {**native,**sources}.items():
    assert prior.digest(Path(name))==expected, name
for row in old['cache_metadata']:
    metadata=Path(row['path']);group=metadata.parent/('__grp__'+metadata.name)
    assert prior.digest(group)==row['group_sha256']
    current=json.loads(metadata.read_text(encoding='utf-8'))
    assert current==row['metadata']
    prior.validate_cache_metadata(current)
    children=json.loads(group.read_text(encoding='utf-8'))['child_paths']
    paths=prior.cache_group_paths(metadata,children)
    for name,path in paths.items():
        assert path.stat().st_size==row['files'][name]['bytes']
        assert prior.digest(path)==row['files'][name]['sha256']
first=probe['worker']['first_kernel'];helper=probe['worker']['helper']
assert first['cache_hit'] and first['matches_historical'] and helper['cache_hit']
first_row=next(row for row in old['cache_metadata'] if row['metadata']['hash']==first['computed_hash'])
assert first_row['metadata']['num_warps']==first['options']['num_warps']
assert first_row['metadata']['waves_per_eu']==first['options']['waves_per_eu']
assert helper['cached_sha256']==native[helper['expected_path']]
checkpoint=prior.PREFIX.with_suffix('.safetensors')
assert prior.digest(checkpoint)==old['checkpoint']['checkpoint_sha256']
assert prior.digest(prior.PREFIX.with_suffix('.json'))==old['checkpoint']['config_sha256']
for name,row in old['inputs'].items():assert prior.digest(Path(row['path']))==row['sha256']
tuning=old['tuning'];assert prior.digest(Path(tuning['path']))==tuning['sha256']
support=[prior.SITE/'triton/_C/libtriton.pyd',prior.SITE/'triton/backends/amd/lib/ocml.bc',prior.SITE/'triton/backends/amd/lib/ockl.bc',
         prior.SITE/'triton/compiler/compiler.py',prior.SITE/'triton/backends/amd/driver.py',
         prior.SITE/'triton/backends/amd/driver.c',Path(helper['library'])]
support.extend((prior.SITE/'triton/backends/amd/include').rglob('*.h'))
support_identity={str(p):dict(bytes=p.stat().st_size,sha256=prior.digest(p)) for p in sorted(support)}
helper_path=Path(helper['expected_path']);binary=helper_path.read_bytes()
assert binary[:2]==b'MZ'
offset=int.from_bytes(binary[60:64],'little')
assert binary[offset:offset+4]==b'PE\0\0'
machine=int.from_bytes(binary[offset+4:offset+6],'little')
assert machine==0x8664
result=dict(classification='UNKNOWN',gpu_initialized=False,weights_loaded=False,kernel_launches=0,
    native_helper_loaded=False,cache_products_executed=False,compiled=False,
    source_files_verified=len(sources),native_files_verified=len(native),
    complete_groups_verified=len(old['cache_metadata']),cache_products_verified=sum(len(row['files']) for row in old['cache_metadata']),
    input_hashes_verified=old['inputs'],checkpoint_sha256=old['checkpoint']['checkpoint_sha256'],
    configuration_sha256=old['checkpoint']['config_sha256'],
    tuning=tuning,first_kernel=first,helper=helper,support_file_identity=support_identity,
    helper_pe=dict(machine_hex=hex(machine),architecture='AMD64',coff_timestamp=int.from_bytes(binary[offset+8:offset+12],'little'),
                  bytes=len(binary),compiler_header_build_provenance_attested=False),
    cache_key_limits=['HIPOptions hashes extern-library paths, not bitcode content; current bitcode hashes recorded, historical build-input hashes unavailable.',
                      'HIP helper key includes C source, platform and include/library directory strings, not compiler or header bytes. PE/hash/key checks do not prove helper loadability.',
                      'Only canonical first splitK2 dispatch was recomputed from actual installed binder/compiler key; later learned supports and unsplit dispatches are not attested.'],
    missing_execution_gates=['No installed cache-only compiler/helper/custom-autotuner interception exists in this candidate.',
                             'No hooked decoder subdivision/sparse-convolution pre-allocation guards; planners remain synthetic contracts.',
                             'Sorting/cumsum internal workspace and whole-call neural liveness remain unbounded by a proved per-call estimate.'],
    next_experiment=dict(cpu_only=True,operation='Exercise one actual installed compile/cache entrypoint under a narrow fail-closed interception, with a deliberate key miss.',
                         acceptance='Reject before every native/compile/build/autotune callback; zero GPU initialization and native callback count. A verified-hit control must preserve exact key/provenance without accessing GPU handles.',
                         excluded='No weights, helper/kernel execution, GPU initialization, decoder forward or benchmark.',
                         resource_limit='One isolated owned CPU process,120s maximum; use existing Job and scratch policies, no infrastructure rebuild.'))
assert not any(n in sys.modules for n in ('torch','triton','flex_gemm','cumesh','o_voxel','trellis2'))
with (EV/'native-cache-audit.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps({key:result[key] for key in ('classification','source_files_verified','native_files_verified','complete_groups_verified','cache_products_verified','gpu_initialized','weights_loaded','kernel_launches')}))
