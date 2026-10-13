"""One exclusive owned CPU observation; no GPU or unfinished acceptance route."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'api'),str(Path(__file__).parent)]
import amd_native_convolution_preflight as preflight
import amd_blaslt_teardown as teardown
EV=ROOT/'docs/compatibility/amd-native-python-tail-contract-evidence'

def acceptance(torch,meta,source,scratch):
    from amd_native_convolution_policy import NativePolicy
    from amd_forward_diagnostics import Trace
    from amd_native_python_tail_support import replay,observe
    private=preflight.private_cache(scratch)
    policy=NativePolicy(torch,private,meta,source,Trace(lambda line:None))
    try:
        policy.permit.phase='convolution'
        with torch.inference_mode():
            case=replay(torch,lambda weight,output:None);record=observe(torch,case)
        # The audit deliberately permits writes only inside owned scratch.
        with (scratch/'observation.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
        return dict(passed=True,observation_complete=True,operators=[row['operator'] for row in record['operators']],
            gpu_initialized=torch.cuda.is_initialized(),native_gpu_executed=False,weights_loaded=False)
    finally:policy.close()

def preserve_capture(output,result):
    output=output.resolve()
    if output!=EV.resolve()/'cpu-probe':raise ValueError('Exclusive owned observation directory required')
    if (not result.get('passed') or not result.get('worker_exit_verified') or not result.get('job_cleanup',{}).get('owned_tree_exit_verified')
        or result['job_cleanup'].get('forced_job_cleanup')):raise ValueError('Clean owned worker exit required before capture copy')
    source=output/'scratch/observation.json'
    if not source.exists():return False
    if source.resolve()!=source or source.stat().st_size>262144:raise ValueError('Unexpected observation path/size')
    with source.open('rb') as src,(output/'observation.json').open('xb') as dst:dst.write(src.read())
    return True

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--scratch',type=Path)
    args=parser.parse_args()
    teardown.verify_workspace();preflight.EV=EV
    if args.worker:
        if args.scratch is None or args.scratch.resolve()!=EV.resolve()/'cpu-probe/scratch':raise ValueError('Owned exclusive CPU scratch required')
        raise SystemExit(preflight.worker(args.scratch,constructor_tests=acceptance))
    code=preflight.supervise(worker_script=Path(__file__).resolve(),probe_name='cpu-probe')
    if code==0 and not preserve_capture(EV/'cpu-probe',json.loads((EV/'cpu-probe/results.json').read_text())):raise RuntimeError('Missing completed observational record')
    raise SystemExit(code)
