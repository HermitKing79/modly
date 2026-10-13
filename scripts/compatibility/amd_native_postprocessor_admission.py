"""Exclusive CPU acceptance of the fixed native-postprocessor contracts."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'api'),str(Path(__file__).parent)]
import amd_native_convolution_preflight as preflight
import amd_blaslt_teardown as teardown
EV=ROOT/'docs/compatibility/amd-native-postprocessor-admission-evidence'

def acceptance(torch,meta,source,scratch):
    from test_amd_native_convolution_hooks import run_constructor_suite
    from test_amd_native_scalar_dispatch import run_scalar_suite
    from test_amd_native_postprocessors import run_postprocessor_suite
    private=preflight.private_cache(scratch)
    constructor=run_constructor_suite(torch,private,meta,source)
    scalar=run_scalar_suite(torch,private,meta,source)
    processors=run_postprocessor_suite(torch,private,meta,source,red=scratch.parent.name=='cpu-red')
    return dict(passed=constructor['passed'] and scalar['passed'] and processors['passed'],constructor=constructor,scalar=scalar,postprocessors=processors,
        gpu_initialized=torch.cuda.is_initialized(),native_gpu_executed=False,weights_loaded=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--scratch',type=Path)
    parser.add_argument('--name',choices=('cpu-red','cpu-acceptance','cpu-acceptance-reviewed','cpu-final','cpu-probe'),default='cpu-probe');a=parser.parse_args()
    teardown.verify_workspace();preflight.EV=EV
    if a.worker:
        if a.scratch is None or a.scratch.resolve() not in tuple(EV.resolve()/name/'scratch' for name in ('cpu-red','cpu-acceptance','cpu-acceptance-reviewed','cpu-final','cpu-probe')):raise ValueError('Owned exclusive CPU scratch required')
        raise SystemExit(preflight.worker(a.scratch,constructor_tests=acceptance))
    raise SystemExit(preflight.supervise(worker_script=Path(__file__).resolve(),probe_name=a.name))
