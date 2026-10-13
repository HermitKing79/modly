"""Corrected hook contract: owned CPU constructor proof, then one conditional run."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'api'),str(Path(__file__).parent)]
import amd_native_convolution_preflight as preflight
import amd_native_convolution_smoke as smoke
import amd_blaslt_teardown as teardown
EV=ROOT/'docs/compatibility/amd-native-convolution-hook-correction-evidence'
CPU_ACCEPTANCE='cpu-acceptance-reviewed'

def constructor_tests(torch,meta,source,scratch):
    from test_amd_native_convolution_hooks import run_constructor_suite
    private=preflight.private_cache(scratch)
    return run_constructor_suite(torch,private,meta,source)

def verify_cpu_acceptance():
    row=json.loads((EV/CPU_ACCEPTANCE/'results.json').read_text())
    result=row.get('worker',{}).get('constructor_acceptance',{})
    if (row.get('passed') is not True or result.get('passed') is not True or result.get('tests_run')!=13
        or result.get('complete_constructor_passed') is not True or result.get('gpu_initialized') is not False
        or result.get('callbacks_executed')!=0 or result.get('errors')!=0 or result.get('failures')!=0):
        raise RuntimeError('Complete installed production constructor acceptance required')

def profile():
    result=smoke.profile();result.worker_script=Path(__file__).resolve()
    result.committed_files += ['scripts/compatibility/amd_native_convolution_hook_correction.py','scripts/compatibility/test_amd_native_convolution_hooks.py']
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu-tests',action='store_true');p.add_argument('--cpu-name',choices=('cpu-red','cpu-red-isolated','cpu-probe','cpu-acceptance','cpu-acceptance-reviewed'),default=CPU_ACCEPTANCE)
    p.add_argument('--worker',action='store_true');p.add_argument('--scratch',type=Path);p.add_argument('--output',type=Path);p.add_argument('--harness-commit');a=p.parse_args()
    teardown.verify_workspace();preflight.EV=EV;smoke.EV=EV;smoke.CPU_PROBE=CPU_ACCEPTANCE
    if a.worker and a.scratch is not None:
        if a.scratch.resolve() not in tuple(EV.resolve()/name/'scratch' for name in ('cpu-red','cpu-red-isolated','cpu-probe','cpu-acceptance','cpu-acceptance-reviewed')):raise ValueError('Exclusive CPU scratch required')
        raise SystemExit(preflight.worker(a.scratch,constructor_tests=constructor_tests))
    if a.cpu_tests:raise SystemExit(preflight.supervise(worker_script=Path(__file__).resolve(),probe_name=a.cpu_name))
    verify_cpu_acceptance()
    if a.worker:raise SystemExit(smoke.worker(a.output))
    smoke.identity()
    import psutil
    if psutil.virtual_memory().available<smoke.contract.HOST_RAM:raise ValueError('Host RAM admission failed before attempt marker')
    raise SystemExit(smoke.supervisor.supervise(a.output,a.harness_commit,profile=profile()))
