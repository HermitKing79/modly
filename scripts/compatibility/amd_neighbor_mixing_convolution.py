"""One approved three-tap launch using the proven cached-leaf worker unchanged."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import amd_cached_kernel_convolution as base
import amd_neighbor_mixing_oracle as numerical
EV=base.ROOT/'docs/compatibility/amd-cached-neighbour-mixing-evidence'

def configure():
    base.EV=EV;base.NUMERICAL=numerical

def cpu_acceptance(torch,meta,source,scratch):
    result=base.cpu_acceptance(torch,meta,source,scratch)
    # The original preflight first verifies the unchanged base fixture. Record
    # the actual three-tap weight identity as well, avoiding an ambiguous log.
    meta['fixture']['weights']={**result['metadata']['weight'],'shape':[1024,3,3,3,1024]}
    meta['numerical_case']=numerical.CASE
    return result

def profile():
    result=base.profile();result.worker_script=Path(__file__).resolve()
    result.committed_files += ['scripts/compatibility/'+n+'.py' for n in ('amd_neighbor_mixing_convolution','amd_neighbor_mixing_oracle','test_amd_neighbor_mixing')]
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--cpu',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--scratch',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--harness-commit');a=p.parse_args();configure();base.teardown.verify_workspace()
    if a.worker and a.scratch is not None:
        if a.cpu or a.output is not None or a.scratch.resolve()!=EV.resolve()/'cpu-acceptance/scratch':raise RuntimeError('Exclusive CPU scratch required')
        return base.p.worker(a.scratch,constructor_tests=cpu_acceptance)
    if a.cpu:
        if a.worker:raise RuntimeError('Invalid CPU launch')
        base.p.EV=EV
        return base.p.supervise(worker_script=Path(__file__).resolve(),probe_name='cpu-acceptance')
    if a.worker:return base.worker(a.output)
    base.identity()
    return base.supervisor.supervise(a.output,a.harness_commit,profile=profile())

if __name__=='__main__':raise SystemExit(main())
