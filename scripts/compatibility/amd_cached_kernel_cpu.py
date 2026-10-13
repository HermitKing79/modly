"""Dedicated CPU-only entry point required by the existing owned supervisor."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import amd_cached_kernel_convolution as proof
p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--scratch',type=Path);a=p.parse_args()
proof.teardown.verify_workspace();proof.p.EV=proof.EV
if not a.worker or a.scratch is None or a.scratch.resolve()!=proof.EV.resolve()/'cpu-acceptance/scratch':raise RuntimeError('Owned CPU worker only')
raise SystemExit(proof.p.worker(a.scratch,constructor_tests=proof.cpu_acceptance))
