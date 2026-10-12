"""One reviewed DINO pass with synchronized, single-call BLAS workspace teardown."""
import argparse
import inspect
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parent))
import amd_dinov3_conditioning as previous

EVIDENCE=previous.local.ROOT/'docs/compatibility/amd-blaslt-teardown-evidence'
PINS=EVIDENCE/'binding-contract.json'

def verify_workspace():
    value=os.environ.get('WORKSPACE_DIR')
    if not value or Path(value).resolve()!=previous.local.ROOT.resolve():
        raise RuntimeError('Set WORKSPACE_DIR to the proof checkout before consuming the exclusive attempt')

def verify_binding_evidence():
    pins=json.loads(PINS.read_text())
    review=EVIDENCE/'binary-review.json'
    if (pins.get('verified_static_binding') is not True
            or previous.local.digest(review)!=pins['binary_review_sha256']
            or json.loads(review.read_text()).get('verified_static_binding') is not True
            or any(previous.local.digest(Path(path))!=sha for path,sha in pins['sources'].items())):
        raise RuntimeError('Reviewed native binding/source identity changed')
    return pins

def verify_runtime_binding(torch):
    if torch.cuda.is_initialized():raise RuntimeError('Binding validation must precede GPU initialization')
    binding=getattr(torch._C,'_cuda_clearCublasWorkspaces',None)
    if (not inspect.isbuiltin(binding) or getattr(binding,'__module__',None)!='torch._C'
            or getattr(binding,'__name__',None)!='_cuda_clearCublasWorkspaces'):
        raise RuntimeError('Expected unmodified native no-argument workspace binding absent')
    pins=verify_binding_evidence()
    path=Path(torch._C.__file__).resolve()
    if (str(path) not in pins['sources'] or str(torch.__version__)!='2.13.0+rocm10.0.0'
            or torch.version.hip!='7.15.26333' or torch.version.cuda is not None):
        raise RuntimeError('Workspace binding runtime differs from reviewed AMD build')
    return dict(verified=True,binding=binding.__name__,module=binding.__module__,extension_path=str(path),
                gpu_initialized=False,native_call_executed=False,binary_review_sha256=pins['binary_review_sha256'])

def classify(record,process_ok,healthy,preserved):
    original=previous.contract.classify(record,process_ok,healthy,preserved)
    if (type(record.get('workspace_clear_calls')) is not int or record['workspace_clear_calls']!=1
            or record.get('workspace_binding',{}).get('verified') is not True
            or record.get('allocation_diagnosis',{}).get('history_disabled') is not True):return 'PARTIAL'
    return original

def profile():
    result=previous.profile();result.evidence=EVIDENCE;result.worker_script=Path(__file__).resolve()
    result.source_identity=lambda:(previous.local.verify_sources(),verify_binding_evidence())
    result.classify=classify
    result.committed_files += ['scripts/compatibility/'+name+'.py' for name in ('amd_blaslt_teardown','amd_dinov3_allocation')]
    result.committed_files += [(EVIDENCE/name).relative_to(previous.local.ROOT).as_posix() for name in
                              ('binding-contract.json','binary-review.json','cpu-binding.json')]
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--harness-commit')
    args=parser.parse_args();verify_workspace();verify_binding_evidence()
    if args.worker:raise SystemExit(previous.worker(args.output,allocation_evidence=EVIDENCE,workspace_teardown=True))
    probe=json.loads((EVIDENCE/'cpu-binding.json').read_text())
    if probe.get('verified') is not True or probe.get('gpu_initialized') is not False or probe.get('native_call_executed') is not False:
        raise RuntimeError('Successful CPU-only native exposure check required')
    previous.verify_preflight()
    raise SystemExit(previous.supervisor.supervise(args.output,args.harness_commit,profile=profile()))
