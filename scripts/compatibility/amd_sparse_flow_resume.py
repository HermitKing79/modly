"""A separately authorised attempt of the unchanged reviewed sparse-flow worker."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import amd_sparse_flow_forward as previous

ORIGINAL=previous.local.EVIDENCE
EVIDENCE=previous.local.ROOT/'docs/compatibility/amd-sparse-flow-resume-evidence'
REQUIRED_RAM=11411181632
REVIEW=ORIGINAL/'independent-contract-review.json'

def cpu_admission(memory=None):
    if memory is None:
        import psutil
        memory=psutil.virtual_memory()
    if (type(memory.total) is not int or type(memory.available) is not int
        or memory.available<0 or memory.total<0 or memory.available>memory.total):
        raise ValueError('Invalid read-only host memory sample')
    return dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        host_total_bytes=memory.total,host_available_bytes=memory.available,required_margin_bytes=REQUIRED_RAM,
        admitted=memory.available>=REQUIRED_RAM,
        torch_imported=any(name=='torch' or name.startswith('torch.') for name in sys.modules),
        scope='Read-only CPU admission; original worker repeats its unchanged margin after runtime import')

def verify_reviewed_code():
    review=json.loads(REVIEW.read_text())
    if review.get('candidate_review_status')!='PASS':raise RuntimeError('Original independent review missing')
    for name,expected in review['reviewed_file_sha256'].items():
        path=previous.local.ROOT/'scripts/compatibility'/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Reviewed sparse-flow code changed: '+name)
    return True

def configure():
    verify_reviewed_code()
    if previous.local.EVIDENCE not in (ORIGINAL,EVIDENCE):raise RuntimeError('Unexpected evidence directory configuration')
    previous.local.EVIDENCE=EVIDENCE
    # PINS and PROBE retain the original verified artifact paths. Only this
    # separately authorised attempt's canonical output/scratch directory changes.

def profile():
    configure();result=previous.profile()
    result.worker_script=Path(__file__).resolve()
    result.committed_files+=['scripts/compatibility/amd_sparse_flow_resume.py',
        REVIEW.relative_to(previous.local.ROOT).as_posix(),
        (EVIDENCE/'first-admission.json').relative_to(previous.local.ROOT).as_posix()]
    return result

def main(args):
    previous.teardown.verify_workspace();configure()
    if args.worker:return previous.worker(args.output)
    admission=cpu_admission()
    if not admission['admitted'] or admission['torch_imported']:
        raise RuntimeError('Read-only CPU admission failed: '+json.dumps(admission))
    with (EVIDENCE/'pre-execution-admission.json').open('x',encoding='utf-8') as stream:
        json.dump(admission,stream,indent=2)
    # Original supervisor verifies published files, creates a new exclusive
    # marker, contains the child before preparation and gates GPU authorization.
    previous.verify_preflight()
    return previous.supervisor.supervise(args.output,args.harness_commit,profile=profile())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--harness-commit')
    raise SystemExit(main(p.parse_args()))
