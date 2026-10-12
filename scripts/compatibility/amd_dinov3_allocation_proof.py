"""One DINO pass, durable CPU features, bounded allocation diagnosis; no retries."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parent))
import amd_dinov3_conditioning as previous

EVIDENCE=previous.local.ROOT/'docs/compatibility/amd-dinov3-allocation-evidence'
PINS=EVIDENCE/'diagnostic-source-contract.json'


def verify_diagnostics():
    sources=json.loads(PINS.read_text())['sources']
    if any(previous.local.digest(Path(path))!=sha for path,sha in sources.items()):
        raise RuntimeError('Inspected allocator/safetensors implementation changed')
    return sources


def classify(record,process_ok,healthy,preserved):
    result=previous.contract.classify(record,process_ok,healthy,preserved)
    if (record.get('feature_checkpoint',{}).get('verified') is not True
            or record.get('allocation_diagnosis',{}).get('snapshot_verified') is not True
            or record['allocation_diagnosis'].get('history_disabled') is not True):return 'PARTIAL'
    return result


def profile():
    result=previous.profile();result.evidence=EVIDENCE;result.worker_script=Path(__file__).resolve()
    result.source_identity=lambda:(previous.local.verify_sources(),verify_diagnostics())
    result.classify=classify
    result.committed_files += ['scripts/compatibility/'+name+'.py' for name in
                              ('amd_dinov3_allocation_proof','amd_dinov3_allocation','amd_feature_checkpoint')]
    result.committed_files.append(PINS.relative_to(previous.local.ROOT).as_posix())
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--harness-commit')
    args=parser.parse_args();verify_diagnostics()
    if args.worker:raise SystemExit(previous.worker(args.output,allocation_evidence=EVIDENCE))
    previous.verify_preflight()
    raise SystemExit(previous.supervisor.supervise(args.output,args.harness_commit,profile=profile()))
