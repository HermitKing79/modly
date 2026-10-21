"""Phase-one-only adapter of the unchanged qualified owned CPU supervisor."""
import sys
sys.dont_write_bytecode=True
import argparse
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from inputs import ROOT,P,HOST,pinned,authenticate,plain


def make_host():
    host=pinned('full_colour_original_CPU_host',HOST,'ce49d185ec23ca89a034eb6dda41d654cf97ca508501736c76029cf28397c475')
    host.E=E;host.P=P;host.ROOT=ROOT;host.BASE='1012a6c04976085c5cf8d4c6911d9010d0e8cc59';host.BRANCH='codex/amd-brick-full-colour-glb'
    host.FOLDERS={'export':'suitability'};host.authenticate=authenticate
    def review_path(path):
        if Path(path)==E/'independent-launch-correction-review.json':path=E/'independent-candidate-review.json'
        return plain(path)
    host.plain=review_path
    return host


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    # Fixed Python-only stage; no exposed render/export continuation.
    raise SystemExit(make_host().main('export',args.preflight))
