"""Only two CPU stages, using the unchanged qualified owned supervisor."""
import sys
sys.dont_write_bytecode=True
import argparse
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from inputs import ROOT,P,HOST,pinned,authenticate,plain

def make_host():
    h=pinned('full_sampled_original_CPU_host',HOST,'ce49d185ec23ca89a034eb6dda41d654cf97ca508501736c76029cf28397c475')
    h.E=E;h.P=P;h.ROOT=ROOT;h.BASE='3db265ca500cfe46909d12f52c151924763cd2f6';h.BRANCH='codex/amd-brick-full-sampled-glb'
    h.FOLDERS=dict(export='export',verify='verify');h.authenticate=authenticate
    def review_path(path):
        if Path(path)==E/'independent-launch-correction-review.json':path=E/'independent-candidate-review.json'
        return plain(path)
    h.plain=review_path;return h

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('export','verify'));ap.add_argument('--preflight',action='store_true');a=ap.parse_args();raise SystemExit(make_host().main(a.stage,a.preflight))
