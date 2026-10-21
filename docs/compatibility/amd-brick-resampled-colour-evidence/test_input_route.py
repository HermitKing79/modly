"""Actual unchanged CPU supervisor adapter, without launching a worker."""
import sys
sys.dont_write_bytecode=True
import unittest
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from run_host import make_host
class RouteTests(unittest.TestCase):
    def test_actual_adapter_has_one_cpu_stage_and_original_limits(self):
        h=make_host();self.assertEqual(h.BASE,'ad619e54b80aef70d832b488adab41ddf69779d8');self.assertEqual(h.BRANCH,'codex/amd-brick-resampled-colour')
        self.assertEqual(h.FOLDERS,{'export':'feasibility'});self.assertEqual(h.CAP,6*1024**3);self.assertEqual(h.DEADLINE,180)
        self.assertEqual(h.E,E);self.assertEqual(h.ROOT,E.parents[2]);self.assertEqual(h.plain(E/'independent-launch-correction-review.json') if (E/'independent-candidate-review.json').exists() else h.plain(E/'run_host.py'),(E/'independent-candidate-review.json').resolve() if (E/'independent-candidate-review.json').exists() else (E/'run_host.py').resolve())
if __name__=='__main__':unittest.main()
