"""CPU-only exact canonical schedule and admission-count regression."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_shape_sampling_contract as c
class ScheduleTests(unittest.TestCase):
    def test_budget_rejects_nonfinite_and_foreign_steps(self):
        for pair in ((float('nan'),float('nan')),(1,float('inf')),(.9,.8),(True,.99)):
            with self.assertRaises(RuntimeError):c.Budget().begin_step(*pair)
    def test_exact_source_schedule_has_21_invocations(self):
        tickets=c.tickets()
        self.assertEqual(len(tickets),21)
        self.assertEqual(sum(r['role']=='positive' for r in tickets),12)
        self.assertEqual(sum(r['role']=='negative' for r in tickets),9)
        self.assertEqual(c.times()[8],.6);self.assertEqual(c.times()[9],.5)
        self.assertEqual([r['step'] for r in tickets if r['role']=='negative'],list(range(1,10)))
    def test_fail_closed_if_required_count_differs(self):
        c.require_count(21,12,9)
        with self.assertRaises(ValueError):c.require_count(22,12,10)
    def test_previous_sparse_parameters_cannot_be_substituted(self):
        self.assertEqual(c.PARAMS,dict(steps=12,rescale_t=3.0,guidance_strength=7.5,guidance_rescale=.5,guidance_interval=(.6,1.0),dino_lock=0.0))
        self.assertEqual(c.SIGMA_MIN,.00001)
if __name__=='__main__':unittest.main()
