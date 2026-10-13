"""CPU-only policy tests: ordered source-faithful tickets, finite bounded results."""
import unittest
try:
    import amd_sparse_sampling_contract as policy
except ModuleNotFoundError:
    policy=None

class SamplingContractTests(unittest.TestCase):
    def test_exact_22_tickets_and_23rd_rejection(self):
        self.assertIsNotNone(policy,'bounded sampling policy is not implemented')
        budget=policy.Budget()
        times=[1,55/56,25/26,15/16,10/11,7/8,5/6,25/32,5/7,5/8,1/2,5/16,0]
        roles=[]
        for step in range(12):
            budget.begin_step(step,times[step],times[step+1])
            for role in (['positive','negative'] if step<10 else ['positive']):
                ticket=budget.next_ticket();self.assertEqual(ticket['role'],role)
                self.assertAlmostEqual(ticket['t'],times[step],places=14)
                budget.begin_forward(ticket);budget.end_forward(ticket,frequency=1);roles.append(role)
            budget.end_step()
        self.assertEqual(roles,['positive','negative']*10+['positive']*2)
        budget.require_complete();self.assertEqual(budget.calls,22)
        with self.assertRaises(RuntimeError):budget.next_ticket()
    def test_bad_order_time_extra_calls_and_frequency_fail_closed(self):
        self.assertIsNotNone(policy,'bounded sampling policy is not implemented')
        for args in [(1,1,55/56),(0,.5,55/56),(0,1,.5)]:
            b=policy.Budget()
            with self.assertRaises(RuntimeError):b.begin_step(*args)
        b=policy.Budget();b.begin_step(0,1,55/56);ticket=b.next_ticket()
        with self.assertRaises(RuntimeError):b.begin_forward({**ticket,'role':'negative'})
        b.begin_forward(ticket)
        with self.assertRaises(RuntimeError):b.begin_forward(ticket)
        with self.assertRaises(RuntimeError):b.end_forward(ticket,frequency=0)
    def test_sampler_only_import_allowlist(self):
        self.assertIsNotNone(policy,'bounded sampling policy is not implemented')
        for name in ('trellis2.pipelines.samplers','trellis2.pipelines.samplers.flow_euler'):
            self.assertFalse(policy.prohibited_import(name))
        for name in ('trellis2.pipelines.trellis2_image_to_3d','trellis2.pipelines.rembg','cumesh','o_voxel','flex_gemm','nvdiffrast'):
            self.assertTrue(policy.prohibited_import(name))

if __name__=='__main__':unittest.main()
