"""Actual A3 fallback must not mistake graph closure for safe geometry."""
import unittest
from qualify import geometry_prerequisites

class FallbackTests(unittest.TestCase):
    def test_available_geometry_failures_do_not_suppress_secondary_candidate(self):
        valid = dict(supported=True, svd_rank=2, planar_within_tolerance=True,
                     zero_length_segment_count=0, projected_self_intersection=False)
        self.assertTrue(geometry_prerequisites(True, 0, valid))
        for field,value in [('supported', False), ('svd_rank', 1),
                            ('planar_within_tolerance', False),
                            ('zero_length_segment_count', 1),
                            ('projected_self_intersection', True)]:
            with self.subTest(field=field):
                self.assertFalse(geometry_prerequisites(True, 0, {**valid,field:value}))
        self.assertFalse(geometry_prerequisites(False, 0, valid))
        self.assertFalse(geometry_prerequisites(True, 1, valid))
        self.assertFalse(geometry_prerequisites(True, 0, None))
        self.assertFalse(geometry_prerequisites(True, 0, {}))

if __name__ == '__main__':
    unittest.main()
