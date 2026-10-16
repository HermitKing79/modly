"""Focused CPU-only top-level preparation and fixed resource contracts."""
import unittest
from unittest.mock import patch

import amd_generation_texture_decode as candidate


class PreparationContractTests(unittest.TestCase):
    def test_live_limits_are_the_original_limits(self):
        self.assertEqual(candidate.LIMITS, dict(allocator_bytes=12*1024**3,
            initial_free_vram_bytes=13*1024**3, device_reserve_bytes=2*1024**3,
            opaque_headroom_bytes=1024**3, live_windows_ram_bytes=12*1024**3,
            prelaunch_windows_ram_bytes=37*1024**3//2, D_free_bytes=50*1024**3,
            tile_rows=32768, native_seconds=60, cpu_seconds=120,
            gpu_seconds=240, host_seconds=480))

    def test_resource_refusal_keeps_both_actual_values(self):
        measured=dict(available_windows_ram_bytes=12*1024**3-1, D_free_bytes=50*1024**3-3)
        with patch.object(candidate, 'read_resources', return_value=measured):
            with self.assertRaises(candidate.ResourceAdmission) as caught:
                candidate.check_resources()
        self.assertEqual(caught.exception.record['measured'], measured)
        self.assertEqual(caught.exception.record['failed'], ['available_windows_ram', 'D_free_space'])

    def test_exact_resource_floor_passes_without_gpu_query(self):
        measured=dict(available_windows_ram_bytes=12*1024**3, D_free_bytes=50*1024**3)
        with patch.object(candidate, 'read_resources', return_value=measured):
            self.assertTrue(candidate.check_resources()['admitted'])

    def test_cli_has_no_execution_or_worker_route(self):
        parser=candidate.parser()
        for option in ('--execute', '--worker', '--gpu'):
            with self.assertRaises(SystemExit):
                parser.parse_args(['--profile','brick-1957891-v1','--output','unused.json',option])


if __name__=='__main__':
    unittest.main(verbosity=2)
