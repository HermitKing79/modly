"""Focused first-frame authorization tests; no Blender or GPU execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image


E = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('brick_pbr_host', E / 'render_host.py')
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


class FirstFrameTests(unittest.TestCase):
    def make(self, path):
        image = Image.new('RGB', (1024, 1024), (45, 25, 15))
        image.paste((190, 150, 90), (180, 180, 840, 840))
        target = path / 'view-pos-x.png'
        image.save(target)
        return dict(name='pos-x', PNG=target.name, device='CPU',
                    denoising_use_gpu=False, source_PLY_sha256=host.SOURCE_SHA,
                    vertex_count=1957891, face_count=4173584,
                    geometry_unchanged_after_first_view=True, exposure_EV=0,
                    material_field_sha256=host.MATERIAL_SHA,
                    material_verified_after_first_view=True,
                    sha256=host.digest(target), bytes=target.stat().st_size)

    def test_actual_first_frame_accepts_only_verified_material_and_CPU(self):
        with tempfile.TemporaryDirectory(dir=host.PRIVATE) as directory:
            path = Path(directory)
            record = self.make(path)
            self.assertTrue(host.first_image(path, record)['verified'])
            for changed in ({'exposure_EV': 1}, {'exposure_EV': 2}, {'material_field_sha256': '0' * 64},
                            {'material_verified_after_first_view': False},
                            {'geometry_unchanged_after_first_view': False},
                            {'device': 'GPU'}, {'denoising_use_gpu': True}):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    host.first_image(path, dict(record, **changed))

    def test_modified_PNG_fails_before_continuation(self):
        with tempfile.TemporaryDirectory(dir=host.PRIVATE) as directory:
            path = Path(directory)
            record = self.make(path)
            with (path / record['PNG']).open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaises(ValueError):
                host.first_image(path, record)

    def test_original_resource_limits_are_retained(self):
        self.assertEqual((host.CAP, host.DEADLINE), (6 * 1024**3, 180))
        self.assertEqual((host.RAM_FLOOR, host.DISK_FLOOR), (12 * 1024**3, 2 * 1024**3))
        self.assertFalse(host.admission(host.RAM_FLOOR - 1, host.DISK_FLOOR)['admitted'])
        self.assertFalse(host.admission(host.RAM_FLOOR, host.DISK_FLOOR - 1)['admitted'])
        self.assertTrue(host.admission(host.RAM_FLOOR, host.DISK_FLOOR)['admitted'])

    def test_first_frame_gate_matches_the_real_renderer(self):
        source = (E / 'render_host.py').read_text()
        self.assertIn("b'BRICK_PBR_READY_GO_REMAINING'", source)
        self.assertNotIn('BRICK_VISUAL_READY', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
