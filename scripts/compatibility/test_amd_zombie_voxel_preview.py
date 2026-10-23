"""Small synthetic CPU fixtures for exact exposed-voxel diagnostic previews."""
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

TEST_ROOT = Path(__file__).resolve().parents[3] / 'artifacts/amd-zombie-occupancy-c570c2e/voxel-preview-test-scratch'

class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'trellis2', 'transformers',
                'flex_gemm', 'cumesh', 'nvdiffrast', 'rembg', 'onnxruntime'}:
            raise AssertionError('Voxel preview imported a model/GPU package: ' + fullname)


sys.meta_path.insert(0, NoModels())
import amd_sparse_occupancy_data as original
try:
    import amd_zombie_voxel_preview as subject
except ModuleNotFoundError as error:
    if error.name != 'amd_zombie_voxel_preview':
        raise
    subject = None


class VoxelPreviewTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(subject, 'The bounded voxel preview helper is missing')
        TEST_ROOT.mkdir(exist_ok=True)

    def test_invalid_and_empty_masks_reject_before_creating_files(self):
        invalid = [np.zeros((64, 64), dtype=bool), np.zeros((64, 64, 64), dtype='f4'),
                   np.full((64, 64, 64), np.nan, dtype='f4'),
                   np.zeros((64, 64, 64), dtype=bool)[:, :, ::-1], None]
        for mask in invalid:
            with self.subTest(type=type(mask).__name__), tempfile.TemporaryDirectory(dir=TEST_ROOT) as tmp:
                with self.assertRaises(ValueError):
                    subject.produce(tmp, mask)
                self.assertEqual(list(Path(tmp).iterdir()), [])
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as tmp:
            empty = np.zeros((64, 64, 64), dtype=bool)
            self.assertEqual(subject.diagnostics(empty)['occupied_count'], 0)
            self.assertIsNone(subject.diagnostics(empty)['bounding_box'])
            with self.assertRaisesRegex(ValueError, 'Empty'):
                subject.produce(tmp, empty)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_export_preserves_original_surface_winding_mask_and_first_preview(self):
        mask = np.zeros((64, 64, 64), dtype=bool)
        mask[2, 3, 4:6] = True
        before = mask.tobytes()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as tmp, tempfile.TemporaryDirectory(dir=TEST_ROOT) as control:
            expected = original.write_preview(control, mask)
            row = subject.produce(tmp, mask)
            self.assertEqual((row['original']['vertices'], row['original']['faces']), (12, 10))
            self.assertEqual((row['vertices'], row['faces']), (12, 10))
            self.assertEqual(row['original']['obj_sha256'], expected['obj_sha256'])
            self.assertEqual((Path(tmp) / 'voxel-preview.obj').read_bytes(),
                             (Path(control) / 'voxel-preview.obj').read_bytes())
            self.assertEqual((Path(tmp) / 'voxel-preview.png').read_bytes(),
                             (Path(control) / 'voxel-preview.png').read_bytes())
            self.assertEqual(mask.tobytes(), before)
            # Winding stays exactly outward in original x/y/z = D/H/W coordinates.
            vertices, faces = original.surface(mask)
            for face in faces:
                a, b, c = [np.asarray(vertices[index - 1]) for index in face[:3]]
                normal = np.cross(b - a, c - a)
                midpoint = np.asarray([vertices[index - 1] for index in face]).mean(axis=0)
                self.assertGreater(float(np.dot(normal, midpoint - [2.5, 3.5, 5])), 0)

    def test_three_views_read_back_with_actual_axes_and_camera_metadata(self):
        mask = np.zeros((64, 64, 64), dtype=bool)
        mask[1:4, 5:7, 8:12] = True
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as tmp:
            row = subject.produce(tmp, mask)
            self.assertEqual(set(row['views']), {'isometric-z-up', 'plus-x', 'plus-y'})
            for name, view in row['views'].items():
                with self.subTest(view=name), Image.open(view['path']) as image:
                    image.load()
                    self.assertEqual((image.mode, image.size), ('RGB', (900, 900)))
                    self.assertGreater(len(image.getcolors(900 * 900)), 2)
                    self.assertEqual(image.info['coordinate_order'], 'x=D,y=H,z=W')
                    self.assertIn('D', image.info['axis_labels'])
                    self.assertIn('H', image.info['axis_labels'])
                    self.assertIn('W', image.info['axis_labels'])
                    self.assertEqual(json.loads(image.info['camera']), view['camera'])
                self.assertLessEqual(Path(view['path']).stat().st_size, 2 * 1024**2)
                self.assertEqual(hashlib.sha256(Path(view['path']).read_bytes()).hexdigest(), view['sha256'])
                camera = view['camera']
                self.assertGreater(camera['screen_up'][2], 0)
                self.assertTrue(np.allclose(np.cross(camera['screen_right'], camera['screen_up']),
                                            camera['toward_camera']))
            self.assertEqual(row['views']['plus-x']['camera']['toward_camera'], [1, 0, 0])
            self.assertEqual(row['views']['plus-y']['camera']['toward_camera'], [0, 1, 0])
            self.assertEqual(row['coordinate_order'], ['D', 'H', 'W'])
            self.assertTrue(row['diagnostic_only'])
            self.assertNotIn('classification', row)

    def test_diagnostics_reports_actual_components_bbox_boundaries_and_slices(self):
        mask = np.zeros((64, 64, 64), dtype=bool)
        mask[0, 0, 0] = mask[2, 3, 4] = mask[2, 3, 5] = True
        before = mask.tobytes()
        row = subject.diagnostics(mask)
        self.assertEqual(row['occupied_count'], 3)
        self.assertEqual(row['components']['count'], 2)
        self.assertEqual(row['components']['largest_sizes'], [2, 1])
        self.assertEqual(row['bounding_box'], {'minimum': [0, 0, 0], 'maximum': [2, 3, 5], 'spans': [3, 4, 6]})
        self.assertEqual(row['volume_boundary_counts'], {'D-': 1, 'D+': 0, 'H-': 1, 'H+': 0, 'W-': 1, 'W+': 0})
        self.assertEqual(row['slice_counts']['D'][0:3], [1, 0, 2])
        self.assertEqual(row['slice_counts']['H'][3], 2)
        self.assertEqual(row['slice_counts']['W'][4:6], [1, 1])
        self.assertTrue(all(len(counts) == 64 and sum(counts) == 3 for counts in row['slice_counts'].values()))
        self.assertEqual(mask.tobytes(), before)
        self.assertNotIn('classification', row)

    def test_original_face_limit_deadline_and_exclusive_output_stop_extra_views(self):
        checker = np.indices((64, 64, 64)).sum(axis=0) % 2 == 0
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as tmp:
            with self.assertRaisesRegex(ValueError, 'face limit'):
                subject.produce(tmp, checker)
            self.assertEqual(list(Path(tmp).iterdir()), [])
        mask = np.zeros((64, 64, 64), dtype=bool)
        mask[1, 2, 3] = True
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as tmp:
            with patch.object(subject.time, 'monotonic', side_effect=[0, 61]):
                with self.assertRaises(TimeoutError):
                    subject.produce(tmp, mask)
            self.assertEqual({path.name for path in Path(tmp).iterdir()}, {'voxel-preview.obj', 'voxel-preview.png'})
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as tmp:
            existing = Path(tmp) / 'voxel-preview.obj'
            existing.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                subject.produce(tmp, mask)
            self.assertEqual(existing.read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main()
