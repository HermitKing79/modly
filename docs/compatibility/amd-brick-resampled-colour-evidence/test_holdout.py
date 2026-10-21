"""Synthetic tests only; no saved geometry or material data are opened."""
import sys
sys.dont_write_bytecode = True
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SOURCE = Path(__file__).with_name('holdout.py')
holdout = None
if SOURCE.is_file():
    spec = importlib.util.spec_from_file_location('tested_position_holdout', SOURCE)
    holdout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(holdout)


def fixture():
    vertices = []
    for i in range(80):
        vertices.extend([[0.,3.*i,0.], [0.,3.*i+1.,0.], [0.,3.*i,1.]])
    vertices = np.asarray(vertices, np.float32)
    faces = np.arange(240, dtype=np.int32).reshape(80,3)
    regions = {f'region_{i:02}':np.arange(80, dtype=np.int32) for i in range(16)}
    return vertices, faces, regions, np.empty((0,3), np.float32)


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(holdout, 'Frozen position-holdout implementation required')

    def test_fixed_regions_original_ids_strict_interiors_and_determinism(self):
        args = fixture()
        before = [args[0].copy(), args[1].copy(), {k:v.copy() for k,v in args[2].items()}]
        result = holdout.build(*args)
        again = holdout.build(*args)
        self.assertEqual(result['query_face'].shape, (2048,))
        self.assertEqual(result['bary'].shape, (2048,3))
        self.assertEqual(result['world'].dtype, np.float32)
        self.assertEqual(result['query_face'].dtype, np.int32)
        self.assertEqual(result['bary'].dtype, np.float64)
        self.assertEqual(result['kinds'].dtype, np.int8)
        np.testing.assert_array_equal(result['query_face'], again['query_face'])
        np.testing.assert_array_equal(result['bary'], again['bary'])
        np.testing.assert_array_equal(result['world'], again['world'])
        np.testing.assert_array_equal(result['kinds'], np.tile([4,5],1024))
        self.assertTrue(np.all((result['bary'] > 0) & (result['bary'] < 1)))
        np.testing.assert_allclose(result['bary'].sum(axis=1), 1, atol=2e-15)
        for index, (name, row) in enumerate(result['cohorts'].items()):
            self.assertEqual((row['query_start'],row['query_stop']), (index*128,(index+1)*128))
            ids = result['query_face'][row['query_start']:row['query_stop']].reshape(64,2)
            np.testing.assert_array_equal(ids[:,0], ids[:,1])
            self.assertEqual(len(np.unique(ids)),64)
            self.assertTrue(np.all(ids[1:,0] > ids[:-1,0]))
            self.assertTrue(np.isin(ids,args[2][name]).all())
            self.assertEqual(row['pool_count'],80)
        expected = np.einsum('ni,nij->nj',result['bary'],args[0][args[1][result['query_face']]].astype(np.float64)).astype(np.float32)
        np.testing.assert_array_equal(result['world'], expected)
        np.testing.assert_array_equal(args[0], before[0]); np.testing.assert_array_equal(args[1], before[1])
        for name in args[2]:
            np.testing.assert_array_equal(args[2][name],before[2][name])
        self.assertIn('previously fixed faces',result['draw_rule']['scope'])

    def test_area_weighting_includes_dominant_area_face(self):
        v,f,r,old = fixture()
        # The last face has a million times the area of every other face.
        v[f[79]] = [[0,1000,0],[0,2000,0],[0,1000,1000]]
        result = holdout.build(v,f,r,old)
        for row in result['cohorts'].values():
            self.assertIn(79,result['query_face'][row['query_start']:row['query_stop']])

    def test_rejects_collision_with_prior_numeric_signed_zero_position(self):
        args = fixture(); first = holdout.build(*args)
        old = first['world'][:1].copy(); old[:,0] = np.float32(-0.)
        self.assertTrue(np.signbit(old[0,0]))
        with self.assertRaisesRegex(ValueError,'collision'):
            holdout.build(*args[:3],old)

    def test_rejects_collisions_after_final_f32_rounding_without_redraw(self):
        step = np.nextafter(np.float32(1),np.float32(2))
        v = np.array([[1,1,1],[step,1,1],[1,step,1]],np.float32)
        f = np.tile(np.array([[0,1,2]],np.int32),(64,1))
        regions = {f'region_{i:02}':np.arange(64,dtype=np.int32) for i in range(16)}
        with self.assertRaisesRegex(ValueError,'collision'):
            holdout.build(v,f,regions,np.empty((0,3),np.float32))

    def test_rejects_bad_types_dimensions_bounds_and_underfilled_regions(self):
        v,f,r,old = fixture()
        cases = [(v.astype(np.float64),f,r,old),(v,f.astype(np.int64),r,old),
                 (v,f,r,np.empty((0,2),np.float32)),(v,f,r,old.astype(np.float64))]
        bad = f.copy(); bad[0,0] = -1; cases.append((v,bad,r,old))
        bad = f.copy(); bad[0,0] = len(v); cases.append((v,bad,r,old))
        bad = v.copy(); bad[0,0] = np.inf; cases.append((bad,f,r,old))
        bad = r.copy(); bad['region_00'] = np.arange(63,dtype=np.int32); cases.append((v,f,bad,old))
        bad = r.copy(); bad['region_00'] = np.arange(80,dtype=np.int64); cases.append((v,f,bad,old))
        bad = r.copy(); bad['region_00'] = np.full(80,0,dtype=np.int32); cases.append((v,f,bad,old))
        bad = r.copy(); bad['region_00'] = np.arange(1,81,dtype=np.int32); cases.append((v,f,bad,old))
        bad = r.copy(); del bad['region_00']; cases.append((v,f,bad,old))
        for index,args in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    holdout.build(*args)

    def test_rejects_degenerate_pool_face_even_if_it_would_not_be_selected(self):
        v,f,r,old = fixture(); f[0,1] = f[0,0]
        with self.assertRaisesRegex(ValueError,'area'):
            holdout.build(v,f,r,old)

    def test_region_dictionary_order_cannot_change_frozen_draws(self):
        v,f,r,old = fixture()
        result = holdout.build(v,f,r,old)
        changed = holdout.build(v,f,dict(reversed(list(r.items()))),old)
        np.testing.assert_array_equal(result['world'],changed['world'])
        self.assertEqual(list(result['cohorts']),list(changed['cohorts']))


if __name__ == '__main__':
    unittest.main()
