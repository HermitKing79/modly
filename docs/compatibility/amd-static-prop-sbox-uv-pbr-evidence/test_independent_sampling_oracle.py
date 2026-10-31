"""Synthetic tests only: no engine, project, source mesh, or private capture input."""
import importlib.util
import hashlib
from copy import deepcopy
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


MODULE = Path(__file__).with_name("independent_sampling_oracle.py")
SPEC = importlib.util.spec_from_file_location("phase22_oracle", MODULE)
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


def camera():
    return dict(position=[0., 0., 0.], forward=[1., 0., 0.],
                left=[0., 1., 0.], up=[0., 0., 1.], field_of_view=90.,
                z_near=.1, z_far=100.)


def binary_texture(size=64):
    yy, xx = np.indices((size, size))
    cells = ((xx // 4) * 3 + (yy // 4) * 5 + (xx // 4) * (yy // 4)) % 8
    rgb = np.stack([(cells & (1 << i) != 0) * 255 for i in range(3)], axis=-1)
    return np.asarray(rgb, dtype=np.uint8)


def dependency_fixture(contract):
    material = contract["material_resource"]
    paths = dict(SourceFile="C:/proof/"+material, CompiledFile="C:/proof/"+material+"_c")
    return [dict(Path=material, IsCompiled=True, IsCompiledAndUpToDate=True, IsCompileFailed=False, **paths),
            dict(Path=material, UnresolvedReferences=[], **paths),
            dict(Path=material, Relation="References", Total=2,
                 Assets=[dict(Path=contract["dependency_texture_resource"], Type="Texture"),
                         dict(Path=contract["shader_resource"], Type="Shader")])]


class ProjectionTests(unittest.TestCase):
    def test_source_basis_projection_has_independent_numeric_answer(self):
        pixels, depths = oracle.project_points(np.array([[4., 2., 1.], [4., -2., -1.]]),
                                              camera(), 200, 100)
        np.testing.assert_allclose(pixels, [[50., 25.], [150., 75.]], atol=1e-12)
        np.testing.assert_array_equal(depths, [4., 4.])

    def test_quaternion_rotation_is_not_reflected(self):
        q = [0., 0., np.sqrt(.5), np.sqrt(.5)]
        np.testing.assert_allclose(oracle.quaternion_matrix(q) @ [1., 0., 0.],
                                   [0., 1., 0.], atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(oracle.quaternion_matrix(q)), 1.)

    def test_projection_sentinels_reject_wrong_fov(self):
        points = np.array([[4., 1., 1.], [4., -1., 1.], [4., 0., -1.]])
        pixels, _ = oracle.project_points(points, camera(), 200, 100)
        correct = oracle.projection_error(points, pixels / [200, 100], camera(), 200, 100)
        self.assertLess(correct, 1e-10)
        wrong = dict(camera(), field_of_view=80.)
        self.assertGreater(oracle.projection_error(points, pixels / [200, 100], wrong, 200, 100), 1.)


class RasterTests(unittest.TestCase):
    def test_perspective_interpolation_matches_manual_ray_plane_barycentrics(self):
        # Projected vertices are (20,80),(80,80),(50,20); depths deliberately differ.
        positions = np.array([[2., 1.2, -1.2], [4., -2.4, -2.4], [8., 0., 4.8]])
        uv = np.array([[0., 0.], [1., 0.], [0., 1.]])
        result = oracle.rasterize(positions, uv, np.array([[0, 1, 2]]), camera(), 100, 100)
        x, y = 50, 50
        self.assertEqual(result["face"][y, x], 0)
        a = np.array([[20., 80.], [80., 80.], [50., 20.]])
        linear = np.linalg.solve(np.vstack([a.T, np.ones(3)]), [x + .5, y + .5, 1.])
        weights = (linear / [2., 4., 8.]); weights /= weights.sum()
        np.testing.assert_allclose(result["uv"][y, x], weights @ uv, atol=1e-12)
        self.assertGreater(np.linalg.norm(result["uv"][y, x] - linear @ uv), .1)

    def test_z_buffer_selects_front_surface_and_culls_reversed_surface(self):
        near = np.array([[2., 1., -1.], [2., -1., -1.], [2., 0., 1.]])
        far = near * 2
        positions = np.vstack([far, near])
        result = oracle.rasterize(positions, np.zeros((6, 2)), np.array([[0, 1, 2], [3, 4, 5]]),
                                  camera(), 64, 64)
        self.assertEqual(result["face"][32, 32], 1)
        reverse = oracle.rasterize(near, np.zeros((3, 2)), np.array([[2, 1, 0]]), camera(), 64, 64)
        self.assertTrue(np.all(reverse["face"] == -1))

    def test_unhandled_clip_crossing_fails_instead_of_silently_dropping_face(self):
        positions = np.array([[.01, 1., -1.], [2., -1., -1.], [2., 0., 1.]])
        with self.assertRaises(oracle.OracleError):
            oracle.rasterize(positions, np.zeros((3, 2)), np.array([[0, 1, 2]]), camera(), 64, 64)


class TextureTests(unittest.TestCase):
    def test_full_compiled_mip_hash_and_grid_witnesses_authenticate_pixels(self):
        rgba = np.zeros((8, 8, 4), np.uint8); rgba[..., 3] = 255
        rgba[:4, :4, 0] = 255; rgba[4:, 4:, 2] = 255
        contract = dict(material_resource="test.vmat", shader_resource="test.shader",
                        runtime_texture_resource_path=None, dependency_texture_resource="test.generated.vtex")
        record = dict(MaterialPath="test.vmat", ShaderResourcePath="test.shader", TextureResourcePath=None,
                      RuntimeTextureName="g_tDiagnosticColor", MaterialValid=True, MaterialError=False,
                      TextureValid=True, TextureError=False, IsLoaded=True, Width=8, Height=8, Depth=1, Mips=1,
                      ImageFormat="RGBA8888", Flags="fixture", ImageSourceSha256="source-hash",
                      MipZeroRgbaByteCount=rgba.size,
                      MipZeroRgbaSha256=hashlib.sha256(rgba.tobytes()).hexdigest(), Samples=[])
        for row in range(5):
            for col in range(5):
                x=col*7//4; y=row*7//4
                record["Samples"].append(dict(X=x, Y=y, **dict(zip("RGBA", map(int, rgba[y, x])))))
        probe = dict(Classification="READ_ONLY_PHASE22_COMPILED_SHADER_MATERIAL_TEXTURE_SNAPSHOT",
                     NoSceneOrResourceMutationRequested=True, Records=[record])
        dependencies = dependency_fixture(contract)
        result = oracle.validate_texture_probe(probe, rgba, "source-hash", contract, dependencies)
        self.assertTrue(result["decoded_rgba_identical"])
        self.assertIsNone(result["runtime_texture_resource_path"])
        self.assertEqual(result["dependency_texture_resource_path"], "test.generated.vtex")
        bad_state = deepcopy(dependencies); bad_state[0]["IsCompiledAndUpToDate"] = False
        for bad in ([], dependencies[:2], bad_state):
            with self.assertRaises(oracle.OracleError):
                oracle.validate_texture_probe(probe, rgba, "source-hash", contract, bad)
        record["TextureResourcePath"] = "test.generated.vtex"
        with self.assertRaises(oracle.OracleError):
            oracle.validate_texture_probe(probe, rgba, "source-hash", contract, dependencies)
        record["TextureResourcePath"] = None
        record["MipZeroRgbaSha256"] = "0"*64
        with self.assertRaises(oracle.OracleError):
            oracle.validate_texture_probe(probe, rgba, "source-hash", contract, dependencies)

    def test_dependency_records_reject_missing_changed_unresolved_and_failed_assets(self):
        contract = dict(material_resource="test.vmat", shader_resource="test.shader",
                        runtime_texture_resource_path=None, dependency_texture_resource="test.generated.vtex")
        original = dependency_fixture(contract)
        self.assertEqual(oracle.validate_dependency_records(original, contract)["texture_resource_path"], "test.generated.vtex")
        mutations = []
        for flag, value in (("IsCompiled", False), ("IsCompiledAndUpToDate", False), ("IsCompileFailed", True)):
            changed = deepcopy(original); changed[0][flag] = value; mutations.append(changed)
        changed = deepcopy(original); changed[1]["UnresolvedReferences"] = ["missing.png"]; mutations.append(changed)
        changed = deepcopy(original); changed[2]["Assets"][0]["Path"] = "other.generated.vtex"; mutations.append(changed)
        changed = deepcopy(original); changed[2]["Assets"][1]["Path"] = "other.shader"; mutations.append(changed)
        changed = deepcopy(original); changed[2]["Assets"] = []; mutations.append(changed)
        changed = deepcopy(original); changed[1]["CompiledFile"] = "C:/elsewhere.vmat_c"; mutations.append(changed)
        mutations.extend([[], original[:2], original+original])
        for changed in mutations:
            with self.subTest(changed=changed):
                with self.assertRaises(oracle.OracleError):
                    oracle.validate_dependency_records(changed, contract)

    def test_changed_dependency_file_hash_rejected_before_any_file_parse(self):
        with mock.patch.object(oracle, "digest", return_value="changed-dependency-hash"):
            with self.assertRaises(oracle.OracleError):
                oracle.read_dependency_evidence(Path("NEVER_OPENED.json"), "frozen-dependency-hash")

    def test_classifier_binary_palette_and_ambiguity(self):
        rgb = np.array([[[255, 0, 0], [0, 255, 255], [255, 255, 255], [0, 0, 0], [255, 128, 0]]], np.uint8)
        np.testing.assert_array_equal(oracle.classify_rgb(rgb), [[1, 6, 7, 0, -1]])

    def test_wrap_top_left_rows_and_point_lookup(self):
        classes = np.array([[1, 2], [4, 7]], np.int8)
        uv = np.array([[.1, .1], [.1, .6], [1.6, -.4]])
        np.testing.assert_array_equal(oracle.sample_classes(classes, uv), [1, 4, 7])

    def test_plateau_margin_never_admits_color_boundary(self):
        classes = np.ones((16, 16), np.int8)
        classes[:, 8:] = 2
        margin = oracle.category_margins(classes, 5)
        self.assertEqual(margin[8, 7], 0)
        self.assertEqual(margin[8, 6], 1)
        self.assertGreaterEqual(margin[8, 3], 3)

    def test_flip_stretch_and_wrong_pixels_are_identifiable_and_reject_actual_faults(self):
        rng = np.random.default_rng(220031)
        classes = rng.integers(0, 8, size=(16, 16), dtype=np.int8)
        yy, xx = np.indices((128, 128))
        uv = np.stack([(xx + .37) / 128, (yy + .21) / 128], axis=-1)
        expected = oracle.sample_classes(classes, uv)
        mask = np.ones(expected.shape, bool)
        positive = oracle.control_results(classes, uv, expected, mask, expected)
        for name, stats in positive.items():
            self.assertGreater(stats["reference_hamming_fraction"], .2, name)
            self.assertLess(stats["actual_agreement"], .8, name)
        for name in oracle.CONTROL_NAMES:
            faulty = oracle.control_expected(classes, uv, name)
            self.assertLess(np.mean(faulty == expected), .8, name)
            stats = oracle.control_results(classes, uv, faulty, mask, expected)
            self.assertAlmostEqual(stats[name]["actual_agreement"], 1., msg=name)
            self.assertFalse(oracle.controls_pass(stats, positive_accuracy=float(np.mean(faulty == expected))))


class TopologyAndGateTests(unittest.TestCase):
    def test_continuity_pairs_reject_a_wrong_color_on_either_incident_face(self):
        p = np.array([[4., 2., -2.], [4., -2., -2.], [4., -2., 2.], [4., 2., 2.]])
        uv = np.array([[0., 1.], [1., 1.], [1., 0.], [0., 0.]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        topology = oracle.derive_topology(p, uv, faces, require_closed=False)
        raster = oracle.rasterize(p, uv, faces, camera(), 128, 128)
        classes = np.ones((16, 16), np.int8)
        expected, mask, _ = oracle.reference_mask(raster, classes, oracle.category_margins(classes))
        good = oracle.continuity_witnesses(topology, faces, raster, expected, expected, mask)
        self.assertGreater(good["pairs"], 0)
        self.assertEqual(good["pairs"], good["matched_pairs"])
        faulty = expected.copy(); faulty[raster["face"] == 1] = 4
        bad = oracle.continuity_witnesses(topology, faces, raster, expected, faulty, mask)
        self.assertEqual(bad["pairs"], good["pairs"])
        self.assertEqual(bad["matched_pairs"], 0)

    def test_continuous_shared_edge_and_deliberate_uv_seam_are_distinguished(self):
        p = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        uv = p[:, :2]
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        topology = oracle.derive_topology(p, uv, faces, require_closed=False)
        self.assertEqual(len(topology["nonseamed"]), 1)
        self.assertEqual(topology["seam_count"], 0)
        pp = np.vstack([p, p[[0, 2]]])
        uu = np.vstack([uv, [[.1, 0.], [.9, 1.]]])
        ff = np.array([[0, 1, 2], [4, 5, 3]])
        seam = oracle.derive_topology(pp, uu, ff, require_closed=False)
        self.assertEqual(len(seam["nonseamed"]), 0)
        self.assertEqual(seam["seam_count"], 1)

    def test_coverage_gate_cannot_pass_perfect_but_tiny_sample(self):
        stats = dict(eligible_pixels=4999, tested_faces_3=99, accuracy=1.,
                     per_class={str(i): dict(samples=700, accuracy=1.) for i in range(8)},
                     per_face={"0": dict(samples=4999, accuracy=1.)})
        failures = oracle.view_failures(stats)
        self.assertTrue(any("5000" in x for x in failures))
        self.assertTrue(any("100" in x for x in failures))

    def test_unknown_actual_pixel_is_a_mismatch_and_never_removes_reference_sample(self):
        expected = np.ones((4, 4), np.int8)
        actual = expected.copy(); actual[0, 0] = -1
        stats = oracle.score_pixels(expected, actual, np.ones((4, 4), bool), np.zeros((4, 4), int))
        self.assertEqual(stats["eligible_pixels"], 16)
        self.assertAlmostEqual(stats["accuracy"], 15 / 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
