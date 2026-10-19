"""Tiny synthetic CPU fixtures only. Never import bpy or original artifacts."""
from io import BytesIO
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

import numpy as np
from PIL import Image

import render_patch as renderer


def png_fixture(pixels):
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return stream.getvalue()


def filtered_fixture(pixels, filters):
    """Forward PNG filtering, independently computed from original scanlines."""
    height, width, channels = pixels.shape
    raw = bytearray()
    for y in range(height):
        mode = filters[y]
        raw.append(mode)
        row = pixels[y].reshape(-1)
        previous = pixels[y - 1].reshape(-1) if y else np.zeros_like(row)
        for x, original in enumerate(row):
            a = int(row[x - channels]) if x >= channels else 0
            b = int(previous[x])
            c = int(previous[x - channels]) if x >= channels else 0
            if mode == 0:
                prediction = 0
            elif mode == 1:
                prediction = a
            elif mode == 2:
                prediction = b
            elif mode == 3:
                prediction = (a + b) // 2
            else:
                distances = [(abs(b - c), a), (abs(a - c), b), (abs(a + b - 2 * c), c)]
                prediction = min(enumerate(distances), key=lambda item: (item[1][0], item[0]))[1][1]
            raw.append((int(original) - prediction) % 256)

    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data) & 0xffffffff)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6 if channels == 4 else 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b""))


def triangle_fixture():
    vertices = np.array([[1, 2, 3], [1, 4, 3], [1, 2, 5]], dtype="<f4")
    faces = np.array([[0, 1, 2]], dtype="<i4")
    uv = np.array([[.1, .2], [.9, .2], [.1, .8]], dtype="<f4")
    return vertices, faces, uv


def glb_fixture(vertices, faces, uv, source_maps, reverse=False, alpha_mode="OPAQUE", translation=None, wrong_filter=False):
    binary = bytearray()
    views, accessors = [], []

    def view(payload):
        while len(binary) % 4:
            binary.append(0)
        index = len(views)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload)})
        binary.extend(payload)
        return index

    def accessor(array, component, kind):
        index = len(accessors)
        accessors.append({"bufferView": view(array.tobytes()), "componentType": component, "count": len(array), "type": kind})
        return index

    gltf_positions = vertices[:, [0, 2, 1]].copy()
    gltf_positions[:, 2] *= -1
    gltf_uv = uv.copy()
    gltf_uv[:, 1] = 1 - gltf_uv[:, 1]
    position_accessor = accessor(gltf_positions, 5126, "VEC3")
    uv_accessor = accessor(gltf_uv, 5126, "VEC2")
    indices = faces[:, [0, 2, 1]] if reverse else faces
    index_accessor = accessor(indices.astype("<u2").reshape(-1), 5123, "SCALAR")
    images = [{"mimeType": "image/png", "bufferView": view(png_fixture(source_maps[name]))} for name in ("base-color.png", "MR.png")]
    node = {"mesh": 0}
    if translation is not None:
        node["translation"] = translation
    document = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [node],
                "meshes": [{"primitives": [{"attributes": {"POSITION": position_accessor, "TEXCOORD_0": uv_accessor}, "indices": index_accessor, "material": 0}]}],
                "materials": [{"alphaMode": alpha_mode, "doubleSided": True, "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicRoughnessTexture": {"index": 1}}}],
                "textures": [{"source": 0,"sampler":0}, {"source": 1,"sampler":0}], "images": images,
                "samplers": [{"magFilter": 9728 if wrong_filter else 9729,"minFilter":9987,"wrapS":33071,"wrapT":33071}],
                "bufferViews": views, "accessors": accessors, "buffers": [{"byteLength": len(binary)}]}
    encoded = json.dumps(document).encode("utf-8")
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\0" * (-len(binary) % 4)
    chunks = struct.pack("<II", len(encoded), 0x4e4f534a) + encoded + struct.pack("<II", len(binary), 0x004e4942) + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


class RendererParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = renderer.PRIVATE / "unit-scratch"
        cls.scratch.mkdir(parents=True, exist_ok=True)
        if cls.scratch.is_symlink() or cls.scratch.resolve().parent != renderer.PRIVATE.resolve():
            raise ValueError("Plain owned unit-test scratch required")

    def setUp(self):
        self.vertices, self.faces, self.uv = triangle_fixture()
        self.reference = renderer.canonical_triangles(self.vertices, self.faces, self.uv)
        self.maps = {"base-color.png": np.array([[[21, 80, 150, 254], [250, 30, 40, 255]], [[6, 9, 12, 255], [255, 0, 90, 254]]], dtype=np.uint8),
                     "MR.png": np.array([[[255, 240, 1], [255, 250, 0]], [[255, 255, 2], [255, 254, 3]]], dtype=np.uint8)}

    def check_glb(self, **kwargs):
        with tempfile.TemporaryDirectory(prefix="modly-uv-glb-fixture-", dir=self.scratch) as directory:
            path = Path(directory) / "fixture.glb"
            path.write_bytes(glb_fixture(self.vertices, self.faces, self.uv, self.maps, **kwargs))
            return renderer.glb_check(path, self.reference, self.maps)

    def test_independent_PIL_RGB_and_RGBA_PNGs_keep_rows_channels_and_alpha(self):
        for pixels in self.maps.values():
            np.testing.assert_array_equal(renderer.png_pixels(png_fixture(pixels)), pixels)

    def test_all_five_filters_decode_asymmetric_original_rows(self):
        pixels = np.random.default_rng(1204).integers(0, 256, (5, 7, 4), dtype=np.uint8)
        np.testing.assert_array_equal(renderer.png_pixels(filtered_fixture(pixels, range(5))), pixels)

    def test_PNG_bad_CRC_and_truncation_are_rejected(self):
        encoded = png_fixture(self.maps["base-color.png"])
        damaged = bytearray(encoded)
        damaged[-1] ^= 1
        for invalid in (bytes(damaged), encoded[:-5]):
            with self.assertRaises(ValueError):
                renderer.png_pixels(invalid)

    def test_cyclic_winding_and_duplicate_corner_vertices_are_accepted(self):
        cyclic = self.faces[:, [1, 2, 0]]
        result = renderer.check_triangles(self.vertices, cyclic, self.uv, self.reference)
        self.assertTrue(result["cyclic_triangle_positions_exact"])
        positions = self.vertices[cyclic.reshape(-1)]
        uv = self.uv[cyclic.reshape(-1)]
        renderer.check_triangles(positions, np.array([[0, 1, 2]], dtype="<i4"), uv, self.reference)

    def test_reversed_winding_changed_position_and_UV_are_rejected(self):
        with self.assertRaises(ValueError):
            renderer.check_triangles(self.vertices, self.faces[:, [0, 2, 1]], self.uv, self.reference)
        changed = self.vertices.copy()
        changed[0, 0] = np.nextafter(changed[0, 0], np.float32(2))
        with self.assertRaises(ValueError):
            renderer.check_triangles(changed, self.faces, self.uv, self.reference)
        uv = self.uv.copy()
        uv[0, 0] += .001
        with self.assertRaises(ValueError):
            renderer.check_triangles(self.vertices, self.faces, uv, self.reference)

    def test_GLB_embedded_RGB_bytes_UV_winding_and_ignored_encoded_alpha(self):
        result = self.check_glb()
        self.assertTrue(result["standard_glTF2"])
        self.assertEqual(result["alpha_mode"], "OPAQUE")
        self.assertEqual(result["maps"]["base-color.png"]["alpha_encoded"]["min"], 254)
        self.assertTrue(all(record["decoded_source_RGB_exact"] for record in result["maps"].values()))

    def test_GLB_reverse_transformed_and_transparent_material_are_rejected(self):
        for options in ({"reverse": True}, {"translation": [0, .1, 0]}, {"alpha_mode": "BLEND"}, {"wrong_filter": True}):
            with self.assertRaises(ValueError):
                self.check_glb(**options)

    def test_wrong_map_RGB_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="modly-uv-glb-fixture-", dir=self.scratch) as directory:
            path = Path(directory) / "fixture.glb"
            path.write_bytes(glb_fixture(self.vertices, self.faces, self.uv, self.maps))
            expected = {name: value.copy() for name, value in self.maps.items()}
            expected["base-color.png"][0, 0, 0] += 1
            with self.assertRaises(ValueError):
                renderer.glb_check(path, self.reference, expected)


if __name__ == "__main__":
    unittest.main()
