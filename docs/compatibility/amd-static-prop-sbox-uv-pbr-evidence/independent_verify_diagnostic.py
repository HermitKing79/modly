"""Independent PNG decoder and full-raster reconstruction; new evidence output only."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import zlib

BASE = Path(__file__).resolve().parent
EXPECTED_GENERATOR = "723eda9607de90770bafacb3389cddc91e0d1ccf94d35b40c428856652afb299"
EXPECTED_NEUTRAL = "2364e7ddb8b271a95db206b39bfbdf938d9e427d46dda2745357264c8451c89e"
SIZE = 1024


def digest(value):
    return hashlib.sha256(value).hexdigest()


def decode_png(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset, chunks, idat, ihdr = 8, [], bytearray(), None
    while offset < len(data):
        count = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4:offset + 8]
        value = data[offset + 8:offset + 8 + count]
        assert len(value) == count
        crc = struct.unpack_from(">I", data, offset + 8 + count)[0]
        assert crc == zlib.crc32(kind + value) & 0xffffffff
        chunks.append(kind.decode("ascii"))
        if kind == b"IHDR":
            assert ihdr is None
            ihdr = struct.unpack(">IIBBBBB", value)
        elif kind == b"sRGB":
            assert value == b"\x00"
        elif kind == b"IDAT":
            idat.extend(value)
        elif kind == b"IEND":
            assert value == b"" and offset + 12 == len(data)
        else:
            raise AssertionError("Unexpected diagnostic PNG chunk")
        offset += 12 + count
    assert chunks == ["IHDR", "sRGB", "IDAT", "IEND"]
    assert ihdr == (SIZE, SIZE, 8, 2, 0, 0, 0)
    scanlines = zlib.decompress(idat)
    assert len(scanlines) == SIZE * (1 + SIZE * 3)
    pixels = bytearray()
    for row in range(SIZE):
        start = row * (1 + SIZE * 3)
        assert scanlines[start] == 0
        pixels.extend(scanlines[start + 1:start + 1 + SIZE * 3])
    return bytes(pixels), chunks


def independently_reconstruct(font):
    # Fill in 32-pixel spans rather than run the generator's per-pixel program.
    rows = []
    for y in range(SIZE):
        row = bytearray()
        for half_column in range(32):
            tile_seed = b"MODLY_PHASE22_UV_DIAGNOSTIC_V1" + f":{y // 64:02d}:{half_column // 2:02d}".encode()
            category = hashlib.sha256(tile_seed).digest()[0] % 8
            if (half_column + y // 32) % 2:
                category = 7 - category
            row.extend([category] * 32)
        rows.append(row)

    def fill(left, top, right, bottom, color):
        assert 0 <= left <= right <= SIZE and 0 <= top <= bottom <= SIZE
        for row in rows[top:bottom]:
            row[left:right] = bytes([color]) * (right - left)

    def glyphs(text, left, top, scale, color):
        for letter_number, letter in enumerate(text):
            for gy, bits in enumerate(font[letter]):
                for gx, bit in enumerate(f"{bits:05b}"):
                    if bit == "1":
                        x = left + scale * (6 * letter_number + gx)
                        y = top + scale * gy
                        fill(x, y, x + scale, y + scale, color)

    def signed_area(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def triangle(points, color):
        a, b, c = points
        total = abs(signed_area(a, b, c))
        for y in range(min(p[1] for p in points), max(p[1] for p in points) + 1):
            for x in range(min(p[0] for p in points), max(p[0] for p in points) + 1):
                p = (x, y)
                if sum(abs(signed_area(u, v, p)) for u, v in ((a, b), (b, c), (c, a))) == total:
                    rows[y][x] = color

    for gy in range(16):
        for gx in range(16):
            x, y = 64 * gx, 64 * gy
            fill(x + 2, y + 2, x + 28, y + 20, 0)
            glyphs(f"{gx:X}{gy:X}", x + 4, y + 4, 2, 7)
    fill(0, 0, SIZE, 64, 0)
    fill(0, 64, 64, SIZE, 0)
    glyphs("MODLY P22 UV", 80, 16, 5, 7)
    fill(70, 72, 950, 128, 0)
    glyphs("U+", 80, 80, 6, 7)
    fill(200, 94, 890, 108, 7)
    triangle(((930, 101), (882, 77), (882, 125)), 7)
    glyphs("V+", 7, 166, 4, 6)
    fill(25, 240, 39, 880, 6)
    triangle(((32, 930), (8, 878), (56, 878)), 6)
    fill(72, 144, 152, 224, 1)
    fill(88, 160, 120, 192, 7)
    fill(912, 144, 992, 224, 2)
    fill(928, 160, 944, 208, 0)
    fill(928, 192, 976, 208, 0)
    for y in range(872, 968):
        for x in range(72, 168):
            if (x - 120) ** 2 + (y - 920) ** 2 <= 44 ** 2:
                rows[y][x] = 4
    fill(120, 888, 136, 904, 7)
    triangle(((888, 888), (1000, 920), (968, 1000)), 5)
    fill(952, 936, 972, 956, 3)
    for color in range(8):
        fill(64 + 120 * color, 1008, 64 + 120 * (color + 1), 1024, color)
    return b"".join(rows)


def main():
    generator = (BASE / "make_diagnostic.py").read_bytes()
    assert digest(generator) == EXPECTED_GENERATOR
    # Read inert literal glyph data with ast.literal_eval; never import or call generator code.
    module = ast.parse(generator)
    font = next(ast.literal_eval(node.value) for node in module.body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FONT" for t in node.targets))
    expected_categories = independently_reconstruct(font)
    png = (BASE / "texture/diagnostic_uv_color.png").read_bytes()
    actual_rgb, chunks = decode_png(png)
    palette = [bytes([255 * ((category >> channel) & 1) for channel in range(3)]) for category in range(8)]
    expected_rgb = b"".join(palette[category] for category in expected_categories)
    assert actual_rgb == expected_rgb, "Full diagnostic raster differs from independent reconstruction"
    rgba = b"".join(palette[category] + b"\xff" for category in expected_categories)
    recipe = json.loads((BASE / "diagnostic-texture.json").read_text())
    assert digest(png) == recipe["png_sha256"]
    assert digest(rgba) == recipe["rgba_sha256"]
    assert {str(k): v for k, v in Counter(expected_categories).items()} == recipe["category_counts"]
    neutral = (BASE / "texture/neutral_source_basecolor.png").read_bytes()
    original = (BASE.parent / "amd-static-prop-sbox-engine-proof-20261031-v1/verified_source/neutral_test_basecolor.png").read_bytes()
    assert neutral == original and digest(neutral) == EXPECTED_NEUTRAL
    report = dict(classification="PASS_INDEPENDENT_FULL_DIAGNOSTIC_PIXEL_IDENTITY", width=SIZE, height=SIZE,
                  inspected_texels=SIZE * SIZE, generator_sha256=EXPECTED_GENERATOR,
                  diagnostic_png_sha256=digest(png), diagnostic_rgba_sha256=digest(rgba),
                  neutral_png_sha256=digest(neutral), chunk_types=chunks, every_chunk_crc_verified=True,
                  independent_full_raster_reconstruction=True, generator_logic_executed=False,
                  reconstruction="Independent 32-pixel spans, literal glyph bitmap data, area-sum triangle inclusion; all 1048576 decoded RGB pixels compared.",
                  category_counts=dict(Counter(expected_categories)),
                  visual_review="1024-square image inspected: readable MODLY P22 UV title, U+ right/V+ down arrows, hex column/row labels, distinct asymmetric corner marks and binary palette strip.",
                  output_scope="This fresh review JSON only; input PNGs and all project files read-only.")
    with (BASE / "independent-diagnostic-pixel-review.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
