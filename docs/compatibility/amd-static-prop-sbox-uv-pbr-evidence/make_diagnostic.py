"""Deterministic, locally authored UV diagnostic; no external images or fonts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import zlib

SIZE = 1024
TILE = 64
SEED = b"MODLY_PHASE22_UV_DIAGNOSTIC_V1"
PALETTE = tuple(tuple(255 if index & (1 << channel) else 0 for channel in range(3)) for index in range(8))
FONT = {
    "0": (14,17,19,21,25,17,14), "1": (4,12,4,4,4,4,14),
    "2": (14,17,1,2,4,8,31), "3": (30,1,1,14,1,1,30),
    "4": (2,6,10,18,31,2,2), "5": (31,16,16,30,1,1,30),
    "6": (14,16,16,30,17,17,14), "7": (31,1,2,4,8,8,8),
    "8": (14,17,17,14,17,17,14), "9": (14,17,17,15,1,1,14),
    "A": (14,17,17,31,17,17,17), "B": (30,17,17,30,17,17,30),
    "C": (14,17,16,16,16,17,14), "D": (30,17,17,17,17,17,30),
    "E": (31,16,16,30,16,16,31), "F": (31,16,16,30,16,16,16),
    "L": (16,16,16,16,16,16,31), "M": (17,27,21,21,17,17,17),
    "O": (14,17,17,17,17,17,14), "P": (30,17,17,30,16,16,16),
    "U": (17,17,17,17,17,17,14), "V": (17,17,17,17,17,10,4),
    "Y": (17,17,10,4,4,4,4), "+": (0,4,4,31,4,4,0),
    " ": (0,0,0,0,0,0,0),
}


def artwork():
    """Return one category byte per texel. PNG row increases with source GLB v."""
    image = bytearray(SIZE * SIZE)
    cell = [[hashlib.sha256(SEED + f":{row:02d}:{col:02d}".encode("ascii")).digest()[0] & 7
             for col in range(16)] for row in range(16)]
    for y in range(SIZE):
        for x in range(SIZE):
            complement = 7 if ((x // 32) + (y // 32)) & 1 else 0
            image[y * SIZE + x] = cell[y // TILE][x // TILE] ^ complement

    def rectangle(x0, y0, x1, y1, category):
        assert 0 <= category < 8
        for y in range(max(0, y0), min(SIZE, y1)):
            left, right = max(0, x0), min(SIZE, x1)
            image[y * SIZE + left:y * SIZE + right] = bytes([category]) * max(0, right - left)

    def text(value, x, y, scale, category):
        for char in value:
            for row, bits in enumerate(FONT[char]):
                for col in range(5):
                    if bits & (1 << (4 - col)):
                        rectangle(x + col * scale, y + row * scale,
                                  x + (col + 1) * scale, y + (row + 1) * scale, category)
            x += 6 * scale

    def triangle(a, b, c, category):
        def cross(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        for y in range(max(0, min(a[1], b[1], c[1])), min(SIZE, max(a[1], b[1], c[1]) + 1)):
            for x in range(max(0, min(a[0], b[0], c[0])), min(SIZE, max(a[0], b[0], c[0]) + 1)):
                sides = [cross(a,b,(x,y)), cross(b,c,(x,y)), cross(c,a,(x,y))]
                if all(v >= 0 for v in sides) or all(v <= 0 for v in sides):
                    image[y * SIZE + x] = category

    # Each original 64-pixel tile is labelled column-then-row in hexadecimal.
    for row in range(16):
        for col in range(16):
            rectangle(col * TILE + 2, row * TILE + 2, col * TILE + 28, row * TILE + 20, 0)
            text(f"{col:X}{row:X}", col * TILE + 4, row * TILE + 4, 2, 7)
    # Large non-antialiased orientation marks and asymmetric corner regions.
    rectangle(0, 0, SIZE, 64, 0)
    rectangle(0, 64, 64, SIZE, 0)
    text("MODLY P22 UV", 80, 16, 5, 7)
    rectangle(70, 72, 950, 128, 0)
    text("U+", 80, 80, 6, 7)
    rectangle(200, 94, 890, 108, 7)
    triangle((930,101),(882,77),(882,125),7)
    text("V+", 7, 166, 4, 6)
    rectangle(25, 240, 39, 880, 6)
    triangle((32,930),(8,878),(56,878),6)
    rectangle(72, 144, 152, 224, 1)
    rectangle(88, 160, 120, 192, 7)
    rectangle(912, 144, 992, 224, 2)
    rectangle(928, 160, 944, 208, 0)
    rectangle(928, 192, 976, 208, 0)
    for y in range(872, 968):
        for x in range(72, 168):
            if (x - 120) ** 2 + (y - 920) ** 2 <= 44 ** 2:
                image[y * SIZE + x] = 4
    rectangle(120, 888, 136, 904, 7)
    triangle((888,888),(1000,920),(968,1000),5)
    rectangle(952, 936, 972, 956, 3)
    for category in range(8):
        rectangle(64 + category * 120, 1008, 64 + (category + 1) * 120, 1024, category)
    return image


def png_bytes(categories):
    def chunk(kind, data):
        return struct.pack(">I",len(data)) + kind + data + struct.pack(">I",zlib.crc32(kind + data) & 0xffffffff)
    raw = bytearray()
    for y in range(SIZE):
        raw.append(0)
        for value in categories[y * SIZE:(y + 1) * SIZE]:
            raw.extend(PALETTE[value])
    header = struct.pack(">IIBBBBB",SIZE,SIZE,8,2,0,0,0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR",header) + chunk(b"sRGB",b"\x00")
            + chunk(b"IDAT",zlib.compress(raw,9)) + chunk(b"IEND",b""))


def generate():
    base = Path(__file__).resolve().parent
    output = base / "texture"
    assert json.loads((base / "preservation-before.json").read_text())["classification"] == "PASS_AUTHENTICATED_PHASE21_PARENT"
    output.mkdir(exist_ok=False)
    categories = artwork()
    png = png_bytes(categories)
    rgba = b"".join(bytes((*PALETTE[value],255)) for value in categories)
    with (output / "diagnostic_uv_color.png").open("xb") as stream:
        stream.write(png)
    original = base.parent / "amd-static-prop-sbox-engine-proof-20261031-v1/verified_source/neutral_test_basecolor.png"
    neutral = original.read_bytes()
    assert hashlib.sha256(neutral).hexdigest() == "2364e7ddb8b271a95db206b39bfbdf938d9e427d46dda2745357264c8451c89e"
    with (output / "neutral_source_basecolor.png").open("xb") as stream:
        stream.write(neutral)
    record = dict(classification="GENERATED_PENDING_INDEPENDENT_PIXEL_REVIEW", width=SIZE,height=SIZE,
                  png_sha256=hashlib.sha256(png).hexdigest(),rgba_sha256=hashlib.sha256(rgba).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  seed=SEED.decode(),palette_rgb=PALETTE,category_counts=dict(Counter(categories)),
                  axes="u+ right; v+ down in PNG rows and original glTF UV convention",
                  recipe="64-pixel SHA256-coded tiles, 32-pixel complementary checks, hexadecimal column/row labels, large U+/V+ arrows, asymmetric corner marks",
                  png_format="1024x1024 RGB8 opaque, explicit sRGB chunk, no external texture/font",
                  neutral_png_sha256=hashlib.sha256(neutral).hexdigest(),source_images_modified=False,
                  generation_attempt=1)
    with (base / "diagnostic-texture.json").open("x",encoding="utf-8") as stream:
        json.dump(record,stream,indent=2)
    print(json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate",action="store_true")
    args = parser.parse_args()
    if args.generate:
        generate()
    else:
        print("INERT: use --generate for the single new private diagnostic texture")
