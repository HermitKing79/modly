"""Private comparison sheets from authenticated existing images; no model work."""
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont, ImageStat

E = Path(__file__).resolve().parent
ROOT = E.parents[2]
P = ROOT.parent / 'artifacts/amd-brick-pbr-preview-9755adc'
R = P / 'renders'
OLD = ROOT.parent / 'artifacts/amd-brick-visual-inspection-b536fb6'
PHOTO_SHA = 'a31ba3b084757faa66e4bb91a85bd956fcc1eaaaf298a091fd08044351cd0293'
CLAY_SHA = 'be2454863b66546131a3cc702953de06f8f0f507d0e062a4564c2d2f10f077f2'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_image(path, expected=None):
    if path.is_symlink() or path.stat().st_size > 16 * 1024**2:
        raise ValueError('Plain bounded private image required')
    if expected is not None and sha(path) != expected:
        raise ValueError('Image identity mismatch')
    with Image.open(path) as image:
        image.load()
        allowed = {'PNG', 'WEBP'} if expected == PHOTO_SHA else {'PNG'}
        if image.format not in allowed or max(image.size) > 2048:
            raise ValueError('Bounded original photo or PNG required')
        if image.mode == 'RGBA':
            background = Image.new('RGBA', image.size, (64, 64, 64, 255))
            background.alpha_composite(image)
            return background.convert('RGB')
        if image.mode != 'RGB':
            raise ValueError('RGB/RGBA expected')
        return image.copy()


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', size)


def write_png(path, image):
    with path.open('xb') as stream:
        image.save(stream, format='PNG')
    with Image.open(path) as check:
        check.load()
        assert check.size == image.size and check.mode == 'RGB'
        stats = ImageStat.Stat(check).stddev
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path),
                dimensions=list(image.size), channel_stddev=stats)


def sheet(path, tiles, title, columns):
    width = columns * 500
    rows = (len(tiles) + columns - 1) // columns
    image = Image.new('RGB', (width, 98 + 540 * rows), '#eeeeee')
    draw = ImageDraw.Draw(image)
    draw.text((14, 9), title, fill='black', font=font(24))
    draw.text((14, 44), 'Genuine learned fields | unchanged full-resolution mesh | Cycles CPU',
              fill='#222222', font=font(18))
    for index, (tile, label) in enumerate(tiles):
        tile = tile.copy()
        tile.thumbnail((480, 480), Image.Resampling.LANCZOS)
        x, y = (index % columns) * 500, 82 + (index // columns) * 540
        image.paste(tile, (x + (500 - tile.width) // 2, y + (480 - tile.height) // 2))
        draw.text((x + 10, y + 485), label, fill='black', font=font(18))
    return write_png(path, image)


def main():
    if sys.stdin.buffer.readline(64) != b'GO\n':
        raise ValueError('Owned host authorization required')
    host = json.loads((R / 'host-result.json').read_text())
    metadata = json.loads((R / 'render-metadata.json').read_text())
    old = json.loads((OLD / 'renders/render-metadata.json').read_text())
    assert host['classification'] == metadata['classification'] == 'PASS'
    assert host['source_unchanged'] and metadata['geometry_unchanged']
    assert not host['job_cleanup']['forced_job_cleanup'] and host['returncode'] == 0
    assert metadata['geometry_payloads_before'] == metadata['geometry_payloads_after']
    identity = json.loads((OLD / 'source-image-identity.json').read_text())
    photo_path = Path(identity['images'][0]['path'])
    photo = read_image(photo_path, PHOTO_SHA)
    clay_path = OLD / 'renders/view-pos-x.png'
    clay = read_image(clay_path, CLAY_SHA)
    new_views = {view['name']: view for view in metadata['views']}
    old_views = {view['name']: view for view in old['views']}
    rendered, inputs = {}, {}
    for name in ('pos-x', 'iso-a', 'iso-b'):
        view, previous = new_views[name], old_views[name]
        for key in ('camera_location', 'camera_quaternion', 'ortho_scale'):
            assert view[key] == previous[key], (name, key)
        path = R / view['PNG']
        rendered[name] = read_image(path, view['sha256'])
        assert rendered[name].size == (1024, 1024)
        inputs[name] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
    crop_box = (470, 360, 660, 590)
    detail = rendered['pos-x'].crop(crop_box).resize((760, 920), Image.Resampling.NEAREST)
    detailed = Image.new('RGB', (800, 1000), '#eeeeee')
    detailed.paste(detail, (20, 42))
    draw = ImageDraw.Draw(detailed)
    draw.text((14, 8), '+X material detail: 4x image crop', fill='black', font=font(19))
    draw.text((14, 968), 'No additional render or geometry modification', fill='black', font=font(17))
    result = write_png(P / 'surface-detail.png', detailed)
    first = [(photo, 'Authenticated original photograph'), (clay, '+X prior clay'),
             (rendered['pos-x'], '+X learned RGB + metallic/roughness/alpha')]
    comparison = sheet(P / 'comparison-sheet.png', first,
                       'BRICK: REFERENCE / CLAY / GENUINE LEARNED PBR', 3)
    contact = sheet(P / 'contact-sheet.png', first + [
        (rendered['iso-a'], 'Isometric A: +X -Y +Z'),
        (rendered['iso-b'], 'Isometric B: -X +Y -Z'),
        (detail, '+X 4x nearest-neighbour crop')],
        'GENUINE BRICK LEARNED-PBR INSPECTION', 3)
    assert sha(photo_path) == PHOTO_SHA and sha(clay_path) == CLAY_SHA
    record = dict(classification='PASS', images_private_only=True,
                  photo_path=str(photo_path), photo_sha256=PHOTO_SHA,
                  prior_clay_path=str(clay_path), prior_clay_sha256=CLAY_SHA,
                  actual_cameras_match_prior_clay_exactly=True,
                  photograph_camera_not_calibrated=True, render_images=inputs,
                  derived_images={Path(x['path']).name: x for x in (result, comparison, contact)},
                  detail_crop_box_pixels=list(crop_box), detail_is_image_crop_not_new_geometry=True)
    with (P / 'image-verification.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
