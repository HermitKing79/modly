"""Reopen one candidate PNG and compare it with the accepted same-frame preview."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
E = Path(__file__).resolve().parent
sys.path.insert(0, str(E))
from inputs import P, REFERENCE, begin, authenticate, verify_stage, plain, digest, save, require


def comparison_label(triangles):
    require(type(triangles) is int and 0 < triangles <= 4_173_584, 'Measured candidate triangle count')
    return f'50% target candidate: {triangles:,} triangles / +2 EV'


def main():
    begin()
    from PIL import Image, ImageDraw, ImageStat
    watched = authenticate()
    assessed = verify_stage('assess')
    rendered = verify_stage('render')
    require(rendered['render_count'] == 1 and rendered['fresh_GLTF_imports'] == 0, 'One candidate frame and zero GLB imports')
    require(rendered['assessed_result_SHA256'] == digest(P/'assess/result.json'), 'Rendered the exact assessed candidate')
    require(rendered['candidate_geometry_exact_before_and_after'] and rendered['native_corner_bytes_exact_before_and_after'] and rendered['imported_material_unchanged'], 'Actual candidate and material preserved')
    watched += [P/'assess/result.json', P/'render/result.json', *[P/'render'/name for name in rendered['files']]]
    watched = list(dict.fromkeys(watched))
    before = {str(path): digest(path) for path in watched}
    out = plain(P/'inspect')
    require(all(path.name in ('scratch', 'stdout.txt', 'stderr.txt') for path in out.iterdir()), 'Fresh host inspection output')
    image_path = plain(P/'render/view-pos-x-ev2.png')
    require(set(rendered['files']) == {image_path.name} and digest(image_path) == rendered['files'][image_path.name]['sha256'], 'Exact single candidate frame')
    require(digest(plain(REFERENCE)) == rendered['reference_preview_SHA256'], 'Exact accepted same-frame reference')
    with Image.open(image_path) as image:
        image.load()
        require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (640, 640), 'Actual640RGB candidate PNG')
        require(max(ImageStat.Stat(image).stddev) > 5, 'Nonblank candidate frame')
        current = image.copy()
    with Image.open(REFERENCE) as image:
        image.load()
        require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (640, 640), 'Accepted640RGB native-colour reference')
        prior = image.copy()
    sheet = Image.new('RGB', (1280, 720), 'white')
    sheet.paste(prior, (0, 60))
    sheet.paste(current, (640, 60))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), 'Accepted original: 4,173,584 triangles / +2 EV', fill='black')
    draw.text((652, 10), comparison_label(rendered['triangles']), fill='black')
    draw.text((12, 32), 'Matched camera, studio, native colour bytes and flat material; no image resizing', fill='black')
    comparison = out/'comparison-pos-x.png'
    with comparison.open('xb') as stream:
        sheet.save(stream, format='PNG')
        stream.flush()
        __import__('os').fsync(stream.fileno())
    with Image.open(comparison) as image:
        image.load()
        require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (1280, 720), 'Independent comparison reopening')
    require(before == {str(path): digest(path) for path in watched}, 'Original and rendered inputs unchanged')
    save(out/'result.json', dict(classification='PASS', PNG_verified=True, fresh_GLTF_imports=0, render_count=1,
         triangles=rendered['triangles'], target_triangles=2_086_792, original_triangles=4_173_584,
         original_reference=dict(bytes=REFERENCE.stat().st_size, sha256=digest(REFERENCE)),
         rendered_image=dict(bytes=image_path.stat().st_size, sha256=digest(image_path)),
         files={comparison.name: dict(bytes=comparison.stat().st_size, sha256=digest(comparison))},
         input_preservation=True, no_image_resizing=True, geometry_or_material_modified=False,
         interpretation='Matched single-view comparison; independent visual inspection remains required and hidden surfaces are not certified'))


if __name__ == '__main__':
    main()
