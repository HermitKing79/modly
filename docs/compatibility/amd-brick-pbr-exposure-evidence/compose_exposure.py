"""Private EV-only image comparison; encoded-display statistics, not radiometry."""
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image

E = Path(__file__).resolve().parent
ROOT = E.parents[2]
P = ROOT.parent / 'artifacts/amd-brick-pbr-exposure-b0b7be1'
R = P / 'renders'
PREVIOUS = ROOT / 'docs/compatibility/amd-brick-pbr-preview-evidence'
HELPER_SHA = '461717271d7e4ba89a6bf6b4b1acaabe1ff718dfe128eb901c8d7c58d2bf5b53'
BASELINE_PNG_SHA = 'a9bc79d78d72c1d08e602910fe5780ccb606f51e8d5b574a8e1b9af78cd90387'


def statistics(rgb, mask):
    pixels = np.asarray(rgb, dtype=np.uint8)[mask].astype(np.float64) / 255
    if not len(pixels):
        raise ValueError('Nonempty fixed image region required')
    luma = pixels @ np.array([.2126, .7152, .0722])
    high, low = pixels.max(axis=1), pixels.min(axis=1)
    saturation = np.divide(high - low, high, out=np.zeros_like(high), where=high > 0)
    return dict(pixels=len(pixels), mean_encoded_RGB=pixels.mean(axis=0).tolist(),
                mean_encoded_luma=float(luma.mean()),
                encoded_luma_percentiles_5_50_95=np.percentile(luma, [5, 50, 95]).tolist(),
                any_channel_at_255_count=int((high == 1).sum()),
                any_channel_at_least_250_count=int((high >= 250 / 255).sum()),
                all_channels_at_most_16_count=int((high <= 16 / 255).sum()),
                mean_display_HSV_saturation=float(saturation.mean()))


def main():
    if sys.stdin.buffer.readline(64) != b'GO\n':
        raise ValueError('Owned host authorization required')
    import hashlib
    helper_path = PREVIOUS / 'compose_preview.py'
    assert hashlib.sha256(helper_path.read_bytes()).hexdigest() == HELPER_SHA
    spec = importlib.util.spec_from_file_location('qualified_private_image_helpers', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    host = json.loads((R / 'host-result.json').read_text())
    metadata = json.loads((R / 'render-metadata.json').read_text())
    assert host['classification'] == metadata['classification'] == 'PASS'
    assert host['source_unchanged'] and metadata['geometry_unchanged']
    assert host['returncode'] == 0 and not host['job_cleanup']['forced_job_cleanup']
    assert metadata['settings_before'] == metadata['settings_after']
    assert metadata['material_before'] == metadata['material_after']
    identity = json.loads((helper.OLD / 'source-image-identity.json').read_text())
    photo_path = Path(identity['images'][0]['path'])
    photo = helper.read_image(photo_path, helper.PHOTO_SHA)
    clay = helper.read_image(helper.OLD / 'renders/view-pos-x.png', helper.CLAY_SHA)
    clay_pixels = np.asarray(clay, dtype=np.int16)
    # A fixed old-clay contrast mask excludes the constant world background.
    # It is a proxy region, not a new geometric silhouette or surface classifier.
    mask = np.max(np.abs(clay_pixels - clay_pixels[0, 0]), axis=2) > 3
    assert mask.shape == (1024, 1024) and 10000 < mask.sum() < 600000
    keys = [('pos-x', 0), ('pos-x-ev1', 1), ('pos-x-ev2', 2)]
    by_name = {view['name']: view for view in metadata['views']}
    assert list(by_name) == [name for name, ev in keys]
    variants, metrics = {}, {}
    for name, exposure in keys:
        view = by_name[name]
        assert view['exposure_EV'] == exposure
        image = helper.read_image(R / view['PNG'], view['sha256'])
        assert image.size == (1024, 1024)
        variants[name] = image
        metrics[name] = dict(exposure_EV=exposure, **statistics(image, mask))
    previous_path = ROOT.parent / 'artifacts/amd-brick-pbr-preview-9755adc/renders/view-pos-x.png'
    previous = helper.read_image(previous_path, BASELINE_PNG_SHA)
    baseline_difference = np.abs(np.asarray(previous, dtype=np.int16) -
                                 np.asarray(variants['pos-x'], dtype=np.int16))
    baseline = dict(pixel_identical=bool(not baseline_difference.any()),
                    maximum_channel_code_difference=int(baseline_difference.max()),
                    mean_absolute_channel_code_difference=float(baseline_difference.mean()),
                    previous_sha256=BASELINE_PNG_SHA,
                    new_sha256=helper.sha(R / 'view-pos-x.png'))
    tiles = [(photo, 'Authenticated reference photograph'),
             (variants['pos-x'], '+X baseline: EV 0'),
             (variants['pos-x-ev1'], '+X exposure +1 EV'),
             (variants['pos-x-ev2'], '+X exposure +2 EV')]
    comparison = helper.sheet(P / 'exposure-comparison.png', tiles,
                             'GENUINE PBR: EXPOSURE ONLY - NO MATERIAL CHANGE', 4)
    contact = helper.sheet(P / 'contact-sheet.png', tiles,
                          'BRICK PBR: REFERENCE / EV 0 / +1 / +2', 2)
    crop = (470, 360, 660, 590)
    details = [(variants[name].crop(crop).resize((760, 920), Image.Resampling.NEAREST),
                'Fixed +X crop: EV ' + str(ev)) for name, ev in keys]
    detail = helper.sheet(P / 'detail-comparison.png', details,
                         'SAME SURFACE CROP - IMAGE ENLARGEMENT', 3)
    with Image.open(photo_path) as original:
        rgba = np.asarray(original.convert('RGBA'))
        reference_statistics = statistics(rgba[:, :, :3], rgba[:, :, 3] >= 250)
    images = {name: dict(path=str(R / by_name[name]['PNG']), bytes=(R / by_name[name]['PNG']).stat().st_size,
                        sha256=helper.sha(R / by_name[name]['PNG'])) for name, ev in keys}
    result = dict(classification='PASS', images_private_only=True, render_images=images,
                  derived_images={Path(x['path']).name: x for x in (comparison, contact, detail)},
                  exposure_statistics=metrics, baseline_against_previous=baseline,
                  fixed_region=dict(source='authenticated previous +X clay',
                                    rule='max(abs(clay_RGB-corner_RGB)) > 3 uint8 code values',
                                    pixels=int(mask.sum()), geometric_silhouette_claimed=False),
                  statistic_convention='Rec709-weighted encoded display code values; not linear radiance or a physical reflectance measurement',
                  photograph_foreground_statistics=reference_statistics,
                  photograph_projection_and_lighting_not_calibrated=True,
                  photograph_comparison_is_not_a_colour_fidelity_score=True,
                  detail_crop_box_pixels=list(crop), detail_is_image_crop=True,
                  colour_management_comparison_performed=False,
                  input_photo_sha256=helper.PHOTO_SHA, original_PBR_material_unchanged=True)
    with (P / 'image-verification.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
