"""Bounded CPU image admission only. No preprocessing, model or generation run."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.amd_stage_contract import admitted_image

MAX_BYTES = 2 * 1024**2
MAX_OUTPUT = 64 * 1024
PREPROCESS = {'image_mode': 'alpha-only', 'resolution': 512}
_VALIDATION_LOCK = threading.Lock()


def _result():
    return dict(schema='modly.amd.image-test.v1', status='invalid', eligible=False, validation='not_run',
        preprocessing_complete=False, generation_qualified=False, metadata={},
        limits=dict(max_bytes=MAX_BYTES, max_dimension=4096, max_pixels=16*1024**2,
                    foreground_alpha_gt=204, min_foreground_extent=3),
        reasons=[], input_contract=None)


def _reason(result, code, message, correction, *, unavailable=False):
    result['reasons'].append(dict(code=code, message=message, correction=correction))
    if unavailable:
        result.update(status='unavailable', eligible=None)
    return result


def _local_test(data):
    result = _result()
    if type(data) is not bytes or not data:
        return _reason(result, 'empty', 'No image bytes received.', 'Choose a pre-segmented RGBA image.')
    if len(data) > MAX_BYTES:
        return _reason(result, 'too_large', 'Encoded image exceeds 2 MiB.', 'Export an RGBA image of at most 2 MiB.')
    try:
        from PIL import Image
    except ImportError:
        return _reason(result, 'decoder_unavailable', 'CPU image decoder is unavailable.',
            'Configure the existing compatible local CPU image runtime.', unavailable=True)
    try:
        with Image.open(io.BytesIO(data)) as image:
            metadata = dict(format=image.format, mode=image.mode, width=image.width, height=image.height,
                frames=getattr(image, 'n_frames', 1), has_alpha_channel='A' in image.getbands(),
                transparent_pixels_present=None, foreground_bbox=None, foreground_extent=None,
                resized_size=None, resized_foreground_bbox=None, resized_foreground_extent=None)
            result['metadata'] = metadata
            if metadata['frames'] != 1:
                return _reason(result, 'animated', 'Multiple image frames are unsupported.', 'Export one still RGBA frame.')
            if image.mode != 'RGBA':
                return _reason(result, 'not_rgba', 'The decoded image is not RGBA; no conversion was applied.',
                    'Remove the background and export a true RGBA PNG or WebP with transparency.')
            if min(image.size) < 1 or max(image.size) > 4096 or image.width*image.height > 16*1024**2:
                return _reason(result, 'dimensions', 'Image dimensions exceed the CPU input limits.',
                    'Resize to at most 4096 pixels per side and 16 Mi pixels total.')
            alpha = image.getchannel('A')
            metadata['transparent_pixels_present'] = alpha.getextrema()[0] == 0
            box = alpha.point(lambda value: 255 if value > 204 else 0).getbbox()
            if box is not None:
                metadata['foreground_bbox'] = [box[0], box[1], box[2]-1, box[3]-1]
                metadata['foreground_extent'] = [box[2]-box[0], box[3]-box[1]]
        try:
            contract = admitted_image(data, preprocess=dict(PREPROCESS))
        except ValueError:
            if not metadata['transparent_pixels_present']:
                code, message, correction = ('transparent_background', 'No fully transparent background pixel exists.',
                    'Supply a segmented cutout with a fully transparent background; opaque RGB conversion is insufficient.')
            elif box is None or min(metadata['foreground_extent']) < 3:
                code, message, correction = ('foreground', 'Foreground above alpha 204 must span at least 3 pixels on both axes.',
                    'Use a substantial visible foreground with alpha 205–255 and a transparent background.')
            else:
                # The original contract rechecks the mask after its 1024 resize.
                scale = min(1, 1024/max(metadata['width'], metadata['height']))
                code = 'resized_foreground' if scale < 1 else 'original_contract'
                message = 'Original CPU input contract rejected the image.'
                correction = 'Use a larger, stronger cutout that remains nondegenerate after the 1024-pixel resize.'
            return _reason(result, code, message, correction)
        metadata.update(resized_size=contract['resized_size'],
                        resized_foreground_bbox=contract['resized_foreground_bbox'],
                        resized_foreground_extent=[contract['resized_foreground_bbox'][2]-contract['resized_foreground_bbox'][0]+1,
                                                   contract['resized_foreground_bbox'][3]-contract['resized_foreground_bbox'][1]+1])
        result.update(status='valid', eligible=True, validation='original_cpu_input_contract', input_contract=contract)
        return result
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return _reason(result, 'corrupt', 'Image bytes could not be decoded safely.', 'Export a complete single-frame RGBA PNG or WebP.')


def _configured_test(data):
    from services.amd_runtime import absolute_file
    result = _result()
    try:
        executable = absolute_file(os.environ['MODLY_AMD_IMAGE_TEST_PYTHON'], 'CPU image Python')
        if executable.name.lower() != 'python.exe':
            raise ValueError('Explicit Windows Python executable required')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')
        with subprocess.Popen([str(executable), '-B', '-I', str(Path(__file__).resolve()), '--cpu-worker'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)) as process:
            output = []
            failures = []
            def write():
                try:
                    process.stdin.write(data)
                    process.stdin.close()
                except OSError:
                    failures.append(True)
            def read():
                try:
                    output.append(process.stdout.read(MAX_OUTPUT+1))
                    if len(output[0]) > MAX_OUTPUT:
                        process.kill()
                except OSError:
                    failures.append(True)
            writer = threading.Thread(target=write, daemon=True)
            reader = threading.Thread(target=read, daemon=True)
            writer.start(); reader.start()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=1)
                raise ValueError('CPU validation deadline expired')
            writer.join(timeout=1); reader.join(timeout=1)
            if writer.is_alive() or reader.is_alive() or failures or process.returncode != 0 or not output or len(output[0]) > MAX_OUTPUT:
                raise ValueError('Incomplete CPU validator response')
        row = json.loads(output[0])
        if (type(row) is not dict or row.get('schema') != 'modly.amd.image-test.v1'
                or row.get('status') not in ('valid', 'invalid', 'unavailable')
                or row.get('preprocessing_complete') is not False or row.get('generation_qualified') is not False
                or set(row) != set(result)):
            raise ValueError('Invalid CPU validator response')
        return row
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return _reason(result, 'runtime_unavailable', 'Configured CPU image validation did not complete.',
            'Check the existing local Python/Pillow configuration; no runtime was installed or substituted.', unavailable=True)


def test_image(data):
    if not _VALIDATION_LOCK.acquire(blocking=False):
        return _reason(_result(), 'validation_busy', 'Another CPU image validation is still running.',
            'Wait for the current validation to finish, then explicitly test this image again.', unavailable=True)
    try:
        result = _local_test(data)
        if result['status'] == 'unavailable' and os.environ.get('MODLY_AMD_IMAGE_TEST_PYTHON'):
            return _configured_test(data)
        return result
    finally:
        _VALIDATION_LOCK.release()


if __name__ == '__main__':
    if sys.argv[1:] != ['--cpu-worker']:
        raise SystemExit(2)
    data = sys.stdin.buffer.read(MAX_BYTES+1)
    print(json.dumps(_local_test(data), separators=(',', ':'), allow_nan=False))
