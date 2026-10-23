"""Real CPU image-byte contracts; no model, worker, GPU, or WSL execution."""
import importlib.abc
import io
import json
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

BLOCKED = {'torch', 'triton', 'trellis2', 'transformers', 'cumesh', 'flex_gemm', 'o_voxel', 'nvdiffrast'}


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BLOCKED:
            raise AssertionError('CPU image validation attempted model import: ' + name)


def setUpModule():
    global guard
    guard = NoModels()
    sys.meta_path.insert(0, guard)


def tearDownModule():
    sys.meta_path.remove(guard)

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from services import amd_image_test as subject
except ImportError:
    subject = None


def encoded(image, format='PNG', **options):
    stream = io.BytesIO()
    image.save(stream, format=format, **options)
    return stream.getvalue()


@unittest.skipIf(Image is None, 'Pillow unavailable in this runtime; capability check tested separately')
class ImageTestContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(subject, 'Local AMD image validator is missing')
        image = Image.new('RGBA', (12, 10), (40, 80, 120, 0))
        image.paste((40, 80, 120, 255), (2, 2, 10, 8))
        self.valid = encoded(image)

    def test_real_rgba_reports_original_contract_without_generation_claim(self):
        result = subject.test_image(self.valid)
        self.assertEqual(result['schema'], 'modly.amd.image-test.v1')
        self.assertEqual(result['status'], 'valid')
        self.assertIs(result['eligible'], True)
        self.assertEqual(result['validation'], 'original_cpu_input_contract')
        self.assertFalse(result['preprocessing_complete'])
        self.assertFalse(result['generation_qualified'])
        self.assertEqual(result['metadata']['format'], 'PNG')
        self.assertEqual(result['metadata']['mode'], 'RGBA')
        self.assertEqual(result['metadata']['foreground_bbox'], [2, 2, 9, 7])
        self.assertEqual(result['metadata']['foreground_extent'], [8, 6])
        self.assertTrue(result['metadata']['transparent_pixels_present'])
        self.assertEqual(result['input_contract']['source_size'], [12, 10])
        self.assertEqual(result['reasons'], [])

    def test_jpeg_and_palette_transparency_are_not_converted_to_rgba(self):
        palette = Image.new('P', (12, 10))
        palette.info['transparency'] = 0
        for blob in (encoded(Image.new('RGB', (12, 10)), 'JPEG'), encoded(palette)):
            with self.subTest(bytes=len(blob)):
                result = subject.test_image(blob)
                self.assertEqual(result['status'], 'invalid')
                self.assertEqual(result['reasons'][0]['code'], 'not_rgba')
                self.assertIsNone(result['input_contract'])
                self.assertIn('RGBA', result['reasons'][0]['correction'])

    def test_opaque_empty_faint_threshold_and_thin_foregrounds_refused(self):
        cases = [('transparent_background', Image.new('RGBA', (12, 10), (0, 0, 0, 255))),
                 ('foreground', Image.new('RGBA', (12, 10), (0, 0, 0, 0)))]
        for alpha in (128, 204):
            faint = Image.new('RGBA', (12, 10))
            faint.paste((1, 2, 3, alpha), (2, 2, 10, 8))
            cases.append(('foreground', faint))
        thin = Image.new('RGBA', (12, 10))
        thin.paste((1, 2, 3, 255), (2, 2, 10, 4))
        cases.append(('foreground', thin))
        for code, image in cases:
            with self.subTest(code=code, extrema=image.getchannel('A').getextrema()):
                result = subject.test_image(encoded(image))
                self.assertEqual(result['status'], 'invalid')
                self.assertEqual(result['reasons'][0]['code'], code)

    def test_alpha205_is_foreground_and_bbox_is_inclusive(self):
        image = Image.new('RGBA', (8, 8))
        image.paste((1, 2, 3, 205), (2, 2, 5, 5))
        result = subject.test_image(encoded(image))
        self.assertEqual(result['status'], 'valid')
        self.assertEqual(result['metadata']['foreground_bbox'], [2, 2, 4, 4])
        self.assertEqual(result['metadata']['foreground_extent'], [3, 3])

    def test_animation_and_oversized_dimensions_fail_before_pixel_admission(self):
        first = Image.new('RGBA', (12, 10), (1, 2, 3, 0))
        second = Image.new('RGBA', (12, 10), (4, 5, 6, 255))
        animated = encoded(first, save_all=True, append_images=[second], duration=50, loop=0)
        result = subject.test_image(animated)
        self.assertEqual(result['reasons'][0]['code'], 'animated')
        oversized = encoded(Image.new('RGBA', (4097, 3)))
        result = subject.test_image(oversized)
        self.assertEqual(result['reasons'][0]['code'], 'dimensions')

    def test_mask_disappearing_after_upstream_resize_is_refused(self):
        image = Image.new('RGBA', (4096, 1024))
        image.paste((1, 2, 3, 255), (100, 100, 103, 103))
        result = subject.test_image(encoded(image))
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(result['reasons'][0]['code'], 'resized_foreground')

    def test_corrupt_and_empty_bytes_have_safe_actionable_results(self):
        for blob, code in ((b'', 'empty'), (b'not image bytes', 'corrupt'), (self.valid[:45], 'corrupt')):
            with self.subTest(code=code):
                result = subject.test_image(blob)
                self.assertEqual(result['status'], 'invalid')
                self.assertEqual(result['reasons'][0]['code'], code)
                self.assertTrue(result['reasons'][0]['correction'])

    def test_oversize_rejected_before_decoder_and_original_refusal_stays_invalid(self):
        result = subject.test_image(b'x' * (2*1024**2 + 1))
        self.assertEqual(result['reasons'][0]['code'], 'too_large')
        with patch.object(subject, 'admitted_image', side_effect=ValueError('strict original rejection')):
            result = subject.test_image(self.valid)
        self.assertEqual(result['status'], 'invalid')
        self.assertIsNone(result['input_contract'])
        self.assertEqual(result['reasons'][0]['code'], 'original_contract')


class ImageTestAvailability(unittest.TestCase):
    def test_concurrent_validation_is_unavailable_without_queue_or_retry(self):
        self.assertIsNotNone(subject, 'Local AMD image validator is missing')
        entered, release = threading.Event(), threading.Event()
        def slow(data):
            entered.set(); release.wait(1)
            return subject._result()
        with patch.object(subject, '_local_test', slow):
            first = threading.Thread(target=subject.test_image, args=(b'first',))
            first.start()
            try:
                self.assertTrue(entered.wait(1))
                second = subject.test_image(b'second')
                self.assertEqual(second['status'], 'unavailable')
                self.assertEqual(second['reasons'][0]['code'], 'validation_busy')
            finally:
                release.set(); first.join(2)

    def test_product_prefix_and_image_multipart_field_reach_real_asgi_route(self):
        from fastapi import FastAPI
        from routers import amd_image_test as route
        app = FastAPI()
        app.include_router(route.router, prefix='/amd')
        body = (b'--image-test\r\nContent-Disposition: form-data; name="image"; filename="cutout.png"\r\n'
                b'Content-Type: image/png\r\n\r\nnot an image\r\n--image-test--\r\n')
        scope = dict(type='http', asgi={'version': '3.0'}, http_version='1.1', method='POST', scheme='http',
            path='/amd/image-test', raw_path=b'/amd/image-test', query_string=b'', root_path='',
            headers=[(b'content-type', b'multipart/form-data; boundary=image-test'),
                     (b'content-length', str(len(body)).encode())], client=('127.0.0.1', 1), server=('localhost', 80))
        responses = []
        async def receive(): return dict(type='http.request', body=body, more_body=False)
        async def send(message): responses.append(message)
        async def inline(function, *args): return function(*args)
        with patch.object(route, 'run_in_threadpool', inline), \
             patch.dict(os.environ, {'MODLY_AMD_IMAGE_TEST_PYTHON': ''}):
            coroutine = app(scope, receive, send)
            with self.assertRaises(StopIteration): coroutine.send(None)
        self.assertEqual(responses[0]['status'], 200)
        row = json.loads(b''.join(message.get('body', b'') for message in responses))
        self.assertEqual(row['schema'], 'modly.amd.image-test.v1')
        self.assertIsInstance(row['metadata'], dict)
        self.assertFalse(row['generation_qualified'])

    def test_missing_pillow_is_unavailable_not_valid_or_invalid(self):
        self.assertIsNotNone(subject, 'Local AMD image validator is missing')
        with patch.dict(sys.modules, {'PIL': None}), patch.dict(os.environ, {'MODLY_AMD_IMAGE_TEST_PYTHON': ''}):
            result = subject.test_image(b'bounded but unexamined image')
        self.assertEqual(result['status'], 'unavailable')
        self.assertIsNone(result['eligible'])
        self.assertEqual(result['validation'], 'not_run')
        self.assertFalse(result['generation_qualified'])
        self.assertIsNone(result['input_contract'])

    def test_untrusted_runtime_path_is_unavailable_without_launch(self):
        self.assertIsNotNone(subject, 'Local AMD image validator is missing')
        with patch.dict(sys.modules, {'PIL': None}), \
             patch.dict(os.environ, {'MODLY_AMD_IMAGE_TEST_PYTHON': '//remote/python.exe'}):
            result = subject.test_image(b'bounded image')
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['reasons'][0]['code'], 'runtime_unavailable')

    def test_incomplete_oversize_timeout_or_foreign_worker_output_cannot_validate(self):
        self.assertIsNotNone(subject, 'Local AMD image validator is missing')
        for variant in ('corrupt', 'oversize', 'timeout', 'foreign-schema'):
            with self.subTest(variant=variant):
                class Process:
                    def __init__(self, *args, **kwargs):
                        self.stdin = io.BytesIO()
                        self.stdout = io.BytesIO(b'x' * 65537 if variant == 'oversize' else
                            b'{"schema":"foreign","status":"valid"}' if variant == 'foreign-schema' else b'not-json')
                        self.returncode = 0
                        self.killed = False
                    def __enter__(self): return self
                    def __exit__(self, *args): self.stdin.close(); self.stdout.close()
                    def kill(self): self.killed = True; self.returncode = -9
                    def wait(self, timeout):
                        if variant == 'timeout' and not self.killed:
                            raise subject.subprocess.TimeoutExpired('fixed CPU script', timeout)
                        return self.returncode
                with patch.dict(sys.modules, {'PIL': None}), \
                     patch.dict(os.environ, {'MODLY_AMD_IMAGE_TEST_PYTHON': 'C:/fixed/python.exe'}), \
                     patch('services.amd_runtime.absolute_file', return_value=Path('C:/fixed/python.exe')), \
                     patch.object(subject.subprocess, 'Popen', Process):
                    result = subject.test_image(b'bounded image')
                self.assertEqual(result['status'], 'unavailable')
                self.assertIsNone(result['eligible'])
                self.assertIsNone(result['input_contract'])

    @unittest.skipUnless(Path('D:/AI/Trellis2AMD/venv/Scripts/python.exe').is_file(), 'Explicit existing Windows CPU runtime absent')
    def test_explicit_existing_windows_runtime_validates_real_bytes(self):
        self.assertIsNotNone(subject, 'Local AMD image validator is missing')
        # A literal PNG fixture avoids making the API runtime depend on Pillow.
        import base64
        blob = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAIElEQVR4nGNgoBQwwhlMzP+RJf7/+wuWYyJkAuUKKAcAw1QECIIlwpwAAAAASUVORK5CYII=')
        with patch.dict(sys.modules, {'PIL': None}), patch.dict(os.environ, {
                'MODLY_AMD_IMAGE_TEST_PYTHON': 'D:/AI/Trellis2AMD/venv/Scripts/python.exe'}):
            result = subject.test_image(blob)
        self.assertEqual(result['status'], 'valid', result)
        self.assertEqual(result['validation'], 'original_cpu_input_contract')
        self.assertEqual(result['metadata']['foreground_extent'], [4, 4])
        self.assertFalse(result['generation_qualified'])

    def test_upload_read_is_bounded_closes_and_ignores_filename(self):
        from routers.amd_image_test import image_test
        class Upload:
            filename = 'C:/private/do-not-read.png'
            def __init__(self): self.closed = False
            async def read(self, size):
                if size != 2097153: raise AssertionError('Unbounded upload read')
                return b'x' * size
            async def close(self): self.closed = True
        upload = Upload()
        # The sandbox blocks Windows asyncio's socketpair before the route
        # starts. Replace only scheduling, preserving real byte validation.
        async def inline(function, *args): return function(*args)
        with patch('routers.amd_image_test.run_in_threadpool', inline):
            coroutine = image_test(upload)
            with self.assertRaises(StopIteration) as done:
                coroutine.send(None)
        result = done.exception.value
        self.assertTrue(upload.closed)
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(result['reasons'][0]['code'], 'too_large')
        self.assertNotIn('private', str(result))


if __name__ == '__main__': unittest.main()
