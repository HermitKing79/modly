"""Synthetic CPU ASGI checks; no server/socket/model/GPU/WSL launch."""
import importlib.abc
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'api'))
BLOCKED = {'torch', 'triton', 'trellis2', 'transformers', 'cumesh', 'flex_gemm', 'o_voxel', 'nvdiffrast'}


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BLOCKED:
            raise AssertionError('CPU preview attempted model import: ' + name)


guard = NoModels()
sys.meta_path.insert(0, guard)


def tearDownModule():
    sys.meta_path.remove(guard)


try:
    import preview_api
except ModuleNotFoundError as error:
    if error.name != 'preview_api': raise
    preview_api = None

from routers import amd_image_test as original_route

try:
    from PIL import Image
except ImportError:
    Image = None


def encoded(image, format='PNG'):
    stream = io.BytesIO()
    image.save(stream, format=format)
    return stream.getvalue()


class PreviewApiTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(preview_api, 'Isolated CPU preview API is missing')
        self.settings = patch.dict(os.environ, MODLY_AMD_PREVIEW_NONCE='unit-preview-only',
            MODLY_AMD_IMAGE_TEST_PYTHON='D:/AI/Trellis2AMD/venv/Scripts/python.exe')
        self.settings.start()
        self.addCleanup(self.settings.stop)

    def request(self, path, *, method='GET', blob=None, field='image', headers=()):
        """Run actual FastAPI parsing/routing without Windows asyncio socketpair.

        Only threadpool scheduling is inlined. The original CPU validator and
        the real multipart parser execute unchanged on synthetic bytes.
        """
        body = b''
        request_headers = list(headers)
        if blob is not None:
            body = (b'--preview-test\r\nContent-Disposition: form-data; name="' + field.encode() +
                b'"; filename="synthetic.png"\r\nContent-Type: image/png\r\n\r\n' + blob + b'\r\n--preview-test--\r\n')
            request_headers.append((b'content-type', b'multipart/form-data; boundary=preview-test'))
        request_headers.append((b'content-length', str(len(body)).encode()))
        scope = dict(type='http', asgi={'version': '3.0'}, http_version='1.1', method=method, scheme='http',
            path=path, raw_path=path.encode(), query_string=b'', root_path='', headers=request_headers,
            client=('127.0.0.1', 1), server=('127.0.0.1', 8000))
        messages = []
        async def receive(): return dict(type='http.request', body=body, more_body=False)
        async def send(message): messages.append(message)
        async def inline(function, *args): return function(*args)
        with patch.object(original_route, 'run_in_threadpool', inline), \
             patch('starlette.datastructures.run_in_threadpool', inline):
            coroutine = preview_api.app(scope, receive, send)
            with self.assertRaises(StopIteration): coroutine.send(None)
        data = b''.join(message.get('body', b'') for message in messages)
        return messages[0]['status'], json.loads(data) if data else None, dict(messages[0]['headers'])

    def test_health_binds_preview_nonce_and_disables_generation(self):
        status, row, _ = self.request('/health')
        self.assertEqual(status, 200)
        self.assertEqual(row, dict(status='ok', preview_nonce='unit-preview-only', generation_enabled=False))

    def test_missing_or_invalid_nonce_and_runtime_fail_health(self):
        for key, value in (('MODLY_AMD_PREVIEW_NONCE', ''), ('MODLY_AMD_PREVIEW_NONCE', 'bad\nnonce'),
                           ('MODLY_AMD_IMAGE_TEST_PYTHON', ''), ('MODLY_AMD_IMAGE_TEST_PYTHON', '//remote/python.exe')):
            with self.subTest(key=key, value=value), patch.dict(os.environ, {key: value}):
                status, _, _ = self.request('/health')
                self.assertEqual(status, 503)

    def test_generation_docs_and_main_application_routes_are_absent(self):
        for path, method in (('/generate', 'POST'), ('/generation/generate', 'POST'), ('/model/all', 'GET'),
                             ('/docs', 'GET'), ('/redoc', 'GET'), ('/openapi.json', 'GET'), ('/', 'GET')):
            with self.subTest(path=path):
                status, _, _ = self.request(path, method=method)
                self.assertEqual(status, 404)

    def test_local_file_origin_can_read_preview_health(self):
        status, _, headers = self.request('/health', headers=[(b'origin', b'null')])
        self.assertEqual(status, 200)
        self.assertIn(headers.get(b'access-control-allow-origin'), (b'null', b'*'))

    def test_missing_or_wrong_multipart_field_is_rejected(self):
        for blob, field in ((None, 'image'), (b'bad', 'file')):
            with self.subTest(field=field):
                status, _, _ = self.request('/amd/image-test', method='POST', blob=blob, field=field)
                self.assertEqual(status, 422)

    def test_encoded_byte_limit_is_enforced_by_unchanged_route(self):
        status, row, _ = self.request('/amd/image-test', method='POST', blob=b'x' * 2097153)
        self.assertEqual(status, 200)
        self.assertEqual(row['reasons'][0]['code'], 'too_large')
        self.assertFalse(row['generation_qualified'])

    @unittest.skipIf(Image is None, 'Pillow unavailable; real pixel checks run in original AMD CPU runtime')
    def test_original_cpu_rgba_jpeg_and_corrupt_byte_contracts(self):
        image = Image.new('RGBA', (8, 8))
        image.paste((1, 2, 3, 255), (2, 2, 6, 6))
        fixtures = [(encoded(image), 'valid', None),
                    (encoded(Image.new('RGB', (8, 8)), 'JPEG'), 'invalid', 'not_rgba'),
                    (b'broken image', 'invalid', 'corrupt')]
        for blob, expected, code in fixtures:
            with self.subTest(expected=expected, code=code):
                status, row, _ = self.request('/amd/image-test', method='POST', blob=blob)
                self.assertEqual(status, 200)
                self.assertEqual(row['schema'], 'modly.amd.image-test.v1')
                self.assertEqual(row['status'], expected)
                self.assertFalse(row['preprocessing_complete'])
                self.assertFalse(row['generation_qualified'])
                if code is not None:
                    self.assertEqual(row['reasons'][0]['code'], code)
                else:
                    self.assertEqual(row['validation'], 'original_cpu_input_contract')
                    self.assertEqual(row['input_contract']['foreground_bbox'], [2, 2, 5, 5])


if __name__ == '__main__': unittest.main()
