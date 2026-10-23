"""Read-only capability/identity check for the explicitly configured runtimes."""
import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'transformers', 'diffusers', 'flex_gemm', 'cupy', 'onnxruntime'}:
            raise ImportError('Image preview cannot import a model/GPU runtime')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def probe(kind):
    sys.meta_path.insert(0, NoModels())
    result = dict(python=str(Path(sys.executable).absolute()), python_sha256=sha(sys.executable),
                  version=sys.version.split()[0], kind=kind, components={}, files={})
    if kind == 'api':
        import fastapi
        import uvicorn
        for module in (fastapi, uvicorn):
            result['components'][module.__name__] = importlib.metadata.version(module.__name__)
            result['files'][str(Path(module.__file__).absolute())] = sha(module.__file__)
        # If Pillow later appears here, the original service would prefer it.
        # Exact probe equality therefore blocks that unqualified route change.
        result['components']['Pillow_available'] = importlib.util.find_spec('PIL') is not None
    elif kind == 'cpu':
        import PIL
        from PIL import Image, features, _imaging
        result['components']['Pillow'] = PIL.__version__
        result['components']['webp'] = features.check('webp')
        base = Path(PIL.__file__).parent
        for name in ('__init__.py', 'Image.py', 'ImageFile.py', 'PngImagePlugin.py', 'JpegImagePlugin.py', 'WebPImagePlugin.py'):
            result['files'][str((base/name).absolute())] = sha(base/name)
        result['files'][str(Path(_imaging.__file__).absolute())] = sha(_imaging.__file__)
        # Real CPU image operations, without disk output or image conversion.
        import io
        buffer = io.BytesIO()
        Image.new('RGBA', (3, 3), (1, 2, 3, 0)).save(buffer, format='PNG')
        with Image.open(io.BytesIO(buffer.getvalue())) as image:
            image.load()
            if image.mode != 'RGBA' or image.size != (3, 3):
                raise ValueError('Original RGBA decoding unavailable')
    else:
        raise ValueError('Only api/cpu probes are supported')
    return result


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Expected api or cpu')
    print(json.dumps(probe(sys.argv[1]), sort_keys=True, allow_nan=False))
