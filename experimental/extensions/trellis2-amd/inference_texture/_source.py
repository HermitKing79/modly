"""Hash-pinned private module loading, without generic sys.modules aliases."""
import builtins
import hashlib
from pathlib import Path
import re
import stat
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[4]
_loaded = {}


def load_pinned(name, relative, expected_sha256, imports=None):
    """Return an isolated source module; mapped imports affect this module only.

    The exact bytes verified are compiled. Private registration supports
    dataclasses; historical names such as admission/run_contract are untouched.
    """
    if (type(name) is not str or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9.]*', name)
            or type(relative) is not str or Path(relative).is_absolute()
            or '..' in Path(relative).parts or type(expected_sha256) is not str
            or not re.fullmatch('[0-9a-f]{64}', expected_sha256)):
        raise ValueError('Exact private module name, relative source and SHA256 required')
    path = ROOT / relative
    if path.resolve(strict=True) != path or not path.is_relative_to(ROOT):
        raise ValueError('Pinned source must remain inside its canonical repository')
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or getattr(info, 'st_file_attributes', 0) & 1024):
        raise ValueError('Pinned source must be a plain single-link file')
    blob = path.read_bytes()
    after = path.lstat()
    if ((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or hashlib.sha256(blob).hexdigest() != expected_sha256):
        raise ValueError('Pinned source SHA256 or identity differs: ' + relative)
    imports = {} if imports is None else dict(imports)
    if any(type(k) is not str or not isinstance(v, ModuleType) for k, v in imports.items()):
        raise ValueError('Private imports require explicit module objects')
    key = (name, relative, expected_sha256, tuple(sorted((k, id(v)) for k, v in imports.items())))
    if key in _loaded:
        return _loaded[key]
    private_name = '_modly_brick_pbr_' + name.replace('.', '_') + '_' + str(len(_loaded))
    module = ModuleType(private_name)
    module.__file__ = str(path)
    module.__package__ = ''
    def local_import(import_name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and import_name in imports:
            return imports[import_name]
        return builtins.__import__(import_name, globals, locals, fromlist, level)
    module.__dict__['__builtins__'] = dict(vars(builtins), __import__=local_import)
    sys.modules[private_name] = module
    try:
        exec(compile(blob, str(path), 'exec'), module.__dict__)
    except BaseException:
        del sys.modules[private_name]
        raise
    _loaded[key] = module
    return module
