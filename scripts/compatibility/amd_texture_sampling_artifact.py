"""Bounded CPU-only texture persistence, independent of Torch and model libraries.

The small serializer emits the public safetensors format (8-byte LE header length,
space-padded JSON header, contiguous LE tensor payloads). It has no native-loader
or safetensors dependency. Only the four approved tensor layouts are accepted.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys

import numpy as np

FORMAT = 'modly-amd-texture-latent-v1'
ARTIFACT_NAME = 'texture-latent.safetensors'
METADATA_NAME = 'texture-latent-metadata.json'
LAYOUTS = {'raw': ('F32', (3743, 32)), 'denormalized': ('F32', (3743, 32)),
           'coords': ('I32', (3743, 4)), 'noise': ('F32', (3743, 32))}
PAYLOAD_BYTES = 1497200
MAX_HEADER_BYTES = 65536
MAX_METADATA_BYTES = 65536
MAX_FILE_BYTES = 8 + MAX_HEADER_BYTES + PAYLOAD_BYTES


def _hash(blob):
    return hashlib.sha256(blob).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate artifact JSON key: ' + key)
        result[key] = value
    return result


def _json_value(value, depth=0):
    if depth > 16:
        raise ValueError('Artifact JSON nesting exceeds bound')
    if value is None or type(value) in (bool, str, int):
        return
    if type(value) is float and np.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_value(item, depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError('Artifact JSON keys must be strings')
            _json_value(item, depth + 1)
            if key.endswith('sha256') and (type(item) is not str or len(item) != 64
                    or any(c not in '0123456789abcdef' for c in item)):
                raise ValueError('Invalid provenance SHA256: ' + key)
        return
    raise ValueError('Artifact metadata requires finite JSON values')


def _json_bytes(value, limit=MAX_METADATA_BYTES):
    _json_value(value)
    blob = json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')
    if len(blob) > limit:
        raise ValueError('Artifact metadata exceeds byte bound')
    return blob


def _decode(blob):
    def nonfinite(_):
        raise ValueError('Nonfinite artifact JSON number')
    try:
        row = json.loads(blob, object_pairs_hook=_unique, parse_constant=nonfinite)
        _json_value(row)
        return row
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('Invalid artifact JSON') from exc


def _plain(path, *, directory=False):
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        raise ValueError('Artifact symlinks, junctions and reparse points are prohibited')
    if directory:
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError('Artifact root must be an existing directory')
    elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError('Artifact must be a plain file with one link')
    if hasattr(os, 'getuid') and info.st_uid != os.getuid():
        raise ValueError('Artifact paths must be owned by the current user')
    return info


def approved_root(root):
    """Treat the explicit caller-provided directory as the approved owned root."""
    root = Path(root)
    if not root.is_absolute():
        raise ValueError('Artifact root must be absolute')
    resolved = root.resolve(strict=True)
    if root != resolved:
        raise ValueError('Artifact root must already be resolved without aliases')
    for ancestor in (root, *root.parents):
        _plain(ancestor, directory=True)
        if (ancestor / '.git').exists() or (ancestor / '.git').is_symlink():
            raise ValueError('Permanent texture artifact root must be outside Git')
    return root


def _owned_file(path, root, name, *, must_exist):
    path = Path(path)
    if not path.is_absolute() or path != root / name:
        raise ValueError('Artifact path must be the exact approved root filename')
    if path.exists() or path.is_symlink():
        _plain(path)
        if path.resolve(strict=True) != path:
            raise ValueError('Artifact path escaped its approved root')
    elif must_exist:
        raise FileNotFoundError(path)
    return path


def _array(name, array):
    dtype, shape = LAYOUTS[name]
    expected_dtype = np.dtype('<i4' if dtype == 'I32' else '<f4')
    if (type(array) is not np.ndarray or array.shape != shape or array.dtype != expected_dtype
            or not array.flags.c_contiguous or array.nbytes != shape[0] * shape[1] * 4):
        raise ValueError('Texture artifact requires exact contiguous LE ' + dtype + ' ' + str(shape))
    if name != 'coords':
        if not np.isfinite(array).all():
            raise ValueError('Texture artifact floats must all be finite')
    else:
        if np.any(array[:, 0] != 0) or np.any(array[:, 1:] < 0) or np.any(array[:, 1:] > 31):
            raise ValueError('Texture coordinates require original bounded batch-zero grid32 values')
        linear = (array[:, 1].astype(np.int64) * 32 + array[:, 2]) * 32 + array[:, 3]
        if np.any(np.diff(linear) <= 0):
            raise ValueError('Texture coordinates must be unique and lexicographically ordered')
    return array.tobytes(order='C')


def _provenance(value):
    if type(value) is not dict or not value:
        raise ValueError('Texture artifact requires nonempty provenance')
    return _decode(_json_bytes(value, 48000))


def _header_metadata(provenance, rows):
    return {'format': FORMAT, 'provenance': _json_bytes(provenance, 48000).decode('utf-8'),
            **{name + '_sha256': rows[name]['sha256'] for name in LAYOUTS}}


def _bounded_read(path, maximum):
    _plain(path)
    with path.open('rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > maximum:
            raise ValueError('Artifact exceeds byte bound or plain-file ownership')
        blob = stream.read(maximum + 1)
    if len(blob) > maximum or len(blob) != info.st_size:
        raise ValueError('Artifact exceeds byte bound or changed during read')
    return blob


def _container(blob):
    if not PAYLOAD_BYTES + 8 < len(blob) <= MAX_FILE_BYTES:
        raise ValueError('Texture container payload/header length mismatch')
    count = struct.unpack('<Q', blob[:8])[0]
    if not 0 < count <= MAX_HEADER_BYTES or count % 8 or 8 + count + PAYLOAD_BYTES != len(blob):
        raise ValueError('Texture safetensors header length mismatch')
    header = _decode(blob[8:8 + count])
    if type(header) is not dict or set(header) != {'__metadata__', *LAYOUTS}:
        raise ValueError('Texture container must contain exactly the approved tensors')
    rows = {}
    cursor = 0
    payload = blob[8 + count:]
    for name, (dtype, shape) in LAYOUTS.items():
        size = shape[0] * shape[1] * 4
        expected = {'dtype': dtype, 'shape': list(shape), 'data_offsets': [cursor, cursor + size]}
        # JSON booleans and floats compare equal to Python ints; the format requires ints.
        if _json_bytes(header[name]) != _json_bytes(expected):
            raise ValueError('Texture tensor layout/offset mismatch: ' + name)
        data = payload[cursor:cursor + size]
        array = np.frombuffer(data, dtype='<i4' if dtype == 'I32' else '<f4').reshape(shape)
        _array(name, array)
        rows[name] = dict(dtype=dtype, shape=list(shape), tensor_bytes=size, sha256=_hash(data))
        cursor += size
    return header, rows, payload


def verify(path, root, expected_metadata=None):
    """Independently validate bytes and provenance after the producing worker exits."""
    root = approved_root(root)
    path = _owned_file(path, root, ARTIFACT_NAME, must_exist=True)
    sidecar = _owned_file(root / METADATA_NAME, root, METADATA_NAME, must_exist=True)
    blob = _bounded_read(path, MAX_FILE_BYTES)
    header, rows, _ = _container(blob)
    metadata = _decode(_bounded_read(sidecar, MAX_METADATA_BYTES))
    if type(metadata) is not dict or set(metadata) != {
            'format', 'artifact_file', 'artifact_bytes', 'artifact_sha256', 'tensors', 'provenance'}:
        raise ValueError('Texture sidecar fields mismatch')
    provenance = _provenance(metadata['provenance'])
    expected = dict(format=FORMAT, artifact_file=ARTIFACT_NAME, artifact_bytes=len(blob),
                    artifact_sha256=_hash(blob), tensors=rows, provenance=provenance)
    if (_json_bytes(metadata) != _json_bytes(expected)
            or _json_bytes(header['__metadata__']) != _json_bytes(_header_metadata(provenance, rows))):
        raise ValueError('Texture payload hashes/provenance/sidecar mismatch')
    if expected_metadata is not None and _json_bytes(metadata) != _json_bytes(expected_metadata):
        raise ValueError('Texture artifact differs from expected saved metadata')
    return dict(verified=True, artifact_path=str(path), metadata_path=str(sidecar), metadata=metadata)


def _exclusive_write(path, blob):
    with path.open('xb') as stream:
        if os.fstat(stream.fileno()).st_nlink != 1:
            raise ValueError('Artifact exclusive file creation ownership changed')
        if stream.write(blob) != len(blob):
            raise OSError('Incomplete texture artifact write')
        stream.flush()
        os.fsync(stream.fileno())


def write(scratch, raw, denormalized, coords, provenance, *, noise):
    """Exclusively persist the approved four tensors and verify byte-exact readback."""
    root = approved_root(scratch)
    tensors = dict(raw=raw, denormalized=denormalized, coords=coords, noise=noise)
    data = {name: _array(name, array) for name, array in tensors.items()}
    provenance = _provenance(provenance)
    rows, descriptors, cursor = {}, {}, 0
    for name, (dtype, shape) in LAYOUTS.items():
        size = len(data[name])
        rows[name] = dict(dtype=dtype, shape=list(shape), tensor_bytes=size, sha256=_hash(data[name]))
        descriptors[name] = dict(dtype=dtype, shape=list(shape), data_offsets=[cursor, cursor + size])
        cursor += size
    header = _json_bytes({'__metadata__': _header_metadata(provenance, rows), **descriptors}, MAX_HEADER_BYTES)
    header += b' ' * (-len(header) % 8)
    if len(header) > MAX_HEADER_BYTES or cursor != PAYLOAD_BYTES:
        raise ValueError('Texture container exceeds its exact payload/header bound')
    blob = struct.pack('<Q', len(header)) + header + b''.join(data.values())
    metadata = dict(format=FORMAT, artifact_file=ARTIFACT_NAME, artifact_bytes=len(blob),
                    artifact_sha256=_hash(blob), tensors=rows, provenance=provenance)
    metadata_blob = _json_bytes(metadata)
    path = _owned_file(root / ARTIFACT_NAME, root, ARTIFACT_NAME, must_exist=False)
    sidecar = _owned_file(root / METADATA_NAME, root, METADATA_NAME, must_exist=False)
    # Precheck both names, and keep exclusive opens: a partial or existing pair is never overwritten.
    for candidate in (path, sidecar):
        if candidate.exists():
            raise FileExistsError(candidate)
    approved_root(root)
    _exclusive_write(path, blob)
    _exclusive_write(sidecar, metadata_blob)
    result = verify(path, root, metadata)
    persisted = _bounded_read(path, MAX_FILE_BYTES)
    if persisted != blob or _bounded_read(sidecar, MAX_METADATA_BYTES) != metadata_blob:
        raise ValueError('Texture artifact byte-exact readback differs')
    return result


def persistence_probe(root, provenance=None):
    """Qualify actual storage using a fresh child and the exact production writer/reader."""
    root = approved_root(root)
    scratch = root / 'persistence-probe'
    scratch.mkdir(exist_ok=False)
    raw = np.arange(3743 * 32, dtype=np.float32).reshape(3743, 32) / np.float32(128)
    denormalized = np.ascontiguousarray(raw * np.float32(.25) + np.float32(.5))
    noise = np.ascontiguousarray(-raw - np.float32(1))
    xyz = np.arange(3743, dtype=np.int32)
    coords = np.column_stack((np.zeros(3743, np.int32), xyz // 1024, xyz // 32 % 32, xyz % 32)).astype(np.int32)
    supplied = {'purpose': 'cpu-only-persistence-probe', 'synthetic': True,
                'noise': {'source': 'synthetic-independent-probe-sequence'}}
    if provenance is not None:
        supplied['caller'] = _provenance(provenance)
    result = write(scratch, raw, denormalized, coords, supplied, noise=noise)
    return {**result, 'synthetic': True, 'torch_imported': 'torch' in sys.modules}


write_texture_artifact = write
verify_texture_artifact = verify
probe_texture_artifact_persistence = persistence_probe


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--probe', type=Path, help='Existing approved root; creates fresh persistence-probe child')
    modes.add_argument('--verify', type=Path, help='Permanent texture-latent.safetensors path')
    parser.add_argument('--root', type=Path, help='Exact approved artifact root for --verify')
    args = parser.parse_args()
    if args.probe is not None:
        result = persistence_probe(args.probe)
    elif args.root is not None:
        result = verify(args.verify, args.root)
    else:
        parser.error('--verify requires --root')
    print(json.dumps(result, sort_keys=True, allow_nan=False))
