"""Bounded CPU-only persistence of the guided decoder's six-channel field.

This standalone writer follows the texture-latent safetensors contract. It never
imports Torch or model code. The independent reader checks the public format,
every tensor byte, provenance and unclamped raw * .5 + .5 transform.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

import numpy as np

FORMAT = 'modly-amd-texture-field-v1'
ARTIFACT_NAME = 'texture-field.safetensors'
METADATA_NAME = 'texture-field-metadata.json'
MAX_ROWS = 1842968
CHANNELS = ('red', 'green', 'blue', 'metallic', 'roughness', 'alpha')
LAYOUTS = {'raw': ('F32', 6), 'features': ('F32', 6), 'coordinates': ('I32', 4)}
MAX_PAYLOAD_BYTES = MAX_ROWS * 64
MAX_HEADER_BYTES = 65536
MAX_METADATA_BYTES = 65536
MAX_FILE_BYTES = 8 + MAX_HEADER_BYTES + MAX_PAYLOAD_BYTES


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
        value = json.loads(blob, object_pairs_hook=_unique, parse_constant=nonfinite)
        _json_value(value)
        return value
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('Invalid artifact JSON') from exc


def _plain(path, *, directory=False, owned=True):
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        raise ValueError('Artifact symlinks, junctions and reparse points are prohibited')
    if directory:
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError('Artifact root must be an existing directory')
    elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError('Artifact must be a plain file with one link')
    if owned and hasattr(os, 'getuid') and info.st_uid != os.getuid():
        raise ValueError('Artifact root/files must be owned by the current user')
    return info


def approved_root(root):
    """Validate the exact existing caller-approved root, on Windows or /mnt/c.

    All ancestors, including the filesystem root, are checked for Git metadata
    and aliases. System-owned Linux mount ancestors need not belong to the
    writer; the approved output root and regular files must belong to it.
    Windows ownership authority remains the caller's host preflight.
    """
    root = Path(root)
    if not root.is_absolute():
        raise ValueError('Artifact root must be absolute')
    resolved = root.resolve(strict=True)
    if root != resolved:
        raise ValueError('Artifact root must already be resolved without aliases')
    for ancestor in (root, *root.parents):
        _plain(ancestor, directory=True, owned=ancestor == root)
        if os.path.lexists(ancestor / '.git'):
            raise ValueError('Permanent texture artifact root must be outside all Git roots')
    return root


def _owned_file(path, root, name, *, must_exist):
    path = Path(path)
    if not path.is_absolute() or path != root / name:
        raise ValueError('Artifact path must be the exact approved root filename')
    if os.path.lexists(path):
        _plain(path)
        if path.resolve(strict=True) != path:
            raise ValueError('Artifact path escaped its approved root')
    elif must_exist:
        raise FileNotFoundError(path)
    return path


def _row_count(raw):
    if (type(raw) is not np.ndarray or raw.ndim != 2 or raw.shape[1] != 6
            or not 1 <= raw.shape[0] <= MAX_ROWS):
        raise ValueError('Texture field requires bounded nonempty [N,6] raw features')
    return raw.shape[0]


def _array(name, array, count):
    dtype, width = LAYOUTS[name]
    expected_dtype = np.dtype('<i4' if dtype == 'I32' else '<f4')
    if (type(array) is not np.ndarray or array.shape != (count, width)
            or array.dtype != expected_dtype or not array.flags.c_contiguous
            or array.nbytes != count * width * 4):
        raise ValueError('Texture field requires exact contiguous LE ' + dtype + ' ' + str((count, width)))
    if name != 'coordinates':
        if not np.isfinite(array).all():
            raise ValueError('Texture field floats must all be finite')
    else:
        if np.any(array[:, 0] != 0) or np.any(array[:, 1:] < 0) or np.any(array[:, 1:] > 511):
            raise ValueError('Texture coordinates require bounded batch-zero grid512 values')
        linear = (array[:, 1].astype(np.int64) * 512 + array[:, 2]) * 512 + array[:, 3]
        if len(np.unique(linear)) != count:
            raise ValueError('Texture coordinates must be unique; original row order is preserved')
    return array


def _transform(raw, features):
    expected = raw * np.float32(.5) + np.float32(.5)
    if not np.array_equal(features.view(np.uint32), expected.view(np.uint32)):
        raise ValueError('Texture features must preserve exact unclamped raw * .5 + .5')


def _provenance(value, count):
    if type(value) is not dict or value.get('mode') not in ('fixture', 'production'):
        raise ValueError('Texture field provenance must explicitly mark fixture or production mode')
    if value['mode'] == 'production' and count != MAX_ROWS:
        raise ValueError('Production texture field must contain the complete original final support')
    return _decode(_json_bytes(value, 48000))


def _statistics(tensors):
    return {name: {channel: {'min': float(array[:, i].min()), 'max': float(array[:, i].max()),
                            'mean': float(array[:, i].mean(dtype=np.float64))}
                   for i, channel in enumerate(CHANNELS)}
            for name, array in tensors.items() if name != 'coordinates'}


def _header_metadata(provenance, rows):
    return {'format': FORMAT, 'provenance': _json_bytes(provenance, 48000).decode('utf-8'),
            **{name + '_sha256': rows[name]['sha256'] for name in LAYOUTS}}


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_nlink


def _bounded_read(path, maximum):
    before = _plain(path)
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        if (_identity(before) != _identity(opened) or not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1 or opened.st_size > maximum):
            raise ValueError('Artifact exceeds byte bound or changed during open')
        blob = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    if (len(blob) > maximum or len(blob) != before.st_size
            or _identity(before) != _identity(after) or _identity(before) != _identity(_plain(path))):
        raise ValueError('Artifact exceeds byte bound or changed during read')
    return blob


def _container(blob):
    if not 8 < len(blob) <= MAX_FILE_BYTES:
        raise ValueError('Texture container exceeds file bound')
    length = struct.unpack('<Q', blob[:8])[0]
    if not 0 < length <= MAX_HEADER_BYTES or length % 8 or 8 + length >= len(blob):
        raise ValueError('Texture safetensors header length mismatch')
    header = _decode(blob[8:8 + length])
    if type(header) is not dict or set(header) != {'__metadata__', *LAYOUTS}:
        raise ValueError('Texture container must contain exactly the approved tensors')
    raw_descriptor = header['raw']
    if (type(raw_descriptor) is not dict or type(raw_descriptor.get('shape')) is not list
            or len(raw_descriptor['shape']) != 2 or type(raw_descriptor['shape'][0]) is not int
            or not 1 <= raw_descriptor['shape'][0] <= MAX_ROWS):
        raise ValueError('Texture field header row count exceeds bound')
    count = raw_descriptor['shape'][0]
    if len(blob) != 8 + length + count * 64:
        raise ValueError('Texture container exact payload length mismatch')
    payload = memoryview(blob)[8 + length:]
    rows, tensors, cursor = {}, {}, 0
    for name, (dtype, width) in LAYOUTS.items():
        size = count * width * 4
        expected = {'dtype': dtype, 'shape': [count, width], 'data_offsets': [cursor, cursor + size]}
        # Canonical bytes distinguish booleans/floats from required JSON integers.
        if _json_bytes(header[name]) != _json_bytes(expected):
            raise ValueError('Texture tensor layout/offset mismatch: ' + name)
        data = payload[cursor:cursor + size]
        array = np.frombuffer(data, dtype='<i4' if dtype == 'I32' else '<f4').reshape(count, width)
        tensors[name] = _array(name, array, count)
        rows[name] = dict(dtype=dtype, shape=[count, width], tensor_bytes=size, sha256=_hash(data))
        cursor += size
    _transform(tensors['raw'], tensors['features'])
    return header, rows, tensors, count


def _metadata(blob, rows, tensors, provenance, count):
    return dict(format=FORMAT, artifact_file=ARTIFACT_NAME, artifact_bytes=len(blob),
                artifact_sha256=_hash(blob), payload_bytes=count * 64, tensors=rows,
                channels=list(CHANNELS), channel_statistics=_statistics(tensors), provenance=provenance)


def verify(file, root, expected_metadata=None):
    """Read and validate every byte independently, with no Torch or AI imports."""
    root = approved_root(root)
    path = _owned_file(file, root, ARTIFACT_NAME, must_exist=True)
    sidecar = _owned_file(root / METADATA_NAME, root, METADATA_NAME, must_exist=True)
    blob = _bounded_read(path, MAX_FILE_BYTES)
    header, rows, tensors, count = _container(blob)
    metadata = _decode(_bounded_read(sidecar, MAX_METADATA_BYTES))
    if type(metadata) is not dict or 'provenance' not in metadata:
        raise ValueError('Texture sidecar fields mismatch')
    provenance = _provenance(metadata['provenance'], count)
    expected = _metadata(blob, rows, tensors, provenance, count)
    if (_json_bytes(metadata) != _json_bytes(expected)
            or _json_bytes(header['__metadata__']) != _json_bytes(_header_metadata(provenance, rows))):
        raise ValueError('Texture field payload hashes/statistics/provenance/sidecar mismatch')
    if expected_metadata is not None and _json_bytes(metadata) != _json_bytes(expected_metadata):
        raise ValueError('Texture field differs from expected saved metadata')
    return dict(verified=True, artifact_path=str(path), metadata_path=str(sidecar), metadata=metadata)


def _exclusive_write(path, blob):
    with path.open('xb') as stream:
        if os.fstat(stream.fileno()).st_nlink != 1:
            raise ValueError('Artifact exclusive file creation ownership changed')
        if stream.write(blob) != len(blob):
            raise OSError('Incomplete texture field write')
        stream.flush()
        os.fsync(stream.fileno())


def write(root, raw, features, coordinates, provenance):
    """Exclusively save CPU arrays, standard-reader roundtrip and byte readback."""
    root = approved_root(root)
    count = _row_count(raw)
    tensors = dict(raw=raw, features=features, coordinates=coordinates)
    for name, array in tensors.items():
        _array(name, array, count)
    _transform(raw, features)
    provenance = _provenance(provenance, count)
    data = {name: array.tobytes(order='C') for name, array in tensors.items()}
    rows, descriptors, cursor = {}, {}, 0
    for name, (dtype, width) in LAYOUTS.items():
        size = len(data[name])
        rows[name] = dict(dtype=dtype, shape=[count, width], tensor_bytes=size, sha256=_hash(data[name]))
        descriptors[name] = dict(dtype=dtype, shape=[count, width], data_offsets=[cursor, cursor + size])
        cursor += size
    header = _json_bytes({'__metadata__': _header_metadata(provenance, rows), **descriptors}, MAX_HEADER_BYTES)
    header += b' ' * (-len(header) % 8)
    if len(header) > MAX_HEADER_BYTES or cursor != count * 64 or cursor > MAX_PAYLOAD_BYTES:
        raise ValueError('Texture container exceeds exact payload/header bound')
    blob = struct.pack('<Q', len(header)) + header + b''.join(data.values())
    metadata = _metadata(blob, rows, tensors, provenance, count)
    metadata_blob = _json_bytes(metadata)
    path = _owned_file(root / ARTIFACT_NAME, root, ARTIFACT_NAME, must_exist=False)
    sidecar = _owned_file(root / METADATA_NAME, root, METADATA_NAME, must_exist=False)
    for candidate in (path, sidecar):
        if os.path.lexists(candidate):
            raise FileExistsError(candidate)
    # Fail before publication if the CPU standard reader is unavailable.
    from safetensors.numpy import load_file
    approved_root(root)
    _exclusive_write(path, blob)
    _exclusive_write(sidecar, metadata_blob)
    restored = load_file(str(path))
    if set(restored) != set(tensors) or any(
            restored[name].dtype != array.dtype or restored[name].shape != array.shape
            or restored[name].tobytes() != data[name] for name, array in tensors.items()):
        raise ValueError('Standard CPU safetensors tensor-byte roundtrip differs')
    result = verify(path, root, metadata)
    if (_bounded_read(path, MAX_FILE_BYTES) != blob
            or _bounded_read(sidecar, MAX_METADATA_BYTES) != metadata_blob):
        raise ValueError('Texture field byte-exact persisted readback differs')
    return result


write_texture_field_artifact = write
verify_texture_field_artifact = verify


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.verify, arguments.root), sort_keys=True, allow_nan=False))
