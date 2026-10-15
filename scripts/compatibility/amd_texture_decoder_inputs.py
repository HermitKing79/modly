"""CPU-only, byte-pinned texture latent and original positional C2S guides.

No Torch/model/native imports or writes. BF16 logits remain original read-only
uint16 payload views; F32 is an exact bit expansion for comparisons only. Runtime
code can copy uint16 bytes and view them as torch.bfloat16 without re-encoding.
"""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import struct
import sys

import numpy as np

import amd_texture_sampling_artifact as artifact

TEXTURE_PATH = 'C:/Users/wattz/Documents/ModlyArtifacts/amd-turret-texture-4f845cd/latent/texture-latent.safetensors'
GEOMETRY_PATH = 'C:/Users/wattz/Documents/ChatGPT/modly/artifacts/amd-wsl-turret-writer-ready-4870856/turret-neural-output.safetensors'
SOURCE_ROOT = 'D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD'
TEXTURE_SHA256 = '5506c4d0199bef64423b41b8bed560e2b7def9d29c5bb1c3072c0ea31ee1ddc4'
GEOMETRY_SHA256 = 'dc4667495ac4f8fd1457adca99563d83ee27c71fce0da2a6eb6698a8a1bd6e22'
TEXTURE_METADATA_SHA256 = '5d1d79c35c8c25e1765cccf4edc678335ad0cde3bd51a68590f7d9412ffae1bc'
GEOMETRY_METADATA_SHA256 = '8f428dea1c98ffabbea24a1986d388094a29bb9c9123a3a0dc48a55f9229a625'
SOURCE_SHA256 = {
    'trellis2/modules/sparse/spatial/spatial2channel.py': 'f36819339e48a89ed368f45feb0987309de23a4fe1d7d43eb347ab30a0b9f09c',
    'trellis2/models/sc_vaes/sparse_unet_vae.py': 'c9ce51db70bb1374c8bdc858129f6ff1170af1447e5210d434b7ebb289dbd2ce',
}
PARENT_ROWS = (3743, 17015, 79169, 378891)
CHILD_ROWS = (17015, 79169, 378891, 1842968)
GEOMETRY_LAYOUT = {'features': ('F32', (1842968, 7)), 'coordinates': ('I32', (1842968, 4)),
                   **{f'subdivision_{i}_coordinates': ('I32', (n, 4)) for i,n in enumerate(PARENT_ROWS)},
                   **{f'subdivision_{i}_logits': ('BF16', (n, 8)) for i,n in enumerate(PARENT_ROWS)}}
GEOMETRY_BYTES = 96433760
TEXTURE_BYTES = 1539912


def platform_path(value, *, platform=None):
    """Translate explicit Windows drive paths to mounted Linux paths, and back."""
    target_windows = (sys.platform if platform is None else platform).startswith('win')
    value = str(value).replace('\\', '/')
    if target_windows:
        match = re.fullmatch(r'/mnt/([a-zA-Z])/(.+)', value)
        if match: value = match[1].upper() + ':/' + match[2]
        result = PureWindowsPath(value)
    else:
        match = re.fullmatch(r'([a-zA-Z]):/(.+)', value)
        if match: value = '/mnt/' + match[1].lower() + '/' + match[2]
        result = PurePosixPath(value)
    if not result.is_absolute() or '..' in result.parts:
        raise ValueError('Absolute canonical input/source path required')
    return result


def _hash(blob):
    return hashlib.sha256(blob).hexdigest()


def _decode(blob):
    # Geometry provenance uses *_sha256 mappings as well as scalar hashes;
    # the latent writer's decoder intentionally permits only scalar suffixes.
    def nonfinite(_): raise ValueError('Nonfinite original JSON number')
    def check(value, depth=0):
        if depth > 16: raise ValueError('Original JSON nesting exceeds bound')
        if value is None or type(value) in (bool,str,int): return
        if type(value) is float and np.isfinite(value): return
        if type(value) in (list,dict):
            for item in (value.values() if type(value) is dict else value): check(item,depth+1)
            return
        raise ValueError('Invalid original JSON value')
    try:
        result = json.loads(blob,object_pairs_hook=artifact._unique,parse_constant=nonfinite)
        check(result)
        return result
    except (UnicodeError,json.JSONDecodeError,RecursionError) as exc:
        raise ValueError('Invalid original JSON') from exc


def _read(path, maximum, digest):
    path = Path(platform_path(path))
    if path != path.resolve(strict=True):
        raise ValueError('Input/source path must be resolved without aliases')
    # These are pinned read-only inputs. Linux / and /mnt ancestors are usually
    # root-owned, and mounted Windows files need not have the guest user's uid.
    # Validate type/aliases/links, without applying an output-writer uid rule.
    for candidate in (path,*path.parents):
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0) & 1024:
            raise ValueError('Input/source symlinks, junctions and reparse points are prohibited')
        if candidate == path:
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError('Input/source must be a plain file with one link')
        elif not stat.S_ISDIR(info.st_mode): raise ValueError('Input/source ancestor must be a plain directory')
    with path.open('rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > maximum:
            raise ValueError('Input/source exceeds plain-file byte bound')
        blob = stream.read(maximum+1)
    if len(blob) > maximum or len(blob) != info.st_size:
        raise ValueError('Input/source exceeds byte bound or changed during read')
    if _hash(blob) != digest:
        raise ValueError('Pinned input/source SHA256 differs: ' + str(path))
    return path, blob


def verify_source_identity(root=SOURCE_ROOT):
    root = Path(platform_path(root))
    return {name: _hash(_read(root / name, 131072, digest)[1]) for name,digest in SOURCE_SHA256.items()}


def bf16_to_f32(bits):
    """Exact BF16 bit expansion: no arithmetic rounding, no original-byte change."""
    if type(bits) is not np.ndarray or bits.dtype != np.dtype('<u2') or not bits.flags.c_contiguous:
        raise ValueError('Original contiguous little-endian BF16 uint16 payload required')
    result = (bits.astype('<u4') << 16).view('<f4')
    if not np.isfinite(result).all(): raise ValueError('Saved guide logits must be finite')
    result.flags.writeable = False
    return result


def _coords(value, *, rows=None, grid=None):
    if (type(value) is not np.ndarray or value.dtype != np.dtype('<i4') or value.ndim != 2
            or value.shape[1] != 4 or (rows is not None and value.shape[0] != rows)
            or not value.flags.c_contiguous or value.shape[0] > CHILD_ROWS[-1]):
        raise ValueError('Expected bounded contiguous original I32 [N,4] coordinates')
    if (np.any(value[:,0] != 0) or np.any(value[:,1:] < 0)
            or np.any(value[:,1:] >= (512 if grid is None else grid))):
        raise ValueError('Invalid batch or coordinate bounds')


def reconstruct_children(parent_coords, logits):
    """Exact pinned C2S order: nonzero(parent,child), x bit0/y bit1/z bit2.

    np.nonzero uses C row order, identical to the source's torch.nonzero on
    [parents,8]. Every selected pair is unique when original parents are unique.
    No coordinate sort or set conversion may be inserted here.
    """
    _coords(parent_coords, grid=256)
    if (type(logits) is not np.ndarray or logits.shape != (len(parent_coords),8)
            or logits.dtype != np.dtype('<f4') or not logits.flags.c_contiguous
            or not np.isfinite(logits).all()):
        raise ValueError('Expected finite exact F32-expanded [parents,8] guide logits')
    parent_index, child_index = np.nonzero(logits > 0)
    if len(parent_index) > CHILD_ROWS[-1]: raise ValueError('Selected child support exceeds approved bound')
    result = parent_coords[parent_index].copy()
    result[:,1:] *= 2
    for axis in range(3): result[:,axis+1] += ((child_index // (2**axis)) % 2).astype('<i4')
    result.flags.writeable = False
    return result


def _container(path, digest, expected_bytes, layout, payload_hashes):
    path, blob = _read(path, expected_bytes, digest)
    if len(blob) != expected_bytes: raise ValueError('Original container byte count differs')
    count = struct.unpack('<Q', blob[:8])[0]
    if not 0 < count <= 65536 or count % 8: raise ValueError('Bounded aligned safetensors header required')
    header = _decode(blob[8:8+count])
    if type(header) is not dict or set(header) != {'__metadata__', *layout}:
        raise ValueError('Original safetensors tensor set differs')
    meta = header['__metadata__']
    if type(meta) is not dict or any(type(k) is not str or type(v) is not str for k,v in meta.items()):
        raise ValueError('Safetensors metadata must be a string mapping')
    arrays, rows, cursor = {}, {}, 0
    if set(payload_hashes) != set(layout): raise ValueError('Original payload hash manifest differs')
    for name,(dtype,shape) in layout.items():
        size = math.prod(shape) * (2 if dtype == 'BF16' else 4)
        expected = dict(dtype=dtype, shape=list(shape), data_offsets=[cursor,cursor+size])
        if artifact._json_bytes(header[name]) != artifact._json_bytes(expected):
            raise ValueError('Original tensor dtype/shape/offsets differ: ' + name)
        data = memoryview(blob)[8+count+cursor:8+count+cursor+size]
        if len(data) != size or _hash(data) != payload_hashes[name]:
            raise ValueError('Original tensor payload SHA256 differs: ' + name)
        a = np.frombuffer(data, dtype={'BF16':'<u2','I32':'<i4','F32':'<f4'}[dtype]).reshape(shape)
        if dtype == 'F32' and not np.isfinite(a).all(): raise ValueError('Nonfinite original tensor: ' + name)
        arrays[name] = a
        rows[name] = dict(dtype=dtype, shape=list(shape), tensor_bytes=size, sha256=_hash(data))
        cursor += size
    if 8+count+cursor != len(blob): raise ValueError('Trailing/overlapping/gapped safetensors payload')
    return path, arrays, rows, meta


def read_inputs(texture_path=TEXTURE_PATH, geometry_path=GEOMETRY_PATH, source_root=SOURCE_ROOT):
    """Read only the two original artifacts and their byte-pinned sidecars.

    Returns dict(denormalized, coords, guides, geometry_coords, metadata). Guides
    are dicts with coords, original logits_bf16 uint16 views and exact logits_f32.
    All returned arrays are read-only. metadata is JSON-safe qualification proof.
    """
    sources = verify_source_identity(source_root)
    texture_path = Path(platform_path(texture_path)); geometry_path = Path(platform_path(geometry_path))
    _, tblob = _read(texture_path.with_name('texture-latent-metadata.json'),65536,TEXTURE_METADATA_SHA256)
    _, gblob = _read(geometry_path.with_name('provenance.json'),65536,GEOMETRY_METADATA_SHA256)
    tm = _decode(tblob); gm = _decode(gblob)
    tp, ta, tr, th = _container(texture_path,TEXTURE_SHA256,TEXTURE_BYTES,artifact.LAYOUTS,
                               {k:v['sha256'] for k,v in tm['tensors'].items()})
    gp, ga, gr, gh = _container(geometry_path,GEOMETRY_SHA256,GEOMETRY_BYTES,GEOMETRY_LAYOUT,
                               gm['output']['tensor_payload_sha256'])
    if (tm != dict(format=artifact.FORMAT, artifact_file=tp.name, artifact_bytes=TEXTURE_BYTES,
                   artifact_sha256=TEXTURE_SHA256, tensors=tr, provenance=tm['provenance'])
            or th != artifact._header_metadata(tm['provenance'],tr)):
        raise ValueError('Texture provenance and payload sidecar differ')
    if (set(gh) != {'summary','provenance'} or _decode(gh['provenance']) != gm['provenance']
            or gm['output']['sha256'] != GEOMETRY_SHA256 or gm['output']['bytes'] != GEOMETRY_BYTES
            or any(gm['output'].get(k) != v for k,v in _decode(gh['summary']).items())):
        raise ValueError('Geometry provenance and payload sidecar differ')
    saved_sources = gm['provenance']['source']
    if any(saved_sources.get(str(platform_path(SOURCE_ROOT, platform='linux') / name)) != digest
           for name,digest in sources.items()):
        raise ValueError('Saved geometry C2S/decoder source identity differs')
    _coords(ta['coords'],rows=PARENT_ROWS[0],grid=32)
    linear = (ta['coords'][:,1].astype(np.int64)*32 + ta['coords'][:,2])*32 + ta['coords'][:,3]
    if np.any(np.diff(linear) <= 0): raise ValueError('Original initial support must be unique and ordered')
    if (tm['provenance']['input']['coords']['sha256'] != tr['coords']['sha256']
            or gm['provenance']['inputs']['coordinates_payload_sha256'] != tr['coords']['sha256']):
        raise ValueError('Saved shape and texture latent coordinate provenance differ')
    guides, proof, parent_expected = [], [], ta['coords']
    for i,n in enumerate(PARENT_ROWS):
        coords = ga[f'subdivision_{i}_coordinates']; bits = ga[f'subdivision_{i}_logits']
        _coords(coords,rows=n,grid=32*2**i)
        values = bf16_to_f32(bits)
        if not np.array_equal(coords,parent_expected): raise ValueError('Original guide parent order differs')
        children = reconstruct_children(coords,values)
        expected = ga[f'subdivision_{i+1}_coordinates'] if i < 3 else ga['coordinates']
        _coords(expected,rows=CHILD_ROWS[i],grid=64*2**i)
        if not np.array_equal(children,expected): raise ValueError('Exact original C2S child coordinate order differs')
        parent_expected = children
        guides.append(dict(coords=coords,logits_bf16=bits,logits_f32=values))
        proof.append(dict(stage=i,parents=n,selected_children=len(children),threshold='strict > 0',
                          logits_dtype='BF16',logits_sha256=gr[f'subdivision_{i}_logits']['sha256'],
                          parent_coords_sha256=_hash(coords),child_coords_sha256=_hash(children),
                          child_order_exact=True,original_bf16_bytes_preserved=True))
    metadata = dict(verified=True,cpu_only=True,parent_rows=list(PARENT_ROWS),child_rows=list(CHILD_ROWS),
                    texture_path=str(tp),texture_sha256=TEXTURE_SHA256,texture_sidecar_sha256=TEXTURE_METADATA_SHA256,
                    geometry_path=str(gp),geometry_sha256=GEOMETRY_SHA256,geometry_sidecar_sha256=GEOMETRY_METADATA_SHA256,
                    sources=sources,texture_tensors=tr,geometry_tensors=gr,guides=proof,
                    coordinate_order='original C2S parent-major; child x bit0, y bit1, z bit2',
                    final_geometry_coordinate_order_exact=True,model_imported=False,gpu_initialized=False)
    return dict(denormalized=ta['denormalized'],coords=ta['coords'],guides=tuple(guides),
                geometry_coords=ga['coordinates'],metadata=metadata)


class GuideOrderValidator:
    """Runtime hook contract for each real C2S call (h and skip, at four stages).

    Caller copies live coords and binarized guide feats to CPU without casting
    their dtype. Checks occur BEFORE invoking original C2S, and AFTER it returns.
    Existing C2S spatial caches are forbidden: that branch ignores guide masks.
    No model/GPU action is performed by this validator.
    """
    def __init__(self, inputs):
        if inputs['metadata'].get('verified') is not True: raise ValueError('Qualified inputs required')
        self.guides = inputs['guides']; self.geometry_coords = inputs['geometry_coords']
        self.calls = [0,0,0,0]; self.pending = None

    def before_c2s(self, stage, live_parent_coords, guide_mask, guide_coords, *, cached=False):
        if (type(stage) is not int or not 0 <= stage < 4 or self.pending is not None
                or sum(self.calls) // 2 != stage or self.calls[stage] >= 2 or cached is not False):
            raise ValueError('C2S stage/call order/cache contract differs')
        expected = self.guides[stage]
        _coords(live_parent_coords,rows=PARENT_ROWS[stage],grid=32*2**stage)
        _coords(guide_coords,rows=PARENT_ROWS[stage],grid=32*2**stage)
        if (not np.array_equal(live_parent_coords,expected['coords'])
                or not np.array_equal(guide_coords,expected['coords'])):
            raise ValueError('Live C2S parent/guide coordinate row order differs')
        if (type(guide_mask) is not np.ndarray or guide_mask.dtype != np.dtype('bool')
                or not guide_mask.flags.c_contiguous
                or not np.array_equal(guide_mask,expected['logits_f32'] > 0)):
            raise ValueError('Live C2S strict >0 original guide mask differs')
        self.pending = stage

    def after_c2s(self, stage, live_child_coords):
        if type(stage) is not int or self.pending != stage: raise ValueError('Unpaired C2S result')
        expected = self.guides[stage+1]['coords'] if stage < 3 else self.geometry_coords
        _coords(live_child_coords,rows=CHILD_ROWS[stage],grid=64*2**stage)
        if not np.array_equal(live_child_coords,expected): raise ValueError('Live C2S child row order differs')
        self.calls[stage] += 1; self.pending = None

    def finalize(self, live_geometry_coords):
        if self.pending is not None or self.calls != [2,2,2,2]: raise ValueError('All eight original C2S calls required')
        _coords(live_geometry_coords,rows=CHILD_ROWS[-1],grid=512)
        if not np.array_equal(live_geometry_coords,self.geometry_coords): raise ValueError('Final geometry alignment differs')
        return dict(verified=True,calls=list(self.calls),final_geometry_order_exact=True)


if __name__ == '__main__':
    import json
    print(json.dumps(read_inputs()['metadata'],sort_keys=True,allow_nan=False))
