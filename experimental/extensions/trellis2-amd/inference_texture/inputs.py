"""Genuine saved brick texture/guides through the original CPU readers."""
import ast
import hashlib
import os
from pathlib import Path
import stat
import struct
import numpy as np
from . import profile as profiles
from ._source import ROOT, load_pinned

TEXTURE_PATH = Path('C:/Users/wattz/Documents/ModlyArtifacts/amd-brick-material-stage1-df41688/latent/texture-latent.safetensors')
GEOMETRY_PATH = ROOT.parent / 'artifacts/amd-brick-neural-completion-c9447dd/neural-output/shape-decoded.safetensors'
_LATENT_SOURCE = 'scripts/compatibility/amd_texture_sampling_artifact.py'
_INPUT_SOURCE = 'scripts/compatibility/amd_texture_decoder_inputs.py'
_GEOMETRY_SOURCE = 'experimental/extensions/trellis2-amd/inference_decoder/artifact_io.py'
_LOADER_SOURCE = 'experimental/extensions/trellis2-amd/texture_decoder_loader.py'
_latent = load_pinned('input_latent', _LATENT_SOURCE, profiles.PINS[_LATENT_SOURCE])
_original = load_pinned('input_reader', _INPUT_SOURCE, profiles.PINS[_INPUT_SOURCE],
                        imports={'amd_texture_sampling_artifact': _latent})
_geometry = load_pinned('input_geometry', _GEOMETRY_SOURCE, profiles.PINS[_GEOMETRY_SOURCE],
                        imports={'run_contract': profiles.contracts})
_guide = load_pinned('brick_guide_validator', _INPUT_SOURCE, profiles.PINS[_INPUT_SOURCE],
                    imports={'amd_texture_sampling_artifact': _latent})
# Only this private copy uses the reauthenticated explicit profile. The original
# imported generic module and all historical files/constants remain untouched.
_guide.PARENT_ROWS = profiles.PARENT_ROWS
_guide.CHILD_ROWS = profiles.CHILD_ROWS
_COORDINATE_HASHES = (
    profiles.INITIAL_COORDINATES_SHA256,
    '16ecc9780a85d722e83fdf52ed3841e7038a69d203f086a0524094a63f18a594',
    '8f1c7d4bfb4f031965f2e095929fa592b394eb60841d02a786805e08f707d81d',
    'ab09fb4a330c0a07aead43193aeec79e6ea9b1efa85c62aeb4ca810a0f79d19e')
_LOGIT_HASHES = (
    '9c31de9c275fa78590743bb9ec7bf0accc3cab7f92e4f3096e99162d8963f07e',
    'c519845ae44a8e30a83038de03578b7cecd8aaf2fcebdb6c1e437e20208a538a',
    '6f74789f5076a42399ec6cf6048a792d1e4953fac3294597bceb717a74e965ba',
    '285f0adba1078694c3112a2ef8c0afbeb0779e951f65ffdaf48cd399cc524304')


def _checkpoint_and_sources():
    # Read the pinned loader as data; importing it could load unrelated modules.
    _, blob = _original._read(ROOT / _LOADER_SOURCE, 131072, profiles.PINS[_LOADER_SOURCE])
    tree = ast.parse(blob)
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in ('SOURCE_SHA256', 'CONFIG_SHA256', 'HEADER_SHA256')}
    if (constants['CONFIG_SHA256'] != profiles.CONFIG_SHA256
            or constants['HEADER_SHA256'] != profiles.HEADER_SHA256):
        raise ValueError('Pinned original texture loader identities differ')
    source_root = Path(_original.platform_path(_original.SOURCE_ROOT))
    sources = {name: _original._hash(_original._read(source_root / 'trellis2' / name, 131072, digest)[1])
               for name, digest in constants['SOURCE_SHA256'].items()}
    prefix = Path(_original.platform_path('D:/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16'))
    _original._read(prefix.with_suffix('.json'), 65536, profiles.CONFIG_SHA256)
    checkpoint = prefix.with_suffix('.safetensors')
    # Header only: never read/deserialise a checkpoint tensor or construct a model.
    for path in (checkpoint, *checkpoint.parents):
        info = path.lstat()
        if path.is_symlink() or getattr(info, 'st_file_attributes', 0) & 1024:
            raise ValueError('Checkpoint aliases are prohibited')
    before = checkpoint.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != 948458812:
        raise ValueError('Original bounded checkpoint file identity differs')
    with checkpoint.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink)
                != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_nlink)):
            raise ValueError('Original checkpoint changed during open')
        prefix_bytes = stream.read(8)
        if len(prefix_bytes) != 8 or struct.unpack('<Q', prefix_bytes)[0] != 26792:
            raise ValueError('Original texture checkpoint header length differs')
        header = prefix_bytes + stream.read(26792)
    after = checkpoint.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or hashlib.sha256(header).hexdigest() != profiles.HEADER_SHA256):
        raise ValueError('Original texture checkpoint header identity differs')
    return sources


def _array(value, dtype, shape, digest):
    if (type(value) is not np.ndarray or value.dtype != np.dtype(dtype)
            or value.shape != shape or not value.flags.c_contiguous
            or value.flags.writeable or hashlib.sha256(value).hexdigest() != digest):
        raise ValueError('Prepared original payload identity differs')


def require_prepared(prepared, profile):
    profile = profiles.require_authenticated(profile)
    if (type(prepared) is not dict or prepared.get('profile') != profile
            or prepared.get('metadata', {}).get('profile') != profile.document()
            or prepared['metadata'].get('profile_sha256') != profile.sha256
            or prepared['metadata'].get('verified') is not True):
        raise ValueError('Authenticated prepared brick inputs required')
    metadata = prepared['metadata']
    expected_metadata = dict(run_contract_sha256=profile.run_contract_sha256,
        texture_sha256=profiles.TEXTURE_SHA256, geometry_sha256=profiles.GEOMETRY_SHA256,
        texture_sidecar_sha256=profiles.TEXTURE_METADATA_SHA256,
        geometry_sidecar_sha256=profiles.GEOMETRY_METADATA_SHA256,
        parent_rows=list(profile.parent_rows), child_rows=list(profile.child_rows),
        sources=dict(_original.SOURCE_SHA256),
        all_four_masks_and_original_child_orders_verified=True, guide_BF16_bytes_preserved=True,
        final_geometry_coordinate_order_exact=True)
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError('Prepared authenticated metadata identity differs')
    _array(prepared['denormalized'], '<f4', (4430, 32),
           'ad6e31cc89b0b94604aa0a9f61b6034eb00005ec82505e93ca2e16d572642648')
    _array(prepared['coords'], '<i4', (4430, 4), profiles.INITIAL_COORDINATES_SHA256)
    _array(prepared['geometry_coords'], '<i4', (1957891, 4), profiles.FINAL_COORDINATES_SHA256)
    if type(prepared['guides']) is not tuple or len(prepared['guides']) != 4:
        raise ValueError('Exactly four genuine guides required')
    for i, g in enumerate(prepared['guides']):
        _array(g['coords'], '<i4', (profile.parent_rows[i], 4), _COORDINATE_HASHES[i])
        _array(g['logits_bf16'], '<u2', (profile.parent_rows[i], 8), _LOGIT_HASHES[i])
        f32 = g['logits_f32']
        if (type(f32) is not np.ndarray or f32.dtype != np.dtype('<f4') or f32.flags.writeable
                or not f32.flags.c_contiguous
                or not np.array_equal(f32.view('<u4'), g['logits_bf16'].astype('<u4') << 16)):
            raise ValueError('Exact read-only BF16 expansion required')
    return prepared


def validate_prepared(prepared, *, profile):
    """Public consumer boundary: authenticate arrays, not a metadata boolean."""
    return require_prepared(prepared, profile)


def prepare(profile, texture_path=None, geometry_path=None):
    profile = profiles.require_authenticated(profile)
    texture_path = TEXTURE_PATH if texture_path is None else Path(texture_path)
    geometry_path = GEOMETRY_PATH if geometry_path is None else Path(geometry_path)
    texture_path = Path(_original.platform_path(texture_path))
    geometry_path = Path(_original.platform_path(geometry_path))
    contract = profiles.contracts.parse(profile.document()['geometry_contract'])
    _, tblob = _original._read(texture_path.with_name('texture-latent-metadata.json'), 65536, profiles.TEXTURE_METADATA_SHA256)
    _, gblob = _original._read(geometry_path.with_name('provenance.json'), 65536, profiles.GEOMETRY_METADATA_SHA256)
    tm, gm = _original._decode(tblob), _original._decode(gblob)
    verified = _latent.verify(texture_path, texture_path.parent, token_count=4430, maximum_tokens=4430)
    if verified['metadata'] != tm or (tm['artifact_sha256'], tm['artifact_bytes']) != (profiles.TEXTURE_SHA256, 1814784):
        raise ValueError('Exact genuine brick texture latent required')
    tp, ta, tr, _ = _original._container(texture_path, profiles.TEXTURE_SHA256, 1814784,
        _latent.layouts(4430, maximum_tokens=4430), {k: v['sha256'] for k, v in tm['tensors'].items()})
    layout = {'features': ('F32', (1957891, 7)), 'coordinates': ('I32', (1957891, 4)),
              **{f'subdivision_{i}_coordinates': ('I32', (n, 4)) for i, n in enumerate(profile.parent_rows)},
              **{f'subdivision_{i}_logits': ('BF16', (n, 8)) for i, n in enumerate(profile.parent_rows)}}
    gp, ga, gr, header = _original._container(geometry_path, profiles.GEOMETRY_SHA256, 104148612,
                                            layout, gm['output']['tensor_payload_sha256'])
    summary = _original._decode(header['summary'])
    if (gm['run_contract'] != contract.document() or _original._decode(header['provenance']) != gm['provenance']
            or gm['output']['sha256'] != profiles.GEOMETRY_SHA256
            or any(gm['output'].get(k) != v for k, v in summary.items())):
        raise ValueError('Saved geometry sidecar/header/run contract differs')
    _geometry.verify_file(gp, gm['provenance'], summary, gm['output']['tensor_payload_sha256'], contract)
    sources = _original.verify_source_identity()
    if any(gm['provenance']['source'].get(str(_original.platform_path(_original.SOURCE_ROOT, platform='linux') / name)) != sha
           for name, sha in sources.items()):
        raise ValueError('Saved guide source identities differ')
    decoder_sources = _checkpoint_and_sources()
    guides = tuple(dict(coords=ga[f'subdivision_{i}_coordinates'],
                        logits_bf16=ga[f'subdivision_{i}_logits'],
                        logits_f32=_original.bf16_to_f32(ga[f'subdivision_{i}_logits'])) for i in range(4))
    if (not np.array_equal(ta['coords'], guides[0]['coords'])
            or tm['provenance']['input']['coords']['sha256'] != tr['coords']['sha256']
            or gm['provenance']['inputs']['coordinates_payload_sha256'] != tr['coords']['sha256']):
        raise ValueError('Original texture/geometry coordinate provenance differs')
    profiles.contracts.validate_chain([(g['coords'], g['logits_f32']) for g in guides], ga['coordinates'], 4430, contract=contract)
    metadata = dict(verified=True, cpu_only=True, profile=profile.document(), profile_sha256=profile.sha256,
                    run_contract_sha256=profile.run_contract_sha256, parent_rows=list(profile.parent_rows),
                    child_rows=list(profile.child_rows), texture_path=str(tp), texture_sha256=profiles.TEXTURE_SHA256,
                    texture_sidecar_sha256=profiles.TEXTURE_METADATA_SHA256, geometry_path=str(gp),
                    geometry_sha256=profiles.GEOMETRY_SHA256, geometry_sidecar_sha256=profiles.GEOMETRY_METADATA_SHA256,
                    texture_tensors=tr, geometry_tensors=gr, sources=sources, decoder_sources=decoder_sources,
                    coordinate_order='original C2S parent-major; child x bit0, y bit1, z bit2',
                    all_four_masks_and_original_child_orders_verified=True, guide_BF16_bytes_preserved=True,
                    final_geometry_coordinate_order_exact=True, checkpoint_header_only=True,
                    model_imported=False, gpu_initialized=False)
    return require_prepared(dict(denormalized=ta['denormalized'], coords=ta['coords'], guides=guides,
                                 geometry_coords=ga['coordinates'], metadata=metadata, profile=profile), profile)


class GuideOrderValidator:
    """Historical eight-call interface, revalidated against the exact brick data."""
    def __init__(self, prepared, *, profile=None):
        profile = prepared.get('profile') if profile is None else profile
        require_prepared(prepared, profile)
        if (_guide.PARENT_ROWS != profiles.PARENT_ROWS or _guide.CHILD_ROWS != profiles.CHILD_ROWS):
            raise ValueError('Private guide bounds changed')
        self._validator = _guide.GuideOrderValidator(prepared)

    @property
    def calls(self):
        return self._validator.calls

    def before_c2s(self, stage, live_parent_coords, guide_mask, guide_coords, *, cached=False):
        return self._validator.before_c2s(stage, live_parent_coords, guide_mask, guide_coords, cached=cached)

    def after_c2s(self, stage, live_child_coords):
        return self._validator.after_c2s(stage, live_child_coords)

    def finalize(self, live_geometry_coords):
        return self._validator.finalize(live_geometry_coords)
