"""Authenticated brick opt-in around the unchanged genuine field serializer.

Writer and independent NumPy reader are separate hash-pinned module instances.
Only fixed byte bounds change within their private namespaces. All original
format, ownership, confinement, finite/transform, exclusive-write and payload
verification functions execute unchanged.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from . import profile as profiles
from ._source import ROOT, load_pinned

_SOURCE = 'scripts/compatibility/amd_texture_field_artifact.py'
_SHA = profiles.PINS[_SOURCE]
_reference = load_pinned('field_reference', _SOURCE, _SHA)
_PUBLIC_NAME = 'amd_texture_field_artifact'
_CANONICAL_OWNERSHIP_CLASS = None


def _canonical_field():
    """Load/authenticate the actual public module, never replace an occupant.

    The original ownership factory and its exact original class stay together
    in their canonical namespace. Private field copies share that single class.
    This does not authorize a root or invent an ownership attestation.
    """
    path = ROOT / _SOURCE
    # Reuse the source loader's plain-file/hash checks, then compile the exact
    # independently rechecked bytes without writing historical bytecode files.
    load_pinned('field_reference', _SOURCE, _SHA)
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != _SHA:
        raise ValueError('Canonical ownership source SHA256 differs')
    module = sys.modules.get(_PUBLIC_NAME)
    if module is None:
        spec = importlib.util.spec_from_file_location(_PUBLIC_NAME, path)
        candidate = importlib.util.module_from_spec(spec)
        module = sys.modules.setdefault(_PUBLIC_NAME, candidate)
        if module is candidate:
            try:
                exec(compile(blob, str(path), 'exec'), vars(module))
            except BaseException:
                if sys.modules.get(_PUBLIC_NAME) is module:
                    del sys.modules[_PUBLIC_NAME]
                raise
    if (type(module) is not ModuleType or module.__name__ != _PUBLIC_NAME
            or Path(getattr(module, '__file__', '')).resolve() != path):
        raise ValueError('Foreign canonical field module origin')
    cls = getattr(module, '_DrvfsOwnership', None)
    if (type(cls) is not type or cls.__module__ != _PUBLIC_NAME
            or cls.__qualname__ != '_DrvfsOwnership'
            or getattr(getattr(cls, 'check', None), '__code__', None) != _reference._DrvfsOwnership.check.__code__
            or getattr(getattr(cls, 'check', None), '__globals__', None) is not vars(module)
            or (_CANONICAL_OWNERSHIP_CLASS is not None and cls is not _CANONICAL_OWNERSHIP_CLASS)):
        raise ValueError('Foreign canonical ownership class implementation')
    for name in ('_owner', '_plain', '_drvfs_mount', 'drvfs_ownership', 'approved_root'):
        actual = getattr(module, name, None)
        expected = getattr(_reference, name)
        if (not callable(actual) or getattr(actual, '__code__', None) != expected.__code__
                or getattr(actual, '__globals__', None) is not vars(module)):
            raise ValueError('Canonical ownership callable implementation differs: ' + name)
    for name in ('os', 'stat', 'Path'):
        if getattr(module, name, None) is not getattr(_reference, name):
            raise ValueError('Canonical ownership dependency differs: ' + name)
    for name in ('MAX_ROWS', 'MAX_PAYLOAD_BYTES', 'MAX_FILE_BYTES', 'MAX_HEADER_BYTES',
                 'MAX_METADATA_BYTES', 'ARTIFACT_NAME', 'METADATA_NAME'):
        if getattr(module, name, None) != getattr(_reference, name):
            raise ValueError('Canonical historical field constant differs: ' + name)
    return module


_historical = _canonical_field()
_CANONICAL_OWNERSHIP_CLASS = _historical._DrvfsOwnership
ARTIFACT_NAME = _historical.ARTIFACT_NAME
METADATA_NAME = _historical.METADATA_NAME
CHANNELS = _historical.CHANNELS
FORMAT = _historical.FORMAT
approved_root = _historical.approved_root
drvfs_ownership = _historical.drvfs_ownership
_RESERVED = {'pbr_profile_document', 'pbr_profile_sha256'}


def _field(kind, profile):
    profiles.require_authenticated(profile)
    canonical = _canonical_field()
    module = load_pinned('field_brick_' + kind, _SOURCE, _SHA)
    original = (1842968, 1842968 * 64, 8 + 65536 + 1842968 * 64)
    qualified = (profile.final_rows, profile.final_rows * 64, 8 + 65536 + profile.final_rows * 64)
    bounds = (module.MAX_ROWS, module.MAX_PAYLOAD_BYTES, module.MAX_FILE_BYTES)
    if bounds == original:
        module.MAX_ROWS, module.MAX_PAYLOAD_BYTES, module.MAX_FILE_BYTES = qualified
        # Ownership remains the exact genuine capability, never a duck-typed
        # stand-in; both isolated modules accept that same original class.
        module._DrvfsOwnership = canonical._DrvfsOwnership
    if ((module.MAX_ROWS, module.MAX_PAYLOAD_BYTES, module.MAX_FILE_BYTES) != qualified
            or module.MAX_HEADER_BYTES != 65536 or module.MAX_METADATA_BYTES != 65536
            or module._DrvfsOwnership is not canonical._DrvfsOwnership):
        raise ValueError('Private fixed profile field bounds/ownership changed')
    return module


def _bind(provenance, profile):
    profiles.require_authenticated(profile)
    if type(provenance) is not dict:
        raise ValueError('Exact field provenance mapping required')
    bound = dict(provenance)
    if bound.get('mode') == 'fixture':
        if bound.get('synthetic') is not True or type(bound.get('purpose')) is not str or not bound['purpose']:
            raise ValueError('Profile fixtures require explicit synthetic identity and purpose')
    elif bound.get('mode') == 'production':
        if bound.get('synthetic', False) is not False:
            raise ValueError('Production field cannot claim synthetic model output')
    else:
        raise ValueError('Explicit production or synthetic fixture mode required')
    expected = dict(pbr_profile_document=profiles._canonical(profile.document()).decode('utf-8'),
                    pbr_profile_sha256=profile.sha256, run_contract_sha256=profile.run_contract_sha256,
                    texture_sha256=profiles.TEXTURE_SHA256, geometry_sha256=profiles.GEOMETRY_SHA256,
                    checkpoint_sha256=profiles.CHECKPOINT_SHA256, config_sha256=profiles.CONFIG_SHA256,
                    guidance_parent_rows=list(profile.parent_rows), guidance_child_rows=list(profile.child_rows))
    for key, value in expected.items():
        if key in bound and bound[key] != value:
            raise ValueError('Field authenticated profile binding differs: ' + key)
        bound[key] = value
    # The original field JSON validator accepts scalar *_sha256 values. Keep
    # the complete profile (including its hash mappings) as canonical JSON text.
    if type(bound.get('inputs')) is dict and 'profile' in bound['inputs']:
        inputs = dict(bound['inputs'])
        if inputs['profile'] == profile.document():
            inputs['profile'] = expected['pbr_profile_document']
        elif inputs['profile'] != expected['pbr_profile_document']:
            raise ValueError('Field input profile binding differs')
        bound['inputs'] = inputs
    return bound


def write(root, raw, features, coordinates, provenance, *, authority=None, profile=None):
    """Original write signature, with one authenticated profile keyword."""
    if profile is None:
        _canonical_field()
        if type(provenance) is dict and _RESERVED.intersection(provenance):
            raise ValueError('Profile-bound fields require authenticated writer opt-in')
        return _historical.write(root, raw, features, coordinates, provenance, authority=authority)
    profile = profiles.require_authenticated(profile)
    module = _field('writer', profile)
    bound = _bind(provenance, profile)
    count = module._row_count(raw)
    if bound['mode'] == 'production':
        if count != profile.final_rows:
            raise ValueError('Production profile requires complete original final support')
        module._array('coordinates', coordinates, count)
        if hashlib.sha256(coordinates).hexdigest() != profiles.FINAL_COORDINATES_SHA256:
            raise ValueError('Production profile requires exact original geometry coordinate order')
    result = module.write(root, raw, features, coordinates, bound, authority=authority)
    # Independent source instance consumes persisted bytes after writer readback.
    return verify(result['artifact_path'], root, result['metadata'], authority=authority, profile=profile)


def verify(file, root=None, expected_metadata=None, *, authority=None, profile=None):
    """Independent complete CPU readback, requiring its own authenticated opt-in."""
    root = Path(file).parent if root is None else root
    if profile is None:
        _canonical_field()
        result = _historical.verify(file, root, expected_metadata, authority=authority)
        if _RESERVED.intersection(result['metadata']['provenance']):
            raise ValueError('Profile-bound artifact requires authenticated independent reader opt-in')
        return result
    profile = profiles.require_authenticated(profile)
    module = _field('reader', profile)
    result = module.verify(file, root, expected_metadata, authority=authority)
    metadata = result['metadata']
    provenance = metadata['provenance']
    if provenance != _bind(provenance, profile):
        raise ValueError('Persisted field authenticated profile metadata missing or differs')
    if provenance['mode'] == 'production' and (
            metadata['tensors']['coordinates']['shape'] != [profile.final_rows, 4]
            or metadata['tensors']['coordinates']['sha256'] != profiles.FINAL_COORDINATES_SHA256):
        raise ValueError('Independent reader requires complete original geometry coordinate order')
    return result


write_texture_field_artifact = write
verify_texture_field_artifact = verify
