"""One experiment's read-only Windows/DrvFS boundary; historical readers stay intact.

The real inputs.prepare() is compiled from its pinned original bytes. Only its
private latent reader's filesystem predicate/read function receives this exact
five-file authority. Container, provenance, guide and checkpoint checks are the
original implementations. No writer or model-execution capability is granted.
"""
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import stat
import sys
from types import ModuleType

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STAGE = Path('/home/modly/native-decoder-preparation-6dd94a85/site-packages')
SID = 'S-1-5-21-3582664715-2766927308-497596791-1001'
SCHEMA = 'modly.windows-exact-input-attestation.v1'
OWNER_SHA = '6849a3ac5375cc1693ce306ea28fa4ce9bed32305b8926bc2f5e6034c75f1a94'
TEXTURE_ROOT = Path('/mnt/c/Users/wattz/Documents/ModlyArtifacts/amd-brick-material-stage1-df41688/latent')
GEOMETRY_ROOT = Path('/mnt/c/Users/wattz/Documents/ChatGPT/modly/artifacts/amd-brick-neural-completion-c9447dd/neural-output')
INPUTS = {
    'texture': (TEXTURE_ROOT/'texture-latent.safetensors', '4c74343c961e74bf3cfde4c24bd2a21b23063e17f0468b7f3049311fa9dc4289', 1814784),
    'texture_metadata': (TEXTURE_ROOT/'texture-latent-metadata.json', '25adb1a50b8b31900ebc56d87f45b7fe23dee9305d97b458f70a60799d59e2a0', 39304),
    'geometry': (GEOMETRY_ROOT/'shape-decoded.safetensors', 'd4de05fc82315e44a739c9815a07a531555e108af05c619ea0426d0c630ef861', 104148612),
    'provenance': (GEOMETRY_ROOT/'provenance.json', 'a42340cef604134b2866a3556f6d74c5f631073eb908fb60a64d90139fac6407', 26328),
    'run_contract': (GEOMETRY_ROOT/'run-contract.json', '4676626d1061e6cd7090b560f952a974a584f8ea9e6218378b7902294b93bd4f', 1449),
}
CORE = {
    'experimental/extensions/trellis2-amd/inference_texture/inputs.py': '603851fd622b4e53d9d484bd0c08d201b4b43b35a3abc1650b38b094a3280ed3',
    'experimental/extensions/trellis2-amd/inference_texture/profile.py': '4cbb0ca7f201201bf8bbd8cd093575eec89868480cb47ed00d4078915bb5b368',
    'experimental/extensions/trellis2-amd/inference_texture/_source.py': '11ae08f720b1f09414bd3ce6ce5fe8ac166f9bc1f712e7768b2db7399652159c',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns)


def read_plain(path, maximum):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve(strict=True), 'Canonical read-only path required')
    for component in (path, *path.parents):
        info = component.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
                'Input aliases/reparse points prohibited')
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 if component == path
                else stat.S_ISDIR(info.st_mode), 'Plain single-link input and directory chain required')
    before = path.lstat()
    require(type(maximum) is int and 0 < before.st_size <= maximum <= 268566528,
            'Bounded nonempty input read required')
    with path.open('rb') as stream:
        require(identity(os.fstat(stream.fileno())) == identity(before), 'Input changed before read')
        blob = stream.read(maximum+1)
        require(identity(os.fstat(stream.fileno())) == identity(before), 'Input changed during read')
    require(identity(path.lstat()) == identity(before) and len(blob) == before.st_size,
            'Input identity or byte count changed')
    return blob


def validate_attestation(document):
    require(type(document) is dict and document.get('schema') == SCHEMA
            and document.get('current_sid') == SID, 'Exact Windows owner attestation required')
    require(document.get('trusted_input_roots') == [str(TEXTURE_ROOT), str(GEOMETRY_ROOT)]
            and document.get('observer', {}).get('sha256') == OWNER_SHA, 'Exact Windows trust roots/observer required')
    rows = document.get('files')
    require(type(rows) is list and len(rows) == 5, 'Five exact authenticated inputs required')
    result, file_ids = {}, set()
    trusted = {SID, 'S-1-5-18', 'S-1-5-32-544'}
    installer = 'S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464'
    for row in rows:
        require(type(row) is dict and row.get('label') in INPUTS and row['label'] not in result,
                'Unknown or repeated input identity')
        path, sha, size = INPUTS[row['label']]
        windows = PureWindowsPath('C:/'+str(path)[7:])
        require((row.get('guest_path'), row.get('windows_path'), row.get('sha256'), row.get('bytes'))
                == (str(path), str(windows), sha, size), 'Exact path/hash/size input binding differs')
        require(row.get('owner_sid') == SID and row.get('current_sid') == SID,
                'Windows input ownership mismatch')
        fid = row.get('windows_identity', {})
        read_identity = row.get('read_identity', {})
        require(all(type(fid.get(k)) is int and fid[k] > 0 for k in ('dev', 'ino', 'mtime_ns'))
                and type(fid.get('nlink')) is int and fid['nlink'] == 1
                and read_identity.get('stable') is True
                and all(read_identity.get(k) == fid for k in ('before', 'opened', 'after_open', 'after')),
                'Stable Windows single-link file ID required')
        key = (fid['dev'], fid['ino'])
        require(key not in file_ids, 'Windows input identity alias')
        file_ids.add(key)
        ancestors = row.get('ancestors')
        expected = [str(windows), *map(str, windows.parents)]
        require(type(ancestors) is list and [a.get('path') for a in ancestors] == expected,
                'Complete canonical Windows ancestor chain required')
        for index, ancestor in enumerate(ancestors):
            attrs = ancestor.get('attributes')
            require(ancestor.get('owner_sid') in ({SID} if index == 0 else trusted | {installer})
                    and ancestor.get('reparse') is False and not ancestor.get('link_type')
                    and not ancestor.get('target') and type(attrs) is int and not attrs & 1024
                    and bool(attrs & 16) == (index != 0), 'Unsafe Windows ancestor ownership/type')
            rules = ancestor.get('rules')
            require(type(rules) is list and bool(rules) and bool(ancestor.get('sddl')), 'Windows DACL evidence required')
            for rule in rules:
                require(rule.get('type') in ('Allow', 'Deny') and type(rule.get('rights')) is int
                        and type(rule.get('sid')) is str, 'Malformed Windows DACL')
                if rule['type'] != 'Allow' or 'InheritOnly' in rule.get('propagation', '') or rule['sid'] in trusted:
                    continue
                mask = (0x2 | 0x4 | 0x10 | 0x100) if index == 0 else 0x40
                mask |= 0x10000 | 0x40000 | 0x80000 | 0x10000000 | 0x40000000
                require(not rule['rights'] & mask, 'Untrusted Windows mutation permission')
        result[row['label']] = row
    return result


def drvfs_mount():
    rows = [line for line in Path('/proc/self/mountinfo').read_text().splitlines() if ' /mnt/c ' in line]
    require(len(rows) == 1 and ' - 9p ' in rows[0] and 'aname=drvfs;path=C:\\;' in rows[0]
            and ';uid=0;gid=0;' in rows[0] and 'metadata' not in rows[0],
            'Exact diagnosed synthetic DrvFS ownership mapping required')
    return rows[0]


class Boundary:
    def __init__(self, attestation):
        self.rows = validate_attestation(attestation)
        require((os.getuid(), os.geteuid(), os.getgid(), os.getegid()) == (1000,)*4,
                'Genuine non-root Modly worker required')
        self.mount = drvfs_mount()
        self.files = {path: (sha, size) for path, sha, size in INPUTS.values()}
        self.directories = {ancestor for path in self.files for ancestor in path.parents}
        self.snapshots = {}
        for path in (*self.directories, *self.files):
            require(path == path.resolve(strict=True), 'Input/ancestor alias prohibited')
            info = path.lstat()
            require((info.st_uid, info.st_gid) == (0, 0), 'Unexpected diagnosed input/ancestor UID/GID')
            require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
                    'Input/ancestor link prohibited')
            require(stat.S_ISDIR(info.st_mode) if path in self.directories
                    else stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Input type/link mismatch')
            self.snapshots[path] = identity(info)
        self.reads = []

    def plain(self, path, *, directory=False):
        path = Path(path)
        require(path in (self.directories if directory else self.files), 'Foreign read-only input path')
        require(drvfs_mount() == self.mount and path == path.resolve(strict=True), 'Input mount or alias changed')
        info = path.lstat()
        require(identity(info) == self.snapshots[path], 'Input ownership/file identity changed')
        return info

    def read(self, path, maximum):
        path = Path(path)
        self.plain(path)
        for ancestor in path.parents:
            self.plain(ancestor, directory=True)
        sha, size = self.files[path]
        require(size <= maximum, 'Authenticated input exceeds caller byte bound')
        blob = read_plain(path, maximum)
        self.plain(path)
        require(len(blob) == size and hashlib.sha256(blob).hexdigest() == sha, 'Authenticated input payload hash differs')
        self.reads.append(dict(path=str(path), bytes=size, sha256=sha))
        return blob


def deny_write(*args, **kwargs):
    raise ValueError('This exact input capability is read-only')


def private_inputs(boundary, profiles, source):
    original_load = source.load_pinned
    latent = original_load('owned_readonly_latent', 'scripts/compatibility/amd_texture_sampling_artifact.py',
                           profiles.PINS['scripts/compatibility/amd_texture_sampling_artifact.py'])
    # This name is exclusive to this adapter. No generic/historical module is patched.
    latent._plain, latent._bounded_read = boundary.plain, boundary.read
    latent.write, latent._exclusive_write = deny_write, deny_write
    facade = ModuleType('owned_input_source')
    facade.ROOT = ROOT
    def private_load(name, relative, expected, imports=None):
        if name == 'input_latent':
            require(relative == 'scripts/compatibility/amd_texture_sampling_artifact.py'
                    and expected == profiles.PINS[relative], 'Original latent source identity differs')
            return latent
        return original_load('owned_' + name, relative, expected, imports=imports)
    facade.load_pinned = private_load
    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and name == '_source':
            return facade
        return builtins.__import__(name, globals, locals, fromlist, level)
    module = ModuleType('inference_texture._owned_readonly_inputs')
    module.__package__ = 'inference_texture'
    path = ROOT/'experimental/extensions/trellis2-amd/inference_texture/inputs.py'
    module.__file__ = str(path)
    module.__dict__['__builtins__'] = dict(vars(builtins), __import__=local_import)
    blob = read_plain(path, 131072)
    require(hashlib.sha256(blob).hexdigest() == CORE[path.relative_to(ROOT).as_posix()], 'Original inputs source changed')
    exec(compile(blob, str(path), 'exec'), vars(module))
    return module


def security_checks(boundary, attestation, inputs, prepared, profile, source):
    tests = []
    def rejects(name, call):
        try:
            call()
        except (ValueError, OSError) as exc:
            tests.append(dict(name=name, passed=True, rejection=type(exc).__name__, message=str(exc)))
        else:
            raise ValueError('Required input rejection absent: '+name)
    for name, change in (
        ('altered_hash', lambda d: d['files'][0].update(sha256='0'*64)),
        ('foreign_path', lambda d: d['files'][0].update(guest_path='/tmp/texture-latent.safetensors')),
        ('Windows_owner_mismatch', lambda d: d['files'][0].update(owner_sid='S-1-5-18')),
        ('Windows_hardlink', lambda d: d['files'][0]['windows_identity'].update(nlink=2)),
        ('Windows_reparse', lambda d: d['files'][0]['ancestors'][0].update(reparse=True))):
        bad = copy.deepcopy(attestation)
        change(bad)
        rejects(name, lambda bad=bad: validate_attestation(bad))
    rejects('foreign_actual_read', lambda: boundary.read(Path('/etc/passwd'), 65536))
    rejects('traversal_actual_reader', lambda: inputs._latent.verify(TEXTURE_ROOT/'../latent/texture-latent.safetensors',
                                        TEXTURE_ROOT/'../latent', token_count=4430, maximum_tokens=4430))
    rejects('foreign_actual_reader', lambda: inputs._latent.verify(Path('/tmp/texture-latent.safetensors'),
                                        Path('/tmp'), token_count=4430, maximum_tokens=4430))
    rejects('bounded_read_too_small', lambda: boundary.read(INPUTS['texture'][0], 10))
    rejects('read_only_writer', lambda: inputs._latent.write(None, None, None))
    rejects('original_reader_altered_hash', lambda: inputs._guide._read(INPUTS['texture'][0], 2000000, '0'*64))
    with INPUTS['texture'][0].open('rb') as stream:
        rejects('descriptor_alias_read', lambda: boundary.read(Path('/proc/self/fd')/str(stream.fileno()), 2000000))
    path = INPUTS['texture'][0]
    saved = boundary.snapshots[path]
    try:
        # Change only test expectation, never UID, filesystem, or actual file.
        boundary.snapshots[path] = (*saved[:3], 1000, *saved[4:])
        rejects('guest_ownership_binding_mismatch', lambda: boundary.plain(path))
    finally:
        boundary.snapshots[path] = saved
    original = source.load_pinned('unmodified_latent_control', 'scripts/compatibility/amd_texture_sampling_artifact.py',
                                 profile.document()['source_pins']['scripts/compatibility/amd_texture_sampling_artifact.py'])
    require(original is not inputs._latent and original._plain is not boundary.plain,
            'Historical reader must remain separate from the read-only capability')
    rejects('historical_UID_predicate_unchanged', lambda: original.approved_root(TEXTURE_ROOT))
    tests.append(dict(name='complete_genuine_inputs_prepare', passed=True))
    inputs.require_prepared(prepared, profile)
    tests.append(dict(name='independent_original_array_revalidation', passed=True))
    # The untouched generic reader still enforces UID ownership: the exception
    # is confined to this private read capability, not its historical behavior.
    original_owned_file = inputs._latent.__dict__['_owned_file']
    require(original_owned_file.__globals__ is inputs._latent.__dict__, 'Original reader function provenance differs')
    return dict(all_passed=True, tests=tests, bounded_input_reads=boundary.reads,
                genuine_prepare_source_sha256=CORE['experimental/extensions/trellis2-amd/inference_texture/inputs.py'],
                writer_capability=False, historical_readers_modified=False)


def prepare(*, manifest_path, attestation_path, expected_manifest_sha256, expected_attestation_sha256):
    manifest_blob = read_plain(manifest_path, 2*1024**2)
    require(hashlib.sha256(manifest_blob).hexdigest() == expected_manifest_sha256, 'CPU candidate manifest hash differs')
    manifest = json.loads(manifest_blob)
    require(type(manifest) is dict and all(manifest.get(k) == v for k, v in CORE.items()), 'Pinned input source closure required')
    for name, sha in manifest.items():
        require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts
                and '\\' not in name and ':' not in name and (ROOT/name).is_relative_to(ROOT), 'Manifest escaped source root')
        require(hashlib.sha256(read_plain(ROOT/name, 2*1024**2)).hexdigest() == sha, 'Candidate source changed: '+name)
    attestation_blob = read_plain(attestation_path, 1024**2)
    require(hashlib.sha256(attestation_blob).hexdigest() == expected_attestation_sha256, 'Windows owner proof hash differs')
    attestation = json.loads(attestation_blob)
    boundary = Boundary(attestation)
    sys.path[:0] = [str(STAGE), str(ROOT/'experimental/extensions/trellis2-amd')]
    import numpy as np
    require(np.__version__ == '2.2.6' and Path(np.__file__).resolve() == STAGE/'numpy/__init__.py',
            'Pinned staged NumPy 2.2.6 required')
    from inference_texture import profile as profiles
    from inference_texture import _source as source
    require(all(manifest.get(k) == v for k, v in profiles.PINS.items()), 'All original source pins required')
    for path, _, size in INPUTS.values():
        boundary.read(path, size)
    profile = profiles.authenticate(profiles.NAME, json.loads(boundary.read(INPUTS['run_contract'][0], 65536)))
    inputs = private_inputs(boundary, profiles, source)
    original_read = inputs._original._read
    def exact_read(path, maximum, sha):
        candidate = Path(inputs._original.platform_path(path))
        if candidate in boundary.files:
            require(boundary.files[candidate][0] == sha, 'Original caller input hash differs')
            return candidate, boundary.read(candidate, maximum)
        return original_read(path, maximum, sha)
    inputs._original._read = exact_read
    prepared = inputs.prepare(profile, texture_path=INPUTS['texture'][0], geometry_path=INPUTS['geometry'][0])
    results = security_checks(boundary, attestation, inputs, prepared, profile, source)
    results['numpy'] = dict(version=np.__version__, path=str(np.__file__))
    results['guest_input_identities'] = {str(p): list(boundary.snapshots[p]) for p in boundary.files}
    return dict(prepared=prepared, acceptance_security_results=results)
