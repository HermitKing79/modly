"""Gated CPU-only saved-input diagnosis and genuine preparation acceptance."""
import argparse
import ast
import datetime
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import stat
import sys
import time
import traceback
import weakref


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DIAGNOSIS_MODES = ('diagnose', 'diagnose-confirmation')
sys.pycache_prefix = str(HERE / '_unused_guest_bytecode')
READER = ROOT / 'scripts/compatibility/amd_texture_sampling_artifact.py'
READER_SHA256 = '5eaf35c08cf2e96f46dfe8bb32fa835c079ef34ce5a322a0afd411ee7cf9474c'
TEXTURE_PATH = Path('/mnt/c/Users/wattz/Documents/ModlyArtifacts/amd-brick-material-stage1-df41688/latent/texture-latent.safetensors')
FORBIDDEN = frozenset({'torch', 'triton', 'flex_gemm', 'trellis2', 'cumesh',
                       'o_voxel', 'hip_utils', 'safetensors', '__triton_launcher'})
TUNABLE_ENV = {'PYTORCH_TUNABLEOP_ENABLED': '0', 'PYTORCH_TUNABLEOP_TUNING': '0',
               'PYTORCH_TUNABLEOP_RECORD_UNTUNED': '0'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save_json(path, row):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def await_go(stream):
    if stream.read(4) != b'GO\n':
        raise RuntimeError('Exact binary CPU host authorization absent or refused')


def metadata(path):
    row = {'path': str(path)}
    try:
        value = path.lstat()
        row.update(uid=value.st_uid, gid=value.st_gid, mode=value.st_mode,
                   mode_octal=oct(value.st_mode), permissions=stat.filemode(value.st_mode),
                   device=value.st_dev, inode=value.st_ino, nlink=value.st_nlink,
                   size=value.st_size, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns,
                   type=('directory' if stat.S_ISDIR(value.st_mode) else 'regular' if stat.S_ISREG(value.st_mode)
                         else 'symlink' if stat.S_ISLNK(value.st_mode) else 'other'),
                   file_attributes=getattr(value, 'st_file_attributes', 0),
                   reparse=bool(getattr(value, 'st_file_attributes', 0) & 1024))
        row['canonical_path'] = str(path.resolve(strict=True))
        row['already_canonical'] = row['canonical_path'] == str(path)
    except (OSError, ValueError) as exc:
        row.update(error_type=type(exc).__name__, error=str(exc))
    return row


def pinned_predicates(path=READER):
    blob = path.read_bytes()
    actual = hashlib.sha256(blob).hexdigest()
    if actual != READER_SHA256:
        raise RuntimeError('Pinned genuine texture reader identity differs')
    tree = ast.parse(blob, filename=str(path))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in ('_plain', 'approved_root')]
    if [node.name for node in functions] != ['_plain', 'approved_root']:
        raise RuntimeError('Genuine predicate function selection differs')
    selected = ast.Module(body=functions, type_ignores=[])
    namespace = {'Path': Path, 'stat': stat, 'os': os, '__builtins__': __builtins__}
    exec(compile(selected, str(path), 'exec'), namespace)
    return namespace, {'reader_path': str(path), 'reader_sha256': actual,
                       'selection': 'Exact unmodified source AST FunctionDef bodies; no reader imports executed',
                       'functions': {node.name: {'line': node.lineno, 'end_line': node.end_lineno,
                           'source_sha256': hashlib.sha256(ast.get_source_segment(blob.decode('utf-8'), node).encode('utf-8')).hexdigest()}
                                     for node in functions}}


def probe_text(path):
    try:
        return {'path': str(path), 'text': path.read_text(encoding='utf-8'), 'metadata': metadata(path)}
    except OSError as exc:
        return {'path': str(path), 'error_type': type(exc).__name__, 'error': str(exc)}


def diagnose(texture_path=TEXTURE_PATH):
    namespace, provenance = pinned_predicates()
    root = texture_path.parent
    ancestors = (root, *root.parents)
    rows = []
    first = None
    # This is precisely approved_root's order. Each actual _plain function body
    # executes independently so all directory metadata survives an early failure.
    for index, path in enumerate(ancestors):
        row = metadata(path)
        row['approved_root_order'] = index
        try:
            namespace['_plain'](path, directory=True)
            row['genuine_plain'] = 'ACCEPTED'
        except Exception as exc:
            row.update(genuine_plain='REJECTED', rejection_type=type(exc).__name__, rejection=str(exc))
            if first is None:
                first = dict(path=str(path), index=index, error_type=type(exc).__name__, error=str(exc))
        rows.append(row)
    try:
        approved = {'outcome': 'ACCEPTED', 'return': str(namespace['approved_root'](root))}
    except Exception as exc:
        approved = {'outcome': 'REJECTED', 'error_type': type(exc).__name__, 'error': str(exc)}
    file_row = metadata(texture_path)
    try:
        namespace['_plain'](texture_path)
        file_row['genuine_plain'] = 'ACCEPTED'
    except Exception as exc:
        file_row.update(genuine_plain='REJECTED', rejection_type=type(exc).__name__, rejection=str(exc))
    mounts = probe_text(Path('/proc/self/mountinfo'))
    covering = []
    for line in mounts.get('text', '').splitlines():
        parts = line.split()
        if '-' not in parts or len(parts) < 7:
            continue
        split = parts.index('-')
        mountpoint = parts[4].replace('\\040', ' ')
        if str(texture_path) == mountpoint or str(texture_path).startswith(mountpoint.rstrip('/') + '/'):
            covering.append({'mountpoint': mountpoint, 'filesystem': parts[split + 1],
                             'source': parts[split + 2], 'mount_options': parts[5],
                             'super_options': parts[split + 3], 'raw': line})
    covering.sort(key=lambda row: len(row['mountpoint']))
    return {'predicate_provenance': provenance, 'texture_file': file_row,
            'ancestor_order': 'artifact root, then parent directories through /',
            'ancestors': rows, 'first_genuine_plain_rejected_directory': first,
            'genuine_approved_root': approved, 'covering_mounts': covering,
            'effective_filesystem': covering[-1] if covering else None,
            'mountinfo': mounts, 'proc_mounts': probe_text(Path('/proc/mounts')),
            'wsl_configuration': probe_text(Path('/etc/wsl.conf')),
            'mnt_c': metadata(Path('/mnt/c'))}


def load_policy(path, expected_sha256):
    spec = importlib.util.spec_from_file_location('owned_readonly_input', path)
    module = importlib.util.module_from_spec(spec)
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != expected_sha256:
        raise RuntimeError('Host-frozen input policy source changed before compilation')
    sys.modules[spec.name] = module
    exec(compile(blob, str(path), 'exec'), module.__dict__)
    return module


def security_tests_passed(evidence):
    if type(evidence) is not dict or evidence.get('all_passed') is not True:
        return False
    tests = evidence.get('tests')
    if type(tests) is dict:
        return bool(tests) and all(value is True for value in tests.values())
    if type(tests) is list:
        return bool(tests) and all(type(row) is dict and row.get('passed') is True for row in tests)
    return False


def full_prepare(args):
    policy_path = HERE / 'readonly_input.py'
    for path, expected in ((policy_path, args.expected_policy_sha256),
                           (args.manifest, args.expected_manifest_sha256),
                           (args.attestation, args.expected_attestation_sha256)):
        if expected is None or digest(path) != expected:
            raise RuntimeError('Host-frozen CPU preparation input/source differs: ' + str(path))
    policy = load_policy(policy_path, args.expected_policy_sha256)
    result = policy.prepare(manifest_path=args.manifest, attestation_path=args.attestation,
                            expected_manifest_sha256=args.expected_manifest_sha256,
                            expected_attestation_sha256=args.expected_attestation_sha256)
    if type(result) is not dict or type(result.get('prepared')) is not dict:
        raise RuntimeError('Genuine prepared-input dictionary and acceptance evidence required')
    prepared = result['prepared']
    metadata_row = prepared.get('metadata')
    security = result.get('acceptance_security_results')
    if (type(metadata_row) is not dict or metadata_row.get('verified') is not True
            or metadata_row.get('cpu_only') is not True
            or metadata_row.get('model_imported') is not False
            or metadata_row.get('gpu_initialized') is not False):
        raise RuntimeError('Genuine CPU-only prepared-input metadata incomplete')
    if not security_tests_passed(security):
        raise RuntimeError('Nonempty passing acceptance security tests required')
    array_refs = []

    def remember_arrays(value):
        if type(value).__module__ == 'numpy' and type(value).__name__ == 'ndarray':
            array_refs.append(weakref.ref(value))
        elif type(value) is dict:
            for child in value.values():
                remember_arrays(child)
        elif type(value) in (tuple, list):
            for child in value:
                remember_arrays(child)

    remember_arrays(prepared)
    if not array_refs:
        raise RuntimeError('Actual genuine prepared NumPy arrays were not observed')
    # The experiment policy authenticates the complete genuine prepare result
    # and returns its JSON evidence, never tensor bytes or model objects.
    evidence = json.loads(json.dumps({'prepared_metadata': metadata_row,
                                     'acceptance_security_results': security}, allow_nan=False))
    del prepared, result, metadata_row, security
    gc.collect()
    if any(reference() is not None for reference in array_refs):
        raise RuntimeError('Genuine prepared arrays survived CPU acceptance release')
    return {**evidence,
            'prepared_arrays_released': True,
            'prepared_array_objects_observed': len(array_refs),
            'input_policy_sha256': digest(policy_path),
            'manifest_sha256': digest(args.manifest),
            'Windows_attestation_sha256': digest(args.attestation)}


def audit_guard(row, forbidden, output, diagnosis):
    result_path = str(output / 'results.json')
    mutations = {'os.remove', 'os.rmdir', 'os.rename', 'os.mkdir', 'os.symlink', 'os.link',
                 'os.chmod', 'os.chown', 'os.truncate', 'os.utime'}

    def deny(event, arguments):
        if event == 'import' and str(arguments[0]).split('.')[0] in forbidden:
            row['forbidden_import_attempts'].append(str(arguments[0]))
            raise RuntimeError('CPU harness forbids native GPU/model import: ' + str(arguments[0]))
        if event.startswith('socket.'):
            raise RuntimeError('CPU harness forbids network entry: ' + event)
        if event in ('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn'):
            raise RuntimeError('CPU harness forbids subprocess entry: ' + event)
        if event in mutations:
            raise RuntimeError('Read-only CPU harness forbids filesystem mutation: ' + event)
        if event == 'open':
            path, mode, flags = arguments
            writes = (isinstance(mode, str) and any(letter in mode for letter in 'wax+')) or (
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writes and (not isinstance(path, str) or path != result_path
                           or not flags & os.O_EXCL or flags & os.O_TRUNC):
                raise RuntimeError('Read-only CPU harness permits only exclusive worker results output')
        if event == 'ctypes.dlopen':
            name = str(arguments[0]).lower()
            if diagnosis or any(token in name for token in ('rocm', 'amdhip', 'cuda', 'dxg', 'hsa-runtime')):
                raise RuntimeError('CPU harness forbids GPU/diagnostic ctypes loader entry')

    return deny


def worker(args):
    output = args.worker
    if output != HERE / args.mode / 'guest':
        raise RuntimeError('Worker output escaped its exclusive mode directory')
    output.mkdir(exist_ok=False)
    # Process-local cache lookup isolation; never create this home path.
    sys.pycache_prefix = '/home/modly/input-ownership-forbidden-pyc-' + args.mode
    began = time.monotonic()
    row = {'schema': 'modly.amd.wsl-input-cpu-worker.v1', 'mode': args.mode,
           'classification': 'CPU_UNKNOWN', 'CPU_only': True, 'gate_authorized': False,
           'GPU_authorized': False, 'model_authorized': False,
           'forbidden_import_attempts': [], 'started_utc': utc_now()}
    forbidden = FORBIDDEN | ({'numpy'} if args.mode in DIAGNOSIS_MODES else set())
    try:
        await_go(sys.stdin.buffer)
        row['gate_authorized'] = True

        sys.addaudithook(audit_guard(row, forbidden, output, args.mode in DIAGNOSIS_MODES))
        for path, expected in ((HERE / 'worker.py', args.expected_worker_sha256),
                               (HERE / 'host.py', args.expected_host_sha256)):
            if digest(path) != expected:
                raise RuntimeError('Host-frozen harness source differs')
        row['runtime'] = {'uid': os.getuid(), 'euid': os.geteuid(), 'gid': os.getgid(),
                          'egid': os.getegid(), 'groups': os.getgroups(),
                          'executable': sys.executable, 'python': sys.version,
                          'prefix': sys.prefix, 'isolated': sys.flags.isolated,
                          'dont_write_bytecode': sys.dont_write_bytecode,
                          'pycache_prefix': sys.pycache_prefix}
        if (sys.prefix != '/opt/modly-wsl/venv' or platform.python_version() != '3.12.3'
                or sys.executable != '/opt/modly-wsl/venv/bin/python'
                or (os.getuid(), os.geteuid(), os.getgid(), os.getegid()) != (1000, 1000, 1000, 1000)
                or sys.flags.isolated != 1 or not sys.dont_write_bytecode):
            raise RuntimeError('Expected isolated Python and Modly UID/GID 1000 required')
        row['bytecode_cache_prefix_absent'] = not Path(sys.pycache_prefix).exists()
        if not row['bytecode_cache_prefix_absent']:
            raise RuntimeError('Isolated bytecode lookup location must remain absent')
        row['owned_guest_comparison'] = {
            'home': metadata(Path('/home/modly')),
            'existing_stage': metadata(Path('/home/modly/native-decoder-preparation-6dd94a85'))}
        row['tunable_environment'] = {key: os.environ.get(key) for key in TUNABLE_ENV}
        if row['tunable_environment'] != TUNABLE_ENV:
            raise RuntimeError('All TunableOp controls must be zero')
        if args.mode in DIAGNOSIS_MODES:
            row['diagnosis'] = diagnose()
            row['classification'] = 'CPU_DIAGNOSED'
        else:
            row.update(full_prepare(args))
            row['classification'] = 'CPU_PREPARED'
    except BaseException as exc:
        row.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        row['forbidden_modules'] = sorted(name for name in sys.modules if name.split('.')[0] in forbidden)
        if row['forbidden_modules'] or row['forbidden_import_attempts']:
            row['classification'] = 'CPU_UNKNOWN'
        row.update(elapsed_seconds=time.monotonic() - began, finished_utc=utc_now())
        save_json(output / 'results.json', row)
    return 0 if row['classification'] in ('CPU_DIAGNOSED', 'CPU_PREPARED') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--mode', choices=(*DIAGNOSIS_MODES, 'acceptance', 'independent'), required=True)
    parser.add_argument('--expected-worker-sha256', required=True)
    parser.add_argument('--expected-host-sha256', required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--attestation', type=Path)
    parser.add_argument('--expected-manifest-sha256')
    parser.add_argument('--expected-attestation-sha256')
    parser.add_argument('--expected-policy-sha256')
    args = parser.parse_args()
    if args.mode not in DIAGNOSIS_MODES and (args.manifest is None or args.attestation is None):
        parser.error('Full CPU acceptance requires manifest and attestation')
    return worker(args)


if __name__ == '__main__':
    raise SystemExit(main())
