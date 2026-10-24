"""Read-only WSL input probe. Args: contract JSON path, canonical contract SHA256.

Host invocation uses the qualified interpreter with -B -I, guest timeout 30s,
host timeout 45s. This script never consumes GO, starts a child or claims paths.
"""
import hashlib
import importlib.abc
import importlib.metadata as metadata
import json
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import sys
import time
import traceback
import zipfile

ROOT = Path('/mnt/c/Users/wattz/Documents/ChatGPT/modly/proof')
DOCS = ROOT / 'docs/compatibility'
CODE = ROOT / 'experimental/extensions/trellis2-amd/inference_decoder'
STAGE = Path('/home/modly/native-decoder-preparation-6dd94a85')
SITE = STAGE / 'site-packages'
PYTHON = '/opt/modly-wsl/venv/bin/python'
FORBIDDEN = {'torch', 'torchvision', 'triton', 'trellis2', 'flex_gemm', 'transformers',
             'cumesh', 'o_voxel', 'nvdiffrast', 'rembg', 'onnxruntime', 'safetensors'}
SOURCE_SHA256 = {
    'experimental/extensions/trellis2-amd/inference_decoder/run_contract.py': 'b5897c709bb0d005476e854ebc2f1a33c874b934d9b16f0fd80de1a65ef7327d',
    'experimental/extensions/trellis2-amd/inference_decoder/decode.py': '122df9e57c756b44da3f73aeb8cae43dc2745ce78e46bbf2adc4a05b1e56cb19',
    'scripts/compatibility/amd_wsl_native_import_policy.py': '8b9207a14757ccda197e11c054d694f9b4ff47006a1cccc46d71f1d494ab96cd',
    'scripts/compatibility/amd_shape_decoder_preflight.py': '327786f654239a1a17278cf73635b662d87a797c81249a77c94539dc2d8d000c',
}
EVIDENCE_SHA256 = {
    'amd-wsl-native-decoder-preparation-evidence/staging.json': '8bfc578bb58e7efd81d2d4fa6baa38ec04ae8ea92379918247494cdbb81babf9',
    'amd-wsl-native-decoder-preparation-evidence/python-dependency-admission.json': '11e846afb991659cd60822496b90354f0297111126b40d791a6ec794d3815fe9',
    'amd-wsl-flexgemm-build-evidence/static-acceptance.json': '67dd2b44ef3707b7f9eb32b14948698141fffe39549556b2602271c7979c67dc',
    'amd-wsl-neural-decoder-preparation-evidence/decoder-contract-readback.json': '77cec60d7769c8f448c5880e620bf7007568c24671450037f74819f02d02a0c3',
}
INPUT_SHA256 = {
    'shape-latent-denormalized.npy': '18778e295e7d9405ca2e80b8fde64dacc790696c361f6418436d2e498149b2b3',
    'coords.npy': 'c432b04ec70c6200fadf8f5f62ce995514e24bf4194b154ac30a76148a7bb4a2',
}
sys.dont_write_bytecode = True


class NoNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise RuntimeError('CPU probe forbids neural/native import: ' + fullname)


def read_only(event, args):
    if event == 'open':
        mode, flags = args[1:3]
        if (isinstance(mode, str) and any(ch in mode for ch in 'wax+')) or \
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise RuntimeError('CPU probe forbids filesystem writes')
    if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system', 'os.posix_spawn',
            'os.mkdir', 'os.remove', 'os.rename', 'os.rmdir', 'os.link', 'os.symlink'):
        raise RuntimeError('CPU probe forbids process, network or filesystem mutations')


sys.meta_path.insert(0, NoNeuralImports())
sys.addaudithook(read_only)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def installed_source_path(windows_path):
    source = PureWindowsPath(windows_path)
    require(source.is_absolute() and source.drive in ('C:', 'D:') and '..' not in source.parts,
            'Foreign or noncanonical installed source path')
    return Path('/mnt/' + source.drive[0].lower(), *source.parts[1:])


def plain(path, *, existing=True):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and str(path) == path.as_posix(), 'Canonical absolute path required')
    for parent in (path, *path.parents):
        if parent.exists() or parent.is_symlink():
            require(not parent.is_symlink(), 'Symlink path prohibited')
    require(path.resolve(strict=existing) == path, 'Path alias prohibited')
    if existing:
        require(stat.S_ISREG(path.stat().st_mode) and path.stat().st_nlink == 1, 'Single regular file required')
    return path


def digest(path):
    path = plain(path)
    before = path.stat()
    with path.open('rb') as stream:
        result = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'File changed during hashing')
    return result


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def json_file(path, expected):
    require(plain(path).stat().st_size <= 8 * 1024**2 and digest(path) == expected, 'Evidence identity or bound differs')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique)


def wheel_payloads(wheel, expected, *, numpy_entry_and_native_only=False):
    require(digest(wheel) == expected, 'Pinned staged wheel changed')
    checked = 0
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            if member.is_dir() or '.dist-info/' in member.filename:
                continue
            if numpy_entry_and_native_only and member.filename != 'numpy/__init__.py' and not member.filename.endswith('.so'):
                continue
            require(not member.filename.startswith('/') and '..' not in Path(member.filename).parts,
                    'Invalid wheel payload path')
            require(member.file_size <= 64 * 1024**2, 'Wheel member exceeds original bounded package scope')
            path = SITE / member.filename
            require(plain(path).stat().st_size == member.file_size and
                    digest(path) == hashlib.sha256(archive.read(member)).hexdigest(), 'Staged wheel payload changed')
            checked += 1
    return checked


def probe(contract_path, expected_sha):
    require(os.getuid() == os.geteuid() == 1000, 'Original modly UID/EUID required')
    require(sys.version_info[:3] == (3, 12, 3) and sys.executable == PYTHON and
            sys.prefix == '/opt/modly-wsl/venv', 'Qualified WSL Python identity required')
    require(Path(__file__).resolve() == DOCS / 'amd-zombie-decode-evidence/wsl_cpu_probe.py', 'Probe source path differs')
    require(re.fullmatch('[0-9a-f]{64}', expected_sha) is not None, 'Canonical contract SHA256 required')
    require(not FORBIDDEN.intersection(sys.modules), 'Neural/native package already imported')
    home = Path('/home/modly')
    home_stat = home.lstat()
    require(home.is_dir() and not home.is_symlink() and home.resolve(strict=True) == home
            and home_stat.st_uid == 1000 and not home_stat.st_mode & 0o022, 'Original private modly home ownership/mode required')
    prefix = Path('/mnt/d/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16')
    checkpoint_stat = plain(prefix.with_suffix('.safetensors')).stat()
    require(checkpoint_stat.st_size == 948490494 and checkpoint_stat.st_mtime_ns == 1791274722716299500,
            'Exact decoder checkpoint metadata changed')
    config_sha256 = digest(prefix.with_suffix('.json'))
    require(config_sha256 == '5f1b856dffce79466fb18f5f0eefc08f08c673b24b3c05b20359c0b6d318a209', 'Exact decoder configuration changed')
    sources = {name: digest(ROOT / name) for name in SOURCE_SHA256}
    require(sources == SOURCE_SHA256, 'Original decoder, input reader or native policy changed')
    evidence = {name: json_file(DOCS / name, sha) for name, sha in EVIDENCE_SHA256.items()}
    staged = evidence['amd-wsl-native-decoder-preparation-evidence/staging.json']
    require(staged['target'] == str(SITE) and staged['canonical_staged_packages'] ==
            {'flex-gemm': '1.0.0', 'numpy': '2.2.6', 'safetensors': '0.6.2'}, 'Original staged package identity differs')
    versions = {name: metadata.version(name) for name in ('torch', 'triton')}
    require(versions == {'torch': '2.9.1+rocm7.2.1.lw.gitff65f5bc',
                         'triton': '3.5.1+rocm7.2.1.gita272dfa8'}, 'Original baseline package metadata differs')
    admission = evidence['amd-wsl-native-decoder-preparation-evidence/python-dependency-admission.json']
    numpy_row = next(row for row in admission['packages'] if row['name'] == 'numpy')
    require(numpy_row['version'] == '2.2.6' and numpy_row['sha256'] ==
            'fd83c01228a688733f1ded5201c678f0c53ecc1006ffbc404db9f7a899ac6249', 'Original NumPy wheel identity differs')
    numpy_files = wheel_payloads(STAGE / 'inputs' / numpy_row['filename'], numpy_row['sha256'],
                                numpy_entry_and_native_only=True)
    native = evidence['amd-wsl-flexgemm-build-evidence/static-acceptance.json']
    require(native['wheel_sha256'] == 'cadaa86c428a93c3aee10c4dd8dfb66604a424c8145670051ca0ce1a0f296285'
            and native['native_sha256'] == '91f66fa2c0f3567cee71246257bea8eb0fe5af9b6c70617d3c5894aa9f103f01', 'Native acceptance identity differs')
    native_files = wheel_payloads(DOCS / 'amd-wsl-flexgemm-build-evidence' / Path(native['wheel']).name, native['wheel_sha256'])
    require(digest(SITE / native['native_member']) == native['native_sha256'], 'Staged original native .so changed')
    installed_sources = {}
    source_contract = evidence['amd-wsl-neural-decoder-preparation-evidence/decoder-contract-readback.json']
    for windows_path, row in source_contract['source_checks'].items():
        guest = installed_source_path(windows_path)
        installed_sources[str(guest)] = digest(guest)
        require(installed_sources[str(guest)] == row['expected_sha256'], 'Pinned installed decoder source changed')
    sys.path[:0] = [str(SITE), str(CODE), str(ROOT / 'scripts/compatibility')]
    import numpy as np
    import run_contract
    require(np.__version__ == '2.2.6' and Path(np.__file__).resolve() == SITE / 'numpy/__init__.py', 'Original staged NumPy must be used')
    require(Path(run_contract.__file__).resolve() == CODE / 'run_contract.py', 'Original run_contract source required')
    contract_path = plain(contract_path)
    contract_file_sha256 = digest(contract_path)
    contract = run_contract.load(contract_path)
    require(contract.sha256 == expected_sha and contract.document()['schema'] == run_contract.SCHEMA, 'Exact original-schema contract required')
    expected_paths = {'shape-latent-denormalized.npy': ROOT / 'artifacts/amd-zombie-shape-0d15dd9/shape/run/scratch/shape-latent-denormalized.npy',
                      'coords.npy': ROOT / 'artifacts/amd-zombie-shape-0d15dd9/shape/run/scratch/shape-latent-coordinates.npy'}
    require({name: str(path) for name, path in contract.input_paths.items()} == {name: str(path) for name, path in expected_paths.items()}
            and dict(contract.input_sha256) == INPUT_SHA256 and contract.initial_count == 1000,
            'Only exact original saved zombie neural inputs are admitted')
    require(contract.source_job_sha256 == '913ad907c55f1f510cce00d1d7573d518883190628d040506a6bd78578dd54bf'
            and digest(ROOT / 'artifacts/amd-zombie-shape-0d15dd9/shape/run/results.json') == contract.source_job_sha256,
            'Original Shape-SLat results identity differs')
    require(contract.maximum_final_rows == 1842968 and contract.maximum_intermediate_rows == 1000000
            and contract.large_convolution_tiling is True and contract.final_support_profile is None
            and str(contract.repo_root) == str(ROOT) and str(contract.native_stage) == str(STAGE), 'Generic zombie admission differs')
    parent = Path(str(contract.output_directory.parent))
    require(parent.parent == Path('/home/modly') and parent.name.startswith('run-') and
            not parent.exists() and not parent.is_symlink(), 'Fresh absent owned guest run-ID root required')
    require(str(contract.output_directory) == str(parent / 'geometry') and str(contract.scratch_directory) == str(parent / 'scratch')
            and str(contract.host_evidence_directory) == str(DOCS / 'amd-repeatable-generation-v1-evidence/new-image/zombie-0468f86/decode/run/guest'),
            'Exact zombie guest/output mirror paths required')
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith(('MemTotal:', 'MemAvailable:'))}
    disks = {name: os.statvfs(name).f_bavail * os.statvfs(name).f_frsize for name in ('/', '/mnt/d')}
    require(memory['MemAvailable'] >= 6972719480 and disks['/'] >= 5 * 1024**3 and disks['/mnt/d'] >= 50 * 1024**3,
            'Original guest RAM/disk admission failed')
    features, coords, summary = run_contract.load_inputs(contract)
    require(not FORBIDDEN.intersection(sys.modules), 'CPU input reader imported neural/native package')
    require(digest(contract_path) == contract_file_sha256 and contract.sha256 == expected_sha and not parent.exists(),
            'Read-only probe preservation failed')
    return {'classification': 'PASS_CPU_INPUT_ONLY', 'run_contract_sha256': contract.sha256,
            'contract_file_sha256': contract_file_sha256, 'probe_sha256': digest(Path(__file__)),
            'uid': os.getuid(), 'euid': os.geteuid(), 'python': sys.version, 'executable': sys.executable,
            'modly_home_uid': home_stat.st_uid, 'modly_home_mode': oct(stat.S_IMODE(home_stat.st_mode)),
            'checkpoint_bytes': checkpoint_stat.st_size, 'checkpoint_mtime_ns': checkpoint_stat.st_mtime_ns,
            'config_sha256': config_sha256, 'checkpoint_payload_read': False,
            'distribution_versions_without_import': versions, 'numpy_version': np.__version__, 'numpy_path': np.__file__,
            'original_source_sha256': sources, 'installed_decoder_source_sha256': installed_sources,
            'preserved_evidence_sha256': EVIDENCE_SHA256, 'numpy_wheel_sha256': numpy_row['sha256'],
            'numpy_installed_entry_and_native_files_verified': numpy_files, 'native_wheel_sha256': native['wheel_sha256'],
            'native_installed_payload_files_verified': native_files, 'native_so_sha256': native['native_sha256'],
            'saved_inputs': summary, 'guest_memory_bytes': memory, 'guest_disk_free_bytes': disks,
            'fresh_guest_run_root_absent': True, 'no_neural_imports': True, 'writes': False,
            'GO_consumed': False, 'gpu_initialized': False, 'model_loaded': False, 'neural_forward_calls': 0,
            'guest_timeout_seconds': 30, 'host_timeout_seconds': 45}


if __name__ == '__main__':
    began = time.monotonic()
    try:
        require(len(sys.argv) == 3, 'Expected contract path and canonical contract SHA256')
        receipt = probe(sys.argv[1], sys.argv[2])
        result = 0
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        receipt = {'classification': 'BLOCKED_CPU_INPUT_ONLY', 'error': type(exc).__name__ + ': ' + str(exc),
                   'no_neural_imports': not bool(FORBIDDEN.intersection(sys.modules)), 'writes': False,
                   'GO_consumed': False, 'gpu_initialized': False, 'model_loaded': False, 'neural_forward_calls': 0}
        result = 1
    receipt['elapsed_seconds'] = time.monotonic() - began
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    raise SystemExit(result)
