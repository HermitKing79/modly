"""CPU-only identity and exact package-plan check; no installation or GPU query."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

EVIDENCE = Path('/mnt/c/Users/wattz/Documents/ChatGPT/modly/proof/docs/compatibility/amd-wsl-flexgemm-build-evidence')
PRIOR = EVIDENCE.parent / 'amd-wsl-neural-decoder-preparation-evidence'
os.environ.update(HIP_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')
root = Path('/opt/modly-wsl/venv')
rows = []
for directory, dirs, files in os.walk(root, followlinks=False):
    for name in dirs + files:
        path = Path(directory) / name
        st = path.lstat()
        rows.append([path.relative_to(root).as_posix(), st.st_size, st.st_mtime_ns, st.st_mode])
dpkg = subprocess.run(['dpkg-query', '-W', '-f=${Package}\t${Version}\n'],
                      capture_output=True, text=True, check=True, timeout=15)
identity = {
    'python': platform.python_version(), 'kernel': platform.release(),
    'venv_entries': len(rows),
    'venv_metadata_sha256': hashlib.sha256(json.dumps(sorted(rows), separators=(',', ':')).encode()).hexdigest(),
    'python_packages': sorted([[d.metadata['Name'], d.version] for d in md.distributions(
        path=[str(root / 'lib/python3.12/site-packages')])]),
    'dpkg_packages': dpkg.stdout.splitlines(), 'torch_imported': False, 'triton_imported': False,
}
previous = json.loads((PRIOR / 'linux-preservation-after.json').read_text())['identity']
if identity != previous:
    raise RuntimeError('Installed Linux identity drift before admission')
plan = json.loads((PRIOR / 'linux-toolchain-plan.json').read_text())
requested = [row['Package'] + '=' + row['Version'] for row in plan['packages']]
command = ['apt-get', '--simulate', '--no-install-recommends', 'install', *requested]
simulation = subprocess.run(command, capture_output=True, text=True, timeout=30)
actual = []
for line in simulation.stdout.splitlines():
    if line.startswith('Inst '):
        match = re.match(r'Inst (\S+)(?: \[([^\]]+)\])? \((\S+) ', line)
        if not match:
            raise RuntimeError('Unparsed package-plan row')
        name, old, new = match.groups()
        actual.append([name, old, new])
expected = [[row['Package'], row['previous_version'], row['Version']] for row in plan['packages']]
if simulation.returncode or sorted(actual) != sorted(expected):
    raise RuntimeError('APT dependency or upgrade drift')

# CPU metadata import only: no availability/device/property query. GPU entry
# points are replaced before any later build-helper import can use them.
import torch
if torch.cuda.is_initialized():
    raise RuntimeError('GPU unexpectedly initialized by Torch import')
def no_gpu(*args, **kwargs):
    raise RuntimeError('CPU-only preparation prohibits GPU initialization or queries')
for name in ('init', '_lazy_init', 'is_available', 'device_count', 'get_device_properties', 'get_device_name'):
    setattr(torch.cuda, name, no_gpu)
for name in ('_cuda_init', '_cuda_getDeviceCount', '_cuda_getDeviceProperties', '_cuda_getArchFlags'):
    if hasattr(torch._C, name):
        setattr(torch._C, name, no_gpu)
abi = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
stat = os.statvfs('/')
ram = {line.split(':')[0]: int(line.split()[1]) * 1024
       for line in Path('/proc/meminfo').read_text().splitlines()
       if line.startswith(('MemTotal:', 'MemAvailable:'))}
import setuptools
import setuptools.command.bdist_wheel
record = {
    'identity': identity, 'identity_matches_prior': True,
    'exact_apt_plan_matches': True, 'simulation': {'command': command,
        'returncode': simulation.returncode, 'stdout': simulation.stdout, 'stderr': simulation.stderr},
    'packages': plan['packages'], 'new_packages': 59, 'upgrades': 2,
    'archive_bytes': plan['archive_bytes_sum'], 'known_installed_bytes': plan['known_installed_bytes_sum'],
    'python': platform.python_version(), 'torch': torch.__version__, 'torch_hip': torch.version.hip,
    'torch_git': torch.version.git_version, 'cxx11_abi': abi, 'torch_config': torch.__config__.show(),
    'rocm_path': str(Path('/opt/rocm').resolve()),
    'bdist_wheel_module': setuptools.command.bdist_wheel.__file__,
    'setuptools': setuptools.__version__, 'gpu_initialized': torch.cuda.is_initialized(),
    'filesystem_total_bytes': stat.f_blocks * stat.f_frsize,
    'filesystem_available_bytes': stat.f_bavail * stat.f_frsize, 'guest_ram_bytes': ram,
    'package_installation': False, 'native_build': False, 'model_loading': False,
}
with (EVIDENCE / 'preflight.json').open('x', encoding='utf-8') as f:
    json.dump(record, f, indent=2)
print(json.dumps({k: record[k] for k in ('identity_matches_prior', 'exact_apt_plan_matches',
    'cxx11_abi', 'torch', 'torch_hip', 'rocm_path', 'bdist_wheel_module',
    'gpu_initialized', 'filesystem_available_bytes', 'guest_ram_bytes')}), flush=True)
