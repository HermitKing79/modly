"""Read-only CPU environment/toolchain inventory. No Torch/native GPU imports."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

def deny_ai(event, args):
    if event == 'import' and str(args[0]).split('.')[0] in {'torch', 'triton', 'numpy', 'flex_gemm'}:
        raise RuntimeError('Read-only inventory prohibits AI/native imports')
sys.addaudithook(deny_ai)
root = Path('/opt/modly-wsl/venv'); rows = []
for directory, dirs, files in os.walk(root, followlinks=False):
    for name in dirs + files:
        path = Path(directory) / name; stat = path.lstat()
        rows.append([path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns, stat.st_mode])
query = subprocess.run(['dpkg-query', '-W', '-f=${Package}\t${Version}\n'],
                       capture_output=True, text=True, timeout=15)
if query.returncode: raise RuntimeError('Read-only package query failed')
identity = {'python': platform.python_version(), 'kernel': platform.release(),
            'venv_entries': len(rows),
            'venv_metadata_sha256': hashlib.sha256(json.dumps(sorted(rows), separators=(',', ':')).encode()).hexdigest(),
            'python_packages': sorted([(d.metadata['Name'], d.version) for d in md.distributions(
                path=[str(root / 'lib/python3.12/site-packages')])]),
            'dpkg_packages': query.stdout.splitlines(),
            'torch_imported': 'torch' in sys.modules, 'triton_imported': 'triton' in sys.modules}
tools = {name: shutil.which(name) for name in ('cc', 'gcc', 'g++', 'clang', 'clang++', 'hipcc', 'cmake', 'ninja')}
paths = ['/opt/rocm/include/hip/hip_runtime.h', '/opt/rocm/include/hip/hip_runtime_api.h',
         '/opt/rocm/include/hipblas/hipblas.h', '/opt/rocm/include/hipblaslt/hipblaslt.h',
         '/opt/rocm/llvm/bin/clang++', '/opt/rocm/llvm/bin/clang', '/opt/rocm/bin/hipcc',
         '/opt/rocm/amdgcn/bitcode', '/usr/include/python3.12/Python.h',
         '/opt/modly-wsl/venv/lib/python3.12/site-packages/torch/include/torch/extension.h']
headers = {name: Path(name).exists() for name in paths}
cmake = root / 'lib/python3.12/site-packages/torch/share/cmake/Torch/TorchConfig.cmake'
flags = [line.strip() for line in cmake.read_text().splitlines()
         if 'GLIBCXX' in line or 'TORCH_CXX_FLAGS' in line]
stat = os.statvfs('/')
record = {'identity': identity, 'os_release': platform.freedesktop_os_release(),
          'tools': tools, 'headers': headers, 'torch_cmake_abi_flags': flags,
          'cmake_config_sha256': hashlib.sha256(cmake.read_bytes()).hexdigest(),
          'filesystem_total_bytes': stat.f_blocks * stat.f_frsize,
          'filesystem_available_bytes': stat.f_bavail * stat.f_frsize,
          'last_gpu_admission_not_refreshed': 'Use prior approved smoke free bytes; no GPU call here',
          'dxg_user_rw': os.access('/dev/dxg', os.R_OK | os.W_OK)}
meminfo = {}
for line in Path('/proc/meminfo').read_text().splitlines():
    if line.startswith(('MemTotal:', 'MemAvailable:')):
        key, value, unit = line.split()
        meminfo[key.rstrip(':')] = int(value) * 1024
record['guest_ram_bytes'] = meminfo
baseline = json.loads(Path('/mnt/c/Users/wattz/Documents/ChatGPT/modly/proof/docs/compatibility/amd-wsl-neural-decoder-preparation-evidence/linux-readiness.json').read_text())
record['identity_matches_before'] = record['identity'] == baseline['identity']
record['headers_tools_and_os_match_before'] = all(record[k] == baseline[k] for k in ('tools', 'headers', 'os_release', 'torch_cmake_abi_flags', 'cmake_config_sha256'))
record['identity_differing_keys'] = [k for k in record['identity'] if record['identity'][k] != baseline['identity'].get(k)]
output = Path('/mnt/c/Users/wattz/Documents/ChatGPT/modly/proof/docs/compatibility/amd-wsl-neural-decoder-preparation-evidence/linux-preservation-after.json')
with output.open('x', encoding='utf-8') as stream: json.dump(record, stream, indent=2)
print(json.dumps({key: record[key] for key in ('tools', 'headers', 'torch_cmake_abi_flags',
                                             'filesystem_total_bytes', 'filesystem_available_bytes')}))
