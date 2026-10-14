"""Read-only standard-library package/device audit; no Torch/native imports."""
import base64
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

def deny_ai(event, args):
    if event == 'import' and str(args[0]).split('.')[0] in {'torch', 'triton', 'numpy'}:
        raise RuntimeError('CPU package audit prohibits GPU/AI imports')
sys.addaudithook(deny_ai)

root = Path('/opt/modly-wsl/venv')
rows = []
for directory, dirs, files in os.walk(root, followlinks=False):
    for name in dirs + files:
        path = Path(directory) / name
        stat = path.lstat()
        rows.append([path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns, stat.st_mode])
distributions = list(md.distributions(path=[str(root / 'lib/python3.12/site-packages')]))
checked = []; mismatches = []; skipped = []
for distribution in distributions:
    for entry in distribution.files or []:
        if entry.hash is None:
            skipped.append([distribution.metadata['Name'], str(entry)])
            continue
        path = distribution.locate_file(entry).resolve()
        path.relative_to(root)
        digest = hashlib.new(entry.hash.mode)
        with path.open('rb') as stream:
            while block := stream.read(1024 * 1024): digest.update(block)
        value = base64.urlsafe_b64encode(digest.digest()).decode().rstrip('=')
        checked.append([distribution.metadata['Name'], str(entry), entry.hash.mode, digest.hexdigest()])
        if value != entry.hash.value:
            mismatches.append([distribution.metadata['Name'], str(entry), entry.hash.value, value])
query = subprocess.run(['dpkg-query', '-W', '-f=${Package}\t${Version}\n'],
                       capture_output=True, text=True, timeout=15)
if query.returncode: raise RuntimeError('Read-only dpkg inventory failed')
# Debian package verification reads recorded checksums; it does not repair files.
verify = subprocess.run(['dpkg', '--verify'], capture_output=True, text=True, timeout=45)
torch_version_source = root / 'lib/python3.12/site-packages/torch/version.py'
identity = {'python': platform.python_version(), 'kernel': platform.release(),
            'venv_entries': len(rows),
            'venv_metadata_sha256': hashlib.sha256(json.dumps(sorted(rows), separators=(',', ':')).encode()).hexdigest(),
            'python_packages': sorted([(d.metadata['Name'], d.version) for d in distributions]),
            'dpkg_packages': query.stdout.splitlines(),
            'torch_imported': 'torch' in sys.modules, 'triton_imported': 'triton' in sys.modules}
record = {'identity': identity, 'os_release': platform.freedesktop_os_release(),
          'prefix': sys.prefix, 'executable': sys.executable,
          'dxg_user_read_write': os.access('/dev/dxg', os.R_OK | os.W_OK),
          'dxg_mode': oct(Path('/dev/dxg').stat().st_mode),
          'torch_distribution': md.version('torch'), 'triton_distribution': md.version('triton'),
          'torch_version_source_sha256': hashlib.sha256(torch_version_source.read_bytes()).hexdigest(),
          'python_record_checked': checked, 'python_record_mismatches': mismatches,
          'python_record_unhashed_entries': skipped, 'dpkg_verify_returncode': verify.returncode,
          'dpkg_verify_stdout': verify.stdout, 'dpkg_verify_stderr': verify.stderr}
output = Path('/mnt/c/Users/wattz/Documents/ChatGPT/modly/proof/docs/compatibility/amd-wsl-rocdxg-gpu-resume-evidence/linux-preflight.json')
with output.open('x', encoding='utf-8') as stream: json.dump(record, stream, indent=2)
print(json.dumps({'checked_python_files': len(checked), 'python_mismatches': len(mismatches),
                  'dpkg_verify_returncode': verify.returncode, 'dpkg_verify_stdout': verify.stdout}))
