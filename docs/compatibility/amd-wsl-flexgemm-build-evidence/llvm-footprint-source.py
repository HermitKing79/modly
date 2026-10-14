"""One pinned LLVM download and streamed tar inventory; never extract/install."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time
import traceback
import urllib.request
import tarfile

EVIDENCE = Path('/mnt/c/Users/wattz/Documents/ChatGPT/modly/proof/docs/compatibility/amd-wsl-flexgemm-build-evidence')
SCRATCH = Path('/home/modly/flexgemm-build-6dd94a85')
VERSION = '22.0.0.26084.70201-81~24.04'
SIZE = 544593292
SHA = '1cbad039d161a03f9c4df81193fdb81d55d9a79e400ce8ce26ee55488512e0bd'
URL = 'https://repo.radeon.com/rocm/apt/7.2.1/pool/main/r/rocm-llvm/rocm-llvm_' + VERSION + '_amd64.deb'
GiB = 1024 ** 3
def free(path):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize
def gate():
    if free('/') < 5 * GiB + SIZE or free('/mnt/d') < 50 * GiB:
        raise RuntimeError('Archive inspection disk floor failed')
gate()
SCRATCH.mkdir(exist_ok=False)
(SCRATCH / 'inputs').mkdir()
archive = SCRATCH / 'inputs' / ('rocm-llvm_' + VERSION + '_amd64.deb')
record = {'archive': str(archive), 'URL': URL, 'expected_sha256': SHA,
          'expected_bytes': SIZE, 'before_guest_free_bytes': free('/'),
          'before_D_free_bytes': free('/mnt/d'), 'package_installed': False,
          'expanded_files_written': False, 'GPU_or_model_actions': False}
start = time.monotonic()
try:
    digest = hashlib.sha256()
    total = 0
    request = urllib.request.Request(URL, headers={'User-Agent': 'Modly-bounded-package-admission/1'})
    with urllib.request.urlopen(request, timeout=30) as response, archive.open('xb') as output:
        if response.status != 200 or int(response.headers['Content-Length']) != SIZE:
            raise RuntimeError('Pinned archive response size/status mismatch')
        while True:
            if time.monotonic() - start > 600:
                raise TimeoutError('One archive download exceeded600s')
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > SIZE or free('/') < 5 * GiB or free('/mnt/d') < 50 * GiB:
                raise RuntimeError('Archive bounds or disk floor violated')
            output.write(chunk)
            digest.update(chunk)
    record.update(download_bytes=total, actual_sha256=digest.hexdigest(), download_seconds=time.monotonic()-start)
    if total != SIZE or digest.hexdigest() != SHA:
        raise RuntimeError('Pinned archive hash mismatch')
    control = subprocess.run(['dpkg-deb', '--field', str(archive)], capture_output=True,
                             text=True, check=True, timeout=15)
    record['control'] = control.stdout
    if 'Package: rocm-llvm\n' not in control.stdout or 'Version: ' + VERSION + '\n' not in control.stdout:
        raise RuntimeError('Pinned archive control identity mismatch')
    regular = 0
    rounded = 0
    entries = 0
    types = {}
    inspect_start = time.monotonic()
    with (SCRATCH / 'dpkg-deb.stderr').open('xb') as stderr:
        proc = subprocess.Popen(['dpkg-deb', '--fsys-tarfile', str(archive)],
                                stdout=subprocess.PIPE, stderr=stderr)
        try:
            with tarfile.open(fileobj=proc.stdout, mode='r|') as stream:
                for member in stream:
                    if time.monotonic() - inspect_start > 180:
                        raise TimeoutError('Expanded LLVM inventory exceeded180s')
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or '..' in path.parts:
                        raise RuntimeError('Unsafe package member path')
                    entries += 1
                    kind = member.type.decode('ascii', errors='replace')
                    types[kind] = types.get(kind, 0) + 1
                    if member.isfile():
                        regular += member.size
                        rounded += ((member.size + 4095) // 4096) * 4096
                    else:
                        rounded += 4096
            proc.stdout.read()
            rc = proc.wait(timeout=10)
            if rc:
                raise RuntimeError('LLVM archive decompression failed: ' + str(rc))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
    record.update(expanded_regular_file_bytes=regular,
                  conservative_4KiB_member_allocation_bytes=rounded,
                  members=entries, member_types=types,
                  archive_inspection_seconds=time.monotonic()-inspect_start,
                  status='verified')
except BaseException:
    record['status'] = 'blocked'
    record['error'] = traceback.format_exc()
    raise
finally:
    record.update(after_guest_free_bytes=free('/'), after_D_free_bytes=free('/mnt/d'),
                  total_seconds=time.monotonic()-start)
    with (EVIDENCE / 'llvm-footprint.json').open('x', encoding='utf-8') as f:
        json.dump(record, f, indent=2)
    print(json.dumps({k: v for k, v in record.items() if k != 'control'}), flush=True)
