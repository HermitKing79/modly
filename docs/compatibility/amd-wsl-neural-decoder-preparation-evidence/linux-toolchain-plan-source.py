"""Local APT simulation and package metadata only; never installs or builds."""
import json
from pathlib import Path
import re
import subprocess

requested = ['hip-dev=7.2.53211.70201-81~24.04', 'hipcc=1.1.1.70201-81~24.04',
             'rocm-llvm=22.0.0.26084.70201-81~24.04', 'rocm-device-libs=1.0.0.70201-81~24.04',
             'python3.12-dev', 'build-essential', 'ninja-build', 'hipblas-dev',
             'hipblaslt-dev', 'hipcub-dev', 'hipsolver-dev', 'hipsparse-dev', 'rocthrust-dev']
command = ['apt-get', '--simulate', '--no-install-recommends', 'install', *requested]
plan = subprocess.run(command, capture_output=True, text=True, timeout=30)
records = []; missing = []
for line in plan.stdout.splitlines():
    if not line.startswith('Inst '): continue
    match = re.match(r'Inst (\S+)(?: \[([^\]]+)\])? \((\S+) ', line)
    if not match: raise RuntimeError('Unparsed simulation row')
    name, previous, version = match.groups()
    capture = subprocess.run(['apt-cache', 'show', name + '=' + version],
                             capture_output=True, text=True, timeout=5)
    fields = dict(row.split(': ', 1) for row in capture.stdout.split('\n\n')[0].splitlines()
                  if ': ' in row and not row.startswith(' '))
    record = {key: fields.get(key) for key in ('Package', 'Version', 'Size', 'Installed-Size',
                                              'Filename', 'SHA256', 'Depends')}
    record['previous_version'] = previous
    records.append(record)
    if record['Installed-Size'] is None: missing.append(name)
result = {'requested': requested, 'command': command, 'returncode': plan.returncode,
          'stdout': plan.stdout, 'stderr': plan.stderr, 'packages': records,
          'new_package_count': sum(row['previous_version'] is None for row in records),
          'upgrade_count': sum(row['previous_version'] is not None for row in records),
          'archive_bytes_sum': sum(int(row['Size']) for row in records if row['Size'] is not None),
          'known_installed_bytes_sum': sum(int(row['Installed-Size']) * 1024 for row in records
                                            if row['Installed-Size'] is not None),
          'missing_installed_size_packages': missing,
          'executed_installation': False, 'compiler_or_gpu_execution': False}
output = Path('/mnt/c/Users/wattz/Documents/ChatGPT/modly/proof/docs/compatibility/amd-wsl-neural-decoder-preparation-evidence/linux-toolchain-plan.json')
with output.open('x', encoding='utf-8') as stream: json.dump(result, stream, indent=2)
print(json.dumps({key: result[key] for key in ('returncode', 'new_package_count', 'upgrade_count',
                                              'archive_bytes_sum', 'known_installed_bytes_sum',
                                              'missing_installed_size_packages')}))
