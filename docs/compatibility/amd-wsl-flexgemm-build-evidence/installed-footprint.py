"""Read-only actual package-owned file allocations; excludes shared directories."""
import json
import os
from pathlib import Path
import subprocess

E=Path('/mnt/c/Users/wattz/Documents/ChatGPT/modly/proof/docs/compatibility/amd-wsl-flexgemm-build-evidence')
packages=json.loads((E/'toolchain-result.json').read_text())['approved_packages']
seen=set();seen_new=set();all_blocks=0;new_blocks=0;logical=0;counts=0
for pkg in packages:
    lines=subprocess.check_output(['dpkg-query','--listfiles',pkg['Package']],text=True).splitlines()
    for name in lines:
        p=Path(name)
        if not p.is_absolute():continue
        try:st=p.lstat()
        except FileNotFoundError:continue
        if p.is_dir() and not p.is_symlink():continue
        inode=(st.st_dev,st.st_ino)
        if inode not in seen:
            seen.add(inode);all_blocks+=st.st_blocks*512;logical+=st.st_size;counts+=1
        if pkg['previous_version'] is None and inode not in seen_new:
            seen_new.add(inode);new_blocks+=st.st_blocks*512
record={'all61_package_owned_nondirectory_allocated_bytes':all_blocks,
        'new59_package_owned_nondirectory_allocated_bytes':new_blocks,
        'all61_logical_bytes':logical,'unique_nondirectory_inodes':counts,
        'meaning':'Observed allocated filesystem blocks; directories/dpkg metadata excluded. All61 includes two replacement packages; this is not claimed as exact net-new disk consumption.',
        'final_guest_filesystem_total_bytes':os.statvfs('/').f_blocks*os.statvfs('/').f_frsize,
        'final_guest_free_bytes':os.statvfs('/').f_bavail*os.statvfs('/').f_frsize,
        'final_D_free_bytes':os.statvfs('/mnt/d').f_bavail*os.statvfs('/mnt/d').f_frsize,
        'GPU_or_model_actions':False}
with (E/'installed-footprint.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
