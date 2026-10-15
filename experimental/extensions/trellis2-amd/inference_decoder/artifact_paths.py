"""Exclusive guest-local run leaves. Never writes through a Windows mount."""
import os
from pathlib import Path
import stat
import sys


def attest_parent(contract):
    if sys.platform != 'linux':
        raise RuntimeError('Decoder execution requires the qualified Linux guest')
    parent = Path(str(contract.output_directory.parent))
    if parent != Path(str(contract.scratch_directory.parent)):
        raise ValueError('Run leaf parents differ')
    for path in (Path('/home'),Path('/home/modly'),parent):
        info=path.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or path.resolve(strict=True) != path):
            raise RuntimeError('Run directory is redirected or not a regular directory')
        if path != Path('/home') and (info.st_uid != os.getuid() or info.st_mode & 0o022):
            raise RuntimeError('Run directory must be owned and protected from foreign writes')
    if parent.stat().st_dev != Path('/home/modly').stat().st_dev:
        raise RuntimeError('Run parent must remain on the guest home filesystem')
    return parent


def claim_directory(contract, kind):
    parent = attest_parent(contract)
    path = Path(str(contract.output_directory if kind == 'output' else contract.scratch_directory))
    if kind not in ('output','scratch') or path.parent != parent:
        raise ValueError('Unknown run leaf')
    path.mkdir(mode=0o700, exist_ok=False)
    info = path.lstat()
    if (path.resolve(strict=True) != path or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_dev != parent.stat().st_dev):
        raise RuntimeError('Fresh run leaf ownership, permissions or filesystem differs')
    return path


def attest_output(contract):
    parent=attest_parent(contract); path=Path(str(contract.output_directory)); info=path.lstat()
    if (path.parent != parent or path.resolve(strict=True) != path or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_dev != parent.stat().st_dev):
        raise RuntimeError('Owned output directory changed')
    return path
