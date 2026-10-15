"""Exact-leaf Windows authority; no permission, mount or environment changes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

E = Path(__file__).resolve().parent
WINDOWS_ROOT = 'C:/Users/wattz/Documents/ModlyArtifacts/amd-turret-texture-owner-067f668'
ROOT = Path(WINDOWS_ROOT if sys.platform == 'win32' else '/mnt/c/' + WINDOWS_ROOT[3:])
PROOF = E/'owner-attestation.json'
EXPECTED_SHA256 = 'f5209e299c5f50d14465269f2b933ea0e5023b4690bf2b9501a29d5931b9a88e'
_contexts = {}
WRITE_MASK = 0x000D0156 | 0x50000000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(root, files=False):
    if sys.platform != 'win32' or root not in (ROOT/'field', ROOT/'probe'):
        raise RuntimeError('Host ownership query is confined to two approved leaves')
    powershell = Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/WindowsPowerShell/v1.0/powershell.exe'
    command = [str(powershell),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
               '-File',str(E/'windows_owner.ps1'),'-Root',str(root)]
    if files:
        command.append('-Files')
    run = subprocess.run(command, capture_output=True, timeout=10)
    if run.returncode or run.stderr:
        raise RuntimeError('Read-only Windows ownership query failed: '+run.stderr.decode(errors='replace'))
    value = json.loads(run.stdout.decode('utf-8-sig'))
    sid = value['current_sid']
    trusted = {sid, 'S-1-5-18', 'S-1-5-32-544'}
    if (value['root'].replace('\\','/') != root.as_posix() or value['C_filesystem'] != 'NTFS'
            or any(row['reparse'] for row in value['ancestors'])):
        raise ValueError('Windows canonical path/filesystem/reparse admission failed')
    for row in [*value['ancestors'][:2], *value['files']]:
        if row['owner_sid'] != sid or row['reparse']:
            raise ValueError('Artifact Windows owner is not the current trusted host user')
        for rule in row['rules']:
            if (rule['type'] == 'Allow' and rule['propagation'] != 'InheritOnly'
                    and rule['rights'] & WRITE_MASK and rule['sid'] not in trusted):
                raise ValueError('Untrusted Windows write/delete/ACL grant')
        info = Path(row['path']).lstat()
        if not Path(row['path']).is_dir() and info.st_nlink != 1:
            raise ValueError('Artifact Windows file is not single-link')
        row['identity'] = {'device': info.st_dev, 'inode': info.st_ino}
    return value


def proof():
    if digest(PROOF) != EXPECTED_SHA256:
        raise ValueError('Trusted host ownership attestation changed')
    record = json.loads(PROOF.read_text())
    if record.get('format') != 'modly-owned-drvfs-v1':
        raise ValueError('Foreign ownership proof format')
    return record


def host_verify(root, files=False):
    record = proof()
    row = next((r for r in record['roots'] if r['windows_path'] == root.as_posix()), None)
    current = snapshot(root, files)
    if row is None or current['current_sid'] != record['windows_user_sid']:
        raise ValueError('Foreign Windows authority/root')
    old = row['host_snapshot']
    for fresh, prior in zip(current['ancestors'][:2], old['ancestors'][:2]):
        if any(fresh[k] != prior[k] for k in ('path','owner_sid','sddl','identity')):
            raise ValueError('Windows artifact root/parent identity or ACL changed')
    return current


def ownership(root):
    if root not in (ROOT/'field', ROOT/'probe'):
        raise ValueError('Foreign ownership leaf')
    if sys.platform == 'win32':
        host_verify(root)
        return None
    if root not in _contexts:
        from amd_texture_field_artifact import drvfs_ownership
        _contexts[root] = drvfs_ownership(root, proof(), EXPECTED_SHA256)
    return _contexts[root]
