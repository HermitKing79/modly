"""One-shot Windows cold copy. Default is inert; no guest/resize/decoder route."""
import argparse
from contextlib import contextmanager, ExitStack
import ctypes
from ctypes import wintypes as wt
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
from amd_forward_owned_job import OwnedJob, ExtendedLimits
from amd_zombie_ram_diagnostic import windows_memory

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('D:/AI/Modly-WSL/Ubuntu24.04/ext4.vhdx')
BACKUP = Path('E:/Modly-Cold-Backup-closed-solid-v1')
DESTINATION = BACKUP / 'ext4.vhdx'
CASE = ROOT.parent / 'artifacts/amd-wsl-cold-backup-v1'
AUTHORITY = ROOT.parent / 'artifacts/amd-wsl-cold-backup-preparation-20261029-v1/authority.json'
AUTHORITY_SHA = '7abb2681a30cf9bd758256b07faf10759e99719e14eb751b11cdcf64b7b82bc9'
PYTHON = Path('D:/AI/Trellis2AMD/venv/Scripts/python.exe')
PYTHON_SHA = '0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14'
PS = Path('C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')
HELPER = Path(__file__).with_name('amd_wsl_cold_backup_inspect.ps1')
GIB = 1024**3
LIVE = 12*GIB
CAP = 512*1024**2
BUFFER = 4*1024**2
DEADLINE = 1800
EVIDENCE_CAP = 4*1024**2
PUBLISHED = ['scripts/compatibility/' + name for name in (
    'amd_wsl_cold_backup.py', 'amd_wsl_cold_backup_inspect.ps1',
    'Invoke-ModlyColdBackup.ps1', 'amd_forward_owned_job.py',
    'amd_zombie_ram_diagnostic.py')]


def utc(): return datetime.now(timezone.utc).isoformat()


def plain(path, existing=True):
    path = Path(os.path.abspath(path))
    if str(path).startswith('\\\\') or path.drive.upper() not in ('C:', 'D:', 'E:'):
        raise RuntimeError('Only canonical local C:/D:/E: paths')
    for part in [*reversed(path.parents), path]:
        try: s = part.lstat()
        except FileNotFoundError:
            if existing: raise
            continue
        if s.st_file_attributes & 0x400:
            raise RuntimeError('Reparse redirection forbidden: ' + str(part))
    return path


def digest(path):
    with plain(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, row):
    data = (json.dumps(row, indent=2, allow_nan=False) + '\n').encode()
    if len(data) > 65536: raise RuntimeError('Receipt exceeds bound')
    with plain(path, False).open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())


def required_space(size):
    return max(31_728_861_184, size) + 2*GIB + EVIDENCE_CAP


def require_space(free, required):
    if type(free) is not int or free < required: raise RuntimeError('Insufficient disk space')


def check_memory(row):
    if (type(row.get('available_bytes')) is not int or row['available_bytes'] < LIVE
            or type(row.get('private_commit_bytes')) is not int or row['private_commit_bytes'] > CAP):
        raise RuntimeError('CPU backup RAM floor/process ceiling refused')


def stream_backup(source, destination, identity, health, independent, checkpoint,
                  block=BUFFER, create=lambda p: p.open('xb')):
    """Caller holds source lock; never retry, remove, overwrite or resume."""
    before = identity(); expected = before['size']; copied = 0; copy_hash = hashlib.sha256()
    checkpoint('copy-start', expected_bytes=expected)
    with create(destination) as out:
        while copied < expected:
            health()
            data = source.read(min(block, expected-copied))
            if not data: raise RuntimeError('Source length changed/short read')
            written = out.write(data)
            if written != len(data): raise OSError('Short destination write')
            copy_hash.update(data); copied += len(data)
            checkpoint('copy-progress', bytes=copied)
        if source.read(1): raise RuntimeError('Source length grew')
        out.flush(); os.fsync(out.fileno())
    if identity() != before: raise RuntimeError('Source identity changed')
    checkpoint('copy-flushed', bytes=copied)
    source.seek(0); source_hash = hashlib.sha256(); verified = 0
    while data := source.read(block):
        health(); source_hash.update(data); verified += len(data)
        if verified > expected: raise RuntimeError('Source length grew during hash')
        checkpoint('source-hash-progress', bytes=verified)
    if verified != expected or source_hash.hexdigest() != copy_hash.hexdigest():
        raise RuntimeError('Complete source hash differs from copied bytes')
    if identity() != before: raise RuntimeError('Source identity changed')
    checkpoint('source-hash-complete', bytes=verified)
    other = independent(destination)
    if other['size'] != expected or other['sha256'] != source_hash.hexdigest():
        raise RuntimeError('Independent destination hash/length differs')
    health()
    if identity() != before: raise RuntimeError('Source identity changed')
    checkpoint('independent-destination-hash-complete', bytes=expected)
    return dict(bytes=expected, copy_sha256=copy_hash.hexdigest(),
                source_sha256=source_hash.hexdigest(), destination_sha256=other['sha256'],
                source_identity=before)


def kernel():
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    signatures = {
        'CreateFileW': ([wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE], wt.HANDLE),
        'CloseHandle': ([wt.HANDLE], wt.BOOL),
        'GetFileInformationByHandle': ([wt.HANDLE, ctypes.c_void_p], wt.BOOL),
        'GetFinalPathNameByHandleW': ([wt.HANDLE, wt.LPWSTR, wt.DWORD, wt.DWORD], wt.DWORD)}
    for name, (args, ret) in signatures.items():
        getattr(k, name).argtypes = args; getattr(k, name).restype = ret
    return k


def handle_identity(handle):
    class Info(ctypes.Structure):
        _fields_ = [('attributes', wt.DWORD), ('created', wt.FILETIME), ('accessed', wt.FILETIME),
                    ('written', wt.FILETIME), ('volume', wt.DWORD), ('size_high', wt.DWORD),
                    ('size_low', wt.DWORD), ('links', wt.DWORD), ('index_high', wt.DWORD), ('index_low', wt.DWORD)]
    k = kernel(); info = Info()
    if not k.GetFileInformationByHandle(handle, ctypes.byref(info)): raise ctypes.WinError()
    name = ctypes.create_unicode_buffer(32768)
    n = k.GetFinalPathNameByHandleW(handle, name, len(name), 0)
    if not 0 < n < len(name): raise ctypes.WinError()
    if info.attributes & 0x400: raise RuntimeError('Reparse handle forbidden')
    return dict(size=(info.size_high << 32) | info.size_low,
                file_id=(info.index_high << 32) | info.index_low,
                volume_serial=format(info.volume, '08X'), links=info.links,
                mtime_ns=(((info.written.dwHighDateTime << 32) | info.written.dwLowDateTime)-116444736000000000)*100,
                final_path=name.value.removeprefix('\\\\?\\'), attributes=info.attributes)


def file_identity(f): return handle_identity(msvcrt.get_osfhandle(f.fileno()))


def verify_destination_identity(created, current):
    # Windows finalizes the write timestamp on closing the last writer.
    # File identity/length/path/attributes must survive that close; thereafter
    # the shared-read verification handle freezes all fields, including mtime.
    if {k:v for k,v in created.items() if k != 'mtime_ns'} != {k:v for k,v in current.items() if k != 'mtime_ns'}:
        raise RuntimeError('Destination identity changed')


@contextmanager
def native_file(path, purpose):
    p = plain(path, purpose != 'create'); k = kernel()
    if purpose == 'source': access, sharing, creation, mode = 0x80000000, 0, 3, 'rb'
    elif purpose == 'create': access, sharing, creation, mode = 0x40000000, 0, 1, 'wb'
    elif purpose == 'verify': access, sharing, creation, mode = 0x80000000, 1, 3, 'rb'
    else: raise ValueError('Unknown handle purpose')
    h = k.CreateFileW(str(p), access, sharing, None, creation, 0x08200000, None)
    if h == wt.HANDLE(-1).value: raise ctypes.WinError()
    try:
        info = handle_identity(h)
        if info['links'] != 1 or os.path.normcase(info['final_path']) != os.path.normcase(str(p)):
            raise RuntimeError('Canonical single-link file required')
        fd = msvcrt.open_osfhandle(h, os.O_BINARY | (os.O_RDONLY if mode == 'rb' else os.O_WRONLY))
    except BaseException:
        k.CloseHandle(h); raise
    with os.fdopen(fd, mode, buffering=0) as f: yield f


class Pins:
    """No-delete directory handles retain the canonical ancestry during I/O."""
    def __init__(self): self.handles = {}; self.k = kernel()
    def add(self, path):
        p = plain(path)
        for part in [*reversed(p.parents), p]:
            key = os.path.normcase(str(part))
            if key in self.handles: continue
            h = self.k.CreateFileW(str(part), 0x81, 3, None, 3, 0x02200000, None)
            if h == wt.HANDLE(-1).value: raise ctypes.WinError()
            try:
                row = handle_identity(h)
                if os.path.normcase(row['final_path'].rstrip('\\')) != key.rstrip('\\'):
                    raise RuntimeError('Directory handle identity differs')
            except BaseException:
                self.k.CloseHandle(h); raise
            self.handles[key] = h
        return handle_identity(self.handles[os.path.normcase(str(p))])
    def close(self):
        errors = []
        for h in reversed(list(self.handles.values())):
            if not self.k.CloseHandle(h): errors.append(ctypes.get_last_error())
        self.handles.clear()
        if errors: raise RuntimeError('Directory handle cleanup failed: ' + str(errors))


def metadata(f):
    f.seek(0)
    if f.read(8) != b'vhdxfile': raise RuntimeError('Not a VHDX')
    f.seek(196608); region = f.read(65536)
    if region[:4] != b'regi': raise RuntimeError('Missing region table')
    count = struct.unpack_from('<I', region, 8)[0]
    if not 0 < count < 2048: raise RuntimeError('Invalid region count')
    entries = []
    for i in range(count):
        guid, offset, length, _ = struct.unpack_from('<16sQII', region, 16+i*32)
        if str(uuid.UUID(bytes_le=guid)) == '8b7ca206-4790-4b9a-b8fe-575f050f886e': entries.append((offset, length))
    if len(entries) != 1: raise RuntimeError('Metadata region not unique')
    offset, length = entries[0]
    if length < 65536 or offset+length > os.fstat(f.fileno()).st_size: raise RuntimeError('Metadata outside file')
    f.seek(offset); table = f.read(65536)
    if table[:8] != b'metadata': raise RuntimeError('Missing metadata signature')
    count = struct.unpack_from('<H', table, 10)[0]
    if not 0 < count < 2048: raise RuntimeError('Invalid metadata count')
    sizes = []
    for i in range(count):
        guid, start, size, _, _ = struct.unpack_from('<16sIIII', table, 32+i*32)
        if str(uuid.UUID(bytes_le=guid)) == '2fa54224-cd1b-4876-b211-5dbed83bf4b8':
            if size != 8 or start+8 > length: raise RuntimeError('Invalid virtual capacity item')
            f.seek(offset+start); sizes.append(struct.unpack('<Q', f.read(8))[0])
    if len(sizes) != 1: raise RuntimeError('Capacity not unique')
    f.seek(0)
    return dict(virtual_capacity_bytes=sizes[0], region_table_sha256=hashlib.sha256(region).hexdigest(),
                metadata_table_sha256=hashlib.sha256(table).hexdigest())


def helper_command(action, path=None):
    cmd = [str(PS), '-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', str(HELPER)]
    if action == 'inventory' and path is None: return cmd + ['-Inventory', '-WaitForOwner']
    if action == 'hash' and path is not None: return cmd + ['-Hash', '-LiteralPath', str(path)]
    raise ValueError('Only read-only inventory or independent hash')


def require_quiet(row):
    if row.get('running') != [] or row.get('vm_processes') != []:
        raise RuntimeError('Conflicting WSL activity; never stop it')


def require_volume(identity, expected):
    if identity.get('volume_serial') != expected:
        raise RuntimeError('Actual handle volume differs from pinned physical drive')


def authenticate_inventory(row, authority):
    require_quiet(row)
    if row.get('registration') != authority.get('registration') or not authority.get('registration'):
        raise RuntimeError('WSL registration differs from authority')
    for field in ('logical', 'disks', 'partitions'):
        for expected in authority[field]:
            matches = [x for x in row.get(field, []) if all(x.get(k) == v for k, v in expected.items())]
            if len(matches) != 1: raise RuntimeError('Physical drive/volume identity differs: ' + field)


def authority():
    if digest(AUTHORITY) != AUTHORITY_SHA: raise RuntimeError('Private authority changed/missing')
    return json.loads(AUTHORITY.read_text(encoding='utf-8'))


def inputs(a):
    checked = {}
    for relative, wanted in a['box_inputs'].items():
        p = plain(ROOT / relative); s = p.stat()
        if (s.st_size, s.st_mtime_ns, s.st_ino) != (wanted['bytes'], wanted['mtime_ns'], wanted['file_id']):
            raise RuntimeError('Box input identity changed')
        if digest(p) != wanted['sha256']: raise RuntimeError('Box input hash changed')
        after = p.stat()
        if (s.st_size, s.st_mtime_ns, s.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise RuntimeError('Box input changed during read')
        checked[relative] = wanted
    return checked


def published(commit):
    if not re.fullmatch('[a-f0-9]{40}', commit or ''): raise RuntimeError('Exact published commit required')
    def git(args): return subprocess.check_output(['git', '-C', str(ROOT), *args], timeout=15)
    if git(['rev-parse', 'HEAD']).decode().strip() != commit: raise RuntimeError('Published checkout differs')
    for name in PUBLISHED:
        original = git(['show', commit+':'+name]).replace(b'\r\n', b'\n')
        if plain(ROOT/name).read_bytes().replace(b'\r\n', b'\n') != original:
            raise RuntimeError('Published backup source differs: ' + name)


class Controller:
    def __init__(self): self.start = time.monotonic(); self.last = 0; self.min_available = None
    def health(self, log=True):
        if time.monotonic()-self.start > DEADLINE: raise TimeoutError('Backup total deadline')
        row = windows_memory(); check_memory(row)
        require_space(shutil.disk_usage('D:/').free, 50*GIB)
        require_space(shutil.disk_usage('E:/').free, 2*GIB)
        require_space(shutil.disk_usage(CASE).free, 2*GIB)
        self.min_available = min(self.min_available or row['available_bytes'], row['available_bytes'])
        if sum(p.stat().st_size for p in CASE.rglob('*') if p.is_file()) > EVIDENCE_CAP:
            raise RuntimeError('Evidence size limit')
        if log and time.monotonic()-self.last >= 1:
            with (CASE/'host-memory.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(utc=utc(), seconds=time.monotonic()-self.start, **row))+'\n')
            self.last = time.monotonic()

    def child(self, cmd, prefix, timeout, payload=None):
        """Fixed helper or this worker; Job includes all owned descendants."""
        allowed = (cmd == helper_command('inventory') or
                   cmd == [str(PYTHON), '-B', '-I', '-S', str(Path(__file__).resolve()), '--worker'])
        if not allowed: raise RuntimeError('Unapproved controller child')
        job = OwnedJob(); child = None; attached = False; began = time.monotonic(); result = {}
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TEMP=str(CASE/'tmp'), TMP=str(CASE/'tmp'),
                   PSModuleAnalysisCachePath='NUL')
        limits = ExtendedLimits(); limits.Basic.Flags = 0x2300
        limits.ProcessMemory = CAP; limits.JobMemory = CAP
        try:
            if not job.api.SetInformationJobObject(job.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)): raise ctypes.WinError()
            with (CASE/(prefix+'-stdout.txt')).open('xb') as out, (CASE/(prefix+'-stderr.txt')).open('xb') as err:
                child = subprocess.Popen(cmd, executable=cmd[0], shell=False, cwd=CASE, env=env,
                                         stdin=subprocess.PIPE, stdout=out, stderr=err, creationflags=0x08000000)
                job.attach(child); attached = True
                if payload is not None:
                    save(CASE/'worker-owner.json', dict(pid=child.pid, parent=os.getpid(), token=payload['token']))
                    child.stdin.write(json.dumps(payload).encode()+b'\n'); child.stdin.flush()
                else:
                    # Inventory cannot query or spawn wsl.exe before Job attachment.
                    child.stdin.write(b'MODLY_COLD_BACKUP_INVENTORY_GO\n'); child.stdin.flush()
                child.stdin.close()
                exited = None
                while True:
                    self.health()
                    if time.monotonic()-began > timeout: raise TimeoutError('Owned stage deadline: '+prefix)
                    if any((CASE/(prefix+'-'+n+'.txt')).stat().st_size > 65536 for n in ('stdout', 'stderr')):
                        raise RuntimeError('Bounded child output exceeded')
                    if child.poll() is not None:
                        if child.returncode != 0: raise RuntimeError('Owned stage failed: '+prefix)
                        if job.active() == 0: break
                        if exited is None: exited = time.monotonic()
                        if time.monotonic()-exited > 5: raise TimeoutError('Owned descendants did not exit')
                    time.sleep(.2)
                result['returncode'] = child.returncode
                peaks = ExtendedLimits()
                if not job.api.QueryInformationJobObject(job.handle, 9, ctypes.byref(peaks), ctypes.sizeof(peaks), None): raise ctypes.WinError()
                result.update(peak_process_commit_bytes=peaks.PeakProcessMemory, peak_job_commit_bytes=peaks.PeakJobMemory)
        finally:
            try: result['cleanup'] = job.finish()
            finally:
                if child and child.stdin and not child.stdin.closed: child.stdin.close()
                if child and child.poll() is None:
                    if not attached: child.kill()
                    child.wait(timeout=5)
                if child:
                    result.update(pid=child.pid, returncode=child.returncode, job_attached=attached)
                result['seconds'] = time.monotonic()-began
                save(CASE/(prefix+'-process.json'), result)
        if result['returncode'] != 0 or result['cleanup']['forced_job_cleanup']:
            raise RuntimeError('Owned stage failed: '+prefix)
        return (CASE/(prefix+'-stdout.txt')).read_bytes().decode('utf-8-sig')


def worker():
    """Only starts after owner Job attachment and exact token/EOF handshake."""
    line = sys.stdin.buffer.read(4097)
    if len(line) > 4096: raise RuntimeError('Bounded owner handshake required')
    payload = json.loads(line); owner = json.loads((CASE/'worker-owner.json').read_text())
    if (not re.fullmatch('[a-f0-9]{32}', payload.get('token', '')) or owner != dict(pid=os.getpid(), parent=os.getppid(), token=payload['token'])):
        raise RuntimeError('Exact owner identity/GO required')
    a = authority(); pins = Pins(); source_before = None; result = dict(classification='FAIL', utc=utc())
    e_serial = next(x['VolumeSerialNumber'] for x in a['logical'] if x['DeviceID'] == 'E:')
    started = time.monotonic(); last = {}; destination_identity = None
    def check():
        if time.monotonic()-started > DEADLINE: raise TimeoutError('Worker deadline')
    def phase(name, **kw):
        check(); now = time.monotonic()
        if name.endswith('progress') and now-last.get(name, 0) < 5: return
        last[name] = now
        with (CASE/'phases.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(phase=name, utc=utc(), elapsed=now-started, **kw))+'\n')
    @contextmanager
    def create(path):
        nonlocal destination_identity
        with native_file(path, 'create') as f:
            require_volume(file_identity(f), e_serial)
            yield f
            destination_identity = file_identity(f)
    def independent(path):
        with native_file(path, 'verify') as f:
            before = file_identity(f)
            verify_destination_identity(destination_identity, before)
            with (CASE/'destination-hash-stdout.txt').open('xb') as out, (CASE/'destination-hash-stderr.txt').open('xb') as err:
                p = subprocess.Popen(helper_command('hash', path), executable=str(PS), shell=False,
                                     stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=0x08000000)
                if p.wait(timeout=DEADLINE) != 0: raise RuntimeError('Independent .NET hashing failed')
            row = json.loads((CASE/'destination-hash-stdout.txt').read_text(encoding='utf-8-sig'))
            if file_identity(f) != before: raise RuntimeError('Destination identity changed')
            result['destination_metadata'] = metadata(f)
            result['destination_identity'] = before
            return row
    try:
        pins.add(SOURCE.parent); pins.add(CASE)
        with native_file(SOURCE, 'source') as src:
            source_before = file_identity(src)
            for k, v in a['source_identity'].items():
                if source_before.get(k) != v: raise RuntimeError('Original VHDX identity differs: '+k)
            result['source_identity'] = source_before; result['source_metadata'] = metadata(src)
            if result['source_metadata'] != a['vhdx_metadata']: raise RuntimeError('Original VHDX metadata changed')
            # Source is now frozen; inventory is Windows-only and never mounts it.
            p = subprocess.run(helper_command('inventory'), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, input=b'MODLY_COLD_BACKUP_INVENTORY_GO\n',
                               timeout=30, check=True, creationflags=0x08000000)
            if len(p.stdout)+len(p.stderr) > 65536: raise RuntimeError('Inventory output limit')
            inventory = json.loads(p.stdout.decode('utf-8-sig')); authenticate_inventory(inventory, a)
            save(CASE/'locked-inventory.json', inventory)
            require_space(shutil.disk_usage('E:/').free, required_space(source_before['size']))
            require_volume(pins.add(BACKUP.parent), e_serial)
            plain(BACKUP, False).mkdir()
            require_volume(pins.add(BACKUP), e_serial)
            result['backup_directory_created'] = True
            row = stream_backup(src, DESTINATION, lambda: file_identity(src), check, independent, phase, create=create)
            if metadata(src) != result['source_metadata'] or result['destination_metadata'] != result['source_metadata']:
                raise RuntimeError('Copied metadata differs')
            result.update(row)
            result['source_after'] = file_identity(src)
            if result['source_after'] != source_before: raise RuntimeError('Source identity changed')
            # Both hashes and metadata are final while the exclusive source lock remains held.
            result['classification'] = 'PASS'
            save(BACKUP/'BYTE-VERIFICATION.json', dict(result, classification='BYTE_VERIFICATION_PASS_PENDING_CONTROLLER',
                                                     source=str(SOURCE), destination=str(DESTINATION),
                                                     authority_sha256=AUTHORITY_SHA, commit=payload['commit']))
    except BaseException as exc:
        result.update(classification='FAIL', error=type(exc).__name__+': '+str(exc))
    finally:
        try: pins.close(); result['handles_closed'] = True
        except Exception as exc: result.update(classification='FAIL', cleanup_error=str(exc))
        result.update(seconds=time.monotonic()-started, source=str(SOURCE), destination=str(DESTINATION),
                      partial_files_preserved=True, no_source_write_requested=True, no_guest_launch=True)
        save(CASE/'worker-result.json', result)
    return 0 if result['classification'] == 'PASS' else 1


def run(commit):
    published(commit)
    if digest(PYTHON) != PYTHON_SHA: raise RuntimeError('Qualified interpreter differs')
    a = authority()
    # No evidence or destination reuse, fallback, resume, or retry.
    for p in (CASE, BACKUP):
        plain(p, False)
        if p.exists(): raise FileExistsError('One-shot path consumed: '+str(p))
    pins = Pins(); created = False
    controller = Controller(); report = dict(classification='FAIL', commit=commit, started=utc(),
                                           source=str(SOURCE), destination=str(DESTINATION), evidence=str(CASE))
    try:
        pins.add(CASE.parent); plain(CASE, False).mkdir(); created = True; pins.add(CASE)
        (CASE/'tmp').mkdir(); pins.add(CASE/'tmp')
        controller.health(); require_space(shutil.disk_usage('E:/').free, required_space(a['source_identity']['size']))
        save(CASE/'box-inputs-before.json', inputs(a))
        inventory = json.loads(controller.child(helper_command('inventory'), 'inventory', 30))
        authenticate_inventory(inventory, a); save(CASE/'registration-and-storage.json', inventory)
        controller.health()
        token = uuid.uuid4().hex
        save(CASE/'contract.json', dict(token=token, commit=commit, authority_sha256=AUTHORITY_SHA,
                                       process_limit_bytes=CAP, wall_seconds=DEADLINE, buffer_bytes=BUFFER,
                                       live_host_floor_bytes=LIVE, neural_gates_changed=False))
        controller.child([str(PYTHON), '-B', '-I', '-S', str(Path(__file__).resolve()), '--worker'],
                         'backup-worker', DEADLINE, dict(token=token, commit=commit))
        worker_row = json.loads((CASE/'worker-result.json').read_text())
        if worker_row['classification'] != 'PASS' or worker_row.get('handles_closed') is not True:
            raise RuntimeError('Worker verification/handle cleanup failed')
        # No guest was started; final metadata query must still show a quiescent system.
        after = json.loads(controller.child(helper_command('inventory'), 'final-inventory', 30))
        authenticate_inventory(after, a); save(CASE/'final-registration-and-storage.json', after)
        save(CASE/'box-inputs-after.json', inputs(a)); controller.health()
        report.update(classification='PASS', verification=worker_row)
    except BaseException as exc:
        report['error'] = type(exc).__name__+': '+str(exc)
    finally:
        try: pins.close(); report['controller_handles_closed'] = True
        except Exception as exc: report.update(classification='FAIL', cleanup_error=str(exc))
        report.update(seconds=time.monotonic()-controller.start, minimum_observed_host_available_bytes=controller.min_available,
                      no_retry=True, no_guest_command=True, no_resize=True, no_neural_execution=True)
        try:
            s = plain(DESTINATION, False).stat()
            report['destination_observation'] = dict(exists=True, bytes=s.st_size, mtime_ns=s.st_mtime_ns)
        except FileNotFoundError: report['destination_observation'] = dict(exists=False)
        except Exception as exc: report['destination_observation'] = dict(error=type(exc).__name__+': '+str(exc))
        if created:
            save(CASE/'final-result.json', report)
            with (CASE/'REPORT.txt').open('x', encoding='utf-8') as f:
                f.write('COLD_BACKUP='+report['classification']+'\n'+json.dumps(report, indent=2)+'\n')
            print('COLD_BACKUP='+report['classification']+'\nEVIDENCE='+str(CASE), flush=True)
        else:
            print(json.dumps(dict(report, evidence_directory_not_created=True)), file=sys.stderr)
    return 0 if report['classification'] == 'PASS' else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true'); p.add_argument('--approve-cold-backup', action='store_true')
    p.add_argument('--commit'); p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.worker:
        if args.run or args.approve_cold_backup or args.commit: p.error('Invalid internal routing')
        return worker()
    if not args.run:
        print('DRY_RUN: no inventory, resource query, child, VHDX access or backup. Separate approval required.')
        return 0
    if not args.approve_cold_backup or not re.fullmatch('[a-f0-9]{40}', args.commit or ''):
        p.error('Separate --approve-cold-backup and exact published --commit required')
    return run(args.commit)


if __name__ == '__main__': raise SystemExit(main())
