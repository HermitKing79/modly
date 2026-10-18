"""CPU-only confinement checks for the repair host; no WSL/model execution."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'docs/compatibility/amd-brick-pbr-plan-repair-evidence/host.py'
ORIGINAL = ROOT.parent/'artifacts/amd-brick-neural-completion-c9447dd/neural-output/run-contract.json'
ORIGINAL_SHA256 = '4676626d1061e6cd7090b560f952a974a584f8ea9e6218378b7902294b93bd4f'
ORIGINAL_GUEST = '/mnt/c/Users/wattz/Documents/ChatGPT/modly/artifacts/amd-brick-neural-completion-c9447dd/neural-output/run-contract.json'
sys.dont_write_bytecode = True
sys.pycache_prefix = str(ROOT/'.tmp'/('forbidden-plan-path-bytecode-'+uuid.uuid4().hex))


def cpu_only(event, args):
    if event == 'import' and str(args[0]).split('.')[0].casefold() in {
            'torch', 'triton', 'trellis2', 'flex_gemm', 'flexgemm', 'o_voxel',
            'cumesh', 'nvdiffrast', 'spconv', 'flash_attn', 'vox2seq'}:
        raise RuntimeError('Plan path tests forbid Torch/model/GPU/native imports')
    if (event.startswith('socket.') or event.startswith('os.exec')
            or event.startswith('os.spawn') or event.startswith('os.fork')
            or event in {'subprocess.Popen', 'os.system', 'os.posix_spawn',
                         'os.startfile', 'os.startfile/2'}):
        raise RuntimeError('Plan path tests forbid processes, WSL and network')


sys.addaudithook(cpu_only)


def load_host():
    spec = importlib.util.spec_from_file_location('_brick_pbr_plan_path_cpu_tests', SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def fixture_directory():
    parent = (ROOT/'.tmp/brick-pbr-plan-path-fixtures').resolve()
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(dir=parent)).resolve()
    try:
        yield directory
    finally:
        if directory.parent != parent or directory.resolve(strict=True) != directory:
            raise ValueError('Fixture cleanup escaped its exact workspace parent')
        shutil.rmtree(directory)


def changed_stat(info, **changes):
    values = {name: getattr(info, name) for name in dir(info) if name.startswith('st_')}
    values.update(changes)
    return SimpleNamespace(**values)


@unittest.skipUnless(os.name == 'nt', 'Production routing requires Windows C: paths')
class PlanPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = load_host()

    def test_exact_original_contract_and_existing_approved_roots_convert(self):
        # Break caught: widening the exception is unnecessary; losing it blocks the plan.
        h = self.host
        self.assertEqual(hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(), ORIGINAL_SHA256)
        self.assertEqual(h.to_guest(ORIGINAL), ORIGINAL_GUEST)
        for path, expected in (
                (h.ROOT/'docs/compatibility/amd-brick-pbr-plan-repair-evidence/worker.py',
                 '/mnt/c/Users/wattz/Documents/ChatGPT/modly/proof/docs/compatibility/amd-brick-pbr-plan-repair-evidence/worker.py'),
                (h.PRIVATE/'execution-plan.json',
                 '/mnt/c/Users/wattz/Documents/ChatGPT/modly/artifacts/amd-brick-pbr-plan-repair-375ed55/execution-plan.json'),
                (h.DESTINATION/'probe/execution-plan.json',
                 '/mnt/c/Users/wattz/Documents/ModlyArtifacts/amd-brick-pbr-plan-repair-375ed55/probe/execution-plan.json')):
            with self.subTest(path=path):
                self.assertEqual(h.to_guest(path), expected)
        h.require_source_cache()

    def test_geometry_exception_does_not_admit_siblings_or_ancestor_directories(self):
        # Break caught: treating GEOMETRY or its artifact tree as an approved root.
        for path in (ORIGINAL.parent, ORIGINAL.parent.parent,
                     ORIGINAL.parent/'shape-decoded.safetensors',
                     ORIGINAL.parent/'foreign.json', ORIGINAL.with_name('RUN-CONTRACT.JSON'),
                     ROOT.parent/'artifacts/unrelated/run-contract.json',
                     Path('C:/Users/wattz/Documents/foreign-run-contract.json')):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.host.to_guest(path)

    def test_raw_relative_traversal_drive_and_noncanonical_aliases_are_rejected(self):
        # Break caught: Path normalization/case-insensitive equality hiding raw aliases.
        canonical = str(ORIGINAL)
        root = str(ROOT)
        aliases = (
            'docs/compatibility/amd-brick-pbr-plan-repair-evidence/worker.py',
            'C:Users/wattz/Documents/ChatGPT/modly/proof/worker.py',
            root+'\\.\\docs\\compatibility\\amd-brick-pbr-plan-repair-evidence\\worker.py',
            root+'\\docs\\..\\docs\\compatibility\\amd-brick-pbr-plan-repair-evidence\\worker.py',
            str(ORIGINAL.parent)+'\\.\\run-contract.json',
            str(ORIGINAL.parent)+'\\..\\neural-output\\run-contract.json',
            canonical+':foreign', canonical+':$DATA', canonical+'.', canonical+' ',
            canonical.replace('C:', 'D:', 1), canonical.replace('C:', 'c:', 1),
            canonical.replace('Users', 'users', 1),
            root.replace('Users', 'users', 1)+'\\docs\\compatibility',
            root.replace('C:', 'c:', 1)+'\\docs\\compatibility',
            '\\\\?\\'+canonical, '\\\\localhost\\C$\\'+canonical[3:],
        )
        for path in aliases:
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.host.to_guest(path)

    def test_exact_original_name_rejects_changed_bytes_without_writing_original(self):
        # Break caught: granting the special path by name without authenticating its bytes.
        real_open = Path.open
        before = ORIGINAL.stat()
        observed = ORIGINAL.read_bytes()
        replacement = observed[:-1]+(b' ' if observed[-1:] != b' ' else b'\n')
        self.assertEqual(len(replacement), len(observed))
        self.assertEqual(json.loads(replacement), json.loads(observed))
        self.assertNotEqual(hashlib.sha256(replacement).hexdigest(), ORIGINAL_SHA256)

        class ChangedRead:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def fileno(self):
                return self.stream.fileno()

            def read(self, size=-1):
                value = self.stream.read(size)
                return replacement if value == observed else value

        def substituted_open(path, *args, **kwargs):
            stream = real_open(path, *args, **kwargs)
            return ChangedRead(stream) if str(path) == str(ORIGINAL) else stream

        with patch.object(Path, 'open', substituted_open), self.assertRaises(ValueError):
            self.host.to_guest(ORIGINAL)
        after = ORIGINAL.stat()
        self.assertEqual((before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink),
                         (after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink))
        self.assertEqual(hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(), ORIGINAL_SHA256)

    def test_owned_links_do_not_become_plain_files_or_canonical_routes(self):
        # Break caught: following a symlink/junction or admitting a multiple-link file.
        h = self.host
        with fixture_directory() as directory:
            target = directory/'target.json'
            target.write_bytes(b'{}\n')
            self.assertEqual(h._read_plain(target, 32), b'{}\n')
            hardlink = directory/'hardlink.json'
            os.link(target, hardlink)
            with self.assertRaises(ValueError):
                h.to_guest(hardlink)
            with self.assertRaises(ValueError):
                h._read_plain(hardlink, 32)
            hardlink.unlink()

            target_directory = directory/'plain-directory'
            target_directory.mkdir()
            (target_directory/'target.json').write_bytes(b'{}\n')

            for link_directory in (False, True):
                with self.subTest(link='directory symlink' if link_directory else 'file symlink'):
                    destination = directory/'symbolic-directory' if link_directory else directory/'symbolic.json'
                    try:
                        os.symlink(target_directory if link_directory else target, destination,
                                   target_is_directory=link_directory)
                    except OSError as exc:
                        if getattr(exc, 'winerror', None) in (5, 1314):
                            self.skipTest('Windows symbolic-link privilege unavailable; reparse metadata tested separately')
                        raise
                    linked_file = destination/'target.json' if link_directory else destination
                    with self.assertRaises(ValueError):
                        h.to_guest(linked_file)
                    with self.assertRaises(ValueError):
                        h._read_plain(linked_file, 32)
                    destination.unlink() if not link_directory else destination.rmdir()

            with self.subTest(link='directory junction'):
                import _winapi
                create_junction = getattr(_winapi, 'CreateJunction', None)
                if create_junction is None:
                    self.skipTest('Python does not expose owned junction creation; reparse metadata tested separately')
                junction = directory/'junction'
                create_junction(str(target_directory), str(junction))
                try:
                    with self.assertRaises(ValueError):
                        h.to_guest(junction/'target.json')
                    with self.assertRaises(ValueError):
                        h._read_plain(junction/'target.json', 32)
                finally:
                    junction.rmdir()

    def test_plain_reader_rejects_reparse_metadata_size_and_changed_open_identity(self):
        # Break caught: missing lstat bounds/reparse checks or trusting a different open file.
        h = self.host
        with fixture_directory() as directory:
            target = directory/'plain.json'
            target.write_bytes(b'{}\n')
            actual = target.lstat()
            real_lstat = Path.lstat
            for changes in ({'st_file_attributes': 1024}, {'st_nlink': 2},
                            {'st_mode': stat.S_IFDIR | 0o700}, {'st_size': 33}):
                with self.subTest(metadata=changes):
                    def injected_lstat(path, *args, **kwargs):
                        if str(path) == str(target):
                            return changed_stat(actual, **changes)
                        return real_lstat(path, *args, **kwargs)
                    with patch.object(Path, 'lstat', injected_lstat), self.assertRaises(ValueError):
                        h._read_plain(target, 32)
            with self.assertRaises(ValueError):
                h._read_plain(target, 2)
            real_fstat = os.fstat
            for field in ('st_ino', 'st_size', 'st_mtime_ns', 'st_nlink'):
                with self.subTest(opened_identity=field):
                    def different_opened_identity(fd):
                        info = real_fstat(fd)
                        return changed_stat(info, **{field: getattr(info, field)+1})
                    with patch.object(h.os, 'fstat', different_opened_identity), self.assertRaises(ValueError):
                        h._read_plain(target, 32)
            calls = 0
            def changed_after_read(fd):
                nonlocal calls
                calls += 1
                info = real_fstat(fd)
                return info if calls == 1 else changed_stat(info, st_mtime_ns=info.st_mtime_ns+1)
            with patch.object(h.os, 'fstat', changed_after_read), self.assertRaises(ValueError):
                h._read_plain(target, 32)


if __name__ == '__main__':
    unittest.main(verbosity=2)
