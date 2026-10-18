"""Focused stdlib tests of the experiment's exact read-only authority."""
import copy
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


class ReadOnlyInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = HERE / 'readonly_input.py'
        cls.policy = None
        if path.exists():
            spec = importlib.util.spec_from_file_location('test_exact_input_policy', path)
            cls.policy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.policy)

    def test_exact_windows_attestation_and_negative_mutations(self):
        self.assertIsNotNone(self.policy, 'The exact read-only input boundary is missing')
        p = self.policy
        files = []
        for label, (path, sha, size) in p.INPUTS.items():
            windows = 'C:\\' + str(path)[7:].replace('/', '\\')
            chain = []
            from pathlib import PureWindowsPath
            for i, ancestor in enumerate((PureWindowsPath(windows), *PureWindowsPath(windows).parents)):
                chain.append(dict(path=str(ancestor), owner_sid=p.SID, reparse=False,
                    link_type=None, target=[], attributes=32 if i == 0 else 16, sddl='verified',
                    rules=[dict(type='Allow', sid=p.SID, rights=2032127, propagation='None')]))
            fid = dict(dev=1, ino=len(files)+1, mtime_ns=1, nlink=1)
            files.append(dict(label=label, guest_path=str(path), windows_path=windows,
                sha256=sha, bytes=size, owner_sid=p.SID, current_sid=p.SID,
                windows_identity=fid, ancestors=chain, read_identity=dict(stable=True,
                    **{k: dict(fid) for k in ('before','opened','after_open','after')})))
        document = dict(schema=p.SCHEMA, current_sid=p.SID,
            trusted_input_roots=[str(p.TEXTURE_ROOT), str(p.GEOMETRY_ROOT)], files=files,
            observer=dict(sha256=p.OWNER_SHA))
        self.assertEqual(set(p.validate_attestation(document)), set(p.INPUTS))
        changes = [lambda d: d.update(current_sid='foreign'),
            lambda d: d['files'][0].update(owner_sid='foreign'),
            lambda d: d['files'][0].update(guest_path='/tmp/texture-latent.safetensors'),
            lambda d: d['files'][0].update(sha256='0'*64),
            lambda d: d['files'][0]['windows_identity'].update(nlink=2),
            lambda d: d['files'][0]['read_identity']['opened'].update(ino=999),
            lambda d: d['files'][0]['ancestors'][0].update(reparse=True),
            lambda d: d['files'][0]['ancestors'].pop(),
            lambda d: d['files'][0]['ancestors'][0]['rules'].append(
                dict(type='Allow', sid='foreign', rights=2, propagation='None'))]
        for change in changes:
            bad = copy.deepcopy(document)
            change(bad)
            with self.subTest(change=change), self.assertRaises(ValueError):
                p.validate_attestation(bad)


if __name__ == '__main__':
    unittest.main()
