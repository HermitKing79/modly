"""Bounded CPU-only qualification; synthetic fields are never model outputs."""
import dataclasses
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = Path('C:/Users/wattz/Documents/ModlyArtifacts/amd-brick-pbr-cpu-3942181-tests')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'experimental/extensions/trellis2-amd'))


def cpu_only(event, args):
    if event == 'import' and str(args[0]).split('.')[0] in {
            'torch', 'triton', 'trellis2', 'flex_gemm', 'o_voxel', 'cumesh'}:
        raise RuntimeError('CPU artifact qualification forbids AI/native imports')
    if event.startswith('socket.'):
        raise RuntimeError('CPU artifact qualification forbids network access')


sys.addaudithook(cpu_only)


class BrickPbrInputsArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile_module = importlib.import_module('inference_texture.profile')
        document = json.loads((ROOT.parent / 'artifacts/amd-brick-neural-completion-c9447dd/neural-output/run-contract.json').read_text())
        cls.profile = cls.profile_module.authenticate('brick-1957891-v1', document)
        cls.inputs = importlib.import_module('inference_texture.inputs')
        cls.prepared = cls.inputs.prepare(cls.profile)

    def fixture(self):
        temporary = tempfile.TemporaryDirectory(prefix='brick-profile-field-', dir=FIXTURE_ROOT)
        root = Path(temporary.name).resolve()
        self.assertEqual(root.parent, FIXTURE_ROOT.resolve(strict=True))
        self.addCleanup(temporary.cleanup)
        return root

    def test_authenticated_profile_is_immutable_and_exact(self):
        profile = importlib.import_module('inference_texture.profile')
        document = json.loads((ROOT.parent / 'artifacts/amd-brick-neural-completion-c9447dd/neural-output/run-contract.json').read_text())
        value = profile.authenticate('brick-1957891-v1', document)
        self.assertEqual(value.initial_rows, 4430)
        self.assertEqual(value.parent_rows, (4430, 21391, 97313, 438684))
        self.assertEqual(value.child_rows, (21391, 97313, 438684, 1957891))
        self.assertEqual(value.final_rows, 1957891)
        self.assertIs(profile.require_authenticated(value), value)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            value.final_rows = 1
        with self.assertRaises(ValueError):
            profile.require_authenticated('brick-1957891-v1')
        bad = dict(document, output_directory='/home/modly/foreign-run/output')
        with self.assertRaises(ValueError):
            profile.authenticate('brick-1957891-v1', bad)
        for forged in (dataclasses.replace(value, final_rows=1842968),
                       dataclasses.replace(value, initial_rows=1),
                       dataclasses.replace(value, parent_rows=(1, 2, 3, 4)),
                       dataclasses.replace(value, run_contract_sha256='0'*64)):
            with self.subTest(forged=forged):
                with self.assertRaises(ValueError):
                    profile.require_authenticated(forged)

    def test_genuine_inputs_and_all_eight_guide_calls(self):
        import numpy as np
        inputs = importlib.import_module('inference_texture.inputs')
        prepared = self.prepared
        self.assertEqual(prepared['denormalized'].shape, (4430, 32))
        self.assertFalse(prepared['denormalized'].flags.writeable)
        self.assertEqual(prepared['geometry_coords'].shape, (1957891, 4))
        validator = inputs.GuideOrderValidator(prepared, profile=self.profile)
        for stage, guide in enumerate(prepared['guides']):
            self.assertEqual(guide['logits_bf16'].dtype, np.dtype('<u2'))
            child = prepared['guides'][stage+1]['coords'] if stage < 3 else prepared['geometry_coords']
            for _ in range(2):
                validator.before_c2s(stage, guide['coords'], guide['logits_f32'] > 0, guide['coords'])
                validator.after_c2s(stage, child)
        self.assertEqual(validator.finalize(prepared['geometry_coords'])['calls'], [2, 2, 2, 2])
        self.assertNotIn('torch', sys.modules)

    def test_prepared_payloads_cannot_be_replaced_by_verified_metadata(self):
        import numpy as np
        for key in ('coords', 'denormalized', 'geometry_coords'):
            bad = dict(self.prepared)
            bad[key] = bad[key].copy()
            bad[key].flat[1] += 1
            bad[key].flags.writeable = False
            with self.subTest(payload=key):
                with self.assertRaises(ValueError):
                    self.inputs.validate_prepared(bad, profile=self.profile)
        bad = dict(self.prepared)
        guides = list(bad['guides'])
        guides[3] = dict(guides[3])
        bits = guides[3]['logits_bf16'].copy()
        bits.flat[0] ^= np.uint16(1)
        bits.flags.writeable = False
        guides[3]['logits_bf16'] = bits
        bad['guides'] = tuple(guides)
        with self.assertRaises(ValueError):
            self.inputs.validate_prepared(bad, profile=self.profile)

    def test_guide_validator_refuses_cache_order_mask_children_and_missing_calls(self):
        g = self.prepared['guides'][0]
        child = self.prepared['guides'][1]['coords']
        make = lambda: self.inputs.GuideOrderValidator(self.prepared, profile=self.profile)
        for stage, coords, mask, guide_coords, cached in (
                (0, g['coords'], g['logits_f32'] > 0, g['coords'], True),
                (1, g['coords'], g['logits_f32'] > 0, g['coords'], False),
                (0, g['coords'][::-1].copy(), g['logits_f32'] > 0, g['coords'], False),
                (0, g['coords'], ~(g['logits_f32'] > 0), g['coords'], False)):
            with self.subTest(stage=stage, cached=cached):
                with self.assertRaises(ValueError):
                    make().before_c2s(stage, coords, mask, guide_coords, cached=cached)
        validator = make()
        validator.before_c2s(0, g['coords'], g['logits_f32'] > 0, g['coords'])
        with self.assertRaises(ValueError):
            validator.after_c2s(0, child[::-1].copy())
        with self.assertRaises(ValueError):
            validator.finalize(self.prepared['geometry_coords'])

    def test_private_loader_preserves_foreign_generic_module_globals(self):
        from types import ModuleType
        from inference_texture._source import load_pinned
        foreign = ModuleType('run_contract')
        foreign.sentinel = object()
        previous = sys.modules.get('run_contract')
        sys.modules['run_contract'] = foreign
        try:
            relative = 'experimental/extensions/trellis2-amd/inference_decoder/artifact_io.py'
            module = load_pinned('test_isolated_geometry', relative,
                                 self.profile_module.PINS[relative],
                                 imports={'run_contract': self.profile_module.contracts})
            self.assertIs(sys.modules['run_contract'], foreign)
            self.assertIs(module.support_counts, self.profile_module.contracts.support_counts)
            with self.assertRaises(ValueError):
                load_pinned('test_wrong_pin', relative, '0'*64)
        finally:
            if previous is None:
                del sys.modules['run_contract']
            else:
                sys.modules['run_contract'] = previous

    def test_profile_writer_independent_reader_and_default_refusal(self):
        import numpy as np
        artifact = importlib.import_module('inference_texture.artifact')
        with tempfile.TemporaryDirectory(prefix='brick-profile-field-', dir=FIXTURE_ROOT) as directory:
            root = Path(directory).resolve()
            raw = np.array([[-3, -2, -1, 0, 1, 2]], dtype='<f4')
            coords = np.array([[0, 511, 1, 2]], dtype='<i4')
            features = raw * np.float32(.5) + np.float32(.5)
            provenance = dict(mode='fixture', synthetic=True, purpose='SYNTHETIC CPU profile persistence')
            result = artifact.write(root, raw, features, coords, provenance, profile=self.profile)
            self.assertTrue(result['verified'])
            self.assertTrue(artifact.verify(result['artifact_path'], root, profile=self.profile)['verified'])
            with self.assertRaises(ValueError):
                artifact.verify(result['artifact_path'], root)
            with self.assertRaises(ValueError):
                artifact.verify(result['artifact_path'], root, profile='brick-1957891-v1')

    def test_historical_default_fixture_remains_unchanged(self):
        import numpy as np
        from inference_texture import artifact
        root = self.fixture()
        raw = np.zeros((2, 6), dtype='<f4')
        coordinates = np.array([[0, 511, 1, 2], [0, 0, 1, 2]], dtype='<i4')
        result = artifact.write(root, raw, raw + np.float32(.5), coordinates, {'mode': 'fixture'})
        self.assertTrue(artifact.verify(result['artifact_path'], root)['verified'])
        self.assertEqual(artifact._historical.MAX_ROWS, 1842968)
        with self.assertRaises(ValueError):
            artifact.verify(result['artifact_path'], root, profile=self.profile)

    def test_qualified_public_ownership_class_is_shared_without_rebinding(self):
        import amd_texture_field_artifact as public
        from inference_texture import artifact
        original_globals = dict(vars(public))
        self.assertIs(artifact._historical, public)
        self.assertIs(artifact.drvfs_ownership, public.drvfs_ownership)
        for kind in ('writer', 'reader'):
            module = artifact._field(kind, self.profile)
            self.assertIs(module._DrvfsOwnership, public._DrvfsOwnership)
            self.assertEqual(module._owner.__code__, public._owner.__code__)
            self.assertEqual(module._plain.__code__, public._plain.__code__)
            self.assertIs(module._DrvfsOwnership.check, public._DrvfsOwnership.check)
        self.assertEqual(dict(vars(public)), original_globals)
        self.assertEqual(public.MAX_ROWS, 1842968)

    def test_foreign_public_module_and_duck_typed_ownership_are_refused(self):
        from types import ModuleType
        from inference_texture import artifact
        genuine = sys.modules.get('amd_texture_field_artifact')
        foreign = ModuleType('amd_texture_field_artifact')
        foreign.__file__ = str(ROOT / 'scripts/compatibility/amd_texture_field_artifact.py')
        sys.modules['amd_texture_field_artifact'] = foreign
        try:
            with self.assertRaises(ValueError):
                artifact._canonical_field()
            self.assertIs(sys.modules['amd_texture_field_artifact'], foreign)
        finally:
            if genuine is None:
                del sys.modules['amd_texture_field_artifact']
            else:
                sys.modules['amd_texture_field_artifact'] = genuine
        original_class = genuine._DrvfsOwnership
        changed_class = type('_DrvfsOwnership', (), {
            '__module__': genuine.__name__, 'check': lambda self, path, info: None})
        genuine._DrvfsOwnership = changed_class
        try:
            with self.assertRaisesRegex(ValueError, 'ownership class implementation'):
                artifact._canonical_field()
        finally:
            genuine._DrvfsOwnership = original_class
        class ForeignOwnership:
            def check(self, path, info):
                raise AssertionError('Foreign ownership code must never execute')
        for module in (artifact._historical, artifact._field('writer', self.profile), artifact._field('reader', self.profile)):
            with self.assertRaisesRegex(ValueError, 'Foreign artifact ownership capability'):
                module._owner(None, None, ForeignOwnership())

    def test_writer_refuses_bad_metadata_nonfinite_transform_and_authority(self):
        import numpy as np
        from inference_texture import artifact
        root = self.fixture()
        raw = np.zeros((2, 6), dtype='<f4')
        features = raw + np.float32(.5)
        coords = np.array([[0, 1, 2, 3], [0, 3, 2, 1]], dtype='<i4')
        fixture = dict(mode='fixture', synthetic=True, purpose='SYNTHETIC refusal checks')
        for changed in (dict(fixture, synthetic=False), dict(fixture, checkpoint_sha256='0'*64),
                        dict(fixture, guidance_parent_rows=[1, 2, 3, 4])):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    artifact.write(root, raw, features, coords, changed, profile=self.profile)
        for bad_raw, bad_features in ((raw + np.float32('nan'), features), (raw, features + 1)):
            with self.assertRaises(ValueError):
                artifact.write(root, bad_raw, bad_features, coords, fixture, profile=self.profile)
        with self.assertRaises(ValueError):
            artifact.write(root, raw, features, coords, fixture, profile=self.profile, authority=object())
        with self.assertRaises(ValueError):
            artifact.write(root, raw, features, coords, {'mode': 'production'}, profile=self.profile)
        self.assertEqual(list(root.iterdir()), [])

    def test_exclusive_hardlink_corruption_and_foreign_binding_refusal(self):
        import numpy as np
        from inference_texture import artifact
        raw = np.zeros((1, 6), dtype='<f4')
        features = raw + np.float32(.5)
        coords = np.array([[0, 1, 2, 3]], dtype='<i4')
        fixture = dict(mode='fixture', synthetic=True, purpose='SYNTHETIC filesystem/integrity refusal')
        root = self.fixture()
        result = artifact.write(root, raw, features, coords, fixture, profile=self.profile)
        with self.assertRaises(FileExistsError):
            artifact.write(root, raw, features, coords, fixture, profile=self.profile)
        path = Path(result['artifact_path'])
        hardlink = root / 'SYNTHETIC-hardlink'
        os.link(path, hardlink)
        with self.assertRaises(ValueError):
            artifact.verify(path, root, profile=self.profile)
        hardlink.unlink()
        with path.open('r+b') as stream:
            stream.seek(-1, 2)
            value = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([value[0] ^ 1]))
        with self.assertRaises(ValueError):
            artifact.verify(path, root, profile=self.profile)
        foreign_root = self.fixture()
        foreign = artifact._bind(fixture, self.profile)
        foreign['pbr_profile_document'] = '{}'
        foreign['pbr_profile_sha256'] = hashlib.sha256(b'{}').hexdigest()
        result = artifact._historical.write(foreign_root, raw, features, coords, foreign)
        with self.assertRaises(ValueError):
            artifact.verify(result['artifact_path'], foreign_root, profile=self.profile)

    def test_independent_parent_reader_after_writer_process_exit(self):
        from inference_texture import artifact
        root = self.fixture()
        code = r'''
import sys,json,numpy as np
from pathlib import Path
def guard(event,args):
    if event=='import' and str(args[0]).split('.')[0] in {'torch','triton','trellis2','flex_gemm','o_voxel','cumesh'}:
        raise RuntimeError('CPU-only writer forbids AI/native imports')
    if event.startswith('socket.'): raise RuntimeError('Network forbidden')
sys.addaudithook(guard)
sys.path.insert(0,sys.argv[1])
from inference_texture import profile,artifact
p=profile.authenticate(profile.NAME,json.loads(Path(sys.argv[2]).read_text()))
raw=np.zeros((1,6),dtype='<f4'); coordinates=np.array([[0,1,2,3]],dtype='<i4')
r=artifact.write(sys.argv[3],raw,raw+np.float32(.5),coordinates,
    dict(mode='fixture',synthetic=True,purpose='SYNTHETIC child-writer CPU artifact'),profile=p)
assert r['verified'] and 'torch' not in sys.modules
print(r['metadata']['artifact_sha256'])
'''
        run = subprocess.run([sys.executable, '-B', '-I', '-c', code,
            str(ROOT / 'experimental/extensions/trellis2-amd'),
            str(ROOT.parent / 'artifacts/amd-brick-neural-completion-c9447dd/neural-output/run-contract.json'),
            str(root)], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        checked = artifact.verify(root / artifact.ARTIFACT_NAME, root, profile=self.profile)
        self.assertEqual(run.stdout.strip(), checked['metadata']['artifact_sha256'])

    def test_full_support_synthetic_roundtrip_with_clean_independent_reader(self):
        import numpy as np
        from inference_texture import artifact
        root = self.fixture()
        raw = np.zeros((self.profile.final_rows, 6), dtype='<f4')
        # Explicitly synthetic numeric payload, preserving genuine row identity.
        raw[:, 0] = np.float32(-3)
        raw[:, 5] = np.float32(2)
        features = raw * np.float32(.5) + np.float32(.5)
        result = artifact.write(root, raw, features, self.prepared['geometry_coords'],
            dict(mode='fixture', synthetic=True, purpose='SYNTHETIC full-support CPU persistence; no model forward'),
            profile=self.profile)
        self.assertEqual(result['metadata']['payload_bytes'], 125305024)
        self.assertEqual(result['metadata']['tensors']['raw']['shape'], [1957891, 6])
        self.assertEqual(result['metadata']['tensors']['coordinates']['sha256'],
                         self.profile_module.FINAL_COORDINATES_SHA256)
        # A fresh independent consumer receives no in-memory Profile or writer
        # state; it authenticates the preserved geometry contract again.
        code = r'''
import sys,json
from pathlib import Path
def guard(event,args):
    if event=='import' and str(args[0]).split('.')[0] in {'torch','triton','trellis2','flex_gemm','o_voxel','cumesh'}:
        raise RuntimeError('Independent CPU reader forbids AI/native imports')
    if event.startswith('socket.'): raise RuntimeError('Network forbidden')
sys.addaudithook(guard)
sys.path.insert(0,sys.argv[1])
from inference_texture import profile,artifact
p=profile.authenticate(profile.NAME,json.loads(Path(sys.argv[2]).read_text()))
r=artifact.verify(sys.argv[3],profile=p)
assert r['verified'] and r['metadata']['payload_bytes']==125305024 and 'torch' not in sys.modules
print(r['metadata']['artifact_sha256'])
'''
        run = subprocess.run([sys.executable, '-B', '-I', '-c', code,
            str(ROOT / 'experimental/extensions/trellis2-amd'),
            str(ROOT.parent / 'artifacts/amd-brick-neural-completion-c9447dd/neural-output/run-contract.json'),
            result['artifact_path']], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.strip(), result['metadata']['artifact_sha256'])
        with self.assertRaises(ValueError):
            artifact.verify(result['artifact_path'], root)
        print(json.dumps(dict(kind='SYNTHETIC_FULL_SUPPORT_CPU_ONLY', payload_bytes=125305024,
                              artifact_bytes=result['metadata']['artifact_bytes'],
                              artifact_sha256=result['metadata']['artifact_sha256'],
                              independent_clean_process_reader=True, model_loaded=False, gpu_executed=False)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
