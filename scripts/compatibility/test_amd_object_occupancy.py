"""CPU-only coverage of the image-only continuation and actual reused contracts."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import amd_object_occupancy as subject
import amd_dinov3_conditioning as dino
import amd_sparse_occupancy_data as data
import amd_sparse_latent_checkpoint as latent
import amd_sparse_sampling_contract as policy


class ContinuationTests(unittest.TestCase):
    def test_actual_image_and_cpu_preview(self):
        image, row = subject.selected_image()
        self.assertEqual(image.mode, 'RGBA')
        self.assertEqual(image.size, (1042, 1042))
        self.assertEqual(image.getchannel('A').getextrema(), (0, 255))
        processed = subject.preprocess_cpu(image)
        self.assertEqual(processed.mode, 'RGB')
        self.assertEqual(subject.digest_bytes(processed.tobytes()), row['crop_pixel_sha256'])

    def test_tampered_image_fails(self):
        with patch.object(subject, 'read_source', return_value=b'foreign image'):
            with self.assertRaises(ValueError):subject.selected_image()

    def test_component_spans_not_global_bbox(self):
        mask = np.zeros((64, 64, 64), dtype=bool)
        mask[0:8, 0:8, 0:8] = True
        self.assertEqual(subject.diagnostic_mask(mask)['classification'], 'PASS')
        mask[:] = False
        mask[0, 0, 0] = mask[63, 63, 63] = True
        self.assertEqual(subject.diagnostic_mask(mask)['classification'], 'FAIL')
        mask[:] = True
        self.assertEqual(subject.diagnostic_mask(mask)['classification'], 'FAIL')
        mask[:] = False
        mask[:, 20:44, 31:33] = True
        self.assertEqual(subject.diagnostic_mask(mask)['classification'], 'FAIL')

    def test_pipeline_stops_at_first_stage_failure(self):
        seen = []
        def run(stage):
            seen.append(stage)
            return stage != 'sampling'
        self.assertFalse(subject.run_stages(run))
        self.assertEqual(seen, ['dino', 'sampling'])

    def test_stage_order_exactly_once(self):
        seen = []
        self.assertTrue(subject.run_stages(lambda stage: seen.append(stage) or True))
        self.assertEqual(seen, ['dino', 'sampling', 'decoder'])

    def test_dino_explicit_save_preserves_default(self):
        self.assertFalse(inspect.signature(dino.worker).parameters['save_features'].default)
        self.assertTrue(subject.should_save_features(True, True))
        self.assertFalse(subject.should_save_features(True, False))
        self.assertTrue(subject.should_save_features(False, False))
        with self.assertRaises(ValueError):subject.should_save_features(True, 'yes')

    def test_actual_profiles_keep_jobs_deadlines_and_authorization(self):
        from contextlib import ExitStack
        row={'verified':True,'path':str(subject.EVIDENCE/'dino/run/scratch/conditioning.safetensors'),
             'file_sha256':'0'*64}
        with ExitStack() as stack:
            for owner,key in [(subject.dino.local,'fixture'),(subject.sampling.forward.local,'FEATURE'),
                (subject.sampling.forward.local,'FEATURE_SHA'),(subject.sampling.forward.local,'verify_feature'),
                (subject.sampling.forward.local,'EVIDENCE'),(subject.sampling.forward,'verify_preflight')]:
                stack.enter_context(patch.object(owner,key,getattr(owner,key)))
            stack.enter_context(patch.object(subject,'feature_reference',return_value=row))
            for stage in ('dino','sampling'):
                p=subject.prepare_profile(stage)
                self.assertEqual((p.cpu_seconds,p.gpu_seconds,p.overall_seconds),subject.LIMITS[stage])
                self.assertEqual(p.evidence,subject.EVIDENCE/stage)
                self.assertEqual(p.worker_script,Path(subject.__file__).resolve())
                self.assertIn('scripts/compatibility/amd_feature_checkpoint.py',p.committed_files)
                self.assertIn('scripts/compatibility/amd_object_occupancy.py',p.committed_files)
            self.assertEqual(p.maximum_forwards,22)
            self.assertEqual(p.maximum_steps,12)
            self.assertEqual(p.maximum_sequences,1)
            self.assertEqual(p.authorize,subject.sampling.AUTHORIZE)

    def test_new_latent_reader_uses_real_container_and_rejects_tampering(self):
        provenance = {key:'0'*64 for key in latent.FIELDS}
        provenance.update(component='SparseStructureFlowModel',seed=str(policy.SEED),sigma_min=str(policy.SIGMA_MIN),
            params=json.dumps(policy.PARAMS),sampler_sources='{}',feature_provenance='{}',runtime='CPU test',harness_commit='0'*40)
        with tempfile.TemporaryDirectory(dir=subject.EVIDENCE) as directory:
            path=Path(directory)/'sparse-structure-latent.safetensors'
            array=np.zeros(latent.SHAPE,dtype='<f4')
            expected=latent.write(path,Path(directory),array,provenance)
            restored,row=data.read_input(path,expected=expected)
            self.assertTrue(np.array_equal(restored,array))
            self.assertEqual(row['file_sha256'],expected['file_sha256'])
            blob=bytearray(path.read_bytes());blob[-1]^=1;path.write_bytes(blob)
            with self.assertRaises(ValueError):data.read_input(path,expected=expected)

    def test_deadlines_and_sampler_settings_unchanged(self):
        self.assertEqual(subject.LIMITS, {'dino':(120,180,360),'sampling':(180,240,480),'decoder':(120,180,360)})
        self.assertEqual(policy.SEED,20261013)
        self.assertEqual(policy.PARAMS['steps'],12)
        self.assertEqual(policy.MEMORY_LIMIT,12*1024**3)


if __name__=='__main__':unittest.main()
