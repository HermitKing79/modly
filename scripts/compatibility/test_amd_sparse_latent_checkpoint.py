"""Synthetic CPU latent artifact regression; never loads an AI checkpoint."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
try:
    import amd_sparse_latent_checkpoint as artifact
except ModuleNotFoundError:artifact=None

class LatentCheckpointTests(unittest.TestCase):
    def provenance(self):
        return dict(component='SparseStructureFlowModel',model_checkpoint_sha256='1'*64,
            model_config_sha256='2'*64,model_source_sha256='3'*64,sampler_sources='{}',
            params='{"dino_lock":0.0,"guidance_interval":[0.6,1.0],"guidance_rescale":0.7,"guidance_strength":7.5,"rescale_t":5.0,"steps":12}',
            seed='20261013',sigma_min='1e-05',noise_sha256='4'*64,feature_file_sha256='5'*64,
            feature_provenance='{"input_png_sha256":"'+ '6'*64+'"}',harness_commit='7'*40,runtime='synthetic CPU fixture')
    def test_exclusive_finite_round_trip_payload_and_corruption(self):
        self.assertIsNotNone(artifact,'bounded latent writer is not implemented')
        array=np.linspace(-1,1,32768,dtype=np.float32).reshape(1,8,16,16,16)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'sparse-structure-latent.safetensors';p=self.provenance()
            row=artifact.write(path,root,array,p)
            self.assertTrue(row['verified']);self.assertLessEqual(row['file_bytes'],262144)
            self.assertEqual(row['shape'],[1,8,16,16,16]);self.assertEqual(row['dtype'],'F32')
            self.assertEqual(row['provenance']['sigma_min'],'1e-05')
            with self.assertRaises(FileExistsError):artifact.write(path,root,array,p)
            blob=bytearray(path.read_bytes());blob[-1]^=1;path.write_bytes(blob)
            with self.assertRaises(ValueError):artifact.verify(path,root,p,row['output_sha256'])
    def test_wrong_shape_dtype_nonfinite_path_and_provenance_fail(self):
        self.assertIsNotNone(artifact,'bounded latent writer is not implemented')
        good=np.ones((1,8,16,16,16),dtype=np.float32)
        for array in (good.astype(np.float64),good.reshape(32768),good*np.float32(np.nan)):
            with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):
                artifact.write(Path(tmp)/'sparse-structure-latent.safetensors',Path(tmp),array,self.provenance())
        with tempfile.TemporaryDirectory() as tmp:
            for path,p in [(Path(tmp)/'foreign.safetensors',self.provenance()),
                           (Path(tmp)/'sparse-structure-latent.safetensors',{**self.provenance(),'seed':'1'}),
                           (Path(tmp)/'sparse-structure-latent.safetensors',{**self.provenance(),'sigma_min':'0.1'})]:
                with self.assertRaises(ValueError):artifact.write(path,Path(tmp),good,p)

if __name__=='__main__':unittest.main()
