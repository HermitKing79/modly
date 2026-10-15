"""CPU-only fresh texture adapter contracts; synthetic fixtures are never inference."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
TEST_ROOT=Path(os.environ.get('MODLY_TEXTURE_TEST_ROOT',tempfile.gettempdir()))/'amd-generation-texture-cpu-tests'


class TextureTests(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.root.resolve().relative_to(TEST_ROOT.resolve());self.temp.cleanup()

    def adapter(self):
        path=Path(__file__).with_name('amd_generation_texture.py')
        self.assertTrue(path.exists(),'fresh texture adapter is missing')
        return importlib.import_module('amd_generation_texture')

    def test_explicit4430_artifact_roundtrip_and_historical_default_rejection(self):
        import amd_texture_sampling_artifact as artifact
        n=4430;raw=np.arange(n*32,dtype=np.float32).reshape(n,32)/np.float32(4096)
        xyz=np.column_stack(np.unravel_index(np.arange(n),(32,32,32))).astype(np.int32)
        coords=np.column_stack((np.zeros(n,np.int32),xyz)).astype(np.int32)
        result=artifact.write(self.root,raw,raw.copy(),coords,{'synthetic':True},noise=-raw,
                              token_count=n,maximum_tokens=n)
        self.assertTrue(artifact.verify(Path(result['artifact_path']),self.root,result['metadata'],
                                       token_count=n,maximum_tokens=n)['verified'])
        self.assertEqual(result['metadata']['tensors']['raw']['shape'],[4430,32])
        self.assertEqual(result['metadata']['tensors']['coords']['tensor_bytes'],70880)
        with self.assertRaises(ValueError):artifact.verify(Path(result['artifact_path']),self.root)
        with self.assertRaises(ValueError):artifact.layouts(4431,maximum_tokens=4430)

    def test_original_normalizations_accept_explicit4430_without_math_change(self):
        import amd_texture_sampling_contract as policy
        source=np.arange(4430*32,dtype=np.float32).reshape(4430,32)/np.float32(4096)
        mean=np.array(policy.SHAPE_NORMALIZATION['mean'],np.float32)
        std=np.array(policy.SHAPE_NORMALIZATION['std'],np.float32)
        normalized=policy.normalize_shape(source,maximum_tokens=4430)
        np.testing.assert_array_equal(normalized,(source-mean[None])/std[None])
        with self.assertRaises(ValueError):policy.normalize_shape(source)
        with self.assertRaises(ValueError):policy.denormalize_texture(np.zeros((4431,32),np.float32),maximum_tokens=4430)

    def test_texture_control_bounds_default_and_noise_first_columns(self):
        import torch
        import amd_texture_sampling_control as control
        from test_amd_shape_sampling_runtime import SparseTensor
        c=control.TextureSamplingControl(torch,None,device='cpu',token_count=2,maximum_tokens=4430)
        self.assertEqual(control.TextureSamplingControl(object(),object()).token_count,3743)
        with self.assertRaises(ValueError):control.TextureSamplingControl(torch,None,token_count=4431,maximum_tokens=4430)
        coords=torch.tensor([[0,0,0,0],[0,0,0,1]],dtype=torch.int32)
        state=torch.ones(2,32);shape=torch.full((2,32),2.)
        c.coords=coords;c.step_x=SparseTensor(state,coords);c.shape_cond=SparseTensor(shape,coords)
        c.budget.begin_step(*c.policy.times()[:2]);c.budget.begin_forward(c.budget.next_ticket())
        with self.assertRaises(RuntimeError):c.validate_concat(SparseTensor(torch.cat((shape,state),dim=-1),coords))
        c.validate_concat(SparseTensor(torch.cat((state,shape),dim=-1),coords))
        self.assertEqual(c.concat_calls,1);self.assertFalse(torch.cuda.is_initialized())

    def test_classifier_fresh_feature_and4430_exact_layouts(self):
        import amd_texture_sampling_contract as policy
        from test_amd_texture_sampling_result import candidate
        row=candidate();n=4430;sha='d'*64
        row['output'].update(shape=[n,32],count=n*32)
        metadata=row['artifacts']['metadata'];metadata['artifact_bytes']=n*400+4096
        for name,tensor in metadata['tensors'].items():
            width=4 if name=='coords' else 32;tensor.update(shape=[n,width],tensor_bytes=n*width*4)
        metadata['provenance']['checkpoint']['conditioning']['file_sha256']=sha
        self.assertEqual(policy.classify(row,True,True,True,token_count=n,maximum_tokens=n,feature_sha256=sha),'PASS')
        self.assertEqual(policy.classify(row,True,True,True),'PARTIAL')
        self.assertEqual(policy.classify(row,True,True,True,token_count=n,maximum_tokens=n,feature_sha256='e'*64),'PARTIAL')

    def test_context_and_artifact_root_are_explicit_and_outside_git(self):
        adapter=self.adapter();base=adapter.APPROVED_ROOT/'new-image/occupancy'
        value=adapter.context(base/'shape',base/'texture',self.root,'a'*40)
        self.assertEqual(value['artifact_root'],str(self.root))
        with self.assertRaises(ValueError):adapter.context(base/'shape',base/'texture',ROOT/'scripts/compatibility','a'*40)
        with self.assertRaises(ValueError):adapter.validate_count(4431)

    def test_actual_fresh_shape_lineage_and_coordinates_are_verified(self):
        adapter=self.adapter()
        evidence=adapter.APPROVED_ROOT/'new-image/occupancy/shape'
        if not (evidence/'run/results.json').exists():self.skipTest('fresh real shape receipt is unavailable')
        bundle=adapter.input_bundle(evidence)
        self.assertEqual(bundle['coords'].shape,(4430,4))
        self.assertEqual(bundle['lineage']['token_count'],4430)
        self.assertEqual(bundle['feature']['file_sha256'],bundle['lineage']['feature_file_sha256'])
        self.assertEqual(bundle['lineage']['shape_coordinates_payload_sha256'],hashlib.sha256(bundle['coords'].tobytes()).hexdigest())

    def test_worker_constructor_is_lazy_and_runtime_only_binds_factories(self):
        adapter=self.adapter();base=adapter.APPROVED_ROOT/'new-image/occupancy'
        runtime=adapter.Runtime(adapter.context(base/'shape',base/'texture',self.root,'a'*40))
        self.assertIsNone(runtime._bundle)
        self.assertEqual(runtime.artifact_root,self.root)
        self.assertEqual(runtime.local.PREFIX.name,'slat_flow_imgshape2tex_dit_1_3B_512_bf16')

    def test_worker_context_rejects_duplicate_keys_foreign_output_and_missing_commit(self):
        adapter=self.adapter();base=adapter.APPROVED_ROOT/'new-image/occupancy'
        value=adapter.context(base/'shape',base/'texture',self.root,'a'*40)
        self.assertEqual(adapter.worker_context(json.dumps(value),self.root),value)
        with self.assertRaises(ValueError):adapter.worker_context(json.dumps(value),self.root/'foreign')
        with self.assertRaises(ValueError):adapter.worker_context(json.dumps({**value,'harness_commit':None}),self.root)
        duplicate=json.dumps(value)[:-1]+',"harness_commit":"'+'b'*40+'"}'
        with self.assertRaises(ValueError):adapter.worker_context(duplicate,self.root)

    def test_preflight_identity_allows_owned_post_run_output(self):
        adapter=self.adapter();base=adapter.APPROVED_ROOT/'new-image/occupancy'
        runtime=adapter.Runtime(adapter.context(base/'shape',base/'texture',self.root,'a'*40))
        (self.root/'latent').mkdir();(self.root/'latent/owned-output.txt').write_text('synthetic')
        runtime._bundle={'coords':np.zeros((2,4),np.int32),'lineage':{'synthetic':True}}
        probe={'artifact_path':str(self.root/'probe'),'metadata':{'synthetic':True},'verified':True,
               'synthetic':True,'torch_imported':False}
        prepared={'schema':adapter.SCHEMA,'lineage':runtime.bundle['lineage'],
                  'artifact_root':str(self.root),'shape_evidence':str(runtime.shape_evidence)}
        with patch.object(adapter.immutable,'verify_sources'),patch.object(adapter.original.teardown,'verify_workspace'), \
             patch.object(adapter.original.teardown,'verify_binding_evidence'),patch.object(adapter.immutable,'verify_sampler',return_value={}), \
             patch.object(adapter.shape,'_json',side_effect=[{},prepared,probe]), \
             patch.object(runtime,'verify_artifact',return_value={k:v for k,v in probe.items() if k not in ('synthetic','torch_imported')}), \
             patch.object(runtime,'identity',return_value={'synthetic':True}):
            self.assertEqual(runtime.verify_preflight(),{'synthetic':True})

    def test_profile_closes_source_dependencies_and_preserves_all_live_budgets(self):
        adapter=self.adapter();base=adapter.APPROVED_ROOT/'new-image/occupancy'
        runtime=adapter.Runtime(adapter.context(base/'shape',base/'texture',self.root,'a'*40))
        profile=runtime.profile()
        self.assertEqual(profile.worker_script,Path(adapter.__file__).resolve())
        self.assertEqual((profile.cpu_seconds,profile.gpu_seconds,profile.overall_seconds),(180,240,480))
        self.assertEqual((profile.maximum_sequences,profile.maximum_steps,profile.maximum_forwards),(1,12,12))
        for name in ('amd_generation_occupancy','amd_object_occupancy','amd_sparse_occupancy',
                     'amd_sparse_occupancy_data','amd_sparse_occupancy_contract','amd_sparse_sampling',
                     'amd_sparse_sampling_contract','amd_generation_shape','amd_generation_texture'):
            self.assertIn('scripts/compatibility/'+name+'.py',profile.committed_files)
        self.assertFalse(any('/new-image/' in name and '/run/' in name for name in profile.committed_files))
        self.assertIsNone(runtime._bundle)

    def test_dynamic_count_runs_real_sampler_with_original_worker_hooks(self):
        import gc
        import torch
        import test_amd_texture_sampling_runtime as existing
        import test_amd_shape_sampling_runtime as shape_fixture
        from amd_shape_sampling_control import PHASE_KEY
        from amd_texture_sampling_control import TextureSamplingControl
        from trellis2.modules.sparse import SparseTensor,sparse_cat
        from trellis2.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler

        class Rope(shape_fixture.Rope):
            def forward(self,x):
                if x.get_spatial_cache(PHASE_KEY) is None:
                    torch.ops.aten.to.dtype_layout(self.freqs,device=torch.device('cuda:0'))
                    angles=torch.arange(128,dtype=torch.float32).reshape(2,64)*.001
                    x.register_spatial_cache(PHASE_KEY,torch.polar(torch.ones_like(angles),angles))
                return x
        class Projection(torch.nn.Module):
            def forward(self,x):return x.replace(x.feats[:,:32])
        class Model(shape_fixture.Fixture):
            def __init__(self):super().__init__();self.input_layer=Projection()
            def forward(self,x,t,cond,concat_cond=None):
                return super().forward(self.input_layer(sparse_cat([x,concat_cond],dim=-1)),t,cond)

        model=Model()
        for block in model.blocks:block.self_attn.rope=Rope()
        control=TextureSamplingControl(torch,FlowEulerGuidanceIntervalSampler,device='cpu',token_count=2)
        coords=torch.tensor([[0,0,0,0],[0,1,2,3]],dtype=torch.int32)
        noise=torch.randn(2,32,generator=torch.Generator().manual_seed(control.policy.SEED))
        sparse=SparseTensor(noise,coords);sparse.register_spatial_cache('layout',[slice(0,2)])
        sparse.register_spatial_cache('seqlen',torch.tensor([2],dtype=torch.int64))
        inputs=dict(cond=torch.ones(1,1029,1024)*.25,shape_cond=sparse.replace(torch.ones(2,32)*.125))
        trace=existing.diagnostics.Trace(lambda line:None,limit=256)
        hooks,memory=existing.actual_hooks(model,control,trace)
        hooks['policy']=control.copy_policy(2);hooks['policy'].phase='forward'
        hooks['policy'].rope=model.blocks[0].self_attn.rope.freqs
        with torch.inference_mode(),existing.CPUSink(),hooks['BoundedDispatch'](), \
             patch.object(torch.nn.Module,'__call__',hooks['module_call']),patch.object(torch,'tensor',hooks['timestep_tensor']):
            result=control.run(model,sparse,inputs,trace,lambda:None,memory)
        self.assertEqual(control.budget.calls,12);self.assertEqual(control.concat_calls,12)
        self.assertEqual([r['role'] for r in control.models],['positive']*12)
        self.assertEqual(list(result.samples.feats.shape),[2,32]);self.assertIs(result.samples.coords,coords)
        self.assertFalse(torch.cuda.is_initialized())
        result.samples.clear_spatial_cache();control.drop_references()
        del result,model,sparse,inputs
        gc.collect()

    def test_clean_cpu_prepare_uses_real_writer_and_no_torch(self):
        import subprocess
        import sys
        adapter=self.adapter();source=adapter.APPROVED_ROOT/'new-image/occupancy/shape'
        if not (source/'run/results.json').exists():self.skipTest('fresh real shape receipt is unavailable')
        with tempfile.TemporaryDirectory(prefix='_texture-cpu-',dir=adapter.APPROVED_ROOT/'new-image') as directory:
            evidence=Path(directory)/'texture'
            result=subprocess.run([sys.executable,'-B',adapter.__file__,'--prepare-only','--shape-evidence',str(source),
                                   '--evidence',str(evidence),'--artifact-root',str(self.root)],capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr)
            record=json.loads(result.stdout);self.assertFalse(record['model_inference'])
            probe=json.loads((evidence/'persistence-preflight.json').read_text())
            self.assertTrue(probe['synthetic']);self.assertFalse(probe['torch_imported'])
            self.assertEqual(probe['metadata']['tensors']['raw']['shape'],[4430,32])
            self.assertTrue((self.root/'latent').is_dir());self.assertFalse(any((self.root/'latent').iterdir()))


if __name__=='__main__':unittest.main(verbosity=2)
