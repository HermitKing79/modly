"""CPU-only genuine sampler/interface replay; no checkpoints or GPU entry."""
import ast,copy,gc,inspect,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import torch
import amd_texture_sampling_control as control
import amd_texture_sampling_contract as policy
import amd_texture_sampling as entry
from test_amd_shape_sampling_runtime import Fixture,hooks,forbidden,SparseTensor,CPUSink
import amd_forward_diagnostics as diagnostics

def actual_hooks(model,c,trace):
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves
    tree=ast.parse(Path(entry.__file__).read_text());nodes=[]
    for name in ('module_call','timestep_tensor'):
        nodes.append(copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)))
    nodes.append(copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='BoundedDispatch')))
    mem=lambda:dict(allocated_bytes=0,reserved_bytes=0,peak_allocated_bytes=0,peak_reserved_bytes=0,available_bytes=4*1024**3)
    p=c.copy_policy(3743);p.phase='forward';p.rope=model.blocks[0].self_attn.rope.freqs
    ns=dict(torch=torch,sampling=c,policy=p,original_call=torch.nn.Module.__call__,original_tensor=torch.tensor,
        modules={id(v):n for n,v in model.named_modules()},record={'calls':{}},trace=trace,describe=entry.describe,forbidden=forbidden,
        contract=policy,strict=entry.strict,local=type('L',(),{'loading':type('P',(),{'gpu_memory':staticmethod(lambda t:mem())})})(),
        op_counts={},TorchDispatchMode=TorchDispatchMode,tree_leaves=tree_leaves)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),entry.__file__,'exec'),ns)
    ns['BoundedDispatch'].__torch_dispatch__=inspect.unwrap(ns['BoundedDispatch'].__torch_dispatch__)
    return ns,mem

class Projection(torch.nn.Module):
    def forward(self,x):
        if x.feats.shape!=(3743,64):raise RuntimeError('Missing genuine noise-first concatenation')
        return x.replace(x.feats[:,:32])
class TextureFixture(Fixture):
    def __init__(self):super().__init__();self.input_layer=Projection()
    def forward(self,x,t,cond,concat_cond=None):
        from trellis2.modules.sparse import sparse_cat
        x=self.input_layer(sparse_cat([x,concat_cond],dim=-1))
        return super().forward(x,t,cond)

class RuntimeTests(unittest.TestCase):
    def test_actual_sampler_twelve_positive_forwards_concat_cache_and_cleanup(self):
        cls=entry.local.import_sampler();model=TextureFixture();c=control.TextureSamplingControl(torch,cls,device='cpu')
        coords=torch.from_numpy(np.load(entry.local.COORDS,allow_pickle=False))
        noise=torch.randn(3743,32,generator=torch.Generator().manual_seed(policy.SEED))
        x=SparseTensor(noise,coords);x.register_spatial_cache('layout',[slice(0,3743)]);x.register_spatial_cache('seqlen',torch.tensor([3743],dtype=torch.int64))
        shape=x.replace(torch.linspace(-1,1,3743*32).reshape(3743,32))
        inputs=dict(cond=torch.ones(1,1029,1024)*.25,shape_cond=shape)
        lines=[];trace=diagnostics.Trace(lines.append,limit=256)
        ns,mem=actual_hooks(model,c,trace)
        with torch.inference_mode(),CPUSink(),ns['BoundedDispatch'](),patch.object(torch.nn.Module,'__call__',ns['module_call']),patch.object(torch,'tensor',ns['timestep_tensor']):
            result=c.run(model,x,inputs,trace,lambda:None,mem)
        self.assertEqual((c.budget.calls,c.t_copies,c.frequency_copies,c.concat_calls),(12,12,12,12))
        self.assertEqual([r['forwards'] for r in c.steps],[1]*12)
        self.assertEqual([r['role'] for r in c.models],['positive']*12)
        self.assertTrue(all(r['cache_reused'] for r in c.models[1:]));self.assertIs(result.samples.coords,coords)
        with tempfile.TemporaryDirectory() as d:
            j=diagnostics.Journal(Path(d)/'trace.jsonl')
            for line in lines:j.consume(line)
            self.assertTrue(j.close())
            # Exercise the real production provenance and writer together with
            # the complete twelve-call diagnostic structure before inference.
            from test_amd_texture_sampling_result import candidate
            import amd_texture_sampling_artifact as writer
            from safetensors.numpy import load_file
            record=candidate();identity=entry.local.identity(hash_weights=False)
            record.update(provenance=dict(checkpoint_sha256=entry.local.CHECKPOINT_SHA,config_sha256=entry.local.CONFIG_SHA,
                conditioning=identity['feature'],saved_inputs=identity['saved_inputs'],pipeline_sha256=identity['pipeline_sha256']),
                runtime=dict(torch=str(torch.__version__)),before_placement=mem(),after_forward=mem(),budget={'allocator_limit_bytes':policy.MEMORY_LIMIT},
                tunable_import={},tunable_before_forward={})
            provenance=entry.make_provenance(record,c,'b'*40)
            raw=result.samples.feats.numpy();denormalized=policy.denormalize_texture(raw)
            written=writer.write(Path(d).resolve(),raw,denormalized,coords.numpy(),provenance,noise=noise.numpy())
            independent=load_file(written['artifact_path'])
            for name,array in [('raw',raw),('denormalized',denormalized),('coords',coords.numpy()),('noise',noise.numpy())]:
                self.assertEqual(independent[name].tobytes(),array.tobytes())
            self.assertTrue(writer.verify(Path(written['artifact_path']),Path(d).resolve(),written['metadata'])['verified'])
        refs=c.issued.copy();result.samples.clear_spatial_cache();c.drop_references();result=x=shape=inputs=coords=noise=None;gc.collect()
        self.assertTrue(all(r() is None for r in refs))
    def test_wrong_concat_order_foreign_conditions_and_memory_are_rejected(self):
        c=control.TextureSamplingControl(torch,None,device='cpu')
        coords=torch.from_numpy(np.load(entry.local.COORDS,allow_pickle=False));a=torch.ones(3743,32);b=torch.zeros_like(a)
        c.coords=coords;c.step_x=SparseTensor(a,coords);c.shape_cond=SparseTensor(b,coords)
        c.budget.begin_step(*policy.times()[:2]);ticket=c.budget.next_ticket();c.budget.begin_forward(ticket)
        for values in (torch.cat([b,a],dim=-1),torch.ones(3743,63)):
            with self.assertRaises(RuntimeError):c.validate_concat(SparseTensor(values,coords))
        valid=SparseTensor(torch.cat([a,b],dim=-1),coords);c.validate_concat(valid)
        with self.assertRaises(RuntimeError):c.validate_concat(valid)
        with self.assertRaises(RuntimeError):c.begin_model((c.step_x,torch.zeros(1),torch.zeros(1,1029,1024)),{},c.copy_policy(3743))
        with self.assertRaises(RuntimeError):c.check_memory(dict(allocated_bytes=policy.MEMORY_LIMIT+1,reserved_bytes=0,available_bytes=4*1024**3))
        with self.assertRaises(RuntimeError):c.check_memory(dict(allocated_bytes=0,reserved_bytes=0,available_bytes=policy.FREE_RESERVE-1))

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeTests))
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(gpu_initialized=False,real_weights_loaded=False,real_model_forwards=0)))
    raise SystemExit(not r.wasSuccessful())
