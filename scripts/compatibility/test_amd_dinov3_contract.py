"""CPU-only safety decisions: never import Torch or load real model tensors."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parent))
import amd_dinov3_contract as contract


CONFIG = dict(model_type='dinov3_vit', architectures=['DINOv3ViTModel'], hidden_size=1024,
              patch_size=16, num_channels=3, num_register_tokens=4, num_hidden_layers=24,
              num_attention_heads=16, torch_dtype='float32')


class ContractTests(unittest.TestCase):
    def test_exact_output(self):
        self.assertEqual(contract.expected_shape(CONFIG), [1, 1029, 1024])

    def test_incompatible_config(self):
        for key, value in [('hidden_size', 768), ('torch_dtype', 'float16'), ('patch_size', 14),
                           ('num_register_tokens', 0), ('num_hidden_layers', 12)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                contract.expected_shape({**CONFIG, key: value})

    def test_nonpersistent_buffer(self):
        good={'rope_embeddings.inv_freq': {'shape':[16], 'dtype':'F32', 'device':'cpu', 'is_meta':False}}
        contract.validate_buffers(good)
        for changed in ({}, {**good, 'extra': good['rope_embeddings.inv_freq']},
                        {'rope_embeddings.inv_freq': {**good['rope_embeddings.inv_freq'], 'device':'meta'}},
                        {'rope_embeddings.inv_freq': {**good['rope_embeddings.inv_freq'], 'dtype':'F16'}}):
            with self.assertRaises(ValueError): contract.validate_buffers(changed)

    def test_exact_float32_transfers(self):
        expected={'shape':[1,3,512,512], 'dtype':'torch.float32', 'device':'cpu'}
        contract.validate_transfer('input', expected, 'cuda:0', {}, 'conditioning')
        for source, target, args, phase in [(expected, 'cuda:1', {}, 'conditioning'),
                (expected, 'cpu', {}, 'conditioning'), (expected, 'cuda:0', {'dtype':'torch.float16'}, 'conditioning'),
                (expected, 'cuda:0', {}, 'cpu'), ({**expected,'shape':[2,3,512,512]},'cuda:0',{},'conditioning')]:
            with self.assertRaises(ValueError):contract.validate_transfer('input',source,target,args,phase)
        out={'shape':[1,1029,1024], 'dtype':'torch.float32', 'device':'cuda:0'}
        contract.validate_transfer('output',out,'cpu',{},'output')
        with self.assertRaises(ValueError):contract.validate_transfer('output',out,'cpu',{},'conditioning')

    def test_single_use_and_counts(self):
        gate=contract.SingleConditioning()
        self.assertEqual(gate.run(lambda: 42),42)
        with self.assertRaises(RuntimeError):gate.run(lambda: 0)
        counts={'extractor':1,'features':1,'embeddings':1,'rope_embeddings':1,
                **{'layer.'+str(i):1 for i in range(24)}}
        contract.validate_counts(counts)
        with self.assertRaises(ValueError):contract.validate_counts({**counts,'layer.2':2})

    def test_exception_consumes_attempt(self):
        gate=contract.SingleConditioning()
        with self.assertRaises(ZeroDivisionError):gate.run(lambda:1/0)
        with self.assertRaises(RuntimeError):gate.run(lambda:None)

    def test_output_and_release(self):
        good={'shape':[1,1029,1024], 'dtype':'torch.float32','device':'cuda:0',
              'all_finite':True,'count':1053696,'conditioning_width':1024}
        contract.validate_output(good)
        for field, value in [('all_finite',False),('device','cpu'),('count',64),('dtype','torch.float16')]:
            with self.assertRaises(ValueError):contract.validate_output({**good,field:value})


if __name__=='__main__':unittest.main()
