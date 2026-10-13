"""Focused CPU contracts; the first tests require no AI framework imports."""
import copy
import unittest
import amd_sparse_flow_contract as policy

class ContractTests(unittest.TestCase):
    def test_exact_parameter_schema(self):
        rows=policy.schema()
        self.assertEqual(len(rows),640)
        self.assertEqual(sum(v['dtype']=='F32' for v in rows.values()),220)
        self.assertEqual(sum(v['dtype']=='BF16' for v in rows.values()),420)
        self.assertEqual(rows['blocks.29.mlp.mlp.0.weight']['shape'],[8192,1536])
        self.assertEqual(rows['blocks.0.self_attn.q_rms_norm.gamma']['dtype'],'F32')
    def test_config_and_header(self):
        config=copy.deepcopy(policy.CONFIG)
        header={k:{'shape':v['shape'],'dtype':'BF16'} for k,v in policy.schema().items()}
        policy.validate_header(config,header)
        for change in ('key','shape','dtype','config'):
            bad=copy.deepcopy(header);cfg=copy.deepcopy(config)
            if change=='key':bad['unapproved']={'shape':[1],'dtype':'BF16'}
            if change=='shape':bad['input_layer.bias']['shape']=[1]
            if change=='dtype':bad['input_layer.bias']['dtype']='F32'
            if change=='config':cfg['args']['resolution']=64
            with self.assertRaises(ValueError):policy.validate_header(cfg,bad)
    def test_single_call_consumed_even_on_error(self):
        permit=policy.SingleForward()
        with self.assertRaises(ZeroDivisionError):permit.run(lambda:1/0)
        with self.assertRaises(RuntimeError):permit.run(lambda:None)
        self.assertEqual(permit.calls,1)
    def test_output_contract(self):
        row=dict(shape=[1,8,16,16,16],dtype='torch.float32',device='cuda:0',all_finite=True,
                 count=32768,requires_grad=False,is_inference_tensor=True,grad_enabled=False)
        policy.validate_output(row)
        for key,value in [('shape',[1,8,64,64,64]),('dtype','torch.bfloat16'),('device','cpu'),
                          ('all_finite',False),('requires_grad',True),('count',1)]:
            with self.assertRaises(ValueError):policy.validate_output({**row,key:value})

if __name__=='__main__':unittest.main()
