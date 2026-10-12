"""CPU-only transfer permits: synthetic descriptors, no GPU or real model loads."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import amd_decoder_forward_contract as contract


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(contract, 'TensorTransfers'), 'Missing exact single-use tensor transfer permit')
        self.permit = contract.TensorTransfers()
        self.source = object()
        self.input = {'shape': [1,8,1,1,1], 'dtype': 'torch.float32', 'device': 'cpu',
                      'stride': [8,1,1,1,1],
                      'layout': 'torch.strided', 'contiguous': True, 'requires_grad': False}
        self.output = {**self.input, 'shape': [1,1,4,4,4], 'device': 'cuda:0', 'stride': [64,64,16,4,1]}

    def begin(self, direction='input', source=None, description=None, phase=None):
        self.permit.begin(direction, source or self.source, description or (self.input if direction=='input' else self.output),
                          phase or ('input-transfer' if direction=='input' else 'output-transfer'))

    def authorize(self, **changes):
        arguments = {'operation': 'aten._to_copy.default', 'args': (self.source,),
                     'kwargs': {'device': 'cuda:0'}, 'phase': 'input-transfer', 'description': self.input}
        arguments.update(changes)
        return self.permit.authorize(**arguments)

    def test_exact_input_copy_is_allowed_once(self):
        self.begin()
        self.authorize()
        self.assertEqual(self.permit.counts, {'input': 1, 'output': 0})
        self.assertIsNone(self.permit.active)
        with self.assertRaises(RuntimeError): self.begin()

    def test_exact_output_copy_to_cpu_is_allowed_once(self):
        self.begin(); self.authorize()
        self.begin('output')
        self.authorize(kwargs={'device': 'cpu'}, phase='output-transfer', description=self.output)
        self.assertEqual(self.permit.counts, {'input': 1, 'output': 1})
        with self.assertRaises(RuntimeError): self.begin('output')

    def test_output_before_input_is_rejected(self):
        with self.assertRaises(RuntimeError): self.begin('output')

    def test_wrong_source_identity_or_positional_arguments_are_rejected(self):
        for args in ((), (object(),), (self.source, False), [self.source]):
            with self.subTest(args=args):
                self.begin()
                with self.assertRaises(RuntimeError): self.authorize(args=args)
                self.permit = contract.TensorTransfers()

    def test_unapproved_operations_remain_rejected(self):
        for op in ('aten.to.dtype_layout','aten.copy_.default','aten.empty_strided.default','aten.add.Tensor','aten.convolution.default'):
            with self.subTest(op=op):
                self.begin()
                with self.assertRaises(RuntimeError): self.authorize(operation=op)
                self.permit = contract.TensorTransfers()

    def test_unexpected_devices_and_arguments_are_rejected(self):
        for kwargs in ({}, {'device':'cuda'}, {'device':'cuda:1'}, {'device':'cpu'}, {'device':'meta'},
                       {'device':'cuda:0','dtype':'torch.float16'}, {'device':'cuda:0','dtype':None},
                       {'device':'cuda:0','layout':'torch.sparse_coo'}, {'device':'cuda:0','non_blocking':True},
                       {'device':'cuda:0','pin_memory':False}, {'device':'cuda:0','memory_format':'torch.channels_last_3d'},
                       {'device':'cuda:0','copy':True}):
            with self.subTest(kwargs=kwargs):
                self.begin()
                with self.assertRaises(RuntimeError): self.authorize(kwargs=kwargs)
                self.permit = contract.TensorTransfers()

    def test_invalid_tensor_metadata_is_rejected_before_dispatch(self):
        for change in ({'shape':[1,8,2,1,1]}, {'dtype':'torch.float16'}, {'device':'cuda:0'},
                       {'layout':'torch.sparse_coo'}, {'contiguous':False}, {'requires_grad':True},
                       {'contiguous':1}, {'extra':0}):
            with self.subTest(change=change):
                with self.assertRaises(RuntimeError): self.begin(description={**self.input, **change})

    def test_wrong_lifecycle_phases_and_overlapping_permits_are_rejected(self):
        for phase in ('cpu','placement','forward','release','output-transfer'):
            with self.subTest(phase=phase):
                with self.assertRaises(RuntimeError): self.begin(phase=phase)
        self.begin()
        with self.assertRaises(RuntimeError): self.begin()
        for phase in ('cpu','placement','forward','release','output-transfer'):
            with self.subTest(dispatch_phase=phase):
                with self.assertRaises(RuntimeError): self.authorize(phase=phase)

    def test_tensor_metadata_cannot_change_after_permit_creation(self):
        self.begin()
        with self.assertRaises(RuntimeError): self.authorize(description={**self.input, 'dtype':'torch.float16'})

    def test_input_stride_is_exact_and_output_stride_is_bound_before_dispatch(self):
        with self.assertRaises(RuntimeError): self.begin(description={**self.input, 'stride':[99,1,1,1,1]})
        self.begin(); self.authorize()
        self.begin('output')
        with self.assertRaises(RuntimeError):
            self.authorize(kwargs={'device':'cpu'},phase='output-transfer',description={**self.output,'stride':[1,1,16,4,1]})

    def test_no_implicit_unarmed_or_repeat_dispatch(self):
        with self.assertRaises(RuntimeError): self.authorize()
        self.begin(); self.authorize()
        with self.assertRaises(RuntimeError): self.authorize()

    def test_output_metadata_and_target_are_restricted(self):
        self.begin(); self.authorize()
        for change in ({'shape':[1,1,2,2,2]}, {'device':'cuda:1'}, {'dtype':'torch.float16'}):
            with self.assertRaises(RuntimeError): self.begin('output', description={**self.output, **change})
        self.begin('output')
        with self.assertRaises(RuntimeError):
            self.authorize(kwargs={'device':'cuda:0'}, phase='output-transfer', description=self.output)

    def test_pass_requires_both_exact_transfer_counts(self):
        from test_amd_decoder_forward import ForwardContractTests
        record=ForwardContractTests().good_record()
        for counts in (None, {'input':0,'output':1}, {'input':1,'output':0}, {'input':2,'output':1}, {'input':True,'output':1}):
            candidate={**record,'transfer_calls':counts}
            self.assertEqual(contract.classify(candidate,True,True,True),'PARTIAL')


if __name__ == '__main__': unittest.main()
