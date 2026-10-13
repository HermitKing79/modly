"""Synthetic CPU-only tests: admission precedes any original allocation."""
import unittest
import amd_shape_decoder_admission as guard


class AdmissionTests(unittest.TestCase):
    def memory(self,reserved=2*guard.GIB):return guard.Memory(reserved,reserved,14*guard.GIB)

    def test_fixed_ceiling_and_initial_free_gate(self):
        guard.initial_admission(self.memory())
        self.assertEqual(guard.ALLOCATOR_CAP,12*guard.GIB)
        with self.assertRaises(ValueError):guard.initial_admission(guard.Memory(0,0,13*guard.GIB-1))

    def test_four_original_stages_normal_expansion(self):
        parent=3743
        for stage in range(4):
            child=parent*2
            for channels in (guard.CHANNELS[stage],8*guard.CHANNELS[stage+1]):
                p=guard.subdivision(self.memory(),stage=stage,parent_tokens=parent,child_tokens=child,input_channels=channels)
                self.assertEqual(p['resolution'],64*2**stage)
                self.assertEqual(p['child_tokens'],child)
            parent=child

    def test_worst_case_expansion_rejected_before_callback(self):
        calls=[]
        with self.assertRaises(ValueError):
            guard.before_operation(guard.subdivision,lambda:calls.append('allocated'),memory=self.memory(),stage=3,
                                   parent_tokens=1916416,child_tokens=15331328,input_channels=512)
        self.assertEqual(calls,[])

    def test_retained_cache_and_projected_next_conv_counted(self):
        with self.assertRaises(ValueError):guard.subdivision(self.memory(11*guard.GIB),stage=0,parent_tokens=3743,child_tokens=29944,input_channels=4096)
        a=guard.subdivision(self.memory(),stage=0,parent_tokens=3743,child_tokens=3743,input_channels=4096,cached=True)
        self.assertEqual(a['geometry_generation_bytes'],0)
        self.assertEqual(a['gather_index_bytes'],16*3743)
        self.assertEqual(a['coordinate_and_index_bytes'],16*3743)
        self.assertGreater(a['feature_bytes'],0)

    def test_malformed_metadata_cannot_reduce_admission(self):
        for cached in ('yes',1,None):
            with self.assertRaises(ValueError):guard.subdivision(self.memory(),stage=0,parent_tokens=3743,child_tokens=3743,input_channels=4096,cached=cached)
        for splitk in (True,1.0):
            with self.assertRaises(ValueError):guard.convolution(self.memory(),tokens=3743,in_channels=1024,out_channels=1024,splitk=splitk,stage=0)
        for channels in (1024.0,True):
            with self.assertRaises(ValueError):guard.convolution(self.memory(),tokens=3743,in_channels=channels,out_channels=1024,splitk=1,stage=0)
        with self.assertRaises(ValueError):guard.subdivision(self.memory(),stage=0,parent_tokens=3743,child_tokens=3743,input_channels=4096.0)

    def test_convolution_source_widths_and_pre_subdivision_wide_output(self):
        p=guard.convolution(self.memory(),tokens=3743,in_channels=1024,out_channels=4096,splitk=1,stage=0)
        self.assertEqual(p['output_bytes'],3743*4096*2)
        for stage,c in enumerate(guard.CHANNELS):
            guard.convolution(self.memory(),tokens=3743,in_channels=c,out_channels=c,splitk=1,stage=stage)

    def test_visible_second_postprocess_and_split_reduction_are_counted(self):
        n=3743
        p=guard.convolution(self.memory(),tokens=n,in_channels=1024,out_channels=1024,splitk=2,stage=0)
        self.assertEqual(p['native_neighbor_bytes'],680*n+112+120*((n+63)//64)+8)
        self.assertEqual(p['split_reduction_bytes'],n*1024*4)
        sub=guard.subdivision(self.memory(),stage=0,parent_tokens=n,child_tokens=2*n,input_channels=4096)
        self.assertEqual(sub['next_splitk_upper_bound'],4)
        self.assertGreaterEqual(sub['next_split_accumulator_and_reduction_bytes'],2*n*512*4*5)

    def test_invalid_stage_shape_counts_and_overflow(self):
        for child in (0,3743*8+1,True,2**63):
            with self.assertRaises(ValueError):guard.subdivision(self.memory(),stage=0,parent_tokens=3743,child_tokens=child,input_channels=4096)
        with self.assertRaises(ValueError):guard.checked_product(2**31-1,2**31-1,4)
        with self.assertRaises(ValueError):guard.convolution(self.memory(),tokens=10,in_channels=1024,out_channels=128,splitk=1,stage=0)
        with self.assertRaises(ValueError):guard.convolution(self.memory(),tokens=10,in_channels=1024,out_channels=1024,splitk=16,stage=0)

    def test_memory_rejects_before_allocation_and_artifact_bound(self):
        calls=[]
        with self.assertRaises(ValueError):
            guard.before_operation(guard.convolution,lambda:calls.append('native'),memory=self.memory(11*guard.GIB),tokens=3743,
                                   in_channels=1024,out_channels=1024,splitk=2,stage=0)
        self.assertEqual(calls,[])
        self.assertLess(guard.output_admission(tokens=20000,subdivision_tokens=[3743,6000,10000,15000]),guard.OUTPUT_LIMIT)


if __name__=='__main__':unittest.main()
