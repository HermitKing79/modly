"""CPU arithmetic matches the existing admitted formula; no GPU imports."""
import importlib.util
from pathlib import Path
import unittest
from admission import conv_plan
spec=importlib.util.spec_from_file_location('offline',Path(__file__).with_name('offline-estimate.py'))
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
class Tests(unittest.TestCase):
    def test_matches_existing_formula_without_changing_guard(self):
        n=123;original=conv_plan(n,64,64,512**3,False,n*(16*64+4*64))
        self.assertEqual(e.buffers(n),{key:original[key] for key in e.buffers(n)})
    def test_rejected_count_only_estimated_not_admitted(self):
        row=dict(site='blocks.3.4.updown',call=1,parents=378891,children=1000001,out_channels=64,
            parent_spatial_shape=[256,256,256],memory=dict(allocated_bytes=1,reserved_bytes=2,free_bytes=14*1024**3))
        result=e.estimate(row)
        self.assertTrue(result['row_gate_rejects']);self.assertFalse(result['GPU_execution_authorized'])
        self.assertEqual(result['buffer_upper_bounds']['im2col_bytes'],1000001*27*64*2)
        row['children']=0;self.assertTrue(e.estimate(row)['zero_children'])
if __name__=='__main__':unittest.main()
