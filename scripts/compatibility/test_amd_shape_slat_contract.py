import copy, sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import amd_shape_slat_contract as c

class ShapeContractTests(unittest.TestCase):
    def test_pool_threshold_axis_order(self):
        logits=np.zeros((1,1,64,64,64),dtype=np.float32)
        logits[0,0,3,8,63]=1
        mask,coords=c.reduce_occupancy(logits,logits[0,0]>0)
        self.assertEqual(mask.shape,(32,32,32))
        np.testing.assert_array_equal(coords,np.array([[0,1,4,31]],dtype=np.int32))
    def test_reject_invalid_saved_mask_and_nonfinite(self):
        logits=np.zeros((1,1,64,64,64),dtype=np.float32)
        with self.assertRaises(ValueError):c.reduce_occupancy(logits,np.ones((64,64,64),dtype=bool))
        logits.flat[0]=np.nan
        with self.assertRaises(ValueError):c.reduce_occupancy(logits,np.zeros((64,64,64),dtype=bool))
    def test_unique_ordered_coordinates(self):
        good=np.array([[0,0,0,1],[0,0,0,2]],dtype=np.int32)
        c.validate_coords(good,2)
        for bad in (good[::-1],np.vstack([good[0],good[0]]),good.astype(np.int64),good+32):
            with self.assertRaises(ValueError):c.validate_coords(bad,2)
    def test_exact_architecture_schema(self):
        s=c.schema();self.assertEqual(len(s),640)
        self.assertEqual(s['input_layer.weight']['shape'],[1536,32])
        self.assertEqual(s['out_layer.weight']['shape'],[32,1536])
        h={k:{**v,'dtype':'BF16'} for k,v in s.items()}
        c.validate_header(c.CONFIG,h)
        for key in ('input_layer.weight','out_layer.weight'):
            bad=copy.deepcopy(h);bad[key]['shape'][0]+=1
            with self.assertRaises(ValueError):c.validate_header(c.CONFIG,bad)
        bad=copy.deepcopy(h);bad.pop('input_layer.bias')
        with self.assertRaises(ValueError):c.validate_header(c.CONFIG,bad)
    def test_one_forward_only(self):
        p=c.SingleForward();self.assertEqual(p.run(lambda:3),3)
        with self.assertRaises(RuntimeError):p.run(lambda:4)
    def test_exact_transfer_contract(self):
        for key,shape,dtype in [('x',[7,32],'torch.float32'),('coords',[7,4],'torch.int32'),('batch_map',[7],'torch.int64'),('t',[1],'torch.float32'),('cond',[1,1029,1024],'torch.float32')]:
            row=dict(shape=shape,dtype=dtype,device='cpu')
            c.validate_transfer(key,row,'cuda:0','inputs',7)
            for field,value in [('dtype','torch.float16'),('device','cuda:1'),('shape',[999])]:
                bad={**row,field:value}
                with self.assertRaises(ValueError):c.validate_transfer(key,bad,'cuda:0','inputs',7)
        with self.assertRaises(ValueError):c.validate_transfer('foreign',{},'cuda:0','inputs',7)

if __name__=='__main__':unittest.main()
