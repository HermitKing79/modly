"""CPU array-only source correspondence negatives; never imports bpy."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest
import numpy as np
from render_worker import validate_candidate

class Transfer(unittest.TestCase):
    def fixture(self):
        source=np.zeros((1957891,3),np.float32);source[:4]=[[0,0,0],[1,0,0],[0,1,0],[2,2,2]]
        colour=np.zeros((1957891,4),np.uint8);colour[:,3]=255;colour[:3,:3]=[[3,42,191],[24,64,97],[201,3,127]]
        ids=np.arange(3,dtype=np.int32);return source[:3].copy(),np.array([[0,1,2]],np.int32),ids,colour[:3].copy(),source,colour
    def test_retained_original_position_and_native_colour_bits(self):
        validate_candidate(*self.fixture())
    def test_colour_reassignment_and_moved_positions_rejected(self):
        for kind in ('colour','position','source_id','unused'):
            args=list(self.fixture())
            if kind=='colour':args[3][0,0]+=1
            elif kind=='position':args[0][0,0]=.01
            elif kind=='source_id':args[2][2]=3
            else:args[1]=np.array([[0,1,0]],np.int32)
            with self.subTest(kind=kind),self.assertRaises(ValueError):validate_candidate(*args)

if __name__=='__main__':unittest.main()
