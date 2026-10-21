"""Tiny CPU fixtures; no original model data or historical source writes."""
import sys
sys.dont_write_bytecode=True
import unittest,tempfile,json,struct
from pathlib import Path
import numpy as np
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from stream_writer import write_glb,linear_rgb
TESTROOT=E.parents[2].parent/'artifacts/amd-brick-full-sampled-glb-3db265c/unit-scratch'
TESTROOT.mkdir(exist_ok=True)

class WriterTests(unittest.TestCase):
    def data(self):
        return (np.array([[0.,-0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],np.float32),np.array([[0,1,2],[0,3,1],[0,2,3]],np.int32),np.array([[.1,.2,.3],[.2,.3,.4],[.3,.4,.5],[.4,.5,.6]],np.float32))
    def test_streamed_layout_transform_uint32_colours_and_exclusive_output(self):
        v,f,c=self.data();before=[x.copy() for x in (v,f,c)]
        with tempfile.TemporaryDirectory(dir=TESTROOT) as t:
            p=Path(t)/'fixture.glb';r=write_glb(p,v,f,c,{'fixture':'a'*64},batch_rows=2)
            b=p.read_bytes();magic,version,total=struct.unpack_from('<4sII',b);self.assertEqual((magic,version,total),(b'glTF',2,len(b)))
            jlen,kind=struct.unpack_from('<II',b,12);self.assertEqual(kind,0x4e4f534a);doc=json.loads(b[20:20+jlen]);off=28+jlen
            pos=np.frombuffer(b,'<f4',12,off).reshape(4,3);expected=v[:,[0,2,1]].copy();expected[:,2]*=-1
            np.testing.assert_array_equal(pos.view(np.uint32),expected.view(np.uint32))
            np.testing.assert_array_equal(np.frombuffer(b,'<f4',12,off+48).reshape(4,3).view(np.uint32),c.view(np.uint32))
            np.testing.assert_array_equal(np.frombuffer(b,'<u4',9,off+96).reshape(3,3),f)
            self.assertEqual(doc['accessors'][2]['componentType'],5125);self.assertEqual(r['bytes'],len(b))
            self.assertEqual(set(doc['meshes'][0]['primitives'][0]['attributes']),{'POSITION','COLOR_0'})
            with self.assertRaises(FileExistsError):write_glb(p,v,f,c,{'fixture':'a'*64})
        for actual,old in zip((v,f,c),before):np.testing.assert_array_equal(actual.view(np.uint32),old.view(np.uint32))

    def test_IEC_output_matches_qualified_original_function(self):
        import importlib.util
        p=E.parent/'amd-foliage-export-decision-evidence/metrics.py';spec=importlib.util.spec_from_file_location('writer_original_IEC',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        encoded=np.array([[-.1,.04045,1.1],[.05,.3,.8],[.4,.5,.6]],np.float32)
        _,old=m.color_prediction(encoded,np.array([[0,1,2]],np.int32),np.array([0],np.int32),np.array([[1/3]*3],np.float64))
        np.testing.assert_array_equal(linear_rgb(encoded).view(np.uint32),old.view(np.uint32))
        with self.assertRaises(ValueError):linear_rgb(encoded.astype(np.float64))

    def test_invalid_indices_nonfinite_colours_and_budget_fail_before_file(self):
        v,f,c=self.data()
        with tempfile.TemporaryDirectory(dir=TESTROOT) as t:
            p=Path(t)/'bad.glb'
            for badv,badf,badc,batch in ((v,np.array([[0,1,9]],np.int32),c,2),(v,f,np.full_like(c,np.nan),2),(v,f,c,32769),(v,np.array([[0,0,1]],np.int32),c,2)):
                with self.assertRaises(ValueError):write_glb(p,badv,badf,badc,{'fixture':'a'*64},batch_rows=batch)
                self.assertFalse(p.exists())
if __name__=='__main__':unittest.main()
