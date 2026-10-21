import sys
sys.dont_write_bytecode=True
import json,struct,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from glb_writer import build_glb


class WriterTests(unittest.TestCase):
    def values(self):
        return [np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32),
                np.array([[0,1,2],[0,3,1],[1,3,2]],np.int32),
                np.array([[0,.1,1],[.3,.4,.5],[.6,.7,.8],[1,0,.25]],np.float32),
                np.array([10,11,12,13],np.int64),np.array([5,6,7],np.int32)]

    def build(self,*values):
        return build_glb(*(values or self.values()),'a'*64,'b'*64)

    def test_bytes_preserve_linear_colour_and_oriented_shared_geometry(self):
        values=self.values();before=[x.tobytes() for x in values]
        payload=self.build(*values)
        magic,version,total=struct.unpack_from('<III',payload)
        self.assertEqual((magic,version,total),(0x46546c67,2,len(payload)))
        length,kind=struct.unpack_from('<II',payload,12)
        self.assertEqual(kind,0x4e4f534a)
        doc=json.loads(payload[20:20+length]);binary=payload[28+length:]
        arrays=[]
        for accessor in doc['accessors']:
            view=doc['bufferViews'][accessor['bufferView']];start=view['byteOffset']
            arrays.append(binary[start:start+view['byteLength']])
        expected=values[0][:,[0,2,1]].copy();expected[:,2]*=-1
        self.assertEqual(arrays[0],expected.tobytes())
        self.assertEqual(arrays[1],values[2].tobytes())
        self.assertEqual(arrays[2],values[1].astype('<u2').tobytes())
        self.assertEqual(doc['accessors'][0]['count'],4)
        self.assertEqual(doc['accessors'][1]['componentType'],5126)
        self.assertEqual(doc['meshes'][0]['primitives'][0]['attributes'],{'POSITION':0,'COLOR_0':1})
        self.assertEqual(doc['meshes'][0]['extras']['sourceVertexIDs'],values[3].tolist())
        self.assertEqual([x.tobytes() for x in values],before)

    def test_material_is_exact_opaque_double_sided_without_maps(self):
        payload=self.build();length=struct.unpack_from('<I',payload,12)[0]
        doc=json.loads(payload[20:20+length]);mat=doc['materials'][0]
        self.assertEqual(mat['alphaMode'],'OPAQUE');self.assertIs(mat['doubleSided'],True)
        self.assertEqual(mat['pbrMetallicRoughness'],dict(baseColorFactor=[1,1,1,1],metallicFactor=.00011082693761409814,roughnessFactor=.999995986777544))
        self.assertFalse(set(doc)&{'images','textures','samplers','extensionsUsed','extensionsRequired'})

    def test_nonfinite_out_of_range_or_encoded_dtype_rejected(self):
        for index,value in [(0,float('nan')),(2,1.1),(2,-.1)]:
            values=self.values();values[index][0,0]=value
            with self.assertRaises(ValueError):self.build(*values)
        values=self.values();values[2]=values[2].astype(np.float64)
        with self.assertRaises(ValueError):self.build(*values)

    def test_invalid_indices_shapes_and_source_identity_rejected(self):
        variants=[]
        v=self.values();v[1][0,0]=4;variants.append(v)
        v=self.values();v[1][0,0]=-1;variants.append(v)
        v=self.values();v[1][0]=[0,0,1];variants.append(v)
        v=self.values();v[2]=v[2][:-1];variants.append(v)
        v=self.values();v[3][0]=v[3][1];variants.append(v)
        v=self.values();v[4]=v[4][:-1];variants.append(v)
        for values in variants:
            with self.assertRaises(ValueError):self.build(*values)
        with self.assertRaises(ValueError):build_glb(*self.values(),'../bad','b'*64)


if __name__=='__main__':unittest.main()
