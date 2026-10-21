import io,json,struct,unittest
import numpy as np
from PIL import Image
from assets import decode_glb,align_uv

def fixture():
    blobs=[]
    for channels in (4,3):
        a=np.arange(4*4*channels,dtype=np.uint8).reshape(4,4,channels)
        out=io.BytesIO();Image.fromarray(a).save(out,format='PNG');blobs.append(out.getvalue())
    v=np.array([[0,0,0],[1,0,0],[0,1,0]],'<f4');uv=v[:,:2].copy();ind=np.array([0,1,2],'<u2')
    parts=[v.tobytes(),uv.tobytes(),ind.tobytes(),*blobs];binary=bytearray();views=[]
    for p in parts:
        while len(binary)%4:binary.append(0)
        views.append(dict(buffer=0,byteOffset=len(binary),byteLength=len(p)));binary.extend(p)
    d={'asset':{'version':'2.0'},'buffers':[{'byteLength':len(binary)}],'bufferViews':views,
       'accessors':[dict(bufferView=0,componentType=5126,count=3,type='VEC3'),dict(bufferView=1,componentType=5126,count=3,type='VEC2'),dict(bufferView=2,componentType=5123,count=3,type='SCALAR')],
       'meshes':[{'primitives':[{'attributes':{'POSITION':0,'TEXCOORD_0':1},'indices':2,'material':0}]}],
       'materials':[{'doubleSided':True,'pbrMetallicRoughness':{'baseColorTexture':{'index':0},'metallicRoughnessTexture':{'index':1}}}],
       'textures':[{'source':0,'sampler':0},{'source':1,'sampler':0}],
       'images':[{'bufferView':3,'mimeType':'image/png'},{'bufferView':4,'mimeType':'image/png'}],
       'samplers':[dict(magFilter=9729,minFilter=9987,wrapS=33071,wrapT=33071)]}
    def encode(doc):
        j=json.dumps(doc).encode();j+=b' '*(-len(j)%4);b=bytes(binary)+b'\0'*(-len(binary)%4)
        return struct.pack('<4sII',b'glTF',2,28+len(j)+len(b))+struct.pack('<II',len(j),0x4e4f534a)+j+struct.pack('<II',len(b),0x004e4942)+b
    return d,encode

class AssetTests(unittest.TestCase):
    def test_actual_container_reader_preserves_png_rows_and_coordinates(self):
        doc,encode=fixture();r=decode_glb(encode(doc),size=4)
        self.assertEqual(r['base'].shape,(4,4,4));np.testing.assert_array_equal(r['base'][0],np.arange(64,dtype=np.uint8).reshape(4,4,4)[-1])
        np.testing.assert_array_equal(r['vertices'],[[0,0,0],[1,0,0],[0,0,1]])
        np.testing.assert_array_equal(r['uv'],[[0,1],[1,1],[0,0]])
    def test_external_map_sampler_change_and_truncated_container_reject(self):
        for mutation in ('uri','sampler','extensions'):
            doc,encode=fixture()
            if mutation=='uri':doc['images'][0]['uri']='foreign.png'
            elif mutation=='sampler':doc['samplers'][0]['minFilter']=9729
            else:doc['extensionsRequired']=['foreign']
            with self.assertRaises(ValueError):decode_glb(encode(doc),size=4)
        doc,encode=fixture()
        with self.assertRaises(ValueError):decode_glb(encode(doc)[:-1],size=4)
    def test_uv_match_is_oriented_and_multiplicity_preserving(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32);f=np.array([[0,1,2]],np.int32);uv=np.array([[.1,.1],[.8,.1],[.1,.8]],np.float32)
        glb={'vertices':v,'faces':np.array([[1,2,0]],np.int32),'uv':uv.copy()}
        a,error=align_uv(v,f,uv,glb);np.testing.assert_array_equal(a,uv);self.assertEqual(error,0)
        glb['faces']=np.array([[0,2,1]],np.int32)
        with self.assertRaises(ValueError):align_uv(v,f,uv,glb)
        glb['faces']=f;glb['uv']=uv+.001
        with self.assertRaises(ValueError):align_uv(v,f,uv,glb)

if __name__=='__main__':unittest.main()
