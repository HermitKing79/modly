"""Bounded core glTF writer: original indices, F32 RGB, no normal/UV generation."""
import sys
sys.dont_write_bytecode=True
import json,os,re,struct
import numpy as np

def need(ok,text):
    if not ok:raise ValueError(text)

def linear_rgb(encoded):
    need(isinstance(encoded,np.ndarray) and encoded.dtype==np.float32 and encoded.ndim==2 and encoded.shape[1]==3 and len(encoded)<=32768 and np.isfinite(encoded).all(),'Finite bounded sampled F32 RGB')
    x=np.clip(encoded.astype(np.float64),0,1)
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4).astype(np.float32)

def write_glb(path,vertices,faces,colours,source_hashes,*,batch_rows=32768):
    need(type(batch_rows) is int and 1<=batch_rows<=32768,'Maximum32768 streamed rows')
    need(isinstance(vertices,np.ndarray) and vertices.dtype==np.float32 and vertices.ndim==2 and vertices.shape[1]==3 and 0<len(vertices)<=2_000_000,'Original bounded F32 vertices')
    need(isinstance(faces,np.ndarray) and faces.dtype==np.int32 and faces.ndim==2 and faces.shape[1]==3 and 0<len(faces)<=4_173_584,'Original bounded I32 triangles')
    need(isinstance(colours,np.ndarray) and colours.dtype==np.float32 and colours.shape==vertices.shape,'Full F32 linear colours')
    need(isinstance(source_hashes,dict) and source_hashes and all(isinstance(k,str) and isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for k,v in source_hashes.items()),'Compact exact source hash identities')
    low=np.full(3,np.inf);high=-low
    for start in range(0,len(vertices),batch_rows):
        v=vertices[start:start+batch_rows];c=colours[start:start+batch_rows]
        need(np.isfinite(v).all() and np.isfinite(c).all() and ((c>=0)&(c<=1)).all(),'Finite geometry and unit linear colours')
        p=v[:,[0,2,1]].copy();p[:,2]*=-1
        low=np.minimum(low,p.min(0).astype(np.float64));high=np.maximum(high,p.max(0).astype(np.float64))
    for start in range(0,len(faces),batch_rows):
        f=faces[start:start+batch_rows];need(np.all(f>=0) and np.all(f<len(vertices)),'Original triangle index bounds')
        xyz=vertices[f].astype(np.float64);area=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
        need(np.all(np.any(area!=0,axis=1)),'Original nondegenerate oriented triangles')
    n,k=len(vertices),len(faces);lengths=[n*12,n*12,k*12];offsets=[0,lengths[0],lengths[0]+lengths[1]]
    doc=dict(asset=dict(version='2.0',generator='Modly source-sampled vertex-colour CPU export'),scene=0,scenes=[dict(nodes=[0])],nodes=[dict(mesh=0)],
        buffers=[dict(byteLength=sum(lengths))],bufferViews=[dict(buffer=0,byteOffset=o,byteLength=b,target=34963 if i==2 else 34962) for i,(o,b) in enumerate(zip(offsets,lengths))],
        accessors=[dict(bufferView=0,componentType=5126,count=n,type='VEC3',min=low.tolist(),max=high.tolist()),dict(bufferView=1,componentType=5126,count=n,type='VEC3'),dict(bufferView=2,componentType=5125,count=k*3,type='SCALAR')],
        materials=[dict(alphaMode='OPAQUE',doubleSided=True,pbrMetallicRoughness=dict(baseColorFactor=[1,1,1,1],metallicFactor=.00011082693761409814,roughnessFactor=.999995986777544))],
        meshes=[dict(primitives=[dict(attributes=dict(POSITION=0,COLOR_0=1),indices=2,material=0,mode=4)],extras=source_hashes)])
    encoded=json.dumps(doc,separators=(',',':'),allow_nan=False).encode('utf-8');encoded+=b' '*(-len(encoded)%4)
    need(len(encoded)<=65536,'Small GLB metadata, no giant ID lists')
    total=28+len(encoded)+sum(lengths);need(total<=128*1024**2 and total%4==0,'Bounded aligned shared-vertex GLB')
    with path.open('xb') as out:
        out.write(struct.pack('<4sII',b'glTF',2,total));out.write(struct.pack('<II',len(encoded),0x4e4f534a));out.write(encoded);out.write(struct.pack('<II',sum(lengths),0x004e4942))
        for start in range(0,n,batch_rows):
            p=vertices[start:start+batch_rows,[0,2,1]].copy();p[:,2]*=-1;out.write(p.astype('<f4',copy=False).tobytes(order='C'))
        for start in range(0,n,batch_rows):out.write(colours[start:start+batch_rows].astype('<f4',copy=False).tobytes(order='C'))
        for start in range(0,k,batch_rows):out.write(faces[start:start+batch_rows].astype('<u4').tobytes(order='C'))
        need(out.tell()==total,'Every streamed byte count accounted for');out.flush();os.fsync(out.fileno())
    need(path.stat().st_size==total,'Exact persisted GLB byte length')
    return dict(bytes=total,vertices=n,triangles=k,buffer_bytes=sum(lengths),JSON_bytes=len(encoded),batch_rows=batch_rows,source_to_glTF='(x,z,-y); determinant+1, unchanged winding',F32_COLOR0=True,uint32_indices=True)
