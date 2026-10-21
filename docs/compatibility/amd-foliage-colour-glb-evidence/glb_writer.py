"""Single-primitive F32 linear vertex-colour GLB, without normal/UV generation."""
import sys
sys.dont_write_bytecode=True
import json,re,struct
import numpy as np

METALLIC=.00011082693761409814
ROUGHNESS=.999995986777544


def build_glb(vertices,faces,linear,source_vertices,source_faces,vertex_ids_sha,face_ids_sha):
    def need(ok,text):
        if not ok:raise ValueError(text)
    need(isinstance(vertices,np.ndarray) and vertices.dtype==np.float32 and vertices.ndim==2 and vertices.shape[1]==3 and 0<len(vertices)<=6000,'Bounded F32 original vertices')
    need(isinstance(faces,np.ndarray) and faces.dtype==np.int32 and faces.ndim==2 and faces.shape[1]==3 and 0<len(faces)<=6000,'Bounded I32 original faces')
    need(isinstance(linear,np.ndarray) and linear.dtype==np.float32 and linear.shape==vertices.shape,'Exact F32 linear COLOR_0 schema')
    need(np.isfinite(vertices).all() and np.isfinite(linear).all() and ((linear>=0)&(linear<=1)).all(),'Finite positions and unit linear colours')
    need(faces.min()>=0 and faces.max()<len(vertices) and (faces[:,0]!=faces[:,1]).all() and (faces[:,1]!=faces[:,2]).all() and (faces[:,2]!=faces[:,0]).all(),'Original valid triangle indices')
    xyz=vertices[faces].astype(np.float64)
    need((np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)>0).all(),'No zero-area original triangles')
    for ids,dtype,count in ((source_vertices,np.int64,len(vertices)),(source_faces,np.int32,len(faces))):
        need(isinstance(ids,np.ndarray) and ids.dtype==dtype and ids.shape==(count,) and ids.min()>=0 and len(np.unique(ids))==count,'Exact unique source IDs')
    need(all(isinstance(x,str) and re.fullmatch('[0-9a-f]{64}',x) for x in (vertex_ids_sha,face_ids_sha)),'Exact source NPY SHA256 identities')
    # Proper rotation from original Blender +Z-up to standard glTF +Y-up.
    position=vertices[:,[0,2,1]].copy();position[:,2]*=-1
    streams=[position.astype('<f4',copy=False).tobytes(),linear.astype('<f4',copy=False).tobytes(),faces.astype('<u2').tobytes()]
    binary=bytearray();views=[]
    for i,raw in enumerate(streams):
        views.append(dict(buffer=0,byteOffset=len(binary),byteLength=len(raw),target=34963 if i==2 else 34962))
        binary.extend(raw);binary.extend(b'\0'*(-len(binary)%4))
    document=dict(asset=dict(version='2.0',generator='Modly bounded CPU foliage colour proof'),scene=0,scenes=[dict(nodes=[0])],nodes=[dict(mesh=0,name='Authenticated foliage')],
        buffers=[dict(byteLength=len(binary))],bufferViews=views,
        accessors=[dict(bufferView=0,componentType=5126,count=len(vertices),type='VEC3',min=position.min(0).tolist(),max=position.max(0).tolist()),
                   dict(bufferView=1,componentType=5126,count=len(vertices),type='VEC3'),
                   dict(bufferView=2,componentType=5123,count=faces.size,type='SCALAR')],
        materials=[dict(name='Genuine learned vertex RGB',alphaMode='OPAQUE',doubleSided=True,pbrMetallicRoughness=dict(baseColorFactor=[1,1,1,1],metallicFactor=METALLIC,roughnessFactor=ROUGHNESS))],
        meshes=[dict(primitives=[dict(attributes=dict(POSITION=0,COLOR_0=1),indices=2,mode=4,material=0)],extras=dict(sourceVertexIDs=source_vertices.tolist(),sourceFaceIDs=source_faces.tolist(),sourceVertexIDs_SHA256=vertex_ids_sha,sourceFaceIDs_SHA256=face_ids_sha))])
    encoded=json.dumps(document,separators=(',',':'),allow_nan=False).encode('utf-8');encoded+=b' '*(-len(encoded)%4)
    payload=struct.pack('<III',0x46546c67,2,28+len(encoded)+len(binary))+struct.pack('<II',len(encoded),0x4e4f534a)+encoded+struct.pack('<II',len(binary),0x004e4942)+binary
    need(len(payload)<=2*1024**2,'Bounded single-patch GLB')
    return payload
