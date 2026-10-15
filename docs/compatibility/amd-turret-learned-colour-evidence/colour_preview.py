"""CPU direct-row RGB preview only. No learned data or geometry is modified."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
import numpy as np
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
MESH=ROOT.parent/'artifacts/amd-turret-cpu-mesh-ad07f96/turret-full.ply'
NEURAL=ROOT.parent/'artifacts/amd-wsl-turret-writer-ready-4870856/turret-neural-output.safetensors'
MATERIAL=Path('C:/Users/wattz/Documents/ModlyArtifacts/amd-turret-texture-owner-067f668/field/texture-field.safetensors')
OUTPUT=ROOT.parent/'artifacts/amd-turret-learned-colour-fa3a98c'
MESH_SHA='89549a36c6acc6f884174a2097ad7e346980d565c999117ea29f89cda9baad1a'
MATERIAL_SHA='7b24c946765d8e5de9718dbc358ce35141a56d836aa54fe7f8945cb1acbf7a80'
NEURAL_SHA='dc4667495ac4f8fd1457adca99563d83ee27c71fce0da2a6eb6698a8a1bd6e22'
COORD_SHA='c87fa21d6f863aced647b48e0bb2edebe32523d36c742e350d39ccfea3fc0934'
N=1842968;F=4012360;BATCH=32768
COLOURED_DTYPE=np.dtype([('xyz','<f4',(3,)),('rgb','u1',(3,))])
FACE_DTYPE=np.dtype([('count','u1'),('indices','<i4',(3,))])
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 with Path(p).open('x',encoding='utf-8') as f:
  json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def quantize_rgb(x):
 if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():raise ValueError('Finite F32 RGB[N,3] required')
 return np.rint(np.clip(x,np.float32(0),np.float32(1))*np.float32(255)).astype(np.uint8)
def ply_header(p,coloured=False):
 with Path(p).open('rb') as f:
  lines=[]
  for _ in range(32):
   line=f.readline(512)
   if not line or len(line)>511:raise ValueError('Bounded PLY header required')
   lines.append(line)
   if line==b'end_header\n':break
  if lines[-1]!=b'end_header\n':raise ValueError('PLY header missing terminator')
  offset=f.tell()
 h=b''.join(lines);ls=h.decode('ascii').splitlines()
 if ls[:2]!=['ply','format binary_little_endian 1.0']:raise ValueError('Little endian binary PLY required')
 nv=int(next(s.split()[2] for s in ls if s.startswith('element vertex ')))
 nf=int(next(s.split()[2] for s in ls if s.startswith('element face ')))
 props=[s for s in ls if s.startswith('property ')]
 expected=['property float x','property float y','property float z']+(['property uchar red','property uchar green','property uchar blue'] if coloured else [])+['property list uchar int vertex_indices']
 if props!=expected or not 0<nv<=N or not 0<nf<=F:raise ValueError('Unexpected PLY schema')
 if Path(p).stat().st_size!=offset+nv*(15 if coloured else 12)+nf*13:raise ValueError('PLY size differs')
 return h,offset,nv,nf
def append_colours(source,target,rgb):
 header,offset,nv,nf=ply_header(source)
 if rgb.dtype!=np.uint8 or rgb.shape!=(nv,3):raise ValueError('Exact RGB u8[N,3] required')
 colour_props=b'property uchar red\nproperty uchar green\nproperty uchar blue\n'
 header=header.replace(b'element face ',colour_props+b'element face ',1)
 vertices=np.memmap(source,mode='r',dtype='<f4',offset=offset,shape=(nv,3))
 began=time.monotonic()
 with Path(target).open('xb') as dst:
  dst.write(header)
  for start in range(0,nv,BATCH):
   if time.monotonic()-began>60:raise TimeoutError('60-second colour write deadline')
   end=min(start+BATCH,nv);rec=np.empty(end-start,dtype=COLOURED_DTYPE)
   rec['xyz']=vertices[start:end];rec['rgb']=rgb[start:end];dst.write(rec.tobytes())
  with Path(source).open('rb') as src:
   src.seek(offset+nv*12);shutil.copyfileobj(src,dst,1024**2)
  dst.flush();os.fsync(dst.fileno())
 del vertices
 return target
def tensor(p,name,dtype,shape):
 with Path(p).open('rb') as f:
  size=int.from_bytes(f.read(8),'little')
  if not 0<size<=65536:raise ValueError('Bounded safetensors header required')
  h=json.loads(f.read(size))
 row=h[name];a,b=row['data_offsets']
 if row['dtype']!=dtype or row['shape']!=list(shape) or not 0<=a<b<=Path(p).stat().st_size-8-size or b-a!=4*int(np.prod(shape)):raise ValueError('Tensor layout mismatch')
 return np.memmap(p,mode='r',dtype={'F32':'<f4','I32':'<i4'}[dtype],offset=8+size+a,shape=shape)
def main():
 began=time.monotonic()
 def no_ai(event,args):
  if event=='import' and str(args[0]).split('.')[0] in {'torch','triton','flex_gemm','o_voxel','trellis2','cupy'}:raise RuntimeError('CPU preview prohibits AI/GPU modules')
  if event in ('subprocess.Popen','os.system','socket.connect','socket.bind'):raise RuntimeError('CPU colour conversion prohibits child/network execution')
 sys.addaudithook(no_ai)
 original={str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':digest(p)} for p in (MESH,MATERIAL,NEURAL)}
 for p,sha in ((MESH,MESH_SHA),(MATERIAL,MATERIAL_SHA),(NEURAL,NEURAL_SHA)):
  if original[str(p)]['sha256']!=sha:raise ValueError('Pinned source identity differs: '+str(p))
 proof=json.loads((E/'row-correspondence-review.json').read_text())
 if proof.get('classification')!='PASS':raise ValueError('Independent row correspondence proof required')
 _,offset,nv,nf=ply_header(MESH)
 if (nv,nf)!=(N,F):raise ValueError('Full original geometry counts required')
 verts=np.memmap(MESH,mode='r',dtype='<f4',offset=offset,shape=(N,3));faces=np.memmap(MESH,mode='r',dtype=FACE_DTYPE,offset=offset+N*12,shape=(F,))
 expected_v=np.load(MESH.parent/'vertices.npy',mmap_mode='r',allow_pickle=False);expected_f=np.load(MESH.parent/'faces.npy',mmap_mode='r',allow_pickle=False)
 if verts.tobytes()!=expected_v.tobytes() or faces['indices'].tobytes()!=expected_f.tobytes() or np.any(faces['count']!=3):raise ValueError('Original PLY/NPY payloads differ')
 coords=tensor(MATERIAL,'coordinates','I32',(N,4));geometry_coords=tensor(NEURAL,'coordinates','I32',(N,4));features=tensor(MATERIAL,'features','F32',(N,6))
 if hashlib.sha256(coords.tobytes()).hexdigest()!=COORD_SHA or coords.tobytes()!=geometry_coords.tobytes():raise ValueError('Geometry/material row coordinates differ')
 if not np.isfinite(features).all():raise ValueError('Nonfinite learned features')
 metadata=json.loads(MATERIAL.with_name('texture-field-metadata.json').read_text())
 if metadata['channels']!=['red','green','blue','metallic','roughness','alpha'] or metadata['artifact_sha256']!=MATERIAL_SHA:raise ValueError('Material channels/provenance differ')
 if OUTPUT.exists() or OUTPUT.is_symlink():raise FileExistsError('Fresh colour output required')
 if shutil.disk_usage(OUTPUT.parent).free<2*1024**3:raise RuntimeError('2 GiB disk admission failed')
 OUTPUT.mkdir(exist_ok=False)
 rgb=quantize_rgb(features[:,:3]);ply=append_colours(MESH,OUTPUT/'turret-learned-colour.ply',rgb)
 _,new_offset,new_n,new_f=ply_header(ply,True);rec=np.memmap(ply,mode='r',dtype=COLOURED_DTYPE,offset=new_offset,shape=(N,));new_faces=np.memmap(ply,mode='r',dtype=FACE_DTYPE,offset=new_offset+N*15,shape=(F,))
 if (new_n,new_f)!=(N,F) or rec['xyz'].tobytes()!=verts.tobytes() or new_faces.tobytes()!=faces.tobytes() or rec['rgb'].tobytes()!=rgb.tobytes():raise ValueError('Derived PLY geometry/colour readback differs')
 for p in (MESH,MATERIAL,NEURAL):
  row=original[str(p)]
  if digest(p)!=row['sha256'] or p.stat().st_size!=row['bytes'] or p.stat().st_mtime_ns!=row['mtime_ns']:raise ValueError('Source artifact changed')
 result={'classification':'PASS','mode':'direct-row learned-colour preview; not trilinear PBR bake','source_mesh':{'path':str(MESH),'sha256':MESH_SHA},'material':{'path':str(MATERIAL),'sha256':MATERIAL_SHA},'neural_sha256':NEURAL_SHA,'source_artifacts':original,'vertex_count':N,'face_count':F,'coloured_ply':{'path':str(ply),'bytes':ply.stat().st_size,'sha256':digest(ply)},'RGB_quantization':'rint(clip(F32 RGB,0,1)*F32(255)).astype(u1)','RGB_payload_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'vertex_payload_sha256':hashlib.sha256(verts.tobytes()).hexdigest(),'face_payload_sha256':hashlib.sha256(faces['indices'].tobytes()).hexdigest(),'face_record_payload_sha256':hashlib.sha256(faces.tobytes()).hexdigest(),'coordinate_payload_sha256':COORD_SHA,'RGB_clamped_values':int(((features[:,:3]<0)|(features[:,:3]>1)).sum()),'geometry_unchanged':True,'all_rows_coloured':True,'elapsed_seconds':time.monotonic()-began,'NumPy_version':np.__version__,'AI_or_WSL_execution':False}
 save(OUTPUT/'conversion.json',result);save(E/'conversion.json',result)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
