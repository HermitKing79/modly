"""Run-derived neural artifact -> original CPU dual-grid mesh; no model imports."""
import argparse,hashlib,json,struct,time
from pathlib import Path
import numpy as np
import amd_flexible_dual_grid_cpu as converter

MAX_ROWS=1_842_968
MAX_INITIAL_ROWS=4430
MAX_BYTES=256*1024**2
_CONTRACT_API=None

def _contract_api():
 global _CONTRACT_API
 if _CONTRACT_API is None:
  import importlib.util,sys
  path=Path(__file__).resolve().parents[2]/'experimental/extensions/trellis2-amd/inference_decoder/run_contract.py'
  name='_modly_generation_mesh_run_contract'
  spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
  previous=sys.modules.get(name);sys.modules[name]=module
  try:spec.loader.exec_module(module)
  except BaseException:
   if previous is None:sys.modules.pop(name,None)
   else:sys.modules[name]=previous
   raise
  _CONTRACT_API=module
 return _CONTRACT_API
def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def _object(pairs):
 result={}
 for key,value in pairs:
  if key in result:raise ValueError('Duplicate artifact header key')
  result[key]=value
 return result
def read_neural(path,expected_sha256,*,contract=None):
 opened=[]
 try:return _read_neural(path,expected_sha256,opened,contract=contract)
 except BaseException:
  for array in opened:array._mmap.close()
  raise
def _read_neural(path,expected_sha256,opened,*,contract=None):
 policy=_contract_api()
 if contract is not None:contract=policy.parse(contract.document())
 limit=policy.final_limit(contract)
 path=Path(path)
 if path.is_symlink() or not path.is_file() or path.stat().st_nlink!=1:raise ValueError('Plain single-link neural artifact required')
 if not 8<path.stat().st_size<=MAX_BYTES or digest(path)!=expected_sha256:raise ValueError('Neural artifact size/hash differs')
 before=path.stat()
 with path.open('rb') as f:
  length=struct.unpack('<Q',f.read(8))[0]
  if not 2<=length<=65536:raise ValueError('Bounded safetensors header required')
  header=json.loads(f.read(length),object_pairs_hook=_object)
 names={'features','coordinates'}|{f'subdivision_{i}_{suffix}' for i in range(4) for suffix in ('coordinates','logits')}
 if set(header)-{'__metadata__'}!=names:raise ValueError('Complete original ten tensors required')
 if contract is not None and contract.final_support_profile is not None:
  try:
   provenance=json.loads(header['__metadata__']['provenance'],object_pairs_hook=_object)
   policy.artifact_binding(provenance,contract)
   policy.support_counts(header['features']['shape'][0],
                         [header[f'subdivision_{i}_coordinates']['shape'][0] for i in range(4)],contract)
  except (KeyError,TypeError,IndexError) as exc:raise ValueError('Bound profile artifact provenance required') from exc
 payload=path.stat().st_size-8-length;arrays={};ranges=[]
 for name in names:
  row=header[name]
  if set(row)!={'dtype','shape','data_offsets'}:raise ValueError('Unexpected tensor descriptor')
  dtype={'F32':'<f4','I32':'<i4','BF16':'<u2'}.get(row['dtype'])
  shape=row['shape'];a,b=row['data_offsets']
  if dtype is None or not isinstance(shape,list) or len(shape)!=2 or any(type(n) is not int or not 0<n<=limit for n in shape):raise ValueError('Invalid tensor shape/dtype')
  if any(type(n) is not int for n in (a,b)) or not 0<=a<b<=payload or b-a!=np.prod(shape)*np.dtype(dtype).itemsize:raise ValueError('Invalid tensor payload layout')
  ranges.append((a,b));arrays[name]=np.memmap(path,mode='r',offset=8+length+a,dtype=dtype,shape=tuple(shape));opened.append(arrays[name])
 previous=0
 for a,b in sorted(ranges):
  if a!=previous:raise ValueError('Overlapping or incomplete tensor payload')
  previous=b
 if previous!=payload:raise ValueError('Unexpected trailing tensor payload')
 features,coords=arrays['features'],arrays['coordinates'];n=len(features)
 if features.dtype!=np.float32 or features.shape!=(n,7) or coords.dtype!=np.int32 or coords.shape!=(n,4) or not np.isfinite(features).all():raise ValueError('Finite original F32[N,7]/I32[N,4] required')
 def coordinate(c,resolution):
  if c.dtype!=np.int32 or c.shape[1]!=4 or np.any(c[:,0]!=0) or np.any(c[:,1:]<0) or np.any(c[:,1:]>=resolution):raise ValueError('Sparse coordinate structure differs')
  keys=(c[:,1].astype('i8')*resolution+c[:,2])*resolution+c[:,3]
  if np.unique(keys).size!=len(keys):raise ValueError('Duplicate sparse coordinate')
 coordinate(coords,512);counts=[]
 for i in range(4):
  parents,mask=arrays[f'subdivision_{i}_coordinates'],arrays[f'subdivision_{i}_logits']
  if len(parents)>(MAX_INITIAL_ROWS if i==0 else 1_000_000):raise ValueError('Subdivision exceeds qualified parent envelope')
  coordinate(parents,32*2**i)
  if header[f'subdivision_{i}_logits']['dtype']!='BF16' or mask.shape!=(len(parents),8) or np.any((mask&0x7f80)==0x7f80):raise ValueError('Finite original BF16 subdivision logits required')
  # BF16 sign/exponent bits implement the source's strict logits>0 exactly,
  # including rejection of both signed zeros; preserve the raw logits intact.
  selected=((mask&0x8000)==0)&((mask&0x7fff)!=0)
  parent_rows,child_ids=np.nonzero(selected);children=len(parent_rows)
  expected=len(arrays[f'subdivision_{i+1}_coordinates']) if i<3 else n
  if children!=expected:raise ValueError('Learned subdivision linkage differs')
  child_coords=parents[parent_rows].copy();child_coords[:,1:]*=2
  for axis in range(3):child_coords[:,axis+1]+=((child_ids//(2**axis))%2).astype('i4')
  next_coords=arrays[f'subdivision_{i+1}_coordinates'] if i<3 else coords
  if not np.array_equal(child_coords,next_coords):raise ValueError('Original C2S child coordinates/order differ')
  counts.append(children)
 policy.support_counts(n,[len(arrays[f'subdivision_{i}_coordinates']) for i in range(4)],contract)
 if path.stat().st_size!=before.st_size or path.stat().st_mtime_ns!=before.st_mtime_ns:raise ValueError('Neural artifact changed while reading')
 return arrays,counts

def extract(source,sha256,output,*,contract=None):
 began=time.monotonic();arrays,subdivisions=read_neural(source,sha256,contract=contract)
 output=Path(output);output.mkdir(exist_ok=False)
 coords=np.ascontiguousarray(arrays['coordinates'][:,1:]);features=arrays['features']
 dual,flags,weight=converter.transform_features(features)
 vertices=converter.vertices(coords,dual);faces,conversion=converter.faces(coords,flags,weight)
 if not len(faces):raise ValueError('Genuine neural output produced an empty surface')
 zero=0
 for start in range(0,len(faces),32768):
  tri=vertices[faces[start:start+32768]].astype('f8')
  zero+=int(np.count_nonzero(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)==0))
 if zero:raise ValueError('Genuine neural output contains degenerate surface triangles; no automatic repair')
 path=output/'mesh.ply';converter.write_ply(path,vertices,faces)
 restored_v,restored_f=converter.read_ply(path)
 if not np.array_equal(restored_v,vertices) or not np.array_equal(restored_f,faces):raise ValueError('CPU mesh readback differs')
 for name,array in (('vertices.npy',vertices),('faces.npy',faces)):
  with (output/name).open('xb') as f:np.save(f,array,allow_pickle=False)
 result={'input_sha256':sha256,'vertices':len(vertices),'triangles':len(faces),'learned_subdivisions':subdivisions,
  'PLY_readback_exact':True,'zero_area_triangles':zero,'bounds':{'min':vertices.min(axis=0).tolist(),'max':vertices.max(axis=0).tolist()},
  'conversion':conversion,'artifacts':{n:{'sha256':digest(output/n),'bytes':(output/n).stat().st_size} for n in ('mesh.ply','vertices.npy','faces.npy')},
  'elapsed_seconds':time.monotonic()-began,'topology_repair':False,'native_extraction':False,'material_decoded':False}
 with (output/'mesh.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 return result
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--sha256',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--run-contract',type=Path)
 args=parser.parse_args();contract=_contract_api().load(args.run_contract) if args.run_contract else None
 print(json.dumps(extract(args.source,args.sha256,args.output,contract=contract)))
