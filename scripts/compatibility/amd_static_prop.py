"""One-shot CPU static-prop export proof; no reconstruction or bake route."""
import sys,json,os,struct,subprocess,math,zlib
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent))
import amd_cgal_proof as owner
import amd_alpha_features_proof as prior
from amd_closure_proof import check
BASE=prior.BASE.parent/'amd-static-prop-export-20261031-v1'
BLENDER=Path('C:/Program Files/Blender Foundation/Blender 5.2/blender.exe')
BLENDER_SHA='284f4041f98e113f3dc10654a7193ffaaa9bfdfec8b87fa116620a48b5f6d4cb'
PROFILE=owner.ROOT/'docs/compatibility/amd-static-prop-profile.json'
VIEWS=('whole45','whole225','detail37','detail8')
RENDERS=tuple(f'{m}-{v}' for m in ('matte','glossy') for v in VIEWS)
STAGES=('authenticate','prepare','export','verify',*RENDERS,'sheet','preserve')
LIVE_EXECUTION_HOLD=True # Failed proof closed out; revised Blender code is offline-only.

def overlap(a,b):
 """Strict positive-area triangle overlap by exact dyadic integer SAT."""
 coords=[float(x).as_integer_ratio() for t in (a,b) for p in t for x in p]
 denominator=max(d for _,d in coords);values=[n*(denominator//d) for n,d in coords]
 triangles=[[(values[6*k+2*i],values[6*k+2*i+1]) for i in range(3)] for k in range(2)]
 for t in triangles:
  ar=(t[1][0]-t[0][0])*(t[2][1]-t[0][1])-(t[1][1]-t[0][1])*(t[2][0]-t[0][0])
  if not ar:raise ValueError('Degenerate UV triangle')
  if ar<0:t.reverse()
 for t,u in (triangles,triangles[::-1]):
  for i in range(3):
   x,y=t[i];xx,yy=t[(i+1)%3]
   if all((xx-x)*(v-y)-(yy-y)*(w-x)<=0 for w,v in u):return False
 return True

def corner_key(positions,uv):
 rows=[tuple(float(x) if x else 0. for x in (*a,*b)) for a,b in zip(positions,uv)]
 return min(tuple(rows[i:]+rows[:i]) for i in range(3))

def uv_check(np,uv,v,f):
 from amd_cpu_uv_bake import uv_metrics
 result=uv_metrics(np,uv);lo=uv.min(axis=1);hi=uv.max(axis=1);pairs=[];tested=0
 for i in range(len(uv)):
  for j in np.flatnonzero(np.all(hi[i]>lo[i+1:],axis=1)&np.all(hi[i+1:]>lo[i],axis=1))+i+1:
   tested+=1
   if overlap(uv[i],uv[j]):pairs.append([i,int(j)])
 if pairs:raise ValueError('Continuous UV overlap: '+str(pairs[:10]))
 cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]);area=np.linalg.norm(cross.astype('f8'),axis=1)/2
 t=uv.astype('f8');u=t[:,1]-t[:,0];w=t[:,2]-t[:,0];density=abs(u[:,0]*w[:,1]-u[:,1]*w[:,0])/2/area
 ratio=float(density.max()/density.min())
 if not math.isfinite(ratio) or ratio>1.1:raise ValueError('Unequal UV area density: '+str(ratio))
 # UV chart connectivity and exact discontinuity along genuine source edges.
 edge={};parent=list(range(len(f)));seams=0
 def root(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 for i,face in enumerate(f):
  for k in range(3):
   a=int(face[k]);b=int(face[(k+1)%3]);entry=(i,tuple(uv[i,k]),tuple(uv[i,(k+1)%3]))
   if a>b:a,b=b,a;entry=(i,entry[2],entry[1])
   if (a,b) in edge:
    old=edge[(a,b)]
    if old[1:]==entry[1:]:parent[root(i)]=root(old[0])
    else:seams+=1
   else:edge[(a,b)]=entry
 return dict(**result,continuous_overlap_pairs=0,pairs_possible=len(uv)*(len(uv)-1)//2,broadphase_pairs=tested,
             area_density_max_over_min=ratio,charts=len({root(i) for i in range(len(f))}),seam_edges=seams,
             coverage_fraction=result['sum_area'],coverage_method='Float64 triangle area sum after exact continuous nonoverlap; no texel approximation')

def png_pixels(path):
 data=path.read_bytes()
 if len(data)>32*1024**2 or data[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Bounded PNG')
 pos=8;compressed=b'';header=None;ended=False
 while pos<len(data):
  length,kind=struct.unpack_from('>I4s',data,pos);payload=data[pos+8:pos+8+length];end=pos+12+length
  if end>len(data) or zlib.crc32(kind+payload)&0xffffffff!=struct.unpack_from('>I',data,end-4)[0]:raise ValueError('PNG CRC')
  if kind==b'IHDR':header=struct.unpack('>IIBBBBB',payload)
  elif kind==b'IDAT':compressed+=payload
  elif kind==b'IEND':ended=True
  pos=end
 if not ended or not header:raise ValueError('Incomplete PNG')
 w,h,bits,color,compression,filtering,interlace=header
 if bits!=8 or color not in (2,6) or (compression,filtering,interlace)!=(0,0,0) or w*h>4096**2:raise ValueError('PNG schema')
 channels=3 if color==2 else 4;stride=w*channels;d=zlib.decompressobj();raw=d.decompress(compressed,h*(stride+1)+1)
 if len(raw)!=h*(stride+1) or not d.eof:raise ValueError('PNG pixels length')
 out=bytearray();previous=bytearray(stride)
 for y in range(h):
  kind=raw[y*(stride+1)];row=bytearray(raw[y*(stride+1)+1:(y+1)*(stride+1)])
  if kind>4:raise ValueError('PNG filter')
  for x in range(stride):
   a=row[x-channels] if x>=channels else 0;b=previous[x];c=previous[x-channels] if x>=channels else 0
   q=a+b-c;pa,pb,pc=abs(q-a),abs(q-b),abs(q-c);paeth=a if pa<=pb and pa<=pc else b if pb<=pc else c
   row[x]=(row[x]+(0,a,b,(a+b)//2,paeth)[kind])&255
  out+=row;previous=row
 return w,h,channels,bytes(out)

def glb(np,path):
 data=path.read_bytes()
 if len(data)>16*1024**2 or struct.unpack_from('<4sII',data)!=(b'glTF',2,len(data)):raise ValueError('GLB header')
 n,kind=struct.unpack_from('<I4s',data,12)
 if kind!=b'JSON' or n%4:raise ValueError('JSON chunk')
 from amd_zombie_d_compare import unique
 def reject_constant(x):raise ValueError('Nonfinite GLB JSON')
 doc=json.loads(data[20:20+n],object_pairs_hook=unique,parse_constant=reject_constant);start=20+n;size,kind=struct.unpack_from('<I4s',data,start)
 def extension_free(value):
  if isinstance(value,dict):
   if 'extensions' in value:raise ValueError('Unsupported nested extension')
   for item in value.values():extension_free(item)
  elif isinstance(value,list):
   for item in value:extension_free(item)
 extension_free(doc)
 if kind!=b'BIN\0' or start+8+size!=len(data):raise ValueError('BIN closure')
 blob=memoryview(data)[start+8:]
 if any(k in doc for k in ('extensionsUsed','extensionsRequired','skins','animations')):raise ValueError('Unsupported GLB features')
 if len(doc['buffers'])!=1 or 'uri' in doc['buffers'][0] or not size-3<=doc['buffers'][0]['byteLength']<=size:raise ValueError('Buffer identity')
 if doc.get('scene',0)!=0 or len(doc['scenes'])!=1 or len(doc['nodes'])!=1 or doc['scenes'][0]['nodes']!=[0]:raise ValueError('Scene graph')
 node=doc['nodes'][0]
 if node.get('mesh')!=0 or any(k in node for k in ('children','matrix','translation','rotation','scale')):raise ValueError('Node transform')
 if len(doc['meshes'])!=1 or len(doc['meshes'][0]['primitives'])!=1:raise ValueError('Mesh count')
 prim=doc['meshes'][0]['primitives'][0]
 if prim.get('mode',4)!=4 or set(prim['attributes'])!={'POSITION','NORMAL','TEXCOORD_0'} or 'targets' in prim:raise ValueError('Primitive schema')
 def accessor(index,width,types):
  a=doc['accessors'][index];bv=doc['bufferViews'][a['bufferView']]
  if a.get('sparse') or a.get('normalized') or a['componentType'] not in types or a['type']!= {1:'SCALAR',2:'VEC2',3:'VEC3'}[width] or bv.get('buffer',0)!=0:raise ValueError('Accessor schema')
  dtype=np.dtype({5126:'<f4',5125:'<u4',5123:'<u2'}[a['componentType']]);offset=bv.get('byteOffset',0);local=a.get('byteOffset',0);stride=bv.get('byteStride',width*dtype.itemsize)
  if a['count']<=0 or a['count']>10000 or local<0 or offset<0 or stride<width*dtype.itemsize or local+(a['count']-1)*stride+width*dtype.itemsize>bv['byteLength'] or offset+bv['byteLength']>len(blob):raise ValueError('Accessor/view bounds')
  return np.ndarray((a['count'],width),dtype,buffer=blob,offset=offset+local,strides=(stride,dtype.itemsize)).copy()
 v=accessor(prim['attributes']['POSITION'],3,{5126});normal=accessor(prim['attributes']['NORMAL'],3,{5126});uv=accessor(prim['attributes']['TEXCOORD_0'],2,{5126});f=accessor(prim['indices'],1,{5123,5125}).reshape(-1,3)
 if len(normal)!=len(v) or len(uv)!=len(v) or f.max()>=len(v) or not all(np.isfinite(a).all() for a in (v,normal,uv)):raise ValueError('Finite attribute correspondence')
 v=np.ascontiguousarray(v[:,[0,2,1]]);v[:,1]*=-1;normal=np.ascontiguousarray(normal[:,[0,2,1]]);normal[:,1]*=-1
 if len(doc['materials'])!=1 or prim['material']!=0:raise ValueError('Material identity')
 material=doc['materials'][0];pbr=material['pbrMetallicRoughness']
 validate_material(material)
 if material.get('alphaMode','OPAQUE')!='OPAQUE' or material.get('doubleSided',False) or any(k in material for k in ('normalTexture','occlusionTexture','emissiveTexture','extensions')) or pbr.get('metallicFactor',1)!=0 or abs(pbr['roughnessFactor']-.35)>1e-6:raise ValueError('Neutral test PBR factors')
 if len(doc.get('images',[]))!=1 or len(doc['textures'])!=1 or pbr['baseColorTexture'].get('texCoord',0)!=0 or pbr['baseColorTexture']['index']!=0 or doc['textures'][0]['source']!=0:raise ValueError('Texture references')
 image=doc['images'][0]
 sampler=doc['textures'][0].get('sampler')
 if sampler is not None:
  s=doc['samplers'][sampler]
  if s.get('wrapS',10497)!=10497 or s.get('wrapT',10497)!=10497 or s.get('magFilter',9729)!=9729 or s.get('minFilter',9987) not in (9729,9987):raise ValueError('Texture sampler')
 if 'uri' in image or image.get('mimeType')!='image/png':raise ValueError('Packed PNG required')
 bv=doc['bufferViews'][image['bufferView']];offset=bv.get('byteOffset',0)
 if offset<0 or offset+bv['byteLength']>len(blob):raise ValueError('Image bounds')
 encoded=bytes(blob[offset:offset+bv['byteLength']])
 return v,f,uv,normal,encoded,doc

def validate_material(material):
 pbr=material['pbrMetallicRoughness']
 if pbr.get('baseColorFactor',[1,1,1,1])!=[1,1,1,1] or material.get('emissiveFactor',[0,0,0])!=[0,0,0] or pbr.get('metallicFactor',1)!=0 or not math.isfinite(pbr.get('roughnessFactor',1)) or abs(pbr.get('roughnessFactor',1)-.35)>1e-6:raise ValueError('Neutral material factors')

def verify():
 import numpy as np
 from amd_cgal_official_check import mesh_counts
 from amd_cpu_uv_bake import geometry_hash
 folder=BASE/'asset';v=np.load(BASE/'inputs/final-v.npy');f=np.load(BASE/'inputs/final-f.npy');uv=np.load(folder/'uv.npy')
 stats=uv_check(np,uv,v,f);gv,gf,guv,gn,png,doc=glb(np,folder/'experimental-static-prop.glb')
 expecteduv=uv.copy();expecteduv[:,:,1]=np.float32(1)-uv[:,:,1]
 keys={corner_key(v[face],expecteduv[i]):i for i,face in enumerate(f)}
 if len(keys)!=len(f):raise ValueError('Ambiguous source triangle identity')
 matched=[]
 for face in gf:
  key=corner_key(gv[face],guv[face])
  if key not in keys:raise ValueError('GLB oriented geometry/UV correspondence')
  matched.append(keys[key])
 if sorted(matched)!=list(range(1000)):raise ValueError('GLB face bijection')
 with (folder/'embedded-neutral.png').open('xb') as stream:stream.write(png)
 w,h,c,pixels=png_pixels(folder/'embedded-neutral.png');sw,sh,sc,sourcepixels=png_pixels(folder/'neutral-test.png')
 rgba=np.frombuffer(pixels,'u1').reshape(h,w,c);rgb=np.frombuffer(sourcepixels,'u1').reshape(sh,sw,sc)
 if (w,h)!=(sw,sh) or not np.array_equal(rgba[:,:,:3],rgb[:,:,:3]) or (c==4 and not np.all(rgba[:,:,3]==255)):raise ValueError('Packed texture pixels changed')
 n=np.cross(gv[gf[:,1]]-gv[gf[:,0]],gv[gf[:,2]]-gv[gf[:,0]]);n=n/np.linalg.norm(n,axis=1)[:,None]
 dots=(gn[gf]*n[:,None,:]).sum(axis=2)
 if not np.allclose(np.linalg.norm(gn,axis=1),1,atol=1e-5) or dots.min()<.99999:raise ValueError('Flat normal readback')
 # Attribute seams duplicate vertices; identify every duplicate via exact original position.
 identity={tuple(x):i for i,x in enumerate(v)}
 if len(identity)!=502 or any(tuple(x) not in identity for x in gv):raise ValueError('Vertex identity')
 ff=np.array([[identity[tuple(x)] for x in gv[face]] for face in gf]);counts=mesh_counts(v.tolist(),ff.tolist())
 exported_uv=uv_check(np,guv[gf],v,ff)
 if any(counts[k] for k in ('boundary_edges','nonmanifold_edges','winding_conflicts','nonmanifold_vertex_links','unused_vertices')) or counts['face_bearing_vertex_components']!=1:raise ValueError('Readback topology')
 save(BASE/'verification.json',dict(classification='PASS_STATIC_PROP_UV_EXPORT_GEOMETRY',UV=stats,serialized_UV=exported_uv,export_attribute_vertices=len(gv),geometric_vertices=502,triangles=len(gf),seam_duplicates=len(gv)-502,
  oriented_triangle_corner_UV_bijection=True,flat_normal_minimum_dot=float(dots.min()),counts=counts,packed_texture_pixels_exact=True,geometry_hash=geometry_hash(np,v,f),GLB_sha256=owner.sha(folder/'experimental-static-prop.glb'),
  research_result='PERMANENT_NO_GO',appearance='USER_APPROVAL_PENDING',normal_bake='SKIPPED_UNQUALIFIED_SOURCE',Sbox='NOT_TESTED',game_ready=False))

def sheet():
 import numpy as np
 from amd_mesh_aware_result import png
 out=BASE/'review';out.mkdir(exist_ok=False)
 html=['<!doctype html><meta charset="utf-8"><title>LOWER-DETAIL STATIC PROP - appearance review pending</title><style>body{background:#ddd;font:16px sans-serif}img{max-width:100%}table{width:100%;table-layout:fixed}td{text-align:center}</style><h1>LOWER-DETAIL STATIC PROP — experimental geometry/export proof</h1><p>Original high-fidelity benchmark: permanently NO-GO. User appearance approval pending. No AI texture or normal bake. All comparison panels: original AI / raw wrap / final 1000 triangles, with flat normals and identical CPU lighting/camera. Close views deliberately magnify the accepted microfacet loss.</p>']
 for material in ('matte','glossy'):
  frames=[]
  for view in VIEWS:
   success(f'{material}-{view}');panels=[]
   for name in ('source','raw','final'):
    path=BASE/f'{material}-{view}'/(name+'.png');w,h,c,data=png_pixels(path)
    if (w,h)!=(512,512):raise ValueError('Render dimensions')
    panels.append(np.frombuffer(data,'u1').reshape(h,w,c)[::2,::2,:3])
   frames.append(np.concatenate(panels,axis=1))
  png(out/(material+'-sheet.png'),np,np.concatenate(frames,axis=0))
  html.append(f'<h2>{material}</h2><p>Columns: source / raw wrap / final. Rows: whole 45°, whole 225°, microbevel region37, isolated feature region8.</p><img src="{material}-sheet.png">')
 qualified=(BASE/'verification.json').exists()
 if qualified:html+=['<h2>Export readback and checker</h2><p>Neutral TEST PBR; not finished AI texture. UV checker is diagnostic only.</p><img width="512" src="../asset/reimport-neutral.png"><img width="512" src="../asset/checker-preview.png"><p><a href="../asset/experimental-static-prop.glb">Experimental GLB</a></p>']
 else:html+=['<h2>EXPORT NO-GO</h2><p>UV area-density validation failed; the numerical ratio was not preserved. The required maximum finite ratio was1.1. No export, checker or normal bake was performed. These are material comparisons of the unchanged saved geometry only.</p>']
 with (out/'index.html').open('x',encoding='utf8') as stream:stream.write('\n'.join(html))
 result=json.loads((BASE/'verification.json').read_text()) if qualified else dict(classification='NO_GO_EXPORT_UV_SCALE',UV='FAIL_AREA_DENSITY_VALIDATION_RATIO_NOT_PERSISTED',UV_layout='NOT_QUALIFIED',GLB='NOT_EXPORTED',geometry_export_readback='NOT_EXECUTED',game_ready=False)
 result.update(profile='LOWER-DETAIL STATIC PROP',appearance_review='PENDING_USER_INSPECTION',raw_neural_result='NO_GO',research_contract='PERMANENT_NO_GO',generated_geometry=False,source_normal_bake='SKIPPED_UNQUALIFIED_SOURCE_WINDING_AND_RAY_PROVENANCE',Sbox='NOT_TESTED_NO_QUALIFIED_IMPORT_ROUTE',remaining=['qualified UV layout with scale consistency','appearance/material-distance approval','production UV padding/mip evaluation','finished PBR authoring/baking','S&box import/units/collision/LOD qualification','asset-class generalisation','licence review before binary integration'],assets={str(path.relative_to(BASE)):dict(bytes=path.stat().st_size,sha256=owner.sha(path)) for path in [*(BASE/'asset').iterdir(),*out.iterdir()] if path.is_file()})
 save(BASE/'asset-manifest.json',result)

save=owner.save
def success(s):
 if json.loads((BASE/(s+'-run/host-result.json')).read_text())['classification']!='PASS':raise ValueError('Failed prerequisite '+s)
def appearance_admission():
 """A failed export never becomes PASS; only unaffected saved-mesh views continue."""
 success('prepare')
 if (BASE/'verification.json').exists():success('verify');return
 r=json.loads((BASE/'export-run/host-result.json').read_text());err=(BASE/'export-run/stderr.txt').read_text()
 if r['classification']!='FAIL' or r.get('error') or r.get('worker_returncode')!=1 or r['cleanup']['forced_job_cleanup'] or not r['cleanup']['owned_tree_exit_verified'] or err.splitlines().count('ValueError: Unequal UV area density')!=1 or 'stats=p.uv_check(np,uv,v,f)' not in err or any((BASE/name).exists() for name in ('asset/experimental-static-prop.glb','asset/uv.npy','asset/blender.json','verification.json')):raise ValueError('Only preserved UV-density failure permits appearance-only views')
def reviewed(stage):
 r=json.loads((BASE/('appearance-only-review.json' if stage in (*RENDERS,'sheet') else 'execution-review.json')).read_text())
 if r['classification']!='APPROVED' or stage not in r.get('allowed_stages',STAGES):raise ValueError('Independent review required')
 for name,h in r['files'].items():
  if owner.sha(Path(name))!=h:raise ValueError('Reviewed dependency changed')

def authenticate(preserve=False):
 a=json.loads((prior.BASE/'authentication.json').read_text())
 maps=[(prior.BASE,json.loads((prior.BASE/'evidence-manifest.json').read_text())),(prior.PRIOR,a['Phase18_files']),(prior.old.ROOT14,a['Phase14_files'])]
 for base,rows in maps:
  for name,row in rows.items():check(base/name,row)
 for name,row in a['original_snapshot'].items():check(Path(name),row)
 for name,row in a['protected_records'].items():
  s=Path(name).stat()
  if (s.st_size,s.st_mtime_ns,s.st_ino)!=(row['bytes'],row['mtime_ns'],row['file_id']):raise ValueError('Protected identity')
 if owner.sha(owner.ROOT/'docs/compatibility/amd-ai-hard-surface-criteria.json')!='397bf6c2dc8908e8a3b1a02ba169d6ec6ab5bc2aaf11ee96a4cb84e81591aad0' or owner.sha(BLENDER)!=BLENDER_SHA:raise ValueError('Frozen criteria/Blender changed')
 # Reuse the hash-qualified isolated CPU NumPy; no installation or environment writes.
 row=json.loads((prior.BASE/'dependency-numpy.json').read_text())
 for r in row['files'].values():check(Path(r['path']),r)
 save(BASE/('preservation.json' if preserve else 'authentication.json'),dict(classification='PASS',Phase19_files=len(maps[0][1]),Phase18_files=len(maps[1][1]),Phase14_files=len(maps[2][1]),original_sources=len(a['original_snapshot']),protected_identities=len(a['protected_records']),Blender_sha256=BLENDER_SHA,profile_sha256=owner.sha(PROFILE)))

def prepare():
 import numpy as np
 from amd_cgal_official_check import read_off
 folder=BASE/'inputs';folder.mkdir(exist_ok=False);receipt=json.loads((prior.old.ROOT14/'input/receipt.json').read_text());center=np.array(receipt['center']);scale=receipt['scale'];D=receipt['diagonal']
 source=owner.ROOT.parent/'artifacts/amd-closed-solid-raw-recovery-v1/mesh'
 quant={};bindings={}
 for name in ('source','raw','final'):
  if name=='source':
   paths=[source/'vertices.npy',source/'faces.npy'];a=json.loads((prior.BASE/'authentication.json').read_text())
   for path in paths:check(path,a['original_snapshot'][str(path)])
   v=np.load(paths[0]);f=np.load(paths[1])
  else:
   paths=[prior.PRIOR/('raw-wrap.off' if name=='raw' else 'simplified.off')];manifest=json.loads((prior.PRIOR/'evidence-manifest.json').read_text());check(paths[0],manifest[paths[0].name])
   v,f=read_off(paths[0]);v=np.asarray(v)/scale+center;f=np.asarray(f)
  expected={'source':(1023995,2047210),'raw':(1406,2808),'final':(502,1000)}[name]
  if v.shape!=(expected[0],3) or f.shape!=(expected[1],3) or not np.isfinite(v).all() or f.min()<0 or f.max()>=len(v):raise ValueError('Copy schema')
  vv=np.asarray(v,dtype='<f4');ff=np.asarray(f,dtype='<u4');err=float(np.linalg.norm(v-vv,axis=1).max()/D)
  if err>1e-7:raise ValueError('Float32 conversion bound')
  quant[name]=err
  bindings[name]={str(path):owner.sha(path) for path in paths}
  for suffix,data in [('v',vv),('f',ff)]:
   with (folder/f'{name}-{suffix}.npy').open('xb') as out:np.save(out,data,allow_pickle=False)
 cameras={}
 for az in (45,225):
  a=math.radians(az);e=math.radians(20);view=np.array([math.cos(a)*math.cos(e),math.sin(a)*math.cos(e),math.sin(e)])
  cameras[f'whole{az}']=dict(target=center.tolist(),location=(center+4*D*view).tolist(),span=1.15*D,azimuth=az,elevation=20)
 oldcams=json.loads((prior.BASE/'detail-review/cameras.json').read_text())
 for region in (37,8):
  c=next(x for x in oldcams if x['region']==region);target=np.array(c['center'])/scale+center;view=np.array(c['view'])
  cameras[f'detail{region}']=dict(target=target.tolist(),location=(target+4*D*view).tolist(),span=c['span']/scale,source_camera_region=region)
 save(BASE/'inputs.json',dict(D=D,center=center.tolist(),cameras=cameras,quantization_max_Euclidean_over_D=quant,bindings=bindings,units='1 exported unit = 1 metre convention; no physical-size inference',normal_bake='SKIPPED_UNQUALIFIED_SOURCE',files={p.name:owner.sha(p) for p in folder.iterdir()}))

def audit(stage):
 from amd_zombie_d_compare import audit_policy,plain
 base=audit_policy(BASE);scratch=plain(BASE/'scratch'/stage);identity=scratch.stat().st_ino;created=set()
 def guard(event,args):
  if event in ('open','os.mkdir','os.remove','os.rmdir') and scratch.stat().st_ino!=identity:raise RuntimeError('Owned scratch identity changed')
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   target=plain(os.fsdecode(args[0]));writing=any(c in str(args[1]) for c in 'wax+') or args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT)
   if writing and target.is_relative_to(scratch) and not target.exists():created.add(target)
  if event=='os.mkdir':
   target=plain(os.fsdecode(args[0]))
   if target.is_relative_to(scratch) and not target.exists():created.add(target)
  if event in ('os.remove','os.rmdir'):
   if len(args)>1 and args[1] not in (None,-1):raise RuntimeError('dir_fd mutation denied')
   target=plain(os.fsdecode(args[0]));target.relative_to(scratch)
   if target==scratch or target not in created or not target.exists() or (target.is_file() and target.stat().st_nlink!=1):raise RuntimeError('Unowned scratch deletion')
   return
  base(event,args)
 return guard

def worker(stage):
 if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO/EOF required')
 sys.addaudithook(audit(stage))
 if stage in ('authenticate','preserve'):authenticate(stage=='preserve');return
 reviewed(stage);success('authenticate');sys.path.insert(0,str(owner.CPU_SITE))
 if stage=='prepare':prepare()
 elif stage=='verify':success('export');verify()
 elif stage=='sheet':appearance_admission();sheet()
 else:raise ValueError('No Python operation')

def main(argv=None):
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=STAGES);p.add_argument('--worker',choices=STAGES);a=p.parse_args(argv)
 if a.worker:worker(a.worker);return 0
 if not a.stage:print('INERT: isolated static prop CPU proof');return 0
 if LIVE_EXECUTION_HOLD and a.stage not in ('authenticate','preserve'):raise RuntimeError('Live proof HOLD; new approval and fresh paths required')
 if a.stage not in ('authenticate','preserve'):reviewed(a.stage);success('authenticate')
 if a.stage=='export' or a.stage in RENDERS:success('prepare')
 if a.stage in RENDERS:appearance_admission()
 scratch=BASE/'scratch'/a.stage;scratch.mkdir(parents=True,exist_ok=False)
 owner.BASE=BASE;owner.__file__=__file__;launch=subprocess.Popen
 def routed(command,**kwargs):
  expected=[sys._base_executable,'-B','-I','-S',str(Path(__file__).resolve()),'--worker',a.stage]
  if command!=expected:raise ValueError('Changed owned worker route')
  if a.stage=='export' or a.stage in RENDERS:command=[str(BLENDER),'--background','--factory-startup','--disable-autoexec','--threads','4','--python-exit-code','1','--python',str(Path(__file__).with_name('amd_static_prop_blender.py')),'--',a.stage]
  kwargs['env']=dict(kwargs['env'],TEMP=str(scratch),TMP=str(scratch));return launch(command,**kwargs)
 owner.subprocess=SimpleNamespace(Popen=routed,PIPE=subprocess.PIPE)
 return owner.supervise(a.stage)
if __name__=='__main__':raise SystemExit(main())
