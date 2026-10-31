"""Read-only interpretation and matched CPU magnifications of saved outputs."""
import json,math,struct,zlib
from pathlib import Path
from collections import Counter,defaultdict
import amd_alpha_features_proof as run
from amd_cgal_proof import save,sha

def source(np):
 p=run.old.ROOT14/'input/mesh.bin'
 return (np.memmap(p,mode='r',offset=16,dtype='<f8',shape=(1023995,3)),
         np.memmap(p,mode='r',offset=16+1023995*24,dtype='<u4',shape=(2047210,3)))

def records():
 with (run.BASE/'census/affected-faces.jsonl').open() as f:return [json.loads(x) for x in f]

def geometry(np,tri):
 n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(n,axis=1)
 if np.any(length<=0) or not np.all(np.isfinite(length)):raise ValueError('Invalid triangle')
 return n/length[:,None],length/20000

def validate_index(np,outcomes,rows,summary):
 if len(outcomes)!=summary['tracked_faces'] or len(np.unique(outcomes['face']))!=len(outcomes):raise ValueError('Outcome identity')
 if np.any(outcomes['face']>=summary['source_faces']) or np.any((outcomes['region']<0)|(outcomes['region']>=51)) or np.any((outcomes['raw']>7)|(outcomes['final']>7)):raise ValueError('Outcome range')
 affected=outcomes[(outcomes['raw']|outcomes['final'])!=0];mapping={int(r['face']):r for r in affected}
 if len(rows)!=len(mapping) or len({r['face'] for r in rows})!=len(rows) or {r['face'] for r in rows}!=set(mapping):raise ValueError('Affected union differs')
 for r in rows:
  m=mapping[r['face']]
  if (r['region'],r['raw_mask'],r['final_mask'])!=(m['region'],m['raw'],m['final']):raise ValueError('Affected mask correspondence')
  if sorted(i['corner'] for i in r['incidences'])!=[k for k in range(3) if (int(m['raw'])|int(m['final']))&(1<<k)]:raise ValueError('Incidence union differs')
 counts=np.zeros(4,dtype='i8');patches=np.zeros((51,4),dtype='i8')
 for k in range(3):
  code=((outcomes['raw']>>k)&1)+2*((outcomes['final']>>k)&1)
  counts+=np.bincount(code,minlength=4);np.add.at(patches,(outcomes['region'],code),1)
 if counts.tolist()!=summary['transitions'] or patches.tolist()!=summary['patch_transitions']:raise ValueError('Transition counts differ')

def analyse():
 import numpy as np
 from amd_cgal_official_check import read_off
 from amd_mesh_aware_result import point_triangle_distances
 rows=records();summary=json.loads((run.BASE/'census/summary.json').read_text());v,f=source(np)
 outcomes=np.fromfile(run.BASE/'census/tracked-face-outcomes.bin',offset=8,dtype=[('face','<u4'),('region','<i4'),('raw','u1'),('final','u1')])
 with (run.BASE/'census/tracked-face-outcomes.bin').open('rb') as stream:
  if int.from_bytes(stream.read(8),'little')!=1925348:raise ValueError('Outcome header')
 validate_index(np,outcomes,rows,summary)
 candidates={}
 for label,name in [('raw','raw-wrap.off'),('final','simplified.off')]:
  a,b=read_off(run.PRIOR/name);tri=np.asarray(a)[np.asarray(b)];n,_=geometry(np,tri);candidates[label]=(tri,n)
 checked=0;failareas=defaultdict(float);facecounts=Counter();patchrows=defaultdict(list);details=[]
 for r in rows:
  fid=r['face'];tri=v[f[fid]];normal,area=geometry(np,tri[None]);normal=normal[0]
  if not np.allclose(tri,r['vertices_native'],atol=1e-12,rtol=0) or abs(area[0]-r['area_over_D2'])>1e-16:raise ValueError('Source correspondence')
  for inc in r['incidences']:
   p=tri[inc['corner']]
   for label,(ct,cn) in candidates.items():
    d=point_triangle_distances(np,np.broadcast_to(p,(len(ct),3)),ct)/100
    dot=np.clip(abs(cn@normal),0,1);compatible=dot>=math.cos(math.radians(10))+1e-12
    near=d<=.002-1e-10;minimum=float(np.degrees(np.arccos(dot[near].max()))) if near.any() else None
    expected=inc[label];best=float(d[compatible].min()) if compatible.any() else None
    if abs(float(d.min())-expected['distance_over_D'])>1e-10:raise ValueError('Independent nearest distance')
    if (minimum is None)!=(expected['minimum_nearby_angle_degrees'] is None) or (minimum is not None and abs(minimum-expected['minimum_nearby_angle_degrees'])>2e-5):raise ValueError('Independent near angle')
    if (best is None)!=(expected['nearest_compatible_distance_over_D'] is None) or (best is not None and abs(best-expected['nearest_compatible_distance_over_D'])>1e-9):raise ValueError('Independent compatible distance')
    passed=bool(best is not None and best<=.002-1e-10)
    if passed!=expected['PASS'] or passed==bool(r[label+'_mask']&(1<<inc['corner'])):raise ValueError('Independent incidence status')
    checked+=1
  c='both' if r['raw_mask'] and r['final_mask'] else 'raw_only' if r['raw_mask'] else 'final_only'
  facecounts[c]+=1;failareas[c]+=r['area_over_D2'];patchrows[r['region']].append(r)
  direct=[x for x in r['nearby_source'] if x['shared_indices']>=2]
  states=Counter(x['contact_state'] for x in r['nearby_source'])
  # These are evidence tags, deliberately not a noise/defect verdict.
  tags=[]
  if any(len(e['faces'])>2 for e in r['edges']):tags.append('SOURCE_JUNCTION')
  if any(len(e['faces'])==1 for e in r['edges']):tags.append('SOURCE_BOUNDARY')
  if states['NON_INDEX_CONNECTED_CONTACT']:tags.append('NON_INDEX_CONNECTED_CONTACT')
  if states['BEYOND_SHARED_INDEX_SIMPLEX']:tags.append('CONTACT_BEYOND_SHARED_INDEX_SIMPLEX')
  if states['NUMERICALLY_NEAR_SHARED_SIMPLEX']:tags.append('NUMERICALLY_UNCERTAIN_NEAR_SHARED_CONTACT')
  if not tags:tags.append('NO_CONFIRMED_DEFECT_IN_MEASURED_LOCAL_CONTEXT')
  details.append(dict(face=fid,region=r['region'],class_name=c,area_over_D2=r['area_over_D2'],tags=tags,
    minimum_altitude_over_D=r['minimum_altitude_over_D'],aspect=r['aspect_longest_over_altitude'],
    shared_edge_neighbour_angles=[x['angle_degrees'] for x in direct],contact_counts=dict(states)))
 source_area=0.
 for i in range(0,len(f),100000):source_area+=float(geometry(np,v[f[i:i+100000]])[1].sum())
 patches=[]
 for region,rr in sorted(patchrows.items()):
  ids=outcomes['face'][outcomes['region']==region];tri=v[f[ids]];n,area=geometry(np,tri)
  moment=np.einsum('ni,nj,n->ij',n,n,area);_,vectors=np.linalg.eigh(moment);axis=vectors[:,-1]
  angles=np.degrees(np.arccos(np.clip(abs(n@axis),0,1)));order=np.argsort(angles);cdf=np.cumsum(area[order])/area.sum()
  affected=np.array([r['face'] for r in rr]);af=v[f[affected]]
  # Components of failed faces only, across genuine exactly-two-incidence edges.
  parent={int(i):int(i) for i in affected}
  def find(i):
   while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
   return i
  for r in rr:
   for e in r['edges']:
    if len(e['faces'])==2 and all(i in parent for i in e['faces']):parent[find(e['faces'][0])]=find(e['faces'][1])
  components=[]
  for root in sorted({find(i) for i in parent}):
   members=[i for i in parent if find(i)==root];tt=v[f[members]]
   components.append(dict(faces=members,area_over_D2=float(geometry(np,tt)[1].sum()),bounds_native=[tt.min(axis=(0,1)).tolist(),tt.max(axis=(0,1)).tolist()]))
  patches.append(dict(region=region,source_faces=len(ids),source_area_over_D2=float(area.sum()),
    source_extent_over_D=(np.ptp(tri.reshape(-1,3),axis=0)/100).tolist(),
    dominant_unsigned_normal=axis.tolist(),area_weighted_normal_p50=float(angles[order[np.searchsorted(cdf,.5)]]),
    area_weighted_normal_p95=float(angles[order[np.searchsorted(cdf,.95)]]),
    failed_faces=len(rr),failed_area_over_D2=sum(r['area_over_D2'] for r in rr),
    failed_extent_over_D=(np.ptp(af.reshape(-1,3),axis=0)/100).tolist(),
    failed_bounds_native=[af.min(axis=(0,1)).tolist(),af.max(axis=(0,1)).tolist()],
    failed_face_components=len(components),components=components,
    raw_failed_incidences=sum(r['raw_mask'].bit_count() for r in rr),final_failed_incidences=sum(r['final_mask'].bit_count() for r in rr)))
 save(run.BASE/'face-classification.json',details)
 save(run.BASE/'analysis.json',dict(classification='COMPLETE_READ_ONLY_MEASUREMENTS',census=summary,
  independent_incidence_output_checks=checked,source_area_over_D2=source_area,affected_face_classes=dict(facecounts),
  affected_area_over_D2=dict(failareas),affected_area_fraction=sum(failareas.values())/source_area,patches=patches,
  face_evidence_tags=dict(Counter(t for d in details for t in d['tags'])),
  context_scope='Full source used for AABB search. Retain neighbours within .002D of any failed-face vertex or centroid; probe minimum is not triangle-pair distance. Contact classification only within this subset; EPIK near-shared contacts remain numerically uncertain.',
  interpretation='No automatic noise classification. Coherent small patches may be meaningful. Source-area counts deduplicate faces; incidence counts do not.',
  frozen_research_result='NO_GO',new_reconstruction_executed=False))

def raster(np,vertices,faces,camera,marked=None,wire=False,budget=None):
 size=512;center=np.array(camera['center']);view=np.array(camera['view']);right=np.array(camera['right']);up=np.array(camera['up']);scale=size/camera['span']
 q=vertices-center;proj=np.column_stack((256+q@right*scale,256-q@up*scale,q@view));del q
 # Full geometry participates, including occluders; this is not a cutaway.
 wanted=[]
 for start in range(0,len(faces),100000):
  p=proj[faces[start:start+100000]];lo=p[:,:,:2].min(axis=1);hi=p[:,:,:2].max(axis=1)
  wanted.extend((start+np.flatnonzero((lo[:,0]<512)&(hi[:,0]>=0)&(lo[:,1]<512)&(hi[:,1]>=0))).tolist())
 depth=np.full((size,size),-np.inf);image=np.full((size,size,3),238,dtype='u1');light=.35*right+.6*up+.72*view;light/=np.linalg.norm(light)
 if budget is not None:
  budget['triangles']+=len(wanted)
  if budget['triangles']>200000:raise ValueError('Render triangle-work cap')
 marked=set() if marked is None else marked
 for fid in wanted:
  p=proj[faces[fid]];x0,y0=np.maximum(0,np.floor(p[:,:2].min(axis=0))).astype(int);x1,y1=np.minimum(511,np.ceil(p[:,:2].max(axis=0))).astype(int)
  if x1<x0 or y1<y0:continue
  if budget is not None:
   budget['pixels']+=int((x1-x0+1)*(y1-y0+1))
   if budget['pixels']>250000000:raise ValueError('Render pixel-work cap')
  u=p[1,:2]-p[0,:2];w=p[2,:2]-p[0,:2];ar=float(u[0]*w[1]-u[1]*w[0])
  if abs(ar)<1e-12:continue
  xx,yy=np.meshgrid(np.arange(x0,x1+1)+.5,np.arange(y0,y1+1)+.5)
  def edge(a,b):return (b[0]-a[0])*(yy-a[1])-(b[1]-a[1])*(xx-a[0])
  a=edge(p[1],p[2])/ar;b=edge(p[2],p[0])/ar;c=1-a-b;z=a*p[0,2]+b*p[1,2]+c*p[2,2]
  dep=depth[y0:y1+1,x0:x1+1];active=(a>=-1e-12)&(b>=-1e-12)&(c>=-1e-12)&(z>dep)
  if not active.any():continue
  dep[active]=z[active];t=vertices[faces[fid]];n=np.cross(t[1]-t[0],t[2]-t[0]);n/=np.linalg.norm(n);shade=int(65+165*abs(n@light))
  rgb=np.full((*a.shape,3),shade,dtype='u1')
  if wire:
   lengths=[np.linalg.norm(p[2,:2]-p[1,:2]),np.linalg.norm(p[0,:2]-p[2,:2]),np.linalg.norm(p[1,:2]-p[0,:2])]
   edge_dist=np.minimum.reduce([abs(ar)*a/max(lengths[0],1e-12),abs(ar)*b/max(lengths[1],1e-12),abs(ar)*c/max(lengths[2],1e-12)])
   if fid in marked:rgb[:]=np.clip(np.array([1.,.42,.08])*shade+20,0,255).astype('u1')
   rgb[edge_dist<.65]=40
  image[y0:y1+1,x0:x1+1][active]=rgb[active]
 return image

def render():
 import numpy as np
 from amd_cgal_official_check import read_off
 from amd_mesh_aware_result import png
 data=json.loads((run.BASE/'analysis.json').read_text());rows=records();v,f=source(np)
 outputs=[]
 for name in ('raw-wrap.off','simplified.off'):
  a,b=read_off(run.PRIOR/name);outputs.append((np.asarray(a),np.asarray(b)))
 directory=run.BASE/'review';directory.mkdir(exist_ok=False);cams=[];html=['<!doctype html><meta charset="utf-8"><title>Phase19 saved-output feature review</title><style>body{background:#ddd;font:16px sans-serif}img{max-width:100%}h2{margin-top:2em}</style><h1>AI-DERIVED RECONSTRUCTION — frozen normal gate FAIL</h1><p>Each panel: original AI / raw 2808 triangles / final 1000 triangles. Top: identical neutral flat shading. Bottom: wireframe; orange marks affected original faces. Full original geometry participates in depth testing. No smoothing, geometry edits or cutaways. Pixel-scale slivers remain uncertain, not automatically noise.</p>']
 budget=dict(triangles=0,pixels=0)
 for patch in data['patches']:
  region=patch['region'];rr=[r for r in rows if r['region']==region];bounds=np.array(patch['failed_bounds_native']);center=bounds.mean(axis=0)
  # Source-only camera, identical for both outputs; outward sign is a declared viewing heuristic.
  n=np.array(patch['dominant_unsigned_normal'])
  if n@center<0:n=-n
  axis=np.eye(3)[np.argmin(abs(n))];right=np.cross(axis,n);right/=np.linalg.norm(right);view=n+.25*right;view/=np.linalg.norm(view);right=np.cross(axis,view);right/=np.linalg.norm(right);up=np.cross(view,right)
  camera=dict(region=region,center=center.tolist(),view=view.tolist(),right=right.tolist(),up=up.tolist(),span=max(1.,float(np.linalg.norm(bounds[1]-bounds[0]))*1.8),resolution=512,full_source_occlusion=True)
  cams.append(camera);images=[];marked={r['face'] for r in rr}
  for wire in (False,True):
   images.append(np.concatenate([raster(np,v,f,camera,marked,wire,budget),*[raster(np,a,b,camera,None,wire,budget) for a,b in outputs]],axis=1))
  sheet=np.concatenate(images,axis=0);name=f'region-{region:02}.png';png(directory/name,np,sheet)
  html.append(f'<h2>Source region {region}</h2><p>{patch["failed_faces"]} affected faces; raw/final failed incidences {patch["raw_failed_incidences"]}/{patch["final_failed_incidences"]}. Span {camera["span"]/100:.6g} D.</p><img src="{name}">')
 save(directory/'cameras.json',cams)
 with (directory/'index.html').open('x',encoding='utf-8') as out:out.write('\n'.join(html))
 save(run.BASE/'render.json',dict(classification='PASS',regions=len(cams),matched_camera_and_lighting=True,geometry_unchanged=True,work=budget,camera_rule='Area-weighted unsigned source normal, sign toward source centroid from source origin, 0.25 tangent obliquity; source failed-region bounds; independent of output coordinates',files={p.name:sha(p) for p in directory.iterdir()}))

def verify_png(path,width=1536,height=1024):
 if path.stat().st_size>8*1024**2:raise ValueError('PNG size cap')
 data=path.read_bytes()
 if data[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('PNG signature')
 pos=8;compressed=b'';names=[];header=None
 while pos<len(data):
  if pos+12>len(data):raise ValueError('Truncated PNG')
  length=struct.unpack('>I',data[pos:pos+4])[0];name=data[pos+4:pos+8];payload=data[pos+8:pos+8+length];end=pos+12+length
  if end>len(data) or zlib.crc32(name+payload)&0xffffffff!=struct.unpack('>I',data[end-4:end])[0]:raise ValueError('PNG CRC')
  names.append(name)
  if name==b'IHDR':header=struct.unpack('>IIBBBBB',payload)
  elif name==b'IDAT':compressed+=payload
  pos=end
 if names!=[b'IHDR',b'IDAT',b'IEND'] or header!=(width,height,8,2,0,0,0):raise ValueError('PNG schema')
 limit=height*(1+width*3);d=zlib.decompressobj();pixels=d.decompress(compressed,limit+1)
 if len(pixels)!=limit or not d.eof or d.unused_data or any(pixels[i*(1+width*3)]!=0 for i in range(height)):raise ValueError('PNG pixels')
 return dict(bytes=len(data),sha256=sha(path),width=width,height=height,decoded_pixels=width*height)

def visual_finalise():
 """Verify existing sheets; add five tighter views, never rerender/overwrite them."""
 import numpy as np
 from amd_cgal_official_check import read_off
 from amd_mesh_aware_result import png
 data=json.loads((run.BASE/'analysis.json').read_text());rows=records();directory=run.BASE/'review';cams=json.loads((directory/'cameras.json').read_text())
 expected={p['region'] for p in data['patches']}
 if len(cams)!=20 or {c['region'] for c in cams}!=expected:raise ValueError('Camera/region coverage')
 verified={f'region-{g:02}.png':verify_png(directory/f'region-{g:02}.png') for g in sorted(expected)}
 contact_pairs=Counter();affected=Counter()
 for r in rows:
  missing=0
  for n in r['nearby_source']:
   if n['shared_indices'] and n['contact_dimension']==-1:contact_pairs['known_shared_index_contact_not_reported']+=1;missing+=1
  if missing:affected['faces_with_missing_known_contact']+=1
 contacts=dict(classification='INCONCLUSIVE_NUMERICAL_CONTACT_DIAGNOSTIC',counts=dict(contact_pairs),affected_faces=dict(affected),
   explanation='EPIK constructed intersections miss known shared-index contacts. Neither NONE nor BEYOND_SHARED proves absence/presence of abnormal contact. Topological edge-incidence and normal/distance census remain independently valid.')
 out=run.BASE/'detail-review';out.mkdir(exist_ok=False);v,f=source(np);outputs=[]
 for name in ('raw-wrap.off','simplified.off'):
  a,b=read_off(run.PRIOR/name);outputs.append((np.asarray(a),np.asarray(b)))
 budget=dict(triangles=0,pixels=0);detail_cameras=[];html=['<!doctype html><meta charset="utf-8"><title>Phase19 feature detail</title><style>body{font:16px sans-serif;background:#ddd}img{max-width:100%}</style><h1>Five finer matched views — original / raw wrap / 1000 triangles</h1><p>Top neutral flat shading; bottom wireframe with affected original faces orange. Source-derived cameras, full-source occlusion, unchanged geometry. These views magnify genuine angular differences; they do not establish semantic importance or game readiness.</p><p><a href="../review/index.html">All20 whole-region views</a></p>']
 for region in (8,18,24,34,37):
  rr=[r for r in rows if r['region']==region];new=[r for r in rr if r['final_mask']&~r['raw_mask']];representative=next(r for r in rr if r['face']==506773) if region==37 else max(new or rr,key=lambda r:r['area_over_D2'])
  camera=dict(next(c for c in cams if c['region']==region));tri=np.array(representative['vertices_native']);camera['center']=tri.mean(axis=0).tolist();camera['span']=max(.2,float(np.linalg.norm(np.ptp(tri,axis=0)))*2.5);camera['representative_face']=representative['face'];detail_cameras.append(camera)
  marked={r['face'] for r in rr};images=[]
  for wire in (False,True):images.append(np.concatenate([raster(np,v,f,camera,marked,wire,budget),*[raster(np,a,b,camera,None,wire,budget) for a,b in outputs]],axis=1))
  name=f'region-{region:02}-detail.png';png(out/name,np,np.concatenate(images,axis=0));verify_png(out/name)
  html.append(f'<h2>Region{region}, source face{representative["face"]}</h2><p>View span{camera["span"]/100:.6g}D.</p><img src="{name}">')
 save(out/'cameras.json',detail_cameras)
 with (out/'index.html').open('x',encoding='utf-8') as stream:stream.write('\n'.join(html))
 save(run.BASE/'visual-completion.json',dict(classification='PASS_EXISTING_IMAGES_AND_NEW_DETAIL_VIEWS',
  prior_render_host_result='FAIL: receipt-only numpy.int64 JSON serialization; preserved without retry',
  original_images=verified,contact_reliability=contacts,detail_views=5,detail_work=budget,geometry_modified=False,
  detail_files={p.name:sha(p) for p in out.iterdir()}))
