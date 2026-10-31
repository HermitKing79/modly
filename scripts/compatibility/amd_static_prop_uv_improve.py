"""Phase20C bounded UV-only analysis/attempt; historical runners stay held."""
import sys,json,math,hashlib,subprocess,os
from pathlib import Path
from collections import defaultdict,Counter
from types import SimpleNamespace
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent))
import amd_static_prop_recovery as prior
old=prior.old;owner=prior.owner;HISTORICAL=prior.BASE
BASE=HISTORICAL.parent/'amd-static-prop-uv-improve-20261031-v1'
PARENT='89bbbe7b6f050c8ef6efa9088a598f0a5d7557d9'
STAGES=('authenticate','diagnose','prepare','uv','uv-audit','export','verify','closeout','preserve')
CURRENT=None;UV_CALLS=0

def topology(np,uv,f,seam_keys=None):
 """UV vertices are edge-connected corner fans, never coordinate-welded."""
 n=len(f);p=list(range(3*n));fp=list(range(n));edges=defaultdict(list)
 def root(a,arr):
  while arr[a]!=a:arr[a]=arr[arr[a]];a=arr[a]
  return a
 def join(a,b,arr):arr[root(a,arr)]=root(b,arr)
 for i,face in enumerate(f):
  for k in range(3):
   a,b=int(face[k]),int(face[(k+1)%3]);ca,cb=i*3+k,i*3+(k+1)%3
   if a>b:a,b=b,a;ca,cb=cb,ca
   edges[(a,b)].append((i,ca,cb))
 seams=[];boundary=[]
 for key,rows in edges.items():
  if len(rows)!=2:raise ValueError('Closed two-incidence source required')
  a,b=rows
  connected=key not in seam_keys if seam_keys is not None else all(np.array_equal(uv.reshape(-1,2)[x],uv.reshape(-1,2)[y]) for x,y in zip(a[1:],b[1:]))
  if connected:
   join(a[0],b[0],fp);join(a[1],b[1],p);join(a[2],b[2],p)
  else:seams.append(list(key));boundary.extend([a,b])
 labels={x:i for i,x in enumerate(sorted({root(i,fp) for i in range(n)}))};charts=[labels[root(i,fp)] for i in range(n)]
 groups=defaultdict(list)
 for i in range(n):groups[charts[i]].append(i)
 br=defaultdict(list);coordinates=defaultdict(set);source_ids=defaultdict(set)
 for i,a,b in boundary:
  ra,rb=root(a,p),root(b,p);br[charts[i]].append((ra,rb))
  if uv is not None:
   for c,r in ((a,ra),(b,rb)):
    key=(charts[i],tuple(float(x) for x in uv.reshape(-1,2)[c]));coordinates[key].add(r);source_ids[key].add(int(f.reshape(-1)[c]))
 anomalies=[]
 for key,roots in coordinates.items():
  degree=sum(sum(x==r for pair in br[key[0]] for x in pair) for r in roots)
  if degree!=2:anomalies.append(dict(chart=key[0],uv=list(key[1]),coordinate_welded_degree=degree,source_vertices=sorted(source_ids[key]),fans=[dict(id=r,boundary_degree=sum(x==r for pair in br[key[0]] for x in pair),faces=sorted({i//3 for i in range(3*n) if root(i,p)==r})) for r in sorted(roots)]))
 fans=defaultdict(list)
 for i in range(3*n):fans[root(i,p)].append(i)
 coincident=defaultdict(list)
 if uv is not None:
  for key,corners in fans.items():coincident[tuple(float(x) for x in uv.reshape(-1,2)[corners[0]])].append(key)
 all_contacts=[dict(uv=list(k),fans=roots) for k,roots in coincident.items() if len(roots)>1]
 reports=[]
 for chart,faces in sorted(groups.items()):
  vertices={root(3*i+k,p) for i in faces for k in range(3)};ee={tuple(sorted((root(3*i+k,p),root(3*i+(k+1)%3,p)))) for i in faces for k in range(3)}
  bg=defaultdict(list)
  for a,b in br[chart]:bg[a].append(b);bg[b].append(a)
  seen=set();loops=0
  for a in bg:
   if a in seen:continue
   loops+=1;stack=[a]
   while stack:
    x=stack.pop()
    if x in seen:continue
    seen.add(x);stack.extend(bg[x])
  chi=len(vertices)-len(ee)+len(faces)
  reports.append(dict(chart=chart,faces=faces,vertices=len(vertices),edges=len(ee),triangles=len(faces),euler=chi,boundary_components=loops,bad_boundary_fan_degrees=sum(len(x)!=2 for x in bg.values()),disk=chi==1 and loops==1 and all(len(x)==2 for x in bg.values())))
 return dict(charts=reports,face_charts=charts,seams=seams,coordinate_anomalies=anomalies,coincident_uv_fans=all_contacts)

def diagnose():
 import numpy as np
 uv=np.load(HISTORICAL/'asset/uv.npy',allow_pickle=False);v=np.load(HISTORICAL/'inputs/final-v.npy',allow_pickle=False);f=np.load(HISTORICAL/'inputs/final-f.npy',allow_pickle=False)
 d=json.loads((HISTORICAL/'asset/uv-diagnostics.json').read_text());top=topology(np,uv,f);tri=v.astype('f8')[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(n,axis=1)/2;n/=np.linalg.norm(n,axis=1)[:,None]
 rows=[]
 for chart in top['charts']:
  ids=chart['faces'];x=np.column_stack((tri[ids].reshape(-1,3),np.ones(3*len(ids))));y=uv[ids].reshape(-1,2).astype('f8')
  coeff,_,rank,s=np.linalg.lstsq(x,y,rcond=None);J=coeff[:3].T;singular=np.linalg.svd(J,compute_uv=False);axis=np.cross(J[0],J[1]);scale2=np.linalg.norm(axis);axis/=scale2
  residual=float(np.linalg.norm(x@coeff-y,axis=1).max());cos=abs(n[ids]@axis);pred=scale2*cos;actual=np.array([d['triangles'][i]['density'] for i in ids]);angle=np.degrees(np.arccos(np.clip(cos,0,1)))
  rows.append(dict(chart=chart['chart'],faces=len(ids),design_rank=int(rank),condition=float(s[0]/s[-1]) if s[-1]>0 else None,maximum_UV_residual=residual,projection_singular_values=singular.tolist(),singular_ratio=float(singular.max()/singular.min()),project_axis=axis.tolist(),angle_min=float(angle.min()),angle_max=float(angle.max()),density_min=float(actual.min()),density_max=float(actual.max()),density_ratio=float(actual.max()/actual.min()),maximum_relative_projection_density_error=float(np.max(abs(pred-actual)/actual)),source_area=float(area[ids].sum())))
 # Genuine 3D edge angle distribution, no winding or geometry modification.
 edge=defaultdict(list)
 for i,face in enumerate(f):
  for k in range(3):edge[tuple(sorted((int(face[k]),int(face[(k+1)%3]))))].append(i)
 angles=np.array([math.degrees(math.acos(float(np.clip(n[a]@n[b],-1,1)))) for a,b in edge.values()])
 owner.save(BASE/'diagnosis.json',dict(classification='READ_ONLY_UV_DIAGNOSIS_COMPLETE',historical_uv_sha256=owner.sha(HISTORICAL/'asset/uv.npy'),old_density_ratio=d['density_ratio'],projection_charts=rows,topology=top,dihedral_quantiles_degrees={str(q):float(np.quantile(angles,q)) for q in (0,.25,.5,.75,.9,.95,.99,1)},geometry_modified=False))

def reviewed(stage):
 r=json.loads((BASE/('preparation-review.json' if stage in ('authenticate','diagnose') else 'plan-review.json' if stage=='prepare' else 'export-review.json' if stage in ('export','verify','closeout','preserve') else 'execution-review.json')).read_text())
 if r['classification']!='APPROVED' or r['parent']!=PARENT or stage not in r['allowed_stages']:raise ValueError('Independent phase approval required')
 for name,h in r['files'].items():
  if owner.sha(Path(name))!=h:raise ValueError('Reviewed dependency changed')

def bind(stage):
 global CURRENT
 CURRENT=stage;prior.BASE=BASE;old.BASE=BASE;old.reviewed=reviewed;old.uv_check=uv_check

def authenticate(preserve=False):
 prior.authenticate(preserve)
 if preserve:
  rows=json.loads((BASE/'phase20B-inputs.json').read_text())
  for name,row in rows.items():old.check(HISTORICAL/name,row)
 else:
  for name,row in json.loads((HISTORICAL/'evidence-manifest.json').read_text()).items():old.check(HISTORICAL/name,row)
  rows={}
  for p in HISTORICAL.rglob('*'):
   if p.is_file():
    s=p.stat();rows[p.relative_to(HISTORICAL).as_posix()]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,file_id=s.st_ino,sha256=owner.sha(p))
  owner.save(BASE/'phase20B-inputs.json',rows)
 owner.save(BASE/('phase20B-preservation.json' if preserve else 'phase20B-authentication.json'),dict(classification='PASS',files=len(rows),historical_hold=old.LIVE_EXECUTION_HOLD))

def prepare():
 """One seam rule; no UV coordinates or parameter search."""
 import numpy as np
 old.success('diagnose')
 folder=BASE/'inputs';folder.mkdir(exist_ok=False)
 spec=json.loads((HISTORICAL/'inputs.json').read_text())
 for name in ('final-v.npy','final-f.npy'):
  path=HISTORICAL/'inputs'/name
  if owner.sha(path)!=spec['files'][name]:raise ValueError('Frozen candidate copy changed')
  with (folder/name).open('xb') as out:out.write(path.read_bytes())
 v=np.load(folder/'final-v.npy',allow_pickle=False);f=np.load(folder/'final-f.npy',allow_pickle=False);t=v.astype('f8')[f];n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);areas=np.linalg.norm(n,axis=1)/2;n/=np.linalg.norm(n,axis=1)[:,None]
 edges=defaultdict(list)
 for i,face in enumerate(f):
  for k in range(3):edges[tuple(sorted((int(face[k]),int(face[(k+1)%3]))))].append(i)
 seams=set()
 for key,rows in edges.items():
  if len(rows)!=2:raise ValueError('Closed source edge incidence')
  if float(n[rows[0]]@n[rows[1]])<math.cos(math.radians(20)):seams.add(key)
 top=topology(np,None,f,seams);reports=[]
 for c in top['charts']:
  ids=c['faces'];axis=(n[ids]*areas[ids,None]).sum(axis=0);axis/=np.linalg.norm(axis);angles=np.degrees(np.arccos(np.clip(n[ids]@axis,-1,1)))
  reports.append(dict(**c,normal_axis=axis.tolist(),maximum_axis_angle=float(angles.max()),source_area=float(areas[ids].sum())))
 owner.save(BASE/'seam-plan.json',dict(classification='OFFLINE_SEAM_RULE_MEASURED',method='Genuine edge dihedral >20 degrees, no UV parameterization yet',seam_edges=[list(x) for x in sorted(seams)],charts=reports,geometry_files={p.name:owner.sha(p) for p in folder.iterdir()},source_D=spec['D'],source_center=spec['center']))

def align_uv(np,uv,f):
 expected=np.load(BASE/'inputs/final-f.npy',allow_pickle=False)
 def key(a):return min(tuple(int(x) for x in a[i:].tolist()+a[:i].tolist()) for i in range(3))
 mapping={key(a):i for i,a in enumerate(expected)};result=np.empty_like(uv);seen=set()
 for row,a in enumerate(f):
  i=mapping.get(key(a))
  if i is None or i in seen:raise ValueError('Oriented source-face identity')
  seen.add(i)
  for k,v in enumerate(expected[i]):result[i,k]=uv[row,list(a).index(v)]
 if len(seen)!=len(expected):raise ValueError('Complete source-face correspondence')
 return result,expected

def measure(np,uv,v,f):
 uv,f=align_uv(np,uv,f);d=prior.diagnostics(np,uv,v,f)
 if not d['finite_uv']:
  d.update(topological_charts=None,directional_stretch_maximum=None,directional_stretch_per_triangle=[],directional_stretch_valid=False,additional_checks='NOT_COMPUTABLE_NONFINITE_UV');return d
 top=topology(np,uv,f)
 plan=json.loads((BASE/'seam-plan.json').read_text());strategy=json.loads((BASE/'strategy.json').read_text())
 if {frozenset(c['faces']) for c in top['charts']}!={frozenset(c['faces']) for c in plan['charts']} or {tuple(e) for e in top['seams']}!={tuple(e) for e in plan['seam_edges']}:d['failures'].append('PLANNED_PARTITION')
 if any(not c['disk'] for c in top['charts']):d['failures'].append('UV_CHART_TOPOLOGY')
 if top['coincident_uv_fans']:d['failures'].append('COINCIDENT_UV_VERTICES')
 for c in top['charts']:
  signs={bool(d['triangles'][i]['signed_uv_area']>0) for i in c['faces']}
  if len(signs)>1:d['failures'].append('UV_FOLD');break
 shape=[]
 if d['finite_uv'] and not d['zero_area_faces']:
  for i,t in enumerate(v.astype('f8')[f]):
   a,b=t[1]-t[0],t[2]-t[0];la=float(np.linalg.norm(a));x=float(b@a/la);h=float(np.linalg.norm(np.cross(a,b))/la)
   intrinsic=np.array([[la,x],[0,h]]);q=uv[i].astype('f8');mat=(q[1:]-q[0]).T@np.linalg.inv(intrinsic);s=np.linalg.svd(mat,compute_uv=False)
   shape.append(float(s.max()/s.min()) if np.isfinite(s).all() and s.min()>0 else math.inf)
  if max(shape)>strategy['maximum_triangle_directional_stretch_ratio']:d['failures'].append('DIRECTIONAL_STRETCH')
 d['directional_stretch_maximum']=prior.finite(max(shape)) if shape else None
 d['directional_stretch_valid']=bool(shape and all(math.isfinite(x) for x in shape))
 d['directional_stretch_per_triangle']=[prior.finite(x) for x in shape];d['topological_charts']=top
 d['classification']='NO_GO_UV' if d['failures'] else 'PASS_UV'
 return d

def uv_check(np,uv,v,f):
 global UV_CALLS
 UV_CALLS+=1;d=measure(np,uv,v,f)
 path=BASE/'asset/uv-diagnostics.json' if CURRENT=='uv' else BASE/f'{CURRENT}-uv-{UV_CALLS}.json'
 if CURRENT=='uv':d['uv_file_sha256']=owner.sha(BASE/'asset/uv.npy')
 owner.save(path,d)
 if d['classification']!='PASS_UV':raise ValueError('UV qualification refused: '+json.dumps(dict(classification=d['classification'],failures=d['failures'],density_ratio=d['density_ratio'])))
 return d

def same_measurements(a,b):
 """Numeric agreement across qualified NumPy builds; identities/verdicts stay exact."""
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same_measurements(a[k],b[k]) for k in a)
 if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same_measurements(x,y) for x,y in zip(a,b))
 if isinstance(a,float):return isinstance(b,(float,int)) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-14)
 return a==b

def uv_audit():
 import numpy as np
 r=json.loads((BASE/'uv-run/host-result.json').read_text());d=json.loads((BASE/'asset/uv-diagnostics.json').read_text())
 if r.get('error') or r['worker_returncode'] not in (0,1) or not r['cleanup']['owned_tree_exit_verified'] or r['cleanup']['forced_job_cleanup']:raise ValueError('UV worker failed outside qualification')
 if r['classification']=='FAIL':
  expected='ValueError: UV qualification refused: '+json.dumps(dict(classification=d['classification'],failures=d['failures'],density_ratio=d['density_ratio']))
  if d['classification']!='NO_GO_UV' or (BASE/'uv-run/stderr.txt').read_text().splitlines().count(expected)!=1:raise ValueError('Exact UV failure required')
 if owner.sha(BASE/'asset/uv.npy')!=d['uv_file_sha256']:raise ValueError('UV identity changed')
 logs=(BASE/'uv-run/stdout.txt').read_text()+(BASE/'uv-run/stderr.txt').read_text()
 warnings=[line for line in logs.splitlines() if 'Unwrap failed to solve' in line or 'Unwrap could not solve' in line]
 uv=np.load(BASE/'asset/uv.npy',allow_pickle=False);v=np.load(BASE/'inputs/final-v.npy',allow_pickle=False);f=np.load(BASE/'inputs/final-f.npy',allow_pickle=False)
 result=measure(np,uv,v,f)
 if not same_measurements(result,{k:v for k,v in d.items() if k!='uv_file_sha256'}):raise ValueError('Independent UV measurements differ')
 t=v.astype('f8')[f];a=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)/2;u=uv.astype('f8');s=u[:,1]-u[:,0];w=u[:,2]-u[:,0];den=abs(s[:,0]*w[:,1]-s[:,1]*w[:,0])/2/a
 with np.errstate(divide='ignore',invalid='ignore'):ratio=float(den.max()/den.min())
 if (math.isfinite(ratio) and ratio<=1.1)!=(d['density_ratio_finite'] and d['density_ratio']<=1.1):raise ValueError('Independent Float64 area-density verdict differs')
 owner.save(BASE/'uv-audit.json',dict(classification='PASS_UV' if d['classification']=='PASS_UV' and not warnings else 'NO_GO_UV',diagnostics_reproduced=True,density_ratio=d['density_ratio'],independent_float64_density_ratio=prior.finite(ratio),independent_ratio_finite=math.isfinite(ratio),audit_invalid_explanation=None if math.isfinite(ratio) else 'Zero or nonfinite UV density; ratio retained as null and UV remains NO-GO',unwrap_failure_warnings=warnings,uv_sha256=d['uv_file_sha256']))

def require_uv_pass():
 old.success('uv-audit')
 if json.loads((BASE/'uv-audit.json').read_text())['classification']!='PASS_UV':raise ValueError('UV qualification NO-GO; export forbidden')

def verify_import():
 """Verify actual Blender reimport arrays plus attribute-seam duplication."""
 import numpy as np
 folder=BASE/'asset';v=np.load(BASE/'inputs/final-v.npy');f=np.load(BASE/'inputs/final-f.npy');uv=np.load(folder/'uv.npy')
 gv,gf,guv,gn,_,_=old.glb(np,folder/'experimental-static-prop.glb')
 attributes=[tuple(float(x) for x in (*a,*b,*c)) for a,b,c in zip(gv,guv,gn)]
 required={attributes[int(i)] for i in gf.ravel()}
 if len(required)!=len(gv) or len(np.unique(gf))!=len(gv):raise ValueError('Unused or redundant export attribute vertices')
 iv=np.load(folder/'reimport-v.npy');iff=np.load(folder/'reimport-f.npy');iuv=np.load(folder/'reimport-uv.npy');normal=np.load(folder/'reimport-normals.npy')
 expected=uv.copy();expected[:,:,1]=np.float32(1)-(np.float32(1)-uv[:,:,1])
 keys={old.corner_key(v[face],expected[i]):i for i,face in enumerate(f)}
 if sorted(keys.get(old.corner_key(iv[face],iuv[i]),-1) for i,face in enumerate(iff))!=list(range(1000)):raise ValueError('Actual reimport oriented geometry/UV correspondence')
 t=iv.astype('f8')[iff];n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);n/=np.linalg.norm(n,axis=1)[:,None];dots=(normal*n[:,None,:]).sum(axis=2)
 if not np.isfinite(normal).all() or not np.allclose(np.linalg.norm(normal,axis=2),1,atol=1e-5) or dots.min()<.99999:raise ValueError('Actual imported flat normals')
 w,h,c,pixels=old.png_pixels(folder/'reimport-packed.png');sw,sh,sc,source=old.png_pixels(folder/'neutral-test.png')
 a=np.frombuffer(pixels,'u1').reshape(h,w,c);b=np.frombuffer(source,'u1').reshape(sh,sw,sc)
 if (w,h)!=(sw,sh) or not np.array_equal(a[:,:,:3],b[:,:,:3]) or (c==4 and not np.all(a[:,:,3]==255)):raise ValueError('Reimported packed texture pixels')
 if not np.array_equal(np.ptp(iv,axis=0),np.ptp(v,axis=0)):raise ValueError('Reimport dimensions changed')
 owner.save(BASE/'reimport-verification.json',dict(classification='PASS_ACTUAL_REIMPORT',expected_unique_attribute_vertices=len(required),export_attribute_vertices=len(gv),required_attribute_seam_duplicates=len(gv)-len(v),all_attribute_vertices_referenced=True,actual_import_uv_roundtrip_exact=True,actual_import_flat_normal_minimum_dot=float(dots.min()),packed_texture_pixels_exact=True,dimensions=np.ptp(v,axis=0).tolist(),metres_per_unit=1,physical_size_qualified=False))

def closeout():
 audit=json.loads((BASE/'uv-audit.json').read_text());d=json.loads((BASE/'asset/uv-diagnostics.json').read_text())
 verified=json.loads((BASE/'verification.json').read_text()) if (BASE/'verification.json').exists() else None
 host=json.loads((BASE/'verify-run/host-result.json').read_text()) if (BASE/'verify-run/host-result.json').exists() else None
 passed=audit['classification']=='PASS_UV' and verified is not None and host is not None and host['classification']=='PASS'
 actual=json.loads((BASE/'reimport-verification.json').read_text()) if passed else None
 owner.save(BASE/'final-manifest.json',dict(classification='PASS_UV_EXPORT_USER_APPEARANCE_PENDING' if passed else 'NO_GO_UV_EXPORT',UV=audit['classification'],density_ratio=d['density_ratio'],directional_stretch_maximum=d['directional_stretch_maximum'],charts=d['charts'],seams=d['seam_edges'],coverage=d['coverage_fraction'],GLB='PASS' if passed else 'NOT_QUALIFIED',GLB_path=str(BASE/'asset/experimental-static-prop.glb') if (BASE/'asset/experimental-static-prop.glb').exists() else None,verification=verified if passed else None,partial_verification=verified if not passed else None,actual_reimport_verification=actual,research='PERMANENT_NO_GO',profile='LOWER-DETAIL STATIC PROP',appearance='USER_APPEARANCE_APPROVAL_PENDING',prior_appearance_assessment='INCONCLUSIVE_LIMITED_CLOSEUP_INFORMATION_UNCHANGED',normal_bake='SKIPPED',Sbox='NOT_TESTED',game_ready=False,geometry_modified=False,retries=0,artifacts={str(p.relative_to(BASE)):dict(bytes=p.stat().st_size,sha256=owner.sha(p)) for p in (BASE/'asset').iterdir() if p.is_file()},remaining=['user appearance approval with informative distance/material views','production PBR and padding/mips','S&box units/import/collision/LOD','separate binary distribution licence review']))

def worker(stage):
 if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Exact owned GO and EOF required')
 bind(stage);sys.addaudithook(prior._ORIGINAL_AUDIT(stage));reviewed(stage)
 if stage in ('authenticate','preserve'):authenticate(stage=='preserve');return
 old.success('authenticate');sys.path.insert(0,str(owner.CPU_SITE))
 if stage=='diagnose':diagnose()
 elif stage=='prepare':prepare()
 elif stage=='uv-audit':uv_audit()
 elif stage=='verify':old.success('export');require_uv_pass();prior._VERIFY();verify_import()
 elif stage=='closeout':closeout()
 else:raise ValueError('Candidate operation not prepared')

def main(argv=None):
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=STAGES);p.add_argument('--worker',choices=STAGES);a=p.parse_args(argv)
 if a.worker:worker(a.worker);return 0
 if not a.stage:print('INERT: Phase20C reviewed UV-only proof');return 0
 bind(a.stage);reviewed(a.stage)
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(owner.ROOT),text=True,timeout=5).strip()!=PARENT:raise ValueError('Verified parent required')
 scratch=BASE/'scratch'/a.stage;scratch.mkdir(parents=True,exist_ok=False)
 if a.stage in ('uv','export'):old.success('prepare')
 if a.stage=='export':require_uv_pass()
 owner.BASE=BASE;owner.__file__=__file__;launch=subprocess.Popen
 def routed(command,**kwargs):
  expected=[sys._base_executable,'-B','-I','-S',str(Path(__file__).resolve()),'--worker',a.stage]
  if command!=expected:raise ValueError('Exact owned CPU route')
  if a.stage in ('uv','export'):command=[str(old.BLENDER),'--background','--factory-startup','--disable-autoexec','--threads','4','--python-exit-code','1','--python',str(Path(__file__).with_name('amd_static_prop_uv_improve_blender.py')),'--',a.stage]
  kwargs['env']=dict(kwargs['env'],TEMP=str(scratch),TMP=str(scratch));return launch(command,**kwargs)
 owner.subprocess=SimpleNamespace(Popen=routed,PIPE=subprocess.PIPE)
 return owner.supervise(a.stage)
if __name__=='__main__':raise SystemExit(main())
