"""Phase20B explicit fresh CPU attempt; historical Phase20 hold stays intact."""
import sys,json,math,hashlib,subprocess,os
from pathlib import Path
from collections import defaultdict,Counter
from types import SimpleNamespace
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent))
import amd_static_prop as old
owner=old.owner;HISTORICAL=old.BASE
BASE=HISTORICAL.parent/'amd-static-prop-recovery-20261031-v1'
PARENT='aec918d34198adc5d2be9e12d0675bf49fef44a4'
RENDERS=old.RENDERS;STAGES=('authenticate','prepare','export','uv-audit','verify',*RENDERS,'sheet','preserve')
_AUTH=old.authenticate;_PREPARE=old.prepare;_VERIFY=old.verify
_ORIGINAL_AUDIT=old.audit
CURRENT=None;UV_CALLS=0

def finite(x):return float(x) if math.isfinite(float(x)) else None
def diagnostics(np,uv,v,f):
 """All diagnostics before acceptance. Acceptance arithmetic matches Phase20."""
 from amd_cpu_uv_bake import geometry_hash
 if uv.shape!=(len(f),3,2) or not 0<len(f)<=1000 or v.shape[1:]!=(3,) or not np.isfinite(v).all() or f.min()<0 or f.max()>=len(v):raise ValueError('Bounded diagnostic schema')
 valid=bool(np.isfinite(uv).all());failures=[]
 if not valid:failures.append('NONFINITE_UV')
 outbounds=np.flatnonzero(np.any((uv<0)|(uv>1),axis=(1,2))).tolist()
 if outbounds:failures.append('BOUNDS')
 t=uv.astype('f8');u=t[:,1]-t[:,0];w=t[:,2]-t[:,0];signed=(u[:,0]*w[:,1]-u[:,1]*w[:,0])/2;areas=abs(signed)
 zeros=np.flatnonzero(areas==0).tolist()
 if zeros:failures.append('ZERO_UV_AREA')
 a=v[f[:,1]]-v[f[:,0]];b=v[f[:,2]]-v[f[:,0]]
 sourcearea=np.linalg.norm(np.cross(a,b).astype('f8'),axis=1)/2
 vf=v.astype('f8');independent=np.linalg.norm(np.cross(vf[f[:,1]]-vf[f[:,0]],vf[f[:,2]]-vf[f[:,0]]),axis=1)/2
 with np.errstate(divide='ignore',invalid='ignore'):density=areas/sourcearea
 densityvalid=bool(np.isfinite(density).all() and np.all(density>0))
 ratio=float(density.max()/density.min()) if densityvalid else math.inf
 if not math.isfinite(ratio) or ratio>1.1:failures.append('AREA_DENSITY')
 pairs=[];tested=0
 if valid and not zeros:
  lo=uv.min(axis=1);hi=uv.max(axis=1)
  for i in range(len(f)):
   for j in np.flatnonzero(np.all(hi[i]>lo[i+1:],axis=1)&np.all(hi[i+1:]>lo[i],axis=1))+i+1:
    tested+=1
    if old.overlap(uv[i],uv[j]):pairs.append([i,int(j)])
  if pairs:failures.append('CONTINUOUS_OVERLAP')
 parent=list(range(len(f)));edge={};seams=[];boundary=[]
 def root(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 if valid:
  for i,face in enumerate(f):
   for k in range(3):
    a,b=int(face[k]),int(face[(k+1)%3]);ends=[uv[i,k].tolist(),uv[i,(k+1)%3].tolist()]
    if a>b:a,b=b,a;ends.reverse()
    row=dict(face=i,source_edge=[a,b],uv=ends)
    if (a,b) in edge:
     prior=edge.pop((a,b))
     if prior['uv']==ends:parent[root(i)]=root(prior['face'])
     else:seams.append([prior['face'],i]);boundary.extend([prior,row])
    else:edge[(a,b)]=row
  boundary.extend(edge.values())
 chart={value:i for i,value in enumerate(sorted({root(j) for j in range(len(f))}))}
 degree=Counter()
 for row in boundary:
  row['chart']=chart[root(row['face'])]
  for p in row['uv']:degree[(row['chart'],tuple(p))]+=1
 order=np.argsort(np.where(np.isfinite(density),density,np.inf))
 records=[dict(face=i,chart=chart[root(i)] if valid else None,source_area=float(sourcearea[i]),independent_source_area=float(independent[i]),signed_uv_area=finite(signed[i]),absolute_uv_area=finite(areas[i]),density=finite(density[i]),density_valid=bool(np.isfinite(density[i]) and density[i]>0)) for i in range(len(f))]
 return dict(classification='NO_GO_UV' if failures else 'PASS_UV',failures=failures,
  density_units='UV-unit squared / original asset-unit squared; source area uses inherited Float32 cross then Float64 norm; independent area uses Float64 cross',
  density_ratio=finite(ratio),density_ratio_finite=math.isfinite(ratio),maximum_allowed_ratio=1.1,
  minimum_density=finite(density.min()),maximum_density=finite(density.max()),minimum_density_face=int(order[0]),maximum_density_face=int(order[-1]),
  worst_low_density_faces=[int(i) for i in order[:10]],worst_high_density_faces=[int(i) for i in order[-10:][::-1]],
  finite_uv=valid,out_of_bounds_faces=outbounds,zero_area_faces=zeros,charts=len(chart) if valid else None,seam_edges=len(seams),seam_face_pairs=seams,
  boundary_segments=boundary,boundary_vertex_degree_not_two=sum(d!=2 for d in degree.values()),
  uv_minimum=finite(np.min(uv)),uv_maximum=finite(np.max(uv)),continuous_overlap_pairs=pairs,overlap_test_complete=valid and not zeros,broadphase_pairs=tested,
  sum_uv_triangle_area=float(areas.sum()) if valid else None,
  coverage_fraction=float(areas.sum()) if valid and not zeros and not outbounds and not pairs else None,coverage_method='Float64 nonoverlapping in-bounds triangle area sum only when continuous overlap testing completed; not texel coverage',
  triangles=records,geometry_hash=geometry_hash(np,v,f),ordered_vertex_payload_sha256=hashlib.sha256(v.tobytes()).hexdigest(),ordered_index_payload_sha256=hashlib.sha256(f.tobytes()).hexdigest())

def uv_check(np,uv,v,f):
 global UV_CALLS
 UV_CALLS+=1;d=diagnostics(np,uv,v,f)
 if CURRENT=='export':
  path=BASE/'asset/uv-diagnostics.json'
  d['uv_file_sha256']=owner.sha(BASE/'asset/uv.npy')
 else:path=BASE/f'{CURRENT}-uv-{UV_CALLS}.json'
 owner.save(path,d)
 if d['classification']!='PASS_UV':raise ValueError('UV qualification refused: '+json.dumps(dict(classification=d['classification'],failures=d['failures'],density_ratio=d['density_ratio']),allow_nan=False))
 return d

def uv_summary(d):
 ratio=str(d['density_ratio']) if d['density_ratio_finite'] else 'nonfinite (recorded as null)'
 return f"{d['classification']}; density ratio {ratio}, required finite <=1.1; {d['charts']} charts, {d['seam_edges']} seams; coverage {d['coverage_fraction']}; failures {d['failures']}. UV coordinates and full diagnostics retained."

def reviewed(stage):
 path=BASE/('appearance-review.json' if stage in (*RENDERS,'sheet') else 'execution-review.json')
 r=json.loads(path.read_text())
 if r['classification']!='APPROVED' or stage not in r['allowed_stages'] or r['parent']!=PARENT:raise ValueError('Fresh independent stage approval required')
 for name,h in r['files'].items():
  if owner.sha(Path(name))!=h:raise ValueError('Reviewed source/evidence changed')

def bind(stage):
 global CURRENT
 CURRENT=stage;old.BASE=BASE;old.reviewed=reviewed;old.appearance_admission=appearance_admission;old.uv_check=uv_check

def authenticate(preserve=False):
 _AUTH(preserve)
 if not old.LIVE_EXECUTION_HOLD:raise ValueError('Historical hold changed')
 if preserve:
  rows=json.loads((BASE/'phase20-inputs.json').read_text())
  for name,row in rows.items():old.check(HISTORICAL/name,row)
 else:
  for name,row in json.loads((HISTORICAL/'evidence-manifest.json').read_text()).items():old.check(HISTORICAL/name,row)
  rows={}
  for path in HISTORICAL.rglob('*'):
   if path.is_file():
    s=path.stat();rows[path.relative_to(HISTORICAL).as_posix()]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,file_id=s.st_ino,sha256=owner.sha(path))
  owner.save(BASE/'phase20-inputs.json',rows)
 owner.save(BASE/('phase20-preservation.json' if preserve else 'phase20-authentication.json'),dict(classification='PASS',files=len(rows),historical_hold=True,parent=PARENT))

def check_export_failure():
 """Only authenticated current UV qualification failure permits inspection/views."""
 r=json.loads((BASE/'export-run/host-result.json').read_text());err=(BASE/'export-run/stderr.txt').read_text()
 d=json.loads((BASE/'asset/uv-diagnostics.json').read_text())
 if r['classification']!='FAIL' or r.get('error') or r['worker_returncode']!=1 or r['cleanup']['forced_job_cleanup'] or not r['cleanup']['owned_tree_exit_verified']:raise ValueError('Unqualified export process failure')
 expected='ValueError: UV qualification refused: '+json.dumps(dict(classification=d['classification'],failures=d['failures'],density_ratio=d['density_ratio']),allow_nan=False)
 if d['classification']!='NO_GO_UV' or not d['failures'] or not set(d['failures'])<= {'NONFINITE_UV','BOUNDS','ZERO_UV_AREA','AREA_DENSITY','CONTINUOUS_OVERLAP'} or err.splitlines().count(expected)!=1:raise ValueError('Current diagnostic/error correspondence required')
 if owner.sha(BASE/'asset/uv.npy')!=d['uv_file_sha256'] or any((BASE/name).exists() for name in ('asset/experimental-static-prop.glb','asset/blender.json','verification.json')):raise ValueError('UV/evidence/export state mismatch')
 return d

def appearance_admission():
 old.success('prepare');old.success('uv-audit')
 if (BASE/'verification.json').exists():old.success('verify');return
 check_export_failure()

def uv_audit():
 import numpy as np
 r=json.loads((BASE/'export-run/host-result.json').read_text())
 if r['classification']=='FAIL':d=check_export_failure()
 elif r['classification']=='PASS':d=json.loads((BASE/'asset/uv-diagnostics.json').read_text())
 else:raise ValueError('Export did not settle')
 uv=np.load(BASE/'asset/uv.npy',allow_pickle=False);v=np.load(BASE/'inputs/final-v.npy',allow_pickle=False);f=np.load(BASE/'inputs/final-f.npy',allow_pickle=False)
 independent=diagnostics(np,uv,v,f)
 if any(independent[k]!=value for k,value in d.items() if k!='uv_file_sha256'):raise ValueError('Saved UV diagnostics do not reproduce')
 # Independent area path and density reduction, without the inherited Float32 cross.
 tri=v.astype('f8')[f];area=np.sqrt(np.sum(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])**2,axis=1))/2
 t=uv.astype('f8');a=t[:,1]-t[:,0];b=t[:,2]-t[:,0];density=abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])/2/area
 inherited=[x['density'] for x in d['triangles']]
 if np.isfinite(density).all() and np.all(density>0) and all(x is not None for x in inherited):
  ratio=float(density.max()/density.min());relative=float(np.max(abs(density-np.array(inherited))/density))
  if relative>1e-4 or (ratio<=1.1)!=(d['density_ratio_finite'] and d['density_ratio']<=1.1):raise ValueError('Independent density arithmetic disagrees')
 else:ratio=float(density.max()/density.min()) if np.isfinite(density).all() and np.all(density>0) else math.inf;relative=None
 owner.save(BASE/'uv-audit.json',dict(classification='PASS_INDEPENDENT_DIAGNOSTICS',UV=d['classification'],density_ratio=d['density_ratio'],independent_float64_density_ratio=finite(ratio),maximum_relative_density_difference=relative,uv_sha256=owner.sha(BASE/'asset/uv.npy'),geometry_hash=d['geometry_hash']))

def sheet():
 import numpy as np
 from amd_mesh_aware_result import png
 old.success('uv-audit');d=json.loads((BASE/'asset/uv-diagnostics.json').read_text());out=BASE/'review';out.mkdir(exist_ok=False)
 html=['<!doctype html><meta charset="utf-8"><title>Phase20B static prop</title><style>body{background:#ddd;font:16px sans-serif}img{max-width:100%}</style><h1>LOWER-DETAIL STATIC PROP — USER_APPEARANCE_APPROVAL_PENDING</h1><p>Columns: original source / raw wrap / final1000 triangles. Rows: whole45°, whole225°, detail37, detail8. Identical CPU camera/light within each comparison, flat normals, no geometry changes.</p><p>Phase11 high-fidelity benchmark permanently NO-GO. Documented microfacet trade-off only; no AI texture or normal bake.</p>',f'<p>{uv_summary(d)}</p>']
 sheets=[]
 for kind in ('matte','glossy'):
  frames=[]
  for view in old.VIEWS:
   old.success(f'{kind}-{view}');panels=[]
   for name in ('source','raw','final'):
    w,h,c,pixels=old.png_pixels(BASE/f'{kind}-{view}'/(name+'.png'))
    if (w,h)!=(512,512):raise ValueError('Comparison dimensions')
    panels.append(np.frombuffer(pixels,'u1').reshape(h,w,c)[::2,::2,:3])
   frames.append(np.concatenate(panels,axis=1))
  image=np.concatenate(frames,axis=0);sheets.append(image);png(out/f'{kind}-sheet.png',np,image);html.append(f'<h2>{kind}</h2><img src="{kind}-sheet.png">')
 png(out/'contact-sheet.png',np,np.concatenate(sheets,axis=1))
 passed=(BASE/'verification.json').exists()
 if passed:
  old.success('verify');html.append('<p>GLB export/readback PASS. Neutral test PBR only.</p><a href="../asset/experimental-static-prop.glb">Experimental GLB</a>')
 else:html.append('<p>EXPORT NO-GO: UV qualification failed. No GLB or readback is claimed. These images compare unchanged saved geometry only.</p>')
 with (out/'index.html').open('x',encoding='utf8') as f:f.write('\n'.join(html))
 result=dict(classification='PASS_STATIC_PROP_GEOMETRY_EXPORT_USER_APPEARANCE_PENDING' if passed else 'NO_GO_EXPORT_UV',UV=d['classification'],density_ratio=d['density_ratio'],density_ratio_retained=True,GLB='PASS' if passed else 'NOT_EXPORTED',appearance='USER_APPEARANCE_APPROVAL_PENDING',research='PERMANENT_NO_GO',normal_bake='SKIPPED_UNQUALIFIED_SIGNED_NORMALS_AND_RAYS',Sbox='NOT_TESTED',game_ready=False,geometry_modified=False,retries=0,remaining=['user material/distance appearance approval','qualified UV scale/layout if failed','production PBR/UV padding/mips','S&box import/units/collision/LOD','separate binary distribution decision'],artifacts={str(p.relative_to(BASE)):dict(bytes=p.stat().st_size,sha256=owner.sha(p)) for p in [*(BASE/'asset').iterdir(),*out.iterdir()] if p.is_file()})
 owner.save(BASE/'asset-manifest.json',result)

def worker(stage):
 if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO and EOF required')
 bind(stage);sys.addaudithook(_ORIGINAL_AUDIT(stage));reviewed(stage)
 if stage in ('authenticate','preserve'):authenticate(stage=='preserve');return
 old.success('authenticate');sys.path.insert(0,str(owner.CPU_SITE))
 if stage=='prepare':_PREPARE()
 elif stage=='uv-audit':uv_audit()
 elif stage=='verify':old.success('export');old.success('uv-audit');_VERIFY()
 elif stage=='sheet':appearance_admission();sheet()
 else:raise ValueError('No CPU Python route')

def main(argv=None):
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=STAGES);p.add_argument('--worker',choices=STAGES);a=p.parse_args(argv)
 if a.worker:worker(a.worker);return 0
 if not a.stage:print('INERT: separately reviewed Phase20B fresh attempt only');return 0
 bind(a.stage);reviewed(a.stage)
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(owner.ROOT),text=True,timeout=5).strip()!=PARENT:raise ValueError('Verified Git parent required')
 if a.stage not in ('authenticate','preserve'):old.success('authenticate')
 if a.stage=='export' or a.stage in RENDERS:old.success('prepare')
 if a.stage in RENDERS:appearance_admission()
 scratch=BASE/'scratch'/a.stage;scratch.mkdir(parents=True,exist_ok=False)
 owner.BASE=BASE;owner.__file__=__file__;launch=subprocess.Popen
 def routed(command,**kwargs):
  expected=[sys._base_executable,'-B','-I','-S',str(Path(__file__).resolve()),'--worker',a.stage]
  if command!=expected:raise ValueError('Exact owned Python route required')
  if a.stage=='export' or a.stage in RENDERS:
   command=[str(old.BLENDER),'--background','--factory-startup','--disable-autoexec','--threads','4','--python-exit-code','1','--python',str(Path(__file__).with_name('amd_static_prop_recovery_blender.py')),'--',a.stage]
  kwargs['env']=dict(kwargs['env'],TEMP=str(scratch),TMP=str(scratch));return launch(command,**kwargs)
 owner.subprocess=SimpleNamespace(Popen=routed,PIPE=subprocess.PIPE)
 return owner.supervise(a.stage)
if __name__=='__main__':raise SystemExit(main())
