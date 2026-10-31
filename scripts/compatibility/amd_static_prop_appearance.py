"""Phase20D: read-only asset authentication and four one-shot CPU comparisons."""
import sys,os,json,math,hashlib,subprocess
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import amd_static_prop_uv_improve as prior
old=prior.old;owner=prior.owner;HISTORICAL=prior.BASE
INPUTS=prior.HISTORICAL
BASE=HISTORICAL.parent/'amd-static-prop-appearance-20261031-v1'
PARENT='9fa586d222f8fba81e6782dff38a458d07aaed80'
GLB=HISTORICAL/'asset/experimental-static-prop.glb'
GLB_SHA='afc880ff5ec2e0c1a9a6977999fa633506b68915c73eddb2c543ee17512c1201'
RENDERS=('matte-detail37','glossy-detail37','matte-detail8','glossy-detail8')
STAGES=('authenticate','diagnose',*RENDERS,'verify','report','preserve')

def bind():
 prior.BASE=BASE;prior.prior.BASE=BASE;old.BASE=BASE

def snapshot(root):
 return {p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,file_id=p.stat().st_ino,sha256=owner.sha(p)) for p in root.rglob('*') if p.is_file()}

def authenticate(preserve=False):
 prior.authenticate(preserve)
 if preserve:rows=json.loads((BASE/'phase20C-inputs.json').read_text())
 else:
  for name,row in json.loads((HISTORICAL/'evidence-manifest.json').read_text()).items():old.check(HISTORICAL/name,row)
  rows=snapshot(HISTORICAL);owner.save(BASE/'phase20C-inputs.json',rows)
 for name,row in rows.items():
  path=HISTORICAL/name;old.check(path,row);s=path.stat()
  if (s.st_size,s.st_mtime_ns,s.st_ino)!=(row['bytes'],row['mtime_ns'],row['file_id']):raise ValueError('Historical identity changed')
 if owner.sha(GLB)!=GLB_SHA:raise ValueError('Qualified GLB identity')
 manifest=json.loads((HISTORICAL/'final-manifest.json').read_text())
 if manifest['classification']!='PASS_UV_EXPORT_USER_APPEARANCE_PENDING' or manifest['research']!='PERMANENT_NO_GO':raise ValueError('Qualified output state changed')
 owner.save(BASE/('phase20C-preservation.json' if preserve else 'phase20C-authentication.json'),dict(classification='PASS',files=len(rows),GLB_sha256=GLB_SHA,research='PERMANENT_NO_GO',historical_execution_hold=old.LIVE_EXECUTION_HOLD))

def image_stats(np,path):
 w,h,c,raw=old.png_pixels(path);a=np.frombuffer(raw,'u1').reshape(h,w,c)[:,:,:3].astype('f8');l=a@np.array([.2126,.7152,.0722])
 return dict(width=w,height=h,sha256=owner.sha(path),luminance_quantiles={str(q):float(np.quantile(l,q)) for q in (0,.05,.5,.95,1)},luminance_std=float(l.std()),near_white_fraction=float(np.all(a>=250,axis=2).mean()),all_255_fraction=float(np.all(a==255,axis=2).mean()),near_black_fraction=float(np.all(a<=5,axis=2).mean()))

def ray(np,v,f,origin,direction):
 """Two-sided first hit; one ray proves visibility only along that ray."""
 origin=np.asarray(origin,'f8');direction=np.asarray(direction,'f8');direction/=np.linalg.norm(direction);best=None
 for start in range(0,len(f),100000):
  t=v[f[start:start+100000]].astype('f8');e1=t[:,1]-t[:,0];e2=t[:,2]-t[:,0];p=np.cross(direction,e2);det=(e1*p).sum(1);valid=abs(det)>1e-18
  inv=np.zeros_like(det);inv[valid]=1/det[valid];s=origin-t[:,0];u=(s*p).sum(1)*inv;q=np.cross(s,e1);vv=(q*direction).sum(1)*inv;d=(q*e2).sum(1)*inv
  ids=np.where(valid&(u>=-1e-8)&(vv>=-1e-8)&(u+vv<=1+1e-8)&(d>0))[0]
  if len(ids):
   i=ids[np.argmin(d[ids])]
   if best is None or d[i]<best['distance']:best=dict(face=int(start+i),distance=float(d[i]),position=(origin+d[i]*direction).tolist())
 return best

def basis(np,view,right,up):
 m=np.array([right,up,view],dtype='f8')
 if not np.allclose(m@m.T,np.eye(3),atol=1e-8) or np.linalg.det(m)<.999999:raise ValueError('Right-handed orthonormal camera required')
 return m

def diagnose():
 import numpy as np
 spec=json.loads((INPUTS/'inputs.json').read_text());stats={}
 for material in ('matte','glossy'):
  for view in ('whole45','whole225','detail37','detail8'):
   for name in ('source','raw','final'):
    path=INPUTS/f'{material}-{view}'/(name+'.png');stats[path.relative_to(INPUTS).as_posix()]=image_stats(np,path)
 oldcams=json.loads((old.prior.BASE/'detail-review/cameras.json').read_text())
 v=np.load(INPUTS/'inputs/source-v.npy',allow_pickle=False);f=np.load(INPUTS/'inputs/source-f.npy',allow_pickle=False);D=spec['D'];cameras={};visibility={}
 for region in (37,8):
  c=next(x for x in oldcams if x['region']==region);oldcam=spec['cameras'][f'detail{region}'];view=np.array(c['view']);right=np.array(c['right']);up=np.array(c['up']);m=basis(np,view,right,up)
  target=np.array(oldcam['target']);loc=np.array(oldcam['location']);face=int(c['representative_face']);tri=v[f[face]].astype('f8');centroid=tri.mean(0)
  # Freeze original source view AND full roll basis; widen to 0.006D for context.
  span=max(float(oldcam['span']),.006*D);loc=target+4*D*view
  corners=(tri-target)@m.T;source_centroid_hit=ray(np,v,f,centroid+4*D*view,-view)
  visibility[str(region)]=dict(representative_face=face,old_center_ray=ray(np,v,f,np.array(oldcam['location']),target-np.array(oldcam['location'])),source_centroid_ray=source_centroid_hit,source_centroid_occlusion_gap=None if source_centroid_hit is None else 4*D-source_centroid_hit['distance'],scope='Two single rays; not complete ROI occlusion certification',representative_triangle_projected_coordinates=corners.tolist(),projected_pixels=((corners[:,:2]/span+.5)*768).tolist(),old_roll_basis_discarded=True)
  cameras[f'detail{region}']=dict(target=target.tolist(),location=loc.tolist(),view=view.tolist(),right=right.tolist(),up=up.tolist(),span=span,resolution=768,source_region=region,representative_face=face)
 owner.save(BASE/'diagnosis.json',dict(classification='READ_ONLY_DIAGNOSIS_COMPLETE',historical_images=stats,visibility=visibility,cameras=cameras,old_settings=dict(transform='Standard',exposure=0,samples=16,lights='Fixed world-space Key500D2 Fill150D2 Rim350D2'),geometry_modified=False))

def approved(stage):
 if stage in ('authenticate','diagnose','preserve'):return
 name='pilot-review.json' if stage==RENDERS[0] else 'continuation-review.json' if stage in RENDERS[1:] else 'verification-review.json'
 r=json.loads((BASE/name).read_text())
 if r['classification']!='APPROVED' or r['parent']!=PARENT or stage not in r['allowed_stages']:raise ValueError('Reviewed stage required')
 for path,h in r['files'].items():
  if owner.sha(Path(path))!=h:raise ValueError('Reviewed source/settings changed')
 if stage in RENDERS[1:]:
  old.success(RENDERS[0])
  if r.get('pilot_useful') is not True:raise ValueError('Useful pilot required')

def verify():
 import numpy as np
 rows={};plan=json.loads((BASE/'render-plan.json').read_text())
 for stage in RENDERS:
  old.success(stage);folder=BASE/stage;r=json.loads((folder/'appearance.json').read_text())
  if r['camera']!=plan['cameras'][stage.split('-',1)[1]] or r['geometry_modified'] or r['GLB_sha256']!=GLB_SHA:raise ValueError('Actual comparison identity')
  for name in ('source','raw','final'):
   if r['before'][name]!=r['after'][name]:raise ValueError('Geometry/UV/normal changed')
   path=folder/(name+'.png')
   if owner.sha(path)!=r['images'][name]:raise ValueError('Rendered image changed')
   rows[str(path.relative_to(BASE))]=image_stats(np,path)
 # Recheck all 24 reused images against diagnosed identities.
 for name,row in json.loads((BASE/'diagnosis.json').read_text())['historical_images'].items():
  if owner.sha(INPUTS/name)!=row['sha256']:raise ValueError('Reused image changed')
 owner.save(BASE/'image-verification.json',dict(classification='PASS_MATCHED_INPUT_OUTPUT_IDENTITIES',images=rows,new_images=len(rows),historical_images_rechecked=24,GLB_sha256=owner.sha(GLB),geometry_modified=False,normal_bake='SKIPPED',research='PERMANENT_NO_GO',appearance='USER_APPEARANCE_APPROVAL_PENDING'))

def report():
 """Local-only review, no remote resources and no additional rendering."""
 import numpy as np
 from html import escape
 from amd_mesh_aware_result import png
 from urllib.parse import quote
 old.success('verify');assessment=json.loads((BASE/'visual-assessment.json').read_text());folder=BASE/'review';folder.mkdir(exist_ok=False)
 labels=('Original AI source · 2,047,210 triangles','Raw Alpha Wrap · 2,808 triangles','Final existing GLB · 1,000 triangles')
 spec=json.loads((INPUTS/'inputs.json').read_text());close=json.loads((BASE/'render-plan.json').read_text())['cameras']
 def link(path):return quote(os.path.relpath(path,folder).replace('\\','/'),safe='/.-')
 def locator(path):
  if path.name!='source.png' or path.parent.name.split('-',1)[1] not in ('whole45','whole225'):return ''
  c=spec['cameras'][path.parent.name.split('-',1)[1]];forward=np.array(c['target'])-np.array(c['location']);forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward);marks=[]
  for region,color in ((37,'#ffce62'),(8,'#74e5ff')):
   delta=np.array(close[f'detail{region}']['target'])-np.array(c['target']);x=512*(.5+delta@right/c['span']);y=512*(.5-delta@up/c['span'])
   marks.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="9" fill="none" stroke="{color}" stroke-width="2"/><text x="{x+12:.3f}" y="{y-8:.3f}" fill="{color}" stroke="#111" stroke-width="3" paint-order="stroke" font-size="19">{region}</text>')
  return '<svg class="locator" viewBox="0 0 512 512" aria-label="Projected source locations 37 and 8">'+''.join(marks)+'</svg>'
 def comparison(title,paths,caption):
  actual_labels=labels[:2]+(('Saved final mesh · 1,000 triangles',) if all(path.is_relative_to(INPUTS) for path in paths) else labels[2:])
  return '<section><h3>'+escape(title)+'</h3><div class="grid">'+''.join('<figure><figcaption>'+escape(label)+'</figcaption><a class="zoom imagewrap" href="'+link(path)+'"><img src="'+link(path)+'" alt="'+escape(title+' / '+label)+'" loading="lazy">'+locator(path)+'</a></figure>' for label,path in zip(actual_labels,paths))+'</div><p>'+escape(caption)+'</p></section>'
 parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Modly static prop — appearance review</title>
<style>body{margin:auto;max-width:1500px;padding:24px;background:#151a22;color:#e8edf5;font:17px/1.55 system-ui,sans-serif}a{color:#a8d9ff}h1{line-height:1.15}nav{display:flex;gap:20px;flex-wrap:wrap;position:sticky;top:0;background:#151a22;padding:12px;z-index:1}.notice{border-left:5px solid #ffc56d;padding:14px 20px;background:#252d39}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}figcaption{font-size:15px;padding:8px;background:#273242}img{width:100%;display:block}section{margin:28px 0}code{overflow-wrap:anywhere}button{font:inherit;padding:8px;cursor:pointer}dialog{width:95vw;height:92vh;max-width:none;max-height:none;background:#151a22;color:white;border:1px solid #789;padding:12px}dialog::backdrop{background:#000c}#viewport{overflow:auto;height:calc(100% - 90px);cursor:grab}#large{width:auto;max-width:none;transform-origin:top left}#caption{margin:6px 0;font-size:14px}summary{cursor:pointer}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #526175}@media(max-width:750px){body{padding:10px}.grid{grid-template-columns:1fr}nav{position:static}}</style>
<h1>LOWER-DETAIL STATIC PROP<br>Appearance review</h1><p class="notice"><strong>USER_APPEARANCE_APPROVAL_PENDING</strong><br>AI-DERIVED RECONSTRUCTION. The original Phase 11 high-fidelity benchmark remains permanently <strong>NO-GO</strong>. This is a separate optional static-prop profile for this saved candidate, not a fully game-ready asset or approval for organic objects.</p>
<nav><a href="#context">Whole-object context</a><a href="#region37">Microfacet strip</a><a href="#region8">Small corner facet</a><a href="#tradeoff">Trade-off / decision</a><a href="#blender">Open GLB</a></nav>
<p>Read each row left to right: original AI surface → raw wrap → saved final GLB. Click any image for a larger view; use zoom buttons and drag to pan. The whole-object views are reused evidence. Close-ups magnify tiny differences and do not represent a normal gameplay distance. No geometry, UV or normal data was changed.</p><h2 id="context">Whole-object context — reused Phase 20B views</h2><p>These existing views preserve useful silhouette context. Their bright tops are clipped; they are not a high-dynamic-range highlight evaluation. Distances are screen-space framing, not a calibrated physical viewing distance.</p>''']
 parts.append('<style>.imagewrap{display:block;position:relative}.locator{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}</style><p>Source-panel markers 37 (upper side bevel) and 8 (lower side bevel) locate the close-ups by projecting their source target points. A marker can lie on a far-side surface; it does not certify visibility. Markers are HTML overlays, not changes to the saved images. Close-up fields span 0.006 source diagonal versus 1.15 in these whole views: approximately 192× spatial magnification at equal display width.</p>')
 for kind in ('matte','glossy'):
  for view in ('whole45','whole225'):
   parts.append(comparison(f'{kind.title()} · {view}',[INPUTS/f'{kind}-{view}'/(x+'.png') for x in ('source','raw','final')],'Historical 512×512 images; same camera, light, exposure and material within this comparison. The final panel here is the authenticated saved geometry, before GLB export.'))
 for region in (37,8):
  parts.append(f'<h2 id="region{region}">Region {region} — '+('coherent microfacet strip' if region==37 else 'small corner facet')+'</h2>')
  for kind in ('matte','glossy'):
   parts.append(comparison(f'{kind.title()} · region {region}',[BASE/f'{kind}-detail{region}'/(x+'.png') for x in ('source','raw','final')],assessment['comparisons'][f'{kind}-detail{region}']))
 parts.append('<h2 id="tradeoff">Documented loss and interpretation</h2><p>'+escape(assessment['tradeoff'])+'</p><p>'+escape(assessment['decision'])+'</p><p>Matte roughness 0.8 / glossy roughness 0.12; metallic 0; shared neutral gray. These are RAM-only review materials, not AI textures. Close-ups use identical orthographic camera, full saved roll basis, scale, lighting, AgX display transform and exposure within each triplet. Source/raw use their historical geometric flat-face presentation; the actual imported GLB retains its existing corner normals and UVs. Source winding remains unqualified; two-sided shading is used consistently.</p>')
 parts.append('<h2 id="blender">Inspect the existing GLB in Blender</h2><ol><li>Open Blender manually. Choose <strong>File → Import → glTF 2.0 (.glb/.gltf)</strong>.</li><li>Select <code>'+escape(str(GLB))+'</code>.</li><li>Select the imported object and press <strong>Numpad .</strong> (Frame Selected). Use Material Preview to inspect the existing neutral test material.</li><li>Do not save over original assets. This instruction launches no application and performs no export.</li></ol><p>Existing GLB SHA-256: <code>'+GLB_SHA+'</code>. Normal baking SKIPPED. S&amp;box NOT TESTED. Finished PBR, production UV padding/mips, engine units/collision/LOD, intended appearance approval and distribution licensing remain separate requirements.</p>')
 parts.append('''<details><summary>Why the old close-ups failed</summary><p>Region 37 source matte/glossy panels had 91.0% / 90.5% pure-white pixels. Region 8 was not clipped, but its source luminance standard deviation was only 0.82 / 0.50 on a 0–255 scale. Standard tone mapping and fixed world-space lights gave inappropriate exposure/contrast for these tight views. The earlier bridge also discarded the source camera roll basis. Two tested centre rays hit the intended source faces; that limited test does not certify visibility across the whole region. The new single frozen lighting setup and full basis were validated by one pilot before the remaining comparisons.</p></details>
<dialog id="viewer"><div><button id="close">Close (Esc)</button> <button id="minus">−</button> <button id="plus">+</button> <button id="fit">Fit width</button> <button id="native">100%</button> <a id="original" target="_blank">Open image</a></div><p id="caption"></p><div id="viewport"><img id="large" alt="Zoomed comparison"></div></dialog>
<script>const dlg=document.getElementById('viewer'),im=document.getElementById('large'),vp=document.getElementById('viewport');let scale=1;function size(s){scale=Math.max(.1,Math.min(8,s));im.style.width=(im.naturalWidth*scale)+'px'}document.querySelectorAll('a.zoom').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();im.src=a.href;im.onload=()=>size(Math.min(1,vp.clientWidth/im.naturalWidth));document.getElementById('caption').textContent=a.querySelector('img').alt;document.getElementById('original').href=a.href;dlg.showModal()}));document.getElementById('close').onclick=()=>dlg.close();document.getElementById('plus').onclick=()=>size(scale*1.5);document.getElementById('minus').onclick=()=>size(scale/1.5);document.getElementById('fit').onclick=()=>size(vp.clientWidth/im.naturalWidth);document.getElementById('native').onclick=()=>size(1);let drag=null;vp.onpointerdown=e=>{if(e.button===0){drag=[e.clientX,e.clientY,vp.scrollLeft,vp.scrollTop];vp.setPointerCapture(e.pointerId);e.preventDefault()}};vp.onpointermove=e=>{if(drag){vp.scrollLeft=drag[2]+drag[0]-e.clientX;vp.scrollTop=drag[3]+drag[1]-e.clientY}};vp.onpointerup=()=>drag=null;</script></html>''')
 with (folder/'index.html').open('x',encoding='utf8') as out:out.write('\n'.join(parts))
 sheet=np.empty((4*384,3*384,3),'u1')
 for row,stage in enumerate(RENDERS):
  for col,name in enumerate(('source','raw','final')):
   w,h,c,data=old.png_pixels(BASE/stage/(name+'.png'));a=np.frombuffer(data,'u1').reshape(h,w,c)[:,:,:3];sheet[row*384:(row+1)*384,col*384:(col+1)*384]=a[::2,::2]
 png(folder/'closeup-sheet.png',np,sheet)
 owner.save(BASE/'review-manifest.json',dict(classification='PASS_REVIEW_EVIDENCE',appearance=assessment['appearance'],user_approval='USER_APPEARANCE_APPROVAL_PENDING',research='PERMANENT_NO_GO',geometry_modified=False,GLB_sha256=owner.sha(GLB),normal_bake='SKIPPED',Sbox='NOT_TESTED',game_ready=False,review_files=snapshot(folder)))

def worker(stage):
 if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO and EOF required')
 bind();sys.addaudithook(prior.prior._ORIGINAL_AUDIT(stage));approved(stage)
 if stage in ('authenticate','preserve'):authenticate(stage=='preserve');return
 old.success('authenticate');sys.path.insert(0,str(owner.CPU_SITE))
 if stage=='diagnose':diagnose()
 elif stage=='verify':verify()
 elif stage=='report':report()
 else:raise ValueError('No general execution route')

def main(argv=None):
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=STAGES);p.add_argument('--worker',choices=STAGES);a=p.parse_args(argv)
 if a.worker:worker(a.worker);return 0
 if not a.stage:print('INERT: Phase20D reviewed CPU appearance evidence only');return 0
 bind();approved(a.stage)
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=owner.ROOT,text=True,timeout=5).strip()!=PARENT:raise ValueError('Exact preparation parent required; historical runner inert after publication')
 scratch=BASE/'scratch'/a.stage;scratch.mkdir(parents=True,exist_ok=False)
 owner.BASE=BASE;owner.__file__=__file__;launch=subprocess.Popen
 def routed(command,**kwargs):
  expected=[sys._base_executable,'-B','-I','-S',str(Path(__file__).resolve()),'--worker',a.stage]
  if command!=expected:raise ValueError('Exact CPU route required')
  if a.stage in RENDERS:command=[str(old.BLENDER),'--background','--factory-startup','--disable-autoexec','--threads','4','--python-exit-code','1','--python',str(Path(__file__).with_name('amd_static_prop_appearance_blender.py')),'--',a.stage]
  kwargs['env']=dict(kwargs['env'],TEMP=str(scratch),TMP=str(scratch));return launch(command,**kwargs)
 owner.subprocess=SimpleNamespace(Popen=routed,PIPE=subprocess.PIPE)
 return owner.supervise(a.stage)
if __name__=='__main__':raise SystemExit(main())
