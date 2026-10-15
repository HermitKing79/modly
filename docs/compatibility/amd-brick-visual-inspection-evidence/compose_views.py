"""Reopen local PNGs and make labelled private sheets; never changes a mesh."""
import argparse,hashlib,json,os
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageStat
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
PRIVATE=ROOT.parent/'artifacts/amd-brick-visual-inspection-b536fb6';R=PRIVATE/'renders'
LABELS=[('pos-x','+X orthographic'),('neg-x','-X orthographic'),('pos-y','+Y orthographic'),('neg-y','-Y orthographic'),('pos-z','+Z orthographic'),('neg-z','-Z orthographic'),('iso-a','Isometric A: +X -Y +Z'),('iso-b','Isometric B: -X +Y -Z'),('silhouette-pos-z','+Z silhouette: black / white')]
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,r):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
def font(size):return ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',size)
def load(p):
 if p.is_symlink() or p.stat().st_size>16*1024**2:raise ValueError('Plain bounded PNG required')
 with Image.open(p) as im:
  if im.format!='PNG' or im.size!=(1024,1024) or im.mode!='RGB':raise ValueError('Exact render schema required')
  im.load();return im.copy()
def record(p):
 with Image.open(p) as im:size=list(im.size)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p),dimensions=size)
def main(reference_view):
 host=json.loads((R/'host-result.json').read_text());meta=json.loads((R/'render-metadata.json').read_text())
 assert host['classification']=='PASS' and meta['classification']=='PASS' and meta['geometry_unchanged'] and meta['mesh_import_count']==1
 assert meta['geometry_payloads_before']==meta['geometry_payloads_after'] and (meta['vertex_count'],meta['face_count'])==(1957891,4173584)
 assert host['source_unchanged'] and not host['job_cleanup']['forced_job_cleanup'] and host['job_cleanup']['owned_tree_exit_verified']
 assert set(meta['outputs'])=={name for name,label in LABELS}
 before={name:digest(R/x['filename']) for name,x in meta['outputs'].items()}
 assert all(before[name]==x['sha256'] and (R/x['filename']).stat().st_size==x['bytes'] for name,x in meta['outputs'].items())
 sheet=Image.new('RGB',(1440,1612),'#f2f2f2');draw=ImageDraw.Draw(sheet)
 draw.text((20,12),'GENUINE BRICK - UNMODIFIED FULL-RESOLUTION MESH',font=font(25),fill='black')
 draw.text((20,49),'1,957,891 vertices / 4,173,584 triangles | flat clay | Cycles CPU',font=font(20),fill='#222222')
 for i,(name,label) in enumerate(LABELS):
  image=load(R/meta['outputs'][name]['filename']);image.thumbnail((468,468),Image.Resampling.LANCZOS)
  x=(i%3)*480+6;y=88+(i//3)*500
  sheet.paste(image,(x,y));draw.text((x+4,y+470),label,font=font(19),fill='black')
 draw.text((20,1584),'Signs name camera position, not photo front. Images stay local; no repair or learned textures.',font=font(17),fill='black')
 target=PRIVATE/'contact-sheet.png'
 with target.open('xb') as f:sheet.save(f,format='PNG')
 identity=json.loads((PRIVATE/'source-image-identity.json').read_text());source=Path(identity['images'][0]['path']);source_sha=identity['images'][0]['sha256']
 assert digest(source)==source_sha
 with Image.open(source) as image:
  image.load();rgba=image.convert('RGBA');reference=Image.new('RGBA',rgba.size,(64,64,64,255));reference.alpha_composite(rgba);reference=reference.convert('RGB')
 selected=load(R/meta['outputs'][reference_view]['filename'])
 comparison=Image.new('RGB',(1600,890),'#f2f2f2');draw=ImageDraw.Draw(comparison)
 for i,(image,label) in enumerate(((reference,'Authenticated source RGBA (grey background)'),(selected,dict(LABELS)[reference_view]+' - original clay geometry'))):
  image.thumbnail((780,780),Image.Resampling.LANCZOS);x=i*800+(800-image.width)//2
  comparison.paste(image,(x,65+(780-image.height)//2));draw.text((i*800+15,16),label,font=font(22),fill='black')
 draw.text((18,857),'Qualitative comparison only: camera and projection are not calibrated to the photograph.',font=font(20),fill='black')
 compare=PRIVATE/'source-comparison.png'
 with compare.open('xb') as f:comparison.save(f,format='PNG')
 # A diagnostic crop is an image view only; it neither classifies edges nor edits geometry.
 detail=load(R/meta['outputs']['pos-x']['filename']);box=(470,360,660,590)
 crop=detail.crop(box).resize((760,920),Image.Resampling.NEAREST)
 diagnostic=Image.new('RGB',(1300,1000),'#f2f2f2');diagnostic.paste(detail.resize((500,500)),(8,210));diagnostic.paste(crop,(532,50))
 draw=ImageDraw.Draw(diagnostic);draw.rectangle(tuple(int(v*500/1024)+(8 if i%2==0 else 210) for i,v in enumerate(box)),outline='red',width=3)
 draw.text((15,15),'+X surface detail: brick/foliage gaps and thin surfaces',font=font(23),fill='black')
 draw.text((15,950),'4x nearest-neighbour image crop. No mesh edit or spatial edge classification.',font=font(20),fill='black')
 diagnostic_path=PRIVATE/'surface-detail.png'
 with diagnostic_path.open('xb') as f:diagnostic.save(f,format='PNG')
 assert before=={name:digest(R/x['filename']) for name,x in meta['outputs'].items()} and digest(source)==source_sha
 receipt=dict(classification='PASS',images_private_only=True,source_sha256=source_sha,source_unchanged=True,reference_view=reference_view,reference_projection_calibrated=False,render_images={name:record(R/x['filename']) for name,x in meta['outputs'].items()},derived_images={p.name:record(p) for p in (target,compare,diagnostic_path)},render_geometry_unchanged=True,diagnostic_crop_source='+X',diagnostic_crop_box_pixels=box,diagnostic_is_not_edge_incidence_map=True)
 save(PRIVATE/'image-verification.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--reference-view',choices=[x[0] for x in LABELS],default='iso-a');a=p.parse_args();main(a.reference_view)
