"""CPU camera-only view of the unchanged verified OBJ, using TRELLIS Z-up."""
import hashlib
import json
import math
from pathlib import Path
from PIL import Image,ImageDraw

evidence=Path(__file__).parent
obj=evidence/'voxel-preview.obj';blob=obj.read_bytes()
assert len(blob)<=12*1024**2
original=json.loads((evidence/'voxel-preview.json').read_text())
assert hashlib.sha256(blob).hexdigest()==original['obj_sha256']
vertices=[];faces=[]
for line in blob.decode('ascii').splitlines():
    fields=line.split()
    if fields and fields[0]=='v':vertices.append(tuple(map(int,fields[1:])))
    elif fields and fields[0]=='f':faces.append(tuple(map(int,fields[1:])))
assert len(faces)==original['faces']<=50000
center=[(min(v[i] for v in vertices)+max(v[i] for v in vertices))/2 for i in range(3)]
yaw=math.pi/4;pitch=math.pi/7
view=(math.sin(yaw)*math.cos(pitch),math.cos(yaw)*math.cos(pitch),math.sin(pitch))
right=(math.cos(yaw),-math.sin(yaw),0)
up=(-math.sin(yaw)*math.sin(pitch),-math.cos(yaw)*math.sin(pitch),math.cos(pitch))
def dot(a,b):return sum(a[i]*b[i] for i in range(3))
points=[]
for v in vertices:
    p=[v[i]-center[i] for i in range(3)]
    points.append((dot(p,right),dot(p,up),dot(p,view)))
scale=740/max(max(p[0] for p in points)-min(p[0] for p in points),max(p[1] for p in points)-min(p[1] for p in points))
image=Image.new('RGB',(900,900),'#f3f5f7');draw=ImageDraw.Draw(image)
for face in sorted(faces,key=lambda f:sum(points[i-1][2] for i in f)):
    world=[vertices[i-1] for i in face]
    a=[world[1][i]-world[0][i] for i in range(3)];b=[world[2][i]-world[0][i] for i in range(3)]
    normal=(a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    facing=dot(normal,view)
    if facing<=0:continue
    shade=int(100+125*facing)
    draw.polygon([(450+points[i-1][0]*scale,450-points[i-1][1]*scale) for i in face],fill=(shade-30,shade-10,shade),outline=(65,75,90))
draw.text((20,20),'Same native 64^3 occupancy / unchanged OBJ / TRELLIS Z-up camera',fill='#1a2530')
target=evidence/'voxel-preview-z-up.png'
with target.open('xb') as f:image.save(f,format='PNG')
assert target.stat().st_size<=2*1024**2
source=Path('D:/AI/Trellis2AMD/ComfyUI/custom_nodes/ComfyUI-Trellis2-AMD/trellis2/utils/render_utils.py')
assert 'torch.tensor([0, 0, 1])' in source.read_text()
row=dict(cpu_only=True,unchanged_obj_sha256=original['obj_sha256'],original_preview_preserved=True,model_or_mask_changes=False,
    camera_up=[0,0,1],yaw_radians=yaw,pitch_radians=pitch,projection='orthographic diagnostic; fixed camera, no model or threshold tuning',
    coordinate_order=['D','H','W'],reference_source=str(source),reference_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    png_path=str(target.resolve()),png_bytes=target.stat().st_size,png_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
with (evidence/'z-up-preview-provenance.json').open('x') as f:json.dump(row,f,indent=2)
sheet=Image.new('RGB',(1200,630),'#f3f5f7');sd=ImageDraw.Draw(sheet)
with Image.open(evidence/'preprocessing-preview.png') as source_image:
    sheet.paste(source_image,(20,60));sd.text((20,20),'Official input / actual 512 conditioning preview',fill='#1a2530')
sheet.paste(image.resize((600,600),Image.Resampling.LANCZOS),(580,30))
sd.text((580,10),'Native occupancy voxel preview / Z up',fill='#1a2530')
with (evidence/'comparison-preview.png').open('xb') as f:sheet.save(f,format='PNG')
print(json.dumps(row))
