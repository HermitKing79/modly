"""Independent bounded CPU check of OBJ faces against the preserved binary mask."""
import hashlib
import json
from pathlib import Path
import struct
import sys
evidence=Path(__file__).parent
def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in ('torch','triton','trellis2','cumesh','flex_gemm','o_voxel'):
        raise RuntimeError('Preview verification prohibits GPU/model imports')
sys.addaudithook(audit)
row=json.loads((evidence/'voxel-preview.json').read_text())
obj=evidence/'voxel-preview.obj'
with obj.open('rb') as f:blob=f.read(12*1024**2+1)
assert len(blob)==row['obj_file_bytes']<=12*1024**2
assert hashlib.sha256(blob).hexdigest()==row['obj_sha256']
vertices=[];faces=[]
for line in blob.decode('ascii').splitlines():
    fields=line.split()
    if not fields or fields[0]=='#':continue
    assert len(fields)==4 if fields[0]=='v' else fields[0]=='f' and len(fields)==5
    if fields[0]=='v':vertices.append(tuple(map(int,fields[1:])))
    else:faces.append(tuple(map(int,fields[1:])))
assert len(vertices)==row['vertices'] and len(faces)==row['faces']<=row['face_limit']
assert len(vertices)==len(set(vertices)) and all(0<=v<=64 for p in vertices for v in p)
raw=(evidence/'occupancy-mask.npy').read_bytes();offset=10+struct.unpack('<H',raw[8:10])[0];mask=raw[offset:]
assert len(mask)==64**3 and set(mask)<={0,1}
expected=set()
for index,occupied in enumerate(mask):
    if not occupied:continue
    point=(index//4096,(index//64)%64,index%64)
    for axis,stride in enumerate((4096,64,1)):
        for sign in (-1,1):
            adjacent=point[axis]+sign
            if not 0<=adjacent<64 or not mask[index+sign*stride]:expected.add((point,axis,sign))
actual=set()
for face in faces:
    assert len(set(face))==4 and all(1<=v<=len(vertices) for v in face)
    points=[vertices[v-1] for v in face]
    spans=[max(p[i] for p in points)-min(p[i] for p in points) for i in range(3)]
    assert sorted(spans)==[0,1,1];axis=spans.index(0)
    a=[points[1][i]-points[0][i] for i in range(3)];b=[points[2][i]-points[0][i] for i in range(3)]
    normal=(a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    assert abs(normal[axis])==1 and sum(abs(v) for v in normal)==1
    origin=[min(p[i] for p in points) for i in range(3)]
    if normal[axis]>0:origin[axis]-=1
    key=(tuple(origin),axis,normal[axis]);assert key not in actual;actual.add(key)
assert actual==expected
png=evidence/'voxel-preview.png';assert png.stat().st_size==row['png_file_bytes']<=2*1024**2
assert hashlib.sha256(png.read_bytes()).hexdigest()==row['png_sha256']
from PIL import Image
with Image.open(png) as image:image.load();assert image.mode=='RGB' and image.size==(900,900)
out=dict(verified=True,exact_exposed_surface=True,vertices=len(vertices),faces=len(faces),unique_faces=True,
    outward_unit_faces=True,png_dimensions=[900,900],obj_sha256=row['obj_sha256'],png_sha256=row['png_sha256'],
    torch_imported='torch' in sys.modules,gpu_initialized=False)
with (evidence/'voxel-preview-independent.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
