"""Bounded CPU-only latent reader, native occupancy artifacts and exposed cells."""
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import amd_sparse_latent_checkpoint as latent

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'docs/compatibility/amd-sparse-structure-sampling-evidence/sparse-structure-latent.safetensors'
INPUT_SHA='8a4da87ae1bbc2d5e5186f300adf11597bf986cb58dfb24a564a2eec287b44f2'

def read_input(path=INPUT,*,expected=None):
    from safetensors.numpy import load
    path=Path(path)
    with path.open('rb') as f:blob=f.read(262145)
    if expected is not None and expected.get('verified') is not True:
        raise ValueError('Generated latent requires a verified artifact reference')
    wanted=INPUT_SHA if expected is None else expected['file_sha256']
    if len(blob)>262144 or hashlib.sha256(blob).hexdigest()!=wanted:raise ValueError('Committed sparse latent SHA/size mismatch')
    old=json.loads((INPUT.parent/'resume-result-analysis.json').read_text()) if expected is None else expected
    provenance={k:old['provenance'][k] for k in latent.FIELDS}
    row=latent.verify(path,path.parent,provenance,old['payload_sha256'] if expected is None else old['output_sha256'])
    array=load(blob)['sparse_latent'].copy()
    return array,row

def components(mask):
    import numpy as np
    remaining=set(int(v) for v in np.flatnonzero(mask));sizes=[]
    while remaining:
        seed=remaining.pop();queue=deque([seed]);size=0
        while queue:
            value=queue.popleft();size+=1;d,h,w=value//4096,(value//64)%64,value%64
            neighbors=[]
            if d:neighbors.append(value-4096)
            if d<63:neighbors.append(value+4096)
            if h:neighbors.append(value-64)
            if h<63:neighbors.append(value+64)
            if w:neighbors.append(value-1)
            if w<63:neighbors.append(value+1)
            for neighbor in neighbors:
                if neighbor in remaining:remaining.remove(neighbor);queue.append(neighbor)
        sizes.append(size)
    return dict(connectivity=6,count=len(sizes),largest_sizes=sorted(sizes,reverse=True)[:20],total_occupied=sum(sizes))

def write_output(scratch,logits,provenance):
    import numpy as np
    scratch=Path(scratch).resolve(strict=True)
    if (type(logits) is not np.ndarray or logits.shape!=(1,1,64,64,64) or logits.dtype!=np.dtype('<f4')
        or not logits.flags.c_contiguous or not np.isfinite(logits).all()):raise ValueError('Require finite contiguous native F32[1,1,64,64,64] logits')
    mask=logits[0,0]>0;coords=np.argwhere(mask).astype('<i4')
    coords=np.concatenate((np.zeros((len(coords),1),dtype='<i4'),coords),axis=1)
    files={}
    for name,array,limit in [('occupancy-logits.npy',logits,2*1024**2),('occupancy-mask.npy',mask,512*1024),('occupancy-coordinates.npy',coords,5*1024**2)]:
        path=scratch/name
        with path.open('xb') as f:np.save(f,array,allow_pickle=False);f.flush();os.fsync(f.fileno())
        if path.stat().st_size>limit:raise ValueError('Occupancy file exceeds limit: '+name)
        restored=np.load(path,allow_pickle=False,max_header_size=4096)
        if restored.dtype!=array.dtype or restored.shape!=array.shape or not np.array_equal(restored,array):raise ValueError('Occupancy readback differs: '+name)
        files[name]=dict(path=str(path),file_bytes=path.stat().st_size,file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),payload_sha256=hashlib.sha256(array.tobytes()).hexdigest(),shape=list(array.shape),dtype=str(array.dtype))
    count=len(coords);bbox=None if count==0 else dict(minimum=coords[:,1:].min(axis=0).tolist(),maximum=coords[:,1:].max(axis=0).tolist())
    degenerate=count in (0,262144) or any(a==b for a,b in zip(bbox['minimum'],bbox['maximum'])) if bbox else True
    row=dict(native_shape=[1,1,64,64,64],threshold='decoded > 0',coordinate_order=['batch','D','H','W'],
        occupied_count=count,occupancy_fraction=count/262144,bounding_box=bbox,empty=count==0,degenerate=degenerate,
        components=components(mask),files=files,provenance=provenance,readback_verified=True)
    with (scratch/'occupancy-provenance.json').open('x',encoding='utf-8') as f:json.dump(row,f,indent=2);f.flush();os.fsync(f.fileno())
    assert json.loads((scratch/'occupancy-provenance.json').read_text())==row
    return row

def surface(mask,face_limit=50000):
    import numpy as np
    if mask.shape!=(64,64,64) or mask.dtype!=bool:raise ValueError('Native binary mask required')
    corners=[[(0,0,0),(0,0,1),(0,1,1),(0,1,0)],[(1,0,0),(1,1,0),(1,1,1),(1,0,1)],
        [(0,0,0),(1,0,0),(1,0,1),(0,0,1)],[(0,1,0),(0,1,1),(1,1,1),(1,1,0)],
        [(0,0,0),(0,1,0),(1,1,0),(1,0,0)],[(0,0,1),(1,0,1),(1,1,1),(0,1,1)]]
    vertices=[];faces=[];ids={}
    for axis in range(3):
        for sign in (-1,1):
            shifted=np.zeros_like(mask);target=[slice(None)]*3;source=[slice(None)]*3
            target[axis]=slice(1,None) if sign==-1 else slice(None,-1)
            source[axis]=slice(None,-1) if sign==-1 else slice(1,None)
            shifted[tuple(target)]=mask[tuple(source)]
            for cell in np.argwhere(mask & ~shifted):
                if len(faces)>=face_limit:raise ValueError('Exposed voxel surface exceeds face limit')
                face=[]
                for offset in corners[2*axis+(sign==1)]:
                    point=tuple(int(cell[i])+offset[i] for i in range(3))
                    if point not in ids:ids[point]=len(vertices)+1;vertices.append(point)
                    face.append(ids[point])
                faces.append(face)
    return vertices,faces

def write_preview(directory,mask):
    """Exact exposed cell faces; bounded ASCII OBJ and CPU raster projection."""
    import math
    from PIL import Image,ImageDraw
    directory=Path(directory).resolve(strict=True);vertices,faces=surface(mask)
    if not faces:raise ValueError('Empty occupancy has no voxel preview')
    lines=['# Diagnostic exposed voxels, native64cubed. OBJ x/y/z = tensor D/H/W.']
    lines += ['v %d %d %d'%v for v in vertices]
    lines += ['f '+' '.join(map(str,f)) for f in faces]
    blob=('\n'.join(lines)+'\n').encode('ascii')
    if len(blob)>12*1024**2:raise ValueError('Voxel OBJ exceeds12MiB cap')
    obj=directory/'voxel-preview.obj'
    with obj.open('xb') as f:f.write(blob)
    assert obj.read_bytes()==blob
    center=[(min(v[i] for v in vertices)+max(v[i] for v in vertices))/2 for i in range(3)]
    points=[];yaw=math.pi/4;pitch=math.pi/7
    for v in vertices:
        x,y,z=[v[i]-center[i] for i in range(3)]
        x,z=x*math.cos(yaw)+z*math.sin(yaw),-x*math.sin(yaw)+z*math.cos(yaw)
        y,z=y*math.cos(pitch)-z*math.sin(pitch),y*math.sin(pitch)+z*math.cos(pitch)
        points.append((x,y,z))
    scale=740/max(max(v[0] for v in points)-min(v[0] for v in points),max(v[1] for v in points)-min(v[1] for v in points))
    image=Image.new('RGB',(900,900),'#f3f5f7');draw=ImageDraw.Draw(image)
    for face in sorted(faces,key=lambda f:sum(points[i-1][2] for i in f)/4):
        p=[points[i-1] for i in face];a=[p[1][i]-p[0][i] for i in range(3)];b=[p[2][i]-p[0][i] for i in range(3)]
        normal=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
        if normal[2]<=0:continue
        shade=int(100+125*normal[2]/math.sqrt(sum(v*v for v in normal)))
        draw.polygon([(450+v[0]*scale,450-v[1]*scale) for v in p],fill=(shade-30,shade-10,shade),outline=(65,75,90))
    draw.text((20,20),'Native 64^3 occupancy: diagnostic exposed voxel faces',fill='#1a2530')
    png=directory/'voxel-preview.png'
    with png.open('xb') as f:image.save(f,format='PNG')
    if png.stat().st_size>2*1024**2:raise ValueError('Voxel PNG exceeds2MiB cap')
    with Image.open(png) as restored:restored.load();assert restored.size==(900,900)
    return dict(vertices=len(vertices),faces=len(faces),face_limit=50000,obj_file_bytes=len(blob),obj_sha256=hashlib.sha256(blob).hexdigest(),
        obj_path=str(obj),png_path=str(png),png_file_bytes=png.stat().st_size,png_sha256=hashlib.sha256(png.read_bytes()).hexdigest(),
        coordinate_order=['D','H','W'],diagnostic_only=True,cpu_only=True)
