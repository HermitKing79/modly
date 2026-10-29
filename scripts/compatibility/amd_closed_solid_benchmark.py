"""Fixed procedural reference specification. No AI or native imports."""
import hashlib,importlib.util,itertools,json,math,os,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT.parent/'artifacts/amd-closed-solid-benchmark-20261029-v1'
CASE=ROOT/'docs/compatibility/amd-repeatable-generation-v1-evidence/new-image/closed-solid-box-v1'
PRIVATE=ROOT.parent/'artifacts/amd-closed-solid-generation-v1'
RAW=ROOT.parent/'artifacts/amd-closed-solid-raw-v1'
REFERENCE=ROOT/'docs/compatibility/amd-closed-solid-reference.json'
DIMENSIONS=(1.0,.65,.8);CHAMFER=.04;SEED=20261013
CAMERA=[2.4,-3.2,2.5];TARGET=[0.,0.,0.];ORTHO=1.7;REFERENCE_RESOLUTION=1024
BLENDER=Path('C:/Program Files/Blender Foundation/Blender 5.2/blender.exe')
BLENDER_SHA='284f4041f98e113f3dc10654a7193ffaaa9bfdfec8b87fa116620a48b5f6d4cb'
PREPARATION_EVIDENCE=('render-run/host-result.json','qualify-run/host-result.json','sources-before.json','sources-after.json',
    'executed-source/amd_closed_solid_benchmark.py','executed-source/amd_closed_solid_cpu.py','executed-source/amd_closed_solid_render.py')
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def save(path,row):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(row,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def geometry():
    """24 exact saved-F32 vertices, 6+12 quads and 8 corner triangles."""
    keys=[(*s,axis) for s in itertools.product((-1,1),repeat=3) for axis in range(3)]
    f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
    vertices=[tuple(f32(k[i]*(DIMENSIONS[i]/2-(0 if i==k[3] else CHAMFER))) for i in range(3)) for k in keys]
    polygons=[]
    for axis in range(3):
        for sign in (-1,1):polygons.append(([i for i,k in enumerate(keys) if k[3]==axis and k[axis]==sign],tuple(sign if j==axis else 0 for j in range(3))))
    for a,b in itertools.combinations(range(3),2):
        for sa,sb in itertools.product((-1,1),repeat=2):polygons.append(([i for i,k in enumerate(keys) if k[3] in (a,b) and k[a]==sa and k[b]==sb],tuple(sa if j==a else sb if j==b else 0 for j in range(3))))
    for signs in itertools.product((-1,1),repeat=3):polygons.append(([i for i,k in enumerate(keys) if k[:3]==signs],signs))
    faces=[]
    for ids,n in polygons:
        center=tuple(sum(vertices[i][j] for i in ids)/len(ids) for j in range(3));axis=min(range(3),key=lambda j:abs(n[j]))
        seed=tuple(int(j==axis) for j in range(3));u=cross(seed,n);w=cross(n,u)
        ids.sort(key=lambda i:math.atan2(dot(sub(vertices[i],center),w),dot(sub(vertices[i],center),u)))
        if dot(cross(sub(vertices[ids[1]],vertices[ids[0]]),sub(vertices[ids[2]],vertices[ids[0]])),n)<0:ids.reverse()
        faces.extend((ids[0],ids[i],ids[i+1]) for i in range(1,len(ids)-1))
    return vertices,faces

def small_topology(v,f):
    edges={};parents=list(range(len(v)));volume=0.;convex=True
    def root(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for tri in f:
        a,b,c=(v[i] for i in tri);n=cross(sub(b,a),sub(c,a));volume+=dot(a,cross(b,c))/6
        if dot(n,n)==0:raise ValueError('Degenerate ground truth')
        convex &= all(dot(n,sub(p,a))<=1e-12 for p in v)
        for j in range(3):
            a,b=tri[j],tri[(j+1)%3];edges.setdefault(tuple(sorted((a,b))),[]).append(1 if a<b else -1);parents[root(a)]=root(b)
    return dict(vertices=len(v),triangles=len(f),components=len({root(i) for i in range(len(v))}),
        boundary=sum(len(x)==1 for x in edges.values()),nonmanifold=sum(len(x)>2 for x in edges.values()),
        winding=sum(len(x)==2 and abs(sum(x))==2 for x in edges.values()),euler=len(v)-len(edges)+len(f),signed_volume=volume,convex=bool(convex))

def cpu_guard(base):
    from amd_zombie_d_handoff import plain
    base=plain(base)
    def audit(event,args):
        if event=='import' and str(args[0]).split('.')[0].lower() in {'torch','triton','trellis2','flex_gemm','o_voxel','cumesh','cupy','nvdiffrast','transformers','diffusers','timm','onnxruntime','comfy'}:raise RuntimeError('CPU benchmark: model/GPU import denied')
        if event in ('subprocess.Popen','os.system','socket.connect','socket.bind','os.fork','os.posix_spawn'):raise RuntimeError('CPU benchmark: child/network denied')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            if any(c in str(args[1]) for c in 'wax+') or args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):plain(os.fsdecode(args[0])).relative_to(base)
        if event=='os.mkdir':plain(args[0]).relative_to(base)
        if event in ('os.remove','os.rename','os.rmdir','os.link','os.symlink','os.truncate'):raise RuntimeError('CPU benchmark: destructive mutation denied')
    return audit
