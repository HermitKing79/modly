"""Bounded read-only investigation of the preserved brick panel's four edges."""
from fractions import Fraction as Q
from pathlib import Path
import ast,hashlib,io,json,math,struct,time

ROOT=Path(__file__).resolve().parents[2]
PANEL=ROOT.parent/'artifacts/amd-brick-seam-uv-bake-87f3755'
MASTER=ROOT.parent/'artifacts/amd-brick-cpu-mesh-8473579/mesh'
DECODE=ROOT.parent/'artifacts/amd-brick-neural-completion-c9447dd/neural-output/shape-decoded.safetensors'
PINS={'prepare/patch.json':'b527aee6a88c08f34b32dac755bddadba2f22fc6c32430ef39a134444e788c63',
 'qualify/uv.json':'86d0c01d54966df891b1ebe027c4bcdbcbc976a3e22bd4411f341e1bc8bd7229',
 'bake/bake.json':'7f06d75e55c0435bea68539b4b7d10194f0677486839d0cc7960d193a325e481',
 'render-qualified/export/patch.glb':'e618af9fea32ce9fedb1e478ca1e0becb7eb55fc9ee4a246bcdf7105a6cf9c21'}
MASTER_PINS={'vertices.npy':'018edd85646f9b504ec52ceb311f69e0c7610e4d1ab306f41c7e73a7a77d3252',
 'faces.npy':'9314a921741bab5c3d87348b4ea26488ee1ab13201b33d67252d6d272997ab1d'}

def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def orient2(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def project(p,axis):return tuple(x for i,x in enumerate(p) if i!=axis)

def strict_inside(p,triangle,normal):
    axis=max(range(3),key=lambda k:abs(normal[k]));t=[project(x,axis) for x in triangle];q=project(p,axis)
    values=[orient2(t[i],t[(i+1)%3],q) for i in range(3)]
    return all(x>0 for x in values) or all(x<0 for x in values)

def plane_cut(triangle,distances):
    points={tuple(Q(x) for x in p) for p,d in zip(triangle,distances) if d==0}
    for i in range(3):
        j=(i+1)%3
        if distances[i]*distances[j]<0:
            ratio=Q(distances[i])/Q(distances[i]-distances[j])
            points.add(tuple(Q(triangle[i][k])+ratio*(triangle[j][k]-triangle[i][k]) for k in range(3)))
    return sorted(points)

def coplanar_clip(a,b,normal):
    axis=max(range(3),key=lambda k:abs(normal[k]));poly=[tuple(Q(x) for x in project(p,axis)) for p in a];clip=[project(p,axis) for p in b]
    direction=1 if orient2(*clip)>0 else -1
    for i in range(3):
        if not poly:break
        left,right=clip[i],clip[(i+1)%3];result=[];previous=poly[-1];pd=direction*orient2(left,right,previous)
        for current in poly:
            cd=direction*orient2(left,right,current)
            if (pd>=0)!=(cd>=0):
                ratio=Q(pd)/Q(pd-cd);result.append(tuple(previous[k]+ratio*(current[k]-previous[k]) for k in range(2)))
            if cd>=0:result.append(current)
            previous,pd=current,cd
        poly=list(dict.fromkeys(result))
    area=abs(sum(poly[i][0]*poly[(i+1)%len(poly)][1]-poly[i][1]*poly[(i+1)%len(poly)][0] for i in range(len(poly)))) if poly else 0
    points=[]
    for p in poly:
        v=[];j=0
        for k in range(3):
            if k==axis:v.append(None)
            else:v.append(p[j]);j+=1
        v[axis]=Q(a[0][axis])-sum(normal[k]*(v[k]-a[0][k]) for k in range(3) if k!=axis)/Q(normal[axis]);points.append(tuple(v))
    dimension=2 if area else (1 if len(set(points))>1 else 0 if points else -1)
    return points,dimension

def on_segment(p,a,b):
    delta=sub(b,a);return cross(sub(p,a),delta)==(0,0,0) and 0<=dot(sub(p,a),delta)<=dot(delta,delta)

def triangle_relation(a,b):
    """Exact predicate for integer/rational coordinates; no eps or repair.

    Production inputs are the exact saved F32 positions lifted to one common
    integer lattice. Contacts along a shared simplex are not reported as
    interior self-intersections. Coplanar area is tested exactly as well.
    """
    a=[tuple(p) for p in a];b=[tuple(p) for p in b]
    if len(a)!=3 or len(b)!=3 or any(len(p)!=3 or any(not isinstance(x,(int,Q)) for x in p) for p in a+b):raise ValueError('Three exact integer/rational3D points required')
    na=cross(sub(a[1],a[0]),sub(a[2],a[0]));nb=cross(sub(b[1],b[0]),sub(b[2],b[0]))
    if na==(0,0,0) or nb==(0,0,0):raise ValueError('Degenerate input triangle')
    if set(a)==set(b):return dict(kind='coincident_triangle',dimension=2,coplanar=True,points=a)
    empty=dict(kind='disjoint',dimension=-1,coplanar=None,points=[])
    if any(max(p[k] for p in a)<min(p[k] for p in b) or max(p[k] for p in b)<min(p[k] for p in a) for k in range(3)):return empty
    da=[dot(nb,sub(p,b[0])) for p in a];db=[dot(na,sub(p,a[0])) for p in b]
    if any(all(x>0 for x in ds) or all(x<0 for x in ds) for ds in (da,db)):return empty
    coplanar=all(x==0 for x in da) and all(x==0 for x in db)
    if coplanar:points,dimension=coplanar_clip(a,b,na)
    else:
        pa,pb=plane_cut(a,da),plane_cut(b,db)
        if not pa or not pb:return empty
        direction=cross(na,nb);axis=max(range(3),key=lambda k:abs(direction[k]))
        if not direction[axis]:raise RuntimeError('Inconsistent exact plane intersection')
        low=max(min(p[axis] for p in pa),min(p[axis] for p in pb));high=min(max(p[axis] for p in pa),max(p[axis] for p in pb))
        if low>high:return empty
        def locate(value):
            if len(pa)==1:return pa[0]
            p,q=pa[0],pa[-1];ratio=Q(value-p[axis])/Q(q[axis]-p[axis]);return tuple(p[k]+ratio*(q[k]-p[k]) for k in range(3))
        points=list(dict.fromkeys([locate(low),locate(high)]));dimension=1 if high>low else 0
    if dimension<0:return empty
    shared=sorted(set(a)&set(b));covered=False
    if len(shared)==1:covered=all(p==shared[0] for p in points)
    if len(shared)==2:covered=all(on_segment(p,*shared) for p in points)
    if covered:kind='shared_edge_only' if dimension==1 else 'shared_vertex_only'
    elif dimension==2:kind='coplanar_positive_area_overlap'
    elif dimension==0:kind='unshared_point_contact'
    else:
        ends=sorted(points);mid=tuple((x+y)/Q(2) for x,y in zip(ends[0],ends[-1]));inside=(strict_inside(mid,a,na),strict_inside(mid,b,nb))
        kind='interior_intersection_segment' if all(inside) else 'edge_face_contact_segment' if any(inside) else 'edge_edge_contact_segment'
    return dict(kind=kind,dimension=dimension,coplanar=coplanar,points=points)

def normal(points):
    n=cross(sub(points[1],points[0]),sub(points[2],points[0]));length=math.sqrt(dot(n,n))
    if not length:raise ValueError('Zero-area source triangle')
    return tuple(x/length for x in n)

def evidence_key(folder,name):return (Path(folder)/name).as_posix()

def authenticate(path,sha,snapshot=False):
    from amd_pixal3d_wheel_payload import readonly
    with readonly(path) as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=sha:raise ValueError('Source hash differs: '+str(path))
        if snapshot:
            if path.stat().st_size>64*1024**2:raise ValueError('Bounded evidence snapshot required')
            stream.seek(0);return stream.read()

def npy_header(stream,np):
    stream.seek(0)
    if stream.read(6)!=b'\x93NUMPY':raise ValueError('NPY magic')
    version=stream.read(2);length=int.from_bytes(stream.read(2 if version[0]==1 else 4),'little')
    if version not in (b'\x01\x00',b'\x02\x00') or not 0<length<4096:raise ValueError('Bounded NPY header')
    header=ast.literal_eval(stream.read(length).decode('ascii'));dt=np.dtype(header['descr']);shape=header['shape']
    if header['fortran_order'] or len(shape)!=2 or not 0<shape[0]<=5_000_000 or not 0<shape[1]<=7 or dt.str not in ('<i4','<f4'):raise ValueError('Qualified row-array format')
    offset=stream.tell();stream.seek(0,2)
    if stream.tell()!=offset+shape[0]*shape[1]*dt.itemsize:raise ValueError('NPY payload length')
    return shape,dt,offset

def read_rows(stream,ids,shape,dtype,offset,np):
    if len(ids)>2000 or any(not 0<=i<shape[0] for i in ids):raise ValueError('Bounded source-row query')
    rows={};size=shape[1]*dtype.itemsize
    for i in ids:
        stream.seek(offset+i*size);data=stream.read(size)
        if len(data)!=size:raise ValueError('Truncated source row')
        rows[i]=np.frombuffer(data,dtype=dtype).tolist()
    return rows

def inspect_panel(output):
    """Fixed small-case analysis only. No repair, UV creation, bake or export."""
    import numpy as np
    from amd_pixal3d_wheel_payload import readonly
    from amd_cpu_uv_bake import glb_arrays,geometry_hash
    output=Path(output).resolve();output.relative_to((ROOT.parent/'artifacts').resolve())
    if output.exists():raise FileExistsError('Exclusive forensic evidence required')
    output.mkdir();began=time.monotonic();receipts={};hashes={}
    def check_time():
        if time.monotonic()-began>120:raise TimeoutError('Bounded local forensic work exceeded120seconds')
    for name,sha in PINS.items():
        data=authenticate(PANEL/name,sha,True);hashes[name]=sha
        if name.endswith('.json'):
            receipt=json.loads(data);receipts[name]=receipt
            if receipt['classification']!='PASS':raise ValueError('Successful source receipt required')
            for file,row in receipt['files'].items():
                target=PANEL/Path(name).parent/file
                if target.resolve().parent!=(PANEL/Path(name).parent).resolve():raise ValueError('Receipt path escape')
                authenticate(target,row['sha256']);hashes[evidence_key(Path(name).parent,file)]=row['sha256'];check_time()
    def array(folder,name):
        key=folder+'/'+name+'.npy';data=authenticate(PANEL/key,hashes[key],True);return np.load(io.BytesIO(data),allow_pickle=False)
    v=array('prepare','vertices');f=array('prepare','faces');sv=array('prepare','source-vertex-ids');sf=array('prepare','source-face-ids');islands=array('qualify','face-island-ids')
    if v.shape!=(457,3) or f.shape!=(781,3) or islands.shape!=(781,) or len(set(map(int,sf)))!=781:raise ValueError('Exact authenticated panel required')
    edge_faces={}
    for i,face in enumerate(f.tolist()):
        for j in range(3):edge_faces.setdefault(tuple(sorted((face[j],face[(j+1)%3]))),[]).append(i)
    bad={e:ids for e,ids in edge_faces.items() if len(ids)>2}
    if len(bad)!=4 or any(len(ids)!=3 for ids in bad.values()):raise ValueError('Expected four three-face fans')
    source_edges={tuple(sorted((int(sv[a]),int(sv[b])))):ids for (a,b),ids in bad.items()};full_incidence={e:[] for e in source_edges};source_faces={}
    with readonly(MASTER/'faces.npy') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=MASTER_PINS['faces.npy']:raise ValueError('Master face hash')
        shape,dt,offset=npy_header(stream,np);stream.seek(offset)
        if shape!=(4173584,3):raise ValueError('Exact master face count')
        for start in range(0,shape[0],32768):
            count=min(32768,shape[0]-start);block=np.frombuffer(stream.read(count*12),dtype=dt).reshape(count,3)
            for e in full_incidence:
                selected=np.flatnonzero(np.any(block==e[0],axis=1)&np.any(block==e[1],axis=1))
                full_incidence[e].extend((start+selected).tolist())
            check_time()
        ids=set(map(int,sf));ids.update(i for group in full_incidence.values() for i in group);ids.update(i^1 for i in list(ids))
        source_faces=read_rows(stream,sorted(ids),shape,dt,offset,np)
    vertex_ids=set(int(x) for face in source_faces.values() for x in face)
    with readonly(MASTER/'vertices.npy') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=MASTER_PINS['vertices.npy']:raise ValueError('Master vertex hash')
        sh,dt,off=npy_header(stream,np);source_v=read_rows(stream,sorted(vertex_ids),sh,dt,off,np)
    if any(source_faces[int(sf[i])]!=sv[f[i]].tolist() for i in range(len(f))) or any(source_v[int(sv[i])]!=v[i].tolist() for i in range(len(v))):raise ValueError('Original source correspondence')
    # Inspect full source quad mates/flags without importing any neural module.
    quads={};needed=set()
    for e,local in source_edges.items():
        for face in full_incidence[e]:
            q=face//2;a,b=source_faces[2*q],source_faces[2*q+1]
            if b[0]==a[0] and b[1]==a[2]:quad=[a[0],a[1],a[2],b[2]]
            elif b[0]==a[2] and b[1]==a[1]:quad=[a[0],a[1],b[2],a[2]]
            else:raise ValueError('Original ordered quad split differs')
            quads[q]=dict(vertices=quad,triangles=[2*q,2*q+1]);needed.update(quad)
    with readonly(DECODE) as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!='d4de05fc82315e44a739c9815a07a531555e108af05c619ea0426d0c630ef861':raise ValueError('Saved decoder output hash')
        stream.seek(0);length=struct.unpack('<Q',stream.read(8))[0]
        if not 0<length<65536:raise ValueError('Bounded decoded header')
        header=json.loads(stream.read(length));decoded={}
        for name,width,dtype in [('coordinates',4,'<i4'),('features',7,'<f4')]:
            row=header[name]
            if row['shape']!=[1957891,width] or row['dtype']!=('I32' if width==4 else 'F32'):raise ValueError('Exact decoded source rows')
            decoded[name]=read_rows(stream,sorted(needed),row['shape'],np.dtype(dtype),8+length+row['data_offsets'][0],np)
    offsets=[[(0,0,0),(0,0,1),(0,1,1),(0,1,0)],[(0,0,0),(1,0,0),(1,0,1),(0,0,1)],[(0,0,0),(0,1,0),(1,1,0),(1,0,0)]]
    for q,record in quads.items():
        ids=record['vertices'];coords=[decoded['coordinates'][i][1:] for i in ids];delta=[tuple(coords[j][k]-coords[0][k] for k in range(3)) for j in range(4)]
        if delta not in offsets:raise ValueError('Source quad grid offsets differ')
        axis=offsets.index(delta);record.update(axis=axis,anchor=ids[0],grid_coordinates=coords,edge_logit=decoded['features'][ids[0]][3+axis],mate_presence=[i in set(map(int,sf)) for i in record['triangles']])
        if record['edge_logit']<=0:raise ValueError('Unflagged source quad')
    # Exact dyadic lifting: no epsilon or float-rounded intersection predicate.
    ratios=[[float(x).as_integer_ratio() for x in row] for row in v];scale=max(d for row in ratios for _,d in row);exact=[tuple(n*(scale//d) for n,d in row) for row in ratios]
    target=set(i for ids in bad.values() for i in ids);index={int(face):i for i,face in enumerate(sf)}
    target.update(index[int(sf[i])^1] for i in list(target) if (int(sf[i])^1) in index)
    details=[]
    for e,ids in sorted(bad.items()):
        incident=[]
        for i in ids:
            face=f[i].tolist();direction=1 if any((face[j],face[(j+1)%3])==e for j in range(3)) else -1
            incident.append(dict(panel_face=i,source_face=int(sf[i]),source_quad=int(sf[i])//2,source_vertices=sv[f[i]].tolist(),normal=normal(v[f[i]].tolist()),direction=direction,island=int(islands[i])))
        pairwise=[]
        for x in range(len(ids)):
            for y in range(x+1,len(ids)):
                i,j=ids[x],ids[y];relation=triangle_relation([exact[k] for k in f[i]],[exact[k] for k in f[j]])
                pairwise.append(dict(panel_faces=[i,j],kind=relation['kind'],coplanar=relation['coplanar'],normal_dot=dot(incident[x]['normal'],incident[y]['normal'])))
        details.append(dict(source_edge=[int(sv[k]) for k in e],panel_edge=e,positions=v[list(e)].tolist(),incident_faces=incident,pair_relations=pairwise,full_source_incident_faces=full_incidence[tuple(sorted(int(sv[k]) for k in e))]))
    hits=[];tested=0;queries=set()
    for i in target:
        for j in range(len(f)):
            if i==j:continue
            pair=tuple(sorted((i,j)))
            if pair in queries:continue
            queries.add(pair);a=[exact[k] for k in f[i]];b=[exact[k] for k in f[j]]
            if any(max(p[k] for p in a)<min(p[k] for p in b) or max(p[k] for p in b)<min(p[k] for p in a) for k in range(3)):continue
            tested+=1;relation=triangle_relation(a,b)
            if relation['kind'] not in ('disjoint','shared_vertex_only','shared_edge_only'):
                hits.append(dict(panel_faces=[i,j],source_faces=[int(sf[i]),int(sf[j])],kind=relation['kind'],points_world=[[float(c/scale) for c in p] for p in relation['points']]))
            check_time()
    # Keep original edge connectivity explicit; UV charts are not sheet labels.
    adjacency=[set() for _ in f]
    for e,ids in edge_faces.items():
        if len(ids)==2:adjacency[ids[0]].add(ids[1]);adjacency[ids[1]].add(ids[0])
    remaining=set(range(len(f)));components=[]
    while remaining:
        todo=[min(remaining)];seen=set(todo);remaining.difference_update(seen)
        while todo:
            for j in adjacency[todo.pop()]-seen:seen.add(j);remaining.discard(j);todo.append(j)
        components.append(dict(faces=sorted(seen),count=len(seen),islands=sorted(set(int(islands[i]) for i in seen))))
    parents=list(range(len(v)))
    def root(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for a,b in edge_faces:parents[root(a)]=root(b)
    conflicts=0
    for (a,b),ids in edge_faces.items():
        if len(ids)==2:
            directions=[any((int(f[i,j]),int(f[i,(j+1)%3]))==(a,b) for j in range(3)) for i in ids]
            conflicts+=directions[0]==directions[1]
    counts=dict(vertices=len(v),triangles=len(f),vertex_connected_components=len({root(i) for i in range(len(v))}),
                boundary_edges=sum(len(ids)==1 for ids in edge_faces.values()),nonmanifold_edges=len(bad),two_face_winding_conflicts=conflicts)
    if counts!=dict(vertices=457,triangles=781,vertex_connected_components=1,boundary_edges=147,nonmanifold_edges=4,two_face_winding_conflicts=0):raise ValueError('Preserved panel counts differ')
    gv,gf,gu,pixels,gmeta=glb_arrays(np,PANEL/'render-qualified/export/patch.glb')
    qv=array('qualify','vertices');qf=array('qualify','faces');qu=array('qualify','uv')
    if geometry_hash(np,gv,gf)!=geometry_hash(np,qv,qf):raise ValueError('GLB oriented geometry differs')
    def canonical(triangle):
        values=[tuple(float(x) for x in p) for p in triangle];rotation=min(range(3),key=lambda i:values[i:]+values[:i]);return tuple(values[rotation:]+values[:rotation]),rotation
    expected={}
    for face in qf:
        key,rotation=canonical(qv[face]);expected[key]=np.roll(qu[face],-rotation,axis=0).astype('f8')
    max_uv=0.0
    for face in gf:
        key,rotation=canonical(gv[face]);actual=np.roll(gu[face],-rotation,axis=0).astype('f8');actual[:,1]=1-actual[:,1]
        max_uv=max(max_uv,float(np.abs(actual-expected[key]).max()))
    if max_uv>2.9802322387695312e-8:raise ValueError('Existing UV roundtrip error exceeded')
    from PIL import Image
    with Image.open(PANEL/'bake/base-color.png') as im:
        if not np.array_equal(pixels,np.array(im.convert('RGBA'))):raise ValueError('Existing base texture pixels differ')
    # Reuse the existing geometry reader, then independently check both PBR
    # payloads and their routing. No baking, exporting or material edits occur.
    data=authenticate(PANEL/'render-qualified/export/patch.glb',PINS['render-qualified/export/patch.glb'],True)
    length=struct.unpack_from('<I',data,12)[0];document=json.loads(data[20:20+length]);start=28+length
    material=document['materials'][document['meshes'][0]['primitives'][0]['material']]
    pbr=material['pbrMetallicRoughness'];embedded={}
    for field,name in [('baseColorTexture','base-color.png'),('metallicRoughnessTexture','MR.png')]:
        texture=pbr[field]
        if texture.get('texCoord',0)!=0:raise ValueError('PBR UV set differs')
        image=document['images'][document['textures'][texture['index']]['source']]
        if image.get('mimeType')!='image/png' or 'uri' in image:raise ValueError('Embedded PBR PNG required')
        view=document['bufferViews'][image['bufferView']];offset=start+view.get('byteOffset',0)
        encoded=data[offset:offset+view['byteLength']];sha=hashlib.sha256(encoded).hexdigest()
        if sha!=hashes['bake/'+name]:raise ValueError('Embedded PBR PNG bytes differ')
        embedded[field]=sha
    result=dict(classification='FORENSICS_COMPLETE_CORRECTION_NOT_PERFORMED',counts=counts,bounds=[v.min(axis=0).tolist(),v.max(axis=0).tolist()],edges=details,source_quads=quads,
        checked_target_faces=sorted(target),tested_triangle_pairs_after_AABB=tested,target_panel_pair_queries=len(queries),local_intersections=hits,
        intersection_scope='Implicated faces and in-panel quad mates against all781panel faces; not a full-master intersection audit',
        manifold_edge_components=components,duplicate_positions=len(v)-len(set(map(tuple,v.tolist()))),
        GLB_readback=dict(oriented_geometry_exact=True,UV_max_error=max_uv,base_texture_pixels_exact=True,embedded_PBR_pngs=embedded,material=material,GLB_sha256=PINS['render-qualified/export/patch.glb'],metadata=gmeta),
        authenticated_stage_hashes=hashes,geometry_modified=False,UVs_modified=False,materials_modified=False,correction_executions=0,elapsed_seconds=time.monotonic()-began)
    # Recheck original bytes after all queries/readback, without writing sources.
    for name,sha in hashes.items():authenticate(PANEL/name,sha)
    with (output/'details.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    return result
