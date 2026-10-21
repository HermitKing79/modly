"""Read-only accepted-container admission; no unwrap, bake or material sampler."""
import hashlib,importlib.util,io,json,os,stat,struct,sys
from pathlib import Path
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-brick-pbr-mip-0f2fd92'
PRIOR=ROOT.parent/'artifacts/amd-brick-seam-uv-bake-87f3755'
PRIOR_E=ROOT/'docs/compatibility/amd-brick-seam-uv-bake-evidence'
GLB=PRIOR/'render-qualified/export/patch.glb'
GLB_SHA='e618af9fea32ce9fedb1e478ca1e0becb7eb55fc9ee4a246bcdf7105a6cf9c21'
AUTH=ROOT/'docs/compatibility/amd-brick-defect-diagnosis-evidence/diagnose.py'
AUTH_SHA='8fa7c617324ba636856b48f47a995ee64cd2a85f3c26d86d8760ec638b163529'
RECEIPTS={'qualify/uv.json':'86d0c01d54966df891b1ebe027c4bcdbcbc976a3e22bd4411f341e1bc8bd7229',
          'bake/bake.json':'7f06d75e55c0435bea68539b4b7d10194f0677486839d0cc7960d193a325e481'}

def require(ok,message):
    if not ok:raise ValueError(message)

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def plain(path):
    path=Path(path);require(path.is_absolute() and path.resolve(strict=True)==path,'Canonical plain absolute path required')
    for item in (path,*path.parents):
        info=item.lstat();require(not stat.S_ISLNK(info.st_mode) and not getattr(info,'st_file_attributes',0)&0x400,'Symlink/reparse forbidden')
        if stat.S_ISREG(info.st_mode):require(info.st_nlink==1,'Hardlinked input forbidden')
        else:require(stat.S_ISDIR(info.st_mode),'Only plain file/directory paths')
    return path

def load_auth():
    require(digest(plain(AUTH))==AUTH_SHA,'Original readonly guard pin')
    s=importlib.util.spec_from_file_location('accepted_readonly_auth',AUTH);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def authenticate():
    """Same complete saved inputs; hashes only, no checkpoint weight reads."""
    auth=load_auth();watched=[AUTH,GLB];identities={}
    require(plain(GLB).stat().st_size==493124 and digest(GLB)==GLB_SHA,'Accepted GLB identity')
    for name,(path,sha) in auth.PINS.items():
        row=auth.identity(plain(path));require(row['sha256']==sha,'Original identity: '+name);identities[name]=row;watched.append(path)
    receipts={}
    for relative,sha in RECEIPTS.items():
        path=plain(PRIOR/relative);require(digest(path)==sha,'Accepted producer receipt identity')
        r=json.loads(path.read_text());require(r['classification']=='PASS','Accepted producer must pass');receipts[relative]=r;watched.append(path)
        for name,row in r['files'].items():
            require(Path(name).name==name and name not in ('.','..'),'Single producer basename')
            item=plain(path.parent/name);require(item.stat().st_size==row['bytes'] and digest(item)==row['sha256'],'Accepted data identity: '+name);watched.append(item)
    return auth,identities,list(dict.fromkeys(watched)),receipts

def decode_glb(payload,size=1024):
    """Bounded core glTF parser and embedded-PNG reader; arrays use V-up."""
    import numpy as np
    from PIL import Image
    require(isinstance(payload,bytes) and 28<=len(payload)<=1024**2,'Small immutable GLB required')
    require(struct.unpack_from('<4sII',payload)==(b'glTF',2,len(payload)),'Complete GLB header')
    offset=12;chunks={}
    while offset<len(payload):
        require(offset+8<=len(payload),'Truncated chunk header');length,kind=struct.unpack_from('<II',payload,offset)
        require(length%4==0 and kind not in chunks and offset+8+length<=len(payload),'Complete unique aligned chunk')
        chunks[kind]=payload[offset+8:offset+8+length];offset+=8+length
    require(set(chunks)=={0x4e4f534a,0x004e4942} and len(chunks[0x4e4f534a])<=131072,'Exact JSON/BIN chunks')
    d=json.loads(chunks[0x4e4f534a]);binary=chunks[0x004e4942]
    require(d['asset']['version']=='2.0' and not d.get('extensionsRequired') and not d.get('extensionsUsed'),'Core glTF only')
    require(len(d['buffers'])==len(d['materials'])==len(d['meshes'])==1 and 'uri' not in d['buffers'][0],'One embedded static mesh/material')
    material=d['materials'][0];require(material.get('alphaMode','OPAQUE')=='OPAQUE' and material.get('doubleSided') is True,'Original opaque two-sided material')
    pbr=material['pbrMetallicRoughness'];require(pbr.get('baseColorFactor',[1]*4)==[1]*4 and pbr.get('metallicFactor',1)==pbr.get('roughnessFactor',1)==1,'Original unit factors')
    decoded={};samplers={};embedded={}
    for name,key,channels in [('base','baseColorTexture',4),('mr','metallicRoughnessTexture',3)]:
        ti=pbr[key];require(ti.get('texCoord',0)==0 and not ti.get('extensions'),'Untransformed UV0')
        texture=d['textures'][ti['index']];sampler=d['samplers'][texture['sampler']]
        require(sampler==dict(magFilter=9729,minFilter=9987,wrapS=33071,wrapT=33071),'Accepted trilinear/clamp sampler only')
        image=d['images'][texture['source']];require('uri' not in image and image['mimeType']=='image/png','Embedded PNG only')
        view=d['bufferViews'][image['bufferView']];start=view.get('byteOffset',0);length=view['byteLength']
        require(view.get('buffer',0)==0 and 0<length<=1024**2 and start>=0 and start+length<=len(binary),'In-buffer bounded PNG')
        png=binary[start:start+length]
        with Image.open(io.BytesIO(png)) as im:
            require(im.format=='PNG' and im.size==(size,size) and im.mode==('RGBA' if channels==4 else 'RGB'),'Original image dimensions/encoding')
            im.load();decoded[name]=np.array(im)[::-1].copy()
        embedded[name]=dict(sha256=hashlib.sha256(png).hexdigest(),bytes=len(png));samplers[name]=sampler
    primitive=d['meshes'][0]['primitives'];require(len(primitive)==1 and primitive[0].get('mode',4)==4,'One triangle primitive')
    primitive=primitive[0]
    def accessor(index,components):
        a=d['accessors'][index];require('sparse' not in a and not a.get('normalized'),'Dense original accessor')
        view=d['bufferViews'][a['bufferView']];dtype={5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]
        require(a['type']=={1:'SCALAR',2:'VEC2',3:'VEC3'}[components] and 0<a['count']<=18000,'Bounded accessor shape')
        width=np.dtype(dtype).itemsize*components;stride=view.get('byteStride',width);start=view.get('byteOffset',0)+a.get('byteOffset',0)
        end=start+(a['count']-1)*stride+width
        require(view.get('buffer',0)==0 and stride>=width and start>=0 and end<=len(binary) and end<=view.get('byteOffset',0)+view['byteLength'],'Bounded accessor range')
        return np.ndarray((a['count'],components),dtype,buffer=binary,offset=start,strides=(stride,np.dtype(dtype).itemsize)).copy()
    v=accessor(primitive['attributes']['POSITION'],3);uv=accessor(primitive['attributes']['TEXCOORD_0'],2);f=accessor(primitive['indices'],1).reshape(-1,3)
    require(v.dtype==uv.dtype==np.float32 and uv.shape==(len(v),2) and np.isfinite(v).all() and np.isfinite(uv).all() and f.max()<len(v),'Finite original GLB geometry')
    v=v[:,[0,2,1]];v[:,1]*=-1;uv[:,1]=1-uv[:,1]
    return dict(**decoded,samplers=samplers,embedded=embedded,vertices=v,faces=f.astype(np.int32),uv=uv)

def align_uv(v,f,uv,glb):
    """Match cyclic oriented original triangles; use actual GLB UV roundtrip."""
    import numpy as np
    require(v.shape==(len(f)*3,3) and uv.shape==(len(v),2),'Accepted seam-corner layout')
    def canonical(vertices,face):
        p=[tuple(map(float,vertices[i])) for i in face];first=min(range(3),key=lambda i:tuple(p[i:]+p[:i]));order=[(first+i)%3 for i in range(3)]
        return tuple(p[first:]+p[:first]),order
    table={}
    for i,face in enumerate(f):
        key,order=canonical(v,face);table.setdefault(key,[]).append((i,order))
    result=np.empty_like(uv);maximum=0.
    require(len(glb['faces'])==len(f),'Complete original triangle count')
    for face in glb['faces']:
        key,order=canonical(glb['vertices'],face);candidates=table.get(key,[]);actual=glb['uv'][face[order]]
        matches=[(j,index,original_order,float(np.max(np.abs(actual-uv[f[index][original_order]])))) for j,(index,original_order) in enumerate(candidates) if np.max(np.abs(actual-uv[f[index][original_order]]))<=2e-7]
        require(len(matches)==1,'Unambiguous original oriented triangle/UV identity')
        j,index,original_order,error=matches[0];result[f[index][original_order]]=actual;maximum=max(maximum,error);candidates.pop(j)
    require(all(not rows for rows in table.values()),'No original triangle omitted')
    return result,maximum

def save(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
