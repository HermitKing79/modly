"""Single bounded phase1 proof. Never changes UVs or samples new materials."""
import sys,os,time,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
OLD=ROOT/'docs/compatibility/amd-brick-pbr-mip-evidence'
sys.path.insert(0,str(OLD))
from assets import authenticate,load_auth,decode_glb,GLB,PRIOR,digest,save,require,align_uv
P=ROOT.parent/'artifacts/amd-brick-mip-safe-atlas-e8244cb'

def main():
    if os.environ.get('MODLY_ATLAS_STAGE')=='revision':
        sys.path.insert(0,str(E))
        from revision import main as revise
        return revise()
    require(sys.stdin.buffer.readline(64)==b'GO\n','Owned binary GO required')
    stage=os.environ['MODLY_ATLAS_STAGE'];require(stage=='feasibility','Only reviewed phase1 is implemented')
    auth=load_auth();auth.P=P/stage;sys.addaudithook(auth.runtime_guard)
    import numpy as np
    from mip_math import pyramid,sample
    sys.path.insert(0,str(E));from footprint import validate_isolation
    began=time.monotonic();out=P/stage/'run'
    _,identities,watched,_=authenticate();before={str(p):digest(p) for p in watched}
    old_private=ROOT.parent/'artifacts/amd-brick-pbr-mip-0f2fd92'
    old_result=json.loads((old_private/'run/result.json').read_text())
    require(digest(old_private/'run/result.json')=='306cb03bf0eaca7384bcbf70a543c7bdb16553b0a326f4238d56bb12a50f0047','Original complete mip failure receipt')
    for name,row in old_result['files'].items():
        p=old_private/'run'/name;require(p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],'Original complete mip evidence: '+name)
    g=decode_glb(GLB.read_bytes());b=PRIOR/'bake'
    v,f,u,fi,rect,raw,covered,owner=[np.load(b/(n+'.npy'),allow_pickle=False) for n in ('vertices','faces','uv','face-island-ids','island-rects','sampled-field','coverage','ownership')]
    guv,roundoff=align_uv(v,f,u,g)
    quv=np.load(old_private/'run/queries-uv.npy',allow_pickle=False)[143870:143871]
    require(np.load(b/'source-face-ids.npy')[496]==3809864 and np.load(b/'query-faces.npy')[10830]==496,'Exact failing source face/query')
    actual=float(sample(pyramid(g['mr'][:,:,1:2].astype(float)/255),quv,np.array([5.]))[0,0])
    # Direct original-base block oracle: no saved C result or pyramid reference.
    q=np.clip(quv[0]*32-.5,0,31);lo=np.floor(q).astype(int);frac=q-lo;num=den=foreign=unowned=0.
    for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
        x=min(lo[0]+dx,31);y=min(lo[1]+dy,31);s=np.s_[y*32:(y+1)*32,x*32:(x+1)*32]
        w=(frac[0] if dx else 1-frac[0])*(frac[1] if dy else 1-frac[1]);mask=(owner[s]==2)&covered[s]
        num+=w*(np.clip(raw[s][:,:,4].astype(float),0,1)*mask).mean();den+=w*mask.mean()
        foreign+=w*((owner[s]>=0)&(owner[s]!=2)).mean();unowned+=w*(owner[s]<0).mean()
    require(den>0,'Source support');reference=float(num/den)
    savedA=np.load(old_private/'run/A-accepted.npy',mmap_mode='r',allow_pickle=False)[143870,4]
    savedC=np.load(old_private/'run/C-source-footprint.npy',mmap_mode='r',allow_pickle=False)[143870,4]
    require(abs(actual-savedA)<2e-12 and abs(reference-savedC)<2e-12,'Independent witness reproduction')
    shifts=np.array([[0,0],[118,60],[77,220],[28,380]],np.float64)
    new_rect=np.array([[0,0,768,1024],[800,32,928,160],[800,192,928,320],[800,352,928,480]],np.int32)
    # Virtual arithmetic only: no UV/output atlas is persisted in phase1.
    virtual=guv[f].astype(float)+shifts[fi,None,:]/1024
    proof=validate_isolation(virtual,fi,new_rect)
    old_spans=[np.ptp(u[f[fi==i]].astype(float),axis=(0,1))*1024 for i in range(4)]
    area=int(np.prod(new_rect[:,2:]-new_rect[:,:2],axis=1).sum())
    require(area<=1024**2,'Atlas capacity')
    require(before=={str(p):digest(p) for p in watched},'Originals unchanged')
    save(out/'result.json',dict(classification='PASS',phase='feasibility_only',no_UV_change=True,no_bake=True,
        GLB_SHA256=digest(GLB),source_identities=identities,original_mip_receipt_SHA256=digest(old_private/'run/result.json'),
        witness=dict(original_face=3809864,local_face=496,saved_query=10830,LOD=5,actual=actual,source_footprint=reference,absolute_error=abs(actual-reference),foreign_weight=float(foreign),unowned_weight=float(unowned)),
        support_proof=proof,ownership_rectangles=new_rect.tolist(),virtual_integer_translations=shifts.tolist(),
        original_island_spans_pixels=[x.tolist() for x in old_spans],owned_area=area,atlas_area=1024**2,unused_area=1024**2-area,
        density_unchanged=True,triangle_coverage_phase_unchanged=True,UV_GLB_roundtrip_error=roundoff,
        all_previous_mip_files_authenticated=len(old_result['files']),elapsed_seconds=time.monotonic()-began))
    print('PHASE1_FEASIBILITY_PASS',flush=True)

if __name__=='__main__':main()
