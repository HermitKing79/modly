"""One disposable CPU mip/filter evaluation of accepted embedded texture bytes."""
import sys,time,json,os,traceback
from pathlib import Path
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from assets import *
ORIGINAL_AUTHENTICATE=authenticate
P=ROOT.parent/'artifacts/amd-brick-mip-safe-atlas-e8244cb/revision'
ORIGINAL_MIP=ROOT.parent/'artifacts/amd-brick-pbr-mip-0f2fd92/run'
sys.path.insert(0,str(E))
from revision_setup import prepare_candidate

def authenticate():
    auth,identities,watched,receipts=ORIGINAL_AUTHENTICATE()
    receipt=json.loads((PRIOR/'bake/bake.json').read_text())
    require(receipt['classification']=='PASS','Actual revised spatial-bake receipt')
    for name,row in receipt['files'].items():
        path=PRIOR/'bake'/name
        require(path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],'Revised bake identity: '+name)
        watched.append(path)
    receipts['bake/bake.json']=receipt
    watched += [GLB,PRIOR/'bake/bake.json',PRIOR/'qualify/raster-ids.npy',PRIOR/'qualify/barycentrics.npy']
    return auth,identities,watched,receipts


def main():
    require(sys.stdin.buffer.readline(64)==b'GO\n','Exact owned-host binary GO')
    auth=load_auth();auth.P=P;sys.addaudithook(auth.runtime_guard)
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from mip_math import pyramid,sample,conditioned,srgb_decode,srgb_encode
    from mip_view import project
    started=time.monotonic();out=P/'run';events=out/'events.jsonl'
    def event(stage,**data):
        with events.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(stage=stage,elapsed_seconds=time.monotonic()-started,**data))+'\n');f.flush();os.fsync(f.fileno())
    def np_save(name,a):
        with (out/(name+'.npy')).open('xb') as f:np.save(f,a,allow_pickle=False)
        require(np.array_equal(np.load(out/(name+'.npy'),allow_pickle=False),a),'Exact output array readback')
    def error(a,b):
        d=np.abs(a-b);return dict(rmse=float(np.sqrt(np.mean(d*d))),p95_absolute=float(np.percentile(d,95)),max_absolute=float(d.max()))
    def encoded(a):return np.column_stack((srgb_encode(a[:,:3]),a[:,3:]))
    def metrics(a,c):return dict(RGB=error(encoded(a)[:,:3],encoded(c)[:,:3]),scalars=error(a[:,3:],c[:,3:]),
                                 metallic=error(a[:,3:4],c[:,3:4]),roughness=error(a[:,4:5],c[:,4:5]),alpha=error(a[:,5:6],c[:,5:6]))
    def passed(m):return m['RGB']['rmse']<=.02 and m['RGB']['p95_absolute']<=.05 and m['RGB']['max_absolute']<=.15 and m['scalars']['rmse']<=.02
    def filtered(chain,uv,lod):
        return np.concatenate([sample(chain,uv[i:i+32768],lod[i:i+32768]) for i in range(0,len(uv),32768)])
    def conditional(num,weight,uv,lod):
        chunks=[conditioned(num,weight,uv[i:i+32768],lod[i:i+32768]) for i in range(0,len(uv),32768)]
        return np.concatenate([x[0] for x in chunks]),np.concatenate([x[1] for x in chunks])[:,0]
    try:
        global PRIOR,GLB,GLB_SHA
        event('one_genuine_spatial_rebake');PRIOR=prepare_candidate(out);GLB=PRIOR/'patch.glb';GLB_SHA=digest(GLB)
        event('authenticate');_,identities,watched,receipts=authenticate();before={str(p):dict(sha256=digest(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in watched}
        glb=decode_glb(GLB.read_bytes());b=PRIOR/'bake';q=PRIOR/'qualify'
        for key,name in [('base','base-color.png'),('mr','MR.png')]:
            require(glb['embedded'][key]['sha256']==receipts['bake/bake.json']['files'][name]['sha256'],'Exact accepted embedded PNG bytes')
        arrays={n:np.load(b/(n+'.npy'),allow_pickle=False) for n in ('vertices','faces','uv','source-vertex-ids','face-island-ids','island-rects','ownership','coverage','sampled-field','query-uv','query-faces','continuous-values','seam-pairing')}
        v,f,original_uv=arrays['vertices'],arrays['faces'],arrays['uv'];uv,uv_error=align_uv(v,f,original_uv,glb)
        require(v.shape==(2343,3) and f.shape==(781,3) and arrays['island-rects'].shape==(4,4),'Accepted patch layout')
        ownership=arrays['ownership'];coverage=arrays['coverage'];raw=arrays['sampled-field'];face_island=arrays['face-island-ids']
        require(raw.dtype==np.float32 and raw.shape==(1024,1024,6) and np.isfinite(raw).all(),'Authentic finite sampled F32 field')
        require(ownership.dtype==np.int16 and coverage.dtype==np.bool_ and ownership.shape==coverage.shape==(1024,1024),'Accepted ownership/coverage arrays')
        ids=np.load(q/'raster-ids.npy',allow_pickle=False);bary=np.load(q/'barycentrics.npy',allow_pickle=False)
        require(np.array_equal(ids>=0,coverage) and np.array_equal(ownership[coverage],face_island[ids[coverage]]),'Complete source coverage/island correspondence')
        event('prepare_queries');old_uv=arrays['query-uv'];old_faces=arrays['query-faces'];point=arrays['continuous-values'];require(len(old_uv)==10961 and point.shape==(10961,6),'All saved continuous queries')
        # Reconstruct barycentric coordinates in the unchanged original atlas,
        # then use actual GLB UVs; no assumption of zero glTF V-flip roundoff.
        def to_actual(points,faces):
            tri=original_uv[f[faces]].astype(float);s=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=2)
            w12=np.linalg.solve(s,(points-tri[:,0])[:,:,None])[:,:,0];weights=np.column_stack((1-w12.sum(1),w12))
            require((weights>=-1e-5).all(),'Original query lies in its triangle')
            return (uv[f[faces]].astype(float)*weights[:,:,None]).sum(1),weights
        old_actual,old_weights=to_actual(old_uv,old_faces)
        # Covered pixel centres nearest each triangle edge provide real saved
        # source positions and positive support; they are not a new sampler.
        yy,xx=np.nonzero(coverage);order=np.argsort(ids[coverage],kind='stable');yy,xx=yy[order],xx[order]
        counts=np.bincount(ids[coverage],minlength=len(f));require((counts>0).all(),'Every original face covered')
        source_corners=arrays['source-vertex-ids'][f];edge_members={}
        for i,corners in enumerate(source_corners):
            for opposite in range(3):edge_members.setdefault(tuple(sorted((int(corners[(opposite+1)%3]),int(corners[(opposite+2)%3])))),[]).append(i)
        selected=[];edge_faces=[];boundary=[];cursor=0
        for i,count in enumerate(counts):
            ys,xs=yy[cursor:cursor+count],xx[cursor:cursor+count];cursor+=count
            for j in range(3):
                k=int(np.argmin(bary[ys,xs,j]));selected.append((ys[k],xs[k]));edge_faces.append(i)
                members=edge_members[tuple(sorted((int(source_corners[i,(j+1)%3]),int(source_corners[i,(j+2)%3]))))]
                boundary.append(len(members)==1 or any(face_island[m]!=face_island[i] for m in members))
        selected=np.asarray(selected,np.int32);edge_faces=np.asarray(edge_faces,np.int32);ey,ex=selected.T
        edge_actual,_=to_actual(np.column_stack(((ex+.5)/1024,(ey+.5)/1024)),edge_faces)
        queries=np.vstack((old_actual,edge_actual));faces=np.concatenate((old_faces,edge_faces));expected=np.vstack((point,raw[ey,ex]));expected_islands=face_island[faces]
        n=len(queries);require(n==13304,'Complete query extension, no omitted edges')
        pair=arrays['seam-pairing'];require(pair.shape==(45,2),'All original seam pairs')
        seam_edge=np.unique(pair.ravel());require(len(seam_edge)==90 and seam_edge.min()>=10781 and seam_edge.max()<10961,'Exact saved seam-edge membership')
        cohorts={'surface':np.arange(10781),'interiors':np.flatnonzero((old_weights[:10781]>.1).all(1)),
                 'seam_edge':seam_edge,'seam_near':np.setdiff1d(np.arange(10781,10961),seam_edge),
                 'triangle_edge':np.arange(10961,n),'island_edge':10961+np.flatnonzero(boundary)}
        # One combined bounded evaluation, including actual projected-size views.
        lods=np.arange(0,5.01,.5);all_uv=np.tile(queries,(len(lods),1));all_lod=np.repeat(lods,n);all_faces=np.tile(faces,len(lods));segments={}
        views={}
        for name,height in [('near',512),('medium',128),('far',32)]:
            view=project(v,f,uv,height);start=len(all_uv);end=start+len(view['query_uv']);segments[name]=[start,end]
            all_uv=np.vstack((all_uv,view['query_uv']));all_lod=np.concatenate((all_lod,view['query_lod']));all_faces=np.concatenate((all_faces,view['query_faces']));views[name]=view
        require(len(all_uv)<=524288,'Fixed total filter query budget')
        expected_all=face_island[all_faces];event('mip_pyramids',queries=len(all_uv))
        actual0=np.concatenate((srgb_decode(glb['base'][:,:,:3]/255),glb['mr'][:,:,2:3]/255,glb['mr'][:,:,1:2]/255,glb['base'][:,:,3:4]/255),2)
        source0=np.clip(raw.astype(float),0,1);source0[:,:,:3]=srgb_decode(source0[:,:,:3])
        actual_chain=pyramid(actual0);A=filtered(actual_chain,all_uv,all_lod)
        owner0=np.stack([ownership==i for i in range(4)]+[ownership<0],2).astype(float);influence=filtered(pyramid(owner0),all_uv,all_lod)
        require(np.max(np.abs(influence.sum(1)-1))<1e-12,'Ownership partition of unity')
        own=influence[np.arange(len(influence)),expected_all];foreign=np.maximum(0,influence[:,:4].sum(1)-own);unowned=influence[:,4]
        B=np.empty_like(A);C=np.empty_like(A);massB=np.empty(len(A));massC=np.empty(len(A))
        for island in range(4):
            event('conditioned_island',island=island);selection=np.flatnonzero(expected_all==island);member=(ownership==island).astype(float)[:,:,None]
            B[selection],massB[selection]=conditional(pyramid(actual0*member),pyramid(member),all_uv[selection],all_lod[selection])
            covered=member*coverage[:,:,None]
            C[selection],massC[selection]=conditional(pyramid(source0*covered),pyramid(covered),all_uv[selection],all_lod[selection])
        require(np.isfinite(A).all() and np.isfinite(B).all() and np.isfinite(C).all(),'Finite complete filtered output')
        # Independent direct original-base block oracle: no mip pyramid inputs.
        event('independent_direct_block_oracle');oracle_indices=np.linspace(0,len(all_uv)-1,64,dtype=np.int32);oracleA=[];oracleC=[];oracleOwner=[]
        for index in oracle_indices:
            coordinate=all_uv[index];lod=all_lod[index];low=int(np.floor(lod));high=min(low+1,5);frac=lod-low;numA=np.zeros(6);numC=np.zeros(6);denC=0.;ow=np.zeros(5)
            for level,lw in [(low,1-frac),(high,frac)]:
                if not lw:continue
                side=1024>>level;block=1<<level;pixel=np.clip(coordinate*side-.5,0,side-1);lo=np.floor(pixel).astype(int);d=pixel-lo
                for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
                    x=min(lo[0]+dx,side-1);y=min(lo[1]+dy,side-1);weight=lw*(d[0] if dx else 1-d[0])*(d[1] if dy else 1-d[1]);r=np.s_[y*block:(y+1)*block,x*block:(x+1)*block]
                    cm=((ownership[r]==expected_all[index])&coverage[r]).astype(float)
                    numA+=weight*actual0[r].mean((0,1));numC+=weight*(source0[r]*cm[:,:,None]).mean((0,1));denC+=weight*cm.mean();ow+=weight*owner0[r].mean((0,1))
            require(denC>0,'Independent source footprint support');oracleA.append(numA);oracleC.append(numC/denC);oracleOwner.append(ow)
        oracle=max(float(np.max(np.abs(A[oracle_indices]-oracleA))),float(np.max(np.abs(C[oracle_indices]-oracleC))),float(np.max(np.abs(influence[oracle_indices]-oracleOwner))))
        require(oracle<=2e-12,'Independent direct block/mip equivalence')
        # Point truth is used only for LOD0; all other quality tests compare C.
        target=np.column_stack((srgb_decode(np.clip(expected[:,:3],0,1)),np.clip(expected[:,3:],0,1)))
        lod0={name:metrics(A[:n][index],target[index]) for name,index in cohorts.items()}
        point_gap=encoded(A[:n])[pair[:,0]]-encoded(A[:n])[pair[:,1]]
        lod0['seam_gap']=dict(RGB=error(point_gap[:,:3],np.zeros_like(point_gap[:,:3])),scalars=error(point_gap[:,3:],np.zeros_like(point_gap[:,3:])))
        require(all(passed(row) for row in lod0.values()),'Original continuous-point LOD0 gates remain unchanged')
        levels=[]
        for k,lod in enumerate(lods):
            s=slice(k*n,(k+1)*n);a,bv,c=A[s],B[s],C[s];fr,un=foreign[s],unowned[s]
            groups={}
            for name,index in cohorts.items():
                if not len(index):raise ValueError('Empty declared cohort')
                groups[name]=dict(count=len(index),total_error=metrics(a[index],c[index]),contamination_error=metrics(a[index],bv[index]),padding_quantization_error=metrics(bv[index],c[index]),foreign_queries=int((fr[index]>1e-7).sum()),unowned_queries=int((un[index]>1e-7).sum()),foreign_max=float(fr[index].max()),unowned_max=float(un[index].max()))
            # Report raw gaps and score only error beyond matched source gaps.
            ea,ec=encoded(a),encoded(c);gap=ea[pair[:,0]]-ea[pair[:,1]];refgap=ec[pair[:,0]]-ec[pair[:,1]];residual=gap-refgap
            seam=dict(RGB=error(residual[:,:3],np.zeros_like(residual[:,:3])),scalars=error(residual[:,3:],np.zeros_like(residual[:,3:])))
            per_island={str(i):dict(count=int((expected_islands==i).sum()),error=metrics(a[expected_islands==i],c[expected_islands==i])) for i in range(4)}
            quality=all(passed(row['total_error']) for row in groups.values()) and all(passed(row['error']) for row in per_island.values()) and passed(seam)
            contamination=bool((fr>1e-7).any() or (un>1e-7).any())
            levels.append(dict(LOD=float(lod),classification='NO-GO' if contamination or not quality else 'PASS',accuracy_gates_pass=quality,cohorts=groups,per_island=per_island,seam_residual=seam,raw_seam_RGB_max=float(np.abs(gap[:,:3]).max()),source_footprint_seam_RGB_max=float(np.abs(refgap[:,:3]).max()),minimum_source_coverage_mass=float(massC[s].min()),source_interior_encoded_RGB_std=np.std(encoded(c)[cohorts['interiors'],:3],0).tolist()))
        event('private_views');images={};panels=[]
        font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',18)
        for name,view in views.items():
            start,end=segments[name];pixels=view['pixels_yx'];ys,xs=pixels.T;rows=[]
            for label,values in [('source',C[start:end]),('accepted',A[start:end])]:
                canvas=np.full((640,640,3),128,np.uint8);canvas[ys,xs]=np.rint(srgb_encode(values[:,:3])*255).astype(np.uint8)
                target=out/(name+'-'+label+'.png');Image.fromarray(canvas).save(target)
                with Image.open(target) as im:im.load();require(np.array_equal(np.array(im),canvas),'Exact PNG readback')
                images[target.name]=dict(sha256=digest(target),bytes=target.stat().st_size)
                box=(int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1);crop=Image.fromarray(canvas).crop(box);scale=min(512/crop.width,512/crop.height)
                crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.NEAREST);panel=Image.new('RGB',(512,512),(128,128,128));panel.paste(crop,((512-crop.width)//2,(512-crop.height)//2));rows.append(panel)
            # Display the immutable accepted view beside this one revised view.
            with Image.open(ORIGINAL_MIP/(name+'-accepted.png')) as old_image:
                old_image.load();old_canvas=np.array(old_image)
            require(old_canvas.shape==(640,640,3),'Original matched-distance comparison')
            crop=Image.fromarray(old_canvas).crop(box)
            crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.NEAREST)
            old_panel=Image.new('RGB',(512,512),(128,128,128));old_panel.paste(crop,((512-crop.width)//2,(512-crop.height)//2));rows[0]=old_panel
            heat=np.full((640,640,3),128,np.uint8);heat[ys,xs]=np.column_stack((np.rint(foreign[start:end]*255),np.rint(unowned[start:end]*255),np.zeros(end-start))).astype(np.uint8)
            Image.fromarray(heat).save(out/(name+'-contamination.png'))
            strip=Image.new('RGB',(1024,558),'#20252c');draw=ImageDraw.Draw(strip);draw.text((12,8),f"{name}: projected height {view['metadata']['actual_projected_height_pixels']:.0f}px; LOD {view['metadata']['LOD_min']:.3f}; NEAREST display enlargement",font=font,fill='white')
            strip.paste(rows[0],(0,46));strip.paste(rows[1],(512,46));panels.append(strip)
        sheet=Image.new('RGB',(1024,64+558*3),'#20252c');draw=ImageDraw.Draw(sheet);draw.text((12,6),'ORIGINAL ACCEPTED GLB (left)  |  ONE REVISED GLB (right)',font=font,fill='white');draw.text((12,32),'Unlit RGB diagnostic; same geometry. Metallic/roughness tested numerically.',font=font,fill='white')
        for i,panel in enumerate(panels):sheet.paste(panel,(0,64+i*558))
        sheet.save(out/'comparison.png')
        # Persist filter outputs and exact inputs for independent verification.
        for name,array in {'queries-uv':all_uv,'queries-lod':all_lod,'queries-island':expected_all,'A-accepted':A,'B-isolated-png':B,'C-source-footprint':C,'ownership-influence':influence,'source-mass':massC,'oracle-indices':oracle_indices}.items():np_save(name,array)
        after={str(p):dict(sha256=digest(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in watched};require(before==after,'Accepted source/input changed')
        classification='PASS' if all(row['classification']=='PASS' for row in levels) else 'NO-GO'
        files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.iterdir() if p.suffix in ('.png','.npy')}
        record=dict(classification=classification,CPU_only=True,originals_unchanged=True,original_bake_repeated=True,UV_rebuilt=True,
            filter_contract='F64 linear-light 2x2 box; bilinear at each mip, linear between mips; clamp atlas outer edges; no mip storage rounding',
            GLB=dict(path=str(GLB),bytes=GLB.stat().st_size,sha256=GLB_SHA,samplers=glb['samplers'],embedded=glb['embedded'],maximum_UV_roundtrip_error=uv_error),
            levels=levels,LOD0_continuous_point_error=lod0,all13304_queries_retained=True,query_counts={k:len(v) for k,v in cohorts.items()},total_evaluations=len(all_uv),
            direct_block_oracle_max_error=oracle,source_positive_support=True,contamination_tolerance=1e-7,
            RGB_gates=dict(rmse=.02,p95_absolute=.05,max_absolute=.15),scalar_RMSE_limit=.02,
            views={name:{**view['metadata'],'query_segment':segments[name],'foreign_weight_max':float(foreign[slice(*segments[name])].max()),'unowned_weight_max':float(unowned[slice(*segments[name])].max())} for name,view in views.items()},
            visual_scope='Original accepted versus one integer-translated, genuinely rebaked revision. Software +X unlit base colour; actual projected-size UV derivative footprints, not Blender PBR or physical orthographic camera-distance changes; panels nearest-enlarged only after filtering',
            no_universal_engine_claim=True,source_input_identities=identities,files=files,elapsed_seconds=time.monotonic()-started)
        save(out/'result.json',record);event('complete',classification=classification);print(json.dumps(dict(classification=classification,elapsed_seconds=record['elapsed_seconds'],levels=[(r['LOD'],r['classification']) for r in levels])),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(classification='BLOCKED',error=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started));raise

if __name__=='__main__':main()
