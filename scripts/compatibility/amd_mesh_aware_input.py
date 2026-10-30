"""Mesh-aware source preparation and unchanged strict plane admission."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import amd_mesh_aware_proof as run
from amd_cgal_proof import save, sha

RAW=run.PRIOR.parent/'amd-closed-solid-raw-recovery-v1/mesh'

def authenticate():
    # Rehash every original binding retained by the independently reviewed proof.
    old=json.loads((run.PRIOR/'authentication.json').read_text())
    checked={}
    for name,expected in old['original_snapshot'].items():
        p=Path(name)
        if not p.is_absolute():p=run.owner.ROOT/p
        s=p.stat()
        if s.st_size!=expected['bytes'] or sha(p)!=expected['sha256']:raise ValueError('Original differs: '+str(p))
        checked[str(p)]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=expected['sha256'])
    for name,e in json.loads((run.PRIOR/'final-build-identities.json').read_text()).items():
        p=run.PRIOR/name
        if p.stat().st_size!=e['bytes'] or sha(p)!=e['sha256']:raise ValueError('Qualified backend differs: '+name)
    frozen=run.owner.ROOT/'docs/compatibility/amd-ai-hard-surface-criteria.json'
    if sha(frozen)!='397bf6c2dc8908e8a3b1a02ba169d6ec6ab5bc2aaf11ee96a4cb84e81591aad0':raise ValueError('Frozen criteria differs')
    if json.loads((run.PRIOR/'official-sample-verification.json').read_text())['classification']!='PASS_OFFICIAL_SAMPLE_SOLVER_AND_EXPORT':raise ValueError('Official baseline receipt')
    save(run.BASE/'authentication.json',dict(classification='PASS',original_snapshot=checked,binding=old['binding'],backend=sha(run.PRIOR/'final-build-identities.json'),frozen_criteria_sha256=sha(frozen),ten_tensor_original_archive_authenticated=True))

def verify_backend():
    import tarfile
    import zipfile
    begin=time.monotonic();results=[]
    rows=json.loads((run.PRIOR/'downloads-provenance.json').read_text())
    rows.append(json.loads((run.PRIOR/'numpy-provenance.json').read_text()))
    for row in rows:
        name=row['name']
        if 'auxiliary' in name:continue # Not linked or used by the qualified SDK.
        path=run.PRIOR/'downloads'/name
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Pinned dependency archive differs')
        count=0;total=0
        def check(member,data):
            nonlocal count,total
            rel=run.owner.safe_member(member)
            if name.startswith('boost_') and not run.owner.selected_boost(rel):return
            dest=(run.SITE/rel) if name.endswith('.whl') else (run.owner.ENV/rel)
            st=dest.stat()
            if dest.is_symlink() or st.st_nlink!=1 or st.st_size!=len(data) or sha(dest)!=hashlib.sha256(data).hexdigest():raise ValueError('Qualified payload differs: '+str(dest))
            count+=1;total+=len(data)
        if name.endswith('.tar.gz'):
            with tarfile.open(path,'r:gz') as archive:
                for member in archive:
                    if member.isfile():check(member.name,archive.extractfile(member).read())
                    elif not member.isdir():raise ValueError('Unexpected archive link')
        else:
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if not member.is_dir():check(member.filename,archive.read(member))
        results.append(dict(archive=name,sha256=row['sha256'],files=count,bytes=total))
    save(run.BASE/'backend-reauthentication.json',dict(classification='PASS',archives=results,elapsed_seconds=time.monotonic()-begin,
        complete_selected_payloads=True,original_environments_modified=False,unused_auxiliary_GMP_MPFR_not_claimed=True))

def adjacency(np,faces):
    """Original index edges only. No welding; 1 and >2 incidences block traversal."""
    edges=faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2).astype('u8')
    lo=edges.min(axis=1);hi=edges.max(axis=1);keys=(lo<<32)|hi
    order=np.argsort(keys,kind='stable');sorted_keys=keys[order]
    start=np.r_[0,np.flatnonzero(sorted_keys[1:]!=sorted_keys[:-1])+1];count=np.diff(np.r_[start,len(keys)])
    pair=start[count==2];a=order[pair];b=order[pair+1]
    valid=(a//3!=b//3)&(lo[a]!=hi[a]);a=a[valid];b=b[valid]
    adj=np.full(len(keys),-1,'i4');adj[a]=b//3;adj[b]=a//3
    return adj.reshape(-1,3),dict(boundary_edges=int((count==1).sum()),nonmanifold_edges=int((count>2).sum()),two_face_edges=int((count==2).sum()))

def prepare():
    import numpy as np
    begin=time.monotonic();out=run.BASE/'input';out.mkdir(exist_ok=False)
    v=np.load(RAW/'vertices.npy',allow_pickle=False);f=np.load(RAW/'faces.npy',allow_pickle=False)
    old=run.PRIOR/'ai-input/samples.npz'
    receipt=json.loads((run.PRIOR/'ai-input/receipt.json').read_text())
    if sha(old)!=receipt['samples_sha256']:raise ValueError('Historical samples differ')
    with np.load(old,allow_pickle=False) as z:
        samples={k:z[k] for k in z.files}
    lo=v.min(axis=0).astype('f8');hi=v.max(axis=0).astype('f8');D=float(np.linalg.norm(hi-lo));center=(lo+hi)/2
    adj,counts=adjacency(np,f)
    if counts['boundary_edges']!=1421 or counts['nonmanifold_edges']!=519:raise ValueError('Original incidence not reproduced')
    # Locked before segmentation: withheld triangle rows are excluded from fitting
    # AND neighbor traversal. Their fresh barycentric coordinates are never fit.
    ids=np.arange(len(f),dtype='u8');locked=((ids*2654435761+1013904223)%17)==0
    active=~locked
    adj[locked]=-1
    valid=adj>=0;adj[valid & locked[np.maximum(adj,0)]]=-1
    areas=np.empty(len(f),'f8');normals=np.empty((len(f),3),'f8')
    for i in range(0,len(f),32768):
        t=v[f[i:i+32768]].astype('f8');n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);mag=np.linalg.norm(n,axis=1)
        if np.any(mag<=0) or not np.isfinite(mag).all():raise ValueError('Invalid source triangle')
        areas[i:i+len(t)]=mag/2;normals[i:i+len(t)]=n/mag[:,None]
    with (out/'mesh.bin').open('xb') as stream:
        np.array([len(v),len(f)],dtype='<u8').tofile(stream)
        ((v.astype('f8')-center)*(100/D)).astype('<f8').tofile(stream)
        f.astype('<u4').tofile(stream);adj.astype('<i4').tofile(stream);active.astype('u1').tofile(stream)
    rng=np.random.default_rng(20261101);locked_ids=np.flatnonzero(locked)
    cdf=np.cumsum(areas[locked]);rows=locked_ids[np.searchsorted(cdf,(np.arange(65536)+.5)/65536*cdf[-1])]
    uv=rng.random((65536,2));r=np.sqrt(uv[:,0]);w=np.column_stack((1-r,r*(1-uv[:,1]),r*uv[:,1]))
    points=(v[f[rows]].astype('f8')*w[:,:,None]).sum(axis=1)
    with (out/'validation.npz').open('xb') as stream:np.savez(stream,points=points,normal=normals[rows],triangle_rows=rows)
    with (out/'source-analysis.npz').open('xb') as stream:np.savez(stream,area=areas,normal=normals,active=active,center=center,diagonal=D)
    save(out/'receipt.json',dict(classification='PASS',vertices=len(v),triangles=len(f),original_incidence=counts,
        diagonal=D,center=center.tolist(),scale=100/D,locked_triangles=int(locked.sum()),fit_triangles=int(active.sum()),
        validation_seed=20261101,validation_points=65536,validation_sha256=sha(out/'validation.npz'),
        input_sha256=sha(out/'mesh.bin'),historical_samples_sha256=sha(old),historical_heldout_points=int(samples['held'].sum()),
        historical_holdout_interpretation='No historical sample coordinate is fitted; underlying non-locked source triangles do inform area-LS planes and adjacency, so this is not spatially blind validation.',
        additional_holdout_interpretation='Locked triangle rows excluded from plane fitting, region propagation and reconstruction input; neighbouring source geometry remains correlated.',
        elapsed_seconds=time.monotonic()-begin))

def read_planes(np,path):
    rows=list(csv.DictReader(path.open()))
    planes=[]
    for r in rows:
        n=np.array([float(r[k]) for k in ('nx','ny','nz')]);offset=float(r['offset'])
        if not np.isfinite(n).all() or not math.isfinite(offset) or abs(np.linalg.norm(n)-1)>1e-8:raise ValueError('Unit finite plane required')
        planes.append(dict(id=len(planes),normal=n.tolist(),offset=offset,faces=int(r.get('faces',0))))
    return planes

def screen():
    import numpy as np
    from amd_cgal_ai_input import evaluate
    z=np.load(run.PRIOR/'ai-input/samples.npz',allow_pickle=False);extra=np.load(run.BASE/'input/validation.npz',allow_pickle=False)
    source=np.load(run.BASE/'input/source-analysis.npz',allow_pickle=False)
    receipt=json.loads((run.BASE/'input/receipt.json').read_text())
    for p,key in ((run.PRIOR/'ai-input/samples.npz','historical_samples_sha256'),(run.BASE/'input/validation.npz','validation_sha256')):
        if sha(p)!=receipt[key]:raise ValueError('Locked validation identity differs')
    D=float(source['diagonal']);center=source['center'];scale=100/D
    raw=read_planes(np,run.BASE/'regions.csv');remaining=np.arange(len(raw));selected=[]
    # Single deterministic residual-feature selection, never RANSAC or a sweep.
    train=(~z['held'])&source['active'][z['triangle_rows']]
    p=z['points'][train];n=z['normal'][train];uncovered=np.ones(len(p),bool)
    cosine=math.cos(math.radians(10))
    for plane in raw:plane['offset']=float(np.dot(plane['normal'],center)+plane['offset']/scale)
    while len(selected)<48 and len(remaining):
        best=None
        for j in remaining:
            row=raw[j];normal=np.array(row['normal']);mask=(abs(p@normal-row['offset'])<=.002*D)&(abs(n@normal)>=cosine)
            score=int((mask&uncovered).sum())
            if mask.sum()>=24 and score and (best is None or score>best[0]):best=(score,int(j),mask)
        if best is None:break
        _,idx,mask=best;selected.append(raw[idx]);remaining=remaining[remaining!=idx];uncovered[mask]=False
    for i,row in enumerate(selected):row['source_region']=row['id'];row['id']=i
    hist,_,_=evaluate(np,z['points'][z['held']],z['normal'][z['held']],selected,D)
    locked,_,_=evaluate(np,extra['points'],extra['normal'],selected,D)
    training,dist,labels=evaluate(np,p,n,selected,D)
    counts=[int(((labels==i)&(dist<=.002*D)).sum()) for i in range(len(selected))]
    qualified=4<=len(selected)<=48 and hist['gate_pass'] and locked['gate_pass'] and all(c>=24 for c in counts)
    save(run.BASE/'plane-screen.json',dict(classification='PASS' if qualified else 'NO_GO',plane_count=len(selected),region_count=len(raw),
        historical_holdout=hist,locked_triangle_validation=locked,training=training,strict_training_support=counts,planes=selected,
        ground_truth_used=False,AI_optimization_authorized=False,finite_support_is_source_patch_not_convex_hull=True,
        residual_feature_selection='One deterministic maximum-new-training-coverage selection from actual adjacency regions, capped at 48.',
        criteria_unchanged=True))
    # Even a passing coverage gate does not authorize MIP until exact post-refit
    # CGAL planes and finite source support are independently screened.
    if not qualified:return
    raise RuntimeError('Final-plane and finite-footprint admission must be implemented and reviewed before MIP')

def residual_assess():
    """Area-aware residual evidence; never use either validation set to fit."""
    import numpy as np
    from amd_cgal_ai_input import evaluate
    source=np.load(run.BASE/'input/source-analysis.npz',allow_pickle=False)
    z=np.load(run.PRIOR/'ai-input/samples.npz',allow_pickle=False)
    receipt=json.loads((run.BASE/'input/receipt.json').read_text())
    if sha(run.PRIOR/'ai-input/samples.npz')!=receipt['historical_samples_sha256']:raise ValueError('Historical sample differs')
    v=np.load(RAW/'vertices.npy',allow_pickle=False);f=np.load(RAW/'faces.npy',allow_pickle=False)
    D=float(source['diagonal']);center=source['center'];scale=100/D
    planes=read_planes(np,run.BASE/'regions.csv')
    train=(~z['held'])&source['active'][z['triangle_rows']]
    p=z['points'][train];n=z['normal'][train];cosine=math.cos(math.radians(10))
    support=[]
    for plane in planes:
        plane['offset']=float(np.dot(plane['normal'],center)+plane['offset']/scale)
        nn=np.array(plane['normal']);mask=(abs(p@nn-plane['offset'])<=.002*D)&(abs(n@nn)>=cosine)
        support.append(int(mask.sum()))
    eligible=[plane for plane,c in zip(planes,support) if c>=24]
    # Source residual inspection uses all nonlocked original face centroids and
    # normals, independent of heldout sample coordinates.
    counts=np.zeros(len(eligible),'i8');areas=np.zeros(len(eligible),'f8')
    source_uncovered=0;source_uncovered_area=0.;bounds_lo=np.full(3,np.inf);bounds_hi=np.full(3,-np.inf)
    residual_rows=[]
    for start in range(0,len(f),32768):
        rows=np.arange(start,min(start+32768,len(f)));rows=rows[source['active'][rows]]
        points=v[f[rows]].astype('f8').mean(axis=1);normals=source['normal'][rows]
        check,dist,label=evaluate(np,points,normals,eligible,D);covered=dist<=.002*D
        for j in range(len(eligible)):
            use=covered&(label==eligible[j]['id']);counts[j]+=int(use.sum());areas[j]+=float(source['area'][rows[use]].sum())
        bad=rows[~covered];source_uncovered+=len(bad);source_uncovered_area+=float(source['area'][bad].sum())
        if len(bad):
            residual_rows.extend(bad.tolist());bounds_lo=np.minimum(bounds_lo,v[f[bad]].min(axis=(0,1)));bounds_hi=np.maximum(bounds_hi,v[f[bad]].max(axis=(0,1)))
    all_hist,_,_=evaluate(np,z['points'][z['held']],z['normal'][z['held']],eligible,D)
    with (run.BASE/'residual-source-rows.npy').open('xb') as stream:np.save(stream,np.array(residual_rows,'i4'),allow_pickle=False)
    save(run.BASE/'residual-assessment.json',dict(classification='ASSESSMENT_ONLY',all_region_count=len(planes),eligible_region_count=len(eligible),
        strict_sample_support_all_regions=support,eligible_region_ids=[p['id'] for p in eligible],
        source_face_support=counts.tolist(),source_area_support=areas.tolist(),source_uncovered_faces=source_uncovered,
        source_uncovered_area=source_uncovered_area,source_uncovered_fraction=float(source_uncovered_area/source['area'][source['active']].sum()),
        source_residual_bounds=[bounds_lo.tolist(),bounds_hi.tolist()] if source_uncovered else None,
        all_eligible_planes_historical_holdout=all_hist,source_rows_sha256=sha(run.BASE/'residual-source-rows.npy'),
        heldout_samples_used_for_fit_or_selection=False,not_an_optimization_admission=True))

def feature_check():
    """Locate recovered patches and separate frozen support from extra exclusions."""
    import numpy as np
    from amd_cgal_ai_input import evaluate
    receipt=json.loads((run.BASE/'input/receipt.json').read_text())
    for path,key in ((run.PRIOR/'ai-input/samples.npz','historical_samples_sha256'),(run.BASE/'input/validation.npz','validation_sha256')):
        if sha(path)!=receipt[key]:raise ValueError('Validation identity differs')
    z=np.load(run.PRIOR/'ai-input/samples.npz',allow_pickle=False)
    src=np.load(run.BASE/'input/source-analysis.npz',allow_pickle=False)
    D=float(src['diagonal']);center=src['center'];scale=100/D
    planes=read_planes(np,run.BASE/'regions.csv')
    for plane in planes:plane['offset']=float(np.dot(plane['normal'],center)+plane['offset']/scale)
    train=~z['held'];allowed=train&src['active'][z['triangle_rows']];cosine=math.cos(math.radians(10))
    rows=[];full_eligible=[]
    for plane in planes:
        n=np.array(plane['normal']);mask=(abs(z['points']@n-plane['offset'])<=.002*D)&(abs(z['normal']@n)>=cosine)
        count=int((mask&train).sum());eligible=count>=24
        if eligible:full_eligible.append(plane)
        rows.append(dict(region=plane['id'],faces=plane['faces'],original_training_support=count,
            training_support_excluding_locked_triangles=int((mask&allowed).sum()),historical_holdout_support=int((mask&z['held']).sum()),
            passes_original_training24=eligible))
    all_planes,all_dist,all_labels=evaluate(np,z['points'][z['held']],z['normal'][z['held']],planes,D)
    all_frozen,_,_=evaluate(np,z['points'][z['held']],z['normal'][z['held']],full_eligible,D)
    # Genuine original patch bounds/area, not infinite-plane support hulls.
    labels=np.fromfile(run.BASE/'region-labels.bin',dtype='<i4')
    v=np.load(RAW/'vertices.npy',allow_pickle=False);f=np.load(RAW/'faces.npy',allow_pickle=False)
    if labels.shape!=(len(f),) or np.any(labels>=len(planes)) or np.any(labels < -1):raise ValueError('Region label scope')
    for row in rows:
        ids=np.flatnonzero(labels==row['region'])
        if len(ids)!=row['faces']:raise ValueError('Native patch membership differs')
        area=src['area'][ids];points=v[f[ids]].astype('f8');centroids=points.mean(axis=1)
        n=np.array(planes[row['region']]['normal']);offset=planes[row['region']]['offset']
        row.update(area=float(area.sum()),centroid=(np.sum(centroids*area[:,None],axis=0)/area.sum()).tolist(),
            bounds=[points.min(axis=(0,1)).tolist(),points.max(axis=(0,1)).tolist()],
            max_vertex_plane_residual=float(abs(points@n-offset).max()),
            maximum_unsigned_normal_angle_degrees=float(np.degrees(np.arccos(np.clip(abs(src['normal'][ids]@n),0,1))).max()))
    save(run.BASE/'feature-check.json',dict(classification='NO_GO' if not all_frozen['gate_pass'] else 'REQUIRES_FURTHER_ADMISSION',
        original_training_count=int(train.sum()),admissible_training_count=int(allowed.sum()),
        all_51_planes_diagnostic_only=all_planes,all_original_training24_eligible=all_frozen,
        full_original_training24_eligible_plane_count=len(full_eligible),patches=rows,
        all_planes_not_a_candidate_or_admission=True,planes_never_fit_from_either_validation_sample_set=True,
        source_labels_sha256=sha(run.BASE/'region-labels.bin'),regions_sha256=sha(run.BASE/'regions.csv'),
        exact_source_feature_recovered_does_not_override_frozen_support=True,
        maximum_allowed_historical_misses=69,actual_selected_plane_misses=91,minimum_additional_coverage_needed=22,
        MIP_executed=False,AI_candidate_created=False))

def augment():
    """One authorized real-triangle sampling supplement, never a new plane fit."""
    import numpy as np
    from amd_cgal_ai_input import evaluate
    supplement=run.BASE/'sampling-supplement.json'
    declared=json.loads(supplement.read_text())
    if declared['candidate_samples_per_patch']!=4096 or declared['retained_samples_per_patch']!=256:raise ValueError('Declared sampling differs')
    z=np.load(run.PRIOR/'ai-input/samples.npz',allow_pickle=False);src=np.load(run.BASE/'input/source-analysis.npz',allow_pickle=False)
    v=np.load(RAW/'vertices.npy',allow_pickle=False);f=np.load(RAW/'faces.npy',allow_pickle=False)
    labels=np.fromfile(run.BASE/'region-labels.bin',dtype='<i4');D=float(src['diagonal']);center=src['center'];scale=100/D
    lo=v.min(axis=0).astype('f8');hi=v.max(axis=0).astype('f8')
    planes=read_planes(np,run.BASE/'regions.csv')
    newp=[];newn=[];newrows=[];newweights=[];counts=[]
    for row in planes:
        i=row['id'];row['offset']=float(np.dot(row['normal'],center)+row['offset']/scale)
        ids=np.flatnonzero((labels==i)&src['active']);cdf=np.cumsum(src['area'][ids])
        chosen=ids[np.searchsorted(cdf,(np.arange(4096)+.5)/4096*cdf[-1])]
        rng=np.random.default_rng(20261102+i);uv=rng.random((4096,2));r=np.sqrt(uv[:,0]);w=np.column_stack((1-r,r*(1-uv[:,1]),r*uv[:,1]))
        pts=(v[f[chosen]].astype('f8')*w[:,:,None]).sum(axis=1)
        block=np.clip(np.floor((pts-lo)/(hi-lo)*16).astype('i8'),0,15)
        historical=(((block[:,0]*73856093)^(block[:,1]*19349663)^(block[:,2]*83492791))%5)==0
        valid=np.flatnonzero(~historical)
        if len(valid)>256:valid=valid[np.floor((np.arange(256)+.5)/256*len(valid)).astype('i8')]
        counts.append(dict(region=i,retained=len(valid),distinct_triangles=int(len(np.unique(chosen[valid])))))
        newp.extend(pts[valid]);newn.extend(src['normal'][chosen[valid]]);newrows.extend(chosen[valid]);newweights.extend(w[valid])
    newp=np.array(newp);newn=np.array(newn);newrows=np.array(newrows,'i4');newweights=np.array(newweights)
    if np.any(~src['active'][newrows]):raise ValueError('Locked triangle entered training')
    train=(~z['held'])&src['active'][z['triangle_rows']]
    p=np.concatenate([z['points'][train],newp]);n=np.concatenate([z['normal'][train],newn])
    masks=[];eligible=[];cosine=math.cos(math.radians(10))
    for plane in planes:
        nn=np.array(plane['normal']);mask=(abs(p@nn-plane['offset'])<=.002*D)&(abs(n@nn)>=cosine)
        if mask.sum()>=24:eligible.append(plane);masks.append(mask)
    masks=np.array(masks);uncovered=np.ones(len(p),bool);selected=[];indices=list(range(len(eligible)))
    while indices and len(selected)<48:
        score=(masks[indices]&uncovered).sum(axis=1);which=int(np.argmax(score))
        if score[which]==0:break
        index=indices.pop(which);selected.append(dict(eligible[index]));uncovered[masks[index]]=False
    for i,row in enumerate(selected):row['source_region']=row['id'];row['id']=i
    training,dist,assigned=evaluate(np,p,n,selected,D);good=dist<=.002*D
    counts_selected=[int((good&(assigned==i)).sum()) for i in range(len(selected))]
    historical,_,_=evaluate(np,z['points'][z['held']],z['normal'][z['held']],selected,D)
    extra=np.load(run.BASE/'input/validation.npz',allow_pickle=False)
    locked,_,_=evaluate(np,extra['points'],extra['normal'],selected,D)
    passed=4<=len(selected)<=48 and min(counts_selected,default=0)>=24 and historical['gate_pass'] and locked['gate_pass']
    out=run.BASE/'augmented';out.mkdir(exist_ok=False)
    with (out/'samples.npz').open('xb') as stream:np.savez(stream,points=p,normal=n,labels=assigned,good=good,new_points=newp,new_triangle_rows=newrows,new_barycentric=newweights,base_training_rows=np.flatnonzero(train),center=center,scale=scale,diagonal=D)
    native=(p-center)*scale
    with (out/'train.ply').open('x') as stream:
        stream.write('ply\nformat ascii 1.0\nelement vertex '+str(int(good.sum()))+'\nproperty double x\nproperty double y\nproperty double z\nproperty double nx\nproperty double ny\nproperty double nz\nproperty int segment_index\nend_header\n')
        for k in np.flatnonzero(good):
            nn=n[k].copy()
            if nn@np.array(selected[int(assigned[k])]['normal'])<0:nn=-nn
            stream.write(' '.join(format(float(x),'.17g') for x in [*native[k],*nn])+' '+str(int(assigned[k]))+'\n')
    save(run.BASE/'augmented-plane-screen.json',dict(classification='PASS' if passed else 'NO_GO',planes=selected,plane_count=len(selected),training=training,strict_training_support=counts_selected,historical_holdout=historical,locked_validation=locked,new_samples=len(newp),per_region_sampling=counts,PLY_sha256=sha(out/'train.ply'),samples_sha256=sha(out/'samples.npz'),supplement_sha256=sha(supplement),historical_and_extra_coordinates_never_fit=True,all_additional_samples_genuine_AI_triangle_barycentrics=True,original_only_support_restriction_was_an_inference_and_is_not_applied=True))

def screen_refined(destination=None):
    import numpy as np
    from amd_cgal_ai_input import evaluate
    previous=json.loads((run.BASE/'augmented-plane-screen.json').read_text())
    if sha(run.BASE/'augmented/samples.npz')!=previous['samples_sha256']:raise ValueError('Augmented data differs')
    z=np.load(run.BASE/'augmented/samples.npz',allow_pickle=False);D=float(z['diagonal']);center=z['center'];scale=float(z['scale'])
    destination=destination or run.BASE
    planes=read_planes(np,destination/'arrangement/prepared-planes.csv')
    members=np.loadtxt(destination/'arrangement/prepared-members.csv',delimiter=',',skiprows=1,dtype='i8')
    good=np.flatnonzero(z['good']);p=z['points'][good];n=z['normal'][good]
    if len(members)!=len(good) or not np.array_equal(np.sort(members[:,1]),np.arange(len(good))):raise ValueError('CGAL membership completeness')
    support=[];all_valid=True
    for plane in planes:
        plane['offset']=float(np.dot(plane['normal'],center)+plane['offset']/scale)
        rows=members[members[:,0]==plane['id'],1];nn=np.array(plane['normal'])
        valid=(abs(p[rows]@nn-plane['offset'])<=.002*D)&(abs(n[rows]@nn)>=math.cos(math.radians(10)))
        strict=int(valid.sum());all_valid=all_valid and strict==len(rows)
        support.append(dict(group=plane['id'],members=len(rows),strict=strict))
    hist=np.load(run.PRIOR/'ai-input/samples.npz',allow_pickle=False);extra=np.load(run.BASE/'input/validation.npz',allow_pickle=False)
    a,_,_=evaluate(np,hist['points'][hist['held']],hist['normal'][hist['held']],planes,D)
    b,_,_=evaluate(np,extra['points'],extra['normal'],planes,D)
    # Finite support is original triangle membership, including the disconnected
    # regions and their open/junction edges; no AABB/hull replaces this domain.
    src=np.load(run.BASE/'input/source-analysis.npz',allow_pickle=False)
    v=np.load(RAW/'vertices.npy',allow_pickle=False);f=np.load(RAW/'faces.npy',allow_pickle=False)
    region=np.fromfile(run.BASE/'region-labels.bin',dtype='<i4')
    source_rows=np.concatenate([hist['triangle_rows'][z['base_training_rows']],z['new_triangle_rows']])[good]
    finite_groups=[]
    for row in support:
        ids=members[members[:,0]==row['group'],1];original=np.unique(source_rows[ids])
        points=v[f[original]].astype('f8');normal=np.array(planes[row['group']]['normal']);offset=planes[row['group']]['offset']
        valid=(abs(points@normal-offset).max(axis=1)<=.002*D)&(abs(src['normal'][original]@normal)>=math.cos(math.radians(10)))
        actual=original[valid]
        finite_groups.append(dict(group=row['group'],distinct_source_triangles=len(original),strict_source_triangles=len(actual),
            source_regions=np.unique(region[actual]).tolist(),strict_area=float(src['area'][actual].sum()),
            bounds=[v[f[actual]].min(axis=(0,1)).tolist(),v[f[actual]].max(axis=(0,1)).tolist()] if len(actual) else None))
    patch_support=[];patch_masks=[]
    for label in range(int(region.max())+1):
        ids=np.flatnonzero(region==label);tri=v[f[ids]].astype('f8');normal=src['normal'][ids]
        best=None
        for plane in planes:
            nn=np.array(plane['normal']);valid=(abs(tri@nn-plane['offset']).max(axis=1)<=.002*D)&(abs(normal@nn)>=math.cos(math.radians(10)))
            area=float(src['area'][ids[valid]].sum())
            if best is None or area>best[0]:best=(area,plane['id'],valid)
        area,group,valid=best;patch_masks.append((ids,valid,group))
        patch_support.append(dict(region=label,group=group,source_triangles=len(ids),strict_triangles=int(valid.sum()),
            source_area=float(src['area'][ids].sum()),strict_area=area,strict_area_fraction=area/float(src['area'][ids].sum()),
            finite_domain='original triangle-index set; disconnected support and source holes retained, never convexified'))
    with (destination/'finite-source-domains.npz').open('xb') as stream:
        np.savez(stream,source_triangle_rows=np.concatenate([ids[mask] for ids,mask,_ in patch_masks]),
            group=np.concatenate([np.full(int(mask.sum()),g,'i4') for _,mask,g in patch_masks]))
    finite_pass=all(g['strict_source_triangles']>=24 and g['strict_area']>0 for g in finite_groups) and all(p['strict_triangles']==p['source_triangles'] for p in patch_support)
    passed=4<=len(planes)<=48 and all_valid and finite_pass and min((q['strict'] for q in support),default=0)>=24 and a['gate_pass'] and b['gate_pass']
    save(destination/'refined-screen.json',dict(classification='PASS' if passed else 'NO_GO',plane_count=len(planes),planes=planes,support=support,historical_holdout=a,locked_validation=b,all_actual_final_CGAL_members_strict=all_valid,finite_source_groups=finite_groups,original_patch_survival=patch_support,finite_source_domain_prerequisite=finite_pass,domains_SHA256=sha(destination/'finite-source-domains.npz'),planes_SHA256=sha(destination/'arrangement/prepared-planes.csv'),members_SHA256=sha(destination/'arrangement/prepared-members.csv'),candidate_arrangement_SHA256=sha(destination/'arrangement/candidates.off'),no_MIP_yet=True,final_output_footprints_still_require_verification=True))
