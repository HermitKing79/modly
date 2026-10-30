"""Bounded AI-only plane assessment, NOT a box fitter or mesh constructor."""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT.parent/'artifacts/amd-ai-hard-surface-assessment-20261030-v1'
RAW=ROOT.parent/'artifacts/amd-closed-solid-raw-recovery-v1'
CRITERIA=ROOT/'docs/compatibility/amd-ai-hard-surface-criteria.json'
CRITERIA_SHA='397bf6c2dc8908e8a3b1a02ba169d6ec6ab5bc2aaf11ee96a4cb84e81591aad0'

def save(path,data):
    with path.open('x',encoding='utf8') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def criteria():
    raw=CRITERIA.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CRITERIA_SHA:raise ValueError('Frozen criteria identity differs')
    return json.loads(raw)

def assessment_guard(base):
    from amd_zombie_d_compare import plain,audit_policy
    ordinary=audit_policy(base);artifacts=ROOT.parent/'artifacts'
    def guard(event,args):
        ordinary(event,args)
        if event=='import' and str(args[0]).startswith('amd_closed_solid_'):
            raise RuntimeError('Ground-truth/benchmark modules unavailable to surface assessment')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=plain(os.fsdecode(args[0]))
            if p.name in ('amd_closed_solid_benchmark.py','amd-closed-solid-reference.json'):
                raise RuntimeError('Ground-truth construction source denied')
            if p.is_relative_to(artifacts) and not (p.is_relative_to(base) or p.is_relative_to(RAW/'mesh')):
                raise RuntimeError('Non-AI artifact unavailable to isolated surface assessment')
    return guard

def fit_planes(points,normals,diagonal,settings):
    """Greedy unsigned-normal/offset clustering. Residuals remain unclassified."""
    import numpy as np
    if points.shape!=normals.shape or points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all() or not np.isfinite(normals).all() or diagonal<=0:
        raise ValueError('Finite measured points/normals required')
    labels=np.full(len(points),-1,np.int32);eligible=np.ones(len(points),bool);planes=[]
    canonical=normals.copy();sign=np.where(canonical[np.arange(len(normals)),np.argmax(abs(canonical),axis=1)]<0,-1.,1.);canonical*=sign[:,None]
    limit=settings['point_to_plane_limit_over_AI_diagonal']*diagonal
    cosine=math.cos(math.radians(settings['normal_angle_degrees']));minimum=settings['minimum_training_support']
    for attempt in range(settings['maximum_hypotheses']):
        ids=np.flatnonzero(eligible)
        if len(ids)<minimum or len(planes)>=settings['maximum_planes']:break
        bins=np.rint(canonical[ids]/settings['normal_bin_width']).astype(np.int16)
        _,inverse,counts=np.unique(bins,axis=0,return_inverse=True,return_counts=True)
        seed=ids[inverse==np.argmax(counts)];normal=canonical[seed].mean(axis=0);normal/=np.linalg.norm(normal)
        angular=ids[abs(normals[ids]@normal)>=cosine];offsets=points[angular]@normal
        slots=np.floor(offsets/limit).astype(np.int64);values,num=np.unique(slots,return_counts=True)
        offset=(values[np.argmax(num)]+.5)*limit
        selected=angular[abs(offsets-offset)<=2*limit]
        for _ in range(settings['plane_refits']):
            if len(selected)<minimum:break
            center=points[selected].mean(axis=0);_,singular,basis=np.linalg.svd(points[selected]-center,full_matrices=False)
            normal=basis[-1];offset=float(center@normal)
            selected=ids[(abs(normals[ids]@normal)>=cosine)&(abs(points[ids]@normal-offset)<=limit)]
        if len(selected)<minimum:eligible[seed]=False;continue
        center=points[selected].mean(axis=0);_,singular,basis=np.linalg.svd(points[selected]-center,full_matrices=False)
        normal=basis[-1];offset=float(center@normal);d=abs(points[selected]@normal-offset)
        plane=dict(id=len(planes),normal=normal.tolist(),offset=offset,training_points=len(selected),
            distance_RMS=float(np.sqrt(np.mean(d*d))),distance_p95=float(np.quantile(d,.95)),distance_max=float(d.max()),
            sample_bounds=[points[selected].min(axis=0).tolist(),points[selected].max(axis=0).tolist()],
            singular_values=singular.tolist(),finite_polygon_footprint='UNVERIFIED; sample bounds are not faces or polygons')
        labels[selected]=len(planes);eligible[selected]=False;planes.append(plane)
    return planes,labels

def evaluate(points,normals,planes,diagonal,settings):
    import numpy as np
    best=np.full(len(points),np.inf);label=np.full(len(points),-1,np.int32)
    cosine=math.cos(math.radians(settings['normal_angle_degrees']))
    for plane in planes:
        n=np.asarray(plane['normal']);distance=abs(points@n-plane['offset']);valid=abs(normals@n)>=cosine
        replace=valid&(distance<best);best[replace]=distance[replace];label[replace]=plane['id']
    limit=settings['point_to_plane_limit_over_AI_diagonal']*diagonal;covered=best<=limit;finite=np.isfinite(best)
    return dict(points=len(points),covered=int(covered.sum()),coverage=float(covered.mean()),
        no_angularly_compatible_plane=int((~finite).sum()),
        finite_distance_quantiles_over_AI_diagonal={str(q):float(np.quantile(best[finite]/diagonal,q)) for q in (.5,.95,.99,1)} if finite.any() else None,
        per_plane=[dict(id=p['id'],nearest_points=int((label==p['id']).sum()),covered=int(((label==p['id'])&covered).sum())) for p in planes],
        infinite_plane_distance_is_not_surface_distance=True),label,best

def authenticate():
    # This separate child may inspect old provenance. It exports no GT geometry/dimensions.
    import amd_closed_solid_forensics as old
    frozen=criteria();arrays={};v=f=None
    try:
        arrays,v,f,auth,snapshot,reader=old.authenticated();reader.verify_snapshot(snapshot)
        mesh=auth['mesh']
        save(BASE/'authentication.json',dict(classification='PASS_AI_SOURCE_AUTHENTICATION',binding=auth['binding'],
            vertices=mesh['vertices'],triangles=mesh['triangles'],mesh_artifacts=mesh['artifacts'],
            comparison_files={str(p.relative_to(RAW)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in (RAW/'view/raw-reference-camera.png',RAW/'view/render.json',RAW/'comparison/comparison.png',RAW/'comparison/comparison.json',RAW/'raw-quality.json')},
            ground_truth_parameters_exported=False,original_snapshot=snapshot))
        if criteria()!=frozen:raise ValueError('Frozen criteria changed during authentication')
    finally:
        for a in [*arrays.values(),v,f]:
            if a is not None:a._mmap.close()

def assess():
    import numpy as np
    began=time.monotonic();frozen=criteria();recorded=json.loads((BASE/'predeclared-criteria.json').read_text())
    if recorded['sha256']!=CRITERIA_SHA or recorded['criteria']!=frozen:raise ValueError('Predeclared criteria differs')
    settings=frozen['assessment']
    auth=json.loads((BASE/'authentication.json').read_text())
    for name,r in auth['mesh_artifacts'].items():
        if digest(RAW/'mesh'/name)!=r['sha256'] or (RAW/'mesh'/name).stat().st_size!=r['bytes']:raise ValueError('Original AI source differs')
    v=np.load(RAW/'mesh/vertices.npy',mmap_mode='r',allow_pickle=False);f=np.load(RAW/'mesh/faces.npy',mmap_mode='r',allow_pickle=False)
    try:
        lo=v.min(axis=0).astype('f8');hi=v.max(axis=0).astype('f8');diagonal=float(np.linalg.norm(hi-lo))
        areas=np.empty(len(f),np.float64)
        for start in range(0,len(f),32768):
            p=v[f[start:start+32768]].astype('f8');areas[start:start+len(p)]=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)*.5
        if not np.isfinite(areas).all() or np.any(areas<=0):raise ValueError('Finite nonzero original triangles required')
        count=settings['area_stratified_samples'];cdf=np.cumsum(areas);ids=np.searchsorted(cdf,(np.arange(count)+.5)/count*cdf[-1])
        tri=v[f[ids]].astype('f8');normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None]
        rng=np.random.default_rng(settings['seed']);uv=rng.random((count,2));r=np.sqrt(uv[:,0]);weights=np.column_stack((1-r,r*(1-uv[:,1]),r*uv[:,1]))
        points=(tri*weights[:,:,None]).sum(axis=1)
        block=np.clip(np.floor((points-lo)/(hi-lo)*settings['spatial_blocks_per_axis']).astype('i8'),0,settings['spatial_blocks_per_axis']-1)
        hashed=(block[:,0]*73856093)^(block[:,1]*19349663)^(block[:,2]*83492791)
        held=(hashed%settings['holdout_hash_modulus'])==settings['holdout_hash_remainder']
        planes,labels=fit_planes(points[~held],normal[~held],diagonal,settings)
        validation,assigned,distance=evaluate(points[held],normal[held],planes,diagonal,settings)
        training,_,_=evaluate(points[~held],normal[~held],planes,diagonal,settings)
        with (BASE/'source-sample-inspection.npz').open('xb') as out:np.savez(out,points=points,triangle_rows=ids,normal=normal,heldout=held,training_plane_labels=labels,heldout_plane_labels=assigned,heldout_plane_residual=distance)
        save(BASE/'plane-assessment.json',dict(classification='READ_ONLY_AI_SURFACE_ASSESSMENT',source_PLY_sha256=auth['mesh_artifacts']['mesh.ply']['sha256'],
            vertices=len(v),triangles=len(f),AI_bounds=[lo.tolist(),hi.tolist()],AI_diagonal=diagonal,
            source_area_including_overlaps=float(cdf[-1]),sampling='Area-stratified original triangles; opposite winding treated identically; overlaps are not independent exterior evidence',
            training=training,spatial_holdout=validation,planes=planes,unassigned_training_samples=int((labels<0).sum()),
            plane_coverage_gate_pass=validation['coverage']>=settings['minimum_heldout_coverage'],
            finite_plane_footprints_verified=False,plane_junction_graph_verified=False,
            qualified_nonconvex_reconstruction_implementation_available=False,
            candidate_created=False,clean_asset_promoted=False,
            criteria_sha256=CRITERIA_SHA,decision='NO_GO_FOR_AD_HOC_BOX_RECONSTRUCTION_UNDER_ENGINEERING_PRIORITY',
            reason='Infinite-plane fitting is not general finite-patch reconstruction; no box template, convexification, fallback or geometry is constructed',
            elapsed_seconds=time.monotonic()-began,ground_truth_read_by_assessment=False))
        for name,r in auth['mesh_artifacts'].items():
            if digest(RAW/'mesh'/name)!=r['sha256']:raise ValueError('AI source changed during assessment')
        if criteria()!=frozen or json.loads((BASE/'predeclared-criteria.json').read_text())!=recorded:raise ValueError('Frozen criteria changed during assessment')
    finally:v._mmap.close();f._mmap.close()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--worker',choices=('authenticate','assess'));a=p.parse_args(argv)
    if a.worker:
        if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Exact owned GO plus EOF required')
        if a.worker=='authenticate':
            import amd_closed_solid_benchmark as b
            sys.addaudithook(b.cpu_guard(BASE))
        else:sys.addaudithook(assessment_guard(BASE))
        sys.path.insert(0,'D:/AI/Trellis2AMD/venv/Lib/site-packages')
        (authenticate if a.worker=='authenticate' else assess)();return 0
    if a.run:
        import amd_zombie_cpu_mesh as owner
        owner.BASE=BASE;owner.__file__=str(Path(__file__).resolve())
        save(BASE/'predeclared-criteria.json',dict(sha256=CRITERIA_SHA,criteria=criteria(),recorded_before_source_assessment=True))
        for stage in ('authenticate','assess'):
            if owner.supervise(stage):return 1
        return 0
    print('INERT: AI-only assessment, no reconstruction or neural execution entry');return 0

if __name__=='__main__':raise SystemExit(main())
