"""AI-only replay, train-only native input, and strict unchanged support screen."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import amd_cgal_proof as owner
BASE=owner.BASE
RAW=owner.ROOT.parent/'artifacts/amd-closed-solid-raw-recovery-v1/mesh'
PRIOR=owner.ROOT.parent/'artifacts/amd-ai-hard-surface-assessment-20261030-v1'


def frozen():
    path=owner.ROOT/'docs/compatibility/amd-ai-hard-surface-criteria.json'
    if owner.sha(path)!='397bf6c2dc8908e8a3b1a02ba169d6ec6ab5bc2aaf11ee96a4cb84e81591aad0':raise ValueError('Frozen Phase11 criteria differs')
    return json.loads(path.read_text())


def guard():
    from amd_zombie_d_compare import plain
    artifacts=owner.ROOT.parent/'artifacts'
    allowed={PRIOR/'source-sample-inspection.npz',PRIOR/'evidence-manifest.json'}
    def audit(event,args):
        if event=='import' and str(args[0]).startswith('amd_closed_solid_'):
            raise RuntimeError('Ground-truth benchmark import unavailable to AI processing')
        if event=='open' and isinstance(args[0],(str,bytes,Path)):
            p=plain(args[0])
            if p.name in ('amd_closed_solid_benchmark.py','amd-closed-solid-reference.json'):
                raise RuntimeError('Ground-truth source unavailable')
            if p.is_relative_to(artifacts) and not (p.is_relative_to(BASE) or p.is_relative_to(RAW) or p in allowed):
                raise RuntimeError('Non-AI artifact unavailable')
    sys.addaudithook(audit)


def original_plane(np, normal, native_offset, center, scale):
    return float(np.dot(normal,center)+native_offset/scale)


def evaluate(np,points,normals,planes,diagonal):
    best=np.full(len(points),np.inf);nearest=np.full(len(points),-1,np.int32)
    cosine=math.cos(math.radians(10))
    for row in planes:
        n=np.asarray(row['normal']);d=abs(points@n-row['offset'])
        use=(abs(normals@n)>=cosine)&(d<best)
        best[use]=d[use];nearest[use]=row['id']
    covered=best<=.002*diagonal
    return dict(points=len(points),covered=int(covered.sum()),coverage=float(covered.mean()),
                no_compatible_normal=int((~np.isfinite(best)).sum()),
                gate_minimum=.995,gate_pass=bool(covered.mean()>=.995),
                infinite_plane_distance_is_not_surface_distance=True),best,nearest


def prepare():
    guard()
    import numpy as np
    frozen()
    if json.loads((BASE/'official-sample-verification.json').read_text())['classification']!='PASS_OFFICIAL_SAMPLE_SOLVER_AND_EXPORT':raise ValueError('Official sample gate')
    auth=json.loads((BASE/'authentication.json').read_text())
    for name,r in auth['mesh_artifacts'].items():
        if owner.sha(RAW/name)!=r['sha256']:raise ValueError('Original AI source hash differs')
    source=PRIOR/'source-sample-inspection.npz'
    expected=json.loads((PRIOR/'evidence-manifest.json').read_text())['source-sample-inspection.npz']
    if owner.sha(source)!=expected['sha256'] or source.stat().st_size!=expected['bytes']:raise ValueError('Historical sample identity differs')
    v=np.load(RAW/'vertices.npy',mmap_mode='r',allow_pickle=False)
    f=np.load(RAW/'faces.npy',mmap_mode='r',allow_pickle=False)
    try:
        lo=v.min(axis=0).astype('f8');hi=v.max(axis=0).astype('f8');center=(lo+hi)/2;diagonal=float(np.linalg.norm(hi-lo));scale=100/diagonal
        areas=np.empty(len(f),np.float64)
        for start in range(0,len(f),32768):
            tri=v[f[start:start+32768]].astype('f8')
            areas[start:start+len(tri)]=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5
        if not np.isfinite(areas).all() or np.any(areas<=0):raise ValueError('Finite positive original areas required')
        cdf=np.cumsum(areas);ids=np.searchsorted(cdf,(np.arange(65536)+.5)/65536*cdf[-1])
        tri=v[f[ids]].astype('f8');normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None]
        uv=np.random.default_rng(20261030).random((65536,2));r=np.sqrt(uv[:,0]);weights=np.column_stack((1-r,r*(1-uv[:,1]),r*uv[:,1]));points=(tri*weights[:,:,None]).sum(axis=1)
        block=np.clip(np.floor((points-lo)/(hi-lo)*16).astype('i8'),0,15)
        held=(((block[:,0]*73856093)^(block[:,1]*19349663)^(block[:,2]*83492791))%5)==0
        with np.load(source,allow_pickle=False) as old:
            if not np.array_equal(ids,old['triangle_rows']) or not np.array_equal(held,old['heldout']):raise ValueError('Original area sampling/split differs')
            if not np.allclose(points,old['points'],atol=2e-15,rtol=0) or not np.allclose(normal,old['normal'],atol=2e-15,rtol=0):raise ValueError('F64 sample replay differs')
        out=BASE/'ai-input';out.mkdir(exist_ok=False)
        # Original coordinates and IDs remain separate from reversible native units.
        with (out/'samples.npz').open('xb') as stream:np.savez(stream,points=points,normal=normal,held=held,triangle_rows=ids,center=center,scale=scale,diagonal=diagonal)
        native=(points-center)*scale;train=np.flatnonzero(~held)
        with (out/'train-native.ply').open('x') as stream:
            stream.write('ply\nformat ascii 1.0\nelement vertex '+str(len(train))+'\nproperty double x\nproperty double y\nproperty double z\nproperty double nx\nproperty double ny\nproperty double nz\nproperty uint sample_id\nend_header\n')
            for i in train:stream.write(' '.join(format(float(x),'.17g') for x in [*native[i],*normal[i]])+' '+str(i)+'\n')
        owner.save(out/'receipt.json',dict(classification='PASS_SOURCE_ONLY_TRAIN_INPUT',source_mesh_sha256=auth['mesh_artifacts']['mesh.ply']['sha256'],
            historical_samples_sha256=expected['sha256'],deterministic_original_sampling_replayed=True,
            training_points=len(train),heldout_points=int(held.sum()),heldout_not_written_to_native_input=True,
            AI_diagonal=diagonal,center=center.tolist(),scale=scale,native_diagonal=100,
            native_input_sha256=owner.sha(out/'train-native.ply'),samples_sha256=owner.sha(out/'samples.npz'),
            ground_truth_used=False,numpy=np.__version__,replay_F64_roundoff_tolerance=2e-15))
        for name,r in auth['mesh_artifacts'].items():
            if owner.sha(RAW/name)!=r['sha256']:raise ValueError('AI source changed')
    finally:v._mmap.close();f._mmap.close()


def assess():
    guard()
    import numpy as np
    frozen()
    receipt=json.loads((BASE/'ai-input/receipt.json').read_text())
    if owner.sha(BASE/'ai-input/samples.npz')!=receipt['samples_sha256']:raise ValueError('Sample identity differs')
    with np.load(BASE/'ai-input/samples.npz',allow_pickle=False) as data:
        points=data['points'];normals=data['normal'];held=data['held'];center=data['center'];scale=float(data['scale']);diagonal=float(data['diagonal'])
    path=BASE/'ai-planes/assigned.ply'
    with path.open() as stream:
        header=[]
        while True:
            line=stream.readline()
            if not line:raise ValueError('Truncated PLY header')
            header.append(line)
            if line.strip()=='end_header':break
        if len(header)>40:raise ValueError('Bounded native PLY header')
        rows=np.loadtxt(stream,dtype='f8',ndmin=2)
    if rows.shape!=(int((~held).sum()),8):raise ValueError('Complete native train point records required')
    ids=rows[:,7].astype('i8');labels=rows[:,6].astype('i8')
    if not np.array_equal(ids,rows[:,7]) or not np.array_equal(labels,rows[:,6]) or not np.array_equal(np.sort(ids),np.flatnonzero(~held)):raise ValueError('Native sample ID membership differs')
    expected=(points[ids]-center)*scale
    if not np.array_equal(rows[:,:3],expected) or not np.array_equal(rows[:,3:6],normals[ids]):raise ValueError('Native changed positions/normals during reorder')
    planes=[];support=[];strict_labels=np.full(len(ids),-1,np.int32)
    with (BASE/'ai-planes/planes.csv').open() as stream:
        for row in csv.DictReader(stream):
            i=int(row['id']);n=np.array([float(row[k]) for k in ('nx','ny','nz')]);offset=float(row['offset'])
            if i!=len(planes) or not np.isfinite(n).all() or not math.isfinite(offset) or abs(np.linalg.norm(n)-1)>1e-10:raise ValueError('Native finite unit plane required')
            offset=original_plane(np,n,offset,center,scale);plane=dict(id=i,normal=n.tolist(),offset=offset)
            assigned=labels==i;distance=abs(points[ids]@n-offset);angular=abs(normals[ids]@n)>=math.cos(math.radians(10))
            strict=assigned&(distance<=.002*diagonal)&angular
            if assigned.sum()!=int(row['training_points']):raise ValueError('Native plane membership count differs')
            strict_labels[strict]=i
            support.append(dict(id=i,CGAL_assigned=int(assigned.sum()),strict_assigned=int(strict.sum()),rejected_native_members=int((assigned&~strict).sum()),
                                source_sample_bounds=[points[ids[strict]].min(axis=0).tolist(),points[ids[strict]].max(axis=0).tolist()] if strict.any() else None))
            planes.append(plane)
    if np.any(labels < -1) or np.any(labels >= len(planes)):raise ValueError('Unknown native plane label')
    training,_,_=evaluate(np,points[~held],normals[~held],planes,diagonal)
    validation,distance,nearest=evaluate(np,points[held],normals[held],planes,diagonal)
    qualified=4<=len(planes)<=48 and all(r['strict_assigned']>=24 for r in support) and validation['gate_pass']
    with (BASE/'ai-planes/strict-support.npz').open('xb') as stream:np.savez(stream,sample_ids=ids,CGAL_labels=labels,strict_labels=strict_labels,heldout_nearest=nearest,heldout_distance=distance)
    owner.save(BASE/'ai-plane-screen.json',dict(classification='PASS_PLANE_SCREEN_ONLY' if qualified else 'NO_GO_PLANE_SCREEN',
        plane_count=len(planes),planes=planes,training=training,spatial_holdout=validation,strict_support=support,
        all_planes_have_24_strict_training_samples=all(r['strict_assigned']>=24 for r in support),
        native_positions_normals_and_sample_IDs_preserved=True,original_quality_thresholds_unchanged=True,
        no_plane_recentering=True,no_ground_truth=True,
        finite_plane_footprints_verified=False,essential_bevel_corner_planes_verified=False,
        AI_optimization_authorized=False,AI_candidate_created=False,
        note='A passing infinite-plane screen still requires essential finite support and post-CGAL-refit checks before optimization. A failing screen is not proof of missing neural source geometry.'))
