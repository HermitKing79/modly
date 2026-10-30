"""Read-only four-quad/logit census of the saved box. No repair or neural route."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct
import sys
import time
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT.parent/'artifacts/amd-closed-solid-support-census-20261030-v1'
PREVIOUS=ROOT.parent/'artifacts/amd-closed-solid-forensics-20261030-v1'
MANIFEST_SHA='72e9714758dd2d2b81d579664a75dc4b8d26349566c9b48f2f22da88d5ef7f9f'
COMPLETION=BASE/'completion-v1'
SAVED_DATA={'edge-census.jsonl':'246907ced0d139f049e6f5a791d0c1f90150d227442830d754c1f6b96b95b453',
    'unique-candidates.json':'7f97f7ccc03874fa01a182cc08dc229f1ebb918f1911126ff50dde0b4d86dca9',
    'preservation-before.json':'1296b6a9d85952f450e3ca6a255b2e993c3eac2ac7cad6cfec3346e17fe5e436',
    'analyze-run/host-result.json':'3615e4ec71c86767b5ce02b726dc8548b760482a2fef66290ae312394fd44d50',
    'analyze-run/stderr.txt':'175ca3235167855c43d87c15e9751bbae846fae4e2bb83a6bafc0ade3d3a7f24',
    'candidate-neighborhoods.json':'a5338d955b09046ec0b16f3a9625b7955c763aae07dc722e474e6078745f932f',
    'authentication.json':'fad081a6f9ae1394789991a981b76db61aae052a2b25c0c0a6a1cee9b659d154',
    'summary.json':'dcebab7b02fc0e3e0620cb727956b6e31e49b5dbec8b93cb472019f40456b938',
    'progress.jsonl':'f3e2ceb2e1a632eb88f7d70d062cd61ff4c1e1278c10db2180debe193230639f',
    'analyze-run/stdout.txt':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'}
OFFSETS=(((0,0,0),(0,0,1),(0,1,1),(0,1,0)),
         ((0,0,0),(1,0,0),(1,0,1),(0,0,1)),
         ((0,0,0),(0,1,0),(1,1,0),(1,0,0)))


def candidates(edge,coords,features,lookup,quad_index,quads):
    """Four possible incident lattice squares; null flag/logit for absent anchor."""
    import numpy as np
    a,b=coords[edge];delta=np.abs(a-b)
    if np.count_nonzero(delta)!=1 or delta.max()!=1:raise ValueError('Unit axial source edge required')
    direction=int(np.argmax(delta));low=np.minimum(a,b);result=[]
    for axis in range(3):
        if axis==direction:continue
        other=next(k for k in range(3) if k not in (axis,direction))
        for shift in (-1,0):
            anchor=low.copy();anchor[other]+=shift;key=tuple(int(x) for x in anchor)
            corners=[tuple(key[k]+p[k] for k in range(3)) for p in OFFSETS[axis]]
            ids=[lookup.get(p) for p in corners];bounds=[all(0<=x<512 for x in p) for p in corners]
            complete=all(i is not None for i in ids) and all(bounds)
            row=lookup.get(key);logit=bits=flag=None;actual=-1
            if row is not None:
                raw=features[row,3+axis]
                if raw.dtype!=np.dtype('f4') or not np.isfinite(raw):raise ValueError('Original finite F32 logit required')
                logit=float(raw);bits=struct.unpack('<I',struct.pack('<f',logit))[0];flag=bool(raw>0)
                actual=int(quad_index[row,axis])
            emitted=bool(complete and flag is True)
            if (actual>=0)!=emitted:raise ValueError('Original emitted quad membership differs')
            if emitted:
                if not np.array_equal(quads[actual],ids):raise ValueError('Original quad corner membership differs')
                pairs={tuple(sorted((int(quads[actual,i]),int(quads[actual,(i+1)%4])))) for i in range(4)}
                if tuple(sorted(map(int,edge))) not in pairs:raise ValueError('Candidate does not contribute to target edge')
            state=('absent_anchor' if row is None else 'active' if emitted else
                   'flagged_incomplete' if flag else 'off_complete' if complete else 'off_incomplete')
            result.append(dict(candidate_id=f'{key[0]},{key[1]},{key[2]}:{axis}',anchor_grid=list(key),axis=axis,
                anchor_exists=row is not None,anchor_vertex_index=row,corner_grid=[list(p) for p in corners],
                corner_vertex_indices=ids,corner_in_grid=bounds,all_corners_present=complete,
                support_class='complete' if complete else 'absent_anchor' if row is None else 'unsupported',
                feature_column=3+axis,flag_on=flag,logit=logit,logit_bits=bits,
                logit_hex=None if bits is None else f'{bits:08x}',quad_row=actual if emitted else None,
                quad_triangle_rows=[2*actual,2*actual+1] if emitted else [],contributes_to_edge=emitted,state=state))
    if len({r['candidate_id'] for r in result})!=4:raise ValueError('Four distinct incident candidates required')
    return result


def incidence_effect(before,active):
    if len(before)!=4 or any(type(n) is not int or n<0 for n in before) or (active and min(before)<1):
        raise ValueError('Four valid global perimeter incidence counts required')
    after=[n+(-1 if active else 1) for n in before]
    return dict(operation='analytical_deactivation' if active else 'analytical_activation',before=before,after=after,
        all_sides_become_two=all(n==2 for n in after),new_boundary_edges=sum(a!=1 and b==1 for a,b in zip(before,after)),
        new_nonmanifold_edges=sum(a<=2 and b>2 for a,b in zip(before,after)),
        erased_edges=sum(a>0 and b==0 for a,b in zip(before,after)),
        boundary_delta=sum(n==1 for n in after)-sum(n==1 for n in before),
        nonmanifold_delta=sum(n>2 for n in after)-sum(n>2 for n in before),
        geometry_or_flags_modified=False)


def perimeter_counts(q,edges,counts,n,*,keys=None):
    import numpy as np
    if keys is None:keys=edges[:,0].astype('u8')*n+edges[:,1].astype('u8')
    pairs=np.array([[q[i],q[(i+1)%4]] for i in range(4)],np.int64);pairs.sort(axis=1)
    query=pairs[:,0].astype('u8')*n+pairs[:,1].astype('u8');pos=np.searchsorted(keys,query)
    found=pos<len(keys);found[found]&=keys[pos[found]]==query[found]
    result=np.zeros(4,np.int32);result[found]=counts[pos[found]]
    return result.tolist()


def deduplicate(rows):
    unique={}
    for row in rows:
        key=row['candidate_id']
        if key in unique and unique[key]!=row:raise ValueError('Duplicate candidate observations disagree')
        unique[key]=row
    return unique


def logit_stats(values):
    import numpy as np
    available=np.array([x for x in values if x is not None],np.float64)
    if not np.isfinite(available).all():raise ValueError('Finite logits only')
    return dict(observations=len(values),available=len(available),unavailable=sum(x is None for x in values),
        negative=int((available<0).sum()),zero=int((available==0).sum()),positive=int((available>0).sum()),
        negative_zero=int(sum(x==0 and math.copysign(1,x)<0 for x in available)),
        quantiles=None if not len(available) else {str(q):float(np.quantile(available,q)) for q in (0,.01,.05,.25,.5,.75,.95,.99,1)},
        absolute_quantiles=None if not len(available) else {str(q):float(np.quantile(abs(available),q)) for q in (0,.01,.05,.25,.5,.75,.95,.99,1)},
        absolute_le={str(t):int((abs(available)<=t).sum()) for t in (.001,.01,.1,1)},
        magnitude_is_calibrated_probability=False,threshold='strict original F32 > 0; signed zeros are off')


def previous_authentication(reader):
    manifest=PREVIOUS/'evidence-manifest.json'
    if reader.digest(manifest)!=MANIFEST_SHA:raise ValueError('Phase 8 manifest identity differs')
    rows=reader.read(manifest);paths=[manifest]
    for name,row in rows.items():
        p=reader.plain(PREVIOUS/name);p.relative_to(PREVIOUS)
        if p.stat().st_size!=row['bytes'] or reader.digest(p)!=row['sha256']:raise ValueError('Phase 8 evidence differs: '+name)
        paths.append(p)
    publication=reader.read(PREVIOUS/'publication.json')
    if publication['commit']!='9795e16201dda8c7e4faa308ecd85a9f448a08c6':raise ValueError('Phase 8 commit differs')
    for name,sha in publication['published_files'].items():
        p=reader.plain(ROOT/name);p.relative_to(ROOT)
        if reader.digest(p)!=sha:raise ValueError('Phase 8 source changed: '+name)
        paths.append(p)
    return reader.snapshot(paths,allow_empty=True)


def summarize(records):
    result={}
    for kind in ('boundary','triple'):
        rows=[r for r in records if r['kind']==kind];observations=[c for r in rows for c in r['candidates']]
        unique=deduplicate(observations);states=Counter(c['state'] for c in observations)
        result[kind]=dict(edges=len(rows),candidate_observations=len(observations),unique_candidates=len(unique),
            candidate_states=dict(states),per_edge_complete_inactive_alternatives=dict(Counter(str(r['complete_inactive_alternatives']) for r in rows)),
            edges_with_complete_inactive_alternative=sum(r['complete_inactive_alternatives']>0 for r in rows),
            per_edge_incidence_balancing_candidates=dict(Counter(str(r['incidence_balancing_candidates']) for r in rows)),
            remaining_candidate_states=dict(Counter(r['remaining_candidate_state'] for r in rows)) if kind=='triple' else None,
            logits=dict(occurrences=logit_stats([c['logit'] for c in observations]),unique=logit_stats([c['logit'] for c in unique.values()]),
                        by_state={s:logit_stats([c['logit'] for c in unique.values() if c['state']==s]) for s in states}))
    return result


def analyze():
    import numpy as np
    import amd_closed_solid_forensics as old
    import amd_flexible_dual_grid_cpu as converter
    import amd_zombie_cpu_mesh as owner
    arrays={};v=f=None;began=time.monotonic()
    def save(name,data):owner.save(BASE/name,data)
    def phase(name):
        if time.monotonic()-began>175:raise TimeoutError('175-second CPU census worker bound')
        with (BASE/'progress.jsonl').open('a',encoding='utf8') as stream:
            stream.write(json.dumps(dict(phase=name,UTC=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-began))+'\n')
            stream.flush()
    try:
        phase('authenticate-ten-tensors-arrays-and-phase8')
        arrays,v,f,auth,snapshot,reader=old.authenticated();previous=previous_authentication(reader);snapshot.update(previous)
        for name,row in snapshot.items():row['file_id']=Path(name).stat().st_ino
        save('preservation-before.json',snapshot);save('authentication.json',auth)
        coords=arrays['coordinates'][:,1:];features=arrays['features'];flags=features[:,3:6]>0
        phase('verify-original-quad-connectivity')
        quads,anchors,axes,missing=old.tuple_quads(coords,flags)
        if not np.array_equal(quads,old.recover_quads(f)):raise ValueError('Original quad order differs')
        dual,_,weight=converter.transform_features(features)
        if not np.array_equal(v,converter.vertices(coords,dual)):raise ValueError('Original vertex positions differ')
        w=weight[quads];choose=w[:,0]*w[:,2]>w[:,1]*w[:,3]
        if not np.array_equal(choose,f[1::2,0]==f[::2,0]):raise ValueError('Original triangle splits differ')
        counter=dict(flagged_quads=int(flags.sum()),valid_quads=len(quads),missing_neighbour_quads=len(missing),
            out_of_grid_quads=sum(any(any(x<0 or x>=512 for x in p) for p in r['corners']) for r in missing),
            split_1=int(choose.sum()),split_2=int((~choose).sum()),global_original_indices=True,batch_rows=32768)
        if counter!=reader.read(PREVIOUS/'quad-attribution.json')['conversion'] or counter!=auth['mesh']['conversion']:
            raise ValueError('Original conversion counters differ')
        del dual,weight,w,choose
        edges,counts,signs,order=old.edge_table(quads);del signs,order
        keys=edges[:,0].astype('u8')*len(v)+edges[:,1].astype('u8')
        full_keys=np.load(old.RAW/'topology/edge_keys.npy',mmap_mode='r');full_counts=np.load(old.RAW/'topology/incidence.npy',mmap_mode='r')
        try:
            target=np.flatnonzero((counts==1)|(counts>2));tri_target=(full_counts==1)|(full_counts>2)
            if not np.array_equal(keys[target],full_keys[tri_target]) or not np.array_equal(counts[target],full_counts[tri_target]):
                raise ValueError('Frozen defective edge identities differ')
        finally:full_keys._mmap.close();full_counts._mmap.close()
        if (len(target),int((counts==1).sum()),int((counts>2).sum()),int(counts.max()))!=(1940,1421,519,3):raise ValueError('Exact original defect scope required')
        quad_index=np.full((len(coords),3),-1,np.int32);quad_index[anchors,axes]=np.arange(len(quads),dtype=np.int32)
        lookup={tuple(c):i for i,c in enumerate(coords.tolist())}
        phase('four-candidate-census');records=[]
        for i,eid in enumerate(target):
            edge=edges[eid];rows=candidates(edge,coords,features,lookup,quad_index,quads)
            if sum(r['contributes_to_edge'] for r in rows)!=int(counts[eid]):raise ValueError('Candidate contribution count differs')
            for c in rows:
                c['incidence_effect']=None
                if c['all_corners_present']:
                    before=perimeter_counts(c['corner_vertex_indices'],edges,counts,len(v),keys=keys)
                    c['incidence_effect']=incidence_effect(before,c['contributes_to_edge'])
            inactive=[r for r in rows if r['state']=='off_complete']
            eligible=[r for r in rows if r['all_corners_present'] and r['incidence_effect']['all_sides_become_two']]
            kind='boundary' if counts[eid]==1 else 'triple'
            remaining=[r for r in rows if not r['contributes_to_edge']]
            records.append(dict(source_edge_key=int(keys[eid]),quad_perimeter_edge_row=int(eid),kind=kind,
                original_vertex_indices=edge.tolist(),grid_endpoints=coords[edge].tolist(),positions=v[edge].tolist(),
                original_incidence=int(counts[eid]),candidates=rows,complete_inactive_alternatives=len(inactive),
                incidence_balancing_candidates=len(eligible),incidence_balancing_candidate_ids=[r['candidate_id'] for r in eligible],
                contributing_quad_rows=[r['quad_row'] for r in rows if r['contributes_to_edge']],
                remaining_candidate_state=remaining[0]['state'] if kind=='triple' else None))
            if i%128==0 and time.monotonic()-began>175:raise TimeoutError('Census deadline')
        if sum(len(r['candidates']) for r in records)!=7760:raise ValueError('Exact 7760 candidate observations required')
        summary=summarize(records)
        old_states=reader.read(PREVIOUS/'boundary-candidate-causes.json')['candidate_states'];states=summary['boundary']['candidate_states']
        reconciled=dict(valid=states.get('active',0),flag_off=states.get('off_complete',0)+states.get('off_incomplete',0),
            missing_anchor=states.get('absent_anchor',0),flagged_missing_corner=states.get('flagged_incomplete',0))
        if reconciled!=old_states:raise ValueError('Original boundary classification not reconciled')
        unique=deduplicate([c for r in records for c in r['candidates']]);balanced=[c for c in unique.values() if c['incidence_effect'] and c['incidence_effect']['all_sides_become_two']]
        neighborhoods=[]
        for c in balanced:
            hits=[r for r in records if c['candidate_id'] in r['incidence_balancing_candidate_ids']]
            neighborhoods.append(dict(candidate=c,touched_defect_edges=[r['source_edge_key'] for r in hits],
                unique_in_incidence_screen_at_all_sides=len(hits)==4 and all(r['incidence_balancing_candidates']==1 for r in hits),
                geometry_intersections_assessed=False,vertex_manifold_assessed=False,orientation_assessed=False,
                no_global_geometry_claim=True))
        summary.update(reconciled_original_boundary_states=reconciled,unique_candidate_count=len(unique),
            incidence_screen_candidates=len(balanced),uniquely_constrained_incidence_candidates=sum(r['unique_in_incidence_screen_at_all_sides'] for r in neighborhoods),
            classification='READ_ONLY_CENSUS_COMPLETE_NOT_REPAIR_APPROVAL',geometry_modified=False,logits_or_flags_modified=False,
            incidence_screen='Sufficient single-quad edge-count condition only; not necessary for multi-quad changes or sufficient for geometry preservation')
        phase('save-private-census')
        with (BASE/'edge-census.jsonl').open('x',encoding='utf8') as stream:
            for r in records:stream.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
        save('unique-candidates.json',list(unique.values()));save('candidate-neighborhoods.json',neighborhoods);save('summary.json',summary)
        phase('verify-preservation');reader.verify_snapshot(snapshot)
        if any(Path(name).stat().st_ino!=r['file_id'] for name,r in snapshot.items()):raise ValueError('Source file identity changed')
        save('preservation-after.json',dict(files=len(snapshot),all_hashes_lengths_mtimes_file_IDs_unchanged=True,
            elapsed_seconds=time.monotonic()-began,WSL=False,GPU=False,checkpoint_loaded=False,geometry_modified=False))
    finally:
        for a in [*arrays.values(),v,f]:
            if a is not None:a._mmap.close()


def finalize_saved():
    """Complete only metadata saved by the one census; never rerun analysis."""
    import numpy as np
    import amd_zombie_d_compare as reader
    import amd_zombie_cpu_mesh as owner
    began=time.monotonic()
    for name,sha in SAVED_DATA.items():
        if reader.digest(BASE/name)!=sha:raise ValueError('Saved census identity differs: '+name)
    snapshot=reader.read(BASE/'preservation-before.json');reader.verify_snapshot(snapshot)
    if any(Path(name).stat().st_ino!=row['file_id'] for name,row in snapshot.items()):raise ValueError('Original source identity differs')
    failed=reader.read(BASE/'analyze-run/host-result.json')
    if (failed['classification']!='FAIL' or failed['worker_returncode']!=1 or failed['cleanup_errors']
        or not failed['worker_exit_verified'] or not failed['job_cleanup']['owned_tree_exit_verified'] or failed['job_cleanup']['forced_job_cleanup']):
        raise ValueError('Natural original census settlement required')
    with reader.plain(BASE/'edge-census.jsonl').open(encoding='utf8') as f:
        records=[json.loads(line,object_pairs_hook=reader.unique) for line in f]
    if len(records)!=1940 or len({r['source_edge_key'] for r in records})!=1940:raise ValueError('Complete unique frozen scope required')
    original=reader.read(BASE/'authentication.json');n=original['binding']['rows']
    raw=ROOT.parent/'artifacts/amd-closed-solid-raw-recovery-v1'
    keys=np.load(raw/'topology/edge_keys.npy',mmap_mode='r');counts=np.load(raw/'topology/incidence.npy',mmap_mode='r')
    try:
        expected={int(k):int(v) for k,v in zip(keys[(counts==1)|(counts>2)],counts[(counts==1)|(counts>2)])}
        if {r['source_edge_key']:r['original_incidence'] for r in records}!=expected:raise ValueError('Frozen edge identities differ')
    finally:keys._mmap.close();counts._mmap.close()
    for r in records:
        a,b=r['original_vertex_indices']
        if a*n+b!=r['source_edge_key'] or len(r['candidates'])!=4:raise ValueError('Saved source edge binding differs')
        if sum(c['contributes_to_edge'] for c in r['candidates'])!=r['original_incidence']:raise ValueError('Saved incidence differs')
        for c in r['candidates']:
            if c['anchor_exists']:
                if struct.unpack('<I',struct.pack('<f',c['logit']))[0]!=c['logit_bits'] or c['flag_on']!=(c['logit']>0):raise ValueError('Saved F32 logit differs')
            elif any(c[k] is not None for k in ('logit','logit_bits','flag_on','anchor_vertex_index')):raise ValueError('Absent anchor must have null signal')
            if c['all_corners_present']:
                if c['incidence_effect']!=incidence_effect(c['incidence_effect']['before'],c['contributes_to_edge']):raise ValueError('Saved analytical effect differs')
    summary=summarize(records);unique=deduplicate([c for r in records for c in r['candidates']])
    if list(unique.values())!=reader.read(BASE/'unique-candidates.json'):raise ValueError('Saved deduplication differs')
    balanced=[c for c in unique.values() if c['incidence_effect'] and c['incidence_effect']['all_sides_become_two']]
    # This exact census saved no balanced neighborhoods before the reporting error.
    if balanced or reader.read(BASE/'candidate-neighborhoods.json')!=[]:raise ValueError('Expected empty original incidence screen differs')
    states=summary['boundary']['candidate_states']
    reconciled=dict(valid=states.get('active',0),flag_off=states.get('off_complete',0)+states.get('off_incomplete',0),missing_anchor=states.get('absent_anchor',0),flagged_missing_corner=states.get('flagged_incomplete',0))
    if reconciled!=reader.read(PREVIOUS/'boundary-candidate-causes.json')['candidate_states']:raise ValueError('Boundary reconciliation differs')
    summary.update(reconciled_original_boundary_states=reconciled,unique_candidate_count=len(unique),incidence_screen_candidates=0,
        uniquely_constrained_incidence_candidates=0,classification='READ_ONLY_CENSUS_COMPLETE_NO_GO_SINGLE_FLAG_CORRECTION',
        geometry_modified=False,logits_or_flags_modified=False,census_reexecuted=False,
        incidence_screen='Sufficient single-quad edge-count condition only; not necessary for multi-quad changes or sufficient for geometry preservation')
    owner.save(COMPLETION/'summary.json',summary)
    reader.verify_snapshot(snapshot)
    if any(Path(name).stat().st_ino!=r['file_id'] for name,r in snapshot.items()):raise ValueError('Source file identity changed')
    for name,sha in SAVED_DATA.items():
        if reader.digest(BASE/name)!=sha:raise ValueError('Saved census changed during completion: '+name)
    owner.save(COMPLETION/'preservation-after.json',dict(files=len(snapshot),all_hashes_lengths_mtimes_file_IDs_unchanged=True,
        elapsed_seconds=time.monotonic()-began,WSL=False,GPU=False,checkpoint_loaded=False,geometry_modified=False,census_reexecuted=False,
        reporting_error='Original summary serialization failed on NumPy int64; preserved partial file and failure receipt unchanged',saved_data_sha256=SAVED_DATA))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument('--run',action='store_true');g.add_argument('--finalize-saved',action='store_true');p.add_argument('--worker',choices=('analyze','finalize'));a=p.parse_args(argv)
    if a.worker:
        if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Exact owned GO plus EOF required')
        import amd_closed_solid_benchmark as b
        sys.addaudithook(b.cpu_guard(COMPLETION if a.worker=='finalize' else BASE));sys.path.insert(0,'D:/AI/Trellis2AMD/venv/Lib/site-packages')
        (finalize_saved if a.worker=='finalize' else analyze)();return 0
    if a.run or a.finalize_saved:
        import amd_zombie_cpu_mesh as owner
        if a.finalize_saved:owner.plain(COMPLETION).mkdir(exist_ok=False)
        owner.BASE=COMPLETION if a.finalize_saved else BASE;owner.__file__=str(Path(__file__).resolve());return owner.supervise('finalize' if a.finalize_saved else 'analyze')
    print('DRY_RUN: read-only saved-output census; no WSL/GPU/repair/neural route');return 0

if __name__=='__main__':raise SystemExit(main())
