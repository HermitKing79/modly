"""Frozen complete-ring checkpoints; no geometry change or collapse."""
import sys
sys.dont_write_bytecode=True
import numpy as np
CAP=6000

def need(ok,message):
    if not ok:raise ValueError(message)

def one_ring(f,keys,order,offsets,current,n,check):
    check();need(current.dtype==np.int32 and 0<len(current)<=CAP and (current[1:]>current[:-1]).all(),'Sorted unique admitted original context')
    tri=f[current];a=tri[:,[0,1,2]];b=tri[:,[1,2,0]]
    wanted=np.unique(np.minimum(a,b).astype(np.uint64)*np.uint64(n)+np.maximum(a,b).astype(np.uint64))
    at=np.searchsorted(keys,wanted);need((at<len(keys)).all() and np.array_equal(keys[at],wanted),'Exact authenticated edge keys')
    count=offsets[at+1]-offsets[at];need((count>0).all() and int(count.max())<=4,'Original maximum incidence4; reject unsupported map before allocation')
    col=np.arange(int(count.max()),dtype=np.int32);where=offsets[at,None]+col
    rows=order[where[col<count[:,None]]]//3
    # At most3*6000*4 source occurrences. A prospective oversized ring has
    # only its IDs/count inspected; no oversized geometry is processed/saved.
    expanded=np.union1d(current,rows).astype(np.int32);check();return expanded

def build_contexts(f,keys,order,offsets,n,baseline_face_ids,baseline_outer_face_ids,check,targets=(128,512,2048),max_faces=CAP):
    need(type(max_faces)is int and 0<max_faces<=CAP,'Unchanged hard6000-face cap')
    need(type(targets)is tuple and len(targets)==3 and all(type(x)is int and x>0 for x in targets) and list(targets)==sorted(set(targets)),'Three frozen increasing checkpoints')
    current=baseline_face_ids.copy();outer=baseline_outer_face_ids.copy();depth=3
    need(0<len(current)<=max_faces and len(current)<targets[0] and np.isin(outer,current).all(),'Admitted authenticated baseline below first checkpoint')
    snapshots=[];records=[dict(depth=depth,face_count=len(current),added_count=len(outer))];baseline_labels=['baseline'];pending=list(targets)
    def retain(labels):
        if depth==3:baseline_labels.extend(x for x in labels if x not in baseline_labels);return
        if snapshots and snapshots[-1]['depth']==depth:snapshots[-1]['labels'].extend(x for x in labels if x not in snapshots[-1]['labels'])
        else:snapshots.append(dict(depth=depth,source_face_ids=current.copy(),outer_face_ids=outer.copy(),labels=list(labels)))
    for _ in range(max_faces-len(current)+1):
        check()
        if len(current)==max_faces:retain(['max']);terminal=dict(reason='exact_cap',largest_depth=depth,largest_face_count=len(current));break
        next_ids=one_ring(f,keys,order,offsets,current,n,check)
        if len(next_ids)>max_faces:
            retain(['max']);terminal=dict(reason='next_complete_ring_over_cap',largest_depth=depth,largest_face_count=len(current),prospective_depth=depth+1,prospective_face_count=len(next_ids));break
        if np.array_equal(next_ids,current):
            retain(['max']);terminal=dict(reason='fixed_point',largest_depth=depth,largest_face_count=len(current));break
        outer=np.setdiff1d(next_ids,current,assume_unique=True);current=next_ids;depth+=1
        records.append(dict(depth=depth,face_count=len(current),added_count=len(outer)))
        labels=[]
        while pending and len(current)>=pending[0]:labels.append(str(pending.pop(0)))
        if labels:retain(labels)
    else:raise ValueError('Monotone full-ring progression incomplete')
    return dict(baseline_depth=3,baseline_labels=baseline_labels,ring_records=records,snapshots=snapshots,terminal=terminal)

def screen(summary):
    k=summary['initial_non_touching_opportunities'];e=summary['eligible_face_population'];f=summary['original_patch_faces']
    need(all(type(x)is int and x>=0 for x in (k,e,f)) and f>0 and type(summary['qualified'])is bool,'Exact integer screening populations')
    removal=2*k;passes_count=k>=10;passes_interior=e>0 and 5*removal>=e;passes_patch=20*removal>=f
    return dict(meaningful_pass=bool(summary['qualified'] and passes_count and passes_interior and passes_patch),passes_10_disjoint=passes_count,passes_20pct_eligible=passes_interior,passes_5pct_patch=passes_patch,estimated_removed_faces=removal,eligible_face_fraction=removal/e if e else None,whole_patch_fraction=removal/f)

def select_first(rows):
    need(len({x['depth'] for x in rows})==len(rows),'No duplicate context qualification')
    return next((x['depth'] for x in sorted(rows,key=lambda x:x['depth']) if x['screening']['meaningful_pass']),None)
