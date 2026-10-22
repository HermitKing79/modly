"""Pinned qualification math; only frozen-choice scalar metadata is adapted."""
import sys
sys.dont_write_bytecode=True
import numpy as np
import common as c
pc=c.pinnedpc()
need=pc.need
complete_stars=pc.complete_stars
full_link=pc.full_link
greedy_capacity=pc.greedy_capacity
raw_edge_components=pc.raw_edge_components

def qualify(v,f,rgba,keys,inc,source_components,context,choice,selection):
    """Merge actual crop facts with global immutable/guard facts, then recheck."""
    check=context['check'];check();face_ids=selection['source_face_ids'];source_faces=np.asarray(f[face_ids],np.int32)
    ids=np.unique(source_faces).astype(np.int32);cv=np.asarray(v[ids],np.float32);cf=np.searchsorted(ids,source_faces).astype(np.int32)
    colours=np.asarray(rgba[ids],np.uint8);outer=np.searchsorted(face_ids,selection['ring_added_face_ids'][-1]).astype(np.int32)
    need(np.array_equal(ids[cf],source_faces) and np.array_equal(cv.view(np.uint32),v[ids].view(np.uint32))
         and np.array_equal(colours,rgba[ids]),'Exact original oriented faces, F32 positions and accepted native colour IDs')
    need((source_components[source_faces]==source_components[source_faces[:,0],None]).all(),'Original source-component face identity')
    need(len(np.unique(source_components[ids]))==1,'Source-connected chosen original patch')
    raw_components=raw_edge_components(source_faces)
    need(raw_components==1,'Selected whole patch must be connected through original edges, not merely shared vertices')
    local_summary,local=context['classify'](cv,cf,ids,source_faces,outer,keys,inc,len(v));check()
    inherited_f=np.asarray(context['protected_faces'][face_ids],np.bool_);inherited_v=np.asarray(context['protected_vertices'][ids],np.bool_)
    inherited_l=np.asarray(context['locked_vertices'][ids],np.bool_)
    protected=local['protected_faces']|inherited_f;pv=local['protected_vertices']|inherited_v
    pv[cf[protected].ravel()]=True;locks=local['vertex_locks']|inherited_l
    locks[cf[pv[cf].any(1)].ravel()]=True
    positions=cv.copy();positions[positions==0]=0
    _,aliases=np.unique(positions,axis=0,return_inverse=True)
    locks=np.isin(aliases,np.unique(aliases[locks]))
    # These are complete GLOBAL stars: selected rows are unique original faces,
    # so equality to verified source occurrence degree excludes omitted faces.
    stars=complete_stars(source_faces,face_ids,context['face_degree'])
    patch_keys=local['edge_keys'];at=np.searchsorted(keys,patch_keys)
    need((at<len(keys)).all() and np.array_equal(keys[at],patch_keys),'Patch edges retain global source keys')
    global_eligible=context['eligible_edge_ids'];where=np.searchsorted(global_eligible,at)
    present=(where<len(global_eligible))&(global_eligible[np.minimum(where,len(global_eligible)-1)]==at)
    touch=at[present];both=[];complete=[];guarded=[];eligible=[];link_rejected=0
    for edge in touch:
        check();edge=int(edge);rows=context['order'][context['offsets'][edge]:context['offsets'][edge+1]]//3
        if not np.isin(rows,face_ids).all():continue
        both.append(edge);a,b=divmod(int(keys[edge]),len(v))
        if a not in stars or b not in stars:continue
        complete.append(edge);la,lb=map(int,np.searchsorted(ids,[a,b]));local_rows=np.searchsorted(face_ids,rows)
        if locks[la] or locks[lb] or protected[local_rows].any():continue
        guarded.append(edge)
        need(inc[edge]==2 and context['consistent'][edge]
             and len(np.unique(context['global_sheet_labels'][rows]))==1,'Global winding/incidence/sheet identity')
        # Reassert the actual opposite directions, independently of the label.
        directions=[]
        for occurrence in context['order'][context['offsets'][edge]:context['offsets'][edge+1]]:
            tri=f[int(occurrence)//3];corner=int(occurrence)%3;directions.append(bool(tri[corner]<tri[(corner+1)%3]))
        need(directions[0]!=directions[1],'Original opposite winding')
        if full_link(f,a,b,stars[a],stars[b]):eligible.append(edge)
        else:link_rejected+=1
    eligible=np.array(eligible,np.int32);capacity=greedy_capacity(f,keys,eligible,stars,len(v),check)
    need(np.isin(keys[eligible],local['edge_keys'][local['eligible_edge_rows']]).all(),
         'Inherited-mask/full-global qualification can only narrow independent local-classifier eligible edges')
    eligible_faces=np.zeros(len(cf),np.bool_)
    if len(eligible):
        rows=np.concatenate((context['order'][context['offsets'][eligible]]//3,context['order'][context['offsets'][eligible]+1]//3))
        eligible_faces[np.searchsorted(face_ids,rows)]=True
    need(2*len(capacity)<=int(eligible_faces.sum())<=int((~protected).sum()),'Initial opportunity is bounded by mutable eligible and unprotected face populations')
    ready=bool(eligible_faces.any() and len(capacity)>=1)
    summary=dict(original_patch_faces=len(cf),original_patch_vertices=len(cv),source_component=int(source_components[ids[0]]),
                 raw_original_edge_connected_components=raw_components,eligible_subset_of_independent_local_classifier=True,
                 selected_component_faces=choice['selected_component_faces'],selected_component_edges=choice['selected_component_edges'],
                 eligible_face_components=choice['eligible_face_components'],seed_edge_id=choice['seed_edge_id'],
                 globally_certified_edges_touching_patch=len(touch),both_edge_faces_represented=len(both),
                 complete_global_endpoint_star_edges=len(complete),post_crop_inherited_guard_edges=len(guarded),
                 lost_incomplete_edge_face_context=len(touch)-len(both),lost_incomplete_endpoint_star_context=len(both)-len(complete),
                 lost_to_actual_crop_or_inherited_guards=len(complete)-len(guarded),
                 full_link_rejected=link_rejected,strict_eligible_edges=len(eligible),eligible_face_population=int(eligible_faces.sum()),
                 immutable_face_floor=int(protected.sum()),optimistic_removable_face_ceiling=int((~protected).sum()),
                 guard_locked_vertices=int(locks.sum()),initial_non_touching_opportunities=len(capacity),
                 initial_topological_removal_estimate_faces=2*len(capacity),
                 estimate_fraction_of_eligible_faces=2*len(capacity)/int(eligible_faces.sum()) if eligible_faces.any() else None,
                 initial_estimate_fraction_of_entire_patch=2*len(capacity)/len(cf),
                 optimistic_removable_ceiling_fraction_of_entire_patch=int((~protected).sum())/len(cf),
                 twenty_percent_is_informational_not_admission=True,qualified=ready)
    check();arrays=dict(vertices=cv,faces=cf,source_faces=source_faces,source_face_ids=face_ids,source_vertex_ids=ids,
                       original_native_RGBA=colours,source_component_labels=np.asarray(source_components[ids],np.int32),
                       inherited_protected_faces=inherited_f,inherited_protected_vertices=inherited_v,inherited_vertex_locks=inherited_l,
                       local_protected_faces=local['protected_faces'],local_protected_vertices=local['protected_vertices'],local_vertex_locks=local['vertex_locks'],
                       protected_faces=protected,protected_vertices=pv,vertex_locks=locks,eligible_faces=eligible_faces,
                       eligible_edge_ids=eligible,non_touching_edge_ids=capacity,global_sheet_labels=np.asarray(context['global_sheet_labels'][face_ids],np.int32))
    star_vertices=np.array(sorted(stars),np.int32);star_offsets=np.zeros(len(star_vertices)+1,np.int32)
    np.cumsum([len(stars[int(g)]) for g in star_vertices],dtype=np.int32,out=star_offsets[1:])
    arrays.update(complete_star_vertex_ids=star_vertices,complete_star_offsets=star_offsets,
                  complete_star_source_face_ids=np.concatenate([stars[int(g)] for g in star_vertices]) if len(star_vertices) else np.empty(0,np.int32))
    return summary,arrays,local_summary
