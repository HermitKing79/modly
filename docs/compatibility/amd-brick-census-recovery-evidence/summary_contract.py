"""Exact schema of the authenticated positive census; no invented defaults."""
import sys
sys.dont_write_bytecode=True
import hashlib,json

MASK_SUMMARY_PAIRS=tuple((n,n) for n in (
    'nonmanifold_vertices','boundary_vertices','alias_vertices','protected_vertices',
    'locked_vertices','multiple_fan_vertices','multiple_sheet_vertices',
    'nonmanifold_touch_faces','boundary_touch_faces','boundary_edge_touch_faces',
    'protected_faces','pre_guard_interior_faces','fully_unlocked_interior_faces',
    'guard_affected_interior_faces'))+(('certified_edge_touch_faces','certified_edge_face_population'),)
OTHER_FIELDS=set(('original_vertices','referenced_vertices','unused_vertices',
    'original_triangles','original_edges','original_nonmanifold_edges',
    'original_boundary_edges','consistent_two_face_edges','orientation_conflict_edges',
    'oriented_sheet_components','corner_fan_components','pre_link_candidate_edges',
    'link_tested_edges','certified_edges','rejected_common_vertex_link',
    'rejected_common_link_edge','link_comparisons','maximum_free_degree','batch_edge_limit'))

def need(ok,message):
    if not ok:raise ValueError(message)

def validated_mapping(pairs):
    need(len(pairs)==15 and all(type(p) is tuple and len(p)==2 for p in pairs),'Exact fifteen mask mappings')
    mapping=dict(pairs)
    need(len(mapping)==len(pairs) and len(set(mapping.values()))==len(pairs),'Duplicate mask/summary mapping')
    need(mapping==dict(MASK_SUMMARY_PAIRS),'Unknown or incorrect mask/summary mapping')
    return mapping

def validate_summary(summary):
    mapping=validated_mapping(MASK_SUMMARY_PAIRS)
    need(type(summary) is dict and set(summary)==set(mapping.values())|OTHER_FIELDS,'Exact completed-positive 34-field summary; no missing/unknown fields or zero-branch defaults')
    need(all(type(x) is int and x>=0 for x in summary.values()),'All summary values are nonnegative genuine integers')
    s=summary;n=s['original_vertices'];f=s['original_triangles'];e=s['original_edges']
    need(s['referenced_vertices']+s['unused_vertices']==n,'Source vertex partition')
    need(sum(s[k] for k in ('original_nonmanifold_edges','original_boundary_edges','consistent_two_face_edges','orientation_conflict_edges'))==e,'Source edge partition')
    need(s['protected_faces']+s['pre_guard_interior_faces']==f and s['fully_unlocked_interior_faces']+s['guard_affected_interior_faces']==s['pre_guard_interior_faces'],'Face stage partitions')
    need(s['pre_link_candidate_edges']==s['link_tested_edges']>0 and s['certified_edges']+s['rejected_common_vertex_link']+s['rejected_common_link_edge']==s['link_tested_edges'],'Exact positive link population partition')
    d=s['maximum_free_degree'];need(0<d<=128 and s['link_comparisons']==s['link_tested_edges']*d*d and s['link_comparisons']<=4_000_000_000,'Recorded padded link-work arithmetic')
    need(s['batch_edge_limit']==min(8192,max(1,8*1024**2//(d*d))),'Recorded link batch bound')
    need(s['protected_vertices']<=s['locked_vertices']<=s['referenced_vertices'] and s['boundary_edge_touch_faces']<=s['boundary_touch_faces']<=f and s['certified_edge_face_population']<=min(f,2*s['certified_edges']),'Population bounds')
    return mapping

def verify_mask_counts(summary,masks):
    mapping=validate_summary(summary)
    need(set(masks)==set(mapping),'Complete exact mask handoff')
    for name,target in mapping.items():need(int(masks[name].sum())==summary[target],'Mapped mask count: '+name+' -> '+target)

def sealed_json(raw,sha256):
    need(type(raw) is bytes and len(raw)<=1024**2,'Bounded sealed JSON bytes')
    def unique(pairs):
        d={}
        for k,v in pairs:
            need(k not in d,'Duplicate JSON key: '+k);d[k]=v
        return d
    result=json.loads(raw,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))
    need(hashlib.sha256(raw).hexdigest()==sha256,'Unchanged authenticated producer bytes')
    return result
