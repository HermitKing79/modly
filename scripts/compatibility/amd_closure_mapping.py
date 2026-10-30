"""Unique polygon correspondence for witness candidates, never eligibility transfer."""
import math

def cyclic_residual(a,b):
    if len(a)!=len(b) or len(a)<3:return math.inf
    n=len(a)
    return min(max(math.dist(a[i],b[(start+sign*i)%n]) for i in range(n))
               for start in range(n) for sign in (-1,1))

def map_faces(old,live,limit=1e-10):
    if len(old)!=len(live) or not old:raise ValueError('Face count')
    for rows in (old,live):
        if [r['variable'] for r in rows]!=list(range(len(rows))) or len({r['key'] for r in rows})!=len(rows):raise ValueError('Ambiguous record identity')
        if any(not math.isfinite(x) for r in rows for p in r['vertices'] for x in p):raise ValueError('Nonfinite polygon')
    mapping=[];used=set()
    for row in live:
        matches=[]
        for prev in old:
            if prev['group']!=row['group'] or len(prev['vertices'])!=len(row['vertices']):continue
            d=cyclic_residual(row['vertices'],prev['vertices'])
            if d<=limit:matches.append((prev,d))
        if len(matches)!=1:raise ValueError('Missing/ambiguous cyclic polygon correspondence')
        prev,d=matches[0]
        if prev['variable'] in used:raise ValueError('Nonbijective polygon correspondence')
        used.add(prev['variable'])
        mapping.append(dict(variable=row['variable'],prior_variable=prev['variable'],group=row['group'],
                            maximum_cyclic_vertex_displacement_native=d,exact_key_match=row['key']==prev['key'],
                            prior_state=prev['state']))
    return dict(classification='UNIQUE_WITNESS_CANDIDATE_MAPPING_ONLY',mapping=mapping,
                maximum_displacement_native=max(r['maximum_cyclic_vertex_displacement_native'] for r in mapping),
                exact_keys=sum(r['exact_key_match'] for r in mapping),eligible_certificates_transferred=False,
                historical_fan_identity_not_recorded=True)
