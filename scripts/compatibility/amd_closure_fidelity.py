"""Conservative impossibility certificate, not a substitute surface-distance test."""
import json
import math
from decimal import Decimal,localcontext
from fractions import Fraction
from pathlib import Path

def decimal_box_witness(point,polygons):
    """Exact rational decision; Decimal is used only for readable distance display."""
    if not polygons or any(not math.isfinite(x) for x in point):raise ValueError('Invalid witness')
    with localcontext() as ctx:
        ctx.prec=200
        guard=Fraction(1,100000000);p=[Fraction.from_float(float(x)) for x in point];bounds=[]
        for row in polygons:
            v=row['vertices']
            if len(v)<3 or any(not math.isfinite(x) for q in v for x in q):raise ValueError('Invalid polygon')
            lo=[min(Fraction.from_float(float(q[k])) for q in v)-guard for k in range(3)]
            hi=[max(Fraction.from_float(float(q[k])) for q in v)+guard for k in range(3)]
            d=[max(lo[k]-p[k],p[k]-hi[k],Fraction(0)) for k in range(3)]
            bounds.append((sum(x*x for x in d),row['variable']))
        squared,face=min(bounds)
        limit=Fraction(100000001,100000000) # conservative allowance above frozen1.0
        display=Decimal(squared.numerator)/Decimal(squared.denominator)
        return dict(minimum_squared_distance_native_exact_fraction=str(squared),distance_lower_bound_native=str(display.sqrt()),
                    nearest_enclosing_box_face=face,outward_expansion_native=str(guard),
                    comparison_limit_native=str(limit),strict_frozen_max_violation=squared>limit*limit,
                    decision_arithmetic='exact rational binary64 coordinates',display_precision_decimal_digits=200,all_boxes_checked=len(polygons))

def run():
    import amd_closure_proof as r
    import sys
    sys.path.insert(0,str(r.old.SITE))
    import numpy as np
    from amd_closure_graph import prune
    b=r.BASE
    cert=[json.loads(x) for x in (b/'complete-certificates.jsonl').read_text().splitlines()]
    faces=[json.loads(x) for x in (b/'diagnostic/actual-polygons.jsonl').read_text().splitlines()]
    fans=[json.loads(x)['faces'] for x in (b/'diagnostic/actual-fans.jsonl').read_text().splitlines()]
    if len(cert)!=1105 or [x['variable'] for x in cert]!=list(range(1105)) or any(x['key']!=faces[i]['key'] or x['coverage']!='CONCLUSIVE' for i,x in enumerate(cert)):raise ValueError('Incomplete exact certification')
    proof=prune(1105,fans,[x['variable'] for x in cert if x['state']!='CERTIFIED_ELIGIBLE'])
    stored=json.loads((b/'certified-closure-screen.json').read_text())
    if proof['remaining']!=stored['remaining'] or not proof['remaining']:raise ValueError('Closure screen differs')
    polygons=[faces[i] for i in proof['remaining']]
    auth=json.loads((b/'authentication.json').read_text())['Phase14_files']
    for name in ('validation-points.bin','input/mesh.bin','candidate-input-pin.json'):
        r.check(r.PRIOR/name,r.old.manifest_identity(auth,name))
    pin=json.loads((r.PRIOR/'candidate-input-pin.json').read_text())
    if r.owner.sha(r.PRIOR/'validation-points.bin')!=pin['validation_points_sha256']:raise ValueError('Frozen source samples differ')
    points=np.fromfile(r.PRIOR/'validation-points.bin',dtype='<f8').reshape(-1,3)
    if points.shape!=(65536,3) or not np.isfinite(points).all():raise ValueError('Frozen sample dimensions')
    with (r.PRIOR/'input/mesh.bin').open('rb') as f:
        dims=np.fromfile(f,dtype='<u8',count=2);vertices=np.fromfile(f,dtype='<f8',count=int(dims[0])*3).reshape(-1,3)
    D=float(np.linalg.norm(np.ptp(vertices,axis=0)))
    if abs(D-100)>1e-9:raise ValueError('Source normalization differs')
    best=np.full(len(points),np.inf)
    for face in polygons:
        v=np.array(face['vertices'],dtype='f8');lo=v.min(axis=0)-1e-8;hi=v.max(axis=0)+1e-8
        gap=np.maximum(np.maximum(lo-points,points-hi),0);best=np.minimum(best,np.sum(gap*gap,axis=1))
    best=np.sqrt(best);w=int(np.argmax(best));independent=decimal_box_witness(points[w],polygons)
    if not independent['strict_frozen_max_violation']:raise ValueError('No conclusive independently verified obstruction')
    criteria=json.loads((r.owner.ROOT/'docs/compatibility/amd-ai-hard-surface-criteria.json').read_text())
    if criteria['candidate_shape_gates']['sampled_max_distance_over_AI_diagonal_max']!=.01:raise ValueError('Frozen limit differs')
    with (b/'source-distance-lower-bounds.bin').open('xb') as f:best.astype('<f8').tofile(f)
    r.owner.save(b/'fidelity-impossibility.json',dict(classification='NO_GO_CANDIDATE_GEOMETRY_OR_FIDELITY',
        all_certification_complete=True,eligible=137,unsupported=968,unresolved=0,
        all_possible_closed_selection_faces=len(polygons),frozen_source_samples=65536,native_AI_diagonal=D,
        frozen_sampled_max_over_D=.01,frozen_sampled_max_native=1.,
        conservative_lower_bound_statistics_over_D=dict(RMS=float(np.sqrt(np.mean(best**2))/100),p95=float(np.quantile(best,.95)/100),p99=float(np.quantile(best,.99)/100),maximum=float(best.max()/100)),
        floating_summary_samples_beyond_max=int((best>1.00000001).sum()),exact_rational_witnesses_independently_checked=1,
        witness=dict(source_sample_row=w,position_native=points[w].tolist(),**independent),
        independent_algorithm='Exact Fraction arithmetic from binary64 coordinates; minimum outward-expanded-AABB squared distance over every permittedface; Decimal only displays distance',
        implication='Every allowed selected polygon and any triangulation is contained in its expanded AABB; even unionofALLpossiblefaces cannot satisfy frozen source-to-candidate sampledmaximum; subsets cannot improve this lowerbound',
        not_a_nonempty_topological_infeasibility_claim=True,production_solves=0,no_mesh_exported=True,
        source_samples_sha256=r.owner.sha(r.PRIOR/'validation-points.bin'),
        certificates_sha256=r.owner.sha(b/'complete-certificates.jsonl'),actual_polygons_sha256=r.owner.sha(b/'diagnostic/actual-polygons.jsonl'),actual_fans_sha256=r.owner.sha(b/'diagnostic/actual-fans.jsonl'),
        neither_historical_nor_locked_source_geometry_blind_to_constraints=True))
