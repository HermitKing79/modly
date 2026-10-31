"""Exact analytical cell audit; never exports or substitutes reconstructed geometry."""
import collections
import csv
import json
import math
import struct
from fractions import Fraction as Q

def rational(x):return Q.from_float(float(x))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def residual(plane,p):return dot(plane[:3],p)-plane[3]

def intersection(a,b,c):
    bc=cross(b[:3],c[:3]);ca=cross(c[:3],a[:3]);ab=cross(a[:3],b[:3]);det=dot(a[:3],bc)
    if not det:raise ValueError('Degenerate triple intersection')
    return tuple((a[3]*bc[k]+b[3]*ca[k]+c[3]*ab[k])/det for k in range(3))

def full_cell(vertices,own,planes):
    if len(set(vertices))!=len(vertices) or len(vertices)<3:raise ValueError('Duplicate/degenerate polygon')
    axis=max(range(3),key=lambda k:abs(planes[own][k]));axes=[k for k in range(3) if k!=axis]
    v=[(p[axes[0]],p[axes[1]]) for p in vertices]
    area=sum(v[i][0]*v[(i+1)%len(v)][1]-v[i][1]*v[(i+1)%len(v)][0] for i in range(len(v)))
    if not area:raise ValueError('Zero area polygon')
    orientation=1 if area>0 else -1;edge_planes=[]
    for i,a in enumerate(v):
        b=v[(i+1)%len(v)]
        sides=[orientation*((b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])) for p in v]
        if min(sides)<0 or max(sides)==0:raise ValueError('Nonconvex or wrongly ordered polygon')
        matches=[j for j,p in enumerate(planes) if j!=own and residual(p,vertices[i])==0 and residual(p,vertices[(i+1)%len(v)])==0]
        if len(matches)!=1:raise ValueError('Edge not a unique plane intersection')
        edge_planes.append(matches[0])
    signs=[]
    for j,p in enumerate(planes):
        values=[residual(p,v) for v in vertices]
        if j==own:
            if any(values):raise ValueError('Own plane mismatch')
            signs.append(0);continue
        positive=any(x>0 for x in values);negative=any(x<0 for x in values)
        if positive and negative:raise ValueError('Missing interior plane cut')
        if not positive and not negative:raise ValueError('Coincident supporting planes')
        signs.append(1 if positive else -1)
    # Convex polygon edges are all active halfspace constraints; all remaining
    # planes contain the whole polygon on one side. Thus it is the FULL cell.
    return dict(signs=signs,edge_planes=edge_planes,projected_area_twice=abs(area))

def point_triangle_squared(p,a,b,c):
    ab=sub(b,a);ac=sub(c,a);ap=sub(p,a);n=cross(ab,ac);n2=dot(n,n)
    choices=[]
    if n2:
        aa=dot(ab,ab);bb=dot(ab,ac);cc=dot(ac,ac);d=aa*cc-bb*bb
        u=(dot(ap,ab)*cc-dot(ap,ac)*bb)/d;v=(dot(ap,ac)*aa-dot(ap,ab)*bb)/d
        if u>=0 and v>=0 and u+v<=1:choices.append(dot(ap,n)**2/n2)
    for s,t in ((a,b),(b,c),(c,a)):
        st=sub(t,s);length=dot(st,st)
        factor=min(Q(1),max(Q(0),dot(sub(p,s),st)/length)) if length else Q(0)
        q=tuple(s[k]+factor*st[k] for k in range(3));choices.append(dot(sub(p,q),sub(p,q)))
    return min(choices)

def zero_by_internal_edges(count,edges,forbidden):
    """A conforming two-sided coplanar edge imposes x_left == x_right."""
    if type(count)!=int or count<1 or count>200000:raise ValueError('Invalid subdivision count')
    if any(type(x)!=int or not 0<=x<count for x in forbidden):raise ValueError('Invalid forbidden cell')
    parent=list(range(count))
    def root(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for a,b in edges:
        if not 0<=a<count or not 0<=b<count or a==b:raise ValueError('Invalid subdivision incidence')
        parent[root(a)]=root(b)
    zero={root(x) for x in forbidden}
    return [i for i in range(count) if root(i) in zero]

def broad_coverage(path,boxes):
    """Necessary-condition screen; enclosing boxes never construct geometry.

    All exact rational bounds are rounded outwards; the decisive witness is
    additionally checked with exact arithmetic. Population statistics are F64.
    """
    if path.stat().st_size!=65536*24 or not boxes:raise ValueError('Frozen sample count / empty boxes')
    bounds=[([math.nextafter(float(x),-math.inf) for x in lo],
             [math.nextafter(float(x),math.inf) for x in hi]) for lo,hi in boxes]
    distances=[]
    for p in struct.iter_unpack('<ddd',path.read_bytes()):
        if not all(map(math.isfinite,p)):raise ValueError('Nonfinite frozen sample')
        best=math.inf
        for lo,hi in bounds:
            d0=max(lo[0]-p[0],p[0]-hi[0],0.)
            d1=max(lo[1]-p[1],p[1]-hi[1],0.)
            d2=max(lo[2]-p[2],p[2]-hi[2],0.)
            best=min(best,d0*d0+d1*d1+d2*d2)
        distances.append(math.sqrt(best))
    ordered=sorted(distances)
    return dict(samples=len(distances),boxes=len(boxes),outward_bounds=True,
                statistics='F64 necessary lower bounds; exact decisive witness recorded separately',
                violations_above_1_native_plus_1e_8=sum(x>1.00000001 for x in distances),
                maximum_native=max(distances),rms_native=math.sqrt(sum(x*x for x in distances)/len(distances)),
                p95_native=ordered[math.ceil(.95*len(ordered))-1],p99_native=ordered[math.ceil(.99*len(ordered))-1])

def read_source_bounds(path):
    lo=[math.inf]*3;hi=[-math.inf]*3;count=None;seen=0
    with path.open() as f:
        if f.readline().strip()!='ply' or f.readline().strip()!='format ascii 1.0':raise ValueError('Expected qualified ASCII PLY')
        for line in f:
            if line.startswith('element vertex '):count=int(line.split()[2])
            if line.strip()=='end_header':break
        if count is None or not 1<=count<=120000:raise ValueError('Point count')
        for line in f:
            if not line.strip():continue
            row=line.split()
            if len(row)!=7:raise ValueError('Training point schema')
            p=list(map(float,row[:3]))
            if not all(map(math.isfinite,p)):raise ValueError('Nonfinite source point')
            for k in range(3):lo[k]=min(lo[k],p[k]);hi[k]=max(hi[k],p[k])
            seen+=1
    if seen!=count:raise ValueError('Training count differs')
    dx,dy,dz=[hi[k]-lo[k] for k in range(3)]
    offset=(.5*math.sqrt(dx*dx+dy*dy+dz*dz))*.05
    return [lo[k]-offset for k in range(3)],[hi[k]+offset for k in range(3)],count,offset

def run():
    import amd_patch_subdivision_proof as r
    from amd_closure_graph import prune
    b=r.BASE;old=r.PRIOR
    faces=[json.loads(x) for x in (old/'diagnostic/actual-polygons.jsonl').read_text().splitlines()]
    cert=[json.loads(x) for x in (old/'complete-certificates.jsonl').read_text().splitlines()]
    fan_rows=[json.loads(x) for x in (old/'diagnostic/actual-fans.jsonl').read_text().splitlines()]
    rows=list(csv.DictReader((old/'diagnostic/owned-session/arrangement/prepared-planes.csv').open()))
    if len(faces)>20000 or len(rows)>48 or sum(len(f['vertices']) for f in faces)>200000:raise ValueError('Complexity cap')
    if [f['variable'] for f in faces]!=list(range(len(faces))) or len(cert)!=len(faces):raise ValueError('Face identities')
    plane_float=[tuple(float(row[k]) for k in ('nx','ny','nz','offset')) for row in rows];source_planes=len(plane_float)
    lo,hi,points,offset=read_source_bounds(r.previous.PRIOR/'augmented/train.ply')
    observed_lo=[min(p[k] for f in faces for p in f['vertices']) for k in range(3)];observed_hi=[max(p[k] for f in faces for p in f['vertices']) for k in range(3)]
    bound_error=max(abs(a-b) for a,b in zip(lo+hi,observed_lo+observed_hi))
    if bound_error>1e-10:raise ValueError('Authenticated source proxy bounds differ')
    for k in range(3):
        for value in (lo[k],hi[k]):plane_float.append(tuple([float(j==k) for j in range(3)]+[value]))
    planes=[tuple(map(rational,p)) for p in plane_float]
    vertices={};fans=collections.defaultdict(list);cells=[];max_delta2=Q(0);occurrences=0;signatures=set();plane_edges=[collections.defaultdict(list) for _ in rows]
    for face in faces:
        ids=[];exact=[];source=[]
        for v in face['vertices']:
            triple=tuple(j for j,p in enumerate(plane_float) if abs(sum(p[k]*v[k] for k in range(3))-p[3])<=1e-8)
            if len(triple)!=3 or face['group'] not in triple:raise ValueError('Ambiguous vertex-plane identity')
            if triple not in vertices:vertices[triple]=intersection(*(planes[j] for j in triple))
            p=vertices[triple];original=tuple(map(rational,v));delta2=dot(sub(p,original),sub(p,original))
            if delta2>Q(1,10**10)**2:raise ValueError('Ideal-cell displacement exceeds identity guard')
            max_delta2=max(max_delta2,delta2);ids.append(triple);exact.append(p);source.append(original);occurrences+=1
        info=full_cell(exact,face['group'],planes);signature=(face['group'],tuple(info['signs']))
        if signature in signatures:raise ValueError('Duplicate plane cell')
        signatures.add(signature)
        for i in range(len(ids)):
            key=tuple(sorted((ids[i],ids[(i+1)%len(ids)])));fans[key].append(face['variable'])
            plane_edges[face['group']][key].append((face['variable'],info['edge_planes'][i],ids[i],ids[(i+1)%len(ids)]))
        cells.append(dict(variable=face['variable'],group=face['group'],key=face['key'],triples=ids,signs=info['signs']))
    if len(set(vertices.values()))!=len(vertices):raise ValueError('Distinct triples share a geometric vertex')
    # Every inner geometric edge must have two cells per intersecting plane;
    # single-face edges must be genuine clipping-box boundaries, never T-junctions.
    boundary_cycles=[]
    for group,edges in enumerate(plane_edges):
        boundary=collections.defaultdict(set)
        for key,incidence in edges.items():
            if len(incidence)==1:
                _,other,a,z=incidence[0]
                if other<source_planes:raise ValueError('Unpaired internal edge / missing cell')
                boundary[a].add(z);boundary[z].add(a)
            elif len(incidence)==2:
                if incidence[0][1]!=incidence[1][1] or incidence[0][1]>=source_planes:raise ValueError('Invalid interior plane incidence')
                other=incidence[0][1]
                if cells[incidence[0][0]]['signs'][other]==cells[incidence[1][0]]['signs'][other]:raise ValueError('Overlapping cells across interior edge')
            else:raise ValueError('Overlapping or duplicate cells')
        if not boundary or any(len(v)!=2 for v in boundary.values()):raise ValueError('Invalid proxy boundary')
        seen=set();stack=[next(iter(boundary))]
        while stack:
            x=stack.pop()
            if x in seen:continue
            seen.add(x);stack.extend(boundary[x]-seen)
        if len(seen)!=len(boundary):raise ValueError('Disconnected or holed plane proxy')
        boundary_cycles.append(dict(group=group,single_cycle=True,vertices=len(boundary)))
    reconstructed=[sorted(v) for v in fans.values()]
    if collections.Counter(tuple(x) for x in reconstructed)!=collections.Counter(tuple(sorted(x['faces'])) for x in fan_rows):raise ValueError('Actual geometric fans differ from CGAL')
    if any(len(x) not in (1,4) for x in reconstructed):raise ValueError('Unexpected arrangement fan')
    # Reuse guarded source-distance witnesses, but independently prove each lies
    # within1e-10 of the old polygon and transfer to the exact cell within1e-10.
    witnesses=[];threshold=Q(1,5)+Q(1,10**8);transfer=Q(2,10**10)
    for face,c in zip(faces,cert):
        if face['key']!=c['key'] or face['variable']!=c['variable']:raise ValueError('Certificate correspondence')
        if c['state']!='CONFIRMED_UNSUPPORTED':continue
        if c['reason']!='OUTSIDE_COMPATIBLE_DOMAIN':raise ValueError('Missing genuine witness')
        p=tuple(map(rational,c['witness']));v=[tuple(map(rational,q)) for q in face['vertices']]
        squared=min(point_triangle_squared(p,v[0],v[i],v[i+1]) for i in range(1,len(v)-1))
        if squared>Q(1,10**10)**2:raise ValueError('Witness outside polygon')
        lower=rational(c['witness_distance_native'])-transfer
        if lower<=threshold:raise ValueError('Unsupported witness transfer insufficient')
        witnesses.append(dict(variable=face['variable'],footprint_squared_distance_exact=str(squared),transferred_source_distance_lower_native=float(lower),guarded_source_distance_not_claimed_exact=True))
    if len(witnesses)!=968:raise ValueError('Incomplete unsupported exclusions')
    proof=prune(len(faces),reconstructed,[w['variable'] for w in witnesses]);remaining=proof['remaining']
    prior_remaining=json.loads((old/'certified-closure-screen.json').read_text())['remaining']
    if remaining!=prior_remaining:raise ValueError('Conforming selection implication changed')
    # Conservative exact witness to ALL surviving ideal-cell boxes. These
    # boxes are diagnostic enclosing sets, never construction geometry.
    row=json.loads((old/'fidelity-impossibility.json').read_text())['witness']['source_sample_row']
    with (r.previous.PRIOR/'validation-points.bin').open('rb') as f:f.seek(row*24);p=tuple(map(rational,struct.unpack('<ddd',f.read(24))))
    distances=[];boxes=[];guard=Q(1,10**8)
    for i in remaining:
        v=[vertices[tuple(t)] for t in cells[i]['triples']]
        low=[min(q[k] for q in v)-guard for k in range(3)];high=[max(q[k] for q in v)+guard for k in range(3)]
        boxes.append((low,high))
        d=[max(low[k]-p[k],p[k]-high[k],Q(0)) for k in range(3)];distances.append((dot(d,d),i))
    squared,closest=min(distances)
    if squared<=Q(100000001,100000000)**2:raise ValueError('No transferred fidelity obstruction')
    coverage=broad_coverage(r.previous.PRIOR/'validation-points.bin',boxes)
    r.owner.save(b/'complete-source-coverage.json',coverage)
    r.owner.save(b/'exact-cell-identities.json',dict(cells=cells,plane_coefficients_rational=[[str(x) for x in p] for p in planes],vertices=[dict(triple=k,coordinates=[str(x) for x in v]) for k,v in vertices.items()],fans=[dict(endpoints=k,faces=v) for k,v in fans.items()],proxy_boundary_cycles=boundary_cycles))
    r.owner.save(b/'witness-transfers.json',witnesses)
    r.owner.save(b/'coarse-constraint-proof.json',proof)
    r.owner.save(b/'cell-audit.json',dict(classification='COMPLETE_FIXED_PLANE_CELL_COMPLEX_WITH_PRESERVED_OBSTRUCTION',candidate_faces=len(faces),source_planes=source_planes,proxy_planes=6,vertex_occurrences=occurrences,unique_triples=len(vertices),exact_missing_plane_cuts=0,ambiguous_vertices=0,duplicate_cells=0,geometric_fans=len(fans),recorded_fan_multiset_exact_match=True,full_connected_convex_cells=True,all_plane_proxies_single_boundary_cycle=True,source_training_points=points,source_proxy_offset=offset,source_derived_proxy_bounds=[lo,hi],proxy_bound_discrepancy_native=bound_error,maximum_exact_vertex_displacement_squared=str(max_delta2),maximum_exact_vertex_displacement_native=math.sqrt(float(max_delta2)),unsupported_witnesses_transferred=len(witnesses),remaining_possible_original_cells=len(remaining),fidelity_witness_row=row,exact_ideal_box_distance_squared=str(squared),ideal_box_distance_lower_native=math.sqrt(float(squared)),closest_ideal_box=closest,frozen_max_native=1.,subdivision_invariance='For embedded closed surfaces on unchanged planes, interior coplanar two-face edges impose equal selection; connected partition propagates across each full original cell; omitted unsupported complement is zero',geometry_exported=False,source_geometry_changed=False,solver_calls=0,original_AI_geometry_inadequacy_not_proven=True))
