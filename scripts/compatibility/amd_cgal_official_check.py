"""Independent OFF/topology and actual SDK-runtime check; no native imports."""
import hashlib
import math
from pathlib import Path
import re
import zipfile
from collections import defaultdict


def mesh_counts(vertices, faces):
    edges=defaultdict(list);links=defaultdict(list);used=set()
    parent=list(range(len(vertices)))
    def find(v):
        while parent[v]!=v:parent[v]=parent[parent[v]];v=parent[v]
        return v
    for row,face in enumerate(faces):
        if len(face)<3 or len(set(face))!=len(face) or any(v<0 or v>=len(vertices) for v in face):raise ValueError('Invalid polygon')
        used.update(face)
        for i,a in enumerate(face):
            b=face[(i+1)%len(face)];edges[tuple(sorted((a,b)))].append((row,a,b))
            parent[find(a)]=find(b);links[a].append((face[i-1],b))
    bad_links=0
    for v,segments in links.items():
        graph=defaultdict(list)
        for a,b in segments:graph[a].append(b);graph[b].append(a)
        seen=set();stack=[next(iter(graph))]
        while stack:
            a=stack.pop()
            if a in seen:continue
            seen.add(a);stack.extend(graph[a])
        if len(seen)!=len(graph) or any(len(n)!=2 for n in graph.values()):bad_links+=1
    return dict(vertices=len(vertices),polygons=len(faces),fan_triangle_count=sum(len(f)-2 for f in faces),
                boundary_edges=sum(len(v)==1 for v in edges.values()),nonmanifold_edges=sum(len(v)>2 for v in edges.values()),
                winding_conflicts=sum(len(v)==2 and v[0][1:]==v[1][1:] for v in edges.values()),
                face_bearing_vertex_components=len({find(v) for v in used}),unused_vertices=len(vertices)-len(used),
                nonmanifold_vertex_links=bad_links,
                duplicate_polygon_vertex_sets=len(faces)-len({tuple(sorted(f)) for f in faces}))


def read_off(path):
    if path.stat().st_size>16*1024**2:raise ValueError('Bounded OFF required')
    lines=[line.split('#')[0].strip() for line in path.read_text().splitlines()]
    lines=[line for line in lines if line]
    if lines.pop(0)!='OFF':raise ValueError('OFF header')
    nv,nf,_=map(int,lines.pop(0).split());vertices=[tuple(map(float,line.split())) for line in lines[:nv]]
    if len(vertices)!=nv or any(len(v)!=3 or not all(map(math.isfinite,v)) for v in vertices):raise ValueError('Finite OFF vertices required')
    faces=[]
    for line in lines[nv:nv+nf]:
        row=list(map(int,line.split()))
        if row[0]!=len(row)-1:raise ValueError('OFF polygon record')
        faces.append(row[1:])
    if len(faces)!=nf or not nv or not nf:raise ValueError('Nonempty complete OFF required')
    return vertices,faces


def verify(base):
    import amd_cgal_proof as p
    log=(base/'sample-official-run/stdout.txt').read_text()
    status=re.findall(r'MODLY_SCIP_STATUS optimal=(\d+) code=(\d+) gap=([^ ]+)',log)
    if status!=[('1','1','0')]:raise ValueError('Exactly one actual optimal zero-gap solve required')
    if 'SCIP version 10.1.0' not in log or '[LP solver: SoPlex 8.1.0]' not in log:raise ValueError('Actual solver versions differ')
    candidates=re.findall(r'MODLY_CANDIDATES faces=(\d+)',log)
    if len(candidates)!=1 or not 4<=int(candidates[0])<=20000:raise ValueError('Candidate complexity receipt')
    modules=re.findall(r'MODLY_MODULE\|([^|]+)\|([^\r\n]+)',log)
    actual={name.lower():Path(path) for name,path in modules};sdk=[]
    with zipfile.ZipFile(base/'downloads/scipoptsuite-10.1.0-win-x64.zip') as z:
        for name,path in actual.items():
            resolved=path.resolve()
            if resolved.is_relative_to(p.SCIP/'bin'):
                expected=z.read('scipoptsuite-10.1.0-win-x64/bin/'+path.name)
                if hashlib.sha256(path.read_bytes()).digest()!=hashlib.sha256(expected).digest():raise ValueError('Loaded SDK payload differs')
                sdk.append(name)
            elif name=='official_sample.exe':
                if resolved!=(p.BUILD/'Release/official_sample.exe').resolve():raise ValueError('Sample process differs')
            elif not str(resolved).lower().startswith('c:\\windows\\'):
                raise ValueError('Unexpected non-SDK/non-Windows runtime module')
    if 'libscip.dll' not in sdk or 'gmp-10.dll' not in sdk or 'mpfr-6.dll' not in sdk:raise ValueError('Required actual runtime modules absent')
    v,f=read_off(base/'official-sample/user_provided_planes_result.off');counts=mesh_counts(v,f)
    for key in ('boundary_edges','nonmanifold_edges','winding_conflicts','unused_vertices','nonmanifold_vertex_links','duplicate_polygon_vertex_sets'):
        if counts[key]:raise ValueError('Official mesh topology: '+key)
    if counts['face_bearing_vertex_components']!=1:raise ValueError('Official sample component count')
    return dict(classification='PASS_OFFICIAL_SAMPLE_SOLVER_AND_EXPORT',**counts,candidate_faces=int(candidates[0]),
                actual_status='SCIP_STATUS_OPTIMAL',gap=0,SCIP='10.1.0',SoPlex='8.1.0',
                authenticated_loaded_SDK_modules=sorted(sdk),runtime_module_count=len(modules),
                sample_mesh_sha256=p.sha(base/'official-sample/user_provided_planes_result.off'),
                sample_never_substitutes_for_AI_output=True,AI_geometry_quality_pass=False,
                complete_self_intersections_or_sample_fidelity_not_measured=True)
