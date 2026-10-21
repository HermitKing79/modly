"""Local saved-graph witnesses, not a repeat full topology assessment."""
import sys
sys.dont_write_bytecode=True
import numpy as np

def canonical(face):
    t=tuple(map(int,face));i=t.index(min(t));return t[i:]+t[:i]

def incidence(keys,counts,a,b,n):
    key=np.uint64(min(a,b))*np.uint64(n)+np.uint64(max(a,b));j=int(np.searchsorted(keys,key))
    return int(counts[j]) if j<len(keys) and keys[j]==key else 0

def local_faces(faces,vertices):
    """Read-only streaming stars; no graph rebuild, no all-face Python objects."""
    vertices=np.array(sorted(set(map(int,vertices))),np.int32);rows=[]
    for s in range(0,len(faces),32768):
        hit=np.isin(faces[s:s+32768],vertices).any(1)
        rows.extend((s+np.flatnonzero(hit)).tolist())
        if len(rows)>32768:raise ValueError('Local star collection exceeds frozen32768-face work bound; no truncation')
    return np.array(rows,np.int32)

def face_records(v,f,rows,source_ids=None):
    records=[]
    for row in rows:
        indices=f[int(row)];source=indices if source_ids is None else source_ids[indices]
        xyz=v[indices].astype(np.float64);cross=np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0]);norm=np.linalg.norm(cross)
        if not np.isfinite(norm) or norm==0:raise ValueError('Invalid witness face')
        records.append(dict(face_id=int(row),vertex_ids=indices.tolist(),source_vertex_ids=source.tolist(),positions=xyz.tolist(),unit_normal=(cross/norm).tolist(),area=float(norm*.5),canonical_oriented_source_ids=list(canonical(source))))
    return records

def angular_stats(records):
    if len(records)<2:return None
    n=np.array([r['unit_normal'] for r in records]);angles=np.degrees(np.arccos(np.clip(n@n.T,-1,1)));a=angles[np.triu_indices(len(n),1)]
    return dict(min_degrees=float(a.min()),max_degrees=float(a.max()),opposed_pairs=int((a>90).sum()),pair_count=len(a))

def run(np,data,cv,cf,ids,rgba,out):
    import common as c
    v,f=data['vertices'],data['faces'];N=len(v);n=len(cv)
    oldkeys=np.load(c.original.CACHE/'edge_keys.npy',mmap_mode='r',allow_pickle=False)
    oldinc=np.load(c.original.CACHE/'incidence.npy',mmap_mode='r',allow_pickle=False)
    keys=np.load(c.PREVIOUS/'assess/cache/edge_keys.npy',mmap_mode='r',allow_pickle=False)
    inc=np.load(c.PREVIOUS/'assess/cache/incidence.npy',mmap_mode='r',allow_pickle=False)
    c.require(int(inc.max())==9,'Recorded maximum incidence nine')
    bad=np.flatnonzero(inc>2)
    low=ids[keys[bad]//np.uint64(n)].astype(np.uint64);high=ids[keys[bad]%np.uint64(n)].astype(np.uint64)
    global_keys=low*np.uint64(N)+high
    at=np.searchsorted(oldkeys,global_keys);safe=np.minimum(at,len(oldkeys)-1)
    oldcount=np.where((at<len(oldkeys))&(oldkeys[safe]==global_keys),oldinc[safe],0)
    novel=bad[oldcount<=2]
    c.require(len(novel)==116786,'Exact saved newly non-manifold edge population')
    nine=novel[inc[novel]==9]
    primary=int(nine[0]) if len(nine) else int(np.flatnonzero(inc==9)[0])
    rest=novel[novel!=primary];ranks=np.linspace(0,len(rest)-1,8,dtype=np.int64)
    selected=[primary]+rest[ranks].tolist()
    endpoints=[(int(keys[j]//np.uint64(n)),int(keys[j]%np.uint64(n))) for j in selected]
    global_endpoints=[(int(ids[a]),int(ids[b])) for a,b in endpoints]
    srcstars=local_faces(f,[x for pair in global_endpoints for x in pair])
    dststars=local_faces(cf,[x for pair in endpoints for x in pair])
    witnesses=[]
    for number,(j,(a,b),(ga,gb)) in enumerate(zip(selected,endpoints,global_endpoints)):
        ca=dststars[(cf[dststars]==a).any(1)&(cf[dststars]==b).any(1)]
        oa=srcstars[(f[srcstars]==ga).any(1)&(f[srcstars]==gb).any(1)]
        c.require(len(ca)==int(inc[j]) and len(oa)==incidence(oldkeys,oldinc,ga,gb,N),'Actual local adjacency reproduces saved incidence')
        original_stars=srcstars[np.isin(f[srcstars],np.array([ga,gb],np.int32)).any(1)]
        c.require(len(original_stars)<=4096 and len(ca)<=4096,'Bounded4096-face witness; no truncation')
        orig_records=face_records(v,f,original_stars);dst_records=face_records(cv,cf,ca,ids)
        oldfacekeys={canonical(x) for x in f[original_stars]}
        signatures=[canonical(ids[cf[k]]) for k in ca]
        duplicate_groups=[list(map(int,ca[[s==key for s in signatures]])) for key in sorted(set(signatures)) if signatures.count(key)>1]
        endpoint_status=[]
        for g in (ga,gb):
            relevant=((oldkeys//np.uint64(N)==g)|(oldkeys%np.uint64(N)==g))
            endpoint_status.append(dict(source_id=g,original_boundary_edges=int(np.count_nonzero(relevant&(oldinc==1))),original_nonmanifold_edges=int(np.count_nonzero(relevant&(oldinc>2)))))
        occurrences=[]
        for row,r in zip(ca,dst_records):
            sf=ids[cf[row]];r['original_pair_incidences']=[incidence(oldkeys,oldinc,int(sf[k]),int(sf[(k+1)%3]),N) for k in range(3)]
            r['matches_original_oriented_triangle']=canonical(sf) in oldfacekeys
            r['native_colour_bytes']=rgba[cf[row]].tolist()
            occurrences.append(r)
        witnesses.append(dict(witness_number=number,saved_edge_rank=int(j),candidate_endpoints=[a,b],source_endpoints=[ga,gb],original_endpoint_edge_incidence=len(oa),candidate_incidence=len(ca),distinct_oriented_triangles=len(set(signatures)),duplicate_groups=duplicate_groups,endpoint_original_junctions=endpoint_status,original_star_faces=orig_records,candidate_incident_faces=occurrences,candidate_normal_pair_angles=angular_stats(dst_records),original_common_edge_faces=oa.tolist(),new_candidate_triangles=sum(not x['matches_original_oriented_triangle'] for x in occurrences)))
    # Prefer a duplicate in the deterministic primary maximum-incidence star.
    dup=None
    for w in witnesses:
        if w['duplicate_groups']:
            dup=dict(selection='first duplicate group in primary-then-eight fixed edge witnesses',edge_witness=w['witness_number'],candidate_face_ids=w['duplicate_groups'][0]);break
    if dup is None:
        # Only a witness lookup, not a new quality count or full topology run.
        contract=c.pin('attribution_old_face_keys',c.OLD/'reduction_contract.py','bdf0fa90da2a4b93020dca1ec73423ec5b8b22b48f951460bd4e9d23a5d8babc')
        k=contract.oriented_keys(cf,n);order=np.argsort(k,kind='stable');where=np.flatnonzero(k[order][1:]==k[order][:-1]);c.require(len(where)>0,'Saved duplicate reproducible');j=int(where[0]);key=k[order[j]]
        dup=dict(selection='lowest cyclic oriented candidate key; stable face order',edge_witness=None,candidate_face_ids=np.flatnonzero(k==key).tolist())
    records=face_records(cv,cf,np.array(dup['candidate_face_ids'],np.int32),ids)
    c.require(len(records)>=2 and len({tuple(x['canonical_oriented_source_ids']) for x in records})==1,'Reproduced same oriented triangle multiplicity')
    src=records[0]['source_vertex_ids'];original_rows=local_faces(f,src)
    matches=[int(j) for j in original_rows if canonical(f[j])==canonical(src)]
    dup.update(records=records,matching_original_oriented_face_ids=matches,classification='same original-ID ordered surface repeated by candidate connectivity')
    payload=dict(primary=witnesses[0],additional=witnesses[1:],duplicate=dup,selection=dict(population_newly_nonmanifold_pairs=len(novel),additional_ranks=ranks.tolist(),sample_size=8),collapse_ancestry_available=False)
    c.save(out/'topology-witnesses.json',payload)
    # A local wireframe schematic, not a shaded render or geometry export.
    from PIL import Image,ImageDraw
    image=Image.new('RGB',(1000,520),'white');draw=ImageDraw.Draw(image)
    primary_w=witnesses[0]
    records=primary_w['original_star_faces']+primary_w['candidate_incident_faces']
    xyz=np.concatenate([np.array(r['positions']) for r in records]);spread=np.ptp(xyz,axis=0);axes=np.argsort(-spread)[:2]
    low=xyz[:,axes].min(0);high=xyz[:,axes].max(0);scale=420/max(float((high-low).max()),1e-12)
    for panel,rs in enumerate((primary_w['original_star_faces'],primary_w['candidate_incident_faces'])):
        draw.text((panel*500+10,10),('Original endpoint stars','Rejected candidate: 9 face occurrences')[panel],fill='black')
        for r in rs:
            points=(np.array(r['positions'])[:,axes]-low)*scale
            coords=[(panel*500+35+float(a),475-float(b)) for a,b in points]
            draw.line(coords+[coords[0]],fill=('#999999','#b22222')[panel],width=1)
        for g in primary_w['source_endpoints']:
            a,b=(v[g,axes]-low)*scale;x=panel*500+35+float(a);y=475-float(b)
            draw.ellipse((x-4,y-4,x+4,y+4),fill='blue');draw.text((x+5,y),str(g),fill='blue')
    draw.text((10,500),'Local orthographic wireframe schematic; same frame; no mesh change, lighting, or Blender render.',fill='black')
    image.save(out/'junction-wireframe.png');image.close()
    summary=[]
    for w in witnesses:
        summary.append({k:w[k] for k in ('witness_number','saved_edge_rank','source_endpoints','original_endpoint_edge_incidence','candidate_incidence','distinct_oriented_triangles','duplicate_groups','endpoint_original_junctions','candidate_normal_pair_angles','new_candidate_triangles')})
    return dict(witnesses=summary,duplicate={k:dup[k] for k in ('selection','edge_witness','candidate_face_ids','matching_original_oriented_face_ids','classification')},sample_is_not_population_causal_proof=True,collapse_ancestry_available=False)
