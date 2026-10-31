"""Nonplanar source-feature inputs and independent CPU result assessment."""
import json
from pathlib import Path
import struct
import amd_alpha_wrap_proof as run
from amd_cgal_proof import save,sha

def prepare():
    import numpy as np
    b=run.BASE;r=run.ROOT14;a=json.loads((b/'authentication.json').read_text())
    names=['input/mesh.bin','input/receipt.json','validation-points.bin','region-labels.bin','owned-session/refined-screen.json']
    pins={}
    for name in names:
        p=r/name;row=run.prior.previous.old.manifest_identity(a['Phase14_files'],name)
        pins[str(p)]=run.check(p,row)
    candidate=run.final_candidate();run.artifact('simplify' if (b/'simplify-run').exists() else 'wrap',candidate)
    pins[str(candidate)]=dict(bytes=candidate.stat().st_size,sha256=sha(candidate))
    with (r/'input/mesh.bin').open('rb') as stream:nv,nf=struct.unpack('<QQ',stream.read(16))
    regions=np.fromfile(r/'region-labels.bin',dtype='<i4')
    if len(regions)!=nf:raise ValueError('Source region correspondence')
    screen=json.loads((r/'owned-session/refined-screen.json').read_text());patches=screen['original_patch_survival']
    if len(patches)!=51:raise ValueError('Historical feature completeness')
    with (b/'feature-input.bin').open('xb') as out:
        out.write(struct.pack('<Q',len(patches)))
        for p in patches:
            ids=np.flatnonzero(regions==p['region']).astype('<u4')
            if len(ids)!=p['source_triangles']:raise ValueError('Source feature face count differs')
            # Actual face normals and all three incidences are computed from
            # original triangle coordinates by the independent native validator.
            out.write(struct.pack('<iQ',p['region'],len(ids)));out.write(ids.tobytes())
    p=b/'feature-input.bin';pins[str(p)]=dict(bytes=p.stat().st_size,sha256=sha(p))
    save(b/'validation-inputs.json',pins)

def assess():
    import numpy as np
    from amd_cgal_official_check import read_off,mesh_counts
    from amd_mesh_aware_result import png,boundary,boundary_distance,point_triangle_distances
    b=run.BASE;folder=b/'validation';raw=run.final_candidate()
    for name,row in json.loads((b/'validation-inputs.json').read_text()).items():run.check(Path(name),row)
    native=json.loads((folder/'native.json').read_text())
    if not native['complete']:raise ValueError('Incomplete native measurements')
    vertices,faces=read_off(raw);v=np.array(vertices,'f8');f=np.array(faces,'i4');counts=mesh_counts(vertices,faces)
    tri=v[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);zero=int((np.linalg.norm(cross,axis=1)==0).sum())
    topology=all(counts[k]==0 for k in ('boundary_edges','nonmanifold_edges','winding_conflicts','unused_vertices','nonmanifold_vertex_links','duplicate_polygon_vertex_sets')) and counts['face_bearing_vertex_components']==1 and zero==0 and len(np.unique(v,axis=0))==len(v) and np.isfinite(v).all()
    native_check=json.loads((b/'final-check.json').read_text());topology=bool(topology and native_check['classification']=='PASS_CLOSED_EMBEDDED_MESH')
    distances={}
    for name in ('source-to-candidate','candidate-to-source'):
        d=np.fromfile(folder/(name+'.bin'),dtype='<f8')
        if d.shape!=(65536,) or not np.isfinite(d).all():raise ValueError('Complete finite distances required')
        row=dict(samples=len(d),RMS=float(np.sqrt(np.mean(d*d))),p95=float(np.quantile(d,.95)),p99=float(np.quantile(d,.99)),maximum=float(d.max()))
        row['PASS']=row['RMS']<=.0015 and row['p95']<=.003 and row['p99']<=.005 and row['maximum']<=.01;distances[name]=row
    views=[];images=[]
    for az in range(0,360,45):
        masks=[]
        for name in ('source','candidate'):
            data=(folder/(name+'-'+str(az)+'.ppm')).read_bytes();prefix=b'P6\n1024 1024\n255\n'
            if not data.startswith(prefix):raise ValueError('Frozen image schema')
            im=np.frombuffer(data[len(prefix):],dtype='u1').reshape(1024,1024,3);png(folder/(name+'-'+str(az)+'.png'),np,im);images.append(im[::2,::2].copy())
            data=(folder/(name+'-'+str(az)+'.pgm')).read_bytes();prefix=b'P5\n1024 1024\n255\n'
            if not data.startswith(prefix):raise ValueError('Frozen mask schema')
            masks.append(np.frombuffer(data[len(prefix):],dtype='u1').reshape(1024,1024)>0)
        x,y=masks;union=int((x|y).sum());iou=float((x&y).sum()/union) if union else 0.
        bx,by=boundary(np,x),boundary(np,y);h=max(boundary_distance(np,bx,by),boundary_distance(np,by,bx))
        views.append(dict(azimuth=az,elevation=20,IoU=iou,boundary_pixels=h,PASS=iou>=.995 and h<=2))
    contact=np.empty((2048,2048,3),'u1')
    for i,im in enumerate(images):contact[(i//4)*512:(i//4+1)*512,(i%4)*512:(i%4+1)*512]=im
    png(folder/'matched-eight-view-contact.png',np,contact)
    supports={name:json.loads((folder/(name+'.json')).read_text()) for name in ('candidate-full-support','source-full-support')}
    features=[json.loads(x) for x in (folder/'features.jsonl').read_text().splitlines()]
    if len(features)!=51:raise ValueError('Incomplete feature measurements')
    samples=np.fromfile(run.ROOT14/'validation-points.bin',dtype='<f8').reshape(65536,3);ids=np.linspace(0,65535,256,dtype='i8');independent=np.full(256,np.inf)
    for triangle in tri:independent=np.minimum(independent,point_triangle_distances(np,samples[ids],triangle)/100)
    error=float(abs(independent-np.fromfile(folder/'source-to-candidate.bin',dtype='<f8')[ids]).max())
    receipt=json.loads((run.ROOT14/'input/receipt.json').read_text());original=v/receipt['scale']+receipt['center']
    out=folder/'AI-DERIVED-UNQUALIFIED.ply'
    with out.open('x') as stream:
        stream.write(f'ply\nformat ascii 1.0\ncomment AI-DERIVED; acceptance in result.json\nelement vertex {len(v)}\nproperty double x\nproperty double y\nproperty double z\nelement face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n')
        for p in original:stream.write(' '.join(format(float(x),'.17g') for x in p)+'\n')
        for face in f:stream.write('3 '+' '.join(map(str,face))+'\n')
    lines=out.read_text().splitlines();start=lines.index('end_header')+1
    rv=np.array([[float(x) for x in line.split()] for line in lines[start:start+len(v)]]);rf=np.array([[int(x) for x in line.split()[1:]] for line in lines[start+len(v):]])
    readback=bool(np.array_equal(rv,original) and np.array_equal(rf,f))
    checks=dict(topology=topology,triangle_budget=len(f)<=1000,distances=all(x['PASS'] for x in distances.values()),views=all(x['PASS'] for x in views),full_surface_support=all(x['classification']=='CERTIFIED' for x in supports.values()),features=all(x['PASS'] for x in features),independent_distances=error<=5e-12,readback=readback)
    html='<html><meta charset="utf-8"><h1>AI source / Alpha Wrapping candidate</h1><p>Left source, right reconstruction; unchanged shared CPU cameras. See result.json for qualification.</p>'
    for az in range(0,360,45):html+=f'<h2>{az} degrees</h2><img width="48%" src="source-{az}.png"><img width="48%" src="candidate-{az}.png">'
    with (folder/'review.html').open('x') as stream:stream.write(html+'</html>')
    save(b/'result.json',dict(classification='PASS_PENDING_INDEPENDENT_VISUAL_REVIEW' if all(checks.values()) else 'NO_GO_GEOMETRY',label='AI-DERIVED RECONSTRUCTION',raw_neural_PASS=False,checks=checks,counts=counts,triangles=len(f),vertices=len(v),distance=distances,views=views,full_support=supports,features=features,independent_distance_max_error=error,export_sha256=sha(out),visual_review_required=True,plane_checks='NOT_APPLICABLE'))
