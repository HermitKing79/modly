"""Independent frozen candidate gates; CPU-only images and readback."""
import json
import math
import struct
import zlib
from pathlib import Path
import amd_mesh_aware_proof as run
from amd_cgal_proof import save,sha

def prepare():
    import numpy as np
    from amd_cgal_official_check import read_off,mesh_counts
    path=run.BASE/'owned-session/candidate/raw-reconstruction.off'
    vertices,faces=read_off(path);counts=mesh_counts(vertices,faces)
    z=np.load(run.PRIOR/'ai-input/samples.npz',allow_pickle=False)
    with (run.BASE/'validation-points.bin').open('xb') as stream:((z['points']-z['center'])*float(z['scale'])).astype('<f8').tofile(stream)
    save(run.BASE/'candidate-input-pin.json',dict(raw_candidate_sha256=sha(path),counts=counts,validation_source_points=65536,validation_points_sha256=sha(run.BASE/'validation-points.bin'),candidate_is_AI_DERIVED_RECONSTRUCTION=True,raw_neural_geometry_PASS=False))

def png(path,np,image):
    def chunk(name,data):return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data)&0xffffffff)
    h,w=image.shape[:2];image=np.asarray(image,dtype='u1')
    raw=b''.join(b'\0'+row.tobytes() for row in image)
    with path.open('xb') as out:out.write(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b''))

def boundary(np,mask):
    inner=mask.copy();inner[1:]&=mask[:-1];inner[:-1]&=mask[1:];inner[:,1:]&=mask[:,:-1];inner[:,:-1]&=mask[:,1:]
    return np.argwhere(mask&~inner)

def boundary_distance(np,a,b):
    if not len(a) or not len(b):return math.inf
    maximum=0.
    for start in range(0,len(a),128):
        delta=a[start:start+128,None,:].astype('f8')-b[None,:,:]
        maximum=max(maximum,float(np.sqrt((delta*delta).sum(axis=2).min(axis=1)).max()))
    return maximum

def point_triangle_distances(np,points,t):
    a,b,c=np.moveaxis(t,-2,0);u=b-a;v=c-a;normal=np.cross(u,v);length=np.linalg.norm(normal,axis=-1)
    if np.any(length<=0):raise ValueError('Nondegenerate triangle required')
    normal=normal/length[...,None]
    signed=((points-a)*normal).sum(axis=-1);projection=points-signed[:,None]*normal
    d00=(u*u).sum(axis=-1);d01=(u*v).sum(axis=-1);d11=(v*v).sum(axis=-1);q=projection-a;d20=(q*u).sum(axis=-1);d21=(q*v).sum(axis=-1);denom=d00*d11-d01*d01
    s=(d11*d20-d01*d21)/denom;w=(d00*d21-d01*d20)/denom
    inside=(s>=0)&(w>=0)&(s+w<=1);distance=np.full(len(points),np.inf)
    for start,end in ((a,b),(b,c),(c,a)):
        edge=end-start;alpha=np.clip(((points-start)*edge).sum(axis=-1)/(edge*edge).sum(axis=-1),0,1)
        distance=np.minimum(distance,np.linalg.norm(points-start-alpha[:,None]*edge,axis=1))
    distance[inside]=abs(signed[inside]);return distance

def prepare_domains():
    import numpy as np
    from amd_mesh_aware_input import RAW
    v=np.load(RAW/'vertices.npy',allow_pickle=False);f=np.load(RAW/'faces.npy',allow_pickle=False)
    src=np.load(run.BASE/'input/source-analysis.npz',allow_pickle=False);D=float(src['diagonal']);scale=100/D;center=src['center']
    final=json.loads((run.BASE/'owned-session/refined-screen.json').read_text());planes=final['planes']
    tri=v[f].astype('f8');parts=[];counts=[];native=[]
    for plane in planes:
        n=np.array(plane['normal']);mask=(abs(tri@n-plane['offset']).max(axis=1)<=.002*D)&(abs(src['normal']@n)>=math.cos(math.radians(10)))
        rows=np.flatnonzero(mask).astype('u4');parts.append(np.column_stack((np.full(len(rows),plane['id'],'u4'),rows)))
        counts.append(dict(group=plane['id'],genuine_source_triangles=len(rows)))
        native.append([*n,(plane['offset']-n@center)*scale])
    records=np.concatenate(parts)
    with (run.BASE/'finite-domain-input.bin').open('xb') as out:
        np.array([len(planes),len(records)],'<u8').tofile(out);np.array(native,'<f8').tofile(out);records.astype('<u4').tofile(out)
    save(run.BASE/'finite-domain-input.json',dict(sha256=sha(run.BASE/'finite-domain-input.bin'),groups=counts,total_memberships=len(records),
        domains='All real original triangles compatible with exact observed plane: all three vertices within .002D,unsigned normal within10degrees; no invented or hull geometry',
        heldout_source_geometry_used_for_evaluation_only=True,training_or_plane_fit_changed=False))

def witnesses():
    """Read-only independent witnesses for the finite-domain certificate refusal."""
    import numpy as np
    from amd_mesh_aware_input import RAW
    from amd_cgal_official_check import read_off
    domain=json.loads((run.BASE/'domain-certification.json').read_text())
    failed=[r['group'] for r in domain['groups'] if not r['certified']]
    with (run.BASE/'finite-domain-input.bin').open('rb') as stream:
        sizes=np.fromfile(stream,dtype='<u8',count=2);planes=np.fromfile(stream,dtype='<f8',count=int(sizes[0])*4).reshape(-1,4);rows=np.fromfile(stream,dtype='<u4').reshape(-1,2)
    v,f=read_off(run.BASE/'validation/triangulated-native.off');tri=np.array(v)[np.array(f)]
    src=np.load(run.BASE/'input/source-analysis.npz',allow_pickle=False);scale=100/float(src['diagonal']);center=src['center']
    source_v=np.load(RAW/'vertices.npy',allow_pickle=False);source_f=np.load(RAW/'faces.npy',allow_pickle=False)
    result=[]
    for group in failed:
        n=planes[group,:3];offset=planes[group,3];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None]
        ids=np.flatnonzero((abs(tri@n-offset).max(axis=1)<=1e-6)&(abs(normal@n)>=math.cos(math.radians(1e-5))))
        reference_rows=rows[rows[:,0]==group,1];reference=(source_v[source_f[reference_rows]].astype('f8')-center)*scale
        worst=None
        for index in ids:
            t=tri[index];probes=np.concatenate([t,(t+np.roll(t,1,axis=0))/2,t.mean(axis=0)[None,:]])
            for point in probes:
                d=point_triangle_distances(np,np.broadcast_to(point,(len(reference),3)),reference)
                j=int(np.argmin(d));minimum=float(d[j]/100)
                if worst is None or minimum>worst['distance_to_compatible_domain_over_D']:
                    worst=dict(group=group,candidate_triangle=int(index),probe_native=point.tolist(),probe_original_units=(point/scale+center).tolist(),distance_to_compatible_domain_over_D=minimum,nearest_compatible_source_triangle=int(reference_rows[j]))
        point=np.array(worst['probe_native']);nearest=np.inf
        for start in range(0,len(source_f),32768):
            t=(source_v[source_f[start:start+32768]].astype('f8')-center)*scale
            nearest=min(nearest,float(point_triangle_distances(np,np.broadcast_to(point,(len(t),3)),t).min()/100))
        worst.update(distance_to_entire_source_over_D=nearest,confirmed_outside_compatible_domain=worst['distance_to_compatible_domain_over_D']>.002+1e-10)
        result.append(worst)
    confirmed=all(r['confirmed_outside_compatible_domain'] for r in result)
    save(run.BASE/'domain-witnesses.json',dict(classification='CONFIRMED_UNSUPPORTED_ASSIGNED_DOMAIN' if confirmed else 'INCOMPLETE_DOMAIN_PROOF',witnesses=result,independent_algorithm='NumPy brute-force Euclidean point-to-triangle projection and edge distances; no AABB reuse',source_and_candidate_unchanged=True,no_new_MIP=True))

def assess():
    import numpy as np
    from amd_cgal_official_check import read_off,mesh_counts
    folder=run.BASE/'validation';raw=run.BASE/'owned-session/candidate/raw-reconstruction.off'
    if sha(raw)!=json.loads((run.BASE/'candidate-input-pin.json').read_text())['raw_candidate_sha256']:raise ValueError('Candidate changed')
    native=json.loads((folder/'native.json').read_text());vertices,faces=read_off(folder/'triangulated-native.off')
    domains=json.loads((run.BASE/'domain-certification.json').read_text())
    counts=mesh_counts(vertices,faces);v=np.array(vertices);f=np.array(faces,dtype='i4')
    tri=v[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);zero=int((np.linalg.norm(cross,axis=1)==0).sum())
    duplicate_positions=len(v)-len(np.unique(v,axis=0))
    topology=all(counts[k]==0 for k in ('boundary_edges','nonmanifold_edges','winding_conflicts','unused_vertices','nonmanifold_vertex_links','duplicate_polygon_vertex_sets')) and counts['face_bearing_vertex_components']==1 and zero==0 and duplicate_positions==0 and np.isfinite(v).all() and native['complete_self_intersections']==0 and len(f)<=1000
    distances={};distance_pass=True
    for name in ('source-to-candidate','candidate-to-source'):
        d=np.fromfile(folder/(name+'.bin'),dtype='<f8')
        if d.shape!=(65536,) or not np.isfinite(d).all():raise ValueError('Complete finite surface distances required')
        stats=dict(samples=len(d),RMS=float(np.sqrt(np.mean(d*d))),p95=float(np.quantile(d,.95)),p99=float(np.quantile(d,.99)),maximum=float(d.max()))
        stats['PASS']=stats['RMS']<=.0015 and stats['p95']<=.003 and stats['p99']<=.005 and stats['maximum']<=.01
        distance_pass=distance_pass and stats['PASS'];distances[name]=stats
    images=[];views=[];view_pass=True
    for az in range(0,360,45):
        masks=[]
        for name in ('source','candidate'):
            path=folder/(name+'-'+str(az)+'.ppm');data=path.read_bytes();header=b'P6\n1024 1024\n255\n'
            if not data.startswith(header):raise ValueError('Fixed raster dimensions')
            image=np.frombuffer(data[len(header):],dtype='u1').reshape(1024,1024,3)
            png(folder/(name+'-'+str(az)+'.png'),np,image);images.append(image[::2,::2].copy())
            content=(folder/(name+'-'+str(az)+'.pgm')).read_bytes();prefix=b'P5\n1024 1024\n255\n'
            if not content.startswith(prefix):raise ValueError('Fixed silhouette dimensions')
            masks.append(np.frombuffer(content[len(prefix):],dtype='u1').reshape(1024,1024)>0)
        a,b=masks;union=int((a|b).sum());iou=float((a&b).sum()/union) if union else 0
        ba=boundary(np,a);bb=boundary(np,b);distance=max(boundary_distance(np,ba,bb),boundary_distance(np,bb,ba))
        passed=iou>=.995 and distance<=2;view_pass=view_pass and passed
        views.append(dict(azimuth_degrees=az,elevation_degrees=20,resolution=[1024,1024],silhouette_IoU=iou,boundary_Hausdorff_pixels=distance,PASS=passed))
    contact=np.empty((2048,2048,3),'u1')
    for i,image in enumerate(images):contact[(i//4)*512:(i//4+1)*512,(i%4)*512:(i%4+1)*512]=image
    png(folder/'matched-eight-view-contact.png',np,contact)
    html='<html><meta charset="utf-8"><title>AI-derived reconstruction review</title><body><h1>Raw AI source versus AI-derived reconstruction</h1><p>CPU, neutral two-sided flat shading. Every pair shares a source-frozen camera, scale and lighting. No texture.</p>'
    for view in views:
        az=view['azimuth_degrees'];html+=f'<h2>{az} degrees: source left, reconstruction right</h2><img width="48%" src="source-{az}.png"><img width="48%" src="candidate-{az}.png">'
    with (folder/'review.html').open('x') as out:out.write(html+'</body></html>')
    src=np.load(run.BASE/'input/source-analysis.npz',allow_pickle=False);D=float(src['diagonal']);original=v/(100/D)+src['center']
    with (folder/'AI-DERIVED-RECONSTRUCTION.ply').open('x') as out:
        out.write(f'ply\nformat ascii 1.0\ncomment AI-DERIVED RECONSTRUCTION; not raw neural geometry\nelement vertex {len(v)}\nproperty double x\nproperty double y\nproperty double z\nelement face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n')
        for row in original:out.write(' '.join(format(float(x),'.17g') for x in row)+'\n')
        for row in f:out.write('3 '+' '.join(map(str,row))+'\n')
    # Explicit finite feature survival, independently of silhouette and aggregate
    # distance counters: every original source patch must retain compatible
    # output faces and all its vertices in the actual finite output footprint.
    final=json.loads((run.BASE/'owned-session/refined-screen.json').read_text());planes=final['planes']
    norms=cross/np.linalg.norm(cross,axis=1)[:,None];groups=np.full(len(f),-1,'i4')
    original_tri=original[f]
    for plane in planes:
        nn=np.array(plane['normal']);matches=(abs(norms@nn)>=math.cos(math.radians(1e-5)))&(abs(original_tri@nn-plane['offset']).max(axis=1)<=1e-8*D)
        if np.any((groups>=0)&matches):raise ValueError('Ambiguous output group')
        groups[matches]=plane['id']
    from amd_mesh_aware_input import RAW
    source_v=np.load(RAW/'vertices.npy',allow_pickle=False);source_f=np.load(RAW/'faces.npy',allow_pickle=False)
    region=np.fromfile(run.BASE/'region-labels.bin',dtype='<i4');features=[]
    for patch in final['original_patch_survival']:
        group=patch['group'];indices=np.unique(source_f[region==patch['region']].ravel());points=source_v[indices].astype('f8')
        best=np.full(len(points),np.inf);matched=original_tri[groups==group]
        for triangle in matched:best=np.minimum(best,point_triangle_distances(np,points,triangle))
        maximum=float(best.max()/D) if len(matched) else None
        passed=maximum is not None and maximum<=.002
        features.append(dict(source_region=patch['region'],actual_output_group=group,output_triangles=len(matched),original_vertices=len(points),maximum_compatible_output_surface_distance_over_D=maximum,
            classification='dominant planar patch' if patch['region']<6 else 'bevel transition' if patch['region']<18 else 'corner patch' if patch['region']<26 else 'small residual patch',PASS=passed))
    features_pass=all(x['PASS'] for x in features) and np.all(groups>=0)
    # Independent brute-force numerical check of native AABB distances, spread
    # across the original area strata. No acceleration tree or native code reused.
    source_samples=np.load(run.PRIOR/'ai-input/samples.npz',allow_pickle=False)['points']
    indices=np.linspace(0,65535,256,dtype='i8');independent=np.full(len(indices),np.inf)
    for triangle in original_tri:independent=np.minimum(independent,point_triangle_distances(np,source_samples[indices],triangle)/D)
    error=float(abs(independent-np.fromfile(folder/'source-to-candidate.bin',dtype='<f8')[indices]).max())
    numerical_pass=error<=5e-12
    # Independent PLY ASCII readback of the original-unit deliverable.
    lines=(folder/'AI-DERIVED-RECONSTRUCTION.ply').read_text().splitlines();header_end=lines.index('end_header')+1
    read_v=np.array([[float(x) for x in line.split()] for line in lines[header_end:header_end+len(v)]])
    read_f=np.array([[int(x) for x in line.split()[1:]] for line in lines[header_end+len(v):]],dtype='i4')
    readback=np.array_equal(read_v,original) and np.array_equal(read_f,f)
    passed=topology and distance_pass and view_pass and features_pass and numerical_pass and readback and native['complete_finite_footprint_certified'] and domains['complete_compatible_domain_footprints'] and domains['certified_triangles']==len(f)
    save(run.BASE/'candidate-result.json',dict(classification='PASS_MEASURED_GATES_PENDING_VISUAL_REVIEW' if passed else 'NO_GO_CANDIDATE',label='AI-DERIVED RECONSTRUCTION',raw_neural_geometry_PASS=False,topology=counts,topology_PASS=bool(topology),zero_area_triangles=zero,duplicate_coordinate_vertices=duplicate_positions,complete_self_intersections=native['complete_self_intersections'],bidirectional_triangle_surface_distances_over_AI_diagonal=distances,views=views,views_PASS=bool(view_pass),features=features,features_PASS=bool(features_pass),independent_bruteforce_source_distance_samples=256,independent_bruteforce_distance_error_over_D=error,independent_bruteforce_PASS=numerical_pass,original_units_PLY_exact_readback=bool(readback),complete_finite_footprint_certified=native['complete_finite_footprint_certified'],complete_compatible_domain_footprints=domains['complete_compatible_domain_footprints'],footprint_queries=native['footprint_queries'],domain_footprint_queries=domains['queries'],full_source_vertex_distance_max_over_D=native['full_source_vertex_distance_max_over_D'],raw_candidate_sha256=sha(raw),triangulated_candidate_sha256=sha(folder/'triangulated-native.off'),original_units_PLY_sha256=sha(folder/'AI-DERIVED-RECONSTRUCTION.ply'),source_only=True,geometry_post_repair=False,visual_review_required=True))
