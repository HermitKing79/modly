"""One read-only A/D wrist feasibility spike; deliberately has no mesh-edit entrypoint."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent / 'artifacts/amd-zombie-ad-wrist-feasibility-20261025-v1'
A = ROOT.parent / 'artifacts/amd-zombie-cpu-mesh-20261025-v1'
D = ROOT.parent / 'artifacts/amd-zombie-d-comparison-9ac0364f-v1'
COMMIT = 'bc7160c7439a5d0ea9653e166620b6d535c498ce'
SHA = {'A': '5301f7dc3e22abf6bfd7c1253a4d6253935ad3054658e2eddbb6ac024e6fc12c',
       'D': '2f8a1f9057c88f588b171c66f75af88b98fa47f1cea7494d6ee55517ce9e8205'}
sys.path.insert(0, str(Path(__file__).parent))
import amd_zombie_cpu_mesh as cpu
import amd_zombie_d_compare as policy
import amd_zombie_topology as topology


def section(np, v, f, x):
    """Triangle/plane segment graph keyed by ORIGINAL crossed edges, winding agnostic.

    Only diagnostic intersection points are calculated. No triangles are split.
    Exact vertex/plane coincidences stop rather than guessing connectivity.
    """
    xyz = v[f]; signed = xyz[:, :, 0] - x
    selected = np.flatnonzero((signed.min(axis=1) < 0) & (signed.max(axis=1) > 0))
    if np.any(signed[selected] == 0):
        raise ValueError('Plane coincides with a vertex; ambiguous section')
    ids = f[selected]; s = signed[selected]
    a = np.concatenate((ids[:, 0], ids[:, 1], ids[:, 2]))
    b = np.concatenate((ids[:, 1], ids[:, 2], ids[:, 0]))
    crossing = np.concatenate((s[:, 0]*s[:, 1], s[:, 1]*s[:, 2], s[:, 2]*s[:, 0])) < 0
    face_rows = np.tile(np.arange(len(ids)), 3)[crossing]
    pairs = np.sort(np.column_stack((a[crossing], b[crossing])), axis=1)
    edges, inverse, degree = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(face_rows, kind='stable')
    if len(inverse) != 2*len(ids): raise ValueError('Two endpoints per triangle required')
    segments = inverse[order].reshape(-1, 2)
    p0 = v[edges[:, 0]].astype(np.float64); p1 = v[edges[:, 1]].astype(np.float64)
    alpha = (x-p0[:, 0])/(p1[:, 0]-p0[:, 0])
    points = p0+alpha[:, None]*(p1-p0)
    parents = list(range(len(points)))
    def root(i):
        while parents[i] != i: parents[i] = parents[parents[i]]; i = parents[i]
        return i
    for a, b in segments.tolist():
        a, b = root(a), root(b)
        if a != b: parents[b] = a
    labels = np.array([root(i) for i in range(len(points))])
    components = []
    for label in np.unique(labels):
        mask = labels == label
        components.append(dict(nodes=int(mask.sum()), degree_one=int((degree[mask] == 1).sum()),
                               degree_over_two=int((degree[mask] > 2).sum()),
                               simple_cycle=bool(np.all(degree[mask] == 2))))
    components.sort(key=lambda row: -row['nodes'])
    metric = dict(x=float(x), intersected_triangles=len(selected), nodes=len(points),
                  components=components, degree_one=int((degree == 1).sum()),
                  degree_over_two=int((degree > 2).sum()),
                  simple_single_loop=bool(len(components) == 1 and np.all(degree == 2)))
    if len(points):
        metric.update(centroid=points.mean(axis=0).tolist(),
                      yz_dimensions=np.ptp(points[:, 1:], axis=0).tolist(),
                      bounds=[points.min(axis=0).tolist(), points.max(axis=0).tolist()])
    return metric, points, segments


def nearest(np, a, b):
    distances=[]
    for start in range(0, len(a), 128):
        distances.append(np.sqrt(((a[start:start+128, None]-b[None])**2).sum(axis=2).min(axis=1)))
    d=np.concatenate(distances)
    return dict(min=float(d.min()), median=float(np.median(d)), p95=float(np.quantile(d, .95)), max=float(d.max()))


def cut_metrics(np, v, f, edges, incidence, side, plane):
    # Analysis-only whole-triangle partition: never clips or exports triangles.
    distal = (side*v[f, 0]).min(axis=1) > plane
    ids=np.flatnonzero(distal); hand=f[ids]
    he,hc,_,_=topology.incidence(np,hand)
    keys=edges[:,0].astype(np.int64)*len(v)+edges[:,1]
    locations=np.searchsorted(keys,he[:,0].astype(np.int64)*len(v)+he[:,1])
    if not np.array_equal(edges[locations],he): raise ValueError('Missing source incidence')
    global_count=incidence[locations]
    exposed=(hc==1)&(global_count>1)
    boundary=he[exposed]
    degree=np.bincount(boundary.ravel(),minlength=len(v))
    return dict(distal_whole_triangles=len(ids), preserved_body_triangles=len(f)-len(ids),
                inherited_distal_boundary_edges=int(((hc==1)&(global_count==1)).sum()),
                newly_exposed_distal_boundary_edges=int(exposed.sum()),
                cut_boundary_vertices=int((degree>0).sum()),
                cut_boundary_branch_vertices=int((degree>2).sum()),
                cut_boundary_open_end_vertices=int((degree==1).sum()),
                cut_edges_from_nonmanifold_source=int((exposed&(global_count>2)).sum()),
                distal_nonmanifold_edges=int((hc>2).sum()),
                geometry_exported_or_modified=False), boundary


def authenticate():
    import numpy as np
    import amd_flexible_dual_grid_cpu as fdg
    _,a_binding=cpu.authenticate()
    _,d_binding=policy.authenticate(COMMIT)
    policy.validate_mesh(cpu.read(D/'mesh/mesh.json'), d_binding, D)
    final=cpu.read(D/'comparison-final.json')
    if final['classification']!='PASS_PAIRED_EVIDENCE_VISUAL_UNASSESSED' or not final['original_files_unchanged']:
        raise ValueError('Successful unmodified D comparison required')
    result={}
    for label,base,binding in [('A',A,a_binding),('D',D,d_binding)]:
        meta=cpu.read(base/'mesh/mesh.json')
        if cpu.digest(base/'mesh/mesh.ply')!=SHA[label] or meta['input_sha256']!=binding['source_sha256']:
            raise ValueError('Immutable source identity differs')
        for name,row in meta['artifacts'].items():
            p=cpu.plain(base/'mesh'/name)
            if p.stat().st_size!=row['bytes'] or cpu.digest(p)!=row['sha256']:
                raise ValueError('Mesh artifact differs')
        vv,ff=fdg.read_ply(base/'mesh/mesh.ply')
        v=np.load(base/'mesh/vertices.npy',mmap_mode='r',allow_pickle=False)
        f=np.load(base/'mesh/faces.npy',mmap_mode='r',allow_pickle=False)
        if not np.array_equal(vv.view(np.uint32),v.view(np.uint32)) or not np.array_equal(ff,f):
            raise ValueError('PLY/NPY bit-exact agreement required')
        for arr in (vv,ff,v,f):arr._mmap.close()
        result[label]=dict(PLY_sha256=SHA[label],vertices=meta['vertices'],triangles=meta['triangles'],
                           decode=binding,PLY_NPY_exact=True)
    cpu.save(BASE/'authentication.json',result)


def analyze():
    import numpy as np
    authenticate()
    outputs={}; overlay={}; models={}
    planes=[.34013,.35013,.36013,.37013,.38013,.39013,.40013,.41013]
    for label,base in [('A',A),('D',D)]:
        v=np.load(base/'mesh/vertices.npy',mmap_mode='r',allow_pickle=False)
        f=np.load(base/'mesh/faces.npy',mmap_mode='r',allow_pickle=False)
        edges,counts,_,winding=topology.incidence(np,f)
        rows={}; cached={}
        for side in (-1,1):
            local=(side*v[edges].mean(axis=1)[:,0]>=.34)
            overlay[f'{label}_{side}_boundary']=v[edges[(counts==1)&local]]
            overlay[f'{label}_{side}_nonmanifold']=v[edges[(counts>2)&local]]
            records=[]
            for plane in planes:
                m,p,s=section(np,v,f,side*plane)
                records.append(m);cached[(side,plane)]=(m,p,s)
            # A fixed central wrist station for detailed source-only cut diagnosis.
            station=.38013
            cut,boundary=cut_metrics(np,v,f,edges,counts,side,station)
            overlay[f'{label}_{side}_cut']=v[boundary]
            m,p,s=cached[(side,station)]
            overlay[f'{label}_{side}_section']=p[s]
            wrist=v[(side*v[:,0]>=.36)&(side*v[:,0]<=.40)]
            centers=[]
            for center in (.355,.395):
                slab=v[np.abs(side*v[:,0]-center)<.004]
                centers.append(slab.mean(axis=0,dtype=np.float64))
            direction=centers[1]-centers[0];direction/=np.linalg.norm(direction)
            # This is a viewing ROI, not anatomical segmentation.
            hand_faces=(side*v[f,0]).min(axis=1)>=.34
            _,fl=topology.component_labels(np,f[hand_faces],len(v))
            rows[str(side)]=dict(section_scan=records,fixed_station=station,hypothetical_whole_triangle_cut=cut,
                                wrist_band_bounds=[wrist.min(axis=0).tolist(),wrist.max(axis=0).tolist()],
                                outward_forearm_axis=direction.tolist(),
                                distal_ROI_face_components=int(len(np.unique(fl))),
                                distal_ROI_triangles=int(hand_faces.sum()),
                                anatomical_segmentation=False)
        models[label]=dict(vertices=len(v),triangles=len(f),bounds=[v.min(axis=0).tolist(),v.max(axis=0).tolist()],
                           boundary_edges=int((counts==1).sum()),nonmanifold_edges=int((counts>2).sum()),
                           two_face_winding_conflicts=int(((counts==2)&(abs(winding)==2)).sum()),wrists=rows)
        outputs[label]=cached
        v._mmap.close();f._mmap.close()
    comparisons={}
    for side in (-1,1):
        results=[]
        for plane in planes:
            ma,a,_=outputs['A'][(side,plane)];md,d,_=outputs['D'][(side,plane)]
            delta=np.array(md['centroid'])-ma['centroid']
            results.append(dict(plane_abs_x=plane,centroid_D_minus_A=delta.tolist(),
                                D_over_A_yz_dimensions=(np.array(md['yz_dimensions'])/ma['yz_dimensions']).tolist(),
                                A_to_D_section_point_distance=nearest(np,a,d),
                                D_to_A_section_point_distance=nearest(np,d,a),
                                centroid_aligned_A_to_D_distance=nearest(np,a,d-delta),
                                both_single_simple_loops=ma['simple_single_loop'] and md['simple_single_loop']))
        axes=[np.array(models[k]['wrists'][str(side)]['outward_forearm_axis']) for k in ('A','D')]
        comparisons[str(side)]=dict(forearm_axis_angle_degrees=float(np.degrees(np.arccos(np.clip(axes[0]@axes[1],-1,1)))),stations=results)
    simple=sum(r['both_single_simple_loops'] for row in comparisons.values() for r in row['stations'])
    # Do not turn a positive exploratory result into an automatic mesh edit.
    decision='NO_GO_AMBIGUOUS_WRIST_SHEETS' if simple==0 else 'STOP_FOR_REVIEW_NO_AUTOMATIC_EDIT'
    result=dict(classification='PASS_READONLY_FEASIBILITY',decision=decision,models=models,comparisons=comparisons,
                coordinate_units='original normalized model coordinates; no physical-size assumption',
                planes=planes,clean_paired_sections=simple,hybrid_created=False,
                source_geometry_modified=False,triangles_clipped_or_filled=False,
                limitation='Cross-sections and whole-triangle cuts are diagnostic only. No self-intersection or semantic sheet classifier.',
                dark_pixels='Not a hole classifier. Cyan denotes measured one-face edges; magenta denotes >2-face edges. Winding conflicts can also affect shading.')
    with (BASE/'diagnostic-lines.npz').open('xb') as stream:np.savez(stream,**overlay)
    cpu.save(BASE/'analysis.json',result)
    print(json.dumps(dict(decision=decision,clean_paired_sections=simple,hybrid_created=False)),flush=True)


def seam_witness():
    """Independent dictionary incidence of the hypothetical cut, no mesh writes."""
    from collections import Counter
    import numpy as np
    records={}
    for label,base in [('A',A),('D',D)]:
        v=np.load(base/'mesh/vertices.npy',mmap_mode='r',allow_pickle=False)
        f=np.load(base/'mesh/faces.npy',mmap_mode='r',allow_pickle=False)
        records[label]={}
        for side in (-1,1):
            t=side*v[f,0]; distal=t.min(axis=1)>.38013
            nearby=t.max(axis=1)>=.38013
            def count(fs):
                c=Counter()
                for a,b,d in fs.tolist():
                    c.update((tuple(sorted((a,b))),tuple(sorted((b,d))),tuple(sorted((d,a)))))
                return c
            original=count(f[nearby]);removed=count(f[distal])
            body_edges=[e for e,n in removed.items() if original[e]>1 and original[e]-n==1]
            donor_edges=[e for e,n in removed.items() if original[e]>1 and n==1]
            def graph(edges):
                degree=Counter(i for e in edges for i in e)
                return dict(new_edges=len(edges),vertices=len(degree),degree_one=sum(d==1 for d in degree.values()),
                            branch_vertices=sum(d>2 for d in degree.values()),all_degree_two=bool(degree and all(d==2 for d in degree.values())))
            records[label][str(side)]=dict(retained_body=graph(body_edges),donor_hand=graph(donor_edges),
                inherited_donor_boundary_edges=sum(n==1 and original[e]==1 for e,n in removed.items()),
                direct_dictionary_incidence=True)
        v._mmap.close();f._mmap.close()
    unambiguous=all(records['A'][str(s)]['retained_body']['all_degree_two'] and records['D'][str(s)]['donor_hand']['all_degree_two'] for s in (-1,1))
    result=dict(decision='STOP_FOR_REVIEW' if unambiguous else 'NO_GO_CLEAN_COPY_ONLY_WRIST_REPLACEMENT',
                inspected_abs_x=.38013,independent_cut_incidence=records,hybrid_created=False,
                actual_new_seam_edges=0,all_source_geometry_unchanged=True,
                scope='Bounded evidence rejects a clean swap here; does not prove all future local reconstruction impossible.')
    cpu.save(BASE/'seam-witness.json',result)
    return result


def compose():
    from PIL import Image,ImageDraw,ImageFont
    seam=seam_witness()
    metadata=cpu.read(BASE/'renders/render-metadata.json')
    if not metadata['all_source_payloads_exact']:raise ValueError('Unmodified render payloads required')
    for row in metadata['views']:
        if cpu.digest(BASE/'renders'/row['name'])!=row['sha256']:raise ValueError('Render changed')
    out=BASE/'review';out.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    def sheet(name,kinds,size):
        image=Image.new('RGB',(size*2,(size+36)*len(kinds)+70),'#242830');draw=ImageDraw.Draw(image)
        draw.text((12,10),'A original (left) / D original (right): unchanged geometry',font=font,fill='white')
        draw.text((12,38),'Cyan: boundary; magenta: >2 faces; amber: diagnostic wrist section',font=font,fill='white')
        for i,kind in enumerate(kinds):
            for j,label in enumerate(('A','D')):
                p=BASE/'renders'/f'{label}-{kind}.png'
                with Image.open(p) as src: image.paste(src.resize((size,size),Image.Resampling.LANCZOS),(j*size,70+i*(size+36)))
                draw.text((j*size+12,70+i*(size+36)+size+5),f'{label} | {kind}',font=font,fill='white')
        image.save(out/name)
    sheet('body-front-back.png',['front','back'],640)
    for side in ('minus','plus'):
        sheet(f'hand-{side}.png',[f'hand-{side}-front',f'hand-{side}-back',f'hand-{side}-topology'],640)
    cpu.save(out/'review.json',dict(decision=seam['decision'],hybrid_created=False,
                                  images={p.name:cpu.digest(p) for p in out.glob('*.png')}))


def supervise(stage):
    owner=cpu.load('_ad_wrist_owned_cpu',Path(cpu.__file__))
    owner.BASE=BASE;owner.__file__=str(Path(__file__).resolve())
    launch=subprocess.Popen
    def routed(command,**kwargs):
        if stage=='render':
            expected=[str(cpu.BLENDER),'--background','--factory-startup','--disable-autoexec','--threads','4',
                      '--python-exit-code','1','--python',str(Path(__file__).with_name('amd_zombie_mesh_render.py'))]
            if command!=expected:raise ValueError('Changed inherited render route')
            command=command[:-1]+[str(Path(__file__).with_name('amd_zombie_ad_wrist_render.py'))]
        elif stage not in ('analyze','compose') or command!=[sys._base_executable,'-B','-I','-S',str(Path(__file__).resolve()),'--worker',stage]:
            raise ValueError('Changed inherited isolated worker route')
        return launch(command,**kwargs)
    owner.subprocess=SimpleNamespace(Popen=routed,PIPE=subprocess.PIPE)
    return owner.supervise(stage)


def worker(stage):
    if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Exact owned GO and EOF required')
    sys.addaudithook(policy.audit_policy(BASE))
    sys.path.append('D:/AI/Trellis2AMD/venv/Lib/site-packages')
    if stage=='analyze':analyze()
    elif stage=='compose':compose()
    else:raise ValueError('Unknown read-only stage')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('analyze','render','compose'))
    parser.add_argument('--worker',choices=('analyze','compose'))
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.stage:raise SystemExit(supervise(args.stage))
    else:print('DRY_RUN: read-only CPU wrist analysis and renders; no hybrid writer or generation entrypoint')
