"""Cheapest authenticated CPU capacity gate. No unwrapping or image selection."""
import sys
sys.dont_write_bytecode=True
import json,time
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from inputs import P,load_auth,load_selection,save,require,digest,array


def main():
    require(sys.stdin.buffer.readline(64)==b'GO\n','Owned binary GO')
    auth=load_auth();auth.P=P;sys.addaudithook(auth.runtime_guard)
    import numpy as np
    from chart_forest import forest
    began=time.monotonic();data,identities,watched=load_selection(np)
    before={str(path):digest(path) for path in watched}
    v,f,sf,sv=[data[n] for n in ('vertices','faces','source-face-ids','source-vertex-ids')]
    xyz=v[f].astype(np.float64)
    areas=np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)/2
    require((areas>0).all(),'No original degenerate face')
    # Both singular values >=rho imply area(J)>=rho². Count every source face.
    area=float(areas.sum());rho=4096
    result=forest(v,f,sf,sv)
    report=result['report'];out=P/'capacity/run'
    for name,key in [('seam-vertices','seam_vertices'),('seam-faces','seam_faces'),('seam-source-local-vertices','source_local_vertex_ids'),('source-corner-indices','source_corner_indices'),('retained-source-edges','retained_source_edges'),('face-chart-ids','face_chart_ids')]:
        array(out/(name+'.npy'),result[key])
    charts=report['chart_count']
    area_floor=area*rho**2
    optimistic=max(area_floor,83*32**2) # Allows atlas-edge/clamp freedom; no padded-policy claim.
    policy_floor=max(area_floor,charts*64**2)
    admitted=optimistic<=1024**2 and charts<=256 and policy_floor<=1024**2
    files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.glob('*.npy')}
    require(before=={str(path):digest(path) for path in watched},'Original inputs unchanged')
    save(out/'result.json',dict(classification='PASS' if admitted else 'BLOCKED',stage='capacity_only_no_unwrap',
        source_input_identities=identities,selection_faces=4531,selection_vertices=3177,
        actual_surface_area_world2=area,minimum_tangent_density=rho,unavoidable_surface_area_texels=area_floor,
        original_edge_components=83,original_nonmanifold_edges=219,original_artificial_boundaries=2308,
        unavoidable_source_connected_chart_floor=83,optimistic_area_lower_bound_texels=optimistic,
        combined_lower_bounds_use_max_not_sum=True,atlas_capacity_texels=1024**2,
        prior_dominant_axis_islands=1127,prior_rectangle_area_texels=6929408,
        candidate_chart_report=report,conditional_64x64_owner_lower_bound_texels=policy_floor,
        policy='Candidate retains32-aligned rectangles, content minimum32pxinside, dimensions32*(ceil(span/32+.5+1e-5)+1); min64 peraxis. This is conditional, not globally minimal mip packing.',
        lower_bound_does_not_guarantee_fit=True,files=files,elapsed_seconds=time.monotonic()-began))
    print('CAPACITY_'+('PASS' if admitted else 'BLOCKED'),flush=True)


if __name__=='__main__':main()
