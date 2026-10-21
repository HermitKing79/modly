"""One no-UV numerical decision gate, no bake/export/renderer/model execution."""
import sys
sys.dont_write_bytecode=True
import hashlib,time,json
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from admission import ROOT,P,C,load_auth,load_selection,digest,save,require,array,pinned


def seam_queries(faces,source_vertex_ids,retained):
    """All original shared-edge face pairs cut by the saved disk candidate.

    No island-ID permission is widened. This diagnostic has no UVs/owners and
    preserves every pair, including nonmanifold multiplicity. 65536 is the
    unchanged sampling batch ceiling, not an island-count reinterpretation.
    """
    import numpy as np
    require(isinstance(faces,np.ndarray) and faces.dtype==np.int32 and faces.ndim==2 and faces.shape[1]==3 and 0<len(faces)<=6000,'Bounded I32 face schema')
    require(isinstance(source_vertex_ids,np.ndarray) and source_vertex_ids.dtype==np.int64 and source_vertex_ids.ndim==1 and len(np.unique(source_vertex_ids))==len(source_vertex_ids),'Unique I64 original vertex IDs')
    require(faces.min()>=0 and faces.max()<len(source_vertex_ids) and source_vertex_ids.min()>=0 and source_vertex_ids.max()<1957891,'Original index bounds')
    require(isinstance(retained,np.ndarray) and retained.dtype==np.int64 and retained.ndim==2 and retained.shape[1]==2 and len(retained)<=18000,'Bounded retained-edge schema')
    require(not len(retained) or (retained[:,0]<retained[:,1]).all(),'Canonical original edge endpoints')
    edges={};retained={tuple(map(int,row)) for row in retained}
    for face_id,triangle in enumerate(source_vertex_ids[faces]):
        for a in range(3):
            b,third=(a+1)%3,(a+2)%3
            if triangle[a]>triangle[b]:a,b=b,a
            key=(int(triangle[a]),int(triangle[b]))
            edges.setdefault(key,[]).append((face_id,a,b,third))
    require(retained.issubset(edges),'Foreign retained source edge')
    require(all(len(edges[key])==2 for key in retained),'Retained edges must be original incidence-two pairs')
    query_faces=[];bary=[];cohort=[];pairs=[]
    for key,members in edges.items():
        if key in retained:continue
        for i,first in enumerate(members):
            for second in members[i+1:]:
                require(len(query_faces)+20<=65536,'All seam queries must fit; no pairs dropped')
                for group in (0,1):
                    for t in (0.,.25,.5,.75,1.):
                        at=t if group==0 else min(.995,max(.005,t));inward=0. if group==0 else .005
                        start=len(query_faces)
                        for face_id,a,b,third in (first,second):
                            weights=np.zeros(3,np.float64)
                            weights[a]=(1-inward)*(1-at);weights[b]=(1-inward)*at;weights[third]=inward
                            query_faces.append(face_id);bary.append(weights);cohort.append(group)
                        if group==0:pairs.append((start,start+1))
    return np.asarray(query_faces,np.int32),np.asarray(bary,np.float64).reshape(-1,3),np.asarray(cohort,np.int8),np.asarray(pairs,np.int32).reshape(-1,2)


def main():
    require(sys.stdin.buffer.readline(64)==b'GO\n','Owned binary GO')
    auth=load_auth();auth.P=P;sys.addaudithook(auth.runtime_guard)
    import numpy as np
    from metrics import color_prediction,evaluate
    began=time.monotonic();data,identities,watched=load_selection(np)
    before={str(path):digest(path) for path in watched}
    v,f,sv,direct=[data[n] for n in ('vertices','faces','source-vertex-ids','direct')]
    sampler_source=ROOT/'docs/compatibility/amd-brick-uv-bake-evidence/sampler.py'
    sampler_module=pinned('decision_original_spatial_sampler',sampler_source,'0a330eca2c991d3b7751d3cbbbe94edac317b34b28ce7216d9d3d374ef20b0c6')
    common=pinned('decision_original_tensor_reader',ROOT/'docs/compatibility/amd-brick-seam-uv-bake-evidence/common.py','e239c7e031bfa5a072d0c02b4804d96b73ed8ca7dc55145be46882ca0b02e96b')
    coordinates=common.tensor(auth.PINS['material'][0],'coordinates',np)
    features=auth.material_features(auth.PINS['material'][0],np)
    sampler=sampler_module.SparseSampler(coordinates,features)
    xyz=v[f].astype(np.float64)
    areas=np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)/2
    require((areas>0).all(),'Original areas must remain positive')
    rng=np.random.Generator(np.random.PCG64(20261021))
    random=rng.random((10000,2));root=np.sqrt(random[:,0])
    wb=np.column_stack((1-root,root*(1-random[:,1]),root*random[:,1]))
    chosen=rng.choice(len(f),10000,p=areas/areas.sum()).astype(np.int32)
    qf=np.concatenate((np.arange(len(f),dtype=np.int32),chosen))
    weights=np.vstack((np.full((len(f),3),1/3),wb))
    retained=np.load(C/'retained-source-edges.npy',allow_pickle=False)
    sf,sb,sc,pairs=seam_queries(f,sv,retained)
    require(len(sf)==44420 and len(pairs)==11105,'Every saved-candidate cut-edge pair included')
    surface=len(qf);qf=np.concatenate((qf,sf));weights=np.vstack((weights,sb));pairs+=surface
    require(len(qf)==58951 and len(qf)<=65536,'Fixed complete query population')
    world=(xyz[qf]*weights[:,:,None]).sum(1).astype(np.float32)
    reference=sampler.sample_world(world)
    require(np.isfinite(reference.values).all(),'Finite genuine source-spatial samples')
    predicted,linear=color_prediction(direct[:,:3],f,qf,weights)
    cohorts=dict(centroids=np.arange(len(f)),area_weighted=np.arange(len(f),surface),
        required_surface=np.arange(surface),seam_edge=np.flatnonzero(sc==0)+surface,
        seam_near=np.flatnonzero(sc==1)+surface)
    report=evaluate(predicted,reference.values,cohorts,cohorts['area_weighted'])
    report['seam_pair_world_max']=float(np.max(abs(world[pairs[:,0]].astype(float)-world[pairs[:,1]])))
    report['seam_pair_reference_max']=float(np.max(abs(reference.values[pairs[:,0]].astype(float)-reference.values[pairs[:,1]])))
    report['seam_pair_vertex_RGB_max']=float(np.max(abs(predicted[pairs[:,0]]-predicted[pairs[:,1]])))
    require(report['seam_pair_world_max']<=1e-7,'Paired source edge positions agree')
    report['opaque_alpha_RMSE_by_cohort']={name:float(np.sqrt(np.mean((1-np.clip(reference.values[index,5].astype(float),0,1))**2))) for name,index in cohorts.items()}
    report['material_behaviour_limits']=['OPAQUE ignores stored alpha; these errors compare actual opacity1 with learned alpha.','A constant alpha/BLEND factor cannot reproduce spatial cutouts or translucency variations. No alphaMode or transparency renderer has been qualified.','Standard COLOR_0 carries RGB/optional alpha, not spatial metallic or roughness. No invented emissive, AO or normal channel.']
    from metrics import channel_ranges
    report['entire_material_field_ranges']=channel_ranges(features)
    report['original_selected_vertex_ranges']=channel_ranges(direct)
    out=P/'run'
    arrays={'query-faces':qf,'barycentrics':weights,'world':world,'source-spatial':reference.values,
        'vertex-RGB':predicted,'linear-COLOR0':linear,'seam-pairs':pairs,'positive-neighbour-count':reference.positive_weight_neighbour_count,
        'raw-weight-sum':reference.raw_weight_sum}
    for name,value in arrays.items():array(out/(name+'.npy'),value)
    require(before=={str(path):digest(path) for path in watched},'All original input identities unchanged')
    files={path.name:dict(bytes=path.stat().st_size,sha256=digest(path)) for path in out.glob('*.npy')}
    reference_qualified=not bool(reference.zero_support.any())
    qualified=reference_qualified and report['RGB_all_pass'] and report['scalars_all_pass']
    save(out/'result.json',dict(classification='PASS_DECISION' if reference_qualified else 'BLOCKED_REFERENCE_SUPPORT',vertex_colour_candidate=('PASS' if qualified else 'FAIL_QUALITY_GATES') if reference_qualified else 'UNQUALIFIED_REFERENCE',
        recommendation=('SEPARATE_VERTEX_COLOUR_PORTABLE_PROOF' if qualified else 'SEPARATE_DISPOSABLE_TARGET_MESH_PROOF') if reference_qualified else 'READ_ONLY_REFERENCE_SUPPORT_DIAGNOSIS',
        seed=20261021,bit_generator='PCG64',sampling_order='10000 uniform pairs for sqrt barycentrics, then 10000 area-weighted face choices',
        original_faces=4531,original_vertices=3177,query_counts={k:len(index) for k,index in cohorts.items()},
        total_queries=len(qf),paired_edge_queries=len(pairs),zero_support=int(reference.zero_support.sum()),
        minimum_positive_neighbours=int(reference.positive_weight_neighbour_count.min()),minimum_weight_sum=float(reference.raw_weight_sum.min()),
        measurements=report,files=files,source_input_identities=identities,inputs_unchanged=True,
        colour_contract='Original encoded field sampled before clamp; derived glTF COLOR_0 exactIEC sRGBdecode thenF32storage, linear-light barycentric interpolation, exactIECencode for encodedRGB errors. No native2.2gamma or bytequantisation.',
        no_geometry_material_or_UV_modification=True,no_bake_or_export=True,elapsed_seconds=time.monotonic()-began))
    print('EXPORT_DECISION_'+('VERTEX_COLOUR_CANDIDATE_PASS' if qualified else 'VERTEX_COLOUR_CANDIDATE_FAIL'),flush=True)


if __name__=='__main__':main()
