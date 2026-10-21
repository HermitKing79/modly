"""Independent full-index readback and bounded conservative quality evidence."""
import sys
sys.dont_write_bytecode=True
import json,time,traceback,gc
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from inputs import *

def main():
    begin();import numpy as np
    import reduction_contract as c
    from projection_metrics import project,raster,compare
    started=time.monotonic();out=P/'assess';stage='authentication';failures=[]
    result=dict(classification='RUNNING',quality_failures=failures,projections={},colour_regions={})
    def checkpoint():save(out/('progress-%02d.json'%len(list(out.glob('progress-*.json')))),dict(stage=stage,elapsed_seconds=time.monotonic()-started,failures=list(failures)))
    try:
        baseline=verify_stage('baseline');reduced=verify_stage('reduce');require(reduced['plan_SHA256']==baseline['private_plan_SHA256'],'Exact authenticated baseline plan used by the sole candidate');data=load(np)
        v,f=data['vertices'],data['faces'];labels=data['labels'];N=len(v)
        native=pinned('half_native_decode',OLD/'quality.py',PINS[OLD/'quality.py'])
        cf=np.fromfile(P/'reduce/original-index-faces.bin',dtype='<i4').reshape(-1,3)
        require(len(cf)==reduced['actual_triangles'] and reduced['candidate_calls']==1,'Exact single actual index output')
        geometry=c.scan(v,cf);ids,compact=c.compact(cf,N);cv=np.asarray(v[ids],dtype='<f4');rgba=np.asarray(data['native_bytes'][ids],dtype=np.uint8)
        require(np.array_equal(cv.view(np.uint32),v[ids].view(np.uint32)) and np.array_equal(ids[compact],cf),'Exact original position bits and triangle source-ID correspondence')
        require(rgba.shape==(len(ids),4) and (rgba[:,3]==255).all(),'Exact original OPAQUE native colour transfer')
        result.update(target_triangles=c.TARGET_TRIANGLES,actual_triangles=len(cf),referenced_vertices=len(ids),
                      triangle_reduction=1-len(cf)/len(f),vertex_reduction=1-len(ids)/1952142,
                      retained_position_and_colour_source_ids_exact=True,geometry=geometry)
        if result['triangle_reduction']<c.GATES['minimum_triangle_reduction']:failures.append('Less than20% meaningful triangle reduction')
        stage='orientation_and_component_lineage';checkpoint()
        expected=np.load(P/'baseline/expected-components.npy',allow_pickle=False)
        lineage=c.component_lineage(cf,labels,expected);result['component_lineage']=lineage
        if not lineage['all_components_retained']:failures.append('Original face-bearing component loss')
        old_keys=c.oriented_keys(f,N);old_keys.sort();new_keys=c.oriented_keys(cf,N);new_keys.sort()
        duplicate_count=int(np.count_nonzero(new_keys[1:]==new_keys[:-1]));result['duplicate_oriented_triangles']=duplicate_count
        if duplicate_count:failures.append('New duplicate oriented triangles')
        reverse=c.oriented_keys(cf[:,[0,2,1]],N)
        def member(query,source):
            at=np.searchsorted(source,query);return (at<len(source))&(source[np.minimum(at,len(source)-1)]==query)
        new_unsorted=c.oriented_keys(cf,N);inherited_reversed=int(np.count_nonzero(member(reverse,old_keys)&~member(new_unsorted,old_keys)))
        result['reversed_preexisting_triangles']=inherited_reversed
        if inherited_reversed:failures.append('Reversal of a pre-existing oriented triangle')
        result['unchanged_oriented_triangles']=int(member(new_unsorted,old_keys).sum())
        result['new_oriented_triangles']=len(cf)-result['unchanged_oriented_triangles']
        del old_keys,new_keys,new_unsorted,reverse;gc.collect()
        stage='complete_candidate_topology';checkpoint()
        topology=pinned('half_topology',ROOT/'docs/compatibility/amd-brick-defect-diagnosis-evidence/topology.py','9d6e29e83b35c60f1f9b49192a290069bf867b51493da3cc5ec017525017d561')
        top=topology.analyze(cv,compact,out/'cache');result['topology']=top
        if top['components']['face_components']!=207:failures.append('Changed number of original face-bearing components')
        original_keys=np.load(CACHE/'edge_keys.npy',mmap_mode='r',allow_pickle=False);original_inc=np.load(CACHE/'incidence.npy',mmap_mode='r',allow_pickle=False)
        new_edge=np.load(out/'cache/edge_keys.npy',mmap_mode='r',allow_pickle=False);new_inc=np.load(out/'cache/incidence.npy',mmap_mode='r',allow_pickle=False)
        def length(keys,vertices,n):
            total=0.
            for s in range(0,len(keys),32768):
                k=keys[s:s+32768];total+=float(np.linalg.norm(vertices[k//n].astype(np.float64)-vertices[k%n],axis=1).sum())
            return total
        before_length=length(original_keys[original_inc==1],v,N);after_length=length(new_edge[new_inc==1],cv,len(cv))
        result['boundary_length']={'before':before_length,'after':after_length,'ratio':after_length/before_length}
        if after_length>before_length*1.05:failures.append('Boundary length grew more than5%')
        original_bad=original_keys[original_inc>2];nb=new_edge[new_inc>2]
        low=ids[nb//len(cv)].astype(np.uint64);high=ids[nb%len(cv)].astype(np.uint64)
        global_bad=np.minimum(low,high)*np.uint64(N)+np.maximum(low,high)
        new_bad=int(np.count_nonzero(~member(global_bad,original_bad)));result['new_nonmanifold_edges']=new_bad
        if new_bad or top['edges']['nonmanifold_incidence_gt_2']>387068:failures.append('New non-manifold edges')
        stage='sampled_conservative_geometry';checkpoint()
        diagonal=float(np.linalg.norm(v.max(0).astype(np.float64)-v.min(0)))
        query=np.load(P/'baseline/query-world.npy',allow_pickle=False);cohorts=json.loads((P/'baseline/cohorts.json').read_text())
        leaf_centroids=v[f[data['foliage']]].mean(1,dtype=np.float64).astype(np.float32)
        first_faces=np.full(int(labels.max())+1,len(f),np.int32)
        for s in range(0,len(f),32768):np.minimum.at(first_faces,labels[f[s:s+32768,0]],np.arange(s,min(s+32768,len(f)),dtype=np.int32))
        component_points=v[f[first_faces[expected]]].mean(1,dtype=np.float64).astype(np.float32)
        q=np.concatenate((query,leaf_centroids,component_points));dist,_=c.upper_bounds(q,cv);dist/=diagonal
        measures={}
        for name,row in cohorts.items():measures[name]=c.stats(dist[row['query_start']:row['query_stop']])
        measures['all4531_foliage_centroids']=c.stats(dist[len(query):len(query)+len(leaf_centroids)])
        measures['all207_component_representatives']=c.stats(dist[-len(component_points):])
        result['geometry_upper_bounds']=measures
        for name,row in measures.items():
            if not c.gate(row,c.GATES['geometry_diagonal']):failures.append('Conservative sampled distance gate: '+name)
        stage='candidate_source_spatial_colour';checkpoint()
        linear=native.decode(rgba[:,:3]);all_points=[];all_faces=[];all_bary=[];ranges={};areas=np.empty(len(cf),np.float64)
        for s in range(0,len(cf),32768):
            p=v[cf[s:s+32768]].astype(np.float64);areas[s:s+len(p)]=.5*np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
        for i,(name,pool) in enumerate(data['selected'].items()):
            present=np.zeros(N,np.bool_);present[f[pool].ravel()]=True
            eligible=np.flatnonzero(present[cf].any(1));require(len(eligible)>=128,'Underfilled original region: '+name)
            rng=np.random.default_rng(np.random.SeedSequence([20261021,i]));chosen=np.sort(rng.choice(eligible,128,replace=False,p=areas[eligible]/areas[eligible].sum()))
            uv=rng.random((128,2));root=np.sqrt(uv[:,0]);bary=np.column_stack((1-root,root*(1-uv[:,1]),root*uv[:,1]))
            points=np.einsum('ni,nij->nj',bary,v[cf[chosen]].astype(np.float64)).astype(np.float32)
            ranges[name]=[len(all_points)*128,(len(all_points)+1)*128];all_points.append(points);all_faces.append(chosen.astype(np.int32));all_bary.append(bary)
        points=np.concatenate(all_points);chosen=np.concatenate(all_faces);bary=np.concatenate(all_bary)
        sampler_module=pinned('half_genuine_spatial_sampler',ROOT/'docs/compatibility/amd-brick-uv-bake-evidence/sampler.py','0a330eca2c991d3b7751d3cbbbe94edac317b34b28ce7216d9d3d374ef20b0c6')
        sampler=sampler_module.SparseSampler(data['coordinates'],data['features'],(512,512,512));samples=sampler.sample_world(points)
        # Source IDs, native byte decoding and linear interpolation are explicit.
        predicted=c.linear_to_encoded(np.einsum('ni,nij->nj',bary,linear[compact[chosen]].astype(np.float64)))
        factors=np.array([.00011082693761409814,.999995986777544,1.])
        target=np.clip(samples.values.astype(np.float64),0,1)
        for name,(a,b) in ranges.items():
            rgb=c.stats(predicted[a:b]-target[a:b,:3]);scalar=[c.stats(factors[j]-target[a:b,j+3]) for j in range(3)]
            good=c.gate(rgb,c.GATES['RGB']) and all(x['rmse']<=.02 for x in scalar) and not samples.zero_support[a:b].any()
            result['colour_regions'][name]=dict(samples=b-a,RGB=rgb,scalars=dict(zip(('metallic','roughness','alpha'),scalar)),zero_support=int(samples.zero_support[a:b].sum()),pass_gate=bool(good))
            if not good:failures.append('Genuine candidate spatial material gate: '+name)
        reverse_faces,reverse_bary,reverse_world=c.reverse_samples(v,cf,areas)
        original_used,_=c.compact(f,N);back,_=c.upper_bounds(reverse_world,v[original_used]);row=c.stats(back/diagonal);result['reverse_geometry_upper_bound']=row
        if not c.gate(row,c.GATES['geometry_diagonal']):failures.append('Reverse conservative sampled distance gate')
        for name,a in [('reverse-faces',reverse_faces),('reverse-barycentric',reverse_bary),('reverse-world',reverse_world),('reverse-upper-distances',back)]:write_array(out/(name+'.npy'),a)
        for name,a in [('query-world',points),('query-faces',chosen),('query-barycentric',bary),('query-source',samples.values),('query-predicted',predicted)]:write_array(out/(name+'.npy'),a)
        del sampler,samples,present,areas,data['prior'],data['resampled'];gc.collect()
        stage='frozen_four_view_projection';checkpoint()
        cameras=json.loads((P/'baseline/views.json').read_text())
        for name,camera in cameras.items():
            projection=project(v,camera);src=raster(v,f,projection,data['foliage']);del projection;gc.collect()
            projection=project(cv,camera);dst=raster(cv,compact,projection);del projection;gc.collect()
            metric=compare(src,dst,diagonal);metric['source_events']=src['pixel_triangle_events'];metric['candidate_events']=dst['pixel_triangle_events'];result['projections'][name]=metric
            for label,value in [('source-mask',src['mask']),('candidate-mask',dst['mask']),('source-depth',src['depth']),('candidate-depth',dst['depth']),('foliage-visible',src['visible_foliage'])]:write_array(out/(name+'-'+label+'.npy'),value)
            if not metric['pass']:failures.append('Frozen projection gate: '+name)
            del src,dst;gc.collect();print('PROJECTION_COMPLETE '+name,flush=True)
        require(any(x['foliage_informative'] for x in result['projections'].values()),'At least one informative original-foliage view required')
        stage='candidate_persistence';checkpoint()
        for name,a in [('vertices',cv),('faces',compact),('source-vertex-ids',ids),('color-bytes',rgba),('linear-colours',linear)]:write_array(out/(name+'.npy'),a)
        # Binary full-support derived PLY, preserving the exact candidate rows.
        header=('ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face %d\nproperty list uchar int vertex_indices\nend_header\n'%(len(cv),len(compact))).encode('ascii')
        ply=out/'candidate.ply'
        with ply.open('xb') as stream:
            stream.write(header)
            for s in range(0,len(cv),32768):
                block=np.empty(min(32768,len(cv)-s),dtype=[('xyz','<f4',3),('rgb','u1',3)]);block['xyz']=cv[s:s+len(block)];block['rgb']=rgba[s:s+len(block),:3];stream.write(block.tobytes())
            for s in range(0,len(compact),32768):
                block=np.empty(min(32768,len(compact)-s),dtype=[('count','u1'),('indices','<i4',3)]);block['count']=3;block['indices']=compact[s:s+len(block)];stream.write(block.tobytes())
            stream.flush();__import__('os').fsync(stream.fileno())
        # Independent mapped payload re-open, no importer or mesh repair.
        pv=np.memmap(ply,mode='r',offset=len(header),shape=len(cv),dtype=[('xyz','<f4',3),('rgb','u1',3)])
        pf=np.memmap(ply,mode='r',offset=len(header)+len(cv)*15,shape=len(compact),dtype=[('count','u1'),('indices','<i4',3)])
        require(ply.stat().st_size==len(header)+len(cv)*15+len(compact)*13 and np.array_equal(pv['xyz'].copy().view(np.uint32),cv.view(np.uint32)) and np.array_equal(pv['rgb'],rgba[:,:3]) and (pf['count']==3).all() and np.array_equal(pf['indices'],compact),'Complete PLY geometry/colour payload readback')
        # A failed conservative upper bound alone does not demonstrate damage.
        demonstrated=[x for x in failures if not x.startswith(('Conservative sampled distance gate:','Reverse conservative sampled distance gate'))]
        verdict='NO-GO' if demonstrated else 'BLOCKED' if failures else 'PASS'
        result.update(classification=verdict,demonstrated_quality_failures=demonstrated,files=manifest(out,[x.name for x in out.iterdir() if x.suffix in ('.npy','.ply')]),
                      estimated_compact_COLOR0_GLB_bytes=len(cv)*24+len(compact)*12+4096,elapsed_seconds=time.monotonic()-started,
                      candidate_source_colour_bytes_unchanged=True,projection_and_sampling_not_Hausdorff_or_all_leaf_guarantees=True)
        save(out/'result.json',result);print('HALF_POLY_ASSESSMENT '+result['classification'],flush=True)
    except BaseException as error:
        result.update(classification='BLOCKED',stage=stage,error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        save(out/'result.json',result);print('HALF_POLY_ASSESSMENT_BLOCKED '+str(error),flush=True)

if __name__=='__main__':main()
