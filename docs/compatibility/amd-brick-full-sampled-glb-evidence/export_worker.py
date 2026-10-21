"""One complete source-resampled colour table followed by one streamed GLB."""
import sys
sys.dont_write_bytecode=True
import json,os,time,traceback
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from inputs import P,CONVERTER,SAMPLER,original,pinned,load,load_auth,require,digest,plain,save,source_hashes

def main():
    require(sys.stdin.buffer.readline(64)==b'GO\n','Exact owned binary GO')
    auth=load_auth();auth.P=P;sys.addaudithook(auth.runtime_guard)
    import numpy as np
    from stream_writer import linear_rgb,write_glb
    from npy_stream import create,sync
    out=P/'export';began=time.monotonic();stage='input_authentication';completed=[];timings={}
    try:
        data,identities,watched=load(np);timings['input_authentication_seconds']=time.monotonic()-began
        before={str(p):digest(p) for p in watched};v,f,coords,field=data['vertices'],data['faces'],data['coordinates'],data['features']
        require(v.shape==(1957891,3) and f.shape==(4173584,3),'Complete authenticated original geometry')
        converter=pinned('full_sample_original_converter',CONVERTER,original.PINS[CONVERTER]);stage='vertex_reconstruction'
        for start in range(0,len(v),32768):
            stop=min(start+32768,len(v));dual,_,_=converter.transform_features(data['geometry_features'][start:stop])
            require(np.array_equal(converter.vertices(coords[start:stop,1:],dual).view(np.uint32),v[start:stop].view(np.uint32)),'Original vertex reconstruction bits')
        sampler_module=pinned('full_sample_original_sparse_sampler',SAMPLER,'0a330eca2c991d3b7751d3cbbbe94edac317b34b28ce7216d9d3d374ef20b0c6')
        stamp=time.monotonic();sampler=sampler_module.SparseSampler(coords,field);timings['sampler_index_seconds']=time.monotonic()-stamp
        colour_path=out/'linear-COLOR0.npy';count_path=out/'positive-support.npy';weight_path=out/'raw-weight-sum.npy'
        colours=create(colour_path,'<f4',(len(v),3));counts=create(count_path,'u1',(len(v),));weights=create(weight_path,'<f4',(len(v),))
        stage='full_vertex_resampling';stamp=time.monotonic();channel_min=np.full(6,np.inf);channel_max=-channel_min;missing_neighbours=0
        with (out/'sampling-progress.jsonl').open('x',encoding='utf-8',newline='\n') as log:
            for start in range(0,len(v),32768):
                stop=min(start+32768,len(v));sample=sampler.sample_world(v[start:stop])
                row=dict(batch=len(completed)+1,start=start,stop=stop,rows=stop-start,zero_support=int(sample.zero_support.sum()),first_zero_source_vertex_ids=(start+np.flatnonzero(sample.zero_support)[:16]).tolist(),min_positive_neighbours=int(sample.positive_weight_neighbour_count.min()),min_raw_weight_sum=float(sample.raw_weight_sum.min()),elapsed_seconds=time.monotonic()-began)
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
                require(not sample.zero_support.any() and np.isfinite(sample.values).all() and np.isfinite(sample.raw_weight_sum).all() and np.all(sample.raw_weight_sum>0),'Finite complete genuine source support; no repair/fallback')
                require(sample.values.shape==(stop-start,6) and sample.values.dtype==np.float32 and np.all((sample.neighbour_rows>=-1)&(sample.neighbour_rows<len(v))),'Six-channel samples and global support indices')
                linear=linear_rgb(sample.values[:,:3]);colours[start:stop]=linear;counts[start:stop]=sample.positive_weight_neighbour_count;weights[start:stop]=sample.raw_weight_sum
                require(np.array_equal(colours[start:stop].view(np.uint32),linear.view(np.uint32)),'Exact original-index colour assignment')
                channel_min=np.minimum(channel_min,sample.values.min(0).astype(np.float64));channel_max=np.maximum(channel_max,sample.values.max(0).astype(np.float64));missing_neighbours+=int((sample.found_neighbour_count<8).sum());completed.append(row)
        timings['sampling_and_staging_seconds']=time.monotonic()-stamp
        require(len(completed)==60 and completed[0]['start']==0 and completed[-1]['stop']==len(v) and completed[-1]['rows']==24579 and all(a['stop']==b['start'] for a,b in zip(completed,completed[1:])),'All60 complete original-order batches without gaps/overlaps')
        for path,view in ((colour_path,colours),(count_path,counts),(weight_path,weights)):sync(path,view)
        del colours,counts,weights,sampler
        stage='colour_artifact_readback';colours=np.load(plain(colour_path),mmap_mode='r',allow_pickle=False);counts=np.load(plain(count_path),mmap_mode='r',allow_pickle=False);weights=np.load(plain(weight_path),mmap_mode='r',allow_pickle=False)
        require(colours.dtype==np.float32 and colours.shape==v.shape and counts.dtype==np.uint8 and counts.shape==weights.shape==(len(v),) and weights.dtype==np.float32,'Complete reopened colour/support structures')
        for start in range(0,len(v),32768):
            require(np.isfinite(colours[start:start+32768]).all() and np.all((colours[start:start+32768]>=0)&(colours[start:start+32768]<=1)) and np.all((counts[start:start+32768]>=1)&(counts[start:start+32768]<=8)) and np.isfinite(weights[start:start+32768]).all() and np.all(weights[start:start+32768]>0),'Every reopened source-supported finite colour row')
        old=data['resampled'];require(np.array_equal(colours[old['unique-original-vertex-ids']].view(np.uint32),old['subset_linear_COLOR0'].view(np.uint32)),'Full sampled colours reproduce every previously qualified subset bit')
        colour_sha=digest(colour_path);source=source_hashes(identities,colour_sha)
        save(out/'colour-ready.json',dict(classification='PASS',vertices=len(v),F32_linear_COLOR0_SHA256=colour_sha,sampling_batches=len(completed),all_rows_supported=True,qualified_subset_bits_exact=True,source_hashes=source))
        stage='single_GLB_export';stamp=time.monotonic();path=out/'brick-source-sampled-colour.glb';container=write_glb(path,v,f,colours,source);timings['GLB_export_seconds']=time.monotonic()-stamp
        require(before=={str(p):digest(p) for p in watched},'Every original source and prior proof unchanged')
        files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in (colour_path,count_path,weight_path,path,out/'colour-ready.json',out/'sampling-progress.jsonl')}
        save(out/'result.json',dict(classification='PASS',scope='Complete CPU sampling/export; independent full-file verification still required',source_vertices=len(v),source_triangles=len(f),files=files,container=container,source_hashes=source,original_input_identities=identities,sampling_batches=completed,source_sample_calls=len(completed),source_rows_sampled=len(v),zero_support=0,minimum_positive_support=int(counts.min()),minimum_raw_weight_sum=float(weights.min()),rows_with_missing_neighbours=missing_neighbours,raw_six_channel_min=channel_min.tolist(),raw_six_channel_max=channel_max.tolist(),all_original_reconstruction_bits_exact=True,qualified_subset_F32_bits_exact=True,timings=timings,inputs_unchanged=True,no_Blender_model_GPU_UV_bake_or_geometry_change=True,elapsed_seconds=time.monotonic()-began))
        print('FULL_SOURCE_SAMPLED_GLB_EXPORT_PASS',flush=True)
    except BaseException as error:
        if not (out/'result.json').exists():save(out/'result.json',dict(classification='NO-GO',stage=stage,error=str(error),traceback=traceback.format_exc(),completed_batches=completed,elapsed_seconds=time.monotonic()-began,retry=False))
        raise

if __name__=='__main__':main()
