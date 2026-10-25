"""Owned Blender CPU capability probe, two-crop experiments and matched renders."""
import sys
if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO/EOF required before Blender imports')
import array,hashlib,json,math,os,time
from pathlib import Path
import bpy
from mathutils import Matrix,Vector
from mathutils.bvhtree import BVHTree
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT.parent/'artifacts/amd-zombie-retopology-feasibility-20261025-v1'
sys.path.insert(0,str(Path(__file__).parent))
import amd_zombie_retopology as api
import amd_flexible_dual_grid_cpu as fdg
MODE=os.environ.get('MODLY_ZOMBIE_RETOPOLOGY_MODE')
if MODE not in ('capabilities','build','render'):raise ValueError('Explicit bounded CPU mode required')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,row):
    with p.open('x') as stream:json.dump(row,stream,indent=2,allow_nan=False)
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.filepaths.temporary_directory=str(BASE)
def mesh_obj(name,v,f):
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(v.tolist(),[],f.tolist());mesh.update()
    obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj);return obj
def payload(obj,v,f):
    co=array.array('f',[0])*(len(v)*3);idx=array.array('i',[0])*(len(f)*3)
    obj.data.vertices.foreach_get('co',co);obj.data.loops.foreach_get('vertex_index',idx)
    assert co.tobytes()==v.tobytes() and idx.tobytes()==f.tobytes() and not obj.modifiers
def arrays(mesh):
    mesh.calc_loop_triangles();v=np.empty(len(mesh.vertices)*3,np.float32);f=np.empty(len(mesh.loop_triangles)*3,np.int32)
    mesh.vertices.foreach_get('co',v);mesh.loop_triangles.foreach_get('vertices',f);return v.reshape(-1,3),f.reshape(-1,3)
def bvh(v,f):return BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True,epsilon=0.) if len(f) else None
def summary(values):
    a=np.asarray(values,np.float64)
    return dict(samples=len(a),median=float(np.median(a)),p95=float(np.quantile(a,.95)),maximum=float(a.max())) if len(a) else dict(samples=0)


if MODE=='capabilities':
    mesh=bpy.data.meshes.new('Capability probe only');obj=bpy.data.objects.new('Capability probe only',mesh);bpy.context.collection.objects.link(obj)
    remesh=obj.modifiers.new('RNA inspection','REMESH');props=remesh.bl_rna.properties
    row=dict(blender_version=bpy.app.version_string,blender_build_hash=bpy.app.build_hash.decode(),
        build_options={k:getattr(bpy.app.build_options,k) for k in ('openvdb','quadriflow') if hasattr(bpy.app.build_options,k)},
        remesh_modes=[item.identifier for item in props['mode'].enum_items],
        remesh_properties=list(props.keys()),quadriflow_operator_properties=list(bpy.ops.object.quadriflow_remesh.get_rna_type().properties.keys()),
        voxel_operator_properties=list(bpy.ops.object.voxel_remesh.get_rna_type().properties.keys()),
        decimate_available=True,BVH_two_sided_raycast_available=True,CPU_only=True,asset_imported=False,
        selected_methods=['native voxel control','view-supported chart with explicit discontinuity seams'],
        deferred_methods={'QuadriFlow':'Does not clean intersecting/nonmanifold source sheets; requires a trustworthy sheet before quadrangulation',
            'decimation alone':'Reduces triangles while retaining or concealing source sheet/incidence defects',
            'manual feature-led retopology':'Stronger control, but beyond this cheap automatic feasibility test; deformation loops still require design'})
    save(BASE/'capabilities.json',row);print(json.dumps(row),flush=True)


def candidate_files(folder,v,f):
    folder.mkdir(exist_ok=False)
    if not len(f):return
    fdg.write_ply(folder/'mesh.ply',v,f)
    api.prior.save_npy(folder/'vertices.npy',v);api.prior.save_npy(folder/'faces.npy',f)
    save(folder/'manifest.json',dict(artifacts={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in folder.iterdir()}))


if MODE=='build':
    caps=read(BASE/'capabilities.json');settings=read(BASE/'experiment-settings.json')
    assert caps['build_options'].get('openvdb') is True and 'VOXEL' in caps['remesh_modes']
    assert settings['baseline_PLY_sha256']==api.BASELINE_SHA
    output=BASE/'candidates';output.mkdir(exist_ok=False);records={}
    for name,r in settings['regions'].items():
        ref=BASE/'reference'/name;assert sha(ref/'mesh.ply')==r['reference_PLY_sha256']
        v=np.load(ref/'vertices.npy',allow_pickle=False);f=np.load(ref/'faces.npy',allow_pickle=False);before=v.tobytes(),f.tobytes()
        crop=mesh_obj(name+' immutable crop',v,f);payload(crop,v,f);tree=bvh(v,f)
        folder=output/name;folder.mkdir(exist_ok=False);records[name]={}
        began=time.perf_counter()
        test=mesh_obj(name+' voxel copy',v,f);bpy.context.view_layer.objects.active=test;test.select_set(True)
        mod=test.modifiers.new('Fixed voxel experiment','REMESH');mod.mode='VOXEL';mod.voxel_size=r['voxel_size'];mod.adaptivity=0.;mod.use_smooth_shade=False;mod.use_remove_disconnected=False
        try:
            bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get();evaluated=test.evaluated_get(dg);mesh=evaluated.to_mesh()
            try:vv,ff=arrays(mesh)
            finally:evaluated.to_mesh_clear()
            candidate_files(folder/'voxel',vv,ff)
            records[name]['voxel']=dict(seconds=time.perf_counter()-began,vertices=len(vv),triangles=len(ff),
                settings=dict(mode='VOXEL',voxel_size=r['voxel_size'],adaptivity=0.,smooth_shade=False,remove_disconnected=False),
                source_orientation_accepted_by_volume_backend_not_exterior_certified=True,
                all_output_components_retained=True,crop_was_not_capped_before_remeshing=True,
                empty=not len(ff))
        except Exception as exc:
            records[name]['voxel']=dict(seconds=time.perf_counter()-began,error=type(exc).__name__+': '+str(exc),no_retry=True)
        test.hide_render=True;test.hide_viewport=True;crop.hide_render=True
        began=time.perf_counter();width=r['hi'][0]-r['lo'][0];height=r['hi'][2]-r['lo'][2]
        nx=math.ceil(width/r['chart_grid_spacing_requested']);nz=math.ceil(height/r['chart_grid_spacing_requested'])
        assert nx*nz<=65536,'Bounded chart grid only'
        xs=np.linspace(r['lo'][0],r['hi'][0],2*nx+1);zs=np.linspace(r['lo'][2],r['hi'][2],2*nz+1)
        points=np.zeros((len(zs),len(xs),3),np.float32);valid=np.zeros(points.shape[:2],bool);hit_ids=np.full(valid.shape,-1,np.int32)
        direction=Vector((0,-r['outside_y'],0))
        for j,z in enumerate(zs):
            for i,x in enumerate(xs):
                hit,normal,face,distance=tree.ray_cast(Vector((x,r['outside_y'],z)),direction,2.)
                if hit is not None:points[j,i]=hit;valid[j,i]=True;hit_ids[j,i]=face
        limit=2*max(width/nx,height/nz)
        vv,ff,receipt=api.chart(np,points,valid,r['outside_y'],limit)
        candidate_files(folder/'chart',vv,ff)
        api.prior.save_npy(folder/'chart/dense-hit-points.npy',points);api.prior.save_npy(folder/'chart/dense-hit-valid.npy',valid);api.prior.save_npy(folder/'chart/dense-hit-face-ids.npy',hit_ids)
        records[name]['chart']=dict(seconds=time.perf_counter()-began,vertices=len(vv),triangles=len(ff),
            settings=dict(grid_intervals=[nx,nz],actual_step=[width/nx,height/nz],outside_y=r['outside_y'],
                winding_ignored_for_rays=True,first_hit_only=True,**receipt),
            dense_reference_rays=int(valid.size),reference_hit_rays=int(valid.sum()),all_output_components_retained=True,
            limitation='One outside-supported chart; hidden sheets/undercuts may be omitted. Discontinuity seams are not automatically interpreted as intentional wounds.')
        # Sampled distances, not a continuous Hausdorff or full fidelity guarantee.
        for kind in ('voxel','chart'):
            path=folder/kind
            if not (path/'mesh.ply').exists():continue
            cv=np.load(path/'vertices.npy',allow_pickle=False);cf=np.load(path/'faces.npy',allow_pickle=False);ct=bvh(cv,cf)
            candidate_samples=np.concatenate((cv,cv[cf].mean(axis=1)))
            distances=[tree.find_nearest(Vector(p))[3] for p in candidate_samples]
            ref_points=points[valid][::2]
            reverse=[ct.find_nearest(Vector(p))[3] for p in ref_points]
            records[name][kind]['sampled_candidate_to_reference_distance']=summary(distances)
            records[name][kind]['sampled_visible_reference_to_candidate_distance']=summary(reverse)
        payload(crop,v,f);assert before==(v.tobytes(),f.tobytes()) and sha(ref/'mesh.ply')==r['reference_PLY_sha256']
    save(BASE/'processing.json',records);print(json.dumps(records),flush=True)


if MODE=='render':
    verified=read(BASE/'independent-mesh-verification.json');assert verified['classification']=='PASS_READBACK_AND_REFERENCE_IDENTITY'
    settings=read(BASE/'experiment-settings.json');output=BASE/'renders';output.mkdir(exist_ok=False)
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=8
    scene.cycles.use_denoising=False;scene.cycles.use_adaptive_sampling=False;scene.cycles.seed=0
    scene.cycles.max_bounces=2;scene.cycles.diffuse_bounces=1;scene.cycles.glossy_bounces=1;scene.cycles.transparent_max_bounces=32
    scene.render.threads_mode='FIXED';scene.render.threads=4;scene.render.resolution_x=scene.render.resolution_y=512;scene.render.resolution_percentage=100
    scene.render.film_transparent=False;scene.render.use_compositing=False;scene.render.use_sequencer=False;scene.render.use_persistent_data=False
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    world=bpy.data.worlds.new('Neutral inspection');world.use_nodes=True;scene.world=world
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(.12,.12,.12,1);world.node_tree.nodes['Background'].inputs['Strength'].default_value=.5
    camera_data=bpy.data.cameras.new('Matched patch views');camera_data.type='ORTHO';camera_data.clip_start=.001;camera_data.clip_end=20
    camera=bpy.data.objects.new('Matched patch views',camera_data);scene.collection.objects.link(camera);scene.camera=camera
    grey=bpy.data.materials.new('Neutral flat clay');grey.use_nodes=True
    node=grey.node_tree.nodes.get('Principled BSDF');node.inputs['Base Color'].default_value=(.35,.35,.35,1);node.inputs['Roughness'].default_value=.9
    def back_material(cull):
        m=bpy.data.materials.new('Backface transparent' if cull else 'Front cyan back red');m.use_nodes=True;m.node_tree.nodes.clear();nodes=m.node_tree.nodes;links=m.node_tree.links
        geom=nodes.new('ShaderNodeNewGeometry');mix=nodes.new('ShaderNodeMixShader');front=nodes.new('ShaderNodeEmission');front.inputs['Color'].default_value=(.02,.7,.8,1)
        if cull:back=nodes.new('ShaderNodeBsdfTransparent')
        else:back=nodes.new('ShaderNodeEmission');back.inputs['Color'].default_value=(.85,.02,.02,1)
        out=nodes.new('ShaderNodeOutputMaterial');links.new(geom.outputs['Backfacing'],mix.inputs[0]);links.new(front.outputs[0],mix.inputs[1]);links.new(back.outputs[0],mix.inputs[2]);links.new(mix.outputs[0],out.inputs['Surface']);return m
    materials={'neutral':grey,'front-back':back_material(False),'culled':back_material(True)}
    lights=[]
    for name,energy in [('Key',130),('Fill',90),('Edge',110)]:
        light=bpy.data.lights.new(name,'AREA');light.energy=energy;light.size=2.5;obj=bpy.data.objects.new(name,light);scene.collection.objects.link(obj);lights.append(obj)
    frames=[];measures={};all_inputs=[]
    for region,r in settings['regions'].items():
        inputs={};before={};objects={};trees={}
        for kind in ('reference','voxel','chart'):
            folder=BASE/'reference'/region if kind=='reference' else BASE/'candidates'/region/kind
            if not (folder/'mesh.ply').exists():continue
            expected=verified['regions'][region][kind]['PLY_sha256'];assert sha(folder/'mesh.ply')==expected
            before[kind]=(folder/'mesh.ply',(folder/'mesh.ply').stat().st_mtime_ns,expected)
            v=np.load(folder/'vertices.npy',allow_pickle=False);f=np.load(folder/'faces.npy',allow_pickle=False)
            inputs[kind]=(v,f);objects[kind]=mesh_obj(region+' '+kind,v,f);payload(objects[kind],v,f);trees[kind]=bvh(v,f)
            all_inputs.append((objects[kind],v,f));objects[kind].hide_render=True
        v,f=inputs['reference'];center=(v.min(axis=0).astype(np.float64)+v.max(axis=0).astype(np.float64))*.5
        scale=max(r['hi'][0]-r['lo'][0],r['hi'][2]-r['lo'][2])*1.35;camera_data.ortho_scale=scale
        sign=r['outside_y'];measures[region]={}
        for ob,offset in zip(lights,[(1,2*sign,2),(-2,sign,.5),(0,-sign,2)]):
            ob.location=Vector(center)+Vector(offset);ob.rotation_euler=(Vector(center)-ob.location).to_track_quat('-Z','Y').to_euler()
        for angle in (-35,0,35):
            theta=math.radians(angle);location=Vector(center)+Vector((2*math.sin(theta),2*sign*math.cos(theta),0))
            direction=(Vector(center)-location).normalized();right=direction.cross(Vector((0,0,1))).normalized();up=right.cross(direction)
            camera.location=location;camera.rotation_mode='QUATERNION';camera.rotation_quaternion=Matrix((right,up,-direction)).transposed().to_quaternion()
            raster={};resolution=128;origins=[]
            for j in range(resolution):
                for i in range(resolution):origins.append(location+right*((i+.5)/resolution-.5)*scale+up*(.5-(j+.5)/resolution)*scale)
            for kind,obj in objects.items():
                depth=np.full((resolution,resolution),np.nan,np.float32);face_ids=np.full(depth.shape,-1,np.int32);facing=np.zeros(depth.shape,bool)
                for k,origin in enumerate(origins):
                    hit,normal,face,distance=trees[kind].ray_cast(origin,direction,4.)
                    if hit is not None:depth.flat[k]=distance;face_ids.flat[k]=face;facing.flat[k]=normal.dot(-direction)>0
                valid=face_ids>=0;raster[kind]=dict(depth=depth,valid=valid,face_ids=face_ids,facing=facing)
                for name,a in [('depth',depth),('face-ids',face_ids),('facing',facing)]:api.prior.save_npy(output/f'{region}-{angle:+03d}-{kind}-{name}.npy',a)
                for mode,material in materials.items():
                    for other in objects.values():other.hide_render=True
                    obj.hide_render=False;obj.data.materials.clear();obj.data.materials.append(material)
                    scene.view_settings.view_transform='AgX' if mode=='neutral' else 'Standard';scene.view_settings.exposure=-.75 if mode=='neutral' else 0
                    filename=f'{region}-{angle:+03d}-{kind}-{mode}.png';scene.render.filepath=str(output/filename)
                    assert scene.cycles.device=='CPU';bpy.ops.render.render(write_still=True)
                    frames.append(dict(region=region,angle=angle,kind=kind,mode=mode,name=filename,sha256=sha(output/filename),
                        camera_location=list(location),target=center.tolist(),ortho_scale=scale))
            ref=raster['reference'];rows={}
            for kind,row in raster.items():
                common=ref['valid']&row['valid'];union=ref['valid']|row['valid']
                rows[kind]=dict(reference_hit_pixels=int(ref['valid'].sum()),candidate_hit_pixels=int(row['valid'].sum()),
                    shared_hit_pixels=int(common.sum()),reference_hit_pixels_lost=int((ref['valid']&~row['valid']).sum()),
                    new_hit_pixels_where_reference_void=int((~ref['valid']&row['valid']).sum()),
                    hit_mask_intersection_over_union=float(common.sum()/union.sum()) if union.any() else 1.,
                    camera_facing_first_hits=int((row['valid']&row['facing']).sum()),backfacing_first_hits=int((row['valid']&~row['facing']).sum()),
                    absolute_depth_difference=summary(abs(ref['depth'][common]-row['depth'][common])))
            if region=='chest-wound' and angle==0:
                origins_np=np.array(origins);x=origins_np[:,0].reshape(128,128);z=origins_np[:,2].reshape(128,128)
                roi=(x>=.03)&(x<=.10)&(z>=.15)&(z<=.23)
                ref_y=location.y+float(direction.y)*ref['depth'];skin=float(np.median(ref_y[ref['valid']&~roi]))
                depth_cut=skin+2*r['chart_grid_spacing_requested'];deep=roi&ref['valid']&(ref_y>depth_cut);void=roi&~ref['valid']
                for kind,row in raster.items():
                    yy=location.y+float(direction.y)*row['depth'];candidate_deep=roi&row['valid']&(yy>depth_cut)
                    rows[kind]['wound_depth_witness']=dict(ROI_world_x=[.03,.10],ROI_world_z=[.15,.23],
                        skin_depth_reference_y=skin,deep_y_threshold=depth_cut,reference_deep_hit_pixels=int(deep.sum()),
                        deep_hit_pixels_retained=int((deep&candidate_deep).sum()),reference_void_pixels=int(void.sum()),
                        reference_void_pixels_filled=int((void&row['valid']).sum()),reference_deep_hit_pixels_lost=int((deep&~row['valid']).sum()),
                        reference_deep_hit_pixels_shallowed=int((deep&row['valid']&~candidate_deep).sum()),
                        limitation='Geometric depth/void witness inside a visually chosen wound ROI, not a semantic wound mask or hidden-cavity proof')
            measures[region][str(angle)]=rows
        for kind,(p,mtime,expected) in before.items():assert p.stat().st_mtime_ns==mtime and sha(p)==expected
        for obj in objects.values():obj.hide_render=True
    for obj,v,f in all_inputs:payload(obj,v,f)
    save(BASE/'view-measurements.json',measures)
    save(output/'render-metadata.json',dict(classification='PASS_MATCHED_CPU_PATCH_PRESENTATION',device='CPU',threads=4,samples=8,
        frames=frames,all_imported_payloads_exact=True,modifiers_applied_during_render=False,smooth_shading=False,
        raster_metrics_resolution=128,depth_and_coverage_metrics_are_samples_not_continuous_equivalence=True))
    print(json.dumps(dict(frames=len(frames),classification='PASS_MATCHED_CPU_PATCH_PRESENTATION')),flush=True)
