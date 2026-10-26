"""Explicit Blender CPU fallback, never a proof of Modly's unavailable native binaries."""
import sys
sys.dont_write_bytecode=True
if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO plus EOF required before Blender imports')
import importlib.util,os
from pathlib import Path
path=Path(__file__).with_name('amd_cpu_uv_bake.py')
spec=importlib.util.spec_from_file_location('_uv_bake_policy',path);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
sys.addaudithook(p.audit_policy)
import array,math,time
import bpy
import numpy as np
from mathutils import Matrix,Vector
STAGE=os.environ.get('MODLY_UV_STAGE')
if STAGE not in ('synthetic','zombie-uv','zombie-bake','preview') or '--' in sys.argv:raise ValueError('One explicit owned CPU stage required')
B=p.BASE

def init():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU'
    scene.cycles.samples=1;scene.cycles.use_denoising=False;scene.cycles.use_adaptive_sampling=False
    scene.cycles.max_bounces=2;scene.cycles.diffuse_bounces=1;scene.cycles.glossy_bounces=1
    scene.render.threads_mode='FIXED';scene.render.threads=4
    scene.render.resolution_x=scene.render.resolution_y=640;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    scene.view_settings.view_transform='AgX';scene.view_settings.exposure=-.75;scene.view_settings.gamma=1
    bpy.context.preferences.filepaths.temporary_directory=str(B/'temp')
    for layer in scene.view_layers:layer.cycles.use_denoising=False
    return scene

def arrays(obj,world=False):
    mesh=obj.data
    v=np.empty((len(mesh.vertices),3),np.float32);mesh.vertices.foreach_get('co',v.ravel())
    if any(len(face.vertices)!=3 for face in mesh.polygons):raise ValueError('Triangles only; no automatic triangulation')
    f=np.empty((len(mesh.polygons),3),np.int32);mesh.loops.foreach_get('vertex_index',f.ravel())
    if world:
        matrix=np.array(obj.matrix_world,dtype=np.float32)
        if not np.array_equal(matrix,np.eye(4,dtype=np.float32)):
            rounded=np.rint(matrix)
            if not np.allclose(matrix,rounded,atol=1e-6) or not np.all(rounded[:3,3]==0) or not np.all(np.abs(rounded[:3,:3]).sum(axis=0)==1):
                raise ValueError('Unexpected GLB transform')
            v=np.ascontiguousarray(v@rounded[:3,:3].T)
    return v,f

def uv_array(obj):
    uv=np.empty((len(obj.data.loops),2),np.float32)
    obj.data.uv_layers.active.data.foreach_get('uv',uv.ravel())
    return uv.reshape(-1,3,2)

def active(obj):
    if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    obj.hide_set(False);obj.hide_render=False;obj.select_set(True);bpy.context.view_layer.objects.active=obj
    for other in bpy.context.scene.objects:
        if other.type=='MESH' and other!=obj:other.hide_render=True

def smart_uv(obj):
    active(obj);v0,f0=arrays(obj);geometry=p.geometry_hash(np,v0,f0)
    bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT')
    result=bpy.ops.uv.smart_project(angle_limit=math.radians(66),island_margin=.002,area_weight=0.,correct_aspect=True,scale_to_bounds=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    if result!={'FINISHED'}:raise ValueError('UV projection failed')
    v,f=arrays(obj)
    if v.shape!=v0.shape or f.shape!=f0.shape or p.geometry_hash(np,v,f)!=geometry:raise ValueError('UV operation changed source surface')
    return dict(method='Blender Smart UV Project',angle_degrees=66,island_margin=.002,
                UV=p.uv_metrics(np,uv_array(obj)),surface_geometry_exact=True,
                coordinate_index_payload_exact=bool(np.array_equal(v.view(np.uint32),v0.view(np.uint32)) and np.array_equal(f,f0)))

def save_array(path,value):
    with path.open('xb') as f:np.save(f,value,allow_pickle=False)

def bake_export(obj,name,resolution,uv_report):
    active(obj);folder=B/name;folder.mkdir(exist_ok=False)
    v,f=arrays(obj);source_hash=p.geometry_hash(np,v,f);uv=uv_array(obj);p.uv_metrics(np,uv)
    if name=='plane':
        rgb=np.column_stack((v[:,0],v[:,1],np.full(len(v),.25))).astype(np.float32)
    else:
        lo=v.min(axis=0);span=v.max(axis=0)-lo
        if np.any(span<=0):raise ValueError('Nondegenerate colour normalization bounds')
        rgb=((v-lo)/span).astype(np.float32)
    corners=rgb[f];rgba=np.column_stack((corners.reshape(-1,3),np.ones(3*len(f),np.float32))).astype(np.float32)
    attr=obj.data.color_attributes.new(name='DIAGNOSTIC_POSITION_RGB_NOT_SKIN',type='FLOAT_COLOR',domain='CORNER')
    attr.data.foreach_set('color',rgba.ravel())
    mat=bpy.data.materials.new('Diagnostic vertex colours only');mat.use_nodes=True;mat.node_tree.nodes.clear()
    nodes=mat.node_tree.nodes;links=mat.node_tree.links
    vc=nodes.new('ShaderNodeVertexColor');vc.layer_name=attr.name
    emission=nodes.new('ShaderNodeEmission');output=nodes.new('ShaderNodeOutputMaterial')
    links.new(vc.outputs['Color'],emission.inputs['Color']);links.new(emission.outputs[0],output.inputs['Surface'])
    obj.data.materials.clear();obj.data.materials.append(mat)
    target=bpy.data.images.new('CPU linear bake '+name,width=resolution,height=resolution,alpha=True,float_buffer=True)
    target.colorspace_settings.name='Non-Color'
    image_node=nodes.new('ShaderNodeTexImage');image_node.image=target;nodes.active=image_node;image_node.select=True
    scene=bpy.context.scene
    if scene.cycles.device!='CPU' or obj.modifiers:raise ValueError('CPU unmodified source required')
    began=time.monotonic()
    if bpy.ops.object.bake(type='EMIT',margin=0,use_clear=True)!={'FINISHED'}:raise ValueError('CPU emission bake failed')
    baked=np.empty((resolution,resolution,4),np.float32);target.pixels.foreach_get(baked.ravel())
    if not np.isfinite(baked).all() or not np.any(baked[...,3]>.5):raise ValueError('Empty or non-finite bake')
    p.write_png(np,folder/'diagnostic.png',baked)
    # Explicit linear->sRGB PNG encoding; no artist/AI colours are inferred.
    texture=bpy.data.images.load(str(folder/'diagnostic.png'),check_existing=False);texture.colorspace_settings.name='sRGB'
    nodes.clear();tex=nodes.new('ShaderNodeTexImage');tex.image=texture;tex.interpolation='Closest'
    bsdf=nodes.new('ShaderNodeBsdfPrincipled');bsdf.inputs['Roughness'].default_value=.85;bsdf.inputs['Metallic'].default_value=0
    bsdf.inputs['Alpha'].default_value=1
    output=nodes.new('ShaderNodeOutputMaterial');links.new(tex.outputs['Color'],bsdf.inputs['Base Color']);links.new(bsdf.outputs[0],output.inputs['Surface'])
    for filename,value in [('vertices.npy',v),('faces.npy',f),('loop-uv.npy',uv),('loop-colors.npy',corners),('linear-bake.npy',baked)]:save_array(folder/filename,value)
    active(obj)
    kwargs=dict(filepath=str(folder/'textured.glb'),export_format='GLB',use_selection=True,export_apply=False,
                export_yup=True,export_texcoords=True,export_normals=True,export_vertex_color='NONE',
                export_materials='EXPORT',export_image_format='AUTO',export_draco_mesh_compression_enable=False)
    if bpy.ops.export_scene.gltf(**kwargs)!={'FINISHED'}:raise ValueError('GLB export failed')
    vv,ff=arrays(obj)
    if p.geometry_hash(np,vv,ff)!=source_hash or not np.array_equal(vv.view(np.uint32),v.view(np.uint32)) or not np.array_equal(ff,f):
        raise ValueError('Bake/export changed geometry')
    # Real Blender re-import, in addition to the independent external GLB parser.
    prior=set(bpy.data.objects)
    if bpy.ops.import_scene.gltf(filepath=str(folder/'textured.glb'))!={'FINISHED'}:raise ValueError('GLB re-import failed')
    imported=[o for o in bpy.data.objects if o not in prior and o.type=='MESH']
    if len(imported)!=1:raise ValueError('Single re-imported mesh required')
    fresh=imported[0];iv,iff=arrays(fresh,True)
    if p.geometry_hash(np,iv,iff)!=source_hash:raise ValueError('Re-imported geometry differs')
    if not fresh.data.uv_layers or not fresh.data.materials:raise ValueError('Re-imported UV/material missing')
    images=[n.image for m in fresh.data.materials for n in m.node_tree.nodes if n.type=='TEX_IMAGE' and n.image]
    if not images:raise ValueError('Re-imported texture image missing')
    p.save(folder/'bake.json',dict(classification='PASS_BLENDER_CPU_PENDING_INDEPENDENT_VERIFICATION',
           resolution=resolution,bake_type='EMIT',bake_device=scene.cycles.device,threads=4,geometry_hash=source_hash,
           source_vertices=len(v),source_triangles=len(f),export_reimport_vertices=len(iv),export_reimport_triangles=len(iff),
           surface_geometry_exact=True,geometry_modifiers=[],UV_report=uv_report,reimported_UV_material_image_present=True,
           diagnostic_pattern='Position-normalized RGB; not AI skin or source appearance',
           baked_image_encoding='Explicit sRGB RGB bytes, linear alpha, PNG top-left rows; glTF UV v origin conversion is exporter-managed.',
           elapsed_bake_export_import_seconds=time.monotonic()-began,GLB_sha256=p.digest(folder/'textured.glb')))
    obj.hide_render=True;fresh.hide_render=True
    return fresh

def aim(obj,target):
    forward=(Vector(target)-obj.location).normalized();right=forward.cross(Vector((0,0,1))).normalized();up=right.cross(forward)
    obj.rotation_mode='QUATERNION';obj.rotation_quaternion=Matrix((right,up,-forward)).transposed().to_quaternion()

def setup_lights(center,extent):
    scene=bpy.context.scene
    world=bpy.data.worlds.new('Neutral CPU inspection');world.use_nodes=True
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(.16,.16,.16,1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value=.5;scene.world=world
    for name,offset,power in [('Key',(2,-3,3),180),('Fill',(-3,-1,1.5),140),('Back',(-1,3,3),160)]:
        light=bpy.data.lights.new(name,'AREA');light.energy=power*extent**2;light.size=2.5*extent
        obj=bpy.data.objects.new(name,light);scene.collection.objects.link(obj);obj.location=Vector(center)+Vector(offset)*extent;aim(obj,center)
    data=bpy.data.cameras.new('Fixed comparison');data.type='ORTHO';data.clip_start=.001;data.clip_end=50
    camera=bpy.data.objects.new('Fixed comparison',data);scene.collection.objects.link(camera);scene.camera=camera
    return camera

def draw(obj,camera,path,location,target,scale):
    active(obj);camera.location=Vector(location);aim(camera,target);camera.data.ortho_scale=scale
    scene=bpy.context.scene;scene.cycles.samples=16;scene.render.filepath=str(path)
    if scene.cycles.device!='CPU':raise ValueError('CPU render required')
    bpy.ops.render.render(write_still=True)

scene=init()
if STAGE=='synthetic':
    mesh=bpy.data.meshes.new('Known plane');mesh.from_pydata([(0,0,0),(1,0,0),(1,1,0),(0,1,0)],[],[(0,1,2),(0,2,3)])
    plane=bpy.data.objects.new('Known UV / vertex colour plane',mesh);scene.collection.objects.link(plane)
    layer=mesh.uv_layers.new(name='KnownUV');layer.data.foreach_set('uv',[0,0,1,0,1,1,0,0,1,1,0,1])
    bake_export(plane,'plane',128,dict(method='Known unit-square UVs'))
    cube_mesh=bpy.data.meshes.new('CPU UV cube')
    vertices=[(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    faces=[(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(3,7,6),(3,6,2),(0,4,7),(0,7,3),(1,2,6),(1,6,5)]
    cube_mesh.from_pydata(vertices,[],faces);cube=bpy.data.objects.new('Smart UV cube',cube_mesh);scene.collection.objects.link(cube)
    uv=smart_uv(cube);fresh=bake_export(cube,'cube',128,uv)
    camera=setup_lights((.5,.5,.5),1)
    draw(fresh,camera,B/'cube/preview.png',(2,-3,2.5),(.5,.5,.5),1.8)
    p.save(B/'synthetic-stage.json',dict(classification='PASS_BLENDER_CPU_ONLY_PENDING_INDEPENDENT_VERIFICATION',native_components_executed=False))
elif STAGE=='zombie-uv':
    if p.read(B/'small-verify.json')['classification']!='PASS':raise ValueError('Independent small proof PASS required')
    source=B/'source/mesh.ply'
    if p.digest(source)!=p.SOURCE_SHA or p.digest(p.SOURCE)!=p.SOURCE_SHA:raise ValueError('Original/copy identity differs')
    if bpy.ops.wm.ply_import(filepath=str(source),merge_verts=False,global_scale=1.,use_scene_unit=False,forward_axis='Y',up_axis='Z')!={'FINISHED'}:raise ValueError('PLY import failed')
    obj=next(o for o in scene.objects if o.type=='MESH');v,f=arrays(obj)
    if (len(v),len(f))!=(356876,777400):raise ValueError('Source counts differ')
    for name,current in [('vertices.npy',v),('faces.npy',f)]:
        original=np.load(p.SOURCE.parent/name,mmap_mode='r',allow_pickle=False)
        try:
            if not np.array_equal(current.view(np.uint32),original.view(np.uint32)):raise ValueError('Imported PLY payload differs')
        finally:original._mmap.close()
    report=smart_uv(obj)
    p.save(B/'zombie-uv.json',dict(classification='PASS_UNCHANGED_SURFACE_UV',source_sha256=p.SOURCE_SHA,**report))
    blend=B/'zombie-uv.blend'
    if blend.exists():raise ValueError('Exclusive blend path required')
    bpy.ops.wm.save_as_mainfile(filepath=str(blend),check_existing=False)
elif STAGE=='zombie-bake':
    if p.read(B/'zombie-uv.json')['classification']!='PASS_UNCHANGED_SURFACE_UV':raise ValueError('UV stage PASS required')
    bpy.ops.wm.open_mainfile(filepath=str(B/'zombie-uv.blend'));scene=bpy.context.scene
    if scene.cycles.device!='CPU' or scene.render.threads!=4:raise ValueError('Preserved CPU settings required')
    obj=next(o for o in scene.objects if o.type=='MESH')
    bake_export(obj,'zombie',2048,p.read(B/'zombie-uv.json'))
elif STAGE=='preview':
    rejected=p.read(B/'rejected-candidate-readback.json')
    if rejected['classification']!='PASS_CONTAINER_READBACK_REJECTED_COLOUR' or rejected['overall_Zombie_D_texture_verdict']!='FAIL_COLOUR_ACCURACY':
        raise ValueError('Explicit rejected-candidate readback required; no acceptance upgrade')
    bpy.ops.wm.ply_import(filepath=str(B/'source/mesh.ply'),merge_verts=False,global_scale=1.,use_scene_unit=False,forward_axis='Y',up_axis='Z')
    original=next(o for o in scene.objects if o.type=='MESH')
    mat=bpy.data.materials.new('Original neutral, no colour data');mat.use_nodes=True
    bsdf=mat.node_tree.nodes.get('Principled BSDF');bsdf.inputs['Base Color'].default_value=(.3,.3,.3,1);bsdf.inputs['Roughness'].default_value=.9
    original.data.materials.append(mat)
    prior=set(bpy.data.objects);bpy.ops.import_scene.gltf(filepath=str(B/'zombie/textured.glb'))
    output=next(o for o in bpy.data.objects if o not in prior and o.type=='MESH')
    ov,of=arrays(original,True);vv,ff=arrays(output,True)
    if p.geometry_hash(np,ov,of)!=p.geometry_hash(np,vv,ff):raise ValueError('Preview surfaces differ')
    folder=B/'previews';folder.mkdir(exist_ok=False)
    center=(ov.min(axis=0)+ov.max(axis=0))*.5;extent=float(np.ptp(ov,axis=0).max());camera=setup_lights(center,extent)
    records=[]
    views=[('front',(center[0],-4,center[2]+.32),center,1.17),('back',(center[0],4,center[2]+.32),center,1.17),
           ('hand-minus',(-.425,-2,.30),(-.425,0,.25),.235),('hand-plus',(.425,-2,.30),(.425,0,.25),.235),
           ('face',(0,-2,.41),(0,0,.40),.22),('wounds',(0,-2,.20),(0,0,.15),.34)]
    for name,location,target,scale in views:
        for label,obj in [('original',original),('diagnostic-texture',output)]:
            path=folder/(label+'-'+name+'.png');draw(obj,camera,path,location,target,scale)
            records.append(p.preview_record(path.name,p.digest(path),location,target,scale))
    p.save(folder/'preview-metadata.json',dict(classification='PASS_MATCHED_GEOMETRY_PREVIEWS',surface_geometry_exact=True,
           texture_is_diagnostic_not_skin=True,preview_of_rejected_colour_candidate=True,
           accepted_Zombie_D_texture=False,device='CPU',threads=4,samples=16,views=records))
