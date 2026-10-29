"""One owned CPU view; procedural truth and future neural mesh stay separate."""
import sys
sys.dont_write_bytecode = True
if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO plus EOF required')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_closed_solid_benchmark as b
args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
if args not in (['--reference'],['--generated']):raise ValueError('Exact reference/generated mode required')
reference=args==['--reference'];base=b.PREP if reference else b.RAW
sys.addaudithook(b.cpu_guard(base))
import hashlib,json,time
import bpy
import numpy as np
from mathutils import Vector
import amd_flexible_dual_grid_cpu as mesh_io
began=time.monotonic()
if b.digest(b.BLENDER)!=b.BLENDER_SHA:raise ValueError('Qualified Blender identity differs')
if reference:
    vertices,faces=b.geometry();v=np.asarray(vertices,dtype='<f4');f=np.asarray(faces,dtype='<i4')
    out=base/'source';out.mkdir(exist_ok=False)
    mesh_io.write_ply(out/'mesh.ply',v,f)
    for name,array in [('vertices',v),('faces',f)]:
        with (out/(name+'.npy')).open('xb') as stream:np.save(stream,array,allow_pickle=False)
    rv,rf=mesh_io.read_ply(out/'mesh.ply')
    if not np.array_equal(rv,v) or not np.array_equal(rf,f):raise ValueError('Ground truth PLY roundtrip differs')
    rv._mmap.close();rf._mmap.close();source=out/'mesh.ply';image=base/'reference.png';metadata=base/'render.json'
else:
    if not (base/'authentication-extract.json').is_file():raise ValueError('Authenticated raw extraction required')
    source=base/'mesh/mesh.ply';v,f=mesh_io.read_ply(source)
    out=base/'view';out.mkdir(exist_ok=False);image=out/'raw-reference-camera.png';metadata=out/'render.json'
if image.exists() or metadata.exists():raise FileExistsError('No render retry or overwrite')
source_hash=b.digest(source)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.wm.ply_import(filepath=str(source),merge_verts=False,global_scale=1.0,use_scene_unit=False,forward_axis='Y',up_axis='Z')
objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
if len(objects)!=1:raise ValueError('One imported untouched mesh required')
obj=objects[0];mesh=obj.data
def check_geometry():
    mesh.calc_loop_triangles();vv=np.empty(len(mesh.vertices)*3,np.float32);mesh.vertices.foreach_get('co',vv)
    ff=np.empty(len(mesh.loop_triangles)*3,np.int32);mesh.loop_triangles.foreach_get('vertices',ff)
    if not np.array_equal(vv.reshape(-1,3),v) or not np.array_equal(ff.reshape(-1,3),f):raise ValueError('Render geometry differs')
    if len(obj.modifiers) or not np.array_equal(np.asarray(obj.matrix_world),np.eye(4)):raise ValueError('Unchanged canonical geometry required')
check_geometry()
for p in mesh.polygons:p.use_smooth=False
mat=bpy.data.materials.new('Neutral matte');mat.use_nodes=True
bsdf=mat.node_tree.nodes.get('Principled BSDF');bsdf.inputs['Base Color'].default_value=(.28,.28,.28,1)
bsdf.inputs['Metallic'].default_value=0;bsdf.inputs['Roughness'].default_value=1
obj.data.materials.clear();obj.data.materials.append(mat)
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=32
scene.cycles.seed=20261029;scene.cycles.use_denoising=False
scene.render.threads_mode='FIXED';scene.render.threads=4
scene.render.resolution_x=scene.render.resolution_y=b.REFERENCE_RESOLUTION;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.render.image_settings.color_depth='8'
scene.render.film_transparent=True
scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.2,.2,.2,1)
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.5
scene.view_settings.view_transform='Standard';scene.view_settings.look='None';scene.view_settings.exposure=0;scene.view_settings.gamma=1
def aim(o):o.rotation_euler=(Vector(b.TARGET)-o.location).to_track_quat('-Z','Y').to_euler()
lights=[]
for name,location,power,size in [('Key',(-3,-4,6),400,4),('Fill',(4,-1,3),100,4)]:
    data=bpy.data.lights.new(name,'AREA');data.energy=power;data.shape='DISK';data.size=size
    light=bpy.data.objects.new(name,data);scene.collection.objects.link(light);light.location=location;aim(light)
    lights.append(dict(name=name,location=list(location),power=power,size=size))
data=bpy.data.cameras.new('Reference orthographic');data.type='ORTHO';data.ortho_scale=b.ORTHO
camera=bpy.data.objects.new('Reference orthographic',data);scene.collection.objects.link(camera);camera.location=b.CAMERA;aim(camera);scene.camera=camera
scene.render.filepath=str(image);bpy.ops.render.render(write_still=True)
check_geometry()
if b.digest(source)!=source_hash:raise ValueError('Source changed while rendering')
b.save(metadata,dict(classification='PASS_CPU_REFERENCE_VIEW',ground_truth=reference,source_PLY_sha256=source_hash,
    vertices=len(v),triangles=len(f),camera=b.CAMERA,target=b.TARGET,ortho_scale=b.ORTHO,resolution=[1024,1024],
    device='CPU',engine='CYCLES',threads=4,samples=32,render_seed=20261029,lights=lights,material_linear_RGB=[.28,.28,.28],roughness=1,
    film_transparent=True,conditioning_background='black from unchanged qualified alpha preprocessing',view_transform='Standard',look='None',
    geometry_unchanged=True,flat_shading=True,modifiers=[],image_sha256=b.digest(image),blender_version=bpy.app.version_string,
    camera_matrix_world=[list(row) for row in camera.matrix_world],elapsed_seconds=time.monotonic()-began))
if not reference:v._mmap.close();f._mmap.close()
