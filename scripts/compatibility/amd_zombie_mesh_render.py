"""Nine local views of the authenticated PLY using Blender Cycles CPU only."""
import array
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import bpy
from mathutils import Matrix,Vector
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT.parent/'artifacts/amd-zombie-cpu-mesh-20261025-v1'
MESH=BASE/'mesh/mesh.ply'
OUTPUT=BASE/'previews'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def aim(obj,target):
    forward=(target-obj.location).normalized();right=forward.cross(Vector((0,0,1))).normalized()
    up=right.cross(forward).normalized()
    obj.rotation_mode='QUATERNION';obj.rotation_quaternion=Matrix((right,up,-forward)).transposed().to_quaternion()
    return right,up


analysis=read(BASE/'mesh-analysis.json');metadata=read(BASE/'mesh/mesh.json')
assert analysis['classification']=='PASS_GEOMETRY_READBACK' and analysis['zero_area_triangles']==0
assert analysis['geometry_repaired'] is False and analysis['geometry_smoothed'] is False
before=MESH.stat();source_sha=digest(MESH)
assert source_sha==analysis['PLY_sha256']==metadata['artifacts']['mesh.ply']['sha256']
OUTPUT.mkdir(exist_ok=False)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU'
scene.cycles.samples=8;scene.cycles.use_denoising=False;scene.cycles.use_adaptive_sampling=False
scene.cycles.max_bounces=2;scene.cycles.diffuse_bounces=1;scene.cycles.glossy_bounces=1
scene.render.threads_mode='FIXED';scene.render.threads=4
scene.render.resolution_x=scene.render.resolution_y=768;scene.render.resolution_percentage=100
scene.render.film_transparent=False;scene.render.use_compositing=False;scene.render.use_sequencer=False
scene.render.use_persistent_data=True
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
scene.render.image_settings.color_depth='8'
scene.view_settings.view_transform='Standard';scene.view_settings.exposure=0;scene.view_settings.gamma=1
bpy.context.preferences.filepaths.temporary_directory=str(BASE)
for layer in scene.view_layers:layer.cycles.use_denoising=False
props=set(bpy.ops.wm.ply_import.get_rna_type().properties.keys())
kwargs=dict(filepath=str(MESH))
for key,value in dict(merge_verts=False,global_scale=1.0,use_scene_unit=False,forward_axis='Y',up_axis='Z').items():
    if key in props:kwargs[key]=value
assert bpy.ops.wm.ply_import(**kwargs)=={'FINISHED'}
objects=[obj for obj in scene.objects if obj.type=='MESH'];assert len(objects)==1
obj=objects[0];mesh=obj.data
assert not obj.modifiers
assert max(abs(obj.matrix_world[i][j]-(i==j)) for i in range(4) for j in range(4))<1e-7
expected=(analysis['vertices'],analysis['triangles'])
assert (len(mesh.vertices),len(mesh.polygons))==expected and len(mesh.loops)==3*expected[1]
coordinates=array.array('f',[0])*(3*expected[0]);mesh.vertices.foreach_get('co',coordinates)
indices=array.array('i',[0])*(3*expected[1]);mesh.loops.foreach_get('vertex_index',indices)
v=np.load(BASE/'mesh/vertices.npy',mmap_mode='r',allow_pickle=False)
f=np.load(BASE/'mesh/faces.npy',mmap_mode='r',allow_pickle=False)
assert hashlib.sha256(coordinates.tobytes()).hexdigest()==hashlib.sha256(v.tobytes()).hexdigest()
assert hashlib.sha256(indices.tobytes()).hexdigest()==hashlib.sha256(f.tobytes()).hexdigest()
v._mmap.close();f._mmap.close();del coordinates,indices,v,f
mesh.polygons.foreach_set('use_smooth',[False]*expected[1])
bpy.context.view_layer.update()
bbox=[[min(point[i] for point in obj.bound_box) for i in range(3)],
      [max(point[i] for point in obj.bound_box) for i in range(3)]]
assert max(abs(bbox[i][j]-analysis['bounds'][('min','max')[i]][j]) for i in range(2) for j in range(3))<1e-6
center=(Vector(bbox[0])+Vector(bbox[1]))*.5;extent=max(bbox[1][i]-bbox[0][i] for i in range(3))
corners=[Vector(p) for p in itertools.product(*zip(*bbox))]
material=bpy.data.materials.new('Untextured matte geometry');material.use_nodes=True
shader=material.node_tree.nodes.get('Principled BSDF')
shader.inputs['Base Color'].default_value=(.43,.43,.43,1);shader.inputs['Roughness'].default_value=.85
mesh.materials.clear();mesh.materials.append(material)
world=bpy.data.worlds.new('Neutral background');world.use_nodes=True
world.node_tree.nodes['Background'].inputs['Color'].default_value=(.14,.16,.19,1)
world.node_tree.nodes['Background'].inputs['Strength'].default_value=.6;scene.world=world
for name,offset,energy in (('Key',(2,-3,3),500),('Fill',(-3,-1,1.5),300),('Back',(-1,3,3),400)):
    light=bpy.data.lights.new(name,'AREA');light.energy=energy*extent**2;light.size=2.5*extent
    lamp=bpy.data.objects.new(name,light);scene.collection.objects.link(lamp);lamp.location=center+Vector(offset)*extent;aim(lamp,center)
data=bpy.data.cameras.new('Orthographic geometry inspection');data.type='ORTHO'
camera=bpy.data.objects.new('Orthographic geometry inspection',data);scene.collection.objects.link(camera);scene.camera=camera
data.clip_start=.001*extent;data.clip_end=20*extent
views=[];began=time.monotonic()
for number in range(8):
    angle=number*math.pi/4
    camera.location=center+Vector((math.sin(angle),-math.cos(angle),.08)).normalized()*(4*extent)
    right,up=aim(camera,center)
    data.ortho_scale=2.3*max(max(abs((point-center).dot(right)),abs((point-center).dot(up))) for point in corners)
    assert scene.cycles.device=='CPU' and scene.render.threads==4 and not obj.modifiers
    assert (len(mesh.vertices),len(mesh.polygons))==expected
    name=f'view-{number*45:03d}.png';scene.render.filepath=str(OUTPUT/name)
    bpy.ops.render.render(write_still=True)
    views.append(dict(name=name,azimuth_degrees=number*45,camera_location=list(camera.location),ortho_scale=data.ortho_scale,sha256=digest(OUTPUT/name)))
    print('CPU_GEOMETRY_VIEW='+name,flush=True)
# A ninth image uses exactly the first camera, emission black against white.
camera.location=center+Vector((0,-1,.08)).normalized()*(4*extent);right,up=aim(camera,center)
data.ortho_scale=2.3*max(max(abs((point-center).dot(right)),abs((point-center).dot(up))) for point in corners)
material.node_tree.nodes.clear()
emission=material.node_tree.nodes.new('ShaderNodeEmission');emission.inputs['Color'].default_value=(0,0,0,1)
out=material.node_tree.nodes.new('ShaderNodeOutputMaterial');material.node_tree.links.new(emission.outputs[0],out.inputs['Surface'])
world.node_tree.nodes['Background'].inputs['Color'].default_value=(1,1,1,1)
world.node_tree.nodes['Background'].inputs['Strength'].default_value=1
scene.cycles.samples=8;scene.render.filepath=str(OUTPUT/'silhouette.png');bpy.ops.render.render(write_still=True)
assert digest(MESH)==source_sha and (MESH.stat().st_size,MESH.stat().st_mtime_ns)==(before.st_size,before.st_mtime_ns)
result=dict(classification='PASS_CPU_PREVIEWS',source_PLY_sha256=source_sha,vertices=expected[0],triangles=expected[1],
    imported_vertex_and_face_payloads_exact=True,geometry_unchanged=True,modifiers=[],smooth_shading=False,
    engine='CYCLES',device='CPU',samples=8,threads=4,resolution=[768,768],up_axis='+Z',
    material='temporary neutral untextured preview only',blender_version=bpy.app.version_string,
    import_settings=kwargs,views=views,silhouette_sha256=digest(OUTPUT/'silhouette.png'),elapsed_seconds=time.monotonic()-began)
with (OUTPUT/'render-metadata.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print('CPU_GEOMETRY_PREVIEWS=PASS',flush=True)
