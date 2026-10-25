"""Original geometry, CPU-only normals comparisons and diagnostic overlays."""
import sys
if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO/EOF required before Blender imports')
import array
import hashlib
import json
import math
import os
from pathlib import Path
import time

import bpy
from mathutils import Matrix,Vector
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT.parent/'artifacts/amd-zombie-topology-diagnosis-20261025-v1'
OLD=ROOT.parent/'artifacts/amd-zombie-cpu-mesh-20261025-v1'
PLY=OLD/'mesh/mesh.ply'
MODE=os.environ.get('MODLY_ZOMBIE_TOPOLOGY_MODE')
if MODE not in ('compare','overlay','details'):raise ValueError('Explicit bounded presentation mode required')
OUTPUT=BASE/MODE;OUTPUT.mkdir(exist_ok=False)


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def aim(obj,target):
    forward=(target-obj.location).normalized();right=forward.cross(Vector((0,0,1))).normalized();up=right.cross(forward)
    obj.rotation_mode='QUATERNION';obj.rotation_quaternion=Matrix((right,up,-forward)).transposed().to_quaternion()


analysis=read(BASE/'topology-analysis.json');old_render=read(OLD/'previews/render-metadata.json')
assert analysis['classification']=='PASS_READONLY_TOPOLOGY_DIAGNOSIS'
assert sha(PLY)==analysis['PLY_sha256']==old_render['source_PLY_sha256']
before=PLY.stat();source_sha=sha(PLY)
v=np.load(OLD/'mesh/vertices.npy',mmap_mode='r',allow_pickle=False)
f=np.load(OLD/'mesh/faces.npy',mmap_mode='r',allow_pickle=False)
vertex_sha=hashlib.sha256(v.tobytes()).hexdigest();face_sha=hashlib.sha256(f.tobytes()).hexdigest()
data=np.load(BASE/'topology.npz',allow_pickle=False)
assert data['boundary_edges'].shape==(19512,2) and data['nonmanifold_edges'].shape==(31449,2)
assert data['face_component'].shape==(len(f),) and data['vertex_component'].shape==(len(v),)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU'
scene.cycles.samples=16 if MODE=='compare' else 8
scene.cycles.use_denoising=False;scene.cycles.use_adaptive_sampling=False
scene.cycles.max_bounces=2;scene.cycles.diffuse_bounces=1;scene.cycles.glossy_bounces=1
scene.render.threads_mode='FIXED';scene.render.threads=4
scene.render.resolution_x=scene.render.resolution_y=768;scene.render.resolution_percentage=100
scene.render.film_transparent=False;scene.render.use_compositing=False;scene.render.use_sequencer=False
scene.render.use_persistent_data=True;scene.render.image_settings.file_format='PNG'
scene.render.image_settings.color_mode='RGB';scene.view_settings.view_transform='AgX'
scene.view_settings.exposure=-.75;scene.view_settings.gamma=1
bpy.context.preferences.filepaths.temporary_directory=str(BASE)
for layer in scene.view_layers:layer.cycles.use_denoising=False
kwargs=dict(filepath=str(PLY),merge_verts=False,global_scale=1.0,use_scene_unit=False,forward_axis='Y',up_axis='Z')
assert bpy.ops.wm.ply_import(**kwargs)=={'FINISHED'}
objects=[o for o in scene.objects if o.type=='MESH'];assert len(objects)==1
obj=objects[0];mesh=obj.data
assert not obj.modifiers and np.allclose(np.array(obj.matrix_world),np.eye(4),atol=1e-7)


def payload_check():
    assert (len(mesh.vertices),len(mesh.polygons),len(mesh.loops))==(len(v),len(f),3*len(f))
    co=array.array('f',[0])*(3*len(v));ix=array.array('i',[0])*(3*len(f))
    mesh.vertices.foreach_get('co',co);mesh.loops.foreach_get('vertex_index',ix)
    assert hashlib.sha256(co.tobytes()).hexdigest()==vertex_sha
    assert hashlib.sha256(ix.tobytes()).hexdigest()==face_sha
    assert not obj.modifiers


payload_check()
center=(Vector(v.min(axis=0))+Vector(v.max(axis=0)))*.5
extent=float((v.max(axis=0)-v.min(axis=0)).max())


def material(name,color,emission=False):
    mat=bpy.data.materials.new(name);mat.use_nodes=True
    if emission:
        mat.node_tree.nodes.clear();node=mat.node_tree.nodes.new('ShaderNodeEmission')
        node.inputs['Color'].default_value=(*color,1);node.inputs['Strength'].default_value=1.6
        out=mat.node_tree.nodes.new('ShaderNodeOutputMaterial');mat.node_tree.links.new(node.outputs[0],out.inputs['Surface'])
    else:
        node=mat.node_tree.nodes.get('Principled BSDF');node.inputs['Base Color'].default_value=(*color,1)
        node.inputs['Roughness'].default_value=.9;node.inputs['Metallic'].default_value=0
    return mat


neutral=material('Neutral matte inspection',(.3,.3,.3))
palette=[(.9,.55,.04),(.05,.8,.2),(.9,.2,.04),(.6,.15,.9),(.05,.55,1),(.9,.7,.02),(.1,.9,.7),(.9,.15,.45),(.55,.9,.04)]
mesh.materials.clear();mesh.materials.append(neutral)
for i,color in enumerate(palette):mesh.materials.append(material('Component '+str(i+1),color,True))
world=bpy.data.worlds.new('Neutral studio');world.use_nodes=True
world.node_tree.nodes['Background'].inputs['Color'].default_value=(.16,.16,.16,1)
world.node_tree.nodes['Background'].inputs['Strength'].default_value=.5;scene.world=world
for name,offset,energy in (('Key',(2,-3,3),180),('Fill',(-3,-1,1.5),140),('Back',(-1,3,3),160)):
    light=bpy.data.lights.new(name,'AREA');light.energy=energy*extent**2;light.size=2.5*extent
    lamp=bpy.data.objects.new(name,light);scene.collection.objects.link(lamp);lamp.location=center+Vector(offset)*extent;aim(lamp,center)
camera_data=bpy.data.cameras.new('Matched orthographic inspection');camera_data.type='ORTHO'
camera=bpy.data.objects.new('Matched orthographic inspection',camera_data);scene.collection.objects.link(camera);scene.camera=camera
camera_data.clip_start=.001*extent;camera_data.clip_end=20*extent
records=[];began=time.monotonic();curves=[];fragment_obj=None


def draw(name,location,target,scale,smooth=True):
    mesh.polygons.foreach_set('use_smooth',[smooth]*len(f));mesh.update()
    camera.location=Vector(location);aim(camera,Vector(target));camera_data.ortho_scale=scale
    scene.render.filepath=str(OUTPUT/(name+'.png'))
    assert scene.cycles.device=='CPU' and not obj.modifiers
    bpy.ops.render.render(write_still=True)
    records.append(dict(name=name+'.png',sha256=sha(OUTPUT/(name+'.png')),smooth_shading=smooth,
        camera_location=list(camera.location),target=list(target),ortho_scale=scale,
        original_hidden_for_isolation=obj.hide_render,diagnostic_edges_visible=any(not c.hide_render for c in curves)))
    print('CPU_TOPOLOGY_RENDER='+name,flush=True)


def edge_object(name,edges,color):
    curve=bpy.data.curves.new(name,'CURVE');curve.dimensions='3D';curve.resolution_u=1
    curve.bevel_depth=.00035;curve.bevel_resolution=0;curve.resolution_u=1
    for edge in edges:
        spline=curve.splines.new('POLY');spline.points.add(1)
        p=v[edge];spline.points.foreach_set('co',[*map(float,p[0]),1,*map(float,p[1]),1])
    ob=bpy.data.objects.new(name,curve);scene.collection.objects.link(ob)
    curve.materials.append(material(name,color,True));curves.append(ob);return ob


if MODE=='compare':
    for row in old_render['views']:
        angle=row['azimuth_degrees']
        draw(f'flat-{angle:03d}',row['camera_location'],center,row['ortho_scale'],False)
        draw(f'smooth-{angle:03d}',row['camera_location'],center,row['ortho_scale'],True)
else:
    edge_object('Boundary cyan',data['boundary_edges'],(.02,.8,1))
    edge_object('Nonmanifold magenta',data['nonmanifold_edges'],(1,.01,.35))
    if MODE=='overlay':
        for row in old_render['views']:
            draw(f'topology-{row["azimuth_degrees"]:03d}',row['camera_location'],center,row['ortho_scale'])
        for c in curves:c.hide_render=True
        mesh.polygons.foreach_set('material_index',data['face_component'].astype(np.int32))
        for row in old_render['views'][::4]:
            draw(f'components-{row["azimuth_degrees"]:03d}',row['camera_location'],center,row['ortho_scale'])
        # A diagnostic subset object only: original mesh remains intact/hidden.
        ids=np.flatnonzero(data['face_component']>0);used=np.unique(f[ids]);mapping=np.full(len(v),-1,np.int32);mapping[used]=np.arange(len(used))
        auxiliary=bpy.data.meshes.new('Isolated existing source fragments')
        auxiliary.from_pydata(v[used].tolist(),[],mapping[f[ids]].tolist());auxiliary.update()
        fragment_obj=bpy.data.objects.new('Diagnostic fragment visibility only',auxiliary);scene.collection.objects.link(fragment_obj)
        for mat in mesh.materials:auxiliary.materials.append(mat)
        auxiliary.polygons.foreach_set('material_index',data['face_component'][ids].astype(np.int32))
        obj.hide_render=True
        for row in old_render['views'][::4]:draw(f'isolated-fragments-{row["azimuth_degrees"]:03d}',row['camera_location'],center,row['ortho_scale'])
        obj.hide_render=False;fragment_obj.hide_render=True
    else:
        features=[('head-front',(0,0,.39),(0,-1,.05),.25),('crown-front',(0,0,.48),(0,-1,.1),.12),
            ('crown-back',(0,.03,.46),(0,1,.12),.16),('chest-wound',(.055,-.03,.19),(0,-1,.08),.19),
            ('abdomen',(-.02,-.03,.075),(0,-1,.05),.23),('hand-xplus',(.455,0,.25),(0,-1,.25),.18),
            ('hand-xminus',(-.455,0,.25),(0,-1,.25),.18),('hair-back',(0,.025,.395),(0,1,.1),.23),
            ('clothing-hem',(0,0,-.11),(0,-1,.1),.36)]
        for name,target,direction,scale in features:
            location=Vector(target)+Vector(direction).normalized()*2
            for c in curves:c.hide_render=True
            draw(name+'-smooth',location,target,scale)
            for c in curves:c.hide_render=False
            draw(name+'-topology',location,target,scale)

payload_check()
assert sha(PLY)==source_sha and (PLY.stat().st_size,PLY.stat().st_mtime_ns)==(before.st_size,before.st_mtime_ns)
result=dict(classification='PASS_UNCHANGED_GEOMETRY_CPU_DIAGNOSTICS',mode=MODE,PLY_sha256=source_sha,
    topology_data_sha256=sha(BASE/'topology.npz'),analysis_sha256=sha(BASE/'topology-analysis.json'),
    original_vertex_and_face_payloads_exact=True,vertices=len(v),triangles=len(f),geometry_modifiers=[],geometry_moved=False,
    shading_only_comparison=True,engine='CYCLES',device='CPU',threads=4,samples=scene.cycles.samples,
    color_management='AgX',exposure=-.75,lights_W=[180,140,160],resolution=[768,768],
    boundary_color='cyan',nonmanifold_color='magenta',component_colors=palette,
    overlay_radius=.00035,overlays_are_separate_diagnostic_objects=True,
    isolated_fragment_views_hide_main_for_visibility=True if MODE=='overlay' else False,
    views=records,elapsed_seconds=time.monotonic()-began)
with (OUTPUT/'render-metadata.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
v._mmap.close();f._mmap.close();data.close()
print('CPU_TOPOLOGY_RENDER=PASS',flush=True)
