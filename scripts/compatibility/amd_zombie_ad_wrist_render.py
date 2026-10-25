"""Matched CPU-only A/D wrist inspection; no hybrid or geometry modifications."""
import sys
sys.dont_write_bytecode=True
if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Exact owned GO and EOF required')
import importlib.util
from pathlib import Path
entry=Path(__file__).with_name('amd_zombie_ad_wrist.py')
spec=importlib.util.spec_from_file_location('_wrist_policy',entry)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
sys.addaudithook(p.policy.audit_policy(p.BASE))
import array
import hashlib
import time
import bpy
from mathutils import Matrix,Vector
import numpy as np

if '--' in sys.argv:raise ValueError('No alternate modes')
analysis=p.cpu.read(p.BASE/'analysis.json')
if analysis['classification']!='PASS_READONLY_FEASIBILITY' or analysis['hybrid_created']:
    raise ValueError('Original-only feasibility evidence required')
out=p.BASE/'renders';out.mkdir(exist_ok=False)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU'
scene.cycles.samples=16;scene.cycles.use_denoising=False;scene.cycles.use_adaptive_sampling=False
scene.cycles.max_bounces=2;scene.cycles.diffuse_bounces=1;scene.cycles.glossy_bounces=1
scene.render.threads_mode='FIXED';scene.render.threads=4
scene.render.resolution_x=scene.render.resolution_y=768;scene.render.resolution_percentage=100
scene.render.film_transparent=False;scene.render.use_compositing=False;scene.render.use_sequencer=False
scene.render.use_persistent_data=True;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
scene.view_settings.view_transform='AgX';scene.view_settings.exposure=-.75;scene.view_settings.gamma=1
bpy.context.preferences.filepaths.temporary_directory=str(p.BASE)
for layer in scene.view_layers:layer.cycles.use_denoising=False

def aim(obj,target):
    forward=(Vector(target)-obj.location).normalized();right=forward.cross(Vector((0,0,1))).normalized();up=right.cross(forward)
    obj.rotation_mode='QUATERNION';obj.rotation_quaternion=Matrix((right,up,-forward)).transposed().to_quaternion()

def material(name,color,emission=False):
    m=bpy.data.materials.new(name);m.use_nodes=True
    if emission:
        m.node_tree.nodes.clear();n=m.node_tree.nodes.new('ShaderNodeEmission')
        n.inputs['Color'].default_value=(*color,1);n.inputs['Strength'].default_value=1
        output=m.node_tree.nodes.new('ShaderNodeOutputMaterial');m.node_tree.links.new(n.outputs[0],output.inputs['Surface'])
    else:
        n=m.node_tree.nodes.get('Principled BSDF');n.inputs['Base Color'].default_value=(*color,1)
        n.inputs['Roughness'].default_value=.9;n.inputs['Metallic'].default_value=0
    return m

meta=p.cpu.read(p.A/'mesh/mesh.json');center=(Vector(meta['bounds']['min'])+Vector(meta['bounds']['max']))*.5
extent=max(meta['bounds']['max'][i]-meta['bounds']['min'][i] for i in range(3))
world=bpy.data.worlds.new('Neutral studio');world.use_nodes=True
world.node_tree.nodes['Background'].inputs['Color'].default_value=(.16,.16,.16,1)
world.node_tree.nodes['Background'].inputs['Strength'].default_value=.5;scene.world=world
lights=[]
for name,offset,energy in [('Key',(2,-3,3),180),('Fill',(-3,-1,1.5),140),('Back',(-1,3,3),160)]:
    light=bpy.data.lights.new(name,'AREA');light.energy=energy*extent**2;light.size=2.5*extent
    obj=bpy.data.objects.new(name,light);scene.collection.objects.link(obj);obj.location=center+Vector(offset)*extent;aim(obj,center)
    lights.append(dict(name=name,location=list(obj.location),energy=light.energy,size=light.size))
camdata=bpy.data.cameras.new('Shared fixed orthographic camera');camdata.type='ORTHO'
camera=bpy.data.objects.new('Shared fixed orthographic camera',camdata);scene.collection.objects.link(camera);scene.camera=camera
camdata.clip_start=.001;camdata.clip_end=20
neutral=material('Untextured neutral',(.3,.3,.3))
lines=np.load(p.BASE/'diagnostic-lines.npz',allow_pickle=False)
records=[];checks=[];began=time.monotonic()

def edge_obj(name,coords,color):
    curve=bpy.data.curves.new(name,'CURVE');curve.dimensions='3D';curve.resolution_u=1
    curve.bevel_depth=.00018;curve.bevel_resolution=0
    for a,b in coords:
        s=curve.splines.new('POLY');s.points.add(1);s.points.foreach_set('co',[*map(float,a),1,*map(float,b),1])
    obj=bpy.data.objects.new(name,curve);scene.collection.objects.link(obj)
    curve.materials.append(material(name,color,True));return obj

def draw(label,kind,location,target,scale,overlay=False):
    camera.location=Vector(location);aim(camera,target);camdata.ortho_scale=scale
    scene.render.filepath=str(out/f'{label}-{kind}.png')
    if scene.cycles.device!='CPU' or scene.render.threads!=4:raise ValueError('CPU settings changed')
    bpy.ops.render.render(write_still=True)
    records.append(dict(name=f'{label}-{kind}.png',source=label,sha256=p.cpu.digest(out/f'{label}-{kind}.png'),
                        location=list(camera.location),target=list(target),ortho_scale=scale,overlay=overlay))
    print('WRIST_CPU_RENDER='+label+'-'+kind,flush=True)

for label,base in [('A',p.A),('D',p.D)]:
    ply=base/'mesh/mesh.ply'
    if p.cpu.digest(ply)!=p.SHA[label]:raise ValueError('Source mesh hash changed')
    kwargs=dict(filepath=str(ply),merge_verts=False,global_scale=1.,use_scene_unit=False,forward_axis='Y',up_axis='Z')
    if bpy.ops.wm.ply_import(**kwargs)!={'FINISHED'}:raise ValueError('Exact PLY import required')
    objects=[o for o in scene.objects if o.type=='MESH'];obj=objects[-1];mesh=obj.data
    v=np.load(base/'mesh/vertices.npy',mmap_mode='r',allow_pickle=False)
    f=np.load(base/'mesh/faces.npy',mmap_mode='r',allow_pickle=False)
    vertex_sha=hashlib.sha256(v.tobytes()).hexdigest();face_sha=hashlib.sha256(f.tobytes()).hexdigest()
    def payload():
        if (len(mesh.vertices),len(mesh.polygons),len(mesh.loops))!=(len(v),len(f),3*len(f)) or obj.modifiers:
            raise ValueError('Source topology changed')
        if max(abs(obj.matrix_world[i][j]-(i==j)) for i in range(4) for j in range(4))>1e-7:
            raise ValueError('Source transform changed')
        co=array.array('f',[0])*(3*len(v));ix=array.array('i',[0])*(3*len(f))
        mesh.vertices.foreach_get('co',co);mesh.loops.foreach_get('vertex_index',ix)
        if hashlib.sha256(co.tobytes()).hexdigest()!=vertex_sha or hashlib.sha256(ix.tobytes()).hexdigest()!=face_sha:
            raise ValueError('Source coordinate/index payload changed')
    payload();mesh.materials.clear();mesh.materials.append(neutral)
    mesh.polygons.foreach_set('use_smooth',[False]*len(f));mesh.update()
    draw(label,'front',(center.x,-4,center.z+.32),center,1.17)
    draw(label,'back',(center.x,4,center.z+.32),center,1.17)
    for side,name in [(-1,'minus'),(1,'plus')]:
        target=(side*.425,0,.25)
        draw(label,f'hand-{name}-front',(side*.425,-2,.30),target,.235)
        draw(label,f'hand-{name}-back',(side*.425,2,.30),target,.235)
        curves=[]
        for kind,color in [('boundary',(.02,.8,1)),('nonmanifold',(1,.01,.35)),('section',(1,.6,.02))]:
            curves.append(edge_obj(label+name+kind,lines[f'{label}_{side}_{kind}'],color))
        draw(label,f'hand-{name}-topology',(side*.425,-2,.30),target,.235,True)
        for curve in curves:curve.hide_render=True
    payload();checks.append(dict(source=label,PLY_sha256=p.cpu.digest(ply),vertex_payload_sha256=vertex_sha,face_payload_sha256=face_sha,exact=True))
    obj.hide_render=True;v._mmap.close();f._mmap.close()
lines.close()
for label in ('A','D'):
    if p.cpu.digest((p.A if label=='A' else p.D)/'mesh/mesh.ply')!=p.SHA[label]:raise ValueError('Original changed')
for kind in {r['name'][2:] for r in records}:
    pair=[r for r in records if r['name'][2:]==kind]
    if len(pair)!=2 or any(pair[0][k]!=pair[1][k] for k in ('location','target','ortho_scale','overlay')):
        raise ValueError('Matched cameras required')
p.cpu.save(out/'render-metadata.json',dict(classification='PASS_MATCHED_UNCHANGED_SOURCES',all_source_payloads_exact=True,
           checks=checks,lights=lights,engine='CYCLES',device='CPU',threads=4,samples=16,
           smooth_shading=False,modifiers=[],geometry_changed=False,view_transform='AgX',exposure=-.75,
           diagnostic_lines_are_annotations_only=True,views=records,elapsed_seconds=time.monotonic()-began))
