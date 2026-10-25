"""Matched CPU presentation of immutable baseline and the approved two-face variant."""
import sys
if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO/EOF required before Blender imports')
import array
import hashlib
import json
import os
from pathlib import Path
import bpy
from mathutils import Matrix,Vector
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT.parent/'artifacts/amd-zombie-winding-prototype-20261025-v1'
OLD=ROOT.parent/'artifacts/amd-zombie-orphan-cleanup-20261025-v1'
MODE=os.environ.get('MODLY_ZOMBIE_WINDING_MODE')
if MODE not in ('silhouettes','local-render'):raise ValueError('Explicit bounded render mode required')
OUTPUT=BASE/MODE;OUTPUT.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
proof=read(BASE/'independent-verification.json');assert proof['classification']=='PASS_EXACT_TWO_FACE_WINDING_EDIT'
paths=[OLD/'variant/mesh.ply',BASE/'variant/mesh.ply']
assert sha(paths[0])==proof['baseline_PLY_sha256'] and sha(paths[1])==proof['variant_PLY_sha256']
before={str(p):(p.stat().st_size,p.stat().st_mtime_ns,sha(p)) for p in paths}
arrays=[(np.load(root/'variant/vertices.npy',mmap_mode='r',allow_pickle=False),np.load(root/'variant/faces.npy',mmap_mode='r',allow_pickle=False)) for root in (OLD,BASE)]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=8
scene.cycles.use_denoising=False;scene.cycles.use_adaptive_sampling=False;scene.cycles.seed=0
scene.cycles.max_bounces=2;scene.cycles.diffuse_bounces=1;scene.cycles.glossy_bounces=1;scene.cycles.transparent_max_bounces=16
scene.render.threads_mode='FIXED';scene.render.threads=4
scene.render.resolution_x=scene.render.resolution_y=768;scene.render.resolution_percentage=100
scene.render.film_transparent=False;scene.render.use_compositing=False;scene.render.use_sequencer=False
scene.render.use_persistent_data=False;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
scene.view_settings.view_transform='Standard';scene.view_settings.exposure=0;scene.view_settings.gamma=1
bpy.context.preferences.filepaths.temporary_directory=str(BASE)
objects=[]
for path in paths:
    prior=set(scene.objects)
    assert bpy.ops.wm.ply_import(filepath=str(path),merge_verts=False,global_scale=1.,use_scene_unit=False,forward_axis='Y',up_axis='Z')=={'FINISHED'}
    obj=next(o for o in scene.objects if o not in prior and o.type=='MESH')
    assert not obj.modifiers and np.array_equal(np.array(obj.matrix_world),np.eye(4))
    obj.data.polygons.foreach_set('use_smooth',[False]*len(obj.data.polygons));objects.append(obj)
def payload(obj,v,f):
    assert (len(obj.data.vertices),len(obj.data.polygons),len(obj.data.loops))==(len(v),len(f),3*len(f))
    co=array.array('f',[0])*(3*len(v));ix=array.array('i',[0])*(3*len(f))
    obj.data.vertices.foreach_get('co',co);obj.data.loops.foreach_get('vertex_index',ix)
    assert co.tobytes()==v.tobytes() and ix.tobytes()==f.tobytes() and not obj.modifiers
for obj,(v,f) in zip(objects,arrays):payload(obj,v,f)
v,f=arrays[0];center=(Vector(v.min(axis=0))+Vector(v.max(axis=0)))*.5
camera_data=bpy.data.cameras.new('Matched orthographic cameras');camera_data.type='ORTHO';camera_data.clip_start=.001;camera_data.clip_end=20
camera=bpy.data.objects.new('Matched orthographic cameras',camera_data);scene.collection.objects.link(camera);scene.camera=camera
world=bpy.data.worlds.new('Inspection background');world.use_nodes=True;scene.world=world
def aim(target):
    forward=(Vector(target)-camera.location).normalized();right=forward.cross(Vector((0,0,1))).normalized();up=right.cross(forward)
    camera.rotation_mode='QUATERNION';camera.rotation_quaternion=Matrix((right,up,-forward)).transposed().to_quaternion()
def emission(name,color):
    m=bpy.data.materials.new(name);m.use_nodes=True;m.node_tree.nodes.clear()
    n=m.node_tree.nodes.new('ShaderNodeEmission');n.inputs['Color'].default_value=(*color,1)
    out=m.node_tree.nodes.new('ShaderNodeOutputMaterial');m.node_tree.links.new(n.outputs[0],out.inputs['Surface']);return m
def diffuse(name,color):
    m=bpy.data.materials.new(name);m.use_nodes=True
    n=m.node_tree.nodes.get('Principled BSDF');n.inputs['Base Color'].default_value=(*color,1);n.inputs['Roughness'].default_value=.9
    return m
def back_material(cull):
    m=bpy.data.materials.new('Backfaces transparent' if cull else 'Cyan front red back');m.use_nodes=True;m.node_tree.nodes.clear();nodes=m.node_tree.nodes;links=m.node_tree.links
    geom=nodes.new('ShaderNodeNewGeometry');mix=nodes.new('ShaderNodeMixShader');front=nodes.new('ShaderNodeEmission');front.inputs['Color'].default_value=(.02,.7,.8,1)
    if cull:back=nodes.new('ShaderNodeBsdfTransparent')
    else:back=nodes.new('ShaderNodeEmission');back.inputs['Color'].default_value=(.85,.02,.02,1)
    out=nodes.new('ShaderNodeOutputMaterial');links.new(geom.outputs['Backfacing'],mix.inputs[0]);links.new(front.outputs[0],mix.inputs[1]);links.new(back.outputs[0],mix.inputs[2]);links.new(mix.outputs[0],out.inputs['Surface']);return m
def show(index):
    for j,obj in enumerate(objects):obj.hide_render=j!=index
def render(name):
    scene.render.filepath=str(OUTPUT/name);assert scene.cycles.device=='CPU';bpy.ops.render.render(write_still=True)

pairs=[];hashes={};witness=[]
if MODE=='silhouettes':
    black=emission('Black two-sided silhouette',(0,0,0))
    for obj in objects:obj.data.materials.clear();obj.data.materials.append(black)
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(1,1,1,1);world.node_tree.nodes['Background'].inputs['Strength'].default_value=1
    for row in read(OLD/'silhouettes/render-metadata.json')['pairs']:
        camera.location=Vector(row['camera_location']);camera_data.ortho_scale=row['ortho_scale'];aim(row['target'])
        record={key:row[key] for key in ('azimuth_degrees','camera_location','target','ortho_scale')}
        for i,kind in enumerate(('source','variant')):
            show(i);name=f'{kind}-{row["azimuth_degrees"]:03d}.png';render(name);record[kind]=name;record[kind+'_sha256']=sha(OUTPUT/name)
        pairs.append(record)
else:
    region=read(BASE/'regional-analysis.json');ids=np.load(BASE/'region-face-ids.npy',allow_pickle=False)
    target=Vector(v[f[[334008,334009]]].mean(axis=(0,1)));local_location=target+Vector((0,2,.3))
    grey=diffuse('Flat neutral clay',(.3,.3,.3));blue=diffuse('Analysis region',(.1,.3,.6));gold=diffuse('Only two approved faces',(.95,.6,.04))
    materials={'neutral':grey,'front-back':back_material(False),'culled':back_material(True)}
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(.16,.16,.16,1);world.node_tree.nodes['Background'].inputs['Strength'].default_value=.5
    for name,offset,energy in [('Key',(2,3,3),180),('Fill',(-3,1,1.5),140),('Back',(-1,-3,3),160)]:
        light=bpy.data.lights.new(name,'AREA');light.energy=energy;light.size=2.5
        ob=bpy.data.objects.new(name,light);scene.collection.objects.link(ob);ob.location=center+Vector(offset);ob.rotation_euler=(center-ob.location).to_track_quat('-Z','Y').to_euler()
    for mode in ('neutral','front-back','culled','regional'):
        camera.location=local_location if mode!='regional' else Vector((0,2,.35))
        look=target if mode!='regional' else Vector((0,.08,.20));camera_data.ortho_scale=.035 if mode!='regional' else .18;aim(look)
        scene.view_settings.view_transform='AgX' if mode in ('neutral','regional') else 'Standard';scene.view_settings.exposure=-.75 if mode in ('neutral','regional') else 0
        for i,kind in enumerate(('source','variant')):
            show(i);obj=objects[i];obj.data.materials.clear()
            obj.data.polygons.foreach_set('material_index',np.zeros(len(f),np.int32))
            if mode=='regional':
                for m in (grey,blue,gold):obj.data.materials.append(m)
                indices=np.zeros(len(f),np.int32);indices[ids]=1;indices[[334008,334009]]=2;obj.data.polygons.foreach_set('material_index',indices)
            else:obj.data.materials.append(materials[mode])
            name=f'{kind}-{mode}.png';render(name);hashes[name]=sha(OUTPUT/name)
            if mode=='front-back':
                direction=(look-camera.location).normalized();depsgraph=bpy.context.evaluated_depsgraph_get()
                for face_id in (334008,334009):
                    vv,ff=arrays[i];points=vv[ff[face_id]].astype(np.float64);normal=np.cross(points[1]-points[0],points[2]-points[0]);normal/=np.linalg.norm(normal)
                    centroid=Vector(points.mean(axis=0));origin=centroid-direction*2
                    hit,position,_,index,hitobj,matrix=scene.ray_cast(depsgraph,origin,direction,distance=4)
                    dot=float(normal@(-np.array(direction)))
                    witness.append(dict(kind=kind,face_id=face_id,normal_dot_to_camera=dot,camera_facing=dot>0,
                        first_hit_face_id=index,selected_face_first_hit=bool(hit and index==face_id),backfaces_transparent_in_culled_view=True))
for obj,(v,f) in zip(objects,arrays):payload(obj,v,f)
for p in paths:assert before[str(p)]==(p.stat().st_size,p.stat().st_mtime_ns,sha(p))
result=dict(classification='COMPLETE_PENDING_PIXEL_REVIEW' if MODE=='silhouettes' else 'PASS_MATCHED_LOCAL_PRESENTATION',
    device='CPU',threads=4,samples=8,all_imported_vertex_face_payloads_exact=True,geometry_moved=False,
    modifiers_applied=False,smooth_shading=False,pairs=pairs,hashes=hashes,backface_witnesses=witness)
with (OUTPUT/'render-metadata.json').open('x') as stream:json.dump(result,stream,indent=2)
for vv,ff in arrays:vv._mmap.close();ff._mmap.close()
print(json.dumps(result),flush=True)
