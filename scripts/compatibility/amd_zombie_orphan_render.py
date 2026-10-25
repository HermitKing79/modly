"""CPU silhouettes and an unrepaired local patch display; never edits source files."""
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
BASE=ROOT.parent/'artifacts/amd-zombie-orphan-cleanup-20261025-v1'
OLD=ROOT.parent/'artifacts/amd-zombie-cpu-mesh-20261025-v1'
MODE=os.environ.get('MODLY_ZOMBIE_ORPHAN_MODE')
if MODE not in ('silhouettes','patch-render'):raise ValueError('Explicit bounded render mode required')
OUTPUT=BASE/MODE;OUTPUT.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
verified=read(BASE/'independent-verification.json')
assert verified['classification']=='PASS_SURFACE_IDENTICAL_ORPHAN_VARIANT'
source=OLD/'mesh/mesh.ply';variant=BASE/'variant/mesh.ply'
assert sha(source)==verified['source_PLY_sha256'] and sha(variant)==verified['variant_PLY_sha256']
before={str(p):(p.stat().st_size,p.stat().st_mtime_ns,sha(p)) for p in (source,variant)}
v0=np.load(OLD/'mesh/vertices.npy',mmap_mode='r',allow_pickle=False)
f0=np.load(OLD/'mesh/faces.npy',mmap_mode='r',allow_pickle=False)
v1=np.load(BASE/'variant/vertices.npy',mmap_mode='r',allow_pickle=False)
f1=np.load(BASE/'variant/faces.npy',mmap_mode='r',allow_pickle=False)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU'
scene.cycles.samples=8;scene.cycles.use_denoising=False;scene.cycles.use_adaptive_sampling=False
scene.cycles.max_bounces=2;scene.cycles.diffuse_bounces=1;scene.cycles.glossy_bounces=1
scene.render.threads_mode='FIXED';scene.render.threads=4
scene.render.resolution_x=scene.render.resolution_y=768;scene.render.resolution_percentage=100
scene.render.film_transparent=False;scene.render.use_compositing=False;scene.render.use_sequencer=False
scene.render.use_persistent_data=False;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
scene.view_settings.view_transform='Standard';scene.view_settings.exposure=0;scene.view_settings.gamma=1
bpy.context.preferences.filepaths.temporary_directory=str(BASE)

def load_mesh(path,name):
    prior=set(scene.objects)
    assert bpy.ops.wm.ply_import(filepath=str(path),merge_verts=False,global_scale=1.0,use_scene_unit=False,forward_axis='Y',up_axis='Z')=={'FINISHED'}
    obj=next(o for o in scene.objects if o not in prior and o.type=='MESH');obj.name=name
    assert not obj.modifiers and np.array_equal(np.array(obj.matrix_world),np.eye(4))
    obj.data.polygons.foreach_set('use_smooth',[False]*len(obj.data.polygons));return obj


def payload(obj,v,f):
    assert (len(obj.data.vertices),len(obj.data.polygons),len(obj.data.loops))==(len(v),len(f),3*len(f))
    co=array.array('f',[0])*(3*len(v));ix=array.array('i',[0])*(3*len(f))
    obj.data.vertices.foreach_get('co',co);obj.data.loops.foreach_get('vertex_index',ix)
    assert hashlib.sha256(co.tobytes()).hexdigest()==hashlib.sha256(v.tobytes()).hexdigest()
    assert hashlib.sha256(ix.tobytes()).hexdigest()==hashlib.sha256(f.tobytes()).hexdigest()
    assert not obj.modifiers


original=load_mesh(source,'Immutable original display');cleaned=load_mesh(variant,'Approved orphan-only variant display')
payload(original,v0,f0);payload(cleaned,v1,f1)
center=(Vector(v0.min(axis=0))+Vector(v0.max(axis=0)))*.5
camera_data=bpy.data.cameras.new('Exact original orthographic cameras');camera_data.type='ORTHO'
camera=bpy.data.objects.new('Exact original orthographic cameras',camera_data);scene.collection.objects.link(camera);scene.camera=camera
camera_data.clip_start=.001;camera_data.clip_end=20
world=bpy.data.worlds.new('Inspection background');world.use_nodes=True;scene.world=world


def aim(target):
    forward=(Vector(target)-camera.location).normalized();right=forward.cross(Vector((0,0,1))).normalized();up=right.cross(forward)
    camera.rotation_mode='QUATERNION';camera.rotation_quaternion=Matrix((right,up,-forward)).transposed().to_quaternion()


def material(name,color,emission=False):
    m=bpy.data.materials.new(name);m.use_nodes=True
    if emission:
        m.node_tree.nodes.clear();node=m.node_tree.nodes.new('ShaderNodeEmission');node.inputs['Color'].default_value=(*color,1)
        out=m.node_tree.nodes.new('ShaderNodeOutputMaterial');m.node_tree.links.new(node.outputs[0],out.inputs['Surface'])
    else:
        node=m.node_tree.nodes.get('Principled BSDF');node.inputs['Base Color'].default_value=(*color,1);node.inputs['Roughness'].default_value=.9
    return m


records=[]
if MODE=='silhouettes':
    black=material('Exact black silhouette',(0,0,0),True)
    for obj in (original,cleaned):obj.data.materials.clear();obj.data.materials.append(black)
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(1,1,1,1);world.node_tree.nodes['Background'].inputs['Strength'].default_value=1
    views=read(OLD/'previews/render-metadata.json')['views']
    for row in views:
        camera.location=Vector(row['camera_location']);camera_data.ortho_scale=row['ortho_scale'];aim(center)
        record=dict(azimuth_degrees=row['azimuth_degrees'],camera_location=list(camera.location),target=list(center),ortho_scale=camera_data.ortho_scale)
        for name,obj,other in [('source',original,cleaned),('variant',cleaned,original)]:
            obj.hide_render=False;other.hide_render=True
            filename=f'{name}-{row["azimuth_degrees"]:03d}.png';scene.render.filepath=str(OUTPUT/filename)
            assert scene.cycles.device=='CPU';bpy.ops.render.render(write_still=True)
            record[name]=filename;record[name+'_sha256']=sha(OUTPUT/filename)
        records.append(record)
    result=dict(classification='COMPLETE_PENDING_INDEPENDENT_PIXEL_CHECK',pairs=records)
else:
    patch=read(BASE/'local-patch.json');original.hide_render=True;cleaned.hide_render=False
    scene.view_settings.view_transform='AgX';scene.view_settings.exposure=-.75
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(.16,.16,.16,1);world.node_tree.nodes['Background'].inputs['Strength'].default_value=.5
    grey=material('Unmodified body',(.3,.3,.3));gold=material('Candidate source quad',(.95,.6,.04));blue=material('Incident source face fans',(.12,.38,.75))
    cleaned.data.materials.clear()
    for m in (grey,gold,blue):cleaned.data.materials.append(m)
    indices=np.zeros(len(f1),np.int32);indices[patch['incident_face_ids']]=2;indices[patch['candidate_face_ids']]=1
    cleaned.data.polygons.foreach_set('material_index',indices)
    for name,offset,energy in [('Key',(2,3,3),180),('Fill',(-3,1,1.5),140),('Back',(-1,-3,3),160)]:
        light=bpy.data.lights.new(name,'AREA');light.energy=energy;light.size=2.5
        ob=bpy.data.objects.new(name,light);scene.collection.objects.link(ob);ob.location=center+Vector(offset)
        forward=(center-ob.location).normalized();ob.rotation_euler=forward.to_track_quat('-Z','Y').to_euler()
    target=Vector(patch['centroid'])
    ray_witness=[]
    for name,location,look,scale in [('posterior-context',center+Vector((0,4,.3)),center,1.17),
        ('candidate-closeup',target+Vector((0,2,.3)),target,.035),('candidate-oblique',target+Vector((1,2,.6)),target,.035)]:
        camera.location=location;aim(look);camera_data.ortho_scale=scale;scene.render.filepath=str(OUTPUT/(name+'.png'))
        if name=='candidate-closeup':
            direction=(Vector(look)-Vector(location)).normalized();depsgraph=bpy.context.evaluated_depsgraph_get()
            for face_id in patch['candidate_face_ids']:
                point=Vector(v1[f1[face_id]].mean(axis=0));origin=point-direction*2
                hit,position,normal,index,hit_obj,matrix=scene.ray_cast(depsgraph,origin,direction,distance=4)
                ray_witness.append(dict(candidate_face_id=face_id,hit=bool(hit),first_hit_face_id=index,
                    selected_face_is_first_hit=bool(hit and index==face_id)))
        bpy.ops.render.render(write_still=True);records.append(dict(name=name+'.png',sha256=sha(OUTPUT/(name+'.png')),camera_location=list(location),target=list(look),ortho_scale=scale))
    result=dict(classification='PASS_UNREPAIRED_PATCH_PRESENTATION',views=records,structural_repair_performed=False,
        candidate_face_ids=patch['candidate_face_ids'],gold_candidate_blue_complete_incident_fans=True,closeup_raycast_witness=ray_witness)
payload(original,v0,f0);payload(cleaned,v1,f1)
for p in (source,variant):assert before[str(p)]==(p.stat().st_size,p.stat().st_mtime_ns,sha(p))
result.update(source_PLY_sha256=sha(source),variant_PLY_sha256=sha(variant),all_imported_vertex_face_payloads_exact=True,
    device='CPU',threads=4,samples=8,resolution=[768,768],modifiers=[],geometry_moved=False)
with (OUTPUT/'render-metadata.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
for a in (v0,f0,v1,f1):a._mmap.close()
print('COPY_ONLY_CPU_PRESENTATION_COMPLETE',flush=True)
