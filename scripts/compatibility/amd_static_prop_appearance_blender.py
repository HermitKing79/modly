"""One matched source/raw/actual-GLB closeup; no UV, normal or mesh edits."""
import sys
sys.dont_write_bytecode=True
if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO and EOF required')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_static_prop_appearance as p
stage=sys.argv[sys.argv.index('--')+1]
if stage not in p.RENDERS:raise ValueError('Only four approved comparisons')
p.bind();sys.addaudithook(p.prior.prior._ORIGINAL_AUDIT(stage));p.approved(stage);p.old.success('authenticate');p.old.success('diagnose')
import bpy,numpy as np,json,hashlib
from mathutils import Vector,Matrix
from amd_cpu_uv_bake import geometry_hash
if bpy.app.version[:3]!=(5,2,1) or p.owner.sha(p.old.BLENDER)!=p.old.BLENDER_SHA:raise ValueError('Qualified Blender required')
B=p.BASE;plan=json.loads((B/'render-plan.json').read_text());spec=json.loads((p.INPUTS/'inputs.json').read_text())
for path,h in plan['identities'].items():
 if p.owner.sha(Path(path))!=h:raise ValueError('Frozen render input changed')
bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene
scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=plan['samples'];scene.cycles.seed=20261031
scene.cycles.use_denoising=False;scene.cycles.use_adaptive_sampling=False;scene.cycles.max_bounces=3;scene.cycles.diffuse_bounces=2;scene.cycles.glossy_bounces=2
scene.render.threads_mode='FIXED';scene.render.threads=4;scene.render.resolution_x=scene.render.resolution_y=768;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.image_settings.color_depth='8'
scene.render.use_compositing=False;scene.render.use_sequencer=False;scene.render.use_persistent_data=True
scene.view_settings.view_transform=plan['transform'];scene.view_settings.look=plan['look'];scene.view_settings.exposure=plan['exposure'];scene.view_settings.gamma=1
bpy.context.preferences.filepaths.temporary_directory=str(B/'scratch'/stage)

def arrays(obj):
 m=obj.data;v=np.empty((len(m.vertices),3),'<f4');f=np.empty((len(m.polygons),3),'<u4')
 m.vertices.foreach_get('co',v.ravel());m.loops.foreach_get('vertex_index',f.ravel())
 if len(m.loops)!=len(f)*3 or any(len(x.vertices)!=3 for x in m.polygons) or obj.modifiers:raise ValueError('Unmodified triangles required')
 return v,f

def signature(obj):
 v,f=arrays(obj);m=obj.data;normal=np.empty((len(m.corner_normals),3),'<f4');m.corner_normals.foreach_get('vector',normal.ravel())
 if normal.shape!=(len(f)*3,3) or not np.isfinite(normal).all():raise ValueError('Complete finite corner normals')
 uv=[]
 for layer in m.uv_layers:
  a=np.empty((len(m.loops),2),'<f4');layer.data.foreach_get('uv',a.ravel());uv.append(dict(name=layer.name,sha256=hashlib.sha256(a.tobytes()).hexdigest()))
 return dict(vertices=len(v),triangles=len(f),vertices_sha256=hashlib.sha256(v.tobytes()).hexdigest(),faces_sha256=hashlib.sha256(f.tobytes()).hexdigest(),normals_sha256=hashlib.sha256(normal.tobytes()).hexdigest(),UV=uv,transform=[list(row) for row in obj.matrix_world],modifiers=len(obj.modifiers))

def source_mesh(name):
 v=np.load(p.INPUTS/'inputs'/f'{name}-v.npy',allow_pickle=False);f=np.load(p.INPUTS/'inputs'/f'{name}-f.npy',allow_pickle=False)
 m=bpy.data.meshes.new(name);m.vertices.add(len(v));m.loops.add(len(f)*3);m.polygons.add(len(f))
 m.vertices.foreach_set('co',v.ravel());m.loops.foreach_set('vertex_index',f.ravel());m.polygons.foreach_set('loop_start',np.arange(len(f),dtype='i4')*3);m.polygons.foreach_set('loop_total',np.full(len(f),3,dtype='i4'));m.update();m.shade_flat()
 obj=bpy.data.objects.new(name,m);scene.collection.objects.link(obj);vv,ff=arrays(obj)
 if not np.array_equal(v,vv) or not np.array_equal(f,ff):raise ValueError('Exact source geometry copy')
 return obj

objects={'source':source_mesh('source'),'raw':source_mesh('raw')}
if bpy.ops.import_scene.gltf(filepath=str(p.GLB),merge_vertices=False)!={'FINISHED'}:raise ValueError('Actual qualified GLB import failed')
imported=[o for o in scene.objects if o.type=='MESH' and o not in objects.values()]
if len(imported)!=1:raise ValueError('Single GLB mesh')
objects['final']=imported[0];fv,ff=arrays(imported[0]);ev=np.load(p.HISTORICAL/'inputs/final-v.npy');ef=np.load(p.HISTORICAL/'inputs/final-f.npy')
if geometry_hash(np,fv,ff)!=geometry_hash(np,ev,ef) or any(abs(imported[0].matrix_world[i][j]-(i==j))>1e-7 for i in range(4) for j in range(4)):raise ValueError('Actual imported GLB geometry/transform correspondence')
before={k:signature(o) for k,o in objects.items()}
kind,region=stage.split('-',1);view=plan['cameras'][region];D=spec['D'];target=Vector(view['target']);right=Vector(view['right']);up=Vector(view['up']);out=Vector(view['view'])
rotation=Matrix((right,up,out)).transposed().to_quaternion()
mat=bpy.data.materials.new('SHARED_'+kind.upper()+'_REVIEW_OVERRIDE_NOT_AI_TEXTURE');mat.use_nodes=True;mat.use_backface_culling=False
bsdf=mat.node_tree.nodes.get('Principled BSDF');bsdf.inputs['Base Color'].default_value=(.35,.35,.35,1);bsdf.inputs['Metallic'].default_value=0;bsdf.inputs['Roughness'].default_value=plan['roughness'][kind]
for obj in objects.values():obj.data.materials.clear();obj.data.materials.append(mat)
world=bpy.data.worlds.new('Controlled low ambient');world.use_nodes=True;world.node_tree.nodes['Background'].inputs['Color'].default_value=(.08,.08,.08,1);world.node_tree.nodes['Background'].inputs['Strength'].default_value=.3;scene.world=world
lights=[]
for row in plan['lights']:
 data=bpy.data.lights.new(row['name'],'AREA');data.energy=row['power_D2']*D*D;data.size=row['size_D']*D;obj=bpy.data.objects.new(row['name'],data);scene.collection.objects.link(obj)
 x,y,z=row['offset_right_up_view_D'];obj.location=target+D*(x*right+y*up+z*out);direction=(target-obj.location).normalized();obj.rotation_euler=direction.to_track_quat('-Z','Y').to_euler()
 lights.append(dict(name=row['name'],location=list(obj.location),energy=data.energy,size=data.size,rotation_euler=list(obj.rotation_euler)))
data=bpy.data.cameras.new('Complete frozen source camera basis');data.type='ORTHO';data.ortho_scale=view['span'];data.clip_start=.00001;data.clip_end=100
camera=bpy.data.objects.new('Camera',data);scene.collection.objects.link(camera);camera.location=view['location'];camera.rotation_mode='QUATERNION';camera.rotation_quaternion=rotation;scene.camera=camera
folder=B/stage;folder.mkdir(exist_ok=False);images={}
for name,obj in objects.items():
 for other in objects.values():other.hide_render=other!=obj
 if scene.cycles.device!='CPU' or obj.modifiers:raise ValueError('CPU unmodified only')
 path=folder/(name+'.png')
 if path.exists():raise ValueError('One-shot exclusive image')
 scene.render.filepath=str(path);bpy.ops.render.render(write_still=True);images[name]=p.owner.sha(path)
after={k:signature(o) for k,o in objects.items()}
if before!=after:raise ValueError('Render changed geometry/UV/normals/transform')
p.owner.save(folder/'appearance.json',dict(classification='MATCHED_CPU_APPEARANCE_USER_REVIEW_PENDING',camera=view,camera_quaternion=list(rotation),lights=lights,material=dict(name=mat.name,roughness=plan['roughness'][kind],base_color=[.35,.35,.35,1],metallic=0,two_sided=True,shared=True,override_in_memory_only=True),transform=scene.view_settings.view_transform,look=scene.view_settings.look,exposure=scene.view_settings.exposure,device=scene.cycles.device,samples=scene.cycles.samples,threads=4,denoising=False,Blender=bpy.app.version_string,GLB_sha256=p.owner.sha(p.GLB),before=before,after=after,images=images,geometry_modified=False,UV_modified=False,normals_modified=False,normal_bake='SKIPPED'))
