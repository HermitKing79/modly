"""Owned headless CPU Blender; geometry remains unmodified and flat shaded."""
import sys
sys.dont_write_bytecode=True
if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO plus EOF required')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_static_prop as p
stage=sys.argv[sys.argv.index('--')+1]
sys.addaudithook(p.audit(stage));p.reviewed(stage);p.success('prepare')
import bpy,numpy as np,json,math
from mathutils import Vector,Matrix
from amd_cpu_uv_bake import geometry_hash
from amd_mesh_aware_result import png
B=p.BASE;stage=sys.argv[sys.argv.index('--')+1]
if stage not in ('export',*p.RENDERS) or bpy.app.version[:3]!=(5,2,1):raise ValueError('Qualified Blender/stage required')
spec=json.loads((B/'inputs.json').read_text())
for name,h in spec['files'].items():
 if p.owner.sha(B/'inputs'/name)!=h:raise ValueError('Copy identity changed')
bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene
scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=16
scene.cycles.use_denoising=False;scene.cycles.use_adaptive_sampling=False;scene.cycles.max_bounces=3
scene.cycles.diffuse_bounces=2;scene.cycles.glossy_bounces=2;scene.cycles.seed=20261031
scene.render.threads_mode='FIXED';scene.render.threads=4;scene.render.resolution_x=scene.render.resolution_y=512
scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.image_settings.color_depth='8'
scene.render.use_compositing=False;scene.render.use_sequencer=False;scene.render.use_persistent_data=True
scene.view_settings.view_transform='Standard';scene.view_settings.exposure=0;scene.view_settings.gamma=1
bpy.context.preferences.filepaths.temporary_directory=str(B/'scratch'/stage)

def arrays(obj):
 mesh=obj.data;v=np.empty((len(mesh.vertices),3),'<f4');f=np.empty((len(mesh.polygons),3),'<u4')
 mesh.vertices.foreach_get('co',v.ravel());mesh.loops.foreach_get('vertex_index',f.ravel())
 if len(mesh.loops)!=len(f)*3 or any(len(x.vertices)!=3 for x in mesh.polygons) or obj.modifiers:raise ValueError('Unmodified triangles required')
 return v,f

def mesh(name):
 v=np.load(B/'inputs'/f'{name}-v.npy');f=np.load(B/'inputs'/f'{name}-f.npy')
 data=bpy.data.meshes.new(name);data.vertices.add(len(v));data.loops.add(len(f)*3);data.polygons.add(len(f))
 data.vertices.foreach_set('co',v.ravel());data.loops.foreach_set('vertex_index',f.ravel())
 data.polygons.foreach_set('loop_start',np.arange(len(f),dtype='i4')*3);data.polygons.foreach_set('loop_total',np.full(len(f),3,dtype='i4'));data.update()
 # Match installed Blender Mesh.from_pydata(shade_flat=True); presentation only.
 # Prepared offline after the held run; not exercised by this proof.
 data.shade_flat()
 obj=bpy.data.objects.new(name,data);scene.collection.objects.link(obj)
 vv,ff=arrays(obj)
 if not np.array_equal(v,vv) or not np.array_equal(f,ff):raise ValueError('Blender copied geometry differs')
 return obj

def active(obj):
 bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);bpy.context.view_layer.objects.active=obj

def material(name,roughness,texture=None):
 mat=bpy.data.materials.new(name);mat.use_nodes=True;mat.use_backface_culling=True
 bsdf=mat.node_tree.nodes.get('Principled BSDF');bsdf.inputs['Base Color'].default_value=(.35,.35,.35,1)
 bsdf.inputs['Roughness'].default_value=roughness;bsdf.inputs['Metallic'].default_value=0
 if texture:
  image=bpy.data.images.load(str(texture),check_existing=False);image.colorspace_settings.name='sRGB';image.pack()
  tex=mat.node_tree.nodes.new('ShaderNodeTexImage');tex.image=image;tex.interpolation='Linear';mat.node_tree.links.new(tex.outputs['Color'],bsdf.inputs['Base Color'])
 return mat

def aim(obj,target):
 forward=(Vector(target)-obj.location).normalized();right=forward.cross(Vector((0,0,1))).normalized();up=right.cross(forward)
 obj.rotation_mode='QUATERNION';obj.rotation_quaternion=Matrix((right,up,-forward)).transposed().to_quaternion()

def lights():
 D=spec['D'];center=Vector(spec['center']);world=bpy.data.worlds.new('Neutral test world');world.use_nodes=True
 world.node_tree.nodes['Background'].inputs['Color'].default_value=(.15,.15,.15,1);world.node_tree.nodes['Background'].inputs['Strength'].default_value=.4;scene.world=world
 for name,offset,power,size in [('Key',(2,-3,3),500,1.5),('Fill',(-3,-1,1),150,2),('Rim',(-1,3,3),350,1)]:
  data=bpy.data.lights.new(name,'AREA');data.energy=power*D*D;data.size=size*D;obj=bpy.data.objects.new(name,data);scene.collection.objects.link(obj);obj.location=center+Vector(offset)*D;aim(obj,center)
 data=bpy.data.cameras.new('Source frozen camera');data.type='ORTHO';data.clip_start=.00001;data.clip_end=100
 obj=bpy.data.objects.new('Source frozen camera',data);scene.collection.objects.link(obj);scene.camera=obj;return obj

def draw(obj,camera,view,path,all_objects):
 for o in all_objects:o.hide_render=o!=obj
 camera.location=view['location'];aim(camera,view['target']);camera.data.ortho_scale=view['span'];scene.render.filepath=str(path)
 state=dict(device=scene.cycles.device,modifiers=len(obj.modifiers),smooth_faces=sum(x.use_smooth for x in obj.data.polygons))
 if state['device']!='CPU' or state['modifiers'] or state['smooth_faces']:raise ValueError('CPU flat unmodified render required: '+json.dumps(state))
 if path.exists():raise ValueError('Exclusive image path')
 bpy.ops.render.render(write_still=True)

if stage=='export':
 obj=mesh('final');active(obj);v,f=arrays(obj);original=geometry_hash(np,v,f);folder=B/'asset';folder.mkdir(exist_ok=False)
 bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT')
 result=bpy.ops.uv.smart_project(angle_limit=math.radians(66),island_margin=.01,area_weight=0.,correct_aspect=True,scale_to_bounds=False)
 bpy.ops.object.mode_set(mode='OBJECT');vv,ff=arrays(obj)
 if result!={'FINISHED'} or not np.array_equal(v,vv) or not np.array_equal(f,ff):raise ValueError('UV changed geometry')
 uv=np.empty((len(f)*3,2),'<f4');obj.data.uv_layers.active.data.foreach_get('uv',uv.ravel());uv=uv.reshape(-1,3,2)
 # Preserve diagnostic UVs even if validation fails. This does not qualify them.
 with (folder/'uv.npy').open('xb') as stream:np.save(stream,uv,allow_pickle=False)
 # CPU exact overlap/density verification before exporting anything.
 stats=p.uv_check(np,uv,v,f)
 neutral=np.full((256,256,3),160,dtype='u1');png(folder/'neutral-test.png',np,neutral)
 y,x=np.indices((1024,1024));checker=np.where(((x//32+y//32)%2)[...,None],np.array([230,230,230]),np.array([45,80,110])).astype('u1');png(folder/'uv-checker.png',np,checker)
 obj.data.materials.append(material('NEUTRAL_TEST_PBR_NOT_AI_TEXTURE',.35,folder/'neutral-test.png'))
 kwargs=dict(filepath=str(folder/'experimental-static-prop.glb'),export_format='GLB',use_selection=True,export_apply=False,export_yup=True,export_texcoords=True,export_normals=True,export_tangents=False,export_vertex_color='NONE',export_materials='EXPORT',export_image_format='AUTO',export_draco_mesh_compression_enable=False)
 if bpy.ops.export_scene.gltf(**kwargs)!={'FINISHED'}:raise ValueError('GLB export failed')
 ev,ef=arrays(obj)
 if not np.array_equal(v,ev) or not np.array_equal(f,ef):raise ValueError('Export mutated copy')
 before=set(bpy.data.objects)
 if bpy.ops.import_scene.gltf(filepath=kwargs['filepath'],merge_vertices=False)!={'FINISHED'}:raise ValueError('GLB reimport failed')
 imported=[o for o in bpy.data.objects if o not in before and o.type=='MESH']
 if len(imported)!=1:raise ValueError('Single imported object')
 fresh=imported[0];iv,iff=arrays(fresh)
 if max(abs(fresh.matrix_world[i][j]-(i==j)) for i in range(4) for j in range(4))>1e-6 or geometry_hash(np,iv,iff)!=original:raise ValueError('Reimport geometry/transform differs')
 if not fresh.data.uv_layers or len(fresh.data.materials)!=1:raise ValueError('UV/material readback')
 images=[n.image for n in fresh.data.materials[0].node_tree.nodes if n.type=='TEX_IMAGE' and n.image]
 if len(images)!=1 or not images[0].packed_file:raise ValueError('Packed image readback')
 pixels=np.empty(256*256*4,'f4');images[0].pixels.foreach_get(pixels)
 if not np.isfinite(pixels).all():raise ValueError('Image pixels finite')
 camera=lights();view=spec['cameras']['whole45']
 draw(fresh,camera,view,folder/'reimport-neutral.png',[obj,fresh])
 fresh.hide_render=True;obj.data.materials.clear();obj.data.materials.append(material('UV_CHECKER_DIAGNOSTIC_ONLY',.8,folder/'uv-checker.png'))
 draw(obj,camera,view,folder/'checker-preview.png',[obj,fresh])
 p.save(folder/'blender.json',dict(classification='PASS_EXPORT_REIMPORT_PENDING_INDEPENDENT_PARSER',Blender=bpy.app.version_string,device='CPU',UV=stats,original_vertices=len(v),original_triangles=len(f),reimport_attribute_vertices=len(iv),reimport_triangles=len(iff),geometry_hash=original,identity_world_transform=True,metres_per_unit=1,modifiers=[],smooth_shading=False,normal_bake='SKIPPED_UNQUALIFIED_SOURCE',packed_images=1))
else:
 p.appearance_admission();kind,viewname=stage.split('-',1);folder=B/stage;folder.mkdir(exist_ok=False)
 objects=[mesh(name) for name in ('source','raw','final')];mat=material(kind+' matched neutral test',.8 if kind=='matte' else .12);mat.use_backface_culling=False
 for obj in objects:obj.data.materials.append(mat)
 camera=lights();view=spec['cameras'][viewname]
 for obj in objects:draw(obj,camera,view,folder/(obj.name+'.png'),objects)
 p.save(folder/'appearance.json',dict(classification='MATCHED_APPEARANCE_EVIDENCE_USER_REVIEW_PENDING',camera=view,material=kind,roughness=.8 if kind=='matte' else .12,metallic=0,source_frozen=True,device='CPU',samples=16,threads=4,denoising=False,flat_normals=True,geometry_modified=False,files={x.name:p.owner.sha(x) for x in folder.glob('*.png')}))
