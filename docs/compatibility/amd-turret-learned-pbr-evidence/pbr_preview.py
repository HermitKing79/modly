"""One same-lighting learned-PBR CPU render, or a non-rendering saved-scene readback."""
import os,sys,json,time
from pathlib import Path
import bpy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pbr_contract as c
VERIFY=os.environ.get('MODLY_PBR_READBACK')=='1';began=time.monotonic()
assert Path(bpy.data.filepath).resolve()==(c.BLEND if VERIFY else c.INPUT).resolve()
assert c.digest(c.INPUT)==c.INPUT_SHA and c.digest(c.FIELD)==c.FIELD_SHA
scene=bpy.context.scene
meshes=[o for o in scene.objects if o.type=='MESH'];assert len(meshes)==1
obj=meshes[0];mesh=obj.data
assert not obj.modifiers and max(abs(obj.matrix_world[i][j]-(i==j)) for i in range(4) for j in range(4))<1e-7
assert (len(mesh.vertices),len(mesh.polygons),len(mesh.loops))==(c.N,c.F,c.F*3)
receipt=json.loads((c.OLD/'conversion.json').read_text())
assert receipt['classification']=='PASS' and receipt['coordinate_payload_sha256']=='c87fa21d6f863aced647b48e0bb2edebe32523d36c742e350d39ccfea3fc0934'
expected=c.raw_channels(c.load_features())
material=mesh.materials[0];assert len(mesh.materials)==1 and material.use_nodes
nodes,links=material.node_tree.nodes,material.node_tree.links
output=next(n for n in nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output)
shader=output.inputs['Surface'].links[0].from_node;assert shader.bl_idname=='ShaderNodeBsdfPrincipled'
assert shader.inputs['Emission Strength'].default_value==0 and not shader.inputs['Alpha'].is_linked and shader.inputs['Alpha'].default_value==1
assert len(shader.inputs['Base Color'].links)==1
rgb_link=shader.inputs['Base Color'].links[0];assert rgb_link.from_node.bl_idname=='ShaderNodeVertexColor' and rgb_link.from_node.layer_name=='Col' and rgb_link.from_socket==rgb_link.from_node.outputs['Color']
rgb_endpoint=[rgb_link.from_node.name,rgb_link.from_socket.name,rgb_link.to_node.name,rgb_link.to_socket.name]
def geometry_rgb():
 co=np.empty(c.N*3,'<f4');fi=np.empty(c.F*3,'<i4');totals=np.empty(c.F,'<i4');starts=np.empty(c.F,'<i4')
 mesh.vertices.foreach_get('co',co);mesh.loops.foreach_get('vertex_index',fi);mesh.polygons.foreach_get('loop_total',totals);mesh.polygons.foreach_get('loop_start',starts)
 assert np.all(totals==3) and np.array_equal(starts,np.arange(c.F,dtype=np.int32)*3)
 result={'vertex_payload_sha256':c.array_digest(co),'face_payload_sha256':c.array_digest(fi)}
 assert all(result[k]==receipt[k] for k in result)
 colour=mesh.color_attributes['Col'];assert colour.domain=='POINT' and colour.data_type=='FLOAT_COLOR'
 rgba=np.empty(c.N*4,np.float32);colour.data.foreach_get('color_srgb',rgba)
 quant=np.rint(np.clip(rgba.reshape(-1,4)[:,:3],0,1)*np.float32(255)).astype(np.uint8)
 assert c.array_digest(quant)==receipt['RGB_payload_sha256']
 result['RGB_payload_sha256']=c.array_digest(quant)
 colour.data.foreach_get('color',rgba);result['linear_RGBA_sha256']=c.array_digest(rgba)
 link=shader.inputs['Base Color'].links[0];assert [link.from_node.name,link.from_socket.name,link.to_node.name,link.to_socket.name]==rgb_endpoint
 return result
def settings():
 cameras=[];lights=[]
 for o in sorted(scene.objects,key=lambda o:o.name):
  if o.type=='CAMERA':cameras.append({'name':o.name,'matrix_world':[list(x) for x in o.matrix_world],'type':o.data.type,'ortho_scale':o.data.ortho_scale,'clip':[o.data.clip_start,o.data.clip_end]})
  if o.type=='LIGHT':lights.append({'name':o.name,'matrix_world':[list(x) for x in o.matrix_world],'type':o.data.type,'energy':o.data.energy,'color':list(o.data.color),'size':o.data.size,'use_shadow':o.data.use_shadow})
 bg=scene.world.node_tree.nodes.get('Background')
 return {'cameras':cameras,'lights':lights,'world_color':list(bg.inputs['Color'].default_value),'world_strength':bg.inputs['Strength'].default_value,'view_transform':scene.view_settings.view_transform,'look':scene.view_settings.look,'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma,'resolution':[scene.render.resolution_x,scene.render.resolution_y,scene.render.resolution_percentage],'engine':scene.render.engine,'device':scene.cycles.device,'samples':scene.cycles.samples,'seed':scene.cycles.seed,'animated_seed':scene.cycles.use_animated_seed,'adaptive_sampling':scene.cycles.use_adaptive_sampling,'max_bounces':scene.cycles.max_bounces,'diffuse_bounces':scene.cycles.diffuse_bounces,'glossy_bounces':scene.cycles.glossy_bounces,'denoising':scene.cycles.use_denoising,'denoiser':scene.cycles.denoiser,'denoising_use_gpu':scene.cycles.denoising_use_gpu,'threads':scene.render.threads,'frame':scene.frame_current}
def inspect_attributes():
 attrs={}
 for name,values in expected.items():
  a=mesh.attributes[name];assert a.data_type=='FLOAT' and a.domain=='POINT' and len(a.data)==c.N
  actual=np.empty(c.N,np.float32);a.data.foreach_get('value',actual)
  assert c.array_digest(actual)==c.array_digest(values)
  attrs[name]={'source_column':c.CHANNELS[name],'data_type':'FLOAT','domain':'POINT','sha256':c.array_digest(actual),'min':float(actual.min()),'max':float(actual.max()),'mean':float(actual.mean(dtype=np.float64)),'outside_0_1':int(((actual<0)|(actual>1)).sum()),'raw_unclamped':True}
 for attr,target in [('Trellis_metallic_raw','Metallic'),('Trellis_roughness_raw','Roughness')]:
  edge=shader.inputs[target].links;assert len(edge)==1
  clamp=edge[0].from_node;assert clamp.bl_idname=='ShaderNodeClamp' and clamp.clamp_type=='MINMAX' and edge[0].from_socket==clamp.outputs['Result']
  assert clamp.inputs['Min'].default_value==0 and clamp.inputs['Max'].default_value==1 and not clamp.inputs['Min'].is_linked and not clamp.inputs['Max'].is_linked
  source=clamp.inputs['Value'].links;assert len(source)==1 and source[0].from_node.bl_idname=='ShaderNodeAttribute' and source[0].from_node.attribute_type=='GEOMETRY' and source[0].from_node.attribute_name==attr and source[0].from_socket==source[0].from_node.outputs['Fac']
 assert not shader.inputs['Alpha'].is_linked and shader.inputs['Alpha'].default_value==1 and shader.inputs['Emission Strength'].default_value==0
 return attrs
geometry_before=geometry_rgb();initial_settings=settings()
assert initial_settings['engine']=='CYCLES' and initial_settings['device']=='CPU' and initial_settings['samples']==64 and initial_settings['resolution']==[640,640,100]
assert initial_settings['denoising'] and initial_settings['denoiser']=='OPENIMAGEDENOISE' and initial_settings['denoising_use_gpu'] is False and initial_settings['threads']==4
side=bpy.data.objects['Preview side'];baseline=json.loads((c.OLD/'render-metadata.json').read_text());side_meta=next(v for v in baseline['views'] if v['name']=='side')
assert max(abs(a-b) for a,b in zip(side.location,side_meta['camera_location']))<1e-6 and abs(side.data.ortho_scale-side_meta['ortho_scale'])<1e-6
assert max(abs(a-b) for a,b in zip(side.rotation_quaternion,side_meta['camera_quaternion']))<1e-6
scene.camera=side
if not VERIFY:
 assert not c.BLEND.exists() and not c.PNG.exists() and not (c.OUT/'render-metadata.json').exists()
 assert shader.inputs['Metallic'].default_value==0 and not shader.inputs['Metallic'].is_linked and abs(shader.inputs['Roughness'].default_value-.8)<1e-6 and not shader.inputs['Roughness'].is_linked
 for name,values in expected.items():
  assert name not in mesh.attributes
  a=mesh.attributes.new(name=name,type='FLOAT',domain='POINT');a.data.foreach_set('value',values)
 for attr,target in [('Trellis_metallic_raw','Metallic'),('Trellis_roughness_raw','Roughness')]:
  node=nodes.new('ShaderNodeAttribute');node.name=attr;node.attribute_type='GEOMETRY';node.attribute_name=attr
  clamp=nodes.new('ShaderNodeClamp');clamp.name=target+' preview range';clamp.clamp_type='MINMAX';clamp.inputs['Min'].default_value=0;clamp.inputs['Max'].default_value=1
  links.new(node.outputs['Fac'],clamp.inputs['Value']);links.new(clamp.outputs['Result'],shader.inputs[target])
 attributes=inspect_attributes();assert settings()==initial_settings and geometry_rgb()==geometry_before
 scene.render.filepath=str(c.PNG);render_start=time.monotonic();bpy.ops.render.render(write_still=True);render_seconds=time.monotonic()-render_start
 assert c.PNG.is_file();assert inspect_attributes()==attributes and geometry_rgb()==geometry_before and settings()==initial_settings
 assert bpy.ops.wm.save_as_mainfile(filepath=str(c.BLEND),check_existing=False,compress=True)=={'FINISHED'}
 assert c.BLEND.is_file() and geometry_rgb()==geometry_before and inspect_attributes()==attributes
 record={'classification':'PASS','mode':'direct-row learned RGB + learned metallic/roughness PBR preview; no trilinear bake','source_blend_sha256':c.INPUT_SHA,'material_sha256':c.FIELD_SHA,'vertex_count':c.N,'face_count':c.F,'geometry_and_RGB':geometry_before,'RGB_link_preserved':rgb_endpoint,'attributes':attributes,'shader_clamps_only':{'Metallic':[0,1],'Roughness':[0,1]},'alpha_retained_raw_and_unlinked':True,'shader_alpha':1,'emission_strength':0,'baseline_scene_settings':initial_settings,'scene_settings_after':settings(),'active_camera':scene.camera.name,'render_seconds':render_seconds,'elapsed_seconds':time.monotonic()-began,'Blender_version':bpy.app.version_string,'outputs':{p.name:{'path':str(p),'bytes':p.stat().st_size,'sha256':c.digest(p)} for p in (c.PNG,c.BLEND)}}
 c.save(c.OUT/'render-metadata.json',record);c.save(c.E/'render-metadata.json',record)
else:
 prior=json.loads((c.OUT/'render-metadata.json').read_text());assert prior['classification']=='PASS'
 assert geometry_rgb()==prior['geometry_and_RGB'] and settings()==prior['baseline_scene_settings'] and inspect_attributes()==prior['attributes'] and scene.camera.name=='Preview side'
 c.save(c.E/'scene-readback.json',{'classification':'PASS','no_render':True,'saved_scene':str(c.BLEND),'geometry_and_RGB':geometry_rgb(),'attributes':inspect_attributes(),'RGB_link_preserved':rgb_endpoint,'settings_match_original':True,'alpha':1,'emission_strength':0,'Blender_version':bpy.app.version_string})
assert c.digest(c.INPUT)==c.INPUT_SHA and c.digest(c.FIELD)==c.FIELD_SHA
print('CPU_LEARNED_PBR_'+('READBACK' if VERIFY else 'RENDER')+'=PASS',flush=True)
