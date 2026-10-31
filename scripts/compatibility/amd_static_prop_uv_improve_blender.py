"""One first-party Blender ABF UV candidate, then separately qualified export."""
import sys
sys.dont_write_bytecode=True
if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO/EOF required')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_static_prop_uv_improve as p
stage=sys.argv[sys.argv.index('--')+1]
if stage not in ('uv','export'):raise ValueError('Only UV or conditional export')
p.bind(stage);sys.addaudithook(p.prior._ORIGINAL_AUDIT(stage));p.reviewed(stage);p.old.success('prepare')
import bpy,numpy as np,json
from amd_cpu_uv_bake import geometry_hash
from amd_mesh_aware_result import png
if bpy.app.version[:3]!=(5,2,1):raise ValueError('Qualified Blender required')
B=p.BASE;plan=json.loads((B/'seam-plan.json').read_text());strategy=json.loads((B/'strategy.json').read_text())
for name,h in plan['geometry_files'].items():
 if p.owner.sha(B/'inputs'/name)!=h:raise ValueError('Frozen copy changed')
bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene
scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.render.threads_mode='FIXED';scene.render.threads=4
bpy.context.preferences.filepaths.temporary_directory=str(B/'scratch'/stage)
v=np.load(B/'inputs/final-v.npy',allow_pickle=False);f=np.load(B/'inputs/final-f.npy',allow_pickle=False)
mesh=bpy.data.meshes.new('Frozen AI-derived static prop');mesh.vertices.add(len(v));mesh.loops.add(3*len(f));mesh.polygons.add(len(f))
mesh.vertices.foreach_set('co',v.ravel());mesh.loops.foreach_set('vertex_index',f.ravel());mesh.polygons.foreach_set('loop_start',np.arange(len(f),dtype='i4')*3);mesh.polygons.foreach_set('loop_total',np.full(len(f),3,dtype='i4'));mesh.update(calc_edges=True);mesh.shade_flat()
obj=bpy.data.objects.new('LOWER_DETAIL_STATIC_PROP_EXPERIMENTAL',mesh);scene.collection.objects.link(obj);obj.select_set(True);bpy.context.view_layer.objects.active=obj

def arrays(o):
 m=o.data;vv=np.empty((len(m.vertices),3),'<f4');ff=np.empty((len(m.polygons),3),'<u4')
 m.vertices.foreach_get('co',vv.ravel());m.loops.foreach_get('vertex_index',ff.ravel())
 if len(m.loops)!=len(ff)*3 or len(o.modifiers):raise ValueError('Unmodified triangles required')
 return vv,ff
def unchanged():
 vv,ff=arrays(obj)
 if not np.array_equal(v,vv) or not np.array_equal(f,ff):raise ValueError('Geometry changed')
def uv_array():
 a=np.empty((len(mesh.loops),2),'<f4');mesh.uv_layers.active.data.foreach_get('uv',a.ravel());return a.reshape(-1,3,2)
def save_uv(name):
 a=uv_array()
 with (B/'asset'/name).open('xb') as out:np.save(out,a,allow_pickle=False)
 return a
unchanged()
if stage=='uv':
 folder=B/'asset';folder.mkdir(exist_ok=False);seams={tuple(x) for x in plan['seam_edges']}
 actual_edges={tuple(sorted(e.vertices)) for e in mesh.edges}
 if len(actual_edges)!=1500 or not seams<=actual_edges:raise ValueError('Derived edge identity')
 for e in mesh.edges:e.use_seam=tuple(sorted(e.vertices)) in seams
 bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT')
 result=bpy.ops.uv.unwrap(method='ANGLE_BASED',fill_holes=False,correct_aspect=True,use_subsurf_data=False,use_original_bounds=False,margin_method='FRACTION',margin=.01)
 bpy.ops.object.mode_set(mode='OBJECT');unchanged();save_uv('uv-abf.npy')
 if result!={'FINISHED'}:raise ValueError('ABF unwrap refused')
 bpy.ops.object.mode_set(mode='EDIT')
 result=bpy.ops.uv.average_islands_scale(scale_uv=False,shear=False)
 bpy.ops.object.mode_set(mode='OBJECT');unchanged();save_uv('uv-normalized.npy')
 if result!={'FINISHED'}:raise ValueError('Uniform island normalization refused')
 bpy.ops.object.mode_set(mode='EDIT')
 result=bpy.ops.uv.pack_islands(rotate=True,rotate_method='ANY',scale=True,merge_overlap=False,margin_method='FRACTION',margin=.01,pin=False,shape_method='CONCAVE',udim_source='CLOSEST_UDIM')
 bpy.ops.object.mode_set(mode='OBJECT');unchanged();uv=save_uv('uv.npy')
 if result!={'FINISHED'}:raise ValueError('UV pack refused')
 p.uv_check(np,uv,v,f)
 p.owner.save(folder/'uv-blender.json',dict(classification='PASS_UV_PENDING_INDEPENDENT_AUDIT',Blender=bpy.app.version_string,CPU=True,geometry_exact=True,vertices=len(v),triangles=len(f),ordered_geometry_hash=geometry_hash(np,v,f),seams=len(seams),strategy=strategy))
else:
 p.require_uv_pass();uv=np.load(B/'asset/uv.npy',allow_pickle=False);layer=mesh.uv_layers.new(name='QualifiedUV');layer.data.foreach_set('uv',uv.ravel());unchanged()
 if not np.array_equal(uv,uv_array()):raise ValueError('Assigned UV identity')
 folder=B/'asset';png(folder/'neutral-test.png',np,np.full((256,256,3),160,dtype='u1'))
 mat=bpy.data.materials.new('NEUTRAL_TEST_PBR_NOT_AI_TEXTURE');mat.use_nodes=True;mat.use_backface_culling=True;bsdf=mat.node_tree.nodes.get('Principled BSDF');bsdf.inputs['Metallic'].default_value=0;bsdf.inputs['Roughness'].default_value=.35
 image=bpy.data.images.load(str(folder/'neutral-test.png'),check_existing=False);image.colorspace_settings.name='sRGB';image.pack();tex=mat.node_tree.nodes.new('ShaderNodeTexImage');tex.image=image;mat.node_tree.links.new(tex.outputs['Color'],bsdf.inputs['Base Color']);mesh.materials.append(mat)
 if bpy.ops.export_scene.gltf(filepath=str(folder/'experimental-static-prop.glb'),export_format='GLB',use_selection=True,export_apply=False,export_yup=True,export_texcoords=True,export_normals=True,export_tangents=False,export_vertex_color='NONE',export_materials='EXPORT',export_image_format='AUTO',export_draco_mesh_compression_enable=False)!={'FINISHED'}:raise ValueError('GLB export refused')
 unchanged();before=set(bpy.data.objects)
 if bpy.ops.import_scene.gltf(filepath=str(folder/'experimental-static-prop.glb'),merge_vertices=False)!={'FINISHED'}:raise ValueError('GLB import refused')
 imported=[o for o in bpy.data.objects if o not in before and o.type=='MESH']
 if len(imported)!=1:raise ValueError('One imported mesh required')
 fresh=imported[0];iv,iff=arrays(fresh)
 if geometry_hash(np,iv,iff)!=geometry_hash(np,v,f) or max(abs(fresh.matrix_world[i][j]-(i==j)) for i in range(4) for j in range(4))>1e-6:raise ValueError('Reimport geometry/transform differs')
 if not fresh.data.uv_layers or len(fresh.data.materials)!=1:raise ValueError('Imported UV/material absent')
 images=[n.image for n in fresh.data.materials[0].node_tree.nodes if n.type=='TEX_IMAGE' and n.image]
 if len(images)!=1 or not images[0].packed_file:raise ValueError('Imported packed image absent')
 iuv=np.empty((len(fresh.data.loops),2),'<f4');fresh.data.uv_layers.active.data.foreach_get('uv',iuv.ravel())
 normals=np.empty((len(fresh.data.loops),3),'<f4');fresh.data.corner_normals.foreach_get('vector',normals.ravel())
 for name,data in [('reimport-v.npy',iv),('reimport-f.npy',iff),('reimport-uv.npy',iuv.reshape(-1,3,2)),('reimport-normals.npy',normals.reshape(-1,3,3))]:
  with (folder/name).open('xb') as out:np.save(out,data,allow_pickle=False)
 with (folder/'reimport-packed.png').open('xb') as out:out.write(images[0].packed_file.data)
 p.owner.save(folder/'export-blender.json',dict(classification='PASS_BLENDER_EXPORT_REIMPORT_PENDING_INDEPENDENT_VERIFY',source_vertices=len(v),source_triangles=len(f),import_attribute_vertices=len(iv),import_triangles=len(iff),geometry_hash=geometry_hash(np,v,f),identity_transform=True,units='metre convention; physical size unqualified',packed_images=1,CPU=True,appearance='USER_APPEARANCE_APPROVAL_PENDING'))
