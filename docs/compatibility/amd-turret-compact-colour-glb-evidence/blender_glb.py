"""One fixture/full export OR independent fresh-scene GLB import. CPU only."""
import json
import os
import struct
import sys
import time
from pathlib import Path
import bpy
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract as c

PHASE = os.environ['MODLY_GLB_PHASE']
FULL = PHASE.startswith('full-')
PATH = c.OUT / ('turret-learned-colour-compact.glb' if FULL else 'fixture.glb')

def material(shader):
    # Only portable constants replace ephemeral metallic/roughness connections.
    for name, value in [('Metallic', 0.0), ('Roughness', 0.8), ('Alpha', 1.0), ('Emission Strength', 0.0)]:
        socket = shader.inputs[name]
        for link in list(socket.links):shader.id_data.links.remove(link)
        socket.default_value = value
    link = shader.inputs['Base Color'].links
    assert len(link) == 1 and link[0].from_node.bl_idname == 'ShaderNodeVertexColor'
    assert link[0].from_node.layer_name == 'Col' and link[0].from_socket.name == 'Color'

def shader_for(obj):
    assert len(obj.data.materials) == 1
    mat = obj.data.materials[0]
    output = next(n for n in mat.node_tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output)
    assert len(output.inputs['Surface'].links) == 1
    shader = output.inputs['Surface'].links[0].from_node
    assert shader.bl_idname == 'ShaderNodeBsdfPrincipled'
    return shader

def arrays(mesh):
    xyz = np.empty((len(mesh.vertices),3), '<f4')
    indices = np.empty(len(mesh.loops), '<i4')
    totals = np.empty(len(mesh.polygons), '<i4')
    starts = np.empty(len(mesh.polygons), '<i4')
    mesh.vertices.foreach_get('co', xyz.reshape(-1))
    mesh.loops.foreach_get('vertex_index', indices)
    mesh.polygons.foreach_get('loop_total', totals)
    mesh.polygons.foreach_get('loop_start', starts)
    assert np.all(totals == 3) and np.array_equal(starts, np.arange(len(totals),dtype='i4')*3)
    assert len(mesh.color_attributes) == 1
    attr = mesh.color_attributes[0]
    assert attr.domain in ('POINT','CORNER')
    values = np.empty((len(attr.data),4), '<f4')
    attr.data.foreach_get('color_srgb', values.reshape(-1))
    assert np.isfinite(values).all() and np.all(abs(values[:,3]-1)<1e-6)
    rgb = np.rint(np.clip(values[:,:3],0,1)*np.float32(255)).astype('u1')
    return xyz, indices.reshape(-1,3), rgb, attr.domain

def inspect_glb():
    with PATH.open('rb') as f:
        magic, version, size = struct.unpack('<III',f.read(12))
        assert magic == 0x46546c67 and version == 2 and size == PATH.stat().st_size
        length, kind = struct.unpack('<II',f.read(8))
        assert kind == 0x4e4f534a and 0 < length < 16*1024**2
        doc = json.loads(f.read(length))
        binary_length, binary_kind = struct.unpack('<II',f.read(8))
        assert binary_kind == 0x004e4942 and f.tell()+binary_length == size
    assert len(doc['buffers']) == 1 and 'uri' not in doc['buffers'][0]
    assert not doc.get('images') and not doc.get('textures') and not doc.get('extensionsRequired')
    assert len(doc['meshes']) == 1 and len(doc['nodes']) == 1
    node = doc['nodes'][0]
    assert node['mesh'] == 0 and all(k not in node for k in ('matrix','translation','rotation','scale'))
    assert len(doc['meshes'][0]['primitives']) == 1 and len(doc['materials']) == 1
    primitive = doc['meshes'][0]['primitives'][0]
    assert primitive.get('mode',4) == 4
    assert set(primitive['attributes']) == {'POSITION','COLOR_0'}
    col = doc['accessors'][primitive['attributes']['COLOR_0']]
    pos = doc['accessors'][primitive['attributes']['POSITION']]
    ind = doc['accessors'][primitive['indices']]
    assert col['componentType'] == 5126 and col['type'] == 'VEC3' and not col.get('normalized',False)
    assert pos['componentType'] == 5126 and pos['type'] == 'VEC3' and pos['count'] == col['count']
    assert 0 < pos['count'] <= (c.N if FULL else 4), 'Unexpected vertex duplication remains'
    if not FULL:assert pos['count']==4, 'Fixture must reuse its four shared vertices'
    assert ind['count'] == (c.F if FULL else 3)*3
    mat = doc['materials'][0]
    assert mat.get('alphaMode','OPAQUE') == 'OPAQUE'
    pbr = mat['pbrMetallicRoughness']
    assert pbr.get('baseColorFactor',[1,1,1,1]) == [1,1,1,1]
    assert pbr['metallicFactor'] == 0 and abs(pbr['roughnessFactor']-.8)<1e-6
    assert not any(k.endswith('Texture') for k in pbr)
    return {'portable_embedded_GLB':True, 'NORMAL_omitted':True, 'COLOR_0':col, 'POSITION':pos, 'indices':ind,
            'material':mat, 'images':0, 'textures':0, 'glTF_axis_conversion':'Blender (x,y,z) to (x,z,-y); determinant +1'}

def verify_binary(xyz, faces, rgb, source_linear):
    """Independent accessor read: exact float RGB before Blender BYTE_COLOR import."""
    with PATH.open('rb') as f:
        f.seek(12);length,kind=struct.unpack('<II',f.read(8));doc=json.loads(f.read(length))
        binary_length,kind=struct.unpack('<II',f.read(8));offset=f.tell()
    primitive=doc['meshes'][0]['primitives'][0]
    def accessor(index):
        a=doc['accessors'][index];v=doc['bufferViews'][a['bufferView']]
        width={'VEC3':3,'SCALAR':1}[a['type']]
        dtype={5126:'<f4',5125:'<u4',5123:'<u2'}[a['componentType']]
        size=np.dtype(dtype).itemsize*width*a['count']
        assert 'sparse' not in a and not a.get('normalized',False) and v['buffer']==0
        assert v.get('byteStride',width*np.dtype(dtype).itemsize)==width*np.dtype(dtype).itemsize
        relative=a.get('byteOffset',0)
        assert relative+size<=v['byteLength'] and v.get('byteOffset',0)+v['byteLength']<=binary_length
        return np.memmap(PATH,mode='r',offset=offset+v.get('byteOffset',0)+relative,dtype=dtype,shape=(a['count'],width))
    pos=accessor(primitive['attributes']['POSITION'])
    colour=accessor(primitive['attributes']['COLOR_0'])
    indices=accessor(primitive['indices']).reshape(-1,3).astype('<i4')
    converted=np.empty_like(pos);converted[:,0]=pos[:,0];converted[:,1]=-pos[:,2];converted[:,2]=pos[:,1]
    srgb=np.where(colour<=.0031308,colour*12.92,1.055*np.power(colour,1/2.4)-.055)
    display=np.rint(np.clip(srgb,0,1)*255).astype('u1')
    result=c.compare(xyz,faces,rgb,converted,indices,display,'POINT',source_linear,colour)
    assert result['maximum_linear_RGB_error']==0,'GLB float RGB must be bit exact'
    return result

began = time.monotonic()
assert bpy.app.version[:2] == (5,2)
xyz, faces, rgb = c.original() if FULL else c.fixture()
if PHASE.endswith('-export'):
    assert not PATH.exists()
    if FULL:
        assert Path(bpy.data.filepath).resolve() == c.SCENE.resolve()
        objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
        assert len(objects) == 1
        obj = objects[0]
        assert not obj.modifiers
        actual = arrays(obj.data)
        assert actual[3] == 'POINT' and all(np.array_equal(a,b) for a,b in zip(actual[:3],(xyz,faces,rgb)))
        linear = np.empty((c.N,4),'<f4')
        obj.data.color_attributes['Col'].data.foreach_get('color',linear.reshape(-1))
        assert c.ahash(linear) == '6c2f745064a95a3b077ee8682346f22971e95a81c56ce7e2b889eae5c36b9fa1'
    else:
        bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
        mesh = bpy.data.meshes.new('Representative three noncoplanar triangles')
        mesh.from_pydata(xyz.tolist(),[],faces.tolist());mesh.update()
        obj = bpy.data.objects.new('Learned RGB fixture',mesh);bpy.context.collection.objects.link(obj)
        col = mesh.color_attributes.new(name='Col',type='FLOAT_COLOR',domain='POINT')
        values = np.ones((len(xyz),4),dtype='f4');values[:,:3]=rgb/np.float32(255)
        col.data.foreach_set('color_srgb',values.reshape(-1))
        mat = bpy.data.materials.new('Original supported RGB node connection');mat.use_nodes=True;mesh.materials.append(mat)
        shader = mat.node_tree.nodes.get('Principled BSDF')
        vc = mat.node_tree.nodes.new('ShaderNodeVertexColor');vc.layer_name='Col'
        mat.node_tree.links.new(vc.outputs['Color'],shader.inputs['Base Color'])
    assert max(abs(obj.matrix_world[i][j]-(i==j)) for i in range(4) for j in range(4)) == 0
    linear = np.empty((len(xyz),4),'<f4')
    obj.data.color_attributes['Col'].data.foreach_get('color',linear.reshape(-1))
    source_linear_path = c.OUT/('source-linear-RGB.npy' if FULL else 'fixture-linear-RGB.npy')
    assert not source_linear_path.exists()
    np.save(source_linear_path,np.ascontiguousarray(linear[:,:3]),allow_pickle=False)
    material(shader_for(obj))
    bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);bpy.context.view_layer.objects.active=obj
    settings = dict(filepath=str(PATH),export_format='GLB',use_selection=True,export_materials='EXPORT',
        export_vertex_color='MATERIAL',export_all_vertex_colors=False,export_image_format='AUTO',
        export_yup=True,export_apply=False,export_normals=False,export_tangents=False,export_texcoords=False,
        export_attributes=False,export_shared_accessors=True,export_animations=False,export_skins=False,
        export_morph=False,use_mesh_edges=False,use_mesh_vertices=False,export_cameras=False,export_lights=False,
        export_draco_mesh_compression_enable=False,export_meshopt_compression_enable=False)
    smooth=np.empty(len(obj.data.polygons),dtype=np.bool_)
    obj.data.polygons.foreach_get('use_smooth',smooth)
    assert not np.any(smooth) and not obj.data.has_custom_normals, 'Authored smooth/custom normals would be discarded'
    assert bpy.ops.export_scene.gltf(**settings) == {'FINISHED'}
    info = inspect_glb()
    record={'classification':'PASS','phase':PHASE,'export_calls':1,'source_geometry_and_RGB_exact':True,
            'source_smooth_faces':int(smooth.sum()),'source_custom_normals':False,'settings':settings,**info}
else:
    assert not bpy.data.filepath
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    info = inspect_glb()
    assert bpy.ops.import_scene.gltf(filepath=str(PATH),import_pack_images=False,merge_vertices=False)=={'FINISHED'}
    meshes = [o for o in bpy.context.scene.objects if o.type=='MESH']
    assert len(meshes)==1 and len(bpy.context.scene.objects)==1
    obj=meshes[0]
    assert not obj.modifiers and max(abs(obj.matrix_world[i][j]-(i==j)) for i in range(4) for j in range(4))<1e-7
    smooth=np.empty(len(obj.data.polygons),dtype=np.bool_)
    obj.data.polygons.foreach_get('use_smooth',smooth)
    assert not np.any(smooth) and not obj.data.has_custom_normals
    target_xyz,target_faces,target_rgb,domain=arrays(obj.data)
    source_linear=np.load(c.OUT/('source-linear-RGB.npy' if FULL else 'fixture-linear-RGB.npy'),mmap_mode='r',allow_pickle=False)
    # Importer constructs BYTE_COLOR; exact original 8-bit sRGB is checked above.
    # The GLB FLOAT32 payload itself must preserve source linear RGB exactly.
    comparison=c.compare(xyz,faces,rgb,target_xyz,target_faces,target_rgb,domain)
    binary_comparison=verify_binary(xyz,faces,rgb,source_linear)
    shader=shader_for(obj)
    links=shader.inputs['Base Color'].links
    assert len(links)==1 and links[0].from_node.bl_idname=='ShaderNodeVertexColor'
    assert links[0].from_socket.name=='Color'
    layer=links[0].from_node.layer_name
    assert layer==obj.data.color_attributes[0].name or (layer=='' and obj.data.color_attributes.render_color_index==0)
    assert not shader.inputs['Metallic'].is_linked and not shader.inputs['Roughness'].is_linked
    assert shader.inputs['Metallic'].default_value==0 and abs(shader.inputs['Roughness'].default_value-.8)<1e-6
    assert shader.inputs['Alpha'].default_value==1 and not shader.inputs['Alpha'].is_linked
    record={'classification':'PASS','phase':PHASE,'fresh_scene_import':True,
            'imported_smooth_faces':int(smooth.sum()),'imported_custom_normals':False,
            'standalone_without_source_blend':True,'learned_RGB_shader_link_verified':True,
            'shader_colour_layer':layer,'render_colour_index':obj.data.color_attributes.render_color_index,
            'binary_payload_verification':binary_comparison,**comparison,**info}
record.update(Blender_version=bpy.app.version_string,elapsed_seconds=time.monotonic()-began,
              path=str(PATH),bytes=PATH.stat().st_size,sha256=c.digest(PATH),CPU_only=True,no_render=True)
c.save(c.E/(PHASE+'.json'),record)
print(json.dumps(record),flush=True)
