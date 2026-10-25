"""Read-only full-reference layered rays and color-coded CPU inspection."""
import sys
if sys.stdin.buffer.read(4)!=b'GO\n':raise RuntimeError('Owned GO/EOF required before Blender imports')
import array,hashlib,json,os,time
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix,Vector
from mathutils.bvhtree import BVHTree
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import amd_zombie_wound_source as api
BASE=api.BASE;OLD=api.OLD
MODE=os.environ.get('MODLY_ZOMBIE_WOUND_SOURCE_MODE')
if MODE not in ('rays','render'):raise ValueError('Explicit read-only CPU mode required')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,row):
    with p.open('x') as stream:json.dump(row,stream,indent=2,allow_nan=False)
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
inputs=read(BASE/'inputs.json');assert sha(OLD/'variant/mesh.ply')==api.BASELINE_SHA
v=np.load(OLD/'variant/vertices.npy',mmap_mode='r',allow_pickle=False);f=np.load(OLD/'variant/faces.npy',mmap_mode='r',allow_pickle=False)
before={str(p):(p.stat().st_size,p.stat().st_mtime_ns,sha(p)) for p in [OLD/'variant/mesh.ply',OLD/'variant/vertices.npy',OLD/'variant/faces.npy']}
bpy.ops.wm.read_factory_settings(use_empty=True);bpy.context.preferences.filepaths.temporary_directory=str(BASE)
if MODE=='rays':
    began=time.perf_counter();tree=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True,epsilon=0.)
    cap=inputs['layer_cap'];epsilon=inputs['layer_epsilon'];resolution=128
    faces=np.full((3,resolution,resolution,cap),-1,np.int32);depth=np.full(faces.shape,np.nan,np.float64)
    origins=np.zeros((3,resolution,resolution,3),np.float64);directions=[];truncated=np.zeros(faces.shape[:3],bool)
    for a,row in enumerate(inputs['views']):
        location=Vector(row['camera_location']);direction=(Vector(row['target'])-location).normalized();directions.append(list(direction))
        right=direction.cross(Vector((0,0,1))).normalized();up=right.cross(direction);scale=row['ortho_scale']
        for j in range(resolution):
            for i in range(resolution):
                origin=location+right*((i+.5)/resolution-.5)*scale+up*(.5-(j+.5)/resolution)*scale
                origins[a,j,i]=origin;current=origin.copy();last=-1.
                for layer in range(cap):
                    hit,normal,face,distance=tree.ray_cast(current,direction,4.)
                    if hit is None:break
                    t=float((hit-origin).dot(direction))
                    if t<=last:raise RuntimeError('Layer continuation did not advance')
                    faces[a,j,i,layer]=face;depth[a,j,i,layer]=t;last=t;current=hit+direction*epsilon
                if faces[a,j,i,-1]>=0:truncated[a,j,i]=tree.ray_cast(current,direction,4.)[0] is not None
    for name,a in [('layer-face-ids.npy',faces),('layer-depths.npy',depth),('ray-origins.npy',origins),('ray-directions.npy',np.asarray(directions,np.float64)),('truncated-rays.npy',truncated)]:api.prior.save_npy(BASE/name,a)
    counts=(faces>=0).sum(axis=3)
    save(BASE/'layer-tracing.json',dict(classification='PASS_BOUNDED_TWO_SIDED_LAYER_TRACING',seconds=time.perf_counter()-began,
        rays=3*resolution*resolution,source_vertices=len(v),source_triangles=len(f),cap=cap,epsilon=epsilon,
        truncated_rays=int(truncated.sum()),layer_count_histogram={str(k):int((counts==k).sum()) for k in np.unique(counts)},
        source_winding_used_for_visibility=False,geometry_modified=False,
        limitations='BVH continuation omits sub-epsilon/coincident events; layers are ray events rather than automatically identified physical sheets'))

if MODE=='render':
    analysis=read(BASE/'analysis.json');assert analysis['source_geometry_changed'] is False
    output=BASE/'renders';output.mkdir(exist_ok=False)
    assert bpy.ops.wm.ply_import(filepath=str(OLD/'variant/mesh.ply'),merge_verts=False,global_scale=1.,use_scene_unit=False,forward_axis='Y',up_axis='Z')=={'FINISHED'}
    obj=next(o for o in bpy.context.scene.objects if o.type=='MESH')
    assert np.array_equal(np.asarray(obj.matrix_world),np.eye(4)) and not obj.modifiers
    def payload():
        assert (len(obj.data.vertices),len(obj.data.polygons),len(obj.data.loops))==(len(v),len(f),len(f)*3)
        co=array.array('f',[0])*(len(v)*3);idx=array.array('i',[0])*(len(f)*3)
        obj.data.vertices.foreach_get('co',co);obj.data.loops.foreach_get('vertex_index',idx)
        assert co.tobytes()==v.tobytes() and idx.tobytes()==f.tobytes() and not obj.modifiers
    payload();obj.data.polygons.foreach_set('use_smooth',[False]*len(f))
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=8
    scene.cycles.use_denoising=False;scene.cycles.use_adaptive_sampling=False;scene.cycles.seed=0
    scene.cycles.max_bounces=2;scene.cycles.diffuse_bounces=1;scene.cycles.glossy_bounces=1
    scene.render.threads_mode='FIXED';scene.render.threads=4;scene.render.resolution_x=scene.render.resolution_y=512;scene.render.resolution_percentage=100
    scene.render.film_transparent=False;scene.render.use_compositing=False;scene.render.use_sequencer=False;scene.render.use_persistent_data=False
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    scene.view_settings.view_transform='AgX';scene.view_settings.exposure=-.75;scene.view_settings.gamma=1
    world=bpy.data.worlds.new('Matched neutral background');world.use_nodes=True;scene.world=world
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(.12,.12,.12,1);world.node_tree.nodes['Background'].inputs['Strength'].default_value=.5
    camera_data=bpy.data.cameras.new('Original matched cameras');camera_data.type='ORTHO';camera_data.clip_start=.001;camera_data.clip_end=20
    camera=bpy.data.objects.new('Original matched cameras',camera_data);scene.collection.objects.link(camera);scene.camera=camera
    target=Vector(inputs['views'][1]['target'])
    for name,offset,energy in [('Key',(1,-2,2),130),('Fill',(-2,-1,.5),90),('Edge',(0,1,2),110)]:
        light=bpy.data.lights.new(name,'AREA');light.energy=energy;light.size=2.5
        ob=bpy.data.objects.new(name,light);scene.collection.objects.link(ob);ob.location=target+Vector(offset);ob.rotation_euler=(target-ob.location).to_track_quat('-Z','Y').to_euler()
    def material(name,color):
        m=bpy.data.materials.new(name);m.use_nodes=True;node=m.node_tree.nodes.get('Principled BSDF');node.inputs['Base Color'].default_value=(*color,1);node.inputs['Roughness'].default_value=.9;return m
    grey=material('Neutral reference',(.35,.35,.35))
    role_colors=[('Uncertain',(.6,.16,.7)),('Exterior candidate',(.12,.48,.9)),('Probable rim',(.95,.47,.04)),('Probable cavity',(.06,.75,.46)),('Outside audit',(.25,.25,.25))]
    materials=[material(name,color) for name,color in role_colors]
    labels=np.load(BASE/'face-role-labels.npy',allow_pickle=False).astype(np.int32);assert labels.shape==(len(f),)
    frames=[]
    for row in inputs['views']:
        camera.location=Vector(row['camera_location']);direction=(Vector(row['target'])-camera.location).normalized();right=direction.cross(Vector((0,0,1))).normalized();up=right.cross(direction)
        camera.rotation_mode='QUATERNION';camera.rotation_quaternion=Matrix((right,up,-direction)).transposed().to_quaternion();camera_data.ortho_scale=row['ortho_scale']
        for mode in ('neutral','roles'):
            obj.data.materials.clear()
            for m in ([grey] if mode=='neutral' else materials):obj.data.materials.append(m)
            obj.data.polygons.foreach_set('material_index',np.zeros(len(f),np.int32) if mode=='neutral' else labels)
            name=f'full-reference-{row["angle"]:+03d}-{mode}.png';scene.render.filepath=str(output/name)
            assert scene.cycles.device=='CPU';bpy.ops.render.render(write_still=True)
            frames.append(dict(angle=row['angle'],mode=mode,name=name,sha256=sha(output/name),camera_location=row['camera_location'],target=row['target'],ortho_scale=row['ortho_scale']))
    payload();save(output/'render-metadata.json',dict(classification='PASS_UNCHANGED_FULL_SOURCE_COLOR_INSPECTION',frames=frames,
        all_source_vertex_and_corner_payloads_exact=True,source_vertices=len(v),source_triangles=len(f),
        device='CPU',threads=4,samples=8,smooth_shading=False,geometry_repaired_or_modified=False,
        color_labels_are_candidates_not_forced_semantics=True))
for name,row in before.items():
    p=Path(name);assert (p.stat().st_size,p.stat().st_mtime_ns,sha(p))==row
v._mmap.close();f._mmap.close()
