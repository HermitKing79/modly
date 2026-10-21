"""One +X CPU preview of the saved verified import; never reimports the GLB."""
import sys
sys.dont_write_bytecode=True
import argparse,gc,json,time,traceback
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from common import *
from contract import verify_geometry
def run(output):
    r,legacy,scratch=start('render',output)
    import bpy
    import numpy as np
    began=time.monotonic();out=P/'render';record=dict(classification='RUNNING',fresh_GLTF_imports=0,render_count=0,views=[],files={})
    try:
        _,_,watched,_=authenticate();before={str(x):digest(x) for x in watched}
        parent=json.loads((IMPORTED/'result.json').read_text());verified=json.loads((P/'qualify/result.json').read_text())
        require(parent['classification']==verified['classification']=='PASS','Actual imported and independently verified source')
        scene_path=plain(SCENE);require(digest(scene_path)==parent['files']['inspected.blend']['sha256'],'Exact verified saved scene')
        require(bpy.ops.wm.open_mainfile(filepath=str(scene_path),load_ui=False,use_scripts=False)=={'FINISHED'},'Open private verified scene, no GLB import')
        objects=list(bpy.context.scene.objects);require(len(objects)==1 and objects[0].type=='MESH','Original sole imported mesh')
        obj=objects[0];vertices=np.load(MESH/'vertices.npy',mmap_mode='r',allow_pickle=False);faces=np.load(MESH/'faces.npy',mmap_mode='r',allow_pickle=False)
        v,f=geometry_readback(obj);verify_geometry(vertices,faces,v,f);del v,f;gc.collect()
        material=r.material_readback(bpy,obj);require(material==parent['material'],'Saved genuine imported shader/material unchanged')
        # Verify unchanged bytes, then qualify the actual native linear getter.
        # No fitted rounding/ULP envelope or colour replacement is introduced.
        stored=np.load(IMPORTED/'actual-colour-bytes.npy',mmap_mode='r',allow_pickle=False)
        actual_bytes=colour_bytes(obj.data.color_attributes[0]);require(np.array_equal(actual_bytes,stored),'All saved corner bytes unchanged on scene reopen');del actual_bytes;gc.collect()
        attribute=obj.data.color_attributes[0];rgba=np.empty((12520752,4),dtype='<f4');attribute.data.foreach_get('color',rgba.ravel())
        require(np.isfinite(rgba).all() and (rgba>=0).all() and (rgba<=1).all() and (rgba[:,3]==1).all(),'Actual native scene-linear colour getter')
        from quality import decode
        oracle=np.load(IMPORTED/'oracle-colour-bytes.npy',mmap_mode='r',allow_pickle=False)
        native_source=decode(oracle[:,:3])
        for s in range(0,len(faces),32768):
            part=faces[s:s+32768];native_source[part.reshape(-1)]=rgba[s*3:(s+len(part))*3,:3]
        for s in range(0,len(faces),32768):
            part=faces[s:s+32768];require(np.array_equal(native_source[part.reshape(-1)].view('<u4'),rgba[s*3:(s+len(part))*3,:3].copy().view('<u4')),'Repeated corner native linear values retain exact source identity')
        del rgba;gc.collect()
        from independent import verify as verify_native
        data,_,_=load(np)
        native_quality=verify_native(data,oracle,decoded_linear=native_source)
        for entry in (native_quality['original_queries'],native_quality['holdout_queries']):entry.pop('predicted_RGB')
        save(out/'native-getter-qualification.json',native_quality)
        require(native_quality['all_gates_pass'],'All112 original gates on actual native colour getter before render')
        record['actual_native_getter_all112_gates_pass']=True
        del data,native_source,oracle;gc.collect()
        b=json.loads(BASELINE.read_text());prior=b['views'][0]
        scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=32
        scene.cycles.use_denoising=True;scene.cycles.denoiser='OPENIMAGEDENOISE';scene.cycles.denoising_use_gpu=False;scene.cycles.use_adaptive_sampling=False;scene.cycles.seed=0;scene.cycles.use_animated_seed=False
        scene.cycles.max_bounces=2;scene.cycles.diffuse_bounces=1;scene.cycles.glossy_bounces=1;scene.cycles.transparent_max_bounces=8
        scene.render.threads_mode='FIXED';scene.render.threads=4;scene.render.resolution_x=scene.render.resolution_y=640;scene.render.resolution_percentage=100
        scene.render.film_transparent=False;scene.render.use_compositing=False;scene.render.use_sequencer=False;scene.render.use_persistent_data=False
        scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.image_settings.color_depth='8';scene.render.image_settings.compression=15
        settings=b['settings_before'];scene.view_settings.view_transform=settings['view_transform'];scene.view_settings.look=settings['look'];scene.view_settings.exposure=2;scene.view_settings.gamma=settings['gamma']
        bpy.context.preferences.filepaths.temporary_directory=str(scratch)
        cp=bpy.context.preferences.addons.get('cycles')
        if cp:
            cp.preferences.compute_device_type='NONE'
            for device in cp.preferences.devices:device.use=device.type=='CPU'
        for layer in scene.view_layers:layer.cycles.use_denoising=True;layer.material_override=None
        world=bpy.data.worlds.new('Accepted neutral studio');world.use_nodes=True
        bg=world.node_tree.nodes['Background'];bg.inputs['Color'].default_value=settings['world_linear_RGBA'];bg.inputs['Strength'].default_value=settings['world_strength'];scene.world=world
        cam_data=bpy.data.cameras.new('Original positive X orthographic');cam_data.type='ORTHO';cam_data.ortho_scale=prior['ortho_scale'];cam_data.clip_start,cam_data.clip_end=prior['clip']
        cam=bpy.data.objects.new(cam_data.name,cam_data);scene.collection.objects.link(cam);cam.location=prior['camera_location'];cam.rotation_mode='QUATERNION';cam.rotation_quaternion=prior['camera_quaternion'];scene.camera=cam
        lamps=[]
        for item in prior['actual_lights']:
            light=bpy.data.lights.new(item['name'],'AREA');light.energy=item['energy'];light.size=item['size'];light.color=item['color']
            lamp=bpy.data.objects.new(light.name,light);scene.collection.objects.link(lamp);lamp.location=item['location'];lamp.rotation_mode='QUATERNION';lamp.rotation_quaternion=item['quaternion'];lamps.append(lamp)
        def state():
            return dict(engine=scene.render.engine,device=scene.cycles.device,samples=scene.cycles.samples,denoising=scene.cycles.use_denoising,denoising_use_gpu=scene.cycles.denoising_use_gpu,resolution=[scene.render.resolution_x,scene.render.resolution_y],exposure=scene.view_settings.exposure,view_transform=scene.view_settings.view_transform,look=scene.view_settings.look,gamma=scene.view_settings.gamma,camera_location=list(cam.location),camera_quaternion=list(cam.rotation_quaternion),ortho_scale=cam_data.ortho_scale,lights=[dict(location=list(x.location),quaternion=list(x.rotation_quaternion),energy=x.data.energy,size=x.data.size,color=list(x.data.color)) for x in lamps])
        expected=state();require(expected['camera_location']==prior['camera_location'] and expected['camera_quaternion']==prior['camera_quaternion'],'Exact original +X camera')
        # The previous complete PBR 1024px/32sample render used 2.98GB; this
        # is one 640px view. A new host admission and hard caps remain mandatory.
        require(time.monotonic()-began<100,'Leave80seconds for bounded single CPU view')
        png=out/'view-pos-x-ev2.png';require(not png.exists(),'Exclusive single preview')
        scene.render.filepath=str(png);stamp=time.monotonic()
        require(bpy.ops.render.render(write_still=True)=={'FINISHED'},'One CPU render')
        record['render_count']=1
        require(state()==expected,'Camera, lighting, exposure and CPU settings unchanged')
        v,f=geometry_readback(obj);verify_geometry(vertices,faces,v,f);del v,f;gc.collect()
        require(r.material_readback(bpy,obj)==material,'Original imported shader/material unchanged after render')
        actual_bytes=colour_bytes(obj.data.color_attributes[0]);require(np.array_equal(actual_bytes,stored),'All native colour bytes unchanged after render');del actual_bytes;gc.collect()
        require(before=={str(x):digest(x) for x in watched},'Original assets unchanged')
        record['files'][png.name]=record_file(png)
        record.update(classification='PASS',render_seconds=time.monotonic()-stamp,settings=expected,saved_import_scene_SHA256=digest(scene_path),original_geometry_unchanged=True,imported_material_unchanged=True,reference_preview_SHA256=digest(REFERENCE),reference_comparison_scope='Contextual: original direct-row gamma2.2 spatial PBR versus source-position sampled IEC-sRGB vertex colour; not pixel-exact equivalence',elapsed_seconds=time.monotonic()-began)
        save(out/'result.json',record);print('FULL_GLB_CPU_VIEW_PASS',flush=True)
    except BaseException as error:
        record.update(classification='NO-GO',error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-began)
        save(out/'failure.json',record);raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);run(a.output)
