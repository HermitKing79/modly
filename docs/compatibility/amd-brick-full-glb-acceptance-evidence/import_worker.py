"""Exactly one stock GLB import; no repair, render, export or colour replacement."""
import sys
sys.dont_write_bytecode=True
import argparse,gc,json,time,traceback
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from common import *
from contract import compaction,verify_geometry
def run(output):
    r,legacy,scratch=start('import',output)
    import bpy
    import numpy as np
    began=time.monotonic();out=P/'import';record=dict(classification='RUNNING',fresh_GLTF_imports=0,render_count=0,geometry_repaired=False,colours_replaced=False,files={})
    try:
        _,_,watched,_=authenticate();before={str(x):digest(x) for x in watched}
        vertices=np.load(MESH/'vertices.npy',mmap_mode='r',allow_pickle=False);faces=np.load(MESH/'faces.npy',mmap_mode='r',allow_pickle=False);colours=np.load(COLOURS,mmap_mode='r',allow_pickle=False)
        ids,inverse=compaction(faces,len(vertices));require(len(ids)==1952142 and len(vertices)-len(ids)==5749,'Only approved original orphans omitted')
        record['original_compacted_vertices']=len(ids);record['original_triangles']=len(faces)
        bpy.ops.wm.read_factory_settings(use_empty=True);bpy.context.preferences.filepaths.temporary_directory=str(scratch)
        # A bounded CPU native setter oracle. This mesh is never imported/rendered.
        oracle=np.lib.format.open_memmap(out/'oracle-colour-bytes.npy',mode='w+',dtype='u1',shape=(len(vertices),4))
        scratch_mesh=bpy.data.meshes.new('CPU native BYTE_COLOR oracle');scratch_mesh.vertices.add(32768)
        native=scratch_mesh.color_attributes.new('Oracle','BYTE_COLOR','POINT');rna=native.data[0].bl_rna.identifier
        rgba=np.ones((32768,4),dtype='<f4')
        for start_row in range(0,len(vertices),32768):
            n=min(32768,len(vertices)-start_row);rgba[:n,:3]=colours[start_row:start_row+n];rgba[n:,:3]=0
            native.data.foreach_set('color',rgba.ravel());oracle[start_row:start_row+n]=colour_bytes(native)[:n]
        oracle.flush();del oracle;gc.collect();bpy.data.meshes.remove(scratch_mesh)
        record['files']['oracle-colour-bytes.npy']={**record_file(out/'oracle-colour-bytes.npy'), 'shape':[1957891,4],'dtype':'uint8'}
        record['native_oracle']=dict(data_type='BYTE_COLOR',domain='POINT',RNA_type=rna,source_rgb='exact original F32 scene-linear COLOR_0',rows=1957891,batch_rows=32768,RGBA_alpha=1,same_RNA_type_as_imported=False)
        stamp=time.monotonic();settings=dict(filepath=str(GLB),merge_vertices=False,import_shading='FLAT')
        save(out/'before-import.json',dict(original_GLB_SHA256=digest(GLB),settings=settings,oracle_SHA256=record['files']['oracle-colour-bytes.npy']['sha256'],GLTF_import_calls_about_to_start=1,no_retry=True))
        record['fresh_GLTF_imports']=1
        print('ONE_FULL_GLB_IMPORT_START',flush=True)
        require(bpy.ops.import_scene.gltf(**settings)=={'FINISHED'},'One genuine stock GLB import')
        record.update(import_completed=True,import_seconds=time.monotonic()-stamp,import_settings=settings,Blender_version=bpy.app.version_string)
        objects=list(bpy.context.scene.objects);require(len(objects)==1 and objects[0].type=='MESH','One original imported object')
        obj=objects[0];actual_v,actual_f=geometry_readback(obj);verify_geometry(vertices,faces,actual_v,actual_f)
        for name,a in (('actual-vertices.npy',actual_v),('actual-faces.npy',actual_f)):record['files'][name]=array_file(out,name,a)
        del actual_v,actual_f;gc.collect()
        material=r.material_readback(bpy,obj);attribute=obj.data.color_attributes[0]
        require(attribute.data_type=='BYTE_COLOR' and attribute.domain=='CORNER' and len(attribute.data)==12520752,'Stock corner-byte colour representation')
        require(attribute.data[0].bl_rna.identifier==rna=='ByteColorAttributeValue','Same native colour setter RNA across domains')
        record['native_oracle']['same_RNA_type_as_imported']=True
        actual=colour_bytes(attribute);oracle=np.load(out/'oracle-colour-bytes.npy',mmap_mode='r',allow_pickle=False)
        for s in range(0,len(faces),32768):require(np.array_equal(actual[3*s:3*min(s+32768,len(faces))],oracle[faces[s:s+32768]].reshape(-1,4)),'Every original corner must retain exactly its expected native colour bytes')
        record['files']['actual-colour-bytes.npy']=array_file(out,'actual-colour-bytes.npy',actual);del actual,oracle;gc.collect()
        record.update(material=material,all_oriented_triangles_exact=True,all_corner_colour_bytes_native_exact=True,omitted_only_original_unused_vertices=5749)
        require(before=={str(x):digest(x) for x in watched},'Original sources unchanged')
        require(time.monotonic()-began<150,'Leave bounded time for private scene persistence')
        scene=out/'inspected.blend';require(not scene.exists(),'Exclusive inspection scene')
        require(bpy.ops.wm.save_as_mainfile(filepath=str(scene),check_existing=True,compress=False,copy=True)=={'FINISHED'},'Private inspection scene save')
        record['files']['inspected.blend']=record_file(scene)
        record.update(classification='PASS',sources_unchanged=True,elapsed_seconds=time.monotonic()-began)
        save(out/'result.json',record);print('FULL_GLB_IMPORT_PASS',flush=True)
    except BaseException as error:
        record.update(classification='NO-GO',error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-began)
        save(out/'failure.json',record);raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);run(a.output)
