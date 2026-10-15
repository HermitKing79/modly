"""Disposable real patches only: original topology, Planar1/5degrees, then stop."""
import sys
sys.dont_write_bytecode=True
import json,math,time
from pathlib import Path
import bpy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as c
core=c.module('planar_stats',c.PRIOR/'core.py')
metrics=c.module('planar_metrics',c.PRIOR/'metrics.py')
ANGLES=(1.,5.)
CRITERIA={'min_triangle_reduction_fraction':.20,'geometry_normalized_rms':.005,
 'geometry_normalized_p95':.01,'geometry_normalized_max':.03,'min_informative_silhouette_iou':.98,
 'max_rgb_rms':.02,'max_rgb_p95':.05,'no_new_components':True,'no_new_nonmanifold_edges':True,
 'max_boundary_length_factor':1.05,'max_patch_triangles':6000}
def ahash(a):
 import hashlib
 return hashlib.sha256(np.ascontiguousarray(a).view('u1')).hexdigest()
def archive(name,data,labels):
 path=c.OUT/(name+'.npz');assert not path.exists()
 np.savez(path,xyz=data[0],faces=data[1],rgba=data[2],face_component_ids=labels)
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':c.digest(path)}
def saved_patch():
 with np.load(c.PATCH['artifacts']['source']['path'],allow_pickle=False) as p:
  data=tuple(p[k].copy() for k in ('xyz','faces','rgba'))
 assert len(data[0])==131 and len(data[1])==272
 return data,{'source':'previously saved unchanged forward patch','original_extraction':c.PATCH['patch_extraction']}
def flat_patch():
 # One independently fixed housing target; read-only arrays, never a master Blender object.
 with c.SOURCE.open('rb') as f:
  for _ in range(64):
   line=f.readline(4097);assert line and len(line)<=4096
   if line.strip()==b'end_header':offset=f.tell();break
  else:raise AssertionError('Missing bounded PLY header')
 n,F=1842968,4012360
 verts=np.memmap(c.SOURCE,mode='r',offset=offset,dtype=[('xyz','<f4',3),('rgb','u1',3)],shape=(n,))
 faces=np.memmap(c.SOURCE,mode='r',offset=offset+n*15,dtype=[('count','u1'),('f','<i4',3)],shape=(F,))
 assert c.SOURCE.stat().st_size==offset+n*15+F*13
 target=np.array([.24,.15,.10]);best=None
 # Mechanical panel candidates: normal within10degrees of X, centroid within0.10 of fixed target.
 for start in range(0,F,65536):
  row=faces[start:start+65536];assert np.all(row['count']==3)
  tri=verts['xyz'][row['f']].astype('f8');normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(normal,axis=1)
  distances=np.sum((tri.mean(axis=1)-target)**2,axis=1)
  eligible=(length>0)&(np.abs(normal[:,0])>=math.cos(math.radians(10))*length)&(distances<=.1**2)
  if eligible.any():
   ids=np.flatnonzero(eligible);i=int(ids[np.argmin(distances[ids])]);item=(float(distances[i]),start+i)
   if best is None or item<best:best=item
 if best is None:return None,{'skipped':'No axis-X panel seed within fixed target neighborhood'}
 marked=np.zeros(n,bool);marked[faces['f'][best[1]]]=True
 for _ in range(3):
  chunks=[]
  for start in range(0,F,65536):
   ids=np.flatnonzero(np.any(marked[faces['f'][start:start+65536]],axis=1))+start
   if len(ids):chunks.append(ids)
  selected=np.concatenate(chunks)
  if not 1<=len(selected)<=6000:return None,{'skipped':'Fixed3-ring neighborhood exceeds6000faces; no truncation or retry','faces':len(selected)}
  marked[faces['f'][selected].reshape(-1)]=True
 global_faces=faces['f'][selected];ids,inv=np.unique(global_faces,return_inverse=True)
 xyz=np.ascontiguousarray(verts['xyz'][ids]);outfaces=inv.reshape(-1,3).astype('<i4')
 linear=np.load(c.LINEAR,allow_pickle=False,mmap_mode='r');rgba=np.ones((len(ids),4),'<f4');rgba[:,:3]=linear[ids]
 tri=xyz[outfaces].astype('f8');normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(normal,axis=1)
 flat_area=float(length[np.abs(normal[:,0])>=math.cos(math.radians(10))*length].sum()/length.sum())
 if len(outfaces)<100 or flat_area<.60:return None,{'skipped':'Fixed panel neighborhood is too small or not predominantly flat','triangles':len(outfaces),'axis_X_flat_area_fraction':flat_area}
 path=c.OUT/'panel-provenance.npz';assert not path.exists()
 np.savez(path,global_vertex_rows=ids,global_face_rows=selected)
 return (xyz,outfaces,rgba),{'target':target.tolist(),'seed_global_face':best[1],'distance_from_target':math.sqrt(best[0]),
  'fixed_vertex_rings':3,'axis_X_flat_area_fraction':flat_area,'selection_independent_of_decimation':True,
  'provenance':{'path':str(path),'bytes':path.stat().st_size,'sha256':c.digest(path)},
  'crop_warning':'Open extraction boundaries; local views and components cannot establish full-model quality'}
def planar(name,data,labels,degrees):
 xyz,faces,rgba=data
 mesh=bpy.data.meshes.new(name);mesh.from_pydata(xyz.tolist(),[],faces.tolist());mesh.update()
 obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj);obj.select_set(True);bpy.context.view_layer.objects.active=obj
 col=mesh.color_attributes.new(name='Col',type='FLOAT_COLOR',domain='POINT');col.data.foreach_set('color',rgba.reshape(-1))
 tag=mesh.attributes.new('SourceComponent','INT','FACE');tag.data.foreach_set('value',labels)
 actual=np.empty_like(xyz);corners=np.empty_like(faces);colors=np.empty_like(rgba)
 mesh.vertices.foreach_get('co',actual.reshape(-1));mesh.loops.foreach_get('vertex_index',corners.reshape(-1));col.data.foreach_get('color',colors.reshape(-1))
 assert actual.tobytes()==xyz.tobytes() and corners.tobytes()==faces.tobytes() and colors.tobytes()==rgba.tobytes()
 mod=obj.modifiers.new('Bounded Planar','DECIMATE');mod.decimate_type='DISSOLVE';mod.angle_limit=math.radians(degrees)
 mod.use_dissolve_boundaries=False;mod.delimit={'NORMAL'}
 began=time.monotonic();assert bpy.ops.object.modifier_apply(modifier=mod.name)=={'FINISHED'};elapsed=time.monotonic()-began
 mesh=obj.data;mesh.calc_loop_triangles()
 polygons=len(mesh.polygons);ngons=sum(len(p.vertices)>3 for p in mesh.polygons)
 assert 0<len(mesh.loop_triangles)<=6000 and len(mesh.vertices)<=18000
 outxyz=np.empty((len(mesh.vertices),3),'<f4');outfaces=np.empty((len(mesh.loop_triangles),3),'<i4');polyids=np.empty(len(outfaces),'<i4')
 mesh.vertices.foreach_get('co',outxyz.reshape(-1));mesh.loop_triangles.foreach_get('vertices',outfaces.reshape(-1));mesh.loop_triangles.foreach_get('polygon_index',polyids)
 color=mesh.color_attributes.get('Col');tag=mesh.attributes.get('SourceComponent')
 assert color and color.domain=='POINT' and color.data_type=='FLOAT_COLOR' and tag and tag.domain=='FACE'
 outcolors=np.empty((len(outxyz),4),'<f4');color.data.foreach_get('color',outcolors.reshape(-1));polylab=np.empty(polygons,'<i4');tag.data.foreach_get('value',polylab)
 outlabels=polylab[polyids];assert np.isin(outlabels,labels).all()
 assert np.isfinite(outcolors).all() and np.all(abs(outcolors[:,3]-1)<1e-6)
 bpy.data.objects.remove(obj,do_unlink=True);bpy.data.meshes.remove(mesh)
 return (outxyz,outfaces,outcolors),outlabels,{'modifier_seconds':elapsed,'polygon_count':polygons,'ngons':ngons,
  'effective_triangle_count':len(outfaces),'angle_degrees':degrees,'dissolve_boundaries':False,'delimit':['NORMAL'],
  'input_Blender_XYZ_faces_RGBA_byte_exact':True}
def quality(before,after,measure):
 reasons=[]
 if after['zero_area_triangles']:reasons.append('zero_area_triangles')
 if after['components']>before['components']:reasons.append('fragmentation')
 if after['nonmanifold_gt2_edges']>before['nonmanifold_gt2_edges']:reasons.append('increased_nonmanifold_incidence')
 if after['boundary_length']>before['boundary_length']*1.05:reasons.append('increased_boundary_length')
 for k,limit in (('rms',.005),('p95',.01),('max',.03)):
  value=measure['worst_direction_geometry_normalized'][k]
  if value is None or value>limit:reasons.append('geometry_'+k)
 for axis,item in measure['silhouettes'].items():
  if item['informative'] and (item['iou']<.98 or item['target_clipped_vertices']):reasons.append('silhouette_'+axis)
 for direction in ('source_to_target','target_to_source'):
  item=measure[direction]
  if not item['complete']:reasons.append(direction+'_missing_surface_component')
  for k,limit in (('rms',.02),('p95',.05)):
   value=item['rgb_channel_error'][k]
   if value is None or value>limit:reasons.append(direction+'_RGB_'+k)
 return reasons
def run_patch(name,data,provenance):
 before,labels=core.stats(data[0],data[1]);assert before['zero_area_triangles']==0 and len(data[1])<=6000
 result={'name':name,'before':before,'provenance':provenance,'source':archive(name+'-source',data,labels),'angles':[]}
 for degrees in ANGLES:
  output,outlabels,info=planar(name,data,labels,degrees)
  after,_=core.stats(output[0],output[1]);measure=metrics.compare(data,output,labels,outlabels)
  reasons=quality(before,after,measure);fraction=1-len(output[1])/len(data[1])
  item={**info,'after':after,'measurements':measure,'quality_pass':not reasons,'quality_failures':reasons,
   'triangle_reduction_fraction':fraction,'worthwhile':fraction>=.20,'artifact':archive(name+'-'+str(int(degrees))+'deg',output,outlabels)}
  result['angles'].append(item)
  c.save(c.E/(name+'-'+str(int(degrees))+'deg.json'),item)
  if reasons:break
 result['quality_pass']=all(x['quality_pass'] for x in result['angles'])
 result['worthwhile']=any(x['quality_pass'] and x['worthwhile'] for x in result['angles'])
 c.save(c.E/(name+'.json'),result);c.save(c.OUT/(name+'.json'),result)
 print(json.dumps({'patch':name,'quality_pass':result['quality_pass'],'counts':[a['effective_triangle_count'] for a in result['angles']]}),flush=True)
 return result
began=time.monotonic();assert bpy.app.version[:3]==(5,2,1)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
first=run_patch('saved-forward',*saved_patch());cases=[first];panel={}
if first['quality_pass']:
 data,panel=flat_patch()
 if data is not None:cases.append(run_patch('armour-panel',data,panel))
result={'decision':'GO' if all(x['quality_pass'] for x in cases) and any(x['worthwhile'] for x in cases) else 'NO-GO',
 'blender':{'version':bpy.app.version_string,'build_hash':bpy.app.build_hash.decode()},'angles_degrees':list(ANGLES),'criteria':CRITERIA,
 'cases':[{'name':p['name'],'quality_pass':p['quality_pass'],'worthwhile':p['worthwhile']} for p in cases],
 'panel_selection':panel,'elapsed_seconds':time.monotonic()-began,'full_mesh_modifications':0,'face_fan_separations':0,'collapse_calls':0,
 'voxel_operations':0,'GLB_exports':0,'stop_on_first_patch_quality_failure':True}
c.save(c.E/'results.json',result);c.save(c.OUT/'results.json',result)
print(json.dumps(result),flush=True)
