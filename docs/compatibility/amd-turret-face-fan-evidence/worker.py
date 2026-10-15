"""One fan-split method, synthetic first, then at most two bounded real patches.

Only the small arrays enter Blender or are separated/decimated. The original
PLY and its global edge incidence are read-only; no full-model preparation.
"""
import sys
sys.dont_write_bytecode=True
import hashlib,json,os,time
from pathlib import Path
import bpy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as c
import core
import metrics

PHASE=os.environ['MODLY_FAN_PHASE']
CRITERIA={'real_ratio':.25,'synthetic_ratio':.05,'max_patch_faces':6000,'fixed_vertex_rings':3,
 'max_prepared_vertex_factor':2.,'max_boundary_length_factor':3.,
 'min_area_in_largest_4x_original_components':.95,
 'max_real_output_fraction':.30,'max_fraction_of_control_faces':.80,
 'geometry_normalized_rms':.005,'geometry_normalized_p95':.01,'geometry_normalized_max':.03,
 'min_axis_silhouette_iou':.98,'max_rgb_rms':.02,'max_rgb_p95':.05,
 'max_lost_sheet_area_fraction':.01}

def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def artifact(name,**arrays):
 path=c.OUT/(name+'.npz');assert not path.exists()
 np.savez(path,**arrays)
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':c.digest(path)}

def decimate(name,data,sheets,ratio):
 xyz,faces,rgba=data
 assert len(faces)<=6000 and len(xyz)<=18000
 mesh=bpy.data.meshes.new(name);mesh.from_pydata(xyz.tolist(),[],faces.tolist());mesh.update()
 obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj)
 bpy.context.view_layer.objects.active=obj;obj.select_set(True)
 col=mesh.color_attributes.new(name='Col',type='FLOAT_COLOR',domain='POINT')
 col.data.foreach_set('color',np.ascontiguousarray(rgba).reshape(-1))
 provenance=mesh.attributes.new('SourceSheet','INT','FACE');provenance.data.foreach_set('value',np.ascontiguousarray(sheets,dtype='i4'))
 # Verify the actual Blender ingestion boundary before any modifier executes.
 actual_xyz=np.empty_like(xyz);actual_faces=np.empty_like(faces);actual_rgba=np.empty_like(rgba)
 actual_labels=np.empty_like(sheets,dtype='i4');actual_totals=np.empty(len(faces),dtype='i4')
 mesh.vertices.foreach_get('co',actual_xyz.reshape(-1));mesh.loops.foreach_get('vertex_index',actual_faces.reshape(-1))
 mesh.polygons.foreach_get('loop_total',actual_totals)
 col.data.foreach_get('color',actual_rgba.reshape(-1));provenance.data.foreach_get('value',actual_labels)
 assert np.all(actual_totals==3) and actual_xyz.tobytes()==xyz.tobytes() and actual_faces.tobytes()==faces.tobytes()
 assert actual_rgba.tobytes()==rgba.tobytes() and np.array_equal(actual_labels,sheets)
 mod=obj.modifiers.new('Single limited Collapse','DECIMATE');mod.decimate_type='COLLAPSE'
 mod.ratio=ratio;mod.use_collapse_triangulate=True;mod.use_symmetry=False
 began=time.monotonic();assert bpy.ops.object.modifier_apply(modifier=mod.name)=={'FINISHED'}
 elapsed=time.monotonic()-began;mesh=obj.data
 assert len(mesh.polygons)>0 and len(mesh.polygons)<=6000
 outxyz=np.empty((len(mesh.vertices),3),'<f4');outfaces=np.empty((len(mesh.polygons),3),'<i4')
 totals=np.empty(len(mesh.polygons),'<i4');mesh.polygons.foreach_get('loop_total',totals);assert np.all(totals==3)
 mesh.vertices.foreach_get('co',outxyz.reshape(-1));mesh.loops.foreach_get('vertex_index',outfaces.reshape(-1))
 attr=mesh.color_attributes.get('Col');assert attr and attr.domain=='POINT' and attr.data_type=='FLOAT_COLOR'
 outrgba=np.empty((len(outxyz),4),'<f4');attr.data.foreach_get('color',outrgba.reshape(-1))
 label=mesh.attributes.get('SourceSheet');assert label and label.domain=='FACE' and label.data_type=='INT'
 labels=np.empty(len(outfaces),'<i4');label.data.foreach_get('value',labels)
 assert np.isin(labels,np.unique(sheets)).all() and np.isfinite(outrgba).all()
 assert np.all(abs(outrgba[:,3]-1)<1e-6)
 result=(outxyz,outfaces,outrgba)
 bpy.data.objects.remove(obj,do_unlink=True);bpy.data.meshes.remove(mesh)
 return result,labels,elapsed

def preparation_details(data,prepared,origin,global_counts):
 xyz,faces,rgba=data;px,pf,pr=prepared
 keys,counts,_=core.edges(faces,len(xyz));pk,pc,_=core.edges(pf,len(px))
 a=origin[(pk//len(px)).astype('i8')];b=origin[(pk%len(px)).astype('i8')]
 oldkeys=np.minimum(a,b).astype('u8')*np.uint64(len(xyz))+np.maximum(a,b).astype('u8')
 where=np.searchsorted(keys,oldkeys);assert np.array_equal(keys[where],oldkeys)
 bad=(global_counts[where]>2)
 return {'ordered_triangle_origins_exact':bool(np.array_equal(origin[pf],faces)),
  'ordered_corner_XYZ_byte_exact':px[pf].tobytes()==xyz[faces].tobytes(),
  'ordered_corner_RGBA_byte_exact':pr[pf].tobytes()==rgba[faces].tobytes(),
  'surface_triangle_payload_sha256':ahash(xyz[faces]),'corner_RGBA_sha256':ahash(rgba[faces]),
  'vertex_duplicates_added':len(px)-len(np.unique(faces)),
  'new_boundaries_from_original_gt2_edges':int(np.count_nonzero((pc==1)&bad)),
  'new_boundaries_from_other_original_edges':int(np.count_nonzero((pc==1)&~bad&(counts[where]>1))),
  'original_bad_edge_descendant_incidence_histogram':{str(int(k)):int(v) for k,v in zip(*np.unique(pc[bad],return_counts=True))}}

def failures(before,prep,control,reduced,control_metric,metric,synthetic):
 problems=[]
 if prep['nonmanifold_gt2_edges']:problems.append('remaining_nonmanifold_edges')
 if reduced['nonmanifold_gt2_edges']:problems.append('post_decimation_nonmanifold_edges')
 if reduced['zero_area_triangles']:problems.append('introduced_zero_area_triangles')
 if synthetic:
  if prep['components']!=4 or reduced['components']!=4:problems.append('synthetic_four_sheets_not_retained')
 else:
  if prep['vertices']>CRITERIA['max_prepared_vertex_factor']*before['vertices']:problems.append('excessive_vertex_duplication')
  if prep['boundary_length']>CRITERIA['max_boundary_length_factor']*before['boundary_length']:problems.append('excessive_boundary_length')
  budget=max(4,4*before['components']);areas=sorted(prep['component_areas'],reverse=True)
  if sum(areas[:budget])/sum(areas)<.95:problems.append('excessive_fragmentation')
  if reduced['triangles']>.30*before['triangles']:problems.append('insufficient_triangle_reduction')
 if reduced['components']>prep['components']:problems.append('new_post_decimation_components')
 if reduced['triangles']>.80*control['triangles']:problems.append('no_material_reduction_advantage_over_control')
 for direction in ('source_to_target','target_to_source'):
  item=metric[direction]
  if not item['complete']:problems.append(direction+'_missing_sheet_or_query')
  if item['lost_sheet_area_fraction']>.01:problems.append(direction+'_lost_surface_sheet_area')
  if item['rgb_channel_error']['rms'] is None or item['rgb_channel_error']['rms']>.02:problems.append(direction+'_RGB_RMS')
  if item['rgb_channel_error']['p95'] is None or item['rgb_channel_error']['p95']>.05:problems.append(direction+'_RGB_P95')
 limits={'rms':1e-6,'p95':1e-6,'max':1e-6} if synthetic else {'rms':.005,'p95':.01,'max':.03}
 for key,limit in limits.items():
  value=metric['worst_direction_geometry_normalized'][key]
  if value is None or value>limit:problems.append('geometry_'+key)
  if not synthetic:
   other=control_metric['worst_direction_geometry_normalized'][key]
   allowance={'rms':.001,'p95':.002,'max':.005}[key]
   if value is not None and other is not None and value>other+allowance:problems.append('geometry_regression_vs_control_'+key)
 for axis,item in metric['silhouettes'].items():
  if item['informative']:
   if item['target_clipped_vertices']>0:problems.append('silhouette_clipping_'+axis)
   if item['iou']<(.995 if synthetic else .98):problems.append('silhouette_'+axis)
   other=control_metric['silhouettes'][axis]['iou']
   if not synthetic and other is not None and item['iou']<other-.005:problems.append('silhouette_regression_vs_control_'+axis)
 return sorted(set(problems))

def run_case(name,data,global_counts=None,extra=None,synthetic=False):
 began=time.monotonic();xyz,faces,rgba=data
 before,_=core.stats(xyz,faces);assert before['zero_area_triangles']==0
 keys,counts,_=core.edges(faces,len(xyz))
 if global_counts is None:global_counts=counts
 px,pf,pr,origin=core.separate(xyz,faces,rgba,global_counts==2)
 prepared=(px,pf,pr);prep,sheets=core.stats(px,pf)
 assert prep['triangles']==before['triangles'] and prep['nonmanifold_gt2_edges']==0
 proof=preparation_details(data,prepared,origin,global_counts)
 assert all(proof[k] for k in ('ordered_triangle_origins_exact','ordered_corner_XYZ_byte_exact','ordered_corner_RGBA_byte_exact'))
 ratio=.05 if synthetic else .25
 control,control_labels,control_seconds=decimate(name+'-control',data,sheets,ratio)
 reduced,reduced_labels,reduced_seconds=decimate(name+'-separated',prepared,sheets,ratio)
 control_stats,_=core.stats(control[0],control[1]);reduced_stats,_=core.stats(reduced[0],reduced[1])
 control_metric=metrics.compare(data,control,sheets,control_labels)
 metric=metrics.compare(data,reduced,sheets,reduced_labels)
 reasons=failures(before,prep,control_stats,reduced_stats,control_metric,metric,synthetic)
 files={}
 for suffix,arrays,labels in [('source',data,sheets),('separated',prepared,sheets),('control-decimated',control,control_labels),('separated-decimated',reduced,reduced_labels)]:
  fields={'xyz':arrays[0],'faces':arrays[1],'rgba':arrays[2],'face_sheet_ids':labels}
  if suffix=='separated':fields['original_vertex_row']=origin
  files[suffix]=artifact(name+'-'+suffix,**fields)
 result={'name':name,'decision':'NO-GO' if reasons else 'GO','reasons':reasons,
  'actual_Blender_ingest_XYZ_triangles_RGBA_sheet_labels_byte_exact':True,
  'synthetic':synthetic,'criteria':CRITERIA,'before':before,'separated':prep,'preservation':proof,
  'ratio':ratio,'requested_triangles':int(len(faces)*ratio),'control_decimated':control_stats,
  'separated_decimated':reduced_stats,'control_measurements':control_metric,'separated_measurements':metric,
  'control_modifier_seconds':control_seconds,'separated_modifier_seconds':reduced_seconds,
  'elapsed_seconds':time.monotonic()-began,'artifacts':files,'patch_extraction':extra}
 c.save(c.E/(name+'.json'),result);c.save(c.OUT/(name+'.json'),result)
 print('CASE='+json.dumps({'name':name,'decision':result['decision'],'reasons':reasons,'source_faces':len(faces),'prepared_components':prep['components'],'control_faces':control_stats['triangles'],'separated_faces':reduced_stats['triangles']}),flush=True)
 return result

def synthetic():
 xyz=[];faces=[];index={}
 def vertex(p):
  if p not in index:index[p]=len(xyz);xyz.append(p)
  return index[p]
 for plane in ('xy','xz'):
  grid=[]
  for j in range(17):
   row=[]
   for i in range(17):
    x,t=i/8-1,j/8-1;row.append(vertex((x,t,0.) if plane=='xy' else (x,0.,t)))
   grid.append(row)
  for j in range(16):
   for i in range(16):
    a,b,d,e=grid[j][i],grid[j][i+1],grid[j+1][i],grid[j+1][i+1]
    faces.extend([(a,b,e),(a,e,d)])
 xyz=np.asarray(xyz,dtype='<f4');faces=np.asarray(faces,dtype='<i4')
 rgba=np.ones((len(xyz),4),'<f4');rgba[:,:3]=.35+.1*xyz
 stats,_=core.stats(xyz,faces);assert (stats['triangles'],stats['nonmanifold_gt2_edges'],stats['components'])==(1024,16,1)
 return run_case('synthetic',(xyz,faces,rgba),synthetic=True)

def source_tables():
 with c.SOURCE.open('rb') as f:
  for _ in range(64):
   line=f.readline(4097);assert line and len(line)<=4096
   if line.strip()==b'end_header':offset=f.tell();break
  else:raise AssertionError('Bounded PLY header required')
 n,F=1842968,4012360
 verts=np.memmap(c.SOURCE,mode='r',offset=offset,dtype=[('xyz','<f4',3),('rgb','u1',3)],shape=(n,))
 rows=np.memmap(c.SOURCE,mode='r',offset=offset+n*15,dtype=[('count','u1'),('faces','<i4',3)],shape=(F,))
 assert c.SOURCE.stat().st_size==offset+n*15+F*13 and np.all(rows['count']==3)
 xyz=verts['xyz'];faces=np.ascontiguousarray(rows['faces'])
 assert c.old.legacy.ahash(xyz)==c.old.legacy.XYZ_SHA and ahash(faces)==c.old.legacy.FACE_SHA
 assert c.old.legacy.ahash(verts['rgb'])==c.old.legacy.RGB_SHA
 linear=np.load(c.LINEAR,allow_pickle=False,mmap_mode='r');assert linear.shape==(n,3) and linear.dtype==np.float32
 allkeys=np.empty((F,3),'<u8')
 for j,k in ((0,1),(1,2),(2,0)):
  allkeys[:,j]=np.minimum(faces[:,j],faces[:,k]).astype('u8')*np.uint64(n)+np.maximum(faces[:,j],faces[:,k]).astype('u8')
 unique,counts=np.unique(allkeys,return_counts=True);del allkeys
 four=unique[counts==4]
 return xyz,faces,linear,unique,counts,four

def patch(name,target,tables):
 xyz,faces,linear,unique,counts,four=tables;n=len(xyz)
 endpoints=np.stack((four//n,four%n),axis=1).astype('i8')
 centers=(xyz[endpoints[:,0]].astype('f8')+xyz[endpoints[:,1]])*.5
 i=int(np.argmin(np.sum((centers-np.asarray(target))**2,axis=1)))
 seed=endpoints[i];distance=float(np.linalg.norm(centers[i]-target))
 assert distance<=.10,'No four-face junction close to fixed target; stop without substituting a sample'
 marked=np.zeros(n,bool);marked[seed]=True
 for _ in range(3):
  selected=np.flatnonzero(np.any(marked[faces],axis=1))
  assert len(selected)<=6000,'Fixed three-ring patch exceeds cap; do not truncate or retry'
  marked[faces[selected].reshape(-1)]=True
 assert 100<=len(selected)<=6000,'Patch too small to assess without expanding the approved sample'
 original_faces=faces[selected];vertices,inv=np.unique(original_faces,return_inverse=True)
 pf=inv.reshape(-1,3).astype('i4');px=np.ascontiguousarray(xyz[vertices]);pr=np.ones((len(vertices),4),'<f4');pr[:,:3]=linear[vertices]
 local_keys,local_counts,_=core.edges(pf,len(px))
 a=vertices[(local_keys//len(px)).astype('i8')];b=vertices[(local_keys%len(px)).astype('i8')]
 global_keys=np.minimum(a,b).astype('u8')*np.uint64(n)+np.maximum(a,b).astype('u8')
 where=np.searchsorted(unique,global_keys);assert np.array_equal(unique[where],global_keys)
 incidence=counts[where]
 seed_faces=np.flatnonzero(np.sum(np.isin(original_faces,seed),axis=1)==2)
 assert len(seed_faces)==4
 extra={'target':target,'seed_global_vertex_rows':seed.tolist(),'seed_midpoint':centers[i].tolist(),
  'distance_from_target':distance,'fixed_vertex_rings':3,'complete_four_face_seed':True,
  'complete_original_seed_endpoint_stars':True,'seed_local_face_rows':seed_faces.tolist(),
  'natural_original_boundary_edges':int(np.count_nonzero((local_counts==1)&(incidence==1))),
  'artificial_extraction_boundary_edges':int(np.count_nonzero((local_counts==1)&(incidence>1))),
  'partially_cropped_original_nonmanifold_edges':int(np.count_nonzero((incidence>2)&(local_counts<incidence))),
  'join_policy':'Only original GLOBAL incidence2; cropped incidence4 is never mistaken for manifold',
  'sample_scope':'Local patch only; does not test complete housing, barrels, base or global silhouette'}
 provenance=artifact(name+'-source-provenance',global_vertex_rows=vertices,global_face_rows=selected,
  global_edge_incidence=incidence,local_edge_keys=local_keys)
 extra['provenance']=provenance
 return run_case(name,(px,pf,pr),incidence,extra)

began=time.monotonic();assert bpy.app.version[:3]==(5,2,1)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
if PHASE=='synthetic':
 result=synthetic()
else:
 assert PHASE=='real'
 tables=source_tables();cases=[]
 for name,target in [('forward-patch',[.10,-.32,.10]),('housing-patch',[.24,.15,.10])]:
  item=patch(name,target,tables);cases.append(item)
  if item['decision']!='GO':break
 result={'decision':'GO' if len(cases)==2 and all(x['decision']=='GO' for x in cases) else 'NO-GO',
  'cases':[{'name':x['name'],'decision':x['decision'],'reasons':x['reasons']} for x in cases],
  'criteria':CRITERIA,'elapsed_seconds':time.monotonic()-began,'full_mesh_separations':0,
  'full_mesh_decimations':0,'GLB_exports':0,'stop_after_first_failing_real_patch':True}
 c.save(c.E/'real.json',result);c.save(c.OUT/'real.json',result)
print('PHASE='+json.dumps({'phase':PHASE,'decision':result['decision'],'seconds':time.monotonic()-began}),flush=True)
