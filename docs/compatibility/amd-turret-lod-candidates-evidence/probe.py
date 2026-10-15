"""One installed-Blender open/nonmanifold fixture, then read-only source census.

This worker never imports the full turret into Blender, repairs topology,
exports a candidate or renders. A proven protected-face floor stops that work.
"""
import json
import math
from pathlib import Path
import sys
import time
import bpy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as c

def geometry(mesh):
 xyz=np.empty((len(mesh.vertices),3),'<f4');faces=np.empty((len(mesh.polygons),3),'<i4')
 counts=np.empty(len(mesh.polygons),'<i4')
 mesh.vertices.foreach_get('co',xyz.reshape(-1));mesh.polygons.foreach_get('loop_total',counts)
 assert np.all(counts==3)
 mesh.loops.foreach_get('vertex_index',faces.reshape(-1))
 attr=mesh.color_attributes.get('Col')
 assert attr and attr.domain=='POINT' and attr.data_type=='FLOAT_COLOR'
 rgba=np.empty((len(attr.data),4),'<f4');attr.data.foreach_get('color',rgba.reshape(-1))
 assert len(rgba)==len(xyz) and np.isfinite(xyz).all() and np.isfinite(rgba).all()
 assert faces.min()>=0 and faces.max()<len(xyz) and np.all(np.abs(rgba[:,3]-1)<1e-6)
 cross=np.cross(xyz[faces[:,1]].astype('f8')-xyz[faces[:,0]],xyz[faces[:,2]].astype('f8')-xyz[faces[:,0]])
 zero=int(np.count_nonzero(np.sum(cross*cross,axis=1)==0))
 return xyz,faces,rgba,zero

def edge_census(faces,n):
 keys=np.empty((len(faces),3),dtype='<u8')
 for j,k in ((0,1),(1,2),(2,0)):
  keys[:,j]=np.minimum(faces[:,j],faces[:,k]).astype('<u8')*np.uint64(n)+np.maximum(faces[:,j],faces[:,k]).astype('<u8')
 unique,counts=np.unique(keys,return_counts=True)
 bad=unique[counts>2]
 locked=np.zeros(n,dtype=np.bool_)
 locked[(bad//np.uint64(n)).astype('i8')]=True;locked[(bad%np.uint64(n)).astype('i8')]=True
 frozen=int(np.count_nonzero(np.sum(locked[faces],axis=1)>=2))
 # Exact union of original triangles incident to any >2-face edge.
 direct=0
 for start in range(0,len(faces),65536):
  query=keys[start:start+65536];where=np.searchsorted(bad,query)
  hit=where<len(bad)
  if len(bad):hit[hit]=bad[where[hit]]==query[hit]
  direct+=int(np.count_nonzero(np.any(hit,axis=1)))
 return {'vertices':n,'triangles':len(faces),'edges':len(unique),
  'boundary_edges':int(np.count_nonzero(counts==1)),'two_face_edges':int(np.count_nonzero(counts==2)),
  'nonmanifold_gt2_edges':len(bad),'maximum_edge_face_incidence':int(counts.max()),
  'vertices_incident_to_gt2_edges':int(locked.sum()),'faces_incident_to_gt2_edges':direct,
  'faces_with_at_least_two_protected_vertices':frozen,
  'source_guard_derived_minimum_face_bound':frozen}

began=time.monotonic()
assert bpy.app.version[:3]==(5,2,1)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
# Four unique faces share one edge, matching the source's maximum incidence4.
xyz=[(0,0,0),(0,0,1),(1,0,.5),(0,1,.5),(-1,0,.5),(0,-1,.5)]
faces=[(0,1,2),(1,0,3),(0,1,4),(1,0,5)]
for y in range(9):
 for x in range(9):xyz.append((3+x/8,y/8,.03*math.sin(x/3)*math.sin(y/3)))
for y in range(8):
 for x in range(8):
  a=6+y*9+x;faces.extend([(a,a+1,a+10),(a,a+10,a+9)])
mesh=bpy.data.meshes.new('Open gradient grid and protected four-face junction')
mesh.from_pydata(xyz,[],faces);mesh.update()
obj=bpy.data.objects.new('Representative fixture only',mesh);bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active=obj;obj.select_set(True)
xyz=np.asarray(xyz,dtype='<f4');rgba=np.ones((len(xyz),4),'<f4')
rgba[:,0]=.15+.05*(xyz[:,0]+1);rgba[:,1]=.2+.1*(xyz[:,1]+1);rgba[:,2]=.25+.2*xyz[:,2]
col=mesh.color_attributes.new(name='Col',type='FLOAT_COLOR',domain='POINT')
col.data.foreach_set('color',rgba.reshape(-1))
before=geometry(mesh);before_census=edge_census(before[1],len(before[0]))
assert before_census['nonmanifold_gt2_edges']==1 and before_census['boundary_edges']>0 and before[3]==0
ratio=100000/4012360
target=int(len(faces)*ratio)
modifier=obj.modifiers.new('Single COLLAPSE qualification','DECIMATE')
modifier.decimate_type='COLLAPSE';modifier.ratio=ratio
modifier.use_collapse_triangulate=True;modifier.use_symmetry=False
assert bpy.ops.object.modifier_apply(modifier=modifier.name)=={'FINISHED'}
after=geometry(obj.data);after_census=edge_census(after[1],len(after[0]))
assert after[3]==0 and 4<=len(after[1])<len(faces)
assert after_census['nonmanifold_gt2_edges']==1 and after_census['maximum_edge_face_incidence']==4
protected_map=[]
for i in range(6):
 matches=np.flatnonzero(np.all(after[0]==before[0][i],axis=1));assert len(matches)==1
 index=int(matches[0]);protected_map.append(index);assert np.array_equal(after[2][index],before[2][i])
def rotations(tri):return {tuple(tri[j:]+tri[:j]) for j in range(3)}
actual_faces={tuple(row) for row in after[1].tolist()}
for face in faces[:4]:assert actual_faces.intersection(rotations([protected_map[i] for i in face]))
assert np.all(after[2][:,:3]>=before[2][:,:3].min(axis=0)-1e-6)
assert np.all(after[2][:,:3]<=before[2][:,:3].max(axis=0)+1e-6)
new_colours=sum(not np.any(np.all(before[2]==row,axis=1)) for row in after[2])
referenced_new_colours=sum(not np.any(np.all(before[2]==after[2][i],axis=1)) for i in np.unique(after[1]))
fixture={'qualification':'PRESERVATION_PASS_TARGET_BLOCKED','synthetic_fixture':True,
 'description':'Open curved grid plus four unique triangles sharing one nonmanifold edge; synthetic linear RGB gradient',
 'Blender_version':bpy.app.version_string,'method':'DECIMATE/COLLAPSE, triangulate on, symmetry off',
 'requested_triangles':target,'ratio_matching_original_100K_target':ratio,'before':before_census,'after':after_census,
 'zero_area_triangles_before':before[3],'zero_area_triangles_after':after[3],
 'protected_four_faces_positions_winding_and_RGBA_byte_exact':True,
 'colour_attribute':'Col/POINT/FLOAT_COLOR','finite_colours_within_source_range':True,
 'new_interpolated_RGBA_rows':new_colours,'referenced_new_interpolated_RGBA_rows':referenced_new_colours,
 'seconds':time.monotonic()-began}
c.save(c.E/'fixture.json',fixture);c.save(c.OUT/'fixture.json',fixture)
print('FIXTURE_RESULT='+json.dumps(fixture),flush=True)

# Read the exact original PLY, independently verifying all geometry/colour payloads.
with c.SOURCE.open('rb') as f:
 header=[]
 for _ in range(64):
  line=f.readline(4097);assert line and len(line)<=4096;header.append(line.decode('ascii').strip())
  if line.strip()==b'end_header':offset=f.tell();break
 else:raise AssertionError('Bounded PLY header missing')
n,F=1842968,4012360
assert 'format binary_little_endian 1.0' in header and f'element vertex {n}' in header and f'element face {F}' in header
verts=np.memmap(c.SOURCE,mode='r',offset=offset,dtype=[('xyz','<f4',3),('rgb','u1',3)],shape=(n,))
rows=np.memmap(c.SOURCE,mode='r',offset=offset+n*15,dtype=[('count','u1'),('faces','<i4',3)],shape=(F,))
assert c.SOURCE.stat().st_size==offset+n*15+F*13 and np.all(rows['count']==3)
faces=np.ascontiguousarray(rows['faces'])
assert c.legacy.ahash(verts['xyz'])==c.legacy.XYZ_SHA and c.legacy.ahash(faces)==c.legacy.FACE_SHA
assert c.legacy.ahash(verts['rgb'])==c.legacy.RGB_SHA
assert np.isfinite(verts['xyz']).all() and faces.min()>=0 and faces.max()<n
assert np.all(faces[:,0]!=faces[:,1]) and np.all(faces[:,1]!=faces[:,2]) and np.all(faces[:,2]!=faces[:,0])
census=edge_census(faces,n)
assert census['nonmanifold_gt2_edges']==338449 and census['maximum_edge_face_incidence']==4
assert census['source_guard_derived_minimum_face_bound']>100000
result={'classification':'BLOCKED','decision':'NO-GO for unmodified-source Blender COLLAPSE at100K/30K',
 'source_PLY_sha256':c.digest(c.SOURCE),'census':census,
 'requested_targets':[100000,30000],'full_turret_decimation_calls':0,'candidate_meshes_exported':0,
 'renders':0,'stop_reason':'Protected nonmanifold topology alone exceeds both triangle targets; tiny installed-runtime fixture corroborates protected-face behavior.',
 'no_mesh_repair_or_method_retry':True,'elapsed_seconds':time.monotonic()-began}
c.save(c.E/'result.json',result);c.save(c.OUT/'result.json',result)
print('LOD_QUALIFICATION='+json.dumps(result),flush=True)
