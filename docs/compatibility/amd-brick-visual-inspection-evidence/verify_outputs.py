"""Independent CPU readback of images/cameras and original ordered mesh bytes."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageStat
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-brick-visual-inspection-b536fb6';R=P/'renders'
def sha(f):
 with Path(f).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
 h=json.loads((R/'host-result.json').read_text());m=json.loads((R/'render-metadata.json').read_text());v=json.loads((P/'image-verification.json').read_text())
 assert h['classification']==m['classification']==v['classification']=='PASS'
 assert h['returncode']==0 and not h['job_cleanup']['forced_job_cleanup'] and h['job_cleanup']['owned_tree_exit_verified'] and h['source_unchanged']
 assert m['mesh_import_count']==1 and m['geometry_unchanged'] and m['geometry_payloads_before']==m['geometry_payloads_after']
 assert (m['vertex_count'],m['face_count'])==(1957891,4173584) and m['engine']=='CYCLES' and m['device']=='CPU' and not m['denoising_use_gpu'] and m['flat_shading']
 mesh=ROOT.parent/'artifacts/amd-brick-cpu-mesh-8473579/mesh'
 assert sha(mesh/'mesh.ply')=='ec9736e2510c26303196c4e3fc2045a1a1d36401b2576ee68f102ee29b551ffd'
 for filename,shape,dtype in [('vertices.npy',(1957891,3),'float32'),('faces.npy',(4173584,3),'int32')]:
  a=np.load(mesh/filename,mmap_mode='r',allow_pickle=False)
  try:
   assert a.shape==shape and str(a.dtype)==dtype
   assert hashlib.sha256(memoryview(a)).hexdigest()==m['geometry_payloads_before'][filename]
  finally:a._mmap.close()
 expected={'pos-x':(1,0,0),'neg-x':(-1,0,0),'pos-y':(0,1,0),'neg-y':(0,-1,0),'pos-z':(0,0,1),'neg-z':(0,0,-1),'iso-a':(1,-1,1),'iso-b':(-1,1,-1),'silhouette-pos-z':(0,0,1)}
 assert {x['name'] for x in m['views']}==set(expected) and len(m['views'])==9
 center=np.array(m['bounding_sphere_center']);stats={};actual_scale=float(np.float32(m['shared_ortho_scale']))
 for view in m['views']:
  d=np.array(view['camera_location'])-center;d/=np.linalg.norm(d)
  reference=np.array(expected[view['name']],dtype=float);reference/=np.linalg.norm(reference)
  assert np.max(np.abs(d-reference))<1e-6 and view['ortho_scale']==actual_scale
  f=R/view['PNG'];assert sha(f)==view['sha256']
  with Image.open(f) as im:
   assert im.format=='PNG' and im.mode=='RGB' and im.size==(1024,1024)
   im.load();stats[view['name']]=dict(extrema=im.getextrema(),channel_stddev=ImageStat.Stat(im).stddev)
 assert {x.name for x in R.glob('*.png')}=={x['PNG'] for x in m['views']}
 for info in v['derived_images'].values():
  assert sha(info['path'])==info['sha256']
  with Image.open(info['path']) as im:im.load();assert list(im.size)==info['dimensions']
 # Only textual JSON receipts enter the public repository, never image bytes.
 for source,dest in [(R/'host-result.json','host-result.json'),(R/'render-metadata.json','render-metadata.json'),(R/'first-view.json','first-view.json'),(R/'first-frame-host-verification.json','first-frame-host-verification.json'),(P/'image-verification.json','image-verification.json'),(P/'composition-host.json','composition-host.json')]:
  with (E/dest).open('xb') as out:out.write(source.read_bytes())
 raw=[R/'attempt-marker.json',R/'host-result.json',R/'render-metadata.json',R/'stdout.txt',R/'stderr.txt',P/'composition-host.json',P/'composition.stdout.txt',P/'composition.stderr.txt']
 receipt=dict(classification='PASS',source_mesh_unchanged=True,original_ordered_position_and_face_payloads_verified=True,exact_camera_axis_and_opposite_isometric_directions_verified=True,camera_direction_F32_tolerance=1e-6,shared_orthographic_scale_verified=True,computed_scale=m['shared_ortho_scale'],all_actual_scales_equal_exact_float32_round=actual_scale,nine_complete_PNGs_reopened=True,all_derived_images_reopened=True,image_pixel_statistics=stats,all_images_private_only=True,raw_private_files={str(f.relative_to(P)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in raw},actual_resources=dict(peak_working_set_bytes=h['process_memory']['peak_working_set_bytes'],peak_committed_bytes=h['peak_process_committed_bytes'],minimum_sampled_available_RAM_bytes=min(x['available_ram_bytes'] for x in h['resource_samples']),minimum_sampled_free_disk_bytes=min(x['disk_free_bytes'] for x in h['resource_samples']),samples=len(h['resource_samples']),supervisor_seconds=h['elapsed_seconds'],renderer_seconds=m['elapsed_seconds']))
 with (E/'independent-readback.json').open('x',encoding='utf-8') as out:json.dump(receipt,out,indent=2,allow_nan=False);out.write('\n')
 print(json.dumps(receipt['actual_resources']));print(json.dumps(v['derived_images'],indent=2))
 print('PASS: nine PNGs, derived sheets, exact camera directions/F32 scale and original ordered mesh bytes; no pixel assets copied into Git.')
if __name__=='__main__':main()
