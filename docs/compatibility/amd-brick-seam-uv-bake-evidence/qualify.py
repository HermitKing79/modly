"""Seam-only atlas qualification; no material sampling or baking."""
import sys,time
from pathlib import Path
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from common import *

def main():
    auth=begin();started=time.monotonic();before=identity_inputs(auth)
    import numpy as np
    from uv_layout import layout,validate_layout,first_overlap
    math=originals();prior=verify(P/'prepare','patch.json');p=P/'prepare';out=P/'qualify'
    v,f,sf,sv,direct,old_uv=[np.load(p/(n+'.npy'),allow_pickle=False) for n in ('vertices','faces','source-face-ids','source-vertex-ids','direct','planar-uv')]
    # Independent replay of the old overlap witness, including original IDs.
    witness=first_overlap(old_uv,f)
    if witness is None or [int(sf[witness[k]]) for k in ('face_i','face_j')]!=prior['overlap']['original_triangle_ids'] or abs(witness['intersection_area']-prior['overlap']['UV_intersection_area'])>1e-12:raise ValueError('Original planar overlap identity changed')
    result=layout(v,f,sf,sv,size=1024,gutter=4,min_density=4096,min_face_texels=1)
    validated=validate_layout(result,v,f,source_face_ids=sf,source_vertex_ids=sv)
    independent=math.assert_injective(result.uv,result.faces)
    original_raster,original_bary=math.rasterize(result.uv,result.faces,1024)
    if not np.array_equal(original_raster,result.raster_ids) or not np.allclose(original_bary,result.barycentrics,rtol=0,atol=1e-7):raise ValueError('Independent raster/UV image orientation differs')
    original_vertices=np.load(auth.PINS['vertices'][0],mmap_mode='r',allow_pickle=False)
    original_faces=np.load(auth.PINS['faces'][0],mmap_mode='r',allow_pickle=False)
    if not np.array_equal(result.vertices.view(np.uint32),original_vertices[result.source_vertex_ids].view(np.uint32)) or not np.array_equal(result.source_vertex_ids[result.faces],original_faces[result.source_face_ids]):raise ValueError('UV splits altered original oriented triangles')
    corner_direct=direct[f].reshape(-1,6)
    data={'vertices':result.vertices,'faces':result.faces,'uv':result.uv,'direct':corner_direct,
          'source-face-ids':result.source_face_ids,'source-vertex-ids':result.source_vertex_ids,
          'face-island-ids':result.face_island_ids,'island-rects':result.island_rects,
          'raster-ids':result.raster_ids,'barycentrics':result.barycentrics,'face-coverage':result.face_coverage}
    if identity_inputs(auth)!=before:raise ValueError('Original inputs changed during UV qualification')
    for name,array in data.items():np_save(out/(name+'.npy'),array)
    record=dict(classification='PASS',CPU_only=True,bake_performed=False,inputs_unchanged=True,
                selected_original_triangles=len(f),selected_original_vertices=len(v),UV_corner_vertices=len(result.vertices),
                original_triangle_positions_winding_multiplicity_bits_preserved=True,original_source_vertex_and_face_ids_exact=True,
                old_overlap_independently_reproduced=witness,UV_report=result.report,independent_original_overlap_check=independent,
                independent_original_raster_equal=True,independent_validation=validated,
                UV_image_orientation='All raster arrays V-up; PNG row flip performed only at image persistence',
                padding_requirement='Each disjoint integer outer rectangle owns its own Telea padding; never sample across island rectangles; 4 pixel gutters each side',
                elapsed_seconds=time.monotonic()-started,prepare_receipt_sha256=digest(p/'patch.json'),
                files=manifest(out,[n+'.npy' for n in data]))
    save(out/'uv.json',record);print('UV_PASS triangles='+str(len(f))+' islands='+str(result.report['island_count'])+' covered='+str(result.report['covered_texels']),flush=True)

if __name__=='__main__':main()
