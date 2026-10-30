"""Geometry-based source retention; prior preferred-group metric stays diagnostic."""
import json
import math
import sys
import numpy as np
from pathlib import Path
import amd_finite_patch_proof as p
from amd_cgal_proof import save,sha
from amd_cgal_official_check import read_off
from amd_mesh_aware_input import RAW
from amd_mesh_aware_result import point_triangle_distances

def assess_geometry():
    contract=json.loads((p.BASE/'feature-validation-contract.json').read_text())
    if contract['point_to_surface_over_D']!=.002 or contract['unsigned_normal_degrees']!=10:raise ValueError('Frozen feature geometry requirements differ')
    src=np.load(p.BASE/'input/source-analysis.npz',allow_pickle=False);D=float(src['diagonal']);center=src['center']
    labels=np.fromfile(p.BASE/'region-labels.bin',dtype='<i4');rows=np.flatnonzero(labels>=0)
    source_v=np.load(RAW/'vertices.npy',allow_pickle=False);source_f=np.load(RAW/'faces.npy',allow_pickle=False)
    if len(labels)!=len(source_f) or len(rows)!=1925348:raise ValueError('Frozen source patch membership differs')
    source_tri=source_v[source_f[rows]].astype('f8');normal=src['normal'][rows]
    centroid=source_tri.mean(axis=1);radius=np.linalg.norm(source_tri-centroid[:,None,:],axis=2).max(axis=1)
    v,f=read_off(p.BASE/'validation/triangulated-native.off');out=np.array(v)/(100/D)+center;triangles=out[np.array(f,dtype='i4')]
    best=np.full((len(rows),3),np.inf);cosine=math.cos(math.radians(10));limit=.002*D
    for tri in triangles:
        n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);length=np.linalg.norm(n)
        if length<=0 or not math.isfinite(float(length)):raise ValueError('Nondegenerate actual output required')
        n/=length;offset=float(n@tri[0])
        # Centroid/radius is only a safe prefilter. Distances are to the actual
        # finite triangle, never its plane or a bounding footprint.
        use=np.flatnonzero((abs(normal@n)>=cosine+1e-12)&(abs(centroid@n-offset)<=limit+radius+1e-12*D))
        for start in range(0,len(use),32768):
            ids=use[start:start+32768];dist=point_triangle_distances(np,source_tri[ids].reshape(-1,3),tri).reshape(-1,3)
            best[ids]=np.minimum(best[ids],dist)
    finite=np.isfinite(best);passed=finite&(best<=limit-1e-10*D);patches=[]
    for region in np.unique(labels[rows]):
        selected=labels[rows]==region;values=best[selected]
        patches.append(dict(region=int(region),source_triangles=int(selected.sum()),vertex_incidences=int(selected.sum()*3),failed_incidences=int((~passed[selected]).sum()),maximum_distance_over_D=float(values.max()/D) if np.isfinite(values).all() else None,PASS=bool(passed[selected].all())))
    gate=bool(passed.all());legacy=json.loads((p.BASE/'candidate-result.json').read_text())
    old_features=legacy['features_PASS']
    required=legacy['topology_PASS'] and legacy['views_PASS'] and legacy['complete_finite_footprint_certified'] and legacy['complete_compatible_domain_footprints'] and legacy['independent_bruteforce_PASS'] and legacy['original_units_PLY_exact_readback'] and all(row['PASS'] for row in legacy['bidirectional_triangle_surface_distances_over_AI_diagonal'].values())
    selection=json.loads((p.BASE/'owned-session/selection.json').read_text())
    required=required and selection['all_exclusions_exactly_zero'] and selection['mapping_reverified_before_deletion'] and selection['selected_face_count']>0
    save(p.BASE/'geometry-feature-retention.json',dict(classification='PASS' if gate else 'FAIL',contract_sha256=sha(p.BASE/'feature-validation-contract.json'),labelled_source_triangles=len(rows),patches=patches,unlabelled_or_locked_triangles=len(source_f)-len(rows),vertex_incidences=len(rows)*3,failed_incidences=int((~passed).sum()),finite_distance_incidences=int(finite.sum()),source_triangle_interiors_certified=False,scope='Every vertex incidence of native-labelled source triangles, against union of actual output triangles satisfying that source face normal within10degrees; no preferred-group requirement',old_preferred_group_result_retained_as_diagnostic=old_features,normal_guard=1e-12,distance_guard_over_D=1e-10))
    save(p.BASE/'measured-result.json',dict(classification='PASS_MEASURED_GATES_PENDING_VISUAL_REVIEW' if required and gate else 'NO_GO_CANDIDATE',label='AI-DERIVED RECONSTRUCTION',raw_neural_geometry_PASS=False,geometric_feature_retention_PASS=gate,all_other_frozen_measured_gates_PASS=bool(required),legacy_preferred_group_retention_diagnostic=old_features,historical_and_locked_source_areas_consumed_by_solver_constraints=True,neither_validation_set_blind_to_constraints=True,visual_review_required=True))
