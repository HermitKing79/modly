"""Read-only, stdlib engine snapshot audit against authenticated GLB.

Discover a unique proper signed-axis permutation from GEOMETRY ONLY, require all
1000 oriented triangles, then check every corner attribute at declared thresholds.
No editor connection, reflection, engine execution, project edit or retry occurs.
"""
import argparse
import collections
import itertools
import json
import math
from pathlib import Path
import re
import sys
import verify_conversion as v

EXPECTED_MODEL = "modly_tests/phase21_static_prop/experimental_static_prop.vmdl"
POSITION_TOLERANCE_METRES = 2e-6
UV_TOLERANCE = .0005
NORMAL_UNIT_ERROR = .015
NORMAL_UNIT_DOT = .9999


class AuditFailure(v.ValidationError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def determinant(perm, signs):
    inversions = sum(perm[i] > perm[j] for i in range(3) for j in range(i + 1, 3))
    return (-1 if inversions % 2 else 1) * math.prod(signs)


def undo_rotation(point, perm, signs):
    obj = [0., 0., 0.]
    for engine_axis in range(3):
        obj[perm[engine_axis]] = signs[engine_axis] * point[engine_axis]
    return tuple(obj)


def obj_to_glb(point, is_normal=False):
    scale = 1 if is_normal else .0254
    return (-point[1]*scale, point[2]*scale, -point[0]*scale)


def geometric_correspondence(source, snapshot):
    unique = list(dict.fromkeys(source['positions']))
    source_ids = {point: i for i, point in enumerate(unique)}
    source_faces = {}
    for face_number, face in enumerate(source['triangles']):
        ids = tuple(source_ids[source['positions'][i]] for i in face)
        key = v.cyclic_key(ids)
        v.require(key not in source_faces, 'Duplicate geometric source face needs multiset attribute pairing')
        source_faces[key] = (face_number, face, ids)
    cell_size = POSITION_TOLERANCE_METRES
    buckets = collections.defaultdict(list)
    for i, point in enumerate(unique):
        buckets[tuple(math.floor(x/cell_size) for x in point)].append(i)
    engine_positions = [tuple(vertex['P']) for vertex in snapshot['Vertices']]
    engine_bounds = v.bounds(engine_positions)
    obj_positions = [(-p[2]/.0254, -p[0]/.0254, p[1]/.0254) for p in source['positions']]
    expected_obj_bounds = v.bounds(obj_positions)
    candidates = []
    for perm in itertools.permutations(range(3)):
        if any(abs(engine_bounds['dimensions'][i] - expected_obj_bounds['dimensions'][perm[i]]) > 2*cell_size/.0254
               for i in range(3)):
            continue
        for signs in itertools.product((-1, 1), repeat=3):
            if determinant(perm, signs) != 1:
                continue
            ids, errors, recovered = [], [], []
            for point in engine_positions:
                original = obj_to_glb(undo_rotation(point, perm, signs))
                recovered.append(original)
                cell = tuple(math.floor(x/cell_size) for x in original)
                matches = []
                for delta in itertools.product((-1, 0, 1), repeat=3):
                    for i in buckets.get(tuple(a+b for a,b in zip(cell,delta)), ()):
                        error = math.dist(original, unique[i])
                        if error <= cell_size:
                            matches.append((i,error))
                if len(matches) != 1:
                    break
                ids.append(matches[0][0])
                errors.append(matches[0][1])
            if len(ids) != len(engine_positions) or set(ids) != set(range(502)):
                continue
            face_ids = [tuple(ids[snapshot['Indices'][i+j]] for j in range(3)) for i in range(0,3000,3)]
            if collections.Counter(v.cyclic_key(face) for face in face_ids) != collections.Counter(source_faces.keys()):
                continue
            candidates.append(dict(perm=perm, signs=signs, ids=ids, face_ids=face_ids,
                                   errors=errors, recovered=recovered, source_faces=source_faces))
    v.require(len(candidates) == 1,
              f'Expected one proper axis rotation with full oriented bijection; found {len(candidates)}. No reflection or arbitrary fit accepted')
    return candidates[0]


def inspect_vmdl(text):
    classes = re.findall(r'_class\s*=\s*"([^"\n]+)"', text)
    v.require(classes.count('RenderMeshFile') == 1, 'Expected exactly one ModelDoc RenderMeshFile')
    forbidden = [name for name in classes if re.search(r'physics|collision|lod', name, re.I)]
    v.require(not forbidden, f'Unexpected collision/physics/LOD ModelDoc nodes: {forbidden}')
    scales = re.findall(r'\bimport_scale\s*=\s*([-+0-9.eE]+)', text)
    v.require(len(scales) == 1 and float(scales[0]) == 1., 'ModelDoc scale must be exactly one')
    for field in ('import_translation', 'import_rotation'):
        values = re.findall(r'\b'+field+r'\s*=\s*\[([^\]]+)\]', text)
        for vector in values:
            components = [float(x) for x in vector.replace(',', ' ').split()]
            v.require(components == [0.,0.,0.], f'Unexpected ModelDoc {field}')
    return dict(classes=classes, physics_collision_lod_nodes=[], import_scale=1.,
                qualification='No collision/LOD configuration or generation nodes; additional compiled LOD coverage is not exposed by selected public readback API')


def compare_snapshot(source, snapshot, vmdl_text):
    v.require(snapshot['ModelPath'].replace('\\','/').lower() == EXPECTED_MODEL, 'Unexpected loaded model path')
    v.require(snapshot['IsError'] is False, 'Error model loaded')
    v.require(snapshot['PhysicsParts'] == 0, 'Compiled model has physics body parts')
    v.require(snapshot['Lod0IndexCount'] == 3000 and len(snapshot['Indices']) == 3000, 'LOD0/raw index count must be 3000')
    vertices = snapshot['Vertices']
    v.require(0 < len(vertices) <= 10000, 'Missing or excessive experimental readback vertices')
    for vertex in vertices:
        v.require(len(vertex['P']) == len(vertex['N']) == 3 and len(vertex['UV']) == 2,
                  'Missing position/normal/UV stream shape')
        v.require(all(v.finite(vertex[field]) for field in ('P','N','UV')), 'Nonfinite experimental readback attribute')
    v.require(all(type(i) is int and 0 <= i < len(vertices) for i in snapshot['Indices']), 'Out-of-range index')
    v.require(set(snapshot['Indices']) == set(range(len(vertices))), 'Unused readback vertices')
    v.require(len({tuple(vertex['P']) for vertex in vertices}) == 502, 'Engine geometric position count differs')
    v.require(snapshot['Materials'] and all(material['Path'] and material['IsError'] is False for material in snapshot['Materials']),
              'Missing/error preview material resource')
    modeldoc = inspect_vmdl(vmdl_text)
    correspondence = geometric_correspondence(source, snapshot)
    perm, signs = correspondence['perm'], correspondence['signs']
    direct_uv_error = flipped_uv_error = normal_error = raw_normal_error = 0.
    min_dot = 1.
    normal_bad_corners = 0
    source_lengths, actual_lengths = [], []
    worst_normal = None
    for face_number, ids in enumerate(correspondence['face_ids']):
        _, source_face, source_ids = correspondence['source_faces'][v.cyclic_key(ids)]
        shift = next(k for k in range(3) if source_ids[k:] + source_ids[:k] == ids)
        source_face = source_face[shift:] + source_face[:shift]
        for corner_number, source_index in enumerate(source_face):
            vertex_index = snapshot['Indices'][3*face_number+corner_number]
            vertex = vertices[vertex_index]
            source_uv = source['uvs'][source_index]
            actual_uv = vertex['UV']
            direct_uv_error = max(direct_uv_error, abs(actual_uv[0]-source_uv[0]), abs(actual_uv[1]-(1-source_uv[1])))
            flipped_uv_error = max(flipped_uv_error, *(abs(a-b) for a,b in zip(actual_uv,source_uv)))
            actual_normal = obj_to_glb(undo_rotation(vertex['N'],perm,signs), True)
            source_normal = source['normals'][source_index]
            sl = math.sqrt(sum(x*x for x in source_normal))
            al = math.sqrt(sum(x*x for x in actual_normal))
            v.require(sl > 0 and al > 0, 'Zero/default normal direction in experimental stream')
            source_lengths.append(sl); actual_lengths.append(al)
            source_unit = tuple(x/sl for x in source_normal)
            actual_unit = tuple(x/al for x in actual_normal)
            error = math.dist(source_unit, actual_unit)
            dot = sum(a*b for a,b in zip(source_unit,actual_unit))
            if error >= normal_error:
                worst_normal = dict(engine_face=face_number, corner=corner_number, source_attribute=source_index,
                                    source_normal=source_normal, engine_normal_in_glb_basis=actual_normal,
                                    unit_error=error, unit_dot=dot)
            normal_error = max(normal_error,error)
            raw_normal_error = max(raw_normal_error,math.dist(source_normal,actual_normal))
            min_dot = min(min_dot,dot)
            normal_bad_corners += error > NORMAL_UNIT_ERROR or dot < NORMAL_UNIT_DOT
    direct_pass, flip_pass = direct_uv_error <= UV_TOLERANCE, flipped_uv_error <= UV_TOLERANCE
    convention = ('OBJ_UV_PRESERVED' if direct_pass else 'GLOBAL_V_FLIP_RELATIVE_TO_OBJ' if flip_pass else 'NO_CONSISTENT_CONVENTION')
    engine_bounds = v.bounds([vertex['P'] for vertex in vertices])
    bounds_error = max(abs(a-b) for key, field in (('min','BoundsMin'),('max','BoundsMax'))
                       for a,b in zip(engine_bounds[key],snapshot[field]))
    axes = 'xyz'
    rotation_description = [('-' if sign < 0 else '+')+'OBJ.'+axes[axis] for axis,sign in zip(perm,signs)]
    result = dict(status='PASS', unique_geometric_positions=502, oriented_triangles=1000,
                  compared_corners=3000, readback_vertices=len(vertices), readback_indices=3000,
                  unused_vertices=0, reversed_missing_extra_faces=0,
                  max_position_error_metres=max(correspondence['errors']),
                  measured_obj_to_engine_rotation=dict(engine_xyz=rotation_description, permutation=list(perm),
                                                       signs=list(signs), determinant=1,
                                                       selection='Unique full oriented geometric bijection; attributes not used to choose rotation'),
                  engine_xyz_inches=engine_bounds, recovered_glb_xyz_metres=v.bounds(correspondence['recovered']),
                  snapshot_bounds_max_error_inches=bounds_error,
                  uv=dict(convention=convention, tolerance=UV_TOLERANCE,
                          max_error_obj_uv_preserved=direct_uv_error, max_error_global_v_flip=flipped_uv_error,
                          consistent_global_flip_qualified=flip_pass and not direct_pass),
                  normals=dict(max_unit_direction_error=normal_error,min_unit_dot=min_dot,
                               thresholds=dict(max_unit_error=NORMAL_UNIT_ERROR,min_unit_dot=NORMAL_UNIT_DOT),
                               affected_corners=normal_bad_corners,max_raw_vector_error=raw_normal_error,
                               source_length_range=[min(source_lengths),max(source_lengths)],
                               engine_length_range=[min(actual_lengths),max(actual_lengths)],worst_corner=worst_normal),
                  physics_parts=0, modeldoc=modeldoc, materials=snapshot['Materials'],
                  material_qualification='Engine preview material only; original roughness.35/metallic0/basecolor engine equivalence NOT TESTED',
                  readback_qualification='GetVertices/GetIndices public experimental APIs; matched returned single mesh/LOD0 data, no claim of extra LOD/draw-call stream coverage')
    failures=[]
    if not (direct_pass or flip_pass): failures.append('No globally consistent UV convention within .0005')
    if normal_bad_corners: failures.append(f'{normal_bad_corners} corners fail unit normal thresholds')
    if bounds_error > POSITION_TOLERANCE_METRES/.0254: failures.append('Snapshot bounds differ from actual vertex bounds')
    if failures:
        result['status']='NO-GO'
        raise AuditFailure('; '.join(failures),result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).resolve().parent
    parser.add_argument('--glb',type=Path,default=base.parent/'amd-static-prop-uv-improve-20261031-v1/asset/experimental-static-prop.glb')
    parser.add_argument('--snapshot',type=Path,default=base/'engine-model-snapshot.json')
    parser.add_argument('--vmdl',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args(argv)
    report=dict(status='NO-GO',thresholds=dict(position_metres=POSITION_TOLERANCE_METRES,uv=UV_TOLERANCE,
                                             unit_normal_error=NORMAL_UNIT_ERROR,unit_normal_dot=NORMAL_UNIT_DOT))
    try:
        source=v.read_glb(args.glb)
        snapshot_bytes=args.snapshot.read_bytes(); vmdl_bytes=args.vmdl.read_bytes()
        report.update(glb_sha256=source['sha256'],snapshot_sha256=v.sha256(snapshot_bytes),
                      vmdl_sha256=v.sha256(vmdl_bytes),vmdl_path=str(args.vmdl.resolve()))
        report['audit']=compare_snapshot(source,json.loads(snapshot_bytes.decode('utf-8-sig')),vmdl_bytes.decode('utf-8-sig'))
        report['status']='PASS'
    except (v.ValidationError,ValueError,KeyError,IndexError,OSError) as error:
        report['error']=f'{type(error).__name__}: {error}'
        if isinstance(error,AuditFailure): report['audit']=error.details
    text=json.dumps(report,indent=2,allow_nan=False)
    with args.report.open('x',encoding='utf-8') as stream: stream.write(text+'\n')
    print(text)
    return 0 if report['status']=='PASS' else 2


if __name__=='__main__': sys.exit(main())
