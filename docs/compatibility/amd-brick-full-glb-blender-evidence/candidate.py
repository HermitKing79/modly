"""Bounded pre-import referenced-index accounting; never changes a mesh."""
import sys
sys.dont_write_bytecode = True

def referenced_vertices(faces, vertex_count, batch_rows=32768):
    import numpy as np
    if (type(vertex_count) is not int or not 0 < vertex_count <= 2000000
            or type(batch_rows) is not int or not 1 <= batch_rows <= 32768
            or not isinstance(faces, np.ndarray) or faces.dtype != np.dtype('<i4')
            or faces.ndim != 2 or faces.shape[1] != 3 or not 0 < len(faces) <= 4173584):
        raise ValueError('Bounded original I32 triangle table and vertex count required')
    used = np.zeros(vertex_count, dtype=np.bool_)
    for start in range(0, len(faces), batch_rows):
        part = faces[start:start + batch_rows]
        if np.any(part < 0) or np.any(part >= vertex_count):
            raise ValueError('Face index outside original vertex table')
        used[part.reshape(-1)] = True
    count = int(np.count_nonzero(used))
    return used, dict(original_vertices=vertex_count, original_triangles=len(faces),
                     referenced_vertices=count, unreferenced_vertices=vertex_count-count,
                     exact_import_vertex_count_gate=count == vertex_count,
                     counting_policy='Distinct triangle index membership; no welding, geometric deduplication or changes')
