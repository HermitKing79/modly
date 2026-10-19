"""CPU-only, read-only topology measurement of the original triangle arrays.

No welding, repair, face removal, mesh export, or model imports occur here.
The caller supplies its owned private output directory; the only written files
are new diagnostic NPY caches. Public results contain counts, not geometry.
"""

from pathlib import Path
import time

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def analyze(vertices, faces, owned_outdir):
    """Measure all original faces and return JSON-safe counts and cache paths.

    Edge incidence counts every one of the three edges of every triangle,
    including occurrences in repeated-index triangles. Components connect
    original vertex indices through those edges; equal coordinates do not weld.
    Isolated unused vertices each form a vertex component, but no face component.
    The cache edge key is ``min_index * vertex_count + max_index`` (uint64).
    Decode a key with integer quotient and remainder by the input vertex count.
    Caches are sorted unique edge keys, matching exact multiplicities, and an
    int32 component label for every original vertex, including unused vertices.
    """
    started = time.perf_counter()
    vertices = np.asanyarray(vertices)
    faces = np.asanyarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.dtype.kind not in "fiu":
        raise ValueError("vertices must be a numeric N by 3 array")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
        raise ValueError("faces must be an integer F by 3 array")
    vertex_count, face_count = len(vertices), len(faces)
    if vertex_count > np.iinfo(np.int32).max:
        raise ValueError("vertex count exceeds the int32 component-cache bound")
    if face_count and (int(faces.min()) < 0 or int(faces.max()) >= vertex_count):
        raise ValueError("face index is outside the original vertex array")
    if face_count * 3 > np.iinfo(np.uint32).max:
        raise ValueError("triangle occurrence count exceeds the uint32 incidence bound")

    output = Path(owned_outdir).resolve()
    paths = {name: output / (name + ".npy") for name in (
        "edge_keys", "incidence", "vertex_components",
    )}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("diagnostic cache files must be new; refusing to overwrite")
    output.mkdir(parents=True, exist_ok=True)

    # The largest allocation is exactly 3 * F * 8 bytes, about 96 MiB for
    # the genuine brick. All face-coordinate work uses bounded small batches.
    all_edges = np.empty(face_count * 3, dtype=np.uint64)
    used = np.zeros(vertex_count, dtype=np.bool_)
    repeated_count = zero_area_count = nonfinite_faces = 0
    nonfinite_vertices = 0
    batch_rows = 32768
    for start in range(0, vertex_count, batch_rows):
        nonfinite_vertices += int(np.count_nonzero(
            ~np.isfinite(vertices[start:start + batch_rows]).all(axis=1)
        ))
    for start in range(0, face_count, batch_rows):
        block = faces[start:start + batch_rows]
        size = len(block)
        used[block.reshape(-1)] = True
        repeated_count += int(np.count_nonzero(
            (block[:, 0] == block[:, 1])
            | (block[:, 1] == block[:, 2])
            | (block[:, 2] == block[:, 0])
        ))
        positions = np.asarray(vertices[block], dtype=np.float64)
        finite = np.isfinite(positions).all(axis=(1, 2))
        nonfinite_faces += int(np.count_nonzero(~finite))
        with np.errstate(invalid="ignore", over="ignore"):
            cross = np.cross(positions[:, 1] - positions[:, 0], positions[:, 2] - positions[:, 0])
            zero_area_count += int(np.count_nonzero(finite & np.all(cross == 0, axis=1)))
        for edge_column, (left, right) in enumerate(((0, 1), (1, 2), (2, 0))):
            low = np.minimum(block[:, left], block[:, right]).astype(np.uint64)
            high = np.maximum(block[:, left], block[:, right]).astype(np.uint64)
            low *= np.uint64(vertex_count)
            low += high
            offset = edge_column * face_count + start
            all_edges[offset:offset + size] = low
    unused_vertices = vertex_count - int(np.count_nonzero(used))
    del used
    all_edges.sort(kind="quicksort")
    if len(all_edges):
        starts = np.empty(len(all_edges), dtype=np.bool_)
        starts[0] = True
        np.not_equal(all_edges[1:], all_edges[:-1], out=starts[1:])
        starts = np.flatnonzero(starts)
        unique_edges = all_edges[starts]
        counts = np.diff(starts, append=len(all_edges))
        max_incidence = int(counts.max())
        del starts
    else:
        unique_edges = np.empty(0, dtype=np.uint64)
        counts = np.empty(0, dtype=np.int64)
        max_incidence = 0
    del all_edges
    edge_counts = {
        "unique": int(len(unique_edges)),
        "boundary_incidence_1": int(np.count_nonzero(counts == 1)),
        "manifold_incidence_2": int(np.count_nonzero(counts == 2)),
        "nonmanifold_incidence_gt_2": int(np.count_nonzero(counts > 2)),
        "max_incidence": max_incidence,
        "self_edges": int(np.count_nonzero(unique_edges % (vertex_count + 1) == 0)),
    }
    np.save(paths["edge_keys"], unique_edges, allow_pickle=False)
    incidence_dtype = np.uint8 if max_incidence <= 255 else np.uint32
    np.save(paths["incidence"], counts.astype(incidence_dtype), allow_pickle=False)
    del counts

    # Sorted keys already group CSR source rows. An undirected traversal of
    # this upper-triangle adjacency is exactly original-index connectivity.
    if vertex_count:
        lower = (unique_edges // np.uint64(vertex_count)).astype(np.int32)
        upper = (unique_edges % np.uint64(vertex_count)).astype(np.int32)
        indptr = np.empty(vertex_count + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(np.bincount(lower, minlength=vertex_count), out=indptr[1:])
        del lower
        graph = csr_matrix((np.ones(len(upper), dtype=np.uint8), upper, indptr),
                           shape=(vertex_count, vertex_count))
        component_count, labels = connected_components(graph, directed=False, return_labels=True)
        labels = labels.astype(np.int32, copy=False)
        del graph, upper, indptr
    else:
        component_count, labels = 0, np.empty(0, dtype=np.int32)
    del unique_edges
    np.save(paths["vertex_components"], labels, allow_pickle=False)
    vertex_sizes = np.bincount(labels, minlength=component_count)
    face_sizes = np.zeros(component_count, dtype=np.int64)
    for start in range(0, face_count, batch_rows):
        face_labels = labels[faces[start:start + batch_rows, 0]]
        component_ids, block_counts = np.unique(face_labels, return_counts=True)
        face_sizes[component_ids] += block_counts
    del labels

    # Components are ranked by vertex count, with deterministic label tie-breaks.
    largest_ids = np.argsort(-vertex_sizes, kind="stable")[:20]
    largest = [{"label": int(label), "vertices": int(vertex_sizes[label]),
                "faces": int(face_sizes[label])} for label in largest_ids]
    vertex_histogram, face_histogram = {}, {}
    positive_face_sizes = face_sizes[face_sizes > 0]
    for name, low, high in (("1", 1, 1), ("2_to_10", 2, 10),
                            ("11_to_100", 11, 100), ("101_to_1000", 101, 1000),
                            ("1001_to_10000", 1001, 10000),
                            ("greater_than_10000", 10001, np.iinfo(np.int64).max)):
        vertex_histogram[name] = int(np.count_nonzero((vertex_sizes >= low) & (vertex_sizes <= high)))
        face_histogram[name] = int(np.count_nonzero((positive_face_sizes >= low) & (positive_face_sizes <= high)))
    return {
        "input_counts": {"vertices": vertex_count, "triangles": face_count,
                         "triangle_edge_occurrences": face_count * 3},
        "edges": edge_counts,
        "degeneracy": {
            "repeated_index_faces": repeated_count,
            "zero_area_faces_float64": zero_area_count,
            "zero_area_definition": "cross(v1-v0,v2-v0) exactly zero in float64; no epsilon",
            "nonfinite_vertex_rows": nonfinite_vertices,
            "faces_with_nonfinite_coordinates": nonfinite_faces,
            "duplicate_faces_assessed": False,
            "duplicate_faces_reason": "Not assessed; independent full face sort omitted to bound CPU and memory.",
        },
        "components": {
            "definition": "Connectivity through shared original vertex indices; no coordinate welding. Unused vertices are isolated vertex components.",
            "face_components_definition": "Face-bearing vertex-index-connected components; excludes all unused vertices.",
            "face_connectivity_limitation": "Faces meeting only at a shared vertex belong to the same component. Edge-connected face components are not assessed.",
            "vertex_components_including_unused": int(component_count),
            "face_components": int(len(positive_face_sizes)),
            "unused_vertices": unused_vertices,
            "largest_order": "descending vertex count, then ascending component label",
            "largest": largest,
            "vertex_size_histogram": vertex_histogram,
            "face_size_histogram": face_histogram,
            "tiny_vertex_components_le_10": int(np.count_nonzero(vertex_sizes <= 10)),
            "tiny_face_components_le_10": int(np.count_nonzero(positive_face_sizes <= 10)),
            "faces_in_tiny_face_components_le_10": int(positive_face_sizes[positive_face_sizes <= 10].sum()),
            "vertices_in_tiny_vertex_components_le_10": int(vertex_sizes[vertex_sizes <= 10].sum()),
        },
        "private_cache": {name: str(path) for name, path in paths.items()},
        "execution": {"CPU_only": True, "numpy": np.__version__, "scipy": scipy.__version__,
                      "elapsed_seconds": round(time.perf_counter() - started, 6),
                      "face_batch_rows": batch_rows, "source_arrays_mutated": False},
    }
